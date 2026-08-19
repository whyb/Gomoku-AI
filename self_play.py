"""
自博弈数据生成 — AlphaZero 风格

核心流程:
  1. 从当前模型 + MCTS 生成对弈数据
  2. 每步: MCTS 搜索 → 得到策略 π_mcts → 采样落子
  3. 终局: 用胜负结果 z 标注所有经验
  4. 对称增强: 每条经验生成 8 个等价版本

产出: (state, π_mcts, z) 三元组用于训练
"""

import time
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from mcts import MCTS, BatchMCTS, get_forced_move
from symmetry import SymmetryAugmenter
from win_reason import reconstruct_game


@dataclass
class GameStep:
    """一局中的一步记录"""
    state: np.ndarray       # (2, H, W) 棋盘状态
    policy: np.ndarray      # (H*W,) MCTS 策略向量
    player: int             # 当前玩家 (1 或 2)


@dataclass
class GameResult:
    """一局的结果"""
    winner: int             # 1, 2, 0(平局)
    steps: List[GameStep]
    total_moves: int
    game_id: int
    main_player: int = 0    # 0=双方都是主模型(self-play), 1=P1是主模型, 2=P2是主模型


class SelfPlayWorker:
    """
    自博弈 Worker — 生成一局对弈的完整数据

    与 train_dy.py 的 env_worker 对比:
    - train_dy.py: Player1=NN(ε-greedy), Player2=Kali-Hac
    - self_play:   Player1=MCTS+NN, Player2=MCTS+NN (同一模型)
    """

    def __init__(self, model: nn.Module, device: str,
                 board_size: int = 10, win_condition: int = 5,
                 num_simulations: int = 400, c_puct: float = 1.5,
                 dirichlet_alpha: float = 0.3,
                 dirichlet_epsilon: float = 0.25,
                 temp_threshold: int = 30,
                 temp_final: float = 0.1,
                 draw_reward: float = 0.0,
                 random_open_moves: int = 0,
                 raw_selfplay_frac: float = 0.0,
                 use_batch_mcts: bool = True,
                 mcts_batch_size: int = 16,
                 fp16: bool = False,
                 use_human_knowledge: bool = False):
        """
        Args:
            model: 神经网络模型
            device: 推理设备
            board_size: 棋盘大小
            win_condition: 连子数
            num_simulations: MCTS 模拟次数
            c_puct: PUCT 探索常数
            dirichlet_alpha: Dirichlet 噪声 alpha
            dirichlet_epsilon: Dirichlet 噪声混合比例
            temp_threshold: 温度退火阈值 (前 N 步 τ=1.0)
            temp_final: 阈值后的温度 (越大越随机, 默认 0.1)
            draw_reward: 平局的价值目标 (默认 0.0; 设为负值可让模型厌恶平局、
                         主动求胜, 防止价值头坍缩到 0)
            random_open_moves: 每局随机开局 N 手 (制造不平衡局面, 增加胜负样本)
            raw_selfplay_frac: 每局以该概率生成"裸棋"对局 (默认 0.0):
              只保留 立即连五/立即堵五 两条硬规则, 关闭 活四双四/活三防守/
              双活三堵三端 等防守短路与树内战术先验, 让模型亲身体验
              "放任对手活三 → 被做成活四/双活三 → 输" 的后果,
              从而学会预防而不是依赖规则保命。对手池对局中对手仍全力防守。
            use_batch_mcts: 是否使用批量 MCTS
            mcts_batch_size: 批量 MCTS 的 batch 大小
            fp16: MCTS 推理是否使用 FP16 混合精度
            use_human_knowledge: 是否在搜索树中用人类知识增强 (默认关闭)
        """
        self.model = model
        self.device = device
        self.board_size = board_size
        self.win_condition = win_condition
        self.temp_threshold = temp_threshold
        self.temp_final = temp_final
        self.draw_reward = draw_reward
        self.random_open_moves = random_open_moves
        self.raw_selfplay_frac = max(0.0, min(1.0, raw_selfplay_frac))

        if use_batch_mcts:
            self.mcts = BatchMCTS(
                model, device,
                num_simulations=num_simulations,
                c_puct=c_puct,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_epsilon=dirichlet_epsilon,
                batch_size=mcts_batch_size,
                fp16=fp16,
                win_condition=win_condition,
                use_human_knowledge=use_human_knowledge
            )
        else:
            self.mcts = MCTS(
                model, device,
                num_simulations=num_simulations,
                c_puct=c_puct,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_epsilon=dirichlet_epsilon,
                fp16=fp16,
                win_condition=win_condition,
                use_human_knowledge=use_human_knowledge
            )

    def _get_forced_move_level(self, board: np.ndarray, player: int,
                               tactical_level: int
                               ) -> Tuple[Optional[int], Optional[str]]:
        """按战术等级取强制走法: 0=全部规则, >=1=仅 立即连五/立即堵五"""
        if tactical_level >= 1:
            return get_forced_move(board, player, self.win_condition,
                                   max_priority=2)
        return get_forced_move(board, player, self.win_condition)

    def play_one_game(self, game_id: int = 0,
                      opening_moves: Optional[List[Tuple[int, int]]] = None
                      ) -> GameResult:
        """
        进行一局自博弈

        Args:
            game_id: 游戏编号
            opening_moves: 可选的开局落子 [(x,y), ...]
        Returns:
            GameResult 包含所有状态、策略和终局结果
        """
        board = np.zeros((self.board_size, self.board_size), dtype=np.int32)
        steps: List[GameStep] = []

        # 随机开局: 若未显式指定开局且启用了随机开局, 生成 N 个不重复的随机落子
        if opening_moves is None:
            opening_moves = self._random_opening()

        # 应用开局
        start_step = 0
        if opening_moves:
            for i, (x, y) in enumerate(opening_moves):
                player = 1 if i % 2 == 0 else 2
                board[x, y] = player
            start_step = len(opening_moves)

        # 裸棋对局: 本局关闭防守短路与树内战术先验
        tactical_level = 1 if np.random.random() < self.raw_selfplay_frac else 0

        current_player = 1 if start_step % 2 == 0 else 2
        max_steps = self.board_size * self.board_size
        winner = 0

        for step in range(start_step, max_steps):
            # 构建状态
            state = self._build_state(board, current_player)

            # 温度退火 (temp_final 可调, 越大中后盘越随机)
            temperature = 1.0 if step < self.temp_threshold else self.temp_final

            # 先手第一手完全均匀随机落子 (不固定天元); 第二手起走 MCTS
            if not board.any():
                total_cells = self.board_size * self.board_size
                actions = list(range(total_cells))
                probs = np.full(total_cells, 1.0 / total_cells, dtype=np.float32)
            else:
                # 规则短路：检查强制走法（裸棋局仅保留 连五/堵五）
                forced_action, reason = self._get_forced_move_level(
                    board, current_player, tactical_level
                )
                if forced_action is not None:
                    actions = [forced_action]
                    probs = np.array([1.0])
                else:
                    # MCTS 搜索
                    actions, probs = self.mcts.search(
                        state, board,
                        temperature=temperature,
                        add_noise=True,
                        use_human_knowledge=(None if tactical_level == 0
                                             else False)
                    )

            # 记录 (state, policy, player)
            full_policy = np.zeros(self.board_size * self.board_size, dtype=np.float32)
            for a, p in zip(actions, probs):
                full_policy[a] = p

            steps.append(GameStep(
                state=state.copy(),
                policy=full_policy,
                player=current_player
            ))

            # 采样动作
            action_idx = np.random.choice(len(actions), p=probs)
            action = actions[action_idx]
            x, y = action // self.board_size, action % self.board_size

            # 落子
            board[x, y] = current_player

            # 检查胜利
            if self._check_win(board, x, y):
                winner = current_player
                break

            current_player = 3 - current_player

        return GameResult(
            winner=winner,
            steps=steps,
            total_moves=len(steps),
            game_id=game_id,
            main_player=0  # self-play: 双方都是主模型
        )

    def generate_training_data(self, game_result: GameResult,
                               augment_symmetry: bool = True
                               ) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        将一局对弈结果转换为训练数据

        AlphaZero 训练目标:
          L = (z - v)² - π^T · log(p) + c||θ||²
          其中 z = 终局结果 (+1/0/-1), π = MCTS 策略

        Args:
            game_result: 对弈结果
            augment_symmetry: 是否进行对称增强
        Returns:
            [(state, mcts_policy, value), ...] 训练样本列表
        """
        data = []
        winner = game_result.winner

        for step in game_result.steps:
            # 价值: 从当前玩家视角
            if winner == 0:
                z = self.draw_reward  # 平局 (默认 0; 可设为负值厌恶平局)
            elif winner == step.player:
                z = 1.0  # 赢了
            else:
                z = -1.0  # 输了

            if augment_symmetry:
                # 对称增强: 生成 8 个等价版本
                policy_2d = step.policy.reshape(self.board_size, self.board_size)
                for t in range(8):
                    s_aug, p_aug = SymmetryAugmenter.augment(
                        step.state, policy_2d, transform_idx=t
                    )
                    data.append((s_aug, p_aug.reshape(-1), z))
            else:
                data.append((step.state, step.policy, z))

        return data

    def _build_state(self, board: np.ndarray, current_player: int) -> np.ndarray:
        """构建神经网络输入"""
        state = np.zeros((2, self.board_size, self.board_size), dtype=np.float32)
        state[0] = (board == current_player).astype(np.float32)
        state[1] = (board == (3 - current_player)).astype(np.float32)
        return state

    def _check_win(self, board: np.ndarray, x: int, y: int) -> bool:
        """检查是否获胜"""
        player = board[x, y]
        if player == 0:
            return False
        h, w = board.shape
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dx, dy in directions:
            count = 1
            for sign in [1, -1]:
                nx, ny = x + sign * dx, y + sign * dy
                while 0 <= nx < h and 0 <= ny < w and board[nx, ny] == player:
                    count += 1
                    nx += sign * dx
                    ny += sign * dy
            if count >= self.win_condition:
                return True
        return False

    def _has_five(self, opening_moves: List[Tuple[int, int]]) -> bool:
        """检查随机开局是否已出现连五 (理论概率极低, 防御性检查)"""
        board = np.zeros((self.board_size, self.board_size), dtype=np.int32)
        for i, (x, y) in enumerate(opening_moves):
            board[x, y] = 1 if i % 2 == 0 else 2
            if self._check_win(board, x, y):
                return True
        return False

    def _random_opening(self) -> Optional[List[Tuple[int, int]]]:
        """生成随机开局落子 (无重复, 且不出现连五); 未启用时返回 None"""
        if self.random_open_moves <= 0:
            return None
        total_cells = self.board_size * self.board_size
        n_moves = min(self.random_open_moves, total_cells)
        for _ in range(8):  # 防御性重试, 避免随机出五连
            picked = np.random.choice(total_cells, size=n_moves, replace=False)
            opening = [(int(c) // self.board_size, int(c) % self.board_size)
                       for c in picked]
            if not self._has_five(opening):
                return opening
        return opening


class SelfPlayManager:
    """
    自博弈管理器 — 协调多个 Worker 生成数据

    流程:
    1. 从主模型同步权重
    2. 生成 N 局自博弈数据
    3. 将数据送入 replay buffer
    4. 定期更新对手池
    """

    def __init__(self, model: nn.Module, device: str,
                 board_size: int = 10, win_condition: int = 5,
                 num_simulations: int = 400,
                 augment_symmetry: bool = True,
                 opponent_pool=None,
                 mcts_batch_size: int = 16,
                 cpu_workers: int = 0,
                 fp16: bool = False,
                 use_human_knowledge: bool = False,
                 c_puct: float = 1.5,
                 dirichlet_alpha: float = 0.3,
                 dirichlet_epsilon: float = 0.25,
                 temp_threshold: int = 30,
                 temp_final: float = 0.1,
                 draw_reward: float = 0.0,
                 random_open_moves: int = 0,
                 raw_selfplay_frac: float = 0.0):
        """
        Args:
            model: 当前训练的模型
            device: 推理设备
            board_size: 棋盘大小
            win_condition: 连子数
            num_simulations: MCTS 模拟次数
            augment_symmetry: 是否对称增强 (已废弃, 统一在 ReplayBuffer 中做)
            opponent_pool: 对手池
            mcts_batch_size: MCTS 批量推理大小
            cpu_workers: CPU 并行 worker 数 (0=串行GPU模式, >0=多进程CPU模式)
            fp16: MCTS 推理是否使用 FP16 混合精度
            use_human_knowledge: 是否在搜索树中用人类知识增强 (默认关闭)
            c_puct / dirichlet_alpha / dirichlet_epsilon / temp_threshold /
            temp_final / draw_reward / random_open_moves: 探索与奖励参数,
            透传给 SelfPlayWorker (详见其 docstring)
            raw_selfplay_frac: 裸棋对局比例, 透传给 SelfPlayWorker
        """
        self.model = model
        self.device = device
        self.board_size = board_size
        self.win_condition = win_condition
        self.augment_symmetry = augment_symmetry
        self.cpu_workers = cpu_workers
        self.model_class = None  # 由外部设置 ('small' | 'standard')
        self.fp16 = fp16
        self.opponent_pool = opponent_pool
        self.game_count = 0
        self.use_human_knowledge = use_human_knowledge
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.temp_threshold = temp_threshold
        self.temp_final = temp_final
        self.draw_reward = draw_reward
        self.random_open_moves = random_open_moves
        self.raw_selfplay_frac = max(0.0, min(1.0, raw_selfplay_frac))
        self._iter_games = []   # 本轮生成的每局结果 (winner, main_player)
        self.last_stats = None  # 最近一轮 generate_games 的对局统计
        self._opponent_worker_cache = {}  # model_id → SelfPlayWorker 缓存

        self.worker = SelfPlayWorker(
            model, device, board_size, win_condition,
            num_simulations=num_simulations,
            mcts_batch_size=mcts_batch_size,
            fp16=fp16,
            use_human_knowledge=use_human_knowledge,
            c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            temp_threshold=temp_threshold,
            temp_final=temp_final,
            draw_reward=draw_reward,
            random_open_moves=random_open_moves,
            raw_selfplay_frac=self.raw_selfplay_frac,
        )

    def update_model(self, model: nn.Module):
        """更新 Worker 的模型权重"""
        self.worker.model = model
        self.worker.mcts.model = model

    def generate_games(self, num_games: int) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        生成 N 局自博弈数据

        当 cpu_workers > 0 时使用多进程并行 (CPU workers)
        否则使用串行 GPU 模式
        """
        self._iter_games = []
        if self.cpu_workers > 0 and self.model_class is not None:
            data = self._generate_games_parallel(num_games)
        else:
            data = self._generate_games_sequential(num_games)
        self.last_games = []             # 本轮对局摘要 (胜因打印用)
        self._compute_stats(num_games)
        return data

    def _record_game(self, winner: int, main_player: int, summary=None):
        """记录一局结果用于 TensorBoard 统计与胜因打印

        summary: win_reason.WinGameSummary 或 None (平局/无法重建),
                 供 train_alphazero.py 打印 P1/P2 的胜因。
        """
        self._iter_games.append((winner, main_player, summary))

    def _compute_stats(self, num_games: int) -> dict:
        """
        汇总本轮对局统计, 存入 self.last_stats 供 TensorBoard 使用:
          - tie_rate:              平局占比 (全部对局)
          - selfplay_win_rate:     自博弈局黑方(P1)胜率 (双方同模型, 正常≈50%)
          - main_first_win_rate:   主模型执黑(先手)胜率 (自博弈记黑方, 对手局按 main_player)
          - main_second_win_rate:  主模型执白(后手)胜率
        """
        games = self._iter_games[:num_games]
        n = len(games)
        stats = {
            'games': n, 'ties': 0,
            'selfplay_games': 0, 'selfplay_p1_wins': 0,
            'main_first_games': 0, 'main_first_wins': 0,
            'main_second_games': 0, 'main_second_wins': 0,
        }
        for winner, main_player, _summary in games:
            if winner == 0:
                stats['ties'] += 1
            if main_player == 0:
                # 自博弈: 主模型同时执双方, 黑方=先手, 白方=后手
                stats['selfplay_games'] += 1
                if winner == 1:
                    stats['selfplay_p1_wins'] += 1
                stats['main_first_games'] += 1
                stats['main_second_games'] += 1
                if winner == 1:
                    stats['main_first_wins'] += 1
                elif winner == 2:
                    stats['main_second_wins'] += 1
            elif main_player == 1:
                stats['main_first_games'] += 1
                if winner == 1:
                    stats['main_first_wins'] += 1
            elif main_player == 2:
                stats['main_second_games'] += 1
                if winner == 2:
                    stats['main_second_wins'] += 1

        def rate(wins, total):
            return wins / total if total else 0.0

        stats['tie_rate'] = stats['ties'] / n if n else 0.0
        stats['selfplay_win_rate'] = rate(
            stats['selfplay_p1_wins'], stats['selfplay_games'])
        stats['main_first_win_rate'] = rate(
            stats['main_first_wins'], stats['main_first_games'])
        stats['main_second_win_rate'] = rate(
            stats['main_second_wins'], stats['main_second_games'])
        self.last_games = games          # 供 train_alphazero.py 打印每局胜因
        self._iter_games = []
        self.last_stats = stats
        return stats

    def _generate_games_sequential(self, num_games: int
                                   ) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """串行生成游戏 (GPU 模式, 原有逻辑)"""
        all_data = []

        for game_idx in range(num_games):
            game_start = time.time()

            opponent_model, opponent_id = self._select_opponent()

            if opponent_model is not None:
                game_result = self._play_against_opponent(opponent_model, opponent_id)
                opponent_type = 'history'
            else:
                game_result = self.worker.play_one_game(game_id=self.game_count)
                opponent_type = 'self'
            # 胜因摘要: 仅非平局需要重建 (平局直接跳过)
            summary = reconstruct_game(game_result, self.win_condition)
            self._record_game(game_result.winner, game_result.main_player,
                              summary)

            game_time = time.time() - game_start
            self.game_count += 1

            data = self.worker.generate_training_data(
                game_result, augment_symmetry=False
            )
            all_data.extend(data)

            if game_idx % 5 == 0 or True:
                winner_str = {1: 'P1', 2: 'P2', 0: 'Tie'}.get(game_result.winner, '?')
                mp = game_result.main_player
                if mp == 0:
                    opp_str = opponent_type
                else:
                    opp_str = f'{opponent_type}(M=P{mp})'
                time_per_move = game_time / max(1, game_result.total_moves)
                samples_per_move = len(data) / max(1, game_result.total_moves)
                print(f"    Game {self.game_count:>4}: "
                      f"moves={game_result.total_moves:>3} | "
                      f"winner={winner_str:>4} | "
                      f"opponent={opp_str:>14} | "
                      f"samples={len(data):>5} ({samples_per_move:.0f}/move) | "
                      f"time={game_time:.1f}s ({time_per_move:.2f}s/move)")

            if self.opponent_pool and self.opponent_pool.should_update():
                self.opponent_pool.add_model(
                    self.model,
                    model_id=f'step_{self.game_count}',
                    step=self.game_count
                )

        return all_data

    def _generate_games_parallel(self, num_games: int
                                 ) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """多进程并行生成游戏 (CPU workers 模式)"""
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # 准备模型权重 (序列化到 CPU)
        model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        # 构建每个 game 的配置
        configs = []
        for i in range(num_games):
            game_id = self.game_count + i
            opp_dict = None

            opponent_model, opponent_id = self._select_opponent()
            if opponent_model is not None:
                opp_dict = {k: v.cpu().clone()
                            for k, v in opponent_model.state_dict().items()}

            configs.append({
                'model_class': self.model_class,
                'model_state_dict': model_state,
                'board_size': self.board_size,
                'win_condition': self.win_condition,
                'num_simulations': self.worker.mcts.num_simulations,
                'mcts_batch_size': self.worker.mcts.batch_size,
                'game_id': game_id,
                'num_games': 1,
                'opponent_state_dict': opp_dict,
                'use_human_knowledge': self.use_human_knowledge,
                'c_puct': self.c_puct,
                'dirichlet_alpha': self.dirichlet_alpha,
                'dirichlet_epsilon': self.dirichlet_epsilon,
                'temp_threshold': self.temp_threshold,
                'temp_final': self.temp_final,
                'draw_reward': self.draw_reward,
                'random_open_moves': self.random_open_moves,
                'raw_selfplay_frac': self.raw_selfplay_frac,
            })

        # 并行执行
        all_data = []
        completed_games = 0
        executor = ProcessPoolExecutor(max_workers=self.cpu_workers)
        try:
            futures = {executor.submit(_cpu_self_play_worker, cfg): i
                       for i, cfg in enumerate(configs)}

            for future in as_completed(futures):
                try:
                    data, results = future.result()
                    all_data.extend(data)
                    for winner, main_player, summary in results:
                        self._record_game(winner, main_player, summary)
                    completed_games += 1
                    if completed_games % 5 == 0:
                        print(f"    [Parallel] {completed_games}/{num_games} games done, "
                              f"{len(all_data)} samples collected")
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(f"    [Parallel] Worker failed: {e}")
        except KeyboardInterrupt:
            print("\n    [Parallel] 收到中断信号, 等待 worker 进程退出...")
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            executor.shutdown(wait=False)
            # 确保所有子进程被清理
            import signal as _signal
            try:
                executor._processes
            except Exception:
                pass

        self.game_count += num_games
        print(f"    [Parallel] Batch complete: {num_games} games → "
              f"{len(all_data)} samples ({len(all_data)/max(1,num_games):.0f}/game)")

        return all_data

    def _select_opponent(self) -> Tuple[Optional[nn.Module], str]:
        """选择对手 (自博弈 vs 历史模型)

        Returns:
            (opponent_model, opponent_id): opponent_model 为 None 表示自博弈,
            opponent_id 用于缓存 key (稳定标识, 非 Python id())
        """
        if self.opponent_pool is None:
            return None, ''

        # 50% 概率与历史模型对弈
        if np.random.random() < 0.5 and len(self.opponent_pool.pool) > 0:
            model = self.opponent_pool.sample_opponent(device=self.device)
            opponent_id = self.opponent_pool._last_selected_id
            return model, opponent_id
        return None, ''

    def _play_against_opponent(self, opponent_model: nn.Module,
                               opponent_id: str = '') -> GameResult:
        """与历史模型对弈一局 (复用缓存的 opponent worker)

        使用 opponent_id (稳定标识) 而非 id(opponent_model) 作为缓存 key,
        避免因 sample_opponent 返回新对象导致缓存永远 miss 的显存泄露。

        主模型随机执 P1(先手)或 P2(后手), 确保两种角色都得到训练。
        """
        # 随机决定主模型执先手还是后手 (50%/50%)
        main_player = 1 if np.random.random() < 0.5 else 2

        cache_key = opponent_id or str(id(opponent_model))

        # 限制缓存大小, 清理不活跃的旧条目 (防御性编程)
        if cache_key not in self._opponent_worker_cache:
            if len(self._opponent_worker_cache) > 20:
                # 删除最旧的条目
                oldest_key = next(iter(self._opponent_worker_cache))
                del self._opponent_worker_cache[oldest_key]

        if cache_key not in self._opponent_worker_cache:
            self._opponent_worker_cache[cache_key] = SelfPlayWorker(
                opponent_model, self.device,
                self.board_size, self.win_condition,
                num_simulations=self.worker.mcts.num_simulations,
                mcts_batch_size=self.worker.mcts.batch_size,
                fp16=self.fp16,
                use_human_knowledge=self.use_human_knowledge,
                c_puct=self.c_puct,
                dirichlet_alpha=self.dirichlet_alpha,
                dirichlet_epsilon=self.dirichlet_epsilon,
                temp_threshold=self.temp_threshold,
                temp_final=self.temp_final,
                draw_reward=self.draw_reward,
                random_open_moves=self.random_open_moves,
            )
        opponent_worker = self._opponent_worker_cache[cache_key]

        board = np.zeros((self.board_size, self.board_size), dtype=np.int32)
        steps = []

        # 随机开局 (与 play_one_game 一致)
        opening_moves = self.worker._random_opening()
        start_step = 0
        if opening_moves:
            for i, (x, y) in enumerate(opening_moves):
                player = 1 if i % 2 == 0 else 2
                board[x, y] = player
            start_step = len(opening_moves)

        # 裸棋局: 主模型只保留 连五/堵五, 对手仍全力防守 (让主模型被惩罚)
        tactical_level = 1 if np.random.random() < self.raw_selfplay_frac else 0

        current_player = 1
        if start_step % 2 == 1:
            current_player = 2
        max_steps = self.board_size * self.board_size
        winner = 0

        for step in range(start_step, max_steps):
            state = self.worker._build_state(board, current_player)
            temperature = (1.0 if step < self.worker.temp_threshold
                           else self.worker.temp_final)

            # 先手第一手完全均匀随机落子 (不固定天元); 第二手起走 MCTS
            if not board.any():
                total_cells = self.board_size * self.board_size
                actions = list(range(total_cells))
                probs = np.full(total_cells, 1.0 / total_cells, dtype=np.float32)
            else:
                if current_player == main_player:
                    # 主模型: 遵守本局的战术等级 (裸棋局无防守短路)
                    forced_action, reason = self.worker._get_forced_move_level(
                        board, current_player, tactical_level
                    )
                    if forced_action is not None:
                        actions = [forced_action]
                        probs = np.array([1.0])
                    else:
                        actions, probs = self.worker.mcts.search(
                            state, board, temperature=temperature,
                            add_noise=True,
                            use_human_knowledge=(None if tactical_level == 0
                                                 else False)
                        )
                else:
                    # 对手: 始终全力防守 (规则短路 + 战术先验)
                    forced_action, reason = get_forced_move(
                        board, current_player, self.win_condition
                    )
                    if forced_action is not None:
                        actions = [forced_action]
                        probs = np.array([1.0])
                    else:
                        actions, probs = opponent_worker.mcts.search(
                            state, board, temperature=temperature,
                            add_noise=True
                        )

            # 记录
            full_policy = np.zeros(self.board_size * self.board_size, dtype=np.float32)
            for a, p in zip(actions, probs):
                full_policy[a] = p
            steps.append(GameStep(state=state.copy(), policy=full_policy,
                                player=current_player))

            # 采样动作
            action_idx = np.random.choice(len(actions), p=probs)
            action = actions[action_idx]
            x, y = action // self.board_size, action % self.board_size
            board[x, y] = current_player

            if self.worker._check_win(board, x, y):
                winner = current_player
                break

            current_player = 3 - current_player

        return GameResult(winner=winner, steps=steps, total_moves=len(steps),
                         game_id=self.game_count, main_player=main_player)


# ============================================================
# 多进程自博弈 Worker (模块级函数, Windows spawn 可 pickle)
# ============================================================

def _cpu_self_play_worker(config: dict) -> list:
    """
    CPU-only 自博弈 Worker — 在子进程中运行, 不碰 GPU

    Args:
        config: dict with keys:
            model_class: 'small' | 'standard'
            model_state_dict: serialized state_dict
            board_size, win_condition, num_simulations, mcts_batch_size
            game_id: starting game id
            num_games: how many games to generate (typically 1)
            opponent_state_dict: optional opponent model state_dict
    Returns:
        list of (state, policy, value) tuples (raw, without symmetry augmentation)
    """
    import torch
    torch.set_num_threads(1)  # 每 worker 只用 1 线程, 靠进程级并行

    from model_alphazero import GomokuNetAlphaZero, GomokuNetAlphaZeroSmall

    device = 'cpu'
    board_size = config['board_size']
    win_condition = config['win_condition']
    num_simulations = config['num_simulations']
    # CPU 推理时 batch_size 不宜过大 (GPU 上 256 合理, CPU 上 8-16 最优)
    mcts_batch_size = min(config.get('mcts_batch_size', 16), 16)

    # 创建主模型
    if config['model_class'] == 'small':
        model = GomokuNetAlphaZeroSmall().to(device)
    else:
        model = GomokuNetAlphaZero().to(device)
    model.load_state_dict(config['model_state_dict'])
    model.eval()

    use_human_knowledge = config.get('use_human_knowledge', False)
    c_puct = config.get('c_puct', 1.5)
    dirichlet_alpha = config.get('dirichlet_alpha', 0.3)
    dirichlet_epsilon = config.get('dirichlet_epsilon', 0.25)
    temp_threshold = config.get('temp_threshold', 30)
    temp_final = config.get('temp_final', 0.1)
    draw_reward = config.get('draw_reward', 0.0)
    random_open_moves = config.get('random_open_moves', 0)
    raw_selfplay_frac = config.get('raw_selfplay_frac', 0.0)

    worker = SelfPlayWorker(
        model, device, board_size, win_condition,
        num_simulations=num_simulations,
        mcts_batch_size=mcts_batch_size,
        use_human_knowledge=use_human_knowledge,
        c_puct=c_puct,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
        temp_threshold=temp_threshold,
        temp_final=temp_final,
        draw_reward=draw_reward,
        random_open_moves=random_open_moves,
        raw_selfplay_frac=raw_selfplay_frac,
    )

    all_data = []
    results = []  # [(winner, main_player), ...] 供 TensorBoard 统计
    for i in range(config.get('num_games', 1)):
        game_id = config['game_id'] + i

        # 如果有对手模型, 与对手对弈
        opponent_dict = config.get('opponent_state_dict')
        if opponent_dict is not None:
            if config['model_class'] == 'small':
                opp_model = GomokuNetAlphaZeroSmall().to(device)
            else:
                opp_model = GomokuNetAlphaZero().to(device)
            opp_model.load_state_dict(opponent_dict)
            opp_model.eval()

            opp_worker = SelfPlayWorker(
                opp_model, device, board_size, win_condition,
                num_simulations=num_simulations,
                mcts_batch_size=mcts_batch_size,
                use_human_knowledge=use_human_knowledge,
                c_puct=c_puct,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_epsilon=dirichlet_epsilon,
                temp_threshold=temp_threshold,
                temp_final=temp_final,
                draw_reward=draw_reward,
                random_open_moves=random_open_moves,
                raw_selfplay_frac=raw_selfplay_frac,
            )

            # Run opponent game
            game_result = _run_opponent_game(
                worker, opp_worker, board_size, win_condition, game_id
            )
        else:
            game_result = worker.play_one_game(game_id=game_id)
        summary = reconstruct_game(game_result, win_condition)
        results.append((game_result.winner, game_result.main_player, summary))

        # 转换为训练数据 (不做对称增强)
        data = worker.generate_training_data(game_result, augment_symmetry=False)
        all_data.extend(data)

    return all_data, results


def _run_opponent_game(worker1, worker2, board_size, win_condition, game_id):
    """在 CPU worker 中运行一场主模型 vs 对手模型的对弈

    主模型随机执 P1(先手)或 P2(后手), 确保两种角色都得到训练。
    worker1 = 主模型, worker2 = 对手模型
    """
    import random as _random
    main_player = 1 if _random.random() < 0.5 else 2

    board = np.zeros((board_size, board_size), dtype=np.int32)
    steps = []

    # 随机开局 (与 play_one_game 一致)
    opening_moves = worker1._random_opening()
    start_step = 0
    if opening_moves:
        for i, (x, y) in enumerate(opening_moves):
            player = 1 if i % 2 == 0 else 2
            board[x, y] = player
        start_step = len(opening_moves)

    # 裸棋局: 主模型只保留 连五/堵五, 对手仍全力防守
    tactical_level = 1 if _random.random() < getattr(
        worker1, 'raw_selfplay_frac', 0.0) else 0

    current_player = 2 if start_step % 2 == 1 else 1
    max_steps = board_size * board_size
    winner = 0

    for step in range(start_step, max_steps):
        state = worker1._build_state(board, current_player)
        temperature = (1.0 if step < worker1.temp_threshold
                       else worker1.temp_final)

        # 先手第一手完全均匀随机落子 (不固定天元); 第二手起走 MCTS
        if not board.any():
            total_cells = board_size * board_size
            actions = list(range(total_cells))
            probs = np.full(total_cells, 1.0 / total_cells, dtype=np.float32)
        else:
            if current_player == main_player:
                forced_action, reason = worker1._get_forced_move_level(
                    board, current_player, tactical_level
                )
                if forced_action is not None:
                    actions = [forced_action]
                    probs = np.array([1.0])
                else:
                    actions, probs = worker1.mcts.search(
                        state, board, temperature=temperature, add_noise=True,
                        use_human_knowledge=(None if tactical_level == 0
                                             else False)
                    )
            else:
                # 对手: 始终全力防守
                forced_action, reason = get_forced_move(
                    board, current_player, win_condition
                )
                if forced_action is not None:
                    actions = [forced_action]
                    probs = np.array([1.0])
                else:
                    actions, probs = worker2.mcts.search(
                        state, board, temperature=temperature, add_noise=True
                    )

        full_policy = np.zeros(board_size * board_size, dtype=np.float32)
        for a, p in zip(actions, probs):
            full_policy[a] = p
        steps.append(GameStep(state=state.copy(), policy=full_policy,
                              player=current_player))

        action_idx = np.random.choice(len(actions), p=probs)
        action = actions[action_idx]
        x, y = action // board_size, action % board_size
        board[x, y] = current_player

        if worker1._check_win(board, x, y):
            winner = current_player
            break

        current_player = 3 - current_player

    return GameResult(winner=winner, steps=steps, total_moves=len(steps),
                      game_id=game_id, main_player=main_player)
