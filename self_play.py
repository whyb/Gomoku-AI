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
                 temp_threshold: int = 30,
                 use_batch_mcts: bool = True,
                 mcts_batch_size: int = 16,
                 fp16: bool = False):
        """
        Args:
            model: 神经网络模型
            device: 推理设备
            board_size: 棋盘大小
            win_condition: 连子数
            num_simulations: MCTS 模拟次数
            c_puct: PUCT 探索常数
            dirichlet_alpha: Dirichlet 噪声 alpha
            temp_threshold: 温度退火阈值 (前 N 步 τ=1.0, 之后 τ=0.1)
            use_batch_mcts: 是否使用批量 MCTS
            mcts_batch_size: 批量 MCTS 的 batch 大小
            fp16: MCTS 推理是否使用 FP16 混合精度
        """
        self.model = model
        self.device = device
        self.board_size = board_size
        self.win_condition = win_condition
        self.temp_threshold = temp_threshold

        if use_batch_mcts:
            self.mcts = BatchMCTS(
                model, device,
                num_simulations=num_simulations,
                c_puct=c_puct,
                dirichlet_alpha=dirichlet_alpha,
                batch_size=mcts_batch_size,
                fp16=fp16
            )
        else:
            self.mcts = MCTS(
                model, device,
                num_simulations=num_simulations,
                c_puct=c_puct,
                dirichlet_alpha=dirichlet_alpha,
                fp16=fp16
            )

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

        # 应用开局
        start_step = 0
        if opening_moves:
            for i, (x, y) in enumerate(opening_moves):
                player = 1 if i % 2 == 0 else 2
                board[x, y] = player
            start_step = len(opening_moves)

        current_player = 1 if start_step % 2 == 0 else 2
        max_steps = self.board_size * self.board_size
        winner = 0

        for step in range(start_step, max_steps):
            # 构建状态
            state = self._build_state(board, current_player)

            # 温度退火
            temperature = 1.0 if step < self.temp_threshold else 0.1

            # 黑方(P1)第一步必须落子天元(正中心) — 标准五子棋/连珠规则
            if not board.any() and current_player == 1:
                center = self.board_size // 2
                action = center * self.board_size + center
                actions = [action]
                probs = np.array([1.0])
            else:
                # 规则短路：检查强制走法（立即获胜 / 必须防守）
                forced_action, reason = get_forced_move(
                    board, current_player, self.win_condition
                )
                if forced_action is not None:
                    actions = [forced_action]
                    probs = np.array([1.0])
                else:
                    # MCTS 搜索
                    actions, probs = self.mcts.search(
                        state, board,
                        temperature=temperature,
                        add_noise=True
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
                z = 0.0  # 平局
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
                 fp16: bool = False):
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
        self._opponent_worker_cache = {}  # model_id → SelfPlayWorker 缓存

        self.worker = SelfPlayWorker(
            model, device, board_size, win_condition,
            num_simulations=num_simulations,
            mcts_batch_size=mcts_batch_size,
            fp16=fp16
        )

    def update_model(self, model: nn.Module):
        """更新 Worker 的模型权重"""
        self.worker.model = model
        self.worker.mcts.model = model

    def generate_games(self, num_games: int) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        生成 N 局自博弈数据

        当 cpu_workers > 0 时使用多进程并行 (CPU workers)
        当 cpu_workers == 0 时使用串行 GPU 模式
        """
        if self.cpu_workers > 0 and self.model_class is not None:
            return self._generate_games_parallel(num_games)
        return self._generate_games_sequential(num_games)

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

            game_time = time.time() - game_start
            self.game_count += 1

            data = self.worker.generate_training_data(
                game_result, augment_symmetry=False
            )
            all_data.extend(data)

            if game_idx % 5 == 0:
                winner_str = {1: 'P1', 2: 'P2', 0: 'Draw'}.get(game_result.winner, '?')
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
                    data = future.result()
                    all_data.extend(data)
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
                fp16=self.fp16
            )
        opponent_worker = self._opponent_worker_cache[cache_key]

        board = np.zeros((self.board_size, self.board_size), dtype=np.int32)
        steps = []
        current_player = 1
        max_steps = self.board_size * self.board_size
        winner = 0

        for step in range(max_steps):
            state = self.worker._build_state(board, current_player)
            temperature = 1.0 if step < self.worker.temp_threshold else 0.1

            # 黑方(P1)第一步必须落子天元(正中心) — 标准五子棋/连珠规则
            if not board.any() and current_player == 1:
                center = self.board_size // 2
                action = center * self.board_size + center
                actions = [action]
                probs = np.array([1.0])
            else:
                # 规则短路：检查强制走法
                forced_action, reason = get_forced_move(
                    board, current_player, self.win_condition
                )
                if forced_action is not None:
                    actions = [forced_action]
                    probs = np.array([1.0])
                # 选择当前玩家的 MCTS (主模型可能执 P1 或 P2)
                elif current_player == main_player:
                    actions, probs = self.worker.mcts.search(
                        state, board, temperature=temperature, add_noise=True
                    )
                else:
                    actions, probs = opponent_worker.mcts.search(
                        state, board, temperature=temperature, add_noise=True
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

    worker = SelfPlayWorker(
        model, device, board_size, win_condition,
        num_simulations=num_simulations,
        mcts_batch_size=mcts_batch_size
    )

    all_data = []
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
                mcts_batch_size=mcts_batch_size
            )

            # Run opponent game
            game_result = _run_opponent_game(
                worker, opp_worker, board_size, win_condition, game_id
            )
        else:
            game_result = worker.play_one_game(game_id=game_id)

        # 转换为训练数据 (不做对称增强)
        data = worker.generate_training_data(game_result, augment_symmetry=False)
        all_data.extend(data)

    return all_data


def _run_opponent_game(worker1, worker2, board_size, win_condition, game_id):
    """在 CPU worker 中运行一场主模型 vs 对手模型的对弈

    主模型随机执 P1(先手)或 P2(后手), 确保两种角色都得到训练。
    worker1 = 主模型, worker2 = 对手模型
    """
    import random as _random
    main_player = 1 if _random.random() < 0.5 else 2

    board = np.zeros((board_size, board_size), dtype=np.int32)
    steps = []
    current_player = 1
    max_steps = board_size * board_size
    winner = 0

    for step in range(max_steps):
        state = worker1._build_state(board, current_player)
        temperature = 1.0 if step < worker1.temp_threshold else 0.1

        # 黑方(P1)第一步必须落子天元(正中心) — 标准五子棋/连珠规则
        if not board.any() and current_player == 1:
            center = board_size // 2
            action = center * board_size + center
            actions = [action]
            probs = np.array([1.0])
        else:
            # 规则短路：检查强制走法
            forced_action, reason = get_forced_move(
                board, current_player, win_condition
            )
            if forced_action is not None:
                actions = [forced_action]
                probs = np.array([1.0])
            elif current_player == main_player:
                actions, probs = worker1.mcts.search(
                    state, board, temperature=temperature, add_noise=True
                )
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
