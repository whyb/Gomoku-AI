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

from mcts import MCTS, BatchMCTS
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
                 mcts_batch_size: int = 16):
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
                batch_size=mcts_batch_size
            )
        else:
            self.mcts = MCTS(
                model, device,
                num_simulations=num_simulations,
                c_puct=c_puct,
                dirichlet_alpha=dirichlet_alpha
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
            game_id=game_id
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
                 mcts_batch_size: int = 16):
        """
        Args:
            model: 当前训练的模型
            device: 推理设备
            board_size: 棋盘大小
            win_condition: 连子数
            num_simulations: MCTS 模拟次数
            augment_symmetry: 是否对称增强
            opponent_pool: 对手池 (可选, 用于与历史模型对弈)
            mcts_batch_size: MCTS 批量推理大小
        """
        self.model = model
        self.device = device
        self.board_size = board_size
        self.win_condition = win_condition
        self.augment_symmetry = augment_symmetry
        self.opponent_pool = opponent_pool
        self.game_count = 0

        self.worker = SelfPlayWorker(
            model, device, board_size, win_condition,
            num_simulations=num_simulations,
            mcts_batch_size=mcts_batch_size
        )

    def update_model(self, model: nn.Module):
        """更新 Worker 的模型权重"""
        self.worker.model = model
        self.worker.mcts.model = model

    def generate_games(self, num_games: int) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        生成 N 局自博弈数据

        Returns:
            所有训练样本的列表 [(state, policy, value), ...]
        """
        all_data = []

        for game_idx in range(num_games):
            game_start = time.time()

            # 决定对手类型
            opponent_model = self._select_opponent()

            if opponent_model is not None:
                # 与历史模型对弈
                game_result = self._play_against_opponent(opponent_model)
                opponent_type = 'history'
            else:
                # 自博弈 (自己 vs 自己)
                game_result = self.worker.play_one_game(game_id=self.game_count)
                opponent_type = 'self'

            game_time = time.time() - game_start
            self.game_count += 1

            # 转换为训练数据
            data = self.worker.generate_training_data(
                game_result,
                augment_symmetry=self.augment_symmetry
            )
            all_data.extend(data)

            # 打印每局摘要 (每 5 局打印一次, 避免刷屏)
            if game_idx % 5 == 0:
                winner_str = {1: 'P1', 2: 'P2', 0: 'Draw'}.get(game_result.winner, '?')
                time_per_move = game_time / max(1, game_result.total_moves)
                samples_per_move = len(data) / max(1, game_result.total_moves)
                print(f"    Game {self.game_count:>4}: "
                      f"moves={game_result.total_moves:>3} | "
                      f"winner={winner_str:>4} | "
                      f"opponent={opponent_type:>6} | "
                      f"samples={len(data):>5} ({samples_per_move:.0f}/move) | "
                      f"time={game_time:.1f}s ({time_per_move:.2f}s/move)")

            # 检查是否需要更新对手池
            if self.opponent_pool and self.opponent_pool.should_update():
                self.opponent_pool.add_model(
                    self.model,
                    model_id=f'step_{self.game_count}',
                    step=self.game_count
                )

        return all_data

    def _select_opponent(self) -> Optional[nn.Module]:
        """选择对手 (自博弈 vs 历史模型)"""
        if self.opponent_pool is None:
            return None

        # 50% 概率与历史模型对弈
        if np.random.random() < 0.5 and len(self.opponent_pool.pool) > 0:
            return self.opponent_pool.sample_opponent(device=self.device)
        return None

    def _play_against_opponent(self, opponent_model: nn.Module) -> GameResult:
        """与历史模型对弈一局"""
        opponent_worker = SelfPlayWorker(
            opponent_model, self.device,
            self.board_size, self.win_condition,
            num_simulations=self.worker.mcts.num_simulations
        )

        board = np.zeros((self.board_size, self.board_size), dtype=np.int32)
        steps = []
        current_player = 1
        max_steps = self.board_size * self.board_size
        winner = 0

        for step in range(max_steps):
            state = self.worker._build_state(board, current_player)
            temperature = 1.0 if step < self.worker.temp_threshold else 0.1

            # 选择当前玩家的 MCTS
            if current_player == 1:
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
                         game_id=self.game_count)
