"""
Arena — 模型对战评估系统

用法:
  两个模型对弈 N 局，统计胜率
  支持:
  - 固定开局评估
  - 随机开局评估
  - 双向对战 (各先手一半)
  - Elo 更新
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Optional, Tuple, Dict, List
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

from elo import EloRating
from opening_book import OpeningBook


class Arena:
    """模型对战评估"""

    def __init__(self, board_size: int = 10, win_condition: int = 5):
        self.board_size = board_size
        self.win_condition = win_condition

    def play_game(self, player1_fn: Callable, player2_fn: Callable,
                  opening_moves: Optional[List[Tuple[int, int]]] = None,
                  temperature: float = 0.3) -> int:
        """
        对弈一局

        Args:
            player1_fn: player1 的动作选择函数 (state) → (actions, probs)
            player2_fn: player2 的动作选择函数
            opening_moves: 可选的开局落子
            temperature: 温度参数
        Returns:
            1 = player1 赢, 2 = player2 赢, 0 = 平局
        """
        board = np.zeros((self.board_size, self.board_size), dtype=np.int32)

        # 应用开局
        start_step = 0
        if opening_moves:
            for i, (x, y) in enumerate(opening_moves):
                player = 1 if i % 2 == 0 else 2
                board[x, y] = player
            start_step = len(opening_moves)

        current_player = 1 if start_step % 2 == 0 else 2
        max_steps = self.board_size * self.board_size

        for step in range(start_step, max_steps):
            # 构建状态
            state = self._build_state(board, current_player)

            # 选择动作
            fn = player1_fn if current_player == 1 else player2_fn
            actions, probs = fn(state)

            # 采样 (低温度 ≈ 贪心)
            if temperature < 0.01:
                action_idx = np.argmax(probs)
            else:
                # 温度采样
                log_probs = np.log(probs + 1e-10) / temperature
                log_probs -= log_probs.max()
                sample_probs = np.exp(log_probs)
                sample_probs /= sample_probs.sum()
                action_idx = np.random.choice(len(actions), p=sample_probs)

            action = actions[action_idx]
            x, y = action // self.board_size, action % self.board_size

            # 落子
            board[x, y] = current_player

            # 检查胜利
            if self._check_win(board, x, y):
                return current_player

            current_player = 3 - current_player

        return 0  # 平局

    def play_match(self, player1_fn: Callable, player2_fn: Callable,
                   num_games: int = 100, use_openings: bool = True,
                   temperature: float = 0.3) -> Dict:
        """
        对弈 N 局 (双向各半)

        Args:
            player1_fn: player1 动作函数
            player2_fn: player2 动作函数
            num_games: 总对局数
            use_openings: 是否使用固定开局
            temperature: 温度参数
        Returns:
            统计结果字典
        """
        openings = None
        if use_openings:
            openings = OpeningBook.get_openings_for_size(self.board_size)
            openings = list(openings.values()) if openings else None

        p1_wins = 0
        p2_wins = 0
        draws = 0
        p1_first_wins = 0
        p1_second_wins = 0

        for game_idx in range(num_games):
            # 选择开局
            opening = None
            if openings:
                opening = openings[game_idx % len(openings)]['moves']

            # 交替先后手
            if game_idx % 2 == 0:
                # player1 先手
                winner = self.play_game(player1_fn, player2_fn, opening, temperature)
                if winner == 1:
                    p1_wins += 1
                    p1_first_wins += 1
                elif winner == 2:
                    p2_wins += 1
                else:
                    draws += 1
            else:
                # player2 先手
                winner = self.play_game(player2_fn, player1_fn, opening, temperature)
                if winner == 2:
                    p1_wins += 1
                    p1_second_wins += 1
                elif winner == 1:
                    p2_wins += 1
                else:
                    draws += 1

        total = num_games
        return {
            'total_games': total,
            'p1_wins': p1_wins,
            'p2_wins': p2_wins,
            'draws': draws,
            'p1_win_rate': p1_wins / total if total > 0 else 0,
            'p2_win_rate': p2_wins / total if total > 0 else 0,
            'draw_rate': draws / total if total > 0 else 0,
            'p1_first_win_rate': p1_first_wins / (total // 2) if total > 1 else 0,
            'p1_second_win_rate': p1_second_wins / (total // 2) if total > 1 else 0,
        }

    def _build_state(self, board: np.ndarray, current_player: int) -> np.ndarray:
        """构建神经网络输入状态"""
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


def evaluate_model_strength(model_new, model_baseline, board_size: int,
                           num_games: int = 200,
                           device: str = 'cuda') -> Dict:
    """
    评估新模型相对于基线模型的强度

    Args:
        model_new: 新模型
        model_baseline: 基线模型
        board_size: 棋盘大小
        num_games: 对局数
        device: 推理设备
    Returns:
        评估结果
    """
    from mcts import MCTS

    mcts_new = MCTS(model_new, device, num_simulations=200)
    mcts_baseline = MCTS(model_baseline, device, num_simulations=200)

    def player_new_fn(state):
        board = np.zeros((board_size, board_size), dtype=np.int32)
        board[state[0] == 1] = 1
        board[state[1] == 1] = 2
        return mcts_new.search(state, board, temperature=0.3, add_noise=False)

    def player_baseline_fn(state):
        board = np.zeros((board_size, board_size), dtype=np.int32)
        board[state[0] == 1] = 1
        board[state[1] == 1] = 2
        return mcts_baseline.search(state, board, temperature=0.3, add_noise=False)

    arena = Arena(board_size)
    result = arena.play_match(player_new_fn, player_baseline_fn,
                              num_games=num_games, temperature=0.3)

    print(f"\n评估结果 ({num_games} 局):")
    print(f"  新模型胜率: {result['p1_win_rate']:.1%}")
    print(f"  基线胜率:   {result['p2_win_rate']:.1%}")
    print(f"  平局率:     {result['draw_rate']:.1%}")
    print(f"  新模型先手胜率: {result['p1_first_win_rate']:.1%}")
    print(f"  新模型后手胜率: {result['p1_second_win_rate']:.1%}")

    return result


if __name__ == '__main__':
    # 测试: 两个随机策略对弈
    board_size = 10
    arena = Arena(board_size)

    def random_player(state):
        board_size = state.shape[1]
        valid = []
        for i in range(board_size):
            for j in range(board_size):
                if state[0, i, j] == 0 and state[1, i, j] == 0:
                    valid.append(i * board_size + j)
        probs = np.ones(len(valid)) / len(valid)
        return valid, probs

    result = arena.play_match(random_player, random_player, num_games=20)
    print(f"随机 vs 随机: P1={result['p1_win_rate']:.1%}, "
          f"P2={result['p2_win_rate']:.1%}, 平={result['draw_rate']:.1%}")
