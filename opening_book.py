"""
固定开局评估集 — 消除开局随机性，公平比较不同模型

用法:
  每个开局指定前 N 步落子，然后两个模型从这个局面开始对弈
  这样可以:
  1. 消除开局随机性对评估的影响
  2. 测试模型在不同开局下的表现
  3. 发现模型的薄弱开局
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


class OpeningBook:
    """固定开局库"""

    # 标准开局定义 (落子坐标列表, 交替黑白)
    STANDARD_OPENINGS = {
        # === 10x10 棋盘开局 ===
        'center_10': {
            'size': 10,
            'moves': [(5, 5)],
            'desc': '天元开局 (10×10)'
        },
        'diagonal_10': {
            'size': 10,
            'moves': [(5, 5), (6, 6)],
            'desc': '对角开局 (10×10)'
        },
        'parallel_10': {
            'size': 10,
            'moves': [(5, 5), (5, 6)],
            'desc': '平行开局 (10×10)'
        },
        'exchange_10': {
            'size': 10,
            'moves': [(5, 5), (6, 6), (5, 6)],
            'desc': '交换开局 (10×10)'
        },

        # === 15x15 棋盘开局 ===
        'center_15': {
            'size': 15,
            'moves': [(7, 7)],
            'desc': '天元开局 (15×15)'
        },
        'diagonal_15': {
            'size': 15,
            'moves': [(7, 7), (8, 8)],
            'desc': '对角开局 (15×15)'
        },
        'parallel_15': {
            'size': 15,
            'moves': [(7, 7), (7, 8)],
            'desc': '平行开局 (15×15)'
        },
        'indirect_15': {
            'size': 15,
            'moves': [(7, 7), (6, 8)],
            'desc': '间接开局 (15×15)'
        },

        # === 5x5 棋盘开局 ===
        'center_5': {
            'size': 5,
            'moves': [(2, 2)],
            'desc': '天元开局 (5×5)'
        },
    }

    @staticmethod
    def get_opening(name: str) -> Dict:
        """获取指定开局"""
        return OpeningBook.STANDARD_OPENINGS[name]

    @staticmethod
    def get_openings_for_size(board_size: int) -> Dict[str, Dict]:
        """获取指定棋盘大小的所有开局"""
        return {
            name: info for name, info in OpeningBook.STANDARD_OPENINGS.items()
            if info['size'] == board_size
        }

    @staticmethod
    def apply_opening(board: np.ndarray, moves: List[Tuple[int, int]]
                      ) -> Tuple[np.ndarray, int]:
        """
        将开局落子应用到棋盘上

        Args:
            board: (H, W) 空棋盘
            moves: 落子坐标列表 [(x, y), ...]
        Returns:
            board: 应用开局后的棋盘
            current_player: 下一步该谁走 (1 或 2)
        """
        for i, (x, y) in enumerate(moves):
            player = 1 if i % 2 == 0 else 2  # 黑先
            board[x, y] = player
        current_player = 1 if len(moves) % 2 == 0 else 2
        return board, current_player

    @staticmethod
    def evaluate_openings(model_fn, board_size: int,
                          openings: Optional[Dict] = None,
                          games_per_opening: int = 20,
                          opponent_model_fn=None) -> Dict:
        """
        用固定开局评估模型表现

        Args:
            model_fn: 模型推理函数 (state) → (actions, probs)
            board_size: 棋盘大小
            openings: 开局字典 (None 则用该棋盘大小的所有标准开局)
            games_per_opening: 每个开局的对局数
            opponent_model_fn: 对手模型推理函数 (None 则用同一个模型自对弈)
        Returns:
            评估结果字典
        """
        if openings is None:
            openings = OpeningBook.get_openings_for_size(board_size)
        if opponent_model_fn is None:
            opponent_model_fn = model_fn

        results = {}
        for name, info in openings.items():
            if info['size'] != board_size:
                continue

            wins_first = 0
            wins_second = 0
            draws = 0

            for game_idx in range(games_per_opening):
                # 交替先后手
                model_first = (game_idx % 2 == 0)
                winner = OpeningManager._play_opening_game(
                    board_size, info['moves'],
                    model_fn if model_first else opponent_model_fn,
                    opponent_model_fn if model_first else model_fn
                )

                if model_first:
                    if winner == 1:
                        wins_first += 1
                    elif winner == 0:
                        draws += 1
                else:
                    if winner == 2:
                        wins_second += 1
                    elif winner == 0:
                        draws += 1

            total = games_per_opening
            results[name] = {
                'desc': info['desc'],
                'wins_as_first': wins_first,
                'wins_as_second': wins_second,
                'draws': draws,
                'win_rate_first': wins_first / (total // 2),
                'win_rate_second': wins_second / (total // 2),
                'overall_win_rate': (wins_first + wins_second) / total,
            }

        return results


class OpeningManager:
    """开局管理器 — 执行开局对弈"""

    @staticmethod
    def _play_opening_game(board_size: int, opening_moves: List[Tuple[int, int]],
                           player1_fn, player2_fn,
                           max_steps: int = None) -> int:
        """
        从开局开始一局对弈

        Returns:
            1 = player1 赢, 2 = player2 赢, 0 = 平局
        """
        if max_steps is None:
            max_steps = board_size * board_size

        board = np.zeros((board_size, board_size), dtype=np.int32)

        # 应用开局
        for i, (x, y) in enumerate(opening_moves):
            player = 1 if i % 2 == 0 else 2
            board[x, y] = player

        current_player = 1 if len(opening_moves) % 2 == 0 else 2
        step = len(opening_moves)

        while step < max_steps:
            # 构建状态
            state = np.zeros((2, board_size, board_size), dtype=np.float32)
            state[0] = (board == current_player).astype(np.float32)
            state[1] = (board == (3 - current_player)).astype(np.float32)

            # 选择动作
            fn = player1_fn if current_player == 1 else player2_fn
            actions, probs = fn(state)

            # 采样动作
            action_idx = np.random.choice(len(actions), p=probs)
            action = actions[action_idx]
            x, y = action // board_size, action % board_size

            # 落子
            board[x, y] = current_player
            step += 1

            # 检查胜利
            if OpeningManager._check_win(board, x, y):
                return current_player

            # 切换玩家
            current_player = 3 - current_player

        return 0  # 平局

    @staticmethod
    def _check_win(board: np.ndarray, x: int, y: int,
                   win_condition: int = 5) -> bool:
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
            if count >= win_condition:
                return True
        return False
