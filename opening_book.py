"""
固定开局评估集 — 公平比较不同模型的续走表现

规则说明:
  先手(黑方)第一手已改为完全均匀随机落子 (不固定天元),
  因此开局库不再指定第一手, 只定义随机首手之后的固定续走
  (以首手位置为基准的相对偏移, 越界时自动镜像保持棋形)。

用法:
  每个开局指定首手之后的 N 步续走, 模型从该局面开始对弈
  这样可以:
  1. 让评估与训练规则一致 (先手均匀随机)
  2. 测试模型在不同局部棋形下的表现
  3. 发现模型的薄弱开局
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


class OpeningBook:
    """固定开局库"""

    # 标准开局定义 — 只含随机首手之后的续走偏移 (以首手为基准, 交替黑白)
    # 注意: 第一手(黑)完全均匀随机落子, 不再固定天元;
    #       原 'center_*' 开局仅指定天元首手, 已随新规则移除
    STANDARD_OPENINGS = {
        # === 10x10 棋盘开局 ===
        'diagonal_10': {
            'size': 10,
            'offsets': [(1, 1)],
            'desc': '对角续走 (随机首手, 白方斜连)'
        },
        'parallel_10': {
            'size': 10,
            'offsets': [(0, 1)],
            'desc': '平行续走 (随机首手, 白方横连)'
        },
        'exchange_10': {
            'size': 10,
            'offsets': [(1, 1), (0, 1)],
            'desc': '交换续走 (随机首手, 白斜+黑横)'
        },

        # === 15x15 棋盘开局 ===
        'diagonal_15': {
            'size': 15,
            'offsets': [(1, 1)],
            'desc': '对角续走 (随机首手, 白方斜连)'
        },
        'parallel_15': {
            'size': 15,
            'offsets': [(0, 1)],
            'desc': '平行续走 (随机首手, 白方横连)'
        },
        'indirect_15': {
            'size': 15,
            'offsets': [(-1, 1)],
            'desc': '间接续走 (随机首手, 白方斜连)'
        },
    }

    @staticmethod
    def _resolve_offset(fx: int, fy: int, dx: int, dy: int,
                        n: int) -> Tuple[int, int]:
        """相对偏移 → 绝对坐标; 越界时沿该轴镜像, 保持相对棋形"""
        nx, ny = fx + dx, fy + dy
        if nx < 0 or nx >= n:
            nx = fx - dx
        if ny < 0 or ny >= n:
            ny = fy - dy
        return nx, ny

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
    def apply_opening(board: np.ndarray, offsets: List[Tuple[int, int]]
                      ) -> Tuple[np.ndarray, int]:
        """
        从空棋盘开始: 先手(黑)第一手均匀随机, 再应用续走偏移

        Args:
            board: (H, W) 空棋盘
            offsets: 以随机首手为基准的续走偏移列表 [(dx, dy), ...]
        Returns:
            board: 应用开局后的棋盘
            current_player: 下一步该谁走 (1 或 2)
        """
        n = board.shape[0]
        first = np.random.randint(n * n)
        fx, fy = first // n, first % n
        board[fx, fy] = 1  # 黑方随机首手

        for i, (dx, dy) in enumerate(offsets):
            player = 2 if i % 2 == 0 else 1  # 续走: 白黑交替
            x, y = OpeningBook._resolve_offset(fx, fy, dx, dy, n)
            board[x, y] = player
        current_player = 2 if len(offsets) % 2 == 0 else 1
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
            ties = 0

            for game_idx in range(games_per_opening):
                # 交替先后手
                model_first = (game_idx % 2 == 0)
                winner = OpeningManager._play_opening_game(
                    board_size, info['offsets'],
                    model_fn if model_first else opponent_model_fn,
                    opponent_model_fn if model_first else model_fn
                )

                if model_first:
                    if winner == 1:
                        wins_first += 1
                    elif winner == 0:
                        ties += 1
                else:
                    if winner == 2:
                        wins_second += 1
                    elif winner == 0:
                        ties += 1

            total = games_per_opening
            results[name] = {
                'desc': info['desc'],
                'wins_as_first': wins_first,
                'wins_as_second': wins_second,
                'ties': ties,
                'win_rate_first': wins_first / (total // 2),
                'win_rate_second': wins_second / (total // 2),
                'overall_win_rate': (wins_first + wins_second) / total,
            }

        return results


class OpeningManager:
    """开局管理器 — 执行开局对弈"""

    @staticmethod
    def _play_opening_game(board_size: int, opening_offsets: List[Tuple[int, int]],
                           player1_fn, player2_fn,
                           max_steps: int = None) -> int:
        """
        从随机首手 + 固定续走开局开始一局对弈

        Returns:
            1 = player1 赢, 2 = player2 赢, 0 = 平局
        """
        if max_steps is None:
            max_steps = board_size * board_size

        board = np.zeros((board_size, board_size), dtype=np.int32)

        # 先手(黑)第一手完全均匀随机 (不固定天元)
        first = np.random.randint(board_size * board_size)
        fx, fy = first // board_size, first % board_size
        board[fx, fy] = 1

        # 应用续走偏移 (白方第二手起, 黑白交替)
        for i, (dx, dy) in enumerate(opening_offsets):
            player = 2 if i % 2 == 0 else 1
            x, y = OpeningBook._resolve_offset(fx, fy, dx, dy, board_size)
            board[x, y] = player

        current_player = 2 if len(opening_offsets) % 2 == 0 else 1
        step = 1 + len(opening_offsets)

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
