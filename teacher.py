"""
传统 AI 教师接口 — 封装 ./another 中的 Kali-Hac 大师级 AI

用途:
  1. 获取教师策略分布 (软标签) — 用于知识蒸馏
  2. 教师自我对弈 — 生成海量高质量训练数据
  3. 单步最佳走法 — 用于快速评估/对比

教师 AI 说明:
  - 基于棋形模式匹配 (连五/活四/冲四/活三/眠三等 14 种棋形)
  - 评分范围约 -5 (死棋) ~ 100000 (连五)
  - 两种难度: '比你6的Level' (wise scoring) / '和我一样6的Level' (regular scoring)

策略分布生成:
  1. 遍历搜索范围内所有空位
  2. 调用 cal_score_wise 计算每个位置的评分
  3. log 变换压缩动态范围 → 温度缩放 → softmax → 概率分布
"""

import sys
import os
import numpy as np
from typing import List, Tuple, Optional

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_ANOTHER_DIR = os.path.join(_BASE_DIR, 'another')
if _ANOTHER_DIR not in sys.path:
    sys.path.insert(0, _ANOTHER_DIR)

import Global_variables as gv
import Alpha_beta_optimize as ai_player
import Calcu_every_step_score as calcu


class TeacherAI:
    """
    传统大师级 AI 教师

    封装 ./another 中的 Kali-Hac AI, 提供干净的 Python 接口。
    注意: 该 AI 使用全局状态 (Global_variables), 因此同一时间
    只能有一个 TeacherAI 实例在运行 (不可并行)。

    Examples:
        >>> teacher = TeacherAI(board_size=15)
        >>> board = np.zeros((15, 15), dtype=np.int32)
        >>> policy = teacher.get_policy(board, current_player=1, temperature=3.0)
        >>> best_move = teacher.get_move(board)
    """

    def __init__(self, board_size: int = 15,
                 mode: str = '比你6的Level'):
        """
        Args:
            board_size: 棋盘大小 (5~20)
            mode: AI 难度模式
                  '比你6的Level' — 使用 cal_score_wise (更强, 推荐)
                  '和我一样6的Level' — 使用 cal_score (标准)
        """
        self.board_size = board_size
        self.mode = mode
        gv.prepare(board_size)

    # ================================================================
    # 内部: 全局状态同步
    # ================================================================

    def _sync_to_global(self, board: np.ndarray):
        """将 numpy board 同步到 Global_variables 全局状态"""
        n = self.board_size
        for i in range(n):
            for j in range(n):
                v = int(board[i, j])
                if v == 1:
                    gv.black[i][j] = 1
                    gv.white[i][j] = 0
                    gv.flag[i][j] = 1
                elif v == 2:
                    gv.black[i][j] = 0
                    gv.white[i][j] = 1
                    gv.flag[i][j] = 1
                else:
                    gv.black[i][j] = 0
                    gv.white[i][j] = 0
                    gv.flag[i][j] = 0
        ai_player.search_range = ai_player.shrink_range()

    # ================================================================
    # 公开接口
    # ================================================================

    def get_move(self, board: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        获取教师的最佳走法 (快速, 仅返回最优解)

        Args:
            board: (H, W) numpy int32 数组, 1=黑 2=白 0=空
        Returns:
            (x, y) 坐标, 或 None (无合法走法)
        """
        self._sync_to_global(board)
        pos = ai_player.machine_thinking(self.mode)
        return pos if pos else None

    def get_policy(self, board: np.ndarray, current_player: int,
                   temperature: float = 3.0) -> np.ndarray:
        """
        获取教师对当前局面的完整策略分布 (软标签)

        流程:
          1. 遍历搜索范围内所有空位
          2. 对每个候选位置调用 cal_score_wise/cal_score 计算评分
          3. 评分 → log 压缩 → 温度缩放 → softmax → 概率分布

        Args:
            board: (H, W) numpy int32 数组, 1=黑 2=白 0=空
            current_player: 当前落子方 (1=黑/先手, 2=白/后手)
            temperature: 温度系数 T
                         T→0: 尖锐分布 (接近 one-hot, 只学最优解)
                         T=2~4: 软化分布 (保留次级候选的相对优劣, 推荐)
                         T→∞: 均匀分布
        Returns:
            policy: (H*W,) float32 概率分布, sum=1.0
        """
        self._sync_to_global(board)
        n = self.board_size

        color = 'black' if current_player == 1 else 'white'
        scores = []
        positions = []

        for i in range(n):
            for j in range(n):
                if gv.flag[i][j] == 0 and ai_player.search_range[i][j] == 1:
                    # 临时放置当前玩家棋子
                    gv.flag[i][j] = 1
                    if current_player == 1:
                        gv.black[i][j] = 1
                    else:
                        gv.white[i][j] = 1

                    # 评估该位置
                    if self.mode == '比你6的Level':
                        score = calcu.cal_score_wise(color, i, j)
                    else:
                        score = calcu.cal_score(color, i, j)

                    # 恢复
                    gv.flag[i][j] = 0
                    if current_player == 1:
                        gv.black[i][j] = 0
                    else:
                        gv.white[i][j] = 0

                    scores.append(score)
                    positions.append(i * n + j)

        # 构建策略分布
        policy = np.zeros(n * n, dtype=np.float32)

        if not positions:
            # 无合法走法 (棋盘满), 均匀分布
            valid = (board == 0).flatten()
            if valid.sum() > 0:
                policy[valid] = 1.0 / valid.sum()
            return policy

        scores = np.array(scores, dtype=np.float64)

        # --- 评分 → 概率的核心变换 ---
        # 教师评分跨度极大 (100000 ~ -5), 直接 softmax 会完全退化
        # 为 one-hot。此处使用 log 变换压缩动态范围:
        #   s' = log(s - s_min + shift)
        # 使各级棋形之间的相对优劣可被温度参数调控。
        score_min = scores.min()
        # shift: 确保所有值 > 0 (log 定义域), 同时控制压缩程度
        shift = max(1.0, abs(score_min) + 10.0)
        log_scores = np.log(scores - score_min + shift)

        # 温度缩放 + 稳定 softmax
        log_scores = log_scores / max(temperature, 1e-6)
        log_scores = log_scores - log_scores.max()  # 数值稳定
        probs = np.exp(log_scores).astype(np.float64)
        probs = probs / probs.sum()

        for pos, prob in zip(positions, probs):
            policy[pos] = float(prob)

        return policy

    def get_raw_scores(self, board: np.ndarray, current_player: int
                       ) -> Tuple[List[int], List[float]]:
        """
        获取教师对所有候选位置的原始评分 (调试/分析用)

        Returns:
            (positions, scores): 位置列表和对应的原始评分
        """
        self._sync_to_global(board)
        n = self.board_size

        color = 'black' if current_player == 1 else 'white'
        positions = []
        scores = []

        for i in range(n):
            for j in range(n):
                if gv.flag[i][j] == 0 and ai_player.search_range[i][j] == 1:
                    gv.flag[i][j] = 1
                    if current_player == 1:
                        gv.black[i][j] = 1
                    else:
                        gv.white[i][j] = 1

                    if self.mode == '比你6的Level':
                        score = calcu.cal_score_wise(color, i, j)
                    else:
                        score = calcu.cal_score(color, i, j)

                    gv.flag[i][j] = 0
                    if current_player == 1:
                        gv.black[i][j] = 0
                    else:
                        gv.white[i][j] = 0

                    positions.append(i * n + j)
                    scores.append(score)

        return positions, scores


# ================================================================
# 教师自我对弈数据生成器
# ================================================================

def _check_win(board: np.ndarray, x: int, y: int,
               win_condition: int = 5) -> bool:
    """检查 (x,y) 落子后是否连成 win_condition 子"""
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


def generate_distill_games(
    teacher: TeacherAI,
    board_size: int,
    win_condition: int,
    num_games: int,
    policy_temperature: float = 3.0,
    random_open_frac: float = 0.2,
    random_open_moves: Tuple[int, int] = (4, 12),
    verbose: bool = True
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    教师自我对弈 — 生成蒸馏训练数据

    流程:
      1. 80% 对局: 教师 vs 教师, 从空棋盘开始
      2. 20% 对局: 先随机走 N 步, 再由教师接手 (增强泛化性)
      3. 每步记录: (state, teacher_policy, game_outcome)

    产出格式与 SelfPlayWorker.generate_training_data 对齐,
    可直接喂入 ReplayBuffer 和训练循环。

    Args:
        teacher: TeacherAI 实例
        board_size: 棋盘大小
        win_condition: 连子数
        num_games: 生成对局数
        policy_temperature: 教师策略温度 (控制软标签的平滑度)
        random_open_frac: 随机开局比例 (0~1)
        random_open_moves: 随机开局步数范围 (min, max)
        verbose: 是否打印进度

    Returns:
        [(state, policy, value), ...] 训练样本列表
        state:  (2, H, W) float32
        policy: (H*W,) float32
        value:  float (+1/-1/0)
    """
    import random

    all_data = []
    n = board_size
    total_cells = n * n

    for game_idx in range(num_games):
        board = np.zeros((n, n), dtype=np.int32)
        game_steps = []  # [(state, policy, player), ...]

        # --- 随机开局 (20% 概率, 暴露多样化局面) ---
        if random.random() < random_open_frac:
            num_random = random.randint(*random_open_moves)
            for step in range(num_random):
                empty = [(i, j) for i in range(n) for j in range(n)
                         if board[i, j] == 0]
                if not empty:
                    break
                x, y = random.choice(empty)
                board[x, y] = 1 if step % 2 == 0 else 2
                if _check_win(board, x, y, win_condition):
                    break  # 随机阶段有人碰巧赢了就结束

        # --- 教师接手对弈 ---
        current_player = 1
        # 根据已有棋子推断当前玩家
        p1_count = (board == 1).sum()
        p2_count = (board == 2).sum()
        current_player = 1 if p1_count == p2_count else 2

        max_steps = total_cells
        winner = 0

        for step in range(max_steps):
            # 构建状态 (channel 0 = 当前玩家, channel 1 = 对手)
            state = np.zeros((2, n, n), dtype=np.float32)
            state[0] = (board == current_player).astype(np.float32)
            state[1] = (board == (3 - current_player)).astype(np.float32)

            # 获取教师策略 (对当前局面)
            policy = teacher.get_policy(board, current_player,
                                        temperature=policy_temperature)

            # 记录
            game_steps.append((state.copy(), policy, current_player))

            # 按策略采样落子
            valid_mask = (board.flatten() == 0)
            # 只在合法动作上采样
            sample_probs = policy.copy()
            sample_probs[~valid_mask] = 0
            if sample_probs.sum() > 0:
                sample_probs = sample_probs / sample_probs.sum()
                action = np.random.choice(total_cells, p=sample_probs)
            else:
                # fallback: 随机走
                valid_indices = np.where(valid_mask)[0]
                action = np.random.choice(valid_indices)

            x, y = action // n, action % n
            board[x, y] = current_player

            if _check_win(board, x, y, win_condition):
                winner = current_player
                break

            current_player = 3 - current_player

        # --- 标注价值 (从当前玩家视角) ---
        for state, policy, player in game_steps:
            if winner == 0:
                z = 0.0
            elif winner == player:
                z = 1.0
            else:
                z = -1.0
            all_data.append((state, policy, z))

        if verbose and (game_idx + 1) % 100 == 0:
            moves = len(game_steps)
            w_str = {1: 'P1', 2: 'P2', 0: 'Tie'}[winner]
            print(f"  [Distill] Game {game_idx + 1:>5}/{num_games}: "
                  f"moves={moves:>3}, winner={w_str:>4}, "
                  f"samples={len(all_data):>8}")

    if verbose:
        print(f"  [Distill] 完成: {num_games} 局 → {len(all_data)} 样本 "
              f"({len(all_data) / max(1, num_games):.0f}/局)")

    return all_data


# ================================================================
# 便捷函数: 评估教师策略与网络策略的一致性
# ================================================================

def evaluate_policy_accuracy(
    teacher: TeacherAI,
    board: np.ndarray,
    current_player: int,
    network_policy: np.ndarray,
    top_k: Tuple[int, ...] = (1, 3, 5)
) -> dict:
    """
    评估网络策略与教师策略的一致性

    Args:
        teacher: TeacherAI 实例
        board: 棋盘状态
        current_player: 当前玩家
        network_policy: (H*W,) 网络输出的策略分布
        top_k: 需要统计的 Top-K 准确率

    Returns:
        dict: {'top1': float, 'top3': float, 'top5': float}
    """
    teacher_policy = teacher.get_policy(board, current_player)
    teacher_best = np.argmax(teacher_policy)
    net_best_k = np.argsort(network_policy)[::-1]

    result = {}
    for k in top_k:
        result[f'top{k}'] = float(teacher_best in net_best_k[:k])

    return result


if __name__ == '__main__':
    # 快速测试
    print("=" * 50)
    print("TeacherAI 测试")
    print("=" * 50)

    board_size = 8
    teacher = TeacherAI(board_size=board_size)

    # 测试空棋盘策略
    board = np.zeros((board_size, board_size), dtype=np.int32)
    policy = teacher.get_policy(board, current_player=1, temperature=3.0)
    top5 = np.argsort(policy)[::-1][:5]
    print(f"\n空棋盘 Top-5 动作 (T=3.0):")
    for rank, action in enumerate(top5, 1):
        x, y = action // board_size, action % board_size
        print(f"  #{rank}: ({x},{y}) p={policy[action]:.4f}")

    # 测试数据生成
    print(f"\n生成 10 局蒸馏数据...")
    data = generate_distill_games(
        teacher, board_size=board_size, win_condition=5,
        num_games=10, policy_temperature=3.0, random_open_frac=0.2
    )
    print(f"总样本数: {len(data)}")
    if data:
        s, p, v = data[0]
        print(f"单样本: state={s.shape}, policy={p.shape}, value={v}")
