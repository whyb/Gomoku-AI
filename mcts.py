"""
MCTS (Monte Carlo Tree Search) — AlphaZero 风格

核心流程:
  1. SELECT:  沿 PUCT 公式选择到叶节点
  2. EXPAND:  用神经网络评估叶节点，得到 (policy, value)
  3. EVALUATE: 如果叶节点是终局，直接返回胜负结果
  4. BACKUP:  将价值沿路径回传，更新 Q/N

PUCT(s,a) = Q(s,a) + c_puct × P(s,a) × √(N(s)) / (1 + N(s,a))
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ============================================================
# 五子棋规则短路 & 候选点过滤辅助函数
# ============================================================

def _check_win_static(board: np.ndarray, x: int, y: int,
                      win_condition: int = 5) -> bool:
    """检查 (x,y) 落子后是否连成五子"""
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


def _find_immediate_win(board: np.ndarray, player: int) -> Optional[int]:
    """遍历所有空位，查找 player 是否有立即获胜的走法"""
    h, w = board.shape
    for x in range(h):
        for y in range(w):
            if board[x, y] == 0:
                board[x, y] = player
                win = _check_win_static(board, x, y)
                board[x, y] = 0
                if win:
                    return x * w + y
    return None


def _find_opponent_threats(board: np.ndarray, player: int,
                           win_condition: int = 5) -> Dict[str, set]:
    """
    查找对手（player 的对方）已有的活四、冲四威胁点。
    返回: {'live4': set(actions), 'rush4': set(actions)}
    """
    h, w = board.shape
    opponent = 3 - player
    live4 = set()
    rush4 = set()
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    for x in range(h):
        for y in range(w):
            if board[x, y] != opponent:
                continue
            for dx, dy in directions:
                # 找到该方向连续 opponent 子的起点
                sx, sy = x, y
                while (0 <= sx - dx < h and 0 <= sy - dy < w and
                       board[sx - dx, sy - dy] == opponent):
                    sx -= dx
                    sy -= dy

                # 统计连续 opponent 子数
                count = 0
                ex, ey = sx, sy
                while 0 <= ex < h and 0 <= ey < w and board[ex, ey] == opponent:
                    count += 1
                    ex += dx
                    ey += dy

                if count != win_condition - 1:   # 只关心恰好 4 子
                    continue

                # 检查两端是否为空
                left_x, left_y = sx - dx, sy - dy
                right_x, right_y = ex, ey

                left_empty = (0 <= left_x < h and 0 <= left_y < w and
                              board[left_x, left_y] == 0)
                right_empty = (0 <= right_x < h and 0 <= right_y < w and
                               board[right_x, right_y] == 0)

                if left_empty and right_empty:
                    live4.add(left_x * w + left_y)
                    live4.add(right_x * w + right_y)
                elif left_empty:
                    rush4.add(left_x * w + left_y)
                elif right_empty:
                    rush4.add(right_x * w + right_y)

    return {'live4': live4, 'rush4': rush4}


def get_forced_move(board: np.ndarray, player: int,
                    win_condition: int = 5) -> Tuple[Optional[int], Optional[str]]:
    """
    查找强制走法（规则短路），优先级：
      1. 自己能连五 → 直接获胜
      2. 对手有活四 → 必须堵（若只有一个堵点）
      3. 对手有冲四 → 必须堵
      4. 对手有连五点 → 必须堵（兜底）
    返回: (action, reason) 或 (None, None)
    """
    h, w = board.shape

    # 1. 自己立即获胜
    win_action = _find_immediate_win(board, player)
    if win_action is not None:
        return win_action, 'win'

    # 2. 对手已有连五点（理论上前一步就该发现，兜底）
    opponent = 3 - player
    must_block = set()
    for x in range(h):
        for y in range(w):
            if board[x, y] == 0:
                board[x, y] = opponent
                win = _check_win_static(board, x, y, win_condition)
                board[x, y] = 0
                if win:
                    must_block.add(x * w + y)
    if len(must_block) == 1:
        return must_block.pop(), 'block_win'

    # 3. 对手活四 / 冲四
    threats = _find_opponent_threats(board, player, win_condition)

    live4_blocks = threats['live4']
    if len(live4_blocks) == 1:
        return live4_blocks.pop(), 'block_live4'
    # 若对手有多个活四，已无法防守，交给 MCTS 决定

    rush4_blocks = threats['rush4']
    if len(rush4_blocks) >= 1:
        return next(iter(rush4_blocks)), 'block_rush4'

    return None, None


def get_candidate_mask(board: np.ndarray, radius: int = 2) -> np.ndarray:
    """
    获取候选落子点掩码（bool 数组，flattened）。
    只考虑已有棋子周围 radius 格内的空点。
    空棋盘时返回中心 3x3 区域。
    """
    h, w = board.shape
    if not board.any():
        mask = np.zeros((h, w), dtype=bool)
        cx, cy = h // 2, w // 2
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < h and 0 <= ny < w:
                    mask[nx, ny] = True
        return mask.reshape(-1)

    mask = np.zeros((h, w), dtype=bool)
    occupied = np.argwhere(board != 0)
    for x, y in occupied:
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w and board[nx, ny] == 0:
                    mask[nx, ny] = True
    return mask.reshape(-1)


class MCTSNode:
    """MCTS 树节点"""

    __slots__ = [
        'parent', 'action', 'children', 'N', 'W', 'Q', 'P',
        'is_expanded', 'current_player'
    ]

    def __init__(self, prior: float, parent=None, action: int = -1):
        self.parent = parent
        self.action = action       # 到达此节点的动作
        self.children: Dict[int, 'MCTSNode'] = {}

        self.N = 0                 # 访问次数
        self.W = 0.0               # 累计价值
        self.Q = 0.0               # 平均价值 = W / N
        self.P = prior             # 先验概率 (来自策略网络)
        self.is_expanded = False
        self.current_player = 1    # 此节点的当前玩家

    def select_child_puct(self, c_puct: float) -> Tuple[int, 'MCTSNode']:
        """PUCT 选择: 选择 UCB 最高的子节点"""
        best_score = -float('inf')
        best_action = -1
        best_child = None

        sqrt_parent_n = math.sqrt(self.N + 1)

        for action, child in self.children.items():
            # Q 值从当前玩家视角 (对手的 Q 需要取反)
            q_value = -child.Q  # 子节点是对手走的，所以取反
            u_value = c_puct * child.P * sqrt_parent_n / (1 + child.N)
            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, policy: np.ndarray, valid_actions: np.ndarray,
               current_player: int):
        """
        扩展节点: 根据策略网络的输出创建子节点

        Args:
            policy: 策略网络输出的 logits (board_size² 维)
            valid_actions: 合法动作的 bool 数组
            current_player: 当前玩家 (1 或 2)
        """
        self.current_player = current_player
        self.is_expanded = True

        # 对非法动作 mask 后 softmax
        masked_policy = policy.copy()
        masked_policy[~valid_actions] = -float('inf')
        policy_probs = F.softmax(
            torch.tensor(masked_policy, dtype=torch.float32), dim=0
        ).numpy()

        # 仅为合法动作创建子节点
        for action in np.where(valid_actions)[0]:
            self.children[action] = MCTSNode(
                prior=policy_probs[action],
                parent=self,
                action=action
            )

    def backup(self, value: float):
        """
        回传价值: 从当前节点向上更新到根节点

        Args:
            value: 从当前玩家视角的价值 ([-1, 1])
        """
        node = self
        while node is not None:
            node.N += 1
            node.W += value
            node.Q = node.W / node.N
            # 每层交替: 当前玩家的价值对父节点来说是对手的价值
            value = -value
            node = node.parent


class MCTS:
    """
    AlphaZero 风格 MCTS

    与纯 ε-greedy 的区别:
    - ε-greedy: 每步只看当前局面，1 次 NN 推理
    - MCTS: 每步模拟 800 次未来局面，结合 NN 的 policy + value
    - MCTS 的策略质量远高于纯 NN 输出
    """

    def __init__(self, model, device, num_simulations: int = 800,
                 c_puct: float = 1.5, dirichlet_alpha: float = 0.3,
                 dirichlet_epsilon: float = 0.25, fp16: bool = False,
                 use_candidate_mask: bool = True, candidate_radius: int = 2,
                 early_stop_threshold: float = 0.95, early_stop_min_sims: int = 50):
        """
        Args:
            model: 神经网络模型 (输出 policy logits, value)
            device: 推理设备
            num_simulations: 每步搜索模拟次数 (AlphaZero 用 800)
            c_puct: PUCT 探索常数 (越大越偏探索)
            dirichlet_alpha: Dirichlet 噪声 alpha 参数
            dirichlet_epsilon: 噪声混合比例
            fp16: 是否用 FP16 混合精度推理 (仅前向, 无梯度)
            use_candidate_mask: 是否只搜索已有棋子周围的候选点
            candidate_radius: 候选点半径 (默认 2 格)
            early_stop_threshold: 最佳动作访问占比超过此值时提前终止
            early_stop_min_sims: 最早在第几次模拟后启用提前终止
        """
        self.model = model
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.fp16 = fp16
        self.use_candidate_mask = use_candidate_mask
        self.candidate_radius = candidate_radius
        self.early_stop_threshold = early_stop_threshold
        self.early_stop_min_sims = early_stop_min_sims
        # autocast device_type: 兼容 'cuda' 和 ROCm 的 'cuda' 设备标识
        self._amp_device = 'cuda' if ('cuda' in device or device != 'cpu') else 'cpu'

    @torch.no_grad()
    def _evaluate(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        用神经网络评估局面

        Args:
            state: (2, H, W) 棋盘状态
        Returns:
            policy: (H*W,) 策略 logits
            value: 标量价值 [-1, 1]
        """
        state_tensor = torch.tensor(
            state, dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.amp.autocast(self._amp_device, enabled=self.fp16):
            logits, value = self.model(state_tensor)
        policy = logits.squeeze(0).float().cpu().numpy()
        value = value.item()
        return policy, value

    def _get_valid_actions(self, board: np.ndarray) -> np.ndarray:
        """获取合法动作 (棋盘上为空的位置，且在有棋子周围 radius 格内)"""
        valid = (board.reshape(-1) == 0)
        if self.use_candidate_mask:
            candidate = get_candidate_mask(board, self.candidate_radius)
            valid = valid & candidate
            # 如果候选过滤后没有合法动作，fallback 到全部空位
            if not valid.any():
                valid = (board.reshape(-1) == 0)
        return valid

    def _apply_action(self, board: np.ndarray, action: int,
                      player: int) -> np.ndarray:
        """在棋盘副本上落子"""
        new_board = board.copy()
        h, w = new_board.shape
        x, y = action // w, action % w
        new_board[x, y] = player
        return new_board

    def _check_win(self, board: np.ndarray, x: int, y: int,
                   win_condition: int = 5) -> bool:
        """检查是否获胜 (4 方向检测)"""
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

    def _is_terminal(self, board: np.ndarray, last_action: int) -> Tuple[bool, float]:
        """
        检查是否终局

        Returns:
            (is_terminal, value_from_current_player_perspective)
        """
        h, w = board.shape
        x, y = last_action // w, last_action % w
        player = board[x, y]

        # 检查是否获胜
        if self._check_win(board, x, y):
            return True, 1.0  # 当前玩家赢

        # 检查是否平局 (棋盘满了)
        if not (board == 0).any():
            return True, 0.0  # 平局

        return False, 0.0

    def search(self, state: np.ndarray, board: np.ndarray,
               temperature: float = 1.0,
               add_noise: bool = True) -> Tuple[List[int], np.ndarray]:
        """
        MCTS 主搜索循环

        Args:
            state: (2, H, W) 神经网络输入状态
            board: (H, W) 棋盘原始数组 (用于判断合法动作和终局)
            temperature: 温度参数 (控制探索程度)
            add_noise: 是否在根节点添加 Dirichlet 噪声
        Returns:
            actions: 可选动作列表
            probs: 动作概率分布
        """
        root = MCTSNode(prior=0)
        valid_actions = self._get_valid_actions(board)
        valid_count = int(valid_actions.sum())

        # 快速路径: 只剩 0 或 1 个合法动作
        if valid_count == 1:
            action = int(np.where(valid_actions)[0][0])
            return [action], np.array([1.0])
        if valid_count == 0:
            return [], np.array([])

        # 评估根节点
        policy, value = self._evaluate(state)

        # 根节点添加 Dirichlet 噪声 (AlphaZero 核心技巧)
        if add_noise:
            if valid_count > 0:
                noise = np.random.dirichlet(
                    [self.dirichlet_alpha] * valid_count
                )
                noise_idx = 0
                for i in np.where(valid_actions)[0]:
                    policy[i] = (1 - self.dirichlet_epsilon) * policy[i] + \
                                self.dirichlet_epsilon * noise[noise_idx]
                    noise_idx += 1

        # 从棋盘推导当前玩家: P1 先手, 棋子数相等→P1, 否则 P2
        current_player = 1 if (board == 1).sum() == (board == 2).sum() else 2
        root.expand(policy, valid_actions, current_player=current_player)

        # 主搜索循环
        for sim in range(self.num_simulations):
            node = root
            sim_board = board.copy()
            search_path = [node]

            # 1. SELECT: 沿 PUCT 选择到叶节点
            while node.is_expanded and node.children:
                action, node = node.select_child_puct(self.c_puct)
                sim_board = self._apply_action(
                    sim_board, action, node.parent.current_player
                )
                search_path.append(node)

            # 获取叶节点的父节点 (用于确定当前玩家)
            parent = search_path[-2] if len(search_path) >= 2 else root
            last_action = node.action

            # 2. 检查是否终局
            if last_action >= 0:
                # 落子的是 parent 的 current_player
                x, y = last_action // sim_board.shape[1], last_action % sim_board.shape[1]
                is_terminal, terminal_value = self._is_terminal(sim_board, last_action)
            else:
                is_terminal = False
                terminal_value = 0.0

            if is_terminal:
                # 终局: 价值从落子方视角
                value = terminal_value
                # 但 backup 时要从 node.parent.current_player 的对手视角
                # 因为 node 是 "对手下一步" 的节点
                value_for_backup = -terminal_value  # 对手赢了 = 当前玩家输了
            else:
                # 3. EXPAND & EVALUATE: 用 NN 评估
                # 构建当前节点的状态表示
                current_player = 3 - parent.current_player  # 轮到对手了
                sim_state = self._build_state(sim_board, current_player)
                policy, value = self._evaluate(sim_state)
                valid = self._get_valid_actions(sim_board)
                if valid.any():
                    node.expand(policy, valid, current_player)
                value_for_backup = value

            # 4. BACKUP: 回传价值
            node.backup(value_for_backup)

            # 5. 提前终止: 若某动作已占绝对优势，停止搜索
            if sim >= self.early_stop_min_sims and sim % 20 == 0:
                total_n = sum(child.N for child in root.children.values())
                if total_n > 0:
                    best_n = max(child.N for child in root.children.values())
                    if best_n / total_n >= self.early_stop_threshold:
                        break

        # 从根节点的访问次数计算策略向量
        actions = sorted(root.children.keys())
        visit_counts = np.array([root.children[a].N for a in actions], dtype=np.float64)

        if visit_counts.sum() == 0:
            # 兜底: 均匀分布
            probs = np.ones(len(actions)) / len(actions)
        elif temperature < 0.01:
            # 近似贪心
            probs = np.zeros(len(actions))
            probs[np.argmax(visit_counts)] = 1.0
        else:
            # 按温度缩放
            log_counts = np.log(visit_counts + 1e-10)
            log_counts = log_counts / temperature
            log_counts -= log_counts.max()  # 数值稳定
            probs = np.exp(log_counts)
            probs /= probs.sum()

        return actions, probs

    def _build_state(self, board: np.ndarray, current_player: int) -> np.ndarray:
        """
        从棋盘和当前玩家构建神经网络输入

        Args:
            board: (H, W) 棋盘
            current_player: 当前玩家 (1 或 2)
        Returns:
            (2, H, W) 状态表示
        """
        state = np.zeros((2, board.shape[0], board.shape[1]), dtype=np.float32)
        state[0] = (board == current_player).astype(np.float32)
        state[1] = (board == (3 - current_player)).astype(np.float32)
        return state


class BatchMCTS:
    """
    批量 MCTS: 将多个模拟的 NN 推理合并为一个 batch

    单线程 MCTS 的瓶颈是串行 NN 推理:
      800 次模拟 × 每次 1 次 NN 推理 = 800 次 GPU 调用
    批量 MCTS:
      800 次模拟 / 16 batch × 每次 1 次 NN 推理 = 50 次 GPU 调用
    加速比约 5-10×
    """

    def __init__(self, model, device, num_simulations: int = 800,
                 c_puct: float = 1.5, dirichlet_alpha: float = 0.3,
                 dirichlet_epsilon: float = 0.25,
                 batch_size: int = 16, fp16: bool = False,
                 use_candidate_mask: bool = True, candidate_radius: int = 2,
                 early_stop_threshold: float = 0.95, early_stop_min_sims: int = 50):
        self.model = model
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.batch_size = batch_size
        self.fp16 = fp16
        self.use_candidate_mask = use_candidate_mask
        self.candidate_radius = candidate_radius
        self.early_stop_threshold = early_stop_threshold
        self.early_stop_min_sims = early_stop_min_sims
        self._amp_device = 'cuda' if ('cuda' in device or device != 'cpu') else 'cpu'

    @torch.no_grad()
    def _evaluate_batch(self, states: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """批量 NN 推理 (支持 FP16 混合精度)"""
        batch = torch.tensor(np.stack(states), dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.amp.autocast(self._amp_device, enabled=self.fp16):
            logits, values = self.model(batch)
        policies = logits.float().cpu().numpy()
        values = values.float().cpu().numpy()
        return policies, values

    def search(self, state: np.ndarray, board: np.ndarray,
               temperature: float = 1.0,
               add_noise: bool = True) -> Tuple[List[int], np.ndarray]:
        """批量 MCTS 搜索 (接口与 MCTS.search 一致)"""
        root = MCTSNode(prior=0)
        valid_actions = (board.reshape(-1) == 0)
        if self.use_candidate_mask:
            candidate = get_candidate_mask(board, self.candidate_radius)
            valid_actions = valid_actions & candidate
            if not valid_actions.any():
                valid_actions = (board.reshape(-1) == 0)
        h, w = board.shape
        valid_count = int(valid_actions.sum())

        # 快速路径
        if valid_count == 1:
            action = int(np.where(valid_actions)[0][0])
            return [action], np.array([1.0])
        if valid_count == 0:
            return [], np.array([])

        # 评估根节点
        policy, value = self._evaluate_batch([state])
        policy = policy[0]
        value = value[0]

        # Dirichlet 噪声
        if add_noise:
            if valid_count > 0:
                noise = np.random.dirichlet([self.dirichlet_alpha] * valid_count)
                noise_idx = 0
                for i in np.where(valid_actions)[0]:
                    policy[i] = (1 - self.dirichlet_epsilon) * policy[i] + \
                                self.dirichlet_epsilon * noise[noise_idx]
                    noise_idx += 1

        # 从棋盘推导当前玩家: P1 先手, 棋子数相等→P1, 否则 P2
        current_player = 1 if (board == 1).sum() == (board == 2).sum() else 2
        root.expand(policy, valid_actions, current_player=current_player)

        sim_count = 0
        while sim_count < self.num_simulations:
            # 收集一批叶子节点
            batch_leaves = []
            batch_states = []
            batch_boards = []

            for _ in range(self.batch_size):
                if sim_count >= self.num_simulations:
                    break

                node = root
                sim_board = board.copy()
                search_path = [node]

                # SELECT
                while node.is_expanded and node.children:
                    action, node = node.select_child_puct(self.c_puct)
                    # 原地落子 (sim_board 已经是独立拷贝，无需再 copy)
                    h, w = sim_board.shape
                    x, y = action // w, action % w
                    sim_board[x, y] = node.parent.current_player
                    search_path.append(node)

                parent = search_path[-2] if len(search_path) >= 2 else root
                last_action = node.action

                # 检查终局
                if last_action >= 0:
                    is_terminal, terminal_value = self._check_terminal_static(
                        sim_board, last_action
                    )
                else:
                    is_terminal, terminal_value = False, 0.0

                if is_terminal:
                    # 直接 backup
                    node.backup(-terminal_value)
                else:
                    current_player = 3 - parent.current_player
                    sim_state = self._build_state_static(sim_board, current_player)
                    batch_leaves.append((node, search_path, sim_board, current_player))
                    batch_states.append(sim_state)
                    batch_boards.append(sim_board)

                sim_count += 1

            # 批量 NN 推理
            if batch_states:
                policies, values = self._evaluate_batch(batch_states)

                for (node, search_path, sim_board, cp), policy, value in \
                        zip(batch_leaves, policies, values):
                    valid = self._get_valid_actions(sim_board) if hasattr(self, '_get_valid_actions') else (sim_board.reshape(-1) == 0)
                    # BatchMCTS 也应用候选过滤
                    if self.use_candidate_mask:
                        candidate = get_candidate_mask(sim_board, self.candidate_radius)
                        valid = valid & candidate
                        if not valid.any():
                            valid = (sim_board.reshape(-1) == 0)
                    if valid.any():
                        node.expand(policy, valid, cp)
                    node.backup(value)

            # 提前终止检查
            if sim_count >= self.early_stop_min_sims and sim_count % 20 == 0:
                total_n = sum(child.N for child in root.children.values())
                if total_n > 0:
                    best_n = max(child.N for child in root.children.values())
                    if best_n / total_n >= self.early_stop_threshold:
                        break

        # 计算策略
        actions = sorted(root.children.keys())
        visit_counts = np.array([root.children[a].N for a in actions], dtype=np.float64)

        if visit_counts.sum() == 0:
            probs = np.ones(len(actions)) / len(actions)
        elif temperature < 0.01:
            probs = np.zeros(len(actions))
            probs[np.argmax(visit_counts)] = 1.0
        else:
            log_counts = np.log(visit_counts + 1e-10) / temperature
            log_counts -= log_counts.max()
            probs = np.exp(log_counts)
            probs /= probs.sum()

        return actions, probs

    @staticmethod
    def _apply_action_static(board, action, player):
        new_board = board.copy()
        h, w = new_board.shape
        x, y = action // w, action % w
        new_board[x, y] = player
        return new_board

    @staticmethod
    def _check_terminal_static(board, last_action):
        h, w = board.shape
        x, y = last_action // w, last_action % w
        player = board[x, y]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dx, dy in directions:
            count = 1
            for sign in [1, -1]:
                nx, ny = x + sign * dx, y + sign * dy
                while 0 <= nx < h and 0 <= ny < w and board[nx, ny] == player:
                    count += 1
                    nx += sign * dx
                    ny += sign * dy
            if count >= 5:
                return True, 1.0
        if not (board == 0).any():
            return True, 0.0
        return False, 0.0

    @staticmethod
    def _build_state_static(board, current_player):
        state = np.zeros((2, board.shape[0], board.shape[1]), dtype=np.float32)
        state[0] = (board == current_player).astype(np.float32)
        state[1] = (board == (3 - current_player)).astype(np.float32)
        return state
