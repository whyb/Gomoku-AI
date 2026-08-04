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
    滑动窗口检测对手所有威胁棋形。

    用长度为 win_condition 的滑动窗沿 4 个方向扫描，自动覆盖：
      - 连四:   X X X X _    (连续)
      - 跳四:   X X _ X X    (一间隔)
      - 跳四:   X _ X X X    (一间隔)
      - 跳四:   _ X X X X    (一端空)
      - 活三:   _ X X X _    (连续三, 两端空)
      - 跳活三: _ X _ X X _  (三带一间隔)
      - 跳活三: _ X X _ X _  (三带一间隔)
      - 冲三:   O X X X _    (一端堵)
      以及上述所有棋形在 win_condition≠5 时的等价形态。

    返回: {
        'must_block': 对手已有 win_condition-1 子，堵住空位即可阻止连珠,
        'threats':    对手已有 win_condition-2 子，潜在威胁点,
    }
    """
    h, w = board.shape
    opponent = 3 - player
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    must_block = set()
    threats = set()

    for x in range(h):
        for y in range(w):
            for dx, dy in directions:
                end_x = x + (win_condition - 1) * dx
                end_y = y + (win_condition - 1) * dy
                if not (0 <= end_x < h and 0 <= end_y < w):
                    continue

                # 快速跳过: 窗口中没有对手棋子则不可能是威胁
                has_opp = False
                has_own = False
                for k in range(win_condition):
                    cx, cy = x + k * dx, y + k * dy
                    v = board[cx, cy]
                    if v == opponent:
                        has_opp = True
                        break  # 有对手棋子, 需详细检查
                    elif v == player:
                        has_own = True
                if not has_opp:
                    continue  # 窗口中没有对手棋子, 跳过

                opp_cnt = 0
                own_cnt = 0
                empties = []
                for k in range(win_condition):
                    cx, cy = x + k * dx, y + k * dy
                    v = board[cx, cy]
                    if v == opponent:
                        opp_cnt += 1
                    elif v == player:
                        own_cnt += 1
                        break  # 我方棋子堵住了这条线, 无需继续
                    else:
                        empties.append((cx, cy))

                if own_cnt > 0:
                    continue  # 我方棋子堵住了这条线

                if opp_cnt == win_condition - 1:
                    # 对手差 1 子连珠 → 必须堵
                    for ex, ey in empties:
                        must_block.add(ex * w + ey)
                elif opp_cnt == win_condition - 2:
                    # 对手差 2 子 → 潜在威胁 (活三/冲三等)
                    for ex, ey in empties:
                        threats.add(ex * w + ey)

    return {'must_block': must_block, 'threats': threats}


def get_forced_move(board: np.ndarray, player: int,
                    win_condition: int = 5) -> Tuple[Optional[int], Optional[str]]:
    """
    查找强制走法（规则短路），优先级：
      1. 自己能连五 → 直接获胜
      2. 对手有必堵点 (冲四/连四/跳四) → 必须堵 (即使有多个也挑一个)
      3. 对手已有连五点（兜底检测）
    返回: (action, reason) 或 (None, None)
    """
    h, w = board.shape

    # 1. 自己立即获胜
    win_action = _find_immediate_win(board, player)
    if win_action is not None:
        return win_action, 'win'

    # 2. 对手威胁检测 (滑动窗口, 覆盖连四/跳四等所有 4-in-W 形态)
    threats = _find_opponent_threats(board, player, win_condition)
    must_block = threats['must_block']

    if len(must_block) == 1:
        return must_block.pop(), 'block'
    elif len(must_block) >= 2:
        # 多个必堵点: 优先选同时覆盖 threats (活三/冲三) 的交叉点
        # 旧逻辑: 冲四永远强制堵, 跳过 MCTS 以提速
        threat_points = threats['threats']
        overlap = must_block & threat_points
        if overlap:
            return overlap.pop(), 'block'
        return must_block.pop(), 'block'

    # 3. 兜底: 对手已有连五点 (理论上前面已覆盖，此处保底)
    opponent = 3 - player
    opponent_win_points = set()
    for x in range(h):
        for y in range(w):
            if board[x, y] == 0:
                board[x, y] = opponent
                win = _check_win_static(board, x, y, win_condition)
                board[x, y] = 0
                if win:
                    opponent_win_points.add(x * w + y)
    if len(opponent_win_points) == 1:
        return opponent_win_points.pop(), 'block_win'

    return None, None


def get_candidate_mask(board: np.ndarray, radius: int = 2) -> np.ndarray:
    """
    获取候选落子点掩码（bool 数组，flattened）。
    只考虑已有棋子周围 radius 格内的空点。
    空棋盘时返回中心 3x3 区域。

    实现: 用 numpy 零填充 + 滑动窗口 OR (等价于方形膨胀) 向量化计算,
    单次调用耗时与棋子数量无关 (原 Python 双层循环会随棋子数线性变慢)。
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

    # 方形半径膨胀: 零填充边界后滑动窗口 OR, 语义与原实现一致
    occupied = (board != 0)
    padded = np.pad(occupied, radius, mode='constant', constant_values=False)
    mask = np.zeros((h, w), dtype=bool)
    for dx in range(2 * radius + 1):
        for dy in range(2 * radius + 1):
            mask |= padded[dx:dx + h, dy:dy + w]
    # 与原始语义一致: 只标记空点
    mask &= (board == 0)
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
               current_player: int,
               add_noise: bool = False,
               dirichlet_alpha: float = 0.3,
               dirichlet_epsilon: float = 0.25):
        """
        扩展节点: 根据策略网络的输出创建子节点

        Args:
            policy: 策略网络输出的 logits (board_size² 维)
            valid_actions: 合法动作的 bool 数组
            current_player: 当前玩家 (1 或 2)
            add_noise: 是否添加 Dirichlet 噪声 (仅根节点)
            dirichlet_alpha: Dirichlet 噪声 alpha 参数
            dirichlet_epsilon: 噪声混合比例
        """
        self.current_player = current_player
        self.is_expanded = True

        valid_indices = np.flatnonzero(valid_actions)
        valid_count = len(valid_indices)

        # 纯 numpy softmax (float32, 避免 torch tensor 创建开销)
        masked = np.where(valid_actions, policy, -1e9).astype(np.float32)
        masked -= masked.max()
        exp = np.exp(masked.astype(np.float64))  # exp 用 float64 保精度, 避免 float32 下溢
        exp = np.where(valid_actions, exp, 0.0).astype(np.float32)
        s = exp.sum()
        probs = exp / s if s > 0 else valid_actions.astype(np.float32) / max(1, valid_actions.sum())

        # Dirichlet 噪声: 在概率空间混合 (AlphaZero 标准做法)
        if add_noise and valid_count > 0:
            noise = np.random.dirichlet([dirichlet_alpha] * valid_count)
            for idx, i in enumerate(valid_indices):
                probs[i] = (1 - dirichlet_epsilon) * probs[i] + dirichlet_epsilon * noise[idx]
            probs /= probs.sum()

        for a in valid_indices:
            self.children[int(a)] = MCTSNode(prior=float(probs[a]), parent=self, action=int(a))

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
                 early_stop_threshold: float = 0.95, early_stop_min_sims: int = 50,
                 win_condition: int = 5, use_human_knowledge: bool = False):
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
            use_human_knowledge: 是否在搜索树中用人类知识增强 (默认关闭, 保持搜索纯净)
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
        self.win_condition = win_condition
        self.use_human_knowledge = use_human_knowledge
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

        # 评估根节点 → logits
        policy, _ = self._evaluate(state)

        # 从棋盘推导当前玩家: P1 先手, 棋子数相等→P1, 否则 P2
        current_player = 1 if (board == 1).sum() == (board == 2).sum() else 2
        # 传入 logits, expand 内部做 softmax + 噪声混合
        root.expand(policy, valid_actions, current_player=current_player,
                    add_noise=add_noise,
                    dirichlet_alpha=self.dirichlet_alpha,
                    dirichlet_epsilon=self.dirichlet_epsilon)

        # 主搜索循环
        # 按棋盘内容缓存 _get_valid_actions 结果: 不同路径可能到达相同盘面
        # (同一棋子集合的不同落子顺序), 避免对同一盘面重复计算掩码
        valid_cache: Dict[bytes, np.ndarray] = {}
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

            # 获取叶节点的父节点
            parent = search_path[-2] if len(search_path) >= 2 else root
            last_action = node.action

            # 2. 检查是否终局
            if last_action >= 0:
                is_terminal, terminal_value = self._is_terminal(sim_board, last_action)
            else:
                is_terminal = False
                terminal_value = 0.0

            if is_terminal:
                value_for_backup = -terminal_value
            else:
                # 3. EXPAND & EVALUATE: NN 推理 + 扩展
                current_player = 3 - parent.current_player
                sim_state = self._build_state(sim_board, current_player)
                policy, value = self._evaluate(sim_state)

                # 可选: 人类知识增强 (--mcts_human_knowledge, 默认关闭)
                if self.use_human_knowledge:
                    win_action = _find_immediate_win(sim_board, current_player)
                    if win_action is not None:
                        policy[win_action] = policy.max() + 10.0
                    threat_info = _find_opponent_threats(sim_board, current_player,
                                                         self.win_condition)
                    for action in threat_info['must_block']:
                        policy[action] = policy.max() + 10.0
                    for action in threat_info['threats']:
                        if policy[action] < policy.max() + 1.0:
                            policy[action] = max(policy[action], policy.max() + 1.0)

                board_key = sim_board.tobytes()
                valid = valid_cache.get(board_key)
                if valid is None:
                    valid = self._get_valid_actions(sim_board)
                    valid_cache[board_key] = valid
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
                 early_stop_threshold: float = 0.95, early_stop_min_sims: int = 50,
                 win_condition: int = 5, use_human_knowledge: bool = False):
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
        self.win_condition = win_condition
        self.use_human_knowledge = use_human_knowledge
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

    def _get_valid_actions(self, board: np.ndarray) -> np.ndarray:
        """获取合法动作 (棋盘上为空的位置，且在有棋子周围 radius 格内)"""
        valid = (board.reshape(-1) == 0)
        if self.use_candidate_mask:
            candidate = get_candidate_mask(board, self.candidate_radius)
            valid = valid & candidate
            if not valid.any():
                valid = (board.reshape(-1) == 0)
        return valid

    def search(self, state: np.ndarray, board: np.ndarray,
               temperature: float = 1.0,
               add_noise: bool = True) -> Tuple[List[int], np.ndarray]:
        """批量 MCTS 搜索 (接口与 MCTS.search 一致)"""
        root = MCTSNode(prior=0)
        valid_actions = self._get_valid_actions(board)
        h, w = board.shape
        valid_count = int(valid_actions.sum())

        # 快速路径
        if valid_count == 1:
            action = int(np.where(valid_actions)[0][0])
            return [action], np.array([1.0])
        if valid_count == 0:
            return [], np.array([])

        # 评估根节点 → logits
        policy, _ = self._evaluate_batch([state])
        policy = policy[0]

        # 从棋盘推导当前玩家: P1 先手, 棋子数相等→P1, 否则 P2
        current_player = 1 if (board == 1).sum() == (board == 2).sum() else 2
        # 传入 logits, expand 内部做 softmax + 噪声混合
        root.expand(policy, valid_actions, current_player=current_player,
                    add_noise=add_noise,
                    dirichlet_alpha=self.dirichlet_alpha,
                    dirichlet_epsilon=self.dirichlet_epsilon)

        sim_count = 0
        # 同一批模拟会反复选中同一叶子 (相同 sim_board), 掩码计算昂贵,
        # 按棋盘内容缓存 _get_valid_actions 结果, 每个唯一盘面只算一次
        valid_cache: Dict[bytes, np.ndarray] = {}
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

                # 注意: 不按 id 去重。同一 batch 内树尚未变化, PUCT 确定性选择
                # 会反复选中同一叶子; 这些重复叶子共享同一次 NN 批推理 (快)，
                # 若用 visited 去重则每个 batch 只剩 1 片叶子, 批量加速完全失效。

                parent = search_path[-2] if len(search_path) >= 2 else root
                last_action = node.action

                # 检查终局
                if last_action >= 0:
                    is_terminal, terminal_value = self._check_terminal_static(
                        sim_board, last_action, self.win_condition
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
                    # 可选: 人类知识增强 (--mcts_human_knowledge, 默认关闭)
                    if self.use_human_knowledge:
                        win_action = _find_immediate_win(sim_board, cp)
                        if win_action is not None:
                            policy[win_action] = policy.max() + 10.0
                        threat_info = _find_opponent_threats(sim_board, cp,
                                                             self.win_condition)
                        for action in threat_info['must_block']:
                            policy[action] = policy.max() + 10.0
                        for action in threat_info['threats']:
                            if policy[action] < policy.max() + 1.0:
                                policy[action] = max(policy[action], policy.max() + 1.0)

                    board_key = sim_board.tobytes()
                    valid = valid_cache.get(board_key)
                    if valid is None:
                        valid = self._get_valid_actions(sim_board)
                        valid_cache[board_key] = valid
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
    def _check_terminal_static(board, last_action, win_condition=5):
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
            if count >= win_condition:
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
