import torch
import math
import numpy as np
import copy

class TreeNode:
    def __init__(self, parent, prior_p):
        self.parent = parent
        self.children = {}  # {action: TreeNode}
        self.n_visits = 0
        self.Q = 0
        self.u = 0
        self.P = prior_p  # 神经网络输出的先验概率

    def expand(self, action_priors):
        """扩展子节点"""
        for action, prob in action_priors:
            if action not in self.children:
                self.children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """选择UCB值最大的子节点"""
        return max(self.children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def get_value(self, c_puct):
        """计算 UCB 值"""
        self.u = (c_puct * self.P * math.sqrt(self.parent.n_visits) / (1 + self.n_visits))
        return self.Q + self.u

    def update(self, leaf_value):
        """反向传播更新当前节点"""
        self.n_visits += 1
        # Q值更新公式：平均值
        self.Q += 1.0 * (leaf_value - self.Q) / self.n_visits
        if self.parent:
            self.parent.update(leaf_value)

    def is_leaf(self):
        return len(self.children) == 0

class MCTS:
    def __init__(self, model, c_puct=5, n_playout=400, board_size=15):
        self._model = model
        self._c_puct = c_puct
        self._n_playout = n_playout
        self._board_size = board_size
        self._root = None

    def get_move_probs(self, env, temp=1e-3):
        """
        运行 MCTS 并返回动作概率分布
        :param env: 当前的游戏环境对象 (需要有 copy 方法和 step 方法)
        :param temp: 温度参数，控制探索程度 (训练初期用 1.0，后期或竞技用 1e-3)
        """
        # 这里的 state 必须适配神经网络的输入格式
        # 注意：你需要确保 env.get_state_representation() 返回的是 canonical form (当前玩家视角)
        state_repr = env.get_state_representation()
        
        # 定义根节点
        device = next(self._model.parameters()).device
        state_tensor = torch.FloatTensor(state_repr).unsqueeze(0).to(device)
        
        with torch.no_grad():
            action_probs, _ = self._model(state_tensor)
            # 加上 Dirichlet 噪声以增加根节点的探索性 (仅在训练时推荐)
            action_probs = torch.softmax(action_probs, dim=1).cpu().numpy().flatten()
            
        # 过滤非法动作
        valid_acts = self._get_valid_moves(env)
        noise = np.random.dirichlet(0.3 * np.ones(len(valid_acts))) # AlphaZero 参数
        
        # 重新归一化概率
        probs_dict = {}
        for idx, act in enumerate(valid_acts):
            # 0.75 * 原概率 + 0.25 * 噪声
            val = 0.75 * action_probs[act] + 0.25 * noise[idx]
            probs_dict[act] = val
            
        self._root = TreeNode(None, 1.0)
        self._root.expand(probs_dict.items())

        # 开始模拟 (Simulations)
        for _ in range(self._n_playout):
            self._playout(copy.deepcopy(env))

        # 计算访问次数作为最终策略
        act_visits = [(act, node.n_visits) for act, node in self._root.children.items()]
        acts, visits = zip(*act_visits)
        act_probs = np.zeros(self._board_size * self._board_size)
        
        # 温度参数处理
        if temp == 0:
            best_act = acts[np.argmax(visits)]
            act_probs[best_act] = 1.0
        else:
            visits = np.array(visits)
            # 防止溢出
            visits_temp = visits ** (1.0 / temp)
            probs = visits_temp / np.sum(visits_temp)
            for act, prob in zip(acts, probs):
                act_probs[act] = prob

        return act, act_probs

    def _playout(self, env):
        node = self._root
        
        # 1. Selection
        while not node.is_leaf():
            action, node = node.select(self._c_puct)
            _, done, _ = env.step(action)
            if done:
                break

        # 2. Evaluation & Expansion
        # 获取当前局面的网络评估
        state_repr = env.get_state_representation()
        device = next(self._model.parameters()).device
        state_tensor = torch.FloatTensor(state_repr).unsqueeze(0).to(device)
        
        with torch.no_grad():
            action_probs, leaf_value = self._model(state_tensor)
        leaf_value = leaf_value.item()
        
        # 如果游戏没结束，扩展节点
        # 注意：需要检查环境是否结束，因为 Selection 阶段可能已经走到终局
        check_done = env.is_game_over()
        valid_moves = self._get_valid_moves(env)

        if check_done:
            # 游戏结束
            if env.winning_line:
                # 有人获胜。因为是 current_player 视角，而上一手(对手)刚刚获胜，
                # 所以当前局面对于 current_player 是必输 (-1)。
                leaf_value = -1.0
            else:
                # 平局
                leaf_value = 0.0
        
        elif not valid_moves: # 平局或无子可落
            leaf_value = 0.0
            
        else:
            # 正常扩展
            # 如果节点已经是叶子节点（未扩展），则扩展它
            if node.is_leaf():
                action_probs = torch.softmax(action_probs, dim=1).cpu().numpy().flatten()
                probs_dict = {a: action_probs[a] for a in valid_moves}
                node.expand(probs_dict.items())
            else:
                pass

        # 3. Backpropagation
        # 这里的 leaf_value 是网络对 "当前状态" 的评估。
        # MCTS 树中，节点存储的是 "父节点选择该动作后" 的价值。
        # 在 AlphaZero 中，Value Head 预测的是当前局面对于当前玩家的胜率 (-1 到 1)
        # 每次递归回溯，需要对 value 取反 (Negamax)
        node.update(-leaf_value)

    def _get_valid_moves(self, env):
        # 假设 env.board 是 numpy 数组，0 表示空
        board_flat = env.board.flatten()
        return [i for i in range(len(board_flat)) if board_flat[i] == 0]
