import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np
import random
from config import Config

# 游戏环境
class Gomoku:
    def __init__(self, board_size=None, win_condition=None):
        self.board_size = board_size or Config.BOARD_SIZE
        self.win_condition = win_condition or Config.WIN_CONDITION
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1
        self.winning_line = []
        self.step_count = 0
        self.reward_log = {1: {'count': 0, 'rewards': {}}, 2: {'count': 0, 'rewards': {}}}
        # 全局奖励记录和计数器
        self.global_reward_summary = {1: {}, 2: {}}
        self.game_count = 0
        self.summary_interval = 1000 # 设置打印间隔，例如每1000局打印一次

    def reset(self):
        # 在新局开始前，记录上一局的奖励
        self._aggregate_rewards()
        self.game_count += 1
        
        # 每隔一定局数打印一次全局总结
        if (self.game_count % self.summary_interval) == 0:
            self._print_global_summary()

        self.board.fill(0)
        self.current_player = 1
        self.winning_line = []
        self.step_count = 0
        self.reward_log = {1: {'count': 0, 'rewards': {}}, 2: {'count': 0, 'rewards': {}}} # 清空奖励记录
        return self.board

    def is_game_over(self):
        return len(self.winning_line) > 0 or self.step_count >= self.board_size * self.board_size

    def is_winning_move(self, x, y):
        # 检查胜利条件
        def count_consecutive(player, dx, dy):
            count = 0
            line = []
            # 正向计数
            for step in range(1, self.win_condition):
                nx, ny = x + dx * step, y + dy * step
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == player:
                    count += 1
                    line.append((nx, ny))
                else:
                    break
            # 反向计数
            for step in range(1, self.win_condition):
                nx, ny = x - dx * step, y - dy * step
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == player:
                    count += 1
                    line.append((nx, ny))
                else:
                    break
            return count, line

        player = self.board[x, y]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dx, dy in directions:
            count, line = count_consecutive(player, dx, dy)
            if count + 1 >= self.win_condition:
                self.winning_line = [(x, y)] + line
                return True
        return False

    def _detect_patterns(self, player, x, y):
        """
        通用棋形检测函数，可准确识别各种棋形。返回一个字典，包含所有棋形的数量和信息。
        核心逻辑：使用 6 格窗口，并检查窗口两端的边界条件，以正确区分活四和冲四。
        """
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        patterns = {
            'live4': [], 'rush4': [], 'live3': [], 'rush3': [],
            'live2': [], 'rush2': []
        }
        opponent = 3 - player

        # 遍历 4 个方向
        for dx, dy in directions:
            # 对于每个方向，以落子点 (x, y) 为中心，向两边延伸
            for i in range(-5, 1): # 6 格窗口的起始位置
                nx_start, ny_start = x + i * dx, y + i * dy
                if not (0 <= nx_start < self.board_size and 0 <= ny_start < self.board_size):
                    continue
                
                # 构建 6 格窗口
                window = []
                coords = []
                for j in range(6):
                    nx, ny = nx_start + j * dx, ny_start + j * dy
                    if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                        window.append(self.board[nx, ny])
                        coords.append((nx, ny))
                    else:
                        window.append(opponent) # 边界视为阻挡

                p_count = window.count(player)
                o_count = window.count(opponent)
                e_count = window.count(0)

                if p_count == 4 and e_count == 2:
                    # 活四：4个自己的棋子，2个空位
                    # 检查两端是否为空
                    if window[0] == 0 and window[5] == 0:
                        stones = {c for i, c in enumerate(coords) if window[i] == player}
                        patterns['live4'].append(stones)
                
                if p_count == 4 and e_count == 1:
                    # 冲四：4个自己的棋子，1个空位
                    # 活四的窗口不包含对手，因此这里只检查冲四
                    stones = {c for i, c in enumerate(coords) if window[i] == player}
                    patterns['rush4'].append(stones)
                
                if p_count == 3 and e_count == 2:
                    # 活三：3个自己的棋子，2个空位
                    if window[0] == 0 and window[5] == 0:
                        stones = {c for i, c in enumerate(coords) if window[i] == player}
                        patterns['live3'].append(stones)
                
                if p_count == 3 and e_count == 1:
                    # 眠三：3个自己的棋子，1个空位
                    if window[0] == 0 and window[5] == 0:
                        stones = {c for i, c in enumerate(coords) if window[i] == player}
                        patterns['rush3'].append(stones)

                if p_count == 2 and e_count == 2:
                    # 活二：2个自己的棋子，2个空位
                    if window[0] == 0 and window[5] == 0:
                        stones = {c for i, c in enumerate(coords) if window[i] == player}
                        patterns['live2'].append(stones)
                
                if p_count == 2 and e_count == 1:
                    # 冲二：2个自己的棋子，1个空位
                    if window[0] == 0 and window[5] == 0:
                        stones = {c for i, c in enumerate(coords) if window[i] == player}
                        patterns['rush2'].append(stones)


        # 去重
        for key in patterns:
            patterns[key] = [frozenset(s) for s in patterns[key]]
            patterns[key] = list(set(patterns[key]))

        return patterns

    def _is_double_live3(self, player, x, y, patterns):
        """基于通用检测结果判断双活三"""
        count = 0
        for i in range(len(patterns['live3'])):
            for j in range(i + 1, len(patterns['live3'])):
                if len(patterns['live3'][i].intersection(patterns['live3'][j])) <= 1:
                    count += 1
        return count >= 1

    def _is_rush4_live3(self, player, x, y, patterns):
        """基于通用检测结果判断冲四活三"""
        if not patterns['rush4'] or not patterns['live3']:
            return False
        for rush4_stones in patterns['rush4']:
            for live3_stones in patterns['live3']:
                if len(rush4_stones.intersection(live3_stones)) <= 2:
                    return True
        return False

    def _is_double_rush4(self, player, x, y, patterns):
        """基于通用检测结果判断双冲四"""
        return len(patterns['rush4']) >= 2

    def calculate_reward(self, x, y):
        """
        使用优先级逻辑计算奖励，避免奖励叠加。
        优先级：必胜棋形 > 强攻棋形 > 防守奖励 > 基础棋形奖励。
        """
        player = self.board[x, y]
        opponent = 3 - player
        
        # 1. 检查是否形成了必胜棋形（活四、双活三、冲四活三、双冲四）
        patterns = self._detect_patterns(player, x, y)
        reward_type = None
        
        if patterns['live4']:
            reward_type = "live4"
        elif self._is_rush4_live3(player, x, y, patterns):
            reward_type = "rush4_live3"
        elif self._is_double_live3(player, x, y, patterns):
            reward_type = "double_live3"
        elif self._is_double_rush4(player, x, y, patterns):
            reward_type = "double_rush4"
        
        if reward_type:
            reward = Config.REWARD[reward_type]
            self._log_reward(player, reward_type, reward)
            return reward

        # 2. 检查是否阻挡了对手的必胜棋形（防守奖励）
        # 防守奖励逻辑：检查对手在落子前是否有必胜威胁
        temp_board_before_move = self.board.copy()
        temp_board_before_move[x, y] = 0 # 模拟回退一步
        if self._detect_opponent_win_threat(temp_board_before_move, opponent, x, y):
            reward = Config.REWARD["block_win"]
            self._log_reward(player, "block_win", reward)
            return reward

        # 3. 如果没有形成高级棋形或阻挡，则计算基础棋形奖励
        total_reward = 0

        # 记录每种棋形
        if len(patterns['rush4']) > 0:
            total_reward += len(patterns['rush4']) * Config.REWARD["rush4"]
            self._log_reward(player, "rush4", len(patterns['rush4']) * Config.REWARD["rush4"])
        
        if len(patterns['live3']) > 0:
            total_reward += len(patterns['live3']) * Config.REWARD["live3"]
            self._log_reward(player, "live3", len(patterns['live3']) * Config.REWARD["live3"])

        if len(patterns['rush3']) > 0:
            total_reward += len(patterns['rush3']) * Config.REWARD["rush3"]
            self._log_reward(player, "rush3", len(patterns['rush3']) * Config.REWARD["rush3"])

        if len(patterns['live2']) > 0:
            total_reward += len(patterns['live2']) * Config.REWARD["live2"]
            self._log_reward(player, "live2", len(patterns['live2']) * Config.REWARD["live2"])
        
        if len(patterns['rush2']) > 0:
            total_reward += len(patterns['rush2']) * Config.REWARD["rush2"]
            self._log_reward(player, "rush2", len(patterns['rush2']) * Config.REWARD["rush2"])

        return total_reward
    
    def _detect_opponent_win_threat(self, board, opponent, block_x, block_y):
        """
        检查对手在落子前是否有一个可以立即获胜的空位。
        这里的逻辑是检查对手是否在落子前，在某个位置已经形成了活四或冲四。
        如果我方在 block_x, block_y 落子正好阻止了这个活四或冲四的完成，则奖励。
        这个实现较为复杂，简单起见，我们只检查对手是否有活四或冲四的威胁点。
        """
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i, j] == 0: # 找到空位
                    # 模拟对手落子
                    board[i, j] = opponent
                    # 检查对手是否获胜
                    if self._is_winning_move(board, i, j, opponent):
                        board[i, j] = 0 # 恢复棋盘
                        # 如果这个胜利点与我们的落子点相邻，则认为我们成功防守
                        if abs(i-block_x) <= 1 and abs(j-block_y) <= 1:
                            return True
                    board[i, j] = 0 # 恢复棋盘

        return False

    def _is_winning_move(self, board, x, y, player):
        # 内部辅助函数
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dx, dy in directions:
            count = 1
            # 正向
            for step in range(1, self.win_condition):
                nx, ny = x + dx * step, y + dy * step
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size and board[nx, ny] == player:
                    count += 1
                else:
                    break
            # 反向
            for step in range(1, self.win_condition):
                nx, ny = x - dx * step, y - dy * step
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size and board[nx, ny] == player:
                    count += 1
                else:
                    break
            if count >= self.win_condition:
                return True
        return False

    def step(self, action):
        x, y = action // self.board_size, action % self.board_size
        if self.board[x, y] != 0:
            return -1, True, 0  # 无效落子
        current_player = self.current_player
        self.board[x, y] = current_player
        self.step_count += 1
        
        if self.is_winning_move(x, y):
            # 胜利时返回：(玩家, 是否结束, (基础奖励, 落子步数))
            reward = Config.REWARD["win"]
            self._log_reward(current_player, "win", reward)
            self._aggregate_rewards()
            self.game_count += 1
            if self.game_count % self.summary_interval == 0:
                self._print_global_summary()
            return current_player, True, (reward, self.step_count)
        else:
            # 计算连珠奖励
            reward = self.calculate_reward(x, y)
            # 切换玩家
            next_player = 3 - current_player
            self.current_player = next_player
            return next_player, False, reward
    
    def get_state_representation(self):
        # 多通道输入表示
        if not np.all((self.board == 0) | (self.board == 1) | (self.board == 2)):
            print("无效棋盘值:", self.board)
        player1_board = (self.board == 1).astype(np.float32)
        player2_board = (self.board == 2).astype(np.float32)
        return np.stack([player1_board, player2_board], axis=0) # 形状: (2, board_size, board_size)

    def _log_reward(self, player, reward_type, reward_value):
        """记录每一步的奖励信息"""
        if reward_type not in self.reward_log[player]['rewards']:
            self.reward_log[player]['rewards'][reward_type] = 0
        self.reward_log[player]['rewards'][reward_type] += reward_value
        self.reward_log[player]['count'] += 1

    def _aggregate_rewards(self):
        """将本局的奖励累加到全局记录器中"""
        for player in [1, 2]:
            for reward_type, value in self.reward_log[player]['rewards'].items():
                if reward_type not in self.global_reward_summary[player]:
                    self.global_reward_summary[player][reward_type] = []
                self.global_reward_summary[player][reward_type].append(value)
            
            # 记录总奖励
            total_reward = sum(self.reward_log[player]['rewards'].values())
            if '总奖励' not in self.global_reward_summary[player]:
                self.global_reward_summary[player]['总奖励'] = []
            self.global_reward_summary[player]['总奖励'].append(total_reward)

    def _print_reward_summary(self):
        """打印游戏结束后的奖励总结"""
        if self.reward_log[1]['count'] > 0 or self.reward_log[2]['count'] > 0:
            print("\n--- 游戏奖励总结 ---")
            for player in [1, 2]:
                summary = self.reward_log[player]
                total_reward = sum(summary['rewards'].values())
                if summary['count'] > 0:
                    print(f"玩家 {player} (共 {summary['count']} 步):")
                    for reward_type, value in summary['rewards'].items():
                        print(f"  - {reward_type}: {value}")
                    print(f"  - 总奖励: {total_reward}\n")
            print("--------------------\n")
    
    def _print_global_summary(self):
        """打印全局奖励总结"""
        print(f"\n--- 全局奖励总结 (共 {self.game_count} 局) ---")
        for player in [1, 2]:
            summary = self.global_reward_summary[player]
            if summary:
                print(f"玩家 {player}:")
                for reward_type, values in summary.items():
                    if values:
                        # 计算平均值
                        avg_reward = np.mean(values)
                        # 计算标准差
                        std_dev = np.std(values)
                        print(f"  - {reward_type}: 平均值={avg_reward:.2f}, 标准差={std_dev:.2f}")
        print("-------------------------------------------\n")
        
        # 打印完后清空全局记录器，重新开始统计
        self.global_reward_summary = {1: {}, 2: {}}

    def print_board(self):
        """打印当前棋盘状态，用 X 表示玩家1，O 表示玩家2，. 表示空位"""
        # 打印列索引
        print("   " + " ".join(f"{i:2}" for i in range(self.board_size)))
        print("  +" + "--" * (self.board_size * 2 - 1) + "+")
        
        # 打印每行内容
        for i in range(self.board_size):
            row = [f"{i:2}|"]  # 行索引
            for j in range(self.board_size):
                if self.board[i, j] == 1:
                    row.append(" X")
                elif self.board[i, j] == 2:
                    row.append(" O")
                else:
                    row.append(" .")
            row.append(" |")
            print("".join(row))
        
        # 打印底部边框
        print("  +" + "--" * (self.board_size * 2 - 1) + "+")


# 卷积神经网络
class GomokuNetV2(nn.Module):
    def __init__(self, board_size):
        super(GomokuNetV2, self).__init__()
        self.board_size = board_size
        
        # 卷积层
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        
        # 动态计算池化后的尺寸和全连接层输入尺寸
        self.pooled_size = (board_size + 1) // 2  # 第一次池化后的尺寸
        self.pooled_size2 = (self.pooled_size + 1) // 2  # 第二次池化后的尺寸
        self.fc1_input_size = 128 * self.pooled_size2 * self.pooled_size2
        
        # 全连接层
        self.fc1 = nn.Linear(self.fc1_input_size, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, board_size * board_size)

    def forward(self, x):
        # 输入形状调整：(batch_size, board_size*board_size) -> (batch_size, 1, board_size, board_size)
        x = x.view(-1, 1, self.board_size, self.board_size)
        
        # 第一次卷积、批归一化、激活和池化
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        
        # 第二次卷积、批归一化、激活和池化
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        
        # 展平特征图
        x = x.view(-1, self.fc1_input_size)
        
        # 全连接层
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class ResidualConvBlock(nn.Module):
    """残差卷积块：增强局部特征提取能力"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 当输入输出通道不同时，用1x1卷积调整维度
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)  # 残差连接
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual  # 残差相加
        return F.relu(x)


class PositionalEncoding(nn.Module):
    """位置编码：为棋盘位置添加空间位置信息（Transformer必备）"""
    def __init__(self, d_model, board_size, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # 适配棋盘尺寸：生成 (board_size^2, d_model) 的位置编码
        self.pe = pe[:board_size*board_size, :, :].transpose(0, 1)  # 形状：(1, N, d_model)，N=棋盘格数

    def forward(self, x):
        # x形状：(batch_size, N, d_model)，N=board_size^2
        x = x + self.pe.to(x.device)  # 叠加位置编码
        return x


class GomokuNetV3(nn.Module):
    """融合残差网络和Transformer的五子棋AI网络，输出格式与V2保持一致"""
    def __init__(self, board_size, channels=64, num_res_blocks=2, num_heads=2, d_model=64):
        super().__init__()
        self.board_size = board_size
        self.n = board_size * board_size
        self.d_model = d_model

        # 1. 输入特征提取：通道数从1改为2（玩家1和玩家2）
        self.input_proj = nn.Conv2d(2, channels, kernel_size=3, padding=1)

        # 2. 残差卷积模块：提取局部特征
        self.res_blocks = nn.Sequential(
            *[ResidualConvBlock(channels, channels) for _ in range(num_res_blocks)]
        )

        # 3. 维度转换：为Transformer准备输入
        self.proj_to_transformer = nn.Linear(channels, d_model)
        self.pos_encoder = PositionalEncoding(d_model, board_size)

        # 4. Transformer编码器：建模全局依赖
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=1)

        # 5. 策略头：预测落子概率（与V3一致）
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 6. 价值头：预测当前局面的胜率
        self.value_head = nn.Sequential(
            nn.Linear(d_model * self.n, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh() # 输出-1到1之间的值，代表胜率
        )

    def forward(self, x):
        # 输入形状：(batch_size, 2, board_size, board_size)
        
        # 1. 输入特征提取
        x = self.input_proj(x)

        # 2. 残差卷积提取局部特征
        x = self.res_blocks(x)

        # 3. 转换为Transformer输入格式
        x = x.flatten(2).transpose(1, 2)
        x = self.proj_to_transformer(x)

        # 4. 叠加位置编码 + Transformer
        x = self.pos_encoder(x)
        transformer_output = self.transformer_encoder(x)
        
        # 5. 策略头输出
        policy_logits = self.policy_head(transformer_output).squeeze(-1)
        
        # 6. 价值头输出
        value_input = transformer_output.flatten(1)
        value = self.value_head(value_input)
        
        # 返回两个输出
        return policy_logits, value.squeeze(-1)

def get_valid_action(logits, board_flat, board_size, epsilon=0.1):
    # 优化后的探索策略：根据棋子数量动态调整相邻位置探索概率
    valid_mask = (board_flat == 0)
    valid_indices = torch.where(valid_mask)[0]
    if valid_indices.numel() == 0:
        return -1

    # 计算当前棋盘上的棋子数量
    total_cells = board_size * board_size
    empty_cells = valid_indices.numel()
    piece_count = total_cells - empty_cells  # 已落子数量
    
    # 计算相邻位置探索的概率：随棋子数量增加从100%线性降至0%
    # 当棋子数量为1时概率100%，棋子充满棋盘时概率0%
    if piece_count <= 1:
        adjacent_prob = 1.0  # 只有1个或0个棋子时，100%从相邻位置探索
    elif piece_count >= total_cells - 1:
        adjacent_prob = 0.0  # 棋盘快满时，0%概率
    else:
        # 线性衰减：(总格子数-1 - 当前棋子数)/(总格子数-2)
        adjacent_prob = (total_cells - 1 - piece_count) / (total_cells - 2)

    # logits 已经被 flatten()，它是一个一维张量
    logits = logits.cpu().flatten()
    valid_indices = valid_indices.cpu()
    
    valid_logits = logits[valid_indices]
    
    # 贪心选择
    greedy_action = valid_indices[torch.argmax(valid_logits)]

    if random.random() < epsilon:
        # 探索模式
        # 收集所有相邻位置的有效落子
        adjacent_actions = []
        board_cpu = board_flat.cpu().reshape(board_size, board_size)
        
        for idx in valid_indices:
            x, y = idx // board_size, idx % board_size
            if is_adjacent_to_piece(board_cpu, x, y, board_size):
                adjacent_actions.append(idx)
        
        # 根据当前棋子数量决定探索相邻位置的概率
        if adjacent_actions and random.random() < adjacent_prob:
            return random.choice(adjacent_actions)
        else:
            return random.choice(valid_indices)
    else:
        # 利用模式
        return greedy_action

def is_adjacent_to_piece(board, x, y, board_size):
    """检查(x,y)位置是否与已有棋子相邻（8个方向）"""
    # 8个方向：上、下、左、右、四个对角线
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        # 检查邻接位置是否在棋盘内且有棋子
        if 0 <= nx < board_size and 0 <= ny < board_size and board[nx, ny] != 0:
            return True
    return False

def load_model_if_exists(model, file_path):
    if os.path.exists(file_path):
        try:
            state = torch.load(file_path, map_location=torch.device('cpu'))
            model.load_state_dict(state)
            print(f"Loaded model weights from {file_path}")
            return True
        except Exception as e:
            print(f"Failed to load model from {file_path}: {e}")
            return False
    else:
        print(f"No saved model weights found at {file_path}")
        return False
