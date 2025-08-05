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

    def reset(self):
        self.board.fill(0)
        self.current_player = 1
        self.winning_line = []
        self.step_count = 0
        return self.board

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

    def calculate_reward(self, x, y):
        # 细化奖励计算，并考虑复合棋形
        player = self.board[x, y]
        opponent = 3 - player
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        total_reward = 0
        live_counts = {2: 0, 3: 0, 4: 0}
        rush_counts = {2: 0, 3: 0, 4: 0}

        for dx, dy in directions:
            count = 1
            forward_blocked = False
            backward_blocked = False

            # 正向计数
            for step in range(1, self.win_condition):
                nx, ny = x + dx * step, y + dy * step
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == player:
                    count += 1
                else:
                    if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == opponent:
                        forward_blocked = True
                    break

            # 反向计数
            for step in range(1, self.win_condition):
                nx, ny = x - dx * step, y - dy * step
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == player:
                    count += 1
                else:
                    if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == opponent:
                        backward_blocked = True
                    break
            
            if count >= 2:
                if not forward_blocked and not backward_blocked:
                    live_counts[min(count, 4)] += 1
                elif forward_blocked != backward_blocked: # 只有一端被阻挡
                    rush_counts[min(count, 4)] += 1
        
        # 复合棋形奖励
        if live_counts[3] >= 2: total_reward += Config.REWARD["双活三"]
        if rush_counts[4] >= 1 and live_counts[3] >= 1: total_reward += Config.REWARD["冲四活三"]
        
        # 单独棋形奖励
        for num in range(2, 5):
            total_reward += live_counts[num] * Config.REWARD[f"live{num}"]
            total_reward += rush_counts[num] * Config.REWARD[f"冲{num}"]

        # 检查是否阻断对手胜利 (逻辑不变)
        temp_board = self.board.copy()
        temp_board[x, y] = opponent
        if self._is_winning_move(temp_board, x, y, opponent):
            total_reward += Config.REWARD["block_win"]

        return total_reward

    def _is_winning_move(self, board, x, y, player):
        # 内部辅助函数：检查指定棋盘上的落子是否胜利
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
        self.board[x, y] = self.current_player
        self.step_count += 1

        # 胜利奖励
        if self.is_winning_move(x, y):
            reward = Config.REWARD["win"]
            return self.current_player, True, reward
        
        # 计算连珠奖励
        reward = self.calculate_reward(x, y)
        
        # 切换玩家
        self.current_player = 3 - self.current_player
        return self.board[x, y], False, reward
    
    def get_state_representation(self):
        # 多通道输入表示
        player1_board = (self.board == 1).astype(np.float32)
        player2_board = (self.board == 2).astype(np.float32)
        return np.stack([player1_board, player2_board], axis=0) # 形状: (2, board_size, board_size)

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


# 卷积神经网络（修复维度匹配问题）
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
        
        # 6. 价值头：预测当前局面的胜率（新增）
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
    # 优化后的探索策略
    valid_mask = (board_flat == 0)
    valid_indices = torch.where(valid_mask)[0]
    if valid_indices.numel() == 0:
        return -1

    # logits 已经被 flatten()，它是一个一维张量。
    logits = logits.cpu().flatten()
    valid_indices = valid_indices.cpu() 
    
    valid_logits = logits[valid_indices]
    
    # 贪心选择
    greedy_action = valid_indices[torch.argmax(valid_logits)]

    if random.random() < epsilon:
        # 探索模式
        # 优先在相邻位置中选择，如果没找到，则在所有空位中选择
        adjacent_actions = []
        board_cpu = board_flat.cpu().reshape(board_size, board_size)
        
        for idx in valid_indices:
            x, y = idx // board_size, idx % board_size
            if is_adjacent_to_piece(board_cpu, x, y, board_size):
                adjacent_actions.append(idx)
        
        if adjacent_actions and random.random() < 0.9: # 90%的概率在相邻位置探索
            return random.choice(adjacent_actions)
        else: # 10%的概率在所有空位中探索
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
        model.load_state_dict(torch.load(file_path))
        print(f"Loaded model weights from {file_path}")
    else:
        print(f"No saved model weights found at {file_path}")