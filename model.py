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

    def is_winning_move(self, x, y):
        # 检查胜利条件（复用原逻辑，适配动态尺寸）
        def count_consecutive(player, dx, dy):
            count = 0
            line = [(x, y)]
            for step in range(1, self.win_condition):
                nx, ny = x + dx * step, y + dy * step
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == player:
                    count += 1
                    line.append((nx, ny))
                else:
                    break
            return count, line

        player = self.board[x, y]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dx, dy in directions:
            count1, line1 = count_consecutive(player, dx, dy)
            count2, line2 = count_consecutive(player, -dx, -dy)
            if count1 + count2 >= self.win_condition - 1:
                self.winning_line = line1 + line2[1:]
                return True
        return False

    def calculate_reward(self, x, y):
        # 细化奖励计算（区分活/冲状态）
        player = self.board[x, y]
        opponent = 3 - player
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        total_reward = 0

        # 检查自身连珠奖励
        for dx, dy in directions:
            # 正向计数（同方向）
            same_forward = 0
            for step in range(1, self.win_condition):
                nx, ny = x + dx * step, y + dy * step
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == player:
                    same_forward += 1
                else:
                    break
            # 反向计数（反方向）
            same_backward = 0
            for step in range(1, self.win_condition):
                nx, ny = x - dx * step, y - dy * step
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == player:
                    same_backward += 1
                else:
                    break
            total_same = same_forward + same_backward + 1  # 包含当前子

            # 检查两端是否被阻挡（判断"活"或"冲"）
            forward_blocked = False
            nx, ny = x + dx * (same_forward + 1), y + dy * (same_forward + 1)
            if not (0 <= nx < self.board_size and 0 <= ny < self.board_size) or self.board[nx, ny] == opponent:
                forward_blocked = True

            backward_blocked = False
            nx, ny = x - dx * (same_backward + 1), y - dy * (same_backward + 1)
            if not (0 <= nx < self.board_size and 0 <= ny < self.board_size) or self.board[nx, ny] == opponent:
                backward_blocked = True

            # 活棋：两端至少一端未被阻挡；冲棋：两端都被阻挡
            if total_same == 4:
                if forward_blocked and backward_blocked:
                    total_reward += Config.REWARD["冲四"]
                else:
                    total_reward += Config.REWARD["live4"]
            elif total_same == 3:
                if forward_blocked and backward_blocked:
                    total_reward += Config.REWARD["冲三"]
                else:
                    total_reward += Config.REWARD["live3"]
            elif total_same == 2:
                if forward_blocked and backward_blocked:
                    total_reward += Config.REWARD["冲二"]
                else:
                    total_reward += Config.REWARD["live2"]

        # 检查是否阻断对手胜利
        temp_board = self.board.copy()
        temp_board[x, y] = opponent  # 模拟对手落子
        if self._is_winning_move(temp_board, x, y, opponent):
            total_reward += Config.REWARD["block_win"]

        return total_same, total_reward

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
            return -1, True, 0, 0  # 无效落子
        self.board[x, y] = self.current_player
        self.step_count += 1

        # 胜利奖励
        if self.is_winning_move(x, y):
            total_reward = Config.REWARD["win"]
            return self.current_player, True, total_reward, self.win_condition
        
        # 计算连珠奖励
        total_count, total_reward = self.calculate_reward(x, y)
        
        # 切换玩家
        self.current_player = 3 - self.current_player
        return self.board[x, y], False, total_reward, total_count

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
        self.n = board_size * board_size  # 棋盘总格子数
        self.d_model = d_model

        # 1. 输入特征提取：将棋盘状态映射到高维特征
        self.input_proj = nn.Conv2d(1, channels, kernel_size=3, padding=1)  # (1, B, B) -> (C, B, B)

        # 2. 残差卷积模块：提取局部特征（连子、形状等）
        self.res_blocks = nn.Sequential(
            *[ResidualConvBlock(channels, channels) for _ in range(num_res_blocks)]
        )

        # 3. 维度转换：为Transformer准备输入（ flatten + 线性投影）
        self.proj_to_transformer = nn.Linear(channels, d_model)  # 每个格子的特征 -> d_model维度
        self.pos_encoder = PositionalEncoding(d_model, board_size)  # 位置编码

        # 4. Transformer编码器：建模全局依赖（任意格子间的关系）
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True  # 设为True，输入形状为 (batch, seq_len, d_model)
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=1)

        # 5. 策略头：预测落子概率（仅返回策略logits，与V2保持一致）
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 每个格子的落子分数
        )

    def forward(self, x):
        # 输入形状：(batch_size, board_size*board_size) -> 转换为 (batch_size, 1, board_size, board_size)
        x = x.view(-1, 1, self.board_size, self.board_size)  # (B, 1, B_size, B_size)

        # 1. 输入特征提取
        x = self.input_proj(x)  # (B, C, B_size, B_size)

        # 2. 残差卷积提取局部特征
        x = self.res_blocks(x)  # (B, C, B_size, B_size)

        # 3. 转换为Transformer输入格式：(B, N, d_model)，N=B_size^2
        x = x.flatten(2)  # (B, C, N) -> 展平为 (B, C, N)，N=B_size^2
        x = x.transpose(1, 2)  # (B, N, C) -> 每个格子作为一个序列元素
        x = self.proj_to_transformer(x)  # (B, N, d_model)

        # 4. 叠加位置编码 + Transformer
        x = self.pos_encoder(x)  # 叠加位置信息
        x = self.transformer_encoder(x)  # (B, N, d_model)，输出全局特征

        # 5. 策略头输出：(B, N) -> 每个位置的落子概率，仅返回这一项与V2保持一致
        policy_logits = self.policy_head(x).squeeze(-1)  # (B, N)
        
        # 确保输出形状与V2完全一致：(batch_size, board_size*board_size)
        return policy_logits

def get_valid_action(logits, board, epsilon=0.1):
    board_size = board.shape[0]
    logits = logits.flatten()
    valid_actions = [(logits[i].item(), i) for i in range(board_size * board_size) if board[i // board_size, i % board_size] == 0]
    valid_actions.sort(reverse=True, key=lambda x: x[0])
    if random.random() < epsilon:
        return random.choice(valid_actions)[1] if valid_actions else -1
    else:
        return valid_actions[0][1] if valid_actions else -1

def load_model_if_exists(model, file_path):
    if os.path.exists(file_path):
        model.load_state_dict(torch.load(file_path))
        print(f"Loaded model weights from {file_path}")
    else:
        print(f"No saved model weights found at {file_path}")
