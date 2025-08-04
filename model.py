import os
import torch
import torch.nn as nn
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
