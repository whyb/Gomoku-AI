import os
import math
import torch
import torch.nn as nn
import numpy as np
import random

BOARD_SIZE = 8  # 定义棋盘大小
WIN_CONDITION = 5  # 胜利条件

# 游戏环境
class Gomoku:
    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.current_player = 1
        self.winning_line = []
        self.step_count = 0  # 记录步数

    def reset(self):
        self.board.fill(0)
        self.current_player = 1
        self.winning_line = []
        self.step_count = 0  # 重置步数

    def is_winning_move(self, x, y):
        # 检查五子连珠的胜利条件
        def count_consecutive(player, dx, dy):
            count = 0
            line = [(x, y)]
            for step in range(1, WIN_CONDITION):
                nx, ny = x + dx * step, y + dy * step
                if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and self.board[nx, ny] == player:
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
            if count1 + count2 >= WIN_CONDITION - 1:
                self.winning_line = line1 + line2[1:]
                return True
        return False

    def calculate_reward(self, x, y):
        def count_consecutive(player, dx, dy):
            count = 0
            for step in range(1, WIN_CONDITION):
                nx, ny = x + dx * step, y + dy * step
                if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and self.board[nx, ny] == player:
                    count += 1
                else:
                    break
            return count

        player = self.board[x, y]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        rewards = {2: 10, 3: 100, 4: 500, 5: 10000}
        total_reward = 0
        for dx, dy in directions:
            count1 = count_consecutive(player, dx, dy)
            count2 = count_consecutive(player, -dx, -dy)
            total_count = count1 + count2 + 1
            if total_count in rewards:
                total_reward += rewards[total_count]
        return total_count, total_reward

    def step(self, action):
        # 解析动作坐标, 将传入的 action 转换为棋盘上的坐标
        x, y = action // BOARD_SIZE, action % BOARD_SIZE
        # 检查目标位置是否已被占用
        if self.board[x, y] != 0:
            return -1, True, 0, 0
        # 落子
        self.board[x, y] = self.current_player
        self.step_count += 1  # 步数加一
        #bonus_reward = (BOARD_SIZE * BOARD_SIZE - self.step_count) * 10  # 步数越少奖励越多
        #bonus_reward = bonus_reward if self.current_player == 1 else -bonus_reward
        bonus_reward = 0
        if self.is_winning_move(x, y):
            base_reward = 10000
            total_reward = base_reward + bonus_reward
            return self.current_player, True, total_reward, 5
        
        # 计算局部奖励
        total_count, total_reward = self.calculate_reward(x, y)
        total_reward = total_reward + bonus_reward
        
        # 切换到另外一个棋手 1 变 2，2 变 1
        self.current_player = 3 - self.current_player
        return self.board[x, y], False, total_reward, total_count

    def simulate_move(self, action):
        x, y = action // BOARD_SIZE, action % BOARD_SIZE
        if self.board[x, y] != 0:
            return False
        self.board[x, y] = self.current_player
        self.current_player = 3 - self.current_player
        return True

    def evaluate_state(self):
        return self.evaluate_board()

    def print_board(self):
        for i in range(BOARD_SIZE):
            row = ''
            for j in range(BOARD_SIZE):
                if (i, j) in self.winning_line:
                    row += '\033[91mX\033[0m ' if self.board[i, j] == 1 else '\033[91mO\033[0m ' if self.board[i, j] == 2 else '. '
                else:
                    row += 'X ' if self.board[i, j] == 1 else 'O ' if self.board[i, j] == 2 else '. '
            print(row)
        print()

# Version #1
class GomokuNetV1(nn.Module):
    def __init__(self):
        super(GomokuNetV1, self).__init__()
        self.fc1 = nn.Linear(BOARD_SIZE * BOARD_SIZE, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, BOARD_SIZE * BOARD_SIZE)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 卷积神经网络（CNN）
class GomokuNetV2(nn.Module):
    def __init__(self):
        super(GomokuNetV2, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * BOARD_SIZE * BOARD_SIZE, 256)
        self.fc2 = nn.Linear(256, BOARD_SIZE * BOARD_SIZE)

    def forward(self, x):
        x = torch.relu(self.conv1(x.view(-1, 1, BOARD_SIZE, BOARD_SIZE)))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 128 * BOARD_SIZE * BOARD_SIZE)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_valid_action(logits, board, epsilon=0.1):
    logits = logits.flatten()  # 展平 logits，确保其形状为(BOARD_SIZE * BOARD_SIZE,)
    valid_actions = [(logits[i].item(), i) for i in range(BOARD_SIZE * BOARD_SIZE) if board[i // BOARD_SIZE, i % BOARD_SIZE] == 0]
    valid_actions.sort(reverse=True, key=lambda x: x[0])  # 根据 logits 从大到小排序

    '''
    转换为棋盘坐标：
    行：row = action // BOARD_SIZE，对于 action = 8，row = 8 // 3 = 2
    列：col = action % BOARD_SIZE，对于 action = 8，col = 8 % 3 = 2
    '''
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
