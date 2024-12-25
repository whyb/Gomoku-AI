import os
import torch
import torch.nn as nn
import numpy as np
import random

BOARD_SIZE = 15 # 定义棋盘大小
WIN_CONDITION = 5 # 胜利条件

# 游戏环境
class Gomoku:
    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.current_player = 1
        self.winning_line = []

    def reset(self):
        self.board.fill(0)
        self.current_player = 1
        self.winning_line = []

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

    def step(self, action):
        x, y = action // BOARD_SIZE, action % BOARD_SIZE
        if self.board[x, y] != 0:
            return -1, True
        self.board[x, y] = self.current_player
        if self.is_winning_move(x, y):
            return self.current_player, True
        self.current_player = 3 - self.current_player
        
        # 中间奖励score机制
        score = self.evaluate_board()
        return score, False

    def evaluate_board(self):
        # TODO 实现一个函数评估当前棋盘状态，返回一个分数
        return 0  # 这里需要根据实际情况实现具体的评估逻辑

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


# def get_valid_action(logits, board, epsilon=0.1):
#     valid_actions = []
#     for i in range(BOARD_SIZE * BOARD_SIZE):
#         x, y = i // BOARD_SIZE, i % BOARD_SIZE
#         if board[x, y] == 0:
#             valid_actions.append((logits[i], i))
#     valid_actions.sort(reverse=True)

#     if random.random() < epsilon:
#         return random.choice(valid_actions)[1] if valid_actions else -1
#     else:
#         return valid_actions[0][1] if valid_actions else -1


def get_valid_action(logits, board, epsilon=0.1):
    logits = logits.flatten()  # 展平logits，确保其形状为(BOARD_SIZE * BOARD_SIZE,)
    valid_actions = [(logits[i].item(), i) for i in range(BOARD_SIZE * BOARD_SIZE) if board[i // BOARD_SIZE, i % BOARD_SIZE] == 0]
    valid_actions.sort(reverse=True, key=lambda x: x[0])  # 根据 logits 从大到小排序

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
