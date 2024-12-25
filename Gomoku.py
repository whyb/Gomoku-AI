import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# 定义棋盘大小和胜利条件
BOARD_SIZE = 15
WIN_CONDITION = 5

# 标志位，控制是否使用GPU
USE_GPU = torch.cuda.is_available()
print("USE_GPU:", USE_GPU)

# 游戏环境
class Gomoku:
    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.current_player = 1

    def reset(self):
        self.board.fill(0)
        self.current_player = 1

    def is_winning_move(self, x, y):
        # 检查五子连珠的胜利条件
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
        for dx, dy in directions:
            if count_consecutive(player, dx, dy) + count_consecutive(player, -dx, -dy) >= WIN_CONDITION - 1:
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
        return 0, False

    def print_board(self):
        for row in self.board:
            print(' '.join(['X' if x == 1 else 'O' if x == 2 else '.' for x in row]))
        print()

# 神经网络模型
class GomokuNet(nn.Module):
    def __init__(self):
        super(GomokuNet, self).__init__()
        self.fc1 = nn.Linear(BOARD_SIZE * BOARD_SIZE, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, BOARD_SIZE * BOARD_SIZE)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_valid_action(logits, board, epsilon=0.1):
    valid_actions = []
    for i in range(BOARD_SIZE * BOARD_SIZE):
        x, y = i // BOARD_SIZE, i % BOARD_SIZE
        if board[x, y] == 0:
            valid_actions.append((logits[i], i))
    valid_actions.sort(reverse=True)

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

# 训练过程
def train():
    device = torch.device("cuda" if USE_GPU else "cpu")
    env = Gomoku()
    model1 = GomokuNet().to(device)
    model2 = GomokuNet().to(device)  # 作为陪练模型
    optimizer1 = optim.Adam(model1.parameters())
    optimizer2 = optim.Adam(model2.parameters())
    criterion = nn.CrossEntropyLoss()

    # 尝试加载模型权重
    load_model_if_exists(model1, 'gobang_best_model.pth')

    epsilon = 0.1  # 设置Epsilon-Greedy策略中的epsilon值

    for round in range(10000):  # 增加训练回合数
        env.reset()
        done = False

        while not done:
            state = torch.FloatTensor(env.board.flatten()).to(device)

            if env.current_player == 1:
                logits = model1(state)
                optimizer = optimizer1
                action = get_valid_action(logits, env.board, epsilon)
            else:
                logits = model2(state)
                optimizer = optimizer2
                action = get_valid_action(logits, env.board, 0.3)  # Player2 增加随机性

            if action == -1:
                break
            reward, done = env.step(action)

            if reward != -1:
                target = torch.LongTensor([action]).to(device)
                loss = criterion(logits.unsqueeze(0), target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done and reward != 0:
                print(f"Round {round}, Player {reward} wins")
                env.print_board()  # 打印棋盘最终状态

        # 每一千个回合重置Player2
        if (round + 1) % 1000 == 0:
            torch.save(model1.state_dict(), f'gobang_model_player1_{round + 1}.pth')
            model2 = GomokuNet().to(device)  # 重置Player2
            optimizer2 = optim.Adam(model2.parameters())

    # 保存最终的Player1模型
    torch.save(model1.state_dict(), 'gobang_best_model.pth')

train()
