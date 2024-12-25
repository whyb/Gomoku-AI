import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 定义棋盘大小和胜利条件
BOARD_SIZE = 15
WIN_CONDITION = 5

# 标志位，控制是否使用GPU
USE_GPU = torch.cuda.is_available()

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

def get_valid_action(logits, board):
    valid_actions = []
    for i in range(BOARD_SIZE * BOARD_SIZE):
        x, y = i // BOARD_SIZE, i % BOARD_SIZE
        if board[x, y] == 0:
            valid_actions.append((logits[i], i))
    valid_actions.sort(reverse=True)
    return valid_actions[0][1] if valid_actions else -1

# 训练过程
def train():
    device = torch.device("cuda" if USE_GPU else "cpu")
    env = Gomoku()
    model1 = GomokuNet().to(device)
    model2 = GomokuNet().to(device)  # 作为陪练模型
    optimizer1 = optim.Adam(model1.parameters())
    optimizer2 = optim.Adam(model2.parameters())
    criterion = nn.CrossEntropyLoss()

    for round in range(1000):
        env.reset()
        done = False

        while not done:
            state = torch.FloatTensor(env.board.flatten()).to(device)

            if env.current_player == 1:
                logits = model1(state)
                optimizer = optimizer1
            else:
                logits = model2(state)
                optimizer = optimizer2

            action = get_valid_action(logits, env.board)
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
                print(f"round {round}, Player {reward} wins")

        # 每一千个回合保存一次模型
        if (round + 1) % 1000 == 0:
            torch.save(model1.state_dict(), f'gobang_model_player1_{round + 1}.pth')

    # 保存最终的Player1模型
    torch.save(model1.state_dict(), 'gobang_best_model.pth')

train()
