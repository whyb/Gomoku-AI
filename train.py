import torch
import torch.nn as nn
import torch.optim as optim
from model import Gomoku, GomokuNetV2, get_valid_action, load_model_if_exists

NEED_PRINT_BOARD = False  # 打印棋盘
# 标志位，控制是否使用GPU
USE_GPU = torch.cuda.is_available()
print("USE_GPU:", USE_GPU)

def train():
    device = torch.device("cuda" if USE_GPU else "cpu")
    env = Gomoku()
    model1 = GomokuNetV2().to(device)
    model2 = GomokuNetV2().to(device)  # 作为陪练模型
    optimizer1 = optim.Adam(model1.parameters())
    optimizer2 = optim.Adam(model2.parameters())
    criterion = nn.CrossEntropyLoss()

    # 尝试加载模型权重
    load_model_if_exists(model1, 'gobang_best_model.pth')
    load_model_if_exists(model2, 'gobang_best_model.pth')

    epsilon = 0.1  # 设置Epsilon-Greedy策略中的epsilon值

    for round in range(10000):  # 增加训练回合数
        env.reset()
        done = False

        while not done:
            state = torch.FloatTensor(env.board.flatten()).unsqueeze(0).to(device)  # 增加batch维度

            if env.current_player == 1:
                logits = model1(state)
                optimizer = optimizer1
                action = get_valid_action(logits.cpu().detach().numpy(), env.board, epsilon)
            else:
                logits = model2(state)
                optimizer = optimizer2
                action = get_valid_action(logits.cpu().detach().numpy(), env.board, 0.3)  # Player2 增加随机性

            if action == -1:
                break
            reward, done = env.step(action)

            if reward != -1:
                target = torch.LongTensor([action]).to(device)
                loss = criterion(logits.view(1, -1), target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done and reward != 0:
                print(f"Round {round}, Player {reward} win!")
                if NEED_PRINT_BOARD:
                    env.print_board()  # 打印棋盘最终状态

        # 每一千个回合重置Player2
        if (round + 1) % 1000 == 0:
            torch.save(model1.state_dict(), f'gobang_model_player1_{round + 1}.pth')
            model2 = GomokuNetV2().to(device)  # 重置Player2
            optimizer2 = optim.Adam(model2.parameters())
            load_model_if_exists(model2, f'gobang_model_player1_{round + 1}.pth')

    # 保存最终的Player1模型
    torch.save(model1.state_dict(), 'gobang_best_model.pth')

if __name__ == "__main__":
    train()
