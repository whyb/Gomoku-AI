import torch
from model import Gomoku, GomokuNetV2, load_model_if_exists, get_valid_action

NEED_PRINT_BOARD = True  # 打印棋盘
# 标志位，控制是否使用GPU
USE_GPU = torch.cuda.is_available()
print("USE_GPU:", USE_GPU)

def validator():
    device = torch.device("cuda" if USE_GPU else "cpu")
    env = Gomoku()
    model1 = GomokuNetV2().to(device)
    model2 = GomokuNetV2().to(device)

    # 加载模型权重
    load_model_if_exists(model1, 'gobang_best_model.pth')
    load_model_if_exists(model2, 'gobang_best_model.pth')

    env.reset()
    done = False
    current_model = model1

    while not done:
        state = torch.FloatTensor(env.board.flatten()).to(device)
        logits = current_model(state)
        action = get_valid_action(logits.cpu().detach().numpy(), env.board)
        if action == -1:
            print("No valid actions available. Ending the game.")
            break
        reward, done = env.step(action)
        if NEED_PRINT_BOARD:  # 打印中间状态
            env.print_board()
        if done:
            print(f"Player {reward} win!")
            break
        current_model = model2 if current_model == model1 else model1

if __name__ == "__main__":
    validator()
