import torch
from model import Gomoku, GomokuNet, load_model_if_exists

NEED_PRINT_BOARD = True # 打印棋盘
# 标志位，控制是否使用GPU
USE_GPU = torch.cuda.is_available()
print("USE_GPU:", USE_GPU)

def validator():
    device = torch.device("cuda" if USE_GPU else "cpu")
    env = Gomoku()
    model = GomokuNet().to(device)

    # 加载模型权重
    load_model_if_exists(model, 'gobang_best_model.pth')

    env.reset()
    done = False
    while not done:
        state = torch.FloatTensor(env.board.flatten()).to(device)
        logits = model(state)
        action = torch.argmax(logits).item()
        reward, done = env.step(action)
        if done:
            print(f"Player {reward} win!")
            if NEED_PRINT_BOARD:
                env.print_board()  # 打印棋盘最终状态

if __name__ == "__main__":
    validator()
