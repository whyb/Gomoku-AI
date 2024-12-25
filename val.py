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
    load_model_if_exists(model1, 'gobang_model_player1_12000.pth')
    load_model_if_exists(model2, 'gobang_model_player1_9000.pth')

    player1_win_count = 0
    player2_win_count = 0

    for round in range(1000):
        env.reset()
        done = False
        current_model = model1

        while not done:
            state = torch.FloatTensor(env.board.flatten()).unsqueeze(0).to(device)  # 增加batch维度
            if env.current_player == 1:
                logits = model1(state)
                action = get_valid_action(logits.cpu().detach().numpy(), env.board, 0.01)
            else:
                logits = model2(state)
                action = get_valid_action(logits.cpu().detach().numpy(), env.board, 0.4)  # Player2 增加随机性

            if action == -1:
                print("No valid actions available. Ending the game.")
                break
            reward, done = env.step(action)
            #if NEED_PRINT_BOARD:  # 打印中间状态
                #env.print_board()
            if done:
                if reward == 1:
                    player1_win_count += 1
                elif reward == 2:
                    player2_win_count += 1
                total_game_count = player1_win_count + player2_win_count
                player1_win_rate = (player1_win_count / total_game_count) * 100 if total_game_count > 0 else 0
                player2_win_rate = (player2_win_count / total_game_count) * 100 if total_game_count > 0 else 0
                print(f"Validator Round {round},\tPlayer {reward} win!\tPlayer 1 win rate: {player1_win_rate:.2f}%, Player 2 win rate: {player2_win_rate:.2f}%")
                if NEED_PRINT_BOARD:
                    env.print_board()
                break
            current_model = model2 if current_model == model1 else model1

if __name__ == "__main__":
    validator()
