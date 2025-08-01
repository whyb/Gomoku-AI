import torch
import argparse
from model import Gomoku, GomokuNetV2, load_model_if_exists, get_valid_action
from config import Config, update_config_from_cli

NEED_PRINT_BOARD = True

def validator():
    parser = argparse.ArgumentParser()
    parser.add_argument("--board_size", type=int, help="Size of the game board")
    args = parser.parse_args()
    config = update_config_from_cli(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Gomoku(config.BOARD_SIZE)
    model1 = GomokuNetV2(config.BOARD_SIZE).to(device)
    model2 = GomokuNetV2(config.BOARD_SIZE).to(device)

    load_model_if_exists(model1, 'gobang_best_model.pth')
    load_model_if_exists(model2, 'gobang_best_model.pth')

    player1_win_count = 0
    player2_win_count = 0

    for round in range(1000):
        env.reset()
        done = False
        while not done:
            state = torch.FloatTensor(env.board.flatten()).unsqueeze(0).to(device)
            if env.current_player == 1:
                logits = model1(state)
                action = get_valid_action(logits.cpu().detach().numpy(), env.board, 0.0001)
            else:
                logits = model2(state)
                action = get_valid_action(logits.cpu().detach().numpy(), env.board, 0.1)

            if action == -1:
                break
            current_player, done, reward, _ = env.step(action)
            if done:
                if current_player == 1:
                    player1_win_count += 1
                else:
                    player2_win_count += 1
                total = player1_win_count + player2_win_count
                rate1 = (player1_win_count / total) * 100 if total else 0
                rate2 = (player2_win_count / total) * 100 if total else 0
                print(f"Round {round},\tPlayer {current_player} win!\tWin rate: P1 {rate1:.2f}%, P2 {rate2:.2f}%")
                if NEED_PRINT_BOARD:
                    env.print_board()
                break

if __name__ == "__main__":
    validator()
