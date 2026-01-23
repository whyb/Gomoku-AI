import argparse
import time
import random
import torch
from concurrent.futures import ProcessPoolExecutor
from model import Gomoku, get_valid_action
from model_dy import GomokuNetDyn, load_model_if_exists
from config import Config, update_config_from_cli

def play_round(board_size, win_condition, model_state_dict):
    env = Gomoku(board_size, win_condition)
    model = GomokuNetDyn()
    model.load_state_dict(model_state_dict)
    model.eval()
    total_cells = board_size * board_size
    first_player = random.choice([1, 2])
    env.current_player = first_player
    steps = 0
    while True:
        state = torch.FloatTensor(env.get_state_representation()).unsqueeze(0)
        if env.current_player == 1:
            with torch.no_grad():
                logits, _ = model(state)
            board_flat = torch.tensor(env.board.flatten())
            action = get_valid_action(logits, board_flat, board_size, epsilon=0.1)
        else:
            valid = [i * board_size + j for i in range(board_size) for j in range(board_size) if env.board[i][j] == 0]
            action = random.choice(valid) if valid else -1
        next_player, done, reward = env.step(action)
        steps += 1
        if done or steps >= total_cells:
            return first_player, env.current_player

def validator():
    parser = argparse.ArgumentParser()
    parser.add_argument("--board_size", type=int, default=15, help="Size of the game board")
    parser.add_argument("--win_condition", type=int, default=5, help="Number of consecutive stones to win")
    parser.add_argument("--total_rounds", type=int, default=200, help="Number of rounds")
    parser.add_argument("--model_path", type=str, default='gobang_best_model_dy.pth', help="Path to the weight file")
    args = parser.parse_args()
    update_config_from_cli(args)

    model = GomokuNetDyn()
    load_model_if_exists(model, args.model_path)
    state = {k: v.cpu() for k, v in model.state_dict().items()}

    p1_first = 0
    p1_second = 0
    with ProcessPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(play_round, Config.BOARD_SIZE, Config.WIN_CONDITION, state) for _ in range(args.total_rounds)]
        for f in futures:
            first_player, winner = f.result()
            if winner == 1:
                if first_player == 1:
                    p1_first += 1
                else:
                    p1_second += 1
    total = args.total_rounds
    print(f"总局数: {total}")
    print(f"P1先手胜局: {p1_first}")
    print(f"P1后手胜局: {p1_second}")

if __name__ == "__main__":
    validator()

