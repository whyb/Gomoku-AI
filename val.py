import torch
import argparse
import random
import os
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager

from model import Gomoku, GomokuNetV3, load_model_if_exists, get_valid_action
from config import Config, update_config_from_cli

NEED_PRINT_BOARD = False

@contextmanager
def suppress_stdout():
    """
    一个上下文管理器，用于临时抑制标准输出（打印）。
    """
    original_stdout = sys.stdout
    with open(os.devnull, 'w') as devnull:
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = original_stdout

# 封装了单局游戏的核心逻辑
def play_one_game(board_size, win_condition, model1_path, model2_path, first_player_id):
    """
    运行一局五子棋游戏，并返回结果。
    这个函数会在每个独立的进程中执行。

    返回一个元组：(赢家ID, 先手玩家ID)
    """
    device = torch.device("cpu")
    env = Gomoku(board_size, win_condition)
    
    model1 = GomokuNetV3(board_size).to(device)
    model2 = GomokuNetV3(board_size).to(device)

    with suppress_stdout():
        load_model_if_exists(model1, model1_path)
        load_model_if_exists(model2, model2_path)
    
    model1.eval()
    model2.eval()
    
    env.reset()
    env.current_player = first_player_id

    done = False
    max_steps = board_size * board_size

    while not done and env.step_count < max_steps:
        # 特殊处理第一步落子：使用纯随机值
        if env.step_count == 0:
            valid_actions = np.where(env.board.flatten() == 0)[0]
            if len(valid_actions) > 0:
                action = random.choice(valid_actions)
            else:
                break
        else:
            state = env.get_state_representation()
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            with torch.no_grad():
                if env.current_player == 1:
                    logits, _ = model1(state_tensor)
                    board_flat = torch.tensor(env.board.flatten(), device=device)
                    action = get_valid_action(logits, board_flat, board_size, epsilon=0.0001)
                else: # env.current_player == 2
                    logits, _ = model2(state_tensor)
                    board_flat = torch.tensor(env.board.flatten(), device=device)
                    action = get_valid_action(logits, board_flat, board_size, epsilon=0.05)

        if action == -1:
            break
            
        winner, done, _ = env.step(action)
    
    if done:
        return winner, first_player_id
    else:
        return 0, first_player_id # 平局

def validator():
    parser = argparse.ArgumentParser()
    parser.add_argument("--board_size", type=int, help="Size of the game board")
    parser.add_argument("--win_condition", type=int, help="Number of consecutive stones to win")
    parser.add_argument("--total_rounds", type=int, default=10000, 
                        help="Total number of rounds to play (default: 10000)")
    parser.add_argument("--print_interval", type=int, default=1000, 
                        help="Interval for printing progress (default: 1000)")
    args = parser.parse_args()
    config = update_config_from_cli(args)

    # 新增四个统计变量，分别记录先手和后手的胜局数
    player1_win_first = 0
    player1_win_second = 0
    player2_win_first = 0
    player2_win_second = 0
    draw_count = 0
    
    # 统计总局数，以计算胜率
    first_player1_rounds = 0
    first_player2_rounds = 0
    
    print(f"===== 验证配置 =====")
    print(f"棋盘尺寸: {config.BOARD_SIZE}x{config.BOARD_SIZE}")
    print(f"胜利条件: 连{config.WIN_CONDITION}子")
    print(f"总局数: {args.total_rounds}")
    print(f"打印间隔: {args.print_interval}局")
    print("开局第一步为随机落子，以增加对局多样性。")
    print(f"===================\n")
    
    model1_path = 'gobang_best_model.pth'
    model2_path = 'gobang_cainiao_model.pth'

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = []
        for round_num in range(args.total_rounds):
            first_player_id = 1 if random.random() < 0.5 else 2
            
            future = executor.submit(
                play_one_game, 
                config.BOARD_SIZE, 
                config.WIN_CONDITION, 
                model1_path, 
                model2_path,
                first_player_id
            )
            futures.append(future)

        completed_rounds = 0
        for future in as_completed(futures):
            # 接收包含赢家和先手玩家ID的元组
            winner, first_player_id = future.result()
            
            if first_player_id == 1:
                first_player1_rounds += 1
            else:
                first_player2_rounds += 1
            
            if winner == 1:
                if first_player_id == 1:
                    player1_win_first += 1
                else:
                    player1_win_second += 1
            elif winner == 2:
                if first_player_id == 2:
                    player2_win_first += 1
                else:
                    player2_win_second += 1
            else:
                draw_count += 1
            
            completed_rounds += 1
            
            if completed_rounds % args.print_interval == 0:
                print(f"已完成 {completed_rounds}/{args.total_rounds} 局")
                print(f"--- Player 1 (执黑) 统计 ---")
                if first_player1_rounds > 0:
                    win_rate_first = (player1_win_first / first_player1_rounds) * 100
                    print(f"  先手胜率: {win_rate_first:.2f}% ({player1_win_first}/{first_player1_rounds})")
                if first_player2_rounds > 0:
                    win_rate_second = (player1_win_second / first_player2_rounds) * 100
                    print(f"  后手胜率: {win_rate_second:.2f}% ({player1_win_second}/{first_player2_rounds})")
                
                print(f"--- Player 2 (执白) 统计 ---")
                if first_player2_rounds > 0:
                    win_rate_first = (player2_win_first / first_player2_rounds) * 100
                    print(f"  先手胜率: {win_rate_first:.2f}% ({player2_win_first}/{first_player2_rounds})")
                if first_player1_rounds > 0:
                    win_rate_second = (player2_win_second / first_player1_rounds) * 100
                    print(f"  后手胜率: {win_rate_second:.2f}% ({player2_win_second}/{first_player1_rounds})")
                
                print(f"平局数: {draw_count}\n")
    
    print("\n===== 验证完成 =====")
    total_rounds = player1_win_first + player1_win_second + player2_win_first + player2_win_second + draw_count
    
    print(f"总场次: {total_rounds}")
    print(f"先手总局数 (Player1先手): {first_player1_rounds}")
    print(f"后手总局数 (Player2先手): {first_player2_rounds}")
    
    print("\n--- Player 1 (执黑) 胜率 ---")
    if first_player1_rounds > 0:
        win_rate_first = (player1_win_first / first_player1_rounds) * 100
        print(f"  先手胜率: {win_rate_first:.2f}% ({player1_win_first} / {first_player1_rounds})")
    if first_player2_rounds > 0:
        win_rate_second = (player1_win_second / first_player2_rounds) * 100
        print(f"  后手胜率: {win_rate_second:.2f}% ({player1_win_second} / {first_player2_rounds})")

    print("\n--- Player 2 (执白) 胜率 ---")
    if first_player2_rounds > 0:
        win_rate_first = (player2_win_first / first_player2_rounds) * 100
        print(f"  先手胜率: {win_rate_first:.2f}% ({player2_win_first} / {first_player2_rounds})")
    if first_player1_rounds > 0:
        win_rate_second = (player2_win_second / first_player1_rounds) * 100
        print(f"  后手胜率: {win_rate_second:.2f}% ({player2_win_second} / {first_player1_rounds})")

    print(f"\n平局数: {draw_count}")

if __name__ == "__main__":
    validator()