import torch
import argparse
from model import Gomoku, GomokuNetV2, load_model_if_exists, get_valid_action
from config import Config, update_config_from_cli

NEED_PRINT_BOARD = True

def validator():
    # 1. 增加--win_condition命令行参数支持
    parser = argparse.ArgumentParser()
    parser.add_argument("--board_size", type=int, help="Size of the game board")
    parser.add_argument("--win_condition", type=int, help="Number of consecutive stones to win")
    parser.add_argument("--total_rounds", type=int, default=10000, 
                      help="Total number of rounds to play (default: 10000)")
    parser.add_argument("--print_interval", type=int, default=1000, 
                      help="Interval for printing progress (default: 1000)")
    args = parser.parse_args()
    config = update_config_from_cli(args)  # 从命令行更新配置

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 2. 初始化环境时传入胜利条件，与训练保持一致
    env = Gomoku(config.BOARD_SIZE, config.WIN_CONDITION)
    model1 = GomokuNetV2(config.BOARD_SIZE).to(device)
    model2 = GomokuNetV2(config.BOARD_SIZE).to(device)

    # 加载训练好的模型（4x4棋盘的模型）
    load_model_if_exists(model1, 'gobang_best_model.pth')
    load_model_if_exists(model2, 'gobang_model_player1_10000.pth')

    player1_win_count = 0
    player2_win_count = 0
    draw_count = 0  # 新增：统计平局次数

    # 打印验证配置信息
    print(f"===== 验证配置 =====")
    print(f"棋盘尺寸: {config.BOARD_SIZE}x{config.BOARD_SIZE}")
    print(f"胜利条件: 连{config.WIN_CONDITION}子")
    print(f"总局数: {args.total_rounds}")
    print(f"打印间隔: {args.print_interval}局")
    print(f"使用设备: {device}")
    print(f"===================\n")

    for round in range(args.total_rounds):
        env.reset()
        done = False
        step_count = 0  # 记录每局步数，防止无限循环
        max_steps = config.BOARD_SIZE * config.BOARD_SIZE  # 最大步数为棋盘格子数

        while not done and step_count < max_steps:
            state = torch.FloatTensor(env.board.flatten()).unsqueeze(0).to(device)
            if env.current_player == 1:
                logits = model1(state)
                action = get_valid_action(logits.cpu().detach().numpy(), env.board, 0.0001)  # P1几乎不探索
            else:
                logits = model2(state)
                action = get_valid_action(logits.cpu().detach().numpy(), env.board, 0.1)  # P2保留探索

            if action == -1:
                break
                
            current_player, done, reward, _ = env.step(action)
            step_count += 1

        # 4. 处理结束状态（胜利或平局）
        if done:
            if current_player == 1:
                player1_win_count += 1
            else:
                player2_win_count += 1
            result = f"Player {current_player} win"
        else:
            draw_count += 1
            result = "Draw"  # 平局

        if NEED_PRINT_BOARD:
                env.print_board()

        # 按指定间隔打印进度
        if (round + 1) % args.print_interval == 0:
            total = player1_win_count + player2_win_count + draw_count
            rate1 = (player1_win_count / total) * 100 if total else 0
            rate2 = (player2_win_count / total) * 100 if total else 0
            rated = (draw_count / total) * 100 if total else 0
            print(f"已完成 {round + 1}/{args.total_rounds} 局 | "
                  f"P1: {rate1:.2f}% ({player1_win_count}) | "
                  f"P2: {rate2:.2f}% ({player2_win_count}) | "
                  f"平局: {rated:.2f}% ({draw_count})")

    # 6. 输出最终统计结果
    print("\n===== 验证完成 =====")
    total = player1_win_count + player2_win_count + draw_count
    print(f"总场次: {total}")
    print(f"Player1 胜率: {player1_win_count/total*100:.2f}% ({player1_win_count})")
    print(f"Player2 胜率: {player2_win_count/total*100:.2f}% ({player2_win_count})")
    print(f"平局率: {draw_count/total*100:.2f}% ({draw_count})")

if __name__ == "__main__":
    validator()
    