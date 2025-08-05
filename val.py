import torch
import argparse
from model import Gomoku, GomokuNetV3, load_model_if_exists, get_valid_action
from config import Config, update_config_from_cli

NEED_PRINT_BOARD = False

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Gomoku(config.BOARD_SIZE, config.WIN_CONDITION)
    
    # 使用相同的模型架构
    model1 = GomokuNetV3(config.BOARD_SIZE).to(device)
    model2 = GomokuNetV3(config.BOARD_SIZE).to(device)

    # 加载训练好的模型
    load_model_if_exists(model1, 'gobang_best_model.pth')
    load_model_if_exists(model2, 'gobang_cainiao_model.pth')

    # 确保模型处于评估模式
    model1.eval()
    model2.eval()

    player1_win_count = 0
    player2_win_count = 0
    draw_count = 0

    print(f"===== 验证配置 =====")
    print(f"棋盘尺寸: {config.BOARD_SIZE}x{config.BOARD_SIZE}")
    print(f"胜利条件: 连{config.WIN_CONDITION}子")
    print(f"总局数: {args.total_rounds}")
    print(f"打印间隔: {args.print_interval}局")
    print(f"使用设备: {device}")
    print(f"===================\n")

    for round_num in range(args.total_rounds):
        env.reset()
        done = False
        step_count = 0
        max_steps = config.BOARD_SIZE * config.BOARD_SIZE

        while not done and step_count < max_steps:
            # 确保输入格式与训练时一致：多通道表示
            state = env.get_state_representation()
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            with torch.no_grad():
                if env.current_player == 1:
                    # model1是主要模型，以极低的探索率进行评估
                    logits, _ = model1(state_tensor)
                    board_flat = torch.tensor(env.board.flatten(), device=device)
                    action = get_valid_action(logits, board_flat, config.BOARD_SIZE, epsilon=0.0001)
                else:
                    # model2是历史模型，以略高的探索率提供一些变化
                    logits, _ = model2(state_tensor)
                    board_flat = torch.tensor(env.board.flatten(), device=device)
                    action = get_valid_action(logits, board_flat, config.BOARD_SIZE, epsilon=0.05) # 探索率设为0.05，比训练时的0.1略低

            if action == -1:
                break
                
            winner, done, _ = env.step(action)
            step_count += 1
        
        # 记录结果
        if done:
            if winner == 1:
                player1_win_count += 1
            else:
                player2_win_count += 1
        else:
            draw_count += 1

        if NEED_PRINT_BOARD:
            env.print_board()

        # 按指定间隔打印进度
        if (round_num + 1) % args.print_interval == 0:
            total = player1_win_count + player2_win_count + draw_count
            rate1 = (player1_win_count / total) * 100 if total else 0
            rate2 = (player2_win_count / total) * 100 if total else 0
            rated = (draw_count / total) * 100 if total else 0
            print(f"已完成 {round_num + 1}/{args.total_rounds} 局 | "
                  f"Player1胜率: {rate1:.2f}% ({player1_win_count}) | "
                  f"Player2胜率: {rate2:.2f}% ({player2_win_count}) | "
                  f"平局: {rated:.2f}% ({draw_count})")

    # 输出最终统计结果
    print("\n===== 验证完成 =====")
    total = player1_win_count + player2_win_count + draw_count
    print(f"总场次: {total}")
    print(f"Player1 胜率: {player1_win_count / total * 100:.2f}% ({player1_win_count})")
    print(f"Player2 胜率: {player2_win_count / total * 100:.2f}% ({player2_win_count})")
    print(f"平局率: {draw_count / total * 100:.2f}% ({draw_count})")

if __name__ == "__main__":
    validator()