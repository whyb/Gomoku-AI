"""
AlphaZero 模型验证脚本 — 适配 model_alphazero.py

模型:
  - standard: GomokuNetAlphaZero (SE-ResNet-128, 10 层, 推荐)
  - small:    GomokuNetAlphaZeroSmall (SE-ResNet-64, 6 层)

对战目标 (--target):
  - self:     模型 vs 模型自我对弈 (双方都使用同一模型)
  - teacher:  模型(P1/黑) vs Kali-Hac 教师(P2/白)

输出两行核心结论, 始终只报验证模型的胜率/平局率:
  - teacher 模式: 验证模型先手 / 教师先手时验证模型的胜率
  - self 模式:    验证模型1先手 / 验证模型2先手的胜率

随机性说明:
  - 模型前向推理是确定性的 (固定权重 + eval 模式)
  - 模型落子前先走战术守卫 (与 webdemo 一致): 必赢/必堵/活四双四/活三防守直接短路
  - 无强制走法时对 NN logits 叠加 +10/+8 软战术先验再贪心 (与 webdemo / --mcts_human_knowledge 一致)
  - --epsilon 控制模型落子的随机探索率: 0=纯贪心(推荐评估), >0 时每步
    以该概率随机落子 (会明显拉低胜率)
  - --seed 固定随机种子后, 每局结果可完全复现

用法:
  python val_az.py --board_size 15 --model standard \
      --model_path alpaz_standard_15x15_best.pth --target self
  python val_az.py --board_size 15 --target teacher
"""

import os
import argparse
import random
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed

from model import Gomoku, get_valid_action
from model_alphazero import GomokuNetAlphaZero, GomokuNetAlphaZeroSmall
from mcts import get_forced_move, _tactical_prior, _apply_tactical_prior
from config import Config, update_config_from_cli


def build_perspective_state(env):
    """
    按当前玩家视角构建 (2, H, W) 状态

    与 self_play.py 的训练数据一致:
      channel 0 = 当前玩家棋子
      channel 1 = 对手棋子
    """
    n = env.board_size
    board = env.board
    current = env.current_player
    state = np.zeros((2, n, n), dtype=np.float32)
    state[0] = (board == current).astype(np.float32)
    state[1] = (board == 3 - current).astype(np.float32)
    return state


def load_az_model(model, model_path):
    """
    加载 AlphaZero 模型权重

    兼容两种格式:
      - 纯权重: torch.save(model.state_dict(), path)
      - 完整 checkpoint: torch.save({..., 'model_state_dict': ...}, path)
    """
    if not os.path.exists(model_path):
        print(f"未找到权重文件: {model_path}, 使用随机初始化权重")
        return False
    state = torch.load(model_path, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)
    print(f"已加载权重: {model_path}")
    return True


def play_round(board_size, win_condition, model_state_dict,
               model_class='standard', target='self',
               epsilon=0.0, game_id=0, seed=None):
    """
    运行一局对弈, 返回 (先手玩家, 胜者); 胜者 0 表示平局。

    Args:
        target:
          'self'    — 模型 vs 模型 (自我对弈)
          'teacher' — 模型(P1/黑) vs Kali-Hac 教师(P2/白)
    """
    env = Gomoku(board_size, win_condition)

    if model_class == 'small':
        model = GomokuNetAlphaZeroSmall()
    else:
        model = GomokuNetAlphaZero()
    model.load_state_dict(model_state_dict)
    model.eval()

    if seed is not None:
        # 每局使用独立种子, 保证 (seed, game_id) 决定整局结果
        random.seed(seed * 100000 + game_id)
        np.random.seed(seed * 100000 + game_id + 1)

    teacher = None
    if target == 'teacher':
        from teacher import TeacherAI
        teacher = TeacherAI(board_size=board_size)

    total_cells = board_size * board_size
    first_player = random.choice([1, 2])
    env.current_player = first_player
    steps = 0

    while True:
        state_tensor = torch.FloatTensor(
            build_perspective_state(env)
        ).unsqueeze(0)

        if target == 'teacher' and env.current_player == 2:
            # 教师执白落子
            if not env.board.any():
                # 空棋盘教师无法评分, 先手第一手随机落子 (均匀分布)
                action = np.random.randint(total_cells)
            else:
                pos = teacher.get_move(env.board)
                if pos is None:
                    valid = np.flatnonzero(env.board.flatten() == 0)
                    if len(valid) == 0:
                        return first_player, 0  # 无合法落子 → 平局
                    action = int(np.random.choice(valid))
                else:
                    action = pos[0] * board_size + pos[1]
        else:
            # 模型落子 (self 模式双方都是模型; teacher 模式模型执黑)
            if not env.board.any() and env.current_player == 1:
                # 先手第一手随机落子 (均匀分布), 不固定天元
                action = np.random.randint(total_cells)
            else:
                # 战术守卫: 必赢/必堵/活四双四/活三防守 直接短路 (与实际对弈一致)
                forced_action, _ = get_forced_move(
                    env.board, env.current_player, win_condition
                )
                if forced_action is not None:
                    action = forced_action
                else:
                    with torch.no_grad():
                        logits, _ = model(state_tensor)
                    # 软战术先验: 无强制走法时对 NN logits 叠加 +10/+8
                    # (与 webdemo 的 aiLogic / --mcts_human_knowledge 的 _apply_tactical_prior 一致)
                    logits_np = logits.detach().cpu().numpy().reshape(-1)
                    _apply_tactical_prior(
                        logits_np,
                        _tactical_prior(env.board, env.current_player, win_condition)
                    )
                    logits = torch.from_numpy(logits_np)
                    board_flat = torch.tensor(env.board.flatten())
                    action = get_valid_action(logits, board_flat, board_size,
                                              epsilon=epsilon)

        if action == -1:
            return first_player, 0  # 棋盘已满 → 平局

        next_player, done, reward = env.step(action)
        steps += 1
        if done:
            return first_player, next_player  # 落子方获胜
        if steps >= total_cells:
            return first_player, 0  # 棋盘下满无人获胜 → 平局


def summary_lines(target, first1, first2,
                  p1_win_first, p1_win_second, p2_win_first,
                  ties_first1, ties_first2):
    """
    生成两行核心结论, 只报验证模型的胜率/平局率

    target='self':
      验证模型1先手的胜率 / 验证模型2先手的胜率
    target='teacher':
      验证模型先手的胜率 / 教师先手时验证模型的胜率
    """
    lines = []
    if target == 'self':
        label1 = '验证模型1'
        label2 = '验证模型2'
    else:
        label1 = '验证模型'
        label2 = '教师'

    if first1 > 0:
        lines.append(
            f"{label1}先手的胜率：{p1_win_first / first1 * 100:.1f}% "
            f"({p1_win_first}/{first1}), 平局 "
            f"{ties_first1 / first1 * 100:.1f}% ({ties_first1}/{first1})"
        )
    if first2 > 0:
        if target == 'self':
            lines.append(
                f"{label2}先手的胜率：{p2_win_first / first2 * 100:.1f}% "
                f"({p2_win_first}/{first2}), 平局 "
                f"{ties_first2 / first2 * 100:.1f}% ({ties_first2}/{first2})"
            )
        else:
            lines.append(
                f"{label2}先手时，{label1}的胜率："
                f"{p1_win_second / first2 * 100:.1f}% "
                f"({p1_win_second}/{first2}), 平局 "
                f"{ties_first2 / first2 * 100:.1f}% ({ties_first2}/{first2})"
            )
    return lines


def validator():
    parser = argparse.ArgumentParser(description='AlphaZero Gomoku 模型验证')
    parser.add_argument("--board_size", type=int, default=15,
                        help="Size of the game board")
    parser.add_argument("--win_condition", type=int, default=5,
                        help="Number of consecutive stones to win")
    parser.add_argument("--total_rounds", type=int, default=200,
                        help="Total number of rounds to play (default: 200)")
    parser.add_argument("--print_interval", type=int, default=50,
                        help="Interval for printing progress (default: 50)")
    parser.add_argument("--model", type=str, default='standard',
                        choices=['small', 'standard'],
                        help="模型大小: small=GomokuNetAlphaZeroSmall(6层/64ch, 快), "
                             "standard=GomokuNetAlphaZero(10层/128ch, 强, 推荐)")
    parser.add_argument("--model_path", type=str,
                        default='alpaz_standard_15x15_best.pth',
                        help="权重文件路径 (兼容纯权重或含 model_state_dict 的 checkpoint)")
    parser.add_argument("--target", type=str, default='self',
                        choices=['self', 'teacher'],
                        help="对打目标: self=模型自我对弈, "
                             "teacher=与 Kali-Hac 教师对打 (默认: self)")
    parser.add_argument("--epsilon", type=float, default=0.0,
                        help="模型落子随机探索率 (0=纯贪心, 评估推荐; "
                             ">0 时每步按该概率随机落子)")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子 (固定后每局结果可复现)")
    args = parser.parse_args()
    update_config_from_cli(args)

    print("===== 验证配置 =====")
    print(f"棋盘尺寸: {Config.BOARD_SIZE}x{Config.BOARD_SIZE}")
    print(f"胜利条件: 连{Config.WIN_CONDITION}子")
    print(f"总局数: {args.total_rounds}")
    print(f"打印间隔: {args.print_interval}局")
    if args.target == 'self':
        target_label = '模型自我对弈 (P1/P2 均为当前模型)'
    else:
        target_label = 'Kali-Hac 教师 (当前模型执黑=P1, 教师执白=P2)'
    model_label = ('standard (GomokuNetAlphaZero)'
                   if args.model == 'standard'
                   else 'small (GomokuNetAlphaZeroSmall)')
    print(f"对战目标: {target_label}")
    print(f"模型: {model_label}")
    print(f"模型探索率 epsilon: {args.epsilon} (0=纯贪心, 结果确定)")
    if args.seed is not None:
        print(f"随机种子: {args.seed}")
    print("先手第一手随机落子 (均匀分布), 不固定天元。")
    print("===================\n")

    # 加载模型
    if args.model == 'small':
        model = GomokuNetAlphaZeroSmall()
    else:
        model = GomokuNetAlphaZero()
    load_az_model(model, args.model_path)
    state = {k: v.cpu() for k, v in model.state_dict().items()}

    # 统计变量: 只关心验证模型自身的胜率/平局率
    p1_win_first = 0
    p1_win_second = 0
    p2_win_first = 0
    ties_first1 = 0
    ties_first2 = 0
    first_player1_rounds = 0
    first_player2_rounds = 0

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(
            play_round, Config.BOARD_SIZE, Config.WIN_CONDITION, state,
            args.model, args.target, args.epsilon, i, args.seed
        ) for i in range(args.total_rounds)]

        completed_rounds = 0
        for future in as_completed(futures):
            first_player_id, winner = future.result()

            if first_player_id == 1:
                first_player1_rounds += 1
            else:
                first_player2_rounds += 1

            if winner == 1:
                if first_player_id == 1:
                    p1_win_first += 1
                else:
                    p1_win_second += 1
            elif winner == 2:
                if first_player_id == 2:
                    p2_win_first += 1
            else:
                if first_player_id == 1:
                    ties_first1 += 1
                else:
                    ties_first2 += 1

            completed_rounds += 1

            if completed_rounds % args.print_interval == 0:
                print(f"已完成 {completed_rounds}/{args.total_rounds} 局")
                for line in summary_lines(
                        args.target, first_player1_rounds,
                        first_player2_rounds, p1_win_first, p1_win_second,
                        p2_win_first, ties_first1, ties_first2):
                    print(line)
                print()

    print("\n===== 验证完成 =====")
    total_rounds = first_player1_rounds + first_player2_rounds
    print(f"总场次: {total_rounds}")
    for line in summary_lines(
            args.target, first_player1_rounds, first_player2_rounds,
            p1_win_first, p1_win_second, p2_win_first,
            ties_first1, ties_first2):
        print(line)


if __name__ == "__main__":
    validator()
