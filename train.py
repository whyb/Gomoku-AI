import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from collections import deque
from copy import deepcopy
from model import Gomoku, GomokuNetV2, get_valid_action, load_model_if_exists
from config import Config, update_config_from_cli

def setup_device():
    use_gpu = torch.cuda.is_available()
    print(f"[初始化] 使用GPU: {use_gpu}")
    return torch.device("cuda" if use_gpu else "cpu")


def setup_players_and_optimizers(device, board_size):
    model1 = GomokuNetV2(board_size).to(device)  # 主训练模型
    model2 = GomokuNetV2(board_size).to(device)  # 动态对手模型
    optimizer1 = optim.Adam(model1.parameters(), lr=Config.LEARNING_RATE)
    optimizer2 = optim.Adam(model2.parameters(), lr=Config.LEARNING_RATE)
    print(f"[初始化] 模型创建完成 - 棋盘尺寸: {board_size}x{board_size}")
    return model1, model2, optimizer1, optimizer2


def get_dynamic_epsilon(steps_taken, total_cells, min_epsilon=0.05):
    """
    动态探索率：第一手100%，随落子数增加线性降低至5%
    steps_taken: 当前已落子数（0表示第一手）
    total_cells: 棋盘总格子数（board_size * board_size）
    """
    if steps_taken == 0:
        return 1.0  # 第一手强制100%探索
    # 线性衰减：从1.0到0.05，总衰减步数为（总格子数-1）
    decay_ratio = min(steps_taken / (total_cells - 1), 1.0)
    epsilon = 1.0 - (1.0 - min_epsilon) * decay_ratio
    return max(epsilon, min_epsilon)  # 确保不低于5%


def select_action(env, model, state, epsilon, device):
    with torch.no_grad():
        logits = model(state)
    
    board_flat = torch.tensor(env.board.flatten(), device=device, dtype=torch.float32)
    valid_mask = (board_flat == 0)
    
    valid_logits = logits[valid_mask.unsqueeze(0)]
    valid_indices = torch.where(valid_mask)[0]
    
    if valid_indices.numel() == 0:
        return logits, -1
    
    valid_logits_np = valid_logits.cpu().numpy().flatten()
    valid_indices_np = valid_indices.cpu().numpy()
    valid_actions = list(zip(valid_logits_np, valid_indices_np))
    valid_actions.sort(reverse=True, key=lambda x: x[0])
    
    # 探索策略：epsilon概率随机选择有效动作
    if random.random() < epsilon:
        return logits, random.choice(valid_actions)[1]
    else:
        top_k = min(3, len(valid_actions))
        return logits, random.choice(valid_actions[:top_k])[1]


def update_model_batch(model, optimizer, criterion, batch, device):
    states, actions, rewards = zip(*batch)
    states = torch.cat(states).to(device)
    actions = torch.tensor(actions, device=device)
    rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
    
    rewards_norm = torch.tanh(rewards / 1000)
    logits = model(states)
    loss = criterion(logits, actions) * rewards_norm.mean()
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return loss.item(), rewards.mean().item()


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--board_size", type=int, help="Size of the game board")
    parser.add_argument("--win_condition", type=int, help="Number of consecutive stones to win")
    args = parser.parse_args()
    config = update_config_from_cli(args)

    # 核心配置
    BATCH_SIZE = 128
    REPLAY_BUFFER_SIZE = 20000
    UPDATE_FREQ = 4
    PRINT_INTERVAL = 500
    HISTORY_POOL_SIZE = 5
    device = setup_device()
    total_cells = config.BOARD_SIZE * config.BOARD_SIZE  # 棋盘总格子数（用于计算探索率）
    
    # 初始化环境和模型
    env = Gomoku(config.BOARD_SIZE, config.WIN_CONDITION)
    model1, model2, optimizer1, optimizer2 = setup_players_and_optimizers(device, config.BOARD_SIZE)
    
    # 历史模型池和经验回放池
    history_pool = deque(maxlen=HISTORY_POOL_SIZE)
    replay_buffer1 = deque(maxlen=REPLAY_BUFFER_SIZE)
    replay_buffer2 = deque(maxlen=REPLAY_BUFFER_SIZE)
    
    # 加载模型
    load_model_if_exists(model1, 'gobang_best_model.pth')
    best_model = GomokuNetV2(config.BOARD_SIZE).to(device)
    best_model.load_state_dict(model1.state_dict())
    history_pool.append(deepcopy(best_model))
    criterion = nn.CrossEntropyLoss()

    # 统计变量
    total_win1 = 0
    total_win2 = 0
    recent_win1 = 0
    recent_win2 = 0
    recent_steps = []
    # 统计先手胜率（用于监控平衡度）
    first_player1_wins = 0
    first_player2_wins = 0
    total_first1 = 0
    total_first2 = 0

    print("\n[训练开始] --------------")
    print(f"最大回合数: {config.MAX_EPISODES}")
    print(f"棋盘尺寸: {config.BOARD_SIZE}x{config.BOARD_SIZE}（总格子数: {total_cells}）")
    print(f"批量大小: {BATCH_SIZE} | 历史模型池大小: {HISTORY_POOL_SIZE}")
    print(f"模型保存间隔: {config.SAVE_INTERVAL}局 | 进度打印间隔: {PRINT_INTERVAL}局\n")

    for episode in range(config.MAX_EPISODES):
        env.reset()
        done = False
        total_steps = 0  # 记录当前局已落子数（用于计算探索率）
        episode_reward1 = 0
        episode_reward2 = 0
        winner = 0

        # 随机决定本局先手（50% P1先手，50% P2先手）
        first_player = random.choice([1, 2])
        env.current_player = first_player  # 假设Gomoku类允许直接设置current_player
        if first_player == 1:
            total_first1 += 1
        else:
            total_first2 += 1

        while not done:
            state = torch.FloatTensor(env.board.flatten()).unsqueeze(0).to(device)
            
            # 根据当前已落子数计算动态探索率
            current_epsilon = get_dynamic_epsilon(total_steps, total_cells)

            if env.current_player == 1:
                # Player1落子（主模型）
                logits, action = select_action(env, model1, state, current_epsilon, device)
            else:
                # Player2落子（多样化对手）
                opponent_choices = [model2, best_model] + list(history_pool)
                selected_opponent = random.choice(opponent_choices)
                logits, action = select_action(env, selected_opponent, state, current_epsilon, device)

            if action == -1:
                break
                
            current_player, done, reward, _ = env.step(action)
            
            # 累积经验
            if current_player == 1:
                replay_buffer1.append((state.cpu().detach(), action, reward))
                episode_reward1 += reward
            else:
                replay_buffer2.append((state.cpu().detach(), action, reward))
                episode_reward2 += reward
            
            total_steps += 1  # 落子数+1，用于下一轮探索率计算
            
            # 批量更新
            if (total_steps % UPDATE_FREQ == 0):
                loss1 = None
                loss2 = None
                if len(replay_buffer1) >= BATCH_SIZE:
                    batch = random.sample(replay_buffer1, BATCH_SIZE)
                    loss1, mean_reward1 = update_model_batch(model1, optimizer1, criterion, batch, device)
                
                if len(replay_buffer2) >= BATCH_SIZE:
                    batch = random.sample(replay_buffer2, BATCH_SIZE)
                    loss2, mean_reward2 = update_model_batch(model2, optimizer2, criterion, batch, device)

        # 记录本局结果（含先手胜率统计）
        if done:
            winner = current_player
            if winner == 1:
                total_win1 += 1
                recent_win1 += 1
                if first_player == 1:
                    first_player1_wins += 1
            else:
                total_win2 += 1
                recent_win2 += 1
                if first_player == 2:
                    first_player2_wins += 1
        recent_steps.append(total_steps)

        # 定期打印进度（增加先手胜率统计）
        if (episode + 1) % PRINT_INTERVAL == 0:
            avg_steps = sum(recent_steps) / len(recent_steps) if recent_steps else 0
            recent_win_rate1 = recent_win1 / PRINT_INTERVAL * 100 if PRINT_INTERVAL > 0 else 0
            recent_win_rate2 = recent_win2 / PRINT_INTERVAL * 100 if PRINT_INTERVAL > 0 else 0
            total_win_rate1 = total_win1 / (episode + 1) * 100 if (episode + 1) > 0 else 0
            
            # 先手胜率计算
            first1_win_rate = (first_player1_wins / total_first1 * 100) if total_first1 > 0 else 0
            first2_win_rate = (first_player2_wins / total_first2 * 100) if total_first2 > 0 else 0

            print(f"[进度] 第 {episode + 1}/{config.MAX_EPISODES} 局")
            print(f"  胜率: 最近 {PRINT_INTERVAL} 局 P1: {recent_win_rate1:.1f}% / P2: {recent_win_rate2:.1f}%")
            print(f"  总体胜率: P1: {total_win_rate1:.1f}% ({total_win1}/{total_win2})")
            print(f"  先手胜率: P1先手: {first1_win_rate:.1f}% | P2先手: {first2_win_rate:.1f}%")
            print(f"  平均步数: {avg_steps:.1f}")
            if 'loss1' in locals() and loss1 is not None:
                print(f"  损失值: P1: {loss1:.4f} / P2: {loss2:.4f}")
            print("  ------------------------")
            
            recent_win1 = 0
            recent_win2 = 0
            recent_steps = []

        # 模型保存与历史池更新
        if (episode + 1) % config.SAVE_INTERVAL == 0:
            torch.save(model1.state_dict(), f'gobang_model_player1_{episode + 1}.pth')
            history_pool.append(deepcopy(model1))
            print(f"[历史池更新] 已保存第 {episode + 1} 局模型，当前池大小: {len(history_pool)}")
            
            if history_pool:
                model2.load_state_dict(random.choice(history_pool).state_dict())
                optimizer2 = optim.Adam(model2.parameters(), lr=Config.LEARNING_RATE)
            
            best_model.load_state_dict(model1.state_dict())
            print(f"[模型保存] 第 {episode + 1} 局模型已保存")

    torch.save(model1.state_dict(), 'gobang_best_model.pth')
    print("\n[训练结束] 最终模型已保存")
    print(f"总胜率: P1: {total_win1/(total_win1+total_win2)*100:.1f}% ({total_win1}/{total_win2})")
    print(f"先手胜率: P1先手: {first_player1_wins/total_first1*100:.1f}% | P2先手: {first_player2_wins/total_first2*100:.1f}%")


if __name__ == "__main__":
    train()
