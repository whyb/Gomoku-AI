import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from collections import deque
from model import Gomoku, GomokuNetV2, get_valid_action, load_model_if_exists
from config import Config, update_config_from_cli

def setup_device():
    use_gpu = torch.cuda.is_available()
    print(f"[初始化] 使用GPU: {use_gpu}")
    return torch.device("cuda" if use_gpu else "cpu")


def setup_players_and_optimizers(device, board_size):
    model1 = GomokuNetV2(board_size).to(device)
    model2 = GomokuNetV2(board_size).to(device)
    optimizer1 = optim.Adam(model1.parameters(), lr=Config.LEARNING_RATE)
    optimizer2 = optim.Adam(model2.parameters(), lr=Config.LEARNING_RATE)
    print(f"[初始化] 模型创建完成 - 棋盘尺寸: {board_size}x{board_size}")
    return model1, model2, optimizer1, optimizer2


def get_epsilon(step, start, end, decay):
    return end + (start - end) * math.exp(-1. * step / decay)


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
    
    if random.random() < epsilon:
        return logits, random.choice(valid_actions)[1]
    else:
        top_k = min(3, len(valid_actions))
        return logits, random.choice(valid_actions[:top_k])[1]


def update_model_batch(model, optimizer, criterion, batch, device, is_model1=True):
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

    # 配置参数
    BATCH_SIZE = 128
    REPLAY_BUFFER_SIZE = 10000
    UPDATE_FREQ = 4
    PRINT_INTERVAL = 500  # 每500局打印一次进度
    device = setup_device()
    
    # 初始化环境和模型
    env = Gomoku(config.BOARD_SIZE, config.WIN_CONDITION)
    model1, model2, optimizer1, optimizer2 = setup_players_and_optimizers(device, config.BOARD_SIZE)
    
    # 经验回放池
    replay_buffer1 = deque(maxlen=REPLAY_BUFFER_SIZE)
    replay_buffer2 = deque(maxlen=REPLAY_BUFFER_SIZE)
    
    # 加载模型
    load_model_if_exists(model1, 'gobang_best_model.pth')
    best_model = GomokuNetV2(config.BOARD_SIZE).to(device)
    best_model.load_state_dict(model1.state_dict())
    criterion = nn.CrossEntropyLoss()

    # 统计变量
    total_win1 = 0
    total_win2 = 0
    recent_win1 = 0  # 最近PRINT_INTERVAL局的胜利数
    recent_win2 = 0
    recent_steps = []  # 最近局的步数统计

    print("\n[训练开始] --------------")
    print(f"最大回合数: {config.MAX_EPISODES}")
    print(f"批量大小: {BATCH_SIZE}")
    print(f"模型保存间隔: {config.SAVE_INTERVAL}局")
    print(f"进度打印间隔: {PRINT_INTERVAL}局\n")

    for episode in range(config.MAX_EPISODES):
        env.reset()
        done = False
        total_steps = 0
        episode_reward1 = 0
        episode_reward2 = 0
        winner = 0  # 0:未分胜负, 1:玩家1胜, 2:玩家2胜

        epsilon1 = get_epsilon(episode, config.EPSILON1_START, config.EPSILON1_END, config.EPSILON_DECAY)
        epsilon2 = get_epsilon(episode, config.EPSILON2_START, config.EPSILON2_END, config.EPSILON_DECAY)

        while not done:
            state = torch.FloatTensor(env.board.flatten()).unsqueeze(0).to(device)
            
            if env.current_player == 1:
                logits, action = select_action(env, model1, state, epsilon1, device)
            else:
                if random.random() < 0.5:
                    logits, action = select_action(env, model2, state, epsilon2, device)
                else:
                    logits, action = select_action(env, best_model, state, epsilon2 * 0.5, device)

            if action == -1:
                break
                
            current_player, done, reward, _ = env.step(action)
            
            # 累积奖励
            if current_player == 1:
                replay_buffer1.append((state.cpu().detach(), action, reward))
                episode_reward1 += reward
            else:
                replay_buffer2.append((state.cpu().detach(), action, reward))
                episode_reward2 += reward
            
            total_steps += 1
            
            # 批量更新
            if (total_steps % UPDATE_FREQ == 0):
                loss1 = None
                loss2 = None
                if len(replay_buffer1) >= BATCH_SIZE:
                    batch = random.sample(replay_buffer1, BATCH_SIZE)
                    loss1, mean_reward1 = update_model_batch(model1, optimizer1, criterion, batch, device, True)
                
                if len(replay_buffer2) >= BATCH_SIZE:
                    batch = random.sample(replay_buffer2, BATCH_SIZE)
                    loss2, mean_reward2 = update_model_batch(model2, optimizer2, criterion, batch, device, False)

        # 记录本局结果
        if done:
            winner = current_player
            if winner == 1:
                total_win1 += 1
                recent_win1 += 1
            else:
                total_win2 += 1
                recent_win2 += 1
        recent_steps.append(total_steps)

        # 定期打印进度 (每PRINT_INTERVAL局)
        if (episode + 1) % PRINT_INTERVAL == 0:
            # 计算最近区间的统计值
            avg_steps = sum(recent_steps) / len(recent_steps) if recent_steps else 0
            recent_win_rate1 = recent_win1 / PRINT_INTERVAL * 100 if PRINT_INTERVAL > 0 else 0
            recent_win_rate2 = recent_win2 / PRINT_INTERVAL * 100 if PRINT_INTERVAL > 0 else 0
            total_win_rate1 = total_win1 / (episode + 1) * 100 if (episode + 1) > 0 else 0

            print(f"[进度] 第 {episode + 1}/{config.MAX_EPISODES} 局")
            print(f"  胜率: 最近 {PRINT_INTERVAL} 局 P1: {recent_win_rate1:.1f}% / P2: {recent_win_rate2:.1f}%")
            print(f"  总体胜率: P1: {total_win_rate1:.1f}% ({total_win1}/{total_win2})")
            print(f"  平均步数: {avg_steps:.1f} | 探索率: P1: {epsilon1:.4f} / P2: {epsilon2:.4f}")
            if 'loss1' in locals() and loss1 is not None:
                print(f"  损失值: P1: {loss1:.4f} / P2: {loss2:.4f}")
            print("  ------------------------")
            
            # 重置最近统计
            recent_win1 = 0
            recent_win2 = 0
            recent_steps = []

        # 模型保存
        if (episode + 1) % config.SAVE_INTERVAL == 0:
            torch.save(model1.state_dict(), f'gobang_model_player1_{episode + 1}.pth')
            best_model.load_state_dict(model1.state_dict())
            model2.load_state_dict(best_model.state_dict())
            optimizer2 = optim.Adam(model2.parameters(), lr=Config.LEARNING_RATE)
            print(f"[模型保存] 第 {episode + 1} 局模型已保存")

    torch.save(model1.state_dict(), 'gobang_best_model.pth')
    print("\n[训练结束] 最终模型已保存")
    print(f"总胜率: P1: {total_win1/(total_win1+total_win2)*100:.1f}% ({total_win1}/{total_win2})")


if __name__ == "__main__":
    train()
