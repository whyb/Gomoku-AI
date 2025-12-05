import os
import sys
import random
import time
import argparse
import math
from collections import deque
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim

from model import Gomoku, GomokuNetV3, get_valid_action, load_model_if_exists
from config import Config, update_config_from_cli

# 引入 another 对手（确保可以正确导入）
_BASE_DIR = os.path.dirname(__file__)
_ANOTHER_DIR = os.path.join(_BASE_DIR, 'another')
if _ANOTHER_DIR not in sys.path:
    sys.path.append(_ANOTHER_DIR)
import Alpha_beta_optimize as ai_player
import Global_variables as gv

CPU_PARALLEL_ENVS = 6
GPU_PARALLEL_ENVS = 8
CPU_BATCH_SIZE = 128
GPU_BATCH_SIZE = 256

def get_device_config():
    use_cuda = torch.cuda.is_available()
    main_device = torch.device("cuda" if use_cuda else "cpu")
    print(f"[初始化] 主设备: {main_device} | 支持CUDA: {use_cuda}")
    return main_device, use_cuda

# 线性衰减辅助函数
def linear_schedule(start, end, current, decay):
    if decay <= 0:
        return end
    ratio = max(0.0, min(1.0, current / decay))
    return start + (end - start) * ratio

def setup_player_and_optimizer(device, board_size):
    model = GomokuNetV3(board_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    print(f"[初始化] 模型创建完成 - 棋盘尺寸: {board_size}x{board_size}")
    return model, optimizer

def reset_another_ai(board_size, board_array):
    gv.prepare(board_size)
    for i in range(board_size):
        for j in range(board_size):
            v = board_array[i][j]
            if v == 1:
                gv.black[i][j] = 1
                gv.flag[i][j] = 1
            elif v == 2:
                gv.white[i][j] = 1
                gv.flag[i][j] = 1
            else:
                gv.flag[i][j] = 0
    ai_player.search_range = ai_player.shrink_range()

def env_worker(env_id, board_size, win_condition, model_queue, experience_queue, total_cells, device_type):
    env = Gomoku(board_size, win_condition)
    device = torch.device("cuda" if device_type == "gpu" else "cpu")

    local_model = GomokuNetV3(board_size).to(device)
    local_model.eval()

    latest_global_episodes = 0

    while True:
        if not model_queue.empty():
            try:
                model_params = model_queue.get(block=False)
                local_model.load_state_dict({k: v.to(device) for k, v in model_params['model'].items()})
                if 'global_episodes' in model_params:
                    latest_global_episodes = model_params['global_episodes']
            except Exception:
                pass

        env.reset()
        reset_another_ai(board_size, env.board)
        done = False
        total_steps = 0
        episode_experience = []
        first_player = random.choice([1, 2])
        env.current_player = first_player

        STEPS_PER_ENV = env.board_size * env.board_size
        while not done and total_steps < STEPS_PER_ENV:
            state = env.get_state_representation()
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            if env.current_player == 1:
                with torch.no_grad():
                    logits, _ = local_model(state_tensor)
                board_flat = torch.tensor(env.board.flatten(), device=device)
                # 动态探索率（随全局局数线性衰减）
                eps = linear_schedule(Config.EPSILON1_START, Config.EPSILON1_END, latest_global_episodes, Config.EPSILON_DECAY)
                action = get_valid_action(logits, board_flat, board_size, epsilon=eps)
            else:
                reset_another_ai(board_size, env.board)
                # 陪练难度混合（随训练进度逐步提高强度，降低随机）
                strong_prob = linear_schedule(Config.OPP_STRONG_PROB_START, Config.OPP_STRONG_PROB_END, latest_global_episodes, Config.OPP_STRONG_DECAY)
                random_prob = linear_schedule(Config.OPP_RANDOM_PROB_START, Config.OPP_RANDOM_PROB_END, latest_global_episodes, Config.OPP_RANDOM_DECAY)
                r = random.random()
                if r < random_prob:
                    valid_actions = [i * board_size + j for i in range(board_size) for j in range(board_size) if env.board[i][j] == 0]
                    action = random.choice(valid_actions) if valid_actions else -1
                else:
                    level = '比你6的Level' if r < (random_prob + strong_prob) else '和我一样6的Level'
                    pos = ai_player.machine_thinking(level)
                    if not pos:
                        valid_actions = [i * board_size + j for i in range(board_size) for j in range(board_size) if env.board[i][j] == 0]
                        action = random.choice(valid_actions) if valid_actions else -1
                    else:
                        i, j = pos
                        action = i * board_size + j

            current_player = env.current_player
            next_player, done, reward = env.step(action)
            if done:
                base_win_reward, step_count = reward
                speed_bonus = Config.SPEED_REWARD_COEFFICIENT * (total_cells - step_count)
                final_reward = base_win_reward + speed_bonus
                episode_experience.append((state_tensor.cpu().detach(), action, final_reward, current_player))
            else:
                episode_experience.append((state_tensor.cpu().detach(), action, reward, current_player))
            total_steps += 1

        if done:
            winner = env.current_player
            loser = 3 - winner
            base_lose_penalty = Config.REWARD["lose"]
            step_based_penalty = Config.LOSE_STEP_PENALTY * env.step_count
            total_lose_penalty = base_lose_penalty + step_based_penalty
            for i in range(len(episode_experience)):
                s, a, r, p = episode_experience[i]
                if p == loser:
                    episode_experience[i] = (s, a, total_lose_penalty, p)

        if episode_experience:
            experience_queue.put({
                'env_id': env_id,
                'device_type': device_type,
                'experiences': episode_experience,
                'first_player': first_player,
                'winner': env.current_player if done else 0
            })
        #env.print_board()

def update_model_batch(model, optimizer, policy_criterion, value_criterion, batch, device):
    states, actions, rewards = zip(*batch)
    states = torch.cat(states).to(device)
    actions = torch.tensor(actions, device=device)
    rewards = torch.tensor(rewards, device=device, dtype=torch.float32)

    logits, values = model(states)
    policy_loss = (policy_criterion(logits, actions) * rewards).mean()
    value_loss = value_criterion(values, rewards)
    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item(), rewards.mean().item()

def train():
    try:
        from torch.multiprocessing import set_start_method, Process, Queue
        set_start_method('spawn')
    except Exception:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--board_size", type=int, help="Size of the game board")
    parser.add_argument("--win_condition", type=int, help="Number of consecutive stones to win")
    args = parser.parse_args()
    config = update_config_from_cli(args)

    # 默认使用8x8棋盘（可被命令行覆盖）
    if not hasattr(args, 'board_size') or args.board_size is None:
        Config.BOARD_SIZE = 8

    main_device, use_cuda = get_device_config()
    batch_size = GPU_BATCH_SIZE if main_device.type == 'cuda' else CPU_BATCH_SIZE

    REPLAY_BUFFER_SIZE = 30000
    PRINT_INTERVAL = 50
    total_cells = Config.BOARD_SIZE * Config.BOARD_SIZE

    model, optimizer = setup_player_and_optimizer(main_device, Config.BOARD_SIZE)
    load_model_if_exists(model, 'gobang_best_model.pth')
    best_model = GomokuNetV3(Config.BOARD_SIZE).to(main_device)
    best_model.load_state_dict(model.state_dict())
    history_pool = deque([deepcopy(best_model)], maxlen=3)

    policy_criterion = nn.CrossEntropyLoss(reduction='none')
    value_criterion = nn.MSELoss()

    # 学习率调度器：先线性升温，再余弦衰减到 LR_MIN
    def lr_lambda(step):
        if step < Config.LR_WARMUP_STEPS:
            return max(1e-8, step / float(Config.LR_WARMUP_STEPS))
        t = min(step - Config.LR_WARMUP_STEPS, Config.LR_DECAY_STEPS)
        cos_decay = 0.5 * (1 + math.cos(math.pi * t / float(Config.LR_DECAY_STEPS)))
        base = Config.LEARNING_RATE
        target = Config.LR_MIN
        return (target / base) + (1 - (target / base)) * cos_decay
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    from torch.multiprocessing import Process, Queue
    model_queue = Queue()
    experience_queue = Queue()

    processes = []
    total_envs = 0
    cpu_envs = CPU_PARALLEL_ENVS if CPU_PARALLEL_ENVS > 0 else (2 if not use_cuda else 0)
    if cpu_envs > 0:
        for env_id in range(cpu_envs):
            p = Process(target=env_worker, args=(f"cpu-{env_id}", Config.BOARD_SIZE, Config.WIN_CONDITION, model_queue, experience_queue, total_cells, "cpu"), daemon=True)
            p.start()
            processes.append(p)
            total_envs += 1
    if use_cuda:
        for env_id in range(GPU_PARALLEL_ENVS):
            p = Process(target=env_worker, args=(f"gpu-{env_id}", Config.BOARD_SIZE, Config.WIN_CONDITION, model_queue, experience_queue, total_cells, "gpu"), daemon=True)
            p.start()
            processes.append(p)
            total_envs += 1

    print(f"[并行初始化] 启动 {total_envs} 个并行游戏环境")

    total_episodes = 0
    def send_model_params():
        model_params = {
            'model': {k: v.cpu() for k, v in model.state_dict().items()},
            'global_episodes': total_episodes,
        }
        while not model_queue.empty():
            try:
                model_queue.get_nowait()
            except Exception:
                pass
        for _ in range(total_envs):
            model_queue.put(model_params)

    send_model_params()

    total_win1 = 0
    total_win2 = 0
    first_player1_wins = 0
    first_player2_wins = 0
    total_first1 = 0
    total_first2 = 0
    update_steps = 0
    last_save_step = 0
    cpu_episodes = 0
    gpu_episodes = 0
    start_time = time.time()
    last_print_time = start_time
    last_episodes = 0
    last_cpu_episodes = 0
    last_gpu_episodes = 0

    print("\n[训练开始] --------------")
    print(f"最大回合数: {Config.MAX_EPISODES}")
    print(f"并行环境数: {total_envs} | 批量大小: {batch_size}")
    print(f"棋盘尺寸: {Config.BOARD_SIZE}x{Config.BOARD_SIZE} | 模型保存间隔: {Config.SAVE_INTERVAL}局\n")

    try:
        while total_episodes < Config.MAX_EPISODES:
            current_time = time.time()
            if current_time - last_print_time >= 10:
                time_diff = current_time - last_print_time
                episodes_diff = total_episodes - last_episodes
                cpu_diff = cpu_episodes - last_cpu_episodes
                gpu_diff = gpu_episodes - last_gpu_episodes
                speed = episodes_diff / time_diff if time_diff > 0 else 0
                print(f"[速度统计] 过去 {time_diff:.1f} 秒完成 {episodes_diff} 局 | 平均速度: {speed:.2f} 局/秒")
                if use_cuda:
                    print(f"[设备贡献] CPU: {cpu_diff} 局 | GPU: {gpu_diff} 局")
                last_print_time = current_time
                last_episodes = total_episodes
                last_cpu_episodes = cpu_episodes
                last_gpu_episodes = gpu_episodes

            experiences = []
            while len(experiences) < batch_size and total_episodes < Config.MAX_EPISODES:
                if not experience_queue.empty():
                    exp_data = experience_queue.get()
                    experiences.extend(exp_data['experiences'])
                    winner = exp_data['winner']
                    first_player = exp_data['first_player']
                    total_episodes += 1
                    if exp_data['device_type'] == 'cpu':
                        cpu_episodes += 1
                    else:
                        gpu_episodes += 1
                    if winner == 1:
                        total_win1 += 1
                        if first_player == 1:
                            first_player1_wins += 1
                    elif winner == 2:
                        total_win2 += 1
                        if first_player == 2:
                            first_player2_wins += 1
                    if first_player == 1:
                        total_first1 += 1
                    else:
                        total_first2 += 1

            replay_buffer1 = deque(maxlen=REPLAY_BUFFER_SIZE)
            for state, action, reward, player in experiences:
                if player == 1:
                    replay_buffer1.append((state, action, reward))

            loss1 = None
            if len(replay_buffer1) >= batch_size:
                batch1 = random.sample(replay_buffer1, batch_size)
                loss1, _ = update_model_batch(model, optimizer, policy_criterion, value_criterion, batch1, main_device)
                scheduler.step()
                update_steps += 1

            if (total_episodes % PRINT_INTERVAL) == 0 and total_episodes > 0:
                total_win_rate1 = (total_win1 / total_episodes * 100 if total_episodes else 0)
                first1_rate = (first_player1_wins / total_first1 * 100 if total_first1 else 0)
                first2_rate = (first_player2_wins / total_first2 * 100 if total_first2 else 0)
                print(f"[进度] 第 {total_episodes}/{Config.MAX_EPISODES} 局")
                print(f"  AI总胜率: P1: {total_win_rate1:.1f}% ({total_win1}/{total_win2})")
                print(f"  AI先手的胜率: AI先手: {first1_rate:.1f}% | 陪练先手的胜率: {first2_rate:.1f}%")
                if loss1 is not None:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"  损失值: P1: {loss1:.4f} | 学习率: {current_lr:.6f}")
                print("  ------------------------")

            if (update_steps - last_save_step) >= Config.SAVE_INTERVAL and update_steps > 0:
                filename = f'gobang_model_player1_step_{update_steps}.pth'
                torch.save(model.state_dict(), filename)
                history_pool.append(deepcopy(model))
                best_model.load_state_dict(model.state_dict())
                send_model_params()
                print(f"[模型保存] 第 {total_episodes} 局模型已保存")
                last_save_step = update_steps
    except KeyboardInterrupt:
        print("\n[中断] 收到 Ctrl+C，正在安全退出...")
    finally:
        for p in processes:
            p.terminate()
        torch.save(model.state_dict(), 'gobang_best_model.pth')
        print("\n[训练结束] 最终模型已保存")
        total = total_win1 + total_win2
        if total > 0:
            print(f"总胜率: P1: {total_win1/total*100:.1f}% ({total_win1}/{total_win2})")
            if total_first1 > 0:
                print(f"AI先手的胜率: {first_player1_wins/total_first1*100:.1f}%")
            if total_first2 > 0:
                print(f"陪练先手的胜率: {first_player2_wins/total_first2*100:.1f}%")
        total_time = time.time() - start_time
        if total_time > 0:
            total_speed = total_episodes / total_time
            print(f"总训练速度: {total_speed:.2f} 局/秒 (共 {total_episodes} 局，耗时 {total_time:.1f} 秒)")
            if use_cuda:
                print(f"设备贡献: CPU: {cpu_episodes} 局 ({(cpu_episodes/total_episodes*100) if total_episodes else 0:.1f}%) | GPU: {gpu_episodes} 局 ({(gpu_episodes/total_episodes*100) if total_episodes else 0:.1f}%)")

if __name__ == "__main__":
    train()
