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

from model import Gomoku, get_valid_action
from model_dy import GomokuNetDyn, load_model_if_exists
from config import Config, update_config_from_cli

_BASE_DIR = os.path.dirname(__file__)
_ANOTHER_DIR = os.path.join(_BASE_DIR, 'another')
if _ANOTHER_DIR not in sys.path:
    sys.path.append(_ANOTHER_DIR)
import Alpha_beta_optimize as ai_player
import Global_variables as gv

CPU_PARALLEL_ENVS = 16
GPU_PARALLEL_ENVS = 0
CPU_BATCH_SIZE = 128
GPU_BATCH_SIZE = 256

def get_device_config():
    use_cuda = torch.cuda.is_available()
    main_device = torch.device("cuda" if use_cuda else "cpu")
    print(f"[初始化] 主设备: {main_device} | 支持CUDA: {use_cuda}")
    return main_device, use_cuda

def linear_schedule(start, end, current, decay):
    if decay <= 0:
        return end
    ratio = max(0.0, min(1.0, current / decay))
    return start + (end - start) * ratio

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
    try:
        env = Gomoku(board_size, win_condition)
        print(f"[子进程启动] {env_id} 启动, 设备: {device_type}")
    except Exception as e:
        experience_queue.put({'env_id': env_id, 'error': f'初始化环境失败: {e}'})
        return
    device = torch.device("cuda" if device_type == "gpu" else "cpu")
    local_model = GomokuNetDyn().to(device)
    local_model.eval()
    latest_global_episodes = 0

    while True:
        try:
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
                    eps = linear_schedule(Config.EPSILON1_START, Config.EPSILON1_END, latest_global_episodes, Config.EPSILON_DECAY)
                    action = get_valid_action(logits, board_flat, board_size, epsilon=eps)
                else:
                    reset_another_ai(board_size, env.board)
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
                # 为失败方添加一个终局惩罚，但只加在失败方的最后一步上，
                # 同时保留中间步骤的棋形奖励，方便后续计算蒙特卡洛回报。
                winner = env.current_player
                loser = 3 - winner
                base_lose_penalty = Config.REWARD["lose"]
                step_based_penalty = Config.LOSE_STEP_PENALTY * env.step_count
                total_lose_penalty = base_lose_penalty + step_based_penalty
                # 找到失败方最后一步并加上惩罚
                last_idx = None
                for idx in range(len(episode_experience) - 1, -1, -1):
                    if episode_experience[idx][3] == loser:
                        last_idx = idx
                        break
                if last_idx is not None:
                    s, a, r, p = episode_experience[last_idx]
                    episode_experience[last_idx] = (s, a, r + total_lose_penalty, p)

            if episode_experience:
                try:
                    experience_queue.put({
                        'env_id': env_id,
                        'device_type': device_type,
                        'experiences': episode_experience,
                        'first_player': first_player,
                        'winner': env.current_player if done else 0
                    })
                except Exception as e:
                    # 若队列提交失败，把错误消息发回主进程
                    try:
                        experience_queue.put({'env_id': env_id, 'error': f'提交经验失败: {e}'})
                    except Exception:
                        pass
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            try:
                experience_queue.put({'env_id': env_id, 'error': f'子进程异常: {e}\n{tb}'})
            except Exception:
                pass
            # 发生未捕获异常时退出子进程循环
            return

def update_model_batch(model, optimizer, policy_criterion, value_criterion, batch, device):
    states, actions, rewards = zip(*batch)
    states = torch.cat(states).to(device)
    actions = torch.tensor(actions, device=device, dtype=torch.long)
    rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
    logits, values = model(states)
    # CrossEntropyLoss returns per-sample loss; multiply by return (can be negative)
    policy_loss = (policy_criterion(logits, actions) * rewards).mean()
    # Ensure values shape matches rewards
    if values.dim() > 1:
        values = values.view(-1)
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

    if not hasattr(args, 'board_size') or args.board_size is None:
        Config.BOARD_SIZE = 8

    main_device, use_cuda = get_device_config()
    batch_size = GPU_BATCH_SIZE if main_device.type == 'cuda' else CPU_BATCH_SIZE

    REPLAY_BUFFER_SIZE = 30000
    PRINT_INTERVAL = 50
    total_cells = Config.BOARD_SIZE * Config.BOARD_SIZE

    model = GomokuNetDyn().to(main_device)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    load_model_if_exists(model, 'gobang_best_model_dy.pth')
    best_model = GomokuNetDyn().to(main_device)
    best_model.load_state_dict(model.state_dict())
    history_pool = deque([deepcopy(best_model)], maxlen=3)

    policy_criterion = nn.CrossEntropyLoss(reduction='none')
    value_criterion = nn.MSELoss()

    # 回放缓冲区——长期保留，跨批次复用经验
    replay_buffer1 = deque(maxlen=REPLAY_BUFFER_SIZE)
    GAMMA = 0.99

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
    # 打印子进程状态以便诊断
    for p in processes:
        print(f"  子进程: {getattr(p, 'name', 'proc')} alive={p.is_alive()}")

    def send_model_params(total_episodes):
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

    send_model_params(0)

    total_win1 = 0
    total_win2 = 0
    total_episodes = 0
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
                # 打印子进程存活信息，帮助判断是否有子进程异常退出
                alive_count = sum(1 for p in processes if p.is_alive())
                if alive_count != total_envs:
                    print(f"[警告] 当前存活子进程: {alive_count}/{total_envs}")
                # 检查队列中是否有错误信息
                try:
                    while not experience_queue.empty():
                        msg = experience_queue.get_nowait()
                        if isinstance(msg, dict) and 'error' in msg:
                            print(f"[子进程错误] {msg.get('env_id')}: {msg.get('error')}")
                        else:
                            # 如果不是错误消息，把它放回队列以便后续处理
                            experience_queue.put(msg)
                            break
                except Exception:
                    pass
                if use_cuda:
                    print(f"[设备贡献] CPU: {cpu_diff} 局 | GPU: {gpu_diff} 局")
                last_print_time = current_time
                last_episodes = total_episodes
                last_cpu_episodes = cpu_episodes
                last_gpu_episodes = gpu_episodes

            # 持续消费经验队列（每次循环尽可能多读入新经验）
            while not experience_queue.empty():
                try:
                    exp_data = experience_queue.get_nowait()
                except Exception:
                    break
                # 如果子进程回传错误消息，打印并跳过
                if isinstance(exp_data, dict) and 'error' in exp_data and 'experiences' not in exp_data:
                    print(f"[子进程消息] {exp_data.get('env_id')}: {exp_data.get('error')}")
                    continue

                episode_exps = exp_data.get('experiences', [])  # 单局按顺序经验列表
                if not episode_exps:
                    continue
                winner = exp_data.get('winner', 0)
                first_player = exp_data.get('first_player', 0)
                total_episodes += 1
                if exp_data.get('device_type') == 'cpu':
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

                # 计算蒙特卡洛回报（从后向前累积）并加入长期回放缓冲区
                returns = []
                G = 0.0
                for s, a, r, p in reversed(episode_exps):
                    G = r + GAMMA * G
                    returns.append((s, a, G, p))
                returns.reverse()

                for s, a, G, p in returns:
                    if p == 1:
                        replay_buffer1.append((s, a, G))
                    else:
                        try:
                            s_swapped = s.clone()
                            s_swapped[:, 0:1, :, :] = s[:, 1:2, :, :]
                            s_swapped[:, 1:2, :, :] = s[:, 0:1, :, :]
                            replay_buffer1.append((s_swapped, a, G))
                        except Exception:
                            pass

            # 如果回放缓冲区不足以训练，则短暂等待并继续循环
            if len(replay_buffer1) < batch_size:
                time.sleep(0.01)
                continue

            loss1 = None
            # 从回放缓冲区采样一批用于训练，并从缓冲区中移除被采样的数据，保证新经验被持续消费
            buf_list = list(replay_buffer1)
            random.shuffle(buf_list)
            batch1 = buf_list[:batch_size]
            remaining = buf_list[batch_size:]
            replay_buffer1 = deque(remaining, maxlen=REPLAY_BUFFER_SIZE)
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
                filename = f'gobang_model_player1_dy_step_{update_steps}.pth'
                torch.save(model.state_dict(), filename)
                history_pool.append(deepcopy(model))
                best_model.load_state_dict(model.state_dict())
                send_model_params(total_episodes)
                print(f"[模型保存] 第 {total_episodes} 局模型已保存")
                last_save_step = update_steps
    except KeyboardInterrupt:
        print("\n[中断] 收到 Ctrl+C，正在安全退出...")
    finally:
        for p in processes:
            p.terminate()
        torch.save(model.state_dict(), 'gobang_best_model_dy.pth')
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

