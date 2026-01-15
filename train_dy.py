import os
import sys
import random
import time
import argparse
import math
from collections import deque, defaultdict
from copy import deepcopy
import threading

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

CPU_PARALLEL_ENVS = 0
GPU_PARALLEL_ENVS = 16
CPU_BATCH_SIZE = 128
GPU_BATCH_SIZE = 2048

# --- 性能统计辅助类 ---
class PerformanceTracker:
    def __init__(self):
        self.stats = defaultdict(list)
    
    def add(self, name, duration):
        self.stats[name].append(duration)
        if len(self.stats[name]) > 100:  # 仅保留最近100次采样
            self.stats[name].pop(0)

    def get_avg(self, name):
        if not self.stats[name]: return 0
        return sum(self.stats[name]) / len(self.stats[name])

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

            # 耗时统计记录
            ts_thinking = 0
            ts_inference = 0

            STEPS_PER_ENV = env.board_size * env.board_size
            while not done and total_steps < STEPS_PER_ENV:
                state = env.get_state_representation()
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

                if env.current_player == 1:
                    # 统计推理耗时
                    t_start = time.time()
                    with torch.no_grad():
                        logits, _ = local_model(state_tensor)
                    ts_inference += (time.time() - t_start)

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
                        
                        # 统计AI搜索耗时
                        t_start = time.time()
                        pos = ai_player.machine_thinking(level)
                        ts_thinking += (time.time() - t_start)

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
                        'winner': env.current_player if done else 0,
                        # 传回耗时数据
                        'worker_stats': {
                            'avg_thinking': ts_thinking / max(1, total_steps),
                            'avg_inference': ts_inference / max(1, total_steps)
                        }
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

    # 初始化性能跟踪
    perf_tracker = PerformanceTracker()

    REPLAY_BUFFER_SIZE = 60000
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

    # 共享变量用于打印日志
    training_info = {"loss1": None, "update_steps": 0, "last_save_step": 0}
    train_thread_running = True

    # --- 异步训练线程函数 ---
    def async_train_worker():
        nonlocal replay_buffer1
        while train_thread_running:
            if len(replay_buffer1) >= batch_size:
                # 采样并更新缓冲区
                buf_list = list(replay_buffer1)
                random.shuffle(buf_list)
                batch1 = buf_list[:batch_size]
                remaining = buf_list[batch_size:]
                replay_buffer1 = deque(remaining, maxlen=REPLAY_BUFFER_SIZE)
                
                # 执行训练
                t_train_start = time.time()
                l1, _ = update_model_batch(model, optimizer, policy_criterion, value_criterion, batch1, main_device)
                perf_tracker.add('main_gpu_train', time.time() - t_train_start)
                
                # 更新状态
                training_info["loss1"] = l1
                training_info["update_steps"] += 1
                scheduler.step()
            else:
                time.sleep(0.01) # 缓冲区不足时稍作等待

    # 启动异步训练线程
    train_thread = threading.Thread(target=async_train_worker, daemon=True)
    train_thread.start()

    total_win1 = 0
    total_win2 = 0
    total_draws = 0
    total_episodes = 0
    first_player1_wins = 0
    first_player1_draws = 0
    first_player2_wins = 0
    first_player2_draws = 0
    total_first1 = 0
    total_first2 = 0
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
                
                # 打印性能元凶分析
                print(f"[性能耗时看板 - 最近100次采样平均]")
                print(f"  1. 子进程-AI搜索 (Thinking): {perf_tracker.get_avg('child_thinking')*1000:.2f} ms/步")
                print(f"  2. 子进程-模型推理 (Inference): {perf_tracker.get_avg('child_inference')*1000:.2f} ms/步")
                print(f"  3. 主进程-Queue读取 (QueueGet): {perf_tracker.get_avg('main_queue_get')*1000:.2f} ms/循环")
                print(f"  4. 主进程-GPU异步训练 (GPUTrain): {perf_tracker.get_avg('main_gpu_train')*1000:.2f} ms/批次")

                # 打印子进程存活信息，帮助判断是否有子进程异常退出
                alive_count = sum(1 for p in processes if p.is_alive())
                if alive_count != total_envs:
                    print(f"[警告] 当前存活子进程: {alive_count}/{total_envs}")
                
                if use_cuda:
                    print(f"[设备贡献] CPU: {cpu_diff} 局 | GPU: {gpu_diff} 局")
                last_print_time = current_time
                last_episodes = total_episodes
                last_cpu_episodes = cpu_episodes
                last_gpu_episodes = gpu_episodes

            # 主循环主要负责极其高效地“收割”经验
            t_q_start = time.time()
            while not experience_queue.empty():
                try:
                    exp_data = experience_queue.get_nowait()
                except Exception:
                    break
                
                if isinstance(exp_data, dict) and 'error' in exp_data and 'experiences' not in exp_data:
                    print(f"[子进程消息] {exp_data.get('env_id')}: {exp_data.get('error')}")
                    continue

                if 'worker_stats' in exp_data:
                    perf_tracker.add('child_thinking', exp_data['worker_stats']['avg_thinking'])
                    perf_tracker.add('child_inference', exp_data['worker_stats']['avg_inference'])

                episode_exps = exp_data.get('experiences', [])
                if not episode_exps:
                    continue
                winner = exp_data.get('winner', 0)
                first_player = exp_data.get('first_player', 0)
                total_episodes += 1
                
                if first_player == 1:
                    total_first1 += 1
                    if winner == 1:
                        total_win1 += 1
                        first_player1_wins += 1
                    elif winner == 2:
                        total_win2 += 1
                    else:
                        total_draws += 1
                        first_player1_draws += 1
                else:
                    total_first2 += 1
                    if winner == 1:
                        total_win1 += 1
                    elif winner == 2:
                        total_win2 += 1
                        first_player2_wins += 1
                    else:
                        total_draws += 1
                        first_player2_draws += 1
                
                if exp_data.get('device_type') == 'cpu':
                    cpu_episodes += 1
                else:
                    gpu_episodes += 1

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
            perf_tracker.add('main_queue_get', time.time() - t_q_start)

            # 打印与保存逻辑（在主循环中按局数触发）
            if (total_episodes % PRINT_INTERVAL) == 0 and total_episodes > 0:
                # 计算各种率
                win_rate1 = (total_win1 / total_episodes * 100)

                # AI先手情况下的统计
                f1_win_rate = (first_player1_wins / total_first1 * 100 if total_first1 else 0)
                f1_draw_rate = (first_player1_draws / total_first1 * 100 if total_first1 else 0)

                # 陪练先手情况下的统计 (陪练的胜率)
                f2_win_rate = (first_player2_wins / total_first2 * 100 if total_first2 else 0)
                f2_draw_rate = (first_player2_draws / total_first2 * 100 if total_first2 else 0)

                print(f"[进度] 第 {total_episodes}/{Config.MAX_EPISODES} 局")
                print(f"  AI总胜率: P1: {win_rate1:.1f}% ({total_win1}/{total_episodes - total_win1})")
                print(f"  AI先手的胜率: AI先手: {f1_win_rate:.1f}% |  平局率: {f1_draw_rate:.1f}%")
                print(f"  陪练先手的胜率: {f2_win_rate:.1f}% | 平局率: {f2_draw_rate:.1f}%")
                
                if training_info["loss1"] is not None:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"  损失值: P1: {training_info['loss1']:.4f} | 学习率: {current_lr:.6f} | 更新步数: {training_info['update_steps']}")
                print("  ------------------------")

            # 保存模型
            u_steps = training_info["update_steps"]
            if (u_steps - training_info["last_save_step"]) >= Config.SAVE_INTERVAL and u_steps > 0:
                filename = f'gobang_model_player1_dy_step_{u_steps}.pth'
                torch.save(model.state_dict(), filename)
                history_pool.append(deepcopy(model))
                best_model.load_state_dict(model.state_dict())
                send_model_params(total_episodes)
                print(f"[模型保存] 第 {total_episodes} 局模型已保存 (更新步数: {u_steps})")
                training_info["last_save_step"] = u_steps
            
            # 主循环不再等待训练，但为了避免 CPU 占用过高，增加极小延迟
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n[中断] 收到 Ctrl+C，正在安全退出...")
    finally:
        train_thread_running = False
        train_thread.join(timeout=1)
        for p in processes:
            p.terminate()
        torch.save(model.state_dict(), 'gobang_best_model_dy.pth')
        print("\n[训练结束] 最终模型已保存")
        total_time = time.time() - start_time
        if total_time > 0:
            print(f"总训练速度: {total_episodes / total_time:.2f} 局/秒")

if __name__ == "__main__":
    train()