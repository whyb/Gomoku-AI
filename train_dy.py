import os
import sys
import random
import time
import argparse
import math
import numpy as np
from collections import deque
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from model import Gomoku, get_valid_action
from model_dy import GomokuNetDyn, load_model_if_exists
from config import Config, update_config_from_cli
from mcts import MCTS

_BASE_DIR = os.path.dirname(__file__)
_ANOTHER_DIR = os.path.join(_BASE_DIR, 'another')
if _ANOTHER_DIR not in sys.path:
    sys.path.append(_ANOTHER_DIR)
import Alpha_beta_optimize as ai_player
import Global_variables as gv

CPU_PARALLEL_ENVS = 0
GPU_PARALLEL_ENVS = 16
CPU_BATCH_SIZE = 128
GPU_BATCH_SIZE = 512

class AlphaZeroLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.value_criterion = nn.MSELoss()

    def forward(self, policy_logits, value, target_pis, target_vs):
        """
        policy_logits: 网络输出的策略 (batch, 225) - 未经过 softmax
        value: 网络输出的胜率 (batch, 1) or (batch)
        target_pis: MCTS 搜索得到的概率分布 (batch, 225)
        target_vs: 真实游戏结果 (batch)
        """
        # 1. Value Loss (MSE)
        value_loss = self.value_criterion(value.view(-1), target_vs.view(-1))

        # 2. Policy Loss (Cross Entropy with Soft Targets)
        # 公式: - sum( target_pi * log(predicted_pi) )
        log_probs = F.log_softmax(policy_logits, dim=1)
        policy_loss = -torch.mean(torch.sum(target_pis * log_probs, dim=1))

        # 总损失
        total_loss = policy_loss + value_loss
        return total_loss, policy_loss.item(), value_loss.item()

# --- 高效 Replay Buffer (List + Swap Remove) ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity

    def push(self, item):
        self.buffer.append(item)
        # 如果超出容量，移除最早的元素 (FIFO)，保持缓冲区大小合理
        # 注意：在当前逻辑中，数据被采样后即移除，capacity 主要作为安全上限
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample_and_remove(self, batch_size):
        """
        随机采样 batch_size 个元素并从 buffer 中移除。
        使用 索引交换 (Swap) 的方式避免列表中间删除带来的 O(N) 开销。
        """
        batch = []
        count = min(len(self.buffer), batch_size)
        for _ in range(count):
            # 随机选择一个索引
            idx = random.randint(0, len(self.buffer) - 1)
            # 将选中元素与最后一个元素交换
            self.buffer[idx], self.buffer[-1] = self.buffer[-1], self.buffer[idx]
            # 弹出最后一个元素（即刚才选中的元素）
            batch.append(self.buffer.pop())
        return batch

    def __len__(self):
        return len(self.buffer)

# ---  DataLoader 辅助类 ---
class BatchDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    """
    自定义 batch 整理函数：将 list of tuples 转换为堆叠后的 Tensor
    """
    # 此时 batch 的结构是 [(state, probs, reward), ...]
    states, target_pis, rewards = zip(*batch)
    
    states = torch.cat(states) # (Batch, 2, 15, 15)
    
    # 将 numpy 数组列表转为 Tensor
    target_pis = torch.tensor(np.array(target_pis), dtype=torch.float32) # (Batch, 225)
    
    # 奖励转为 Tensor
    rewards = torch.tensor(rewards, dtype=torch.float32)
    
    return states, target_pis, rewards

def alpha_zero_loss(policy_logits, value, target_pis, target_vs):
    """
    policy_logits: 网络输出 (未经过 softmax)
    target_pis: MCTS 搜索后的分布
    value: 网络输出的胜率预测
    target_vs: 最终真实结果 (胜+1, 负-1)
    """
    # 1. 策略损失：KL 散度 (相当于 soft-target cross entropy)
    # log_softmax + (target * log_probs)
    log_probs = F.log_softmax(policy_logits, dim=1)
    policy_loss = -torch.mean(torch.sum(target_pis * log_probs, dim=1))
    
    # 2. 价值损失：MSE
    value_loss = F.mse_loss(value.view(-1), target_vs)
    
    return policy_loss + value_loss

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
    
    # 初始化 MCTS
    # 15x15 棋盘，训练时 playout 大概设置为 200-400 之间
    mcts_playout = 300 if board_size >= 15 else 200
    mcts = MCTS(local_model, c_puct=5, n_playout=mcts_playout, board_size=board_size)

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
            p1_step_count = 0 
            p2_step_count = 0 
            episode_experience = []
            first_player = random.choice([1, 2])
            env.current_player = first_player

            STEPS_PER_ENV = env.board_size * env.board_size
            while not done and total_steps < STEPS_PER_ENV:
                state = env.get_state_representation()
                # 保持原逻辑：在此处转为 Tensor，放入队列
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

                # --- 被训练 AI (Player 1) 逻辑 ---
                if env.current_player == 1:
                    # --- AlphaZero 核心逻辑 ---
                    # 温度参数控制：前期探索大(temp=1)，后期趋向确定的策略(temp->0)
                    temp = 1.0 if len(episode_experience) < 10 else 0.5 # 比如前10步比较随机
                    
                    # 获取 MCTS 概率分布 (mcts_probs 是长度为 225 的 numpy array)
                    action, mcts_probs = mcts.get_move_probs(env, temp=temp)
                
                # --- 陪练 AI (Player 2) 逻辑 ---
                else:
                    p2_step_count += 1
                    if p2_step_count == 1: 
                        valid_actions = [i for i in range(total_cells) if env.board.flatten()[i] == 0]
                        action = random.choice(valid_actions)
                    else:
                        reset_another_ai(board_size, env.board)
                        strong_prob = linear_schedule(Config.OPP_STRONG_PROB_START, Config.OPP_STRONG_PROB_END, latest_global_episodes, Config.OPP_STRONG_DECAY)
                        random_prob = linear_schedule(Config.OPP_RANDOM_PROB_START, Config.OPP_RANDOM_PROB_END, latest_global_episodes, Config.OPP_RANDOM_DECAY)
                        
                        r = random.random()
                        if r < random_prob:
                            valid_actions = [i for i in range(total_cells) if env.board.flatten()[i] == 0]
                            action = random.choice(valid_actions) if valid_actions else -1
                        else:
                            level = '比你6的Level' if r < (random_prob + strong_prob) else '和我一样6的Level'
                            pos = ai_player.machine_thinking(level)
                            if not pos:
                                valid_actions = [i for i in range(total_cells) if env.board.flatten()[i] == 0]
                                action = random.choice(valid_actions) if valid_actions else -1
                            else:
                                i, j = pos
                                action = i * board_size + j
                    
                    # 构造陪练的 one-hot 分布。
                    # 虽然我们主要训练 P1，但如果你也要训练 P2，这里也需要存储分布
                    mcts_probs = np.zeros(total_cells, dtype=np.float32)
                    if action != -1: 
                        mcts_probs[action] = 1.0

                current_player = env.current_player
                next_player, done, reward = env.step(action)
                
                # --- 核心存储修改：存储 mcts_probs 而不是 action ---
                if done:
                    base_win_reward, step_count = reward
                    speed_bonus = Config.SPEED_REWARD_COEFFICIENT * (total_cells - step_count)
                    final_reward = base_win_reward + speed_bonus
                    # 将 action 替换为 mcts_probs
                    episode_experience.append((state_tensor.cpu().detach(), mcts_probs, final_reward, current_player))
                else:
                    # 将 action 替换为 mcts_probs
                    episode_experience.append((state_tensor.cpu().detach(), mcts_probs, reward, current_player))

                total_steps += 1

            if done:
                winner = env.current_player
                loser = 3 - winner
                base_lose_penalty = Config.REWARD["lose"]
                step_based_penalty = Config.LOSE_STEP_PENALTY * env.step_count
                total_lose_penalty = base_lose_penalty + step_based_penalty
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
            return



def update_model_batch(model, optimizer, criterion, states, target_pis, rewards):
    model.train()
    policy_logits, values = model(states)
    
    # 调用我们新定义的 AlphaZeroLoss
    loss, loss_pi, loss_v = criterion(policy_logits, values, target_pis, rewards)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return loss.item(), loss_pi, loss_v

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

    #policy_criterion = nn.CrossEntropyLoss(reduction='none')
    criterion = AlphaZeroLoss()
    # value_criterion is integrated into AlphaZeroLoss

    # --- 使用高效的自定义 Replay Buffer ---
    replay_buffer1 = ReplayBuffer(capacity=REPLAY_BUFFER_SIZE)
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
    total_draws = 0
    total_episodes = 0
    first_player1_wins = 0
    first_player1_draws = 0
    first_player2_wins = 0
    first_player2_draws = 0
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
                alive_count = sum(1 for p in processes if p.is_alive())
                if alive_count != total_envs:
                    print(f"[警告] 当前存活子进程: {alive_count}/{total_envs}")
                try:
                    while not experience_queue.empty():
                        msg = experience_queue.get_nowait()
                        if isinstance(msg, dict) and 'error' in msg:
                            print(f"[子进程错误] {msg.get('env_id')}: {msg.get('error')}")
                        else:
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

            # 消费经验
            while not experience_queue.empty():
                try:
                    exp_data = experience_queue.get_nowait()
                except Exception:
                    break
                if isinstance(exp_data, dict) and 'error' in exp_data and 'experiences' not in exp_data:
                    print(f"[子进程消息] {exp_data.get('env_id')}: {exp_data.get('error')}")
                    continue

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

                # 计算蒙特卡洛回报
                returns = []
                G = 0.0
                for s, a, r, p in reversed(episode_exps):
                    G = r + GAMMA * G
                    returns.append((s, a, G, p))
                returns.reverse()

                for s, a, G, p in returns:
                    if p == 1:
                        # 直接 push 到新的 List-based buffer
                        replay_buffer1.push((s, a, G))
                    else:
                        try:
                            s_swapped = s.clone()
                            s_swapped[:, 0:1, :, :] = s[:, 1:2, :, :]
                            s_swapped[:, 1:2, :, :] = s[:, 0:1, :, :]
                            replay_buffer1.push((s_swapped, a, G))
                        except Exception:
                            pass

            if len(replay_buffer1) < batch_size:
                time.sleep(0.01)
                continue

            # --- 从 Buffer 中高效采样并使用 DataLoader 处理数据 ---
            # 1. 高效采样 (O(1) 移除)
            batch_data = replay_buffer1.sample_and_remove(batch_size)
            
            # 2. 包装为 Dataset 并使用 DataLoader
            # num_workers=0 表示在主进程处理，避免频繁 spawn 子进程开销，
            # 但 pin_memory=True 可以让数据传输到 GPU 异步非阻塞
            train_loader = DataLoader(
                BatchDataset(batch_data), 
                batch_size=batch_size, 
                shuffle=False, # 数据在 buffer 采样时已经是随机的
                collate_fn=collate_fn, 
                pin_memory=use_cuda
            )

            loss1 = None
            # DataLoader 循环（此处只会执行一次，因为 batch_size 等于 loader 大小）
            for states, actions, rewards in train_loader:
                # 异步传输数据到 GPU
                states = states.to(main_device, non_blocking=True)
                actions = actions.to(main_device, non_blocking=True)
                rewards = rewards.to(main_device, non_blocking=True)
                
                loss1, loss_pi, loss_v = update_model_batch(model, optimizer, criterion, states, actions, rewards)
            
            scheduler.step()
            update_steps += 1

            if (total_episodes % PRINT_INTERVAL) == 0 and total_episodes > 0:
                win_rate1 = (total_win1 / total_episodes * 100)
                f1_win_rate = (first_player1_wins / total_first1 * 100 if total_first1 else 0)
                f1_draw_rate = (first_player1_draws / total_first1 * 100 if total_first1 else 0)
                f2_win_rate = (first_player2_wins / total_first2 * 100 if total_first2 else 0)
                f2_draw_rate = (first_player2_draws / total_first2 * 100 if total_first2 else 0)

                print(f"[进度] 第 {total_episodes}/{Config.MAX_EPISODES} 局")
                print(f"  AI总胜率: P1: {win_rate1:.1f}% ({total_win1}/{total_episodes - total_win1})")
                print(f"  AI先手的胜率: AI先手: {f1_win_rate:.1f}% |  平局率: {f1_draw_rate:.1f}%")
                print(f"  陪练先手的胜率: {f2_win_rate:.1f}% | 平局率: {f2_draw_rate:.1f}%")
                
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