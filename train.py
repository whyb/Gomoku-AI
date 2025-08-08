import math
import random
import torch
#import torch_directml
import torch.nn as nn
import torch.optim as optim
import argparse
from collections import deque
from copy import deepcopy
from torch.multiprocessing import Process, Queue, set_start_method
import time
from model import Gomoku, GomokuNetV3, get_valid_action, load_model_if_exists, is_adjacent_to_piece
from config import Config, update_config_from_cli

CPU_PARALLEL_ENVS = 0
GPU_PARALLEL_ENVS = 8
CPU_BATCH_SIZE = 128
GPU_BATCH_SIZE = 512

def get_device_config():
    use_cuda = torch.cuda.is_available()
    main_device = torch.device("cuda" if use_cuda else "cpu")
    print(f"[初始化] 主设备: {main_device} | 支持CUDA: {use_cuda}")
    return main_device, use_cuda

# def get_device_config2():
#     index = 1
#     use_dml = torch_directml.is_available()
#     if use_dml:
#         for i in range(torch_directml.device_count()):
#             print("[", i, "]", torch_directml.device_name(i))
#     main_device = torch_directml.device(index) if use_dml else torch.device("cpu")
#     print(f"[初始化] 主设备: {torch_directml.device_name(index)} | 支持DML: {use_dml}")
#     return main_device, use_dml

def setup_players_and_optimizers(device, board_size):
    model1 = GomokuNetV3(board_size).to(device)
    model2 = GomokuNetV3(board_size).to(device)
    optimizer1 = optim.Adam(model1.parameters(), lr=Config.LEARNING_RATE)
    optimizer2 = optim.Adam(model2.parameters(), lr=Config.LEARNING_RATE)
    print(f"[初始化] 模型创建完成 - 棋盘尺寸: {board_size}x{board_size}")
    return model1, model2, optimizer1, optimizer2

def get_dynamic_epsilon(steps_taken, total_cells, min_epsilon=0.05):
    if steps_taken == 0:
        return 1.0
    decay_ratio = min(steps_taken / (total_cells - 1), 1.0)
    epsilon = 1.0 - (1.0 - min_epsilon) * decay_ratio
    return max(epsilon, min_epsilon)


def env_worker(env_id, board_size, win_condition, model_queue, experience_queue, total_cells, device_type):
    env = Gomoku(board_size, win_condition)
    device = torch.device("cuda" if device_type == "gpu" else "cpu")
    #device, use_cuda = get_device_config2()
    
    local_model1 = GomokuNetV3(board_size).to(device)
    local_model2 = GomokuNetV3(board_size).to(device)
    local_best = GomokuNetV3(board_size).to(device)
    local_history = []
    
    local_model1.eval()
    local_model2.eval()
    local_best.eval()

    while True:
        if not model_queue.empty():
            try:
                model_params = model_queue.get(block=False)
                local_model1.load_state_dict({k: v.to(device) for k, v in model_params['model1'].items()})
                local_model2.load_state_dict({k: v.to(device) for k, v in model_params['model2'].items()})
                local_best.load_state_dict({k: v.to(device) for k, v in model_params['best'].items()})
                local_history = []
                for hist_model in model_params['history']:
                    hist_model_device = GomokuNetV3(board_size).to(device)
                    hist_model_device.load_state_dict({k: v.to(device) for k, v in hist_model.items()})
                    hist_model_device.eval()
                    local_history.append(hist_model_device)
            except:
                pass

        env.reset()
        done = False
        total_steps = 0
        episode_experience = []
        first_player = random.choice([1, 2])
        env.current_player = first_player

        STEPS_PER_ENV = env.board_size * env.board_size * 0.75 #每个并行游戏环境env最多执行的落子步数上限
        while not done and total_steps < STEPS_PER_ENV:
            state = env.get_state_representation()
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            current_epsilon = get_dynamic_epsilon(total_steps, total_cells)
            
            with torch.no_grad():
                if env.current_player == 1:
                    logits, _ = local_model1(state_tensor)
                else:
                    opponent = random.choice([local_model2, local_best] + local_history)
                    logits, _ = opponent(state_tensor)

            board_flat = torch.tensor(env.board.flatten(), device=device)
            action = get_valid_action(logits, board_flat, board_size, epsilon=current_epsilon)
            
            current_player = env.current_player
            next_player, done, reward = env.step(action)
            # 处理胜利时的奖励（区分普通奖励和胜利奖励）
            if done:
                # 胜利时的reward是元组：(基础win奖励, 落子步数)
                base_win_reward, step_count = reward
                # 计算速度奖励：棋盘总格子数 - 实际步数（步数越少，奖励越高）
                total_cells = env.board_size * env.board_size
                speed_bonus = (total_cells - step_count) * Config.SPEED_REWARD_COEFFICIENT
                # 总胜利奖励 = 基础奖励 + 速度奖励
                final_reward = base_win_reward + speed_bonus
                episode_experience.append((state_tensor.cpu().detach(), action, final_reward, current_player))
            else:
                # 普通奖励直接记录
                episode_experience.append((state_tensor.cpu().detach(), action, reward, current_player))
            total_steps += 1

        # 处理失败方的惩罚（随步数增加而加重）
        if done:
            winner = env.current_player
            loser = 3 - winner  # 确定失败方
            # 失败惩罚 = 基础惩罚 + 步数×惩罚系数（步数越多，惩罚越重）
            base_lose_penalty = Config.REWARD["lose"]
            step_based_penalty = total_steps * Config.LOSE_STEP_PENALTY  # 每步增加的惩罚
            total_lose_penalty = base_lose_penalty + step_based_penalty

            # 遍历经验列表，更新失败方的所有步骤奖励
            for i in range(len(episode_experience)):
                state, action, reward, player = episode_experience[i]
                if player == loser:
                    # 失败方的每一步都叠加总失败惩罚（强化“拖延必败”的信号）
                    episode_experience[i] = (state, action, total_lose_penalty, player)

        if episode_experience:
            experience_queue.put({
                'env_id': env_id,
                'device_type': device_type,
                'experiences': episode_experience,
                'first_player': first_player,
                'winner': env.current_player if done else 0
            })

def update_model_batch(model, optimizer, policy_criterion, value_criterion, batch, device):
    states, actions, rewards = zip(*batch)
    states = torch.cat(states).to(device)
    actions = torch.tensor(actions, device=device)
    rewards = torch.tensor(rewards, device=device, dtype=torch.float32)

    # 策略梯度损失
    logits, values = model(states)
    
    # 计算策略损失（加权交叉熵）
    policy_loss = policy_criterion(logits, actions) * rewards
    policy_loss = policy_loss.mean()
    
    # 计算价值损失（均方误差）
    value_loss = value_criterion(values, rewards)
    
    # 总损失 = 策略损失 + 价值损失
    loss = policy_loss + value_loss
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return loss.item(), rewards.mean().item()


def train():
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--board_size", type=int, help="Size of the game board")
    parser.add_argument("--win_condition", type=int, help="Number of consecutive stones to win")
    args = parser.parse_args()
    config = update_config_from_cli(args)

    main_device, use_cuda = get_device_config()
    batch_size = GPU_BATCH_SIZE if main_device.type == 'cuda' else CPU_BATCH_SIZE
    #main_device, use_cuda = get_device_config2()
    #batch_size = GPU_BATCH_SIZE
    
    REPLAY_BUFFER_SIZE = 30000
    UPDATE_FREQ = 4
    PRINT_INTERVAL = 50
    HISTORY_POOL_SIZE = 3
    total_cells = config.BOARD_SIZE * config.BOARD_SIZE
    
    model1, model2, optimizer1, optimizer2 = setup_players_and_optimizers(main_device, config.BOARD_SIZE)
    history_pool = deque(maxlen=HISTORY_POOL_SIZE)
    replay_buffer1 = deque(maxlen=REPLAY_BUFFER_SIZE)
    replay_buffer2 = deque(maxlen=REPLAY_BUFFER_SIZE)
    
    load_model_if_exists(model1, 'gobang_best_model.pth')
    best_model = GomokuNetV3(config.BOARD_SIZE).to(main_device)
    best_model.load_state_dict(model1.state_dict())
    history_pool.append(deepcopy(best_model))

    # 策略损失（带奖励加权的交叉熵）和价值损失（均方误差）
    policy_criterion = nn.CrossEntropyLoss(reduction='none')
    value_criterion = nn.MSELoss()

    model_queue = Queue()
    experience_queue = Queue()

    processes = []
    total_envs = 0
    for env_id in range(CPU_PARALLEL_ENVS):
        p = Process(
            target=env_worker,
            args=(
                f"cpu-{env_id}", config.BOARD_SIZE, config.WIN_CONDITION,
                model_queue, experience_queue, total_cells, "cpu"
            ),
            daemon=True
        )
        p.start()
        processes.append(p)
        total_envs += 1

    if use_cuda:
        for env_id in range(GPU_PARALLEL_ENVS):
            p = Process(
                target=env_worker, 
                args=(
                    f"gpu-{env_id}", config.BOARD_SIZE, config.WIN_CONDITION,
                    model_queue, experience_queue, total_cells, "gpu"
                ),
                daemon=True
            )
            p.start()
            processes.append(p)
            total_envs += 1
    
    print(f"[并行初始化] 启动 {total_envs} 个并行游戏环境")

    def send_model_params():
        model_params = {
            'model1': {k: v.cpu() for k, v in model1.state_dict().items()},
            'model2': {k: v.cpu() for k, v in model2.state_dict().items()},
            'best': {k: v.cpu() for k, v in best_model.state_dict().items()},
            'history': [{k: v.cpu() for k, v in m.state_dict().items()} for m in history_pool]
        }
        while not model_queue.empty():
            try: model_queue.get_nowait()
            except: pass
        for _ in range(total_envs):
            model_queue.put(model_params)
    send_model_params()

    total_win1 = 0
    total_win2 = 0
    total_episodes = 0
    first_player1_wins = 0
    first_player2_wins = 0
    total_first1 = 0
    total_first2 = 0
    update_steps = 0
    cpu_episodes = 0
    gpu_episodes = 0
    start_time = time.time()
    last_print_time = start_time
    last_episodes = 0
    last_cpu_episodes = 0
    last_gpu_episodes = 0

    print("\n[训练开始] --------------")
    print(f"最大回合数: {config.MAX_EPISODES}")
    print(f"并行环境数: {total_envs} | 批量大小: {batch_size}")
    print(f"棋盘尺寸: {config.BOARD_SIZE}x{config.BOARD_SIZE} | 模型保存间隔: {config.SAVE_INTERVAL}局\n")

    try:
        while total_episodes < config.MAX_EPISODES:
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
            while len(experiences) < batch_size and total_episodes < config.MAX_EPISODES:
                if not experience_queue.empty():
                    exp_data = experience_queue.get()
                    experiences.extend(exp_data['experiences'])
                    
                    winner = exp_data['winner']
                    first_player = exp_data['first_player']
                    total_episodes += 1
                    
                    if exp_data['device_type'] == 'cpu': cpu_episodes += 1
                    else: gpu_episodes += 1
                    
                    if winner == 1:
                        total_win1 += 1
                        if first_player == 1: first_player1_wins += 1
                    elif winner == 2:
                        total_win2 += 1
                        if first_player == 2: first_player2_wins += 1
                    
                    if first_player == 1: total_first1 += 1
                    else: total_first2 += 1

            for state, action, reward, player in experiences:
                if player == 1: replay_buffer1.append((state, action, reward))
                else: replay_buffer2.append((state, action, reward))

            loss1 = loss2 = None
            if len(replay_buffer1) >= batch_size and len(replay_buffer2) >= batch_size:
                batch1 = random.sample(replay_buffer1, batch_size)
                batch2 = random.sample(replay_buffer2, batch_size)
                
                loss1, _ = update_model_batch(model1, optimizer1, policy_criterion, value_criterion, batch1, main_device)
                loss2, _ = update_model_batch(model2, optimizer2, policy_criterion, value_criterion, batch2, main_device)
                update_steps += 1
            
            if (total_episodes % PRINT_INTERVAL) == 0 and total_episodes > 0:
                total_win_rate1 = (total_win1 / total_episodes * 100 if total_episodes else 0)
                first1_rate = (first_player1_wins / total_first1 * 100 if total_first1 else 0)
                first2_rate = (first_player2_wins / total_first2 * 100 if total_first2 else 0)
                print(f"[进度] 第 {total_episodes}/{config.MAX_EPISODES} 局")
                print(f"  胜率: P1: {total_win_rate1:.1f}% ({total_win1}/{total_win2})")
                print(f"  先手胜率: P1先手: {first1_rate:.1f}% | P2先手: {first2_rate:.1f}%")
                if loss1 is not None:
                    print(f"  损失值: P1: {loss1:.4f} / P2: {loss2:.4f}")
                print("  ------------------------")

            if (total_episodes % config.SAVE_INTERVAL) == 0 and total_episodes > 0:
                torch.save(model1.state_dict(), f'gobang_model_player1_{total_episodes}.pth')
                history_pool.append(deepcopy(model1))
                print(f"[历史池更新] 已保存第 {total_episodes} 局模型，当前池大小: {len(history_pool)}")
                
                if history_pool:
                    # 玩家2更新逻辑：从历史池中随机选择一个模型作为陪练
                    model2.load_state_dict(random.choice(history_pool).state_dict())
                    optimizer2 = optim.Adam(model2.parameters(), lr=Config.LEARNING_RATE)
                
                best_model.load_state_dict(model1.state_dict())
                send_model_params()
                print(f"[模型保存] 第 {total_episodes} 局模型已保存")
    finally:
        for p in processes: p.terminate()
        torch.save(model1.state_dict(), 'gobang_best_model.pth')
        print("\n[训练结束] 最终模型已保存")
        total = total_win1 + total_win2
        if total > 0:
            print(f"总胜率: P1: {total_win1/total*100:.1f}% ({total_win1}/{total_win2})")
            print(f"先手胜率: P1先手: {first_player1_wins/total_first1*100:.1f}% | P2先手: {first_player2_wins/total_first2*100:.1f}%")
        total_time = time.time() - start_time
        if total_time > 0:
            total_speed = total_episodes / total_time
            print(f"总训练速度: {total_speed:.2f} 局/秒 (共 {total_episodes} 局，耗时 {total_time:.1f} 秒)")
            if use_cuda:
                print(f"设备贡献: CPU: {cpu_episodes} 局 ({cpu_episodes/total_episodes*100:.1f}%) | GPU: {gpu_episodes} 局 ({gpu_episodes/total_episodes*100:.1f}%)")


if __name__ == "__main__":
    train()