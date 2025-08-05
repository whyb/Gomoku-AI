import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from collections import deque
from copy import deepcopy
from torch.multiprocessing import Process, Queue, set_start_method
import time
from model import Gomoku, GomokuNetV3, get_valid_action, load_model_if_exists
from config import Config, update_config_from_cli

# 设备特定配置参数
CPU_PARALLEL_ENVS = 8    # CPU环境数量
GPU_PARALLEL_ENVS = 1    # GPU环境数量
STEPS_PER_ENV = 20       # 每个环境每轮生成的步数
CPU_BATCH_SIZE = 256     # CPU批处理大小
GPU_BATCH_SIZE = 256     # GPU批处理大小


def get_device_config():
    """获取设备配置，返回主设备和是否支持CUDA"""
    use_cuda = torch.cuda.is_available()
    main_device = torch.device("cuda" if use_cuda else "cpu")
    print(f"[初始化] 主设备: {main_device} | 支持CUDA: {use_cuda}")
    return main_device, use_cuda


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


def is_adjacent_to_piece(board, x, y, board_size):
    """检查(x,y)位置是否与已有棋子相邻（8个方向）"""
    # 8个方向：上、下、左、右、四个对角线
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),          (0, 1),
        (1, -1),  (1, 0), (1, 1)
    ]
    
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        # 检查邻接位置是否在棋盘内且有棋子
        if 0 <= nx < board_size and 0 <= ny < board_size:
            if board[nx, ny] != 0:
                return True
    return False


def env_worker(env_id, board_size, win_condition, model_queue, experience_queue, 
               total_cells, device_type):
    """环境工作器，支持CPU和GPU两种设备类型"""
    env = Gomoku(board_size, win_condition)
    # 根据设备类型设置计算设备
    device = torch.device("cuda" if device_type == "gpu" else "cpu")
    
    # 初始化本地模型（明确在指定设备上）
    local_model1 = GomokuNetV3(board_size).to(device)
    local_model2 = GomokuNetV3(board_size).to(device)
    local_best = GomokuNetV3(board_size).to(device)
    local_history = []

    # 确保模型处于评估模式
    local_model1.eval()
    local_model2.eval()
    local_best.eval()

    while True:
        # 获取并加载模型参数（根据设备类型转换参数）
        if not model_queue.empty():
            try:
                model_params = model_queue.get(block=False)
                # 加载参数时明确转移到当前设备
                local_model1.load_state_dict(
                    {k: v.to(device) for k, v in model_params['model1'].items()}
                )
                local_model2.load_state_dict(
                    {k: v.to(device) for k, v in model_params['model2'].items()}
                )
                local_best.load_state_dict(
                    {k: v.to(device) for k, v in model_params['best'].items()}
                )
                # 历史模型同样转移到当前设备
                local_history = []
                for hist_model in model_params['history']:
                    hist_model_device = GomokuNetV3(board_size).to(device)
                    hist_model_device.load_state_dict(
                        {k: v.to(device) for k, v in hist_model.items()}
                    )
                    hist_model_device.eval()
                    local_history.append(hist_model_device)
            except:
                pass

        # 生成游戏经验
        env.reset()
        done = False
        total_steps = 0
        episode_experience = []
        first_player = random.choice([1, 2])
        env.current_player = first_player

        while not done and total_steps < STEPS_PER_ENV:
            # 状态张量明确在当前设备上
            state = torch.FloatTensor(env.board.flatten()).unsqueeze(0).to(device)
            current_epsilon = get_dynamic_epsilon(total_steps, total_cells)

            # 选择动作（确保所有计算在当前设备上）
            if env.current_player == 1:
                with torch.no_grad():  # 禁用梯度计算，节省内存
                    logits = local_model1(state)
            else:
                if local_history:
                    opponent = random.choice([local_model2, local_best] + local_history)
                else:
                    opponent = local_model2
                with torch.no_grad():
                    logits = opponent(state)

            # 计算有效动作
            board_flat = torch.tensor(env.board.flatten(), device=device)
            valid_mask = (board_flat == 0)
            valid_indices = torch.where(valid_mask)[0]
            if valid_indices.numel() == 0:
                break

            valid_logits = logits[valid_mask.unsqueeze(0)]
            # 转换为CPU进行后续处理
            valid_actions = list(
                zip(valid_logits.cpu().numpy().flatten(), valid_indices.cpu().numpy())
            )
            valid_actions.sort(reverse=True, key=lambda x: x[0])

            # 选择动作：探索率部分添加位置约束
            action = -1
            if random.random() < current_epsilon:
                # 检查是否是第一次落子（棋盘为空）
                is_first_move = (env.board == 0).all()
                
                if is_first_move:
                    # 第一次落子：全棋盘随机
                    action = random.choice(valid_actions)[1]
                else:
                    # 非第一次：筛选出与已有棋子相邻的位置
                    adjacent_actions = []
                    for logit, idx in valid_actions:
                        x = idx // board_size
                        y = idx % board_size
                        # 检查该位置是否与已有棋子相邻
                        if is_adjacent_to_piece(env.board, x, y, board_size):
                            adjacent_actions.append((logit, idx))
                    
                    # 如果有相邻位置，从相邻位置中随机选择
                    if adjacent_actions:
                        action = random.choice(adjacent_actions)[1]
                    else:
                        # 若无相邻位置（极端情况），退回到所有有效位置
                        action = random.choice(valid_actions)[1]
            else:
                # 利用策略：从top-k中选择
                top_k = min(3, len(valid_actions))
                action = random.choice(valid_actions[:top_k])[1]

            # 执行动作并记录经验（转换为CPU张量以便主进程处理）
            current_player = env.current_player
            next_player, done, reward, _ = env.step(action)
            episode_experience.append(
                (state.cpu().detach(), action, reward, current_player)
            )
            total_steps += 1

        # 发送经验
        if episode_experience:
            experience_queue.put({
                'env_id': env_id,
                'device_type': device_type,  # 记录经验来自哪种设备
                'experiences': episode_experience,
                'first_player': first_player,
                'winner': env.current_player if done else 0
            })


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
    # Windows多进程设置
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--board_size", type=int, help="Size of the game board")
    parser.add_argument("--win_condition", type=int, help="Number of consecutive stones to win")
    args = parser.parse_args()
    config = update_config_from_cli(args)

    # 获取设备配置
    main_device, use_cuda = get_device_config()
    # 根据主设备选择批处理大小
    batch_size = GPU_BATCH_SIZE if main_device.type == 'cuda' else CPU_BATCH_SIZE
    
    # 核心配置
    REPLAY_BUFFER_SIZE = 30000
    UPDATE_FREQ = 4
    PRINT_INTERVAL = 500
    HISTORY_POOL_SIZE = 3
    total_cells = config.BOARD_SIZE * config.BOARD_SIZE
    
    # 初始化模型
    model1, model2, optimizer1, optimizer2 = setup_players_and_optimizers(
        main_device, config.BOARD_SIZE
    )
    history_pool = deque(maxlen=HISTORY_POOL_SIZE)
    replay_buffer1 = deque(maxlen=REPLAY_BUFFER_SIZE)
    replay_buffer2 = deque(maxlen=REPLAY_BUFFER_SIZE)
    
    # 加载模型
    load_model_if_exists(model1, 'gobang_best_model.pth')
    best_model = GomokuNetV3(config.BOARD_SIZE).to(main_device)
    best_model.load_state_dict(model1.state_dict())
    history_pool.append(deepcopy(best_model))
    criterion = nn.CrossEntropyLoss()

    # 进程通信队列
    model_queue = Queue()
    experience_queue = Queue()

    # 启动并行环境
    processes = []
    total_envs = 0
    
    # 启动CPU环境
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
    
    # 启动GPU环境（如果支持CUDA）
    #if use_cuda:
    #    for env_id in range(GPU_PARALLEL_ENVS):
    #        p = Process(
    #            target=env_worker,
    #            args=(
    #                f"gpu-{env_id}", config.BOARD_SIZE, config.WIN_CONDITION, 
    #                model_queue, experience_queue, total_cells, "gpu"
    #            ),
    #            daemon=True
    #        )
    #        p.start()
    #        processes.append(p)
    #        total_envs += 1
    
    print(
        f"[并行初始化] 启动 {total_envs} 个并行游戏环境 "
        f"(CPU: {CPU_PARALLEL_ENVS}, GPU: {GPU_PARALLEL_ENVS if use_cuda else 0})"
    )

    # 发送模型参数（根据设备类型转换参数）
    def send_model_params():
        # 主进程模型参数准备 - 保存为CPU张量以便所有设备都能加载
        model_params = {
            'model1': {k: v.cpu() for k, v in model1.state_dict().items()},
            'model2': {k: v.cpu() for k, v in model2.state_dict().items()},
            'best': {k: v.cpu() for k, v in best_model.state_dict().items()},
            'history': [
                {k: v.cpu() for k, v in m.state_dict().items()} 
                for m in history_pool
            ]
        }
        # 清空队列
        while not model_queue.empty():
            try:
                model_queue.get_nowait()
            except:
                pass
        # 为每个环境发送一份参数
        for _ in range(total_envs):
            model_queue.put(model_params)
    send_model_params()

    # 统计变量
    total_win1 = 0
    total_win2 = 0
    total_episodes = 0
    first_player1_wins = 0
    first_player2_wins = 0
    total_first1 = 0
    total_first2 = 0
    update_steps = 0
    
    # 设备特定统计
    cpu_episodes = 0
    gpu_episodes = 0
    
    # 速度计算变量
    start_time = time.time()
    last_print_time = start_time  # 上一次打印速度的时间
    last_episodes = 0  # 上一次打印时的总局数
    last_cpu_episodes = 0  # 上一次打印时的CPU局数
    last_gpu_episodes = 0  # 上一次打印时的GPU局数

    print("\n[训练开始] --------------")
    print(f"最大回合数: {config.MAX_EPISODES}")
    print(f"并行环境数: {total_envs} | 批量大小: {batch_size}")
    print(
        f"棋盘尺寸: {config.BOARD_SIZE}x{config.BOARD_SIZE} | "
        f"模型保存间隔: {config.SAVE_INTERVAL}局\n"
    )

    try:
        while total_episodes < config.MAX_EPISODES:
            # 检查是否需要打印速度（每隔10秒）
            current_time = time.time()
            if current_time - last_print_time >= 10:
                # 计算时间差和局数差
                time_diff = current_time - last_print_time
                episodes_diff = total_episodes - last_episodes
                # 计算各设备在这段时间内的贡献
                cpu_diff = cpu_episodes - last_cpu_episodes
                gpu_diff = gpu_episodes - last_gpu_episodes
                
                # 计算每秒局数（避免除零）
                speed = episodes_diff / time_diff if time_diff > 0 else 0
                print(
                    f"[速度统计] 过去 {time_diff:.1f} 秒完成 {episodes_diff} 局 | "
                    f"平均速度: {speed:.2f} 局/秒"
                )
                if use_cuda:
                    print(
                        f"[设备贡献] CPU: {cpu_diff} 局 | GPU: {gpu_diff} 局"
                    )
                # 更新记录
                last_print_time = current_time
                last_episodes = total_episodes
                last_cpu_episodes = cpu_episodes
                last_gpu_episodes = gpu_episodes

            # 收集经验
            experiences = []
            while len(experiences) < batch_size and total_episodes < config.MAX_EPISODES:
                if not experience_queue.empty():
                    exp_data = experience_queue.get()
                    experiences.extend(exp_data['experiences'])
                    
                    # 更新统计
                    winner = exp_data['winner']
                    first_player = exp_data['first_player']
                    total_episodes += 1
                    
                    # 更新设备特定统计
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

            # 分离经验到回放池
            for state, action, reward, player in experiences:
                if player == 1:
                    replay_buffer1.append((state, action, reward))
                else:
                    replay_buffer2.append((state, action, reward))

            # 模型更新
            loss1 = loss2 = None
            if len(replay_buffer1) >= batch_size and len(replay_buffer2) >= batch_size:
                batch1 = random.sample(replay_buffer1, batch_size)
                batch2 = random.sample(replay_buffer2, batch_size)
                
                loss1, _ = update_model_batch(
                    model1, optimizer1, criterion, batch1, main_device
                )
                loss2, _ = update_model_batch(
                    model2, optimizer2, criterion, batch2, main_device
                )
                update_steps += 1

                if update_steps % UPDATE_FREQ == 0:
                    send_model_params()

            # 打印进度
            if total_episodes % PRINT_INTERVAL == 0 and total_episodes > 0:
                total_win_rate1 = (
                    total_win1 / total_episodes * 100 if total_episodes else 0
                )
                first1_rate = (
                    first_player1_wins / total_first1 * 100 if total_first1 else 0
                )
                first2_rate = (
                    first_player2_wins / total_first2 * 100 if total_first2 else 0
                )

                print(f"[进度] 第 {total_episodes}/{config.MAX_EPISODES} 局")
                print(
                    f"  胜率: P1: {total_win_rate1:.1f}% ({total_win1}/{total_win2})"
                )
                print(
                    f"  先手胜率: P1先手: {first1_rate:.1f}% | "
                    f"P2先手: {first2_rate:.1f}%"
                )
                if loss1 is not None:
                    print(
                        f"  损失值: P1: {loss1:.4f} / P2: {loss2:.4f}"
                    )
                print("  ------------------------")

            # 模型保存
            if total_episodes % config.SAVE_INTERVAL == 0 and total_episodes > 0:
                torch.save(
                    model1.state_dict(), 
                    f'gobang_model_player1_{total_episodes}.pth'
                )
                history_pool.append(deepcopy(model1))
                print(
                    f"[历史池更新] 已保存第 {total_episodes} 局模型，"
                    f"当前池大小: {len(history_pool)}"
                )
                
                if history_pool:
                    model2.load_state_dict(
                        random.choice(history_pool).state_dict()
                    )
                    optimizer2 = optim.Adam(
                        model2.parameters(), lr=Config.LEARNING_RATE
                    )
                
                best_model.load_state_dict(model1.state_dict())
                send_model_params()
                print(f"[模型保存] 第 {total_episodes} 局模型已保存")

    finally:
        # 清理进程
        for p in processes:
            p.terminate()
        torch.save(model1.state_dict(), 'gobang_best_model.pth')
        print("\n[训练结束] 最终模型已保存")
        total = total_win1 + total_win2
        if total > 0:
            print(
                f"总胜率: P1: {total_win1/total*100:.1f}% "
                f"({total_win1}/{total_win2})"
            )
            print(
                f"先手胜率: P1先手: {first_player1_wins/total_first1*100:.1f}% | "
                f"P2先手: {first_player2_wins/total_first2*100:.1f}%"
            )
        # 打印总训练速度和设备贡献
        total_time = time.time() - start_time
        if total_time > 0:
            total_speed = total_episodes / total_time
            print(
                f"总训练速度: {total_speed:.2f} 局/秒 "
                f"(共 {total_episodes} 局，耗时 {total_time:.1f} 秒)"
            )
            if use_cuda:
                print(
                    f"设备贡献: CPU: {cpu_episodes} 局 "
                    f"({cpu_episodes/total_episodes*100:.1f}%) | "
                    f"GPU: {gpu_episodes} 局 "
                    f"({gpu_episodes/total_episodes*100:.1f}%)"
                )


if __name__ == "__main__":
    train()
