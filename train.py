import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from model import Gomoku, GomokuNetV2, get_valid_action, load_model_if_exists
from config import Config, update_config_from_cli

def setup_device():
    use_gpu = torch.cuda.is_available()
    print("USE_GPU:", use_gpu)
    return torch.device("cuda" if use_gpu else "cpu")


def setup_players_and_optimizers(device, board_size):
    # 初始化两个玩家的模型和优化器（独立参数）
    model1 = GomokuNetV2(board_size).to(device)
    model2 = GomokuNetV2(board_size).to(device)
    optimizer1 = optim.Adam(model1.parameters(), lr=Config.LEARNING_RATE)
    optimizer2 = optim.Adam(model2.parameters(), lr=Config.LEARNING_RATE)
    return model1, model2, optimizer1, optimizer2


def get_epsilon(step, start, end, decay):
    # 探索率指数衰减策略
    return end + (start - end) * math.exp(-1. * step / decay)


def select_action(env, model, state, epsilon):
    # 生成动作概率分布
    logits = model(state)  # 形状为 (1, board_size*board_size)
    
    # 将张量转换为numpy数组处理（避免直接索引张量导致的错误）
    logits_np = logits.cpu().detach().numpy().flatten()
    
    # 筛选有效动作（仅空位可落子）
    valid_actions = [
        (logits_np[i], i) 
        for i in range(env.board_size * env.board_size) 
        if env.board[i // env.board_size, i % env.board_size] == 0
    ]
    
    if not valid_actions:
        return logits, -1  # 无有效动作
    
    valid_actions.sort(reverse=True, key=lambda x: x[0])  # 按分数排序
    
    # epsilon-greedy策略选择动作
    if random.random() < epsilon:
        return logits, random.choice(valid_actions)[1]
    else:
        # 从top3中随机选择（增加一定探索性）
        top_k = min(3, len(valid_actions))
        return logits, random.choice(valid_actions[:top_k])[1]


def update_model(reward, logits, optimizer, action, env, criterion, device, step, current_model):
    # 每步更新模型
    target = torch.LongTensor([action]).to(device)
    
    # 奖励归一化（缓解奖励值差异过大导致的训练不稳定）
    reward_norm = torch.tanh(torch.FloatTensor([reward]).to(device) / 1000)
    
    # 计算损失（交叉熵损失 * 归一化奖励）
    loss = criterion(logits.view(1, -1), target) * reward_norm
    
    # 反向传播与参数更新
    optimizer.zero_grad()
    loss.backward()
    
    # 梯度裁剪（作用于当前模型参数，防止梯度爆炸）
    torch.nn.utils.clip_grad_norm_(current_model.parameters(), max_norm=1.0)
    
    optimizer.step()


def train():
    # 解析命令行参数（支持动态设置棋盘尺寸和胜利条件）
    parser = argparse.ArgumentParser()
    parser.add_argument("--board_size", type=int, help="Size of the game board")
    parser.add_argument("--win_condition", type=int, help="Number of consecutive stones to win")
    args = parser.parse_args()
    config = update_config_from_cli(args)

    device = setup_device()
    env = Gomoku(config.BOARD_SIZE, config.WIN_CONDITION)
    model1, model2, optimizer1, optimizer2 = setup_players_and_optimizers(device, config.BOARD_SIZE)
    
    # 加载已有模型（若存在）
    load_model_if_exists(model1, 'gobang_best_model.pth')
    
    # 保留历史最优模型作为陪练
    best_model = GomokuNetV2(config.BOARD_SIZE).to(device)
    best_model.load_state_dict(model1.state_dict())
    criterion = nn.CrossEntropyLoss()

    for episode in range(config.MAX_EPISODES):
        env.reset()
        done = False
        total_steps = 0
        
        # 动态获取当前探索率（随训练进程衰减）
        epsilon1 = get_epsilon(episode, config.EPSILON1_START, config.EPSILON1_END, config.EPSILON_DECAY)
        epsilon2 = get_epsilon(episode, config.EPSILON2_START, config.EPSILON2_END, config.EPSILON_DECAY)

        while not done:
            # 转换当前棋盘状态为模型输入格式
            state = torch.FloatTensor(env.board.flatten()).unsqueeze(0).to(device)
            
            # 根据当前玩家选择模型和参数
            if env.current_player == 1:
                logits, action = select_action(env, model1, state, epsilon1)
                optimizer = optimizer1
                current_model = model1  # 当前训练的模型
            else:
                # 50%概率使用当前model2，50%使用历史最优模型作为陪练
                if random.random() < 0.5:
                    logits, action = select_action(env, model2, state, epsilon2)
                else:
                    logits, action = select_action(env, best_model, state, epsilon2 * 0.5)  # 陪练模型探索率降低
                optimizer = optimizer2
                current_model = model2  # 当前训练的模型

            # 处理无效动作（棋盘已满）
            if action == -1:
                break
            
            # 执行动作并获取反馈
            current_player, done, reward, total_count = env.step(action)
            
            # 更新当前模型
            update_model(reward, logits, optimizer, action, env, criterion, device, total_steps, current_model)
            total_steps += 1

        # 定期保存模型并更新最优模型
        if (episode + 1) % config.SAVE_INTERVAL == 0:
            # 保存当前model1
            torch.save(model1.state_dict(), f'gobang_model_player1_{episode + 1}.pth')
            
            # 更新最优模型（此处可根据实际评估指标优化，如胜率提升）
            best_model.load_state_dict(model1.state_dict())
            
            # 重置model2为当前最优模型的副本（作为新陪练）
            model2.load_state_dict(best_model.state_dict())
            optimizer2 = optim.Adam(model2.parameters(), lr=Config.LEARNING_RATE)
            
            print(f"Episode {episode + 1} - Model saved. Steps: {total_steps}, Epsilon1: {epsilon1:.4f}, Epsilon2: {epsilon2:.4f}")

    # 训练结束后保存最终模型
    torch.save(model1.state_dict(), 'gobang_best_model.pth')
    print("Training completed. Final model saved.")


if __name__ == "__main__":
    train()
