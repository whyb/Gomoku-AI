import random
import torch
import torch.nn as nn
import torch.optim as optim
from model import Gomoku, GomokuNetV2, get_valid_action, load_model_if_exists


import random


def get_random_smaller_thousand_multiple(number):
    """
    生成比输入数字小的、但不等于 0 的以 1000 为倍数的整数列表，并根据线性分布随机选择一个。
    :param number: 输入的数字
    :return: 随机选择的倍数，如果不满足条件返回相应的信息
    """
    if number < 1000:
        return "input number must >= 1000."
    # 当 number 等于 1000 时，将 multiples 列表初始化为包含 1000
    if number == 1000:
        multiples = [1000]
    else:
        multiples = [i for i in range(1000, number, 1000)]
    # 计算每个元素的权重，这里使用线性分布，数字越大权重越大
    weights = [i for i in range(1, len(multiples) + 1)]
    # 随机选择一个元素，根据计算出的权重
    return random.choices(multiples, weights=weights)[0] if multiples else "has no number"


def setup_device():
    """
    检查是否使用 GPU 并设置设备
    :return: 设备对象
    """
    use_gpu = torch.cuda.is_available()
    print("USE_GPU:", use_gpu)
    return torch.device("cuda" if use_gpu else "cpu")


def setup_players_and_optimizers(device):
    """
    初始化玩家模型和优化器
    :param device: 计算设备
    :return: 玩家 1 的模型、玩家 2 的模型、玩家 1 的优化器、玩家 2 的优化器
    """
    model1 = GomokuNetV2().to(device)
    model2 = GomokuNetV2().to(device)
    optimizer1 = optim.Adam(model1.parameters())
    optimizer2 = optim.Adam(model2.parameters())
    return model1, model2, optimizer1, optimizer2


def load_model_weights(model, optimizer, model_path):
    """
    加载模型权重并设置优化器
    :param model: 要加载权重的模型
    :param optimizer: 对应的优化器
    :param model_path: 权重文件路径
    """
    load_model_if_exists(model, model_path)
    optimizer = optim.Adam(model.parameters())


def select_action(env, model, optimizer, state, epsilon):
    """
    为玩家选择动作
    :param env: 游戏环境
    :param model: 玩家的模型
    :param optimizer: 玩家的优化器
    :param state: 当前状态
    :param epsilon: epsilon-greedy 策略中的 epsilon 值
    :return: 选择的动作
    """
    logits = model(state)
    action = get_valid_action(logits.cpu().detach().numpy(), env.board, epsilon)
    return logits, optimizer, action


def update_model(reward, logits, optimizer, action, env, criterion, device):
    """
    根据奖励更新模型参数
    :param reward: 获得的奖励
    :param logits: 模型的输出
    :param optimizer: 优化器
    :param action: 采取的动作
    :param env: 游戏环境
    :param criterion: 损失函数
    :param device: 设备
    """
    if reward!= 0:  # 当奖励不为 0 时更新模型
        target = torch.LongTensor([action]).to(device)
        # 改进：根据分数调整损失函数
        loss = criterion(logits.view(1, -1), target) * torch.FloatTensor([reward]).to(device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def print_game_result(env, round, reward, current_player):
    """
    打印游戏结果和最终棋盘状态
    :param env: 游戏环境
    :param round: 回合数
    :param reward: 奖励
    """
    if abs(reward) == 10000:  # 五子连珠获胜或失败
        print(f"Round {round}, Player {current_player} win with 5 in a row!")
    # elif abs(reward) == 1000:  # 一般获胜或失败
    #     print(f"Round {round}, Player {current_player} win!")
    elif abs(reward) == 500:  # 四子连珠获胜或失败
        print(f"\tRound {round}, Player {current_player} has 4 in a row!")
    elif abs(reward) == 100:  # 三子连珠获胜或失败
        print(f"\tRound {round}, Player {current_player} has 3 in a row!")
    elif abs(reward) == 10:  # 二子连珠获胜或失败
        print(f"\tRound {round}, Player {current_player} has 2 in a row!")
    #env.print_board()


def train():
    """
    主训练函数
    """
    device = setup_device()
    env = Gomoku()
    model1, model2, optimizer1, optimizer2 = setup_players_and_optimizers(device)
    load_model_if_exists(model1, 'gobang_best_model.pth')
    load_model_if_exists(model2, 'gobang_best_model.pth')
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()
    epsilon = 0.1


    for round in range(100000):
        env.reset()
        done = False
        while not done:
            state = torch.FloatTensor(env.board.flatten()).unsqueeze(0).to(device)
            if env.current_player == 1:
                logits1, optimizer1, action = select_action(env, model1, optimizer1, state, epsilon)
            else:
                logits2, optimizer2, action = select_action(env, model2, optimizer2, state, 0.3)
            if action == -1:
                break
            current_player, done, score = env.step(action)
            if current_player == 1:
                update_model(score, logits1, optimizer1, action, env, criterion1, device)
            else:
                update_model(score, logits2, optimizer2, action, env, criterion2, device)

            print_game_result(env, round, score, current_player)


        if (round + 1) % 1000 == 0:
            torch.save(model1.state_dict(), f'gobang_model_player1_{round + 1}.pth')
            model2 = GomokuNetV2().to(device)
            optimizer2 = optim.Adam(model2.parameters())
            random_pth_number = get_random_smaller_thousand_multiple(round + 1)
            load_model_weights(model2, optimizer2, f'gobang_model_player1_{random_pth_number}.pth')


    torch.save(model1.state_dict(), 'gobang_best_model.pth')


if __name__ == "__main__":
    train()
