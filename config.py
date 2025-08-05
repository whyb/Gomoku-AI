# config.py
class Config:
    # 棋盘配置
    BOARD_SIZE = 10  # 默认10x10，可通过命令行参数修改
    WIN_CONDITION = 5  # 胜利条件（连珠数）
    
    # 训练参数
    LEARNING_RATE = 0.001
    # 探索率配置
    EPSILON1_START = 1   # Player1初始探索率
    EPSILON1_END = 0.01   # Player1最终探索率
    EPSILON2_START = 1  # Player2初始探索率
    EPSILON2_END = 0.1    # Player2最终探索率
    EPSILON_DECAY = 100000  # 探索率衰减步数
    SAVE_INTERVAL = 10000  # 模型保存间隔
    MAX_EPISODES = 4000000  # 最大训练回合数
    
    # 奖励配置 (新增复合棋形奖励)
    REWARD = {
        "win": 10000,
        "block_win": 5000,   # 阻断对手胜利
        "live4": 1000,       # 活四
        "冲4": 800,         # 冲四（差一子成五）
        "live3": 300,        # 活三
        "冲3": 200,         # 冲三
        "live2": 50,         # 活二
        "冲2": 30,          # 冲二
        "双活三": 1500,      # 复合棋形奖励：同时形成两个活三
        "冲四活三": 2000,    # 复合棋形奖励：同时形成冲四和活三
    }

# 通过命令行可以修改棋盘尺寸
def update_config_from_cli(args):
    if hasattr(args, 'board_size') and args.board_size:
        Config.BOARD_SIZE = args.board_size
    if hasattr(args, 'win_condition') and args.win_condition:
        Config.WIN_CONDITION = args.win_condition
    return Config