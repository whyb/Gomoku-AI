# config.py
class Config:
    # 棋盘配置
    BOARD_SIZE = 10  # 默认10x10，可通过命令行参数修改
    WIN_CONDITION = 5  # 胜利条件（连珠数）
    
    # 训练参数
    LEARNING_RATE = 9.9e-3 # 0.001
    # 动态学习率调度参数
    LR_MIN = 3e-5
    LR_WARMUP_STEPS = 10000
    LR_DECAY_STEPS = 300000
    # 探索率配置
    EPSILON1_START = 0.33 #0.9   # Player1初始探索率
    EPSILON1_END = 0.05 #0.05    # Player1最终探索率
    EPSILON2_START = 0.9   # Player2初始探索率
    EPSILON2_END = 0.1     # Player2最终探索率
    EPSILON_DECAY = 200000  # 探索率按总局数线性衰减
    SAVE_INTERVAL = 5000  # 模型保存间隔
    MAX_EPISODES = 10000000  # 最大训练回合数
    
    # 奖励配置 (新增复合棋形奖励)
    REWARD = {
        "win": 50,            # 胜利奖励（最高优先级）
        "lose": -25,          # 失败惩罚
        "double_live3": 8,    # 双活3奖励（接近胜利）
        "rush4_live3": 12,    # 冲4+活3是必胜棋形，奖励应较高
        "double_rush4": 12,   # 双冲4强进攻棋形
        "live4": 10,          # 活四（下一步可胜）
        "live3": 4,           # 活三（需要防守，否则对手可能形成活四）
        "rush4": 6,           # 冲四（对手必须防守，否则失败）
        "rush3": 2,           # 冲三
        "block_win": 8,       # 挡住对手必胜（防守奖励）
        "live2": 1,           # 活2
        "rush2": 0.5          # 冲2
    }

    # 速度奖励系数（激励AI让越快胜利越好）
    # 计算示例：15x15棋盘（225格），若10步胜利，速度奖励 = (225-10)×2 = 430，总奖励达630；
    # 若50步胜利，速度奖励 = (225-50)×2 = 350，总奖励则等于200+350
    SPEED_REWARD_COEFFICIENT = 0.2

    # 速度惩罚系数
    LOSE_STEP_PENALTY = 0.05  # 每多走一步，失败惩罚随步数增加

    # 陪练难度混合（从易到难）
    OPP_STRONG_PROB_START = 0.99 #0.3
    OPP_STRONG_PROB_END = 1.0 #0.9
    OPP_STRONG_DECAY = 200000
    OPP_RANDOM_PROB_START = 0.005 #0.4
    OPP_RANDOM_PROB_END = 0.00005 #0.05
    OPP_RANDOM_DECAY = 200000
    

# 通过命令行可以修改棋盘尺寸
def update_config_from_cli(args):
    if hasattr(args, 'board_size') and args.board_size:
        Config.BOARD_SIZE = args.board_size
    if hasattr(args, 'win_condition') and args.win_condition:
        Config.WIN_CONDITION = args.win_condition
    return Config
