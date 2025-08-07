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
    MAX_EPISODES = 90000000  # 最大训练回合数
    
    # 奖励配置 (新增复合棋形奖励)
    REWARD = {
        "win": 2000,          # 胜利奖励（最高优先级）
        "lose": -1000,        # 失败惩罚
        "双活3": 800,         # 双活3奖励（接近胜利）
        "冲4活3": 1000,       # 冲4+活3是必胜棋形，奖励应极高
        "双冲4": 1000,        # 双冲4也是强进攻棋形
        "live4": 800,         # 活四（下一步可胜）
        "live3": 200,         # 活三（需要防守，否则对手可能形成活四）
        "冲4": 300,           # 冲四（对手必须防守，否则失败）
        "冲3": 100,           # 冲三
        "block_win": 1500,     # 挡住对手必胜（降低防守奖励，低于进攻）
        "live2": 50,
        "冲2": 20
    }

    # 速度奖励系数（激励AI让越快胜利越好）
    # 计算示例：15x15棋盘（225格），若10步胜利，速度奖励 = (225-10)×20 = 4300，总奖励达6300；
    # 若50步胜利，速度奖励 = (225-50)×20 = 3500，总奖励则等于2000+3500
    SPEED_REWARD_COEFFICIENT = 20

    # 速度惩罚系数
    LOSE_STEP_PENALTY = 10  # 每多走一步，失败惩罚增加10
    

# 通过命令行可以修改棋盘尺寸
def update_config_from_cli(args):
    if hasattr(args, 'board_size') and args.board_size:
        Config.BOARD_SIZE = args.board_size
    if hasattr(args, 'win_condition') and args.win_condition:
        Config.WIN_CONDITION = args.win_condition
    return Config