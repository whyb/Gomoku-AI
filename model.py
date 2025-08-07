import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np
import random
from config import Config

# 游戏环境
class Gomoku:
    def __init__(self, board_size=None, win_condition=None):
        self.board_size = board_size or Config.BOARD_SIZE
        self.win_condition = win_condition or Config.WIN_CONDITION
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1
        self.winning_line = []
        self.step_count = 0

    def reset(self):
        self.board.fill(0)
        self.current_player = 1
        self.winning_line = []
        self.step_count = 0
        return self.board

    def is_winning_move(self, x, y):
        # 检查胜利条件
        def count_consecutive(player, dx, dy):
            count = 0
            line = []
            # 正向计数
            for step in range(1, self.win_condition):
                nx, ny = x + dx * step, y + dy * step
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == player:
                    count += 1
                    line.append((nx, ny))
                else:
                    break
            # 反向计数
            for step in range(1, self.win_condition):
                nx, ny = x - dx * step, y - dy * step
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == player:
                    count += 1
                    line.append((nx, ny))
                else:
                    break
            return count, line

        player = self.board[x, y]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dx, dy in directions:
            count, line = count_consecutive(player, dx, dy)
            if count + 1 >= self.win_condition:
                self.winning_line = [(x, y)] + line
                return True
        return False

    def _detect_double_live3(self, player, x, y):
        """检测是否形成两个独立的活3"""
        # 1. 对角线方向的细分检测（8个方向组合）
        # 基础4个方向加上其垂直方向，确保复杂交叉组合能被检测
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        live3_info = []  # 存储活3的方向和涉及的棋子位置
        opponent = 3 - player

        for dx, dy in directions:
            # 计算当前方向的连珠及两端状态
            count = 1
            stones = [(x, y)]  # 记录当前连珠涉及的棋子
            forward_empty = False
            backward_empty = False

            # 正向检查
            nx, ny = x + dx, y + dy
            while 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == player:
                count += 1
                stones.append((nx, ny))
                nx += dx
                ny += dy
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == 0:
                forward_empty = True

            # 反向检查
            nx, ny = x - dx, y - dy
            while 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == player:
                count += 1
                stones.append((nx, ny))
                nx -= dx
                ny -= dy
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == 0:
                backward_empty = True

            # 活3判定：保持原逻辑（连珠数=3，且两端均为空）
            if count == 3 and forward_empty and backward_empty:
                live3_info.append({"dir": (dx, dy), "stones": set(stones)})

        # 2. 允许共享1个棋子（但不能共享2个及以上）
        if len(live3_info) >= 2:
            for i in range(len(live3_info)):
                for j in range(i + 1, len(live3_info)):
                    # 计算两个活3的共享棋子数量
                    shared_stones = live3_info[i]["stones"].intersection(live3_info[j]["stones"])
                    # 允许共享1个棋子（双活3常共享中心子），但不能共享更多
                    if len(shared_stones) <= 1:
                        # 同时确保方向不同（避免同一方向的重复检测）
                        if live3_info[i]["dir"] != live3_info[j]["dir"]:
                            print(f"检测到双活3: ({x},{y})，玩家{player}")
                            return True
        return False
    
    def _detect_rush4_live3(self, player, x, y):
        """检测是否形成独立的冲4和活3组合（修复索引错误）"""
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        rush4_list = []
        live3_list = []
        opponent = 3 - player

        for dx, dy in directions:
            line = []
            positions = []
            
            # 构建包含当前落子的完整线条（核心修复）
            # 先向左/上追溯（包含当前位置）
            nx, ny = x, y
            while 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                line.append(self.board[nx, ny])
                positions.append((nx, ny))
                nx -= dx
                ny -= dy
            # 反转后当前落子在末尾，需要重新调整
            line.reverse()
            positions.reverse()
            
            # 再向右/下追溯（从当前位置的下一个开始）
            nx, ny = x + dx, y + dy
            while 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                line.append(self.board[nx, ny])
                positions.append((nx, ny))
                nx += dx
                ny += dy
            
            # 关键修复：确保当前玩家的棋子在line中
            # 从整个列表中查找，而不是限定范围
            try:
                current_idx = line.index(player)
            except ValueError:
                # 如果找不到当前玩家棋子，说明构建线条有误，跳过此方向
                continue
            
            # 检测冲4（包括"3连珠+1空格+1连珠"）
            for i in range(len(line) - 4 + 1):
                window = line[i:i+4]
                player_count = window.count(player)
                empty_count = window.count(0)
                opponent_count = window.count(opponent)
                
                # 非连续冲4（3子+1空格）
                if player_count == 3 and empty_count == 1 and opponent_count == 0:
                    empty_pos = window.index(0)
                    if (empty_pos > 0 and window[empty_pos-1] == player) and \
                       (empty_pos < 3 and window[empty_pos+1] == player):
                        stones = set()
                        for j in range(4):
                            if window[j] == player:
                                stones.add(positions[i+j])
                        rush4_list.append({"dir": (dx, dy), "stones": stones})
                        break
                
                # 连续冲4（4子相连）
                if player_count == 4:
                    left_open = (i == 0) or (line[i-1] == 0) if i > 0 else True
                    right_open = (i+4 == len(line)) or (line[i+4] == 0) if (i+4) < len(line) else True
                    if left_open or right_open:
                        stones = set(positions[i:i+4])
                        rush4_list.append({"dir": (dx, dy), "stones": stones})
                        break
            
            # 检测活3（保持原有逻辑）
            count = 1
            stones = [(x, y)]
            forward_empty = False
            backward_empty = False
            
            # 正向检查
            nx, ny = x + dx, y + dy
            while 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == player:
                count += 1
                stones.append((nx, ny))
                nx += dx
                ny += dy
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == 0:
                forward_empty = True
            
            # 反向检查
            nx, ny = x - dx, y - dy
            while 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == player:
                count += 1
                stones.append((nx, ny))
                nx -= dx
                ny -= dy
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == 0:
                backward_empty = True
            
            if count == 3 and forward_empty and backward_empty:
                live3_list.append({"dir": (dx, dy), "stones": set(stones)})

        # 检查冲4和活3的组合
        for rush4 in rush4_list:
            for live3 in live3_list:
                if rush4["dir"] != live3["dir"]:
                    shared = rush4["stones"].intersection(live3["stones"])
                    if len(shared) <= 1:
                        print(f"检测到冲4活3: ({x},{y})，玩家{player}")
                        return True
        return False

    def _detect_double_rush4(self, player, x, y):
        """检测是否形成双冲四：同一玩家在不同方向有两处冲四，且需不同防守点"""
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        rush4_dirs = []  # 存储冲四的方向及防守点
        opponent = 3 - player

        for dx, dy in directions:
            # 计算当前方向的连珠数量
            count = 1
            forward_empty = False
            backward_empty = False
            forward_blocked = False
            backward_blocked = False

            # 正向检查
            nx, ny = x + dx, y + dy
            while 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == player:
                count += 1
                nx += dx
                ny += dy
            # 正向端点状态（空/阻挡）
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                if self.board[nx, ny] == 0:
                    forward_empty = True
                elif self.board[nx, ny] == opponent:
                    forward_blocked = True

            # 反向检查
            nx, ny = x - dx, y - dy
            while 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == player:
                count += 1
                nx -= dx
                ny -= dy
            # 反向端点状态（空/阻挡）
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                if self.board[nx, ny] == 0:
                    backward_empty = True
                elif self.board[nx, ny] == opponent:
                    backward_blocked = True

            # 冲四判定：连珠数=4，且只有一端被阻挡（另一端为空）
            if count == 4:
                if (forward_empty and backward_blocked) or (backward_empty and forward_blocked):
                    # 记录防守点（被阻挡的反方向为空位）
                    defense_pos = (nx, ny) if backward_empty else (x + dx * (count), y + dy * (count))
                    rush4_dirs.append((dx, dy, defense_pos))

        # 去重方向（避免相反方向重复计算，如(1,0)和(-1,0)视为同一方向）
        unique_rush4 = []
        seen_dirs = set()
        for dir_info in rush4_dirs:
            dx, dy, pos = dir_info
            # 用绝对值表示方向（如(1,0)和(-1,0)都记为(1,0)）
            key = (abs(dx), abs(dy))
            if key not in seen_dirs:
                seen_dirs.add(key)
                unique_rush4.append(dir_info)

        # 双冲四：至少2个不同方向的冲四，且防守点不同
        if len(unique_rush4) >= 2:
            defenses = [info[2] for info in unique_rush4]
            # 检查防守点是否不同（避免同一位置防守两个冲四）
            if len(set(defenses)) >= 2:
                print(f"检测到双冲4: ({x},{y})，玩家{player}")
                return True
        return False

    def calculate_reward(self, x, y):
        # 细化奖励计算
        player = self.board[x, y]
        opponent = 3 - player
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        total_reward = 0
        live_counts = {2: 0, 3: 0, 4: 0}
        rush_counts = {2: 0, 3: 0, 4: 0}

        # 棋形计数逻辑
        for dx, dy in directions:
            count = 1
            forward_blocked = False
            backward_blocked = False

            # 正向计数
            for step in range(1, self.win_condition):
                nx, ny = x + dx * step, y + dy * step
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == player:
                    count += 1
                else:
                    if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == opponent:
                        forward_blocked = True
                    break

            # 反向计数
            for step in range(1, self.win_condition):
                nx, ny = x - dx * step, y - dy * step
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == player:
                    count += 1
                else:
                    if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == opponent:
                        backward_blocked = True
                    break
            
            if count >= 2:
                if not forward_blocked and not backward_blocked:
                    live_counts[min(count, 4)] += 1
                elif forward_blocked != backward_blocked: # 只有一端被阻挡
                    rush_counts[min(count, 4)] += 1

        # 双活3奖励
        if self._detect_double_live3(player, x, y):
            total_reward += Config.REWARD["双活3"]
        
        # 冲4活3奖励
        if self._detect_rush4_live3(player, x, y):
            total_reward += Config.REWARD["冲4活3"]
        
        # 双冲四奖励（高于单冲四，低于冲4活3）
        if self._detect_double_rush4(player, x, y):
            total_reward += Config.REWARD["双冲4"]
        
        # 单独棋形奖励
        for num in range(2, 5):
            total_reward += live_counts[num] * Config.REWARD[f"live{num}"]
            total_reward += rush_counts[num] * Config.REWARD[f"冲{num}"]

        # 阻断对手胜利奖励
        temp_board = self.board.copy()
        temp_board[x, y] = opponent
        if self._is_winning_move(temp_board, x, y, opponent):
            total_reward += Config.REWARD["block_win"]

        return total_reward

    def _is_winning_move(self, board, x, y, player):
        # 内部辅助函数
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dx, dy in directions:
            count = 1
            # 正向
            for step in range(1, self.win_condition):
                nx, ny = x + dx * step, y + dy * step
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size and board[nx, ny] == player:
                    count += 1
                else:
                    break
            # 反向
            for step in range(1, self.win_condition):
                nx, ny = x - dx * step, y - dy * step
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size and board[nx, ny] == player:
                    count += 1
                else:
                    break
            if count >= self.win_condition:
                return True
        return False

    def step(self, action):
        x, y = action // self.board_size, action % self.board_size
        if self.board[x, y] != 0:
            return -1, True, 0  # 无效落子
        current_player = self.current_player
        self.board[x, y] = current_player
        self.step_count += 1
        
        if self.is_winning_move(x, y):
            # 胜利时返回：(玩家, 是否结束, (基础奖励, 落子步数))
            return current_player, True, (Config.REWARD["win"], self.step_count)
        else:
            # 计算连珠奖励
            reward = self.calculate_reward(x, y)
            # 切换玩家
            next_player = 3 - current_player
            self.current_player = next_player
            return next_player, False, reward
    
    def get_state_representation(self):
        # 多通道输入表示
        player1_board = (self.board == 1).astype(np.float32)
        player2_board = (self.board == 2).astype(np.float32)
        return np.stack([player1_board, player2_board], axis=0) # 形状: (2, board_size, board_size)

    def print_board(self):
        """打印当前棋盘状态，用 X 表示玩家1，O 表示玩家2，. 表示空位"""
        # 打印列索引
        print("   " + " ".join(f"{i:2}" for i in range(self.board_size)))
        print("  +" + "--" * (self.board_size * 2 - 1) + "+")
        
        # 打印每行内容
        for i in range(self.board_size):
            row = [f"{i:2}|"]  # 行索引
            for j in range(self.board_size):
                if self.board[i, j] == 1:
                    row.append(" X")
                elif self.board[i, j] == 2:
                    row.append(" O")
                else:
                    row.append(" .")
            row.append(" |")
            print("".join(row))
        
        # 打印底部边框
        print("  +" + "--" * (self.board_size * 2 - 1) + "+")


# 卷积神经网络（修复维度匹配问题）
class GomokuNetV2(nn.Module):
    def __init__(self, board_size):
        super(GomokuNetV2, self).__init__()
        self.board_size = board_size
        
        # 卷积层
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        
        # 动态计算池化后的尺寸和全连接层输入尺寸
        self.pooled_size = (board_size + 1) // 2  # 第一次池化后的尺寸
        self.pooled_size2 = (self.pooled_size + 1) // 2  # 第二次池化后的尺寸
        self.fc1_input_size = 128 * self.pooled_size2 * self.pooled_size2
        
        # 全连接层
        self.fc1 = nn.Linear(self.fc1_input_size, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, board_size * board_size)

    def forward(self, x):
        # 输入形状调整：(batch_size, board_size*board_size) -> (batch_size, 1, board_size, board_size)
        x = x.view(-1, 1, self.board_size, self.board_size)
        
        # 第一次卷积、批归一化、激活和池化
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        
        # 第二次卷积、批归一化、激活和池化
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        
        # 展平特征图
        x = x.view(-1, self.fc1_input_size)
        
        # 全连接层
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class ResidualConvBlock(nn.Module):
    """残差卷积块：增强局部特征提取能力"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 当输入输出通道不同时，用1x1卷积调整维度
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)  # 残差连接
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual  # 残差相加
        return F.relu(x)


class PositionalEncoding(nn.Module):
    """位置编码：为棋盘位置添加空间位置信息（Transformer必备）"""
    def __init__(self, d_model, board_size, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # 适配棋盘尺寸：生成 (board_size^2, d_model) 的位置编码
        self.pe = pe[:board_size*board_size, :, :].transpose(0, 1)  # 形状：(1, N, d_model)，N=棋盘格数

    def forward(self, x):
        # x形状：(batch_size, N, d_model)，N=board_size^2
        x = x + self.pe.to(x.device)  # 叠加位置编码
        return x


class GomokuNetV3(nn.Module):
    """融合残差网络和Transformer的五子棋AI网络，输出格式与V2保持一致"""
    def __init__(self, board_size, channels=64, num_res_blocks=2, num_heads=2, d_model=64):
        super().__init__()
        self.board_size = board_size
        self.n = board_size * board_size
        self.d_model = d_model

        # 1. 输入特征提取：通道数从1改为2（玩家1和玩家2）
        self.input_proj = nn.Conv2d(2, channels, kernel_size=3, padding=1)

        # 2. 残差卷积模块：提取局部特征
        self.res_blocks = nn.Sequential(
            *[ResidualConvBlock(channels, channels) for _ in range(num_res_blocks)]
        )

        # 3. 维度转换：为Transformer准备输入
        self.proj_to_transformer = nn.Linear(channels, d_model)
        self.pos_encoder = PositionalEncoding(d_model, board_size)

        # 4. Transformer编码器：建模全局依赖
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=1)

        # 5. 策略头：预测落子概率（与V3一致）
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 6. 价值头：预测当前局面的胜率
        self.value_head = nn.Sequential(
            nn.Linear(d_model * self.n, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh() # 输出-1到1之间的值，代表胜率
        )

    def forward(self, x):
        # 输入形状：(batch_size, 2, board_size, board_size)
        
        # 1. 输入特征提取
        x = self.input_proj(x)

        # 2. 残差卷积提取局部特征
        x = self.res_blocks(x)

        # 3. 转换为Transformer输入格式
        x = x.flatten(2).transpose(1, 2)
        x = self.proj_to_transformer(x)

        # 4. 叠加位置编码 + Transformer
        x = self.pos_encoder(x)
        transformer_output = self.transformer_encoder(x)
        
        # 5. 策略头输出
        policy_logits = self.policy_head(transformer_output).squeeze(-1)
        
        # 6. 价值头输出
        value_input = transformer_output.flatten(1)
        value = self.value_head(value_input)
        
        # 返回两个输出
        return policy_logits, value.squeeze(-1)

def get_valid_action(logits, board_flat, board_size, epsilon=0.1):
    # 优化后的探索策略：根据棋子数量动态调整相邻位置探索概率
    valid_mask = (board_flat == 0)
    valid_indices = torch.where(valid_mask)[0]
    if valid_indices.numel() == 0:
        return -1

    # 计算当前棋盘上的棋子数量
    total_cells = board_size * board_size
    empty_cells = valid_indices.numel()
    piece_count = total_cells - empty_cells  # 已落子数量
    
    # 计算相邻位置探索的概率：随棋子数量增加从90%线性降至0%
    # 当棋子数量为1时概率90%，棋子充满棋盘时概率0%
    if piece_count <= 1:
        adjacent_prob = 0.9  # 只有1个或0个棋子时，100%从相邻位置探索
    elif piece_count >= total_cells - 1:
        adjacent_prob = 0.0  # 棋盘快满时，0%概率
    else:
        # 线性衰减：(总格子数-1 - 当前棋子数)/(总格子数-2)
        adjacent_prob = (total_cells - 1 - piece_count) / (total_cells - 2)

    # logits 已经被 flatten()，它是一个一维张量
    logits = logits.cpu().flatten()
    valid_indices = valid_indices.cpu() 
    
    valid_logits = logits[valid_indices]
    
    # 贪心选择
    greedy_action = valid_indices[torch.argmax(valid_logits)]

    if random.random() < epsilon:
        # 探索模式
        # 收集所有相邻位置的有效落子
        adjacent_actions = []
        board_cpu = board_flat.cpu().reshape(board_size, board_size)
        
        for idx in valid_indices:
            x, y = idx // board_size, idx % board_size
            if is_adjacent_to_piece(board_cpu, x, y, board_size):
                adjacent_actions.append(idx)
        
        # 根据当前棋子数量决定探索相邻位置的概率
        if adjacent_actions and random.random() < adjacent_prob:
            return random.choice(adjacent_actions)
        else:
            return random.choice(valid_indices)
    else:
        # 利用模式
        return greedy_action

def is_adjacent_to_piece(board, x, y, board_size):
    """检查(x,y)位置是否与已有棋子相邻（8个方向）"""
    # 8个方向：上、下、左、右、四个对角线
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        # 检查邻接位置是否在棋盘内且有棋子
        if 0 <= nx < board_size and 0 <= ny < board_size and board[nx, ny] != 0:
            return True
    return False

def load_model_if_exists(model, file_path):
    if os.path.exists(file_path):
        model.load_state_dict(torch.load(file_path))
        print(f"Loaded model weights from {file_path}")
    else:
        print(f"No saved model weights found at {file_path}")