import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResidualBlock(nn.Module):
    """通用的残差卷积块，不限制输入尺寸"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out, inplace=True)

class DynamicTransformerBlock(nn.Module):
    """自适应尺寸的 Transformer 块"""
    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [Batch, Sequence, Channels]
        # 使用 PyTorch 2.0+ 自动触发 Flash Attention (如果硬件支持)
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x

class GomokuNetDynV2(nn.Module):
    """
    自适应任意棋盘尺寸的高级网络
    支持动态输入 (B, 2, H, W)
    """
    def __init__(self, channels=128, num_res_blocks=4, num_transformer_layers=2, nhead=4):
        super().__init__()
        self.channels = channels
        
        # 1. 初始特征提取
        self.stem = nn.Sequential(
            nn.Conv2d(2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 2. 局部特征增强 (残差卷积)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_res_blocks)]
        )
        
        # 3. 全局依赖建模 (Transformer)
        self.transformer_layers = nn.ModuleList([
            DynamicTransformerBlock(channels, nhead) for _ in range(num_transformer_layers)
        ])
        
        # 4. 策略头 (Policy Head) - 输出每个格子的胜率
        # 使用 1x1 卷积保持尺寸一致，最后 flatten
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, kernel_size=1)
        )
        
        # 5. 价值头 (Value Head) - 输出局面评分
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), # 无论棋盘多大，都池化为 1x1
            nn.Flatten(),
            nn.Linear(channels, channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 2, 1),
            nn.Tanh()
        )

    def get_2d_pos_encoding(self, h, w, c, device):
        """动态生成 2D 正余弦位置编码"""
        grid_y, grid_x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
        
        dim_t = torch.arange(c // 2, device=device).float()
        dim_t = 10000**(2 * (dim_t // 2) / (c // 2))

        # 计算 X 和 Y 的编码
        pos_x = grid_x.unsqueeze(-1) / dim_t
        pos_y = grid_y.unsqueeze(-1) / dim_t
        
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=-1).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=-1).flatten(2)
        
        # 拼接 X, Y 编码得到 (H, W, C)
        pos_2d = torch.cat((pos_y, pos_x), dim=-1)
        return pos_2d.flatten(0, 1).unsqueeze(0) # [1, H*W, C]

    def forward(self, x):
        # x: [B, 2, H, W]
        b, _, h, w = x.size()
        
        # 卷积阶段
        feat = self.stem(x)
        feat = self.res_blocks(feat)
        
        # Transformer 阶段
        # 展平为序列: [B, C, H, W] -> [B, H*W, C]
        feat_seq = feat.flatten(2).transpose(1, 2)
        
        # 添加动态位置编码
        pos_encoding = self.get_2d_pos_encoding(h, w, self.channels, x.device)
        feat_seq = feat_seq + pos_encoding
        
        for layer in self.transformer_layers:
            feat_seq = layer(feat_seq)
            
        # 还原回空间维度供 Policy Head 使用
        feat_spatial = feat_seq.transpose(1, 2).reshape(b, self.channels, h, w)
        
        # 策略输出
        policy_logits = self.policy_head(feat_spatial) # [B, 1, H, W]
        policy_logits = policy_logits.view(b, -1)     # [B, H*W]
        
        # 价值输出
        value = self.value_head(feat_spatial).squeeze(-1) # [B]
        
        return policy_logits, value

def load_model_if_exists(model, file_path):
    import os
    if os.path.exists(file_path):
        try:
            state = torch.load(file_path, map_location=torch.device('cpu'))
            model.load_state_dict(state)
            print(f"Loaded dynamic model weights from {file_path}")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
    return False