"""
AlphaZero 风格模型架构 — ResNet + SE (Squeeze-Excitation)

与 GomokuNetDyn (model_dy.py) 的对比:

  特性              GomokuNetDyn         GomokuNetAlphaZero
  ─────────────────────────────────────────────────────────
  通道数            64                   128
  残差块            4 × ConvBlock        10 × SEResBlock
  SE 注意力         无                   有 (reduction=8)
  全局建模          无                   无 (纯卷积, 速度快)
  参数量            ~200K                ~1.5M
  推理速度          快                   中等
  棋力上限          中等                 高

SE (Squeeze-Excitation) 的作用:
  自适应地为每个通道分配权重, 让网络关注最有用的特征
  在 ImageNet 上 +1-2% top-1 accuracy, 在棋类中效果显著
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation 模块

    流程:
      1. Squeeze: 全局平均池化, (B,C,H,W) → (B,C)
      2. Excitation: 两层 FC + Sigmoid, (B,C) → (B,C,1,1)
      3. Scale: 原始特征 × 注意力权重
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEResBlock(nn.Module):
    """
    带 SE 注意力的残差块

    结构:
      Conv3×3 → BN → ReLU → Conv3×3 → BN → SE → + skip → ReLU
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels, reduction)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.relu(out + residual)
        return out


class GomokuNetAlphaZero(nn.Module):
    """
    AlphaZero 风格网络

    架构:
      Input (B, 2, H, W)
        ↓
      Stem: Conv3×3(2→128) → BN → ReLU
        ↓
      Body: 10 × SEResBlock(128)
        ↓
      ┌─────────────┬─────────────┐
      │ Policy Head  │ Value Head  │
      │ Conv1×1→32   │ Conv1×1→32  │
      │ Conv1×1→1    │ AvgPool→FC  │
      │ → (B, H×W)   │ → (B, 1)    │
      └─────────────┴─────────────┘

    输出:
      - policy_logits: (B, H×W) — 落子偏好 (未经 softmax)
      - value: (B,) — 局面价值 [-1, 1]

    支持任意棋盘大小 (全卷积, 无固定 FC)
    """

    def __init__(self, in_channels: int = 2, channels: int = 128,
                 num_blocks: int = 10, reduction: int = 8):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # Body: 残差块堆叠
        self.body = nn.Sequential(
            *[SEResBlock(channels, reduction) for _ in range(num_blocks)]
        )

        # Policy Head: 两层 1×1 卷积
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)
            # 输出 (B, 1, H, W), 后续 flatten → (B, H×W)
        )

        # Value Head: 卷积 + 全局池化 + FC
        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.value_fc = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 2, H, W) 棋盘状态
        Returns:
            policy_logits: (B, H×W)
            value: (B,)
        """
        x = self.stem(x)
        x = self.body(x)

        # Policy: (B, 1, H, W) → (B, H×W)
        policy = self.policy_head(x).flatten(1)

        # Value: (B, 32, H, W) → (B, 32) → (B, 1) → (B,)
        value = self.value_conv(x)
        value = F.adaptive_avg_pool2d(value, 1).flatten(1)
        value = self.value_fc(value).squeeze(-1)

        return policy, value


class GomokuNetAlphaZeroSmall(nn.Module):
    """
    小型 AlphaZero 网络 (用于快速实验)

    参数量: ~300K (vs 大型 ~1.5M)
    适合: 5×5, 10×10 棋盘的快速迭代
    """

    def __init__(self, in_channels: int = 2, channels: int = 64,
                 num_blocks: int = 6, reduction: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.body = nn.Sequential(
            *[SEResBlock(channels, reduction) for _ in range(num_blocks)]
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1)
        )
        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.value_fc = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        policy = self.policy_head(x).flatten(1)
        value = self.value_conv(x)
        value = F.adaptive_avg_pool2d(value, 1).flatten(1)
        value = self.value_fc(value).squeeze(-1)
        return policy, value


# ================================================================
# 知识蒸馏工具
# ================================================================

class DistillationConfig:
    """知识蒸馏超参数配置

    各参数的设计理由见 train_alphazero.py 中的 --distill_* 帮助文档。
    """

    def __init__(self,
                 temperature: float = 3.0,
                 value_weight: float = 0.5,
                 lr: float = 3e-3,
                 lr_min: float = 1e-5,
                 warmup_steps: int = 500,
                 decay_steps: int = 50000,
                 batch_size: int = 2048,
                 distill_games: int = 50000,
                 random_open_frac: float = 0.2,
                 l2_coeff: float = 1e-4):
        self.temperature = temperature
        self.value_weight = value_weight
        self.lr = lr
        self.lr_min = lr_min
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.batch_size = batch_size
        self.distill_games = distill_games
        self.random_open_frac = random_open_frac
        self.l2_coeff = l2_coeff


def distillation_loss(student_logits: torch.Tensor,
                      student_value: torch.Tensor,
                      teacher_policy: torch.Tensor,
                      game_outcome: torch.Tensor,
                      temperature: float = 3.0,
                      value_weight: float = 0.5,
                      l2_coeff: float = 1e-4,
                      model_params: list = None
                      ) -> tuple:
    """
    知识蒸馏损失函数 (Hinton 2015 风格)

    总损失:
      L = L_policy + λ_value × L_value + λ_l2 × ||θ||²

    其中:
      L_policy = KL( softmax(s_teacher / T) || softmax(s_student / T) ) × T²
                 → 让学生在温度 T 下模仿教师的软标签
      L_value  = MSE(v_student, z_game)   (z ∈ {-1, 0, +1})

    温度 T² 缩放: logits / T 后梯度缩小 1/T², 乘以 T² 恢复原始量级,
    使策略损失与价值损失的尺度匹配。

    Args:
        student_logits:  (B, H×W) 学生网络策略 logits
        student_value:   (B,) 或 (B,1) 学生网络价值预测
        teacher_policy:  (B, H×W) 教师策略分布 (概率, 已归一化)
        game_outcome:    (B,) 终局结果 (+1/-1/0)
        temperature:     蒸馏温度 T
        value_weight:    价值损失权重 λ_value
        l2_coeff:        L2 正则系数 λ_l2
        model_params:    模型参数列表 (用于 L2 正则, 可选)

    Returns:
        (total_loss, policy_loss, value_loss)
    """
    # --- 策略损失: KL 散度 (教师→学生) ---
    # 教师软目标: softmax(log_teacher / T) — teacher_policy 已经是概率,
    # 但为了一致性我们将其视为"logits 经 T 软化后的结果"
    # 实际上 teacher_policy 是教师原始评分经 log→温度→softmax 得到的概率,
    # 这里直接用 KL 散度度量学生输出与教师策略的差异。

    # 学生 log prob (带温度)
    student_log_probs = torch.nn.functional.log_softmax(
        student_logits / temperature, dim=1
    )

    # 教师 soft prob (带温度, 从原始概率再软化一次)
    # teacher_policy 已经是概率分布, 再软化: p_i^(1/T) / Σ_j p_j^(1/T)
    teacher_soft = torch.pow(teacher_policy + 1e-10, 1.0 / temperature)
    teacher_soft = teacher_soft / teacher_soft.sum(dim=1, keepdim=True)

    # KL 散度 (teacher_soft 作为 target, student 去匹配)
    policy_loss = torch.sum(
        teacher_soft * (torch.log(teacher_soft + 1e-10) - student_log_probs),
        dim=1
    ).mean()

    # 温度平方缩放 (恢复梯度量级)
    policy_loss = policy_loss * (temperature ** 2)

    # --- 价值损失: MSE ---
    if student_value.dim() > 1:
        student_value = student_value.squeeze(-1)
    value_loss = torch.nn.functional.mse_loss(student_value, game_outcome)

    # --- 总损失 ---
    total_loss = policy_loss + value_weight * value_loss

    # --- L2 正则 ---
    if l2_coeff > 0 and model_params is not None:
        l2_reg = sum(p.pow(2).sum() for p in model_params)
        total_loss = total_loss + l2_coeff * l2_reg

    return total_loss, policy_loss, value_loss


if __name__ == '__main__':
    # 测试模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for board_size in [5, 10, 15]:
        model = GomokuNetAlphaZero().to(device)
        x = torch.randn(4, 2, board_size, board_size).to(device)
        policy, value = model(x)
        params = sum(p.numel() for p in model.parameters())
        print(f"Board {board_size}×{board_size}: policy={policy.shape}, "
              f"value={value.shape}, params={params:,}")

    # 测试小型模型
    model_small = GomokuNetAlphaZeroSmall().to(device)
    x = torch.randn(4, 2, 10, 10).to(device)
    policy, value = model_small(x)
    params = sum(p.numel() for p in model_small.parameters())
    print(f"\nSmall model: policy={policy.shape}, value={value.shape}, "
          f"params={params:,}")

    # 测试蒸馏损失
    print("\n--- 蒸馏损失测试 ---")
    B, H, W = 4, 10, 10
    test_logits = torch.randn(B, H * W)
    test_value = torch.randn(B)
    test_teacher = torch.softmax(torch.randn(B, H * W), dim=1)
    test_outcome = torch.zeros(B)
    test_outcome[0] = 1.0
    test_outcome[1] = -1.0

    total, p_loss, v_loss = distillation_loss(
        test_logits, test_value, test_teacher, test_outcome,
        temperature=3.0, value_weight=0.5
    )
    print(f"Total={total:.4f}, Policy={p_loss:.4f}, Value={v_loss:.4f}")
