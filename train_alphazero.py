"""
AlphaZero 风格训练主循环

与 train_dy.py 的核心区别:
  1. 自博弈: 模型 vs 自己 (而非 vs Kali-Hac)
  2. MCTS: 每步 800 次模拟 (而非 ε-greedy)
  3. 损失函数: L = (z-v)² - π^T·log(p) (而非 CE×reward)
  4. 对称增强: 8× 数据 (而非仅通道交换)
  5. 对手池: 与历史模型对战 (而非固定对手)
  6. 温度退火: 开局探索 → 中后局利用 (而非全局 ε-greedy)
  7. AMP 混合精度训练
  8. Elo 评估体系

用法:
  python train_alphazero.py --board_size 10 --num_simulations 400
"""

import os
import sys
import time
import signal
import argparse
import random
import warnings
import threading
import numpy as np
from collections import deque
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

# 解决 Windows ROCm MIOpen bug: "Invalid elapsed time detected in EvaluateInvokers"
# 必须在 import torch 之前设置，使用宽松的搜索模式避免 GPU 计时异常
os.environ.setdefault('MIOPEN_FIND_ENFORCE', '3')       # SEARCH_DB_UPDATE，不完全搜索
os.environ.setdefault('MIOPEN_FIND_MODE', '1')           # 快速模式

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.amp import autocast, GradScaler

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# GradScaler.step(optimizer) 调用 optimizer.step() 在先，scheduler.step() 在后，
# 但 PyTorch 的 _step_count 检测无法穿透 GradScaler，产生误报警告。此处禁用它。
warnings.filterwarnings(
    'ignore',
    message='Detected call of `lr_scheduler.step()` before `optimizer.step()`'
)

from config import Config, update_config_from_cli
from model_alphazero import (GomokuNetAlphaZero, GomokuNetAlphaZeroSmall,
                              DistillationConfig, distillation_loss)
from model_dy import load_model_if_exists
from mcts import MCTS, BatchMCTS, get_forced_move
from self_play import SelfPlayWorker, SelfPlayManager
from symmetry import SymmetryAugmenter
from opponent_pool import OpponentPool
from elo import EloRating
from arena import Arena
from opening_book import OpeningBook
from teacher import TeacherAI, generate_distill_games

# ============================================================
# 常量 (基准测试最优值: AMD RX 7900 XTX / RTX 3060 Ti)
# ============================================================
REPLAY_BUFFER_SIZE = 200000    # Replay buffer 容量
SAVE_INTERVAL = 2000           # 每 N 步保存一次
EVAL_INTERVAL = 5000           # 每 N 步评估一次
GAMES_PER_EVAL = 1             # 评估对局数
GAMES_PER_ITERATION = 16       # 每次迭代生成的对局数
L2_COEFF = 1e-4                # L2 正则系数
GAMMA = 1.0                    # 折扣因子 (AlphaZero 用 1.0, 不折扣)


class ReplayBuffer:
    """经验回放缓冲区 (预分配 numpy 环形缓冲, 零拷贝采样)"""

    def __init__(self, capacity: int = REPLAY_BUFFER_SIZE, board_size: int = 10):
        self.capacity = capacity
        self.board_size = board_size
        self._states = np.empty((capacity, 2, board_size, board_size), dtype=np.float32)
        self._policies = np.empty((capacity, board_size * board_size), dtype=np.float32)
        self._values = np.empty(capacity, dtype=np.float32)
        self._head = 0   # 下一个写入位置
        self._size = 0   # 当前有效数据量

    def add(self, data: List[Tuple[np.ndarray, np.ndarray, float]]):
        """添加一批训练数据 (不使用对称增强, 增强在采样时进行)"""
        n = len(data)
        if n == 0:
            return
        # 分批写入环形缓冲 (处理 n > capacity 的极端情况)
        start = 0
        while start < n:
            chunk = min(n - start, self.capacity - self._head)
            end = start + chunk
            for i, (s, p, v) in enumerate(data[start:end]):
                idx = self._head + i
                self._states[idx] = s
                self._policies[idx] = p
                self._values[idx] = v
            self._head = (self._head + chunk) % self.capacity
            self._size = min(self._size + chunk, self.capacity)
            start = end

    def sample(self, batch_size: int, board_size: int,
               augment: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """采样一个 batch (含对称增强, 零拷贝转换为 torch 张量)"""
        indices = np.random.choice(self._size, batch_size, replace=False)
        states = self._states[indices]    # (B, 2, H, W) 零拷贝视图
        policies = self._policies[indices]  # (B, H*W)
        values = self._values[indices]      # (B,)

        # 对称增强
        if augment:
            policies_2d = policies.reshape(-1, board_size, board_size)
            states, policies_2d = SymmetryAugmenter.augment_batch(states, policies_2d)
            policies = policies_2d.reshape(-1, board_size * board_size)

        # 零拷贝: torch.from_numpy 共享内存, 避免先 CPU 再 .to(device) 的额外拷贝
        return (torch.from_numpy(states.copy()),
                torch.from_numpy(policies.copy()),
                torch.from_numpy(values.copy()))

    def __len__(self):
        return self._size

    def save(self, filepath: str):
        """保存 ReplayBuffer 到磁盘 (压缩 numpy 格式)"""
        np.savez_compressed(
            filepath,
            states=self._states[:self._size],
            policies=self._policies[:self._size],
            values=self._values[:self._size],
            head=self._head,
            size=self._size,
            capacity=self.capacity,
            board_size=self.board_size
        )

    def load(self, filepath: str):
        """从磁盘加载 ReplayBuffer"""
        data = np.load(filepath, allow_pickle=False)
        loaded_size = int(data['size'])
        loaded_capacity = int(data['capacity'])
        if loaded_capacity != self.capacity:
            self.capacity = loaded_capacity
            self._states = np.empty((loaded_capacity, 2, self.board_size, self.board_size), dtype=np.float32)
            self._policies = np.empty((loaded_capacity, self.board_size * self.board_size), dtype=np.float32)
            self._values = np.empty(loaded_capacity, dtype=np.float32)
        n = min(loaded_size, self.capacity)
        self._states[:n] = data['states'][:n]
        self._policies[:n] = data['policies'][:n]
        self._values[:n] = data['values'][:n]
        self._head = int(data['head']) % self.capacity
        self._size = n
        print(f"  加载 ReplayBuffer: {self._size:,} 样本 (capacity={self.capacity:,})")
        return self._size


def update_model_alphazero(model: nn.Module, optimizer: torch.optim.Optimizer,
                           scaler: GradScaler, batch: Tuple,
                           device: str, l2_coeff: float = L2_COEFF,
                           fp16: bool = False
                           ) -> Tuple[float, float, float]:
    """
    AlphaZero 风格损失函数

    L = (z - v)² - π^T · log(p) + c · ||θ||²

    与 train_dy.py 的区别:
    - 策略损失: CE(π_mcts, p_net) 而非 CE(action) × reward
    - 价值损失: MSE(v, z) 其中 z ∈ {-1, 0, 1} 而非 MC 回报
    - 有 L2 正则
    - 使用 AMP 混合精度
    """
    states, mcts_policies, values = batch
    states = states.to(device)
    mcts_policies = mcts_policies.to(device)
    values = values.to(device)

    optimizer.zero_grad()

    amp_dtype = torch.float16 if fp16 else None
    with autocast('cuda' if 'cuda' in device else 'cpu', dtype=amp_dtype):
        logits, v_pred = model(states)

        # 策略损失: KL 散度 (MCTS 策略是 teacher, 网络是 student)
        log_probs = F.log_softmax(logits, dim=1)
        policy_loss = -torch.sum(mcts_policies * log_probs) / len(states)

        # 价值损失
        value_loss = F.mse_loss(v_pred.squeeze(-1), values)

        # L2 正则
        l2_reg = sum(p.pow(2).sum() for p in model.parameters())

        loss = policy_loss + value_loss + l2_coeff * l2_reg

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()

    return loss.item(), policy_loss.item(), value_loss.item()


# ============================================================
# 知识蒸馏训练 (教师引导 → 快速达到大师水平)
# ============================================================

# 蒸馏专用常量
DISTILL_GAMES_PER_BATCH = 500    # 每批生成的游戏数 (增量, Ctrl+C 安全)
DISTILL_SAVE_STEPS = 2000        # 每 N 训练步保存一次
DISTILL_EVAL_STEPS = 500         # 每 N 训练步评估 Top-K
DISTILL_LOG_STEPS = 100          # 每 N 训练步打印日志


def _update_model_distill(model, optimizer, scaler, batch, device,
                          distill_cfg, fp16=False):
    """
    蒸馏模式下的单步训练 — 使用 KL 散度匹配教师策略

    与 update_model_alphazero 的区别:
      - 策略损失: KL(teacher_policy || student_policy) 而非 CE(π_mcts, p_net)
      - 价值损失: 相同 (MSE with game outcome)
      - 温度参数控制软标签平滑度
    """
    states, teacher_policies, values = batch
    states = states.to(device)
    teacher_policies = teacher_policies.to(device)
    values = values.to(device)

    optimizer.zero_grad()

    amp_dtype = torch.float16 if fp16 else None
    with autocast('cuda' if 'cuda' in device else 'cpu', dtype=amp_dtype):
        logits, v_pred = model(states)

        loss, p_loss, v_loss = distillation_loss(
            logits, v_pred, teacher_policies, values,
            temperature=distill_cfg.temperature,
            value_weight=distill_cfg.value_weight,
            l2_coeff=distill_cfg.l2_coeff,
            model_params=model.parameters()
        )

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()

    return loss.item(), p_loss.item(), v_loss.item()


def _compute_topk_accuracy(model, batch, device, k_values=(1, 3, 5)):
    """计算学生网络与教师策略的 Top-K 一致率"""
    states, teacher_policies, _ = batch
    states = states.to(device)
    teacher_policies = teacher_policies.to(device)

    with torch.no_grad():
        logits, _ = model(states)
        _, student_topk = torch.topk(logits, max(k_values), dim=1)

    _, teacher_top1 = torch.topk(teacher_policies, 1, dim=1)
    teacher_top1 = teacher_top1.expand(-1, max(k_values))

    results = {}
    for k in k_values:
        match = (student_topk[:, :k] == teacher_top1[:, :k]).any(dim=1)
        results[f'top{k}'] = match.float().mean().item()
    return results


def _save_distill_checkpoint(model, optimizer, scheduler, scaler,
                              replay_buffer, buffer_path,
                              update_step, games_generated,
                              board_size, win_condition, model_tag,
                              distill_cfg, checkpoint_path,
                              distill_model_path):
    """保存蒸馏完整状态 (训练状态 + 数据)"""
    # 训练 checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'update_step': update_step,
        'games_generated': games_generated,
        'board_size': board_size,
        'win_condition': win_condition,
        'model_tag': model_tag,
        'distill_temperature': distill_cfg.temperature,
        'distill_value_weight': distill_cfg.value_weight,
        'distill_lr': distill_cfg.lr,
        'distill_games': distill_cfg.distill_games,
        'save_time': time.time(),
    }, checkpoint_path)
    # 纯权重 (供 MCTS 微调加载)
    torch.save(model.state_dict(), distill_model_path)
    # Replay Buffer 数据
    replay_buffer.save(buffer_path)


def _load_distill_checkpoint(model, optimizer, scheduler, scaler,
                              replay_buffer, checkpoint_path, buffer_path,
                              device):
    """加载蒸馏 checkpoint + replay buffer, 返回 (update_step, games_generated)"""
    update_step = 0
    games_generated = 0
    ckpt = None

    if os.path.exists(checkpoint_path):
        print(f"发现蒸馏 checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        update_step = ckpt.get('update_step', 0)
        games_generated = ckpt.get('games_generated', 0)
        print(f"  从 step={update_step}, games={games_generated} 恢复")
    else:
        print("从零开始蒸馏训练")
        return update_step, games_generated

    if ckpt is not None:
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])

    # 加载 replay buffer
    if os.path.exists(buffer_path):
        replay_buffer.load(buffer_path)

    return update_step, games_generated


def train_distill(args):
    """
    知识蒸馏训练主循环 (增量生成 + 可续训 + Ctrl+C 安全)

    两阶段流程:
      Phase 1 (本函数): 教师引导 → 学生快速模仿大师
      Phase 2 (train 函数): 关闭 --distill → 自动加载蒸馏权重 → MCTS 微调

    关键设计:
      - 分批生成游戏 (每批 500 局), 生成一批 → 加入 buffer → 训练几步 → 循环
      - 每批生成后保存 checkpoint (含 replay buffer), Ctrl+C 随时安全退出
      - 支持 --save_interval_hours 定时自动保存
      - 续训时自动跳过已生成的游戏
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    board_size = args.board_size
    win_condition = args.win_condition

    # --- 蒸馏超参数 ---
    distill_cfg = DistillationConfig(
        temperature=args.distill_temperature,
        value_weight=args.distill_value_weight,
        lr=args.distill_lr,
        lr_min=args.distill_lr_min,
        warmup_steps=args.distill_warmup_steps,
        decay_steps=args.distill_decay_steps,
        batch_size=args.distill_batch_size,
        distill_games=args.distill_games,
        random_open_frac=args.distill_random_frac,
        l2_coeff=L2_COEFF,
    )

    # --- 模型 ---
    if args.model == 'small':
        model = GomokuNetAlphaZeroSmall().to(device)
        model_tag = 'small'
    else:
        model = GomokuNetAlphaZero().to(device)
        model_tag = 'standard'

    prefix = f'alpaz_{model_tag}_{board_size}x{board_size}'
    distill_model_path = f'{prefix}_distill.pth'
    checkpoint_path = f'{prefix}_distill_checkpoint.pth'
    buffer_path = f'{prefix}_distill_buffer.npz'

    # --- 优化器 + AMP + LR ---
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=distill_cfg.lr,
        weight_decay=distill_cfg.l2_coeff
    )
    scaler = GradScaler('cuda')

    def lr_lambda_distill(step):
        cycle_length = max(1, distill_cfg.decay_steps)
        step_in_cycle = step % cycle_length
        if step_in_cycle < distill_cfg.warmup_steps:
            return step_in_cycle / max(1, distill_cfg.warmup_steps)
        progress = min(1.0, (step_in_cycle - distill_cfg.warmup_steps) /
                       max(1, cycle_length - distill_cfg.warmup_steps))
        cos_val = 0.5 * (1 + np.cos(np.pi * progress))
        final_factor = distill_cfg.lr_min / distill_cfg.lr
        return final_factor + (1.0 - final_factor) * cos_val

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_distill)

    # --- Replay Buffer (环形缓冲, 固定容量, 与游戏数量无关) ---
    replay_buffer = ReplayBuffer(capacity=REPLAY_BUFFER_SIZE, board_size=board_size)

    # --- 恢复训练状态 ---
    update_step, games_generated = _load_distill_checkpoint(
        model, optimizer, scheduler, scaler, replay_buffer,
        checkpoint_path, buffer_path, device
    )

    # --- 打印配置 ---
    print("=" * 60)
    print("知识蒸馏训练 — 教师引导模式")
    print("=" * 60)
    print(f"棋盘大小:      {board_size}×{board_size}")
    print(f"连子数:        {win_condition}")
    print(f"模型:          {model_tag} ({sum(p.numel() for p in model.parameters()):,} 参数)")
    print(f"教师 AI:       Kali-Hac (Mode='比你6的Level')")
    print(f"蒸馏温度 T:    {distill_cfg.temperature}")
    print(f"价值损失权重:  {distill_cfg.value_weight}")
    print(f"学习率:        {distill_cfg.lr}")
    print(f"Batch size:    {distill_cfg.batch_size}")
    print(f"目标对局数:    {distill_cfg.distill_games:,}")
    print(f"已生成对局:    {games_generated:,}")
    print(f"随机开局比例:  {distill_cfg.random_open_frac:.0%}")
    print(f"保存间隔:      {args.save_interval_hours} 小时")
    print(f"对称增强:      开启 (8×)")
    print(f"设备:          {device}")
    if device == 'cuda':
        print(f"GPU:           {torch.cuda.get_device_name(0)}")
    if games_generated > 0:
        print(f">>> 续训模式: 从 game {games_generated} 继续生成")
        print(f">>> Buffer 已有: {len(replay_buffer):,} 样本")
    print("=" * 60)

    # --- 初始化教师 ---
    print("\n初始化教师 AI...")
    teacher = TeacherAI(board_size=board_size)
    print("教师就绪。")

    # ============================================================
    # 训练循环: 生成数据 → 训练 → 保存 (交替进行)
    # ============================================================
    batch_size = distill_cfg.batch_size
    total_games_target = distill_cfg.distill_games
    games_per_batch = min(DISTILL_GAMES_PER_BATCH, total_games_target)
    max_steps = distill_cfg.decay_steps
    save_interval_sec = args.save_interval_hours * 3600

    start_time = time.time()
    last_save_time = time.time()
    running = True

    def signal_handler(sig, frame):
        nonlocal running
        print("\n\n收到中断信号 (Ctrl+C)，正在安全保存...")
        running = False
    signal.signal(signal.SIGINT, signal_handler)

    # 滑动窗口统计 (格式对齐正常训练的日志)
    recent_losses = deque(maxlen=100)
    recent_p_losses = deque(maxlen=100)
    recent_v_losses = deque(maxlen=100)
    recent_gen_times = deque(maxlen=20)       # 每批生成耗时
    recent_train_times = deque(maxlen=20)     # 每批训练耗时
    recent_game_lengths = deque(maxlen=100)   # 每局平均步数
    recent_gen_moves = deque(maxlen=20)       # 每批生成的样本数
    total_samples_generated = 0               # 累计样本数

    # 早停 & 最佳模型追踪
    best_top1 = 0.0
    best_model_path = f'{prefix}_distill_best.pth'
    patience_counter = 0
    patience_limit = 20       # 连续 20 次评估无提升 → 停止
    min_improvement = 0.005   # Top1 提升不足 0.5% 不算有效提升
    loss_spike_threshold = 3.0  # loss 超过近期均值 3 倍视为爆炸

    print(f"\n开始蒸馏训练... (Ctrl+C 安全退出并保存)")
    print("-" * 100)

    try:
        while running:
            # ====================================================
            # Phase A: 增量生成教师数据
            # ====================================================
            if games_generated < total_games_target:
                remaining = total_games_target - games_generated
                batch_games = min(games_per_batch, remaining)

                t_gen_start = time.time()
                try:
                    game_data = generate_distill_games(
                        teacher, board_size=board_size, win_condition=win_condition,
                        num_games=batch_games,
                        policy_temperature=distill_cfg.temperature,
                        random_open_frac=distill_cfg.random_open_frac,
                        verbose=False  # 内部 verbose 关闭, 统一由外部汇总
                    )
                except KeyboardInterrupt:
                    print("\n生成中断, 保存已完成的游戏...")
                    running = False
                    break

                gen_time = time.time() - t_gen_start
                games_generated += batch_games
                num_samples_this_batch = len(game_data)
                total_samples_generated += num_samples_this_batch
                avg_game_len = num_samples_this_batch / max(1, batch_games)

                replay_buffer.add(game_data)
                del game_data

                recent_gen_times.append(gen_time)
                recent_game_lengths.append(avg_game_len)
                recent_gen_moves.append(num_samples_this_batch)

                # 数据生成全部完成: 立即保存
                if games_generated >= total_games_target:
                    print(f"\n[数据] 全部 {total_games_target:,} 局生成完成!"
                          f" 共 {total_samples_generated:,} 样本, "
                          f"耗时 {time.time() - start_time:.0f}s")
                    _save_distill_checkpoint(
                        model, optimizer, scheduler, scaler,
                        replay_buffer, buffer_path,
                        update_step, games_generated,
                        board_size, win_condition, model_tag,
                        distill_cfg, checkpoint_path, distill_model_path
                    )
                    last_save_time = time.time()

            # ====================================================
            # Phase B: 训练 (如果有足够数据)
            # ====================================================
            if len(replay_buffer) < batch_size:
                if games_generated >= total_games_target:
                    break
                continue

            t_train_start = time.time()
            trained_this_batch = 0
            train_steps_per_batch = 200

            while trained_this_batch < train_steps_per_batch and running:
                model.train()
                batch = replay_buffer.sample(batch_size, board_size, augment=True)

                loss, p_loss, v_loss = _update_model_distill(
                    model, optimizer, scaler, batch, device,
                    distill_cfg, fp16=args.fp16
                )
                scheduler.step()
                update_step += 1
                trained_this_batch += 1

                recent_losses.append(loss)
                recent_p_losses.append(p_loss)
                recent_v_losses.append(v_loss)

                # --- 步数触发保存 ---
                if update_step % DISTILL_SAVE_STEPS == 0:
                    t_save = time.time()
                    _save_distill_checkpoint(
                        model, optimizer, scheduler, scaler,
                        replay_buffer, buffer_path,
                        update_step, games_generated,
                        board_size, win_condition, model_tag,
                        distill_cfg, checkpoint_path, distill_model_path
                    )
                    last_save_time = time.time()
                    print(f"  [保存] Checkpoint @ step={update_step}, "
                          f"games={games_generated} "
                          f"({time.time() - t_save:.1f}s)")

                # --- 时间触发保存 ---
                if save_interval_sec > 0:
                    if time.time() - last_save_time >= save_interval_sec:
                        t_save = time.time()
                        _save_distill_checkpoint(
                            model, optimizer, scheduler, scaler,
                            replay_buffer, buffer_path,
                            update_step, games_generated,
                            board_size, win_condition, model_tag,
                            distill_cfg, checkpoint_path, distill_model_path
                        )
                        last_save_time = time.time()
                        print(f"  [自动保存] 定时触发 @ {time.time() - start_time:.0f}s "
                              f"(间隔 {save_interval_sec/3600:.1f}h) "
                              f"({time.time() - t_save:.1f}s)")

            train_time = time.time() - t_train_start
            recent_train_times.append(train_time)

            # ====================================================
            # 汇总日志 (每 500 局生成后打印, 格式对齐正常训练)
            # ====================================================
            # 每 500 局汇总打印
            if games_generated > 0 and games_generated % games_per_batch == 0:
                # --- 评估 Top-K ---
                model.eval()
                eval_batch = replay_buffer.sample(
                    min(batch_size, len(replay_buffer)),
                    board_size, augment=False
                )
                acc = _compute_topk_accuracy(model, eval_batch, device)

                # --- 计算滑动平均 ---
                avg_loss = (sum(recent_losses) / len(recent_losses)
                            if recent_losses else 0)
                avg_p = (sum(recent_p_losses) / len(recent_p_losses)
                         if recent_p_losses else 0)
                avg_v = (sum(recent_v_losses) / len(recent_v_losses)
                         if recent_v_losses else 0)
                avg_gen_time = (sum(recent_gen_times) / len(recent_gen_times)
                                if recent_gen_times else 0)
                avg_train_time = (sum(recent_train_times) / len(recent_train_times)
                                  if recent_train_times else 0)
                avg_game_len_recent = (sum(recent_game_lengths) / len(recent_game_lengths)
                                       if recent_game_lengths else 0)

                lr = optimizer.param_groups[0]['lr']
                elapsed = time.time() - start_time

                # --- 吞吐量 ---
                games_per_sec = games_generated / max(1, elapsed)
                samples_per_sec = total_samples_generated / max(1, elapsed)
                steps_per_sec = update_step / max(1, elapsed)

                # --- 每步教师评分耗时 (ms/move) ---
                ms_per_move = (avg_gen_time * 1000 / max(1,
                    sum(recent_gen_moves) / max(1, len(recent_gen_moves)))
                    if recent_gen_moves else 0)

                # --- GPU 内存 ---
                gpu_mem_used = 0
                gpu_mem_total = 1
                if device == 'cuda':
                    gpu_mem_used = torch.cuda.memory_allocated() / 1024**2
                    gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3

                # --- 打印 ---
                print(f"[Game {games_generated:>6}/{total_games_target:<6}] "
                      f"Loss={avg_loss:.4f}(P={avg_p:.4f} V={avg_v:.4f}) | "
                      f"LR={lr:.6f} | Top1={acc['top1']:.1%}")
                print(f"  Speed: "
                      f"{games_per_sec:.2f} games/s | "
                      f"{samples_per_sec:.0f} samples/s | "
                      f"{steps_per_sec:.2f} train_steps/s")
                print(f"  Timing: "
                      f"gen={avg_gen_time:.1f}s | "
                      f"train={avg_train_time:.1f}s | "
                      f"game_len={avg_game_len_recent:.1f} | "
                      f"{ms_per_move:.1f}ms/move")
                print(f"  System: "
                      f"Buffer={len(replay_buffer):,} | "
                      f"Step={update_step} | "
                      f"Samples={total_samples_generated:,} | "
                      f"GPU={gpu_mem_used:.0f}MB | "
                      f"Total={elapsed/60:.1f}min")
                print("-" * 100)

                # --- 提前收敛 (数据全部生成后) ---
                if games_generated >= total_games_target:
                    if acc['top1'] >= 0.85 and acc['top3'] >= 0.95:
                        print(f"  [收敛] Top-1={acc['top1']:.2%}, "
                              f"Top-3={acc['top3']:.2%} — 训练完成")
                        break

                    # 最佳模型追踪 + 早停 + 崩溃检测
                    if acc['top1'] > best_top1 + min_improvement:
                        if best_top1 > 0:
                            print(f"  [新高] Top1 {best_top1:.1%} → {acc['top1']:.1%}")
                        best_top1 = acc['top1']
                        patience_counter = 0
                        torch.save(model.state_dict(), best_model_path)
                    else:
                        patience_counter += 1

                    avg_loss_recent = (sum(recent_losses) / len(recent_losses)
                                       if recent_losses else 0)
                    # 崩溃检测: 绝对零 + 相对暴跌
                    top1_crashed = (acc['top1'] < 0.01 and best_top1 > 0.10)
                    top1_plummeted = (best_top1 > 0.20 and
                                      acc['top1'] < best_top1 * 0.3)
                    loss_exploded = (avg_loss_recent > loss_spike_threshold and
                                     best_top1 > 0.10)

                    if top1_crashed or top1_plummeted or loss_exploded:
                        reason = ("Top1→0%" if top1_crashed else
                                  f"Top1 {best_top1:.1%}→{acc['top1']:.1%}" if top1_plummeted else
                                  f"Loss={avg_loss_recent:.1f}")
                        print(f"  ⚠️ [崩溃检测] {reason}, "
                              f"恢复最佳 checkpoint (Top1={best_top1:.1%})")
                        if os.path.exists(best_model_path):
                            model.load_state_dict(torch.load(best_model_path,
                                                map_location=device, weights_only=True))
                        running = False
                        break

                    if patience_counter >= patience_limit:
                        print(f"  [早停] Top1 连续 {patience_limit} 次无提升 "
                              f"(最佳={best_top1:.1%}), 停止训练")
                        running = False
                        break

                # --- 最佳模型追踪 ---
                if acc['top1'] > best_top1 + min_improvement:
                    best_top1 = acc['top1']
                    patience_counter = 0
                    torch.save(model.state_dict(), best_model_path)
                else:
                    patience_counter += 1

                # --- 损失爆炸检测 ---
                avg_loss_recent = (sum(recent_losses) / len(recent_losses)
                                   if recent_losses else 0)
                if (avg_loss_recent > loss_spike_threshold and
                    best_top1 > 0.10 and acc['top1'] < 0.01):
                    print(f"  ⚠️ [崩溃检测] Loss={avg_loss_recent:.1f} 异常飙升, "
                          f"Top1={acc['top1']:.1%} → 0%, "
                          f"模型已退化! 恢复最佳 checkpoint (Top1={best_top1:.1%})")
                    if os.path.exists(best_model_path):
                        model.load_state_dict(torch.load(best_model_path,
                                            map_location=device, weights_only=True))
                    running = False
                    break

                # --- 早停 ---
                if patience_counter >= patience_limit:
                    print(f"  [早停] Top1 连续 {patience_limit} 次无提升 "
                          f"(最佳={best_top1:.1%}), 停止训练")
                    running = False
                    break

            # --- 数据全部生成, 继续纯训练 (定期打印状态) ---
            if games_generated >= total_games_target:
                if update_step >= max_steps:
                    print(f"  [完成] 已达最大步数 {max_steps}")
                    break

                # 纯训练阶段: 每 DISTILL_LOG_STEPS 步打印一次状态
                if update_step % (DISTILL_LOG_STEPS * 5) == 0:
                    model.eval()
                    eval_batch = replay_buffer.sample(
                        min(batch_size, len(replay_buffer)),
                        board_size, augment=False
                    )
                    acc = _compute_topk_accuracy(model, eval_batch, device)

                    avg_loss = (sum(recent_losses) / len(recent_losses)
                                if recent_losses else 0)
                    avg_p = (sum(recent_p_losses) / len(recent_p_losses)
                             if recent_p_losses else 0)
                    avg_v = (sum(recent_v_losses) / len(recent_v_losses)
                             if recent_v_losses else 0)
                    lr = optimizer.param_groups[0]['lr']
                    elapsed = time.time() - start_time
                    steps_per_sec = update_step / max(1, elapsed)

                    gpu_mem_used = 0
                    if device == 'cuda':
                        gpu_mem_used = torch.cuda.memory_allocated() / 1024**2

                    print(f"[Train {update_step:>7}] "
                          f"Loss={avg_loss:.4f}(P={avg_p:.4f} V={avg_v:.4f}) | "
                          f"LR={lr:.6f} | Top1={acc['top1']:.1%} | "
                          f"Buf={len(replay_buffer):,}")
                    print(f"  Speed: {steps_per_sec:.2f} steps/s | "
                          f"GPU={gpu_mem_used:.0f}MB | "
                          f"Total={elapsed/60:.1f}min")
                    print("-" * 100)

                    if acc['top1'] >= 0.85 and acc['top3'] >= 0.95:
                        print(f"  [收敛] Top-1={acc['top1']:.2%}, "
                              f"Top-3={acc['top3']:.2%} — 训练完成")
                        break

                    # 最佳模型追踪 + 早停 + 崩溃检测
                    if acc['top1'] > best_top1 + min_improvement:
                        if best_top1 > 0:
                            print(f"  [新高] Top1 {best_top1:.1%} → {acc['top1']:.1%}")
                        best_top1 = acc['top1']
                        patience_counter = 0
                        torch.save(model.state_dict(), best_model_path)
                    else:
                        patience_counter += 1

                    avg_loss_recent = (sum(recent_losses) / len(recent_losses)
                                       if recent_losses else 0)
                    # 崩溃检测: 绝对零 + 相对暴跌
                    top1_crashed = (acc['top1'] < 0.01 and best_top1 > 0.10)
                    top1_plummeted = (best_top1 > 0.20 and
                                      acc['top1'] < best_top1 * 0.3)
                    loss_exploded = (avg_loss_recent > loss_spike_threshold and
                                     best_top1 > 0.10)

                    if top1_crashed or top1_plummeted or loss_exploded:
                        reason = ("Top1→0%" if top1_crashed else
                                  f"Top1 {best_top1:.1%}→{acc['top1']:.1%}" if top1_plummeted else
                                  f"Loss={avg_loss_recent:.1f}")
                        print(f"  ⚠️ [崩溃检测] {reason}, "
                              f"恢复最佳 checkpoint (Top1={best_top1:.1%})")
                        if os.path.exists(best_model_path):
                            model.load_state_dict(torch.load(best_model_path,
                                                map_location=device, weights_only=True))
                        running = False
                        break

                    if patience_counter >= patience_limit:
                        print(f"  [早停] Top1 连续 {patience_limit} 次无提升 "
                              f"(最佳={best_top1:.1%}), 停止训练")
                        running = False
                        break

    except KeyboardInterrupt:
        # 最外层兜底
        print("\n\n收到中断信号，正在最终保存...")
    except Exception as e:
        print(f"\n\n训练异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # ====================================================
        # 最终保存 (无论如何都会执行)
        # ====================================================
        print("\n保存最终状态...")
        _save_distill_checkpoint(
            model, optimizer, scheduler, scaler,
            replay_buffer, buffer_path,
            update_step, games_generated,
            board_size, win_condition, model_tag,
            distill_cfg, checkpoint_path, distill_model_path
        )
        print(f"Checkpoint → {checkpoint_path}")
        print(f"权重       → {distill_model_path}")
        print(f"训练数据   → {buffer_path}")

    # --- 最终评估 ---
    model.eval()
    if len(replay_buffer) >= batch_size:
        final_batch = replay_buffer.sample(
            min(batch_size * 4, len(replay_buffer)),
            board_size, augment=False
        )
        final_acc = _compute_topk_accuracy(model, final_batch, device)
    else:
        final_acc = {'top1': 0.0, 'top3': 0.0, 'top5': 0.0}

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"蒸馏训练总结")
    print(f"{'=' * 60}")
    print(f"生成对局数:     {games_generated:,} / {total_games_target:,}")
    print(f"总样本数:       {len(replay_buffer):,}")
    print(f"总训练步数:     {update_step:,}")
    print(f"总耗时:         {elapsed/60:.1f} 分钟")
    print(f"最终 Teacher Top-1: {final_acc['top1']:.2%}")
    print(f"最终 Teacher Top-3: {final_acc['top3']:.2%}")
    print(f"最终 Teacher Top-5: {final_acc['top5']:.2%}")
    print(f"\n蒸馏模型已保存:")
    print(f"  最终权重: {distill_model_path}")
    if best_top1 > 0.01 and os.path.exists(best_model_path):
        print(f"  最佳权重: {best_model_path} (Top1={best_top1:.1%})")
    print(f"下一步: 关闭 --distill 标志, 加载权重进行 MCTS 微调:")
    print(f"  python train_alphazero.py --board_size {board_size} "
          f"--model {model_tag}")
    print(f"  (将自动加载 {distill_model_path} 作为初始权重)")
    if best_top1 > 0.01 and os.path.exists(best_model_path):
        print(f"  提示: 也可手动加载 {best_model_path} (蒸馏最佳)")
    print(f"{'=' * 60}")


def train(args):
    """主训练函数 (AlphaZero MCTS 自博弈)"""
    # ============================================================
    # 初始化
    # ============================================================
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    board_size = args.board_size
    win_condition = args.win_condition
    num_simulations = args.num_simulations
    mcts_batch_size = args.mcts_batch_size
    batch_size = args.batch_size

    # 根据 --model 参数选择模型
    if args.model == 'small':
        model = GomokuNetAlphaZeroSmall().to(device)
        model_tag = 'small'
    else:
        model = GomokuNetAlphaZero().to(device)
        model_tag = 'standard'

    # 所有持久化文件路径 (统一前缀, 方便管理)
    prefix = f'alpaz_{model_tag}_{board_size}x{board_size}'
    model_path = f'{prefix}_model.pth'
    best_model_path = f'{prefix}_best.pth'             # MCTS 最佳模型
    pool_path = f'{prefix}_opponent_pool.pth'
    elo_path = f'{prefix}_elo.json'
    checkpoint_path = f'{prefix}_checkpoint.pth'
    distill_model_path = f'{prefix}_distill.pth'         # 蒸馏最终权重
    distill_best_path = f'{prefix}_distill_best.pth'     # 蒸馏最佳权重 (优先)

    # ============================================================
    # 加载 checkpoint (完整恢复训练状态)
    # ============================================================
    update_step = 0
    total_games = 0
    total_samples = 0
    total_mcts_sims = 0
    resume_info = ""
    ckpt = None  # 在外层作用域保存，后续恢复优化器/scheduler 时复用

    if os.path.exists(checkpoint_path):
        print(f"发现 checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # 恢复模型权重
        model.load_state_dict(ckpt['model_state_dict'])
        # 恢复训练计数
        update_step = ckpt.get('update_step', 0)
        total_games = ckpt.get('total_games', 0)
        total_samples = ckpt.get('total_samples', 0)
        total_mcts_sims = ckpt.get('total_mcts_sims', 0)
        # 恢复 LR 调度器配置 (需在创建 scheduler 之前，确保闭包捕获正确的值)
        for key in ['decay_steps', 'warmup_steps', 'lr_min', 'learning_rate']:
            if key in ckpt:
                setattr(args, key, ckpt[key])
        resume_info = f"从 step={update_step}, games={total_games} 恢复, decay_steps={args.decay_steps}"
        print(f"  {resume_info}")
    elif os.path.exists(distill_best_path):
        # 优先加载蒸馏最佳权重 (distill_best.pth)
        print(f"发现蒸馏最佳权重: {distill_best_path}")
        state = torch.load(distill_best_path, map_location=device, weights_only=False)
        if 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)
        print(f"  已加载蒸馏最佳权重 (策略已接近大师水平, MCTS 将在此基础上微调)")
    elif os.path.exists(distill_model_path):
        # 加载蒸馏权重作为 MCTS 微调的起点
        print(f"发现蒸馏权重: {distill_model_path}")
        state = torch.load(distill_model_path, map_location=device, weights_only=False)
        if 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)
        print(f"  已加载蒸馏权重作为初始模型 "
              f"(策略已接近大师水平, MCTS 将在此基础上微调)")
    elif os.path.exists(model_path):
        load_model_if_exists(model, model_path)
        print(f"加载模型 (无 checkpoint): {model_path}")
    else:
        print("从零开始训练")

    # 优化器 + AMP
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=L2_COEFF
    )
    scaler = GradScaler('cuda')

    # LR 调度器: 余弦退火 + 自动热重启 (每个周期峰值减半, 预热仅首周期)
    def lr_lambda(step):
        cycle_length = max(1, args.decay_steps)
        cycle = step // cycle_length
        step_in_cycle = step % cycle_length

        # 预热仅在第一周期
        if step_in_cycle < args.warmup_steps and cycle == 0:
            return step_in_cycle / max(1, args.warmup_steps)

        progress = min(1.0, (step_in_cycle - args.warmup_steps) /
                       max(1, cycle_length - args.warmup_steps))
        cos_val = 0.5 * (1 + np.cos(np.pi * progress))

        # 每个周期峰值减半: lr_peak = initial_lr × 0.5^cycle
        restart_factor = 0.5 ** cycle
        final_factor = args.lr_min / args.learning_rate
        return final_factor + (restart_factor - final_factor) * cos_val

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 恢复优化器和调度器状态 (复用上方已加载的 ckpt)
    if ckpt is not None:
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            print("  恢复优化器状态")
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            print("  恢复学习率调度器状态")
        if 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
            print("  恢复 AMP scaler 状态")

    # 打印启动信息
    print("=" * 60)
    print("AlphaZero 风格 Gomoku 训练")
    print("=" * 60)
    print(f"棋盘大小:    {board_size}×{board_size}")
    print(f"连子数:      {win_condition}")
    print(f"模型:        {model_tag} ({sum(p.numel() for p in model.parameters()):,} 参数)")
    print(f"MCTS 模拟:   {num_simulations} 次/步")
    print(f"MCTS batch:  {mcts_batch_size}")
    print(f"训练 batch:  {batch_size}")
    print(f"Replay Buffer: {REPLAY_BUFFER_SIZE:,}")
    print(f"对称增强:    开启 (8×)")
    print(f"FP16:        {'开启' if args.fp16 else '关闭'}")
    print(f"设备:        {device}")
    if device == 'cuda':
        print(f"GPU:         {torch.cuda.get_device_name(0)}")
        print(f"显存:        {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    if resume_info:
        print(f">>> {resume_info}")
        print(f">>> LR={optimizer.param_groups[0]['lr']:.6f}")
    print("=" * 60)

    # Replay Buffer (内存中, 不持久化 — 太大了)
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE, board_size=board_size)

    # 对手池
    opponent_pool = OpponentPool(
        max_size=20,
        update_interval=500,
        selection_strategy='newer_biased'
    )
    if os.path.exists(pool_path):
        opponent_pool.load(pool_path)

    # Elo 系统
    elo = EloRating()
    if os.path.exists(elo_path):
        elo.load(elo_path)
    elo.add_model('current', elo=1500)
    elo.add_model('random', elo=800)

    # 自博弈管理器
    self_play_manager = SelfPlayManager(
        model, device, board_size, win_condition,
        num_simulations=num_simulations,
        augment_symmetry=True,
        opponent_pool=opponent_pool,
        mcts_batch_size=mcts_batch_size,
        cpu_workers=args.cpu_workers,
        fp16=args.fp16,
        use_human_knowledge=args.mcts_human_knowledge
    )
    self_play_manager.model_class = model_tag  # 告知管理器模型类型 (用于序列化)

    # Arena (评估用)
    arena = Arena(board_size, win_condition)

    # ============================================================
    # 定义 checkpoint 保存/加载函数
    # ============================================================
    def save_checkpoint(tag='auto'):
        """保存完整 checkpoint"""
        torch.save({
            # 模型
            'model_state_dict': model.state_dict(),
            # 优化器
            'optimizer_state_dict': optimizer.state_dict(),
            # 学习率调度器
            'scheduler_state_dict': scheduler.state_dict(),
            # AMP scaler
            'scaler_state_dict': scaler.state_dict(),
            # 训练计数
            'update_step': update_step,
            'total_games': total_games,
            'total_samples': total_samples,
            'total_mcts_sims': total_mcts_sims,
            # 训练配置 (用于验证一致性)
            'board_size': board_size,
            'win_condition': win_condition,
            'num_simulations': num_simulations,
            'batch_size': batch_size,
            'model_tag': model_tag,
            # LR 调度器配置 (用于续训时自动恢复)
            'decay_steps': args.decay_steps,
            'warmup_steps': args.warmup_steps,
            'lr_min': args.lr_min,
            'learning_rate': args.learning_rate,
            # 时间戳
            'save_time': time.time(),
        }, checkpoint_path)

        # 同时保存纯模型权重 (兼容 ONNX 导出等)
        torch.save(model.state_dict(), model_path)
        # 保存对手池和 Elo
        opponent_pool.save(pool_path)
        elo.save(elo_path)

    # ============================================================
    # 训练循环
    # ============================================================
    start_time = time.time()
    last_save_time = time.time()   # 时间触发自动保存
    last_eval_time = time.time()   # 时间触发自动评估
    save_interval_sec = args.save_interval_hours * 3600
    eval_ran = False               # 追踪是否至少运行过一次评估
    running = True

    # 滑动窗口统计 (最近 N 轮)
    recent_iter_times = deque(maxlen=20)
    recent_selfplay_times = deque(maxlen=20)
    recent_train_times = deque(maxlen=20)
    recent_game_lengths = deque(maxlen=100)
    recent_games_per_iter = deque(maxlen=20)
    recent_losses_mcts = deque(maxlen=200)   # MCTS 训练损失滑动窗口

    # 崩溃检测 & 最佳模型 (MCTS 用 Elo/胜率做指标)
    best_elo = elo.get_rating('current') if elo.get_rating('current') > 0 else 1500
    mcts_patience = 0
    mcts_patience_limit = 30        # 连续 30 次评估无提升 → 停止
    mcts_min_improvement = 10       # Elo 提升不足 10 分不算有效提升
    mcts_loss_spike = 5.0           # Policy loss 超过此值视为异常
    mcts_collapse_elo_drop = 200    # Elo 相对最佳跌超 200 分 → 崩溃

    def signal_handler(sig, frame):
        nonlocal running
        print("\n\n收到中断信号，正在保存...")
        running = False
    signal.signal(signal.SIGINT, signal_handler)

    # GPU 信息
    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"MCTS batch size: {self_play_manager.worker.mcts.batch_size if hasattr(self_play_manager.worker.mcts, 'batch_size') else 'N/A'}")
    print("\n开始训练... (Ctrl+C 停止)\n")
    print("-" * 100)

    # ============================================================
    # 后台训练线程 — 与自博弈流水线并行
    # ============================================================
    buffer_lock = threading.Lock()
    train_running = True
    train_stats = {'loss': 0.0, 'p_loss': 0.0, 'v_loss': 0.0, 'steps': 0}
    sp_running = threading.Event()   # GPU 自博弈模式下的互斥信号
    use_cpu_workers = args.cpu_workers > 0

    def training_worker():
        """后台训练线程: 持续从 buffer 采样并训练, 每次循环训练多步"""
        nonlocal update_step
        while train_running:
            # GPU 自博弈模式下暂停训练 (共享 GPU)
            if sp_running.is_set():
                time.sleep(0.05)
                continue

            # 连续训练多步 (不只在两次自博弈之间训一步)
            trained_this_round = 0
            while trained_this_round < 50 and train_running and not sp_running.is_set():
                with buffer_lock:
                    buf_len = len(replay_buffer)
                    if buf_len >= batch_size * 2:
                        batch = replay_buffer.sample(batch_size, board_size, augment=True)
                    else:
                        batch = None

                if batch is None:
                    break

                model.train()
                loss, p_loss, v_loss = update_model_alphazero(
                    model, optimizer, scaler, batch, device,
                    fp16=args.fp16
                )
                scheduler.step()
                with buffer_lock:
                    update_step += 1
                    train_stats['loss'] += loss
                    train_stats['p_loss'] += p_loss
                    train_stats['v_loss'] += v_loss
                    train_stats['steps'] += 1
                    recent_losses_mcts.append(p_loss)
                trained_this_round += 1

            if trained_this_round == 0:
                time.sleep(0.1)

    train_thread = threading.Thread(target=training_worker, daemon=True)
    train_thread.start()

    while running and total_games < args.max_games:
        iter_start = time.time()

        # ---- 阶段 1: 自博弈生成数据 ----
        # GPU 模式: 暂停训练 (共用 GPU); CPU 并行模式: 训练继续
        if not use_cpu_workers:
            sp_running.set()
        sp_start = time.time()
        model.eval()
        with torch.no_grad():
            game_data = self_play_manager.generate_games(GAMES_PER_ITERATION)
        sp_time = time.time() - sp_start
        if not use_cpu_workers:
            sp_running.clear()

        num_games_this_iter = GAMES_PER_ITERATION
        num_samples_this_iter = len(game_data)
        raw_steps = num_samples_this_iter
        avg_game_len = raw_steps / max(1, num_games_this_iter)

        total_games += num_games_this_iter
        total_samples += num_samples_this_iter
        total_mcts_sims += raw_steps * num_simulations

        recent_selfplay_times.append(sp_time)
        recent_game_lengths.append(avg_game_len)
        recent_games_per_iter.append(num_games_this_iter)

        with buffer_lock:
            replay_buffer.add(game_data)

        # ---- 阶段 2: 训练统计收集 ----
        # CPU 并行模式: 训练在自博弈期间已同步进行, 等待追上
        # GPU 模式: 训练刚恢复, 等它跑几步
        train_start = time.time()
        if use_cpu_workers:
            # 等待训练线程追上 (处理完 buffer 中积压的数据)
            wait_start = time.time()
            while time.time() - wait_start < 10.0:
                with buffer_lock:
                    if len(replay_buffer) < batch_size * 4:
                        break
                time.sleep(0.1)
        else:
            # GPU 模式: 让训练线程跑一小会
            time.sleep(0.5)

        with buffer_lock:
            ts = train_stats['steps']
            if ts > 0:
                avg_loss = train_stats['loss'] / ts
                avg_p_loss = train_stats['p_loss'] / ts
                avg_v_loss = train_stats['v_loss'] / ts
                num_train_steps = ts
            else:
                avg_loss = avg_p_loss = avg_v_loss = 0.0
                num_train_steps = 0
            train_stats['loss'] = 0.0
            train_stats['p_loss'] = 0.0
            train_stats['v_loss'] = 0.0
            train_stats['steps'] = 0

        train_time = time.time() - train_start
        recent_train_times.append(train_time)

        iter_time = time.time() - iter_start
        recent_iter_times.append(iter_time)

        # ---- 阶段 3: 定期保存 ----
        need_save = False
        save_reason = ''
        if update_step > 0 and update_step % SAVE_INTERVAL == 0:
            need_save = True
            save_reason = 'step'
        elif (save_interval_sec > 0 and
              time.time() - last_save_time >= save_interval_sec):
            need_save = True
            save_reason = 'time'
            last_save_time = time.time()

        if need_save:
            save_start = time.time()
            save_checkpoint('periodic')
            save_dur = time.time() - save_start
            print(f"  [保存] Checkpoint 已保存 (trigger={save_reason}, "
                  f"step={update_step}, games={total_games}, "
                  f"耗时 {save_dur:.1f}s)")

            # 添加到对手池 (仅在 step 触发时添加，避免过频)
            if save_reason == 'step':
                opponent_pool.add_model(
                    model, model_id=f'step_{update_step}',
                    step=update_step
                )

        # ---- 阶段 4: 定期评估 ----
        need_eval = False
        if update_step > 0 and update_step % EVAL_INTERVAL == 0:
            need_eval = True
        elif time.time() - last_eval_time >= 7200:  # 每 2 小时评估一次
            need_eval = True
            last_eval_time = time.time()

        if need_eval:
            eval_start = time.time()
            model.eval()

            def mcts_player(state):
                board = np.zeros((board_size, board_size), dtype=np.int32)
                board[state[0] == 1] = 1
                board[state[1] == 1] = 2
                # 推断当前玩家
                p1_count = (board == 1).sum()
                p2_count = (board == 2).sum()
                current_player = 1 if p1_count == p2_count else 2
                forced_action, _ = get_forced_move(board, current_player, win_condition)
                if forced_action is not None:
                    return [forced_action], np.array([1.0])
                mcts = MCTS(model, device, num_simulations=100, fp16=args.fp16,
                            win_condition=win_condition)
                return mcts.search(state, board, temperature=0.3, add_noise=False)

            def random_player(state):
                valid = []
                for i in range(board_size):
                    for j in range(board_size):
                        if state[0, i, j] == 0 and state[1, i, j] == 0:
                            valid.append(i * board_size + j)
                probs = np.ones(len(valid)) / len(valid)
                return valid, probs

            result = arena.play_match(mcts_player, random_player,
                                      num_games=GAMES_PER_EVAL)
            eval_time = time.time() - eval_start
            print(f"  [评估] vs 随机: 胜率={result['p1_win_rate']:.1%} "
                  f"(先手={result['p1_first_win_rate']:.1%}, "
                  f"后手={result['p1_second_win_rate']:.1%}) | "
                  f"评估耗时 {eval_time:.1f}s")

            # 更新 Elo
            for _ in range(10):
                if result['p1_win_rate'] > 0.5:
                    elo.update('current', 'random')
                else:
                    elo.update('random', 'current')
            eval_ran = True

            # --- MCTS 崩溃检测 & 最佳模型 ---
            current_elo = elo.get_rating('current')
            avg_p_loss_recent = (sum(recent_losses_mcts) / len(recent_losses_mcts)
                                 if recent_losses_mcts else 0)

            # 1) 最佳 Elo 追踪
            if current_elo > best_elo + mcts_min_improvement:
                if best_elo > 1500:
                    print(f"  [新高] Elo {best_elo:.0f} → {current_elo:.0f} "
                          f"(+{current_elo - best_elo:.0f})")
                best_elo = current_elo
                mcts_patience = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"  [保存] 最佳模型 → {best_model_path}")
            else:
                mcts_patience += 1

            # 2) Policy loss 异常飙升
            if avg_p_loss_recent > mcts_loss_spike:
                print(f"  ⚠️ [异常] Policy loss={avg_p_loss_recent:.2f} > "
                      f"{mcts_loss_spike}, 可能存在训练不稳定")

            # 3) Elo 崩塌检测
            if best_elo > 1600 and current_elo < best_elo - mcts_collapse_elo_drop:
                print(f"  ⚠️ [崩溃检测] Elo 从 {best_elo:.0f} 暴跌至 "
                      f"{current_elo:.0f} (跌 {best_elo - current_elo:.0f} > "
                      f"{mcts_collapse_elo_drop}), 恢复最佳 checkpoint!")
                if os.path.exists(best_model_path):
                    model.load_state_dict(torch.load(best_model_path,
                                        map_location=device, weights_only=True))
                    print(f"  已恢复最佳模型 (Elo={best_elo:.0f})")
                mcts_patience = 0  # 重置, 给恢复后的模型学习机会

            # 4) 早停
            if mcts_patience >= mcts_patience_limit:
                print(f"  [早停] Elo 连续 {mcts_patience_limit} 次评估无提升 "
                      f"(最佳={best_elo:.0f}), 停止训练")
                running = False

        # ---- 打印详细状态 ----
        elapsed = time.time() - start_time
        lr = optimizer.param_groups[0]['lr']
        buffer_size = len(replay_buffer)
        elo_current = elo.get_rating('current')
        pool_size = len(opponent_pool.pool)

        # 打印训练速度统计 (CPU 并行模式每轮打印, GPU 模式每 5 轮打印)
        stats_interval = GAMES_PER_ITERATION if use_cpu_workers else GAMES_PER_ITERATION * 5
        if total_games % stats_interval == 0 or True:
            # 计算滑动平均
            avg_iter_time = sum(recent_iter_times) / len(recent_iter_times) if recent_iter_times else 0
            avg_sp_time = sum(recent_selfplay_times) / len(recent_selfplay_times) if recent_selfplay_times else 0
            avg_train_time = sum(recent_train_times) / len(recent_train_times) if recent_train_times else 0
            avg_game_len_recent = sum(recent_game_lengths) / len(recent_game_lengths) if recent_game_lengths else 0

            # 吞吐量
            games_per_sec = total_games / max(1, elapsed)
            samples_per_sec = total_samples / max(1, elapsed)
            steps_per_sec = update_step / max(1, elapsed)
            sims_per_sec = total_mcts_sims / max(1, elapsed)

            # GPU 内存
            gpu_mem_used = 0
            gpu_mem_pct = 0
            if device == 'cuda':
                gpu_mem_used = torch.cuda.memory_allocated() / 1024**2  # MB
                gpu_mem_pct = gpu_mem_used / (gpu_mem * 1024) * 100

            # 每局平均 MCTS 搜索次数
            mcts_sims_per_game = total_mcts_sims / max(1, total_games)

            print(f"[Game {total_games:>6}] "
                  f"Loss={avg_loss:.4f}(P={avg_p_loss:.4f} V={avg_v_loss:.4f}) | "
                  f"LR={lr:.6f} | Elo={elo_current:.0f}")

            print(f"  Speed: "
                  f"{games_per_sec:.2f} games/s | "
                  f"{samples_per_sec:.0f} samples/s | "
                  f"{steps_per_sec:.2f} train_steps/s | "
                  f"{sims_per_sec:.0f} MCTS sims/s")

            print(f"  Timing: "
                  f"iter={avg_iter_time:.1f}s | "
                  f"selfplay={avg_sp_time:.1f}s | "
                  f"train={avg_train_time:.1f}s | "
                  f"game_len={avg_game_len_recent:.1f}")

            print(f"  System: "
                  f"Buffer={buffer_size:,} | "
                  f"Pool={pool_size} | "
                  f"Step={update_step} | "
                  f"Samples={total_samples:,} | "
                  f"MCTS_sims={total_mcts_sims:,.0f} | "
                  f"GPU={gpu_mem_used:.0f}MB({gpu_mem_pct:.0f}%) | "
                  f"Total={elapsed/60:.1f}min")
            print("-" * 100)

    # ============================================================
    # 清理
    # ============================================================
    train_running = False
    train_thread.join(timeout=5)
    print("\n训练结束，保存 checkpoint...")
    save_checkpoint('final')

    # 如果训练期间从未评估过，退出前强制运行一次 (Elo 表需要数据)
    if not eval_ran:
        print("\n训练期间未触发评估，退出前强制评估...")
        model.eval()

        def mcts_player_final(state):
            board = np.zeros((board_size, board_size), dtype=np.int32)
            board[state[0] == 1] = 1
            board[state[1] == 1] = 2
            # 推断当前玩家
            p1_count = (board == 1).sum()
            p2_count = (board == 2).sum()
            current_player = 1 if p1_count == p2_count else 2
            forced_action, _ = get_forced_move(board, current_player, win_condition)
            if forced_action is not None:
                return [forced_action], np.array([1.0])
            mcts = MCTS(model, device, num_simulations=100, fp16=args.fp16,
                        win_condition=win_condition)
            return mcts.search(state, board, temperature=0.3, add_noise=False)

        def random_player_final(state):
            valid = []
            for i in range(board_size):
                for j in range(board_size):
                    if state[0, i, j] == 0 and state[1, i, j] == 0:
                        valid.append(i * board_size + j)
            probs = np.ones(len(valid)) / len(valid)
            return valid, probs

        result = arena.play_match(mcts_player_final, random_player_final,
                                  num_games=GAMES_PER_EVAL)
        print(f"  [最终评估] vs 随机: 胜率={result['p1_win_rate']:.1%} "
              f"(先手={result['p1_first_win_rate']:.1%}, "
              f"后手={result['p1_second_win_rate']:.1%})")
        for _ in range(10):
            if result['p1_win_rate'] > 0.5:
                elo.update('current', 'random')
            else:
                elo.update('random', 'current')

    opponent_pool.print_pool_status()
    elo.print_leaderboard()

    elapsed = time.time() - start_time
    print(f"\n=== 训练统计 ===")
    print(f"最佳 Elo:     {best_elo:.0f}")
    if os.path.exists(best_model_path):
        print(f"最佳模型:     {best_model_path}")
    print(f"总对局数:    {total_games:,}")
    print(f"总样本数:    {total_samples:,}")
    print(f"总训练步数:  {update_step:,}")
    print(f"总 MCTS 模拟: {total_mcts_sims:,.0f}")
    print(f"总耗时:      {elapsed/60:.1f} 分钟")
    print(f"平均速度:    {total_games/elapsed:.2f} 局/秒, "
          f"{update_step/elapsed:.2f} 步/秒")
    if device == 'cuda':
        print(f"峰值 GPU 内存: {torch.cuda.max_memory_allocated()/1024**2:.0f} MB")


def main():
    parser = argparse.ArgumentParser(description='AlphaZero Gomoku 训练')
    parser.add_argument('--board_size', type=int, default=10,
                        help='棋盘大小 (默认 10)')
    parser.add_argument('--win_condition', type=int, default=5,
                        help='连子数 (默认 5)')
    parser.add_argument('--num_simulations', type=int, default=200,
                        help='MCTS 模拟次数 (默认 200)')
    parser.add_argument('--mcts_batch_size', type=int, default=256,
                        help='MCTS 批量推理大小 (默认 256)')
    parser.add_argument('--learning_rate', type=float, default=2e-3,
                        help='初始学习率 (默认 2e-3)')
    parser.add_argument('--lr_min', type=float, default=1e-5,
                        help='最小学习率 (默认 1e-5)')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help='LR 预热步数 (默认 1000)')
    parser.add_argument('--decay_steps', type=int, default=100000,
                        help='LR 衰减步数 (默认 100000)')
    parser.add_argument('--max_games', type=int, default=100000,
                        help='最大对局数 (默认 100000)')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='训练 batch 大小 (默认 1024, 基准测试最优)')
    parser.add_argument('--model', type=str, default='small',
                        choices=['small', 'standard'],
                        help='模型大小: small(6层/64ch,快) 或 standard(10层/128ch,强)')
    parser.add_argument('--fp16', action='store_true',
                        help='使用 FP16 混合精度训练 (推荐 AMD GPU)')
    parser.add_argument('--cpu_workers', type=int, default=0,
                        help='CPU 并行 worker 数 (0=串行GPU模式, >0=多进程CPU自博弈)')
    parser.add_argument('--mcts_human_knowledge', action='store_true',
                        help='在 MCTS 搜索树中启用人类知识增强 (连五/挡四/活三检测), 默认关闭')
    parser.add_argument('--save_interval_hours', type=float, default=1.0,
                        help='自动保存间隔 (小时, 默认 1.0, 设为 0 禁用)')

    # ============================================================
    # 知识蒸馏参数 (--distill 启用时生效)
    # ============================================================
    parser.add_argument('--distill', action='store_true',
                        help='启用知识蒸馏模式: 使用传统 AI 作为教师, '
                             '快速将学生网络提升到大师水平。蒸馏完成后, '
                             '关闭此标志加载蒸馏权重进行 MCTS 微调。')
    parser.add_argument('--distill_temperature', type=float, default=3.0,
                        help='蒸馏温度 T (默认 3.0)。T↑ → 教师策略更平滑, '
                             '学生学到更多次级候选的相对优劣; '
                             'T↓ → 策略更尖锐, 趋于 one-hot 模仿。'
                             '推荐: 2.0~4.0')
    parser.add_argument('--distill_value_weight', type=float, default=0.5,
                        help='价值损失权重 λ_value (默认 0.5)。'
                             '蒸馏阶段价值头信号较弱 (教师无估值), '
                             '降低此权重让训练聚焦于策略匹配。'
                             '推荐: 0.1~1.0')
    parser.add_argument('--distill_lr', type=float, default=3e-3,
                        help='蒸馏学习率 (默认 3e-3)。'
                             '监督蒸馏收敛远快于 RL, 可用更高 LR。'
                             '推荐: 1e-3~5e-3')
    parser.add_argument('--distill_lr_min', type=float, default=1e-5,
                        help='蒸馏最小学习率 (默认 1e-5)')
    parser.add_argument('--distill_warmup_steps', type=int, default=500,
                        help='蒸馏 LR 预热步数 (默认 500)')
    parser.add_argument('--distill_decay_steps', type=int, default=50000,
                        help='蒸馏 LR 衰减步数 (默认 50000)')
    parser.add_argument('--distill_batch_size', type=int, default=2048,
                        help='蒸馏训练 batch 大小 (默认 2048)。'
                             '蒸馏数据 IID, 大 batch 提供更稳定梯度。'
                             '推荐: 1024~4096')
    parser.add_argument('--distill_games', type=int, default=50000,
                        help='教师自我对弈局数 (默认 50000)。'
                             '10×10: 50000 局 ≈ 2M 样本, 足以收敛。'
                             '15×15: 20000~30000 局即可。'
                             '5×5:   10000 局足够。')
    parser.add_argument('--distill_random_frac', type=float, default=0.2,
                        help='随机开局比例 (默认 0.2)。'
                             '混入随机走子产生的局面, 强制学生在非均衡'
                             '状态下也模仿教师, 增强泛化性。')
    args = parser.parse_args()

    update_config_from_cli(args)

    if args.distill:
        train_distill(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
