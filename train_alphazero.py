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
from model_alphazero import GomokuNetAlphaZero, GomokuNetAlphaZeroSmall
from model_dy import load_model_if_exists
from mcts import MCTS, BatchMCTS
from self_play import SelfPlayWorker, SelfPlayManager
from symmetry import SymmetryAugmenter
from opponent_pool import OpponentPool
from elo import EloRating
from arena import Arena
from opening_book import OpeningBook

# ============================================================
# 常量 (基准测试最优值: AMD RX 7900 XTX / RTX 3060 Ti)
# ============================================================
REPLAY_BUFFER_SIZE = 200000    # Replay buffer 容量
SAVE_INTERVAL = 2000           # 每 N 步保存一次
EVAL_INTERVAL = 5000           # 每 N 步评估一次
GAMES_PER_EVAL = 40            # 评估对局数
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


def train(args):
    """主训练函数"""
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
    pool_path = f'{prefix}_opponent_pool.pth'
    elo_path = f'{prefix}_elo.json'
    checkpoint_path = f'{prefix}_checkpoint.pth'

    # ============================================================
    # 加载 checkpoint (完整恢复训练状态)
    # ============================================================
    update_step = 0
    total_games = 0
    total_samples = 0
    total_mcts_sims = 0
    resume_info = ""

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
        resume_info = f"从 step={update_step}, games={total_games} 恢复"
        print(f"  {resume_info}")
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

    # LR 调度器: 预热 + 余弦退火
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.decay_steps - args.warmup_steps)
        return max(args.lr_min / args.learning_rate,
                   0.5 * (1 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 恢复优化器和调度器状态
    if os.path.exists(checkpoint_path):
        ckpt_resume = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'optimizer_state_dict' in ckpt_resume:
            optimizer.load_state_dict(ckpt_resume['optimizer_state_dict'])
            print("  恢复优化器状态")
        if 'scheduler_state_dict' in ckpt_resume:
            scheduler.load_state_dict(ckpt_resume['scheduler_state_dict'])
            print("  恢复学习率调度器状态")
        if 'scaler_state_dict' in ckpt_resume:
            scaler.load_state_dict(ckpt_resume['scaler_state_dict'])
            print("  恢复 AMP scaler 状态")
        del ckpt_resume

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
        fp16=args.fp16
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
    running = True

    # 滑动窗口统计 (最近 N 轮)
    recent_iter_times = deque(maxlen=20)
    recent_selfplay_times = deque(maxlen=20)
    recent_train_times = deque(maxlen=20)
    recent_game_lengths = deque(maxlen=100)
    recent_games_per_iter = deque(maxlen=20)

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
        if update_step > 0 and update_step % SAVE_INTERVAL == 0:
            save_start = time.time()
            save_checkpoint('periodic')
            save_time = time.time() - save_start
            print(f"  [保存] Checkpoint 已保存 (step={update_step}, "
                  f"games={total_games}, 耗时 {save_time:.1f}s)")

            # 添加到对手池
            opponent_pool.add_model(
                model, model_id=f'step_{update_step}',
                step=update_step
            )

        # ---- 阶段 4: 定期评估 ----
        if update_step > 0 and update_step % EVAL_INTERVAL == 0:
            eval_start = time.time()
            model.eval()

            def mcts_player(state):
                board = np.zeros((board_size, board_size), dtype=np.int32)
                board[state[0] == 1] = 1
                board[state[1] == 1] = 2
                mcts = MCTS(model, device, num_simulations=100, fp16=args.fp16)
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

        # ---- 打印详细状态 ----
        elapsed = time.time() - start_time
        lr = optimizer.param_groups[0]['lr']
        buffer_size = len(replay_buffer)
        elo_current = elo.get_rating('current')
        pool_size = len(opponent_pool.pool)

        # 打印训练速度统计 (CPU 并行模式每轮打印, GPU 模式每 5 轮打印)
        stats_interval = GAMES_PER_ITERATION if use_cpu_workers else GAMES_PER_ITERATION * 5
        if total_games % stats_interval == 0:
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
    opponent_pool.print_pool_status()
    elo.print_leaderboard()

    elapsed = time.time() - start_time
    print(f"\n=== 训练统计 ===")
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
                        help='MCTS 模拟次数 (默认 200, 基准测试最优)')
    parser.add_argument('--mcts_batch_size', type=int, default=256,
                        help='MCTS 批量推理大小 (默认 256, 基准测试最优)')
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
    args = parser.parse_args()

    update_config_from_cli(args)
    train(args)


if __name__ == '__main__':
    main()
