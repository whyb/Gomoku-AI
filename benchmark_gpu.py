"""
GPU 性能基准测试 — 寻找 AMD RX 7900 XTX 的最优训练参数

测试维度:
  1. NN 推理 batch size (16/32/64/128/256/512)
  2. MCTS 模拟次数 (100/200/400/800)
  3. 训练 batch size (128/256/512/1024/2048)
  4. AMP 混合精度 vs FP32
  5. 模型大小 (Small / Standard)
"""

import os
import sys
import time
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F


def get_gpu_info():
    """获取 GPU 信息"""
    if not torch.cuda.is_available():
        return {"available": False, "name": "N/A", "mem_gb": 0}
    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "mem_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "compute_capability": torch.cuda.get_device_capability(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def benchmark_nn_inference(model, device, board_size, batch_sizes, num_warmup=10, num_iters=50):
    """测试不同 batch size 的 NN 推理速度"""
    print("\n=== NN 推理基准测试 ===")
    print(f"{'Batch':>6}  {'Time(ms)':>10}  {'Throughput':>12}  {'Speedup':>8}")
    print("-" * 45)

    results = {}
    baseline_time = None

    for bs in batch_sizes:
        x = torch.randn(bs, 2, board_size, board_size, device=device)

        # warmup
        model.eval()
        with torch.no_grad():
            for _ in range(num_warmup):
                model(x)
        if device == 'cuda':
            torch.cuda.synchronize()

        # benchmark
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_iters):
                model(x)
        if device == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        avg_ms = (t1 - t0) / num_iters * 1000
        throughput = bs * num_iters / (t1 - t0)
        if baseline_time is None:
            baseline_time = avg_ms / bs  # per-sample time at bs=1

        speedup = (baseline_time * bs) / avg_ms if baseline_time > 0 else 0
        results[bs] = {"avg_ms": avg_ms, "throughput": throughput, "speedup": speedup}

        print(f"{bs:>6}  {avg_ms:>10.2f}  {throughput:>10.0f}/s  {speedup:>7.2f}x")

        # 内存溢出检测
        if device == 'cuda':
            mem_used = torch.cuda.memory_allocated() / 1024**2
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
            if mem_used / mem_total > 0.85:
                print(f"  [!] GPU 显存接近上限 ({mem_used:.0f}/{mem_total:.0f} MB), 跳过更大 batch")
                break

    return results


def benchmark_training_step(model, device, board_size, batch_sizes, use_amp=True):
    """测试不同 batch size 的训练步速度"""
    print(f"\n=== 训练步基准测试 (AMP={'ON' if use_amp else 'OFF'}) ===")
    print(f"{'Batch':>6}  {'Time(ms)':>10}  {'Samples/s':>12}  {'Speedup':>8}")
    print("-" * 45)

    results = {}
    baseline_time = None

    for bs in batch_sizes:
        states = torch.randn(bs, 2, board_size, board_size, device=device)
        policies = torch.randn(bs, board_size * board_size, device=device)
        policies = F.softmax(policies, dim=1)
        values = torch.randn(bs, device=device)

        optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

        # warmup
        model.train()
        for _ in range(5):
            optimizer.zero_grad()
            if use_amp and device == 'cuda':
                with torch.amp.autocast('cuda'):
                    logits, v = model(states)
                    loss = -torch.sum(policies * F.log_softmax(logits, dim=1)) / bs + F.mse_loss(v, values)
                scaler = torch.amp.GradScaler('cuda')
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, v = model(states)
                loss = -torch.sum(policies * F.log_softmax(logits, dim=1)) / bs + F.mse_loss(v, values)
                loss.backward()
                optimizer.step()
        if device == 'cuda':
            torch.cuda.synchronize()

        # benchmark
        if use_amp and device == 'cuda':
            scaler = torch.amp.GradScaler('cuda')

        t0 = time.perf_counter()
        num_iters = 20
        for _ in range(num_iters):
            optimizer.zero_grad()
            if use_amp and device == 'cuda':
                with torch.amp.autocast('cuda'):
                    logits, v = model(states)
                    loss = -torch.sum(policies * F.log_softmax(logits, dim=1)) / bs + F.mse_loss(v, values)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, v = model(states)
                loss = -torch.sum(policies * F.log_softmax(logits, dim=1)) / bs + F.mse_loss(v, values)
                loss.backward()
                optimizer.step()
        if device == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        avg_ms = (t1 - t0) / num_iters * 1000
        throughput = bs * num_iters / (t1 - t0)
        if baseline_time is None:
            baseline_time = avg_ms / bs

        speedup = (baseline_time * bs) / avg_ms if baseline_time > 0 else 0
        results[bs] = {"avg_ms": avg_ms, "throughput": throughput, "speedup": speedup}

        print(f"{bs:>6}  {avg_ms:>10.2f}  {throughput:>10.0f}/s  {speedup:>7.2f}x")

        if device == 'cuda':
            mem_used = torch.cuda.memory_allocated() / 1024**2
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
            if mem_used / mem_total > 0.85:
                print(f"  [!] GPU 显存接近上限 ({mem_used:.0f}/{mem_total:.0f} MB), 跳过更大 batch")
                break

    return results


def benchmark_mcts_search(model, device, board_size, sim_counts, batch_size=16):
    """测试 MCTS 搜索速度"""
    from mcts import BatchMCTS

    print(f"\n=== MCTS 搜索基准测试 (NN batch={batch_size}) ===")
    print(f"{'Sims':>6}  {'Time(s)':>10}  {'Sims/s':>12}  {'s/move':>10}")
    print("-" * 45)

    results = {}

    for num_sims in sim_counts:
        mcts = BatchMCTS(
            model, device,
            num_simulations=num_sims,
            batch_size=batch_size
        )

        board = np.zeros((board_size, board_size), dtype=np.int32)
        board[board_size // 2, board_size // 2] = 1  # 天元开局
        state = np.zeros((2, board_size, board_size), dtype=np.float32)
        state[0, board_size // 2, board_size // 2] = 1.0

        # warmup
        mcts.search(state, board, temperature=1.0, add_noise=True)

        # benchmark
        t0 = time.perf_counter()
        num_trials = 3
        for _ in range(num_trials):
            mcts.search(state, board, temperature=1.0, add_noise=True)
        if device == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        avg_time = (t1 - t0) / num_trials
        sims_per_sec = num_sims / avg_time
        results[num_sims] = {"time_s": avg_time, "sims_per_sec": sims_per_sec}

        print(f"{num_sims:>6}  {avg_time:>10.2f}  {sims_per_sec:>10.0f}/s  {avg_time:>10.3f}")

    return results


def find_optimal_config(board_size=15):
    """主测试流程: 寻找最优配置"""
    gpu_info = get_gpu_info()
    print("=" * 60)
    print("GPU 性能基准测试 — 寻找最优训练参数")
    print("=" * 60)
    print(f"GPU:         {gpu_info['name']}")
    print(f"显存:        {gpu_info['mem_gb']:.1f} GB")
    print(f"PyTorch:     {gpu_info['torch_version']}")
    print(f"CUDA/ROCm:   {gpu_info['cuda_version']}")
    print(f"棋盘大小:    {board_size}x{board_size}")
    print("=" * 60)

    device = 'cuda' if gpu_info['available'] else 'cpu'

    # ============================================================
    # 测试 1: 模型大小对比
    # ============================================================
    from model_alphazero import GomokuNetAlphaZero, GomokuNetAlphaZeroSmall

    print("\n" + "=" * 60)
    print("测试 1: 模型大小对比")
    print("=" * 60)

    models = {}
    for name, cls in [('Small(6层)', GomokuNetAlphaZeroSmall),
                       ('Standard(10层)', GomokuNetAlphaZero)]:
        m = cls().to(device)
        params = sum(p.numel() for p in m.parameters())
        print(f"\n{name}: {params:,} 参数")
        models[name] = m

    # ============================================================
    # 测试 2: NN 推理 batch size
    # ============================================================
    print("\n" + "=" * 60)
    print("测试 2: NN 推理 batch size (决定 MCTS 速度)")
    print("=" * 60)

    batch_sizes_infer = [8, 16, 32, 64, 128, 256]

    infer_results = {}
    for name, model in models.items():
        print(f"\n--- {name} ---")
        infer_results[name] = benchmark_nn_inference(
            model, device, board_size, batch_sizes_infer
        )

    # ============================================================
    # 测试 3: 训练 batch size
    # ============================================================
    print("\n" + "=" * 60)
    print("测试 3: 训练 batch size")
    print("=" * 60)

    batch_sizes_train = [128, 256, 512, 1024, 2048]

    train_results = {}
    for name, model in models.items():
        for amp in [True, False]:
            tag = f"{name}+AMP" if amp else name
            print(f"\n--- {tag} ---")
            train_results[tag] = benchmark_training_step(
                model, device, board_size, batch_sizes_train, use_amp=amp
            )

    # ============================================================
    # 测试 4: MCTS 搜索速度
    # ============================================================
    print("\n" + "=" * 60)
    print("测试 4: MCTS 搜索速度 (不同模拟次数)")
    print("=" * 60)

    sim_counts = [100, 200, 400, 800]

    # 找到最优 NN batch size (吞吐量最高的)
    best_infer_bs = {}
    for name in models:
        best_bs = max(infer_results[name].keys(),
                      key=lambda k: infer_results[name][k]['throughput'])
        best_infer_bs[name] = best_bs
        print(f"{name} 最优推理 batch size: {best_bs}")

    mcts_results = {}
    for name, model in models.items():
        bs = best_infer_bs[name]
        print(f"\n--- {name} (NN batch={bs}) ---")
        mcts_results[name] = benchmark_mcts_search(
            model, device, board_size, sim_counts, batch_size=bs
        )

    # ============================================================
    # 汇总: 推荐配置
    # ============================================================
    print("\n" + "=" * 60)
    print("汇总: 推荐配置")
    print("=" * 60)

    for name in models:
        infer = infer_results[name]
        best_infer_bs = max(infer.keys(), key=lambda k: infer[k]['throughput'])
        best_infer = infer[best_infer_bs]

        # 找最优训练 batch (吞吐量最高)
        train_key = f"{name}+AMP"
        if train_key in train_results:
            train = train_results[train_key]
            best_train_bs = max(train.keys(), key=lambda k: train[k]['throughput'])
            best_train = train[best_train_bs]
        else:
            best_train_bs = 256
            best_train = {"throughput": 0}

        mcts = mcts_results.get(name, {})

        print(f"\n--- {name} ---")
        print(f"  NN 推理:  batch={best_infer_bs:>4}, {best_infer['throughput']:.0f} samples/s")
        print(f"  训练:     batch={best_train_bs:>4}, {best_train['throughput']:.0f} samples/s")
        if 400 in mcts:
            print(f"  MCTS(400): {mcts[400]['sims_per_sec']:.0f} sims/s, {mcts[400]['time_s']:.2f}s/move")
        if 200 in mcts:
            print(f"  MCTS(200): {mcts[200]['sims_per_sec']:.0f} sims/s, {mcts[200]['time_s']:.2f}s/move")

        # 估算每局耗时
        avg_moves = board_size * board_size * 0.3  # 约 30% 棋盘填满
        if 400 in mcts:
            game_time_400 = mcts[400]['time_s'] * avg_moves
            print(f"  估算每局(400sim): {game_time_400:.1f}s ({3600/game_time_400:.0f} 局/小时)")
        if 200 in mcts:
            game_time_200 = mcts[200]['time_s'] * avg_moves
            print(f"  估算每局(200sim): {game_time_200:.1f}s ({3600/game_time_200:.0f} 局/小时)")

    # ============================================================
    # 推荐最终配置
    # ============================================================
    print("\n" + "=" * 60)
    print("推荐命令:")
    print("=" * 60)

    # 选择综合最优模型
    for name in models:
        mcts = mcts_results.get(name, {})
        if 200 in mcts and mcts[200]['sims_per_sec'] > 0:
            sims_per_sec = mcts[200]['sims_per_sec']
            # 推荐: 每步 MCTS 时间控制在 0.3-1.0 秒
            recommended_sims_200 = 200
            recommended_sims_400 = 400

            infer = infer_results[name]
            best_bs = max(infer.keys(), key=lambda k: infer[k]['throughput'])

            train_key = f"{name}+AMP"
            if train_key in train_results:
                train = train_results[train_key]
                best_train_bs = max(train.keys(), key=lambda k: train[k]['throughput'])
            else:
                best_train_bs = 512

            if 'Small' in name:
                model_flag = '--model small'
            else:
                model_flag = ''

            print(f"\n# {name}:")
            print(f"python train_alphazero.py --board_size {board_size} "
                  f"--num_simulations 200 {model_flag}")
            print(f"  (MCTS batch={best_bs}, train batch={best_train_bs}, AMP=ON)")

    print()


if __name__ == '__main__':
    board_size = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    find_optimal_config(board_size)
