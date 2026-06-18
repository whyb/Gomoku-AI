"""
训练曲线可视化 — 绘制 loss, value, policy, Elo 等指标

用法:
  python visualize.py --log_file training_log.csv
  python visualize.py --elo_file elo_10x10.json
"""

import os
import json
import argparse
import numpy as np
from typing import Dict, List, Optional


def plot_training_curves(log_file: str, save_dir: str = 'plots'):
    """
    绘制训练曲线 (loss, policy_loss, value_loss, learning_rate)

    Args:
        log_file: CSV 格式的训练日志
        save_dir: 图表保存目录
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print("需要安装 matplotlib: pip install matplotlib")
        return

    os.makedirs(save_dir, exist_ok=True)

    # 读取日志
    data = {
        'step': [], 'loss': [], 'policy_loss': [], 'value_loss': [],
        'elo': [], 'win_rate': [], 'lr': []
    }

    with open(log_file, 'r') as f:
        header = f.readline().strip().split(',')
        for line in f:
            values = line.strip().split(',')
            for h, v in zip(header, values):
                if h in data:
                    try:
                        data[h].append(float(v))
                    except ValueError:
                        data[h].append(0)

    if not data['step']:
        print(f"日志为空: {log_file}")
        return

    # 绘制 Loss 曲线
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('训练曲线', fontsize=16)

    # 总 Loss
    axes[0, 0].plot(data['step'], data['loss'], label='Total Loss', alpha=0.7)
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('总损失')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Policy Loss
    axes[0, 1].plot(data['step'], data['policy_loss'], label='Policy Loss',
                    color='orange', alpha=0.7)
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('策略损失')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Value Loss
    axes[1, 0].plot(data['step'], data['value_loss'], label='Value Loss',
                    color='green', alpha=0.7)
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('价值损失')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Elo
    if data['elo']:
        axes[1, 1].plot(data['step'], data['elo'], label='Elo',
                        color='red', alpha=0.7)
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Elo')
        axes[1, 1].set_title('Elo 评分')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"训练曲线已保存: {save_path}")


def plot_elo_history(elo_file: str, save_dir: str = 'plots'):
    """
    绘制 Elo 历史曲线

    Args:
        elo_file: Elo JSON 文件路径
        save_dir: 图表保存目录
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print("需要安装 matplotlib: pip install matplotlib")
        return

    os.makedirs(save_dir, exist_ok=True)

    with open(elo_file, 'r') as f:
        data = json.load(f)

    history = data.get('history', [])
    if not history:
        print(f"Elo 历史为空: {elo_file}")
        return

    # 提取 Elo 变化
    timestamps = [h['timestamp'] for h in history]
    winner_elos = [h['winner_elo_after'] for h in history]
    loser_elos = [h['loser_elo_after'] for h in history]

    # 归一化时间
    t0 = timestamps[0]
    times = [(t - t0) / 3600 for t in timestamps]  # 小时

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(times, winner_elos, label='赢家 Elo', alpha=0.7, color='green')
    ax.plot(times, loser_elos, label='输家 Elo', alpha=0.7, color='red')
    ax.set_xlabel('时间 (小时)')
    ax.set_ylabel('Elo')
    ax.set_title('Elo 评分历史')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'elo_history.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Elo 历史已保存: {save_path}")


def plot_win_rate(log_file: str, save_dir: str = 'plots'):
    """绘制胜率曲线"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print("需要安装 matplotlib: pip install matplotlib")
        return

    os.makedirs(save_dir, exist_ok=True)

    data = {'games': [], 'win_rate': []}
    with open(log_file, 'r') as f:
        header = f.readline().strip().split(',')
        for line in f:
            values = line.strip().split(',')
            for h, v in zip(header, values):
                if h in data:
                    try:
                        data[h].append(float(v))
                    except ValueError:
                        data[h].append(0)

    if not data['games']:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(data['games'], data['win_rate'], label='vs 随机', alpha=0.7)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% 线')
    ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='90% 线')
    ax.set_xlabel('对局数')
    ax.set_ylabel('胜率')
    ax.set_title('模型胜率 (vs 随机玩家)')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'win_rate.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"胜率曲线已保存: {save_path}")


class TrainingLogger:
    """训练日志记录器 (CSV 格式)"""

    def __init__(self, log_file: str = 'training_log.csv'):
        self.log_file = log_file
        self.header_written = False

    def log(self, step: int, loss: float, policy_loss: float,
            value_loss: float, elo: float = 0, win_rate: float = 0,
            lr: float = 0):
        """记录一条训练日志"""
        if not self.header_written:
            with open(self.log_file, 'w') as f:
                f.write('step,loss,policy_loss,value_loss,elo,win_rate,lr\n')
            self.header_written = True

        with open(self.log_file, 'a') as f:
            f.write(f'{step},{loss:.6f},{policy_loss:.6f},{value_loss:.6f},'
                    f'{elo:.1f},{win_rate:.4f},{lr:.8f}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练曲线可视化')
    parser.add_argument('--log_file', type=str, default='training_log.csv',
                        help='训练日志文件路径')
    parser.add_argument('--elo_file', type=str, default=None,
                        help='Elo JSON 文件路径')
    parser.add_argument('--save_dir', type=str, default='plots',
                        help='图表保存目录')
    args = parser.parse_args()

    if args.log_file and os.path.exists(args.log_file):
        plot_training_curves(args.log_file, args.save_dir)
        plot_win_rate(args.log_file, args.save_dir)

    if args.elo_file and os.path.exists(args.elo_file):
        plot_elo_history(args.elo_file, args.save_dir)
