"""
对手池 (Opponent Pool) — 管理历史模型版本用于自博弈

作用:
  1. 防止策略震荡: 和固定对手对战容易过拟合，和多个历史版本对战更稳健
  2. 提升鲁棒性: 模型需要打败所有历史版本，而不只是最新的
  3. 课程式训练: 可以从弱到强逐步挑战

参考: AlphaGo Zero 的对手池机制
"""

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any
from collections import deque


class OpponentEntry:
    """对手池中的一个模型条目"""

    def __init__(self, state_dict: Dict, model_id: str,
                 elo: float = 1500.0, step: int = 0,
                 model_class_name: str = ''):
        self.state_dict = {k: v.clone().cpu() for k, v in state_dict.items()}
        self.model_id = model_id
        self.elo = elo
        self.step = step
        self.model_class_name = model_class_name  # 记录模型类名
        self.created_at = time.time()
        self.games_as_opponent = 0


class OpponentPool:
    """
    对手池管理器

    策略:
    - 保存训练过程中的模型快照
    - 自博弈时随机选择历史模型作为对手
    - 较新的模型有更高的被选中概率
    - 可选: 按 Elo 权重选择
    """

    def __init__(self, max_size: int = 20,
                 update_interval: int = 500,
                 selection_strategy: str = 'newer_biased'):
        """
        Args:
            max_size: 池中最大模型数
            update_interval: 每多少局自博弈后将当前模型加入池
            selection_strategy: 选择策略
                - 'uniform': 均匀随机
                - 'newer_biased': 较新的模型概率更高
                - 'elo_biased': Elo 较高的模型概率更高
        """
        self.pool: List[OpponentEntry] = deque(maxlen=max_size)
        self.max_size = max_size
        self.update_interval = update_interval
        self.selection_strategy = selection_strategy
        self.games_since_update = 0

    def add_model(self, model: nn.Module, model_id: str,
                  elo: float = 1500.0, step: int = 0):
        """
        将当前模型快照加入池

        Args:
            model: PyTorch 模型
            model_id: 模型标识 (如 'step_10000')
            elo: 当前 Elo 评分
            step: 训练步数
        """
        entry = OpponentEntry(
            state_dict=model.state_dict(),
            model_id=model_id,
            elo=elo,
            step=step,
            model_class_name=type(model).__name__
        )
        self.pool.append(entry)
        print(f"[对手池] 添加模型 {model_id} (Elo={elo:.0f}, 池大小={len(self.pool)})")

    def sample_opponent(self, model_class=None, device: str = 'cpu',
                        *model_args, **model_kwargs
                        ) -> Optional[nn.Module]:
        """
        随机采样一个对手模型

        Args:
            model_class: 模型类 (如 GomokuNetAlphaZero), None 则用存储时的类
            device: 模型运行设备 ('cuda' 或 'cpu')
            *model_args, **model_kwargs: 模型构造参数
        Returns:
            加载了权重的模型，如果池为空返回 None
        """
        if not self.pool:
            return None

        entry = self._select_entry()

        # 根据存储时的模型类名创建正确的模型
        if model_class is not None:
            model = model_class(*model_args, **model_kwargs)
        else:
            model = self._create_model_by_name(entry.model_class_name)

        model.load_state_dict(entry.state_dict)
        model = model.to(device)
        model.eval()

        entry.games_as_opponent += 1
        return model

    @staticmethod
    def _create_model_by_name(class_name: str) -> nn.Module:
        """根据类名创建模型实例"""
        from model_alphazero import GomokuNetAlphaZero, GomokuNetAlphaZeroSmall
        from model_dy import GomokuNetDyn

        class_map = {
            'GomokuNetAlphaZero': GomokuNetAlphaZero,
            'GomokuNetAlphaZeroSmall': GomokuNetAlphaZeroSmall,
            'GomokuNetDyn': GomokuNetDyn,
        }
        if class_name in class_map:
            return class_map[class_name]()
        raise ValueError(f"未知的模型类名: {class_name}")

    def _select_entry(self) -> OpponentEntry:
        """根据策略选择一个对手条目"""
        if self.selection_strategy == 'uniform':
            return random.choice(self.pool)

        elif self.selection_strategy == 'newer_biased':
            # 较新的模型概率更高: weight = index + 1
            weights = np.array([i + 1 for i in range(len(self.pool))],
                             dtype=np.float64)
            weights /= weights.sum()
            idx = np.random.choice(len(self.pool), p=weights)
            return self.pool[idx]

        elif self.selection_strategy == 'elo_biased':
            # Elo 高的模型概率更高: softmax(elo / 200)
            elos = np.array([e.elo for e in self.pool], dtype=np.float64)
            elos = elos / 200.0  # 温度缩放
            elos -= elos.max()   # 数值稳定
            weights = np.exp(elos)
            weights /= weights.sum()
            idx = np.random.choice(len(self.pool), p=weights)
            return self.pool[idx]

        else:
            return random.choice(self.pool)

    def should_update(self) -> bool:
        """检查是否应该添加新的模型快照"""
        self.games_since_update += 1
        if self.games_since_update >= self.update_interval:
            self.games_since_update = 0
            return True
        return False

    def get_latest_model_id(self) -> Optional[str]:
        """获取池中最新模型的 ID"""
        if not self.pool:
            return None
        return self.pool[-1].model_id

    def get_pool_info(self) -> List[Dict[str, Any]]:
        """获取池中所有模型的信息"""
        return [
            {
                'model_id': e.model_id,
                'elo': round(e.elo, 1),
                'step': e.step,
                'games_as_opponent': e.games_as_opponent,
                'age_seconds': round(time.time() - e.created_at)
            }
            for e in self.pool
        ]

    def print_pool_status(self):
        """打印池状态"""
        print(f"\n[对手池状态] 大小: {len(self.pool)}/{self.max_size}")
        for i, e in enumerate(self.pool):
            print(f"  [{i}] {e.model_id} | Elo={e.elo:.0f} | "
                  f"对战次数={e.games_as_opponent}")

    def save(self, filepath: str):
        """
        保存对手池到文件 (含所有模型权重)

        Args:
            filepath: 保存路径 (建议用 .pth 后缀)
        """
        data = {
            'max_size': self.max_size,
            'update_interval': self.update_interval,
            'selection_strategy': self.selection_strategy,
            'games_since_update': self.games_since_update,
            'entries': []
        }
        for entry in self.pool:
            data['entries'].append({
                'state_dict': {k: v.cpu() for k, v in entry.state_dict.items()},
                'model_id': entry.model_id,
                'elo': entry.elo,
                'step': entry.step,
                'model_class_name': entry.model_class_name,
                'created_at': entry.created_at,
                'games_as_opponent': entry.games_as_opponent,
            })
        torch.save(data, filepath)
        print(f"[对手池] 已保存到 {filepath} ({len(self.pool)} 个模型)")

    def load(self, filepath: str):
        """
        从文件加载对手池

        Args:
            filepath: 加载路径
        """
        if not os.path.exists(filepath):
            print(f"[对手池] 文件不存在: {filepath}")
            return

        data = torch.load(filepath, map_location='cpu', weights_only=False)
        self.max_size = data.get('max_size', self.max_size)
        self.update_interval = data.get('update_interval', self.update_interval)
        self.selection_strategy = data.get('selection_strategy', self.selection_strategy)
        self.games_since_update = data.get('games_since_update', 0)

        self.pool.clear()
        for entry_data in data.get('entries', []):
            entry = OpponentEntry(
                state_dict=entry_data['state_dict'],
                model_id=entry_data['model_id'],
                elo=entry_data.get('elo', 1500.0),
                step=entry_data.get('step', 0),
                model_class_name=entry_data.get('model_class_name', ''),
            )
            entry.created_at = entry_data.get('created_at', time.time())
            entry.games_as_opponent = entry_data.get('games_as_opponent', 0)
            self.pool.append(entry)

        print(f"[对手池] 已从 {filepath} 加载 ({len(self.pool)} 个模型)")
