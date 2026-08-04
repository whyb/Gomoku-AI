"""
Elo 评分系统 — 衡量模型棋力

原理:
  E_A = 1 / (1 + 10^((R_B - R_A) / 400))  — A 的预期胜率
  R_A' = R_A + K × (S_A - E_A)              — 更新后的评分
  其中 S_A = 1(赢) / 0.5(平) / 0(输), K = 32(默认)
"""

import time
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict


@dataclass
class ModelRecord:
    """模型记录"""
    model_id: str
    elo: float = 1500.0
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    created_at: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)  # 额外信息 (board_size, step 等)

    @property
    def win_rate(self) -> float:
        if self.games_played == 0:
            return 0.0
        return self.wins / self.games_played


class EloRating:
    """Elo 评分系统"""

    def __init__(self, k_factor: float = 32, default_elo: float = 1500.0):
        self.ratings: Dict[str, ModelRecord] = {}
        self.k_factor = k_factor
        self.default_elo = default_elo
        self.history: List[Dict] = []  # 对战历史

    def add_model(self, model_id: str, elo: float = None, **metadata):
        """注册新模型"""
        if model_id not in self.ratings:
            self.ratings[model_id] = ModelRecord(
                model_id=model_id,
                elo=elo or self.default_elo,
                metadata=metadata
            )

    def get_rating(self, model_id: str) -> float:
        """获取模型评分"""
        if model_id not in self.ratings:
            return self.default_elo
        return self.ratings[model_id].elo

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """A 对 B 的预期胜率"""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def update(self, winner_id: str, loser_id: str, tie: bool = False):
        """
        更新评分

        Args:
            winner_id: 赢家模型 ID (tie 时无所谓)
            loser_id: 输家模型 ID
            tie: 是否平局
        """
        self.add_model(winner_id)
        self.add_model(loser_id)

        ra = self.ratings[winner_id].elo
        rb = self.ratings[loser_id].elo
        ea = self.expected_score(ra, rb)
        eb = self.expected_score(rb, ra)

        if tie:
            sa, sb = 0.5, 0.5
        else:
            sa, sb = 1.0, 0.0

        self.ratings[winner_id].elo = ra + self.k_factor * (sa - ea)
        self.ratings[loser_id].elo = rb + self.k_factor * (sb - eb)

        # 更新统计
        self.ratings[winner_id].games_played += 1
        self.ratings[loser_id].games_played += 1
        if tie:
            self.ratings[winner_id].ties += 1
            self.ratings[loser_id].ties += 1
        else:
            self.ratings[winner_id].wins += 1
            self.ratings[loser_id].losses += 1

        # 记录历史
        self.history.append({
            'winner': winner_id,
            'loser': loser_id,
            'tie': tie,
            'winner_elo_after': self.ratings[winner_id].elo,
            'loser_elo_after': self.ratings[loser_id].elo,
            'timestamp': time.time()
        })

    def leaderboard(self, top_n: int = 20) -> List[Tuple[str, float, int]]:
        """
        排行榜

        Returns:
            [(model_id, elo, games_played), ...]
        """
        sorted_models = sorted(
            self.ratings.values(),
            key=lambda m: -m.elo
        )
        return [
            (m.model_id, round(m.elo, 1), m.games_played)
            for m in sorted_models[:top_n]
        ]

    def print_leaderboard(self, top_n: int = 10):
        """打印排行榜"""
        lb = self.leaderboard(top_n)
        print("\n" + "=" * 60)
        print(f"{'排名':<4} {'模型ID':<30} {'Elo':<8} {'对局数':<8} {'胜率':<8}")
        print("=" * 60)
        for i, (mid, elo, games) in enumerate(lb, 1):
            record = self.ratings[mid]
            wr = f"{record.win_rate:.1%}" if games > 0 else "N/A"
            print(f"{i:<4} {mid:<30} {elo:<8} {games:<8} {wr:<8}")
        print("=" * 60)

    def save(self, filepath: str):
        """保存到文件"""
        data = {
            'k_factor': self.k_factor,
            'default_elo': self.default_elo,
            'ratings': {
                mid: asdict(record)
                for mid, record in self.ratings.items()
            },
            'history': self.history[-1000:]  # 只保留最近 1000 条
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, filepath: str):
        """从文件加载 (兼容旧版存档中的 draws/draw 字段)"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.k_factor = data['k_factor']
        self.default_elo = data['default_elo']
        records = {}
        for mid, record in data['ratings'].items():
            # 旧版存档字段名 draws → ties
            if 'draws' in record and 'ties' not in record:
                record['ties'] = record.pop('draws')
            records[mid] = ModelRecord(**record)
        self.ratings = records
        self.history = data.get('history', [])
        # 旧版存档字段名 draw → tie
        for h in self.history:
            if 'draw' in h and 'tie' not in h:
                h['tie'] = h.pop('draw')


if __name__ == '__main__':
    # 示例用法
    elo = EloRating()
    elo.add_model('model_step_10000', board_size=10)
    elo.add_model('model_step_20000', board_size=10)
    elo.add_model('model_step_30000', board_size=10)

    # 模拟 10 场对战
    import random
    models = list(elo.ratings.keys())
    for _ in range(10):
        a, b = random.sample(models, 2)
        winner = random.choice([a, b])
        loser = b if winner == a else a
        elo.update(winner, loser)

    elo.print_leaderboard()
