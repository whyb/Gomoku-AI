"""
D4 对称增强 — 五子棋棋盘的 8 种等价变换

正方形的对称群 D4 包含:
  - 4 种旋转: 0°, 90°, 180°, 270°
  - 4 种翻转+旋转: 水平翻转, 然后 0°/90°/180°/270° 旋转

每盘棋的所有经验都可以生成 8 个等价版本 → 免费 8× 数据
"""

import numpy as np
from typing import Tuple


class SymmetryAugmenter:
    """五子棋 D4 对称增强"""

    NUM_TRANSFORMS = 8

    @staticmethod
    def augment(state: np.ndarray, policy_2d: np.ndarray,
                transform_idx: int = -1) -> Tuple[np.ndarray, np.ndarray]:
        """
        对 (state, policy) 进行对称变换

        Args:
            state: (C, H, W) 棋盘状态 (C 个通道)
            policy_2d: (H, W) 策略 2D 数组
            transform_idx: 变换索引 (0-7), -1 表示随机选择
        Returns:
            augmented_state: (C, H, W)
            augmented_policy_2d: (H, W)
        """
        if transform_idx < 0:
            transform_idx = np.random.randint(SymmetryAugmenter.NUM_TRANSFORMS)

        if transform_idx < 4:
            # 旋转 k×90°
            k = transform_idx
            aug_state = np.rot90(state, k=k, axes=(1, 2)).copy()
            aug_policy = np.rot90(policy_2d, k=k).copy()
        else:
            # 水平翻转 + 旋转 (transform_idx-4)×90°
            aug_state = np.flip(state, axis=2).copy()
            aug_policy = np.flip(policy_2d, axis=1).copy()
            k = transform_idx - 4
            if k > 0:
                aug_state = np.rot90(aug_state, k=k, axes=(1, 2)).copy()
                aug_policy = np.rot90(aug_policy, k=k).copy()

        return aug_state, aug_policy

    @staticmethod
    def augment_batch(states: np.ndarray, policies_2d: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量增强: 每个样本独立随机变换

        Args:
            states: (B, C, H, W)
            policies_2d: (B, H, W)
        Returns:
            augmented states, augmented policies_2d
        """
        aug_states = []
        aug_policies = []
        for s, p in zip(states, policies_2d):
            s_aug, p_aug = SymmetryAugmenter.augment(s, p)
            aug_states.append(s_aug)
            aug_policies.append(p_aug)
        return np.stack(aug_states), np.stack(aug_policies)

    @staticmethod
    def policy_1d_to_2d(policy_1d: np.ndarray, board_size: int) -> np.ndarray:
        """将 (H*W,) 策略向量转为 (H, W)"""
        return policy_1d.reshape(board_size, board_size)

    @staticmethod
    def policy_2d_to_1d(policy_2d: np.ndarray) -> np.ndarray:
        """将 (H, W) 策略转为 (H*W,) 向量"""
        return policy_2d.reshape(-1)

    @staticmethod
    def augment_experience(state: np.ndarray, policy_1d: np.ndarray,
                           board_size: int,
                           transform_idx: int = -1
                           ) -> Tuple[np.ndarray, np.ndarray]:
        """
        增强单条经验数据 (方便直接调用)

        Args:
            state: (C, H, W)
            policy_1d: (H*W,)
            board_size: 棋盘大小
            transform_idx: 变换索引, -1 为随机
        Returns:
            augmented_state, augmented_policy_1d
        """
        policy_2d = SymmetryAugmenter.policy_1d_to_2d(policy_1d, board_size)
        s_aug, p_aug = SymmetryAugmenter.augment(state, policy_2d, transform_idx)
        return s_aug, SymmetryAugmenter.policy_2d_to_1d(p_aug)


def verify_symmetry():
    """验证对称增强的正确性 (调试用)"""
    board_size = 5
    # 构造一个不对称的棋盘
    board = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.float32)

    state = np.stack([board, (board == 0).astype(np.float32)])
    policy = np.arange(25, dtype=np.float32)

    print("原始 policy 2D:")
    print(policy.reshape(5, 5))

    for i in range(8):
        s_aug, p_aug = SymmetryAugmenter.augment_experience(state, policy, 5, i)
        print(f"\n变换 {i}: policy 2D:")
        print(p_aug.reshape(5, 5))
        print(f"state shape: {s_aug.shape}, sum: {s_aug.sum():.1f}")


if __name__ == '__main__':
    verify_symmetry()
