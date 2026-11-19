"""Interpolate world-frame corrections from paired TUM keyframe poses."""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def _read_poses(path):
    path = Path(path).expanduser()
    try:
        poses = np.loadtxt(path, ndmin=2)
    except (OSError, ValueError) as exc:
        raise ValueError(f"无法读取关键帧 TUM {path}（{exc}）") from exc
    if len(poses) < 2:
        raise ValueError(f"{path} 的关键帧 TUM 至少两行")
    if poses.shape[1] != 8:
        raise ValueError(f"{path} 的关键帧 TUM 每行必须有 8 列")
    if not np.isfinite(poses).all():
        raise ValueError(f"{path} 的时间和位姿必须是有限数")
    norms = np.linalg.norm(poses[:, 4:8], axis=1)
    if (norms < 1e-12).any() or not np.isfinite(norms).all():
        raise ValueError(f"{path} 含无法归一化的四元数")
    return poses


class PoseCorrection:
    """C(t) maps points in the original FAST-LIO world into the optimized world."""

    def __init__(self, raw_path, optimized_path):
        raw = _read_poses(raw_path)
        optimized = _read_poses(optimized_path)
        if len(raw) != len(optimized):
            raise ValueError("原始和优化关键帧 TUM 行数不同")
        if (np.diff(raw[:, 0]) <= 0).any() or (np.diff(optimized[:, 0]) <= 0).any():
            raise ValueError("原始和优化关键帧 TUM 的时间戳均须严格递增")
        if not np.allclose(raw[:, 0], optimized[:, 0], rtol=0, atol=1e-6):
            raise ValueError("原始和优化关键帧 TUM 时间戳不匹配（容差 1 微秒）")
        self.stamps = raw[:, 0]
        # Rotation.from_quat normalizes valid TUM quaternions on construction.
        raw_rotations = Rotation.from_quat(raw[:, 4:8])
        rotations = Rotation.from_quat(optimized[:, 4:8]) * raw_rotations.inv()
        self.translations = optimized[:, 1:4] - rotations.apply(raw[:, 1:4])
        self.slerp = Slerp(self.stamps, rotations)

    def at(self, stamp):
        # Keep the closest correction before the first and after the last keyframe.
        t = np.clip(stamp, self.stamps[0], self.stamps[-1])
        right = np.minimum(np.searchsorted(self.stamps, t, side="right"), len(self.stamps) - 1)
        left = right - 1
        alpha = np.asarray((t - self.stamps[left]) / (self.stamps[right] - self.stamps[left]))[..., None]
        offset = (1 - alpha) * self.translations[left] + alpha * self.translations[right]
        return self.slerp(t), offset
