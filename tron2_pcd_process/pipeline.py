"""处理链 1/2：体素去重 + 帧计数过滤 + 地面偏移自动估计。

对应 ROS1 版 `bag_mapping_pipeline.py` 的核心逻辑，逐函数对照迁移：

    voxel_keys              —— 体素线性编码
    VoxelAccumulator        —— 流式累积"每帧独有的体素"（在线/离线共用）
    estimate_ground_offset  —— 轨迹附近点的相对高度主峰 → 雷达光心到地面高度
    otsu_threshold          —— 帧数两类分割（min_frames=0 时自动）

动态去除默认要求同一 0.05 m 体素出现在 ≥3 帧；手动阈值可设为 1~10，
0 使用 Otsu 自动（钳到 2~4）。不需要 ROS，也不需要回环位姿。
"""

import numpy as np
from typing import Dict, List

from .config import AUTO_MIN_FRAMES, MAX_EXPLICIT_MIN_FRAMES

SPAN = 20000
"""体素整数网格单轴偏移，坐标范围 ±SPAN*resolution（0.05 m 时 ±1000 km）。"""

SIDE = 2 * SPAN + 1


def voxel_keys(xyz: np.ndarray, resolution: float) -> np.ndarray:
    """世界坐标 → 体素线性键（int64）。必须与 visibility 用的编码一致。"""
    grid = np.floor(np.asarray(xyz, dtype=np.float64) / resolution).astype(np.int64) + SPAN
    return (grid[:, 0] * SIDE + grid[:, 1]) * SIDE + grid[:, 2]


class _KeyIndex:
    """键 → 槽位 的 Python dict 索引。

    用 dict 而不是 numpy 排序：每帧只 O(本帧键数) 的哈希查找，
    不会随总体素数增长（numpy 排序方案在 240 万体素时每帧要几秒）。
    键是 int64，Python int 可直接做哈希。
    """

    def __init__(self):
        self.slot: Dict[int, int] = {}

    def __len__(self) -> int:
        return len(self.slot)

    def locate(self, keys: np.ndarray):
        """返回 (已有槽位, 已有来源下标, 新键, 新键来源下标)。"""
        existing_slot, existing_src, new_keys, new_src = [], [], [], []
        slot = self.slot
        for index, key in enumerate(keys.tolist()):
            found = slot.get(key)
            if found is None:
                new_keys.append(key)
                new_src.append(index)
            else:
                existing_slot.append(found)
                existing_src.append(index)
        return (np.asarray(existing_slot, dtype=np.int64),
                np.asarray(existing_src, dtype=np.int64),
                np.asarray(new_keys, dtype=np.int64),
                np.asarray(new_src, dtype=np.int64))


class VoxelAccumulator:
    """流式累积"每个体素被多少帧看到"。

    与 RepresentativeAccumulator 共用同一套"预分配 + dict 索引"结构，
    保证两者的体素键序列**逐位一致**（下游要求 keys 对齐）。
    """

    def __init__(self, resolution: float, initial_capacity: int = 4096):
        self.resolution = float(resolution)
        self._index = _KeyIndex()
        self._capacity = int(initial_capacity)
        self._keys = np.zeros(self._capacity, dtype=np.int64)
        self._counts = np.zeros(self._capacity, dtype=np.int64)
        self._size = 0

    def __len__(self) -> int:
        return self._size

    @property
    def keys(self) -> np.ndarray:
        return self._keys[:self._size]

    @property
    def counts(self) -> np.ndarray:
        return self._counts[:self._size]

    def _grow(self, needed: int) -> None:
        while self._capacity < needed:
            self._capacity *= 2
        self._keys = np.resize(self._keys, self._capacity)
        self._counts = np.resize(self._counts, self._capacity)

    def add_frame(self, xyz: np.ndarray) -> int:
        if len(xyz) == 0:
            return 0
        frame_keys = np.unique(voxel_keys(np.asarray(xyz, np.float32), self.resolution))
        existing, _e_src, new, _n_src = self._index.locate(frame_keys)
        if len(existing):
            self._counts[existing] += 1
        if len(new):
            base = self._size
            if base + len(new) > self._capacity:
                self._grow(base + len(new))
            self._keys[base:base + len(new)] = new
            self._counts[base:base + len(new)] = 1
            self._size = base + len(new)
            self._index.slot.update(
                zip(new.tolist(), range(base, base + len(new))))
        return int(len(new))

    def sorted_view(self):
        """按键升序返回 (keys, counts)——离线路径与原作者版对齐时用。"""
        order = np.argsort(self.keys, kind="stable")
        return self.keys[order], self.counts[order]


class RepresentativeAccumulator:
    """流式版本的体素代表点：每帧首点累加，最后 ÷ 帧数。

    性能要点（实测 240 万体素 / 5744 帧）：
      * 按 2× 容量预分配，扩容时才 np.concatenate，避免每帧重建大数组
      * 累加是原地切片写入 `sum[slots] += pts`，向量化
      * 键查找走 dict（O(本帧键数)），不做全表排序
    实测整包 ~35 s；早期每帧 concat 版本要 153 s，逐元素版本要 10 分钟以上。
    """

    def __init__(self, resolution: float, initial_capacity: int = 4096):
        self.resolution = float(resolution)
        self._index = _KeyIndex()
        self._capacity = int(initial_capacity)
        self._keys = np.zeros(self._capacity, dtype=np.int64)
        self._sum = np.zeros((self._capacity, 3), dtype=np.float64)
        self._hits = np.zeros(self._capacity, dtype=np.int64)
        self._size = 0

    def __len__(self) -> int:
        return self._size

    @property
    def keys(self) -> np.ndarray:
        return self._keys[:self._size]

    def _grow(self, needed: int) -> None:
        while self._capacity < needed:
            self._capacity *= 2
        self._keys = np.resize(self._keys, self._capacity)
        self._sum = np.resize(self._sum, (self._capacity, 3))
        self._hits = np.resize(self._hits, self._capacity)

    def add_frame(self, xyz: np.ndarray) -> None:
        if len(xyz) == 0:
            return
        # 与 ROS1 版一致：在 **float32** 点上看首点（原版 xyz 就是 float32）
        xyz32 = np.asarray(xyz, dtype=np.float32)
        uniq, first = np.unique(voxel_keys(xyz32, self.resolution), return_index=True)
        pts = xyz32[first].astype(np.float64)
        existing, e_src, new, n_src = self._index.locate(uniq)
        if len(existing):
            self._sum[existing] += pts[e_src]
            self._hits[existing] += 1
        if len(new):
            base = self._size
            if base + len(new) > self._capacity:
                self._grow(base + len(new))
            self._keys[base:base + len(new)] = new
            self._sum[base:base + len(new)] = pts[n_src]
            self._hits[base:base + len(new)] = 1
            self._size = base + len(new)
            # 批量建索引：一次 dict.update，比逐条赋值快一个量级
            self._index.slot.update(
                zip(new.tolist(), range(base, base + len(new))))

    @property
    def means(self) -> np.ndarray:
        """代表点（各帧首点均值），与 ROS1 版一致按 float32 取整。"""
        if self._size == 0:
            return np.zeros((0, 3), dtype=np.float64)
        hits = np.maximum(self._hits[:self._size], 1)[:, None]
        return (self._sum[:self._size] / hits).astype(np.float32).astype(np.float64)


def estimate_ground_offset(means, counts, traj_xy, traj_z, near_radius=1.5,
                           lo=-2.5, hi=0.5, bin_width=0.02):
    """轨迹附近点的 (z - 当地轨迹 z) 主峰即地面；offset = -peak。

    用 1 m 格子把点挂到同格的轨迹高度上，全程向量化：
    轨迹发散（格子数暴涨）时也不会退化成循环遍历。
    """
    if len(traj_xy) < 2 or len(means) == 0:
        return None, None

    cell = 1.0
    span = 200000
    side = 2 * span + 1

    tcx = np.floor(traj_xy[:, 0] / cell).astype(np.int64) + span
    tcy = np.floor(traj_xy[:, 1] / cell).astype(np.int64) + span
    tk = tcx * side + tcy

    order = np.argsort(tk, kind="stable")
    tk_sorted = tk[order]
    tz_sorted = traj_z[order].astype(np.float64)
    uniq, starts = np.unique(tk_sorted, return_index=True)
    counts_t = np.diff(np.append(starts, len(tk_sorted)))
    tz_cell = np.add.reduceat(tz_sorted, starts) / counts_t

    pcx = np.floor(means[:, 0] / cell).astype(np.int64) + span
    pcy = np.floor(means[:, 1] / cell).astype(np.int64) + span
    pk = pcx * side + pcy

    pos = np.searchsorted(uniq, pk)
    pos = np.clip(pos, 0, len(uniq) - 1)
    matched = uniq[pos] == pk
    if not matched.any():
        return None, None

    d = means[matched, 2] - tz_cell[pos[matched]]
    hist, edges = np.histogram(d, bins=np.arange(lo, hi + bin_width, bin_width))
    if hist.sum() == 0:
        return None, None
    kernel = np.array([1, 2, 3, 2, 1], dtype=np.float64)
    kernel /= kernel.sum()
    smooth = np.convolve(hist.astype(np.float64), kernel, mode="same")
    peak = int(np.argmax(smooth))
    center = edges[peak] + bin_width / 2
    window = d[np.abs(d - center) <= 2 * bin_width]
    refined = float(np.median(window)) if len(window) else float(center)
    offset = -refined
    hist_info = {
        "peak_center_m": round(float(center), 4),
        "refined_m": round(refined, 4),
        "offset_m": round(offset, 4),
        "matched_voxels": int(matched.sum()),
        "traj_cells": int(len(uniq)),
        "bins": [round(float(v), 3) for v in edges[:-1]],
        "counts": hist.tolist(),
    }
    return offset, hist_info


def otsu_threshold(values, lo=2, hi=40):
    """对帧数做两类分割，返回阈值（保留 count >= 阈值）。"""
    if len(values) == 0:
        return lo
    v = np.clip(np.asarray(values, dtype=np.float64), lo, hi)
    hist, edges = np.histogram(v, bins=256, range=(lo, hi))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return lo
    p = hist / total
    omega = np.cumsum(p)
    centers = (edges[:-1] + edges[1:]) / 2
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mu_t * omega - mu) ** 2 / (omega * (1 - omega))
    sigma_b[~np.isfinite(sigma_b)] = -1
    idx = int(np.argmax(sigma_b))
    return float(centers[idx])


def resolve_min_frames(counts, configured: int):
    """min_frames=1..10 用显式值；0 走 Otsu 并钳到 2~4。

    Otsu 在"低帧数占多数"的长跑数据上会偏高（实测 5 分钟趟给出 13，
    把弱结构也删了），因此钳到 2~4。
    """
    is_integer = int(configured) == configured
    if (not is_integer or
            not (AUTO_MIN_FRAMES <= configured <= MAX_EXPLICIT_MIN_FRAMES)):
        raise ValueError("min_frames must be 0 (auto) or between 1 and 10")
    if configured > AUTO_MIN_FRAMES:
        return int(configured), "explicit"
    auto = otsu_threshold(counts)
    min_frames = int(min(max(2, round(auto)), 4))
    return min_frames, f"auto-otsu(raw={auto:.1f}, clamped 2..4)"


def frame_count_percentiles(counts) -> dict:
    if len(counts) == 0:
        return {q: 0 for q in (50, 90, 99)}
    return {q: int(np.percentile(counts, q)) for q in (50, 90, 99)}
