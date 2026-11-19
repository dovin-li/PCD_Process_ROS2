"""处理链 2/2：逐段地面基准的高度窗投影 → 2D 覆盖栅格。

对应 ROS1 版 `slope_map_projector.py` 的核心逻辑。语义与原版一致：

    * 轨迹按"弧长 1.0 m 或高差 0.10 m"分段，每段一个地面基准 z；
      这是坡道上不"一刀切"的关键。
    * 点相对当地地面高度落在 [height_low, height_high] 且离轨迹 <= 8 m
      → 障碍格（0）。
    * 地面带 [-0.30, height_low) 的点 → 可通行（254）。
    * 射线自由空间填充：从段位姿向观测点连线，遇障碍即停。
    * 输出 0=障碍 / 205=未知 / 254=可通行。

与 ROS1 版的差异（都是有意为之，见 docs/与ROS1版差异.md）：
    * 只接受 numpy 数组，不再 memmap PCD 文件（在线节点没有文件可读）；
      离线节点负责把 PCD 读成数组后调用这里。
    * 栅格统计与写盘拆开（`project_to_grid` 返回数组，`write_map` 落盘），
      在线节点可以只算不写。
"""

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .config import PcdProcessConfig
from .points import Trajectory, write_pgm_yaml

FLOOR_LO_M = -0.30
"""地面带下沿（相对当地地面）：地面本身有 6~10 cm 噪声余量。"""


@dataclass
class FreeSpaceLayer:
    """可通行层（按 world 坐标存稀疏点，不同栅格尺寸之间可复用）。"""

    points_xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float64))
    segment_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))


@dataclass
class ProjectionResult:
    occupied: np.ndarray
    free: np.ndarray
    ground_free: np.ndarray
    ray_free: np.ndarray
    xmin: float
    ymin: float
    width: int
    height: int
    resolution_m: float
    segment_xy: np.ndarray
    segment_z: np.ndarray
    segment_sample_counts: np.ndarray
    segment_hits: np.ndarray
    projected_points: int
    finite_points: int
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    elapsed_s: float

    def metrics(self) -> dict:
        occupied = self.occupied
        free = self.free
        return {
            "trajectory_rows": int(self.segment_sample_counts.sum()),
            "projected_points": int(self.projected_points),
            "finite_points": int(self.finite_points),
            "resolution_m": float(self.resolution_m),
            "trajectory_segments": [
                {
                    "segment_id": int(i),
                    "representative_z_m": float(z),
                    "trajectory_samples": int(samples),
                    "projected_points": int(hits),
                }
                for i, (z, samples, hits) in enumerate(
                    zip(self.segment_z, self.segment_sample_counts, self.segment_hits))
            ],
            "grid_width": int(self.width),
            "grid_height": int(self.height),
            "origin": [float(self.xmin), float(self.ymin), 0.0],
            "point_bounds_min_xyz": [float(v) for v in self.bounds_min],
            "point_bounds_max_xyz": [float(v) for v in self.bounds_max],
            "occupied_cells": int(occupied.sum()),
            "free_cells": int(free.sum()),
            "free_cells_ground": int((self.ground_free & ~occupied).sum()),
            "free_cells_raycast": int((self.ray_free & ~occupied).sum()),
            "unknown_cells": int((~occupied & ~free).sum()),
        }

    def pgm_image(self) -> np.ndarray:
        image = np.full((self.height, self.width), 205, dtype=np.uint8)
        image[self.free] = 254
        image[self.occupied] = 0
        return image


# ------------------------------------------------------------------ 轨迹分段

def smooth_trajectory_z(positions, window_m):
    """按弧长窗口做 z 中值平滑（不是按采样点数）。"""
    positions = np.asarray(positions, dtype=np.float64)
    if window_m <= 0.0:
        raise ValueError("smooth window must be positive")
    segment_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    distance = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    half_window = float(window_m) * 0.5
    result = np.empty(len(positions), dtype=np.float64)
    for index, center in enumerate(distance):
        left = int(np.searchsorted(distance, center - half_window, side="left"))
        right = int(np.searchsorted(distance, center + half_window, side="right"))
        result[index] = np.median(positions[left:max(left + 1, right), 2])
    return distance, result


def segment_trajectory(distance, smoothed_z, max_length_m, max_z_span_m):
    """弧长或高差任一超限就切段（坡道上换地面基准）。"""
    if max_length_m <= 0.0 or max_z_span_m <= 0.0:
        raise ValueError("segment length and Z span must be positive")
    ids = np.zeros(len(distance), dtype=np.int32)
    start = 0
    segment_id = 0
    for index in range(1, len(distance)):
        z_span = float(np.ptp(smoothed_z[start:index + 1]))
        if distance[index] - distance[start] >= max_length_m or z_span >= max_z_span_m:
            segment_id += 1
            start = index
        ids[index] = segment_id
    return ids


def make_segment_references(positions, smoothed_z, segment_ids):
    """每段收敛成一个稳定代表：xy 中值 + z 中值。"""
    centers, heights, lengths = [], [], []
    for segment_id in range(int(segment_ids[-1]) + 1):
        indices = np.flatnonzero(segment_ids == segment_id)
        if not len(indices):
            continue
        centers.append(np.median(positions[indices, :2], axis=0))
        heights.append(float(np.median(smoothed_z[indices])))
        lengths.append(int(len(indices)))
    if not centers:
        raise ValueError("trajectory segmentation produced no segments")
    return np.asarray(centers), np.asarray(heights), np.asarray(lengths)


def build_segments(traj: Trajectory, cfg: PcdProcessConfig):
    """轨迹 → (segment_xy, segment_z, samples)。"""
    if len(traj) < 2:
        raise ValueError("轨迹至少需要 2 条位姿才能分段投影")
    positions = traj.positions
    distance, smoothed = smooth_trajectory_z(positions, cfg.smooth_window_m)
    ids = segment_trajectory(distance, smoothed, cfg.segment_length_m, cfg.segment_z_span_m)
    return make_segment_references(positions, smoothed, ids)


# ------------------------------------------------------------------ 高度窗分类

def _classify(points, segment_xy, segment_z, tree, ground_offset_m, cfg):
    """返回 (keep, selected_segment, ground_mask, ground_segment)。"""
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        empty_b = np.zeros(0, dtype=bool)
        empty_i = np.zeros(0, dtype=np.int32)
        return empty_b, empty_i, empty_b, empty_i
    k = min(int(cfg.neighbor_count), len(segment_z))
    distances, indices = tree.query(points[:, :2], k=k, workers=1)
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    rel = points[:, 2, None] - (segment_z[indices] - ground_offset_m)
    # 只与空间上最近的轨迹段比较，防止远处低坡段把上坡地面判成障碍
    local = distances <= (distances[:, :1] + cfg.candidate_distance_tolerance_m)
    valid = local & (rel >= cfg.height_low_m) & (rel <= cfg.height_high_m)
    if cfg.max_route_distance_m is not None:
        valid &= distances <= float(cfg.max_route_distance_m)
    ranked = np.where(valid, distances, np.inf)
    selected = np.argmin(ranked, axis=1)
    keep = np.isfinite(ranked[np.arange(len(points)), selected])
    selected_segment = indices[np.arange(len(points)), selected].astype(np.int32)

    band = local & (rel >= FLOOR_LO_M) & (rel < cfg.height_low_m)
    ground_mask = band.any(axis=1)
    ground_segment = indices[ground_mask, 0].astype(np.int32)
    return keep, selected_segment, ground_mask, ground_segment


def collect_free_space(points, segment_xy, segment_z, tree, ground_offset_m, cfg):
    """从点云里取"地面带"和"障碍窗"的点，供射线填充用（与栅格无关）。"""
    keep, _sel, ground_mask, ground_seg = _classify(
        points, segment_xy, segment_z, tree, ground_offset_m, cfg)
    ground_pts = np.asarray(points, dtype=np.float64)[ground_mask]
    hit_pts = np.asarray(points, dtype=np.float64)[keep]
    if cfg.raycast_source == "ground" and len(ground_pts):
        pts, segs = ground_pts, ground_seg
    else:
        pts = np.vstack([hit_pts, ground_pts]) if len(ground_pts) else hit_pts
        segs = (np.concatenate([_sel[keep], ground_seg])
                if len(ground_seg) else _sel[keep])
    return FreeSpaceLayer(points_xy=pts[:, :2], segment_ids=segs)


# ------------------------------------------------------------------ 栅格投影

def compute_bounds(points_xy, resolution, margin_m) -> tuple:
    if len(points_xy) == 0:
        raise ValueError("height-filtered projection produced no points")
    minimum = np.min(points_xy, axis=0)
    maximum = np.max(points_xy, axis=0)
    xmin = math.floor((minimum[0] - margin_m) / resolution) * resolution
    ymin = math.floor((minimum[1] - margin_m) / resolution) * resolution
    xmax = math.ceil((maximum[0] + margin_m) / resolution) * resolution
    ymax = math.ceil((maximum[1] + margin_m) / resolution) * resolution
    width = int(round((xmax - xmin) / resolution)) + 1
    height = int(round((ymax - ymin) / resolution)) + 1
    return xmin, ymin, width, height


def rasterize_hits(points_xy, resolution, xmin, ymin, width, height, occupancy_radius_m):
    gx = np.floor((points_xy[:, 0] - xmin) / resolution).astype(np.int64)
    gy = np.floor((points_xy[:, 1] - ymin) / resolution).astype(np.int64)
    inside = (gx >= 0) & (gx < width) & (gy >= 0) & (gy < height)
    occupied = np.zeros((height, width), dtype=bool)
    occupied[gy[inside], gx[inside]] = True
    radius_cells = int(math.ceil(max(0.0, occupancy_radius_m) / resolution))
    if radius_cells:
        structure = np.ones((2 * radius_cells + 1, 2 * radius_cells + 1), dtype=bool)
        occupied = ndimage.binary_dilation(occupied, structure=structure)
        occupied = ndimage.binary_closing(occupied, structure=structure)
    return occupied, int(np.count_nonzero(inside))


def raycast_free_cells(points_xy, segment_ids, segment_xy, xmin, ymin, width, height,
                       resolution, occupied, skip_near_m=0.20, max_range_m=None,
                       block=20000):
    """从段位姿向点连线，射线经过的格子标可通行；遇障碍格即停。

    标准 ray casting 语义：不停会"穿墙"，在地图外画出白色条纹。
    """
    free = np.zeros((height, width), dtype=bool)
    if len(points_xy) == 0:
        return free
    flat = free.reshape(-1)
    occ_flat = occupied.reshape(-1)
    for start in range(0, len(points_xy), block):
        stop = min(start + block, len(points_xy))
        pts = points_xy[start:stop]
        seg = segment_ids[start:stop]
        ox = segment_xy[seg, 0]
        oy = segment_xy[seg, 1]
        dx = pts[:, 0] - ox
        dy = pts[:, 1] - oy
        planar = np.hypot(dx, dy)
        usable = planar > skip_near_m
        if max_range_m is not None:
            usable &= planar <= float(max_range_m)
        if not usable.any():
            continue
        ox, oy, dx, dy, planar = ox[usable], oy[usable], dx[usable], dy[usable], planar[usable]
        steps = np.maximum(1, np.ceil(planar / resolution).astype(np.int64))
        total = int(steps.sum())
        ray_id = np.repeat(np.arange(len(planar), dtype=np.int64), steps)
        starts = np.repeat(np.cumsum(steps) - steps, steps)
        frac = (np.arange(total, dtype=np.float64) - starts + 0.5) / steps[ray_id]
        sx = (ox[ray_id] + frac * dx[ray_id]).astype(np.float32)
        sy = (oy[ray_id] + frac * dy[ray_id]).astype(np.float32)
        gx = np.floor((sx - xmin) / resolution).astype(np.int32)
        gy = np.floor((sy - ymin) / resolution).astype(np.int32)
        valid = (gx >= 0) & (gx < width) & (gy >= 0) & (gy < height)
        valid &= frac * planar[ray_id] > skip_near_m
        if not valid.any():
            continue
        rid = ray_id[valid]
        lin = gy[valid].astype(np.int64) * width + gx[valid]
        kept = np.ones(len(lin), dtype=bool)
        hits = occ_flat[lin]
        if hits.any():
            hit_positions = np.flatnonzero(hits)
            first_ray, first_occurrence = np.unique(rid[hit_positions], return_index=True)
            first_stop_index = hit_positions[first_occurrence]
            limit = np.full(int(rid.max()) + 1, np.iinfo(np.int64).max, dtype=np.int64)
            limit[first_ray] = first_stop_index
            kept = np.arange(len(lin), dtype=np.int64) < limit[rid]
        flat[np.unique(lin[kept])] = True
    return free


def filter_isolated(points_xy, min_support, radius_m):
    """孤立点不参与射线，否则每个离群点都会拉出一条穿图的白色细线。"""
    if len(points_xy) < 2 or min_support <= 0:
        return np.ones(len(points_xy), dtype=bool)
    k = min(len(points_xy), min_support + 1)
    tree = cKDTree(points_xy)
    dists, _ = tree.query(points_xy, k=k, workers=1)
    if dists.ndim == 1:
        dists = dists[:, None]
    near = (dists[:, 1:] <= radius_m).sum(axis=1)
    return near >= min_support


def project_to_grid(points, traj: Trajectory, ground_offset_m: float,
                    cfg: PcdProcessConfig,
                    segments=None,
                    free_layer: Optional[FreeSpaceLayer] = None) -> ProjectionResult:
    """点云 + 轨迹 → 2D 覆盖栅格（含可通行层与射线填充）。

    segments   : (segment_xy, segment_z, samples)，不给就现算
    free_layer : 可通行层（稀疏点）；给了就跳过 classify 里的地面/障碍判定
    """
    started = time.monotonic()
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if len(points) == 0:
        raise ValueError("点云为空或全是 NaN")
    seg_xy, seg_z, seg_samples = segments or build_segments(traj, cfg)
    tree = cKDTree(seg_xy)
    if len(seg_xy) < 1:
        raise ValueError("trajectory has no usable segment references")

    # 障碍点：分块做 kNN，避免 110 万点一次性 × k16 爆内存
    block = 200_000
    keep_blocks, seg_blocks, ground_pts, ground_seg = [], [], [], []
    segment_hits = np.zeros(len(seg_xy), dtype=np.int64)
    for start in range(0, len(points), block):
        chunk = points[start:start + block]
        keep, selected, gmask, gseg = _classify(
            chunk, seg_xy, seg_z, tree, ground_offset_m, cfg)
        if gmask.any():
            ground_pts.append(chunk[gmask])
            ground_seg.append(gseg)
        if keep.any():
            keep_blocks.append(chunk[keep])
            seg_blocks.append(selected[keep])
            segment_hits += np.bincount(selected[keep], minlength=len(segment_hits))
    if not keep_blocks:
        raise ValueError("高度窗与轨迹距离过滤后没有障碍点")

    hits = np.concatenate(keep_blocks, axis=0)
    hit_segs = np.concatenate(seg_blocks)
    ground_pts = np.concatenate(ground_pts, axis=0) if ground_pts else np.zeros((0, 3))
    ground_segs = np.concatenate(ground_seg) if ground_seg else np.zeros(0, dtype=np.int32)

    all_xy = np.vstack([hits[:, :2], ground_pts[:, :2]]) if len(ground_pts) else hits[:, :2]
    xmin, ymin, width, height = compute_bounds(all_xy, cfg.resolution_m, cfg.margin_m)
    if width * height > cfg.max_grid_cells:
        raise ValueError(f"grid would contain {width * height} cells (limit {cfg.max_grid_cells})")

    occupied, points_used = rasterize_hits(
        hits[:, :2], cfg.resolution_m, xmin, ymin, width, height, cfg.occupancy_radius_m)

    ground_free = np.zeros((height, width), dtype=bool)
    if len(ground_pts):
        gx = np.floor((ground_pts[:, 0] - xmin) / cfg.resolution_m).astype(np.int64)
        gy = np.floor((ground_pts[:, 1] - ymin) / cfg.resolution_m).astype(np.int64)
        inside = (gx >= 0) & (gx < width) & (gy >= 0) & (gy < height)
        ground_free[gy[inside], gx[inside]] = True

    ray_free = np.zeros((height, width), dtype=bool)
    if cfg.raycast_free:
        if free_layer is not None:
            ray_pts = free_layer.points_xy
            ray_segs = free_layer.segment_ids
        elif cfg.raycast_source == "ground" and len(ground_pts):
            ray_pts, ray_segs = ground_pts[:, :2], ground_segs
        else:
            ray_pts = np.vstack([hits[:, :2], ground_pts[:, :2]]) if len(ground_pts) else hits[:, :2]
            ray_segs = (np.concatenate([hit_segs, ground_segs])
                        if len(ground_segs) else hit_segs)
        if len(ray_pts) >= 2:
            supported = filter_isolated(ray_pts, cfg.raycast_min_support,
                                        cfg.raycast_support_radius_m)
            ray_pts, ray_segs = ray_pts[supported], ray_segs[supported]
        ray_free = raycast_free_cells(
            ray_pts, ray_segs, seg_xy, xmin, ymin, width, height,
            cfg.resolution_m, occupied,
            skip_near_m=cfg.raycast_skip_near_m,
            max_range_m=cfg.max_route_distance_m)

    free = (ground_free | ray_free) & ~occupied
    if cfg.raycast_free:
        structure = np.ones((3, 3), dtype=bool)
        free = ndimage.binary_closing(free, structure=structure)
        free &= ~ndimage.binary_dilation(occupied, structure=structure)

    return ProjectionResult(
        occupied=occupied, free=free, ground_free=ground_free, ray_free=ray_free,
        xmin=float(xmin), ymin=float(ymin), width=int(width), height=int(height),
        resolution_m=float(cfg.resolution_m),
        segment_xy=seg_xy, segment_z=seg_z, segment_sample_counts=seg_samples,
        segment_hits=segment_hits, projected_points=int(points_used),
        finite_points=int(len(points)),
        bounds_min=points.min(axis=0), bounds_max=points.max(axis=0),
        elapsed_s=float(time.monotonic() - started),
    )


def write_map(result: ProjectionResult, prefix):
    """把投影结果写成 <prefix>.pgm / <prefix>.yaml，返回 (pgm, yaml)。"""
    return write_pgm_yaml(result.pgm_image(), f"{prefix}.pgm",
                          result.resolution_m, result.xmin, result.ymin)
