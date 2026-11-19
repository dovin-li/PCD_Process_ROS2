"""处理参数：一份配置，离线/在线/命令行三处共用。

默认值 = ROS1 版 `bag_to_map.py` 的固化参数（2026-10-27 现场验收版），
改这里就是改全链路；README 与 config/*.yaml 必须同步。

所有字段都带单位，名字直接对应原版常量，方便逐项对照。
"""

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Mapping, Optional


@dataclass(frozen=True)
class PcdProcessConfig:
    # ---- 点云过滤（pipeline） ----
    voxel_m: float = 0.05
    """体素去重粒度，与 2D 图分辨率一致。"""

    min_frames: int = 3
    """同一体素至少被几帧观测才保留。2=细节多但杂，4=干净但丢矮结构。0=Otsu 自动（钳 2~4）。"""

    max_points_per_frame: int = 0
    """每帧最多取多少点（0=不限），仅在线节点限内存用。"""

    # ---- 地面基准 ----
    ground_offset_m: Optional[float] = None
    """雷达光心到地面高度；None=自动估计（轨迹附近点的相对高度主峰）。"""

    # ---- 2D 投影（projection） ----
    resolution_m: float = 0.05
    height_low_m: float = 0.10
    """障碍窗下沿（相对当地地面）。低于此不算障碍 = 地面噪声余量。"""

    height_high_m: float = 1.00
    """障碍窗上沿（相对当地地面）。高于此不算 = 桌面 / 人上半身 / 横梁。"""

    smooth_window_m: float = 0.50
    """轨迹 z 中值平滑窗（按弧长）。"""

    segment_length_m: float = 1.00
    segment_z_span_m: float = 0.10
    """轨迹分段：每段换一次地面基准；坡道上靠这个做到"不一刀切"。"""

    neighbor_count: int = 16
    candidate_distance_tolerance_m: float = 0.10
    max_route_distance_m: float = 8.0
    """离轨迹更远的点丢弃，防远场飞点撑大地图。"""

    occupancy_radius_m: float = 0.05
    margin_m: float = 1.0
    max_grid_cells: int = 100_000_000

    raycast_free: bool = True
    raycast_skip_near_m: float = 0.20
    raycast_source: str = "ground"          # "ground" | "all"
    raycast_min_support: int = 3
    raycast_support_radius_m: float = 0.30

    # ---- 轨迹体检（不通过拒绝出图） ----
    max_step_m: float = 1.0
    max_z_span_m: float = 5.0
    min_duration_s: float = 10.0

    # ---- 话题 / frame 契约 ----
    odom_topic: str = "/Odometry"
    cloud_topic: str = "/cloud_registered"
    odom_frame: str = "camera_init"
    odom_child_frame: str = "body"
    cloud_frame: str = "camera_init"

    # ---- 在线节点专用 ----
    # 实测（scan_0）：101 个位姿时估 0.88、301 个估 -0.07，1001 个才稳定到 1.185。
    # 因此门槛必须比"能出数"高得多，否则在线图的地面基准一开始就是错的。
    ground_estimate_min_poses: int = 1000
    """攒够这么多位姿后才估地面偏移；估出来就冻结（早期点少主峰会跑偏）。"""

    ground_estimate_min_voxels: int = 300000
    max_traj_poses: int = 200_000
    """位姿轨迹上限，超过后等距抽稀，防止长时间建图内存线性涨。"""

    def to_ros_params(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.__annotations__}

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "PcdProcessConfig":
        """ROS 参数表 → 配置；忽略未知键，None 视为"未设置"（保留默认）。"""
        known = {f.name: f.type for f in fields(cls)}
        changed: Dict[str, Any] = {}
        for key, value in params.items():
            if key not in known or value is None:
                continue
            changed[key] = value
        return cls(**changed)

    def replace(self, **changes: Any) -> "PcdProcessConfig":
        unknown = set(changes) - {f.name for f in fields(self)}
        if unknown:
            raise TypeError(f"unknown config field(s): {sorted(unknown)}")
        return PcdProcessConfig(**{**asdict(self), **changes})

    def validate(self) -> None:
        if self.voxel_m <= 0 or self.resolution_m <= 0:
            raise ValueError("voxel_m / resolution_m must be > 0")
        if self.min_frames < 0:
            raise ValueError("min_frames must be >= 0")
        if not (self.height_low_m < self.height_high_m):
            raise ValueError("height_low_m must be < height_high_m")
        if self.segment_length_m <= 0 or self.segment_z_span_m <= 0:
            raise ValueError("segment_length_m / segment_z_span_m must be > 0")
        if self.max_route_distance_m <= 0:
            raise ValueError("max_route_distance_m must be > 0")
        if self.raycast_source not in ("ground", "all"):
            raise ValueError("raycast_source must be 'ground' or 'all'")


DEFAULT_CONFIG = PcdProcessConfig()
