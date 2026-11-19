"""ROS2 版 PCD Process：FAST-LIO 点云 → 静态 PCD + 2D 占据栅格地图。

常用入口：

    from tron2_pcd_process import DEFAULT_CONFIG, PcdProcessConfig
    from tron2_pcd_process.bag_reader import iter_bag
    from tron2_pcd_process.pipeline import VoxelAccumulator, RepresentativeAccumulator
    from tron2_pcd_process.projection import project_to_grid, write_map

命令行入口（console_scripts）：
    pcd_process_bag        离线：ROS2 bag → PCD + 2D 图
    pcd_process_online     在线：订阅话题边建图边出图
    pcd_process_ros1_to_ros2  ROS1 bag → ROS2 bag

注意：核心模块（config/points/pipeline/projection/bag_reader）不 import rclpy，
可在没有 ROS 的机器上直接使用；只有 *_node 模块需要 rclpy。
"""

__version__ = "0.2.0"

from .config import DEFAULT_CONFIG, PcdProcessConfig  # noqa: E402
from .points import OdometrySample, PointCloudFrame, Trajectory, read_pcd_xyz, write_pcd  # noqa: E402

__all__ = [
    "__version__",
    "PcdProcessConfig",
    "DEFAULT_CONFIG",
    "PointCloudFrame",
    "OdometrySample",
    "Trajectory",
    "read_pcd_xyz",
    "write_pcd",
]
