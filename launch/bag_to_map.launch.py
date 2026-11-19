"""离线处理 launch：ROS2 bag → 静态 PCD + 2D 地图。

等价命令行：
    ros2 launch tron2_pcd_process bag_to_map.launch.py \
        bag:=/bags/scan_0 out_dir:=/tmp/out/scan_0

注意：这是**一次性任务**，节点跑完自行退出，不适合在 launch 里长期挂着。
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument("bag", description="ROS2 bag 目录或 .db3 文件"),
        DeclareLaunchArgument("out_dir", description="输出目录"),
        DeclareLaunchArgument("force", default_value="false",
                             description="跳过轨迹体检（里程计发散也出图）"),
        DeclareLaunchArgument("overwrite", default_value="false"),
        DeclareLaunchArgument("odom_topic", default_value="/Odometry"),
        DeclareLaunchArgument("cloud_topic", default_value="/cloud_registered"),
        DeclareLaunchArgument("voxel_m", default_value="0.05"),
        DeclareLaunchArgument("min_frames", default_value="3"),
        DeclareLaunchArgument("ground_offset_m", default_value=""),
        DeclareLaunchArgument("height_low_m", default_value="0.10"),
        DeclareLaunchArgument("height_high_m", default_value="1.00"),
        DeclareLaunchArgument("resolution_m", default_value="0.05"),
        DeclareLaunchArgument("max_route_distance_m", default_value="8.0"),
    ]

    params = {
        "bag": LaunchConfiguration("bag"),
        "out_dir": LaunchConfiguration("out_dir"),
        "force": LaunchConfiguration("force"),
        "overwrite": LaunchConfiguration("overwrite"),
        "odom_topic": LaunchConfiguration("odom_topic"),
        "cloud_topic": LaunchConfiguration("cloud_topic"),
        "voxel_m": LaunchConfiguration("voxel_m"),
        "min_frames": LaunchConfiguration("min_frames"),
        "height_low_m": LaunchConfiguration("height_low_m"),
        "height_high_m": LaunchConfiguration("height_high_m"),
        "resolution_m": LaunchConfiguration("resolution_m"),
        "max_route_distance_m": LaunchConfiguration("max_route_distance_m"),
    }
    # 地面偏移留空 = 自动估计（bag_node 把空串当未设置处理）
    nodes = [
        Node(
            package="tron2_pcd_process",
            executable="pcd_process_bag",
            name="pcd_process_bag",
            output="screen",
            parameters=[params],
        ),
    ]
    return LaunchDescription(args + nodes)
