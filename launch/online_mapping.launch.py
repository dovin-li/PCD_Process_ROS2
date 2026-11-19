"""在线处理 launch：订阅 FAST-LIO，边建图边出图。

    ros2 launch tron2_pcd_process online_mapping.launch.py \
        out_dir:=/home/nvidia/maps/run1

三条前置条件（缺一不可，都在 FAST-LIO 侧）：
  1. /cloud_registered 的 frame_id 必须是 camera_init（世界系）；
     订阅 /cloud_registered_body 会让整张图错位。
  2. dense_publish_en: true，否则点云被抽稀、墙面偏细。
  3. 起本节点之前先确认 /Odometry 有数据（雷达 + IMU 正常）。
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("out_dir", description="地图输出目录"),
        DeclareLaunchArgument("odom_topic", default_value="/Odometry"),
        DeclareLaunchArgument("cloud_topic", default_value="/cloud_registered"),
        DeclareLaunchArgument("update_period_s", default_value="10.0"),
        DeclareLaunchArgument("save_on_shutdown", default_value="true"),
        DeclareLaunchArgument("voxel_m", default_value="0.05"),
        DeclareLaunchArgument("min_frames", default_value="3"),
        DeclareLaunchArgument("height_low_m", default_value="0.10"),
        DeclareLaunchArgument("height_high_m", default_value="1.00"),
        DeclareLaunchArgument("resolution_m", default_value="0.05"),
        DeclareLaunchArgument("ground_estimate_min_poses", default_value="1000"),
        DeclareLaunchArgument("ground_estimate_min_voxels", default_value="300000"),
        Node(
            package="tron2_pcd_process",
            executable="pcd_process_online",
            name="pcd_process_online",
            output="screen",
            parameters=[{
                "out_dir": LaunchConfiguration("out_dir"),
                "odom_topic": LaunchConfiguration("odom_topic"),
                "cloud_topic": LaunchConfiguration("cloud_topic"),
                "update_period_s": LaunchConfiguration("update_period_s"),
                "save_on_shutdown": LaunchConfiguration("save_on_shutdown"),
                "voxel_m": LaunchConfiguration("voxel_m"),
                "min_frames": LaunchConfiguration("min_frames"),
                "height_low_m": LaunchConfiguration("height_low_m"),
                "height_high_m": LaunchConfiguration("height_high_m"),
                "resolution_m": LaunchConfiguration("resolution_m"),
                "ground_estimate_min_poses":
                    LaunchConfiguration("ground_estimate_min_poses"),
                "ground_estimate_min_voxels":
                    LaunchConfiguration("ground_estimate_min_voxels"),
            }],
        ),
    ])
