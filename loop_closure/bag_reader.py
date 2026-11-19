"""Shared ROS2 bag parser from :mod:`tron2_pcd_process`.

Loop closure and PCD export must decode exactly the same sensor timestamps,
frames and XYZ values, so this package deliberately has no second parser.
"""

from tron2_pcd_process.bag_reader import *  # noqa: F401,F403
