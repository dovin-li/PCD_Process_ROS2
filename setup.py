from glob import glob

from setuptools import setup

package_name = "tron2_pcd_process"

setup(
    name=package_name,
    version="0.2.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/tools", glob("tools/*.py")),
    ],
    install_requires=["setuptools", "numpy", "scipy"],
    zip_safe=True,
    maintainer="Dovin Li",
    maintainer_email="dovin.lee@example.com",
    description="ROS2 port of PCD Process: FAST-LIO bag -> static PCD + 2D occupancy map",
    license="Proprietary",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "pcd_process_bag = tron2_pcd_process.bag_node:main",
            "pcd_process_online = tron2_pcd_process.online_node:main",
            "pcd_process_ros1_to_ros2 = tron2_pcd_process.ros1_to_ros2:main",
        ],
    },
)
