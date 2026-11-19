from glob import glob

from setuptools import find_packages, setup

package_name = "tron2_pcd_process"

setup(
    name=package_name,
    version="0.3.0",
    packages=find_packages(exclude=("test", "test.*")),
    package_data={"loop_closure._solid": ["LICENSE"]},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "requirements-loop.txt"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/tools", glob("tools/*.py")),
    ],
    install_requires=["setuptools", "numpy", "scipy", "PyYAML"],
    zip_safe=True,
    maintainer="Dovin Li",
    maintainer_email="dovin.lee@example.com",
    description="Standalone ROS2 PCD processing, offline loop closure, elevation and cost-map export",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "pcd_process_bag = tron2_pcd_process.bag_node:main",
            "pcd_process_online = tron2_pcd_process.online_node:main",
            "pcd_process_ros1_to_ros2 = tron2_pcd_process.ros1_to_ros2:main",
            "loop_closure_bag = loop_closure.run:main",
            "loop_closure_ros1_bag = loop_closure.run_ros1:main",
            "pcd_process_terrain = terrain_tools.terrain_export:main",
        ],
    },
)
