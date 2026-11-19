# gm_chain

`gm_chain` 是 `pcd_process_terrain` 使用的 ROS2 `grid_map` 官方滤波链执行器。
它是独立的 `ament_cmake` 辅助包，不依赖 TRON2 工作区；主仓库仍是一个
`ament_python` 包，因此需要把此目录单独放进 colcon 工作区构建一次。

## 构建

```bash
mkdir -p ~/gm_chain_ws/src
ln -s "$PWD/extras/gm_chain" ~/gm_chain_ws/src/gm_chain
cd ~/gm_chain_ws
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select gm_chain
source install/setup.bash
```

依赖：`grid_map_core`、`grid_map_filters`、`grid_map_cv`、`filters`、`pluginlib`
和 `class_loader`。当前验收环境是 ROS2 Humble；源码和 CMake 不写死 ROS 安装路径。

## 单独运行

```bash
$(ros2 pkg prefix gm_chain)/lib/gm_chain/gm_chain \
  grid.bin layers.bin --ros-args \
  --params-file $(ros2 pkg prefix gm_chain)/share/gm_chain/chain.yaml
```

正常输出固定为五层：`elevation_inpainted`、`slope`、`step`、
`step_large`、`roughness`。Python 入口会校验尺寸、字节数、输入签名、有限值和
高程对齐误差，失败时不会覆盖上一版地图。
