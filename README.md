# PCD Process ROS2

**ROS2 版 PCD Process。** 把 FAST-LIO 的 world 系点云 + 位姿，处理成 **静态 3D 点云 PCD** 和 **2D 占据栅格地图**。

> 本仓库是 [PCD_Process](https://github.com/dovin-li/PCD_Process)（ROS1 版）的 **ROS2 重写**，
> 算法语义逐项对齐，输入换成 ROS2 bag，并新增在线模式。
> ROS1 版继续维护在 `PCD_Process`，两者互不依赖。

| 对比 | `PCD_Process`（ROS1） | **本仓库（ROS2）** |
|---|---|---|
| 输入 | ROS1 `.bag` | ROS2 `.db3` + `metadata.yaml` |
| 运行环境 | 不需要 ROS（纯 Python） | 离线不需要 ROS；在线需要 ROS2 Humble |
| 在线模式 | 无 | **有**（订阅话题边建图边出图） |
| ROS1→ROS2 转换 | 无 | **有**（`tools/ros1_to_ros2_bag.py`） |
| 处理算法 | 体素去重 + 帧计数过滤 + 逐段投影 | **完全相同**（已验证逐像素一致） |

目录是标准 ROS2 包（`tron2_pcd_process/`），可整体放进任何 colcon workspace 的 `src/` 下编译；
也可以不编译，直接用 `PYTHONPATH=.` 跑（离线模式不需要 ROS）。

---

## 0. 它解决什么问题

建图期间有人走动，会在点云里留下"人形重影"。这个包的去动态机制是**帧计数过滤**：

> 同一个 5 cm 体素至少被 **3 帧**观测到才算结构；人走过只留 1~2 帧，整批剔除。

不需要回环位姿、不需要后处理调参。实测 3.7 GB / 10,527 帧的走廊数据，
体素 3,440,574 → 1,133,611（剔除 67%），耗时 17.5 s。

## 1. 三条链路

```
                  ┌─────────────────────────── 离线（bag）───────────────────────────┐
ROS2 bag (.db3) ──┤ sqlite3 直读 → 体素去重 → 帧计数过滤 → 地面偏移 → 逐段投影 → 出图  ├──► PCD / PGM / YAML
                  └──────────────────────────────────────────────────────────────────┘

                  ┌─────────────────────────── 在线（订阅）───────────────────────────┐
/Odometry ────────┤ 同一份核心代码，边收边攒，随时 ~/save 落盘                          ├──► 同款产物
/cloud_registered ┘                                                                  ┘
```

三条硬结论：

1. **核心处理链不 import rclpy**（`pipeline.py` / `projection.py` / `points.py`），
   所以能在 Mac 上直接跑、能单测、离线在线共用一份代码。
2. **离线节点用 sqlite3 直读 .db3**（标准库），不需要 ROS 运行时也能读 bag。
3. **在线与离线结果应当一致**：同一份数据回放给在线节点，产物应与离线逐位相同。

## 2. 快速开始

### 2.1 离线处理一份 ROS2 bag

```bash
# 方式 A：纯 Python（不需要 source ROS，Mac / 机器人容器都行）
cd src/tron2_pcd_process
PYTHONPATH=. python3 -m tron2_pcd_process.bag_node --bag /bags/scan_0 --out /tmp/out/scan_0

# 方式 B：ROS2 入口（需要 colcon build + source install/setup.bash）
ros2 run tron2_pcd_process pcd_process_bag --ros-args -p bag:=/bags/scan_0 -p out_dir:=/tmp/out/scan_0

# 方式 C：launch
ros2 launch tron2_pcd_process bag_to_map.launch.py bag:=/bags/scan_0 out_dir:=/tmp/out/scan_0
```

### 2.2 在线边跑边出图

```bash
ros2 launch tron2_pcd_process online_mapping.launch.py out_dir:=/home/nvidia/maps/run1

# 随时手动落盘（std_srvs/Trigger）
ros2 service call /pcd_process_online/save std_srvs/srv/Trigger
# Ctrl-C 退出时默认也会自动落盘（save_on_shutdown:=true）
```

### 2.3 历史 ROS1 数据转 ROS2

```bash
# 转换器在 tools/，可在任意机器上跑（不需要 ROS1/ROS2）
PYTHONPATH=tools python3 tools/ros1_to_ros2_bag.py \
    --bag ~/Desktop/1030/scan_0.bag --out /tmp/bags/scan_0_ros2
```

转换器做的事：ROS1 bag 解析 → 消息经 CDR 重编码 → 写 sqlite3 `.db3` + `metadata.yaml`。
产物可以用 `ros2 bag play` 回放，也可以直接喂给 `pcd_process_bag`。

## 3. 输入契约（缺一不可）

| 话题 | 类型 | 要求 |
|---|---|---|
| `/Odometry` | `nav_msgs/msg/Odometry` | `frame_id=camera_init`，`child_frame_id=body` |
| `/cloud_registered` | `sensor_msgs/msg/PointCloud2` | `frame_id` 必须是 **`camera_init`**（世界系） |

**最容易踩的一条**：订阅 `/cloud_registered_body` 不会报错，但整张图会错位——
那个话题是 body 系的。离线节点不强制校验 frame（历史数据可能不规范），
在线节点会打 ERROR 日志。

FAST-LIO 侧两个必须的设置（建图前改，导航时改回）：

```yaml
publish:
    linear_update:
    dense_publish_en: true     # ← 建图必须 true，否则点云被抽稀、墙面偏细
pcd_save:
    pcd_save_en: false         # ← 本包吃 bag，不需要 FAST-LIO 自己落盘
```

## 4. 产物

| 文件 | 内容 |
|---|---|
| `cloud_voxel_all.pcd` | 全部 5 cm 体素，`intensity` = 观测帧数（想换阈值不用重跑） |
| `cloud_static.pcd` | 帧计数过滤后的静态点云（送进投影的就是它） |
| `cloud_noground.pcd` | 再去掉地面的点云（**推荐先看这个**） |
| `traj_camera_init.tum` | 轨迹，TUM 格式 |
| `map.pgm` / `map.yaml` | 2D 地图：**0=障碍 205=未知 254=可通行**，可直接给 `map_server` |
| `metrics.json` | 自动估计值、帧数分布、阈值 |
| `map_metrics.json` | 投影参数、逐段地面基准明细 |
| `summary.json` | 本次运行汇总 |
| `report.md` | 人读的体检结论（先看这个） |

## 5. 参数

权威默认值在 `tron2_pcd_process/config.py` 的 `PcdProcessConfig`；
速查表与"现场最常见两件事"见 `config/params_reference.yaml`。

固化参数（与原 ROS1 版一致）：

| 参数 | 默认 | 说明 |
|---|---|---|
| 体素 | 0.05 m | 去重粒度，与 2D 图分辨率一致 |
| 帧数阈值 | **3** | 2=杂点多，4=干净但丢矮结构 |
| 障碍窗 | 0.10 ~ 1.00 m | 相对当地地面；低于下沿算地面，高于上沿算桌面/横梁 |
| 栅格分辨率 | 0.05 m | 与 `map_server` 契约一致 |
| 轨迹分段 | 1.0 m 弧长 / 0.10 m 高差 | 坡道上每段换一次地面基准 |
| 轨迹距离门限 | 8 m | 更远的点丢弃，防远场飞点撑大地图 |
| 地面偏移 | 自动估计 | 雷达光心到地面高度，本项目实测约 **1.17~1.18 m** |

## 6. 轨迹体检（不通过就拒绝出图）

| 门限 | 健康值参考（scan_0 实测） |
|---|---|
| 相邻位姿最大位移 ≤ 1.0 m | 0.061 m |
| 全程高差 ≤ 5.0 m | 3.30 m |
| 时长 ≥ 10 s | 293 s |

里程计发散（物理冲击导致 z 积分到 −173 km 那种）会被拦下，`--force` 可强制出图。

## 7. 验收基线与对照

2026-11-19 用 `~/Desktop/1030/scan_0.bag`（2.0 GB / 5744 帧 / 293 s）做了 ROS1 版与 ROS2 版逐项对照：

| 指标 | ROS1 版 | ROS2 版 | 差异 |
|---|---|---|---|
| 点云帧数 | 5744 | 5744 | 0 |
| 体素总数 | 2,422,823 | 2,422,824 | +1 |
| 静态体素 | 748,863 | 748,867 | +4 |
| 障碍格 | 138,542 | 138,551 | +9 |
| 2D 栅格 | 3417×1212 | 3417×1212 | 0 |
| 地面偏移 | 1.1794 | 1.1792 | 2e-4 |
| 帧计数中位/90/99 分位 | 1/7/19 | 1/7/19 | 0 |
| 轨迹 | — | 逐位与 ROS1 一致（max diff 0.0） | 0 |
| 耗时 | — | 18~35 s（视机器负载） | — |

差异来源：ROS1 版把 2422 万个体素键**全部拼起来再 unique**，边界体素在 float32
精度下会碰撞；ROS2 版按帧流式去重。ROS2 版得到的体素数恰好等于
"逐帧 unique 的并集"，即更准确的一方。

## 8. 测试

```bash
cd src/tron2_pcd_process
PYTHONPATH=. python3 test/test_pipeline.py     # 无 pytest 也能跑
PYTHONPATH=. python3 -m pytest test -q          # 有 pytest 时
```

覆盖：体素编码、帧计数过滤（动态点剔除）、流式累积与批量结果一致、
地面偏移估计（合成 1.2 m 地面）、Otsu 钳制、PCD 往返、轨迹体检、NaN 剔除、参数校验。

## 9. 已知限制

1. 只支持 sqlite3（`.db3`）存储。`.mcap` 需要先 `ros2 bag convert` 或装 `rosbag2_py`。
2. 没有 `/Odometry` 的 bag 无法处理——没有轨迹就没有逐段地面基准。
3. 帧计数过滤只能去掉"停留时间短"的动态物；长时间静止的人需要人工清理。
4. 2D 图表达不了重叠楼层，跨层数据要按层分段建图。
5. 本版**没有**轨迹走廊清理：机器人走过的地方若残留点在障碍窗内会被标成障碍。
6. 在线节点长期运行时轨迹会抽稀（`max_traj_poses`），超过约 20 万位姿后分辨率下降。

## 10. 与 ROS1 版的关系

- ROS1 版 `~/Desktop/测试/point_process` **保持不动**，仍是历史数据的参考实现。
- 本包是它的 ROS2 重写 + 在线模式扩展；算法改动会同步回 ROS1 版的文档。
- 逐项差异清单见 `docs/与ROS1版差异.md`。
