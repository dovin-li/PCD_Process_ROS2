# Loop Closure（ROS2 离线）

本仓库内置的 FAST-LIO 离线回环模块。它只生成优化前后两份关键帧 TUM 位姿，
再由 `tron2_pcd_process` 对每个世界系点云帧插值施加回环修正、去动态重影并导出 PCD。
不取代 FAST-LIO，也不启动机器人节点。

## 默认算法

1. Mid-360 最近 10 帧融合成关键帧云（对齐 SOLiD 的 Mid-360 配置）。
2. 2 m 里程计近邻优先，SOLiD 描述子补漏，每路最多 3 个候选。
3. Open3D Generalized ICP 验证；overlap、RMSE、位移修正及 30° 描述子航向一致性联合门禁。
4. GTSAM iSAM2 + Cauchy 鲁棒核优化位姿图。

它是“弱漂移时几何近邻稳定工作，漂移超过 2 m 时 SOLiD 尝试补漏”的混合方案。
`--candidate-mode solid|geometry` 可做 A/B 对照。

## 输入契约

ROS2 bag 必须有：

| 话题 | 要求 |
|---|---|
| `/Odometry` | `camera_init -> body`，使用 Header 时间 |
| `/cloud_registered` | `camera_init` 世界系点云，使用 Header 时间 |

点云与里程计不按录包到达顺序硬配对。程序先整理里程计 Header 时间，
再对每帧点云做平移线性插值 + 四元数 slerp。超出里程计时间范围的点云不外推。

## 运行

先在独立虚拟环境安装固定范围的依赖：

```bash
cd PCD_Process_ROS2
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e .
python3 -m pip install -r requirements-loop.txt
```

然后在同一个仓库根目录直接运行（不需要 ROS 运行时）：

```bash
PYTHONPATH=. python3 -m loop_closure.run \
  --bag /bags/scan_ros2 --out /maps/scan/loop
```

colcon 安装后：

```bash
ros2 run tron2_pcd_process loop_closure_bag \
  --bag /bags/scan_ros2 --out /maps/scan/loop
```

默认值已对齐 TRON2 实录：`submap_scans=10`、`min_gap=50`、
`candidate_mode=hybrid`、`proximity_radius=2.0`、`solid_profile=tron2`。
`upstream` 会先撤销当前 Livox bridge 的 Rx(180°) 坐标翻转，用于保守 A/B。

## 输出与 PCD 导出

| 文件 | 含义 |
|---|---|
| `keyscan_odom_poses.txt` | 回环前关键帧 TUM |
| `keyscan_optimized_poses.txt` | iSAM2 优化后同一组 TUM |
| `keyscan_matches.txt` | `query history rmse`，保持三列格式 |
| `loop_diagnostics.jsonl` | 每条接受边的来源、分数、overlap、RMSE 和门禁诊断 |
| `loop_stats.json` | 按来源计数、图优化误差和修正量 |

五个文件会在临时目录全部生成并校验后，再以完整一代替换 `--out`
目录；GICP/GTSAM 中途失败不会覆盖上一代结果。`--out` 应当使用回环结果专用目录。

使用两份 TUM 导出去动态后、保留地面的 XYZI PCD：

```bash
PYTHONPATH=. python3 -m tron2_pcd_process.bag_node \
  --bag /bags/scan_ros2 --out /maps/scan/edit-input --pcd-only \
  --raw-poses /maps/scan/loop/keyscan_odom_poses.txt \
  --pgo-poses /maps/scan/loop/keyscan_optimized_poses.txt
```

`cloud_static.pcd` 中 `intensity` 是独立观测帧数（最高 255）。修正在帧计数之前施加，
因此回环后重合结构会在同一体素累计，而短暂动态重影会被过滤。

## 真实数据验收（2026-11-19）

- 1027 ROS1 走廊：268 关键帧。SOLiD 十帧融合 Top-1 `23/40=57.5%`，
  Top-3 `31/40=77.5%`；最终混合链接受 39 条边（38 geometry + 1 SOLiD）。
- 1119 ROS2 bag：215 关键帧，接受 78 条边（70 geometry + 8 SOLiD）；
  iSAM2 误差 `0.9455 -> 0.1546`，最大平移修正 0.0256 m。

1027 的 Top-1 没有超过 60%，所以不把它宣称为“纯 SOLiD 已达标”；Top-3 候选再经 GICP
是当前交付口径。里程计近邻只是项目代理指标，不是独立物理真值。

## 依赖与测试

必需 Python 包：`numpy`、`open3d`、`gtsam`；ROS1 bag 入口另需 `rosbags`。
Mac 上 GTSAM Python 绑定需使用 NumPy 1.x 环境。
版本范围已写入仓库根目录的 `requirements-loop.txt`，可在独立虚拟环境中安装。

```bash
PYTHONPATH=. python3 -m pytest test/test_backend.py test/test_candidate_modes.py \
  test/test_delivery_contracts.py test/test_mid360_submap.py \
  test/test_solid_frontend.py test/test_time_pairing.py -q
```

当前 28 项单测覆盖：SOLiD 检索/航向、十帧融合、异步时间配对与坐标系门禁、GICP 变换方向、
候选去重/分源诊断、航向一致性门、iSAM2/Cauchy 及公开参数。

SOLiD 纯 Python 核心来自 `sparolab/SOLiD` v2.1.0，固定提交
`867abba490ed00d7d5f7862799bd263f53786e69`，BSD-3-Clause 许可见
`loop_closure/_solid/LICENSE`。
