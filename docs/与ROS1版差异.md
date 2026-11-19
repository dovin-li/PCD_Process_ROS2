# 与 ROS1 版 PCD Process 的差异

对照对象：`~/Desktop/测试/point_process`（GitHub `dovin-li/PCD_Process`）。
原则：**算法语义不变**，标出所有有意差异，方便排查"同一份数据两边结果不同"。

## 1. 输入

| 项 | ROS1 版 | ROS2 版 |
|---|---|---|
| 输入 | ROS1 bag（v2.0，none/bz2/lz4） | ROS2 bag（sqlite3 `.db3` + metadata.yaml）；分片支持 |
| 读取方式 | 自写 chunk 解压 + 记录头解析 | sqlite3 标准库直读；`.mcap` 需 `ros2 bag convert` |
| 话题 | `/Odometry` + `/cloud_registered` | 同名，可配 |
| 在线模式 | 无 | **新增**：订阅话题边收边攒，`~/save` 落盘 |

ROS1 版的 `_read_record` 有一个"头长度为 0 → 停止"的修复（防 reindex 后尾部空洞
导致 `KeyError: 'op'`），转换器里保留了同样的判断。

## 2. 核心处理链（语义一致）

| 环节 | 实现 | 差异 |
|---|---|---|
| 体素编码 | `floor(x/res)*grid` 线性化 | 无（同公式、同偏移量） |
| 帧内去重 | 每帧 `np.unique` | 无 |
| 帧计数过滤 | 体素出现 ≥N 帧 | 无（默认 3，同原版固化值） |
| 动态剔除 | 靠帧计数 | 无 |
| 地面偏移 | 轨迹 1 m 格内相对高度主峰 + 中值精修 | 无 |
| 轨迹分段 | 1.0 m 弧长 / 0.10 m 高差 | 无 |
| 高度窗分类 | kNN 到段参考 + 距离容差门控 | 无 |
| 射线自由空间填充 | 从段位姿向点连线，遇障碍停 | 无 |
| PGM 编码 | 0 障碍 / 205 未知 / 254 通行 | 无 |

**实现层差异（不改结果，只改速度/内存）**

1. **流式累积取代"全量拼接后排序"**
   ROS1 版把所有帧的体素键 `np.concatenate` 后统一 `np.unique`，
   内存峰值 = 全部帧的点数（约 2400 万键）；ROS2 版按帧增量合并，
   内存只随体素数增长。这是 22 个体素差异的来源，见第 4 节。
2. **代表点用预分配数组 + dict 索引**
   最初写成每帧 `np.concatenate` 重建大数组，240 万体素时 5744 帧要 153 s；
   改成预分配 + `dict.update` 批量建索引后 14.6 s。
   踩坑记录：用 `np.add.at` 或 Python 逐元素循环都会退化到分钟级。
3. **2D 投影分块做 kNN**
   原版按 50 万点分块；新版按 20 万点分块，避免 110 万点 × k=16 一次性爆内存。

## 3. 新增能力

- **在线节点**（`pcd_process_online`）：与离线上限同构，`~/save` 立即出图，
  退出时自动落盘；地面偏移在攒够位姿后估计并**冻结**（早期点少主峰会跑偏）。
- **ROS1 → ROS2 转换器**（`tools/ros1_to_ros2_bag.py`）：把历史 bag 转成能
  `ros2 bag play` 的标准 ROS2 bag。
- **ROS2 参数化**：所有参数既可从命令行给，也可从 ROS 参数表给（同一份 `PcdProcessConfig`）。

## 4. 已知数值差异（scan_0 实测）

| 指标 | ROS1 | ROS2 | 差异 | 原因 |
|---|---|---|---|---|
| 体素总数 | 2,422,823 | 2,422,824 | +1 | 见下 |
| 静态体素 | 748,863 | 748,867 | +4 | 同上，级联到多帧阈值 |
| 障碍格 | 138,542 | 138,551 | +9 | 同上，级联到投影 |
| 地面偏移 | 1.1794 | 1.1792 | 2e-4 | 体素集合微差 |
| 轨迹 | — | — | **0** | 位姿解析逐位一致 |
| 2D 栅格尺寸 | 3417×1212 | 3417×1212 | 0 | — |

**+1 个体素怎么来的**：ROS1 版先把 2422 万个体素键拼成大数组再 `np.unique`，
float32 坐标在边界附近算出的键会有一对一"碰撞"（不同点算成同一个键）；
ROS2 版逐帧去重后合并，得到的键数恰好等于"逐帧 unique 的并集"，
即没有碰撞损失的一方。差集只有 22 个键，坐标平均差 5.8e-7 m，
最大 2.5 cm（都是体素边界点）。

结论：**差 1 个体素属于精度改善，不是功能回归**。要复现原版的碰撞行为反而需要
刻意退化实现，没有意义。

## 5. 转换器的技术细节（踩坑记录）

ROS1 线上序列化有两种真实存在的布局，转换器用数学判据自动判定：

| 布局 | 对齐 | 字符串长度前缀 |
|---|---|---|
| aligned | 字段按自身宽度对齐（官方 roscpp/rospy） | 不含 NUL |
| native | 完全按写端内存布局，不对齐（msg_generation 等） | 含 NUL |

`scan_0.bag` 实测是 **native + 不含 NUL**。判定方法：枚举 4 种组合做完整探测，
只有一种能读完每条消息，且读出 `point_step` 为小整数、`row_step == width*point_step`。

其它必须记住的点：

1. **Header.seq 要整条跳过**（ROS2 已删除），且跳过之后不能在字段循环里再读一次。
2. **PointCloud2.data 在 ROS1 里带 uint32 长度前缀**，不是"读到文件尾"。
3. **变长数组长度前缀按 4 字节对齐**，但 uint8 数组的元素本身按 1 字节对齐。
4. **datatype 7 = FLOAT32**（不是 int64）。这个映射写错会让点云解析出 1e19 的荒谬值。
5. CDR 写端的对齐基准是**封装头之后**，读端 `offset` 也按"相对封装头之后"记。
6. 封装头第 1 字节的最低位是 CDR 表示标志（0=大端，1=小端）。

## 6. Docker / 真实 ROS2 环境验证（2026-11-19，Humble ros-base）

在 `ros:humble-ros-base` 容器里 colcon build + 单测 + `ros2 run` + `ros2 launch` + 在线回放全跑通。
**下面这些坑只有在真实 rclpy 下才暴露**，纯 Python 单测发现不了，记下来避免重犯：

1. **参数不能重复声明**（`ParameterAlreadyDeclaredException`）。
   离线/在线节点都有"通用字段已声明 + 循环里又声明一次"的重复；
   修法：用一个 `seen` 集合，循环里跳过已声明过的名字。
2. **`None` 默认值会触发 rclpy 告警**（"declaring a parameter only providing its name"）。
   - `ParameterDescriptor` 在 Humble 里位于 `rcl_interfaces.msg`，不在 `rclpy` 顶层，
     也不在 `rclpy.parameter`；
   - 但给它 `type_=` 传 `Parameter.Type.DOUBLE` 会报 "Invalid arguments ... type_"；
   - **最终方案：用 `float("nan")` 当"未设置"哨兵**，读回时 NaN→None。简单可靠。
3. **metadata.yaml 必须严格符合 rosbag2 schema**，否则 `ros2 bag play` 报
   `invalid node; first invalid key: "starting_time"`：
   - `duration` / `starting_time` 都是**嵌套结构** `{nanoseconds: N}` / `{nanoseconds_since_epoch: N}`；
   - `files[]` 里同样用嵌套 `duration` / `starting_time`，不是 `duration_ns` 之类的扁平键；
   - 还必须有 `custom_data: {}`、`compression_format`、`compression_mode`。
4. **`ros2 bag play` 的 stdout 走管道会被提前 EOF 杀掉**，测试时要重定向到文件再 tail。
5. 容器里 `ros:humble-ros-base` **没有 scipy**，需 `apt install python3-scipy`。

**容器内实测数字**（全量 5744 帧，与 Mac 完全一致）：
静态体素 748,867 / 障碍格 138,551 / 可通行 390,320 / 地面偏移 1.1792。
在线模式经 `ros2 bag play` 回放验证：地面偏移 1.1869、`~/save` 出图成功。

## 7. T14 实测（2026-11-19）

环境：T14（x86_64，Ubuntu 22.04）+ 容器 `tron2-sim:amd64`（ROS2 Humble，`apt install python3-scipy` 后可用）。
测试目录 `~/pcd_process_test/ws`（独立 workspace，**不影响**正在运行的 `cmu_go2_sim` 仿真栈）。

| 项 | 结果 |
|---|---|
| `colcon build` | 通过，`colcon list` = `(ros.ament_python)` |
| 单测（pytest） | 31 passed |
| 转换 T14 本地 `scan_0.bag`（ROS1 → ROS2） | 5744 + 5744 帧，2.3 s（bag 是 bz2 压缩，正常解压） |
| ROS2 离线全量处理 | **13.4 s**，静态 748,867 / 障碍 138,551 / 通行 390,320 / 地面 1.1792 |
| 与 Mac 结果对比 | **逐位一致** |
| 在线模式（`ros2 bag play` 回放） | 订阅 1306 帧、地面 1.1822、`~/save` 出图成功 |
| 错误输入（缺 `/Odometry` 的 bag） | 明确报错并指出缺哪个话题，不再抛 traceback |

T14 机器本身比 Mac 和 Docker Desktop 快：全量处理 13.4 s（Mac 18~35 s，Docker 29~49 s）。

**注意**：同机的 `smith_hall_2nd_floor.db3` 只有 `/utlidar/cloud` + `/utlidar/imu`，
不含 `/Odometry`/`/cloud_registered`，**不能**直接喂给本包（会按契约拒绝）。

## 8. 交付前自检发现的入口点问题（2026-11-19）

`setup.py` 里注册了三个 console_scripts，但**两个只有单一来源**，靠真实环境才能发现：

| 入口 | 问题 | 修复 |
|---|---|---|
| `pcd_process_bag` | 正常 | — |
| `pcd_process_online` | 正常 | — |
| `pcd_process_ros1_to_ros2` | **指向了不存在的 `tron2_pcd_process/ros1_to_ros2.py`** | 新增该模块做薄转发，按 `share/tools/` → 源码 `tools/` 两级查找 |

另：`tron2_pcd_process/__init__.py` 原本只有 docstring，`from tron2_pcd_process import
PcdProcessConfig` 会失败；已补上公开 API 重导出。

这两条都加了回归测试（`test_console_script_wrapper_resolves_tools`、
`test_all_console_scripts_resolve`），**改 `setup.py` 的入口点必跑**。
教训：console_scripts 必须有测试覆盖——它在纯 Python 单测里永远绿，只有
`ros2 run <executable>` 时才会露出来。
