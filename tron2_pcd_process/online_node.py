#!/usr/bin/env python3
"""在线节点：订阅 FAST-LIO 的 /Odometry + /cloud_registered，边建图边出图。

和离线节点的区别只在"数据从哪来"：核心处理链（体素帧计数 → 地面偏移 →
逐段投影）完全是同一份代码，所以同一份 bag 回放给在线节点和直接离线跑，
产物应当逐位一致（验收判据见 README）。

接口：

    订阅  /Odometry          nav_msgs/Odometry          camera_init -> body
    订阅  /cloud_registered  sensor_msgs/PointCloud2    frame 必须 camera_init
    服务  ~/save             std_srvs/Trigger            立即出图并落盘
    话题  ~/stats            std_msgs/String            每 update_period_s 报一次进度

典型用法（真机建图时挂在这个节点后面，随时按一下保存）：

    ros2 run tron2_pcd_process pcd_process_online --ros-args \
        -p out_dir:=/home/nvidia/maps/run1 -p update_period_s:=10.0

在线节点的三条硬约束（都有理由，别改）：
  1. 地面偏移必须"位姿攒够 N 条之后"才估计，且估出来就冻结 ——
     早期只有几百个点时主峰会跑偏，冻结后整张图的地面基准一致。
  2. 位姿轨迹涨到 max_traj_poses 后开始等距抽稀，防止长时间建图内存线性涨。
  3. 2D 只输出，不做平滑滤波 —— 与离线版保持同一算法，验收才对得上。
"""

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

from . import projection
from .config import DEFAULT_CONFIG, PcdProcessConfig
from .pipeline import (
    RepresentativeAccumulator,
    VoxelAccumulator,
    estimate_ground_offset,
    frame_count_percentiles,
    resolve_min_frames,
)
from .points import Trajectory, write_pcd


class OnlineAccumulator:
    """与 ROS 完全解耦的在线累积状态机（可脱离 rclpy 单测）。"""

    def __init__(self, cfg: PcdProcessConfig, out_dir):
        self.cfg = cfg
        self.out_dir = Path(out_dir).expanduser().resolve()
        self.voxels = VoxelAccumulator(cfg.voxel_m)
        self.reps = RepresentativeAccumulator(cfg.voxel_m)
        self.traj_rows: list = []
        self.cloud_frames = 0
        self.odom_count = 0
        self.ground_offset: Optional[float] = None
        self._ground_locked = False
        self.last_frame_stamp: Optional[float] = None
        self.last_export_s: float = 0.0

    # ---- 输入 ----

    def add_odometry(self, row: np.ndarray) -> None:
        self.traj_rows.append(np.asarray(row, dtype=np.float64).reshape(8))
        self.odom_count += 1
        self._maybe_freeze_ground()

    def add_cloud(self, xyz: np.ndarray, stamp: float) -> None:
        xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
        if self.cfg.max_points_per_frame:
            xyz = xyz[:self.cfg.max_points_per_frame]
        if not len(xyz):
            return
        self.voxels.add_frame(xyz)
        self.reps.add_frame(xyz)
        self.cloud_frames += 1
        self.last_frame_stamp = float(stamp)
        self._maybe_freeze_ground()

    def _maybe_freeze_ground(self) -> None:
        """估计地面偏移，并决定何时锁定。

        实测（scan_0 连续估）：
            101 位姿 → 0.88     301 位姿 → -0.07     1001 位姿 → 1.185
            5744 位姿 → 1.179
        早期估计毫无参考价值，所以：
          * 少于 min_poses / min_voxels 时**不出数**（ground_offset 保持 None）
          * 出数之后仍然每帧重估（数据多了更准），直到位姿数达到最终一半才锁定
          * 锁定后不再变 —— 2D 图必须用同一个地面基准，否则会错层
        """
        if self._ground_locked:
            return
        if self.cfg.ground_offset_m is not None:
            self.ground_offset = float(self.cfg.ground_offset_m)
            self._ground_locked = True
            return
        min_poses, min_voxels = self.cfg.ground_estimate_min_poses, \
            self.cfg.ground_estimate_min_voxels
        if self.odom_count < min_poses or len(self.voxels) < min_voxels:
            return
        traj = self.trajectory()
        offset, _hist = estimate_ground_offset(
            self.reps.means, self.voxels.counts,
            traj.positions[:, :2], traj.positions[:, 2])
        if offset is None:
            return
        self.ground_offset = float(offset)
        # 位姿数达到 2× 门槛后再估一轮就锁定，给足"后续数据修正早期偏差"的余地
        if self.odom_count >= 2 * min_poses:
            self._ground_locked = True

    # ---- 查询 ----

    @property
    def ready(self) -> bool:
        return (self.ground_offset is not None and self.cloud_frames > 0
                and self.odom_count >= 2)

    def trajectory(self) -> Trajectory:
        rows = self.traj_rows
        if len(rows) > self.cfg.max_traj_poses:
            step = max(1, len(rows) // self.cfg.max_traj_poses)
            rows = rows[::step] + [rows[-1]]
        traj = Trajectory(np.vstack(rows))
        if len(traj) > 2 and np.any(np.diff(traj.stamps) < 0):
            traj = Trajectory(traj.rows[np.argsort(traj.stamps, kind="stable")])
        return traj

    def stats(self) -> dict:
        return {
            "odom": int(self.odom_count),
            "frames": int(self.cloud_frames),
            "voxels": int(len(self.voxels)),
            "ground_offset_m": None if self.ground_offset is None
            else round(float(self.ground_offset), 4),
            "last_stamp": self.last_frame_stamp,
        }

    # ---- 出图 ----

    def build(self) -> dict:
        """跑一遍完整处理链，返回产物路径与统计（不要求能写盘）。"""
        if not self.ready:
            raise RuntimeError("数据还不够：需要位姿与点云，且地面偏移可估计")
        cfg = self.cfg
        traj = self.trajectory()
        counts = self.voxels.counts
        means = self.reps.means
        min_frames, threshold_mode = resolve_min_frames(counts, cfg.min_frames)
        keep = counts >= min_frames
        static_xyz = means[keep]
        health = traj.health(cfg.max_step_m, cfg.max_z_span_m, cfg.min_duration_s)
        metrics = {
            "voxel_m": cfg.voxel_m,
            "voxels_total": int(len(means)),
            "static_voxels": int(keep.sum()),
            "static_ratio": round(float(keep.mean()) if len(keep) else 0.0, 4),
            "ground_offset_m": round(float(self.ground_offset), 4),
            "min_frames": int(min_frames),
            "threshold_mode": threshold_mode,
            "frame_count_percentiles": frame_count_percentiles(counts),
            "frame_count_max": int(counts.max()) if len(counts) else 0,
            "health": health,
            "cloud_frames": int(self.cloud_frames),
            "odom_messages": int(self.odom_count),
        }
        result: Optional[projection.ProjectionResult] = None
        if len(static_xyz):
            try:
                result = projection.project_to_grid(
                    static_xyz, traj, self.ground_offset, cfg)
                metrics["map"] = result.metrics()
            except ValueError as exc:
                metrics["map_error"] = str(exc)
        return {"traj": traj, "means": means, "counts": counts, "keep": keep,
                "projection": result, "metrics": metrics}

    def export(self, overwrite: bool = True) -> dict:
        """出图并落盘到 out_dir（产物名与离线节点一致）。"""
        built = self.build()
        out = self.out_dir
        out.mkdir(parents=True, exist_ok=True)
        cfg = self.cfg
        means, counts, keep = built["means"], built["counts"], built["keep"]
        write_pcd(out / "cloud_voxel_all.pcd", means,
                  np.clip(counts, 0, 255).astype(np.float32))
        write_pcd(out / "cloud_static.pcd", means[keep],
                  np.clip(counts[keep], 0, 255).astype(np.float32))
        built["traj"].write_tum(out / "traj_camera_init.tum")
        result = built["projection"]
        if result is None:
            raise RuntimeError("投影失败：" + str(built["metrics"].get("map_error")))
        projection.write_map(result, out / "map")
        summary = {
            "out": str(out),
            "mode": "online",
            "ground_offset_m": round(float(self.ground_offset), 4),
            "cloud": built["metrics"],
            "map": result.metrics(),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (out / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self.last_export_s = time.time()
        return summary


# ------------------------------------------------------------------ ROS 节点

def main(argv=None) -> int:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import PointCloud2
    from std_msgs.msg import String
    from std_srvs.srv import Trigger

    rclpy.init(args=argv)
    node = Node("pcd_process_online")
    defaults = DEFAULT_CONFIG.to_ros_params()
    node.declare_parameter("bag", "")            # 占位：与离线节点共用参数表
    node.declare_parameter("out_dir", ".")
    node.declare_parameter("force", False)
    node.declare_parameter("overwrite", True)
    node.declare_parameter("update_period_s", 10.0)
    node.declare_parameter("save_on_shutdown", True)
    # 这三个在 config 里已有默认值，统一在下面对 defaults 循环里声明一次，
    # 重复声明会抛 ParameterAlreadyDeclaredException（在真实 rclpy 里才暴露）。
    declared = {"bag", "out_dir", "force", "overwrite", "update_period_s",
                "save_on_shutdown"}
    for name, value in defaults.items():
        if name in declared:
            continue
        node.declare_parameter(name, value)

    params = {name: node.get_parameter(name).value for name in defaults}
    cfg = PcdProcessConfig.from_params(params)
    cfg = cfg.replace(
        ground_estimate_min_poses=int(node.get_parameter("ground_estimate_min_poses").value),
        ground_estimate_min_voxels=int(node.get_parameter("ground_estimate_min_voxels").value),
        max_traj_poses=int(node.get_parameter("max_traj_poses").value),
    )
    cfg.validate()
    acc = OnlineAccumulator(cfg, node.get_parameter("out_dir").value)
    log = node.get_logger()

    sensor_qos = QoSProfile(
        depth=20,
        history=QoSHistoryPolicy.KEEP_LAST,
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.VOLATILE,
    )

    def on_odom(msg: Odometry) -> None:
        from .points import OdometrySample
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if msg.header.frame_id and cfg.odom_frame and msg.header.frame_id != cfg.odom_frame:
            if not acc.odom_count:
                log.warn(f"Odometry frame_id={msg.header.frame_id!r} "
                         f"与契约 {cfg.odom_frame!r} 不一致（不阻挡，仅提示）")
        pose = msg.pose.pose
        sample = OdometrySample(
            stamp=stamp,
            position=[pose.position.x, pose.position.y, pose.position.z],
            orientation=[pose.orientation.x, pose.orientation.y,
                         pose.orientation.z, pose.orientation.w],
            frame_id=msg.header.frame_id, child_frame_id=msg.child_frame_id)
        acc.add_odometry(sample.as_row())

    def on_cloud(msg: PointCloud2) -> None:
        xyz = _pointcloud2_xyz(msg)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if not acc.cloud_frames and msg.header.frame_id and cfg.cloud_frame \
                and msg.header.frame_id != cfg.cloud_frame:
            log.error(f"点云 frame_id={msg.header.frame_id!r} 不是世界系 "
                      f"{cfg.cloud_frame!r}：请订阅 /cloud_registered，"
                      f"不要订阅 /cloud_registered_body")
        acc.add_cloud(xyz, stamp)

    node.create_subscription(Odometry, cfg.odom_topic, on_odom, sensor_qos)
    node.create_subscription(PointCloud2, cfg.cloud_topic, on_cloud, sensor_qos)
    stats_pub = node.create_publisher(String, "~/stats", 10)

    def do_save(prefix: str = "save") -> str:
        if not acc.ready:
            return "数据还不够：" + json.dumps(acc.stats(), ensure_ascii=False)
        try:
            summary = acc.export()
        except Exception as exc:  # noqa: BLE001 - 服务里不能把异常抛出去
            log.error(f"{prefix} 失败：{exc}")
            return f"失败：{exc}"
        text = (f"{prefix} ok：{summary['cloud']['static_voxels']} 静态体素，"
                f"障碍 {summary['map']['occupied_cells']} 格，"
                f"可通行 {summary['map']['free_cells']} 格 → {acc.out_dir}")
        log.info(text)
        return text

    def on_save(_request, response: Trigger.Response) -> Trigger.Response:
        text = do_save("save")
        response.success = text.startswith("save ok")
        response.message = text
        return response

    node.create_service(Trigger, "~/save", on_save)

    period = max(1.0, float(node.get_parameter("update_period_s").value))

    def tick() -> None:
        msg = String()
        msg.data = json.dumps(acc.stats(), ensure_ascii=False)
        stats_pub.publish(msg)

    node.create_timer(period, tick)
    log.info(f"pcd_process_online 就绪：{cfg.odom_topic} + {cfg.cloud_topic} → {acc.out_dir}")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if bool(node.get_parameter("save_on_shutdown").value):
            if acc.ready:
                try:
                    do_save("shutdown-save")
                except Exception as exc:  # noqa: BLE001
                    log.error(f"退出保存失败：{exc}")
            else:
                # 数据不足时静默退出会让人误以为"跑了但没产物"，明确说清楚
                log.warn("退出时数据不足以出图，未保存：" +
                         json.dumps(acc.stats(), ensure_ascii=False))
        node.destroy_node()
        rclpy.shutdown()
    return 0


def _pointcloud2_xyz(msg) -> np.ndarray:
    """PointCloud2 → (N,3) float32，按字段描述取 x/y/z，不假设字段布局。"""
    names = [f.name for f in msg.fields]
    if not all(key in names for key in ("x", "y", "z")):
        raise ValueError(f"PointCloud2 缺 x/y/z 字段（有 {names}）")
    n = msg.height * msg.width
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)
    dtype = np.dtype({
        "names": names,
        "formats": [np.dtype("<f4") for _ in names],
        "offsets": [int(f.offset) for f in msg.fields],
        "itemsize": int(msg.point_step),
    })
    points = np.frombuffer(bytes(msg.data), dtype=dtype, count=n)
    return np.column_stack([points["x"], points["y"], points["z"]]).astype(np.float32)


if __name__ == "__main__":
    raise SystemExit(main())
