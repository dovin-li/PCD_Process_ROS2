#!/usr/bin/env python3
"""离线节点：ROS2 bag → 静态点云 PCD + 2D 占据栅格地图（一条命令）。

等价于 ROS1 版 `./bin/pcd-map <bag> <out>`，只是输入换成 ROS2 bag。

两种运行方式：

  1) 纯 Python（不需要 ROS，Mac 上就能跑）：
        python3 -m tron2_pcd_process.bag_node --bag run --out out/run1
        # 或装好的入口：
        pcd_process_bag --bag run --out out/run1

  2) ROS2 节点（参数与 1) 同名同义）：
        ros2 run tron2_pcd_process pcd_process_bag --ros-args \
            -p bag:=/bags/scan_0 -p out_dir:=/tmp/out/scan_0

产物（与 ROS1 版逐项对齐）：

    cloud_voxel_all.pcd   全部 5 cm 体素，intensity = 观测帧数
    cloud_static.pcd      帧计数过滤后的静态点云（送进投影的就是它）
    cloud_noground.pcd    额外去地面后的点云
    traj_camera_init.tum  轨迹
    map.pgm / map.yaml    2D 地图（0=障碍 205=未知 254=可通行）
    metrics.json          自动估计值、帧数分布
    map_metrics.json      投影参数与逐段地面基准
    summary.json          本次运行汇总
    report.md             人读的体检结论
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from . import projection
from .bag_reader import iter_bag, list_topics, resolve_storage
from .config import DEFAULT_CONFIG, PcdProcessConfig
from .pipeline import (
    RepresentativeAccumulator,
    VoxelAccumulator,
    estimate_ground_offset,
    frame_count_percentiles,
    resolve_min_frames,
)
from .points import Trajectory, write_pcd


def _add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--bag", required=True, help="ROS2 bag 目录或 .db3 文件")
    parser.add_argument("--out", "--out-dir", dest="out", required=True, help="输出目录")
    parser.add_argument("--force", action="store_true", help="跳过轨迹体检（发散数据也出图）")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已存在的输出")
    parser.add_argument("--odom-topic", default=DEFAULT_CONFIG.odom_topic)
    parser.add_argument("--cloud-topic", default=DEFAULT_CONFIG.cloud_topic)
    parser.add_argument("--voxel", type=float, default=DEFAULT_CONFIG.voxel_m)
    parser.add_argument("--min-frames", type=int, default=DEFAULT_CONFIG.min_frames)
    parser.add_argument("--ground-offset", type=float, default=None,
                        help="雷达光心到地面高度；省略则自动估计")
    parser.add_argument("--resolution", type=float, default=DEFAULT_CONFIG.resolution_m)
    parser.add_argument("--height-low", type=float, default=DEFAULT_CONFIG.height_low_m)
    parser.add_argument("--height-high", type=float, default=DEFAULT_CONFIG.height_high_m)
    parser.add_argument("--max-route-distance", type=float,
                        default=DEFAULT_CONFIG.max_route_distance_m)
    parser.add_argument("--max-points-per-frame", type=int, default=0)
    parser.add_argument("--no-raycast", dest="raycast_free", action="store_true",
                        default=DEFAULT_CONFIG.raycast_free)
    parser.add_argument("--max-step-m", type=float, default=DEFAULT_CONFIG.max_step_m)
    parser.add_argument("--max-z-span-m", type=float, default=DEFAULT_CONFIG.max_z_span_m)
    parser.add_argument("--min-duration-s", type=float, default=DEFAULT_CONFIG.min_duration_s)
    return parser


def _config_from_args(args) -> PcdProcessConfig:
    cfg = DEFAULT_CONFIG.replace(
        odom_topic=args.odom_topic,
        cloud_topic=args.cloud_topic,
        voxel_m=args.voxel,
        min_frames=args.min_frames,
        ground_offset_m=args.ground_offset,
        resolution_m=args.resolution,
        height_low_m=args.height_low,
        height_high_m=args.height_high,
        max_route_distance_m=args.max_route_distance,
        max_points_per_frame=args.max_points_per_frame,
        raycast_free=args.raycast_free,
        max_step_m=args.max_step_m,
        max_z_span_m=args.max_z_span_m,
        min_duration_s=args.min_duration_s,
    )
    cfg.validate()
    return cfg


# ------------------------------------------------------------------ 主流程

def run(bag, out, cfg: PcdProcessConfig, force: bool = False,
        overwrite: bool = False, log=print) -> dict:
    started = time.time()
    bag_path = Path(bag).expanduser()
    files, _meta = resolve_storage(bag_path)
    if not files:
        raise SystemExit(f"{bag_path}: 不是有效的 ROS2 bag")
    out = Path(out).expanduser().resolve()
    expected = [out / name for name in
                ("summary.json", "cloud_static.pcd", "map.pgm", "map.yaml")]
    if not overwrite:
        existing = [str(p) for p in expected if p.exists()]
        if existing:
            raise SystemExit(f"输出已存在：{existing}（加 --overwrite 覆盖）")
    out.mkdir(parents=True, exist_ok=True)

    log(f"== 1/4 读取 bag（{len(files)} 个存储文件）==")
    voxels = VoxelAccumulator(cfg.voxel_m)
    rep = RepresentativeAccumulator(cfg.voxel_m)
    traj_rows: List[np.ndarray] = []
    frame_stamps: List[float] = []
    cloud_frames = 0
    cloud_frame_id = ""
    try:
        stream = iter_bag(bag_path, cfg.odom_topic, cfg.cloud_topic)
        first = next(iter(stream))
    except KeyError as exc:
        raise SystemExit(f"输入不满足契约：{exc.args[0]}\n"
                         f"本包需要 {cfg.odom_topic}（FAST-LIO 位姿）与 "
                         f"{cfg.cloud_topic}（世界系点云）两个话题。")
    except TypeError as exc:
        raise SystemExit(f"话题类型不符合契约：{exc}")
    except StopIteration:
        raise SystemExit(f"bag 里没有可用的 {cfg.odom_topic}/{cfg.cloud_topic} 消息")

    def chain(first_item, rest):
        yield first_item
        yield from rest

    for key, _stamp, obj in chain(first, stream):
        if key == "odom":
            traj_rows.append(obj.as_row())
        else:
            xyz = obj.xyz
            if cfg.max_points_per_frame:
                xyz = xyz[:cfg.max_points_per_frame]
            cloud_frames += 1
            cloud_frame_id = cloud_frame_id or obj.frame_id
            frame_stamps.append(obj.stamp)
            voxels.add_frame(xyz)
            rep.add_frame(xyz)
    if not traj_rows or not cloud_frames:
        raise SystemExit("bag 里没有可用的 Odometry 或点云消息")

    traj = Trajectory(np.vstack(traj_rows))
    if np.any(np.diff(traj.stamps) < 0):
        traj = Trajectory(traj.rows[np.argsort(traj.stamps, kind="stable")])
    means = rep.means
    counts = voxels.counts
    rep_keys = rep.keys
    if len(rep_keys) != len(voxels.keys) or not np.array_equal(rep_keys, voxels.keys):
        raise SystemExit("内部错误：体素集合与代表点集合不一致")

    log(f"   odom {len(traj)} 条 / 点云 {cloud_frames} 帧 / "
        f"体素 {len(voxels)}（frame_id={cloud_frame_id or '?'}）")

    # ---- 地面偏移 ----
    if cfg.ground_offset_m is None:
        ground_offset, ground_hist = estimate_ground_offset(
            means, counts, traj.positions[:, :2], traj.positions[:, 2])
        ground_source = "auto"
    else:
        ground_offset, ground_hist = float(cfg.ground_offset_m), None
        ground_source = "argument"
    if ground_offset is None:
        raise SystemExit("地面偏移未能自动估计；请用 --ground-offset 手工给值")
    log(f"   地面偏移 {ground_offset:.4f} m（{ground_source}）")

    min_frames, threshold_mode = resolve_min_frames(counts, cfg.min_frames)
    keep = counts >= min_frames

    # ---- 产物 1：PCD / TUM ----
    log("== 2/4 写出点云与轨迹 ==")
    write_pcd(out / "cloud_voxel_all.pcd", means,
              np.clip(counts, 0, 255).astype(np.float32))
    write_pcd(out / "cloud_static.pcd", means[keep],
              np.clip(counts[keep], 0, 255).astype(np.float32))
    traj.write_tum(out / "traj_camera_init.tum")

    noground_stats = None
    static_xyz = means[keep]
    if len(static_xyz):
        try:
            from scipy.spatial import cKDTree
            tree_traj = cKDTree(traj.positions[:, :2])
            _dist, nearest = tree_traj.query(static_xyz[:, :2], k=1)
            local_ground = traj.positions[nearest, 2] - ground_offset
            above = (static_xyz[:, 2] - local_ground) >= cfg.height_low_m
            write_pcd(out / "cloud_noground.pcd", static_xyz[above],
                      np.clip(counts[keep][above], 0, 255).astype(np.float32))
            noground_stats = {"points": int(above.sum()),
                              "removed": int((~above).sum()),
                              "height_m": cfg.height_low_m}
        except ImportError:  # pragma: no cover - scipy 是硬依赖，这里只是兜底
            noground_stats = None

    # ---- 轨迹体检 ----
    log("== 3/4 轨迹体检 ==")
    health = traj.health(cfg.max_step_m, cfg.max_z_span_m, cfg.min_duration_s)
    log("   " + json.dumps(health, ensure_ascii=False))
    if not health["ok"] and not force:
        (out / "report.md").write_text(
            "# 拒绝出图\n\n" + "\n".join(f"- {p}" for p in health["problems"]) + "\n",
            encoding="utf-8")
        raise SystemExit("轨迹体检未通过，已停止（加 --force 可强制出图）：\n  " +
                         "\n  ".join(health["problems"]))

    metrics = simplify_metrics(
        bag=bag_path, traj=traj, counts=counts, means_count=int(len(means)),
        cloud_frames=cloud_frames, cloud_frame_id=cloud_frame_id,
        cfg=cfg, ground_offset=ground_offset, ground_hist=ground_hist,
        min_frames=min_frames, threshold_mode=threshold_mode, keep=keep,
        noground_stats=noground_stats, started=started,
    )

    # ---- 投影 ----
    log("== 4/4 投影 2D 地图 ==")
    segments = projection.build_segments(traj, cfg)
    result = projection.project_to_grid(static_xyz, traj, ground_offset, cfg,
                                       segments=segments)
    projection.write_map(result, out / "map")
    map_metrics = {"pcd": str(out / "cloud_static.pcd"),
                   "trajectory": str(out / "traj_camera_init.tum"),
                   **result.metrics()}
    (out / "map_metrics.json").write_text(
        json.dumps(map_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = {
        "bag": str(bag_path),
        "out": str(out),
        "storage_files": [str(f) for f in files],
        "fixed_params": {
            "voxel_m": cfg.voxel_m,
            "height_window_m": [cfg.height_low_m, cfg.height_high_m],
            "grid_resolution_m": cfg.resolution_m,
            "min_frames": cfg.min_frames if cfg.min_frames else "auto(otsu,2..4)",
            "max_route_distance_m": cfg.max_route_distance_m,
        },
        "auto_estimated": {
            "ground_offset_m": round(float(ground_offset), 4),
            "min_frames": int(min_frames),
            "threshold_mode": threshold_mode,
        },
        "health": health,
        "cloud": {
            "frames": cloud_frames,
            "voxels_total": int(len(means)),
            "static_voxels": int(keep.sum()),
            "static_ratio": round(float(keep.mean()) if len(keep) else 0.0, 4),
        },
        "map": {
            "occupied_cells": map_metrics["occupied_cells"],
            "free_cells": map_metrics["free_cells"],
            "grid": [map_metrics["grid_width"], map_metrics["grid_height"]],
            "pgm": str(out / "map.pgm"),
            "yaml": str(out / "map.yaml"),
        },
        "outputs": {
            "pcd_all": str(out / "cloud_voxel_all.pcd"),
            "pcd_static": str(out / "cloud_static.pcd"),
            "pcd_noground": str(out / "cloud_noground.pcd"),
            "traj": str(out / "traj_camera_init.tum"),
            "map_pgm": str(out / "map.pgm"),
            "map_yaml": str(out / "map.yaml"),
        },
        "elapsed_s": round(time.time() - started, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    (out / "report.md").write_text(
        "# ROS2 bag → PCD + 2D 地图\n\n"
        f"- bag：`{bag_path.name}`（{len(files)} 个存储文件）\n"
        f"- 帧数：{cloud_frames}，时长 {health['duration_s']} s\n"
        f"- 自动地面偏移：**{ground_offset:.3f} m**（{ground_source}）\n"
        f"- 帧数阈值：{min_frames}（{threshold_mode}）\n"
        f"- 体素：全部 {len(means)}，保留 {int(keep.sum())}"
        f"（static_ratio {summary['cloud']['static_ratio']}）\n"
        f"- 2D：障碍 {map_metrics['occupied_cells']} 格 / 可通行 {map_metrics['free_cells']} 格"
        f"，栅格 {map_metrics['grid_width']}×{map_metrics['grid_height']}\n"
        f"- 体检：{'通过' if health['ok'] else '未通过，已强制出图'}"
        f"（最大帧间位移 {health['max_step_m']} m，高差 {health['z_span_m']} m）\n"
        f"- 耗时：{summary['elapsed_s']} s\n",
        encoding="utf-8")

    log(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def simplify_metrics(**kwargs) -> dict:
    traj = kwargs["traj"]
    counts = kwargs["counts"]
    cfg = kwargs["cfg"]
    bag = kwargs["bag"]
    return {
        "bag": str(bag),
        "cloud_frames": int(kwargs["cloud_frames"]),
        "cloud_frame_id": kwargs["cloud_frame_id"],
        "odom_messages": int(len(traj)),
        "duration_s": round(float(traj.stamps[-1] - traj.stamps[0]), 2),
        "trajectory_z_span_m": round(float(np.ptp(traj.positions[:, 2])), 3),
        "voxel_m": cfg.voxel_m,
        "voxels_total": int(kwargs["means_count"]),
        "ground_offset_m": round(float(kwargs["ground_offset"]), 4),
        "ground_histogram": kwargs["ground_hist"],
        "frame_count_percentiles": frame_count_percentiles(counts),
        "frame_count_max": int(counts.max()) if len(counts) else 0,
        "min_frames": int(kwargs["min_frames"]),
        "threshold_mode": kwargs["threshold_mode"],
        "static_voxels": int(kwargs["keep"].sum()),
        "static_ratio": round(float(kwargs["keep"].mean()) if len(kwargs["keep"]) else 0.0, 4),
        "noground_output": kwargs["noground_stats"],
        "elapsed_s": round(time.time() - kwargs["started"], 1),
    }


# ------------------------------------------------------------------ 入口

class PcdProcessBagNode:
    """ROS2 包装：参数来自 ROS 参数表，跑完一次即 shutdown。

    真正的工作在 `run()`，这个类只负责参数搬运 —— 所以离线逻辑可脱离 ROS 单测。
    """

    def __init__(self):
        import rclpy
        from rclpy.node import Node

        rclpy.init(args=None)
        self.rclpy = rclpy
        self.node = Node("pcd_process_bag")
        defaults = DEFAULT_CONFIG.to_ros_params()
        self.node.declare_parameter("bag", "")
        self.node.declare_parameter("out_dir", "")
        self.node.declare_parameter("force", False)
        self.node.declare_parameter("overwrite", False)
        seen = {"bag", "out_dir", "force", "overwrite"}   # 已声明过，不能重复声明
        for name, value in defaults.items():
            if name not in seen:
                self.node.declare_parameter(name, _to_param(value))
                seen.add(name)
            # 兼容省略 _m 后缀的简写（如 max_route_distance）；同名则不重复声明
            short = _short(name)
            if short != name and short not in seen:
                self.node.declare_parameter(short, _to_param(value))
                seen.add(short)

    def _get(self, name):
        value = self.node.get_parameter(name).value
        if value is None or (isinstance(value, float) and math.isnan(value)):
            short = _short(name)
            if short != name and self.node.has_parameter(short):
                alt = self.node.get_parameter(short).value
                if not (isinstance(alt, float) and math.isnan(alt)):
                    return alt
            if value is not None:
                return None          # NaN 哨兵 = 未设置
        return value

    def run(self) -> int:
        params = {name: self._get(name) for name in DEFAULT_CONFIG.to_ros_params()}
        # launch 传空串表示"未设置"，转成 None 才走自动/默认分支
        params = {k: (None if v == "" else v) for k, v in params.items()}
        bag = self.node.get_parameter("bag").value
        out = self.node.get_parameter("out_dir").value
        if not bag or not out:
            self.node.get_logger().error("必须提供参数 bag 与 out_dir")
            return 2
        cfg = PcdProcessConfig.from_params(params)
        force = bool(self.node.get_parameter("force").value)
        overwrite = bool(self.node.get_parameter("overwrite").value)
        try:
            cfg.validate()
        except ValueError as exc:
            self.node.get_logger().error(f"参数非法：{exc}")
            return 2
        log = lambda msg: self.node.get_logger().info(str(msg))  # noqa: E731
        try:
            run(bag, out, cfg, force=force, overwrite=overwrite, log=log)
        except SystemExit as exc:
            self.node.get_logger().error(str(exc))
            return 1
        return 0

    def shutdown(self):
        self.node.destroy_node()
        self.rclpy.shutdown()


NAN = float("nan")


def _to_param(value):
    """None 不能直接声明（rclpy 会告警），用 NaN 当"未设置"哨兵。"""
    return NAN if value is None else value


def _short(name: str) -> str:
    return name[:-2] if name.endswith("_m") else name


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        import rclpy  # noqa: F401
        has_ros = True
    except ImportError:
        has_ros = False

    if "--ros-args" in argv or (has_ros and not argv):
        node = PcdProcessBagNode()
        try:
            return node.run()
        finally:
            node.shutdown()

    parser = _add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args(argv)
    cfg = _config_from_args(args)
    run(args.bag, args.out, cfg, force=args.force, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
