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
from scipy.spatial.transform import Rotation

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
from .pose_correction import PoseCorrection


def _add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--bag", required=True, help="ROS2 bag 目录或 .db3 文件")
    parser.add_argument("--out", "--out-dir", dest="out", required=True, help="输出目录")
    parser.add_argument("--raw-poses", help="回环前的关键帧 TUM 位姿，与 --pgo-poses 配对")
    parser.add_argument("--pgo-poses", help="优化后的同一组关键帧 TUM 位姿")
    parser.add_argument("--pose-time-source", choices=("header", "bag"), default="header",
                        help="关键帧 TUM 使用点云的传感器时间（header）或收包时间（bag）")
    parser.add_argument("--pcd-only", action="store_true", help="导出保留地面的静态 PCD 与轨迹，供人工编辑")
    parser.add_argument("--force", action="store_true", help="跳过轨迹体检（发散数据也出图）")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已存在的输出")
    parser.add_argument("--odom-topic", default=DEFAULT_CONFIG.odom_topic)
    parser.add_argument("--cloud-topic", default=DEFAULT_CONFIG.cloud_topic)
    parser.add_argument("--voxel", type=float, default=DEFAULT_CONFIG.voxel_m)
    parser.add_argument(
        "--min-frames", type=int, default=DEFAULT_CONFIG.min_frames,
        help="帧数过滤阈值：1..10 手动设置，0=Otsu 自动（钳 2..4）；默认 3",
    )
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
        overwrite: bool = False, log=print, *, raw_poses=None, pgo_poses=None,
        pcd_only=False, pose_time_source="header") -> dict:
    started = time.time()
    bag_path = Path(bag).expanduser()
    files, _meta = resolve_storage(bag_path)
    if not files:
        raise SystemExit(f"{bag_path}: 不是有效的 ROS2 bag")
    if bool(raw_poses) != bool(pgo_poses):
        raise SystemExit("必须同时提供 --raw-poses 和 --pgo-poses")
    if pose_time_source not in ("header", "bag"):
        raise SystemExit("pose_time_source 必须为 header 或 bag")
    try:
        correction = PoseCorrection(raw_poses, pgo_poses) if raw_poses else None
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    out = Path(out).expanduser().resolve()
    expected = [out / name for name in
                ("summary.json", "cloud_static.pcd", "map.pgm", "map.yaml")]
    had_published_output = any(p.exists() for p in expected)
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
    correction_stamps: List[float] = []
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

    for key, bag_stamp, obj in chain(first, stream):
        if key == "odom":
            if correction is not None and (obj.frame_id, obj.child_frame_id) != (
                    cfg.odom_frame, cfg.odom_child_frame):
                raise SystemExit(f"位姿修正需要 {cfg.odom_frame} → {cfg.odom_child_frame}，"
                                 f"实际里程计坐标系为 {obj.frame_id} → {obj.child_frame_id}")
            traj_rows.append(obj.as_row())
        else:
            xyz = obj.xyz
            if cfg.max_points_per_frame:
                xyz = xyz[:cfg.max_points_per_frame]
            if correction is not None:
                if obj.frame_id != cfg.cloud_frame:
                    raise SystemExit(f"位姿修正需要 {cfg.cloud_frame} 世界系点云，"
                                     f"实际点云坐标系为 {obj.frame_id}")
                correction_stamp = bag_stamp if pose_time_source == "bag" else obj.stamp
                correction_stamps.append(correction_stamp)
                frame_stamps.append(obj.stamp)
                rotation, offset = correction.at(correction_stamp)
                xyz = (rotation.apply(xyz) + offset).astype(np.float32)
            cloud_frames += 1
            cloud_frame_id = cloud_frame_id or obj.frame_id
            voxels.add_frame(xyz)
            rep.add_frame(xyz)
    if not traj_rows or not cloud_frames:
        raise SystemExit("bag 里没有可用的 Odometry 或点云消息")

    traj = Trajectory(np.vstack(traj_rows))
    if np.any(np.diff(traj.stamps) < 0):
        traj = Trajectory(traj.rows[np.argsort(traj.stamps, kind="stable")])
    raw_traj = traj
    if correction is not None:
        times = np.asarray(correction_stamps)
        start, end = correction.stamps[[0, -1]]
        if times.max() < start or times.min() > end:
            raise SystemExit("关键帧与点云时间范围没有重叠，请核对 bag、TUM 与 --pose-time-source")
        pose_metadata = {
            "raw_poses": str(Path(raw_poses).expanduser().resolve()),
            "optimized_poses": str(Path(pgo_poses).expanduser().resolve()),
            "time_source": pose_time_source, "keyframes": len(correction.stamps),
            "frames_before_first": int(np.count_nonzero(times < start)),
            "frames_after_last": int(np.count_nonzero(times > end)),
        }
        query_times = traj.stamps
        if pose_time_source == "bag":
            # Map each sensor stamp to the same cloud's receipt time. Using a
            # separate odometry receipt time would introduce its transport delay.
            order = np.argsort(frame_stamps, kind="stable")
            query_times = np.interp(query_times, np.asarray(frame_stamps)[order],
                                    times[order])
        rotations, offsets = correction.at(query_times)
        corrected_rows = traj.rows.copy()
        corrected_rows[:, 1:4] = rotations.apply(traj.positions) + offsets
        corrected_rows[:, 4:8] = (rotations * Rotation.from_quat(traj.orientations)).as_quat()
        traj = Trajectory(corrected_rows)
    means = rep.means
    counts = voxels.counts
    rep_keys = rep.keys
    if len(rep_keys) != len(voxels.keys) or not np.array_equal(rep_keys, voxels.keys):
        raise SystemExit("内部错误：体素集合与代表点集合不一致")

    log(f"   odom {len(traj)} 条 / 点云 {cloud_frames} 帧 / "
        f"体素 {len(voxels)}（frame_id={cloud_frame_id or '?'}）")

    # ---- 地面偏移 ----
    if pcd_only:
        ground_offset, ground_hist = None, None
    elif cfg.ground_offset_m is None:
        ground_offset, ground_hist = estimate_ground_offset(
            means, counts, traj.positions[:, :2], traj.positions[:, 2])
        ground_source = "auto"
    else:
        ground_offset, ground_hist = float(cfg.ground_offset_m), None
        ground_source = "argument"
    if not pcd_only:
        if ground_offset is None:
            raise SystemExit("地面偏移未能自动估计；请用 --ground-offset 手工给值")
        log(f"   地面偏移 {ground_offset:.4f} m（{ground_source}）")

    min_frames, threshold_mode = resolve_min_frames(counts, cfg.min_frames)
    keep = counts >= min_frames
    static_xyz = means[keep]

    # Validate everything that can invalidate the export before publishing the
    # first file.  In particular, an overwrite attempt that fails validation
    # must leave the previous, internally consistent generation untouched.
    log("== 2/4 轨迹体检 ==")
    health = traj.health(cfg.max_step_m, cfg.max_z_span_m, cfg.min_duration_s)
    log("   " + json.dumps(health, ensure_ascii=False))
    if not health["ok"] and not force:
        if not had_published_output:
            (out / "report.md").write_text(
                "# 拒绝出图\n\n" + "\n".join(f"- {p}" for p in health["problems"]) + "\n",
                encoding="utf-8")
        raise SystemExit("轨迹体检未通过，已停止（加 --force 可强制出图）：\n  " +
                         "\n  ".join(health["problems"]))
    if not len(static_xyz):
        raise SystemExit(f"帧计数过滤后静态体素为 0（min_frames={min_frames}），"
                         "拒绝导出空 PCD；请检查数据或降低 --min-frames")

    # ---- 产物 1：PCD / TUM ----
    log("== 3/4 写出点云与轨迹 ==")
    write_pcd(out / "cloud_voxel_all.pcd", means,
              np.clip(counts, 0, 255).astype(np.float32))
    write_pcd(out / "cloud_static.pcd", means[keep],
              np.clip(counts[keep], 0, 255).astype(np.float32))
    traj.write_tum(out / "traj_camera_init.tum")
    if correction is not None:
        raw_traj.write_tum(out / "traj_raw.tum")

    noground_stats = None
    if len(static_xyz) and not pcd_only:
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

    metrics = simplify_metrics(
        bag=bag_path, traj=traj, counts=counts, means_count=int(len(means)),
        cloud_frames=cloud_frames, cloud_frame_id=cloud_frame_id,
        cfg=cfg, ground_offset=ground_offset, ground_hist=ground_hist,
        min_frames=min_frames, threshold_mode=threshold_mode, keep=keep,
        noground_stats=noground_stats, started=started,
    )
    if correction is not None:
        metrics["pose_correction"] = pose_metadata

    if pcd_only:
        outputs = {
            "pcd_all": str(out / "cloud_voxel_all.pcd"),
            "pcd_static": str(out / "cloud_static.pcd"),
            "traj": str(out / "traj_camera_init.tum"),
        }
        if correction is not None:
            outputs["traj_raw"] = str(out / "traj_raw.tum")
        summary = {
            "bag": str(bag_path), "out": str(out), "health": health,
            "cloud": {"frames": cloud_frames, "voxels_total": len(means),
                      "static_voxels": int(keep.sum())}, "outputs": outputs,
        }
        if correction is not None:
            summary["pose_correction"] = pose_metadata
        (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
        log(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary

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
    if correction is not None:
        summary["pose_correction"] = pose_metadata
        summary["outputs"]["traj_raw"] = str(out / "traj_raw.tum")
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
        "ground_offset_m": (round(float(kwargs["ground_offset"]), 4)
                            if kwargs["ground_offset"] is not None else None),
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

OFFLINE_POSE_PARAMS = {"raw_poses": "", "pgo_poses": "", "pose_time_source": "header",
                       "pcd_only": False}


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
        for name, default in OFFLINE_POSE_PARAMS.items():
            self.node.declare_parameter(name, default)
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
        pose_params = {name: self.node.get_parameter(name).value for name in OFFLINE_POSE_PARAMS}
        try:
            cfg.validate()
        except ValueError as exc:
            self.node.get_logger().error(f"参数非法：{exc}")
            return 2
        log = lambda msg: self.node.get_logger().info(str(msg))  # noqa: E731
        try:
            run(bag, out, cfg, force=force, overwrite=overwrite, log=log, **pose_params)
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
    if bool(args.raw_poses) != bool(args.pgo_poses):
        parser.error("必须同时提供 --raw-poses 和 --pgo-poses")
    cfg = _config_from_args(args)
    run(args.bag, args.out, cfg, force=args.force, overwrite=args.overwrite,
        raw_poses=args.raw_poses, pgo_poses=args.pgo_poses, pcd_only=args.pcd_only,
        pose_time_source=args.pose_time_source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
