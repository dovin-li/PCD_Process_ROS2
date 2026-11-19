"""ROS2 bag 离线回环检测，默认使用 SOLiD、GICP 和 GTSAM iSAM2。

输入契约（与 PCD Process ROS2 相同）：
    /cloud_registered  sensor_msgs/PointCloud2   世界系（camera_init）
    /Odometry          nav_msgs/Odometry         camera_init -> body

输出（写到 out_dir）：
    keyscan_odom_poses.txt        回环前的位姿（tum 风格：stamp x y z qx qy qz qw）
    keyscan_optimized_poses.txt   回环后的位姿
    keyscan_matches.txt           检测到的闭环（i j fitness）
    loop_stats.json               统计摘要

用法：
    PYTHONPATH=. python3 -m loop_closure.run \
        --bag <bag目录或.db3> --out <输出目录>
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from . import icp_align
from . import loop_detection
from . import pose_graph


REQUIRED_OUTPUTS = frozenset({
    "keyscan_odom_poses.txt",
    "keyscan_optimized_poses.txt",
    "keyscan_matches.txt",
    "loop_diagnostics.jsonl",
    "loop_stats.json",
})


def publish_staged(worker, bag_path, out_dir, args):
    """Build one complete output generation, then replace the destination.

    Computation never writes into an existing result directory. Once all five
    public artifacts exist, the complete staging directory replaces the old
    directory on the same filesystem. If computation or publication fails,
    the previous complete generation is restored and the exception propagates.
    """
    destination = Path(out_dir).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.staging-", dir=str(destination.parent)))
    backup = stage.with_name(stage.name.replace(".staging-", ".previous-", 1))
    moved_previous = False
    try:
        result = worker(bag_path, stage, args)
        if result != 0:
            return result
        missing = sorted(name for name in REQUIRED_OUTPUTS if not (stage / name).is_file())
        if missing:
            raise RuntimeError("loop output generation is incomplete: " + ", ".join(missing))

        if destination.exists() or destination.is_symlink():
            os.replace(destination, backup)
            moved_previous = True
        try:
            os.replace(stage, destination)
        except Exception:
            if moved_previous and not destination.exists():
                os.replace(backup, destination)
                moved_previous = False
            raise
        if moved_previous:
            if backup.is_dir() and not backup.is_symlink():
                shutil.rmtree(backup)
            else:
                backup.unlink()
            moved_previous = False
        print("   产物 → %s" % destination)
        return result
    finally:
        if stage.exists():
            shutil.rmtree(stage)
        if moved_previous and backup.exists() and not destination.exists():
            os.replace(backup, destination)


def rot_to_quat(R):
    """3x3 → (qx,qy,qz,qw)，Shepperd 法，数值稳定。"""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w], dtype=np.float64)


def write_tum(path, stamps, poses):
    with open(path, "w") as fh:
        for st, T in zip(stamps, poses):
            q = rot_to_quat(T[:3, :3])
            p = T[:3, 3]
            fh.write("%.9f %.6f %.6f %.6f %.9f %.9f %.9f %.9f\n"
                     % (st, p[0], p[1], p[2], q[0], q[1], q[2], q[3]))


def _slerp_quaternion(q0, q1, alpha):
    """Shortest-path quaternion interpolation for ``(qx, qy, qz, qw)``."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        value = q0 + float(alpha) * (q1 - q0)
        return value / np.linalg.norm(value)
    theta = np.arccos(dot)
    scale = np.sin(theta)
    return (np.sin((1.0 - alpha) * theta) / scale * q0
            + np.sin(alpha * theta) / scale * q1)


def _prepare_odometry(samples):
    """Sort by header time and collapse duplicate stamps deterministically."""
    by_stamp = {float(sample.stamp): sample for sample in samples}
    ordered = [by_stamp[stamp] for stamp in sorted(by_stamp)]
    if not ordered:
        return (np.zeros(0, dtype=np.float64), np.zeros((0, 3), dtype=np.float64),
                np.zeros((0, 4), dtype=np.float64))
    return (
        np.asarray([sample.stamp for sample in ordered], dtype=np.float64),
        np.asarray([sample.position for sample in ordered], dtype=np.float64),
        np.asarray([sample.orientation for sample in ordered], dtype=np.float64),
    )


def _pose_at_header_time(stamps, positions, orientations, stamp):
    """Interpolate ``camera_init -> body`` at a cloud header timestamp.

    Exact endpoints are accepted.  A cloud outside the odometry header-time
    interval returns ``None`` instead of silently extrapolating or reusing the
    last record-order sample.
    """
    if len(stamps) == 0:
        return None
    stamp = float(stamp)
    index = int(np.searchsorted(stamps, stamp, side="left"))
    if index < len(stamps) and stamps[index] == stamp:
        return icp_align.T_from_pose(positions[index], orientations[index])
    if index == 0 or index == len(stamps):
        return None
    before = index - 1
    span = float(stamps[index] - stamps[before])
    if span <= 0.0:
        return None
    alpha = (stamp - float(stamps[before])) / span
    position = positions[before] + alpha * (positions[index] - positions[before])
    orientation = _slerp_quaternion(
        orientations[before], orientations[index], alpha)
    return icp_align.T_from_pose(position, orientation)


def load_bag(bag_path, keyframe_meter=0.2, submap_scans=1):
    """读 bag，按位移抽关键帧。返回 (stamps, world_poses, world_clouds)。

    直接复用 PCD Process ROS2 的 iter_bag（sqlite3 直读，不需要 ROS）。
    先按消息 header 时间整理 odom，再二次扫描 cloud；每帧 cloud 使用相邻
    odom 的平移线性插值 + 四元数 slerp。这样记录/接收顺序交错不会污染
    位姿。超出 odom header 时间范围的 cloud 明确跳过，不做外推。

    cloud 话题自身必须保持 header 时间非递减；若不满足则拒绝生成一个
    时间倒退的 TUM 轨迹。两帧位移 >= keyframe_meter 时收作关键帧。
    """
    from .bag_reader import iter_bag

    odometry = [obj for kind, _, obj in
                iter_bag(bag_path, "/Odometry", None) if kind == "odom"]
    for sample in odometry:
        if sample.frame_id != "camera_init" or sample.child_frame_id != "body":
            raise ValueError(
                "/Odometry frame contract requires camera_init -> body; got "
                f"{sample.frame_id} -> {sample.child_frame_id}")
    odom_stamps, odom_positions, odom_orientations = _prepare_odometry(odometry)

    stamps, poses, clouds = [], [], []
    last_key = None
    last_cloud_stamp = None
    window = loop_detection.ScanWindow(submap_scans)
    for kind, _, obj in iter_bag(bag_path, None, "/cloud_registered"):
        if kind != "cloud":
            continue
        if obj.frame_id != "camera_init":
            raise ValueError(
                "/cloud_registered frame contract requires camera_init; got "
                f"{obj.frame_id}")
        if last_cloud_stamp is not None and obj.stamp < last_cloud_stamp:
            raise ValueError(
                "/cloud_registered header timestamp is not monotonic: "
                f"{obj.stamp:.9f} < {last_cloud_stamp:.9f}")
        last_cloud_stamp = obj.stamp
        T = _pose_at_header_time(
            odom_stamps, odom_positions, odom_orientations, obj.stamp)
        if T is None:
            continue
        window.append(obj.xyz, T)
        if last_key is None or np.linalg.norm(T[:3, 3] - last_key[:3, 3]) >= keyframe_meter:
            stamps.append(obj.stamp)
            poses.append(T)
            clouds.append(window.merged(world_coordinates=True))
            last_key = T
    return stamps, poses, clouds


def to_body(cloud_world, T_world_body):
    """将世界系点云转换到该帧 body 坐标系，p_body = R^T (p_world - t)。"""
    R = T_world_body[:3, :3]
    t = T_world_body[:3, 3]
    return (cloud_world.astype(np.float64) - t) @ R


# Legacy callers supply the original world scans through this module variable.
# The bag entry point passes body-frame scans directly and holds no global cache.
descs_src = []


def detect_loops(descs, poses, keys, sc_thres=0.15, history=3,
                 min_gap=10, fitness_thres=0.3, max_candidates=3, max_gate=2.0):
    body_clouds = [to_body(cloud, T) for cloud, T in zip(descs_src, poses)]
    return loop_detection.detect_loops(descs, body_clouds, poses, sc_thres, min_gap,
                                       fitness_thres, max_candidates, max_gate)


def _run_once(bag_path, out_dir, args):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("== 读 bag 并抽关键帧 ==", flush=True)
    mid360 = loop_detection.Mid360Input.from_args(args)
    stamps, poses, clouds = load_bag(bag_path, keyframe_meter=args.keyframe_meter,
                                     submap_scans=mid360.submap_scans)
    print("   关键帧 %d 个，耗时 %.1f s" % (len(poses), time.time() - t0), flush=True)
    if len(poses) < 5:
        print("关键帧太少，退出")
        return 1

    descriptor, radius, min_score, mode, proximity = loop_detection.descriptor_options(args)
    print("== 准备 %s 模式候选 ==" % mode, flush=True)
    body_clouds = [to_body(cloud, T) for cloud, T in zip(clouds, poses)]
    descs, fill = loop_detection.prepare_descriptors(body_clouds, mode, descriptor, radius, mid360)
    del clouds
    print("   %d 个描述子，耗时 %.1f s" % (len(descs), time.time() - t0), flush=True)

    write_tum(out / "keyscan_odom_poses.txt", stamps, poses)

    print("== 检测闭环 ==", flush=True)
    diagnostics = loop_detection.LoopDiagnostics(mode, descriptor)
    edges, tried = loop_detection.detect_loops(
        descs, body_clouds, poses,
        sc_thres=args.sc_thres, min_gap=args.min_gap,
        fitness_thres=args.fitness_thres, descriptor=descriptor, solid_min_score=min_score,
        candidate_mode=mode, proximity_radius=proximity, diagnostics=diagnostics,
        solid_profile=mid360.profile,
    )
    print("   候选 %d 次，接受闭环 %d 个，耗时 %.1f s"
          % (tried, len(edges), time.time() - t0), flush=True)

    with open(out / "keyscan_matches.txt", "w") as fh:
        for (i, j, T, fit) in edges:
            fh.write("%d %d %.6f\n" % (i, j, fit))
    source_counts = diagnostics.write(out)

    print("== 位姿图优化 ==", flush=True)
    opt, stats = pose_graph.optimize(poses, edges, loop_score=args.loop_noise)
    write_tum(out / "keyscan_optimized_poses.txt", stamps, opt)

    cmp = pose_graph.compare(poses, opt)
    summary = {
        "bag": str(bag_path),
        "keyframes": len(poses),
        "cloud_points_per_keyframe": loop_detection.point_counts(body_clouds),
        "descriptor_fill": fill,
        **source_counts,
        "pose_time_source": "cloud_header",
        "loop_edges": len(edges),
        "candidates_tried": tried,
        "graph": stats,
        "correction": cmp,
        "params": {
            **mid360.stats(),
            "keyframe_meter": args.keyframe_meter,
            "sc_thres": args.sc_thres,
            "min_gap": args.min_gap,
            "fitness_thres": args.fitness_thres,
            "loop_noise": args.loop_noise,
            "max_radius": radius,
            "descriptor": descriptor if mode != "geometry" else None,
            "solid_min_score": min_score,
            "candidate_mode": mode,
            "proximity_radius": proximity,
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(out / "loop_stats.json", "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print("== 完成 ==")
    print("   闭环 %d 个" % len(edges))
    print("   修正量：平移最大 %.3f m / 均 %.3f m；转角最大 %.2f° / 均 %.2f°"
          % (cmp["trans_max_m"], cmp["trans_mean_m"],
             cmp["angle_max_deg"], cmp["angle_mean_deg"]))
    return 0


def run(bag_path, out_dir, args):
    return publish_staged(_run_once, bag_path, out_dir, args)


def main():
    ap = argparse.ArgumentParser(description="ROS2 bag 回环检测（SOLiD + GICP + iSAM2）")
    ap.add_argument("--bag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keyframe-meter", type=float, default=0.2)
    ap.add_argument("--sc-thres", type=float, default=0.15)
    ap.add_argument("--min-gap", type=int, default=loop_detection.DEFAULT_MIN_GAP)
    ap.add_argument("--fitness-thres", type=float, default=0.3)
    ap.add_argument("--loop-noise", type=float, default=0.2)
    loop_detection.add_descriptor_options(ap)
    args = ap.parse_args()
    return run(args.bag, args.out, args)


if __name__ == "__main__":
    sys.exit(main())
