"""ROS1 bag 离线回环检测，使用 /livox/lidar 原始点云与 /Odometry。

TRON2 的 Livox bridge 已将倒装雷达的 y、z 取反。原始扫描按最近到达的
/Odometry 配对，使用该位姿作为雷达位姿近似；每帧内部保留原始运动畸变。
默认将最近 10 次扫描变换至当前关键帧坐标系，供描述子和 GICP 共用。

用法：
    python3 -m loop_closure.run_ros1 --bag <bag> --out <dir>
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import icp_align, loop_detection, pose_graph
from .loop_detection import detect_loops
from .run import publish_staged, write_tum


def read_livox_xyz(raw):
    """Livox CustomMsg → (N,3) float32。

    实测头部布局（ROS1 序列化，非对齐）：
        seq u32(4) | stamp(8) | frame_id string(4+len) | timebase u64(8)
        | point_num u32(4) | lidar_id u8(1) | rsvd u8[3](3) | 填充(4)
    点数据从该处开始，每点 19 字节：
        offset_time u32(4) | x,y,z f32(12) | reflectivity/tag/line u8(3)
    """
    import struct

    p = 0
    p += 4                                            # seq
    p += 8                                            # stamp
    (fl,) = struct.unpack_from("<I", raw, p); p += 4
    p += fl                                           # frame_id
    p += 8                                            # timebase
    (pn,) = struct.unpack_from("<I", raw, p); p += 4
    p += 1 + 3                                        # lidar_id + rsvd
    p += 4                                            # 填充（实测必需，否则 t 字段错位）

    n = int(pn)
    if n <= 0:
        return np.zeros((0, 3), np.float32)

    step = 19
    if len(raw) - p < n * step:
        step = max(1, (len(raw) - p) // n)
    buf = np.frombuffer(raw, dtype=np.uint8, count=n * step, offset=p).reshape(n, step)
    xyz = np.empty((n, 3), dtype=np.float32)
    for k in range(3):
        xyz[:, k] = buf[:, 4 + k * 4: 8 + k * 4].copy().view(np.float32).ravel()
    return xyz[np.isfinite(xyz).all(axis=1)]


def load_ros1(bag_path, keyframe_meter=0.2, submap_scans=1):
    """读 ROS1 bag，按位移抽关键帧。返回 (stamps, world_poses, body_clouds)。"""
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore

    ts = get_typestore(Stores.ROS1_NOETIC)
    stamps, poses, clouds = [], [], []
    last_odom = None
    last_key = None
    window = loop_detection.ScanWindow(submap_scans)

    with Reader(Path(bag_path)) as r:
        lidar_c = [c for c in r.connections if c.topic == "/livox/lidar"]
        odom_c = [c for c in r.connections if c.topic == "/Odometry"]
        if not lidar_c or not odom_c:
            raise SystemExit("bag 里缺 /livox/lidar 或 /Odometry")
        lidar_c, odom_c = lidar_c[0], odom_c[0]

        for conn, t, raw in r.messages(connections=[lidar_c, odom_c]):
            if conn.id == odom_c.id:
                m = ts.deserialize_ros1(raw, conn.msgtype)
                p, q = m.pose.pose.position, m.pose.pose.orientation
                last_odom = (m.header.stamp.sec + m.header.stamp.nanosec * 1e-9,
                             np.array([p.x, p.y, p.z]),
                             np.array([q.x, q.y, q.z, q.w]))
                continue
            if last_odom is None:
                continue
            stamp, pos, quat = last_odom
            T = icp_align.T_from_pose(pos, quat)
            keyframe = last_key is None or np.linalg.norm(T[:3, 3] - last_key[:3, 3]) >= keyframe_meter
            if submap_scans == 1 and not keyframe:
                continue
            xyz = read_livox_xyz(raw)
            if xyz.shape[0] == 0:
                continue
            window.append(xyz, T)
            if not keyframe:
                continue
            stamps.append(stamp)
            poses.append(T)
            clouds.append(window.merged(world_coordinates=False))
            last_key = T
    return stamps, poses, clouds


def _run_once(bag_path, out_dir, args):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("== 读 bag（/livox/lidar 原始点云）==", flush=True)
    mid360 = loop_detection.Mid360Input.from_args(args)
    stamps, poses, clouds = load_ros1(bag_path, args.keyframe_meter, submap_scans=mid360.submap_scans)
    print("   关键帧 %d 个，首帧 %d 点，耗时 %.1f s"
          % (len(poses), len(clouds[0]) if clouds else 0, time.time() - t0), flush=True)
    if len(poses) < 5:
        print("关键帧太少，退出")
        return 1

    descriptor, radius, min_score, mode, proximity = loop_detection.descriptor_options(args)
    print("== 准备 %s 模式候选 ==" % mode, flush=True)
    descs, fill = loop_detection.prepare_descriptors(clouds, mode, descriptor, radius, mid360)
    print("   %d 个描述子，耗时 %.1f s" % (len(descs), time.time() - t0), flush=True)

    write_tum(out / "keyscan_odom_poses.txt", stamps, poses)

    print("== 闭环检测 ==", flush=True)
    diagnostics = loop_detection.LoopDiagnostics(mode, descriptor)
    edges, tried = detect_loops(descs, clouds, poses, args.sc_thres, args.min_gap,
                                args.fitness_thres, descriptor=descriptor,
                                solid_min_score=min_score, candidate_mode=mode,
                                proximity_radius=proximity, diagnostics=diagnostics, solid_profile=mid360.profile)
    print("   候选 %d 次，接受 %d 个，耗时 %.1f s" % (tried, len(edges), time.time() - t0), flush=True)

    with open(out / "keyscan_matches.txt", "w") as fh:
        for (i, j, T, fit) in edges:
            fh.write("%d %d %.6f\n" % (i, j, fit))
    source_counts = diagnostics.write(out)

    print("== 位姿图优化 ==", flush=True)
    opt, stats = pose_graph.optimize(poses, edges)
    write_tum(out / "keyscan_optimized_poses.txt", stamps, opt)
    cmp = pose_graph.compare(poses, opt)

    summary = {
        "bag": str(bag_path),
        "keyframes": len(poses),
        "points_per_frame": int(len(clouds[0])) if clouds else 0,
        "cloud_points_per_keyframe": loop_detection.point_counts(clouds),
        "descriptor_fill": fill,
        **source_counts,
        "loop_edges": len(edges),
        "candidates_tried": tried,
        "graph": stats,
        "correction": cmp,
        "params": {**mid360.stats(), "keyframe_meter": args.keyframe_meter, "sc_thres": args.sc_thres,
                   "min_gap": args.min_gap, "fitness_thres": args.fitness_thres,
                   "max_radius": radius, "descriptor": descriptor if mode != "geometry" else None,
                   "solid_min_score": min_score, "candidate_mode": mode, "proximity_radius": proximity},
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(out / "loop_stats.json", "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print("== 完成 ==")
    print("   闭环 %d 个" % len(edges))
    print("   修正：平移 max %.3f m / 均 %.3f m；转角 max %.2f° / 均 %.2f°"
          % (cmp["trans_max_m"], cmp["trans_mean_m"], cmp["angle_max_deg"], cmp["angle_mean_deg"]))
    return 0


def run(bag_path, out_dir, args):
    return publish_staged(_run_once, bag_path, out_dir, args)


def main():
    ap = argparse.ArgumentParser(description="ROS1 bag 回环检测（Livox 原始点云 + SOLiD + GICP + iSAM2）")
    ap.add_argument("--bag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keyframe-meter", type=float, default=0.5)
    ap.add_argument("--sc-thres", type=float, default=0.25)
    ap.add_argument("--min-gap", type=int, default=loop_detection.DEFAULT_MIN_GAP)
    ap.add_argument("--fitness-thres", type=float, default=0.3)
    loop_detection.add_descriptor_options(ap)
    a = ap.parse_args()
    return run(a.bag, a.out, a)


if __name__ == "__main__":
    sys.exit(main())
