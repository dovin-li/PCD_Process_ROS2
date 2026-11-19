"""在线节点单测：累积状态机、地面偏移冻结策略、与离线的一致性。

不依赖 rclpy —— `OnlineAccumulator` 是纯 Python 状态机，
ROS 节点只是把消息翻译进来。这样在线逻辑也能在 Mac 上单测。
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tron2_pcd_process.config import DEFAULT_CONFIG  # noqa: E402
from tron2_pcd_process.online_node import OnlineAccumulator  # noqa: E402
from tron2_pcd_process.pipeline import VoxelAccumulator  # noqa: E402
from tron2_pcd_process.points import OdometrySample  # noqa: E402


def _synthetic_stream(frames=400, points=400, seed=7):
    """直走 + 地面在 -1.2 m + 一面墙，返回 (odom 列表, 点云列表)。"""
    rng = np.random.default_rng(seed)
    odom, clouds = [], []
    for index in range(frames):
        x = index * 0.02
        cloud = np.column_stack([
            rng.uniform(x - 5, x + 5, points),
            rng.uniform(-3, 3, points),
            np.where(rng.random(points) < 0.5, -1.2, -0.8),   # 地面 + 墙底
        ]).astype(np.float32)
        clouds.append(cloud)
        odom.append(OdometrySample(
            stamp=index * 0.1, position=[x, 0.0, 0.0], orientation=[0, 0, 0, 1.0]).as_row())
    return odom, clouds


def test_not_ready_before_enough_data():
    acc = OnlineAccumulator(DEFAULT_CONFIG, "/tmp/unused")
    assert acc.ready is False
    for _ in range(10):
        acc.add_cloud(np.zeros((10, 3), np.float32), 0.0)
    assert acc.ready is False, "只有点云、没有位姿时不能算出图"


def test_ground_offset_stays_none_until_threshold():
    """位姿不够时不许出地面偏移——早期主峰会跑偏到 0.8 甚至负数。"""
    cfg = DEFAULT_CONFIG.replace(ground_estimate_min_poses=100,
                                 ground_estimate_min_voxels=100)
    acc = OnlineAccumulator(cfg, "/tmp/unused")
    odom, clouds = _synthetic_stream(frames=40)
    for row, cloud in zip(odom, clouds):
        acc.add_odometry(row)
        acc.add_cloud(cloud, 0.0)
    assert acc.odom_count < 100
    assert acc.stats()["ground_offset_m"] is None


def test_ground_offset_estimated_and_locked():
    cfg = DEFAULT_CONFIG.replace(ground_estimate_min_poses=50,
                                 ground_estimate_min_voxels=100)
    acc = OnlineAccumulator(cfg, "/tmp/unused")
    odom, clouds = _synthetic_stream(frames=200)
    for index, (row, cloud) in enumerate(zip(odom, clouds)):
        acc.add_odometry(row)
        acc.add_cloud(cloud, 0.0)
        if index == 120:
            first = acc.ground_offset
    assert first is not None, "过门槛后必须估出地面偏移"
    assert abs(first - 1.2) < 0.1, f"地面偏移 {first} 偏离 1.2 太多"
    assert acc._ground_locked, "位姿数达到 2× 门槛后应锁定"
    locked = acc.ground_offset
    acc.add_odometry(odom[-1])
    acc.add_cloud(clouds[-1], 0.0)
    assert acc.ground_offset == locked, "锁定后地面基准不能再变（否则 2D 图会错层）"


def test_explicit_ground_offset_is_used_and_locked_immediately():
    cfg = DEFAULT_CONFIG.replace(ground_offset_m=1.05)
    acc = OnlineAccumulator(cfg, "/tmp/unused")
    odom, clouds = _synthetic_stream(frames=5)
    acc.add_odometry(odom[0])
    acc.add_cloud(clouds[0], 0.0)
    assert acc.ground_offset == 1.05
    assert acc._ground_locked is True


def test_online_accumulation_matches_offline_voxel_set():
    """同样的输入，在线累积器与离线累积器必须得到相同的体素与帧计数。"""
    acc = OnlineAccumulator(DEFAULT_CONFIG, "/tmp/unused")
    offline = VoxelAccumulator(DEFAULT_CONFIG.voxel_m)
    odom, clouds = _synthetic_stream(frames=60)
    for row, cloud in zip(odom, clouds):
        acc.add_odometry(row)
        acc.add_cloud(cloud, 0.0)
        offline.add_frame(cloud)

    online_keys, online_counts = acc.voxels.sorted_view()
    offline_keys, offline_counts = offline.sorted_view()
    assert np.array_equal(online_keys, offline_keys)
    assert np.array_equal(online_counts, offline_counts)


def test_max_points_per_frame_caps_frames():
    cfg = DEFAULT_CONFIG.replace(max_points_per_frame=5,
                                 ground_estimate_min_poses=1,
                                 ground_estimate_min_voxels=1)
    acc = OnlineAccumulator(cfg, "/tmp/unused")
    cloud = np.zeros((100, 3), np.float32)
    cloud[:, 1] = np.arange(100) * 0.1
    acc.add_cloud(cloud, 0.0)
    assert len(acc.voxels) == 5, "max_points_per_frame 必须生效（在线限内存用）"


def test_export_writes_expected_files(tmp_path=None):
    import tempfile

    cfg = DEFAULT_CONFIG.replace(ground_estimate_min_poses=1,
                                 ground_estimate_min_voxels=1,
                                 min_duration_s=0.0)
    with tempfile.TemporaryDirectory() as tmp:
        acc = OnlineAccumulator(cfg, tmp)
        odom, clouds = _synthetic_stream(frames=80)
        for row, cloud in zip(odom, clouds):
            acc.add_odometry(row)
            acc.add_cloud(cloud, 0.0)
        summary = acc.export()
        out = Path(tmp)
        for name in ("cloud_voxel_all.pcd", "cloud_static.pcd",
                     "traj_camera_init.tum", "map.pgm", "map.yaml", "summary.json"):
            assert (out / name).is_file(), f"缺少产物 {name}"
        assert summary["mode"] == "online"
        assert summary["map"]["occupied_cells"] > 0


if __name__ == "__main__":
    import traceback

    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failed = 0
    for name, func in tests:
        try:
            func()
            print(f"PASS {name}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
