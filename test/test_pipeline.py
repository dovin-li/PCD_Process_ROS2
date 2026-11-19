"""核心处理链单测：不依赖 ROS，不依赖真实 bag。

跑法（Mac / 机器人容器都行）：
    cd src/tron2_pcd_process
    PYTHONPATH=. python3 -m pytest test -q          # 有 pytest
    PYTHONPATH=. python3 test/test_pipeline.py      # 没 pytest 也能跑（内置最小 runner）
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tron2_pcd_process.config import PcdProcessConfig, DEFAULT_CONFIG  # noqa: E402
from tron2_pcd_process.pipeline import (  # noqa: E402
    RepresentativeAccumulator,
    VoxelAccumulator,
    estimate_ground_offset,
    otsu_threshold,
    resolve_min_frames,
    voxel_keys,
)
from tron2_pcd_process.points import (  # noqa: E402
    OdometrySample,
    PointCloudFrame,
    Trajectory,
    read_pcd_xyz,
    write_pcd,
)


def _straight_frames(frames=10, points_per_frame=200, seed=0):
    """合成一段"直走 + 两面平行墙"的点云，用于验证帧计数过滤与去动态。"""
    rng = np.random.default_rng(seed)
    clouds = []
    for index in range(frames):
        x = index * 0.1
        wall = np.column_stack([
            rng.uniform(-10, 10, points_per_frame // 2),
            np.full(points_per_frame // 2, 2.0),
            rng.uniform(-1.5, 0.5, points_per_frame // 2),
        ])
        floor = np.column_stack([
            rng.uniform(-10, 10, points_per_frame // 2),
            rng.uniform(-2, 2, points_per_frame // 2),
            np.full(points_per_frame // 2, -1.2),
        ])
        clouds.append(np.vstack([wall, floor]).astype(np.float32))
        del x
    return clouds


def test_voxel_keys_are_stable_and_ordered():
    a = np.array([[0.0, 0.0, 0.0], [0.049, 0.0, 0.0], [0.051, 0.0, 0.0]], np.float32)
    keys = voxel_keys(a, 0.05)
    assert keys[0] == keys[1], "同一 5cm 体素内的点必须同键"
    assert keys[2] != keys[0], "跨体素边界的点必须是不同键"


def test_frame_count_filters_transient_points():
    """只出现在 1~2 帧的点必须被剔除，>=3 帧的结构必须保留。"""
    clouds = _straight_frames(frames=6)
    transient = np.array([[0.0, 0.0, 5.0]], np.float32)   # 空中孤点，只出现 1 帧
    persistent = np.array([[0.0, 0.0, -0.9]], np.float32)  # 每帧都出现

    acc = VoxelAccumulator(0.05)
    for index, cloud in enumerate(clouds):
        extra = [transient] if index == 0 else []
        acc.add_frame(np.vstack([cloud] + extra + [persistent]))
    counts = acc.counts
    keys = acc.keys

    transient_key = int(voxel_keys(transient, 0.05)[0])
    persistent_key = int(voxel_keys(persistent, 0.05)[0])
    keys_sorted, counts_sorted = acc.sorted_view()
    pos_t = int(np.searchsorted(keys_sorted, transient_key))
    pos_p = int(np.searchsorted(keys_sorted, persistent_key))
    assert keys_sorted[pos_t] == transient_key and counts_sorted[pos_t] == 1
    assert keys_sorted[pos_p] == persistent_key and counts_sorted[pos_p] == 6
    del keys, counts


def test_voxel_accumulator_matches_batch_reference():
    """流式累积必须与"拼接后统一 unique"结果完全一致（数量与帧计数）。"""
    clouds = _straight_frames(frames=8, seed=1)
    acc = VoxelAccumulator(0.05)
    for cloud in clouds:
        acc.add_frame(cloud)

    all_keys = np.concatenate([np.unique(voxel_keys(c, 0.05)) for c in clouds])
    uniq, counts = np.unique(all_keys, return_counts=True)
    assert len(acc.keys) == len(uniq)
    # 累积器按键首次出现顺序存储，比较时统一排序
    keys, counted = acc.sorted_view()
    assert np.array_equal(keys, uniq)
    assert np.array_equal(counted, counts)


def test_representative_accumulator_matches_manual_mean():
    clouds = [
        np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]], np.float32),
        np.array([[0.02, 0.0, 0.0]], np.float32),
    ]
    acc = RepresentativeAccumulator(0.05)
    for cloud in clouds:
        acc.add_frame(cloud)
    # 同一体素被 2 帧看到；每帧取首点 [0,0,0] 与 [0.02,0,0]
    means = acc.means
    assert len(means) == 1
    assert np.allclose(means[0], [0.01, 0.0, 0.0], atol=1e-6)


def test_voxel_and_representative_agree_on_key_set():
    clouds = _straight_frames(frames=5, seed=2)
    vox = VoxelAccumulator(0.05)
    rep = RepresentativeAccumulator(0.05)
    for cloud in clouds:
        vox.add_frame(cloud)
        rep.add_frame(cloud)
    assert np.array_equal(vox.sorted_view()[0], np.sort(rep.keys)), \
        "两套累积器的体素集合必须一致"


def test_ground_offset_estimate_recovers_known_height():
    """地面在 -1.2 m（即光心离地 1.2 m），估计值应接近 1.2。"""
    rng = np.random.default_rng(3)
    frames = 20
    means = np.column_stack([
        rng.uniform(-4, 4, 4000),
        rng.uniform(-4, 4, 4000),
        np.full(4000, -1.2) + rng.normal(0, 0.003, 4000),
    ])
    counts = np.ones(len(means), dtype=np.int64)
    traj_xy = np.column_stack([np.linspace(-4, 4, frames), np.zeros(frames)])
    traj_z = np.full(frames, 0.0)
    offset, hist = estimate_ground_offset(means, counts, traj_xy, traj_z)
    assert offset is not None
    assert abs(offset - 1.2) < 0.02, f"估计 {offset} 偏离真实 1.2 m 太多"
    assert hist["matched_voxels"] > 0


def test_ground_offset_returns_none_without_trajectory():
    offset, hist = estimate_ground_offset(
        np.zeros((10, 3)), np.ones(10), np.zeros((0, 2)), np.zeros(0))
    assert offset is None and hist is None


def test_resolve_min_frames_clamps_auto_threshold():
    counts = np.array([1] * 1000 + [50] * 10)
    min_frames, mode = resolve_min_frames(counts, 0)
    assert 2 <= min_frames <= 4, "自动阈值必须钳在 2..4"
    assert mode.startswith("auto-otsu")
    assert resolve_min_frames(counts, 5) == (5, "explicit")


def test_otsu_separates_two_populations():
    """双峰分布：阈值应落在两峰之间。

    注意两峰都要落在 lo=2 以上——低于 lo 的值会被截断到 lo，
    截断后双峰并成一坨，Otsu 会给出接近 lo 的结果（这是原版就有的行为）。
    """
    values = np.array([5] * 500 + [30] * 500)
    threshold = otsu_threshold(values)
    assert 5 < threshold < 30


def test_pcd_roundtrip(tmp_path=None):
    import tempfile

    points = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]], np.float32)
    intensity = np.array([3.0, 7.0], np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "t.pcd"
        write_pcd(path, points, intensity)
        back, back_i = read_pcd_xyz(path)
    assert np.allclose(back, points)
    assert np.allclose(back_i, intensity)


def test_trajectory_health_flags_divergence():
    rows = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [2.0, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ])
    good = Trajectory(rows).health(1.0, 5.0, 10.0)
    assert good["ok"] is False, "时长 2 s < 10 s 必须判定不通过"
    assert any("时长" in p for p in good["problems"])

    jumped = rows.copy()
    jumped[2, 1] = 50.0
    bad = Trajectory(jumped).health(1.0, 5.0, 0.0)
    assert bad["ok"] is False and any("位移" in p for p in bad["problems"])


def test_pointcloud_frame_drops_nonfinite_points():
    xyz = np.array([[1.0, 2.0, 3.0], [np.nan, 0.0, 0.0], [4.0, 5.0, 6.0]], np.float32)
    frame = PointCloudFrame(stamp=0.0, xyz=xyz)
    assert len(frame) == 2, "NaN 点必须在入口就被剔除（否则体素编码会溢出）"


def test_odometry_sample_normalizes_quaternion():
    sample = OdometrySample(stamp=0.0, position=[1, 2, 3], orientation=[0, 0, 0, 2.0])
    assert abs(np.linalg.norm(sample.orientation) - 1.0) < 1e-12


def test_config_validation_rejects_bad_height_window():
    try:
        PcdProcessConfig(height_low_m=1.0, height_high_m=0.5).validate()
    except ValueError:
        return
    raise AssertionError("height_low > height_high 必须报错")


def test_config_from_params_ignores_unknown_and_none():
    cfg = PcdProcessConfig.from_params({"voxel_m": 0.1, "ground_offset_m": None, "bogus": 1})
    assert cfg.voxel_m == 0.1
    assert cfg.ground_offset_m is None
    assert DEFAULT_CONFIG.voxel_m == 0.05


if __name__ == "__main__":
    import traceback

    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
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
