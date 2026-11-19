"""Exercise full-frame pose correction through the offline bag entry point."""

import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from test_ros1_to_ros2 import SPECS, build_odometry_bytes, build_pointcloud2_bytes, serde
from tron2_pcd_process.bag_node import PcdProcessBagNode
from tron2_pcd_process.points import read_pcd_xyz
from tron2_pcd_process.pose_correction import PoseCorrection


PACKAGE = Path(__file__).resolve().parents[1]


class PoseCorrectionTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.bag = self.root / "room.db3"
        self.raw_file = self.root / "raw.tum"
        self.optimized_file = self.root / "optimized.tum"
        self.landmark = np.array([1.225, 0.475, 0.525])
        self.stamps = np.array([100, 106, 112])
        self.raw_rows = []
        self.corrected_rows = []
        walker = serde.Walker(SPECS)
        with sqlite3.connect(self.bag) as db:
            db.execute("CREATE TABLE topics(id INTEGER, name TEXT, type TEXT)")
            db.executemany("INSERT INTO topics VALUES(?,?,?)", [
                (1, "/Odometry", "nav_msgs/msg/Odometry"),
                (2, "/cloud_registered", "sensor_msgs/msg/PointCloud2"),
            ])
            db.execute("CREATE TABLE messages(topic_id INTEGER, timestamp INTEGER, data BLOB)")
            for index, stamp in enumerate(self.stamps):
                raw_pos = np.array([0.4 + 0.15 * index, -0.25, 1.20])
                raw_rot = Rotation.from_euler("x", 20.0, degrees=True)
                rotation = Rotation.from_euler("z", 45.0 * index, degrees=True)
                translation = np.array([0.30 * index, 0.10 * index, 0.0])
                corrected_pos = rotation.apply(raw_pos) + translation
                self.raw_rows.append(np.r_[stamp, raw_pos, raw_rot.as_quat()])
                self.corrected_rows.append(np.r_[stamp, corrected_pos, (rotation * raw_rot).as_quat()])
                # Three observations form one static voxel only after correction.
                static = rotation.inv().apply(self.landmark - translation)
                transient = np.array([5.825 + index, 2.225, 0.525])
                odom = build_odometry_bytes(sec=stamp, nsec=0, pos=raw_pos, quat=raw_rot.as_quat())
                cloud = build_pointcloud2_bytes([static, transient, transient], sec=stamp, nsec=0)
                for topic_id, type_name, blob in [(1, "nav_msgs/msg/Odometry", odom),
                                                   (2, "sensor_msgs/msg/PointCloud2", cloud)]:
                    data = walker.convert(SPECS[type_name], blob, "native", False)
                    db.execute("INSERT INTO messages VALUES(?,?,?)", (topic_id, int(stamp) * 10**9, data))
        self.raw_rows = np.array(self.raw_rows)
        self.corrected_rows = np.array(self.corrected_rows)
        np.savetxt(self.raw_file, self.raw_rows[[0, -1]], fmt="%.15g")
        np.savetxt(self.optimized_file, self.corrected_rows[[0, -1]], fmt="%.15g")

    def run_bag(self, *args):
        return subprocess.run(
            [sys.executable, "-B", "-m", "tron2_pcd_process.bag_node",
             "--bag", str(self.bag), "--out", str(self.root / "out"),
             "--ground-offset", "1.2", *map(str, args)],
            cwd=PACKAGE, capture_output=True, text=True, check=False)

    def test_world_frames_are_corrected_before_observation_counting(self):
        result = self.run_bag("--raw-poses", self.raw_file, "--pgo-poses", self.optimized_file)
        self.assertEqual(result.returncode, 0, result.stderr)
        points, hits = read_pcd_xyz(self.root / "out" / "cloud_static.pcd")
        np.testing.assert_allclose(points, [self.landmark], atol=1e-6)
        np.testing.assert_array_equal(hits, [3])
        raw = np.loadtxt(self.root / "out" / "traj_raw.tum")
        corrected = np.loadtxt(self.root / "out" / "traj_camera_init.tum")
        np.testing.assert_allclose(raw, self.raw_rows, atol=1e-6)
        np.testing.assert_allclose(corrected, self.corrected_rows, atol=1e-6)

    def test_pcd_only_finishes_before_any_2d_projection(self):
        result = self.run_bag("--raw-poses", self.raw_file, "--pgo-poses", self.optimized_file,
                              "--pcd-only", "--height-low", "4", "--height-high", "5")
        self.assertEqual(result.returncode, 0, result.stderr)
        out = self.root / "out"
        self.assertTrue((out / "cloud_static.pcd").is_file())
        self.assertFalse((out / "map.pgm").exists())
        self.assertFalse((out / "map.yaml").exists())
        summary = json.loads((out / "summary.json").read_text())
        self.assertEqual(summary["cloud"]["frames"], 3)
        self.assertEqual(summary["outputs"]["traj_raw"], str(out / "traj_raw.tum"))
        self.assertEqual(summary["outputs"]["traj"], str(out / "traj_camera_init.tum"))
        self.assertNotIn("map", summary)

    def test_failed_trajectory_health_never_publishes_partial_pcd_outputs(self):
        result = self.run_bag("--raw-poses", self.raw_file, "--pgo-poses", self.optimized_file,
                              "--pcd-only", "--min-duration-s", "1000")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("轨迹体检未通过", result.stderr)
        out = self.root / "out"
        self.assertFalse((out / "cloud_voxel_all.pcd").exists())
        self.assertFalse((out / "cloud_static.pcd").exists())
        self.assertFalse((out / "traj_camera_init.tum").exists())
        self.assertFalse((out / "summary.json").exists())

    def test_pcd_only_rejects_an_empty_static_cloud_before_writing_outputs(self):
        result = self.run_bag("--raw-poses", self.raw_file, "--pgo-poses", self.optimized_file,
                              "--pcd-only", "--min-frames", "100")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("静态体素为 0", result.stderr)
        out = self.root / "out"
        self.assertFalse((out / "cloud_voxel_all.pcd").exists())
        self.assertFalse((out / "cloud_static.pcd").exists())
        self.assertFalse((out / "traj_camera_init.tum").exists())
        self.assertFalse((out / "summary.json").exists())

    def test_bag_clock_uses_cloud_acquisition_for_the_matching_odom_pose(self):
        # Each cloud is recorded 40 seconds after its sensor stamp in this fixture.
        # Odometry has a different transport delay, but the same acquisition stamps.
        with sqlite3.connect(self.bag) as db:
            db.execute("UPDATE messages SET timestamp = timestamp + 40000000000 WHERE topic_id = 2")
            db.execute("UPDATE messages SET timestamp = timestamp + 39000000000 WHERE topic_id = 1")
        for path in [self.raw_file, self.optimized_file]:
            rows = np.loadtxt(path)
            rows[:, 0] += 40
            np.savetxt(path, rows, fmt="%.15g")
        result = self.run_bag("--raw-poses", self.raw_file, "--pgo-poses", self.optimized_file,
                              "--pose-time-source", "bag", "--pcd-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        out = self.root / "out"
        points, hits = read_pcd_xyz(out / "cloud_static.pcd")
        np.testing.assert_allclose(points, [self.landmark], atol=1e-6)
        np.testing.assert_array_equal(hits, [3])
        np.testing.assert_allclose(np.loadtxt(out / "traj_camera_init.tum"),
                                   self.corrected_rows, atol=1e-6)

    def test_malformed_tum_pairs_explain_what_is_invalid(self):
        raw = self.raw_rows[[0, -1]]
        opt = self.corrected_rows[[0, -1]]
        mismatch = opt.copy()
        mismatch[-1, 0] += 0.1
        not_finite = raw.copy()
        not_finite[0, 1] = np.nan
        zero_quaternion = opt.copy()
        zero_quaternion[-1, 4:] = 0
        cases = [
            (raw[:1], opt[:1], "至少两行"),
            (raw, opt[:1], "至少两行|行数"),
            (raw, np.vstack((opt, opt[-1])), "行数"),
            (raw[:, :7], opt, "8 列"),
            (raw, np.column_stack((opt, np.ones(2))), "8 列"),
            (not_finite, opt, "有限"),
            (raw, zero_quaternion, "四元数"),
            (raw[::-1], opt[::-1], "严格递增"),
            (raw[[0, 0]], opt[[0, 0]], "严格递增"),
            (raw, mismatch, "时间戳不匹配"),
        ]
        for raw_rows, optimized_rows, message in cases:
            with self.subTest(message=message, shapes=(raw_rows.shape, optimized_rows.shape)):
                np.savetxt(self.raw_file, raw_rows, fmt="%.15g")
                np.savetxt(self.optimized_file, optimized_rows, fmt="%.15g")
                with self.assertRaisesRegex(ValueError, message):
                    PoseCorrection(self.raw_file, self.optimized_file)

    def test_cli_rejects_one_sided_pair_as_an_argument_error(self):
        for name, path in [("--raw-poses", self.raw_file), ("--pgo-poses", self.optimized_file)]:
            with self.subTest(name=name):
                result = self.run_bag(name, path)
                self.assertEqual(result.returncode, 2)
                self.assertIn("必须同时提供", result.stderr)
                self.assertFalse((self.root / "out").exists())

    def test_disjoint_keyframe_times_reject_the_bag_before_pcd_export(self):
        for path in [self.raw_file, self.optimized_file]:
            rows = np.loadtxt(path)
            rows[:, 0] += 1000
            np.savetxt(path, rows, fmt="%.15g")
        result = self.run_bag("--raw-poses", self.raw_file, "--pgo-poses", self.optimized_file,
                              "--pcd-only")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("时间范围没有重叠", result.stderr)
        self.assertFalse((self.root / "out" / "cloud_static.pcd").exists())

    def test_frames_outside_keyframes_use_the_nearest_endpoint_and_are_counted(self):
        for path in [self.raw_file, self.optimized_file]:
            rows = np.loadtxt(path)
            rows[:, 0] = [101, 111]
            np.savetxt(path, rows, fmt="%.15g")
        result = self.run_bag("--raw-poses", self.raw_file, "--pgo-poses", self.optimized_file,
                              "--pcd-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        out = self.root / "out"
        points, _ = read_pcd_xyz(out / "cloud_static.pcd")
        np.testing.assert_allclose(points, [self.landmark], atol=1e-6)
        correction = json.loads((out / "summary.json").read_text()).get("pose_correction", {})
        self.assertEqual(correction.get("keyframes"), 2)
        self.assertEqual(correction.get("time_source"), "header")
        self.assertEqual(correction.get("frames_before_first"), 1)
        self.assertEqual(correction.get("frames_after_last"), 1)

    def test_rejects_a_non_world_frame_within_the_stream(self):
        with sqlite3.connect(self.bag) as db:
            index, cloud = db.execute("SELECT rowid,data FROM messages WHERE topic_id=2 "
                                      "ORDER BY timestamp DESC LIMIT 1").fetchone()
            cloud = cloud.replace(b"camera_init\0", b"body_sensor\0")
            db.execute("UPDATE messages SET data=? WHERE rowid=?", (cloud, index))
        result = self.run_bag("--raw-poses", self.raw_file, "--pgo-poses", self.optimized_file,
                              "--pcd-only")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("body_sensor", result.stderr)
        self.assertIn("camera_init", result.stderr)
        self.assertFalse((self.root / "out" / "cloud_static.pcd").exists())

    def test_ros_parameter_wrapper_runs_the_same_pcd_stage(self):
        # Model only the ROS parameter API. Bag reading, processing and export run normally.
        parameters = {}
        node = Mock()
        node.declare_parameter.side_effect = parameters.setdefault
        node.get_parameter.side_effect = lambda name: SimpleNamespace(value=parameters[name])
        node.has_parameter.side_effect = parameters.__contains__
        ros = {"rclpy": Mock(), "rclpy.node": SimpleNamespace(Node=lambda _name: node)}
        with patch.dict(sys.modules, ros):
            wrapper = PcdProcessBagNode()
            for name in ("raw_poses", "pgo_poses", "pose_time_source", "pcd_only"):
                self.assertIn(name, parameters)
            out = self.root / "out"
            parameters.update(bag=str(self.bag), out_dir=str(out), raw_poses=str(self.raw_file),
                              pgo_poses=str(self.optimized_file), pcd_only=True)
            self.assertEqual(wrapper.run(), 0)
            points, hits = read_pcd_xyz(out / "cloud_static.pcd")
            np.testing.assert_allclose(points, [self.landmark], atol=1e-6)
            np.testing.assert_array_equal(hits, [3])
            self.assertFalse((out / "map.pgm").exists())


if __name__ == "__main__":
    unittest.main()
