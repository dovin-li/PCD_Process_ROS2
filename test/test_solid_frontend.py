import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from loop_closure import icp_align, loop_detection, run, run_ros1
from loop_closure.points import OdometrySample, PointCloudFrame


def pose(yaw, translation):
    angle = np.deg2rad(yaw)
    c, s = np.cos(angle), np.sin(angle)
    transform = np.eye(4)
    transform[:2, :2] = [[c, -s], [s, c]]
    transform[:3, 3] = translation
    return transform


def scene():
    rng = np.random.default_rng(47)
    clusters = [
        ([3.8, 1.4, 0.5], 250), ([-1.8, 8.2, 3.1], 125),
        ([-6.5, -3.6, 0.4], 170), ([3.1, -6.2, 1.5], 210),
    ]
    return np.vstack([rng.normal(center, [0.6, 0.4, 0.18], (n, 3))
                      for center, n in clusters]).astype(np.float32)


class TestSolidFrontend(unittest.TestCase):
    def test_descriptor_yaw_recovers_rotated_scene_when_odometry_seed_fails_gicp(self):
        points = scene()
        yaw = pose(90, [0.08, -0.05, 0.02])
        query = run.to_body(points, yaw)
        initial = pose(0, yaw[:3, 3])
        descs, _ = loop_detection.describe_scans([points, query], "solid", 60)
        diagnostics = loop_detection.LoopDiagnostics("solid", "solid")
        edges, tried = loop_detection.detect_loops(
            descs, [points, query], [np.eye(4), initial], 0.15, 0, 0.3,
            descriptor="solid", diagnostics=diagnostics)
        self.assertEqual((tried, len(edges)), (1, 1))
        self.assertEqual(diagnostics.records[0]["init_source"], "descriptor_yaw")
        self.assertTrue(diagnostics.records[0]["yaw_fallback"])
        np.testing.assert_allclose(edges[0][2], np.linalg.inv(yaw), atol=0.02)

    def test_yaw_retry_preserves_odometry_translation_pitch_and_roll(self):
        from scipy.spatial.transform import Rotation
        predicted = pose(0, [1.2, -0.4, 0.6])
        predicted[:3, :3] = Rotation.from_euler("ZYX", [35, 16, -11], degrees=True).as_matrix()
        seeds = list(loop_detection.registration_seeds(predicted, np.deg2rad(-75)))
        self.assertEqual([name for name, _ in seeds], ["odometry", "descriptor_yaw"])
        np.testing.assert_array_equal(seeds[0][1], predicted)
        np.testing.assert_array_equal(seeds[1][1][:3, 3], predicted[:3, 3])
        np.testing.assert_allclose(seeds[1][1][:3, :3],
                                   Rotation.from_euler("ZYX", [-75, 16, -11], degrees=True).as_matrix())

    def test_asymmetric_rotated_revisit_recovers_history_id_and_yaw(self):
        frontend = loop_detection
        points = scene()
        for yaw in (0, 90, -90):
            with self.subTest(yaw=yaw):
                current = run.to_body(points, pose(yaw, [0, 0, 0]))
                clouds = [points, points * 2.2, current, current]
                descs, _ = frontend.describe_scans(clouds, "solid", 60.0)
                hits = dict(frontend.solid_candidates(descs, 2, 0.98, 1))
                self.assertFalse(any(hits.get(i) for i in (0, 1, 2)))
                self.assertEqual([candidate.id for candidate in hits[3]], [0])
                self.assertGreater(hits[3][0].score, 0.99)
                self.assertAlmostEqual(np.rad2deg(hits[3][0].yaw_rad), yaw)

    def test_recent_and_future_neighbors_cannot_displace_valid_history(self):
        frontend = loop_detection
        points = scene()
        clouds = [points * 1.01] + [points] * 52
        descs, _ = frontend.describe_scans(clouds, "solid", 60.0)
        hits = dict(frontend.solid_candidates(descs, 48, 0.8, 1))
        self.assertFalse(any(hits.get(i) for i in range(49)))
        self.assertEqual([candidate.id for candidate in hits[49]], [0])

    def test_empty_or_nonfinite_scans_do_not_retrieve_at_zero_threshold(self):
        frontend = loop_detection
        clouds = [np.empty((0, 3)), scene(), np.array([[np.nan, 1, 2]])]
        descs, _ = frontend.describe_scans(clouds, "solid", 60.0)
        hits = dict(frontend.solid_candidates(descs, 0, 0.0, 2))
        self.assertFalse(any(hits.values()))

    def test_current_to_history_alignment_is_inverted_for_current_history_factor(self):
        frontend = loop_detection
        previous = pose(-30, [8, 4, 2])
        history_from_current = pose(60, [0.7, -0.3, 0.1])
        current = previous @ history_from_current
        history_scan = scene()
        query_scan = run.to_body(history_scan, history_from_current)
        candidate = SimpleNamespace(id=0, score=1.0, yaw_rad=np.deg2rad(60))

        def verify_alignment(source, target, init):
            np.testing.assert_array_equal(source, query_scan)
            np.testing.assert_array_equal(target, history_scan)
            np.testing.assert_allclose(init, history_from_current, atol=1e-12)
            return history_from_current, 1.0, 0.02

        with patch.object(icp_align, "align_frames", side_effect=verify_alignment):
            edges, tried = frontend.verify_candidates(
                [(1, [candidate])], [history_scan, query_scan], [previous, current], 0.3)
        self.assertEqual(tried, 1)
        self.assertEqual(len(edges), 1)
        i, j, factor, _ = edges[0]
        self.assertEqual((i, j), (1, 0))
        np.testing.assert_allclose(current @ factor, previous, atol=1e-12)

    def test_scan_context_fallback_ignores_points_beyond_the_configured_radius(self):
        clouds = [scene() + 100, scene(), scene()]
        descs, _ = loop_detection.describe_scans(clouds, "scan-context", 20.0)
        self.assertTrue(np.isfinite(descs[0]).all())
        hits = dict(loop_detection.scan_context_candidates(descs, 0, 0.25, 2))
        self.assertFalse(hits.get(1))
        self.assertEqual([candidate.id for candidate in hits[2]], [1])

    def test_ros2_keyframe_uses_cloud_header_time(self):
        quat = [0.0, 0.0, 0.0, 1.0]
        odom = OdometrySample(5.1, [1, 0, 0], quat, "camera_init", "body")
        cloud = PointCloudFrame(5.1, scene(), "camera_init")
        events = [("odom", 75.0, odom), ("cloud", 76.0, cloud)]
        def iter_topics(_bag, odom_topic, cloud_topic):
            wanted = ({"odom"} if odom_topic else set()) | ({"cloud"} if cloud_topic else set())
            return iter(event for event in events if event[0] in wanted)

        with patch("loop_closure.bag_reader.iter_bag", side_effect=iter_topics):
            stamps, _, _ = run.load_bag("unused")
        self.assertEqual(stamps, [5.1])

    def test_ros2_world_and_ros1_sensor_scans_retrieve_the_same_loop(self):
        points = scene()
        origin = pose(0, [240, 120, 2])
        yaw = pose(90, [0, 0, 0])
        poses = [origin, origin, origin, origin @ yaw, origin @ yaw]
        world_scans = [points + origin[:3, 3]] * 5
        body_scans = [run.to_body(cloud, t) for cloud, t in zip(world_scans, poses)]
        args = SimpleNamespace(keyframe_meter=0.2, max_radius=60.0, min_gap=2,
                               fitness_thres=0.3, sc_thres=0.15, loop_noise=0.2,
                               candidate_mode="solid")

        def register(source, target, init):
            np.testing.assert_allclose(source @ init[:3, :3].T + init[:3, 3],
                                       target, atol=1e-6)
            return init.copy(), 1.0, 0.02

        for entry, reader, scans in ((run, "load_bag", world_scans),
                                     (run_ros1, "load_ros1", body_scans)):
            with self.subTest(entry=entry.__name__), tempfile.TemporaryDirectory() as out:
                with patch.object(entry, reader, return_value=(list(range(5)), poses, scans)), \
                        patch.object(icp_align, "align_frames", side_effect=register), \
                        patch.object(entry.pose_graph, "optimize", return_value=(poses, {})):
                    self.assertEqual(entry.run("unused", out, args), 0)
                summary = json.loads((Path(out) / "loop_stats.json").read_text())
                self.assertEqual(summary["params"]["descriptor"], "solid")
                self.assertEqual(summary["loop_edges"], 2)


if __name__ == "__main__":
    unittest.main()
