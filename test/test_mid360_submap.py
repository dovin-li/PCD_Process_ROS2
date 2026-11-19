import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from loop_closure import loop_detection, run, run_ros1
from loop_closure._solid import Config, Extractor
from loop_closure.points import OdometrySample, PointCloudFrame
from test_solid_frontend import pose, scene


class TestMid360Submap(unittest.TestCase):
    def test_profile_uses_selected_vertical_fov_and_reports_yaw_in_the_input_body_frame(self):
        bridge_cloud = scene() * [1, -1, -1]
        current = run.to_body(bridge_cloud, pose(90, [0, 0, 0]))
        for profile, axes in (("upstream", [1, -1, -1]), ("tron2", [1, 1, 1]),
                              ("tron2-legacy", [1, 1, 1])):
            with self.subTest(profile=profile):
                descs, _ = loop_detection.describe_scans(
                    [bridge_cloud, current], "solid", 60, profile=profile, fov_down=-9, fov_up=38)
                extractor = Extractor(Config(fov_down=-9, fov_up=38, min_range=1, max_range=60))
                np.testing.assert_allclose(descs[0].combined(),
                                           extractor.extract(bridge_cloud * axes).combined())
                hits = dict(loop_detection.solid_candidates(descs, 0, 0.98, 1, profile=profile))
                self.assertAlmostEqual(np.rad2deg(hits[1][0].yaw_rad), 90.0)

    def test_both_readers_use_latest_scans_including_non_keyframes(self):
        positions = [0, 0.25, 0.5, 0.75, 1.0]
        poses = [pose(90 if k == 4 else 0, [x, 0, 0]) for k, x in enumerate(positions)]
        world = [np.array([[5, k, 2]], dtype=np.float32) for k in range(len(poses))]
        body = [run.to_body(scan, p).astype(np.float32) for scan, p in zip(world, poses)]
        odometry = [OdometrySample(k, p[:3, 3], run.rot_to_quat(p[:3, :3]), "camera_init", "body")
                    for k, p in enumerate(poses)]
        clouds = [PointCloudFrame(k, scan, "camera_init") for k, scan in enumerate(world)]

        def ros2_events(_bag, odom_topic, cloud_topic):
            if odom_topic:
                return (("odom", k + 100, o) for k, o in enumerate(odometry))
            return (("cloud", k + 200, c) for k, c in enumerate(clouds))

        raw_conn = SimpleNamespace(id=0, topic="/livox/lidar")
        odom_conn = SimpleNamespace(id=1, topic="/Odometry", msgtype="nav_msgs/msg/Odometry")
        events = []
        for k, o in enumerate(odometry):
            location = SimpleNamespace(x=o.position[0], y=0, z=0)
            q = SimpleNamespace(**dict(zip("xyzw", o.orientation)))
            msg = SimpleNamespace(
                header=SimpleNamespace(stamp=SimpleNamespace(sec=k, nanosec=0)),
                pose=SimpleNamespace(pose=SimpleNamespace(position=location, orientation=q)))
            events.extend([(odom_conn, k, msg), (raw_conn, k, body[k])])
        with patch("loop_closure.bag_reader.iter_bag", side_effect=ros2_events), \
                patch("rosbags.rosbag1.Reader") as reader, \
                patch("rosbags.typesys.get_typestore") as store, \
                patch.object(run_ros1, "read_livox_xyz", side_effect=lambda x: x):
            reader.return_value.__enter__.return_value.connections = [raw_conn, odom_conn]
            reader.return_value.__enter__.return_value.messages.side_effect = lambda **kwargs: iter(events)
            store.return_value.deserialize_ros1.side_effect = lambda raw, message_type: raw
            for size, expected in ((1, world[-1]), (3, np.concatenate(world[-3:]))):
                with self.subTest(size=size):
                    for loader, world_coordinates in ((run.load_bag, True), (run_ros1.load_ros1, False)):
                        stamps, recorded_poses, scans = loader("unused", 1.0, submap_scans=size)
                        self.assertEqual(stamps, [0, 4])
                        np.testing.assert_allclose(scans[0], body[0], atol=1e-6)
                        last = run.to_body(scans[-1], recorded_poses[-1]) if world_coordinates else scans[-1]
                        np.testing.assert_allclose(last, run.to_body(expected, poses[-1]), atol=1e-6)
