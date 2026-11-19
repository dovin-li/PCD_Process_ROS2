import unittest
from unittest.mock import patch

import numpy as np

from loop_closure import run
from loop_closure.points import OdometrySample, PointCloudFrame


def yaw_quaternion(degrees):
    half = np.deg2rad(degrees) * 0.5
    return [0.0, 0.0, np.sin(half), np.cos(half)]


class TestRos2HeaderTimePairing(unittest.TestCase):
    def test_clouds_use_interpolated_header_time_pose_despite_record_order(self):
        odom0 = OdometrySample(
            0.0, [0.0, 0.0, 0.0], yaw_quaternion(0.0),
            "camera_init", "body")
        odom2 = OdometrySample(
            2.0, [2.0, 0.0, 0.0], yaw_quaternion(90.0),
            "camera_init", "body")
        cloud_middle = PointCloudFrame(
            1.0, np.array([[1.0, 2.0, 3.0]], np.float32), "camera_init")
        cloud_outside = PointCloudFrame(
            3.0, np.array([[7.0, 8.0, 9.0]], np.float32), "camera_init")

        # Record/receipt order is deliberately unrelated to message header time:
        # both clouds arrive before either odometry sample, and cloud header order
        # remains valid.  The out-of-range cloud must not be extrapolated.
        events = [
            ("cloud", 100.0, cloud_middle),
            ("cloud", 101.0, cloud_outside),
            ("odom", 102.0, odom2),
            ("odom", 104.0, odom0),
        ]
        def iter_topics(_bag, odom_topic, cloud_topic):
            wanted = ({"odom"} if odom_topic else set()) | ({"cloud"} if cloud_topic else set())
            return iter(event for event in events if event[0] in wanted)

        with patch("loop_closure.bag_reader.iter_bag", side_effect=iter_topics):
            stamps, poses, clouds = run.load_bag("unused", keyframe_meter=0.0)

        self.assertEqual(stamps, [1.0])
        np.testing.assert_allclose(
            [transform[:3, 3] for transform in poses],
            [[1.0, 0.0, 0.0]], atol=1e-12)
        np.testing.assert_allclose(
            poses[0][:2, :2],
            [[np.cos(np.deg2rad(45.0)), -np.sin(np.deg2rad(45.0))],
             [np.sin(np.deg2rad(45.0)), np.cos(np.deg2rad(45.0))]],
            atol=1e-12)
        np.testing.assert_array_equal(clouds[0], cloud_middle.xyz)

    def test_wrong_fast_lio_frames_are_rejected_before_keyframes_are_built(self):
        quat = yaw_quaternion(0.0)
        good_odom = OdometrySample(1.0, [0, 0, 0], quat, "camera_init", "body")
        good_cloud = PointCloudFrame(
            1.0, np.array([[1.0, 2.0, 3.0]], np.float32), "camera_init")
        cases = [
            (OdometrySample(1.0, [0, 0, 0], quat, "map", "body"), good_cloud,
             "camera_init.*body"),
            (OdometrySample(1.0, [0, 0, 0], quat, "camera_init", "base_link"), good_cloud,
             "camera_init.*body"),
            (good_odom, PointCloudFrame(
                1.0, np.array([[1.0, 2.0, 3.0]], np.float32), "body"),
             "camera_init"),
        ]
        for odom, cloud, message in cases:
            with self.subTest(frames=(odom.frame_id, odom.child_frame_id, cloud.frame_id)):
                events = [("odom", 10.0, odom), ("cloud", 11.0, cloud)]

                def iter_topics(_bag, odom_topic, cloud_topic):
                    wanted = ({"odom"} if odom_topic else set()) | ({"cloud"} if cloud_topic else set())
                    return iter(event for event in events if event[0] in wanted)

                with patch("loop_closure.bag_reader.iter_bag", side_effect=iter_topics), \
                        self.assertRaisesRegex(ValueError, message):
                    run.load_bag("unused", keyframe_meter=0.0)


if __name__ == "__main__":
    unittest.main()
