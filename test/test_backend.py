"""Loop-closure backend contract tests (no ROS runtime required)."""

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loop_closure import icp_align, pose_graph  # noqa: E402


def _pose(x=0.0, y=0.0, yaw=0.0):
    c, s = np.cos(yaw), np.sin(yaw)
    out = np.eye(4)
    out[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    out[:3, 3] = [x, y, 0.0]
    return out


class BackendContractTest(unittest.TestCase):
    def test_alignment_to_between_factor_inverts_current_to_history(self):
        """ICP current->history is the inverse of Between(current, history)."""
        current_world = _pose(x=2.0)
        history_world = _pose(x=0.0)
        current_to_history = np.linalg.inv(history_world) @ current_world

        factor = icp_align.alignment_to_between(current_to_history)

        expected = np.linalg.inv(current_world) @ history_world
        np.testing.assert_allclose(factor, expected, atol=1e-12)

    def test_gicp_backend_maps_source_points_into_target_frame(self):
        rng = np.random.default_rng(7)
        source = rng.normal(size=(600, 3)).astype(np.float64)
        source[:, 0] *= 2.0
        truth = _pose(x=0.20, y=-0.10, yaw=np.deg2rad(5.0))
        target = source @ truth[:3, :3].T + truth[:3, 3]

        got, fitness, rmse = icp_align.align_frames(
            source, target, init=truth, leaf=0.05, method="gicp")

        np.testing.assert_allclose(got, truth, atol=2e-2)
        self.assertGreater(fitness, 0.9)
        self.assertLess(rmse, 0.03)

    def test_isam2_path_is_used_when_requested(self):
        poses = [_pose(x=float(k) * 1.02) for k in range(6)]
        exact_loop = np.linalg.inv(poses[0]) @ _pose(x=5.0)
        optimized, stats = pose_graph.optimize(
            poses, [(0, 5, exact_loop, 0.03)], use_isam=True)

        self.assertEqual(stats["optimizer"], "isam2")
        self.assertEqual(stats["relinearize_skip"], 1)
        self.assertEqual(stats["loop_kernel"], "cauchy")
        self.assertEqual(stats["loop_edges"], 1)
        self.assertEqual(len(optimized), len(poses))
        self.assertLess(stats["error_after"], stats["error_before"])

    def test_loop_score_scales_the_robust_factor_noise(self):
        np.testing.assert_allclose(
            pose_graph._loop_sigmas(fitness=0.10, loop_score=0.40),
            [0.10, 0.10, 0.10, 0.20, 0.20, 0.20],
        )
        np.testing.assert_allclose(
            pose_graph._loop_sigmas(fitness=0.10, loop_score=0.10),
            [0.025, 0.025, 0.025, 0.05, 0.05, 0.05],
        )

    def test_loop_score_must_be_finite_and_positive(self):
        for score in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(score=score), self.assertRaisesRegex(ValueError, "loop_score"):
                pose_graph._loop_sigmas(fitness=0.10, loop_score=score)

    def test_compare_labels_start_end_distance_without_calling_it_closure_error(self):
        raw = [_pose(x=0.0), _pose(x=2.0)]
        optimized = [_pose(x=0.0), _pose(x=1.5)]
        stats = pose_graph.compare(raw, optimized)
        self.assertEqual(stats["raw_start_end_distance_m"], 2.0)
        self.assertEqual(stats["optimized_start_end_distance_m"], 1.5)
        self.assertNotIn("closure_error", stats)


if __name__ == "__main__":
    unittest.main()
