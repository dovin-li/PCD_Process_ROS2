import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from loop_closure import loop_detection, run
from test_solid_frontend import pose, scene


def default_args():
    return SimpleNamespace(
        keyframe_meter=0.2, sc_thres=0.15, min_gap=50,
        fitness_thres=0.3, loop_noise=0.2,
        submap_scans=1, solid_profile="tron2",
        solid_fov_down=-7.0, solid_fov_up=52.0,
        candidate_mode="geometry", proximity_radius=2.0,
        descriptor="solid", solid_min_score=0.98, max_radius=60.0,
    )


class DeliveryContractsTest(unittest.TestCase):
    def test_failed_recompute_leaves_the_previous_complete_generation_untouched(self):
        poses = [pose(0, [0.2 * index, 0, 0]) for index in range(5)]
        clouds = [scene() for _ in poses]
        stamps = list(range(len(poses)))
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "loop"
            out.mkdir()
            previous = {
                "keyscan_odom_poses.txt": b"old raw\n",
                "keyscan_optimized_poses.txt": b"old optimized\n",
                "keyscan_matches.txt": b"old matches\n",
                "loop_diagnostics.jsonl": b'{"old": true}\n',
                "loop_stats.json": json.dumps({"generation": "old"}).encode(),
            }
            for name, payload in previous.items():
                (out / name).write_bytes(payload)

            with patch.object(run, "load_bag", return_value=(stamps, poses, clouds)), \
                    patch.object(loop_detection, "detect_loops",
                                 side_effect=RuntimeError("synthetic GICP failure")):
                with self.assertRaisesRegex(RuntimeError, "synthetic GICP failure"):
                    run.run("unused", out, default_args())

            self.assertEqual({name: (out / name).read_bytes() for name in previous}, previous)


if __name__ == "__main__":
    unittest.main()
