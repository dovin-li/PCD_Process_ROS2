import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from loop_closure import icp_align, loop_detection, run, run_ros1
from test_solid_frontend import pose, scene


class CandidateModeTest(unittest.TestCase):
    def test_descriptor_candidates_reject_good_gicp_when_its_measured_yaw_disagrees(self):
        cloud = scene()
        previous, predicted = np.eye(4), pose(165, [0.3, 0.4, 0])
        for source in ("solid", "scan-context", "geometry"):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as out:
                candidate = loop_detection.Proposal(0, source, 1.0, np.deg2rad(5))
                diagnostics = loop_detection.LoopDiagnostics(
                    "hybrid", source if source != "geometry" else "solid")
                with patch.object(icp_align, "align_frames", return_value=(predicted, 0.85, 0.1)) as align:
                    edges, tried = loop_detection.verify_candidates(
                        [(1, [candidate])], [cloud, cloud], [previous, predicted], 0.3,
                        diagnostics=diagnostics)
                self.assertEqual(tried, 1)
                self.assertEqual(len(edges), int(source == "geometry"))
                counts = diagnostics.write(Path(out))
                if source == "geometry":
                    self.assertEqual(align.call_count, 1)
                    self.assertEqual(counts["yaw_consistency_rejections"], 0)
                    self.assertAlmostEqual(diagnostics.records[0]["yaw_consistency_error_deg"], 160)
                else:
                    self.assertEqual(align.call_count, 2)
                    self.assertEqual(counts["yaw_consistency_rejections"], 2)
                    self.assertEqual(counts["candidates_rejected_yaw_consistency"][source], 1)

    def test_descriptor_yaw_gate_uses_wrapped_measured_angle_and_keeps_the_boundary(self):
        measured = pose(156, [0, 0, 0])
        candidate = loop_detection.Proposal(0, "solid", 0.99, np.deg2rad(-174))
        diagnostics = loop_detection.LoopDiagnostics("solid", "solid")
        cloud = scene()
        with patch.object(icp_align, "align_frames", return_value=(measured, 0.85, 0.1)) as align:
            edges, _ = loop_detection.verify_candidates(
                [(1, [candidate])], [cloud, cloud], [np.eye(4), pose(50, [0, 0, 0])],
                0.3, diagnostics=diagnostics)
        self.assertEqual(len(edges), 1)
        self.assertEqual(align.call_count, 1)
        self.assertAlmostEqual(diagnostics.records[0]["yaw_consistency_error_deg"], 30)

    def test_both_cli_defaults_select_hybrid_with_two_metre_radius_and_gap_fifty(self):
        for entry in (run, run_ros1):
            with self.subTest(entry=entry.__name__), \
                    patch.object(sys, "argv", ["loop", "--bag", "bag", "--out", "out"]), \
                    patch.object(entry, "run", return_value=0) as invoked:
                entry.main()
                args = invoked.call_args.args[2]
                self.assertEqual(getattr(args, "candidate_mode", None), "hybrid")
                self.assertEqual(args.proximity_radius, 2.0)
                self.assertEqual(args.min_gap, 50)
                self.assertEqual(getattr(args, "submap_scans", None), 10)
                self.assertEqual(args.solid_profile, "tron2")
                self.assertEqual((args.solid_fov_down, args.solid_fov_up), (-7.0, 52.0))

    def test_legacy_profile_spelling_maps_to_published_tron2_axes_in_statistics(self):
        for spelling in ("tron2", "tron2-legacy"):
            with self.subTest(spelling=spelling), \
                    patch.object(sys, "argv", ["loop", "--bag", "bag", "--out", "out",
                                               "--solid-profile", spelling]), \
                    patch.object(run_ros1, "run", return_value=0) as invoked:
                run_ros1.main()
                config = loop_detection.Mid360Input.from_args(invoked.call_args.args[2])
                self.assertEqual(config.stats()["solid_profile"], "tron2")
                self.assertEqual(config.stats()["solid_descriptor_axes"], ["x", "y", "z"])

    def test_entries_forward_cloud_window_and_upstream_profile_to_descriptor_and_registration(self):
        poses = [pose(0, [i * 0.2, 0, 0]) for i in range(5)]
        body = [scene() + i * 0.1 for i in range(5)]
        for entry, reader in ((run, "load_bag"), (run_ros1, "load_ros1")):
            supplied = [cloud + t[:3, 3] for cloud, t in zip(body, poses)] if entry is run else body
            options = ["loop", "--bag", "unused", "--submap-scans", "4", "--solid-profile", "upstream",
                       "--solid-fov-down", "-8", "--solid-fov-up", "41", "--min-gap", "1"]
            with self.subTest(entry=entry.__name__), tempfile.TemporaryDirectory() as out, \
                    patch.object(sys, "argv", options + ["--out", out]), \
                    patch.object(entry, reader, return_value=(range(5), poses, supplied)) as read, \
                    patch.object(loop_detection, "describe_scans", wraps=loop_detection.describe_scans) as describe, \
                    patch.object(icp_align, "align_frames", return_value=(np.eye(4), 0.8, 0.08)) as align, \
                    patch.object(entry.pose_graph, "optimize", return_value=(poses, {})):
                self.assertEqual(entry.main(), 0)
                self.assertEqual(read.call_args.kwargs["submap_scans"], 4)
                self.assertEqual(describe.call_args.kwargs, {"profile": "upstream", "fov_down": -8, "fov_up": 41})
                described = describe.call_args.args[0]
                self.assertTrue(align.called)
                for source, target in [call.args for call in align.call_args_list]:
                    self.assertTrue(any(source is cloud for cloud in described))
                    self.assertTrue(any(target is cloud for cloud in described))
                params = json.loads((Path(out) / "loop_stats.json").read_text())["params"]
                self.assertEqual((params["submap_scans"], params["solid_profile"]), (4, "upstream"))
                self.assertEqual((params["solid_fov_down"], params["solid_fov_up"]), (-8, 41))
                self.assertEqual(params["solid_descriptor_axes"], ["x", "-y", "-z"])

    def test_hybrid_orders_near_history_first_and_removes_shared_pairs(self):
        poses = [pose(0, [x, 0, 0]) for x in (0.5, 0.3, 1.5, 8.0, 0.01, 0.0)]
        hits = [SimpleNamespace(id=index, score=score, yaw_rad=0.0)
                for index, score in ((3, 0.999), (1, 0.992), (0, 0.985))]
        self.assertTrue(hasattr(loop_detection, "candidate_groups"), "hybrid candidate merge is missing")
        expected = {
            "hybrid": ([1, 0, 2, 3], ["geometry", "geometry", "geometry", "solid"]),
            "geometry": ([1, 0, 2], ["geometry", "geometry", "geometry"]),
            "solid": ([3, 1, 0], ["solid", "solid", "solid"]),
        }
        for mode, (ids, sources) in expected.items():
            with self.subTest(mode=mode):
                groups = loop_detection.candidate_groups(
                    [(5, hits)], poses, min_gap=1, proximity_radius=2.0,
                    max_candidates=3, mode=mode, descriptor="solid")
                candidates = dict(groups)[5]
                self.assertEqual([hit.id for hit in candidates], ids)
                self.assertEqual([hit.source for hit in candidates], sources)
                if mode == "hybrid":
                    self.assertEqual([hit.also_source for hit in candidates],
                                     ["solid", "solid", None, None])

    def test_real_entry_writes_source_counts_and_measured_registration_diagnostics(self):
        poses = [pose(0, [x, 0, 0]) for x in (0, 0.5, 5.0, 0, 10.0)]
        cloud = scene()
        body = [cloud] * len(poses)
        world = [cloud + p[:3, 3] for p in poses]
        args = SimpleNamespace(keyframe_meter=0.5, min_gap=2, sc_thres=0.25,
                               fitness_thres=0.3, loop_noise=0.2)

        def register(source, target, init):
            measured = init.copy()
            measured[2, 3] += 0.2
            return measured, 0.9, 0.075

        for entry, reader, scans in ((run, "load_bag", world), (run_ros1, "load_ros1", body)):
            with self.subTest(entry=entry.__name__), tempfile.TemporaryDirectory() as out:
                with patch.object(entry, reader, return_value=(list(range(5)), poses, scans)), \
                        patch.object(icp_align, "align_frames", side_effect=register), \
                        patch.object(entry.pose_graph, "optimize", return_value=(poses, {})):
                    self.assertEqual(entry.run("unused", out, args), 0)
                path = Path(out) / "loop_diagnostics.jsonl"
                self.assertTrue(path.exists(), "accepted edges require a provenance record")
                records = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual([r["source"] for r in records], ["geometry", "solid"])
                self.assertEqual(records[0]["also_source"], "solid")
                self.assertEqual(records[0]["odom_distance_m"], 0.0)
                self.assertGreater(records[1]["score"], 0.98)
                for record in records:
                    self.assertAlmostEqual(record["overlap"], 0.9)
                    self.assertAlmostEqual(record["rmse"], 0.075)
                    self.assertAlmostEqual(record["prediction_delta"], 0.2)
                    self.assertEqual((record["init_source"], record["yaw_fallback"]), ("odometry", False))
                stats = json.loads((Path(out) / "loop_stats.json").read_text())
                self.assertEqual(stats["params"]["candidate_mode"], "hybrid")
                self.assertEqual(stats["accepted_by_source"], {"geometry": 1, "solid": 1})
                self.assertEqual(stats["tried_by_source"], {"geometry": 1, "solid": 1})
                matches = (Path(out) / "keyscan_matches.txt").read_text().splitlines()
                self.assertEqual([len(row.split()) for row in matches], [3, 3])

    def test_geometry_only_runs_without_extracting_descriptors(self):
        poses = [np.eye(4)] * 5
        cloud = scene()
        args = SimpleNamespace(keyframe_meter=0.5, min_gap=2, sc_thres=0.25,
                               fitness_thres=0.3, candidate_mode="geometry")
        with tempfile.TemporaryDirectory() as out, \
                patch.object(run_ros1, "load_ros1", return_value=(range(5), poses, [cloud]*5)), \
                patch.object(loop_detection, "describe_scans", side_effect=AssertionError("unexpected descriptor extraction")), \
                patch.object(icp_align, "align_frames", return_value=(np.eye(4), 0.9, 0.1)), \
                patch.object(run_ros1.pose_graph, "optimize", return_value=(poses, {})):
            self.assertEqual(run_ros1.run("unused", out, args), 0)
            records = [json.loads(row) for row in (Path(out) / "loop_diagnostics.jsonl").read_text().splitlines()]
            self.assertTrue(all(r["source"] == "geometry" and r["score"] is None for r in records))
            self.assertEqual(len(records), 2)
