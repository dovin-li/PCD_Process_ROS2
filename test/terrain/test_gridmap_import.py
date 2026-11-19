import contextlib
import io
import json
import shlex
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from terrain_tools import terrain_export as gm


def write_floor(path, height, origin=(0.0, 0.0), hole=None):
    points = []
    for y in range(4):
        for x in range(5):
            if (x, y) == hole:
                continue
            z = height(x, y) if callable(height) else height
            for dx, dy in ((0.02, 0.02), (0.05, 0.02), (0.02, 0.05), (0.05, 0.05)):
                points.append((origin[0] + x * 0.2 + dx, origin[1] + y * 0.2 + dy, z))
    header = (f"FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nWIDTH {len(points)}\n"
              f"HEIGHT 1\nPOINTS {len(points)}\nDATA binary\n")
    path.write_bytes(header.encode() + np.asarray(points, dtype="<f4").tobytes())


# A file-only stand-in for the external ROS grid_map executable. Copy its input
# elevation into layer one; a horizontal floor has zero slope, step and roughness.
FLAT_FILTER = shlex.join((sys.executable, "-c", "\n".join((
    "from pathlib import Path",
    "import sys",
    "grid = Path(sys.argv[1]).read_bytes()",
    "elevation = grid[32:]",
    "Path(sys.argv[2]).write_bytes(grid[:8] + elevation + bytes(4 * len(elevation)))",
)))) + ' "{grid}" "{layers}"'


def export(pcd, out, runner=None):
    argv = [gm.__file__, "--pcd", str(pcd), "--out", str(out)]
    if runner is not None:
        argv.extend(("--run-cmd", runner))
    with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
        gm.main()


class GridMapImportTests(unittest.TestCase):
    def test_cli_help_shows_the_filter_command(self):
        result = subprocess.run([sys.executable, "-B", gm.__file__, "--help"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--run-cmd", result.stdout)

    def test_import_uses_named_xyz_in_an_edited_pcd(self):
        # Normal has three components, intensity is uint32 and z is float64.
        header = (b"VERSION .7\nFIELDS normal intensity z y x\nSIZE 4 4 8 4 4\n"
                  b"TYPE F U F F F\nCOUNT 3 1 1 1 1\nWIDTH 1\nHEIGHT 1\nPOINTS 1\n")
        payloads = {
            "ascii": b"0 1 0 99 3.125 2 1\n",
            "binary": struct.pack("<3fId2f", 0, 1, 0, 99, 3.125, 2, 1),
        }
        with tempfile.TemporaryDirectory() as directory:
            pcd = Path(directory) / "edited.pcd"
            for encoding, payload in payloads.items():
                with self.subTest(encoding=encoding):
                    pcd.write_bytes(header + f"DATA {encoding}\n".encode() + payload)
                    np.testing.assert_array_equal(gm.load_pcd_xyz(pcd), [[1.0, 2.0, 3.125]])

    def test_nonfinite_coordinates_do_not_enter_the_elevation_grid(self):
        header = (b"FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\n"
                  b"WIDTH 2\nHEIGHT 1\nPOINTS 2\nDATA binary\n")
        with tempfile.TemporaryDirectory() as directory:
            pcd = Path(directory) / "edited.pcd"
            pcd.write_bytes(header + struct.pack("<6f", 1, 2, 3, float("nan"), 5, 6))
            np.testing.assert_array_equal(gm.load_pcd_xyz(pcd), [[1, 2, 3]])
            pcd.write_bytes(header + struct.pack("<6f", 1, 2, float("inf"), float("nan"), 5, 6))
            with self.assertRaisesRegex(ValueError, "有限|finite"):
                gm.load_pcd_xyz(pcd)

    def test_xyz_fields_must_be_scalar(self):
        header = (b"FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 2 1 1\n"
                  b"WIDTH 1\nHEIGHT 1\nPOINTS 1\nDATA ascii\n1 2 3 4\n")
        with tempfile.TemporaryDirectory() as directory:
            pcd = Path(directory) / "invalid.pcd"
            pcd.write_bytes(header)
            with self.assertRaisesRegex(ValueError, "COUNT 1"):
                gm.load_pcd_xyz(pcd)

    def test_orientation_rejects_nonfinite_or_unrelated_filter_layers(self):
        elevation = np.arange(12, dtype=np.float32).reshape(3, 4) / 10.0
        known = np.ones_like(elevation, dtype=bool)
        layers = {name: elevation.copy() for name in gm.LAYER_ORDER}

        nonfinite = {name: value.copy() for name, value in layers.items()}
        nonfinite["roughness"][1, 2] = np.nan
        with self.assertRaisesRegex(SystemExit, "非有限|NaN|finite"):
            gm.orientation_fix(nonfinite, elevation, known)

        unrelated = {name: value.copy() for name, value in layers.items()}
        unrelated["elevation_inpainted"] += 7.0
        with self.assertRaisesRegex(SystemExit, "误差|不匹配"):
            gm.orientation_fix(unrelated, elevation, known)

    def test_sparse_cloud_without_observed_cells_reports_the_density_issue(self):
        with tempfile.TemporaryDirectory() as directory:
            pcd = Path(directory) / "sparse.pcd"
            out = Path(directory) / "terrain"
            pcd.write_bytes(b"FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nWIDTH 1\n"
                            b"HEIGHT 1\nPOINTS 1\nDATA ascii\n1 2 3\n")
            with self.assertRaisesRegex(SystemExit, "点密度|点数"):
                export(pcd, out, FLAT_FILTER)
            self.assertFalse((out / "map.yaml").exists())

    def test_same_size_replacement_cloud_cannot_reuse_old_layers(self):
        with tempfile.TemporaryDirectory() as directory:
            pcd = Path(directory) / "edited.pcd"
            out = Path(directory) / "terrain output"
            write_floor(pcd, 0.0)
            export(pcd, out, FLAT_FILTER)
            export(pcd, out)  # Same cloud, same filter inputs.
            previous = {p: p.read_bytes() for p in out.iterdir() if p.is_file()}

            write_floor(pcd, 2.0)  # Identical name, XY extent, dimensions and point count.
            with self.assertRaisesRegex(SystemExit, "--run-cmd"):
                export(pcd, out)
            self.assertEqual({p: p.read_bytes() for p in previous}, previous)

            export(pcd, out, FLAT_FILTER)
            export(pcd, out)
            self.assertNotEqual((out / "layers.bin").read_bytes(), previous[out / "layers.bin"])

    def test_elevation_and_costmap_share_geometry_and_mark_unobserved_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            pcd = Path(directory) / "edited.pcd"
            out = Path(directory) / "terrain"
            origin = (-5.0, 3.0)
            write_floor(pcd, lambda x, y: 1 + 0.01 * x + 0.04 * y, origin, hole=(3, 1))
            export(pcd, out, FLAT_FILTER)

            metrics = json.loads((out / "metrics.json").read_text())
            self.assertIn("artifacts", metrics, "metrics should identify the elevation and costmap files")
            files = metrics["artifacts"]
            elevation = np.load(out / files["elevation"], allow_pickle=False)
            known = np.load(out / files["known"], allow_pickle=False)
            meta = yaml.safe_load((out / files["elevation_metadata"]).read_text())
            nav = yaml.safe_load((out / files["costmap"]).read_text())
            self.assertEqual(files["elevation"], "elevation.npy")
            self.assertEqual(files["known"], "known.npy")
            self.assertEqual(meta["resolution"], nav["resolution"])
            self.assertEqual(meta["origin_xy"], list(origin))
            self.assertEqual(nav["origin"][:2], list(origin))
            self.assertEqual(meta["shape"], [5, 6])
            self.assertEqual(meta["unknown"], "NaN")
            self.assertEqual(elevation.dtype, np.float32)
            self.assertEqual(known.dtype, np.bool_)
            expected = np.full((5, 6), np.nan, dtype=np.float32)
            y, x = np.indices((4, 5))
            expected[:4, :5] = 1 + 0.01 * x + 0.04 * y
            expected[1, 3] = np.nan
            np.testing.assert_allclose(elevation, expected, equal_nan=True)
            np.testing.assert_array_equal(known, np.isfinite(expected))
            pgm = (out / nav["image"]).read_bytes().split(b"\n", 3)
            self.assertEqual(pgm[1], b"6 5")
            cost_pixels = np.frombuffer(pgm[3], np.uint8).reshape(elevation.shape)
            np.testing.assert_array_equal(np.flipud(cost_pixels) > 0, known)

    def test_drivable_is_finalized_after_reachability_marks_a_high_island_lethal(self):
        with tempfile.TemporaryDirectory() as directory:
            pcd = Path(directory) / "edited.pcd"
            out = Path(directory) / "terrain"
            write_floor(pcd, lambda x, _y: 0.75 if x == 4 else 0.0)
            export(pcd, out, FLAT_FILTER)

            drivable = np.load(out / "drivable.npy", allow_pickle=False)
            known = np.load(out / "known.npy", allow_pickle=False)
            pgm = (out / "map.pgm").read_bytes().split(b"\n", 3)
            passable = np.flipud(np.frombuffer(pgm[3], np.uint8).reshape(drivable.shape)) > 0
            np.testing.assert_array_equal(drivable, passable)
            self.assertTrue(known[:, 4].any())
            self.assertFalse(drivable[:, 4].any())


if __name__ == "__main__":
    unittest.main()
