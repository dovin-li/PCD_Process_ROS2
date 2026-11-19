"""点云帧与位姿的数据结构 + PCD 读写。

只依赖 numpy；不 import rclpy —— 离线 bag 节点和（可选的）在线节点在边界处
把 ROS 消息翻译成本模块的纯数组，核心链路不认识 ROS 类型。
"""

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class PointCloudFrame:
    """一帧世界系点云（FAST-LIO 的 /cloud_registered）。"""

    stamp: float                     # 秒（ROS 时间，float64）
    xyz: np.ndarray                  # (N, 3) float32/float64，世界系（camera_init）
    frame_id: str = ""
    seq: int = 0

    def __post_init__(self) -> None:
        self.xyz = np.asarray(self.xyz)
        if self.xyz.ndim != 2 or self.xyz.shape[1] < 3:
            raise ValueError(f"xyz must be (N, >=3), got {self.xyz.shape}")
        self.xyz = np.ascontiguousarray(self.xyz[:, :3], dtype=np.float32)
        # FAST-LIO 偶尔吐 NaN/inf 点，后续体素编码必须排除，否则整数溢出
        if len(self.xyz) and not np.isfinite(self.xyz).all():
            finite = np.isfinite(self.xyz).all(axis=1)
            self.xyz = self.xyz[finite]

    def __len__(self) -> int:
        return int(self.xyz.shape[0])


@dataclass
class OdometrySample:
    """一条位姿：camera_init -> body（平移 + 四元数）。"""

    stamp: float
    position: np.ndarray             # (3,) float64
    orientation: np.ndarray          # (4,) qx qy qz qw float64
    frame_id: str = ""
    child_frame_id: str = ""

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64).reshape(3)
        self.orientation = np.asarray(self.orientation, dtype=np.float64).reshape(4)
        norm = float(np.linalg.norm(self.orientation))
        if norm < 1e-9:
            raise ValueError("zero-length orientation quaternion")
        self.orientation = self.orientation / norm

    def as_row(self) -> np.ndarray:
        """TUM 行：[t, x, y, z, qx, qy, qz, qw]。"""
        return np.concatenate(([self.stamp], self.position, self.orientation))


@dataclass
class Trajectory:
    """按时间升序的位姿序列，内部用 (N, 8) float64 矩阵存储。"""

    rows: np.ndarray = field(default_factory=lambda: np.zeros((0, 8), dtype=np.float64))

    def __post_init__(self) -> None:
        self.rows = np.asarray(self.rows, dtype=np.float64).reshape(-1, 8)

    def __len__(self) -> int:
        return int(self.rows.shape[0])

    @property
    def stamps(self) -> np.ndarray:
        return self.rows[:, 0]

    @property
    def positions(self) -> np.ndarray:
        return self.rows[:, 1:4]

    @property
    def orientations(self) -> np.ndarray:
        return self.rows[:, 4:8]

    @classmethod
    def from_samples(cls, samples) -> "Trajectory":
        samples = sorted(samples, key=lambda s: s.stamp)
        if not samples:
            return cls()
        return cls(np.vstack([s.as_row() for s in samples]))

    @classmethod
    def read_tum(cls, path) -> "Trajectory":
        values = np.loadtxt(path, comments="#", dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.shape[1] < 8:
            raise ValueError("trajectory must have at least 8 columns (TUM)")
        return cls(values[:, :8])

    def write_tum(self, path) -> Path:
        path = Path(path)
        with path.open("w", encoding="ascii") as stream:
            stream.write("# timestamp tx ty tz qx qy qz qw\n")
            for row in self.rows:
                stream.write("%.9f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n" % tuple(row))
        return path

    def health(self, max_step_m: float, max_z_span_m: float,
               min_duration_s: float) -> dict:
        """轨迹体检：里程计发散直接拒绝出图（原版第 9 节）。"""
        if len(self) == 0:
            return {"poses": 0, "duration_s": 0.0, "max_step_m": None,
                    "z_span_m": None, "ok": False, "problems": ["轨迹为空"]}
        pos = self.positions
        steps = np.hypot(np.diff(pos[:, 0]), np.diff(pos[:, 1]))
        step_max = float(steps.max()) if len(steps) else 0.0
        z_span = float(np.ptp(pos[:, 2]))
        duration = float(self.stamps[-1] - self.stamps[0])
        problems = []
        if step_max > max_step_m:
            problems.append(
                f"相邻位姿最大位移 {step_max:.2f} m > {max_step_m} m（里程计疑似发散）")
        if z_span > max_z_span_m:
            problems.append(
                f"全程高差 {z_span:.2f} m > {max_z_span_m} m（里程计疑似发散）")
        if duration < min_duration_s:
            problems.append(f"时长 {duration:.1f} s < {min_duration_s} s（数据太短）")
        return {
            "poses": int(len(self)),
            "duration_s": round(duration, 2),
            "max_step_m": round(step_max, 4),
            "z_span_m": round(z_span, 4),
            "ok": not problems,
            "problems": problems,
        }


# ------------------------------------------------------------------ PCD 读写

def write_pcd(path, points: np.ndarray, intensity: Optional[np.ndarray] = None) -> Path:
    """写 binary PCD（FIELDS x y z intensity）。与原版逐字节同格式。"""
    path = Path(path)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    n = len(points)
    if intensity is None:
        intensity = np.zeros(n, dtype=np.float32)
    intensity = np.asarray(intensity, dtype=np.float32).reshape(-1)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n"
        "FIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n"
        f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA binary\n"
    )
    record = np.column_stack([points, intensity]).astype("<f4")
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        stream.write(record.tobytes())
    return path


def read_pcd_xyz(path):
    """读 PCD 的 xyz（支持 ascii / binary），返回 (points, intensity_or_None)。

    投影器要的只是 xyz，但把 intensity 一起返回，方便按帧数二次过滤。
    """
    path = Path(path)
    with path.open("rb") as stream:
        header_lines = []
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"{path}: PCD header truncated")
            text = line.decode("ascii", errors="replace").strip()
            header_lines.append(text)
            if text.upper().startswith("DATA"):
                break
        data_offset = stream.tell()

    header = {}
    for text in header_lines:
        if not text or text.startswith("#"):
            continue
        key, _, value = text.partition(" ")
        header[key.upper()] = value.split()

    names = header["FIELDS"]
    sizes = [int(s) for s in header["SIZE"]]
    types = [value.upper() for value in header["TYPE"]]
    counts = [int(c) for c in header.get("COUNT", ["1"] * len(names))]
    if not (len(names) == len(sizes) == len(types) == len(counts)):
        raise ValueError(f"{path}: FIELDS/SIZE/TYPE/COUNT lengths differ")
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError(f"{path}: PCD must contain x, y and z fields")
    if any(count < 1 for count in counts):
        raise ValueError(f"{path}: PCD field COUNT must be positive")
    if any(counts[names.index(name)] != 1 for name in ("x", "y", "z")):
        raise ValueError(f"{path}: PCD x, y and z fields must have COUNT 1")

    data_format = header.get("DATA", ["ascii"])[0].lower()
    component_offsets = np.cumsum([0] + counts[:-1]).tolist()
    column = dict(zip(names, component_offsets))
    if data_format == "ascii":
        raw = np.loadtxt(path, skiprows=len(header_lines), dtype=np.float64, ndmin=2)
        expected_columns = sum(counts)
        if raw.shape[1] != expected_columns:
            raise ValueError(
                f"{path}: ASCII PCD has {raw.shape[1]} columns, expected {expected_columns}")
        idx = [column[name] for name in ("x", "y", "z")]
        inten = raw[:, column["intensity"]] if "intensity" in names else None
        return raw[:, idx].astype(np.float32), inten
    if data_format != "binary":
        raise ValueError(f"{path}: unsupported PCD DATA format {data_format!r}")

    if "POINTS" in header:
        npts = int(header["POINTS"][0])
    else:
        npts = int(header["WIDTH"][0]) * int(header["HEIGHT"][0])
    offset, offsets = 0, []
    for size, count in zip(sizes, counts):
        offsets.append(offset)
        offset += size * count
    formats = {
        ("F", 2): "<f2", ("F", 4): "<f4", ("F", 8): "<f8",
        ("I", 1): "i1", ("I", 2): "<i2", ("I", 4): "<i4", ("I", 8): "<i8",
        ("U", 1): "u1", ("U", 2): "<u2", ("U", 4): "<u4", ("U", 8): "<u8",
    }
    field_formats = []
    for name, type_code, size, count in zip(names, types, sizes, counts):
        base = formats.get((type_code, size))
        if base is None:
            raise ValueError(f"{path}: unsupported PCD field {name}: TYPE {type_code} SIZE {size}")
        field_formats.append(base if count == 1 else (base, (count,)))
    dtype = np.dtype({
        "names": names,
        "formats": field_formats,
        "offsets": offsets,
        "itemsize": offset,
    })
    expected = data_offset + npts * offset
    if path.stat().st_size < expected:
        raise ValueError(f"{path}: PCD data truncated")
    cloud = np.memmap(path, dtype=dtype, mode="r", offset=data_offset, shape=(npts,))
    xyz = np.column_stack([cloud["x"], cloud["y"], cloud["z"]]).astype(np.float32)
    inten = np.asarray(cloud["intensity"]).copy() if "intensity" in names else None
    return xyz, inten


# ------------------------------------------------------------- 栅格图读写

def write_pgm_yaml(image: np.ndarray, pgm_path, resolution: float,
                   origin_x: float, origin_y: float) -> tuple:
    """写 2D 图：0=障碍 / 205=未知 / 254=可通行；PGM 行序翻转成图像坐标系。"""
    pgm_path = Path(pgm_path)
    yaml_path = pgm_path.with_suffix(".yaml")
    flipped = np.flipud(image)
    with pgm_path.open("wb") as stream:
        stream.write(("P5\n%d %d\n255\n" % (flipped.shape[1], flipped.shape[0])).encode("ascii"))
        stream.write(flipped.astype(np.uint8).tobytes())
    yaml_path.write_text(
        "image: %s\nresolution: %.6f\norigin: [%.6f, %.6f, 0.0]\n"
        "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n"
        % (pgm_path.name, resolution, origin_x, origin_y),
        encoding="utf-8",
    )
    return pgm_path, yaml_path


def pgm_size(path) -> tuple:
    """只读 2D 图头，返回 (width, height)。在线节点每轮只更新统计用。"""
    with open(path, "rb") as stream:
        magic = stream.readline()
        if not magic.startswith(b"P5"):
            raise ValueError(f"{path}: not a binary PGM")
        line = stream.readline()
        while line.startswith(b"#"):
            line = stream.readline()
        width, height = (int(v) for v in line.split()[:2])
    return width, height


# ------------------------------------------------------------------ 小工具

def tensor_quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """批量四元数 (N,4) qx qy qz qw → (N,3,3) 旋转矩阵。"""
    q = np.asarray(q, dtype=np.float64)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    out = np.empty((len(q), 3, 3), dtype=np.float64)
    out[:, 0, 0] = 1 - 2 * (yy + zz)
    out[:, 0, 1] = 2 * (xy - wz)
    out[:, 0, 2] = 2 * (xz + wy)
    out[:, 1, 0] = 2 * (xy + wz)
    out[:, 1, 1] = 1 - 2 * (xx + zz)
    out[:, 1, 2] = 2 * (yz - wx)
    out[:, 2, 0] = 2 * (xz - wy)
    out[:, 2, 1] = 2 * (yz + wx)
    out[:, 2, 2] = 1 - 2 * (xx + yy)
    return out


def parse_pointcloud2_header(header_bytes: bytes) -> dict:
    """解析 PointCloud2 原始头（前 16 字节为 seq/sec/nsec）。

    用于离线读 bag 时只读消息头、不反序列化整帧的场景（诊断用）。
    """
    out = {"seq": None, "stamp": None, "frame_id": None}
    if len(header_bytes) < 4:
        return out
    (seq,) = struct.unpack_from("<I", header_bytes, 0)
    sec, nsec = struct.unpack_from("<II", header_bytes, 4)
    (flen,) = struct.unpack_from("<I", header_bytes, 12)
    out.update(
        seq=int(seq),
        stamp=sec + nsec * 1e-9,
        frame_id=header_bytes[16:16 + flen].decode("ascii", errors="replace"),
    )
    return out
