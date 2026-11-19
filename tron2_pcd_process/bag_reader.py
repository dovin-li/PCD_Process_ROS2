"""ROS2 bag 读取层：sqlite3 直读 .db3，mcap 才回退 rosbag2_py。

设计目标（与原 ROS1 版同一条原则）：**不需要 ROS 运行时也能读**。
ROS2 的 .db3 bag 就是一张普通 SQLite 表，sqlite3 是 Python 标准库，
所以离线节点能在 Mac / 没装 ROS 的机器上直接跑。

支持：
    * bag 目录（ROS2 惯例：目录里 .db3 + metadata.yaml）
    * 单个 .db3 文件
    * 分片 bag（多个 .db3，按 metadata.yaml 的 relative_file_paths 顺序读）

不支持（明确报错，不静默跳过）：
    * 压缩过的 storage → 报错并提示先 `ros2 bag convert`
    * .mcap → 需要 rosbag2_py，没有就报错

消息反序列化：按 CDR 规则手工解析 Odometry / PointCloud2 需要的字段，
不依赖 rosidl 生成的 Python 类型 —— 离线路径彻底脱离 ROS 环境。
（在线节点走正常的 rclpy 订阅，不需要这套解析。）

CDR 布局约定（与 Fast CDR 一致）：
    前 4 字节封装头（01 00 = CDR_LE），其后每个字段按**相对封装头之后**的
    偏移做自身宽度对齐；string = u32 长度（含 NUL）+ UTF-8 + NUL；
    变长数组 = u32 长度 + 元素。
"""

import re
import sqlite3
import struct
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np

from .points import OdometrySample, PointCloudFrame

ODOMETRY_TYPE = "nav_msgs/msg/Odometry"
POINTCLOUD2_TYPE = "sensor_msgs/msg/PointCloud2"

# sensor_msgs/PointField.datatype → (struct 码, 字节宽)
# 1 INT8 2 UINT8 3 INT16 4 UINT16 5 INT32 6 UINT32 7 FLOAT32 8 FLOAT64
_DT = {
    1: ("b", 1), 2: ("B", 1), 3: ("h", 2), 4: ("H", 2), 5: ("i", 4), 6: ("I", 4),
    7: ("f", 4), 8: ("d", 8),
}


class CdrReader:
    """CDR 读取器。

    内部 `offset` 是**相对封装头之后**的位置（CDR 对齐约定），
    真实字节位置 = offset + 4。读指针统一用 `_take` 推进。
    """

    HEADER_BYTES = 4

    def __init__(self, blob: bytes):
        if len(blob) < 4:
            raise ValueError("CDR payload too short")
        self.data = blob
        # 封装头第 1 字节 = 表示标识（0=CDR_BE, 1=CDR_LE）
        self.big = bool(blob[0] & 0x01) if len(blob) > 1 else False   # CDR_BE 标志
        self.offset = 0        # 相对封装头之后

    # ---- 底层 ----

    def _align(self, size: int) -> None:
        self.offset += (-self.offset) % size

    def _take(self, n: int) -> bytes:
        start = self.offset + self.HEADER_BYTES
        if start + n > len(self.data):
            raise ValueError("CDR payload truncated")
        self.offset += n
        return self.data[start:start + n]

    def _fmt(self, char: str) -> str:
        return (">" if self.big else "<") + char

    # ---- 类型 ----

    def u8(self) -> int:
        return struct.unpack(self._fmt("B"), self._take(1))[0]

    def boolean(self) -> bool:
        return self.u8() != 0

    def u32(self) -> int:
        self._align(4)
        return struct.unpack(self._fmt("I"), self._take(4))[0]

    def time(self) -> float:
        self._align(4)
        sec = struct.unpack(self._fmt("i"), self._take(4))[0]
        nsec = struct.unpack(self._fmt("I"), self._take(4))[0]
        return sec + nsec * 1e-9

    def floats(self, n: int) -> np.ndarray:
        self._align(8)
        raw = self._take(8 * n)
        return np.frombuffer(raw, dtype=np.dtype(self._fmt("d")), count=n)

    def string(self) -> str:
        length = self.u32()
        if length == 0:
            return ""
        raw = self._take(length)
        if raw.endswith(b"\x00"):
            raw = raw[:-1]
        return raw.decode("utf-8", "replace")

    def array_len(self) -> int:
        return self.u32()

    def array_bytes(self, count: int) -> bytes:
        return self._take(count)


def parse_odometry(blob: bytes) -> OdometrySample:
    r = CdrReader(blob)
    stamp = r.time()                      # Header.stamp（seq 已被 ROS2 删除）
    frame_id = r.string()
    child_frame_id = r.string()
    position = r.floats(3)                # Pose.pose.position
    orientation = r.floats(4)             # Pose.pose.orientation
    return OdometrySample(stamp=stamp, position=position, orientation=orientation,
                          frame_id=frame_id, child_frame_id=child_frame_id)


def parse_pointcloud2(blob: bytes) -> PointCloudFrame:
    r = CdrReader(blob)
    stamp = r.time()
    frame_id = r.string()
    height = r.u32()
    width = r.u32()
    nfields = r.u32()
    fields = []
    for _ in range(nfields):
        name = r.string()
        offset = r.u32()
        datatype = r.u8()
        count = r.u32()
        fields.append((name, offset, datatype, count))
    is_bigendian = r.boolean()
    point_step = r.u32()
    row_step = r.u32()
    data_len = r.array_len()
    r._align(1)                           # uint8 数组按 1 对齐
    raw = r.array_bytes(data_len)

    npts = (height * width) if point_step == 0 else data_len // point_step
    if npts == 0:
        return PointCloudFrame(stamp=stamp, xyz=np.zeros((0, 3), np.float32),
                               frame_id=frame_id)

    by_name = {f[0]: f for f in fields}
    missing = [k for k in ("x", "y", "z") if k not in by_name]
    if missing:
        raise ValueError(f"PointCloud2 缺字段 {missing}（有 {[f[0] for f in fields]}）")
    order = ">" if (is_bigendian or r.big) else "<"
    # 把数据看成 (npts, point_step) 的 uint8 矩阵，再按字段偏移取列：
    # 比 as_strided 稳，且不依赖字段之间的内存布局巧合。
    grid = np.frombuffer(raw, dtype=np.uint8, count=npts * point_step)
    grid = grid.reshape(npts, point_step)
    xyz = np.empty((npts, 3), dtype=np.float32)
    for column, key in enumerate(("x", "y", "z")):
        _name, offset, datatype, _count = by_name[key]
        fmt, width = _DT[datatype]
        cells = np.ascontiguousarray(grid[:, offset:offset + width])
        xyz[:, column] = cells.view(np.dtype(order + fmt)).ravel()
    return PointCloudFrame(stamp=stamp, xyz=xyz, frame_id=frame_id)


# ------------------------------------------------------------------ bag 打开

def _parse_metadata(path: Path) -> dict:
    """极简 metadata.yaml 读取（只取需要的键，避免硬依赖 pyyaml）。"""
    if not path.is_file():
        return {"relative_file_paths": [], "topics": []}
    text = path.read_text(encoding="utf-8", errors="replace")
    info = {"relative_file_paths": [], "topics": []}
    match = re.search(r"relative_file_paths:\s*(.*?)(?:\n\s*\w+:|\Z)", text, re.S)
    if match:
        block = match.group(1)
        info["relative_file_paths"] = re.findall(r"-\s*['\"]?([^'\"\s]+\.db3)['\"]?", block)
    for name, kind in re.findall(r"name:\s*['\"]?([^'\"\s]+)['\"]?\s*\n\s*type:\s*['\"]?([^'\"\s]+)['\"]?", text):
        if name.startswith("/"):
            info["topics"].append({"topic": name, "type": kind})
    return info


def resolve_storage(path) -> Tuple[List[Path], dict]:
    path = Path(path).expanduser()
    if path.is_dir():
        meta = _parse_metadata(path / "metadata.yaml")
        files = [path / name for name in meta["relative_file_paths"]]
        files = [f for f in files if f.is_file()] or sorted(path.glob("*.db3"))
        return files, meta
    if path.is_file() and path.suffix == ".db3":
        return [path], _parse_metadata(path.parent / "metadata.yaml")
    if path.is_file() and path.suffix.lower().startswith(".mcap"):
        return [path], {"storage": "mcap", "relative_file_paths": []}
    raise FileNotFoundError(f"不是有效的 ROS2 bag 路径：{path}")


def _connect(db3: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db3}?mode=ro", uri=True)
    try:
        conn.execute("SELECT id, name, type FROM topics").fetchall()
    except sqlite3.DatabaseError as exc:
        conn.close()
        raise ValueError(f"{db3}: 不是 ROS2 sqlite3 bag（{exc}）") from exc
    return conn


def _topic_table(conn: sqlite3.Connection) -> dict:
    return {int(i): (name, kind)
            for i, name, kind in conn.execute("SELECT id, name, type FROM topics")}


def list_topics(bag_path) -> List[dict]:
    files, _meta = resolve_storage(bag_path)
    out: dict = {}
    for db3 in files:
        if db3.suffix != ".db3":
            continue
        conn = _connect(db3)
        try:
            out.update({name: kind for name, kind in _topic_table(conn).values()})
        finally:
            conn.close()
    return [{"topic": k, "type": v} for k, v in sorted(out.items())]


def _iter_rows(files: List[Path], wanted: dict, start_time, stop_time, chunk_size):
    for db3 in files:
        conn = _connect(db3)
        try:
            id_to_key = {tid: wanted[name]
                         for tid, (name, _k) in _topic_table(conn).items() if name in wanted}
            if not id_to_key:
                continue
            ids = sorted(id_to_key)
            sql = (f"SELECT topic_id, timestamp, data FROM messages "
                   f"WHERE topic_id IN ({','.join('?' * len(ids))})")
            params: List = list(ids)
            if start_time is not None:
                sql += " AND timestamp >= ?"
                params.append(int(start_time * 1e9))
            if stop_time is not None:
                sql += " AND timestamp <= ?"
                params.append(int(stop_time * 1e9))
            sql += " ORDER BY timestamp"
            cursor = conn.execute(sql, params)
            while True:
                rows = cursor.fetchmany(chunk_size)
                if not rows:
                    break
                for topic_id, ts, blob in rows:
                    yield id_to_key[int(topic_id)], ts * 1e-9, blob
        finally:
            conn.close()


def iter_bag(bag_path, odom_topic: Optional[str], cloud_topic: Optional[str],
             start_time: Optional[float] = None, stop_time: Optional[float] = None,
             chunk_size: int = 512) -> Iterator[Tuple[str, float, object]]:
    """按时间顺序产出 ("odom" | "cloud", stamp, 对象)。"""
    files, _meta = resolve_storage(bag_path)
    if not files:
        raise FileNotFoundError(f"{bag_path}: 找不到 .db3 文件")
    if files[0].suffix != ".db3":
        raise NotImplementedError(
            "mcap 读取尚未实现；请先 `ros2 bag convert` 转成 sqlite3（.db3）")

    wanted: dict = {}
    if odom_topic:
        wanted[odom_topic] = "odom"
    if cloud_topic:
        wanted[cloud_topic] = "cloud"

    available: dict = {}
    for db3 in files:
        conn = _connect(db3)
        try:
            available.update({name: kind for name, kind in _topic_table(conn).values()})
        finally:
            conn.close()
    for topic, kind_expected in list(wanted.items()):
        if topic not in available:
            raise KeyError(f"bag 里没有话题 {topic}（现有：{sorted(available)}）")
        expected = ODOMETRY_TYPE if kind_expected == "odom" else POINTCLOUD2_TYPE
        if available[topic] != expected:
            raise TypeError(f"话题 {topic} 类型是 {available[topic]}，需要 {expected}")

    for key, stamp, blob in _iter_rows(files, wanted, start_time, stop_time, chunk_size):
        yield key, stamp, (parse_odometry(blob) if key == "odom" else parse_pointcloud2(blob))
