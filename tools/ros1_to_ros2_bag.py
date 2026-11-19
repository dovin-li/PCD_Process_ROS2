#!/usr/bin/env python3
"""ROS1 bag → ROS2 bag（sqlite3 .db3 + metadata.yaml）。

用途：把历史 ROS1 数据（如 `~/Desktop/1030/scan_0.bag`）转成 ROS2 bag，
喂给 `pcd_process_bag`，并和 ROS1 版 `point_process` 的产物做逐项数值对照。

本机（Mac）没有 ROS1/ROS2，所以这里从零实现了两端：

    * ROS1 bag v2.0 解析：chunk 解压（none / bz2 / lz4）+ 记录头解析
      （逻辑与 ROS1 版 bag_mapping_pipeline.py 相同，含"头长度为 0"的修复）
    * 消息反序列化 → 再序列化：ROS1 TCPROS 格式 → ROS2 CDR 格式，
      字段定义来自 tools/ros1_ros2_serde.py（可扫真实 .msg，也可用内置定义）

默认转换 /Odometry 与 /cloud_registered（PCD Process 的唯一输入契约）。
其余话题能用内置消息定义转换就一并转，转不了就跳过并列出原因。

用法：
    python3 tools/ros1_to_ros2_bag.py --bag scan_0.bag --out scan_0_ros2 \
        --topics /Odometry /cloud_registered
"""

import argparse
import bz2
import json
import sqlite3
import struct
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ros1_ros2_serde import (  # noqa: E402
    MsgSpec,
    Walker,
    build_specs,
    detect_layout,
    normalize_name,
)

MSG_DATA = 2
BAG_HEADER = 3
BAG_INDEX = 4
BAG_CHUNK = 5
BAG_CONNECTION = 7

DEFAULT_TOPICS = ("/Odometry", "/cloud_registered")


# ---------------------------------------------------------------- ROS1 解析

def _parse_header(blob: bytes) -> dict:
    fields, offset = {}, 0
    while offset < len(blob):
        (length,) = struct.unpack_from("<I", blob, offset)
        offset += 4
        name, _, value = blob[offset:offset + length].partition(b"=")
        offset += length
        fields[name.decode()] = value
    return fields


def _iter_records(blob: bytes) -> Iterator[Tuple[dict, bytes]]:
    offset = 0
    while offset + 4 <= len(blob):
        (header_len,) = struct.unpack_from("<I", blob, offset)
        offset += 4
        header = _parse_header(blob[offset:offset + header_len])
        offset += header_len
        (data_len,) = struct.unpack_from("<I", blob, offset)
        offset += 4
        yield header, blob[offset:offset + data_len]
        offset += data_len


def _read_record(stream) -> Tuple[Optional[dict], bytes]:
    raw = stream.read(4)
    if len(raw) < 4:
        return None, b""
    (header_len,) = struct.unpack("<I", raw)
    if header_len == 0:
        # 头长度为 0 = 已走出记录区（录制被强杀 / reindex 后尾部空洞）。
        # 正常 ROS1 记录头长度不可能为 0；不做这个判断会 KeyError: 'op'。
        return None, b""
    header = _parse_header(stream.read(header_len))
    (data_len,) = struct.unpack("<I", stream.read(4))
    payload = stream.read(data_len)
    if len(payload) < data_len:
        return None, b""
    return header, payload


def _decompress(header: dict, payload: bytes) -> bytes:
    kind = header.get("compression", b"none")
    if kind in (b"none", b""):
        return payload
    if kind in (b"\x02", b"bz2"):
        return bz2.decompress(payload)
    if kind == b"lz4":
        try:
            import lz4.frame  # type: ignore
        except ImportError as exc:
            raise SystemExit("bag 用 lz4 压缩，需要 pip install lz4") from exc
        return lz4.frame.decompress(payload)
    raise ValueError(f"不支持的 chunk 压缩：{kind!r}")


def scan_connections(path: Path) -> Dict[int, Tuple[str, str]]:
    """读 connection 表：conn_id → (topic, ros1_type)。"""
    with path.open("rb") as stream:
        stream.readline()
        header, _ = _read_record(stream)
        if header is None or "index_pos" not in header:
            raise SystemExit(f"{path}: 不是有效的 ROS1 bag（缺 bag header）")
        index_pos = struct.unpack("<Q", header["index_pos"])[0]
        stream.seek(index_pos)
        conns: Dict[int, Tuple[str, str]] = {}
        while True:
            header, payload = _read_record(stream)
            if header is None:
                break
            if header.get("op", b"\x00")[0] == BAG_CONNECTION:
                (conn_id,) = struct.unpack("<I", header["conn"])
                info = _parse_header(payload)
                conns[conn_id] = (info["topic"].decode(), info["type"].decode())
    return conns


def iter_messages(path: Path, wanted_ids) -> Iterator[Tuple[int, float, bytes]]:
    """按文件顺序产出 (conn_id, stamp, ros1_bytes)。"""
    with path.open("rb") as stream:
        stream.readline()
        _read_record(stream)                     # bag header
        while True:
            header, payload = _read_record(stream)
            if header is None:
                break
            if header.get("op", b"\x00")[0] != BAG_CHUNK:
                continue
            if "compression" in header and header["compression"] not in (b"none", b""):
                payload = _decompress(header, payload)
            for inner, data in _iter_records(payload):
                if inner.get("op", b"\x00")[0] != MSG_DATA:
                    continue
                (conn_id,) = struct.unpack("<I", inner["conn"])
                if conn_id not in wanted_ids:
                    continue
                sec, nsec = struct.unpack_from("<II", inner["time"])
                yield conn_id, sec + nsec * 1e-9, data


# ---------------------------------------------------------------- 写 ROS2

def _create_db3(path: Path) -> sqlite3.Connection:
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=MEMORY")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("CREATE TABLE schema(id INTEGER PRIMARY KEY, schema_version INTEGER, ros_distro TEXT)")
    conn.execute("INSERT INTO schema VALUES(3, 3, 'humble')")
    conn.execute("CREATE TABLE metadata(id INTEGER PRIMARY KEY, metadata_version INTEGER, "
                 "metadata TEXT)")
    conn.execute("INSERT INTO metadata VALUES(1, 1, ?)",
                 (json.dumps({"offered_qos_profiles": ""}),))
    conn.execute("CREATE TABLE topics(id INTEGER PRIMARY KEY, name TEXT NOT NULL, "
                 "type TEXT NOT NULL, serialization_format TEXT NOT NULL, "
                 "offered_qos_profiles TEXT NOT NULL)")
    conn.execute("CREATE TABLE messages(id INTEGER PRIMARY KEY, topic_id INTEGER NOT NULL, "
                 "timestamp INTEGER NOT NULL, data BLOB NOT NULL)")
    return conn


def ros1_to_ros2(bag: Path, out: Path, topics: List[str],
                 ros_share: Optional[Path] = None,
                 limit: int = 0) -> dict:
    started = time.time()
    specs: Dict[str, MsgSpec] = build_specs(ros_share)
    conns = scan_connections(bag)
    wanted = {}
    for conn_id, (topic, kind) in conns.items():
        if topic in topics:
            wanted[conn_id] = (topic, kind)
    missing = [t for t in topics if t not in {topic for topic, _ in conns.values()}]
    if missing:
        raise SystemExit(f"bag 里没有这些话题：{missing}"
                         f"（现有 {sorted({t for t, _ in conns.values()})}）")

    out.mkdir(parents=True, exist_ok=True)
    db3 = out / f"{out.name}_0.db3"
    conn = _create_db3(db3)

    topic_ids: Dict[str, int] = {}
    for conn_id, (topic, kind) in sorted(wanted.items(), key=lambda kv: kv[1][0]):
        ros2_type = normalize_name(kind)
        if ros2_type not in specs:
            raise SystemExit(f"缺少 {ros2_type} 的消息定义，无法转换 {topic}")
        topic_ids[topic] = len(topic_ids) + 1
        conn.execute("INSERT INTO topics(id, name, type, serialization_format, "
                     "offered_qos_profiles) VALUES(?,?,?,?,?)",
                     (topic_ids[topic], topic, ros2_type, "cdr", ""))
    conn.commit()

    # 先探测 ROS1 字节布局（对齐 / 字符串 NUL / 数组前缀），全 bag 一致
    probe_id = next((cid for cid, (_t, k) in wanted.items()
                     if normalize_name(k) == "sensor_msgs/msg/PointCloud2"), None)
    probes = []
    if probe_id is not None:
        for cid, _stamp, data in iter_messages(bag, {probe_id}):
            probes.append(data)
            if len(probes) >= 3:
                break
    mode, string_has_nul = detect_layout(specs, probes)
    walker = Walker(specs)

    stats = {topic: 0 for topic in topic_ids}
    t0 = t_min = t_max = None
    batch = []
    max_count = limit or None
    for conn_id, stamp, data in iter_messages(bag, set(wanted)):
        topic, kind = wanted[conn_id]
        ros2_type = normalize_name(kind)
        payload = walker.convert(specs[ros2_type], data, mode, string_has_nul)
        timestamp_ns = int(round(stamp * 1e9))
        batch.append((topic_ids[topic], timestamp_ns, payload))
        stats[topic] += 1
        t0 = stamp if t0 is None else t0
        t_min = stamp if t_min is None else min(t_min, stamp)
        t_max = stamp if t_max is None else max(t_max, stamp)
        if len(batch) >= 2000:
            conn.executemany("INSERT INTO messages(topic_id, timestamp, data) VALUES(?,?,?)",
                             batch)
            batch.clear()
        if max_count is not None and sum(stats.values()) >= max_count:
            break
    if batch:
        conn.executemany("INSERT INTO messages(topic_id, timestamp, data) VALUES(?,?,?)", batch)
    conn.execute("CREATE INDEX timestamp_idx ON messages(timestamp ASC)")
    conn.commit()
    conn.close()

    # metadata.yaml 必须严格符合 ROS2 的 rosbag2_bagfile_information 结构，
    # 否则 `ros2 bag play` 会报 "invalid node; first invalid key: starting_time"。
    # 关键点：duration / starting_time 都是嵌套 {"nanoseconds": N} 结构，
    # 不能写成标量或别的键名。
    duration_ns = int(round((t_max - t_min) * 1e9)) if t0 else 0
    start_ns = int(round((t_min or 0) * 1e9))
    metadata = {
        "rosbag2_bagfile_information": {
            "version": 5,
            "storage_identifier": "sqlite3",
            "duration": {"nanoseconds": duration_ns},
            "starting_time": {"nanoseconds_since_epoch": start_ns},
            "message_count": int(sum(stats.values())),
            "topics_with_message_count": [
                {"topic_metadata": {"name": topic,
                                    "type": _topic_type(topic, wanted),
                                    "serialization_format": "cdr",
                                    "offered_qos_profiles": ""},
                 "message_count": int(count)}
                for topic, count in stats.items()],
            "compression_format": "",
            "compression_mode": "",
            "relative_file_paths": [db3.name],
            "files": [{"path": db3.name,
                       "starting_time": {"nanoseconds_since_epoch": start_ns},
                       "duration": {"nanoseconds": duration_ns},
                       "message_count": int(sum(stats.values()))}],
            "custom_data": {},
        }
    }
    (out / "metadata.yaml").write_text(_dump_yaml(metadata), encoding="utf-8")
    return {
        "ros1_layout": {"mode": mode, "string_has_nul": string_has_nul},
        "bag": str(bag),
        "out": str(out),
        "db3": str(db3),
        "size_bytes": db3.stat().st_size,
        "topics": stats,
        "duration_s": round((t_max - t_min), 3) if t0 else 0.0,
        "elapsed_s": round(time.time() - started, 1),
    }


def _topic_type(topic: str, wanted) -> str:
    for _conn_id, (name, kind) in wanted.items():
        if name == topic:
            return normalize_name(kind)
    return ""


def _dump_yaml(obj, indent: int = 0) -> str:
    """够用的 YAML 输出（dict / list / 标量），避免硬依赖 pyyaml。"""
    pad = "  " * indent
    if isinstance(obj, dict):
        lines = []
        for key, value in obj.items():
            if isinstance(value, (dict, list)) and value:
                lines.append(f"{pad}{key}:")
                lines.append(_dump_yaml(value, indent + 1))
            elif isinstance(value, list) and not value:
                lines.append(f"{pad}{key}: []")
            elif isinstance(value, str) and value == "":
                lines.append(f"{pad}{key}: ''")
            else:
                lines.append(f"{pad}{key}: {_scalar(value)}")
        return "\n".join(lines)
    if isinstance(obj, list):
        lines = []
        for item in obj:
            if isinstance(item, dict):
                chunk = _dump_yaml(item, indent + 1).splitlines()
                lines.append(f"{pad}- {chunk[0].strip()}")
                lines.extend(chunk[1:])
            else:
                lines.append(f"{pad}- {_scalar(item)}")
        return "\n".join(lines)
    return f"{pad}{_scalar(obj)}"


def _scalar(value) -> str:
    if isinstance(value, str):
        return value if value else "''"
    return str(value)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="ROS1 bag → ROS2 bag (.db3)")
    parser.add_argument("--bag", required=True, help="ROS1 .bag 文件")
    parser.add_argument("--out", required=True, help="输出 ROS2 bag 目录")
    parser.add_argument("--topics", nargs="*", default=list(DEFAULT_TOPICS))
    parser.add_argument("--ros-share", default=None,
                        help="ROS2 安装的 share 目录（扫 .msg 用；不给则用内置定义）")
    parser.add_argument("--limit", type=int, default=0, help="只转前 N 条消息（调试用）")
    args = parser.parse_args(argv)
    result = ros1_to_ros2(Path(args.bag).expanduser().resolve(),
                          Path(args.out).expanduser().resolve(),
                          list(args.topics),
                          Path(args.ros_share) if args.ros_share else None,
                          limit=args.limit)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
