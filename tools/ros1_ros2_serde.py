"""ROS1 ⇄ ROS2 消息转码（转换器的测试基础设施，不是产品代码）。

需求场景：在**没有 ROS 环境**的 Mac 上把 ROS1 bag 变成 ROS2 db3，
不能依赖 rosidl 生成的类型。因此这里实现最小可用的两端编解码。

## 两种 ROS1 线上布局（实测都存在）

| 布局 | 对齐 | 字符串长度前缀 |
|---|---|---|
| `aligned` | 字段按自身宽度 4/8 字节对齐（官方 roscpp / rospy） | 不含 NUL |
| `native`  | 完全按写端内存布局推进，不对齐（msg_generation 等生成器） | 含 NUL |

实测 `~/Desktop/1030/scan_0.bag` = **native + 含 NUL**。判定不靠猜：
`detect_layout` 枚举四种组合做完整探测，只有一种能把每条消息**恰好读完**
且读出语义自洽的几何参数（`point_step` 是小整数、`row_step == width*point_step`）。

## ROS2 CDR 写端

4 字节封装头（`00 01 00 00` = 小端），此后每个字段按自身宽度对齐，
字符串 = uint32 长度（含 NUL）+ UTF-8 + NUL，变长数组 = uint32 长度 + 元素。

偏移从**封装头之后**算起是 CDR 的通行约定（Fast CDR / rmw 的实现一致）。
本模块的 `CdrWriter` 显式记录这一基准，避免"从 0 算还是从 4 算"的歧义。
"""

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PRIMITIVES = {
    "bool": "?", "byte": "B", "char": "B",
    "int8": "b", "uint8": "B",
    "int16": "h", "uint16": "H",
    "int32": "i", "uint32": "I",
    "int64": "q", "uint64": "Q",
    "float32": "f", "float64": "d",
}

_RENAME = {"byte": "uint8", "Header": "std_msgs/Header", "time": "builtin_interfaces/Time"}
STATE_KEY = "__state__"


# ------------------------------------------------------------------ 消息定义

@dataclass
class TypeSpec:
    name: str
    kind: str                        # "primitive" | "msg"
    type_name: str
    array: bool = False
    fixed_len: Optional[int] = None
    constant: bool = False


@dataclass
class MsgSpec:
    full_name: str
    fields: List[TypeSpec] = field(default_factory=list)


def normalize_name(type_name: str) -> str:
    """类型串 → ROS2 全名；**原始类型（string / int32 ...）原样返回**。"""
    if type_name in PRIMITIVES or type_name in ("string", "time", "duration"):
        return type_name
    type_name = _RENAME.get(type_name, type_name)
    if type_name == "time":
        return "builtin_interfaces/msg/Time"
    if type_name.startswith("builtin_interfaces/") or "/msg/" in type_name:
        return type_name
    if "/" in type_name:
        package, _, name = type_name.partition("/")
        return f"{package}/msg/{name.split('.')[0]}"
    return "std_msgs/msg/" + type_name


def _read_msg_file(path: Path) -> List[TypeSpec]:
    fields: List[TypeSpec] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("="):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        type_part, name = parts[0], parts[1].strip()
        if "=" in name:
            fields.append(TypeSpec(name.split("=")[0].strip(), "primitive", type_part,
                                   constant=True))
            continue
        is_array, fixed_len = False, None
        if "[" in type_part:
            base, _, rest = type_part.partition("[")
            inner = rest.rstrip("]")
            type_part, is_array = base, True
            fixed_len = int(inner) if inner else None
        fields.append(TypeSpec(name, "primitive" if type_part in PRIMITIVES else "msg",
                               type_part, is_array, fixed_len))
    return fields


def build_specs(ros_share: Optional[Path] = None) -> Dict[str, MsgSpec]:
    if ros_share is None or not Path(ros_share).exists():
        return _builtin_specs()
    specs: Dict[str, MsgSpec] = {}
    for msg_file in Path(ros_share).glob("*/share/*/msg/*.msg"):
        package = msg_file.parent.parent.parent.name
        name = f"{package}/msg/{msg_file.stem}"
        specs[name] = MsgSpec(name, _read_msg_file(msg_file))
    specs.update(_timespec())
    return specs


def _timespec() -> Dict[str, MsgSpec]:
    return {"builtin_interfaces/msg/Time": MsgSpec("builtin_interfaces/msg/Time", [
        TypeSpec("sec", "primitive", "int32"),
        TypeSpec("nanosec", "primitive", "uint32")])}


def _builtin_specs() -> Dict[str, MsgSpec]:
    def p(name, type_name, array=False, fixed_len=None):
        return TypeSpec(name, "primitive", type_name, array, fixed_len)

    def m(name, full, array=False):
        return TypeSpec(name, "msg", full, array)

    specs: Dict[str, MsgSpec] = {
        # 内置定义描述 **ROS1** 布局（Header 含 seq），转码时会跳掉 seq
        "std_msgs/msg/Header": MsgSpec("std_msgs/msg/Header", [
            p("seq", "uint32"),
            m("stamp", "builtin_interfaces/msg/Time"),
            p("frame_id", "string")]),
        "geometry_msgs/msg/Point": MsgSpec("geometry_msgs/msg/Point", [
            p("x", "float64"), p("y", "float64"), p("z", "float64")]),
        "geometry_msgs/msg/Vector3": MsgSpec("geometry_msgs/msg/Vector3", [
            p("x", "float64"), p("y", "float64"), p("z", "float64")]),
        "geometry_msgs/msg/Quaternion": MsgSpec("geometry_msgs/msg/Quaternion", [
            p("x", "float64"), p("y", "float64"), p("z", "float64"), p("w", "float64")]),
        "geometry_msgs/msg/Pose": MsgSpec("geometry_msgs/msg/Pose", [
            m("position", "geometry_msgs/msg/Point"),
            m("orientation", "geometry_msgs/msg/Quaternion")]),
        "geometry_msgs/msg/PoseWithCovariance": MsgSpec(
            "geometry_msgs/msg/PoseWithCovariance", [
                m("pose", "geometry_msgs/msg/Pose"),
                p("covariance", "float64", array=True, fixed_len=36)]),
        "geometry_msgs/msg/Twist": MsgSpec("geometry_msgs/msg/Twist", [
            m("linear", "geometry_msgs/msg/Vector3"),
            m("angular", "geometry_msgs/msg/Vector3")]),
        "geometry_msgs/msg/TwistWithCovariance": MsgSpec(
            "geometry_msgs/msg/TwistWithCovariance", [
                m("twist", "geometry_msgs/msg/Twist"),
                p("covariance", "float64", array=True, fixed_len=36)]),
        "nav_msgs/msg/Odometry": MsgSpec("nav_msgs/msg/Odometry", [
            m("header", "std_msgs/msg/Header"),
            p("child_frame_id", "string"),
            m("pose", "geometry_msgs/msg/PoseWithCovariance"),
            m("twist", "geometry_msgs/msg/TwistWithCovariance")]),
        "sensor_msgs/msg/PointField": MsgSpec("sensor_msgs/msg/PointField", [
            p("name", "string"), p("offset", "uint32"),
            p("datatype", "uint8"), p("count", "uint32")]),
        "sensor_msgs/msg/PointCloud2": MsgSpec("sensor_msgs/msg/PointCloud2", [
            m("header", "std_msgs/msg/Header"),
            p("height", "uint32"), p("width", "uint32"),
            m("fields", "sensor_msgs/msg/PointField", array=True),
            p("is_bigendian", "bool"), p("point_step", "uint32"),
            p("row_step", "uint32"),
            p("data", "uint8", array=True),
            p("is_dense", "bool")]),
        "sensor_msgs/msg/Imu": MsgSpec("sensor_msgs/msg/Imu", [
            m("header", "std_msgs/msg/Header"),
            m("orientation", "geometry_msgs/msg/Quaternion"),
            p("orientation_covariance", "float64", array=True, fixed_len=9),
            m("angular_velocity", "geometry_msgs/msg/Vector3"),
            p("angular_velocity_covariance", "float64", array=True, fixed_len=9),
            m("linear_acceleration", "geometry_msgs/msg/Vector3"),
            p("linear_acceleration_covariance", "float64", array=True, fixed_len=9)]),
        "sensor_msgs/msg/LaserScan": MsgSpec("sensor_msgs/msg/LaserScan", [
            m("header", "std_msgs/msg/Header"),
            p("angle_min", "float32"), p("angle_max", "float32"),
            p("angle_increment", "float32"), p("time_increment", "float32"),
            p("scan_time", "float32"), p("range_min", "float32"),
            p("range_max", "float32"),
            p("ranges", "float32", array=True),
            p("intensities", "float32", array=True)]),
        "sensor_msgs/msg/Image": MsgSpec("sensor_msgs/msg/Image", [
            m("header", "std_msgs/msg/Header"),
            p("height", "uint32"), p("width", "uint32"),
            p("encoding", "string"), p("is_bigendian", "uint8"),
            p("step", "uint32"),
            p("data", "uint8", array=True)]),
    }
    specs.update(_timespec())
    return specs


# ------------------------------------------------------------------ 读端

class LayoutMismatch(Exception):
    """当前布局假设与数据不符。"""


class Ros1Reader:
    """ROS1 读端：mode 决定是否对齐，string_has_nul 决定字符串约定。"""

    def __init__(self, data: bytes, mode: str = "aligned", string_has_nul: bool = False):
        if mode not in ("aligned", "native"):
            raise ValueError(f"unknown ros1 read mode: {mode}")
        self.data = data
        self.offset = 0
        self.mode = mode
        self.string_has_nul = string_has_nul

    def align(self, size: int) -> None:
        if self.mode == "aligned":
            self.offset += (-self.offset) % size

    def take(self, n: int) -> bytes:
        if self.offset + n > len(self.data):
            raise ValueError("ROS1 payload truncated")
        chunk = self.data[self.offset:self.offset + n]
        self.offset += n
        return chunk

    def u32(self) -> int:
        self.align(4)
        return struct.unpack("<I", self.take(4))[0]

    def string(self) -> str:
        length = self.u32()
        raw = self.take(length)
        return (raw[:-1] if self.string_has_nul else raw).decode("utf-8", "replace")


# ------------------------------------------------------------------ 写端

class CdrWriter:
    """ROS2 CDR 写端。偏移基准 = 封装头之后（CDR 约定）。"""

    HEADER = b"\x00\x01\x00\x00"

    def __init__(self):
        self.buf = bytearray(self.HEADER)
        self._unaligned_depth = 0

    def _align(self, size: int) -> None:
        # 对齐基准 = 封装头之后的位置（CDR 约定）
        body = len(self.buf) - len(self.HEADER)
        self.buf.extend(b"\x00" * ((-body) % size))

    def start_submessage(self) -> None:
        """CDR 规则：嵌套消息按自身最大字段宽度（≤8）重新对齐。"""
        if self._unaligned_depth == 0:
            self._align(8)

    def u32(self, value: int) -> None:
        self._align(4)
        self.buf.extend(struct.pack("<I", value))

    def pack(self, fmt: str, value) -> None:
        size = struct.calcsize("<" + fmt)
        self._align(size)
        self.buf.extend(struct.pack("<" + fmt, value))

    def string(self, value: str) -> None:
        # 长度前缀 = 含 NUL 的字节数；前缀本身按 4 字节对齐
        raw = value.encode("utf-8") + b"\x00"
        self.u32(len(raw))
        self.buf.extend(raw)

    def raw(self, data: bytes, align: int = 1) -> None:
        self._align(align)
        self.buf.extend(data)

    def getvalue(self) -> bytes:
        return bytes(self.buf)


class _ProbeSink:
    """顶替 CdrWriter 的只读探针：记录顶层字段值，不产生字节。"""

    def __init__(self):
        self.values: Dict[str, object] = {}

    def add(self, name, value) -> None:
        if name:
            self.values[name] = value

    def u32(self, _v) -> None:
        pass

    def pack(self, _f, _v) -> None:
        pass

    def string(self, _v) -> None:
        pass

    def raw(self, _d, align: int = 1) -> None:  # noqa: ARG002
        pass


# ------------------------------------------------------------------ 转码器

class Walker:
    def __init__(self, specs: Dict[str, MsgSpec]):
        self.specs = specs

    # 转码：ROS1 → CDR
    def convert(self, spec: MsgSpec, blob: bytes, mode: str = "aligned",
                string_has_nul: bool = False) -> bytes:
        src = Ros1Reader(blob, mode, string_has_nul)
        out = CdrWriter()
        self._walk(spec, src, out, drop_seq=True, top=spec)
        if src.offset != len(blob):
            raise LayoutMismatch(f"{spec.full_name}: 剩余 {len(blob) - src.offset} 字节未消费")
        return out.getvalue()

    # 只读探测：只解出顶层标量
    def probe(self, spec: MsgSpec, blob: bytes, mode: str = "aligned",
              string_has_nul: bool = False) -> Dict[str, object]:
        src = Ros1Reader(blob, mode, string_has_nul)
        sink = _ProbeSink()
        self._walk(spec, src, sink, drop_seq=True, top=spec)
        if src.offset != len(blob):
            raise LayoutMismatch(f"{spec.full_name}: 剩余 {len(blob) - src.offset} 字节未消费")
        sink.values["__consumed__"] = src.offset
        return sink.values

    def _walk(self, spec: MsgSpec, src: Ros1Reader, out, drop_seq: bool,
              top: Optional[MsgSpec] = None) -> None:
        if drop_seq and spec.full_name == "std_msgs/msg/Header":
            src.take(4)                        # ROS1 Header.seq：ROS2 已删除
        # 顶层 PointCloud2 需要 width/point_step/row_step 来决定 data 数组长度
        width = point_step = row_step = 0
        for f in spec.fields:
            if f.constant:
                continue
            if drop_seq and spec.full_name == "std_msgs/msg/Header" and f.name == "seq":
                continue                       # 已在上面整体跳过
            if f.kind == "primitive" and f.name == "data" and f.type_name == "uint8" \
                    and f.array:
                # PointCloud2.data：ROS1 无长度前缀，长度 = row_step（或 width*point_step）。
                # ROS2 CDR 需要显式长度前缀，因此带上。
                # ROS1 的 PointCloud2.data 实际**带有** uint32 长度前缀（= row_step），
                # 读走它，再按长度前缀声明的字节数读数据
                src.align(4)
                declared = int.from_bytes(src.take(4), "little")
                n = declared if declared else (int(row_step) or int(width * point_step))
                payload = src.take(n)
                # ROS2 CDR：长度前缀按 4 对齐，数据本身按 1 对齐（u8 元素）
                out.u32(n)   # ← 唯一一处写前缀；_primitive 里对 data 不再写
                out.raw(payload, align=1)
                continue
            if f.kind == "primitive":
                value = self._primitive(f, src, out, (top is not None) and f.name)
                if top is not None:
                    if f.name == "width":
                        width = int(value)
                    elif f.name == "point_step":
                        point_step = int(value)
                    elif f.name == "row_step":
                        row_step = int(value)
                continue
            target = normalize_name(f.type_name)
            if target == "builtin_interfaces/msg/Time":
                # 按 Time 自己的字段逐项读（int32 sec + uint32 nanosec），
                # 交给 _primitive 处理对齐，不硬编码 8 字节
                for ts in self.specs["builtin_interfaces/msg/Time"].fields:
                    self._primitive(ts, src, out, None)
                continue
            sub = self.specs.get(target)
            if sub is None:
                raise KeyError(f"缺少消息定义：{target}")
            if f.array:
                n = f.fixed_len if f.fixed_len is not None else src.u32()
                if f.fixed_len is None:
                    out.u32(n)
                for _ in range(n):
                    self._walk(sub, src, out, drop_seq=(target == "std_msgs/msg/Header"))
            else:
                self._walk(sub, src, out, drop_seq=(target == "std_msgs/msg/Header"))

    def _primitive(self, f: TypeSpec, src: Ros1Reader, out, top_name) -> None:
        if f.type_name == "string":
            if f.array:
                n = f.fixed_len if f.fixed_len is not None else src.u32()
                if f.fixed_len is None:
                    out.u32(n)
                for _ in range(n):
                    out.string(src.string())
                return
            out.string(src.string())
            return

        fmt = PRIMITIVES[f.type_name]
        width = struct.calcsize("<" + fmt)
        if f.array:
            # 变长数组的长度前缀：ROS1 的 uint8[]（如 PointCloud2.data）**没有**前缀，
            # 长度由 width*height*point_step 隐式决定；其余数组类型带 uint32 前缀。
            if f.fixed_len is not None:
                n = f.fixed_len
            elif f.type_name == "uint8":
                n = None                       # 由调用方/上下文决定，见 _read_uint8_tail
            else:
                n = src.u32()
                out.u32(n)
            if n is None:
                return                         # 交给 _walk 的 data 特例
            src.align(width)
            out.raw(src.take(n * width), align=width)
            return
        src.align(width)
        value = struct.unpack("<" + fmt, src.take(width))[0]
        if top_name and hasattr(out, "add"):
            out.add(top_name, value)
        else:
            out.pack(fmt, value)
        return value


# ------------------------------------------------------------------ 布局探测

def _fits(walker: Walker, spec: MsgSpec, blob: bytes, mode: str, has_nul: bool) -> bool:
    """完整探测必须恰好读完，且读出语义自洽的几何参数。

    正确布局下 PointCloud2 必然满足：point_step 是小整数、row_step == width*point_step。
    """
    try:
        values = walker.probe(spec, blob, mode, has_nul)
    except (LayoutMismatch, ValueError, struct.error, KeyError):
        return False
    try:
        width = int(values["width"])
        point_step = int(values["point_step"])
        row_step = int(values["row_step"])
    except (KeyError, TypeError, ValueError):
        return False
    return (0 < point_step <= 1000 and width > 0
            and row_step == width * point_step
            and int(values["__consumed__"]) == len(blob))


def detect_layout(specs: Dict[str, MsgSpec], samples) -> Tuple[str, bool]:
    """探测 (read_mode, string_has_nul)。多帧交叉验证；官方布局优先。"""
    pc_spec = specs.get("sensor_msgs/msg/PointCloud2")
    if pc_spec is None or not samples:
        return "aligned", False
    walker = Walker(specs)
    candidates: List[Tuple[str, bool]] = []
    for mode in ("aligned", "native"):
        for has_nul in (False, True):
            if all(_fits(walker, pc_spec, blob, mode, has_nul) for blob in samples):
                candidates.append((mode, has_nul))
    candidates.sort(key=lambda item: ({"aligned": 0, "native": 1}[item[0]], item[1]))
    if candidates:
        return candidates[0]
    raise LayoutMismatch("无法判定 ROS1 字节布局（bag 损坏或消息定义不符）")


def convert_message(spec: MsgSpec, blob: bytes, specs: Dict[str, MsgSpec],
                    mode: str = "aligned", string_has_nul: bool = False) -> bytes:
    return Walker(specs).convert(spec, blob, mode, string_has_nul)
