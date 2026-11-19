"""ROS1 → ROS2 序列化单测：把转换器踩过的坑逐条钉死。

这些用例是回归防线——任何一个被改坏，转换出来的 db3 会在
`ros2 bag play` 或 `pcd_process_bag` 里静默产生错值（不是报错），
所以必须用"已知正确答案"的字节级断言来锁。
"""

import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import ros1_ros2_serde as serde  # noqa: E402
from tron2_pcd_process.bag_reader import CdrReader, parse_odometry, parse_pointcloud2  # noqa: E402

SPECS = serde.build_specs(None)


# ---------------------------------------------------------------- 构造样例

def _ros1_time(sec: int, nsec: int) -> bytes:
    return struct.pack("<iI", sec, nsec)


def _ros1_string(text: str, nul: bool = False) -> bytes:
    raw = text.encode() + (b"\x00" if nul else b"")
    return struct.pack("<I", len(raw)) + raw


def build_odometry_bytes(seq=312, sec=1793340989, nsec=756613016,
                         frame="camera_init", child="body",
                         pos=(-0.0404, 0.0117, 0.0128), quat=(-0.0005, 0.0008, -0.001, 0.99999),
                         nul=False) -> bytes:
    """按 ROS1 native 布局（不对齐、字符串不含 NUL）拼一条 Odometry。"""
    out = bytearray()
    out += struct.pack("<I", seq)
    out += _ros1_time(sec, nsec)
    out += _ros1_string(frame, nul)
    out += _ros1_string(child, nul)
    out += struct.pack("<3d", *pos)
    out += struct.pack("<4d", *quat)
    out += struct.pack("<36d", *([0.0] * 36))       # pose.covariance
    out += struct.pack("<6d", *([0.0] * 6))         # twist
    out += struct.pack("<36d", *([0.0] * 36))       # twist.covariance
    return bytes(out)


def build_pointcloud2_bytes(points, frame="camera_init", nul=False,
                            sec=1793340989, nsec=756613016) -> bytes:
    """按 ROS1 native 布局拼一条 PointCloud2（x/y/z/intensity，point_step=16）。"""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    data = bytearray()
    for x, y, z in points:
        data += struct.pack("<4f", float(x), float(y), float(z), 0.0)
    names = [("x", 0), ("y", 4), ("z", 8), ("intensity", 12)]

    out = bytearray()
    out += struct.pack("<I", 1)                     # seq
    out += _ros1_time(sec, nsec)
    out += _ros1_string(frame, nul)
    out += struct.pack("<II", 1, len(points))       # height, width
    out += struct.pack("<I", len(names))            # nfields
    for name, offset in names:
        out += _ros1_string(name, nul)
        out += struct.pack("<IBI", offset, 7, 1)    # datatype 7 = FLOAT32
    out += struct.pack("<B", 0)                     # is_bigendian
    out += struct.pack("<II", 16, len(data))        # point_step, row_step
    out += struct.pack("<I", len(data))             # ← ROS1 的变长数组长度前缀
    out += bytes(data)
    out += struct.pack("<B", 1)                     # is_dense
    return bytes(out)


# ---------------------------------------------------------------- 用例

def test_layout_detected_as_native_without_nul():
    samples = [build_pointcloud2_bytes(np.random.default_rng(i).normal(size=(50, 3))) 
               for i in range(3)]
    mode, has_nul = serde.detect_layout(SPECS, samples)
    assert mode == "native", f"应识别为 native 布局，实际 {mode}"
    assert has_nul is False, "字符串长度前缀不应包含 NUL"


def test_odometry_roundtrip_values():
    """坑 1：Header.seq 要跳过（且在字段循环里不能再读一次）。"""
    blob = build_odometry_bytes()
    cdr = serde.Walker(SPECS).convert(SPECS["nav_msgs/msg/Odometry"], blob,
                                      mode="native", string_has_nul=False)
    odom = parse_odometry(cdr)
    assert odom.frame_id == "camera_init"
    assert odom.child_frame_id == "body"
    assert abs(odom.stamp - (1793340989 + 756613016 * 1e-9)) < 1e-9
    assert np.allclose(odom.position, [-0.0404, 0.0117, 0.0128], atol=1e-9)


def test_cdr_header_has_encapsulation_and_starts_with_stamp():
    """CDR 输出必须以 00 01 00 00 开头，紧接着是 stamp（seq 已删除）。"""
    blob = build_odometry_bytes()
    cdr = serde.Walker(SPECS).convert(SPECS["nav_msgs/msg/Odometry"], blob,
                                      mode="native", string_has_nul=False)
    assert cdr[:4] == b"\x00\x01\x00\x00"
    sec, nsec = struct.unpack_from("<iI", cdr, 4)
    assert (sec, nsec) == (1793340989, 756613016)
    # ROS1 715 → CDR 724：
    #   去 Header.seq(-4) / 补封装头(+4) / 两个字符串各补 NUL(+2)
    #   / child_frame_id 前补 4 字节对齐填充(+4) / framework 尾对齐(+3)
    assert len(cdr) == 724


def test_string_lengths_include_nul_in_cdr():
    blob = build_odometry_bytes(frame="camera_init")
    cdr = serde.Walker(SPECS).convert(SPECS["nav_msgs/msg/Odometry"], blob, "native", False)
    # 封装头 4 + stamp 8 之后是 frame_id 的长度前缀
    length = struct.unpack_from("<I", cdr, 12)[0]
    assert length == len(b"camera_init") + 1, "CDR 字符串长度必须含结尾 NUL"
    assert cdr[16:16 + length] == b"camera_init\x00"


def test_pointcloud2_roundtrip_values():
    """坑 2/3/4：data 长度前缀、数组对齐、datatype 7 = FLOAT32。"""
    rng = np.random.default_rng(11)
    points = rng.uniform(-20, 20, size=(37, 3)).astype(np.float32)
    blob = build_pointcloud2_bytes(points)
    cdr = serde.Walker(SPECS).convert(SPECS["sensor_msgs/msg/PointCloud2"], blob,
                                      mode="native", string_has_nul=False)
    frame = parse_pointcloud2(cdr)
    assert frame.frame_id == "camera_init"
    assert len(frame) == len(points)
    assert np.allclose(frame.xyz, points, atol=1e-6), "点云坐标必须逐值一致"


def test_pointcloud2_datatype_map_is_float32():
    """坑 4：datatype 7 是 FLOAT32（4 字节），写成 int64 会解析出 1e19 的荒谬值。"""
    from tron2_pcd_process.bag_reader import _DT

    assert _DT[7] == ("f", 4)
    assert _DT[8] == ("d", 8)
    assert _DT[1] == ("b", 1) and _DT[2] == ("B", 1)


def test_cdr_writer_alignment_is_relative_to_encapsulation_header():
    """坑 5：对齐基准是封装头之后。"""
    w = serde.CdrWriter()
    w.pack("i", 1)                      # body offset 0..4
    w.pack("I", 2)                      # 4..8
    w.pack("d", 3.0)                    # 需要 8 对齐 → body 8，无填充
    assert len(w.buf) == 4 + 8 + 8
    w2 = serde.CdrWriter()
    w2.pack("i", 1)                     # body 0..4
    w2.pack("d", 2.0)                   # 需要 body 8 对齐 → 补 4 字节
    assert len(w2.buf) == 4 + 4 + 4 + 8


def test_cdr_reader_honors_little_endian_flag():
    """坑 6：封装头第 1 字节最低位 = CDR 表示标志。"""
    from tron2_pcd_process.bag_reader import CdrReader as Reader

    reader = Reader(b"\x00\x01\x00\x00" + struct.pack("<iI", 100, 200))
    assert reader.big is False
    assert reader.time() == 100 + 200e-9


def test_detect_layout_rejects_garbage():
    """判据必须是数学必然而不是"能读完"：损坏数据要报错而不是猜。"""
    try:
        serde.detect_layout(SPECS, [b"\x00" * 200])
    except serde.LayoutMismatch:
        return
    raise AssertionError("损坏数据必须抛出 LayoutMismatch")


def test_console_script_wrapper_resolves_tools():
    """setup.py 注册的 pcd_process_ros1_to_ros2 必须能找到真实的转换器脚本。

    没有这条测试时，入口点指向了一个不存在的模块，`ros2 run` 会直接崩。
    """
    from tron2_pcd_process import ros1_to_ros2

    tools = ros1_to_ros2._locate()
    assert (tools / "ros1_to_ros2_bag.py").is_file()
    assert (tools / "ros1_ros2_serde.py").is_file()


def test_all_console_scripts_resolve():
    """三个 console_scripts 都要能定位到真实的 main。"""
    import importlib

    from setuptools import find_packages  # noqa: F401  (仅确认环境)

    root = Path(__file__).resolve().parent.parent
    setup_text = (root / "setup.py").read_text()
    # console_scripts 的写法是 "'name = module.path:function'"
    targets = []
    for line in setup_text.splitlines():
        if " = " not in line or ":" not in line:
            continue
        quoted = line.strip().strip('",')
        if quoted.count(":") != 1:
            continue
        spec = quoted.split(" = ", 1)[-1]
        module, _, func = spec.partition(":")
        if module.startswith("tron2_pcd_process"):
            targets.append((module, func))
    assert len(targets) >= 3, f"setup.py 里只找到 {targets}"
    for module, func in targets:
        mod = importlib.import_module(module)
        entry = getattr(mod, func, None)
        assert callable(entry), f"{module}:{func} 不存在或不可调用"


if __name__ == "__main__":
    import traceback

    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failed = 0
    for name, func in tests:
        try:
            func()
            print(f"PASS {name}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
