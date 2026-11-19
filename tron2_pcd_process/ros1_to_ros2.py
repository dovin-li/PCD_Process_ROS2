"""`pcd_process_ros1_to_ros2` 入口：把 ROS1 bag 转成 ROS2 bag。

真正的实现在 `tools/ros1_to_ros2_bag.py`（连同 `tools/ros1_ros2_serde.py`）。
那两个文件被安装到 `share/tron2_pcd_process/tools/`，本模块负责找到它们并转发。

为什么要有这一层：`setup.py` 的 console_scripts 必须指向**包内**的可导入模块，
而转换器要保持"不装 ROS 也能当脚本跑"（`python3 tools/ros1_to_ros2_bag.py --bag ...`），
所以做成一层薄包装，两条路都通。
"""

import runpy
import sys
from pathlib import Path


def _tools_dir() -> Path:
    """优先用安装后的 share/tools（ros2 run 场景），退回源码目录（开发场景）。"""
    import ament_index_python  # 只有装了 ROS 才有；源码目录下会失败，走 except

    share = ament_index_python.get_package_share_directory("tron2_pcd_process")
    candidate = Path(share) / "tools"
    if (candidate / "ros1_to_ros2_bag.py").is_file():
        return candidate
    raise FileNotFoundError(candidate)


def _locate() -> Path:
    try:
        return _tools_dir()
    except Exception:  # noqa: BLE001 - 没有 ament 就走源码路径
        # 普通 pip/venv 安装的数据文件位于 <sys.prefix>/share/...，不经过 ament 索引。
        installed = Path(sys.prefix) / "share" / "tron2_pcd_process" / "tools"
        if (installed / "ros1_to_ros2_bag.py").is_file():
            return installed
        here = Path(__file__).resolve()
        # 源码布局：<pkg>/tron2_pcd_process/ros1_to_ros2.py → <pkg>/tools/
        candidate = here.parent.parent / "tools"
        if (candidate / "ros1_to_ros2_bag.py").is_file():
            return candidate
        raise SystemExit(
            f"找不到转换器脚本：{installed} 或 {candidate}\n"
            "请确认 package 安装完整，或直接从源码运行 tools/ros1_to_ros2_bag.py")


def main(argv=None) -> int:
    tools = _locate()
    script = tools / "ros1_to_ros2_bag.py"
    # runpy 让脚本里的 __main__ 分支正常执行，同时保持它自己的 sys.path 处理
    sys.argv = [str(script)] + list(sys.argv[1:] if argv is None else argv)
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
