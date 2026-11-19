"""用 grid_map 官方插件链把 PCD 变成 Nav2 可用的**连续代价图**。

分工（严格区分）：
  * 高程栅格：本脚本只做 I/O（点→格子的高度），不做任何地形判据
  * 坡度 / 台阶 / 粗糙度 / 通行代价：**全部由 grid_map 官方滤波器计算**（见 chain.yaml）
  * 硬门槛（>30° 或 >0.20 m 判致命）与灰阶映射：本脚本

用法：
    pcd_process_terrain --pcd scene.pcd --out out_dir --run-cmd "..."
其中 --run-cmd 是执行 grid_map 链的命令模板（在装有 grid_map 的机器上跑），
包含 {grid} 与 {layers} 两个占位符。首次生成或更换点云时须执行滤波链；
同一输入只调整代价参数时，可省略该参数以复用经校验的图层。
PCD 读取器来自本仓库，支持 ASCII 和未压缩 binary PCD，并按字段名提取 XYZ；
人工编辑器保留的 intensity、normal 等额外字段不会影响导入。
elevation.npy 是以米为单位的高程，known.npy 标记已观测格，几何见 elevation.yaml。
Nav2 加载 map.yaml 与 map.pgm，所有产品路径都可在 metrics.json 中找到。
"""

import argparse
import fcntl
import hashlib
import json
import re
import shlex
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import yaml
from scipy import ndimage

from tron2_pcd_process.points import read_pcd_xyz as read_project_pcd_xyz

MAX_SLOPE_DEG = 34.0      # 连续坡面的硬上限（≤30° 必须可通行，留 4° 余量）
STEP_AS_STEP_M = 0.15     # 高差 > 此值视为"离散台阶"，不再用坡度判它
MAX_STEP_M = 0.22         # 台阶门槛（实测口径）

OBSTACLE_HEIGHT_M = 0.30     # 格内最高点高出地面多少算硬障碍（与 CMU 同口径）
OBSTACLE_POINT_FRACTION = 0.5  # 离地点占该格点数的比例达到多少算硬障碍

# 注意：这里的 step 是 grid_map MinInRadius（半径 0.30 m）测出的"局部高差"，
# 实测标定（合成场景）：
#   0.20 m 踏步楼梯 = 0.196 | 30° 连续坡 = 0.203   ← 必须通过
#   0.25 m 踏步楼梯 = 0.245 | 35° 连续坡 = 0.246   ← 必须拒绝
# 所以门槛取两者之间的 0.22，而不是直接写 0.20（写 0.20 会把 30° 坡误杀）。
# 换用真实户外楼梯数据后需重新标定这一口径。
ROUGH_SCALE_M = 0.15      # 粗糙度归一化尺度（代价曲线用）
RISER_STEP_M = 0.10       # 高差 > 此值视为"一级踏步"
SLOPE_WEIGHT = 0.50       # 坡度在连续代价里的权重
STEP_WEIGHT = 0.375       # 台阶权重（0.30 × 1.25：按用户要求把高度惩罚加重四分之一）
ROUGH_WEIGHT = 0.20       # 粗糙度权重
RISER_PENALTY = 1.00      # 每跨一级踏步的固定加罚（竖面直接按满代价计）（按面积摊会把楼梯算得比坡道便宜）
RISER_FACE_DEG = 45.0     # 竖面判据：坡度 > 此值才算"踏步竖面"，连续坡面不加罚
STEP_WINDOW_M = 0.20      # 与插件链里 min_elev_small 的半径一致
PROTRUSION_RATIO = 1.20   # 台阶凸起度：实测高差 / 坡度在同一窗口的理论高差
PROTRUSION_MIN_STEP_M = 0.08  # 凸起度判据生效的最小高差（避免把平地噪声当台阶）
LAYER_ORDER = ("elevation_inpainted", "slope", "step", "step_large", "roughness")
MAX_LAYER_ALIGNMENT_MAE_M = 0.10
GROUND_QUANTILE = 0.20
TOP_QUANTILE = 0.95


def load_pcd_xyz(path):
    xyz, _intensity = read_project_pcd_xyz(path)
    finite = np.isfinite(xyz).all(axis=1)
    if not finite.any():
        raise ValueError("PCD 没有有限 XYZ 坐标")
    if not finite.all():
        xyz = xyz[finite]
    return xyz.astype(np.float64, copy=False)


def build_grid(points, resolution, min_points):
    """Build the elevation input grid without any ROS or workspace dependency."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise SystemExit("点云为空或不是 N×3 XYZ")
    xmin = float(np.floor(points[:, 0].min() / resolution) * resolution)
    ymin = float(np.floor(points[:, 1].min() / resolution) * resolution)
    xmax = float(np.ceil(points[:, 0].max() / resolution) * resolution)
    ymax = float(np.ceil(points[:, 1].max() / resolution) * resolution)
    width = int(round((xmax - xmin) / resolution)) + 1
    height = int(round((ymax - ymin) / resolution)) + 1

    gx = np.floor((points[:, 0] - xmin) / resolution).astype(np.int64)
    gy = np.floor((points[:, 1] - ymin) / resolution).astype(np.int64)
    inside = (gx >= 0) & (gx < width) & (gy >= 0) & (gy < height)
    gx, gy, z = gx[inside], gy[inside], points[inside, 2]

    n_cells = width * height
    flat = gy * width + gx
    counts = np.bincount(flat, minlength=n_cells).astype(np.int64)
    order = np.lexsort((z, flat))
    z_sorted = z[order]
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    lo_idx = starts + np.floor(GROUND_QUANTILE * np.maximum(counts - 1, 0)).astype(np.int64)
    hi_idx = starts + np.floor(TOP_QUANTILE * np.maximum(counts - 1, 0)).astype(np.int64)
    lo_idx = np.clip(lo_idx, 0, max(len(z_sorted) - 1, 0))
    hi_idx = np.clip(hi_idx, 0, max(len(z_sorted) - 1, 0))

    ground = np.full((height, width), np.nan, dtype=np.float64)
    top = np.full((height, width), np.nan, dtype=np.float64)
    scaled_min = max(3, int(round(min_points * (resolution / 0.25) ** 2)))
    known = (counts >= scaled_min).reshape(height, width)
    has = counts > 0
    ground.reshape(-1)[has] = z_sorted[lo_idx[has]]
    top.reshape(-1)[has] = z_sorted[hi_idx[has]]

    flat_ground = np.full(n_cells, np.nan)
    flat_ground[known.reshape(-1)] = ground.reshape(-1)[known.reshape(-1)]
    relative_z = z - flat_ground[flat]
    band = np.isfinite(relative_z) & (relative_z > 0.10) & (relative_z < 2.0)
    above = np.bincount(flat[band], minlength=n_cells).reshape(height, width)
    ground[~known] = np.nan
    top[~known] = np.nan
    return ground, top, counts.reshape(height, width), known, above, (xmin, ymin, width, height)

def connectivity_stats(passable_mask, resolution):
    """统计可通行区的连通块，返回 (标注图, 块大小排序列表)。

    为什么必须做：实测老管线导出的交付图里真实可通行格只有 176 格、却碎成 45 块
    （最大块 35 格 ≈ 2.2 m²）。这种图交给 Nav2 只会得到"no valid path"，
    而如果未知区被当成自由，它会"看起来能跑"——这正是历史上闭环验收失真的原因。
    """
    from scipy import ndimage
    labels, n = ndimage.label(passable_mask)
    if n == 0:
        return labels, []
    sizes = ndimage.sum(passable_mask, labels, range(1, n + 1)).astype(int)
    order = np.argsort(sizes)[::-1]
    stats = [(int(order[i] + 1), int(sizes[order[i]]),
              float(sizes[order[i]] * resolution * resolution)) for i in range(len(order))]
    return labels, stats



def _count_above(points, bounds, resolution, ground, shape, height):
    """统计每格内"离地高度 > height"的点数（与硬障碍门限同源）。"""
    rows, cols = shape
    gi = np.floor((points[:, 1] - bounds[1]) / resolution).astype(np.int64)
    gj = np.floor((points[:, 0] - bounds[0]) / resolution).astype(np.int64)
    ok = (gi >= 0) & (gi < rows) & (gj >= 0) & (gj < cols)
    gi, gj = gi[ok], gj[ok]
    z = points[ok, 2]
    gz = np.nan_to_num(ground[gi, gj])
    above = z - gz > height
    flat = gi * cols + gj
    counts = np.bincount(flat[above], minlength=rows * cols)
    return counts.reshape(shape).astype(np.int64)



def reachable_from_ground(elevation, known, max_climb, seed_band=None):
    """从地面出发做可达性扩散：每步允许的爬升不超过 max_climb。

    为什么需要它（审查意见 S6）：所有局内判据（坡度/台阶/硬障碍）都是"同一格内或
    相邻格之间"的比较，**判断不了"这一格是不是地面"**。当一整格都落在桌面上时：
    该格的"地面"取到桌面高度、"最高点"也是桌面高度，两者相减为 0，坡度也是 0
    → 被判成"平坦、低代价、可通行"。桌沿的落差通常会生成一圈致命格把它围住，
    但圈上有缺口（点稀疏/贴墙）时规划器就能把车规划上桌面。

    本函数换一个问法：**机器人能不能从地面爬到这里**。
    - 种子：高程接近全图低分位（地面高度带）的格子
    - 扩散：8 邻域，单步高差 ≤ max_climb（默认=台阶能力 0.22 m）
    楼梯每级 0.20 m ≤ 0.22 → 可达（合规台阶仍能过）；
    桌面 0.75 m > 0.22 → 不可达（判为非地面）。
    返回 (可达掩码, 种子格数)。
    """
    from collections import deque
    # 种子带必须明显小于爬升能力：否则"刚好等于门槛的高台"会自己成为种子，
    # 整片高台直接被判可达（实测 0.30 m 落差在 seed_band=0.30 时整片漏判）。
    if seed_band is None:
        seed_band = 0.5 * max_climb
    rows, cols = elevation.shape
    flat = np.asarray(elevation, dtype=np.float64).ravel()
    valid = np.asarray(known, dtype=bool).ravel()
    reached = np.zeros(flat.shape, bool)
    if not valid.any():
        return reached.reshape(rows, cols), 0
    floor_level = float(np.percentile(flat[valid], 10))
    seeds = valid & (flat <= floor_level + seed_band)
    if not seeds.any():
        return reached.reshape(rows, cols), 0
    reached |= seeds
    dq = deque(np.flatnonzero(seeds).tolist())
    while dq:
        idx = dq.popleft()
        r, c = divmod(idx, cols)
        z = flat[idx]
        for dr in (-1, 0, 1):
            nr = r + dr
            if nr < 0 or nr >= rows:
                continue
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nc = c + dc
                if nc < 0 or nc >= cols:
                    continue
                nidx = nr * cols + nc
                if reached[nidx] or not valid[nidx]:
                    continue
                if abs(flat[nidx] - z) > max_climb:
                    continue
                reached[nidx] = True
                dq.append(nidx)
    return reached.reshape(rows, cols), int(seeds.sum())


def build_elevation(points, resolution, min_points, obstacle_height=MAX_STEP_M,
                    smooth_size=0):
    """只做 I/O：每格取地面高度（20% 分位）与是否有数据。

    未知格用**最近邻**填充（不能用全局中位数：那会在图边界造出假断崖，
    实测把 30° 坡的 step 层算成 5.9 m）。
    """
    from scipy import ndimage
    ground, top, count, known, _above, bounds = build_grid(points, resolution, min_points)
    if known.any():
        idx = ndimage.distance_transform_edt(~known, return_distances=False, return_indices=True)
        filled = np.nan_to_num(ground)[idx[0], idx[1]]
        elevation = np.where(known, np.nan_to_num(ground), filled)
    else:
        elevation = np.zeros_like(ground)
    # bounds 是 (min_x, min_y, max_x, max_y) 或类似元组，取前两个作为网格左下角
    origin = (float(bounds[0]), float(bounds[1]))
    # 地面微噪声去噪：老管线在做判据前对地面图做 3×3 中值，新管线曾直接把原始地面
    # 喂给滤波链 → 真实点云（0.10 m 网格每格约 2~3 点）的单格抖动被当成台阶/粗糙度，
    # 实测导致 91% 的格子代价饱和、地图被均匀高代价糊住并出现周期性条纹。
    # 中值能保住真实台阶棱边（阶跃不被抹平），只压单格噪声。
    if smooth_size and smooth_size >= 3:
        cleaned = ndimage.median_filter(np.nan_to_num(elevation), size=smooth_size, mode="nearest")
        elevation = np.where(known, cleaned, elevation)
    # 硬障碍：纯高程图分不出"地面"和"桌面/柜面"（后者在格内最高点高出地面很多，
    # 但如果只扫到台面，最低点也一样高 → 高程看起来是平的）。老管线用
    # 「格内最高点 - 地面 > 0.30 m」或「格内离地 0.10~2.0 m 的点 ≥3 个」兜住这类障碍，
    # 新管线曾漏掉这一项，实测真实数据里有 27% 的参考障碍被放行。
    obstacle_rel = np.where(known, np.nan_to_num(top) - np.nan_to_num(ground), 0.0)
    # 只按"绝对点数"判会把"擦到墙边"的格子也算障碍（格内点数少，2~3 个墙点就够数），
    # 实测会误伤 6.5% 的机器人实际轨迹。改成要求"离地点占该格点数的比例"够高。
    #
    # S5 修复：分子必须与高度门限同源。原实现用 build_grid 的 above（统计 rel>0.10 m），
    # 而门限是 0.22 m —— 一个整格 0.12 m 的低台也能把"占比"顶满，判据被低处点帮着达标。
    # 这里按"离地 > obstacle_height 的点数"自己重算分子。
    count_safe = np.maximum(count, 1)
    above_gate = _count_above(points, bounds, resolution, ground, count.shape, obstacle_height)
    frac_above = above_gate / count_safe
    #
    # S4 修复：占比口径没有绝对点数下限。0.10 m 网格的 known 门槛只有 3 点，
    # 于是"3 点格里 2 个离地点"就能判障碍——判据正好压在密度下限上。
    # 要求格内点数至少 5（或 2 倍 known 门槛，取大者）。
    scaled_min = max(3, int(round(min_points * (resolution / 0.25) ** 2)))
    dense_enough = count >= max(5, 2 * scaled_min)
    hard_obstacle = known & dense_enough & (obstacle_rel > obstacle_height) & (
        frac_above >= OBSTACLE_POINT_FRACTION)
    # 说明：地面估计用 20% 分位，粗网格下窄高物体会被平均掉（0.30 m 立柱在 0.20 m 网格上
    # 高程读到 0.00）。所以"格内最高点 - 地面"这一项是兜住窄障碍的关键，门限必须按
    # 机器人台阶能力（0.22 m）而不能比它松。
    return elevation.astype(np.float32), known, origin, hard_obstacle


PAD_CELLS = 2   # 给滤波器留边缘余量：否则边界值不可靠，会留出"可钻"的缺口


def write_grid_bin(path, elevation, resolution, origin):
    padded = np.pad(elevation, PAD_CELLS, mode="edge")
    rows, cols = padded.shape
    with open(path, "wb") as fh:
        np.array([rows, cols], np.int32).tofile(fh)
        np.array([resolution, origin[0] - PAD_CELLS * resolution,
                  origin[1] - PAD_CELLS * resolution], np.float64).tofile(fh)
        padded.astype(np.float32).tofile(fh)


def read_layers(path, shape):
    rows, cols = shape
    shape_in = np.fromfile(path, dtype=np.int32, count=2)
    if tuple(shape_in) != (rows, cols):
        raise SystemExit("grid_map 输出尺寸 %s 与输入 %s 不一致" % (tuple(shape_in), shape))
    size = rows * cols
    if path.stat().st_size != 8 + len(LAYER_ORDER) * size * 4:
        raise SystemExit("grid_map 输出的图层字节数与网格尺寸不一致")
    data = np.fromfile(path, dtype=np.float32, offset=8)
    return {name: data[i * size:(i + 1) * size].reshape(rows, cols)
            for i, name in enumerate(LAYER_ORDER)}


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def orientation_fix(layers, elevation, known=None):
    """自动识别 grid_map 输出的朝向（不靠猜）。

    grid_map 的缓冲方向与写入顺序不一定一致，之前手工猜 fliplr 就猜错了（实际是上下翻转），
    结果把坡道和楼梯的位置读反。这里用高程层与输入高程做匹配，取误差最小的变换。
    """
    target = np.asarray(elevation, dtype=np.float64)
    observed = np.isfinite(target) if known is None else np.asarray(known, dtype=bool)
    if observed.shape != target.shape:
        raise SystemExit("grid_map 朝向校验的 known 尺寸与高程不一致")
    observed &= np.isfinite(target)
    if not observed.any():
        raise SystemExit("grid_map 朝向校验没有已观测的有限高程格")
    candidates = {
        "原样": lambda a: a,
        "左右翻转": np.fliplr,
        "上下翻转": np.flipud,
        "旋转180": lambda a: np.rot90(a, 2),
    }
    best_name, best_err = None, None
    for name, fn in candidates.items():
        candidate = fn(layers["elevation_inpainted"])
        if candidate.shape != target.shape or not np.isfinite(candidate[observed]).all():
            continue
        err = float(np.mean(np.abs(candidate[observed] - target[observed])))
        if best_err is None or err < best_err:
            best_name, best_err = name, err
    if best_name is None:
        raise SystemExit("grid_map 高程图层在已观测格含 NaN/非有限值，拒绝发布")
    if best_err > MAX_LAYER_ALIGNMENT_MAE_M:
        raise SystemExit("grid_map 高程图层与输入不匹配：最佳朝向平均误差 "
                         "%.3f m > %.3f m" % (best_err, MAX_LAYER_ALIGNMENT_MAE_M))
    fixed = {k: candidates[best_name](v) for k, v in layers.items()}
    for layer_name in LAYER_ORDER:
        layer = fixed[layer_name]
        if layer.shape != target.shape:
            raise SystemExit("grid_map 图层 %s 尺寸与高程不一致" % layer_name)
        if not np.isfinite(layer[observed]).all():
            raise SystemExit("grid_map 图层 %s 在已观测格含 NaN/非有限值，拒绝发布" % layer_name)
    return fixed, best_name, best_err


# 代价 → 灰阶：必须避开 Nav2 的阈值边界，否则"贵的可通行"会被当成致命。
# 映射 240(代价0) … 20(代价1)，配合 yaml 里 free_thresh=0.10 / occupied_thresh=0.90：
#   灰阶 240 → 占用率 0.059 < 0.10 → 自由
#   灰阶  40 → 占用率 0.843 < 0.90 → 仍可通行（代价最大值必须留在致命线以内！）
COST_GRAY_FREE = 240.0
COST_GRAY_OCC = 40.0


def cost_to_gray(cost, lethal):
    """cost 0..1 → 灰阶 240(自由)…40(最贵但仍可通行)；致命写 0。

    注意：代价 1.0 的灰阶必须仍低于 occupied_thresh，否则"最贵的可通行"会变成"不可通行"
    （实测把 0.20 m 楼梯的竖面惩罚 1.0 映射成灰阶 20 → 占用率 0.92 → 整条楼梯被判死）。
    """
    span = COST_GRAY_FREE - COST_GRAY_OCC
    gray = np.clip(np.round(COST_GRAY_FREE - span * np.clip(cost, 0.0, 1.0)), 1, 254).astype(np.uint8)
    gray[lethal] = 0
    return gray


def write_costmap(out_dir, gray, resolution, origin, unknown_mask):
    image = gray.copy()
    # 未知格写 0（致命）：在 mode: scale 下，灰阶 205 的占用率是 0.196，
    # 落在 free(0.10) 与 occupied(0.90) 之间 → Nav2 得到占用值 20，被当成"便宜可通行"，
    # 而不是 -1 unknown（那只有 mode: trinary 才有）。没有点数据的区域不允许穿越。
    image[unknown_mask] = 0
    image = np.flipud(image)                       # PGM 行 0 在上，yaml origin 在左下
    rows, cols = image.shape
    with open(out_dir / "map.pgm", "wb") as fh:
        fh.write(b"P5\n%d %d\n255\n" % (cols, rows))
        fh.write(image.tobytes())
    with open(out_dir / "map.yaml", "w") as fh:
        yaml.safe_dump({"image": "map.pgm", "resolution": float(resolution),
                        "origin": [float(origin[0]), float(origin[1]), 0.0],
                        "negate": 0, "occupied_thresh": 0.90, "free_thresh": 0.10,
                        "mode": "scale"}, fh, sort_keys=False)


def main():
    ap = argparse.ArgumentParser(description="将保留地面的 PCD 导出为米制高程网格与 Nav2 连续代价图")
    ap.add_argument("--pcd", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resolution", type=float, default=0.20)
    ap.add_argument("--min-points", type=int, default=4)
    ap.add_argument("--obstacle-height", type=float, default=MAX_STEP_M,
                    help="格内最高点高出地面的门限；默认等于台阶门限 0.22 m，这样 0.25 m 立柱这类窄障碍也能兜住（原来 0.30 m 会漏）")
    ap.add_argument("--max-climb", type=float, default=MAX_STEP_M,
                    help="从地面可达性扩散时允许的单步爬升（默认=台阶能力 0.22 m）。"
                         "用来剔除桌面这类孤立高台：一整格都在桌面上时格内判据全是 0，"
                         "只能靠'能不能从地面爬上去'来识别")
    ap.add_argument("--no-reachability", action="store_true",
                    help="关闭地面可达性判据（默认开启）")
    ap.add_argument("--min-component-cells", type=int, default=0,
                    help="把小于该格数的可通行碎片当噪声剔除（0=不剔）。建议开启：碎片会拖垮规划连通性")
    ap.add_argument("--drivable-step-m", type=float, default=0.16,
                    help="可驾驶台阶上限（默认 0.16 m = 0.20 m 能力 × 0.8）。"
                         "超过它的格子即使不算致命，也导出为不可驾驶，供局部层当障碍")
    ap.add_argument("--drivable-slope-deg", type=float, default=24.0,
                    help="可驾驶坡度上限（默认 24° = 30° 能力 × 0.8）")
    ap.add_argument("--step-weight", type=float, default=STEP_WEIGHT,
                    help="台阶在连续代价里的权重（默认 %.3f）。调大 = 越高越贵，"
                         "台阶仍然是可通过的，只是代价上升" % STEP_WEIGHT)
    ap.add_argument("--slope-weight", type=float, default=SLOPE_WEIGHT,
                    help="坡度权重（默认 %.2f）" % SLOPE_WEIGHT)
    ap.add_argument("--rough-weight", type=float, default=ROUGH_WEIGHT,
                    help="粗糙度权重（默认 %.2f）" % ROUGH_WEIGHT)
    ap.add_argument("--smooth-elev", type=int, default=0,
                    help="高程中值去噪窗口（0=关）。注意 3×3 在 0.20 m 网格上等于 0.60 m 足迹，"
                         "会把 0.30 m 见方的立柱/桌腿整块抹平（实测：0.30×0.30×0.25 m 立柱在 "
                         "0.20 m 网格上被判为可通行）。只有网格 <=0.10 m 时才建议开启")
    ap.add_argument("--no-hard-obstacle", action="store_true",
                    help=f"关闭硬障碍判据（格内最高点高出地面 >{OBSTACLE_HEIGHT_M:.2f} m "
                         f"且离地点占比 ≥{100 * OBSTACLE_POINT_FRACTION:.0f}%%）。"
                         "默认开启：实测对真实 bag 的机器人轨迹 0 误伤（100%% 可通行），"
                         "同时能抓住老管线判为障碍的约 21%% 格子")
    ap.add_argument("--run-cmd", default="", help="执行 grid_map 的命令模板，用 {grid} 指代输入、{layers} 指代输出；"
                                                "省略时复用当前输入已验证的滤波结果")
    args = ap.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    # 同目录的重试和并发导入共享一套成品。仅在全部计算成功后发布。
    with (out_dir / ".gridmap.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with tempfile.TemporaryDirectory(dir=out_dir, prefix=".gridmap-") as staging:
            stage = Path(staging)
            build_maps(args, stage, out_dir)
            for product in stage.iterdir():
                product.replace(out_dir / product.name)


def build_maps(args, out_dir, cache_dir):
    points = load_pcd_xyz(args.pcd)
    elevation, known, (ox, oy), hard_obstacle = build_elevation(
        points, args.resolution, args.min_points, args.obstacle_height,
        args.smooth_elev)
    if not known.any():
        raise SystemExit("当前分辨率下没有满足最少点数的格子，请检查点云密度或调整分辨率")
    grid_bin, layers_bin = out_dir / "grid.bin", out_dir / "layers.bin"
    write_grid_bin(grid_bin, elevation, args.resolution, (ox, oy))

    if args.run_cmd:
        layers_bin.touch()  # Docker 文件挂载需要宿主先有一个普通文件。
        paths = {"grid": grid_bin, "layers": layers_bin}
        command = re.sub(r"""(["']?)\{(grid|layers)\}\1""",
                         lambda m: shlex.quote(str(paths[m.group(2)])), args.run_cmd)
        subprocess.run(command, shell=True, check=True)
    else:
        layers_bin = cache_dir / "layers.bin"
        if not layers_bin.is_file():
            raise SystemExit("缺少已验证的 grid_map 图层，请用 --run-cmd 生成")

    # XYZ 绑定本次读取的数据；grid.bin 含高程、分辨率和原点。层文件也须逐字节匹配。
    signature = json.dumps({"xyz_sha256": hashlib.sha256(points).hexdigest(),
                            "grid_sha256": file_sha256(grid_bin),
                            "layers_sha256": file_sha256(layers_bin)}, indent=2).encode()
    if args.run_cmd:
        (out_dir / "layers.json").write_bytes(signature)
    else:
        manifest = cache_dir / "layers.json"
        if not manifest.is_file() or manifest.read_bytes() != signature:
            raise SystemExit("现有 grid_map 图层与本次输入不匹配，请用 --run-cmd 重新生成")

    padded_shape = (elevation.shape[0] + 2 * PAD_CELLS, elevation.shape[1] + 2 * PAD_CELLS)
    layers = read_layers(layers_bin, padded_shape)
    layers = {k: v[PAD_CELLS:-PAD_CELLS, PAD_CELLS:-PAD_CELLS] for k, v in layers.items()}
    layers, orientation, error = orientation_fix(layers, elevation, known)
    print("朝向校正：%s（与输入高程平均误差 %.3f m）" % (orientation, error))
    # 台阶层：raw 用于判"是否是台阶"，中值用于判"台阶多高"
    #   raw 尖峰：MinInRadius 窗口跨两级踏步时高差翻倍（0.20 m 楼梯出现 0.5 m 尖峰）
    #   中值：保留真实踏步高度、压掉尖峰
    raw_step = np.clip(layers["step"], 0, None)
    step = ndimage.median_filter(raw_step, size=3)
    step_large = np.clip(layers["step_large"], 0, None)
    slope_deg = np.degrees(layers["slope"])
    rough = layers["roughness"]

    # 判据（对应硬性需求）：
    #   ① 台阶高度 > 0.20 m  → 致命（0.20 m 以内必须能过）
    #   ② 是否"台阶"用**周期性**判：台阶升上去又回到原高度，坡是单调上升。
    #      step(0.20 m 半径) / step(0.60 m 半径)：台阶 ≈ 1，30° 坡 ≈ 0.3。
    #      该判据不依赖分辨率（0.05/0.10 m 都成立），也不受中值滤波影响
    #   ③ 台阶不受坡度限制（竖面本身必然很陡）；连续坡面才用坡度判 > 34° 致命
    slope_gate = ndimage.median_filter(slope_deg, size=3)
    # 台阶凸起度 = 窗口内实测高差 / 同坡度在窗口内的理论高差
    #   台阶：高差集中在竖面，实测 >> 理论（实测 0.196 vs 34° 理论 0.135 → 1.45）
    #   连续坡：二者相等（≈1.0）
    # 这个比值对分辨率不敏感（分子分母都随窗口线性缩放）。
    # 注意：不能用"大半径/小半径高差比"——楼梯整体是上升的，大窗口会看到两级、比值失效。
    theoretical_rise = STEP_WINDOW_M * np.tan(np.radians(np.clip(slope_gate, 0, 89)))
    protrusion = raw_step / np.maximum(theoretical_rise, 1e-6)
    is_step = (raw_step > PROTRUSION_MIN_STEP_M) & (protrusion > PROTRUSION_RATIO)
    # 台阶区域豁免坡度判据：0.20 m 踏步 / 0.30 m 踏面的楼梯平均梯度就是 33.7°，
    # 实测坡度 34.2° 刚好越过 34° 门槛 → 踏面被判致命、整条楼梯断开。
    # 做法：把"竖面"往外扩一个踏面的距离，扩到的格也当台阶处理。
    step_zone = ndimage.binary_dilation(
        is_step, iterations=max(1, int(round(STEP_WINDOW_M / args.resolution))))
    # 致命台阶判据必须取 max(raw, 中值)：3×3 中值是为压 MinInRadius 跨两级踏面的尖峰，
    # 但它会把"1 格宽的台阶/落差"整条抹成 0（实测 1 格宽 0.25 m 带 → 中值后 max=0），
    # 而 is_step 用的是 raw，step_zone 仍为真 → 该格既躲过台阶判据又躲过坡度判据。
    step_for_gate = np.maximum(raw_step, step)
    obstacle_gate = np.zeros_like(hard_obstacle) if args.no_hard_obstacle else hard_obstacle
    lethal = known & (obstacle_gate
                      | (step_for_gate > MAX_STEP_M + 1e-6)
                      | ((~step_zone) & (slope_gate > MAX_SLOPE_DEG)))

    # 可驾驶判据（不是致命判据）：按 TRON2 能力 0.20 m / 30° 打 0.8 折扣得 0.16 m / 24°。
    # 致命图给全局规划做"绝对不能走"；可驾驶图给局部层做"这格车过不去"。
    drivable = known & ~(
        (step_for_gate > args.drivable_step_m + 1e-6)
        | ((~step_zone) & (slope_gate > args.drivable_slope_deg))
        | obstacle_gate)
    # 代价 = 连续项（坡度/台阶/粗糙度）+ 跨台阶的离散惩罚
    continuous = (args.slope_weight * np.clip(slope_deg / MAX_SLOPE_DEG, 0, 1)
                  + args.step_weight * np.clip(step / MAX_STEP_M, 0, 1)
                  + args.rough_weight * np.clip(rough / ROUGH_SCALE_M, 0, 1))
    riser = is_step.astype(float) * RISER_PENALTY
    cost = np.where(known & ~lethal, np.clip(continuous + riser, 0, 1), 0.0)
    gray = cost_to_gray(cost, lethal)
    # 地面可达性：剔除"从地面爬不上去"的孤立高台（桌面/柜面）
    if not args.no_reachability:
        reachable, n_seeds = reachable_from_ground(elevation, known, args.max_climb)
        raised = known & ~reachable
        if raised.any():
            print("   地面可达性：%d 格从地面不可达（单步 >%.2f m），按非地面判死"
                  "（种子 %d 格）" % (int(raised.sum()), args.max_climb, n_seeds))
        lethal = lethal | raised

    # 连通性：交付前必须知道可通行区分成几块、最大块多大
    passable_mask = ~lethal & known
    labels, comp_stats = connectivity_stats(passable_mask, args.resolution)
    if comp_stats:
        print("   可通行区连通性：%d 块，最大 %d 格（%.1f m²），第二 %d 格" % (
            len(comp_stats), comp_stats[0][1], comp_stats[0][2],
            comp_stats[1][1] if len(comp_stats) > 1 else 0))
        if args.min_component_cells > 0:
            drop = np.zeros_like(passable_mask)
            for label_id, size, _area in comp_stats:
                if size < args.min_component_cells:
                    drop |= (labels == label_id)
            if drop.any():
                print("   剔除 %d 格碎片（<%d 格）" % (int(drop.sum()), args.min_component_cells))
                lethal = lethal | drop

    # 关键：可达性与碎片剔除都在 gray 生成之后才更新 lethal，
    # 必须把最终的 lethal 重新写回灰阶，否则这些判死**不会进入交付图**
    # （实测：桌子场景打印"53 格不可达判死"，但导出的 PGM 里桌面仍是可通行）。
    gray = cost_to_gray(cost, lethal)
    drivable &= ~lethal
    np.save(out_dir / "drivable.npy", drivable)
    print("   可驾驶判据：台阶 >%.2f m 或 坡度 >%.1f° 判为不可驾驶，共 %d 格"
          % (args.drivable_step_m, args.drivable_slope_deg, int((known & ~drivable).sum())))
    write_costmap(out_dir, gray, args.resolution, (ox, oy), ~known)

    # 滤波器为了计算边缘填补了空洞；交付的高程图只保留有测量值的格子。
    observed = elevation.copy()
    observed[~known] = np.nan
    np.save(out_dir / "elevation.npy", observed, allow_pickle=False)
    np.save(out_dir / "known.npy", known, allow_pickle=False)
    elevation_meta = {"elevation": "elevation.npy", "known": "known.npy", "unit": "m",
                      "resolution": args.resolution, "origin_xy": [ox, oy],
                      "shape": list(elevation.shape), "axis_order": ["y", "x"], "unknown": "NaN"}
    (out_dir / "elevation.yaml").write_text(yaml.safe_dump(elevation_meta, sort_keys=False))

    stats = {
        "pcd": str(args.pcd),
        "artifacts": {"elevation": "elevation.npy", "known": "known.npy",
                      "drivable": "drivable.npy",
                      "elevation_metadata": "elevation.yaml", "costmap": "map.yaml",
                      "costmap_image": "map.pgm"},
        "resolution_m": args.resolution,
        "known_cells": int(known.sum()),
        "lethal_cells": int(lethal.sum()),
        "passable_cells": int((known & ~lethal).sum()),
        "slope_deg": {k: round(float(v), 2) for k, v in
                      zip(("p50", "p90", "max"),
                          np.nanpercentile(slope_deg[known], [50, 90, 100]))},
        "step_m": {k: round(float(v), 3) for k, v in
                   zip(("p50", "p90", "max"),
                       np.nanpercentile(step[known], [50, 90, 100]))},
        "cost": {k: round(float(v), 3) for k, v in
                 zip(("p50", "max"),
                     np.nanpercentile(cost[known], [50, 100]))},
        "thresholds": {
            "max_slope_deg": MAX_SLOPE_DEG,
            "max_step_m": MAX_STEP_M,
            "drivable_slope_deg": args.drivable_slope_deg,
            "drivable_step_m": args.drivable_step_m,
            "slope_weight": args.slope_weight,
            "step_weight": args.step_weight,
            "rough_weight": args.rough_weight,
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
