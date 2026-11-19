"""Scan Context 描述子：把一帧点云压成一个 (RING, SECTOR) 的二维矩阵。

原始实现来自 Kim & Kim (IROS 2018)，这里用 numpy 重写，语义对齐 C++ 版：
    - 极坐标分环（按半径）/ 分扇区（按方位角）
    - 每格取该格内**最大高度**（不是平均，保留结构高度）
    - 空格的填充值由 `empty_value` 控制（C++ 用 0）

关键参数（与 aloam_velodyne 的 sc_pgo_mid360.yaml 保持一致）：
    sc_max_radius = 20.0 m
    sc_dist_thres = 0.15
还有两个上游隐含值：RING=20, SECTOR=60。
"""
import numpy as np

# 默认分辨率针对"稀疏点云"场景（FAST-LIO 降采样后 ~1000-2500 点/帧）。
# C++ 版用 20x60=1200 格，是按 10000 点/帧的全分辨率点云设计的；
# 稀疏数据下每格不到 1 个点，描述子几乎是空的，匹配不上（实测 215 帧只有 3 帧有候选）。
# 12x36=432 格：2000 点/帧时每格约 5 个点，够形成稳定描述子。
RING = 12
SECTOR = 36


LIDAR_HEIGHT = 2.0   # C++ 版的 LIDAR_HEIGHT：把点云 z 抬到正值域再做最大高度


def make_descriptor(xyz, max_radius=20.0, rings=RING, sectors=SECTOR, lidar_height=LIDAR_HEIGHT):
    """点云 (N,3) → 描述子 (rings, sectors) float32。

    超过 max_radius 的点直接丢弃（与 C++ 的 sc_max_radius 一致）。
    高度取 z，空格填 0。
    """
    # 初值必须是 -inf：点云 z 常有负值（雷达下方的地面/低矮结构），
    # 用 0 当初值会让"最大高度为负"的格子被 0 覆盖，从而被错当成空格。
    # 实测：z 中位 -0.43 时，用 0 初值非零格只有 0.10，改成 -inf 后回到 0.34。
    desc = np.full((rings, sectors), -np.inf, dtype=np.float32)
    if xyz.shape[0] == 0:
        return np.zeros((rings, sectors), dtype=np.float32)

    x = xyz[:, 0].astype(np.float64)
    y = xyz[:, 1].astype(np.float64)
    # C++ 版：pt.z = z + LIDAR_HEIGHT。雷达高约 2 m，加完后整帧 z 落在正值，
    # "最大高度"才等价于"最高的结构"。不加这步，格子最大值常常是负数，
    # 描述子里大量 0（无观测）与真实低结构混在一起，区分度崩掉。
    z = xyz[:, 2].astype(np.float64) + float(lidar_height)

    r = np.hypot(x, y)
    keep = r < max_radius
    if not np.any(keep):
        return np.zeros_like(desc)
    x, y, z, r = x[keep], y[keep], z[keep], r[keep]

    # 环索引：r/max_radius * rings，落在 [0, rings-1]
    ring = np.floor(r / max_radius * rings).astype(np.int32)
    np.clip(ring, 0, rings - 1, out=ring)

    # 扇区索引：atan2 归一到 [0, 2pi)
    theta = np.arctan2(y, x)
    theta = np.where(theta < 0, theta + 2 * np.pi, theta)
    sector = np.floor(theta / (2 * np.pi) * sectors).astype(np.int32)
    np.clip(sector, 0, sectors - 1, out=sector)

    # 每格取最大高度：用 lexsort 找每格最大值比 np.maximum.at 快，
    # 但这里格数只有 1200、点数一万级，np.maximum.at 足够且无坑。
    np.maximum.at(desc, (ring, sector), z.astype(np.float32))
    # 没被任何点命中的格子还原为 0（表示"无观测"）
    desc[~np.isfinite(desc)] = 0.0
    return desc


def ring_key(desc, rings=RING, sectors=SECTOR):
    """环键（C++ 的 ringkey）：每环非零格的平均值，用于快速预筛。

    空环记 0。返回 (rings,) float32。
    """
    masks = desc != 0
    counts = masks.sum(axis=1)
    sums = desc.sum(axis=1)
    key = np.zeros(rings, dtype=np.float32)
    nz = counts > 0
    key[nz] = (sums[nz] / counts[nz]).astype(np.float32)
    return key


def sector_key(desc, rings=RING, sectors=SECTOR):
    """扇区键（C++ 的 sectorkey）：每扇区非零格的平均值，用于估计 yaw 偏移。"""
    masks = desc != 0
    counts = masks.sum(axis=0)
    sums = desc.sum(axis=0)
    key = np.zeros(sectors, dtype=np.float32)
    nz = counts > 0
    key[nz] = (sums[nz] / counts[nz]).astype(np.float32)
    return key


def distance(desc_a, desc_b, sectors=SECTOR):
    """两个描述子的距离（对齐 C++ 的 distDirectSC，0 最好、1 最差）。

    必须**逐列（扇区）**归一化后取平均，不能把整个矩阵拉平：
      * 拉平后，高值列会完全主导结果，低值列的信息被淹没；
      * 空列在拉平时贡献 0 相似度，把距离系统性抬到 0.7 上下，
        看起来"所有帧都差 0.7"，毫无区分力（实测 69 个最近邻里只有 2 个真的邻近）。
    C++ 的做法是跳过空列，只在双方都非空的扇区上算余弦相似度再平均。
    """
    a = desc_a.astype(np.float64)
    b = desc_b.astype(np.float64)
    na = np.linalg.norm(a, axis=0)          # 每列（扇区）的模
    nb = np.linalg.norm(b, axis=0)
    valid = (na > 1e-9) & (nb > 1e-9)
    if not np.any(valid):
        return 1.0
    dots = np.sum(a[:, valid] * b[:, valid], axis=0)
    sims = dots / (na[valid] * nb[valid])
    return float(1.0 - np.mean(sims))


def best_shift(desc_query, desc_cand, sectors=SECTOR):
    """按扇区循环移位找最小距离，返回 (最小距离, 最优移位量 sector)。"""
    best_d, best_s = 2.0, 0
    for s in range(sectors):
        shifted = np.roll(desc_cand, s, axis=1)
        d = distance(desc_query, shifted, sectors=sectors)
        if d < best_d:
            best_d, best_s = d, s
    return best_d, best_s
