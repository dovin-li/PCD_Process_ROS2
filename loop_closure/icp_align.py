"""ICP 精配准：给 Scan Context 的候选闭环算相对位姿。

优先用 open3d；没有 open3d 时退回自己写的点到点 ICP（numpy）。
两条路都返回 4x4 变换 + fitness（重叠区平均残差，越小越好）。
"""
import numpy as np


def _voxel_down(xyz, leaf):
    """体素降采样，返回去重后的点（取每格首点）。"""
    if leaf <= 0 or xyz.shape[0] == 0:
        return xyz
    keys = np.floor(xyz / leaf).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return xyz[np.sort(idx)]


def icp_open3d(src_xyz, dst_xyz, init=np.eye(4), max_dist=0.5, max_iter=50,
               method="gicp"):
    """Open3D registration; production default is Generalized ICP.

    The returned matrix always maps ``src_xyz`` into the coordinate frame of
    ``dst_xyz``.  This direction is deliberately explicit because a GTSAM
    ``BetweenFactor(src, dst)`` needs its inverse (see
    :func:`alignment_to_between`).
    """
    import open3d as o3d

    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(src_xyz.astype(np.float64))
    dst = o3d.geometry.PointCloud()
    dst.points = o3d.utility.Vector3dVector(dst_xyz.astype(np.float64))

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=max_iter)
    if method == "gicp":
        reg = o3d.pipelines.registration.registration_generalized_icp(
            src, dst, max_dist, init,
            o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
            criteria,
        )
    elif method == "point_to_point":
        reg = o3d.pipelines.registration.registration_icp(
            src, dst, max_dist, init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria,
        )
    else:
        raise ValueError(f"unsupported registration method: {method}")
    return np.asarray(reg.transformation), float(reg.fitness), float(reg.inlier_rmse)


def icp_numpy(src_xyz, dst_xyz, init=np.eye(4), max_iter=30, tol=1e-6):
    """纯 numpy 点到点 ICP（SVD 求解），open3d 不可用时兜底。

    返回 (T 4x4, fitness, rmse)。fitness 定义为最近邻距离 < 0.3 的点占比。
    """
    from scipy.spatial import cKDTree

    src = src_xyz.astype(np.float64)
    dst = dst_xyz.astype(np.float64)
    if src.shape[0] == 0 or dst.shape[0] == 0:
        return np.eye(4), 0.0, float("inf")

    T = init.copy()
    tree = cKDTree(dst)
    prev_err = np.inf
    for _ in range(max_iter):
        moved = src @ T[:3, :3].T + T[:3, 3]
        dist, idx = tree.query(moved, k=1)
        good = dist < 0.3
        if good.sum() < 10:
            break
        a = moved[good]
        b = dst[idx[good]]
        ca, cb = a.mean(axis=0), b.mean(axis=0)
        H = (a - ca).T @ (b - cb)
        U, S, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        D = np.diag([1.0, 1.0, d])
        R = Vt.T @ D @ U.T
        t = cb - R @ ca
        dT = np.eye(4)
        dT[:3, :3] = R
        dT[:3, 3] = t
        T = dT @ T
        err = float(dist[good].mean())
        if abs(prev_err - err) < tol:
            break
        prev_err = err

    moved = src @ T[:3, :3].T + T[:3, 3]
    dist, _ = tree.query(moved, k=1)
    fitness = float((dist < 0.3).mean())
    rmse = float(dist[dist < 0.3].mean()) if np.any(dist < 0.3) else float("inf")
    return T, fitness, rmse


def align_frames(src_xyz, dst_xyz, init=np.eye(4), leaf=0.2, method="gicp"):
    """统一入口：降采样后配准，默认 GICP。

    ``gicp`` 是回环约束的正式后端；只有显式指定
    ``point_to_point`` 才运行旧的点到点 ICP。没有 Open3D 时不得静默把
    GICP 降级成另一种算法。
    """
    s = _voxel_down(src_xyz, leaf)
    d = _voxel_down(dst_xyz, leaf)
    try:
        return icp_open3d(s, d, init=init, method=method)
    except ImportError as exc:
        if method == "gicp":
            raise RuntimeError("GICP requires Open3D") from exc
        return icp_numpy(s, d, init=init)


def alignment_to_between(target_from_source):
    """Convert an ICP ``target <- source`` result to Between(source, target).

    With world poses ``T_w_source`` and ``T_w_target``, registration returns
    ``inv(T_w_target) @ T_w_source`` while GTSAM expects
    ``inv(T_w_source) @ T_w_target``.  They are exact inverses.
    """
    value = np.asarray(target_from_source, dtype=np.float64).reshape(4, 4)
    return np.linalg.inv(value)


def T_from_pose(position, quat):
    """(x,y,z) + (qx,qy,qz,qw) → 4x4 变换。"""
    x, y, z, w = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        R = np.eye(3)
    else:
        s = 2.0 / n
        R = np.array([
            [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
        ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [float(position[0]), float(position[1]), float(position[2])]
    return T


def yaw_from_T(T):
    return float(np.arctan2(T[1, 0], T[0, 0]))
