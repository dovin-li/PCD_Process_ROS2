"""GTSAM 位姿图：把里程计边和回环边一起优化，输出修正后的位姿。

与 aloam_velodyne 的 C++ 版对齐：
    odomNoise  = diag(1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4)   ← 平移非常可信
    priorNoise = diag(1e-12 ... )                            ← 首节点强约束
    loopNoiseScore ≈ 0.2                                     ← 回环边由 ICP fitness 缩放

注意：gtsam 的 Python 绑定要求 numpy<2，固定版本范围见仓库根目录 requirements-loop.txt。
"""
import numpy as np

# 里程计边的 sigma：相邻关键帧（0.2 m 间隔）的 FAST-LIO 相对误差实测量级。
# 注意：C++ 版的 1e-6 是"单节点先验"的量级，直接拿来当连续里程计边会让优化器
# 完全忽略回环（实测：0.5 m 漂移 + 0.05 m 精确回环，修正量只有 2.6e-6 m）。
# 所以这里用物理上合理的量级：平移 1 cm、旋转 0.5°。
# GTSAM Pose3 tangent order is rotation first, then translation.
ODOM_SIGMAS = np.array([0.0087, 0.0087, 0.0087, 0.01, 0.01, 0.01])
PRIOR_SIGMAS = np.array([1e-12] * 6)


def _loop_sigmas(fitness, loop_score):
    """Map GICP RMSE and the public noise scale to Pose3 tangent sigmas.

    ``loop_score=0.2`` preserves the established default.  Larger values make
    loop factors weaker; smaller positive values trust them more.
    """
    score = float(loop_score)
    if not np.isfinite(score) or score <= 0:
        raise ValueError("loop_score must be finite and positive")
    rmse = float(fitness)
    if not np.isfinite(rmse) or rmse < 0:
        raise ValueError("loop fitness must be finite and non-negative")
    base = max(rmse, 0.02) * score / 0.2
    sigma_rot = float(np.clip(base * 0.5, 0.02, 0.20))
    return np.array([sigma_rot, sigma_rot, sigma_rot, base, base, base])


def _pose3(T):
    import gtsam
    R = gtsam.Rot3(T[0, 0], T[0, 1], T[0, 2],
                   T[1, 0], T[1, 1], T[1, 2],
                   T[2, 0], T[2, 1], T[2, 2])
    return gtsam.Pose3(R, gtsam.Point3(float(T[0, 3]), float(T[1, 3]), float(T[2, 3])))


def _mat4(pose):
    return np.asarray(pose.matrix(), dtype=np.float64)


def optimize(poses, loop_edges, loop_score=0.2, use_isam=True):
    """位姿图优化。

    poses       : 长度 N 的 4x4 数组列表（里程计位姿，世界系）
    loop_edges  : [(i, j, T_ij, fitness), ...]  i<j，T_ij 是 i→j 的相对变换
    loop_score  : 回环边长噪声的基准尺（越大越不信任回环）
    返回        : (优化后位姿 list[4x4], 残差统计 dict)
    """
    import gtsam

    # Validate the public knob even when this run happens to contain no loops.
    _loop_sigmas(0.02, loop_score)

    n = len(poses)
    if n == 0:
        return [], {"error": "no poses"}

    g = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(PRIOR_SIGMAS)
    odom_noise = gtsam.noiseModel.Diagonal.Sigmas(ODOM_SIGMAS)

    g.add(gtsam.PriorFactorPose3(0, _pose3(poses[0]), prior_noise))
    for k in range(n):
        initial.insert(k, _pose3(poses[k]))

    for k in range(1, n):
        rel = np.linalg.inv(poses[k - 1]) @ poses[k]
        g.add(gtsam.BetweenFactorPose3(k - 1, k, _pose3(rel), odom_noise))

    used_loops = 0
    for (i, j, T_ij, fitness) in loop_edges:
        if not (0 <= i < n and 0 <= j < n) or i == j:
            continue
        try:
            sig = _loop_sigmas(fitness, loop_score)
            diagonal = gtsam.noiseModel.Diagonal.Sigmas(sig)
            estimator = gtsam.noiseModel.mEstimator.Cauchy.Create(1.0)
            noise = gtsam.noiseModel.Robust.Create(estimator, diagonal)
        except Exception:
            continue
        g.add(gtsam.BetweenFactorPose3(i, j, _pose3(T_ij), noise))
        used_loops += 1

    err_before = float(g.error(initial))
    if use_isam:
        params = gtsam.ISAM2Params()
        params.relinearizeSkip = 1
        params.setRelinearizeThreshold(0.01)
        isam = gtsam.ISAM2(params)
        isam.update(g, initial)
        # Empty updates let iSAM2 revisit variables selected by its nonlinear
        # relinearization policy without rebuilding a batch graph.
        for _ in range(4):
            isam.update()
        result = isam.calculateEstimate()
        optimizer_name = "isam2"
    else:
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(100)
        result = gtsam.LevenbergMarquardtOptimizer(g, initial, params).optimize()
        optimizer_name = "levenberg_marquardt"

    out = [_mat4(result.atPose3(k)) for k in range(n)]
    err_after = float(g.error(result))

    return out, {
        "poses": n,
        "loop_edges": used_loops,
        "optimizer": optimizer_name,
        "relinearize_skip": 1 if use_isam else None,
        "loop_kernel": "cauchy",
        "loop_noise_scale": float(loop_score),
        "error_before": err_before,
        "error_after": err_after,
    }


def compare(odom_poses, opt_poses):
    """量化回环修正量及优化前后的首尾间距。

    首尾间距只是几何量，不是“闭合误差”：开放路线本来就不应该首尾重合。
    """
    n = min(len(odom_poses), len(opt_poses))
    if n == 0:
        return {}
    diffs, angles = [], []
    for k in range(n):
        a, b = odom_poses[k], opt_poses[k]
        diffs.append(float(np.linalg.norm(a[:3, 3] - b[:3, 3])))
        Ra, Rb = a[:3, :3], b[:3, :3]
        c = np.clip((np.trace(Ra.T @ Rb) - 1.0) / 2.0, -1.0, 1.0)
        angles.append(float(np.degrees(np.arccos(c))))
    diffs = np.array(diffs)
    angles = np.array(angles)
    return {
        "frames": n,
        "trans_max_m": float(diffs.max()),
        "trans_mean_m": float(diffs.mean()),
        "angle_max_deg": float(angles.max()),
        "angle_mean_deg": float(angles.mean()),
        "raw_start_end_distance_m": float(np.linalg.norm(
            odom_poses[n - 1][:3, 3] - odom_poses[0][:3, 3])),
        "optimized_start_end_distance_m": float(np.linalg.norm(
            opt_poses[n - 1][:3, 3] - opt_poses[0][:3, 3])),
    }
