"""Offline loop retrieval on sensor-frame scans and geometric verification.

The vendored NumPy core in _solid comes from sparolab/SOLiD 2.1.0,
commit 867abba490ed00d7d5f7862799bd263f53786e69, python/solid/pure/core.py.
Its BSD-3-Clause license is in _solid/LICENSE.
"""
import json
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import icp_align, scan_context
from ._solid import Candidate, Config, Extractor


SOLID_MIN_SCORE = 0.98
DEFAULT_MIN_GAP = 50
DEFAULT_PROXIMITY_RADIUS = 2.0
DEFAULT_SUBMAP_SCANS = 10
DEFAULT_SOLID_PROFILE = "tron2"
MID360_FOV_DOWN = -7.0
MID360_FOV_UP = 52.0
DESCRIPTOR_YAW_GATE_DEG = 30.0


def solid_profile_name(profile):
    if profile == "tron2-legacy":
        return "tron2"
    if profile not in ("tron2", "upstream"):
        raise ValueError(f"unknown SOLiD coordinate profile: {profile}")
    return profile


@dataclass(frozen=True)
class Mid360Input:
    submap_scans: int = DEFAULT_SUBMAP_SCANS
    profile: str = DEFAULT_SOLID_PROFILE
    fov_down: float = MID360_FOV_DOWN
    fov_up: float = MID360_FOV_UP

    @classmethod
    def from_args(cls, args):
        return cls(getattr(args, "submap_scans", DEFAULT_SUBMAP_SCANS),
                   solid_profile_name(getattr(args, "solid_profile", DEFAULT_SOLID_PROFILE)),
                   getattr(args, "solid_fov_down", MID360_FOV_DOWN),
                   getattr(args, "solid_fov_up", MID360_FOV_UP))

    def stats(self):
        return {"submap_scans": self.submap_scans, "solid_profile": self.profile,
                "solid_fov_down": self.fov_down, "solid_fov_up": self.fov_up,
                "solid_descriptor_axes": ["x", "-y", "-z"] if self.profile == "upstream"
                else ["x", "y", "z"]}


class ScanWindow:
    """Most recent paired scans, including non-keyframes and the current scan."""

    def __init__(self, size):
        if size < 1:
            raise ValueError("submap_scans must be at least 1")
        self.scans = deque(maxlen=size)

    def append(self, cloud, pose):
        self.scans.append((cloud, pose))

    def merged(self, *, world_coordinates):
        current, key_pose = self.scans[-1]
        if len(self.scans) == 1:
            return current
        if world_coordinates:
            # FAST-LIO has already transformed every point to camera_init.
            return np.concatenate([cloud for cloud, _ in self.scans])
        key_from_world = np.linalg.inv(key_pose)
        parts = []
        for cloud, scan_pose in list(self.scans)[:-1]:
            key_from_scan = key_from_world @ scan_pose
            parts.append((cloud @ key_from_scan[:3, :3].T + key_from_scan[:3, 3])
                         .astype(current.dtype, copy=False))
        return np.concatenate([*parts, current])


@dataclass(frozen=True)
class Proposal:
    id: int
    source: str
    score: Optional[float] = None
    yaw_rad: Optional[float] = None
    also_source: Optional[str] = None


class LoopDiagnostics:
    def __init__(self, mode, descriptor):
        sources = []
        if mode != "solid":
            sources.append("geometry")
        if mode != "geometry":
            sources.append(descriptor)
        self.attempts = dict.fromkeys(sources, 0)
        self.accepted = dict.fromkeys(sources, 0)
        self.yaw_retries = 0
        self.yaw_consistency_rejections = 0
        self.rejected_yaw_candidates = dict.fromkeys(sources, 0)
        self.records = []

    def write(self, out_dir):
        with (out_dir / "loop_diagnostics.jsonl").open("w", encoding="utf-8") as stream:
            for record in self.records:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        return {"tried_by_source": self.attempts, "accepted_by_source": self.accepted,
                "gicp_yaw_retries": self.yaw_retries,
                "descriptor_yaw_gate_deg": DESCRIPTOR_YAW_GATE_DEG,
                "yaw_consistency_rejections": self.yaw_consistency_rejections,
                "candidates_rejected_yaw_consistency": self.rejected_yaw_candidates}


def point_counts(clouds):
    counts = [len(cloud) for cloud in clouds]
    return {"min": min(counts), "median": float(np.median(counts)), "max": max(counts)}


def add_descriptor_options(parser):
    parser.add_argument("--submap-scans", type=int, default=DEFAULT_SUBMAP_SCANS,
                        help="number of latest input scans merged for each descriptor and GICP (Mid-360: 10)")
    parser.add_argument("--solid-profile", type=solid_profile_name, choices=("tron2", "upstream"),
                        default=DEFAULT_SOLID_PROFILE,
                        help="SOLiD axes on TRON2 data: upstream undoes the bridge Rx(180 deg), "
                             "tron2 uses the published body axes")
    parser.add_argument("--solid-fov-down", type=float, default=MID360_FOV_DOWN,
                        help="lower SOLiD elevation bound in degrees in the selected profile")
    parser.add_argument("--solid-fov-up", type=float, default=MID360_FOV_UP,
                        help="upper SOLiD elevation bound in degrees in the selected profile")
    parser.add_argument("--candidate-mode", choices=("hybrid", "solid", "geometry"), default="hybrid",
                        help="combine nearest odometry history and descriptors, or select one source")
    parser.add_argument("--proximity-radius", type=float, default=DEFAULT_PROXIMITY_RADIUS,
                        help="maximum odometry distance in metres for geometry candidates")
    parser.add_argument("--descriptor", choices=("solid", "scan-context"), default="solid")
    parser.add_argument("--solid-min-score", type=float, default=SOLID_MIN_SCORE,
                        help="minimum R-SOLiD cosine similarity")
    parser.add_argument("--max-radius", type=float,
                        help="descriptor range in metres (SOLiD 60, Scan Context 20)")


def descriptor_options(args):
    descriptor = getattr(args, "descriptor", "solid")
    radius = getattr(args, "max_radius", None)
    if radius is None:
        radius = 60.0 if descriptor == "solid" else 20.0
    return (descriptor, radius, getattr(args, "solid_min_score", SOLID_MIN_SCORE),
            getattr(args, "candidate_mode", "hybrid"),
            getattr(args, "proximity_radius", DEFAULT_PROXIMITY_RADIUS))


def describe_scans(clouds, descriptor, max_radius, *, profile=DEFAULT_SOLID_PROFILE,
                   fov_down=MID360_FOV_DOWN, fov_up=MID360_FOV_UP):
    """Extract descriptors from sensor/body-frame points, retaining keyframe order."""
    if descriptor == "solid":
        if not np.isfinite(max_radius) or max_radius <= 1.0:
            raise ValueError("SOLiD max_radius must exceed the Mid-360 minimum range of 1 m")
        if not -90 <= fov_down < fov_up <= 90:
            raise ValueError("SOLiD vertical FOV must satisfy -90 <= down < up <= 90 degrees")
        profile = solid_profile_name(profile)
        # TRON2 publishes the inverted unit in virtual body axes (x, -y, -z).
        # Rotate back only for the native sensor descriptor; registration stays in body.
        if profile == "upstream":
            rotation = np.array([1, -1, -1], dtype=np.float32)
            clouds = (cloud * rotation for cloud in clouds)
        extractor = Extractor(Config(fov_down=fov_down, fov_up=fov_up, min_range=1.0,
                                     max_range=max_radius, voxel_size=0.4))
        descriptors = [extractor.extract(cloud) for cloud in clouds]
        fills = [np.count_nonzero(d.combined()) / (d.rsolid.size + d.asolid.size)
                 for d in descriptors]
    elif descriptor == "scan-context":
        descriptors = [scan_context.make_descriptor(cloud, max_radius=max_radius)
                       for cloud in clouds]
        fills = [np.count_nonzero(d) / d.size for d in descriptors]
    else:
        raise ValueError(f"unknown loop descriptor: {descriptor}")
    return descriptors, float(np.mean(fills)) if fills else 0.0


def prepare_descriptors(clouds, mode, descriptor, max_radius, mid360=Mid360Input()):
    if mode == "geometry":
        return [], None
    descriptors, fill = describe_scans(clouds, descriptor, max_radius, profile=mid360.profile,
                                       fov_down=mid360.fov_down, fov_up=mid360.fov_up)
    return descriptors, round(fill, 3)


def solid_candidates(descriptors, min_gap, min_score, max_candidates, *, profile=DEFAULT_SOLID_PROFILE):
    """Rank only eligible history by the upstream pure database's cosine score."""
    if min_gap < 0 or max_candidates < 1 or not -1 <= min_score <= 1:
        raise ValueError("invalid SOLiD history, candidate count, or cosine threshold")
    if not descriptors:
        return
    ranges = np.stack([d.rsolid for d in descriptors])
    norms = np.linalg.norm(ranges, axis=1)
    valid = np.isfinite(norms) & (norms > 0)
    keys = np.zeros_like(ranges)
    np.divide(ranges, norms[:, None], out=keys, where=valid[:, None])

    for current in range(min_gap + 1, len(descriptors)):
        if not valid[current]:
            continue
        eligible = current - min_gap
        # The pure upstream Database computes yaw for all passing scans. Sorting
        # the same cosine scores first limits angular alignment to the top hits.
        scores = np.einsum("ij,j->i", keys[:eligible], keys[current])
        hits = np.flatnonzero(valid[:eligible] & (scores >= min_score))
        order = np.argsort(-scores[hits], kind="stable")[:max_candidates]
        query_angles = descriptors[current].asolid
        candidates = []
        for history in hits[order]:
            yaw = Extractor.pose_yaw_deg(query_angles, descriptors[history].asolid)
            # Upstream reports (shift + 1) bins. GICP frame convention uses
            # zero for identical scans and positive yaw for current -> history.
            yaw = (yaw - 360.0 / query_angles.size + 180.0) % 360.0 - 180.0
            if profile == "upstream":
                # Rx(pi) conjugates Rz(yaw) to Rz(-yaw) in the published body axes.
                yaw = -yaw
            candidates.append(Candidate(int(history), float(np.clip(scores[history], -1, 1)),
                                        float(np.deg2rad(yaw))))
        if candidates:
            yield current, candidates


def scan_context_candidates(descriptors, min_gap, max_distance, max_candidates):
    if not descriptors:
        return
    keys = np.stack([scan_context.ring_key(d) for d in descriptors])
    for current in range(min_gap + 1, len(descriptors)):
        delta = keys[:current - min_gap] - keys[current]
        nearest = np.argsort(np.einsum("ij,ij->i", delta, delta), kind="stable")[:40]
        candidates = []
        for history in nearest:
            distance, shift = scan_context.best_shift(descriptors[current], descriptors[history])
            if distance < max_distance:
                # best_shift rolls the historical descriptor into the query frame.
                yaw = -2.0 * np.pi * shift / scan_context.SECTOR
                candidates.append(Candidate(int(history), 1.0 - distance, yaw))
        if candidates:
            candidates.sort(key=lambda hit: -hit.score)
            yield current, candidates[:max_candidates]


def candidate_groups(descriptor_groups, poses, min_gap, proximity_radius, max_candidates,
                     mode, descriptor):
    """Yield at most three neighbors first, then up to three descriptor proposals."""
    if mode not in ("hybrid", "solid", "geometry"):
        raise ValueError(f"unknown candidate mode: {mode}")
    if min_gap < 0 or max_candidates < 1:
        raise ValueError("invalid history gap or candidate count")
    if mode != "solid" and (not np.isfinite(proximity_radius) or proximity_radius <= 0):
        raise ValueError("proximity radius must be finite and positive")
    limit = min(max_candidates, 3)
    ranked = {} if mode == "geometry" else dict(descriptor_groups)
    if mode != "solid":
        positions = np.asarray([p[:3, 3] for p in poses])
    for current in range(min_gap + 1, len(poses)):
        matches = {hit.id: hit for hit in ranked.get(current, [])[:limit]}
        proposals = []
        if mode != "solid":
            distances = np.linalg.norm(positions[:current - min_gap] - positions[current], axis=1)
            nearby = np.flatnonzero(distances < proximity_radius)
            order = np.argsort(distances[nearby], kind="stable")[:limit]
            for index in nearby[order]:
                history = int(index)
                shared = matches.pop(history, None)
                if shared is None:
                    proposals.append(Proposal(history, "geometry"))
                else:
                    proposals.append(Proposal(history, "geometry", shared.score,
                                              shared.yaw_rad, also_source=descriptor))
        proposals.extend(Proposal(hit.id, descriptor, hit.score, hit.yaw_rad)
                         for hit in matches.values())
        if proposals:
            yield current, proposals


def registration_seeds(predicted, yaw_rad):
    yield "odometry", predicted
    if yaw_rad is not None and np.isfinite(yaw_rad):
        # Left-multiplying Rz(delta) replaces ZYX yaw and preserves roll/pitch.
        angle = yaw_rad - icp_align.yaw_from_T(predicted)
        c, s = np.cos(angle), np.sin(angle)
        seed = predicted.copy()
        seed[:2, :3] = np.array([[c, -s], [s, c]]) @ predicted[:2, :3]
        yield "descriptor_yaw", seed


def verify_candidates(candidates_by_frame, clouds, poses, max_rmse, max_gate=2.0,
                      min_fitness=0.6, diagnostics=None):
    """Return measured (current, history, current_from_history, RMSE) factors."""
    edges, tried = [], 0
    for current, candidates in candidates_by_frame:
        for candidate in candidates:
            history = candidate.id
            predicted = np.linalg.inv(poses[history]) @ poses[current]
            source = getattr(candidate, "source", None)
            descriptor_yaw = getattr(candidate, "yaw_rad", None)
            tried += 1
            if diagnostics is not None:
                diagnostics.attempts[candidate.source] += 1
            rejected_yaw = False
            for init_source, seed in registration_seeds(predicted, descriptor_yaw):
                if init_source == "descriptor_yaw" and diagnostics is not None:
                    diagnostics.yaw_retries += 1
                measured, overlap, rmse = icp_align.align_frames(
                    clouds[current], clouds[history], init=seed)
                prediction_delta = float(np.linalg.norm(measured[:3, 3] - predicted[:3, 3]))
                if (not np.isfinite(measured).all() or not np.isfinite(overlap)
                        or not np.isfinite(rmse) or overlap < min_fitness
                        or rmse > max_rmse or prediction_delta > max_gate):
                    continue
                yaw_error = None
                if descriptor_yaw is not None and np.isfinite(descriptor_yaw):
                    delta = np.rad2deg(icp_align.yaw_from_T(measured) - descriptor_yaw)
                    yaw_error = float(abs((delta + 180.0) % 360.0 - 180.0))
                if source in ("solid", "scan-context") and (
                        yaw_error is None or yaw_error > DESCRIPTOR_YAW_GATE_DEG + 1e-12):
                    rejected_yaw = True
                    if diagnostics is not None:
                        diagnostics.yaw_consistency_rejections += 1
                    continue
                break
            else:
                if rejected_yaw and diagnostics is not None:
                    diagnostics.rejected_yaw_candidates[source] += 1
                continue
            factor = icp_align.alignment_to_between(measured)
            edges.append((current, history, factor, rmse))
            if diagnostics is not None:
                diagnostics.accepted[candidate.source] += 1
                diagnostics.records.append({
                    "current": current, "history": history,
                    "source": candidate.source, "also_source": candidate.also_source,
                    "score": candidate.score, "descriptor_yaw_rad": candidate.yaw_rad,
                    "init_source": init_source, "yaw_fallback": init_source == "descriptor_yaw",
                    "yaw_consistency_error_deg": yaw_error,
                    "odom_distance_m": float(np.linalg.norm(predicted[:3, 3])),
                    "overlap": float(overlap), "rmse": float(rmse),
                    "prediction_delta": prediction_delta,
                })
            break
    return edges, tried


def detect_loops(descs, clouds, poses, sc_thres, min_gap, fitness_thres, max_candidates=3,
                 max_gate=2.0, min_fitness=0.6, *, descriptor="scan-context",
                 solid_min_score=SOLID_MIN_SCORE, candidate_mode="solid",
                 proximity_radius=DEFAULT_PROXIMITY_RADIUS, diagnostics=None,
                 solid_profile=DEFAULT_SOLID_PROFILE):
    # Existing callers with precomputed Scan Context descriptors keep their API.
    if candidate_mode == "geometry":
        candidates = ()
    elif descriptor == "solid":
        candidates = solid_candidates(descs, min_gap, solid_min_score, max_candidates, profile=solid_profile)
    elif descriptor == "scan-context":
        if min_gap < 0 or max_candidates < 1:
            raise ValueError("invalid Scan Context history or candidate count")
        candidates = scan_context_candidates(descs, min_gap, sc_thres, max_candidates)
    else:
        raise ValueError(f"unknown loop descriptor: {descriptor}")
    groups = candidate_groups(candidates, poses, min_gap, proximity_radius, max_candidates,
                              candidate_mode, descriptor)
    return verify_candidates(groups, clouds, poses, fitness_thres, max_gate, min_fitness, diagnostics)
