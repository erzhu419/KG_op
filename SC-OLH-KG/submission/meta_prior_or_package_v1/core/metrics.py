"""Metrics and Pareto utilities."""

from __future__ import annotations

import numpy as np


def pareto_filter(points, return_indices=False):
    """Return nondominated points under minimization."""
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        if return_indices:
            return pts, np.array([], dtype=int)
        return pts
    keep = np.ones(len(pts), dtype=bool)
    for i, p in enumerate(pts):
        if not keep[i]:
            continue
        dominated_by_p = np.all(p <= pts, axis=1) & np.any(p < pts, axis=1)
        keep[dominated_by_p] = False
        if np.any(np.all(pts <= p, axis=1) & np.any(pts < p, axis=1)):
            keep[i] = False
    idx = np.where(keep)[0]
    if return_indices:
        return pts[idx], idx
    return pts[idx]


def crowding_distance_select(points) -> int:
    """Select an index by maximum crowding distance."""
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n <= 2:
        return 0
    dist = np.zeros(n, dtype=float)
    for j in range(pts.shape[1]):
        order = np.argsort(pts[:, j])
        dist[order[0]] = dist[order[-1]] = np.inf
        lo = pts[order[0], j]
        hi = pts[order[-1], j]
        scale = max(hi - lo, 1e-12)
        for k in range(1, n - 1):
            dist[order[k]] += (pts[order[k + 1], j] - pts[order[k - 1], j]) / scale
    return int(np.argmax(dist))


def compute_hypervolume_2d(points, ref_point):
    """2D dominated hypervolume for minimization."""
    pts = pareto_filter(points)
    if len(pts) == 0:
        return 0.0
    ref = np.asarray(ref_point, dtype=float)
    pts = pts[np.argsort(pts[:, 0])]
    hv = 0.0
    prev_y = ref[1]
    for x, y in pts:
        if x >= ref[0] or y >= ref[1]:
            continue
        hv += max(ref[0] - x, 0.0) * max(prev_y - y, 0.0)
        prev_y = min(prev_y, y)
    return float(max(hv, 0.0))


def summarize_stage_times(iteration_log):
    keys = [
        "t_posterior_solve",
        "t_candidate_gen",
        "t_kg_compute",
        "t_simulate",
        "t_update",
        "t_eval",
    ]
    out = {}
    denom = max(len(iteration_log), 1)
    for key in keys:
        vals = [float(row.get(key, 0.0)) for row in iteration_log]
        out[key] = {
            "total": float(sum(vals)),
            "mean": float(sum(vals) / denom),
        }
    total = sum(v["total"] for v in out.values())
    for key in keys:
        out[key]["share"] = float(out[key]["total"] / total) if total > 0 else 0.0
    return out
