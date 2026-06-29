"""Pareto dominance utilities.

Translated from: perato.m, perato_con.m, crowding_distance.m
"""
import numpy as np


def is_dominated(a, b):
    """Check if solution a is dominated by solution b (minimization).

    b dominates a iff b[i] <= a[i] for all i, and b[j] < a[j] for some j.
    """
    return np.all(b <= a) and np.any(b < a)


def pareto_front_indices(objectives):
    """Find indices of Pareto non-dominated solutions (minimization).

    Parameters
    ----------
    objectives : np.ndarray of shape (N, m)
        Objective values for N solutions with m objectives.

    Returns
    -------
    list of int
        Indices of non-dominated solutions.
    """
    N = len(objectives)
    if N == 0:
        return []
    dominated = np.zeros(N, dtype=bool)
    for i in range(N):
        if dominated[i]:
            continue
        for j in range(N):
            if i == j or dominated[j]:
                continue
            if is_dominated(objectives[i], objectives[j]):
                dominated[i] = True
                break
    return [i for i in range(N) if not dominated[i]]


def pareto_front_2d(f1, f2):
    """Find Pareto front indices for 2 objectives (minimization).

    Uses efficient O(N log N) sweep.
    """
    N = len(f1)
    if N == 0:
        return []
    # Sort by f1 ascending, then f2 ascending
    idx = np.lexsort((f2, f1))
    front = [idx[0]]
    best_f2 = f2[idx[0]]
    for i in range(1, N):
        if f2[idx[i]] < best_f2:
            front.append(idx[i])
            best_f2 = f2[idx[i]]
    return front


def crowding_distance(objectives):
    """Compute crowding distance for a set of objective values.

    Translated from crowding_distance.m

    Parameters
    ----------
    objectives : np.ndarray of shape (N, m)
        Objective values for N solutions.

    Returns
    -------
    np.ndarray of shape (N,)
        Crowding distance for each solution.
    """
    N, m = objectives.shape
    if N <= 2:
        return np.full(N, np.inf)

    cd = np.zeros(N)
    for j in range(m):
        sorted_idx = np.argsort(objectives[:, j])
        fmax = objectives[sorted_idx[-1], j]
        fmin = objectives[sorted_idx[0], j]
        # Boundary points get infinite distance
        cd[sorted_idx[0]] = np.inf
        cd[sorted_idx[-1]] = np.inf
        if fmax - fmin > 1e-12:
            for k in range(1, N - 1):
                cd[sorted_idx[k]] += (
                    (objectives[sorted_idx[k+1], j] - objectives[sorted_idx[k-1], j])
                    / (fmax - fmin)
                )
    return cd


def compute_hypervolume_2d(front, ref_point):
    """Compute 2D hypervolume indicator.

    Parameters
    ----------
    front : np.ndarray of shape (N, 2)
        Non-dominated front points (minimization).
    ref_point : np.ndarray of shape (2,)
        Reference point (upper bound).

    Returns
    -------
    float
        Hypervolume indicator.
    """
    if len(front) == 0:
        return 0.0
    # Filter points dominated by ref_point
    valid = np.all(front < ref_point, axis=1)
    pts = front[valid]
    if len(pts) == 0:
        return 0.0
    # Sort by f1 ascending
    pts = pts[np.argsort(pts[:, 0])]
    hv = 0.0
    prev_f2 = ref_point[1]
    for p in pts:
        if p[1] < prev_f2:
            hv += (ref_point[0] - p[0]) * (prev_f2 - p[1])
            prev_f2 = p[1]
    return hv


def compute_igd(estimated_pf, true_pf):
    """Compute Inverted Generational Distance.

    IGD = (1/|PF*|) * sum_{p in PF*} min_{q in estimated_pf} ||p - q||

    Parameters
    ----------
    estimated_pf : np.ndarray of shape (N, m)
    true_pf : np.ndarray of shape (M, m)

    Returns
    -------
    float
    """
    if len(estimated_pf) == 0 or len(true_pf) == 0:
        return float('inf')
    total = 0.0
    for p in true_pf:
        dists = np.linalg.norm(estimated_pf - p, axis=1)
        total += np.min(dists)
    return total / len(true_pf)
