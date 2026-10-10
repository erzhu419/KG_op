"""Candidate generation helpers."""

from __future__ import annotations

import numpy as np
from scipy.stats import qmc


def boundary_solutions(problem):
    """Problem-independent boundary/center initial design."""
    lo, hi = problem.int_bounds()
    lo = np.asarray(lo, dtype=int)
    hi = np.asarray(hi, dtype=int)
    center = np.round((lo + hi) / 2.0).astype(int)
    rows = [tuple(lo), tuple(hi), tuple(center)]
    for j in range(problem.d):
        x_hi = lo.copy()
        x_hi[j] = hi[j]
        rows.append(tuple(x_hi))
        x_q1 = lo.copy()
        x_q1[j] = np.round(lo[j] + 0.25 * (hi[j] - lo[j])).astype(int)
        rows.append(tuple(x_q1))
        x_mid = lo.copy()
        x_mid[j] = center[j]
        rows.append(tuple(x_mid))
        x_q3 = lo.copy()
        x_q3[j] = np.round(lo[j] + 0.75 * (hi[j] - lo[j])).astype(int)
        rows.append(tuple(x_q3))
    return unique_candidates(rows)


def latin_hypercube_candidates(problem, n, rng=None):
    """Generate LHD candidates and map them to the integer grid."""
    if n <= 0:
        return []
    rng = rng or np.random.default_rng()
    try:
        sampler = qmc.LatinHypercube(d=problem.d, seed=int(rng.integers(1, 2**31 - 1)))
        rows = sampler.random(n)
    except Exception:
        rows = rng.random((n, problem.d))
    return [problem.continuous_to_int(row) for row in rows]


def random_candidates(problem, n, rng=None):
    rng = rng or np.random.default_rng()
    return [problem.sample_random(rng) for _ in range(max(0, int(n)))]


def structured_candidates(problem, n, rng=None):
    """Problem-provided candidate anchors for structured synthetic domains."""
    if n <= 0 or not hasattr(problem, "structured_candidates"):
        return []
    return [
        tuple(int(v) for v in x)
        for x in problem.structured_candidates(n=int(n), rng=rng)
    ]


def axis_candidates(problem, n, rng=None):
    """Evenly spaced anchors from a problem-provided axis grid."""
    del rng
    if n <= 0 or not hasattr(problem, "all_axis_solutions"):
        return []
    rows = [tuple(int(v) for v in x) for x in problem.all_axis_solutions()]
    rows = unique_candidates(rows)
    if len(rows) <= int(n):
        return rows
    idx = np.linspace(0, len(rows) - 1, int(n))
    return [rows[int(round(i))] for i in idx]


def axis_landmark_candidates(problem, n, rng=None):
    """Low-edge axis landmarks that do not require evaluating the objective.

    The paper-style RZDT families often place the best scalarized feasible
    point near a low-risk boundary rather than exactly at 0/25/50/75/100.
    These anchors keep the grid problem-independent while giving the optimizer
    a fair chance to inspect that boundary neighborhood.
    """
    del rng
    if n <= 0:
        return []
    lo, hi = problem.int_bounds()
    lo = np.asarray(lo, dtype=int)
    hi = np.asarray(hi, dtype=int)
    if problem.d <= 0:
        return []
    span = int(hi[0] - lo[0])
    if span <= 0:
        return [tuple(lo)]

    ordered_fractions = [
        0.0,
        1.0,
        0.05,
        0.10,
        0.125,
        0.14,
        0.15,
        0.20,
        0.25,
        1.0 / 3.0,
        0.50,
        2.0 / 3.0,
        0.75,
        0.875,
        0.95,
    ]
    rows = []
    for frac in ordered_fractions:
        x = lo.copy()
        x[0] = int(np.clip(round(lo[0] + frac * span), lo[0], hi[0]))
        rows.append(tuple(x))

    if len(unique_candidates(rows)) < int(n):
        for frac in np.linspace(0.0, 1.0, int(n)):
            x = lo.copy()
            x[0] = int(np.clip(round(lo[0] + frac * span), lo[0], hi[0]))
            rows.append(tuple(x))
    return unique_candidates(rows)[: int(n)]


def posterior_sample_candidates(
    problem,
    gpr_models,
    n_batches=0,
    pool_size=500,
    keep_per_batch=20,
    rng=None,
    use_constraint=False,
    variance_lookup=None,
    epistemic_lookup=None,
    tau=0.0,
    alpha_z=1.6448536269514722,
    beta_g=0.0,
    certification_mode="legacy",
):
    """Cheap posterior-sampling candidate generator.

    This avoids making the first SC-OLH-KG prototype depend on a full inner
    NSGA-II loop.  Each batch samples parametric coefficients, scores a random
    pool, and keeps promising points.
    """
    if n_batches <= 0:
        return []
    rng = rng or np.random.default_rng()
    candidates = []
    for _ in range(int(n_batches)):
        pool = random_candidates(problem, pool_size, rng)
        X = np.asarray(pool, dtype=int)
        sampled = []
        for model in gpr_models:
            p = model.p
            mean = model.a[:p]
            cov = 0.5 * (model.C[:p, :p] + model.C[:p, :p].T)
            eig = np.linalg.eigvalsh(cov)
            if np.min(eig) < 1e-12:
                cov = cov + (1e-12 - np.min(eig)) * np.eye(p)
            try:
                sampled.append(rng.multivariate_normal(mean, cov))
            except np.linalg.LinAlgError:
                sampled.append(mean + 0.01 * rng.standard_normal(p))
        Phi = gpr_models[0].basis_matrix(X)
        obj = Phi @ sampled[0]
        if len(gpr_models) > 1 and use_constraint:
            con = Phi @ sampled[1]
            if variance_lookup is not None:
                sig = np.sqrt(np.array([variance_lookup(1, x) for x in X]))
                if (
                    str(certification_mode).lower() == "theory"
                    and epistemic_lookup is not None
                ):
                    s_g = np.sqrt(np.array([epistemic_lookup(x) for x in X]))
                    feasible = (
                        con
                        + np.sqrt(max(float(beta_g), 0.0)) * s_g
                        + alpha_z * sig
                        <= tau
                    )
                else:
                    feasible = con + alpha_z * sig <= tau
            else:
                feasible = con <= tau
        else:
            feasible = np.ones(len(pool), dtype=bool)
        order = np.lexsort((obj, ~feasible))
        for idx in order[:keep_per_batch]:
            candidates.append(tuple(int(v) for v in X[idx]))
    return candidates


def unique_candidates(candidates):
    seen = set()
    out = []
    for x in candidates:
        x_tuple = tuple(int(v) for v in x)
        if x_tuple not in seen:
            seen.add(x_tuple)
            out.append(x_tuple)
    return out
