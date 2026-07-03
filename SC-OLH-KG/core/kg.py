"""Knowledge-gradient utilities with candidate feature caching."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from core.gpr import ParametricGPR


def compute_h(a, b):
    """Compute `E[max_j a_j + b_j Z] - max_j a_j` for `Z ~ N(0, 1)`.

    This is the Frazier-Powell line-envelope calculation.  Unlike the legacy
    implementation, it avoids the O(M^2) dominated-line pre-pass; slopes are
    sorted and duplicate slopes keep only the largest intercept.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) <= 1:
        return 0.0

    order = np.lexsort((-a, b))
    a_sorted = a[order]
    b_sorted = b[order]

    # Collapse equal slopes: for max lines, only the highest intercept matters.
    keep_a = []
    keep_b = []
    i = 0
    while i < len(a_sorted):
        j = i + 1
        best = a_sorted[i]
        while j < len(a_sorted) and abs(b_sorted[j] - b_sorted[i]) <= 1e-14:
            best = max(best, a_sorted[j])
            j += 1
        keep_a.append(best)
        keep_b.append(b_sorted[i])
        i = j

    a_k = np.asarray(keep_a, dtype=float)
    b_k = np.asarray(keep_b, dtype=float)
    if len(a_k) <= 1:
        return 0.0

    hull_a = []
    hull_b = []
    cuts = []
    for aj, bj in zip(a_k, b_k):
        while hull_a:
            if abs(bj - hull_b[-1]) <= 1e-14:
                z = np.inf if aj > hull_a[-1] else -np.inf
            else:
                z = (hull_a[-1] - aj) / (bj - hull_b[-1])
            if not cuts or z > cuts[-1]:
                break
            hull_a.pop()
            hull_b.pop()
            if cuts:
                cuts.pop()
        if not hull_a:
            cuts.append(-np.inf)
        else:
            cuts.append((hull_a[-1] - aj) / (bj - hull_b[-1]))
        hull_a.append(float(aj))
        hull_b.append(float(bj))

    if len(hull_a) <= 1:
        return 0.0

    h_val = 0.0
    for idx, (aj, bj) in enumerate(zip(hull_a, hull_b)):
        z_lo = cuts[idx]
        z_hi = np.inf if idx == len(hull_a) - 1 else cuts[idx + 1]
        if z_hi <= z_lo:
            continue
        p = norm.cdf(z_hi) - norm.cdf(z_lo)
        if p <= 1e-15:
            continue
        phi_lo = 0.0 if np.isneginf(z_lo) else norm.pdf(z_lo)
        phi_hi = 0.0 if np.isposinf(z_hi) else norm.pdf(z_hi)
        h_val += aj * p + bj * (phi_lo - phi_hi)
    return max(float(h_val - np.max(a)), 0.0)


def compute_kg_factor(
    gpr: ParametricGPR,
    candidate_set,
    x,
    sigma2_hat: float,
    candidate_aug=None,
    mu_vec=None,
) -> float:
    """Compute one KG factor for one sampling point."""
    if len(candidate_set) == 0:
        return 0.0
    A = candidate_aug
    if A is None:
        A = gpr.augmented_feature_matrix(candidate_set)
    mu = mu_vec
    if mu is None:
        mu = A @ gpr.a
    e = gpr.augmented_feature(x)
    Ce = gpr.C @ e
    denom = np.sqrt(max(float(sigma2_hat) + float(e @ Ce), 1e-15))
    sigma_tilde = (A @ Ce) / denom
    return compute_h(-mu, -sigma_tilde)


def compute_kg_vectorized(
    gpr: ParametricGPR,
    candidate_set,
    sigma2_hats,
    sample_points=None,
) -> np.ndarray:
    """Compute KG for many sampling points against one candidate set.

    The candidate augmented matrix and posterior means are built once.  This is
    the main speed fix over repeatedly calling the scalar legacy function.
    """
    if sample_points is None:
        sample_points = candidate_set
    if len(candidate_set) == 0 or len(sample_points) == 0:
        return np.zeros(0, dtype=float)

    candidate_aug = gpr.augmented_feature_matrix(candidate_set)
    sample_aug = gpr.augmented_feature_matrix(sample_points)
    mu_vec = candidate_aug @ gpr.a

    sigma2 = np.asarray(sigma2_hats, dtype=float)
    if sigma2.ndim == 0:
        sigma2 = np.full(len(sample_points), float(sigma2))
    if len(sigma2) != len(sample_points):
        raise ValueError("sigma2_hats length must match sample_points")

    Ce = gpr.C @ sample_aug.T
    denom_vals = sigma2 + np.einsum("ij,ji->i", sample_aug, Ce)
    denom = np.sqrt(np.maximum(denom_vals, 1e-15))
    sigma_tilde = (candidate_aug @ Ce) / denom[None, :]
    return np.array([
        compute_h(-mu_vec, -sigma_tilde[:, j])
        for j in range(len(sample_points))
    ], dtype=float)
