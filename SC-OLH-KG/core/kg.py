"""Knowledge-gradient utilities with candidate feature caching."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from core.gpr import ParametricGPR


@dataclass(frozen=True)
class LineEnvelopeCertificate:
    """Checkable certificate for the Frazier-Powell line envelope.

    The active interval for hull line `i` is `[cuts[i], cuts[i+1]]`, with
    `+inf` used as the implicit final upper cut.  The Gaussian first moment is
    `E[Z 1{lo <= Z <= hi}]`, so each contribution is
    `a_i * prob_mass_i + b_i * first_moment_i`.
    """

    input_size: int
    baseline: float
    envelope_expectation: float
    h_value: float
    hull_indices: tuple[int, ...]
    hull_intercepts: tuple[float, ...]
    hull_slopes: tuple[float, ...]
    cuts: tuple[float, ...]
    prob_masses: tuple[float, ...]
    first_moments: tuple[float, ...]
    contributions: tuple[float, ...]


def _as_line_arrays(a, b):
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if len(a) != len(b):
        raise ValueError("a and b must have the same length")
    return a, b


def _build_line_envelope(a, b, slope_tol=1e-14):
    """Return active line indices, intercepts, slopes, and lower cuts."""
    a, b = _as_line_arrays(a, b)
    if len(a) == 0:
        return [], [], [], []

    order = np.lexsort((-a, b))
    a_sorted = a[order]
    b_sorted = b[order]

    # Collapse equal slopes: for max lines, only the highest intercept matters.
    keep_idx = []
    keep_a = []
    keep_b = []
    i = 0
    while i < len(a_sorted):
        j = i + 1
        best_a = float(a_sorted[i])
        best_idx = int(order[i])
        while j < len(a_sorted) and abs(b_sorted[j] - b_sorted[i]) <= slope_tol:
            if float(a_sorted[j]) > best_a:
                best_a = float(a_sorted[j])
                best_idx = int(order[j])
            j += 1
        keep_idx.append(best_idx)
        keep_a.append(best_a)
        keep_b.append(float(b_sorted[i]))
        i = j

    hull_idx = []
    hull_a = []
    hull_b = []
    cuts = []
    for original_idx, aj, bj in zip(keep_idx, keep_a, keep_b):
        while hull_a:
            z = (hull_a[-1] - aj) / (bj - hull_b[-1])
            if not cuts or z > cuts[-1]:
                break
            hull_idx.pop()
            hull_a.pop()
            hull_b.pop()
            if cuts:
                cuts.pop()
        if not hull_a:
            cuts.append(-np.inf)
        else:
            cuts.append((hull_a[-1] - aj) / (bj - hull_b[-1]))
        hull_idx.append(int(original_idx))
        hull_a.append(float(aj))
        hull_b.append(float(bj))
    return hull_idx, hull_a, hull_b, cuts


def _integrate_line_envelope(a, hull_a, hull_b, cuts):
    if len(hull_a) == 0:
        return 0.0, 0.0, [], [], []
    baseline = float(np.max(a))
    prob_masses = []
    first_moments = []
    contributions = []
    envelope_expectation = 0.0
    for idx, (aj, bj) in enumerate(zip(hull_a, hull_b)):
        z_lo = cuts[idx]
        z_hi = np.inf if idx == len(hull_a) - 1 else cuts[idx + 1]
        if z_hi <= z_lo:
            prob_masses.append(0.0)
            first_moments.append(0.0)
            contributions.append(0.0)
            continue
        p = float(norm.cdf(z_hi) - norm.cdf(z_lo))
        phi_lo = 0.0 if np.isneginf(z_lo) else float(norm.pdf(z_lo))
        phi_hi = 0.0 if np.isposinf(z_hi) else float(norm.pdf(z_hi))
        first_moment = phi_lo - phi_hi
        contribution = float(aj * p + bj * first_moment)
        prob_masses.append(p)
        first_moments.append(float(first_moment))
        contributions.append(contribution)
        envelope_expectation += contribution
    h_value = max(float(envelope_expectation - baseline), 0.0)
    return h_value, baseline, prob_masses, first_moments, contributions


def compute_h(a, b):
    """Compute `E[max_j a_j + b_j Z] - max_j a_j` for `Z ~ N(0, 1)`.

    This is the Frazier-Powell line-envelope calculation.  Unlike the legacy
    implementation, it avoids the O(M^2) dominated-line pre-pass; slopes are
    sorted and duplicate slopes keep only the largest intercept.
    """
    a, b = _as_line_arrays(a, b)
    if len(a) <= 1:
        return 0.0

    _, hull_a, hull_b, cuts = _build_line_envelope(a, b)
    if len(hull_a) <= 1:
        return 0.0
    h_value, _, _, _, _ = _integrate_line_envelope(a, hull_a, hull_b, cuts)
    return h_value


def compute_h_certificate(a, b) -> LineEnvelopeCertificate:
    """Compute `compute_h` together with a checkable line-envelope certificate."""
    a, b = _as_line_arrays(a, b)
    if len(a) == 0:
        return LineEnvelopeCertificate(
            input_size=0,
            baseline=0.0,
            envelope_expectation=0.0,
            h_value=0.0,
            hull_indices=(),
            hull_intercepts=(),
            hull_slopes=(),
            cuts=(),
            prob_masses=(),
            first_moments=(),
            contributions=(),
        )
    hull_idx, hull_a, hull_b, cuts = _build_line_envelope(a, b)
    h_value, baseline, probs, moments, contributions = _integrate_line_envelope(
        a, hull_a, hull_b, cuts)
    return LineEnvelopeCertificate(
        input_size=int(len(a)),
        baseline=float(baseline),
        envelope_expectation=float(sum(contributions)),
        h_value=float(h_value),
        hull_indices=tuple(int(v) for v in hull_idx),
        hull_intercepts=tuple(float(v) for v in hull_a),
        hull_slopes=tuple(float(v) for v in hull_b),
        cuts=tuple(float(v) for v in cuts),
        prob_masses=tuple(float(v) for v in probs),
        first_moments=tuple(float(v) for v in moments),
        contributions=tuple(float(v) for v in contributions),
    )


def validate_h_certificate(a, b, certificate=None, atol=1e-8):
    """Validate a `compute_h_certificate` result against all original lines.

    The endpoint checks are exact for affine differences on finite intervals.
    On the two Gaussian tails, slope dominance plus boundary dominance certifies
    the whole half-line.  The return shape is intentionally JSON-friendly.
    """
    a, b = _as_line_arrays(a, b)
    cert = certificate or compute_h_certificate(a, b)
    errors = []
    if cert.input_size != len(a):
        errors.append("input_size mismatch")
    if len(a) == 0:
        return {"valid": not errors, "errors": errors}
    if len(cert.hull_indices) != len(cert.hull_intercepts):
        errors.append("hull index/intercept length mismatch")
    if len(cert.hull_indices) != len(cert.hull_slopes):
        errors.append("hull index/slope length mismatch")
    if len(cert.hull_indices) != len(cert.cuts):
        errors.append("hull/cut length mismatch")
    if len(cert.hull_indices) != len(cert.contributions):
        errors.append("hull/contribution length mismatch")
    if errors:
        return {"valid": False, "errors": errors}

    m = len(cert.hull_indices)
    if m == 0:
        errors.append("nonempty input has empty hull")
        return {"valid": False, "errors": errors}
    if not np.isneginf(cert.cuts[0]):
        errors.append("first cut is not -inf")
    for i, original_idx in enumerate(cert.hull_indices):
        if original_idx < 0 or original_idx >= len(a):
            errors.append(f"hull index {i} out of bounds")
            continue
        if abs(cert.hull_intercepts[i] - float(a[original_idx])) > atol:
            errors.append(f"hull intercept {i} does not match input")
        if abs(cert.hull_slopes[i] - float(b[original_idx])) > atol:
            errors.append(f"hull slope {i} does not match input")
    for i in range(1, m):
        if not cert.hull_slopes[i - 1] < cert.hull_slopes[i] + atol:
            errors.append(f"hull slopes not increasing at {i}")
        if not cert.cuts[i - 1] < cert.cuts[i]:
            errors.append(f"cuts not strictly increasing at {i}")

    def value(ai, bi, z):
        return float(ai + bi * z)

    for i in range(m):
        ai = cert.hull_intercepts[i]
        bi = cert.hull_slopes[i]
        lo = cert.cuts[i]
        hi = np.inf if i == m - 1 else cert.cuts[i + 1]
        endpoints = []
        if np.isfinite(lo):
            endpoints.append(lo)
        if np.isfinite(hi):
            endpoints.append(hi)
        for j in range(len(a)):
            aj = float(a[j])
            bj = float(b[j])
            if np.isneginf(lo) and bj < bi - atol:
                errors.append(f"left-tail slope violation: active={i} line={j}")
            if np.isposinf(hi) and bj > bi + atol:
                errors.append(f"right-tail slope violation: active={i} line={j}")
            for z in endpoints:
                if value(aj, bj, z) > value(ai, bi, z) + atol:
                    errors.append(
                        f"endpoint dominance violation: active={i} line={j}")
                    break

    expected_h = compute_h(a, b)
    if abs(cert.h_value - expected_h) > max(atol, atol * abs(expected_h)):
        errors.append("h_value mismatch")
    if abs(cert.baseline - float(np.max(a))) > atol:
        errors.append("baseline mismatch")
    if abs(cert.envelope_expectation - sum(cert.contributions)) > atol:
        errors.append("envelope/contribution mismatch")
    if abs(cert.h_value - max(cert.envelope_expectation - cert.baseline, 0.0)) > atol:
        errors.append("KG formula mismatch")
    return {"valid": not errors, "errors": errors}


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
