"""Chance-certification utilities shared by acquisition and recommendation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2, nct, norm, t


@dataclass(frozen=True)
class CertificationResult:
    """Vectorized conservative chance-bound components."""

    margin: np.ndarray
    mu: np.ndarray
    epistemic_var: np.ndarray
    aleatoric_var: np.ndarray
    beta_g: float
    z_alpha: float
    tau: float
    mode: str


@dataclass(frozen=True)
class GuardDecompositionResult:
    """Auditable decomposition of a conservative chance margin.

    A joint robust certificate can contain an extra source/task ambiguity
    correction beyond the nominal epistemic and aleatoric guards. Positive
    correction is epistemic uncertainty; negative correction is retained as
    favorable coupling so the decomposition reconstructs the exact margin.
    """

    mean_excess: np.ndarray
    epistemic_guard: np.ndarray
    joint_epistemic_guard: np.ndarray
    aleatoric_guard: np.ndarray
    favorable_coupling: np.ndarray
    reconstructed_margin: np.ndarray
    dominant_mode: np.ndarray


@dataclass(frozen=True)
class GaussianReplicationCertificate:
    """Fixed-policy finite-sample Gaussian chance certificate.

    The policy must be frozen before ``constraint_samples`` are drawn.  A
    Bonferroni split gives simultaneous one-sided bounds for the unknown mean
    and standard deviation.  Consequently, ``upper_margin <= 0`` certifies
    ``mu + z_(1-alpha) sigma <= tau`` with error probability at most
    ``delta`` under iid Gaussian replication noise.
    """

    upper_margin: float
    sample_mean: float
    sample_std: float
    mean_upper: float
    sigma_upper: float
    mean_radius: float
    z_alpha: float
    t_quantile: float
    chi2_quantile: float
    sample_count: int
    delta: float
    mean_delta: float
    variance_delta: float
    certified: bool
    status: str


@dataclass(frozen=True)
class GaussianQuantileToleranceCertificate:
    """Exact one-sided Gaussian quantile tolerance certificate.

    For iid Gaussian replications, ``sample_mean + k * sample_std`` is an
    upper confidence bound for ``mu + z_(1-alpha) sigma`` when ``k`` is the
    corresponding noncentral-Student-t tolerance factor.
    """

    upper_margin: float
    sample_mean: float
    sample_std: float
    quantile_upper: float
    tolerance_factor: float
    z_alpha: float
    noncentral_t_quantile: float
    noncentrality: float
    degrees_of_freedom: int
    sample_count: int
    delta: float
    certified: bool
    status: str
    method: str


def decompose_chance_margin(result: CertificationResult) -> GuardDecompositionResult:
    """Split a chance margin into mean, epistemic, and aleatoric guards."""

    mean_excess = np.asarray(result.mu, dtype=float) - float(result.tau)
    epistemic_guard = np.sqrt(max(float(result.beta_g), 0.0)) * np.sqrt(
        np.maximum(np.asarray(result.epistemic_var, dtype=float), 0.0)
    )
    aleatoric_guard = float(result.z_alpha) * np.sqrt(
        np.maximum(np.asarray(result.aleatoric_var, dtype=float), 0.0)
    )
    nominal = mean_excess + epistemic_guard + aleatoric_guard
    correction = np.asarray(result.margin, dtype=float) - nominal
    joint_epistemic_guard = np.maximum(correction, 0.0)
    favorable_coupling = np.minimum(correction, 0.0)
    reconstructed = (
        mean_excess
        + epistemic_guard
        + joint_epistemic_guard
        + aleatoric_guard
        + favorable_coupling
    )

    epistemic_total = epistemic_guard + joint_epistemic_guard
    mean_guard = np.maximum(mean_excess, 0.0)
    modes = np.full(np.shape(mean_excess), "interior", dtype=object)
    positive = np.asarray(result.margin, dtype=float) > 0.0
    epistemic_dominant = (
        positive
        & (epistemic_total >= aleatoric_guard)
        & (epistemic_total >= mean_guard)
    )
    aleatoric_dominant = (
        positive
        & ~epistemic_dominant
        & (aleatoric_guard >= mean_guard)
    )
    modes[epistemic_dominant] = "epistemic"
    modes[aleatoric_dominant] = "aleatoric"
    return GuardDecompositionResult(
        mean_excess=mean_excess,
        epistemic_guard=epistemic_guard,
        joint_epistemic_guard=joint_epistemic_guard,
        aleatoric_guard=aleatoric_guard,
        favorable_coupling=favorable_coupling,
        reconstructed_margin=reconstructed,
        dominant_mode=modes,
    )


def conservative_chance_margin(
    mu_g,
    epistemic_var_g,
    v_c_plus,
    tau: float,
    alpha: float,
    beta_g: float = 0.0,
    mode: str = "theory",
) -> CertificationResult:
    """Return `mu_g + sqrt(beta_g)s_g + z_alpha sqrt(v_C^+) - tau`.

    `legacy` mode keeps the same interface but zeroes the epistemic term,
    matching the older `mu + z sqrt(v)` bound for ablation.
    """

    mu = np.asarray(mu_g, dtype=float)
    epistemic = np.maximum(np.asarray(epistemic_var_g, dtype=float), 0.0)
    aleatoric = np.maximum(np.asarray(v_c_plus, dtype=float), 1e-12)
    mode = str(mode or "theory").lower()
    beta = 0.0 if mode == "legacy" else max(float(beta_g), 0.0)
    z_alpha = float(norm.ppf(1 - float(alpha)))
    margin = (
        mu
        + np.sqrt(beta) * np.sqrt(epistemic)
        + z_alpha * np.sqrt(aleatoric)
        - float(tau)
    )
    return CertificationResult(
        margin=np.asarray(margin, dtype=float),
        mu=mu,
        epistemic_var=epistemic,
        aleatoric_var=aleatoric,
        beta_g=float(beta),
        z_alpha=z_alpha,
        tau=float(tau),
        mode=mode,
    )


def conservative_chance_margin_scalar(
    mu_g: float,
    epistemic_var_g: float,
    v_c_plus: float,
    tau: float,
    alpha: float,
    beta_g: float = 0.0,
    mode: str = "theory",
) -> float:
    """Scalar wrapper for the conservative chance margin."""

    result = conservative_chance_margin(
        [mu_g],
        [epistemic_var_g],
        [v_c_plus],
        tau=tau,
        alpha=alpha,
        beta_g=beta_g,
        mode=mode,
    )
    return float(result.margin[0])


def gaussian_replication_chance_certificate(
    constraint_samples,
    *,
    tau: float,
    alpha: float,
    delta: float = 0.05,
    mean_delta_fraction: float = 0.5,
) -> GaussianReplicationCertificate:
    """Certify one frozen policy from independent Gaussian replications.

    Let ``n >= 2`` and ``S`` be the unbiased sample standard deviation.  The
    certificate uses

    ``mu_U = y_bar + t_(1-delta_mu,n-1) S / sqrt(n)``

    and

    ``sigma_U = S sqrt((n-1) / chi2_(delta_sigma,n-1))``.

    The returned chance margin is ``mu_U + z_(1-alpha) sigma_U - tau``.
    The search data used to choose the policy must not be reused as
    ``constraint_samples``; this keeps the fixed-policy coverage statement
    free of post-selection bias.
    """

    values = np.asarray(constraint_samples, dtype=float).reshape(-1)
    if np.any(~np.isfinite(values)):
        raise ValueError("constraint samples must be finite")
    if not 0.0 < float(alpha) <= 0.5:
        raise ValueError(
            "alpha must lie in (0, 0.5] so the Gaussian safety "
            "quantile is nonnegative")
    if not 0.0 < float(delta) < 1.0:
        raise ValueError("delta must lie strictly between zero and one")
    fraction = float(mean_delta_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError(
            "mean_delta_fraction must lie strictly between zero and one")

    n = int(len(values))
    delta_mu = float(delta) * fraction
    delta_sigma = float(delta) - delta_mu
    z_alpha = float(norm.ppf(1.0 - float(alpha)))
    if n < 2:
        sample_mean = (
            float(np.mean(values)) if n else float("nan"))
        return GaussianReplicationCertificate(
            upper_margin=float("inf"),
            sample_mean=sample_mean,
            sample_std=float("nan"),
            mean_upper=float("inf"),
            sigma_upper=float("inf"),
            mean_radius=float("inf"),
            z_alpha=z_alpha,
            t_quantile=float("inf"),
            chi2_quantile=0.0,
            sample_count=n,
            delta=float(delta),
            mean_delta=delta_mu,
            variance_delta=delta_sigma,
            certified=False,
            status="insufficient_replications",
        )

    degrees_of_freedom = n - 1
    sample_mean = float(np.mean(values))
    sample_std = float(np.std(values, ddof=1))
    t_quantile = float(t.ppf(
        1.0 - delta_mu, degrees_of_freedom))
    chi2_quantile = float(chi2.ppf(
        delta_sigma, degrees_of_freedom))
    if (
        not np.isfinite(t_quantile)
        or not np.isfinite(chi2_quantile)
        or chi2_quantile <= 0.0
    ):
        raise FloatingPointError(
            "nonfinite Student-t or chi-square certificate quantile")

    mean_radius = float(
        t_quantile * sample_std / np.sqrt(float(n)))
    mean_upper = float(sample_mean + mean_radius)
    sigma_upper = float(
        sample_std
        * np.sqrt(float(degrees_of_freedom) / chi2_quantile)
    )
    upper_margin = float(
        mean_upper + z_alpha * sigma_upper - float(tau))
    return GaussianReplicationCertificate(
        upper_margin=upper_margin,
        sample_mean=sample_mean,
        sample_std=sample_std,
        mean_upper=mean_upper,
        sigma_upper=sigma_upper,
        mean_radius=mean_radius,
        z_alpha=z_alpha,
        t_quantile=t_quantile,
        chi2_quantile=chi2_quantile,
        sample_count=n,
        delta=float(delta),
        mean_delta=delta_mu,
        variance_delta=delta_sigma,
        certified=bool(upper_margin <= 0.0),
        status="certified" if upper_margin <= 0.0 else "not_certified",
    )


def gaussian_quantile_tolerance_certificate(
    constraint_samples,
    *,
    tau: float,
    alpha: float,
    delta: float = 0.05,
) -> GaussianQuantileToleranceCertificate:
    """Directly bound a frozen policy's Gaussian chance quantile.

    If ``Y_i`` are iid ``Normal(mu, sigma^2)``, ``n >= 2``, and ``S`` is the
    unbiased sample standard deviation, then

    ``T = sqrt(n) * (mu + z sigma - sample_mean) / S``

    has a noncentral Student-t law with ``n - 1`` degrees of freedom and
    noncentrality ``z * sqrt(n)``. Hence

    ``sample_mean + nct.ppf(1-delta, n-1, z*sqrt(n)) * S / sqrt(n)``

    covers ``mu + z sigma`` with probability ``1-delta``. This avoids the
    extra Bonferroni split needed by separate mean and variance bounds.
    Search samples must not be reused for this fixed-policy certificate.
    """

    values = np.asarray(constraint_samples, dtype=float).reshape(-1)
    if np.any(~np.isfinite(values)):
        raise ValueError("constraint samples must be finite")
    if not 0.0 < float(alpha) <= 0.5:
        raise ValueError(
            "alpha must lie in (0, 0.5] so the Gaussian safety "
            "quantile is nonnegative")
    if not 0.0 < float(delta) < 1.0:
        raise ValueError("delta must lie strictly between zero and one")

    n = int(len(values))
    z_alpha = float(norm.ppf(1.0 - float(alpha)))
    if n < 2:
        sample_mean = (
            float(np.mean(values)) if n else float("nan"))
        return GaussianQuantileToleranceCertificate(
            upper_margin=float("inf"),
            sample_mean=sample_mean,
            sample_std=float("nan"),
            quantile_upper=float("inf"),
            tolerance_factor=float("inf"),
            z_alpha=z_alpha,
            noncentral_t_quantile=float("inf"),
            noncentrality=float("inf"),
            degrees_of_freedom=max(n - 1, 0),
            sample_count=n,
            delta=float(delta),
            certified=False,
            status="insufficient_replications",
            method="gaussian_noncentral_t_tolerance",
        )

    degrees_of_freedom = n - 1
    sample_mean = float(np.mean(values))
    sample_std = float(np.std(values, ddof=1))
    noncentrality = float(z_alpha * np.sqrt(float(n)))
    noncentral_t_quantile = float(nct.ppf(
        1.0 - float(delta),
        degrees_of_freedom,
        noncentrality,
    ))
    if not np.isfinite(noncentral_t_quantile):
        raise FloatingPointError(
            "nonfinite noncentral Student-t tolerance quantile")
    tolerance_factor = float(
        noncentral_t_quantile / np.sqrt(float(n)))
    quantile_upper = float(
        sample_mean + tolerance_factor * sample_std)
    upper_margin = float(quantile_upper - float(tau))
    return GaussianQuantileToleranceCertificate(
        upper_margin=upper_margin,
        sample_mean=sample_mean,
        sample_std=sample_std,
        quantile_upper=quantile_upper,
        tolerance_factor=tolerance_factor,
        z_alpha=z_alpha,
        noncentral_t_quantile=noncentral_t_quantile,
        noncentrality=noncentrality,
        degrees_of_freedom=degrees_of_freedom,
        sample_count=n,
        delta=float(delta),
        certified=bool(upper_margin <= 0.0),
        status="certified" if upper_margin <= 0.0 else "not_certified",
        method="gaussian_noncentral_t_tolerance",
    )
