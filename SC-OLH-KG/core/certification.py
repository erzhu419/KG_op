"""Chance-certification utilities shared by acquisition and recommendation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


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
