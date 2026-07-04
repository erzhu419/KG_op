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
