"""Noise-limited oracle diagnostics for chance-constraint certification.

These utilities deliberately assume more information than an optimizer has:
the true chance margin and the true observation standard deviation.  They are
therefore lower bounds on the replication burden, not deployable certificates.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm, t as student_t


def oracle_mean_radius(
    sigma,
    replicates,
    *,
    confidence_alpha=0.01,
    variance_known=True,
):
    """Return the one-sided radius for an oracle sample-mean estimate."""

    sigma = np.maximum(np.asarray(sigma, dtype=float), 0.0)
    replicates = int(replicates)
    if replicates < 1:
        raise ValueError("replicates must be positive")
    alpha = float(np.clip(confidence_alpha, 1e-12, 0.5 - 1e-12))
    if variance_known:
        quantile = float(norm.ppf(1.0 - alpha))
    elif replicates < 2:
        return np.full_like(sigma, np.inf, dtype=float)
    else:
        quantile = float(student_t.ppf(1.0 - alpha, replicates - 1))
    return quantile * sigma / math.sqrt(replicates)


def oracle_certification_upper(
    true_margin,
    sigma,
    replicates,
    *,
    confidence_alpha=0.01,
    variance_known=True,
):
    """Optimistic upper bound when only the constraint mean must be learned."""

    true_margin = np.asarray(true_margin, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if true_margin.shape != sigma.shape:
        raise ValueError("true_margin and sigma must have the same shape")
    return true_margin + oracle_mean_radius(
        sigma,
        replicates,
        confidence_alpha=confidence_alpha,
        variance_known=variance_known,
    )


def required_replications_known_variance(
    true_margin,
    sigma,
    *,
    confidence_alpha=0.01,
):
    """Optimistic replication count needed for a known-variance certificate.

    Unsafe points have infinite required replication.  A feasible point with
    margin ``m < 0`` needs ``ceil((z sigma / -m)^2)`` replications.
    """

    margin = np.asarray(true_margin, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 0.0)
    if margin.shape != sigma.shape:
        raise ValueError("true_margin and sigma must have the same shape")
    quantile = float(norm.ppf(
        1.0 - float(np.clip(confidence_alpha, 1e-12, 0.5 - 1e-12))))
    required = np.full(margin.shape, np.inf, dtype=float)
    strictly_feasible = margin < 0.0
    ratio = np.zeros_like(margin, dtype=float)
    ratio[strictly_feasible] = (
        quantile * sigma[strictly_feasible] / -margin[strictly_feasible]
    ) ** 2
    required[strictly_feasible] = np.maximum(
        np.ceil(ratio[strictly_feasible]), 1.0)
    exact_safe = (margin == 0.0) & (sigma == 0.0)
    required[exact_safe] = 1.0
    return required


def _finite_quantiles(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            "minimum": None,
            "q25": None,
            "median": None,
            "q90": None,
            "maximum": None,
        }
    quantiles = np.quantile(values, [0.0, 0.25, 0.5, 0.9, 1.0])
    return {
        "minimum": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q90": float(quantiles[3]),
        "maximum": float(quantiles[4]),
    }


def oracle_certifiability_metrics(
    true_margin,
    sigma,
    replicate_budgets=(1, 3, 5, 10, 20, 50, 100),
    *,
    confidence_alpha=0.01,
):
    """Summarize the best possible certificate under direct replication."""

    margin = np.asarray(true_margin, dtype=float).reshape(-1)
    sigma = np.asarray(sigma, dtype=float).reshape(-1)
    if len(margin) == 0 or len(margin) != len(sigma):
        raise ValueError("nonempty margin and sigma arrays must match")
    if not np.all(np.isfinite(margin)) or not np.all(np.isfinite(sigma)):
        raise ValueError("margin and sigma must be finite")
    budgets = tuple(sorted(set(int(value) for value in replicate_budgets)))
    if not budgets or budgets[0] < 1:
        raise ValueError("replicate budgets must be positive")

    feasible = margin <= 0.0
    feasible_count = int(np.sum(feasible))
    required = required_replications_known_variance(
        margin,
        sigma,
        confidence_alpha=confidence_alpha,
    )
    budget_rows = {}
    for budget in budgets:
        known = oracle_certification_upper(
            margin,
            sigma,
            budget,
            confidence_alpha=confidence_alpha,
            variance_known=True,
        ) <= 0.0
        unknown = oracle_certification_upper(
            margin,
            sigma,
            budget,
            confidence_alpha=confidence_alpha,
            variance_known=False,
        ) <= 0.0
        budget_rows[str(budget)] = {
            "known_variance_count": int(np.sum(known)),
            "known_variance_pool_rate": float(np.mean(known)),
            "known_variance_feasible_recall": float(
                np.sum(known & feasible) / max(feasible_count, 1)),
            "unknown_variance_count": int(np.sum(unknown)),
            "unknown_variance_pool_rate": float(np.mean(unknown)),
            "unknown_variance_feasible_recall": float(
                np.sum(unknown & feasible) / max(feasible_count, 1)),
        }

    return {
        "pool_size": int(len(margin)),
        "confidence_alpha": float(confidence_alpha),
        "true_feasible_count": feasible_count,
        "true_feasible_rate": float(np.mean(feasible)),
        "minimum_true_margin": float(np.min(margin)),
        "minimum_constraint_sigma": float(np.min(sigma)),
        "median_constraint_sigma": float(np.median(sigma)),
        "maximum_constraint_sigma": float(np.max(sigma)),
        "known_variance_required_replications": _finite_quantiles(
            required[feasible]),
        "replicate_budgets": budget_rows,
        "interpretation": (
            "optimistic_oracle_lower_bound_on_replication_burden"
        ),
        "promotion_eligible": False,
    }
