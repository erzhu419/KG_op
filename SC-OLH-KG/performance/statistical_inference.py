"""Small deterministic helpers for preregistered confirmatory inference."""

from __future__ import annotations

import numpy as np
from scipy.stats import beta


def exact_binomial_interval(successes, trials, *, confidence_level=0.95):
    """Two-sided Clopper-Pearson interval for independent Bernoulli units."""

    successes = int(successes)
    trials = int(trials)
    confidence_level = float(confidence_level)
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("binomial counts must satisfy 0 <= successes <= trials")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    tail = 0.5 * (1.0 - confidence_level)
    lower = (
        0.0 if successes == 0
        else float(beta.ppf(tail, successes, trials - successes + 1))
    )
    upper = (
        1.0 if successes == trials
        else float(beta.ppf(
            1.0 - tail, successes + 1, trials - successes))
    )
    return [lower, upper]


def exact_binomial_lower_bound(successes, trials, *, confidence_level=0.95):
    """One-sided exact lower confidence bound for a Bernoulli success rate."""

    successes = int(successes)
    trials = int(trials)
    confidence_level = float(confidence_level)
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("binomial counts must satisfy 0 <= successes <= trials")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    if successes == 0:
        return 0.0
    return float(beta.ppf(
        1.0 - confidence_level,
        successes,
        trials - successes + 1,
    ))


def exact_binomial_upper_bound(successes, trials, *, confidence_level=0.95):
    """One-sided exact upper confidence bound for a Bernoulli success rate."""

    successes = int(successes)
    trials = int(trials)
    confidence_level = float(confidence_level)
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("binomial counts must satisfy 0 <= successes <= trials")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    if successes == trials:
        return 1.0
    return float(beta.ppf(
        confidence_level,
        successes + 1,
        trials - successes,
    ))


def holm_adjust(p_values):
    """Return Holm step-down adjusted p-values in their original order."""

    values = [float(value) for value in p_values]
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [1.0] * len(values)
    running = 0.0
    family_size = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (family_size - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def bootstrap_mean_ci(values, *, seed, repetitions=10_000):
    """Percentile CI for a mean over the supplied independent units."""

    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return None
    if len(values) == 1:
        value = float(values[0])
        return [value, value]
    rng = np.random.default_rng(int(seed))
    indexes = rng.integers(
        0, len(values), size=(int(repetitions), len(values)))
    means = np.mean(values[indexes], axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def apply_holm_family(rows, *, pvalue_field, family_field):
    """Add Holm p-values separately within each declared inference family."""

    families = {}
    for index, row in enumerate(rows):
        families.setdefault(str(row[family_field]), []).append(index)
    adjusted_field = f"{pvalue_field}_holm"
    for family_id, indexes in sorted(families.items()):
        adjusted = holm_adjust([rows[index][pvalue_field] for index in indexes])
        for index, value in zip(indexes, adjusted):
            rows[index][adjusted_field] = float(value)
            rows[index]["holm_family_id"] = family_id
            rows[index]["holm_family_size"] = len(indexes)
    return rows
