"""Small deterministic helpers for preregistered confirmatory inference."""

from __future__ import annotations

import numpy as np


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
