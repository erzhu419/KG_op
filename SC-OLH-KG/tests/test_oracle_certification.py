from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.oracle_certification import (  # noqa: E402
    oracle_certifiability_metrics,
    oracle_certification_upper,
    required_replications_known_variance,
)


def test_known_variance_required_replications_matches_closed_form():
    margin = np.array([-1.0, -0.1, 0.2])
    sigma = np.array([0.1, 0.1, 0.1])
    required = required_replications_known_variance(
        margin, sigma, confidence_alpha=0.05)
    assert required[0] == 1.0
    assert required[1] == 3.0
    assert np.isinf(required[2])
    upper = oracle_certification_upper(
        margin, sigma, int(required[1]), confidence_alpha=0.05)
    assert upper[1] <= 0.0


def test_oracle_certifiability_is_monotone_in_replication_budget():
    margin = np.array([-0.20, -0.05, 0.10])
    sigma = np.array([0.10, 0.10, 0.10])
    metrics = oracle_certifiability_metrics(
        margin,
        sigma,
        replicate_budgets=(1, 3, 10, 100),
        confidence_alpha=0.01,
    )
    known = [
        metrics["replicate_budgets"][str(value)][
            "known_variance_count"]
        for value in (1, 3, 10, 100)
    ]
    unknown = [
        metrics["replicate_budgets"][str(value)][
            "unknown_variance_count"]
        for value in (1, 3, 10, 100)
    ]
    assert known == sorted(known)
    assert unknown == sorted(unknown)
    assert unknown[0] == 0
    assert metrics["promotion_eligible"] is False
