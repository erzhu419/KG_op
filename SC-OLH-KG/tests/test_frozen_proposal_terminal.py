import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.benchmark_frozen_proposal_only import (  # noqa: E402
    _select_shortlist,
)
from performance.benchmark_lodo_meta_prior import (  # noqa: E402
    build_scalarized_problem,
)


def test_proposal_only_v9_freezes_three_candidates_without_truth():
    problem = build_scalarized_problem(
        "FactorShockStatePolicyRZDT1",
        5,
        100,
        0.04,
        0.05,
        (0.5, 0.5),
    )
    points = [
        (5, 5, 5, 5, 5),
        (20, 20, 20, 20, 20),
        (50, 50, 50, 50, 50),
        (90, 90, 90, 90, 90),
    ]
    observations = np.asarray([
        [0.5, -0.10],
        [0.1, -0.08],
        [0.4, -0.20],
        [0.0, 0.10],
    ])
    shortlist, audit = _select_shortlist(
        problem,
        points,
        observations,
        shortlist_mode="posterior_objective_challenger_then_safe",
        shortlist_size=3,
        maximum_violation_probability=0.5,
    )
    assert len(shortlist) == 3
    assert len({tuple(row["point"]) for row in shortlist}) == 3
    assert shortlist[0]["selected_posterior_objective"] == 0.1
    assert audit["posterior_contract"] == (
        "one_shot_gaussian_plugin_not_fitted_surrogate")
    assert audit["target_observations_used"] is True
    assert audit["target_oracle_used"] is False
