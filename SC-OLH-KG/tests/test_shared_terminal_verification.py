import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algorithms.single_olhkg import (  # noqa: E402
    SingleOLHKGAlgorithm,
    SingleOLHKGConfig,
)
from core.terminal_verification import (  # noqa: E402
    build_verification_aware_shortlist,
    parse_verification_candidate_budgets,
    select_objective_verification_challenger,
    select_posterior_safe_interior,
    verify_frozen_policy,
    verify_frozen_shortlist,
)
from problems.rzdt import (  # noqa: E402
    FactorShockStatePolicyRZDT1,
    RZDT1,
)
from problems.single_objective import ScalarizedProblem  # noqa: E402


class _TwoPolicyGaussianProblem:
    d = 1
    alpha = 0.05
    tau = 0.0
    simulation_noise_model = "iid_gaussian"

    def simulate(self, point, rng):
        unsafe = int(point[0]) == 0
        constraint = (1.0 if unsafe else -1.0) + rng.normal(0.0, 0.01)
        return np.asarray([float(point[0]), constraint], dtype=float)


def test_v9_shortlist_freezes_challenger_primary_and_distinct_support():
    problem = ScalarizedProblem(FactorShockStatePolicyRZDT1(
        d=5, L=100, sigma=0.04, alpha=0.05))
    points = [
        (5, 5, 5, 5, 5),
        (20, 20, 20, 20, 20),
        (50, 50, 50, 50, 50),
        (90, 90, 90, 90, 90),
    ]
    primary = points[0]
    shortlist, audit = build_verification_aware_shortlist(
        problem,
        primary,
        points,
        objective_mean=[0.8, 0.1, 0.4, 0.0],
        probability_violation=[0.01, 0.40, 0.03, 0.90],
        shortlist_size=3,
        maximum_violation_probability=0.5,
        probability_slack=0.05,
        require_provider=True,
    )
    selected = [tuple(row["point"]) for row in shortlist]
    assert selected[0] == points[1]
    assert selected[1] == primary
    assert len(selected) == len(set(selected)) == 3
    assert shortlist[0]["shortlist_role"] == (
        "posterior_objective_verification_challenger")
    assert shortlist[1]["shortlist_role"] == (
        "posterior_feasible_primary_fallback")
    assert audit["shortlist_mode"] == (
        "posterior_objective_challenger_then_safe")
    assert audit["target_oracle_used"] is False
    assert audit["verification_samples_used"] is False


def test_explicit_candidate_budget_parser_preserves_order():
    assert parse_verification_candidate_budgets(
        "80,128,128", default=(1, 2)
    ) == (80, 128, 128)
    assert parse_verification_candidate_budgets(
        "", default=(80, 96)
    ) == (80, 96)


def test_shared_single_policy_verifier_is_v64_byte_equivalent():
    problem = ScalarizedProblem(
        RZDT1(d=5, L=20, sigma=0.01, heteroscedastic=True))
    algorithm = SingleOLHKGAlgorithm(
        problem,
        SingleOLHKGConfig(
            N=2,
            n0=2,
            seed=19,
            terminal_verification_budget=12,
            terminal_verification_delta=0.05,
            terminal_verification_method="normal_quantile_tolerance",
        ),
    )
    point = (0,) * 5
    wrapped = algorithm._terminal_fixed_policy_verification(point)
    shared = verify_frozen_policy(
        problem,
        point,
        seed=19,
        search_evaluation_count=0,
        verification_budget=12,
        delta=0.05,
        method="normal_quantile_tolerance",
    )
    assert wrapped == shared


def test_ordered_verification_spends_fallback_only_after_primary_failure():
    problem = _TwoPolicyGaussianProblem()
    shortlist = [
        {
            "shortlist_position": 1,
            "shortlist_role": "posterior_bayes_primary",
            "point": [0],
        },
        {
            "shortlist_position": 2,
            "shortlist_role": "posterior_safe_interior_diversified",
            "point": [1],
        },
    ]
    deployed, audit = verify_frozen_shortlist(
        problem,
        shortlist,
        seed=7,
        search_evaluation_count=13,
        candidate_budgets=(80, 96),
        familywise_delta=0.05,
        method="normal_quantile_tolerance",
    )
    assert deployed == (1,)
    assert audit["selected_shortlist_rank"] == 2
    assert audit["attempt_count"] == 2
    assert audit["verification_budget"] == 176
    assert audit["total_evaluation_count"] == 189
    assert audit["per_candidate_delta"] == 0.025
    assert audit["attempts"][0]["certified"] is False
    assert audit["attempts"][1]["certified"] is True
    assert audit["posterior_updated_from_verification"] is False
    assert audit["search_samples_reused"] is False
    assert audit["target_oracle_used"] is False


def test_safe_interior_selector_uses_only_posterior_and_common_risk_coordinate():
    problem = ScalarizedProblem(FactorShockStatePolicyRZDT1(
        d=5, L=100, sigma=0.04, alpha=0.05))
    primary = (2, 2, 2, 2, 2)
    points = [
        (1, 1, 1, 1, 1),
        (10, 10, 10, 10, 10),
        (90, 90, 90, 90, 90),
    ]
    support = select_posterior_safe_interior(
        problem,
        primary,
        points,
        [0.10, 0.12, 0.80],
        probability_slack=0.05,
        require_provider=True,
    )
    assert support["point"] == points[1]
    assert support["eligible_count"] == 2
    assert support["coordinate_source"] == "cumulative_risk_psi=(A,N)"
    assert support["target_labels_used"] is False
    assert support["target_oracle_used"] is False
    assert support["verification_samples_used"] is False


def test_safe_interior_selector_can_rank_safe_sublevel_by_objective():
    problem = ScalarizedProblem(FactorShockStatePolicyRZDT1(
        d=5, L=100, sigma=0.04, alpha=0.05))
    primary = (2, 2, 2, 2, 2)
    points = [
        (1, 1, 1, 1, 1),
        (10, 10, 10, 10, 10),
        (90, 90, 90, 90, 90),
    ]
    support = select_posterior_safe_interior(
        problem,
        primary,
        points,
        [0.10, 0.12, 0.80],
        objective_mean=[0.1, 0.5, 0.0],
        selection_mode="objective_ranked",
        probability_slack=0.05,
        require_provider=True,
    )
    assert support["point"] == points[0]
    assert support["selection_mode"] == "objective_ranked"
    assert support["selected_posterior_objective"] == 0.1
    assert support["selection_contract"] == (
        "posterior_violation_sublevel_minimum_posterior_objective")


def test_objective_safe_selector_uses_global_safety_reference():
    problem = ScalarizedProblem(FactorShockStatePolicyRZDT1(
        d=5, L=100, sigma=0.04, alpha=0.05))
    primary = (2, 2, 2, 2, 2)
    points = [
        primary,
        (10, 10, 10, 10, 10),
        (90, 90, 90, 90, 90),
    ]
    support = select_posterior_safe_interior(
        problem,
        primary,
        points,
        [0.01, 0.04, 0.07],
        objective_mean=[1.0, 0.5, 0.0],
        selection_mode="objective_safe_ranked",
        probability_slack=0.05,
        require_provider=True,
    )
    assert support["point"] == points[1]
    assert support["eligible_count"] == 1
    assert support["eligibility_reference"] == "global_minimum"
    assert support["selection_contract"] == (
        "posterior_global_violation_sublevel_"
        "minimum_posterior_objective")


def test_objective_verification_challenger_uses_posterior_median_feasibility():
    points = [(1,), (2,), (3,)]
    selected = select_objective_verification_challenger(
        points,
        objective_mean=[0.3, 0.1, 0.2],
        probability_violation=[0.1, 0.7, 0.4],
        maximum_violation_probability=0.5,
    )
    assert selected["point"] == points[2]
    assert selected["eligible_count"] == 2
    assert selected["selected_probability_violation"] == 0.4
    assert selected["verification_required_for_deployment"] is True
    assert selected["target_labels_used"] is False
    assert selected["target_oracle_used"] is False
    assert selected["verification_samples_used"] is False
