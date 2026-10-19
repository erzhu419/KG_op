"""Fixed three-candidate terminal tournament registered for local preparation."""
from __future__ import annotations

import copy

from core.terminal_verification import (
    _verification_method,
    verify_frozen_policy,
    verify_paired_objective_dominance,
)


def verify_frozen_three_candidate_tournament(
    problem,
    frozen_shortlist,
    *,
    seed,
    search_evaluation_count,
    method="normal_quantile_tolerance",
    mean_delta_fraction=0.5,
    shortlist_mode="posterior_objective_challenger_then_safe",
):
    """Compare the frozen third candidate with the certified first-stage winner.

    Positions are fixed: primary, initial incumbent, third candidate. All three
    receive safety checks; only two certified policies enter a comparison.
    The second comparison uses block one even when block zero was unnecessary.
    """
    records = copy.deepcopy(list(frozen_shortlist))
    points = [tuple(int(value) for value in row["point"]) for row in records]
    if len(points) != 3 or len(set(points)) != 3:
        raise ValueError("the registered tournament requires three distinct frozen policies")
    budgets = [80, 120, 120]
    familywise_delta = 0.05
    per_candidate_delta = familywise_delta / 3
    comparison_delta = 0.05 / 6
    attempts, comparisons = [], []

    def safety(index):
        result = verify_frozen_policy(
            problem, points[index], seed=seed,
            search_evaluation_count=search_evaluation_count,
            verification_budget=budgets[index], delta=per_candidate_delta,
            candidate_index=index, method=method,
            mean_delta_fraction=mean_delta_fraction,
        )
        attempts.append(result)
        return bool(result["certified"])

    def compare(challenger, incumbent, block_index):
        result = verify_paired_objective_dominance(
            problem, points[challenger], points[incumbent], seed=seed,
            comparison_budget=8, delta=comparison_delta,
            comparison_block_index=block_index,
        )
        winner = challenger if result["challenger_dominates"] else incumbent
        comparisons.append(dict(result, block_index=block_index,
            challenger_index=challenger, incumbent_index=incumbent,
            winner_index=winner))
        return winner

    primary_safe, incumbent_safe = safety(0), safety(1)
    if primary_safe and incumbent_safe:
        winner = compare(0, 1, 0)
    elif primary_safe:
        winner = 0
    elif incumbent_safe:
        winner = 1
    else:
        winner = None
    initial_stage_winner = winner
    third_safe = safety(2)
    if third_safe:
        winner = 2 if winner is None else compare(2, winner, 1)

    certified = winner is not None
    deployed = points[winner] if certified else points[0]
    safety_budget = sum(row["verification_budget"] for row in attempts)
    objective_budget = sum(row["simulation_calls"] for row in comparisons)
    actual_budget = safety_budget + objective_budget
    _, method_label = _verification_method(method)
    return deployed, {
        "enabled": True,
        "status": "certified" if certified else "abstained_uncertified",
        "method": f"frozen_three_candidate_tournament_{method_label}",
        "protocol": "frozen_three_candidate_two_objective_comparisons",
        "shortlist_mode": str(shortlist_mode),
        "policy_frozen_before_verification": True,
        "shortlist_frozen_before_verification": True,
        "search_samples_reused": False,
        "posterior_updated_from_verification": False,
        "verification_budget": int(actual_budget),
        "safety_verification_budget": int(safety_budget),
        "objective_comparison_budget": int(objective_budget),
        "objective_comparison_budget_per_policy": 8,
        "objective_delta_per_comparison": comparison_delta,
        "objective_familywise_delta": 0.05 / 3,
        "maximum_objective_comparisons": 2,
        "verification_budget_per_candidate": budgets[0],
        "fallback_verification_budget": budgets[1],
        "candidate_verification_budgets": budgets,
        "max_verification_budget": 352,
        "search_evaluation_count": int(search_evaluation_count),
        "total_evaluation_count": int(search_evaluation_count) + int(actual_budget),
        "familywise_delta": familywise_delta,
        "per_candidate_delta": per_candidate_delta,
        "mean_delta_fraction": float(mean_delta_fraction),
        "frozen_shortlist_size": 3,
        "frozen_shortlist": records,
        "attempts": attempts,
        "attempt_count": len(attempts),
        "objective_incumbent_guard_enabled": True,
        "objective_incumbent_position": 2,
        "objective_comparisons": comparisons,
        "initial_stage_winner_index": initial_stage_winner,
        "certified": certified,
        "selected_shortlist_rank": winner + 1 if certified else None,
        "recommendation_changed": deployed != points[0],
        "target_oracle_used": False,
        "certification_scope": "familywise_frozen_three_candidate_independent_paired_objective_tournament",
        "verification_samples_logged": False,
    }
