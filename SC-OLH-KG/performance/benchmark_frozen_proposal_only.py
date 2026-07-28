#!/usr/bin/env python3
"""Evaluate a frozen source proposal without an online optimization backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.designs import (  # noqa: E402
    common_sobol_integer_design,
    integer_design_fingerprint,
    load_frozen_source_informed_design,
)
from core.terminal_verification import (  # noqa: E402
    build_verification_aware_shortlist,
    freeze_objective_incumbent_shortlist,
    parse_verification_candidate_budgets,
    select_initial_empirical_objective_incumbent,
    verify_frozen_shortlist,
)
from performance.benchmark_lodo_meta_prior import build_scalarized_problem  # noqa: E402
from performance.benchmark_quality import json_safe, parse_weights  # noqa: E402


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _true_metrics(problem, point):
    point = tuple(map(int, point))
    objective = float(problem.true_objective(point))
    constraint_mean = float(problem.true_constraint_mean(point))
    constraint_sigma = float(problem.true_sigma(point)[1])
    chance_margin = (
        constraint_mean
        + float(norm.ppf(1.0 - float(problem.alpha))) * constraint_sigma
        - float(problem.tau)
    )
    _, optimum = problem.true_best_feasible()
    feasible = bool(chance_margin <= 0.0)
    return {
        "x_recommended": list(point),
        "true_objective": objective,
        "true_constraint_mean": constraint_mean,
        "true_constraint_sigma": constraint_sigma,
        "true_chance_margin": float(chance_margin),
        "true_feasible": feasible,
        "feasible_regret": (
            max(0.0, objective - float(optimum)) if feasible else None
        ),
        "constraint_violation": max(0.0, float(chance_margin)),
    }


def _select_shortlist(
    problem,
    points,
    observations,
    *,
    shortlist_mode="proposal_only_empirical_safe_interior",
    shortlist_size=2,
    maximum_violation_probability=0.5,
    probability_slack=0.05,
    candidate_universe="frozen_source_informed_n0",
):
    """Use only one-shot target observations to rank the frozen proposal."""

    observations = np.asarray(observations, dtype=float)
    z_alpha = float(norm.ppf(1.0 - float(problem.alpha)))
    empirical_margin = (
        observations[:, 1]
        + z_alpha * float(problem.sigma_level)
        - float(problem.tau)
    )
    estimated_feasible = empirical_margin <= 0.0
    if np.any(estimated_feasible):
        eligible = np.flatnonzero(estimated_feasible)
        primary_index = int(eligible[np.argmin(
            observations[eligible, 0])])
        primary_reason = "minimum_observed_objective_among_nominal_safe"
    else:
        primary_index = int(np.lexsort((
            observations[:, 0],
            empirical_margin,
        ))[0])
        primary_reason = "minimum_nominal_one_shot_chance_margin"
    normalized_mode = str(
        shortlist_mode
    ).strip().lower().replace("-", "_")
    if normalized_mode == "posterior_objective_challenger_then_safe":
        nominal_scale = max(float(problem.sigma_level), 1e-12)
        probability_violation = norm.cdf(
            empirical_margin / nominal_scale)
        shortlist, audit = build_verification_aware_shortlist(
            problem,
            points[primary_index],
            points,
            observations[:, 0],
            probability_violation,
            shortlist_size=int(shortlist_size),
            maximum_violation_probability=float(
                maximum_violation_probability),
            probability_slack=float(probability_slack),
            support_selection_mode="diverse",
            require_provider=True,
            selector_posterior="one_shot_nominal_empirical_plugin",
            candidate_universe=str(candidate_universe),
        )
        return shortlist, {
            **audit,
            "primary_reason": primary_reason,
            "posterior_contract": (
                "one_shot_gaussian_plugin_not_fitted_surrogate"),
            "target_observations_used": True,
            "target_oracle_used": False,
        }
    if normalized_mode != "proposal_only_empirical_safe_interior":
        raise ValueError(
            "unknown proposal-only terminal shortlist mode")

    support_order = np.argsort(empirical_margin, kind="stable")
    support_index = next(
        (int(index) for index in support_order if int(index) != primary_index),
        primary_index,
    )
    rows = []
    for position, role, index, reason in (
        (
            1,
            "proposal_only_empirical_primary",
            primary_index,
            primary_reason,
        ),
        (
            2,
            "proposal_only_empirical_safe_interior",
            support_index,
            "minimum_nominal_one_shot_chance_margin",
        ),
    ):
        point = tuple(map(int, points[index]))
        rows.append({
            "shortlist_position": int(position),
            "shortlist_role": role,
            "posterior_rank": None,
            "point": list(point),
            "point_fingerprint": integer_design_fingerprint([point]),
            "selector_posterior": "none_one_shot_nominal_empirical",
            "selection_reason": reason,
            "observed_objective": float(observations[index, 0]),
            "observed_constraint": float(observations[index, 1]),
            "estimated_chance_margin": float(empirical_margin[index]),
            "target_labels_used": True,
            "target_oracle_used": False,
            "verification_samples_used": False,
        })
    return rows, {
        "shortlist_mode": normalized_mode,
        "shortlist_size": int(len(rows)),
        "primary_reason": primary_reason,
        "target_observations_used": True,
        "target_oracle_used": False,
    }


def _load_initial_design(args, problem):
    mode = str(getattr(
        args, "initial_design", "source_informed"
    )).strip().lower()
    if mode == "source_informed":
        if not str(getattr(args, "initial_design_file", "")).strip():
            raise ValueError(
                "source_informed proposal-only runs require "
                "--initial-design-file"
            )
        points, contract = load_frozen_source_informed_design(
            args.initial_design_file,
            heldout=args.heldout,
            seed=args.seed,
            n0=args.n0,
            dimension=args.d,
        )
        return points, contract, int(args.offline_source_calls)
    if mode == "common_sobol":
        points = tuple(common_sobol_integer_design(
            problem,
            int(args.n0),
            int(args.seed),
        ))
        fingerprint = integer_design_fingerprint(points)
        return points, {
            "design_kind": "common_sobol",
            "proposal_mode": "none",
            "structural_prior_profile": "none",
            "source_dimension": int(args.d),
            "target_dimension": int(args.d),
            "fingerprint": fingerprint,
            "source_archive_fingerprint": None,
            "source_archive_oracle_aided": False,
            "target_labels_used": False,
            "target_oracle_used": False,
        }, 0
    raise ValueError("initial_design must be source_informed or common_sobol")


def run_one(args):
    started = time.time()
    problem = build_scalarized_problem(
        args.heldout,
        args.d,
        args.L,
        args.sigma,
        args.alpha,
        parse_weights(args.weights),
    )
    points, design_contract, offline_source_calls = _load_initial_design(
        args,
        problem,
    )
    initial_design_mode = str(getattr(
        args, "initial_design", "source_informed"
    )).strip().lower()
    rng = np.random.default_rng(int(args.seed))
    observations = [
        np.asarray(problem.simulate(point, rng), dtype=float)
        for point in points
    ]
    shortlist_mode = str(getattr(
        args,
        "terminal_verification_shortlist_mode",
        "proposal_only_empirical_safe_interior",
    ))
    shortlist, shortlist_audit = _select_shortlist(
        problem,
        points,
        observations,
        shortlist_mode=shortlist_mode,
        shortlist_size=int(getattr(
            args, "terminal_verification_shortlist_size", 2)),
        maximum_violation_probability=float(getattr(
            args,
            "terminal_objective_challenger_max_violation_probability",
            0.5,
        )),
        probability_slack=float(getattr(
            args, "terminal_safe_interior_probability_slack", 0.05)),
        candidate_universe=(
            "frozen_source_informed_n0"
            if initial_design_mode == "source_informed"
            else "common_sobol_n0"
        ),
    )
    incumbent_audit = None
    if bool(getattr(args, "terminal_objective_incumbent_guard", False)):
        incumbent = select_initial_empirical_objective_incumbent(
            points,
            observations,
            n0=int(args.n0),
        )
        shortlist, incumbent_audit = freeze_objective_incumbent_shortlist(
            shortlist,
            incumbent,
            shortlist_size=len(shortlist),
        )
    candidate_budgets = parse_verification_candidate_budgets(
        getattr(args, "terminal_verification_candidate_budgets", ""),
        default=(
            int(args.terminal_verification_primary_budget),
            int(args.terminal_verification_support_budget),
        ),
    )
    if len(candidate_budgets) != len(shortlist):
        raise ValueError(
            "proposal-only terminal shortlist and candidate budgets differ")
    deployed, verification = verify_frozen_shortlist(
        problem,
        shortlist,
        seed=int(args.seed),
        search_evaluation_count=int(args.n0),
        candidate_budgets=candidate_budgets,
        familywise_delta=float(args.terminal_verification_delta),
        method=str(args.terminal_verification_method),
        mean_delta_fraction=float(
            args.terminal_verification_mean_delta_fraction),
        shortlist_mode=shortlist_mode,
        objective_incumbent_position=(
            None
            if incumbent_audit is None
            else incumbent_audit["objective_incumbent_position"]
        ),
        objective_comparison_budget=int(getattr(
            args, "terminal_objective_comparison_budget", 0)),
        objective_comparison_delta=float(getattr(
            args, "terminal_objective_comparison_delta", 0.05)),
    )
    initial_truth = [_true_metrics(problem, point) for point in points]
    deployed_metrics = _true_metrics(problem, deployed)
    feasible_initial = [
        row["feasible_regret"]
        for row in initial_truth
        if row["feasible_regret"] is not None
    ]
    verification_calls = int(verification["verification_budget"])
    return {
        "schema_version": 1,
        "status": "ok",
        "method": (
            "frozen_crossdim_proposal_only"
            if initial_design_mode == "source_informed"
            else "common_sobol_proposal_only"
        ),
        "heldout_target_domain": str(args.heldout),
        "seed": int(args.seed),
        "source_archive_fingerprint": (
            None
            if design_contract["source_archive_fingerprint"] is None
            else str(design_contract["source_archive_fingerprint"])
        ),
        "initial_design_fingerprint": str(design_contract["fingerprint"]),
        "information_contract": {
            "initial_design": initial_design_mode,
            "uses_source_archive": bool(
                initial_design_mode == "source_informed"),
            "source_dimension": int(
                design_contract.get("source_dimension", args.source_d)),
            "target_dimension": int(args.d),
            "dimension_holdout": bool(
                int(design_contract.get("source_dimension", args.source_d))
                != int(args.d)
            ),
            "frozen_initial_points": [list(map(int, point)) for point in points],
            "frozen_initial_points_fingerprint": integer_design_fingerprint(
                points),
            "offline_source_calls": int(offline_source_calls),
            "target_initial_calls_n0": int(args.n0),
            "target_search_calls": int(args.n0),
            "target_verification_calls": verification_calls,
            "target_safety_verification_calls": int(
                verification.get(
                    "safety_verification_budget", verification_calls)
            ),
            "target_objective_comparison_calls": int(
                verification.get("objective_comparison_budget", 0)),
            "target_total_calls": int(args.n0) + verification_calls,
            "online_optimization_calls_after_n0": 0,
            "online_backend": "none",
            "selection_rule": "nominal_one_shot_empirical",
            "source_oracle_aided": False,
            "target_oracle_used_for_selection": False,
            "target_true_sigma_used_for_selection": False,
            "terminal_verification_identical_across_methods": True,
            "terminal_verification_updates_optimizer": False,
            "terminal_verification_shortlist_mode": shortlist_mode,
            "terminal_verification_candidate_budgets": list(
                candidate_budgets),
            "terminal_verification_familywise_delta": float(
                args.terminal_verification_delta),
        },
        "initial_observations": [
            {
                "point": list(map(int, point)),
                "observation": observation.tolist(),
            }
            for point, observation in zip(points, observations)
        ],
        "frozen_terminal_shortlist": shortlist,
        "terminal_shortlist_selection_audit": shortlist_audit,
        "terminal_verification": verification,
        "terminal_objective_incumbent": incumbent_audit,
        "n_search_simulations": int(args.n0),
        "n_safety_verification_simulations": int(
            verification.get(
                "safety_verification_budget", verification_calls)
        ),
        "n_objective_comparison_simulations": int(
            verification.get("objective_comparison_budget", 0)),
        "n_verification_simulations": int(verification_calls),
        "n_target_simulations_total": int(args.n0) + verification_calls,
        "n_simulations": int(args.n0) + verification_calls,
        "initial_truth_audit": {
            "computed_after_shortlist_freeze_and_verification": True,
            "used_for_selection_or_certification": False,
            "true_feasible_count": int(sum(
                row["true_feasible"] for row in initial_truth)),
            "true_feasible_rate": float(np.mean([
                row["true_feasible"] for row in initial_truth
            ])),
            "best_true_feasible_regret": (
                float(min(feasible_initial)) if feasible_initial else None
            ),
            "rows": initial_truth,
        },
        **deployed_metrics,
        "wall_time_sec": float(time.time() - started),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--initial-design",
        choices=("source_informed", "common_sobol"),
        default="source_informed",
    )
    parser.add_argument("--initial-design-file", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-d", type=int, default=50)
    parser.add_argument("--d", type=int, default=1000)
    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=0.04)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--weights", default="0.5,0.5")
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--offline-source-calls", type=int, default=384)
    parser.add_argument(
        "--terminal-verification-primary-budget", type=int, default=80)
    parser.add_argument(
        "--terminal-verification-support-budget", type=int, default=96)
    parser.add_argument(
        "--terminal-verification-candidate-budgets",
        default="",
        help=(
            "Optional comma-separated ordered budgets. When set, this "
            "overrides the legacy primary/support pair."
        ),
    )
    parser.add_argument(
        "--terminal-verification-shortlist-mode",
        choices=(
            "proposal_only_empirical_safe_interior",
            "posterior_objective_challenger_then_safe",
        ),
        default="proposal_only_empirical_safe_interior",
    )
    parser.add_argument(
        "--terminal-verification-shortlist-size", type=int, default=2)
    parser.add_argument(
        "--terminal-objective-incumbent-guard",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--terminal-objective-comparison-budget", type=int, default=0)
    parser.add_argument(
        "--terminal-objective-comparison-delta", type=float, default=0.05)
    parser.add_argument(
        "--terminal-objective-challenger-max-violation-probability",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--terminal-safe-interior-probability-slack",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--terminal-verification-delta", type=float, default=0.05)
    parser.add_argument(
        "--terminal-verification-mean-delta-fraction",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--terminal-verification-method",
        choices=("component_bonferroni", "normal_quantile_tolerance"),
        default="normal_quantile_tolerance",
    )
    args = parser.parse_args()
    payload = run_one(args)
    _atomic_json(args.out, payload)
    print(json.dumps({
        "status": payload["status"],
        "method": payload["method"],
        "initial_design": args.initial_design,
        "heldout": args.heldout,
        "seed": args.seed,
        "out": args.out,
    }, indent=2))


if __name__ == "__main__":
    main()
