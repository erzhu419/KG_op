#!/usr/bin/env python3
"""Optimize one controlled heteroscedastic scenario and audit certification.

One run freezes the search-primary recommendation before independent terminal
verification.  The compact result therefore contains a paired, within-run
estimate of whether certification rescued, preserved, or degraded deployment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import norm, spearmanr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algorithms.single_olhkg import (  # noqa: E402
    SingleOLHKGAlgorithm,
    SingleOLHKGConfig,
)
from performance.benchmark_quality import json_safe  # noqa: E402
from problems.controlled_heteroscedastic import (  # noqa: E402
    CONTROLLED_HETERO_SCENARIOS,
    ControlledHeteroscedasticProblem,
)
from problems.single_objective import ScalarizedProblem  # noqa: E402


BACKENDS = {
    "sobol": "sobol_new",
    "risk_ts": "risk_ts",
    "joint_voi": "sobol_exact_joint_voi",
}


def _variance_audit(algorithm, problem):
    base = problem.base
    rows = [
        base._constant_policy(z0, z1, z2)
        for z0 in np.linspace(0.05, 0.95, 7)
        for z1 in np.linspace(0.05, 0.95, 7)
        for z2 in np.linspace(0.05, 0.95, 7)
    ]
    true_variance = np.asarray([
        float(problem.true_sigma(x)[1] ** 2) for x in rows
    ])
    predicted = np.asarray(
        algorithm.variance_model.predict_variance_many(
            1, rows, problem),
        dtype=float,
    )
    upper = np.asarray(
        algorithm.variance_model.predict_certification_variance_many(
            1, rows, problem),
        dtype=float,
    )
    log_error = (
        np.log(np.maximum(predicted, 1e-12))
        - np.log(np.maximum(true_variance, 1e-12))
    )
    correlation = (
        np.nan
        if (
            np.std(true_variance) <= 1e-15
            or np.std(predicted) <= 1e-15
        )
        else spearmanr(true_variance, predicted).statistic
    )
    return {
        "evaluation_count": int(len(rows)),
        "log_variance_rmse": float(np.sqrt(np.mean(log_error ** 2))),
        "variance_spearman": (
            None if not np.isfinite(correlation) else float(correlation)
        ),
        "upper_variance_coverage": float(np.mean(upper >= true_variance)),
        "median_predicted_true_ratio": float(np.median(
            predicted / np.maximum(true_variance, 1e-12))),
        "median_upper_true_ratio": float(np.median(
            upper / np.maximum(true_variance, 1e-12))),
        "truth_join_timing": "post_run_only",
        "target_oracle_used_for_decision": False,
    }


def _best_evaluated_truth(algorithm, problem):
    _, true_best = problem.true_best_feasible()
    rows = []
    for x in dict.fromkeys(
        tuple(int(value) for value in point)
        for point, _ in algorithm.history
    ):
        margin = (
            problem.true_constraint_mean(x)
            + norm.ppf(1.0 - problem.alpha) * problem.true_sigma(x)[1]
            - problem.tau
        )
        objective = problem.true_objective(x)
        rows.append({
            "x": list(x),
            "true_feasible": bool(margin <= 0.0),
            "true_margin": float(margin),
            "true_objective": float(objective),
            "regret": (
                float(objective - true_best)
                if margin <= 0.0 and np.isfinite(true_best)
                else None
            ),
        })
    feasible = [row for row in rows if row["true_feasible"]]
    best = min(
        feasible,
        key=lambda row: row["regret"],
        default=None,
    )
    return {
        "evaluated_distinct_count": int(len(rows)),
        "evaluated_true_feasible_count": int(len(feasible)),
        "found_true_feasible": bool(feasible),
        "best_evaluated_feasible_regret": (
            None if best is None else float(best["regret"])
        ),
        "oracle_hit_at_0_01": bool(
            best is not None and float(best["regret"]) <= 0.01
        ),
        "truth_join_timing": "post_run_only",
        "target_oracle_used_for_decision": False,
    }


def _paired_deployment(primary, final, terminal):
    primary_feasible = bool(primary.get("true_feasible", False))
    final_feasible = bool(final.get("true_feasible", False))
    primary_regret = (
        float(primary["simple_regret"]) if primary_feasible else None)
    final_regret = float(final["simple_regret"]) if final_feasible else None
    if primary_feasible and final_feasible:
        regret_change = float(final_regret - primary_regret)
    else:
        regret_change = None
    return {
        "recommendation_changed": bool(
            terminal.get("recommendation_changed", False)),
        "primary_true_feasible": primary_feasible,
        "deployment_true_feasible": final_feasible,
        "primary_feasible_regret": primary_regret,
        "deployment_feasible_regret": final_regret,
        "deployment_minus_primary_regret": regret_change,
        "feasibility_rescue": bool(not primary_feasible and final_feasible),
        "feasibility_loss": bool(primary_feasible and not final_feasible),
        "strict_objective_win": bool(
            regret_change is not None and regret_change < -1e-12),
        "strict_objective_loss": bool(
            regret_change is not None and regret_change > 1e-12),
        "preserved": bool(
            primary_feasible == final_feasible
            and (
                regret_change is None
                or abs(regret_change) <= 1e-12
            )
        ),
    }


def run_cell(args):
    base = ControlledHeteroscedasticProblem(
        d=args.d,
        L=args.L,
        sigma=args.sigma,
        alpha=args.alpha,
        scenario=args.scenario,
    )
    problem = ScalarizedProblem(base, weights=(0.5, 0.5))
    backend = BACKENDS[args.backend]
    exact = args.backend == "joint_voi"
    config = SingleOLHKGConfig(
        implementation_contract_id=(
            "controlled_hetero_v51_search_v64_terminal_v1"),
        theory_contract_id=(
            "controlled_hetero_paired_search_verification_v1"),
        N=args.N,
        n0=args.n0,
        initial_design="common_sobol",
        K1=args.K1,
        K2=0,
        posterior_pool_size=args.posterior_pool_size,
        posterior_keep=min(16, args.posterior_pool_size),
        axis_candidate_count=0,
        structured_candidate_count=0,
        state_candidate_count=args.state_candidate_count,
        state_inverse_pool_size=args.state_inverse_pool_size,
        state_inverse_neighbors=1,
        variance_mode=args.variance_mode,
        hvd_use_cumulative_provider=True,
        hvd_cumulative_target_evidence_mode="replication_only",
        hvd_singleton_evidence_mode="in_sample_residual",
        certification_mode="theory",
        beta_g=args.beta_g,
        use_state_coupling=True,
        use_state_basis=True,
        state_basis_mode="state",
        constraint_state_basis_mode="state",
        encoder_kind="synthetic",
        recommendation_axis_oracle=False,
        use_problem_initial_samples=False,
        use_boundary_initial_samples=False,
        use_recommendation_refinement=False,
        recommendation_observed_fallback=True,
        recommend_observed_only=True,
        decision_backend=backend,
        decision_risk_penalty=args.risk_penalty,
        decision_aleatoric_mode="posterior_central",
        decision_violation_loss_mode="positive_part",
        decision_ambiguity_mode="posterior_nominal",
        decision_recommend_observed_only=True,
        adaptive_replication_voi=bool(exact),
        evaluate_or_replicate_new_action_count=4,
        evaluate_or_replicate_new_action_policy=(
            "canonical_plus_posterior_risk"),
        replication_candidate_count=4 if exact else 0,
        replication_max_per_solution=5,
        acquisition_mode="exact_mc" if exact else "additive",
        exact_kg_mc_samples=args.exact_mc_samples if exact else 0,
        exact_kg_jobs=args.exact_jobs if exact else 1,
        exact_kg_parallel_backend="process_fork",
        exact_kg_sampling_mode="antithetic_nested",
        exact_kg_clip_negative=False,
        exact_kg_terminal_mode="bayes_risk",
        terminal_bayes_violation_penalty=args.risk_penalty,
        finalist_replication_budget=0,
        finalist_empirical_override="disabled",
        finalist_frontier_policy="legacy",
        posterior_dominance_enabled=False,
        terminal_verification_budget=(
            args.verification_primary_budget if args.verify else 0),
        terminal_verification_delta=args.verification_delta,
        terminal_verification_method="normal_quantile_tolerance",
        terminal_verification_policy=(
            "ordered_frozen_shortlist" if args.verify else "fixed_policy"),
        terminal_verification_shortlist_size=2 if args.verify else 1,
        terminal_verification_fallback_budget=(
            args.verification_support_budget if args.verify else 0),
        terminal_verification_shortlist_mode=(
            "posterior_primary_safe_interior"
            if args.verify else "posterior_ranked"
        ),
        terminal_safe_interior_probability_slack=0.05,
        terminal_safe_interior_require_provider=bool(args.verify),
        checkpoint_interval=0,
        checkpoint_dir="",
        truth_pool_diagnostics=True,
        truth_pool_max_candidates=0,
        evaluate_interval=0,
        progress_logging=True,
        progress_label=(
            f"controlled/{args.scenario}/{args.variance_mode}/"
            f"{args.backend}/seed{args.seed}"
        ),
        seed=args.seed,
    )
    algorithm = SingleOLHKGAlgorithm(problem, config)
    raw = algorithm.run(verbose=False)
    primary = dict(raw.get("optimization_recommendation_truth") or {})
    final = {
        key: raw.get(key)
        for key in (
            "x_recommended",
            "true_objective",
            "true_constraint_mean",
            "true_constraint_sigma",
            "true_chance_margin",
            "true_feasible",
            "true_best_x",
            "true_best_objective",
            "simple_regret",
        )
    }
    terminal = dict(raw.get("terminal_verification") or {})
    scenario = CONTROLLED_HETERO_SCENARIOS[args.scenario]
    return {
        "schema_version": 1,
        "status": "ok",
        "experiment": "controlled_heteroscedastic_optimum",
        "scenario": args.scenario,
        "variance_location": scenario["location"],
        "variance_geometry": scenario["geometry"],
        "provider_exact": bool(scenario["provider_exact"]),
        "variance_mode": args.variance_mode,
        "backend": args.backend,
        "seed": int(args.seed),
        "d": int(args.d),
        "n0": int(args.n0),
        "search_budget": int(args.N),
        "verification_primary_budget": int(
            args.verification_primary_budget if args.verify else 0),
        "verification_support_budget": int(
            args.verification_support_budget if args.verify else 0),
        "search_primary": primary,
        "verified_deployment": final,
        "paired_deployment_effect": _paired_deployment(
            primary, final, terminal),
        "best_evaluated_truth": _best_evaluated_truth(
            algorithm, problem),
        "posterior_certificate": dict(
            raw.get("certificate_outcome_audit") or {}),
        "independent_terminal_certificate": {
            "certified": bool(
                raw.get("terminal_verification_certified", False)),
            "method": terminal.get("method"),
            "verification_budget": int(
                terminal.get("verification_budget", 0)),
            "selected_rank": terminal.get("selected_rank"),
            "false_certificate": bool(
                raw.get("terminal_verification_certified", False)
                and not bool(final.get("true_feasible", False))
            ),
            "recommendation_changed": bool(
                terminal.get("recommendation_changed", False)),
        },
        "variance_audit": _variance_audit(
            algorithm, problem),
        "adaptive_outcome": dict(
            raw.get("adaptive_outcome_audit") or {}),
        "n_search_simulations": int(raw["n_search_simulations"]),
        "n_verification_simulations": int(
            raw["n_verification_simulations"]),
        "n_total_simulations": int(raw["n_simulations"]),
        "wall_time_sec": float(raw["total_time_sec"]),
        "candidate_source_counts": dict(
            raw.get("candidate_source_counts") or {}),
        "oracle_contract": base.oracle_contract(),
        "information_contract": {
            "source_archive_used": False,
            "source_proposal_used": False,
            "initial_design": "common_sobol",
            "problem_specific_initial_hook_used": False,
            "problem_specific_refinement_used": False,
            "target_oracle_used_for_search": bool(
                args.variance_mode == "oracle"),
            "oracle_variance_row_is_diagnostic_upper_bound": bool(
                args.variance_mode == "oracle"),
            "post_run_truth_used_for_metrics_only": True,
            "verification_updates_optimizer": False,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=tuple(CONTROLLED_HETERO_SCENARIOS),
        required=True,
    )
    parser.add_argument(
        "--variance-mode",
        choices=("pooled", "orthogonal", "factor", "oracle"),
        required=True,
    )
    parser.add_argument(
        "--backend", choices=tuple(BACKENDS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--d", type=int, default=1000)
    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=0.04)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--K1", type=int, default=24)
    parser.add_argument("--posterior-pool-size", type=int, default=128)
    parser.add_argument("--state-candidate-count", type=int, default=8)
    parser.add_argument("--state-inverse-pool-size", type=int, default=256)
    parser.add_argument("--beta-g", type=float, default=2.0)
    parser.add_argument("--risk-penalty", type=float, default=5.0)
    parser.add_argument("--exact-mc-samples", type=int, default=8)
    parser.add_argument("--exact-jobs", type=int, default=12)
    parser.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--verification-primary-budget", type=int, default=80)
    parser.add_argument(
        "--verification-support-budget", type=int, default=96)
    parser.add_argument("--verification-delta", type=float, default=0.05)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.n0 > args.N:
        raise ValueError("n0 cannot exceed N")
    if args.d < 3:
        raise ValueError("controlled suite requires d >= 3")
    started = time.perf_counter()
    result = run_cell(args)
    result["runner_wall_time_sec"] = float(time.perf_counter() - started)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(json_safe(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": result["status"],
        "scenario": result["scenario"],
        "variance_mode": result["variance_mode"],
        "backend": result["backend"],
        "seed": result["seed"],
        "out": str(args.out),
    }))


if __name__ == "__main__":
    main()
