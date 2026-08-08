#!/usr/bin/env python3
"""Target-only functional-space SCBO for ordered-profile stress tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.botorch_adapters import (  # noqa: E402
    BoTorchBaseline,
    BoTorchBaselineConfig,
)
from core.designs import integer_design_fingerprint  # noqa: E402
from core.terminal_verification import verify_frozen_shortlist_binomial  # noqa: E402
from performance.benchmark_profile_stress_suite import (  # noqa: E402
    _atomic_json,
    _oracle_library,
    _registered_audit_library,
    _stable_seed,
)
from problems.profile_coefficient_space import (  # noqa: E402
    CosineCoefficientProfileProblem,
)
from problems.randomized_profiles import (  # noqa: E402
    PROFILE_STRESS_REGIMES,
    RandomizedOrderedProfileProblem,
)


CONTRACT_ID = "target_only_functional_profile_scbo_v1"


def run_task(
    *,
    regime,
    target_seed,
    design_seed,
    dimension=1000,
    active_rank=None,
    alpha=0.05,
    safe_mass=0.08,
    n0=10,
    N=13,
    coefficient_count=8,
    coefficient_scale=0.25,
    level_bounds=(0.05, 0.95),
    schema_mode="declared",
    family_seed=20260808,
    verification_budgets=(80, 80, 80),
    familywise_delta=0.05,
    raw_samples=1024,
    num_restarts=10,
    maxiter=100,
    batch_candidates=128,
    ts_candidates=0,
    checkpoint_path="",
    checkpoint_resume=False,
    progress_logging=False,
):
    """Run one frozen target-only DCT-space constrained BO task."""

    regime = str(regime)
    if regime not in PROFILE_STRESS_REGIMES:
        raise ValueError(f"unknown stress regime: {regime}")
    n0 = int(n0)
    N = int(N)
    if N < n0:
        raise ValueError("functional SCBO requires N >= n0")
    regime_seed = int(family_seed) + _stable_seed(regime)
    target = RandomizedOrderedProfileProblem(
        regime=regime,
        role="target",
        task_seed=int(target_seed),
        family_seed=regime_seed,
        d=int(dimension),
        active_rank=active_rank,
        alpha=float(alpha),
        safe_mass=float(safe_mass),
    )
    functional = CosineCoefficientProfileProblem(
        target,
        coefficient_count=int(coefficient_count),
        coefficient_scale=float(coefficient_scale),
        level_bounds=tuple(float(value) for value in level_bounds),
        schema_mode=schema_mode,
    )
    config = BoTorchBaselineConfig(
        N=N,
        n0=n0,
        seed=int(design_seed),
        method="botorch_scbo",
        raw_samples=int(raw_samples),
        num_restarts=int(num_restarts),
        maxiter=int(maxiter),
        batch_candidates=int(batch_candidates),
        ts_candidates=int(ts_candidates),
        strict_failures=True,
        use_problem_initial_samples=False,
        use_boundary_initial_samples=False,
        initial_design="sobol",
        checkpoint_path=str(checkpoint_path or ""),
        checkpoint_resume=bool(checkpoint_resume),
        checkpoint_interval=1,
        progress_logging=bool(progress_logging),
        progress_label=(
            f"functional-scbo:{regime}:d{int(dimension)}:"
            f"seed{int(target_seed)}:K{int(coefficient_count)}:N{N}"
        ),
        torch_device="cpu",
    )
    started = time.perf_counter()
    optimizer = BoTorchBaseline(functional, config)
    backend = optimizer.run(
        freeze_terminal_shortlist=True,
        evaluate_truth=False,
        terminal_probability_slack=0.05,
        terminal_require_provider=False,
        terminal_shortlist_mode="posterior_objective_challenger_then_safe",
        terminal_shortlist_size=3,
        terminal_maximum_violation_probability=0.5,
    )

    raw_shortlist = []
    for row in backend["frozen_terminal_shortlist"]:
        coefficient_point = tuple(int(value) for value in row["point"])
        raw_point = functional.raw_point(coefficient_point)
        raw_shortlist.append({
            **{
                key: value for key, value in row.items()
                if key not in {"point", "point_fingerprint"}
            },
            "coefficient_point": list(coefficient_point),
            "coefficient_point_fingerprint": integer_design_fingerprint(
                [coefficient_point]),
            "point": list(raw_point),
            "point_fingerprint": integer_design_fingerprint([raw_point]),
            "coordinate_contract": functional.contract_id,
        })
    deployed, verification = verify_frozen_shortlist_binomial(
        target,
        raw_shortlist,
        seed=int(design_seed) + 71_003,
        search_evaluation_count=N,
        candidate_budgets=tuple(int(value) for value in verification_budgets),
        familywise_delta=float(familywise_delta),
        all_success_only=True,
    )

    # Truth is intentionally unavailable until the optimizer, shortlist, and
    # independent verification decision have all been frozen.
    audit_library = _registered_audit_library(target)
    _, oracle_best = _oracle_library(target, audit_library)
    history = []
    for index, row in enumerate(backend["history"]):
        coefficient_point = tuple(int(value) for value in row["x"])
        raw_point = functional.raw_point(coefficient_point)
        history.append({
            "evaluation_index": int(index),
            "source": (
                "target_only_coefficient_sobol_initial"
                if index < n0 else "target_only_functional_scbo"
            ),
            "coefficient_point": list(coefficient_point),
            "coefficient_point_fingerprint": integer_design_fingerprint(
                [coefficient_point]),
            "raw_point_fingerprint": integer_design_fingerprint([raw_point]),
            "observation": [float(value) for value in row["y"]],
            "true_feasible": bool(target.is_truly_feasible(raw_point)),
            "true_objective": float(target.true_objective(raw_point)),
            "true_chance_margin": float(target.true_chance_margin(raw_point)),
        })
    initial_feasible = [row for row in history[:n0] if row["true_feasible"]]
    search_feasible = [row for row in history if row["true_feasible"]]
    best_initial = (
        None if not initial_feasible else min(
            initial_feasible, key=lambda row: row["true_objective"])
    )
    best_search = (
        None if not search_feasible else min(
            search_feasible, key=lambda row: row["true_objective"])
    )
    deployed_truth = None
    if deployed is not None:
        deployed_truth = {
            "feasible": bool(target.is_truly_feasible(deployed)),
            "objective": float(target.true_objective(deployed)),
            "chance_margin": float(target.true_chance_margin(deployed)),
            "feasibility_probability": float(
                target.true_feasibility_probability(deployed)),
        }
    initial_regret = (
        None if best_initial is None else max(
            0.0,
            float(best_initial["true_objective"] - oracle_best["objective"]),
        )
    )
    search_regret = (
        None if best_search is None else max(
            0.0,
            float(best_search["true_objective"] - oracle_best["objective"]),
        )
    )
    deployed_regret = (
        None
        if deployed_truth is None or not deployed_truth["feasible"]
        else max(
            0.0,
            float(deployed_truth["objective"] - oracle_best["objective"]),
        )
    )
    initial_penalized_loss = (
        float(initial_regret)
        if initial_regret is not None else float(
            1.0 + min(
                max(row["true_chance_margin"], 0.0)
                for row in history[:n0]
            ) / max(float(target.safe_radius), 1e-12)
        )
    )
    search_penalized_loss = (
        float(search_regret)
        if search_regret is not None else float(
            1.0 + min(
                max(row["true_chance_margin"], 0.0) for row in history
            ) / max(float(target.safe_radius), 1e-12)
        )
    )
    verification_calls = int(verification["verification_budget"])
    maximum_verification_calls = int(sum(
        int(value) for value in verification_budgets))
    return {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "status": "ok",
        "regime": regime,
        "target_seed": int(target_seed),
        "design_seed": int(design_seed),
        "arm": "target_only_dct_space_scbo",
        "nominal_dimension": int(target.d),
        "effective_rank": int(target.effective_rank),
        "active_rank_override": (
            None if active_rank is None else int(active_rank)),
        "alpha": float(target.alpha),
        "safe_mass": float(target.safe_mass),
        "n0": n0,
        "N": N,
        "source_calls": 0,
        "target_search_calls": N,
        "verification_calls": verification_calls,
        "maximum_verification_calls": maximum_verification_calls,
        "all_in_calls_unamortized": int(N + verification_calls),
        "all_in_budget_cap_unamortized": int(
            N + maximum_verification_calls),
        "all_in_calls_amortized": int(N + verification_calls),
        "all_in_budget_cap_amortized": int(
            N + maximum_verification_calls),
        "target_information_contract": target.information_contract(),
        "functional_coordinate_contract": functional.information_contract(),
        "backend_contract": {
            "method": backend["method"],
            "algorithm_fidelity": backend["algorithm_fidelity"],
            "initial_design": backend["initial_design"],
            "truth_metrics_evaluated": backend["truth_metrics_evaluated"],
            "target_oracle_used": backend["target_oracle_used"],
            "posterior_certificate_kind": backend[
                "posterior_certificate_kind"],
            "botorch_fit_failures": backend["botorch_fit_failures"],
            "botorch_candidate_failures": backend[
                "botorch_candidate_failures"],
            "total_time_sec": float(backend["total_time_sec"]),
            "runtime_fingerprint": backend["runtime_fingerprint"],
        },
        "search_records": history,
        "shortlist": raw_shortlist,
        "verification": verification,
        "initial_design_contains_true_feasible": bool(best_initial is not None),
        "initial_design_true_feasible_count": int(len(initial_feasible)),
        "initial_design_finite_audit_library_regret": initial_regret,
        "initial_design_penalized_loss": initial_penalized_loss,
        "contains_true_feasible": bool(best_search is not None),
        "search_contains_true_feasible": bool(best_search is not None),
        "true_feasible_count_in_search": int(len(search_feasible)),
        "finite_library_regret": search_regret,
        "search_finite_audit_library_regret": search_regret,
        "feasible_and_epsilon_optimal_005": bool(
            search_regret is not None and search_regret <= 0.05),
        "penalized_loss": search_penalized_loss,
        "search_penalized_loss": search_penalized_loss,
        "finite_audit_library_oracle_objective": float(
            oracle_best["objective"]),
        "finite_audit_library_size": int(len(audit_library)),
        "independently_certified": bool(verification["certified"]),
        "deployed_truth": deployed_truth,
        "certified_deployed_true_feasible": bool(
            verification["certified"]
            and deployed_truth is not None
            and deployed_truth["feasible"]
        ),
        "certified_deployed_finite_audit_library_regret": deployed_regret,
        "false_certificate": bool(
            verification["certified"]
            and deployed_truth is not None
            and not deployed_truth["feasible"]
        ),
        "oracle_role": "post_run_audit_only",
        "wall_time_sec": float(time.perf_counter() - started),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--regime", choices=tuple(PROFILE_STRESS_REGIMES), required=True)
    parser.add_argument("--target-seed", type=int, required=True)
    parser.add_argument("--design-seed", type=int, required=True)
    parser.add_argument("--d", type=int, default=1000)
    parser.add_argument("--active-rank", type=int)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--safe-mass", type=float, default=0.08)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--N", type=int, default=13)
    parser.add_argument("--coefficient-count", type=int, default=8)
    parser.add_argument("--coefficient-scale", type=float, default=0.25)
    parser.add_argument("--level-bounds", default="0.05,0.95")
    parser.add_argument(
        "--schema-mode", choices=("declared", "schema_blind"),
        default="declared")
    parser.add_argument("--family-seed", type=int, default=20260808)
    parser.add_argument("--verification-budgets", default="80,80,80")
    parser.add_argument("--familywise-delta", type=float, default=0.05)
    parser.add_argument("--raw-samples", type=int, default=1024)
    parser.add_argument("--num-restarts", type=int, default=10)
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--batch-candidates", type=int, default=128)
    parser.add_argument("--ts-candidates", type=int, default=0)
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--checkpoint-resume", action="store_true")
    parser.add_argument("--progress-logging", action="store_true")
    args = parser.parse_args()
    payload = run_task(
        regime=args.regime,
        target_seed=args.target_seed,
        design_seed=args.design_seed,
        dimension=args.d,
        active_rank=args.active_rank,
        alpha=args.alpha,
        safe_mass=args.safe_mass,
        n0=args.n0,
        N=args.N,
        coefficient_count=args.coefficient_count,
        coefficient_scale=args.coefficient_scale,
        level_bounds=tuple(
            float(value) for value in args.level_bounds.split(",")),
        schema_mode=args.schema_mode,
        family_seed=args.family_seed,
        verification_budgets=tuple(
            int(value) for value in args.verification_budgets.split(",")),
        familywise_delta=args.familywise_delta,
        raw_samples=args.raw_samples,
        num_restarts=args.num_restarts,
        maxiter=args.maxiter,
        batch_candidates=args.batch_candidates,
        ts_candidates=args.ts_candidates,
        checkpoint_path=args.checkpoint_path,
        checkpoint_resume=args.checkpoint_resume,
        progress_logging=args.progress_logging,
    )
    _atomic_json(args.out, payload)
    print(json.dumps({
        "status": payload["status"],
        "out": str(args.out),
        "regime": payload["regime"],
        "target_seed": payload["target_seed"],
        "N": payload["N"],
        "coefficient_count": payload[
            "functional_coordinate_contract"]["coefficient_count"],
        "certified": payload["independently_certified"],
    }, indent=2, sort_keys=True))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
