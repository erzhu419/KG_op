#!/usr/bin/env python3
"""Source-free functional SCBO control for the OPSD region holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.botorch_adapters import (  # noqa: E402
    BoTorchBaseline,
    BoTorchBaselineConfig,
)
from core.designs import integer_design_fingerprint  # noqa: E402
from core.terminal_verification import verify_frozen_shortlist_binomial  # noqa: E402
from performance.benchmark_external_energy_v2 import (  # noqa: E402
    TARGET_MARKETS,
    _atomic_json,
    _sha256_file,
    market_region,
)
from problems.energy_reliability import OPSDStorageReliabilityProblem  # noqa: E402
from problems.profile_coefficient_space import (  # noqa: E402
    CosineCoefficientProfileProblem,
)


CONTRACT_ID = "opsd_region_heldout_functional_scbo_v1"


def run_task(
    *,
    data_path,
    target_market,
    target_seed,
    year=2018,
    dimension=1000,
    alpha=0.05,
    n0=10,
    N=13,
    coefficient_count=8,
    coefficient_scale=0.25,
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
    target_market = str(target_market)
    if target_market not in TARGET_MARKETS:
        raise ValueError(f"unknown registered target market: {target_market}")
    n0 = int(n0)
    N = int(N)
    if N < n0:
        raise ValueError("energy functional SCBO requires N >= n0")
    target_seed = int(target_seed)
    design_seed = 4_100_000 + target_seed
    target = OPSDStorageReliabilityProblem(
        data_path,
        market=target_market,
        year=int(year),
        d=int(dimension),
        alpha=float(alpha),
    )
    functional = CosineCoefficientProfileProblem(
        target,
        coefficient_count=int(coefficient_count),
        coefficient_scale=float(coefficient_scale),
        level_bounds=(0.05, 0.95),
        schema_mode="declared",
        nominal_sigma=1e-7,
    )
    config = BoTorchBaselineConfig(
        N=N,
        n0=n0,
        seed=design_seed,
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
            f"energy-functional-scbo:{target_market}:"
            f"seed{target_seed}:K{int(coefficient_count)}:N{N}"
        ),
        torch_device="cpu",
    )
    started = time.perf_counter()
    backend = BoTorchBaseline(functional, config).run(
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
        seed=design_seed + 71_003,
        search_evaluation_count=N,
        candidate_budgets=tuple(int(value) for value in verification_budgets),
        familywise_delta=float(familywise_delta),
        split="verification",
        all_success_only=True,
    )
    deployment_truth = None
    if deployed is not None:
        population = target.split_population(deployed, "verification")
        probability = float((population[:, 1] <= target.tau).mean())
        deployment_truth = {
            "finite_empirical_window_count": int(len(population)),
            "finite_empirical_feasibility_probability": probability,
            "truly_feasible_on_empirical_window_distribution": bool(
                probability >= 1.0 - target.alpha),
            "finite_empirical_objective_mean": float(
                population[:, 0].mean()),
            "used_for_search_or_selection": False,
        }
    search_records = []
    for index, row in enumerate(backend["history"]):
        coefficient_point = tuple(int(value) for value in row["x"])
        raw_point = functional.raw_point(coefficient_point)
        search_records.append({
            "evaluation_index": int(index),
            "source": (
                "target_only_coefficient_sobol_initial"
                if index < n0 else "target_only_functional_scbo"
            ),
            "coefficient_point": list(coefficient_point),
            "coefficient_point_fingerprint": integer_design_fingerprint(
                [coefficient_point]),
            "point_fingerprint": integer_design_fingerprint([raw_point]),
            "observation": [float(value) for value in row["y"]],
        })
    verification_calls = int(verification["verification_budget"])
    maximum_verification_calls = int(sum(
        int(value) for value in verification_budgets))
    return {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "status": "ok",
        "target_market": target_market,
        "target_region": market_region(target_market),
        "target_seed": target_seed,
        "design_seed": design_seed,
        "arm": "target_only_dct_space_scbo",
        "year": int(year),
        "nominal_dimension": int(dimension),
        "alpha": float(alpha),
        "n0": n0,
        "N": N,
        "source_calls": 0,
        "target_search_calls": N,
        "verification_calls": verification_calls,
        "maximum_verification_calls": maximum_verification_calls,
        "all_in_calls_unamortized": int(N + verification_calls),
        "all_in_budget_cap_unamortized": int(
            N + maximum_verification_calls),
        "amortization_targets": 1,
        "all_in_calls_amortized": int(N + verification_calls),
        "all_in_budget_cap_amortized": int(
            N + maximum_verification_calls),
        "data_sha256": _sha256_file(data_path),
        "information_contract": target.information_contract(),
        "functional_coordinate_contract": functional.information_contract(),
        "backend_contract": {
            "method": backend["method"],
            "algorithm_fidelity": backend["algorithm_fidelity"],
            "initial_design": backend["initial_design"],
            "truth_metrics_evaluated": backend["truth_metrics_evaluated"],
            "target_oracle_used": backend["target_oracle_used"],
            "botorch_fit_failures": backend["botorch_fit_failures"],
            "botorch_candidate_failures": backend[
                "botorch_candidate_failures"],
            "total_time_sec": float(backend["total_time_sec"]),
            "runtime_fingerprint": backend["runtime_fingerprint"],
        },
        "target_outcomes_used_to_define_coordinate": False,
        "target_search_observations_used_by_backend": True,
        "target_oracle_used_during_search": False,
        "search_records": search_records,
        "shortlist": raw_shortlist,
        "verification": verification,
        "independently_certified": bool(verification["certified"]),
        "deployment_truth": deployment_truth,
        "false_certificate": bool(
            verification["certified"]
            and deployment_truth is not None
            and not deployment_truth[
                "truly_feasible_on_empirical_window_distribution"]
        ),
        "objective_if_certified": (
            None if deployment_truth is None
            else deployment_truth["finite_empirical_objective_mean"]
        ),
        "certificate_scope": (
            "fixed_empirical_distribution_over_admissible_window_start_indices"
        ),
        "future_process_generalization_claimed": False,
        "wall_time_sec": float(time.perf_counter() - started),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--target-market", choices=TARGET_MARKETS, required=True)
    parser.add_argument("--target-seed", type=int, required=True)
    parser.add_argument("--year", type=int, default=2018)
    parser.add_argument("--d", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--N", type=int, default=13)
    parser.add_argument("--coefficient-count", type=int, default=8)
    parser.add_argument("--coefficient-scale", type=float, default=0.25)
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
        data_path=args.data,
        target_market=args.target_market,
        target_seed=args.target_seed,
        year=args.year,
        dimension=args.d,
        alpha=args.alpha,
        n0=args.n0,
        N=args.N,
        coefficient_count=args.coefficient_count,
        coefficient_scale=args.coefficient_scale,
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
        "target_market": payload["target_market"],
        "target_seed": payload["target_seed"],
        "certified": payload["independently_certified"],
        "false_certificate": payload["false_certificate"],
    }, indent=2, sort_keys=True))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
