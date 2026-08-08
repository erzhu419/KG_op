#!/usr/bin/env python3
"""Forecast-indexed OPSD profile benchmark frozen after the OR review.

Unlike energy V2, the profile coordinate is an observable state-feedback map
from forecast stress to target state of charge.  The 1000-point decision grid
and the 168-hour physical simulation horizon are deliberately separate.
"""

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
from core.designs import (  # noqa: E402
    common_sobol_integer_design,
    integer_design_fingerprint,
    next_sobol_integer_candidate,
)
from core.profile_atlas import (  # noqa: E402
    ProfileAtlasConfig,
    SourceProfileRecord,
    SourceScoredProfileAtlas,
)
from core.terminal_verification import (  # noqa: E402
    verify_frozen_shortlist_binomial,
)
from performance.benchmark_external_energy_v2 import (  # noqa: E402
    TARGET_MARKETS,
    _atomic_json,
    _natural_constant_points,
    _sha256_file,
    _structural_initial_profiles,
    market_region,
    region_heldout_source_markets,
)
from performance.benchmark_profile_stress_suite import (  # noqa: E402
    _select_shortlist,
)
from problems.energy_forecast_policy import (  # noqa: E402
    OPSDForecastIndexedStorageProblem,
)
from problems.profile_coefficient_space import (  # noqa: E402
    CosineCoefficientProfileProblem,
)
from problems.randomized_profiles import (  # noqa: E402
    generate_structural_profile_library,
)


CONTRACT_ID = "opsd_forecast_indexed_region_holdout_v3"
DESIGN_CONTRACT_ID = "opsd_forecast_indexed_source_atlas_design_v3"
ARMS = (
    "source_atlas",
    "generic_dct_maximin",
    "random_low_frequency",
    "natural_constant_grid",
    "raw_sobol",
    "target_only_dct_space_scbo",
)


def _profile_point(problem, profile):
    return problem.continuous_to_int(np.interp(
        problem.nodes,
        profile.nodes,
        profile.values,
        left=float(profile.values[0]),
        right=float(profile.values[-1]),
    ))


def build_source_archive(
    data_path,
    *,
    target_market,
    library,
    year=2018,
    dimension=1000,
    horizon=168,
    replications=3,
    alpha=0.05,
    seed=20260808,
):
    """Evaluate a shared stress-response library outside the target region."""

    replications = int(replications)
    if replications < 1:
        raise ValueError("source replications must be positive")
    source_markets = region_heldout_source_markets(target_market)
    rows = []
    for market_index, market in enumerate(source_markets):
        problem = OPSDForecastIndexedStorageProblem(
            data_path,
            market=market,
            year=int(year),
            d=int(dimension),
            horizon=int(horizon),
            alpha=float(alpha),
        )
        for profile_index, profile in enumerate(library):
            point = _profile_point(problem, profile)
            samples = []
            for replication in range(replications):
                rng = np.random.default_rng(np.random.SeedSequence([
                    int(seed), market_index, profile_index, replication, 3109,
                ]))
                samples.append(problem.simulate(point, rng))
            samples = np.vstack(samples)
            rows.append(SourceProfileRecord(
                task_id=f"OPSD-V3:{market}:search_{int(year) - 1}",
                profile_id=profile.profile_id,
                profile=profile.values,
                objective_samples=tuple(float(value) for value in samples[:, 0]),
                constraint_samples=tuple(float(value) for value in samples[:, 1]),
                alpha=problem.alpha,
                tau=problem.tau,
                descriptor=(),
                nodes=profile.nodes,
            ))
    return tuple(rows), source_markets


def materialize_source_atlas(
    *,
    data_path,
    target_market,
    year=2018,
    dimension=1000,
    horizon=168,
    alpha=0.05,
    n0=10,
    library_size=64,
    source_replications=3,
    family_seed=20260808,
):
    """Freeze one target-outcome-free forecast-response initial design."""

    library = generate_structural_profile_library(
        int(library_size),
        dimension=128,
        seed=int(family_seed) + 991,
        maximum_frequency=40,
    )
    records, source_markets = build_source_archive(
        data_path,
        target_market=target_market,
        library=library,
        year=year,
        dimension=dimension,
        horizon=horizon,
        replications=source_replications,
        alpha=alpha,
        seed=family_seed + 1237,
    )
    atlas = SourceScoredProfileAtlas(ProfileAtlasConfig(
        n0=int(n0),
        max_frequency=8,
        frequency_penalty=0.25,
        include_diagonal_quadratic=True,
        safety_metric_weight=1.0,
        objective_metric_weight=1.0,
        first_center_safety_weight=0.5,
    )).fit(records, target_descriptor=None).selected()
    target = OPSDForecastIndexedStorageProblem(
        data_path,
        market=target_market,
        year=int(year),
        d=int(dimension),
        horizon=int(horizon),
        alpha=float(alpha),
        outcome_access=False,
    )
    by_id = {profile.profile_id: profile for profile in library}
    profiles = tuple(by_id[item.profile_id] for item in atlas.members)
    points = tuple(_profile_point(target, profile) for profile in profiles)
    if len(points) != int(n0) or len(set(points)) != int(n0):
        raise RuntimeError("frozen energy V3 atlas is not a unique n0 design")
    return {
        "schema_version": 1,
        "contract_id": DESIGN_CONTRACT_ID,
        "status": "frozen_before_target_outcomes",
        "target_market": str(target_market),
        "target_region": market_region(target_market),
        "year": int(year),
        "nominal_dimension": int(dimension),
        "simulation_horizon_hours": int(horizon),
        "policy_semantics": target.policy_semantics,
        "stress_coordinate_contract": target.stress_coordinate_contract,
        "alpha": float(alpha),
        "n0": int(n0),
        "library_size": int(library_size),
        "source_replications": int(source_replications),
        "family_seed": int(family_seed),
        "source_markets": list(source_markets),
        "source_regions": [market_region(market) for market in source_markets],
        "source_calls": int(
            len(source_markets) * int(library_size) * int(source_replications)
        ),
        "data_sha256": _sha256_file(data_path),
        "selected_profile_ids": [profile.profile_id for profile in profiles],
        "points": [list(map(int, point)) for point in points],
        "initial_design_fingerprint": integer_design_fingerprint(points),
        "frontend_diagnostics": dict(atlas.diagnostics),
        "target_outcomes_used": False,
        "target_oracle_used": False,
        "target_region_excluded_from_source_archive": True,
    }


def load_frozen_source_atlas_design(
    path,
    *,
    data_path,
    target_market,
    year,
    dimension,
    horizon,
    alpha,
    n0,
    library_size,
    source_replications,
):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {
        "contract_id": DESIGN_CONTRACT_ID,
        "status": "frozen_before_target_outcomes",
        "target_market": str(target_market),
        "year": int(year),
        "nominal_dimension": int(dimension),
        "simulation_horizon_hours": int(horizon),
        "policy_semantics": OPSDForecastIndexedStorageProblem.policy_semantics,
        "stress_coordinate_contract": (
            OPSDForecastIndexedStorageProblem.stress_coordinate_contract),
        "alpha": float(alpha),
        "n0": int(n0),
        "library_size": int(library_size),
        "source_replications": int(source_replications),
        "target_outcomes_used": False,
        "target_oracle_used": False,
        "target_region_excluded_from_source_archive": True,
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    digest = _sha256_file(data_path)
    if payload.get("data_sha256") != digest:
        mismatches["data_sha256"] = (payload.get("data_sha256"), digest)
    if mismatches:
        raise ValueError(f"frozen energy V3 design mismatch: {mismatches}")
    points = tuple(
        tuple(int(value) for value in point) for point in payload["points"])
    if len(points) != int(n0) or len(set(points)) != int(n0):
        raise ValueError("frozen energy V3 design is not a unique n0 design")
    if integer_design_fingerprint(points) != payload["initial_design_fingerprint"]:
        raise ValueError("frozen energy V3 design fingerprint mismatch")
    return points, payload


def _verify_and_summarize(
    target,
    shortlist,
    *,
    design_seed,
    N,
    verification_budgets,
    familywise_delta,
):
    deployed, verification = verify_frozen_shortlist_binomial(
        target,
        shortlist,
        seed=int(design_seed) + 71_003,
        search_evaluation_count=int(N),
        candidate_budgets=tuple(int(value) for value in verification_budgets),
        familywise_delta=float(familywise_delta),
        split="verification",
        all_success_only=True,
    )
    deployment_truth = None
    if deployed is not None:
        population = target.split_population(deployed, "verification")
        probability = float(np.mean(population[:, 1] <= target.tau))
        deployment_truth = {
            "finite_empirical_window_count": int(len(population)),
            "finite_empirical_feasibility_probability": probability,
            "truly_feasible_on_empirical_window_distribution": bool(
                probability >= 1.0 - target.alpha),
            "finite_empirical_objective_mean": float(
                np.mean(population[:, 0])),
            "used_for_search_or_selection": False,
        }
    return verification, deployment_truth


def _run_functional_backend(
    target,
    *,
    target_market,
    target_seed,
    design_seed,
    n0,
    N,
    verification_budgets,
    familywise_delta,
    coefficient_count,
    coefficient_scale,
    checkpoint_path,
    checkpoint_resume,
    raw_samples,
    num_restarts,
    maxiter,
    batch_candidates,
):
    functional = CosineCoefficientProfileProblem(
        target,
        coefficient_count=int(coefficient_count),
        coefficient_scale=float(coefficient_scale),
        level_bounds=(0.05, 0.95),
        schema_mode="declared",
        nominal_sigma=1e-7,
    )
    config = BoTorchBaselineConfig(
        N=int(N),
        n0=int(n0),
        seed=int(design_seed),
        method="botorch_scbo",
        raw_samples=int(raw_samples),
        num_restarts=int(num_restarts),
        maxiter=int(maxiter),
        batch_candidates=int(batch_candidates),
        ts_candidates=0,
        strict_failures=True,
        use_problem_initial_samples=False,
        use_boundary_initial_samples=False,
        initial_design="sobol",
        checkpoint_path=str(checkpoint_path or ""),
        checkpoint_resume=bool(checkpoint_resume),
        checkpoint_interval=1,
        progress_logging=False,
        progress_label=(
            f"energy-v3-functional:{target_market}:seed{target_seed}:N{N}"),
        torch_device="cpu",
    )
    backend = BoTorchBaseline(functional, config).run(
        freeze_terminal_shortlist=True,
        evaluate_truth=False,
        terminal_probability_slack=0.05,
        terminal_require_provider=False,
        terminal_shortlist_mode="posterior_objective_challenger_then_safe",
        terminal_shortlist_size=3,
        terminal_maximum_violation_probability=0.5,
    )
    shortlist = []
    for row in backend["frozen_terminal_shortlist"]:
        coefficient_point = tuple(int(value) for value in row["point"])
        raw_point = functional.raw_point(coefficient_point)
        shortlist.append({
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
    verification, truth = _verify_and_summarize(
        target,
        shortlist,
        design_seed=design_seed,
        N=N,
        verification_budgets=verification_budgets,
        familywise_delta=familywise_delta,
    )
    records = []
    for index, row in enumerate(backend["history"]):
        coefficient_point = tuple(int(value) for value in row["x"])
        raw_point = functional.raw_point(coefficient_point)
        records.append({
            "evaluation_index": int(index),
            "source": (
                "target_only_coefficient_sobol_initial"
                if index < int(n0) else "target_only_functional_scbo"
            ),
            "coefficient_point": list(coefficient_point),
            "coefficient_point_fingerprint": integer_design_fingerprint(
                [coefficient_point]),
            "point_fingerprint": integer_design_fingerprint([raw_point]),
            "observation": [float(value) for value in row["y"]],
        })
    return {
        "search_records": records,
        "shortlist": shortlist,
        "verification": verification,
        "deployment_truth": truth,
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
            "runtime_fingerprint": backend["runtime_fingerprint"],
        },
        "initial_design_fingerprint": integer_design_fingerprint([
            tuple(int(value) for value in row["x"])
            for row in backend["history"][:int(n0)]
        ]),
        "frontend_diagnostics": {
            "contract_id": functional.contract_id,
            "source_outcomes_used": False,
            "target_outcomes_used_to_define_coordinate": False,
            "target_oracle_used": False,
        },
    }


def run_task(
    *,
    data_path,
    target_market,
    target_seed,
    arm,
    year=2018,
    dimension=1000,
    horizon=168,
    alpha=0.05,
    n0=10,
    N=13,
    library_size=64,
    source_replications=3,
    family_seed=20260808,
    verification_budgets=(80, 80, 80),
    familywise_delta=0.05,
    amortization_targets=20,
    design_path=None,
    coefficient_count=8,
    coefficient_scale=0.25,
    checkpoint_path="",
    checkpoint_resume=False,
    raw_samples=1024,
    num_restarts=10,
    maxiter=100,
    batch_candidates=128,
):
    target_market = str(target_market)
    arm = str(arm)
    if target_market not in TARGET_MARKETS:
        raise ValueError(f"unknown registered target market: {target_market}")
    if arm not in ARMS:
        raise ValueError(f"unknown energy V3 arm: {arm}")
    n0, N = int(n0), int(N)
    if N < n0 or n0 < 3:
        raise ValueError("energy V3 requires N >= n0 >= 3")
    target_seed = int(target_seed)
    design_seed = 4_300_000 + target_seed
    target = OPSDForecastIndexedStorageProblem(
        data_path,
        market=target_market,
        year=int(year),
        d=int(dimension),
        horizon=int(horizon),
        alpha=float(alpha),
    )
    started = time.perf_counter()
    source_markets = ()
    source_calls = 0
    selected_profile_ids = []

    if arm == "target_only_dct_space_scbo":
        result = _run_functional_backend(
            target,
            target_market=target_market,
            target_seed=target_seed,
            design_seed=design_seed,
            n0=n0,
            N=N,
            verification_budgets=verification_budgets,
            familywise_delta=familywise_delta,
            coefficient_count=coefficient_count,
            coefficient_scale=coefficient_scale,
            checkpoint_path=checkpoint_path,
            checkpoint_resume=checkpoint_resume,
            raw_samples=raw_samples,
            num_restarts=num_restarts,
            maxiter=maxiter,
            batch_candidates=batch_candidates,
        )
    else:
        library = generate_structural_profile_library(
            int(library_size),
            dimension=128,
            seed=int(family_seed) + 991,
            maximum_frequency=40,
        )
        atlas = None
        if arm == "source_atlas":
            if design_path is None:
                raise ValueError(
                    "confirmatory energy V3 source_atlas requires frozen design")
            points, frozen = load_frozen_source_atlas_design(
                design_path,
                data_path=data_path,
                target_market=target_market,
                year=year,
                dimension=dimension,
                horizon=horizon,
                alpha=alpha,
                n0=n0,
                library_size=library_size,
                source_replications=source_replications,
            )
            source_markets = tuple(frozen["source_markets"])
            source_calls = int(frozen["source_calls"])
            selected_profile_ids = list(frozen["selected_profile_ids"])
            frontend = dict(frozen["frontend_diagnostics"])
        elif arm == "raw_sobol":
            points = tuple(common_sobol_integer_design(
                target, n0, design_seed, seed_offset=19_873))
            frontend = {
                "contract_id": "raw_common_sobol_v1",
                "source_outcomes_used": False,
                "target_outcomes_used": False,
                "target_oracle_used": False,
            }
        elif arm == "natural_constant_grid":
            points = _natural_constant_points(target, n0)
            frontend = {
                "contract_id": "natural_constant_policy_grid_v3",
                "source_outcomes_used": False,
                "target_outcomes_used": False,
                "target_oracle_used": False,
            }
        else:
            profiles, frontend = _structural_initial_profiles(
                arm, library, atlas, n0, design_seed)
            points = tuple(_profile_point(target, profile) for profile in profiles)
            selected_profile_ids = [profile.profile_id for profile in profiles]
        if len(points) != n0 or len(set(points)) != n0:
            raise RuntimeError("energy V3 initial design must contain n0 unique points")

        records = []
        observed = list(points)
        for index, point in enumerate(points):
            rng = np.random.default_rng(np.random.SeedSequence([
                design_seed, index, 3211,
            ]))
            records.append({
                "point": tuple(point),
                "observation": np.asarray(
                    target.simulate(point, rng), dtype=float),
                "source": "initial_design",
                "evaluation_index": index,
            })
        while len(records) < N:
            point = next_sobol_integer_candidate(
                target,
                design_seed,
                observed=observed,
                seed_offset=330_107,
            )
            index = len(records)
            rng = np.random.default_rng(np.random.SeedSequence([
                design_seed, index, 3211,
            ]))
            observed.append(tuple(point))
            records.append({
                "point": tuple(point),
                "observation": np.asarray(
                    target.simulate(point, rng), dtype=float),
                "source": "neutral_sobol_continuation",
                "evaluation_index": index,
            })
        shortlist = _select_shortlist(records, target.tau, size=3)
        verification, truth = _verify_and_summarize(
            target,
            shortlist,
            design_seed=design_seed,
            N=N,
            verification_budgets=verification_budgets,
            familywise_delta=familywise_delta,
        )
        result = {
            "search_records": [{
                "evaluation_index": row["evaluation_index"],
                "source": row["source"],
                "point_fingerprint": integer_design_fingerprint([row["point"]]),
                "observation": row["observation"].tolist(),
            } for row in records],
            "shortlist": shortlist,
            "verification": verification,
            "deployment_truth": truth,
            "initial_design_fingerprint": integer_design_fingerprint(points),
            "frontend_diagnostics": frontend,
        }

    verification = result["verification"]
    truth = result["deployment_truth"]
    verification_calls = int(verification["verification_budget"])
    maximum_verification_calls = int(sum(
        int(value) for value in verification_budgets))
    amortized_source = float(source_calls / int(amortization_targets))
    return {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "status": "ok",
        "target_market": target_market,
        "target_region": market_region(target_market),
        "target_seed": target_seed,
        "design_seed": design_seed,
        "arm": arm,
        "year": int(year),
        "nominal_dimension": int(dimension),
        "simulation_horizon_hours": int(horizon),
        "alpha": float(alpha),
        "n0": n0,
        "N": N,
        "source_markets": list(source_markets),
        "source_regions": [market_region(market) for market in source_markets],
        "source_calls": int(source_calls),
        "target_search_calls": N,
        "verification_calls": verification_calls,
        "maximum_verification_calls": maximum_verification_calls,
        "all_in_calls_unamortized": int(
            source_calls + N + verification_calls),
        "all_in_budget_cap_unamortized": int(
            source_calls + N + maximum_verification_calls),
        "amortization_targets": int(amortization_targets),
        "all_in_calls_amortized": float(
            amortized_source + N + verification_calls),
        "all_in_budget_cap_amortized": float(
            amortized_source + N + maximum_verification_calls),
        "library_size": int(library_size),
        "source_replications": int(source_replications),
        "data_sha256": _sha256_file(data_path),
        "information_contract": target.information_contract(),
        "selected_profile_ids": selected_profile_ids,
        "source_archive_target_region_excluded": bool(
            all(market_region(market) != market_region(target_market)
                for market in source_markets)),
        "target_outcomes_used_to_fit_frontend": False,
        "target_oracle_used_during_search": False,
        **result,
        "independently_certified": bool(verification["certified"]),
        "false_certificate": bool(
            verification["certified"]
            and truth is not None
            and not truth["truly_feasible_on_empirical_window_distribution"]
        ),
        "objective_if_certified": (
            None if truth is None else truth["finite_empirical_objective_mean"]
        ),
        "certificate_scope": target.verification_distribution_scope,
        "future_process_generalization_claimed": False,
        "wall_time_sec": float(time.perf_counter() - started),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--target-market", choices=TARGET_MARKETS, required=True)
    parser.add_argument("--target-seed", type=int, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--design-path")
    parser.add_argument("--year", type=int, default=2018)
    parser.add_argument("--d", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=168)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--N", type=int, default=13)
    parser.add_argument("--verification-budgets", default="80,80,80")
    parser.add_argument("--familywise-delta", type=float, default=0.05)
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--checkpoint-resume", action="store_true")
    args = parser.parse_args()
    payload = run_task(
        data_path=args.data,
        target_market=args.target_market,
        target_seed=args.target_seed,
        arm=args.arm,
        year=args.year,
        dimension=args.d,
        horizon=args.horizon,
        alpha=args.alpha,
        n0=args.n0,
        N=args.N,
        verification_budgets=tuple(
            int(value) for value in args.verification_budgets.split(",")),
        familywise_delta=args.familywise_delta,
        design_path=args.design_path,
        checkpoint_path=args.checkpoint_path,
        checkpoint_resume=args.checkpoint_resume,
    )
    _atomic_json(args.out, payload)
    print(json.dumps({
        "status": payload["status"],
        "target_market": payload["target_market"],
        "target_seed": payload["target_seed"],
        "arm": payload["arm"],
        "certified": payload["independently_certified"],
    }, sort_keys=True))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
