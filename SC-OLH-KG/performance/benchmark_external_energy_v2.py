#!/usr/bin/env python3
"""Region-held-out OPSD profile-design benchmark.

This experiment isolates the initial-design frontend. Every arm receives the
same neutral Sobol continuation and exact finite-distribution verifier. The
source-scored arm learns only from four preregistered source markets outside
the target region; target outcomes are unavailable until the design is frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.designs import (  # noqa: E402
    common_sobol_integer_design,
    integer_design_fingerprint,
    next_sobol_integer_candidate,
)
from core.profile_atlas import (  # noqa: E402
    ProfileAtlasConfig,
    SourceProfileRecord,
    SourceScoredProfileAtlas,
    generic_dct_maximin,
    resample_profile,
)
from core.terminal_verification import verify_frozen_shortlist_binomial  # noqa: E402
from performance.benchmark_profile_stress_suite import (  # noqa: E402
    _select_shortlist,
)
from performance.benchmark_quality import json_safe  # noqa: E402
from problems.energy_reliability import OPSDStorageReliabilityProblem  # noqa: E402
from problems.randomized_profiles import (  # noqa: E402
    StructuralProfile,
    generate_structural_profile_library,
)


CONTRACT_ID = "opsd_region_heldout_profile_design_v2"
ENERGY_REGIONS = {
    "denmark": ("DK_1", "DK_2"),
    "great_britain": ("GB_GBN",),
    "italy": (
        "IT_CNOR", "IT_CSUD", "IT_NORD", "IT_SARD", "IT_SICI", "IT_SUD",
    ),
    "norway": ("NO_1", "NO_2", "NO_3", "NO_4", "NO_5"),
    "sweden": ("SE_1", "SE_2", "SE_3", "SE_4"),
}
REGION_REPRESENTATIVES = {
    "denmark": "DK_1",
    "great_britain": "GB_GBN",
    "italy": "IT_NORD",
    "norway": "NO_1",
    "sweden": "SE_1",
}
TARGET_MARKETS = tuple(
    market for markets in ENERGY_REGIONS.values() for market in markets
)
ARMS = (
    "source_atlas",
    "generic_dct_maximin",
    "random_low_frequency",
    "natural_constant_grid",
    "raw_sobol",
)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def market_region(market):
    market = str(market)
    for region, markets in ENERGY_REGIONS.items():
        if market in markets:
            return region
    raise ValueError(f"market is outside the registered energy suite: {market}")


def region_heldout_source_markets(target_market):
    """One fixed representative from every non-target region."""

    target_region = market_region(target_market)
    return tuple(
        REGION_REPRESENTATIVES[region]
        for region in ENERGY_REGIONS
        if region != target_region
    )


def _profile_point(problem, profile):
    values = resample_profile(
        profile.values,
        (np.arange(problem.d, dtype=float) + 0.5) / float(problem.d),
        source_nodes=profile.nodes,
    )
    return problem.continuous_to_int(values)


def build_source_archive(
    data_path,
    *,
    target_market,
    library,
    year=2018,
    dimension=1000,
    replications=3,
    alpha=0.05,
    seed=20260808,
):
    """Evaluate the shared profile library on non-target-region markets."""

    replications = int(replications)
    if replications < 1:
        raise ValueError("source replications must be positive")
    rows = []
    source_markets = region_heldout_source_markets(target_market)
    for market_index, market in enumerate(source_markets):
        problem = OPSDStorageReliabilityProblem(
            data_path,
            market=market,
            year=int(year),
            d=int(dimension),
            alpha=float(alpha),
        )
        for profile_index, profile in enumerate(library):
            point = _profile_point(problem, profile)
            samples = []
            for replication in range(replications):
                rng = np.random.default_rng(np.random.SeedSequence([
                    int(seed), market_index, profile_index, replication, 2609,
                ]))
                samples.append(problem.simulate(point, rng))
            samples = np.vstack(samples)
            rows.append(SourceProfileRecord(
                task_id=f"OPSD:{market}:search_{int(year) - 1}",
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
    alpha=0.05,
    n0=10,
    library_size=64,
    source_replications=3,
    family_seed=20260808,
):
    """Freeze one target-outcome-free region-held-out source design."""

    library = generate_structural_profile_library(
        int(library_size),
        dimension=128,
        seed=int(family_seed) + 991,
        maximum_frequency=40,
    )
    source_records, source_markets = build_source_archive(
        data_path,
        target_market=target_market,
        library=library,
        year=year,
        dimension=dimension,
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
    )).fit(source_records, target_descriptor=None).selected()
    target = OPSDStorageReliabilityProblem(
        data_path,
        market=target_market,
        year=int(year),
        d=int(dimension),
        alpha=float(alpha),
        outcome_access=False,
    )
    by_id = {profile.profile_id: profile for profile in library}
    profiles = tuple(by_id[item.profile_id] for item in atlas.members)
    points = tuple(_profile_point(target, profile) for profile in profiles)
    if len(points) != int(n0) or len(set(points)) != int(n0):
        raise RuntimeError("frozen energy source atlas is not a unique n0 design")
    return {
        "schema_version": 1,
        "contract_id": "opsd_region_heldout_source_atlas_design_v2",
        "status": "frozen_before_target_outcomes",
        "target_market": str(target_market),
        "target_region": market_region(target_market),
        "year": int(year),
        "nominal_dimension": int(dimension),
        "alpha": float(alpha),
        "n0": int(n0),
        "library_size": int(library_size),
        "source_replications": int(source_replications),
        "family_seed": int(family_seed),
        "source_markets": list(source_markets),
        "source_regions": [market_region(market) for market in source_markets],
        "source_calls": int(
            len(source_markets) * library_size * source_replications),
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
    alpha,
    n0,
    library_size,
    source_replications,
):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {
        "contract_id": "opsd_region_heldout_source_atlas_design_v2",
        "status": "frozen_before_target_outcomes",
        "target_market": str(target_market),
        "year": int(year),
        "nominal_dimension": int(dimension),
        "alpha": float(alpha),
        "n0": int(n0),
        "library_size": int(library_size),
        "source_replications": int(source_replications),
        "target_outcomes_used": False,
        "target_oracle_used": False,
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if payload.get("data_sha256") != _sha256_file(data_path):
        mismatches["data_sha256"] = (
            payload.get("data_sha256"), _sha256_file(data_path))
    if mismatches:
        raise ValueError(f"frozen energy V2 design mismatch: {mismatches}")
    points = tuple(
        tuple(int(value) for value in point) for point in payload["points"]
    )
    if len(points) != int(n0) or len(set(points)) != int(n0):
        raise ValueError("frozen energy V2 design is not a unique n0 design")
    if integer_design_fingerprint(points) != payload["initial_design_fingerprint"]:
        raise ValueError("frozen energy V2 design fingerprint mismatch")
    return points, payload


def _structural_initial_profiles(arm, library, atlas, n0, design_seed):
    by_id = {profile.profile_id: profile for profile in library}
    if arm == "source_atlas":
        return tuple(by_id[item.profile_id] for item in atlas.members), {
            **dict(atlas.diagnostics),
            "target_region_outcomes_used": False,
        }
    if arm == "generic_dct_maximin":
        indices, diagnostics = generic_dct_maximin(
            [profile.values for profile in library],
            int(n0),
            nodes=library[0].nodes,
            max_frequency=8,
            frequency_penalty=0.25,
            include_diagonal_quadratic=True,
        )
        return tuple(library[index] for index in indices), diagnostics
    if arm == "random_low_frequency":
        eligible = tuple(
            profile for profile in library
            if profile.family in {"constant", "ramp", "low_frequency"}
        )
        rng = np.random.default_rng(int(design_seed))
        indices = rng.choice(len(eligible), size=int(n0), replace=False)
        return tuple(eligible[int(index)] for index in indices), {
            "contract_id": "random_low_frequency_library_subset_v1",
            "source_outcomes_used": False,
            "target_outcomes_used": False,
            "target_oracle_used": False,
        }
    raise ValueError(f"arm has no structural profile design: {arm}")


def _natural_constant_points(problem, n0):
    levels = np.rint(np.linspace(0, problem.L, int(n0))).astype(int)
    if len(set(map(int, levels))) != int(n0):
        raise RuntimeError("natural constant grid has duplicate levels")
    return tuple(tuple([int(level)] * problem.d) for level in levels)


def run_task(
    *,
    data_path,
    target_market,
    target_seed,
    arm,
    year=2018,
    dimension=1000,
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
):
    target_market = str(target_market)
    arm = str(arm)
    if target_market not in TARGET_MARKETS:
        raise ValueError(f"unknown registered target market: {target_market}")
    if arm not in ARMS:
        raise ValueError(f"unknown energy V2 arm: {arm}")
    n0 = int(n0)
    N = int(N)
    if N < n0 or n0 < 3:
        raise ValueError("energy V2 requires N >= n0 >= 3")
    target_seed = int(target_seed)
    design_seed = 4_100_000 + target_seed
    target = OPSDStorageReliabilityProblem(
        data_path,
        market=target_market,
        year=int(year),
        d=int(dimension),
        alpha=float(alpha),
    )
    library = generate_structural_profile_library(
        int(library_size),
        dimension=128,
        seed=int(family_seed) + 991,
        maximum_frequency=40,
    )

    atlas = None
    source_markets = ()
    source_calls = 0
    frozen_points = None
    selected_profile_ids = []
    frontend = {
        "contract_id": "not_applicable",
        "source_outcomes_used": False,
        "target_outcomes_used": False,
        "target_oracle_used": False,
    }
    if arm == "source_atlas":
        if design_path is not None:
            frozen_points, frozen = load_frozen_source_atlas_design(
                design_path,
                data_path=data_path,
                target_market=target_market,
                year=year,
                dimension=dimension,
                alpha=alpha,
                n0=n0,
                library_size=library_size,
                source_replications=source_replications,
            )
            source_markets = tuple(frozen["source_markets"])
            source_calls = int(frozen["source_calls"])
            frontend = dict(frozen["frontend_diagnostics"])
            selected_profile_ids = list(frozen["selected_profile_ids"])
        else:
            source_records, source_markets = build_source_archive(
                data_path,
                target_market=target_market,
                library=library,
                year=year,
                dimension=dimension,
                replications=source_replications,
                alpha=alpha,
                seed=family_seed + 1237,
            )
            source_calls = int(
                len(source_markets) * library_size * source_replications)
            atlas = SourceScoredProfileAtlas(ProfileAtlasConfig(
                n0=n0,
                max_frequency=8,
                frequency_penalty=0.25,
                include_diagonal_quadratic=True,
                safety_metric_weight=1.0,
                objective_metric_weight=1.0,
                first_center_safety_weight=0.5,
            )).fit(source_records, target_descriptor=None).selected()

    if arm == "raw_sobol":
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
            "contract_id": "natural_constant_policy_grid_v2",
            "source_outcomes_used": False,
            "target_outcomes_used": False,
            "target_oracle_used": False,
        }
    elif arm == "source_atlas" and design_path is not None:
        if frozen_points is None:
            raise RuntimeError("frozen source atlas design was not loaded")
        points = frozen_points
    else:
        profiles, frontend = _structural_initial_profiles(
            arm, library, atlas, n0, design_seed)
        points = tuple(_profile_point(target, profile) for profile in profiles)
        selected_profile_ids = [profile.profile_id for profile in profiles]
    if len(points) != n0 or len(set(points)) != n0:
        raise RuntimeError("energy V2 initial design must contain n0 unique points")

    records = []
    observed = list(points)
    for index, point in enumerate(points):
        rng = np.random.default_rng(np.random.SeedSequence([
            design_seed, index, 2711,
        ]))
        records.append({
            "point": tuple(point),
            "observation": np.asarray(target.simulate(point, rng), dtype=float),
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
            design_seed, index, 2711,
        ]))
        observed.append(tuple(point))
        records.append({
            "point": tuple(point),
            "observation": np.asarray(target.simulate(point, rng), dtype=float),
            "source": "neutral_sobol_continuation",
            "evaluation_index": index,
        })

    shortlist = _select_shortlist(records, target.tau, size=3)
    deployed, verification = verify_frozen_shortlist_binomial(
        target,
        shortlist,
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
        probability = float(np.mean(population[:, 1] <= target.tau))
        deployment_truth = {
            "finite_empirical_window_count": int(len(population)),
            "finite_empirical_feasibility_probability": probability,
            "truly_feasible_on_empirical_window_distribution": bool(
                probability >= 1.0 - target.alpha),
            "finite_empirical_objective_mean": float(np.mean(population[:, 0])),
            "used_for_search_or_selection": False,
        }
    verification_calls = int(verification["verification_budget"])
    maximum_verification_calls = int(sum(
        int(value) for value in verification_budgets))
    amortized_source_calls = float(source_calls / int(amortization_targets))
    return {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "status": "ok",
        "target_market": target_market,
        "target_region": market_region(target_market),
        "target_seed": target_seed,
        "arm": arm,
        "year": int(year),
        "nominal_dimension": int(dimension),
        "alpha": float(alpha),
        "n0": n0,
        "target_search_calls": N,
        "source_markets": list(source_markets),
        "source_regions": [market_region(market) for market in source_markets],
        "source_calls": source_calls,
        "verification_calls": verification_calls,
        "maximum_verification_calls": maximum_verification_calls,
        "all_in_calls_unamortized": int(source_calls + N + verification_calls),
        "all_in_budget_cap_unamortized": int(
            source_calls + N + maximum_verification_calls),
        "amortization_targets": int(amortization_targets),
        "all_in_calls_amortized": float(
            amortized_source_calls + N + verification_calls),
        "all_in_budget_cap_amortized": float(
            amortized_source_calls + N + maximum_verification_calls),
        "library_size": int(library_size),
        "source_replications": int(source_replications),
        "information_contract": target.information_contract(),
        "frontend_diagnostics": frontend,
        "frozen_design_path": (
            None if design_path is None else str(design_path)),
        "selected_profile_ids": selected_profile_ids,
        "initial_design_fingerprint": integer_design_fingerprint(points),
        "source_archive_target_region_excluded": bool(
            all(market_region(market) != market_region(target_market)
                for market in source_markets)),
        "target_outcomes_used_to_fit_frontend": False,
        "target_oracle_used_during_search": False,
        "search_records": [{
            "evaluation_index": row["evaluation_index"],
            "source": row["source"],
            "point_fingerprint": integer_design_fingerprint([row["point"]]),
            "observation": row["observation"].tolist(),
        } for row in records],
        "shortlist": shortlist,
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
            None if deployment_truth is None else
            deployment_truth["finite_empirical_objective_mean"]
        ),
        "certificate_scope": (
            "fixed_empirical_distribution_over_admissible_window_start_indices"
        ),
        "future_process_generalization_claimed": False,
    }


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--target-market", choices=TARGET_MARKETS, required=True)
    parser.add_argument("--target-seed", type=int, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--year", type=int, default=2018)
    parser.add_argument("--d", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--N", type=int, default=13)
    parser.add_argument("--library-size", type=int, default=64)
    parser.add_argument("--source-replications", type=int, default=3)
    parser.add_argument("--family-seed", type=int, default=20260808)
    parser.add_argument("--verification-budgets", default="80,80,80")
    parser.add_argument("--familywise-delta", type=float, default=0.05)
    parser.add_argument("--amortization-targets", type=int, default=20)
    parser.add_argument("--design")
    args = parser.parse_args()
    result = run_task(
        data_path=args.data,
        target_market=args.target_market,
        target_seed=args.target_seed,
        arm=args.arm,
        year=args.year,
        dimension=args.d,
        alpha=args.alpha,
        n0=args.n0,
        N=args.N,
        library_size=args.library_size,
        source_replications=args.source_replications,
        family_seed=args.family_seed,
        verification_budgets=tuple(
            int(value) for value in args.verification_budgets.split(",")),
        familywise_delta=args.familywise_delta,
        amortization_targets=args.amortization_targets,
        design_path=args.design,
    )
    _atomic_json(args.out, result)
    print(json.dumps({
        "status": result["status"],
        "out": str(args.out),
        "target_market": result["target_market"],
        "target_seed": result["target_seed"],
        "arm": result["arm"],
        "independently_certified": result["independently_certified"],
        "false_certificate": result["false_certificate"],
    }, indent=2, sort_keys=True))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
