#!/usr/bin/env python3
"""One-task shard for the randomized ordered-profile stress suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.designs import common_sobol_integer_design, integer_design_fingerprint  # noqa: E402
from core.profile_atlas import (  # noqa: E402
    ProfileAtlasConfig,
    SourceScoredProfileAtlas,
    generic_dct_maximin,
    regular_profile_nodes,
)
from core.terminal_verification import verify_frozen_shortlist_binomial  # noqa: E402
from performance.benchmark_quality import json_safe  # noqa: E402
from problems.randomized_profiles import (  # noqa: E402
    PROFILE_STRESS_REGIMES,
    RandomizedOrderedProfileProblem,
    StructuralProfile,
    generate_structural_profile_library,
    source_profile_records,
)


CONTRACT_ID = "randomized_ordered_profile_stress_v2"
ARMS = (
    "source_atlas",
    "generic_dct_maximin",
    "random_low_frequency",
    "natural_blockwise",
    "raw_sobol",
    "oracle_library_upper_bound",
)


def _stable_seed(text):
    value = 0
    for character in str(text).encode("utf-8"):
        value = (value * 257 + int(character)) % (2 ** 31 - 1)
    return int(value)


def natural_blockwise_profiles(count, *, dimension=128):
    """Outcome-free constants and three-block policies."""

    count = int(count)
    nodes = regular_profile_nodes(int(dimension))
    specifications = [
        (0.10, 0.10, 0.10),
        (0.30, 0.30, 0.30),
        (0.50, 0.50, 0.50),
        (0.70, 0.70, 0.70),
        (0.90, 0.90, 0.90),
        (0.20, 0.50, 0.80),
        (0.80, 0.50, 0.20),
        (0.20, 0.80, 0.50),
        (0.50, 0.20, 0.80),
        (0.80, 0.20, 0.50),
        (0.50, 0.80, 0.20),
        (0.15, 0.65, 0.35),
        (0.65, 0.35, 0.15),
        (0.35, 0.15, 0.65),
    ]
    if count > len(specifications):
        raise ValueError("natural blockwise control has insufficient profiles")
    block = np.minimum((3.0 * nodes).astype(int), 2)
    return tuple(
        StructuralProfile(
            profile_id=f"natural_block_{index:02d}",
            values=tuple(float(levels[group]) for group in block),
            nodes=tuple(float(value) for value in nodes),
            family="natural_blockwise",
        )
        for index, levels in enumerate(specifications[:count])
    )


def _select_shortlist(records, tau, size=3):
    feasible = [row for row in records if row["observation"][1] <= float(tau)]
    roles = []
    if feasible:
        roles.append((
            "observed_feasible_objective",
            min(feasible, key=lambda row: row["observation"][0]),
        ))
        roles.append((
            "observed_safe_interior",
            min(feasible, key=lambda row: row["observation"][1]),
        ))
    roles.append((
        "empirical_bayes_risk",
        min(records, key=lambda row: (
            float(row["observation"][0])
            + 2.0 * max(float(row["observation"][1]) - float(tau), 0.0)
        )),
    ))
    roles.extend(
        ("objective_fallback", row)
        for row in sorted(records, key=lambda row: row["observation"][0])
    )
    selected = []
    seen = set()
    for role, row in roles:
        point = tuple(row["point"])
        if point in seen:
            continue
        seen.add(point)
        selected.append({
            "shortlist_position": len(selected) + 1,
            "shortlist_role": role,
            "point": list(point),
            "point_fingerprint": integer_design_fingerprint([point]),
        })
        if len(selected) == min(int(size), len(records)):
            break
    return selected


def _design_profiles(
    arm,
    *,
    library,
    source_selection,
    n0,
    design_seed,
):
    by_id = {profile.profile_id: profile for profile in library}
    if arm == "source_atlas":
        return tuple(
            by_id[member.profile_id] for member in source_selection.members
        ), dict(source_selection.diagnostics)
    if arm == "generic_dct_maximin":
        indices, diagnostics = generic_dct_maximin(
            [profile.values for profile in library],
            n0,
            nodes=library[0].nodes,
            max_frequency=8,
            frequency_penalty=0.25,
            include_diagonal_quadratic=True,
        )
        return tuple(library[index] for index in indices), diagnostics
    if arm == "random_low_frequency":
        eligible = [
            profile for profile in library
            if profile.family in {"constant", "ramp", "low_frequency"}
        ]
        if len(eligible) < n0:
            raise RuntimeError("low-frequency library is smaller than n0")
        rng = np.random.default_rng(int(design_seed))
        indices = rng.choice(len(eligible), size=n0, replace=False)
        return tuple(eligible[int(index)] for index in indices), {
            "contract_id": "random_low_frequency_library_subset_v1",
            "source_outcomes_used": False,
            "target_outcomes_used": False,
            "target_oracle_used": False,
            "design_seed": int(design_seed),
        }
    if arm == "natural_blockwise":
        return natural_blockwise_profiles(
            n0, dimension=len(library[0].values)), {
                "contract_id": "natural_three_block_profile_control_v1",
                "source_outcomes_used": False,
                "target_outcomes_used": False,
                "target_oracle_used": False,
            }
    raise ValueError(f"arm {arm} is not a structural-profile design")


def _oracle_library(target, library):
    rows = []
    for profile in library:
        point = target.point_from_structural_profile(profile, schema_mode="declared")
        rows.append({
            "profile": profile,
            "point": point,
            "feasible": bool(target.is_truly_feasible(point)),
            "objective": float(target.true_objective(point)),
            "margin": float(target.true_chance_margin(point)),
        })
    feasible = [row for row in rows if row["feasible"]]
    if not feasible:
        raise RuntimeError("registered finite audit library has no feasible policy")
    best = min(feasible, key=lambda row: row["objective"])
    return rows, best


def run_task(
    *,
    regime,
    target_seed,
    arm,
    dimension=1000,
    active_rank=None,
    alpha=0.05,
    safe_mass=0.08,
    n0=10,
    source_task_count=2,
    library_size=64,
    source_replications=3,
    schema_mode="declared",
    descriptor_mode="domain_blind",
    family_seed=20260808,
    design_seed=None,
    verification_budgets=(80, 80, 80),
    familywise_delta=0.05,
    amortization_targets=20,
    atlas_max_frequency=8,
    atlas_frequency_penalty=0.25,
    atlas_first_center_safety_weight=0.5,
):
    regime = str(regime)
    arm = str(arm)
    if regime not in PROFILE_STRESS_REGIMES:
        raise ValueError(f"unknown stress regime: {regime}")
    if arm not in ARMS:
        raise ValueError(f"unknown stress arm: {arm}")
    if schema_mode not in {"declared", "schema_blind"}:
        raise ValueError("schema_mode must be declared or schema_blind")
    if descriptor_mode not in {"conditioned", "domain_blind"}:
        raise ValueError("descriptor_mode must be conditioned or domain_blind")
    n0 = int(n0)
    design_seed = (
        int(design_seed)
        if design_seed is not None else 900_000 + int(target_seed)
    )
    regime_seed = int(family_seed) + _stable_seed(regime)
    library = generate_structural_profile_library(
        int(library_size),
        dimension=128,
        seed=int(family_seed) + 991,
        maximum_frequency=40,
    )
    target = RandomizedOrderedProfileProblem(
        regime=regime,
        role="target",
        task_seed=int(target_seed),
        family_seed=regime_seed,
        d=int(dimension),
        active_rank=active_rank,
        alpha=alpha,
        safe_mass=safe_mass,
    )
    atlas = None
    frontend_diagnostics = {
        "contract_id": "not_applicable",
        "source_outcomes_used": False,
        "target_outcomes_used": False,
        "target_oracle_used": False,
    }
    source_calls = 0
    if arm == "source_atlas":
        sources = tuple(
            RandomizedOrderedProfileProblem(
                regime=regime,
                role="source",
                task_seed=10_000 + index,
                family_seed=regime_seed,
                d=256,
                active_rank=active_rank,
                alpha=alpha,
                safe_mass=safe_mass,
            )
            for index in range(int(source_task_count))
        )
        source_records = source_profile_records(
            sources,
            library,
            replications=int(source_replications),
            seed=int(family_seed) + 1237,
        )
        source_calls = int(
            source_task_count * library_size * source_replications)
        atlas = SourceScoredProfileAtlas(ProfileAtlasConfig(
            n0=n0,
            max_frequency=int(atlas_max_frequency),
            frequency_penalty=float(atlas_frequency_penalty),
            include_diagonal_quadratic=True,
            safety_metric_weight=1.0,
            objective_metric_weight=1.0,
            first_center_safety_weight=float(
                atlas_first_center_safety_weight),
            descriptor_temperature=1.0,
        )).fit(
            source_records,
            target_descriptor=(
                target.observable_descriptor()
                if descriptor_mode == "conditioned" else None
            ),
        ).selected()
        frontend_diagnostics = dict(atlas.diagnostics)
    oracle_rows, oracle_best = _oracle_library(target, library)

    target_oracle_used_for_design = arm == "oracle_library_upper_bound"
    if arm == "raw_sobol":
        points = tuple(common_sobol_integer_design(
            target, n0, design_seed, seed_offset=19_873))
        selected_profile_ids = []
        frontend_diagnostics = {
            "contract_id": "raw_common_sobol_v1",
            "source_outcomes_used": False,
            "target_outcomes_used": False,
            "target_oracle_used": False,
        }
    elif arm == "oracle_library_upper_bound":
        ranked = sorted(
            oracle_rows,
            key=lambda row: (
                not row["feasible"],
                row["objective"] if row["feasible"] else row["margin"],
                row["profile"].profile_id,
            ),
        )[:n0]
        points = tuple(row["point"] for row in ranked)
        selected_profile_ids = [row["profile"].profile_id for row in ranked]
        frontend_diagnostics = {
            "contract_id": "finite_library_oracle_upper_bound_v1",
            "source_outcomes_used": False,
            "target_outcomes_used": True,
            "target_oracle_used": True,
        }
    else:
        profiles, control_diagnostics = _design_profiles(
            arm,
            library=library,
            source_selection=atlas,
            n0=n0,
            design_seed=design_seed,
        )
        if arm != "source_atlas":
            frontend_diagnostics = control_diagnostics
        points = tuple(target.point_from_structural_profile(
            profile, schema_mode=schema_mode) for profile in profiles)
        selected_profile_ids = [profile.profile_id for profile in profiles]
    if len(points) != n0 or len(set(points)) != n0:
        raise RuntimeError("stress design must contain n0 unique target points")

    search_records = []
    for index, point in enumerate(points):
        rng = np.random.default_rng(np.random.SeedSequence([
            design_seed, index, 2017,
        ]))
        observation = np.asarray(target.simulate(point, rng), dtype=float)
        search_records.append({
            "point": point,
            "observation": observation,
            "evaluation_index": index,
        })
    shortlist = _select_shortlist(search_records, target.tau, size=3)
    deployed, verification = verify_frozen_shortlist_binomial(
        target,
        shortlist,
        seed=design_seed + 71_003,
        search_evaluation_count=n0,
        candidate_budgets=tuple(int(value) for value in verification_budgets),
        familywise_delta=float(familywise_delta),
        all_success_only=True,
    )

    true_rows = [{
        "point": tuple(point),
        "feasible": bool(target.is_truly_feasible(point)),
        "objective": float(target.true_objective(point)),
        "margin": float(target.true_chance_margin(point)),
        "feasibility_probability": float(
            target.true_feasibility_probability(point)),
    } for point in points]
    feasible_rows = [row for row in true_rows if row["feasible"]]
    best_design = (
        None if not feasible_rows else min(
            feasible_rows, key=lambda row: row["objective"])
    )
    regret = (
        None if best_design is None else max(
            0.0, float(best_design["objective"] - oracle_best["objective"]))
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
    verification_calls = int(verification["verification_budget"])
    all_in_unamortized = int(source_calls + n0 + verification_calls)
    amortized_source = float(source_calls / int(amortization_targets))
    penalized_loss = (
        float(regret)
        if regret is not None else float(
            1.0 + min(max(row["margin"], 0.0) for row in true_rows)
            / max(target.safe_radius, 1e-12)
        )
    )
    return {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "status": "ok",
        "regime": regime,
        "target_seed": int(target_seed),
        "design_seed": int(design_seed),
        "arm": arm,
        "schema_mode": schema_mode,
        "descriptor_mode": descriptor_mode,
        "nominal_dimension": int(target.d),
        "effective_rank": int(target.effective_rank),
        "alpha": float(target.alpha),
        "safe_mass": float(target.safe_mass),
        "n0": n0,
        "source_task_count": int(source_task_count),
        "source_profiles_per_task": int(library_size),
        "source_replications_per_profile": int(source_replications),
        "atlas_max_frequency": int(atlas_max_frequency),
        "atlas_frequency_penalty": float(atlas_frequency_penalty),
        "atlas_first_center_safety_weight": float(
            atlas_first_center_safety_weight),
        "source_calls": source_calls,
        "target_search_calls": n0,
        "verification_calls": verification_calls,
        "all_in_calls_unamortized": all_in_unamortized,
        "amortization_targets": int(amortization_targets),
        "amortized_source_calls_per_target": amortized_source,
        "all_in_calls_amortized": float(
            amortized_source + n0 + verification_calls),
        "target_information_contract": target.information_contract(),
        "frontend_diagnostics": frontend_diagnostics,
        "atlas_diagnostics": (
            dict(atlas.diagnostics) if atlas is not None else None),
        "selected_profile_ids": selected_profile_ids,
        "design_fingerprint": integer_design_fingerprint(points),
        "target_oracle_used_for_design": target_oracle_used_for_design,
        "target_outcomes_used_for_design": target_oracle_used_for_design,
        "search_records": [{
            "evaluation_index": row["evaluation_index"],
            "point_fingerprint": integer_design_fingerprint([row["point"]]),
            "observation": row["observation"].tolist(),
        } for row in search_records],
        "shortlist": shortlist,
        "verification": verification,
        "contains_true_feasible": bool(best_design is not None),
        "true_feasible_count_in_design": int(len(feasible_rows)),
        "best_true_feasible_objective": (
            None if best_design is None else float(best_design["objective"])),
        "finite_library_oracle_objective": float(oracle_best["objective"]),
        "finite_library_regret": regret,
        "feasible_and_epsilon_optimal_005": bool(
            regret is not None and regret <= 0.05),
        "penalized_loss": penalized_loss,
        "independently_certified": bool(verification["certified"]),
        "deployed_truth": deployed_truth,
        "false_certificate": bool(
            verification["certified"]
            and deployed_truth is not None
            and not deployed_truth["feasible"]),
        "oracle_role": (
            "finite_library_upper_bound"
            if target_oracle_used_for_design else "post_run_audit_only"),
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
    parser.add_argument("--out", required=True)
    parser.add_argument("--regime", choices=tuple(PROFILE_STRESS_REGIMES), required=True)
    parser.add_argument("--target-seed", type=int, required=True)
    parser.add_argument("--design-seed", type=int)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--d", type=int, default=1000)
    parser.add_argument("--active-rank", type=int)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--safe-mass", type=float, default=0.08)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--source-task-count", type=int, default=2)
    parser.add_argument("--library-size", type=int, default=64)
    parser.add_argument("--source-replications", type=int, default=3)
    parser.add_argument(
        "--schema-mode", choices=("declared", "schema_blind"), default="declared")
    parser.add_argument(
        "--descriptor-mode", choices=("conditioned", "domain_blind"),
        default="domain_blind")
    parser.add_argument("--family-seed", type=int, default=20260808)
    parser.add_argument("--verification-budgets", default="80,80,80")
    parser.add_argument("--familywise-delta", type=float, default=0.05)
    parser.add_argument("--amortization-targets", type=int, default=20)
    parser.add_argument("--atlas-max-frequency", type=int, default=8)
    parser.add_argument("--atlas-frequency-penalty", type=float, default=0.25)
    parser.add_argument(
        "--atlas-first-center-safety-weight", type=float, default=0.5)
    args = parser.parse_args()
    payload = run_task(
        regime=args.regime,
        target_seed=args.target_seed,
        arm=args.arm,
        dimension=args.d,
        active_rank=args.active_rank,
        alpha=args.alpha,
        safe_mass=args.safe_mass,
        n0=args.n0,
        source_task_count=args.source_task_count,
        library_size=args.library_size,
        source_replications=args.source_replications,
        schema_mode=args.schema_mode,
        descriptor_mode=args.descriptor_mode,
        family_seed=args.family_seed,
        design_seed=args.design_seed,
        verification_budgets=tuple(
            int(value) for value in args.verification_budgets.split(",")),
        familywise_delta=args.familywise_delta,
        amortization_targets=args.amortization_targets,
        atlas_max_frequency=args.atlas_max_frequency,
        atlas_frequency_penalty=args.atlas_frequency_penalty,
        atlas_first_center_safety_weight=(
            args.atlas_first_center_safety_weight),
    )
    _atomic_json(args.out, payload)
    print(json.dumps({
        "status": payload["status"],
        "out": str(args.out),
        "regime": payload["regime"],
        "arm": payload["arm"],
        "target_seed": payload["target_seed"],
        "contains_true_feasible": payload["contains_true_feasible"],
        "independently_certified": payload["independently_certified"],
    }, indent=2, sort_keys=True))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
