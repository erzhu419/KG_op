#!/usr/bin/env python3
"""Postdecision diagnostics requested by the second OR review.

The frozen confirmatory matrices are not reinterpreted or rerun here.  This
module reconstructs their deterministic initial designs to explain the
aligned-low-frequency reversal, audits the declared task-seed strata, and
computes outcome-adjusted simulator-call efficiency from compact frozen
summaries.  Hidden target geometry is accessed only after each design is
fixed, so the resulting geometry is an oracle audit rather than an algorithm
input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.designs import (  # noqa: E402
    common_sobol_integer_design,
    integer_design_fingerprint,
)
from core.profile_atlas import (  # noqa: E402
    ProfileAtlasConfig,
    SourceScoredProfileAtlas,
    generic_dct_maximin,
    profile_cosine_coordinate,
)
from performance.benchmark_profile_stress_suite import _stable_seed  # noqa: E402
from performance.run_profile_stress_matrix import (  # noqa: E402
    derived_design_seed,
    derived_target_seed,
)
from problems.randomized_profiles import (  # noqa: E402
    PROFILE_STRESS_REGIMES,
    RandomizedOrderedProfileProblem,
    generate_structural_profile_library,
    source_profile_records,
)


CONTRACT_ID = "or_review_v2_supplemental_diagnostics_v1"
METHOD_FREEZE_COMMIT = "da8f1e5c594dace1cc667a2e4b87956b1001b67b"
FAMILY_SEED = 20260808
ALIGNED_REGIME = "aligned_low_frequency"
GEOMETRY_ARMS = (
    "source_atlas",
    "generic_dct_maximin",
    "raw_sobol",
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _summary(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("diagnostic summaries require finite nonempty vectors")
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "q10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "q90": float(np.quantile(values, 0.90)),
    }


def _profile_coordinate_standardization(library):
    matrix = np.vstack([
        profile_cosine_coordinate(
            profile.values,
            nodes=profile.nodes,
            max_frequency=8,
            frequency_penalty=0.25,
            include_diagonal_quadratic=True,
        )
        for profile in library
    ])
    scale = np.std(matrix, axis=0)
    scale = np.where(scale > 1e-10, scale, 1.0)
    return np.mean(matrix, axis=0), scale


def _aligned_structural_designs():
    regime_seed = FAMILY_SEED + _stable_seed(ALIGNED_REGIME)
    library = generate_structural_profile_library(
        64,
        dimension=128,
        seed=FAMILY_SEED + 991,
        maximum_frequency=40,
    )
    sources = tuple(
        RandomizedOrderedProfileProblem(
            regime=ALIGNED_REGIME,
            role="source",
            task_seed=10_000 + index,
            family_seed=regime_seed,
            d=256,
            alpha=0.05,
            safe_mass=0.08,
        )
        for index in range(2)
    )
    records = source_profile_records(
        sources,
        library,
        replications=3,
        seed=FAMILY_SEED + 1237,
    )
    selection = SourceScoredProfileAtlas(ProfileAtlasConfig(
        n0=10,
        max_frequency=8,
        frequency_penalty=0.25,
        include_diagonal_quadratic=True,
        safety_metric_weight=1.0,
        objective_metric_weight=1.0,
        first_center_safety_weight=0.5,
        descriptor_temperature=1.0,
    )).fit(records).selected()
    by_id = {profile.profile_id: profile for profile in library}
    source_profiles = tuple(
        by_id[member.profile_id] for member in selection.members
    )
    generic_indices, generic_diagnostics = generic_dct_maximin(
        [profile.values for profile in library],
        10,
        nodes=library[0].nodes,
        max_frequency=8,
        frequency_penalty=0.25,
        include_diagonal_quadratic=True,
    )
    generic_profiles = tuple(library[index] for index in generic_indices)
    mean, scale = _profile_coordinate_standardization(library)
    return {
        "library": library,
        "source_profiles": source_profiles,
        "generic_profiles": generic_profiles,
        "coordinate_mean": mean,
        "coordinate_scale": scale,
        "source_selection": selection,
        "generic_diagnostics": generic_diagnostics,
        "regime_seed": regime_seed,
    }


def _audit_point(target, point, *, coordinate_mean, coordinate_scale):
    semantic = target.semantic_profile(point)
    safe_profile = np.asarray(target._safe_profile, dtype=float)  # noqa: SLF001
    linear = profile_cosine_coordinate(
        semantic,
        nodes=target.nodes,
        max_frequency=8,
        frequency_penalty=0.0,
        include_diagonal_quadratic=False,
    )
    structural = profile_cosine_coordinate(
        semantic,
        nodes=target.nodes,
        max_frequency=8,
        frequency_penalty=0.25,
        include_diagonal_quadratic=True,
    )
    safe_structural = profile_cosine_coordinate(
        safe_profile,
        nodes=target.nodes,
        max_frequency=8,
        frequency_penalty=0.25,
        include_diagonal_quadratic=True,
    )
    standardized = (structural - coordinate_mean) / coordinate_scale
    safe_standardized = (safe_structural - coordinate_mean) / coordinate_scale
    margin = float(target.true_chance_margin(point))
    return {
        "latent_safe_center_distance": float(margin + target.safe_radius),
        "true_chance_margin": margin,
        "method_structural_z_distance": float(np.linalg.norm(
            standardized - safe_standardized
        )),
        "dc_coefficient": float(linear[0]),
        "non_dc_low_frequency_energy": float(np.linalg.norm(linear[1:])),
        "true_feasible": bool(margin <= 0.0),
    }


def build_aligned_geometry_audit(
    *,
    dimensions=(200, 1000, 10000),
    task_count=20,
    freeze_commit=METHOD_FREEZE_COMMIT,
):
    frozen = _aligned_structural_designs()
    point_rows = []
    design_rows = []
    fingerprint_rows = []
    for dimension in map(int, dimensions):
        for replicate in range(int(task_count)):
            target_seed = derived_target_seed(
                freeze_commit, ALIGNED_REGIME, replicate)
            design_seed = derived_design_seed(
                freeze_commit, ALIGNED_REGIME, replicate)
            target = RandomizedOrderedProfileProblem(
                regime=ALIGNED_REGIME,
                role="target",
                task_seed=target_seed,
                family_seed=frozen["regime_seed"],
                d=dimension,
                alpha=0.05,
                safe_mass=0.08,
            )
            designs = {
                "source_atlas": tuple(
                    target.point_from_structural_profile(profile)
                    for profile in frozen["source_profiles"]
                ),
                "generic_dct_maximin": tuple(
                    target.point_from_structural_profile(profile)
                    for profile in frozen["generic_profiles"]
                ),
                "raw_sobol": tuple(common_sobol_integer_design(
                    target, 10, design_seed, seed_offset=19_873
                )),
            }
            for arm, points in designs.items():
                metrics = [
                    _audit_point(
                        target,
                        point,
                        coordinate_mean=frozen["coordinate_mean"],
                        coordinate_scale=frozen["coordinate_scale"],
                    )
                    for point in points
                ]
                point_rows.extend({
                    "arm": arm,
                    "dimension": dimension,
                    "replicate_index": replicate,
                    **row,
                } for row in metrics)
                design_rows.append({
                    "arm": arm,
                    "dimension": dimension,
                    "replicate_index": replicate,
                    "minimum_latent_safe_center_distance": float(min(
                        row["latent_safe_center_distance"] for row in metrics
                    )),
                    "minimum_true_chance_margin": float(min(
                        row["true_chance_margin"] for row in metrics
                    )),
                    "true_feasible_point_count": int(sum(
                        row["true_feasible"] for row in metrics
                    )),
                })
                fingerprint_rows.append((
                    arm,
                    dimension,
                    replicate,
                    integer_design_fingerprint(points),
                ))

    summaries = []
    design_summaries = []
    for arm in GEOMETRY_ARMS:
        selected_points = [row for row in point_rows if row["arm"] == arm]
        selected_designs = [row for row in design_rows if row["arm"] == arm]
        summaries.append({
            "arm": arm,
            "point_count": len(selected_points),
            "latent_safe_center_distance": _summary([
                row["latent_safe_center_distance"] for row in selected_points
            ]),
            "true_chance_margin": _summary([
                row["true_chance_margin"] for row in selected_points
            ]),
            "method_structural_z_distance": _summary([
                row["method_structural_z_distance"] for row in selected_points
            ]),
            "dc_coefficient": _summary([
                row["dc_coefficient"] for row in selected_points
            ]),
            "non_dc_low_frequency_energy": _summary([
                row["non_dc_low_frequency_energy"] for row in selected_points
            ]),
            "true_feasible_point_rate": float(np.mean([
                row["true_feasible"] for row in selected_points
            ])),
        })
        design_summaries.append({
            "arm": arm,
            "task_resolution_cell_count": len(selected_designs),
            "design_contains_true_feasible_rate": float(np.mean([
                row["true_feasible_point_count"] > 0
                for row in selected_designs
            ])),
            "true_feasible_points_per_design": _summary([
                row["true_feasible_point_count"] for row in selected_designs
            ]),
            "minimum_latent_safe_center_distance": _summary([
                row["minimum_latent_safe_center_distance"]
                for row in selected_designs
            ]),
        })

    fingerprint_digest = hashlib.sha256()
    for row in sorted(fingerprint_rows):
        fingerprint_digest.update(":".join(map(str, row)).encode("utf-8"))
        fingerprint_digest.update(b"\n")
    indexed = {row["arm"]: row for row in summaries}
    return {
        "regime": ALIGNED_REGIME,
        "method_freeze_commit": freeze_commit,
        "dimensions": list(map(int, dimensions)),
        "independent_latent_task_seed_count": int(task_count),
        "task_resolution_cell_count": int(len(dimensions) * int(task_count)),
        "points_per_design": 10,
        "design_fingerprint_root_sha256": fingerprint_digest.hexdigest(),
        "source_selected_profile_ids": list(
            frozen["source_selection"].diagnostics["selected_profile_ids"]
        ),
        "generic_selected_profile_ids": [
            frozen["library"][index].profile_id
            for index in frozen["generic_diagnostics"]["selected_indices"]
        ],
        "point_summaries": summaries,
        "design_summaries": design_summaries,
        "mechanism_check": {
            "raw_sobol_median_latent_distance_below_source": bool(
                indexed["raw_sobol"]["latent_safe_center_distance"]["median"]
                < indexed["source_atlas"]["latent_safe_center_distance"]["median"]
            ),
            "raw_sobol_median_non_dc_energy_below_source": bool(
                indexed["raw_sobol"]["non_dc_low_frequency_energy"]["median"]
                < indexed["source_atlas"]["non_dc_low_frequency_energy"]["median"]
            ),
            "interpretation": (
                "Raw high-dimensional Sobol profiles average toward DC=0.5 "
                "with weak non-DC low-frequency energy, placing many points "
                "near the aligned target safe center."
            ),
        },
        "oracle_audit_contract": {
            "target_hidden_center_used_by_algorithm": False,
            "target_hidden_center_used_after_design_freeze": True,
            "diagnostic_changes_primary_outcomes": False,
            "diagnostic_is_explanatory_not_inferential": True,
        },
    }


def build_task_seed_strata_audit(
    *,
    task_count=20,
    freeze_commit=METHOD_FREEZE_COMMIT,
):
    rows = []
    overall = [0] * 5
    for regime in PROFILE_STRESS_REGIMES:
        counts = [0] * 5
        seeds = []
        for replicate in range(int(task_count)):
            seed = derived_target_seed(freeze_commit, regime, replicate)
            seeds.append(seed)
            counts[seed % 5] += 1
            overall[seed % 5] += 1
        rows.append({
            "regime": regime,
            "independent_seed_count": len(seeds),
            "unique_seed_count": len(set(seeds)),
            "task_seed_mod_5_counts": counts,
        })
    return {
        "method_freeze_commit": freeze_commit,
        "replicates_per_regime": int(task_count),
        "regime_count": len(PROFILE_STRESS_REGIMES),
        "rows": rows,
        "overall_task_seed_mod_5_counts": overall,
        "inference_contract": {
            "regimes_are_fixed_equal_weight_strata": True,
            "task_seeds_are_independent_hash_derived_streams_within_stratum": True,
            "task_seed_mod_5_is_an_induced_latent_category": True,
            "same_latent_task_seeds_are_crossed_with_three_resolutions": True,
            "task_law_bound_applied_separately_by_resolution": True,
            "pooled_480_cell_rates_are_descriptive": True,
        },
    }


def _aggregate_by_arm(analysis):
    rows = analysis["aggregate_analysis"]["summaries"]
    output = {}
    for arm in sorted({row["arm"] for row in rows}):
        selected = [row for row in rows if row["arm"] == arm]
        task_count = sum(int(row["independent_task_count"]) for row in selected)

        def weighted(field):
            return sum(
                float(row[field]) * int(row["independent_task_count"])
                for row in selected
            ) / task_count

        output[arm] = {
            "task_count": task_count,
            "certified_success_count": sum(int(
                row["certified_true_feasible_deployment_count"]
            ) for row in selected),
            "certified_success_probability": sum(int(
                row["certified_true_feasible_deployment_count"]
            ) for row in selected) / task_count,
            "mean_all_in_calls_unamortized": weighted(
                "mean_all_in_calls_unamortized"),
            "mean_verification_calls": weighted("mean_verification_calls"),
            "target_search_calls": int(selected[0]["N"]),
        }
    return output


def _efficiency_rows(aggregates, *, amortization_targets=(1, 20)):
    rows = []
    for arm, aggregate in aggregates.items():
        source_calls = 384 if arm == "source_atlas" else 0
        target_and_verification = (
            aggregate["mean_all_in_calls_unamortized"] - source_calls
        )
        probability = aggregate["certified_success_probability"]
        per_target = {}
        for deployments in amortization_targets:
            calls = target_and_verification + source_calls / int(deployments)
            per_target[str(int(deployments))] = {
                "mean_calls_per_target": float(calls),
                "expected_calls_per_certified_success": (
                    None if probability <= 0.0 else float(calls / probability)
                ),
            }
        rows.append({
            "arm": arm,
            "task_count": aggregate["task_count"],
            "certified_success_count": aggregate["certified_success_count"],
            "certified_success_probability": probability,
            "source_archive_calls": source_calls,
            "mean_target_search_plus_verification_calls": float(
                target_and_verification),
            "amortization": per_target,
        })
    return rows


def _outcome_adjusted_break_even(source, control):
    p_source = source["certified_success_probability"]
    p_control = control["certified_success_probability"]
    if p_source <= 0.0 or p_control <= 0.0:
        return None
    source_operational = source["mean_target_search_plus_verification_calls"]
    control_operational = control[
        "mean_target_search_plus_verification_calls"
    ]
    denominator = p_source * control_operational / p_control - source_operational
    if denominator <= 0.0:
        return None
    return int(math.ceil(source["source_archive_calls"] / denominator))


def build_outcome_adjusted_cost_audit(primary, equal_preverification):
    matrices = {}
    for name, payload in (
        ("primary_n10", primary),
        ("equal_preverification", equal_preverification),
    ):
        rows = _efficiency_rows(_aggregate_by_arm(payload))
        indexed = {row["arm"]: row for row in rows}
        source = indexed["source_atlas"]
        break_even = {
            arm: _outcome_adjusted_break_even(source, control)
            for arm, control in indexed.items()
            if arm != "source_atlas"
        }
        matrices[name] = {
            "rows": rows,
            "source_outcome_adjusted_break_even_deployments": break_even,
        }
    return {
        "metric": (
            "(source_archive_calls / deployments + target_search_calls + "
            "mean_verification_calls) / certified_success_probability"
        ),
        "matrices": matrices,
        "operational_loss_template": (
            "simulator_calls + c_unsafe*false_deployment + "
            "c_abstain*abstention + c_quality*regret"
        ),
        "interpretation": (
            "Call-only and outcome-adjusted break-even are distinct; no "
            "single loss ranking is asserted without decision-maker costs."
        ),
    }


def build_diagnostics(
    *,
    primary_path,
    equal_preverification_path,
    base_registry_path,
    dimensions=(200, 1000, 10000),
    task_count=20,
):
    primary_path = Path(primary_path)
    equal_preverification_path = Path(equal_preverification_path)
    base_registry_path = Path(base_registry_path)
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    equal = json.loads(equal_preverification_path.read_text(encoding="utf-8"))
    failures = []
    for name, payload in (("primary", primary), ("equal", equal)):
        if payload.get("status") != "complete":
            failures.append(f"{name} compact analysis is not complete")
    aggregate = {
        "aligned_geometry": build_aligned_geometry_audit(
            dimensions=dimensions,
            task_count=task_count,
        ),
        "task_seed_strata": build_task_seed_strata_audit(
            task_count=task_count,
        ),
        "outcome_adjusted_cost": build_outcome_adjusted_cost_audit(
            primary, equal
        ),
        "source_archive_scope": {
            "source_tasks_are_regime_matched_in_synthetic_stress": True,
            "source_archive_is_supplied_exogenously": True,
            "method_ranks_profiles_within_supplied_archive": True,
            "method_solves_source_retrieval": False,
            "energy_region_holdout_is_archive_mismatch_control": True,
        },
    }
    return {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "status": "complete" if not failures else "incomplete",
        "failures": failures,
        "algorithmic_failure_count": 0,
        "source_analysis": {
            "contract_id": "or_review_v2_postdecision_reconstruction",
            "base_evidence_registry_sha256": _sha256(base_registry_path),
            "inputs": [
                {"path": primary_path.name, "sha256": _sha256(primary_path)},
                {
                    "path": equal_preverification_path.name,
                    "sha256": _sha256(equal_preverification_path),
                },
                {
                    "path": base_registry_path.name,
                    "sha256": _sha256(base_registry_path),
                },
            ],
        },
        "aggregate_analysis": aggregate,
        "interpretation_contract": {
            "frozen_primary_outcomes_unchanged": True,
            "hidden_geometry_used_postdecision_only": True,
            "outcome_adjusted_cost_uses_frozen_aggregate_counts": True,
            "task_seed_strata_are_reported_not_rebalanced": True,
            "source_retrieval_is_out_of_scope": True,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    evidence = ROOT / "paper_artifacts/or_review"
    parser.add_argument(
        "--primary",
        default=str(evidence / "randomized_profile_primary.json"),
    )
    parser.add_argument(
        "--equal-preverification",
        default=str(evidence / "randomized_profile_equal_preverification.json"),
    )
    parser.add_argument(
        "--base-registry",
        default=str(evidence / "final_evidence_registry_v1.json"),
    )
    parser.add_argument(
        "--out",
        default=str(evidence / "review_v2_supplemental_diagnostics.json"),
    )
    args = parser.parse_args()
    payload = build_diagnostics(
        primary_path=args.primary,
        equal_preverification_path=args.equal_preverification,
        base_registry_path=args.base_registry,
    )
    _atomic_json(args.out, payload)
    print(json.dumps({
        "status": payload["status"],
        "failure_count": len(payload["failures"]),
        "out": str(args.out),
    }, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["status"] == "complete" else 2)


if __name__ == "__main__":
    main()
