#!/usr/bin/env python3
"""Audit the frozen proposal with the rank-aligned atlas theorem."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

from baselines.transfer_archive import (
    FrozenTransferArchive,
    dimension_equivariant_profile_features,
    frozen_archive_from_meta_prior,
    resample_normalized_profiles,
)
from performance.benchmark_lodo_meta_prior import (
    build_scalarized_problem,
    train_meta_prior,
)
from performance.benchmark_sota_fairness import oracle_free_lodo_config
from performance.proposal_coverage import (
    geometric_atlas_coverage_audit,
    normalized_ranks,
    rank_aligned_atlas_coverage_audit,
    source_only_rank_alignment_calibration,
)
from performance.structural_ablation import apply_structural_prior_profile


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def audit_rank_atlas(
    archive,
    design_payload,
    problem,
    *,
    calibration_delta=0.05,
    source_support_profiles=None,
):
    archive.validate()
    designs = list(design_payload.get("designs", {}).values())
    if not designs:
        raise ValueError("proposal payload contains no designs")
    fingerprints = {
        str(row["fingerprint"]) for row in designs
    }
    if len(fingerprints) != 1:
        raise ValueError("paper proposal is not one frozen deterministic atlas")
    first_points = tuple(
        tuple(map(int, point)) for point in designs[0]["points"])
    if any(
        tuple(tuple(map(int, point)) for point in row["points"])
        != first_points
        for row in designs
    ):
        raise ValueError("equal fingerprints conceal unequal proposal points")
    atlas_normalized = np.asarray(first_points, dtype=float) / float(problem.L)
    atlas_coordinate = dimension_equivariant_profile_features(
        atlas_normalized)

    source_rank_vectors = []
    source_rows = []
    nearest_indexes = None
    for task in archive.tasks:
        source_coordinate = dimension_equivariant_profile_features(task.X)
        distances = np.linalg.norm(
            atlas_coordinate[:, None, :]
            - source_coordinate[None, :, :],
            axis=2,
        )
        indexes = np.argmin(distances, axis=1)
        if nearest_indexes is None:
            nearest_indexes = indexes
        source_margins = np.asarray(task.chance_margin(), dtype=float)[indexes]
        source_ranks = normalized_ranks(source_margins)
        source_rank_vectors.append(source_ranks)
        source_rows.append({
            "domain": task.name,
            "nearest_profile_indexes": indexes.astype(int).tolist(),
            "nearest_coordinate_distances": distances[
                np.arange(len(indexes)), indexes
            ].astype(float).tolist(),
            "observed_chance_margins": source_margins.tolist(),
            "normalized_risk_ranks": source_ranks.tolist(),
            "observed_feasible_count": int(np.sum(source_margins <= 0.0)),
            "target_oracle_used": False,
        })
    calibration = source_only_rank_alignment_calibration(
        source_rank_vectors, delta=calibration_delta)
    aggregate_source_rank = np.mean(
        np.vstack(source_rank_vectors), axis=0)

    z_alpha = float(norm.ppf(1.0 - float(problem.alpha)))
    target_margins = np.asarray([
        float(problem.true_constraint_mean(point))
        + z_alpha * float(problem.true_sigma(point)[1])
        - float(problem.tau)
        for point in first_points
    ], dtype=float)
    target_rank = normalized_ranks(target_margins)
    target_feasible = target_margins <= 0.0
    theorem = rank_aligned_atlas_coverage_audit(
        aggregate_source_rank,
        target_rank,
        target_feasible,
        range(len(first_points)),
        source_only_alignment_bound=calibration[
            "source_only_alignment_error_bound"],
    )
    empirical_theorem = rank_aligned_atlas_coverage_audit(
        aggregate_source_rank,
        target_rank,
        target_feasible,
        range(len(first_points)),
        source_only_alignment_bound=calibration[
            "empirical_pairwise_sup_rank_error"],
    )

    source_profiles = np.asarray(
        archive.tasks[0].X
        if source_support_profiles is None
        else source_support_profiles,
        dtype=float,
    )
    if any(
        source_support_profiles is None
        and not np.array_equal(
            source_profiles, np.asarray(task.X, dtype=float))
        for task in archive.tasks[1:]
    ):
        raise ValueError(
            "geometric audit requires the shared source profile library")
    mapped_source = resample_normalized_profiles(
        source_profiles, int(problem.d))
    mapped_points = [
        tuple(np.clip(
            np.rint(row * float(problem.L)), 0, problem.L
        ).astype(int).tolist())
        for row in mapped_source
    ]
    target_library = []
    target_index = {}
    for point in [*mapped_points, *first_points]:
        if point not in target_index:
            target_index[point] = len(target_library)
            target_library.append(point)
    target_library_normalized = (
        np.asarray(target_library, dtype=float) / float(problem.L)
    )
    target_library_coordinate = dimension_equivariant_profile_features(
        target_library_normalized)
    target_library_margins = np.asarray([
        float(problem.true_constraint_mean(point))
        + z_alpha * float(problem.true_sigma(point)[1])
        - float(problem.tau)
        for point in target_library
    ], dtype=float)
    geometric_source_coordinate = dimension_equivariant_profile_features(
        source_profiles)
    geometric_support_contract = (
        "full_shared_source_profile_library"
        if source_support_profiles is None
        else "frozen_source_consensus_plus_universal_seed"
    )
    if source_support_profiles is not None:
        # risk_objective_initial_candidates always places the frozen,
        # target-label-free universal library seed before source templates.
        geometric_source_coordinate = np.vstack([
            geometric_source_coordinate,
            atlas_coordinate[0],
        ])
    geometric = geometric_atlas_coverage_audit(
        atlas_coordinate,
        geometric_source_coordinate,
        target_library_coordinate,
        target_library_margins <= 0.0,
        atlas_target_indices=[
            target_index[point] for point in first_points
        ],
        target_margin=target_library_margins,
    )
    return {
        "schema_version": 1,
        "status": "complete",
        "heldout_target_domain": type(problem).__name__,
        "target_dimension": int(problem.d),
        "n0": len(first_points),
        "unique_design_fingerprint_count": len(fingerprints),
        "source_archive_fingerprint": archive.fingerprint,
        "source_only_calibration": calibration,
        "source_rows": source_rows,
        "post_run_target_audit": {
            "true_chance_margins": target_margins.tolist(),
            "normalized_risk_ranks": target_rank.tolist(),
            "true_feasible_count": int(np.sum(target_feasible)),
            "target_truth_used_post_run_only": True,
            "target_truth_used_for_proposal_or_selection": False,
        },
        "finite_sample_rank_theorem_audit": theorem,
        "empirical_rank_diagnostic_without_dkw": empirical_theorem,
        "geometric_atlas_theorem_audit": geometric,
        "geometric_source_support": geometric_support_contract,
        "universal_seed_in_geometric_support": bool(
            source_support_profiles is not None),
    }


def reconstruct_source_template_support(
    manifest,
    archive,
    design_payload,
    heldout,
):
    config = oracle_free_lodo_config(manifest)
    config["d"] = int(design_payload["dimension"])
    source_config = dict(config)
    source_config["d"] = int(design_payload["source_dimension"])
    source_config["meta_source_dimension"] = int(
        design_payload["source_dimension"])
    source_config["meta_source_design_mode"] = str(
        design_payload["source_design_mode"])
    apply_structural_prior_profile(
        source_config,
        str(design_payload["structural_prior_profile"]),
    )
    prior = train_meta_prior(
        source_config, heldout, 0, teacher=False)
    reconstructed = frozen_archive_from_meta_prior(prior, source_seed=0)
    if reconstructed.fingerprint != archive.fingerprint:
        raise ValueError(
            "reconstructed prior does not match the frozen source archive")
    profiles = np.asarray([
        item["profile"] for item in prior.source_consensus_templates
    ], dtype=float)
    if profiles.ndim != 2 or len(profiles) == 0:
        raise ValueError("frozen prior has no consensus template support")
    return profiles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--design", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--d", type=int, required=True)
    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=0.04)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--calibration-delta", type=float, default=0.05)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    archive = FrozenTransferArchive.load(args.archive)
    design = json.loads(Path(args.design).read_text(encoding="utf-8"))
    source_support_profiles = (
        None
        if args.manifest is None
        else reconstruct_source_template_support(
            args.manifest,
            archive,
            design,
            args.heldout,
        )
    )
    problem = build_scalarized_problem(
        args.heldout,
        args.d,
        args.L,
        args.sigma,
        args.alpha,
        (0.5, 0.5),
    )
    payload = audit_rank_atlas(
        archive,
        design,
        problem,
        calibration_delta=args.calibration_delta,
        source_support_profiles=source_support_profiles,
    )
    payload["heldout_target_domain"] = str(args.heldout)
    _atomic_json(args.out, payload)
    print(json.dumps({
        "status": payload["status"],
        "heldout": args.heldout,
        "finite_sample_theorem_conditions_hold": payload[
            "finite_sample_rank_theorem_audit"
        ]["theorem_conditions_hold"],
        "observed_atlas_contains_feasible": payload[
            "geometric_atlas_theorem_audit"
        ]["observed_atlas_contains_feasible"],
        "geometric_finite_library_conditions_hold": payload[
            "geometric_atlas_theorem_audit"
        ]["finite_library_theorem_conditions_hold"],
        "out": args.out,
    }, indent=2))


if __name__ == "__main__":
    main()
