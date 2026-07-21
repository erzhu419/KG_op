"""Parallel LODO benchmark for learned admissible SC/HVD meta-priors."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algorithms.single_olhkg import SingleOLHKGAlgorithm, SingleOLHKGConfig  # noqa: E402
from core.admissibility import domain_tuned_audit  # noqa: E402
from core.designs import common_sobol_integer_design  # noqa: E402
from performance.benchmark_quality import json_safe, parse_csv, parse_weights, write_csv  # noqa: E402
from problems.rzdt import make_problem  # noqa: E402
from problems.single_objective import ScalarizedProblem  # noqa: E402
from representation.meta_prior import (  # noqa: E402
    AdmissibleProblemAdapter,
    LearnedMetaPrior,
    MetaPriorProblemAdapter,
)


def finite_stats(values):
    vals = []
    for value in values:
        if value is None:
            continue
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(val):
            vals.append(val)
    if not vals:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(vals),
        "mean": float(statistics.fmean(vals)),
        "median": float(statistics.median(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def mean_bool(values):
    vals = [bool(v) for v in values]
    return float(sum(vals) / len(vals)) if vals else None


def build_scalarized_problem(
    name, d, L, sigma, alpha, weights, *, problem_kwargs=None,
):
    return ScalarizedProblem(
        make_problem(
            name,
            d=d,
            L=L,
            sigma=sigma,
            alpha=alpha,
            **dict(problem_kwargs or {}),
        ),
        weights=weights,
    )


def _source_episode_record_counts(total_records, n_augments, budget_mode):
    total = int(total_records)
    count = max(1, int(n_augments))
    mode = str(budget_mode).strip().lower()
    if total <= 0:
        raise ValueError("source_records_per_domain must be positive")
    if mode == "per_episode":
        return [total] * count
    if mode != "per_base_domain":
        raise ValueError(
            "meta_source_budget_mode must be per_episode or per_base_domain")
    if total < count:
        raise ValueError(
            "per_base_domain budget must provide at least one record per episode")
    quotient, remainder = divmod(total, count)
    return [
        quotient + (1 if index < remainder else 0)
        for index in range(count)
    ]


def _source_augmented_problem_specs(args_dict, source_names, seed):
    rng = np.random.default_rng(int(args_dict["meta_seed"]) + 4243 * int(seed))
    n_aug = max(1, int(args_dict.get("meta_source_augments", 1)))
    base_sigma = float(args_dict["sigma"])
    base_alpha = float(args_dict["alpha"])
    base_weights = parse_weights(args_dict["weights"])
    sigma_jitter = max(float(args_dict.get("meta_source_sigma_jitter", 0.0)), 0.0)
    alpha_jitter = max(float(args_dict.get("meta_source_alpha_jitter", 0.0)), 0.0)
    weight_jitter = max(float(args_dict.get("meta_source_weight_jitter", 0.0)), 0.0)
    geometry_shift_scale = max(float(args_dict.get(
        "meta_source_geometry_shift_scale", 0.0)), 0.0)
    geometry_radius_jitter = max(float(args_dict.get(
        "meta_source_geometry_log_radius_jitter", 0.0)), 0.0)
    budget_mode = str(args_dict.get(
        "meta_source_budget_mode", "per_episode")).lower()
    record_counts = _source_episode_record_counts(
        args_dict["source_records_per_domain"], n_aug, budget_mode)
    geometry_requested = bool(
        geometry_shift_scale > 0.0 or geometry_radius_jitter > 0.0)
    geometry_domains = {
        "FactorShockStatePolicyRZDT1",
        "InventorySupplyChain",
        "QueueResourceControl",
    }
    specs = []
    for name in source_names:
        if geometry_requested and name not in geometry_domains:
            raise ValueError(
                f"source task geometry is unavailable for {name}")
        for aug_idx in range(n_aug):
            if aug_idx == 0:
                sigma = base_sigma
                alpha = base_alpha
                weights = base_weights
                label = name
                shift = np.zeros(3, dtype=float)
                radius_scale = 1.0
            else:
                sigma = base_sigma * float(np.exp(rng.normal(0.0, sigma_jitter)))
                alpha = base_alpha * float(np.exp(rng.normal(0.0, alpha_jitter)))
                alpha = float(np.clip(alpha, 0.01, 0.20))
                if weight_jitter > 0.0:
                    weights = np.maximum(
                        base_weights + rng.normal(0.0, weight_jitter, size=len(base_weights)),
                        1e-4,
                    )
                    weights = weights / max(float(np.sum(weights)), 1e-12)
                else:
                    weights = base_weights
                label = f"{name}#episode{aug_idx}"
                shift = rng.normal(
                    0.0, geometry_shift_scale, size=3)
                radius_scale = float(np.exp(rng.normal(
                    0.0, geometry_radius_jitter)))
            problem_kwargs = {}
            if geometry_requested:
                problem_kwargs = {
                    "task_geometry_shift": np.asarray(
                        shift, dtype=float).tolist(),
                    "task_geometry_radius_scale": float(radius_scale),
                }
            specs.append({
                "label": label,
                "base_domain": name,
                "sigma": float(sigma),
                "alpha": float(alpha),
                "weights": np.asarray(weights, dtype=float),
                "problem_kwargs": problem_kwargs,
                "record_count": int(record_counts[aug_idx]),
                "episode_index": int(aug_idx),
            })
    return specs


def meta_source_seed(args_dict, target_seed):
    mode = str(args_dict.get("meta_source_seed_mode", "frozen")).lower()
    if mode == "frozen":
        return 0
    if mode == "per_target":
        return int(target_seed)
    raise ValueError(
        "meta_source_seed_mode must be 'frozen' or 'per_target'")


def train_meta_prior(args_dict, heldout, seed, *, teacher=False):
    domains = parse_csv(args_dict["domains"])
    source_names = [name for name in domains if name != heldout]
    if not source_names:
        raise ValueError(f"heldout={heldout} leaves no source domains")
    source_seed = meta_source_seed(args_dict, seed)
    source_dimension = int(args_dict.get(
        "meta_source_dimension", args_dict["d"]))
    source_specs = _source_augmented_problem_specs(
        args_dict,
        source_names,
        source_seed,
    )
    source_problems = [
        (
            spec["label"],
            build_scalarized_problem(
                spec["base_domain"],
                source_dimension,
                args_dict["L"],
                spec["sigma"],
                spec["alpha"],
                spec["weights"],
                problem_kwargs=spec["problem_kwargs"],
            ),
        )
        for spec in source_specs
    ]
    prior = LearnedMetaPrior(
        local_dim=args_dict["meta_local_dim"],
        shared_dim=args_dict["meta_shared_dim"],
        anchor_count=args_dict["meta_anchor_count"],
        kmeans_iters=args_dict["meta_kmeans_iters"],
        soft_temperature=args_dict["meta_soft_temperature"],
        ridge=args_dict["meta_ridge"],
        boundary_weight=args_dict["meta_boundary_weight"],
        boundary_temperature=args_dict["meta_boundary_temperature"],
        variance_weight=args_dict["meta_variance_weight"],
        feasible_penalty=args_dict["meta_feasible_penalty"],
        feasible_bonus=args_dict["meta_feasible_bonus"],
        elite_fraction=args_dict["meta_elite_fraction"],
        boundary_fraction=args_dict["meta_boundary_fraction"],
        teacher_records_per_domain=(
            args_dict["meta_teacher_records_per_domain"] if teacher else 0
        ),
        teacher_weight=args_dict["meta_teacher_weight"],
        teacher_pool_size=args_dict["meta_teacher_pool_size"],
        teacher_elite_fraction=args_dict["meta_teacher_elite_fraction"],
        teacher_boundary_fraction=args_dict["meta_teacher_boundary_fraction"],
        anchor_sampling_temperature=(
            args_dict["meta_teacher_anchor_sampling_temperature"]
            if teacher
            else args_dict["meta_anchor_sampling_temperature"]
        ),
        hvd_noise_floor_scale=(
            args_dict["meta_teacher_hvd_noise_floor_scale"]
            if teacher
            else args_dict["meta_hvd_noise_floor_scale"]
        ),
        universal_shape_count=args_dict["meta_universal_shape_count"],
        component_stage=args_dict["meta_component_stage"],
        spectral_active_dim=args_dict["meta_spectral_active_dim"],
        spectral_max_library_size=args_dict["meta_spectral_max_library_size"],
        spectral_low_frequency_components=args_dict[
            "meta_spectral_low_frequency_components"],
        spectral_low_frequency_prior=bool(args_dict.get(
            "meta_spectral_low_frequency_prior", True)),
        spectral_graph_neighbors=args_dict["meta_spectral_graph_neighbors"],
        spectral_orthogonalization=str(args_dict.get(
            "meta_spectral_orthogonalization", "symmetric")),
        spectral_relevance_floor=args_dict["meta_spectral_relevance_floor"],
        spectral_gate_boundary_weight=args_dict[
            "meta_spectral_gate_boundary_weight"],
        spectral_gate_dangerous_weight=args_dict[
            "meta_spectral_gate_dangerous_weight"],
        spectral_gate_selection_tolerance=args_dict[
            "meta_spectral_gate_selection_tolerance"],
        spectral_gate_calibration_quantile=args_dict[
            "meta_spectral_gate_calibration_quantile"],
        spectral_frequency_adaptation=args_dict[
            "meta_spectral_frequency_adaptation"],
        spectral_frequency_cutoffs=args_dict[
            "meta_spectral_frequency_cutoffs"],
        spectral_frequency_ridges=args_dict[
            "meta_spectral_frequency_ridges"],
        spectral_frequency_source_penalty=args_dict[
            "meta_spectral_frequency_source_penalty"],
        spectral_frequency_temperature=args_dict[
            "meta_spectral_frequency_temperature"],
        spectral_frequency_refit_interval=args_dict[
            "meta_spectral_frequency_refit_interval"],
        spectral_risk_alignment=args_dict[
            "meta_spectral_risk_alignment"],
        spectral_alignment_active_dim=args_dict[
            "meta_spectral_alignment_active_dim"],
        spectral_alignment_subspace_dim=args_dict[
            "meta_spectral_alignment_subspace_dim"],
        spectral_alignment_domain_penalty=args_dict[
            "meta_spectral_alignment_domain_penalty"],
        spectral_alignment_source_procrustes=args_dict[
            "meta_spectral_alignment_source_procrustes"],
        spectral_alignment_target_ridge=args_dict[
            "meta_spectral_alignment_target_ridge"],
        spectral_alignment_target_min_gain=args_dict[
            "meta_spectral_alignment_target_min_gain"],
        spectral_alignment_target_min_bins=args_dict[
            "meta_spectral_alignment_target_min_bins"],
        spectral_alignment_refit_interval=args_dict[
            "meta_spectral_alignment_refit_interval"],
        spectral_alignment_source_episodes=args_dict[
            "meta_spectral_alignment_source_episodes"],
        spectral_alignment_admission=args_dict[
            "meta_spectral_alignment_admission"],
        spectral_alignment_latent_proposals=args_dict[
            "meta_spectral_alignment_latent_proposals"],
        spectral_alignment_inverse_pool_size=args_dict[
            "meta_spectral_alignment_inverse_pool_size"],
        spectral_alignment_episode_pilot_size=args_dict[
            "meta_spectral_alignment_episode_pilot_size"],
        spectral_alignment_episode_evaluation_size=args_dict[
            "meta_spectral_alignment_episode_evaluation_size"],
        spectral_alignment_episode_ridge=args_dict[
            "meta_spectral_alignment_episode_ridge"],
        spectral_additive_adaptation=args_dict[
            "meta_spectral_additive_adaptation"],
        spectral_additive_max_groups=args_dict[
            "meta_spectral_additive_max_groups"],
        spectral_additive_target_max_groups=args_dict[
            "meta_spectral_additive_target_max_groups"],
        spectral_additive_source_penalty=args_dict[
            "meta_spectral_additive_source_penalty"],
        spectral_additive_complexity_penalty=args_dict[
            "meta_spectral_additive_complexity_penalty"],
        spectral_additive_temperature=args_dict[
            "meta_spectral_additive_temperature"],
        spectral_additive_refit_interval=args_dict[
            "meta_spectral_additive_refit_interval"],
        spectral_additive_max_saturation_fraction=args_dict[
            "meta_spectral_additive_max_saturation_fraction"],
        spectral_coefficient_shrinkage=args_dict[
            "meta_spectral_coefficient_shrinkage"],
        spectral_shrinkage_strength=args_dict[
            "meta_spectral_shrinkage_strength"],
        spectral_shrinkage_floor=args_dict[
            "meta_spectral_shrinkage_floor"],
        spectral_adaptive_sparsity=args_dict[
            "meta_spectral_adaptive_sparsity"],
        spectral_adaptive_min_pip=args_dict[
            "meta_spectral_adaptive_min_pip"],
        spectral_adaptive_max_pip=args_dict[
            "meta_spectral_adaptive_max_pip"],
        spectral_adaptive_spike_ratio=args_dict[
            "meta_spectral_adaptive_spike_ratio"],
        spectral_adaptive_damping=args_dict[
            "meta_spectral_adaptive_damping"],
        spectral_adaptive_max_iter=args_dict[
            "meta_spectral_adaptive_max_iter"],
        spectral_adaptive_tolerance=args_dict[
            "meta_spectral_adaptive_tolerance"],
        spectral_adaptive_residual_floor_scale=args_dict[
            "meta_spectral_adaptive_residual_floor_scale"],
        spectral_adaptive_gate_tolerance=args_dict[
            "meta_spectral_adaptive_gate_tolerance"],
        spectral_adaptive_multiplicity_correction=args_dict[
            "meta_spectral_adaptive_multiplicity_correction"],
        spectral_adaptive_max_effective_fraction=args_dict[
            "meta_spectral_adaptive_max_effective_fraction"],
        spectral_adaptive_saturation_fraction=args_dict[
            "meta_spectral_adaptive_saturation_fraction"],
        ordered_cumulative_exposure=bool(args_dict.get(
            "meta_ordered_cumulative_exposure", False)),
        ordered_exposure_max_frequency=int(args_dict.get(
            "meta_ordered_exposure_max_frequency", 8)),
        ordered_exposure_active_dim=int(args_dict.get(
            "meta_ordered_exposure_active_dim", 2)),
        ordered_exposure_frequency_penalty=float(args_dict.get(
            "meta_ordered_exposure_frequency_penalty", 0.10)),
        ordered_exposure_basis_mode=str(args_dict.get(
            "meta_ordered_exposure_basis_mode", "full_quadratic")),
        ordered_exposure_orthogonal_coordinates=bool(args_dict.get(
            "meta_ordered_exposure_orthogonal_coordinates", True)),
        ordered_exposure_adaptive_sparsity=bool(args_dict.get(
            "meta_ordered_exposure_adaptive_sparsity", False)),
        ordered_exposure_replace_local_kernel=bool(args_dict.get(
            "meta_ordered_exposure_replace_local_kernel", False)),
        ordered_exposure_semiparametric_residual=bool(args_dict.get(
            "meta_ordered_exposure_semiparametric_residual", False)),
        ordered_exposure_latent_structure_selection=bool(args_dict.get(
            "meta_ordered_exposure_latent_structure_selection", False)),
        ordered_exposure_group_shared_shrinkage=bool(args_dict.get(
            "meta_ordered_exposure_group_shared_shrinkage", False)),
        ordered_exposure_group_ridge_learning=bool(args_dict.get(
            "meta_ordered_exposure_group_ridge_learning", False)),
        coordinate_mode=args_dict["meta_coordinate_mode"],
        coordinate_relevance_floor=args_dict["meta_coordinate_relevance_floor"],
        observable_mean_coordinate=bool(args_dict.get(
            "meta_observable_mean_coordinate", False)),
        observable_mean_ridges=args_dict.get(
            "meta_observable_mean_ridges", "0.01,0.1,1.0,10.0,100.0"),
        observable_mean_mode=str(args_dict.get(
            "meta_observable_mean_mode", "latent")),
        observable_mean_latent_dim=int(args_dict.get(
            "meta_observable_mean_latent_dim", 2)),
        observable_mean_training_target=str(args_dict.get(
            "meta_observable_mean_training_target", "constraint_mean")),
        observable_mean_input_mode=str(args_dict.get(
            "meta_observable_mean_input_mode", "policy_profile")),
        observable_mean_descriptor_mode=str(args_dict.get(
            "meta_observable_mean_descriptor_mode", "ordered")),
        observable_mean_feature_mode=str(args_dict.get(
            "meta_observable_mean_feature_mode", "linear")),
        observable_mean_latent_transform=str(args_dict.get(
            "meta_observable_mean_latent_transform", "identity")),
        observable_mean_target_residual_rank=int(args_dict.get(
            "meta_observable_mean_target_residual_rank", 0)),
        observable_mean_target_residual_prior_scale=float(args_dict.get(
            "meta_observable_mean_target_residual_prior_scale", 1.0)),
        observable_mean_target_residual_pool_size=int(args_dict.get(
            "meta_observable_mean_target_residual_pool_size", 128)),
        observable_mean_target_residual_rcond=float(args_dict.get(
            "meta_observable_mean_target_residual_rcond", 1e-8)),
        observable_mean_role_assignment_posterior=bool(args_dict.get(
            "meta_observable_mean_role_assignment_posterior", False)),
        observable_mean_role_assignment_prior=str(args_dict.get(
            "meta_observable_mean_role_assignment_prior", "uniform")),
        observable_mean_role_assignment_prior_temperature_scale=float(
            args_dict.get(
                "meta_observable_mean_role_assignment_prior_temperature_scale",
                1.0,
            )),
        observable_mean_role_assignment_inactive_variance=float(args_dict.get(
            "meta_observable_mean_role_assignment_inactive_variance", 1e-12)),
        observable_variance_input_mode=str(args_dict.get(
            "meta_observable_variance_input_mode", "legacy_policy_proxy")),
        source_observation_mode=str(args_dict.get(
            "meta_source_observation_mode", "analytic")),
        source_observation_replicates=int(args_dict.get(
            "meta_source_observation_replicates", 1)),
        source_design_mode=str(args_dict.get(
            "meta_source_design_mode", "random")),
        source_universal_fraction=float(args_dict.get(
            "meta_source_universal_fraction", 0.75)),
        source_consensus_template_count=int(args_dict.get(
            "meta_source_consensus_template_count", 0)),
        seed=int(args_dict["meta_seed"]) + int(source_seed),
    )
    prior.fit_from_source_problems(
        source_problems,
        n_records_per_domain=args_dict["source_records_per_domain"],
        n_records_by_problem=[
            spec["record_count"] for spec in source_specs
        ],
        rng=np.random.default_rng(
            int(args_dict["meta_seed"]) + 1009 * int(source_seed)),
        hierarchical_boundary_config=(
            {
                "descriptor_mode": args_dict["tcb_v2_descriptor_mode"],
                "coordinate": args_dict["tcb_v2_coordinate"],
                "geometry": args_dict["tcb_v2_geometry"],
                "rank": args_dict["tcb_v2_rank"],
                "ridge": args_dict["tcb_v2_ridge"],
                "domain_penalty": args_dict["tcb_v2_domain_penalty"],
                "boundary_temperature": args_dict[
                    "tcb_v2_boundary_temperature"],
                "adaptation_ridge": args_dict[
                    "tcb_v2_adaptation_ridge"],
                "upper_alpha": args_dict["tcb_v2_upper_alpha"],
                "calibration_prior_df": args_dict[
                    "tcb_v2_calibration_prior_df"],
                "hierarchy_iterations": args_dict[
                    "tcb_v2_hierarchy_iterations"],
                "effect_ridge": args_dict["tcb_v2_effect_ridge"],
                "rotation_mode": args_dict["tcb_v2_rotation_mode"],
                "rotation_ridge": args_dict["tcb_v2_rotation_ridge"],
                "target_residual_rank": args_dict[
                    "tcb_v2_target_residual_rank"],
                "residual_ridge": args_dict["tcb_v2_residual_ridge"],
            }
            if bool(args_dict.get("tcb_v2_enabled", False))
            else None
        ),
    )
    prior.training_diagnostics.update({
        "source_seed_mode": str(args_dict.get(
            "meta_source_seed_mode", "frozen")),
        "source_seed": int(source_seed),
        "target_seed_used_for_source_training": bool(
            str(args_dict.get(
                "meta_source_seed_mode", "frozen")).lower() == "per_target"),
        "source_policy_dimension": int(source_dimension),
        "target_policy_dimension": int(args_dict["d"]),
        "dimension_holdout": bool(source_dimension != int(args_dict["d"])),
        "source_base_domains": list(source_names),
        "source_base_domain_count": int(len(source_names)),
        "source_episode_count_per_base_domain": int(
            args_dict.get("meta_source_augments", 1)),
        "source_episode_budget_mode": str(args_dict.get(
            "meta_source_budget_mode", "per_episode")),
        "source_episode_record_budget": int(sum(
            spec["record_count"] for spec in source_specs)),
        "source_episode_cost_matched": bool(
            str(args_dict.get(
                "meta_source_budget_mode", "per_episode"
            )).lower() == "per_base_domain"
        ),
        "source_episode_specs": [
            {
                "label": spec["label"],
                "base_domain": spec["base_domain"],
                "episode_index": spec["episode_index"],
                "record_count": spec["record_count"],
                "task_geometry_shift": spec["problem_kwargs"].get(
                    "task_geometry_shift", [0.0, 0.0, 0.0]),
                "task_geometry_radius_scale": spec["problem_kwargs"].get(
                    "task_geometry_radius_scale", 1.0),
            }
            for spec in source_specs
        ],
        "source_episode_target_data_used": False,
        "source_episode_target_oracle_used": False,
    })
    if str(args_dict.get(
        "meta_source_observation_mode", "analytic"
    )).lower() == "replicated":
        from baselines.transfer_archive import frozen_archive_from_meta_prior
        archive = frozen_archive_from_meta_prior(
            prior, source_seed=source_seed)
        prior.training_diagnostics.update({
            "source_archive_fingerprint": archive.fingerprint,
            "source_archive_domains": list(archive.source_domains),
            "source_archive_profiles_per_domain": (
                archive.profiles_per_domain),
            "source_archive_simulator_calls": int(archive.simulator_calls),
            "source_archive_oracle_aided": False,
        })
    return prior


def build_target_problem(args_dict, heldout, line, seed):
    weights = parse_weights(args_dict["weights"])
    target_problem_kwargs = {}
    if heldout == "FactorShockStatePolicyRZDT1":
        target_problem_kwargs["shared_shock_scale"] = float(
            args_dict.get("target_shared_shock_scale", 1.0))
    target = build_scalarized_problem(
        heldout,
        args_dict["d"],
        args_dict["L"],
        args_dict["sigma"],
        args_dict["alpha"],
        weights,
        problem_kwargs=target_problem_kwargs,
    )
    line = str(line)
    meta_diag = None
    if line == "strict":
        return AdmissibleProblemAdapter(target, variant="strict_universal"), meta_diag
    if line in ("lodo", "lodo_teacher"):
        prior = train_meta_prior(
            args_dict,
            heldout,
            seed,
            teacher=(line == "lodo_teacher"),
        )
        meta_diag = prior.diagnostics()
        return MetaPriorProblemAdapter(
            target,
            prior,
            proposal_pool_size=args_dict["meta_proposal_pool_size"],
            refinement_count=args_dict["meta_refinement_count"],
        ), meta_diag
    if line == "domain":
        return target, meta_diag
    raise ValueError(f"unknown line {line!r}")


def post_run_variance_calibration_audit(
    algorithm, problem, *, seed, audit_size=128,
):
    """Audit aleatoric calibration on a fixed target-oracle holdout pool.

    The pool and truth are constructed only after optimization has terminated;
    neither is exposed to proposal generation, posterior updates, acquisition,
    or recommendation.
    """

    points = common_sobol_integer_design(
        problem,
        max(8, int(audit_size)),
        int(seed),
        seed_offset=880_301,
    )
    true_variance = np.asarray([
        float(problem.true_sigma(point)[1]) ** 2 for point in points
    ], dtype=float)
    if algorithm.task_ensemble is None:
        predicted = algorithm.variance_model.predict_variance_many(
            1, points, problem)
        certified = algorithm.variance_model.predict_certification_variance_many(
            1, points, problem)
        posterior_source = "single_hvd"
    else:
        _, _, per_expert = algorithm.task_ensemble.expert_moments_many(
            1, points, certification=False)
        weights = np.asarray(
            algorithm.task_ensemble.variance_structure_weights(),
            dtype=float,
        )
        weights = np.maximum(weights, 0.0)
        weights /= max(float(np.sum(weights)), 1e-12)
        predicted = weights @ np.asarray(per_expert, dtype=float)
        certified = algorithm.task_ensemble.robust_moments_many(
            1, points, certification=True).aleatoric_upper
        posterior_source = (
            "replication_variance_task_posterior_hvd_mixture"
            if algorithm.task_ensemble.variance_structure_isolated
            else "task_posterior_hvd_mixture"
        )
    predicted = np.maximum(np.asarray(predicted, dtype=float), 1e-12)
    certified = np.maximum(np.asarray(certified, dtype=float), 1e-12)
    truth = np.maximum(true_variance, 1e-12)
    predicted_ratio = predicted / truth
    certified_ratio = certified / truth
    log_error = np.log(predicted) - np.log(truth)
    certified_log_error = np.log(certified) - np.log(truth)
    return {
        "status": "audited",
        "audit_size": int(len(points)),
        "posterior_source": posterior_source,
        "variance_structure_posterior_mode": str(getattr(
            algorithm.task_ensemble,
            "variance_structure_posterior_mode",
            "single_hvd" if algorithm.task_ensemble is None else "shared",
        )) if algorithm.task_ensemble is not None else "single_hvd",
        "log_variance_rmse": float(np.sqrt(np.mean(log_error ** 2))),
        "certified_log_variance_rmse": float(np.sqrt(
            np.mean(certified_log_error ** 2))),
        "median_predicted_true_ratio": float(np.median(predicted_ratio)),
        "median_certified_true_ratio": float(np.median(certified_ratio)),
        "variance_upper_coverage": float(np.mean(certified >= truth)),
        "underprediction_rate": float(np.mean(predicted < truth)),
        "target_oracle_used_for_decision": False,
        "target_oracle_used_for_post_run_audit": True,
    }


def _source_informed_archive_contract(args_dict, meta_diag):
    mode = str(args_dict.get(
        "initial_design_archive_match_mode", "exact")).strip().lower()
    if mode not in {"exact", "paired_frozen_control"}:
        raise ValueError(
            "initial_design_archive_match_mode must be exact or "
            "paired_frozen_control")
    if str(args_dict.get("initial_design", "auto")) != "source_informed":
        if mode != "exact":
            raise ValueError(
                "paired_frozen_control requires a source-informed design")
        return {
            "status": "not_applicable",
            "mode": mode,
            "matches": True,
            "target_data_used": False,
            "target_oracle_used": False,
        }

    expected = str(args_dict.get(
        "initial_design_source_archive_fingerprint", ""))
    training = dict((meta_diag or {}).get("training") or {})
    actual = str(training.get("source_archive_fingerprint", ""))
    if not expected:
        raise ValueError(
            "source-informed SC-OLH-KG requires an archive fingerprint")
    if not actual:
        raise ValueError("source training is missing an archive fingerprint")
    matches = actual == expected
    if mode == "exact" and not matches:
        raise ValueError(
            "SC-OLH-KG source archive does not match the frozen initial "
            f"proposal archive: {actual} != {expected}"
        )
    if mode == "paired_frozen_control":
        if str(args_dict.get(
            "meta_source_budget_mode", "per_episode"
        )).strip().lower() != "per_base_domain":
            raise ValueError(
                "paired_frozen_control requires a cost-matched source budget")
        if training.get("source_episode_target_data_used", True):
            raise ValueError(
                "paired_frozen_control source episodes used target data")
        if training.get("source_episode_target_oracle_used", True):
            raise ValueError(
                "paired_frozen_control source episodes used target oracle")
    return {
        "status": "audited",
        "mode": mode,
        "proposal_archive_fingerprint": expected,
        "posterior_archive_fingerprint": actual,
        "matches": bool(matches),
        "proposal_frozen_across_arms": mode == "paired_frozen_control",
        "target_data_used": False,
        "target_oracle_used": False,
    }


def run_one(task):
    args_dict = task["args"]
    heldout = task["heldout"]
    line = task["line"]
    seed = int(task["seed"])
    basis_label = str(args_dict.get(
        "basis_label",
        "state" if bool(args_dict.get("use_state_basis", True)) else "raw",
    ))
    basis_grid = bool(args_dict.get("basis_grid", False)) or bool(
        args_dict.get("basis_pair_grid", False))
    problem, meta_diag = build_target_problem(args_dict, heldout, line, seed)
    initial_design_archive_contract = _source_informed_archive_contract(
        args_dict, meta_diag)
    checkpoint_path = str(args_dict.get("checkpoint_path") or "").strip()
    runtime_checkpoint_dir = str(args_dict.get("runtime_checkpoint_dir") or "").strip()
    if not runtime_checkpoint_dir and checkpoint_path:
        safe_variant = f"{line}_{heldout}_{basis_label}_seed{int(seed)}"
        safe_variant = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_"
            for ch in safe_variant
        )
        runtime_checkpoint_dir = str(
            Path(checkpoint_path).with_suffix("").parent
            / "runtime"
            / safe_variant
        )
    if line == "strict":
        use_problem_initial = False
        use_boundary_initial = False
        use_recommendation_refinement = False
        state_candidate_count = 0
    else:
        use_problem_initial = True
        use_boundary_initial = False
        use_recommendation_refinement = True
        state_candidate_count = int(args_dict["state_candidate_count"])

    config = SingleOLHKGConfig(
        implementation_contract_id=str(args_dict.get(
            "implementation_contract_id", "unversioned")),
        theory_contract_id=str(args_dict.get(
            "theory_contract_id", "unversioned")),
        N=args_dict["N"],
        n0=args_dict["n0"],
        initial_design=str(args_dict.get("initial_design", "auto")),
        initial_design_points=tuple(
            tuple(map(int, point))
            for point in args_dict.get("initial_design_points", ())
        ),
        initial_design_fingerprint=str(args_dict.get(
            "initial_design_fingerprint", "")),
        initial_design_source_archive_fingerprint=str(args_dict.get(
            "initial_design_source_archive_fingerprint", "")),
        K1=args_dict["K1"],
        K2=args_dict["K2"],
        posterior_pool_size=args_dict["posterior_pool_size"],
        posterior_keep=args_dict["posterior_keep"],
        axis_candidate_count=args_dict["axis_candidate_count"],
        structured_candidate_count=args_dict["structured_candidate_count"],
        state_candidate_count=state_candidate_count,
        state_inverse_pool_size=args_dict["state_inverse_pool_size"],
        state_inverse_neighbors=args_dict["state_inverse_neighbors"],
        n_thr=args_dict["n_thr"],
        variance_mode=args_dict["variance_mode"],
        hvd_use_cumulative_provider=bool(args_dict.get(
            "hvd_use_cumulative_provider", True)),
        hvd_cumulative_transfer_mode=str(args_dict.get(
            "hvd_cumulative_transfer_mode", "scalar")),
        hvd_source_task_weight_mode=str(args_dict.get(
            "hvd_source_task_weight_mode", "independent")),
        hvd_cumulative_target_evidence_mode=str(args_dict.get(
            "hvd_cumulative_target_evidence_mode", "replication_only")),
        hvd_singleton_evidence_mode=str(args_dict.get(
            "hvd_singleton_evidence_mode", "in_sample_residual")),
        lambda_feas=args_dict["lambda_feas"],
        lambda_var=args_dict["lambda_var"],
        lambda_mean=args_dict["lambda_mean"],
        lambda_constraint_epistemic=args_dict["lambda_constraint_epistemic"],
        lambda_coupling=args_dict["lambda_coupling"] if line != "strict" else 0.0,
        beta_g=args_dict["beta_g"],
        certification_mode=args_dict["certification_mode"],
        recommendation_calibration=bool(args_dict["recommendation_calibration"]),
        certification_calibration=bool(args_dict["certification_calibration"]),
        recommendation_axis_oracle=False,
        use_problem_initial_samples=use_problem_initial,
        use_boundary_initial_samples=use_boundary_initial,
        use_recommendation_refinement=use_recommendation_refinement,
        use_state_coupling=True,
        use_state_basis=bool(args_dict["use_state_basis"]),
        state_basis_mode=args_dict["state_basis_mode"],
        constraint_state_basis_mode=args_dict["constraint_state_basis_mode"],
        raw_basis_dim=args_dict["raw_basis_dim"],
        raw_projection_seed=args_dict["raw_projection_seed"],
        numeric_backend=args_dict["numeric_backend"],
        numeric_backend_device=args_dict["numeric_backend_device"],
        torch_dtype=args_dict["torch_dtype"],
        torch_min_rows=args_dict["torch_min_rows"],
        encoder_kind=args_dict["encoder_kind"],
        encoder_latent_dim=args_dict["encoder_latent_dim"],
        encoder_fit_pool_size=args_dict["encoder_fit_pool_size"],
        lf_os_max_library_size=args_dict["lf_os_max_library_size"],
        lf_os_low_frequency_components=args_dict["lf_os_low_frequency_components"],
        lf_os_max_active=args_dict["lf_os_max_active"],
        lf_os_graph_neighbors=args_dict["lf_os_graph_neighbors"],
        lf_os_residual_floor_scale=args_dict["lf_os_residual_floor_scale"],
        acquisition_mode=args_dict["acquisition_mode"],
        decision_backend=args_dict.get("decision_backend", "legacy"),
        decision_risk_penalty=args_dict.get("decision_risk_penalty", 5.0),
        decision_aleatoric_mode=args_dict.get(
            "decision_aleatoric_mode", "certification_upper"),
        decision_violation_loss_mode=args_dict.get(
            "decision_violation_loss_mode", "positive_part"),
        decision_ambiguity_mode=args_dict.get(
            "decision_ambiguity_mode", "kl_robust"),
        decision_source_utility_weight=args_dict.get(
            "decision_source_utility_weight", 1.0),
        decision_backend_seed_offset=args_dict.get(
            "decision_backend_seed_offset", 470003),
        decision_recommend_observed_only=bool(args_dict.get(
            "decision_recommend_observed_only", True)),
        exact_kg_mc_samples=args_dict["exact_kg_mc_samples"],
        exact_kg_jobs=args_dict["exact_kg_jobs"],
        exact_kg_parallel_backend=args_dict["exact_kg_parallel_backend"],
        exact_kg_sampling_mode=args_dict["exact_kg_sampling_mode"],
        exact_kg_clip_negative=bool(args_dict["exact_kg_clip_negative"]),
        decision_contract_mode=args_dict.get(
            "decision_contract_mode", "legacy"),
        exact_kg_use_score=args_dict["exact_kg_use_score"],
        exact_kg_blend=args_dict["exact_kg_blend"],
        exact_kg_terminal_mode=args_dict["exact_kg_terminal_mode"],
        terminal_bayes_violation_penalty=args_dict[
            "terminal_bayes_violation_penalty"],
        terminal_frontier_candidate_count=args_dict[
            "terminal_frontier_candidate_count"],
        tcb_v2_mode=(
            str(args_dict.get("tcb_v2_mode", "off"))
            if (
                line in ("lodo", "lodo_teacher")
                and bool(args_dict.get("tcb_v2_enabled", False))
            )
            else "off"
        ),
        tcb_v2_frontier_count=int(args_dict.get(
            "tcb_v2_frontier_count", 1)),
        finalist_terminal_value_mode=str(args_dict.get(
            "finalist_terminal_value_mode", "model_default")),
        task_posterior_mode=(
            args_dict["task_posterior_mode"]
            if line in ("lodo", "lodo_teacher")
            else "off"
        ),
        source_discrepancy_update=bool(args_dict.get(
            "source_discrepancy_update", True)),
        task_posterior_initial_design=args_dict[
            "task_posterior_initial_design"],
        task_posterior_boundary_bracket_fraction=args_dict[
            "task_posterior_boundary_bracket_fraction"],
        task_posterior_mandatory_universal_count=args_dict[
            "task_posterior_mandatory_universal_count"],
        task_posterior_pilot_count=args_dict[
            "task_posterior_pilot_count"],
        task_posterior_temperature=args_dict["task_posterior_temperature"],
        task_posterior_temperature_decay=args_dict[
            "task_posterior_temperature_decay"],
        task_posterior_boundary_score_weight=args_dict[
            "task_posterior_boundary_score_weight"],
        task_posterior_objective_score_weight=args_dict[
            "task_posterior_objective_score_weight"],
        task_posterior_constraint_score_weight=args_dict[
            "task_posterior_constraint_score_weight"],
        task_posterior_safe_generalized=bool(args_dict[
            "task_posterior_safe_generalized"]),
        task_posterior_safe_boundary_score_weight=args_dict[
            "task_posterior_safe_boundary_score_weight"],
        task_posterior_safe_pairwise_score_weight=args_dict[
            "task_posterior_safe_pairwise_score_weight"],
        task_posterior_safe_pairwise_max_history=args_dict[
            "task_posterior_safe_pairwise_max_history"],
        task_posterior_safe_pairwise_probability_floor=args_dict[
            "task_posterior_safe_pairwise_probability_floor"],
        task_posterior_kl_radius_numerator=args_dict[
            "task_posterior_kl_radius_numerator"],
        task_posterior_confidence_delta=args_dict[
            "task_posterior_confidence_delta"],
        task_posterior_max_kl_radius=args_dict[
            "task_posterior_max_kl_radius"],
        task_posterior_robust_certificate_mode=str(args_dict.get(
            "task_posterior_robust_certificate_mode", "separable")),
        certification_head_authority=str(args_dict.get(
            "certification_head_authority", "task_joint")),
        task_posterior_prior_protection_numerator=args_dict[
            "task_posterior_prior_protection_numerator"],
        task_posterior_prior_protection_max=args_dict[
            "task_posterior_prior_protection_max"],
        task_posterior_local_kernel_expert=bool(args_dict[
            "task_posterior_local_kernel_expert"]),
        task_posterior_candidate_count=args_dict[
            "task_posterior_candidate_count"],
        task_posterior_recommendation_count=args_dict[
            "task_posterior_recommendation_count"],
        task_posterior_proposal_pool_size=args_dict[
            "task_posterior_proposal_pool_size"],
        task_posterior_proposal_exploration=args_dict[
            "task_posterior_proposal_exploration"],
        task_posterior_proposal_min_per_expert=args_dict[
            "task_posterior_proposal_min_per_expert"],
        task_posterior_sensitivity_mode=args_dict[
            "task_posterior_sensitivity_mode"],
        task_variance_posterior_mode=str(args_dict.get(
            "task_variance_posterior_mode", "shared")),
        task_latent_inference_mode=args_dict.get(
            "task_latent_inference_mode", "shadow"),
        task_latent_calibration_mode=args_dict.get(
            "task_latent_calibration_mode", "source_profiles"),
        constraint_uncertain_candidate_count=args_dict[
            "constraint_uncertain_candidate_count"],
        constraint_uncertain_pool_size=args_dict["constraint_uncertain_pool_size"],
        constraint_uncertain_state_pool_fraction=args_dict[
            "constraint_uncertain_state_pool_fraction"],
        constraint_uncertain_use_calibration=bool(
            args_dict["constraint_uncertain_use_calibration"]),
        constraint_epistemic_margin_softening=args_dict[
            "constraint_epistemic_margin_softening"],
        replication_candidate_count=args_dict[
            "replication_candidate_count"],
        replication_max_per_solution=args_dict[
            "replication_max_per_solution"],
        evaluate_or_replicate_new_action_count=int(args_dict.get(
            "evaluate_or_replicate_new_action_count", 1)),
        evaluate_or_replicate_new_action_policy=str(args_dict.get(
            "evaluate_or_replicate_new_action_policy", "canonical_sobol")),
        evaluate_or_replicate_baseline_new_action_count=int(args_dict.get(
            "evaluate_or_replicate_baseline_new_action_count", 0)),
        policy_improvement_mode=str(args_dict.get(
            "policy_improvement_mode", "off")),
        policy_improvement_mc_error_bound=float(args_dict.get(
            "policy_improvement_mc_error_bound", 0.0)),
        policy_improvement_rollout_depth=int(args_dict.get(
            "policy_improvement_rollout_depth", 1)),
        policy_improvement_rollout_max_arms=int(args_dict.get(
            "policy_improvement_rollout_max_arms", 4)),
        policy_improvement_rollout_mc_samples=int(args_dict.get(
            "policy_improvement_rollout_mc_samples", 2)),
        policy_improvement_rollout_mc_error_bound=float(args_dict.get(
            "policy_improvement_rollout_mc_error_bound", 0.0)),
        replication_margin_softening=args_dict[
            "replication_margin_softening"],
        adaptive_replication_voi=bool(args_dict.get(
            "adaptive_replication_voi", False)),
        posterior_dominance_enabled=bool(args_dict.get(
            "posterior_dominance_enabled", False)),
        posterior_dominance_delta=float(args_dict.get(
            "posterior_dominance_delta", 0.05)),
        posterior_dominance_min_mean_gain=float(args_dict.get(
            "posterior_dominance_min_mean_gain", 0.0)),
        posterior_dominance_initialization=str(args_dict.get(
            "posterior_dominance_initialization", "risk")),
        certification_recheck_top_k=args_dict[
            "certification_recheck_top_k"],
        certification_recheck_min_replicates=args_dict[
            "certification_recheck_min_replicates"],
        certification_recheck_soft_margin_scale=args_dict[
            "certification_recheck_soft_margin_scale"],
        certification_recheck_variance_prior_df=args_dict[
            "certification_recheck_variance_prior_df"],
        finalist_replication_budget=args_dict[
            "finalist_replication_budget"],
        finalist_replication_count=args_dict[
            "finalist_replication_count"],
        finalist_observed_safety_count=int(args_dict.get(
            "finalist_observed_safety_count", 1)),
        finalist_replication_min_replicates=args_dict[
            "finalist_replication_min_replicates"],
        finalist_replication_delta=args_dict[
            "finalist_replication_delta"],
        finalist_replication_variance_prior_df=args_dict[
            "finalist_replication_variance_prior_df"],
        finalist_replication_expert_stratified=bool(args_dict[
            "finalist_replication_expert_stratified"]),
        finalist_replication_adaptive_race=bool(args_dict[
            "finalist_replication_adaptive_race"]),
        finalist_replication_fixed_universe=bool(args_dict[
            "finalist_replication_fixed_universe"]),
        finalist_replication_policy=args_dict.get(
            "finalist_replication_policy", "legacy"),
        finalist_empirical_override=args_dict.get(
            "finalist_empirical_override", "legacy"),
        finalist_frontier_policy=args_dict.get(
            "finalist_frontier_policy", "legacy"),
        finalist_terminal_max_arms=args_dict.get(
            "finalist_terminal_max_arms", 4),
        finalist_terminal_mc_samples=args_dict.get(
            "finalist_terminal_mc_samples", 2),
        observed_incumbent_use_replicate_variance=bool(args_dict[
            "observed_incumbent_use_replicate_variance"]),
        safe_interior_candidate_count=args_dict["safe_interior_candidate_count"],
        safe_interior_pool_size=args_dict["safe_interior_pool_size"],
        safe_interior_margin=args_dict["safe_interior_margin"],
        observed_neighbor_candidate_count=args_dict[
            "observed_neighbor_candidate_count"],
        observed_neighbor_radius=args_dict["observed_neighbor_radius"],
        observed_neighbor_safe_margin_scale=args_dict[
            "observed_neighbor_safe_margin_scale"],
        recommendation_infeasible_penalty=args_dict[
            "recommendation_infeasible_penalty"],
        recommendation_infeasible_strategy=args_dict[
            "recommendation_infeasible_strategy"],
        recommend_observed_only=bool(args_dict["recommend_observed_only"]),
        recommendation_slack_initial=args_dict["recommendation_slack_initial"],
        recommendation_slack_decay=args_dict["recommendation_slack_decay"],
        recommendation_calibration_scope=args_dict["recommendation_calibration_scope"],
        recommendation_calibration_max_effective_fraction=args_dict[
            "recommendation_calibration_max_effective_fraction"],
        recommendation_calibration_min_obs=args_dict["recommendation_calibration_min_obs"],
        recommendation_calibration_max_leverage=args_dict[
            "recommendation_calibration_max_leverage"],
        recommendation_calibration_max_theory_margin=args_dict[
            "recommendation_calibration_max_theory_margin"],
        certification_calibration_min_obs=args_dict["certification_calibration_min_obs"],
        certification_calibration_beta=args_dict["certification_calibration_beta"],
        certification_calibration_policy=args_dict[
            "certification_calibration_policy"],
        certification_calibration_max_leverage=args_dict[
            "certification_calibration_max_leverage"],
        certification_calibration_max_theory_margin=args_dict[
            "certification_calibration_max_theory_margin"],
        certification_calibration_raise_delta=args_dict[
            "certification_calibration_raise_delta"],
        calibration_standardize_features=bool(
            args_dict["calibration_standardize_features"]),
        recommendation_observed_fallback=bool(args_dict["recommendation_observed_fallback"]),
        observed_incumbent_margin_scale=args_dict["observed_incumbent_margin_scale"],
        use_source_recommendation_slack=bool(args_dict["use_source_recommendation_slack"]),
        source_mean_prior_fallback=bool(args_dict["source_mean_prior_fallback"]),
        source_mean_prior_z=args_dict["source_mean_prior_z"],
        source_mean_prior_margin_tol=args_dict["source_mean_prior_margin_tol"],
        source_constraint_mean_coefficient_prior=bool(args_dict.get(
            "source_constraint_mean_coefficient_prior", False)),
        source_constraint_mean_hyperlaw_mode=str(args_dict.get(
            "source_constraint_mean_hyperlaw_mode", "single_gaussian_draw")),
        source_constraint_mean_adaptation_mode=str(args_dict.get(
            "source_constraint_mean_adaptation_mode", "frozen")),
        source_constraint_mean_deviation_mode=str(args_dict.get(
            "source_constraint_mean_deviation_mode", "raw_independent")),
        source_constraint_mean_misspecification_mode=str(args_dict.get(
            "source_constraint_mean_misspecification_mode", "none")),
        source_constraint_mean_misspecification_prior_df=float(args_dict.get(
            "source_constraint_mean_misspecification_prior_df", 4.0)),
        source_constraint_mean_misspecification_ridge=float(args_dict.get(
            "source_constraint_mean_misspecification_ridge", 1.0)),
        source_constraint_mean_misspecification_max_scale=float(args_dict.get(
            "source_constraint_mean_misspecification_max_scale", 100.0)),
        source_constraint_mean_misspecification_delta=float(args_dict.get(
            "source_constraint_mean_misspecification_delta", 0.05)),
        source_constraint_mean_confidence_mode=str(args_dict.get(
            "source_constraint_mean_confidence_mode", "model")),
        source_constraint_mean_confidence_delta=float(args_dict.get(
            "source_constraint_mean_confidence_delta", 0.05)),
        source_constraint_mean_contrast_scale=float(args_dict.get(
            "source_constraint_mean_contrast_scale", 1.0)),
        source_constraint_mean_role_epistemic_mode=str(args_dict.get(
            "source_constraint_mean_role_epistemic_mode", "none")),
        source_constraint_mean_null_weight=float(args_dict.get(
            "source_constraint_mean_null_weight", 0.5)),
        source_constraint_mean_null_geometry=str(args_dict.get(
            "source_constraint_mean_null_geometry", "isotropic")),
        source_constraint_mean_null_geometry_ridge=float(args_dict.get(
            "source_constraint_mean_null_geometry_ridge", 1e-3)),
        source_constraint_mean_evidence_temperature=float(args_dict.get(
            "source_constraint_mean_evidence_temperature", 1.0)),
        source_constraint_mean_structure_score_mode=str(args_dict.get(
            "source_constraint_mean_structure_score_mode",
            "marginal_likelihood")),
        source_constraint_mean_residual_rank_posterior=bool(args_dict.get(
            "source_constraint_mean_residual_rank_posterior", False)),
        source_constraint_mean_residual_rank_prior=str(args_dict.get(
            "source_constraint_mean_residual_rank_prior",
            "0.70,0.20,0.10")),
        source_constraint_mean_residual_rank_inactive_variance=float(
            args_dict.get(
                "source_constraint_mean_residual_rank_inactive_variance",
                1e-12)),
        boundary_coordinate_candidate_count=int(args_dict.get(
            "boundary_coordinate_candidate_count", 0)),
        boundary_coordinate_pool_size=int(args_dict.get(
            "boundary_coordinate_pool_size", 512)),
        boundary_coordinate_safe_fraction=float(args_dict.get(
            "boundary_coordinate_safe_fraction", 0.30)),
        boundary_coordinate_boundary_fraction=float(args_dict.get(
            "boundary_coordinate_boundary_fraction", 0.40)),
        boundary_coordinate_coverage_fraction=float(args_dict.get(
            "boundary_coordinate_coverage_fraction", 0.30)),
        truth_pool_diagnostics=bool(args_dict["truth_pool_diagnostics"]),
        truth_pool_good_regret=args_dict["truth_pool_good_regret"],
        truth_pool_max_candidates=args_dict["truth_pool_max_candidates"],
        llm_prior_enabled=bool(args_dict["llm_prior_enabled"]),
        llm_prior_base_url=args_dict["llm_prior_base_url"],
        llm_prior_model=args_dict["llm_prior_model"],
        llm_prior_api_key_env=args_dict["llm_prior_api_key_env"],
        llm_prior_candidate_count=args_dict["llm_prior_candidate_count"],
        llm_prior_inverse_pool_size=args_dict["llm_prior_inverse_pool_size"],
        llm_prior_interval=args_dict["llm_prior_interval"],
        llm_prior_min_obs=args_dict["llm_prior_min_obs"],
        llm_prior_timeout_sec=args_dict["llm_prior_timeout_sec"],
        llm_prior_gate_floor=args_dict["llm_prior_gate_floor"],
        llm_prior_max_observations=args_dict["llm_prior_max_observations"],
        checkpoint_dir=runtime_checkpoint_dir,
        checkpoint_resume=bool(args_dict["runtime_checkpoint_resume"]),
        checkpoint_interval=args_dict["runtime_checkpoint_interval"],
        evaluate_interval=int(args_dict.get("evaluate_interval", 5)),
        progress_logging=bool(args_dict["progress_logging"]),
        progress_label=(
            f"{line}:{heldout}:{basis_label}:seed{int(seed)}"
            if bool(args_dict["progress_logging"])
            else ""
        ),
        progress_units_per_iteration=args_dict["progress_units_per_iteration"],
        progress_exact_updates=args_dict["progress_exact_updates"],
        eval_pool_size=args_dict["eval_pool_size"],
        seed=seed,
    )
    started = time.time()
    alg = SingleOLHKGAlgorithm(problem, config)
    result = alg.run(verbose=False)
    for field in ("implementation_contract_id", "theory_contract_id"):
        expected = str(args_dict.get(field, "unversioned"))
        actual = str(result.get(field, "unversioned"))
        if actual != expected:
            raise RuntimeError(
                f"{field} changed across the target run: "
                f"expected {expected!r}, observed {actual!r}"
            )
    true_feasible = bool(result["true_feasible"])
    posterior_feasible = bool(result.get("posterior_feasible", False))
    initial_design = result.get("task_initial_design") or {}
    initial_truth = initial_design.get("truth_audit") or {}
    audit = (
        problem.admissibility_audit()
        if hasattr(problem, "admissibility_audit")
        else domain_tuned_audit().to_dict()
    )
    source_training = (
        (meta_diag or {}).get("training", {})
        if line in ("lodo", "lodo_teacher") else {}
    )
    variance_calibration = post_run_variance_calibration_audit(
        alg,
        problem,
        seed=seed,
        audit_size=int(args_dict.get("variance_audit_size", 128)),
    )
    return {
        "line": line,
        "heldout": heldout,
        "seed": seed,
        "implementation_contract_id": str(result.get(
            "implementation_contract_id", "unversioned")),
        "theory_contract_id": str(result.get(
            "theory_contract_id", "unversioned")),
        "theory_contract_timing": str(result.get(
            "theory_contract_timing", "missing")),
        "exact_kg_mc_samples": int(args_dict.get(
            "exact_kg_mc_samples", 0)),
        "exact_kg_sampling_mode": str(args_dict.get(
            "exact_kg_sampling_mode", "missing")),
        "evaluate_or_replicate_new_action_count": int(args_dict.get(
            "evaluate_or_replicate_new_action_count", 0)),
        "replication_max_per_solution": int(args_dict.get(
            "replication_max_per_solution", 0)),
        "initial_design_archive_contract": initial_design_archive_contract,
        "source_target_adaptation_contract": {
            "source_prior_frozen_online": True,
            "source_constraint_coefficient_prior_used": bool(args_dict.get(
                "source_constraint_mean_coefficient_prior", False)),
            "source_constraint_mean_adaptation_mode": str(args_dict.get(
                "source_constraint_mean_adaptation_mode", "frozen")),
            "source_constraint_coefficient_prior_target_oracle_used": False,
            "gradient_finetuning": False,
            "adaptation_kind": (
                "posterior_conditioning_expert_reweighting_and_"
                "target_representation_refit"
                if line in ("lodo", "lodo_teacher")
                else "target_posterior_conditioning"
            ),
            "online_parameters_changed": [
                "target_objective_gpr_posterior",
                "target_constraint_gpr_posterior",
                "target_cumulative_hvd_posterior",
                *(
                    [
                        "finite_source_expert_weights",
                        "each_expert_target_gpr_posterior",
                        "each_expert_target_hvd_posterior",
                        "budgeted_target_basis_and_risk_alignment_gate",
                        *(
                            ["source_eta_coefficient_posterior"]
                            if bool(args_dict.get(
                                "source_constraint_mean_coefficient_prior",
                                False,
                            ))
                            else []
                        ),
                    ]
                    if line in ("lodo", "lodo_teacher") else []
                ),
            ],
            "target_calls_used_for_adaptation": int(args_dict["N"]),
            "target_initial_design": str(args_dict.get(
                "initial_design", "auto")),
            "target_initial_design_fingerprint": initial_design.get(
                "fingerprint"),
            "target_initial_design_source_archive_fingerprint": (
                initial_design.get("source_archive_fingerprint")
            ),
            "initial_design_archive_match_mode": (
                initial_design_archive_contract["mode"]),
            "initial_design_archive_matches_posterior_archive": bool(
                initial_design_archive_contract["matches"]),
            "target_oracle_used_for_adaptation": False,
            "source_domains": source_training.get(
                "source_archive_domains", []),
            "source_profiles_per_domain": source_training.get(
                "source_archive_profiles_per_domain", {}),
            "source_simulator_calls": int(source_training.get(
                "source_archive_simulator_calls", 0)),
            "source_archive_fingerprint": source_training.get(
                "source_archive_fingerprint"),
            "source_oracle_aided": bool(source_training.get(
                "source_archive_oracle_aided", False)),
        },
        "N": int(args_dict["N"]),
        "n0": int(args_dict["n0"]),
        "K1": int(args_dict["K1"]),
        "K2": int(args_dict["K2"]),
        "basis_label": basis_label,
        "state_basis_enabled": bool(args_dict["use_state_basis"]),
        "state_basis_mode": args_dict["state_basis_mode"],
        "constraint_state_basis_mode": args_dict["constraint_state_basis_mode"],
        "meta_component_stage": args_dict["meta_component_stage"],
        "meta_spectral_orthogonalization": str(args_dict.get(
            "meta_spectral_orthogonalization", "symmetric")),
        "meta_observable_mean_coordinate": bool(args_dict.get(
            "meta_observable_mean_coordinate", False)),
        "meta_observable_mean_mode": str(args_dict.get(
            "meta_observable_mean_mode", "latent")),
        "meta_observable_mean_latent_dim": int(args_dict.get(
            "meta_observable_mean_latent_dim", 2)),
        "meta_observable_mean_training_target": str(args_dict.get(
            "meta_observable_mean_training_target", "constraint_mean")),
        "meta_observable_mean_input_mode": str(args_dict.get(
            "meta_observable_mean_input_mode", "policy_profile")),
        "meta_observable_mean_descriptor_mode": str(args_dict.get(
            "meta_observable_mean_descriptor_mode", "ordered")),
        "meta_observable_mean_feature_mode": str(args_dict.get(
            "meta_observable_mean_feature_mode", "linear")),
        "meta_observable_mean_latent_transform": str(args_dict.get(
            "meta_observable_mean_latent_transform", "identity")),
        "meta_observable_mean_target_residual_rank": int(args_dict.get(
            "meta_observable_mean_target_residual_rank", 0)),
        "meta_observable_mean_target_residual_prior_scale": float(
            args_dict.get(
                "meta_observable_mean_target_residual_prior_scale", 1.0)),
        "meta_observable_mean_target_residual_pool_size": int(args_dict.get(
            "meta_observable_mean_target_residual_pool_size", 128)),
        "meta_observable_mean_target_residual_rcond": float(args_dict.get(
            "meta_observable_mean_target_residual_rcond", 1e-8)),
        "meta_observable_mean_role_assignment_posterior": bool(args_dict.get(
            "meta_observable_mean_role_assignment_posterior", False)),
        "meta_observable_mean_role_assignment_prior": str(args_dict.get(
            "meta_observable_mean_role_assignment_prior", "uniform")),
        "meta_observable_mean_role_assignment_prior_temperature_scale": float(
            args_dict.get(
                "meta_observable_mean_role_assignment_prior_temperature_scale",
                1.0,
            )),
        "meta_observable_mean_role_assignment_inactive_variance": float(
            args_dict.get(
                "meta_observable_mean_role_assignment_inactive_variance",
                1e-12)),
        "meta_observable_variance_input_mode": str(args_dict.get(
            "meta_observable_variance_input_mode", "legacy_policy_proxy")),
        "source_constraint_mean_coefficient_prior": bool(args_dict.get(
            "source_constraint_mean_coefficient_prior", False)),
        "source_constraint_mean_hyperlaw_mode": str(args_dict.get(
            "source_constraint_mean_hyperlaw_mode", "single_gaussian_draw")),
        "source_constraint_mean_adaptation_mode": str(args_dict.get(
            "source_constraint_mean_adaptation_mode", "frozen")),
        "source_constraint_mean_deviation_mode": str(args_dict.get(
            "source_constraint_mean_deviation_mode", "raw_independent")),
        "source_constraint_mean_misspecification_mode": str(args_dict.get(
            "source_constraint_mean_misspecification_mode", "none")),
        "source_constraint_mean_misspecification_prior_df": float(
            args_dict.get(
                "source_constraint_mean_misspecification_prior_df", 4.0)),
        "source_constraint_mean_misspecification_ridge": float(args_dict.get(
            "source_constraint_mean_misspecification_ridge", 1.0)),
        "source_constraint_mean_misspecification_max_scale": float(
            args_dict.get(
                "source_constraint_mean_misspecification_max_scale", 100.0)),
        "source_constraint_mean_misspecification_delta": float(
            args_dict.get(
                "source_constraint_mean_misspecification_delta", 0.05)),
        "source_constraint_mean_confidence_mode": str(args_dict.get(
            "source_constraint_mean_confidence_mode", "model")),
        "source_constraint_mean_confidence_delta": float(args_dict.get(
            "source_constraint_mean_confidence_delta", 0.05)),
        "source_constraint_mean_contrast_scale": float(args_dict.get(
            "source_constraint_mean_contrast_scale", 1.0)),
        "source_constraint_mean_role_epistemic_mode": str(args_dict.get(
            "source_constraint_mean_role_epistemic_mode", "none")),
        "source_constraint_mean_null_weight": float(args_dict.get(
            "source_constraint_mean_null_weight", 0.5)),
        "source_constraint_mean_null_geometry": str(args_dict.get(
            "source_constraint_mean_null_geometry", "isotropic")),
        "source_constraint_mean_null_geometry_ridge": float(args_dict.get(
            "source_constraint_mean_null_geometry_ridge", 1e-3)),
        "source_constraint_mean_evidence_temperature": float(args_dict.get(
            "source_constraint_mean_evidence_temperature", 1.0)),
        "source_constraint_mean_structure_score_mode": str(args_dict.get(
            "source_constraint_mean_structure_score_mode",
            "marginal_likelihood")),
        "source_constraint_mean_residual_rank_posterior": bool(args_dict.get(
            "source_constraint_mean_residual_rank_posterior", False)),
        "source_constraint_mean_residual_rank_prior": str(args_dict.get(
            "source_constraint_mean_residual_rank_prior",
            "0.70,0.20,0.10")),
        "source_constraint_mean_residual_rank_inactive_variance": float(
            args_dict.get(
                "source_constraint_mean_residual_rank_inactive_variance",
                1e-12)),
        "boundary_coordinate_candidate_count": int(args_dict.get(
            "boundary_coordinate_candidate_count", 0)),
        "boundary_coordinate_pool_size": int(args_dict.get(
            "boundary_coordinate_pool_size", 512)),
        "boundary_coordinate_safe_fraction": float(args_dict.get(
            "boundary_coordinate_safe_fraction", 0.30)),
        "boundary_coordinate_boundary_fraction": float(args_dict.get(
            "boundary_coordinate_boundary_fraction", 0.40)),
        "boundary_coordinate_coverage_fraction": float(args_dict.get(
            "boundary_coordinate_coverage_fraction", 0.30)),
        "meta_source_observation_mode": str(args_dict.get(
            "meta_source_observation_mode", "analytic")),
        "meta_source_observation_replicates": int(args_dict.get(
            "meta_source_observation_replicates", 1)),
        "meta_source_design_mode": str(args_dict.get(
            "meta_source_design_mode", "random")),
        "meta_source_universal_fraction": float(args_dict.get(
            "meta_source_universal_fraction", 0.75)),
        "meta_source_consensus_template_count": int(args_dict.get(
            "meta_source_consensus_template_count", 0)),
        "meta_ordered_cumulative_exposure": bool(args_dict.get(
            "meta_ordered_cumulative_exposure", False)),
        "meta_spectral_low_frequency_prior": bool(args_dict.get(
            "meta_spectral_low_frequency_prior", True)),
        "structural_prior_profile": str(args_dict.get(
            "structural_prior_profile", "inherit")),
        "hvd_ablation_profile": str(args_dict.get(
            "hvd_ablation_profile", "inherit")),
        "target_shared_shock_scale": float(args_dict.get(
            "target_shared_shock_scale", 1.0)),
        "meta_ordered_exposure_max_frequency": int(args_dict.get(
            "meta_ordered_exposure_max_frequency", 8)),
        "meta_ordered_exposure_frequency_penalty": float(args_dict.get(
            "meta_ordered_exposure_frequency_penalty", 0.10)),
        "meta_ordered_exposure_basis_mode": str(args_dict.get(
            "meta_ordered_exposure_basis_mode", "full_quadratic")),
        "meta_ordered_exposure_orthogonal_coordinates": bool(args_dict.get(
            "meta_ordered_exposure_orthogonal_coordinates", True)),
        "meta_ordered_exposure_adaptive_sparsity": bool(args_dict.get(
            "meta_ordered_exposure_adaptive_sparsity", False)),
        "task_posterior_sensitivity_mode": args_dict[
            "task_posterior_sensitivity_mode"],
        "task_variance_posterior_mode": str(args_dict.get(
            "task_variance_posterior_mode", "shared")),
        "task_latent_inference_mode": args_dict.get(
            "task_latent_inference_mode", "shadow"),
        "task_latent_calibration_mode": args_dict.get(
            "task_latent_calibration_mode", "source_profiles"),
        "task_posterior_safe_generalized": bool(args_dict[
            "task_posterior_safe_generalized"]),
        "task_posterior_boundary_bracket_fraction": float(args_dict[
            "task_posterior_boundary_bracket_fraction"]),
        "task_posterior_mandatory_universal_count": int(args_dict[
            "task_posterior_mandatory_universal_count"]),
        "task_posterior_prior_protection_numerator": float(args_dict[
            "task_posterior_prior_protection_numerator"]),
        "task_posterior_prior_protection_max": float(args_dict[
            "task_posterior_prior_protection_max"]),
        "task_posterior_local_kernel_expert": bool(args_dict[
            "task_posterior_local_kernel_expert"]),
        "exact_kg_terminal_mode": str(args_dict[
            "exact_kg_terminal_mode"]),
        "decision_contract_mode": str(args_dict.get(
            "decision_contract_mode", "legacy")),
        "exact_kg_sampling_mode": str(args_dict[
            "exact_kg_sampling_mode"]),
        "exact_kg_clip_negative": bool(args_dict[
            "exact_kg_clip_negative"]),
        "decision_backend": str(args_dict.get(
            "decision_backend", "legacy")),
        "decision_risk_penalty": float(args_dict.get(
            "decision_risk_penalty", 5.0)),
        "decision_aleatoric_mode": str(args_dict.get(
            "decision_aleatoric_mode", "certification_upper")),
        "decision_violation_loss_mode": str(args_dict.get(
            "decision_violation_loss_mode", "positive_part")),
        "decision_ambiguity_mode": str(args_dict.get(
            "decision_ambiguity_mode", "kl_robust")),
        "decision_source_utility_weight": float(args_dict.get(
            "decision_source_utility_weight", 1.0)),
        "decision_backend_diagnostics": result.get(
            "decision_backend_diagnostics"),
        "decision_backend_contract": result.get(
            "decision_backend_contract"),
        "decision_backend_terminal_rule": result.get(
            "decision_backend_terminal_rule"),
        "decision_backend_terminal_value": result.get(
            "decision_backend_terminal_value"),
        "decision_backend_terminal_status": result.get(
            "decision_backend_terminal_status"),
        "decision_backend_terminal_certified_count": result.get(
            "decision_backend_terminal_certified_count"),
        "decision_backend_terminal_margin": result.get(
            "decision_backend_terminal_margin"),
        "decision_backend_terminal_used": bool(result.get(
            "decision_backend_terminal_used", False)),
        "decision_backend_terminal_risk": result.get(
            "decision_backend_terminal_risk"),
        "decision_backend_terminal_expected_violation": result.get(
            "decision_backend_terminal_expected_violation"),
        "decision_backend_terminal_probability_violation": result.get(
            "decision_backend_terminal_probability_violation"),
        "decision_backend_terminal_violation_loss": result.get(
            "decision_backend_terminal_violation_loss"),
        "decision_backend_terminal_violation_loss_mode": result.get(
            "decision_backend_terminal_violation_loss_mode"),
        "decision_backend_terminal_aleatoric_mode": result.get(
            "decision_backend_terminal_aleatoric_mode"),
        "decision_backend_terminal_ambiguity_mode": result.get(
            "decision_backend_terminal_ambiguity_mode"),
        "decision_backend_terminal_nominal_expected_violation": result.get(
            "decision_backend_terminal_nominal_expected_violation"),
        "decision_backend_terminal_robust_expected_violation": result.get(
            "decision_backend_terminal_robust_expected_violation"),
        "decision_backend_terminal_nominal_probability_violation": result.get(
            "decision_backend_terminal_nominal_probability_violation"),
        "decision_backend_terminal_robust_probability_violation": result.get(
            "decision_backend_terminal_robust_probability_violation"),
        "terminal_bayes_pool_audit": result.get(
            "terminal_bayes_pool_audit"),
        "target_design_fingerprint": result.get(
            "target_design_fingerprint"),
        "online_action_trace": result.get("online_action_trace"),
        "online_action_sequence_fingerprint": result.get(
            "online_action_sequence_fingerprint"),
        "online_action_trace_target_oracle_used": bool(result.get(
            "online_action_trace_target_oracle_used", True)),
        "simulation_randomness_contract": result.get(
            "simulation_randomness_contract"),
        "proposal_randomness_contract": result.get(
            "proposal_randomness_contract"),
        "adaptive_replication_voi_enabled": bool(args_dict.get(
            "adaptive_replication_voi", False)),
        "adaptive_replication_voi": result.get(
            "adaptive_replication_voi"),
        "adaptive_replication_selected_count": int((
            result.get("adaptive_replication_voi") or {}
        ).get("selected_replication_count", 0)),
        "adaptive_new_point_selected_count": int((
            result.get("adaptive_replication_voi") or {}
        ).get("selected_new_point_count", 0)),
        "posterior_dominance_enabled": bool(args_dict.get(
            "posterior_dominance_enabled", False)),
        "posterior_dominance": result.get("posterior_dominance"),
        "posterior_dominance_switch_count": int((
            result.get("posterior_dominance") or {}
        ).get("switch_count", 0)),
        "posterior_dominance_terminal_used": bool(
            result.get("posterior_dominance_terminal_used", False)),
        "adaptive_outcome_audit": result.get(
            "adaptive_outcome_audit"),
        "certificate_outcome_audit": result.get(
            "certificate_outcome_audit"),
        "exact_kg_diagnostics": result.get("exact_kg_diagnostics"),
        "terminal_bayes_violation_penalty": float(args_dict[
            "terminal_bayes_violation_penalty"]),
        "terminal_frontier_candidate_count": int(args_dict[
            "terminal_frontier_candidate_count"]),
        "tcb_v2_enabled": bool(args_dict.get("tcb_v2_enabled", False)),
        "tcb_v2_mode": str(args_dict.get("tcb_v2_mode", "off")),
        "tcb_v2_frontier_count": int(args_dict.get(
            "tcb_v2_frontier_count", 1)),
        "tcb_v2_descriptor_mode": str(args_dict.get(
            "tcb_v2_descriptor_mode", "learned_risk")),
        "finalist_terminal_value_mode": str(args_dict.get(
            "finalist_terminal_value_mode", "model_default")),
        "terminal_pool_shared": bool(result.get(
            "terminal_pool_shared", False)),
        "terminal_pool_size": result.get("terminal_pool_size"),
        "certification_recheck_top_k": int(args_dict[
            "certification_recheck_top_k"]),
        "certification_recheck_min_replicates": int(args_dict[
            "certification_recheck_min_replicates"]),
        "certification_recheck_soft_margin_scale": float(args_dict[
            "certification_recheck_soft_margin_scale"]),
        "finalist_replication_budget": int(args_dict[
            "finalist_replication_budget"]),
        "finalist_replication_count": int(args_dict[
            "finalist_replication_count"]),
        "finalist_observed_safety_count": int(args_dict.get(
            "finalist_observed_safety_count", 1)),
        "finalist_replication_min_replicates": int(args_dict[
            "finalist_replication_min_replicates"]),
        "finalist_replication_delta": float(args_dict[
            "finalist_replication_delta"]),
        "finalist_replication_expert_stratified": bool(args_dict[
            "finalist_replication_expert_stratified"]),
        "finalist_replication_adaptive_race": bool(args_dict[
            "finalist_replication_adaptive_race"]),
        "finalist_replication_fixed_universe": bool(args_dict[
            "finalist_replication_fixed_universe"]),
        "finalist_replication_policy": str(args_dict.get(
            "finalist_replication_policy", "legacy")),
        "finalist_empirical_override": str(args_dict.get(
            "finalist_empirical_override", "legacy")),
        "finalist_frontier_policy": str(args_dict.get(
            "finalist_frontier_policy", "legacy")),
        "finalist_terminal_max_arms": int(args_dict.get(
            "finalist_terminal_max_arms", 4)),
        "finalist_terminal_mc_samples": int(args_dict.get(
            "finalist_terminal_mc_samples", 2)),
        "finalist_replication": result.get("finalist_replication"),
        "replicated_finalist_used": result.get(
            "replicated_finalist_used"),
        "replicated_finalist_reason": result.get(
            "replicated_finalist_reason"),
        "replicated_finalist_empirical_certificate": result.get(
            "replicated_finalist_empirical_certificate"),
        "replicated_finalist_selected": result.get(
            "replicated_finalist_selected"),
        "replicated_finalist_rows": result.get(
            "replicated_finalist_rows"),
        "observed_incumbent_use_replicate_variance": bool(args_dict[
            "observed_incumbent_use_replicate_variance"]),
        "transfer_cell": f"{line}-{basis_label}",
        "variant": (
            f"{line}-{basis_label}:{heldout}"
            if basis_grid
            else f"{line}:{heldout}"
        ),
        "audit_admissible_mainline": bool(audit.get("admissible_mainline", False)),
        "audit": audit,
        "meta_prior": meta_diag,
        "meta_basis": result.get("meta_basis"),
        "mean_risk_coordinate_contract": result.get(
            "mean_risk_coordinate_contract"),
        "task_posterior": result.get("task_posterior"),
        "source_discrepancy_update": bool(args_dict.get(
            "source_discrepancy_update", True)),
        "task_meta_coherence": result.get("task_meta_coherence"),
        "task_initial_design": initial_design,
        "initial_boundary_bracket_generated": initial_design.get(
            "boundary_bracket_generated"),
        "initial_mandatory_universal_generated": initial_design.get(
            "mandatory_universal_generated"),
        "initial_true_feasible_count": initial_truth.get(
            "initial_design_true_feasible_count"),
        "initial_true_feasible_rate": initial_truth.get(
            "initial_design_true_feasible_rate"),
        "initial_has_true_feasible": initial_truth.get(
            "initial_design_has_true_feasible"),
        "initial_true_min_margin": initial_truth.get(
            "initial_design_true_min_margin"),
        "initial_true_median_margin": initial_truth.get(
            "initial_design_true_median_margin"),
        "initial_best_feasible_regret": (
            result.get("adaptive_outcome_audit") or {}
        ).get("initial_best_feasible_regret"),
        "adaptive_rescue": bool((
            result.get("adaptive_outcome_audit") or {}
        ).get("adaptive_rescue", False)),
        "adaptive_loss": bool((
            result.get("adaptive_outcome_audit") or {}
        ).get("adaptive_loss", False)),
        "adaptive_preservation": bool((
            result.get("adaptive_outcome_audit") or {}
        ).get("adaptive_preservation", False)),
        "adaptive_improves_initial_best": bool((
            result.get("adaptive_outcome_audit") or {}
        ).get("adaptive_improves_initial_best", False)),
        "adaptive_regret_change": (
            result.get("adaptive_outcome_audit") or {}
        ).get("adaptive_regret_change"),
        "posterior_certificate_vacuous": bool((
            result.get("certificate_outcome_audit") or {}
        ).get("posterior_certificate_vacuous", False)),
        "posterior_certified_evaluated_count": (
            result.get("certificate_outcome_audit") or {}
        ).get("posterior_certified_count"),
        "certificate_precision": (
            result.get("certificate_outcome_audit") or {}
        ).get("certificate_precision"),
        "certificate_recall_on_evaluated_feasible": (
            result.get("certificate_outcome_audit") or {}
        ).get("certificate_recall_on_evaluated_feasible"),
        "false_certificate_count": (
            result.get("certificate_outcome_audit") or {}
        ).get("false_certificate_count"),
        "adaptive_sparsity": result.get("adaptive_sparsity"),
        "hvd_use_cumulative_provider": bool(args_dict.get(
            "hvd_use_cumulative_provider", True)),
        "hvd_cumulative_transfer_mode": str(args_dict.get(
            "hvd_cumulative_transfer_mode", "scalar")),
        "hvd_source_task_weight_mode": str(args_dict.get(
            "hvd_source_task_weight_mode", "independent")),
        "hvd_cumulative_target_evidence_mode": str(args_dict.get(
            "hvd_cumulative_target_evidence_mode", "replication_only")),
        "hvd_singleton_evidence_mode": str(args_dict.get(
            "hvd_singleton_evidence_mode", "in_sample_residual")),
        "task_posterior_robust_certificate_mode": str(args_dict.get(
            "task_posterior_robust_certificate_mode", "separable")),
        "certification_head_authority": str(args_dict.get(
            "certification_head_authority", "task_joint")),
        "gpr_numerics": result.get("gpr_numerics"),
        "source_conditioned_confidence": result.get(
            "source_conditioned_confidence"),
        "true_feasible": true_feasible,
        "posterior_feasible": posterior_feasible,
        "false_feasible": bool(posterior_feasible and not true_feasible),
        "true_objective": float(result["true_objective"]),
        "true_best_objective": float(result["true_best_objective"]),
        "simple_regret": float(result["simple_regret"]),
        "feasible_simple_regret": (
            float(result["simple_regret"]) if true_feasible else None
        ),
        "true_chance_margin": float(result["true_chance_margin"]),
        "constraint_violation": float(max(result["true_chance_margin"], 0.0)),
        "posterior_chance_margin": float(result["posterior_chance_margin"]),
        "posterior_theory_chance_margin": result.get("posterior_theory_chance_margin"),
        "posterior_calibrated_chance_margin": result.get(
            "posterior_calibrated_chance_margin"),
        "posterior_certification_source": result.get("posterior_certification_source"),
        "certification_margin_decomposition": result.get(
            "certification_margin_decomposition"),
        "posterior_bayes_risk_used": result.get(
            "posterior_bayes_risk_used"),
        "posterior_bayes_risk": result.get("posterior_bayes_risk"),
        "posterior_bayes_objective": result.get(
            "posterior_bayes_objective"),
        "posterior_bayes_expected_violation": result.get(
            "posterior_bayes_expected_violation"),
        "posterior_bayes_kl_radius": result.get(
            "posterior_bayes_kl_radius"),
        "recommendation_slack": result.get("recommendation_slack"),
        "recommendation_infeasible_strategy": result.get(
            "recommendation_infeasible_strategy"),
        "recommendation_effective_infeasible_penalty": result.get(
            "recommendation_effective_infeasible_penalty"),
        "recommend_observed_only": bool(args_dict["recommend_observed_only"]),
        "observed_neighbor_candidate_count": int(
            args_dict["observed_neighbor_candidate_count"]),
        "observed_neighbor_radius": float(args_dict["observed_neighbor_radius"]),
        "observed_neighbor_safe_margin_scale": float(
            args_dict["observed_neighbor_safe_margin_scale"]),
        "recommendation_calibration": result.get("recommendation_calibration"),
        "calibrated_recommendation_used": result.get("calibrated_recommendation_used"),
        "calibrated_recommendation_reason": result.get(
            "calibrated_recommendation_reason"),
        "calibrated_recommendation_scope": result.get(
            "calibrated_recommendation_scope"),
        "calibrated_recommendation_rejected_by_observed": result.get(
            "calibrated_recommendation_rejected_by_observed"),
        "calibrated_recommendation_rejected_candidate_margin": result.get(
            "calibrated_recommendation_rejected_candidate_margin"),
        "calibrated_recommendation_rejected_candidate_mu_obj": result.get(
            "calibrated_recommendation_rejected_candidate_mu_obj"),
        "n_calibration_candidates": result.get("n_calibration_candidates"),
        "n_calibration_certified_guarded": result.get(
            "n_calibration_certified_guarded"),
        "n_calibration_guarded": result.get("n_calibration_guarded"),
        "n_calibration_feasible": result.get("n_calibration_feasible"),
        "n_calibration_raw_feasible": result.get("n_calibration_raw_feasible"),
        "calibrated_constraint_margin": result.get(
            "calibrated_constraint_margin"),
        "calibrated_guarded_constraint_margin": result.get(
            "calibrated_guarded_constraint_margin"),
        "calibration_max_leverage": result.get("calibration_max_leverage"),
        "calibration_max_theory_margin": result.get(
            "calibration_max_theory_margin"),
        "calibration_selected_leverage": result.get(
            "calibration_selected_leverage"),
        "calibration_selected_theory_margin": result.get(
            "calibration_selected_theory_margin"),
        "calibration_min_leverage": result.get("calibration_min_leverage"),
        "calibration_median_leverage": result.get("calibration_median_leverage"),
        "calibration_min_theory_margin": result.get(
            "calibration_min_theory_margin"),
        "task_adaptive_violation_probability": result.get(
            "task_adaptive_violation_probability"),
        "task_adaptive_expected_violation_loss": result.get(
            "task_adaptive_expected_violation_loss"),
        "task_adaptive_objective_loss": result.get(
            "task_adaptive_objective_loss"),
        "task_adaptive_robust_component": result.get(
            "task_adaptive_robust_component"),
        "task_adaptive_empirical_component": result.get(
            "task_adaptive_empirical_component"),
        "task_adaptive_total_loss": result.get("task_adaptive_total_loss"),
        "task_adaptive_class_weights": result.get(
            "task_adaptive_class_weights"),
        "task_adaptive_affects_theory_certificate": result.get(
            "task_adaptive_affects_theory_certificate"),
        "task_adaptive_expected_empirical_trust": result.get(
            "task_adaptive_expected_empirical_trust"),
        "task_adaptive_prequential_sigma": result.get(
            "task_adaptive_prequential_sigma"),
        "task_adaptive_loo_sigma": result.get("task_adaptive_loo_sigma"),
        "task_adaptive_conformal_sigma": result.get(
            "task_adaptive_conformal_sigma"),
        "task_adaptive_empirical_hvd_variance": result.get(
            "task_adaptive_empirical_hvd_variance"),
        "recommendation_calibration_audit_available": result.get(
            "recommendation_calibration_audit_available"),
        "recommendation_calibration_n_feasible": result.get(
            "recommendation_calibration_n_feasible"),
        "recommendation_calibration_sigma": result.get(
            "recommendation_calibration_sigma"),
        "recommendation_calibration_selected_ridge": result.get(
            "recommendation_calibration_selected_ridge"),
        "recommendation_calibration_effective_rank": result.get(
            "recommendation_calibration_effective_rank"),
        "recommendation_calibration_effective_rank_cap": result.get(
            "recommendation_calibration_effective_rank_cap"),
        "recommendation_calibration_rank_cap_satisfied": result.get(
            "recommendation_calibration_rank_cap_satisfied"),
        "recommendation_calibration_nested_refit": result.get(
            "recommendation_calibration_nested_refit"),
        "recommendation_calibration_features_standardized": result.get(
            "recommendation_calibration_features_standardized"),
        "recommendation_selected_calibrated_rec_margin": result.get(
            "recommendation_selected_calibrated_rec_margin"),
        "recommendation_selected_calibrated_rec_objective": result.get(
            "recommendation_selected_calibrated_rec_objective"),
        "recommendation_selected_calibrated_rec_leverage": result.get(
            "recommendation_selected_calibrated_rec_leverage"),
        "certification_calibration_used": result.get("certification_calibration_used"),
        "certification_calibration_policy": result.get(
            "certification_calibration_policy"),
        "certification_calibration_n_used": result.get(
            "certification_calibration_n_used"),
        "certification_calibration_max_leverage": result.get(
            "certification_calibration_max_leverage"),
        "certification_calibration_max_theory_margin": result.get(
            "certification_calibration_max_theory_margin"),
        "certification_calibration_raise_delta": result.get(
            "certification_calibration_raise_delta"),
        "certification_calibration_n_feasible": result.get(
            "certification_calibration_n_feasible"),
        "source_mean_prior_fallback": result.get("source_mean_prior_fallback"),
        "source_mean_prior_used": result.get("source_mean_prior_used"),
        "source_mean_prior_available": result.get("source_mean_prior_available"),
        "source_mean_prior_n_feasible": result.get("source_mean_prior_n_feasible"),
        "source_mean_prior_min_margin": result.get("source_mean_prior_min_margin"),
        "source_mean_prior_selected_margin": result.get(
            "source_mean_prior_selected_margin"),
        "source_mean_prior_guard_used": result.get("source_mean_prior_guard_used"),
        "source_mean_prior_ranker_used": result.get("source_mean_prior_ranker_used"),
        "source_mean_prior_guard_n_feasible": result.get(
            "source_mean_prior_guard_n_feasible"),
        "recommendation_has_true_feasible": result.get(
            "recommendation_has_true_feasible"),
        "recommendation_missed_true_feasible": result.get(
            "recommendation_missed_true_feasible"),
        "recommendation_has_true_safe_good": result.get(
            "recommendation_has_true_safe_good"),
        "recommendation_missed_true_safe_good": result.get(
            "recommendation_missed_true_safe_good"),
        "recommendation_selected_true_feasible": result.get(
            "recommendation_selected_true_feasible"),
        "recommendation_best_true_feasible_posterior_margin": result.get(
            "recommendation_best_true_feasible_posterior_margin"),
        "recommendation_best_true_feasible_posterior_feasible": result.get(
            "recommendation_best_true_feasible_posterior_feasible"),
        "recommendation_true_best_feasible_regret": result.get(
            "recommendation_true_best_feasible_regret"),
        "recommendation_best_true_feasible_decision_margin": result.get(
            "recommendation_best_true_feasible_decision_margin"),
        "recommendation_best_true_feasible_x": result.get(
            "recommendation_best_true_feasible_x"),
        "recommendation_best_true_feasible_decision_feasible": result.get(
            "recommendation_best_true_feasible_decision_feasible"),
        "recommendation_selected_decision_margin": result.get(
            "recommendation_selected_decision_margin"),
        "recommendation_best_true_feasible_mu_con": result.get(
            "recommendation_best_true_feasible_mu_con"),
        "recommendation_best_true_feasible_epistemic_var": result.get(
            "recommendation_best_true_feasible_epistemic_var"),
        "recommendation_best_true_feasible_aleatoric_var": result.get(
            "recommendation_best_true_feasible_aleatoric_var"),
        "recommendation_best_true_feasible_theory_margin": result.get(
            "recommendation_best_true_feasible_theory_margin"),
        "recommendation_best_true_feasible_calibrated_margin": result.get(
            "recommendation_best_true_feasible_calibrated_margin"),
        "recommendation_best_true_feasible_calibrated_rec_margin": result.get(
            "recommendation_best_true_feasible_calibrated_rec_margin"),
        "recommendation_best_true_feasible_calibrated_rec_objective": result.get(
            "recommendation_best_true_feasible_calibrated_rec_objective"),
        "recommendation_best_true_feasible_calibrated_rec_leverage": result.get(
            "recommendation_best_true_feasible_calibrated_rec_leverage"),
        "recommendation_best_true_feasible_source_margin": result.get(
            "recommendation_best_true_feasible_source_margin"),
        "recommendation_best_true_feasible_certification_source": result.get(
            "recommendation_best_true_feasible_certification_source"),
        "recommendation_selected_mu_con": result.get("recommendation_selected_mu_con"),
        "recommendation_selected_epistemic_var": result.get(
            "recommendation_selected_epistemic_var"),
        "recommendation_selected_aleatoric_var": result.get(
            "recommendation_selected_aleatoric_var"),
        "recommendation_selected_theory_margin": result.get(
            "recommendation_selected_theory_margin"),
        "recommendation_selected_calibrated_margin": result.get(
            "recommendation_selected_calibrated_margin"),
        "recommendation_selected_calibrated_rec_margin": result.get(
            "recommendation_selected_calibrated_rec_margin"),
        "recommendation_selected_calibrated_rec_objective": result.get(
            "recommendation_selected_calibrated_rec_objective"),
        "recommendation_selected_calibrated_rec_leverage": result.get(
            "recommendation_selected_calibrated_rec_leverage"),
        "recommendation_selected_source_margin": result.get(
            "recommendation_selected_source_margin"),
        "recommendation_selected_certification_source": result.get(
            "recommendation_selected_certification_source"),
        "observed_incumbent_used": result.get("observed_incumbent_used"),
        "observed_incumbent_rejected": result.get("observed_incumbent_rejected"),
        "observed_incumbent_reason": result.get("observed_incumbent_reason"),
        "observed_incumbent_chance_margin": result.get(
            "observed_incumbent_chance_margin"),
        "observed_incumbent_sigma": result.get(
            "observed_incumbent_sigma"),
        "observed_incumbent_sigma_source": result.get(
            "observed_incumbent_sigma_source"),
        "observed_incumbent_replicate_count": result.get(
            "observed_incumbent_replicate_count"),
        "certification_recheck_selected_count": int(
            result.get("candidate_source_counts", {}).get(
                "certification_recheck", 0)),
        "n_simulations": int(result["n_simulations"]),
        "n_distinct_solutions": int(result["n_distinct_solutions"]),
        "n_pool": int(result.get("n_pool", 0)),
        "n_posterior_feasible": int(result.get("n_posterior_feasible", 0)),
        "wall_time_sec": float(time.time() - started),
        "algorithm_time_sec": float(result["total_time_sec"]),
        "stage_times": result.get("stage_times", {}),
        "variance_diagnostics": result.get("variance", {}),
        "variance_calibration_audit": variance_calibration,
        "variance_log_rmse": variance_calibration["log_variance_rmse"],
        "certified_variance_log_rmse": variance_calibration[
            "certified_log_variance_rmse"],
        "median_predicted_true_variance_ratio": variance_calibration[
            "median_predicted_true_ratio"],
        "median_certified_true_variance_ratio": variance_calibration[
            "median_certified_true_ratio"],
        "variance_upper_coverage": variance_calibration[
            "variance_upper_coverage"],
        "candidate_source_counts": result.get("candidate_source_counts", {}),
        "truth_pool_diagnostics": result.get("truth_pool_diagnostics", {}),
        "boundary_raw_pool_truth_diagnostics": result.get(
            "boundary_raw_pool_truth_diagnostics", {}),
        "boundary_coordinate_proposal": result.get(
            "boundary_coordinate_proposal", {}),
        "llm_prior": result.get("llm_prior", {}),
        "x_recommended": result["x_recommended"],
    }


def summarize(rows):
    grouped = {}
    for row in rows:
        N_value = row.get("N")
        if N_value is None:
            key = row["variant"]
        else:
            key = f"N{int(N_value)}:{row['variant']}"
        grouped.setdefault(key, []).append(row)
    out = {}
    for variant, items in grouped.items():
        out[variant] = {
            "variant": variant,
            "N": items[0].get("N"),
            "line": items[0]["line"],
            "heldout": items[0]["heldout"],
            "basis_label": items[0].get("basis_label", ""),
            "state_basis_enabled": bool(items[0].get("state_basis_enabled", False)),
            "state_basis_mode": items[0].get("state_basis_mode", ""),
            "constraint_state_basis_mode": items[0].get(
                "constraint_state_basis_mode", ""),
            "transfer_cell": items[0].get("transfer_cell", items[0]["line"]),
            "n_runs": len(items),
            "audit_admissible_mainline_rate": mean_bool(
                row["audit_admissible_mainline"] for row in items),
            "true_feasible_rate": mean_bool(row["true_feasible"] for row in items),
            "posterior_feasible_rate": mean_bool(row["posterior_feasible"] for row in items),
            "false_feasible_rate": mean_bool(row["false_feasible"] for row in items),
            "initial_has_true_feasible_rate": finite_stats(
                row.get("initial_has_true_feasible", None) for row in items),
            "initial_true_feasible_count": finite_stats(
                row.get("initial_true_feasible_count", None) for row in items),
            "initial_true_feasible_rate": finite_stats(
                row.get("initial_true_feasible_rate", None) for row in items),
            "initial_best_feasible_regret": finite_stats(
                row.get("initial_best_feasible_regret", None) for row in items),
            "adaptive_rescue_rate": finite_stats(
                row.get("adaptive_rescue", None) for row in items),
            "adaptive_loss_rate": finite_stats(
                row.get("adaptive_loss", None) for row in items),
            "adaptive_preservation_rate": finite_stats(
                row.get("adaptive_preservation", None) for row in items),
            "adaptive_improves_initial_best_rate": finite_stats(
                row.get("adaptive_improves_initial_best", None)
                for row in items),
            "adaptive_regret_change": finite_stats(
                row.get("adaptive_regret_change", None) for row in items),
            "adaptive_replication_selected_count": finite_stats(
                row.get("adaptive_replication_selected_count", None)
                for row in items),
            "adaptive_new_point_selected_count": finite_stats(
                row.get("adaptive_new_point_selected_count", None)
                for row in items),
            "posterior_dominance_switch_count": finite_stats(
                row.get("posterior_dominance_switch_count", None)
                for row in items),
            "posterior_dominance_terminal_used_rate": finite_stats(
                row.get("posterior_dominance_terminal_used", None)
                for row in items),
            "posterior_certificate_vacuous_rate": finite_stats(
                row.get("posterior_certificate_vacuous", None)
                for row in items),
            "posterior_certified_evaluated_count": finite_stats(
                row.get("posterior_certified_evaluated_count", None)
                for row in items),
            "certificate_precision": finite_stats(
                row.get("certificate_precision", None) for row in items),
            "certificate_recall_on_evaluated_feasible": finite_stats(
                row.get("certificate_recall_on_evaluated_feasible", None)
                for row in items),
            "false_certificate_count": finite_stats(
                row.get("false_certificate_count", None) for row in items),
            "initial_boundary_bracket_generated": finite_stats(
                row.get("initial_boundary_bracket_generated", None)
                for row in items),
            "initial_mandatory_universal_generated": finite_stats(
                row.get("initial_mandatory_universal_generated", None)
                for row in items),
            "simple_regret": finite_stats(row["simple_regret"] for row in items),
            "feasible_simple_regret": finite_stats(
                row["feasible_simple_regret"] for row in items),
            "constraint_violation": finite_stats(
                row["constraint_violation"] for row in items),
            "true_chance_margin": finite_stats(
                row["true_chance_margin"] for row in items),
            "wall_time_sec": finite_stats(row["wall_time_sec"] for row in items),
            "llm_prior_ok_count": finite_stats(
                (row.get("llm_prior") or {}).get("ok_count", 0) for row in items),
            "llm_prior_selected_count": finite_stats(
                (row.get("llm_prior") or {}).get("selected_count", 0) for row in items),
            "llm_prior_gate_mean": finite_stats(
                (row.get("llm_prior") or {}).get("gate_mean", 0.0) for row in items),
            "pool_has_true_feasible_rate": finite_stats(
                (row.get("truth_pool_diagnostics") or {}).get(
                    "pool_has_true_feasible_rate", None)
                for row in items),
            "pool_missed_true_feasible_rate": finite_stats(
                (row.get("truth_pool_diagnostics") or {}).get(
                    "missed_true_feasible_rate", None)
                for row in items),
            "pool_has_true_safe_good_rate": finite_stats(
                (row.get("truth_pool_diagnostics") or {}).get(
                    "pool_has_true_safe_good_rate", None)
                for row in items),
            "pool_selected_true_feasible_rate": finite_stats(
                (row.get("truth_pool_diagnostics") or {}).get(
                    "selected_true_feasible_rate", None)
                for row in items),
            "pool_best_true_feasible_posterior_margin": finite_stats(
                (row.get("truth_pool_diagnostics") or {}).get(
                    "mean_best_true_feasible_posterior_margin", None)
                for row in items),
            "pool_best_true_feasible_posterior_feasible_rate": finite_stats(
                (row.get("truth_pool_diagnostics") or {}).get(
                    "best_true_feasible_posterior_feasible_rate", None)
                for row in items),
            "recommendation_has_true_feasible": finite_stats(
                row.get("recommendation_has_true_feasible", None) for row in items),
            "recommendation_missed_true_feasible": finite_stats(
                row.get("recommendation_missed_true_feasible", None) for row in items),
            "recommendation_best_true_feasible_posterior_margin": finite_stats(
                row.get("recommendation_best_true_feasible_posterior_margin", None)
                for row in items),
            "recommendation_best_true_feasible_posterior_feasible": finite_stats(
                row.get("recommendation_best_true_feasible_posterior_feasible", None)
                for row in items),
            "recommendation_best_true_feasible_decision_margin": finite_stats(
                row.get("recommendation_best_true_feasible_decision_margin", None)
                for row in items),
            "recommendation_best_true_feasible_decision_feasible": finite_stats(
                row.get("recommendation_best_true_feasible_decision_feasible", None)
                for row in items),
            "recommendation_selected_decision_margin": finite_stats(
                row.get("recommendation_selected_decision_margin", None)
                for row in items),
            "source_mean_prior_used": finite_stats(
                row.get("source_mean_prior_used", None) for row in items),
            "source_mean_prior_n_feasible": finite_stats(
                row.get("source_mean_prior_n_feasible", None) for row in items),
            "source_mean_prior_guard_used": finite_stats(
                row.get("source_mean_prior_guard_used", None) for row in items),
            "source_mean_prior_ranker_used": finite_stats(
                row.get("source_mean_prior_ranker_used", None) for row in items),
            "source_mean_prior_guard_n_feasible": finite_stats(
                row.get("source_mean_prior_guard_n_feasible", None) for row in items),
            "calibrated_recommendation_rejected_by_observed": finite_stats(
                row.get("calibrated_recommendation_rejected_by_observed", None)
                for row in items),
            "observed_incumbent_used": finite_stats(
                row.get("observed_incumbent_used", None) for row in items),
            "observed_incumbent_chance_margin": finite_stats(
                row.get("observed_incumbent_chance_margin", None) for row in items),
            "certification_calibration_n_used": finite_stats(
                row.get("certification_calibration_n_used", None) for row in items),
            "certification_calibration_n_feasible": finite_stats(
                row.get("certification_calibration_n_feasible", None)
                for row in items),
            "recommendation_best_true_feasible_mu_con": finite_stats(
                row.get("recommendation_best_true_feasible_mu_con", None)
                for row in items),
            "recommendation_best_true_feasible_epistemic_var": finite_stats(
                row.get("recommendation_best_true_feasible_epistemic_var", None)
                for row in items),
            "recommendation_best_true_feasible_aleatoric_var": finite_stats(
                row.get("recommendation_best_true_feasible_aleatoric_var", None)
                for row in items),
            "recommendation_best_true_feasible_theory_margin": finite_stats(
                row.get("recommendation_best_true_feasible_theory_margin", None)
                for row in items),
            "recommendation_best_true_feasible_calibrated_margin": finite_stats(
                row.get("recommendation_best_true_feasible_calibrated_margin", None)
                for row in items),
            "recommendation_best_true_feasible_calibrated_rec_margin": finite_stats(
                row.get(
                    "recommendation_best_true_feasible_calibrated_rec_margin",
                    None,
                )
                for row in items),
            "recommendation_best_true_feasible_calibrated_rec_objective": finite_stats(
                row.get(
                    "recommendation_best_true_feasible_calibrated_rec_objective",
                    None,
                )
                for row in items),
            "recommendation_best_true_feasible_calibrated_rec_leverage": finite_stats(
                row.get(
                    "recommendation_best_true_feasible_calibrated_rec_leverage",
                    None,
                )
                for row in items),
            "recommendation_best_true_feasible_source_margin": finite_stats(
                row.get("recommendation_best_true_feasible_source_margin", None)
                for row in items),
            "recommendation_selected_mu_con": finite_stats(
                row.get("recommendation_selected_mu_con", None) for row in items),
            "recommendation_selected_epistemic_var": finite_stats(
                row.get("recommendation_selected_epistemic_var", None)
                for row in items),
            "recommendation_selected_aleatoric_var": finite_stats(
                row.get("recommendation_selected_aleatoric_var", None)
                for row in items),
            "recommendation_selected_theory_margin": finite_stats(
                row.get("recommendation_selected_theory_margin", None)
                for row in items),
            "recommendation_selected_calibrated_margin": finite_stats(
                row.get("recommendation_selected_calibrated_margin", None)
                for row in items),
            "recommendation_selected_calibrated_rec_margin": finite_stats(
                row.get("recommendation_selected_calibrated_rec_margin", None)
                for row in items),
            "recommendation_selected_calibrated_rec_objective": finite_stats(
                row.get("recommendation_selected_calibrated_rec_objective", None)
                for row in items),
            "recommendation_selected_calibrated_rec_leverage": finite_stats(
                row.get("recommendation_selected_calibrated_rec_leverage", None)
                for row in items),
            "recommendation_selected_source_margin": finite_stats(
                row.get("recommendation_selected_source_margin", None)
                for row in items),
            "recommendation_calibration_n_feasible": finite_stats(
                row.get("recommendation_calibration_n_feasible", None)
                for row in items),
            "recommendation_calibration_sigma": finite_stats(
                row.get("recommendation_calibration_sigma", None) for row in items),
            "recommendation_calibration_selected_ridge": finite_stats(
                row.get("recommendation_calibration_selected_ridge", None)
                for row in items),
            "recommendation_calibration_effective_rank": finite_stats(
                row.get("recommendation_calibration_effective_rank", None)
                for row in items),
            "recommendation_calibration_effective_rank_cap": finite_stats(
                row.get("recommendation_calibration_effective_rank_cap", None)
                for row in items),
            "recommendation_calibration_rank_cap_satisfied_rate": finite_stats(
                row.get("recommendation_calibration_rank_cap_satisfied", None)
                for row in items),
            "n_calibration_guarded": finite_stats(
                row.get("n_calibration_guarded", None) for row in items),
            "n_calibration_certified_guarded": finite_stats(
                row.get("n_calibration_certified_guarded", None) for row in items),
            "n_calibration_feasible": finite_stats(
                row.get("n_calibration_feasible", None) for row in items),
            "n_calibration_raw_feasible": finite_stats(
                row.get("n_calibration_raw_feasible", None) for row in items),
            "calibrated_constraint_margin": finite_stats(
                row.get("calibrated_constraint_margin", None) for row in items),
            "calibrated_guarded_constraint_margin": finite_stats(
                row.get("calibrated_guarded_constraint_margin", None)
                for row in items),
            "calibration_selected_leverage": finite_stats(
                row.get("calibration_selected_leverage", None) for row in items),
            "calibration_selected_theory_margin": finite_stats(
                row.get("calibration_selected_theory_margin", None)
                for row in items),
            "calibration_min_leverage": finite_stats(
                row.get("calibration_min_leverage", None) for row in items),
            "calibration_median_leverage": finite_stats(
                row.get("calibration_median_leverage", None) for row in items),
            "calibration_min_theory_margin": finite_stats(
                row.get("calibration_min_theory_margin", None) for row in items),
        }
    return out


def flatten_summary(summary):
    row = {
        "variant": summary["variant"],
        "N": summary.get("N"),
        "line": summary["line"],
        "heldout": summary["heldout"],
        "basis_label": summary.get("basis_label", ""),
        "state_basis_enabled": summary.get("state_basis_enabled", False),
        "state_basis_mode": summary.get("state_basis_mode", ""),
        "constraint_state_basis_mode": summary.get("constraint_state_basis_mode", ""),
        "transfer_cell": summary.get("transfer_cell", summary["line"]),
        "n_runs": summary["n_runs"],
        "audit_admissible_mainline_rate": summary["audit_admissible_mainline_rate"],
        "true_feasible_rate": summary["true_feasible_rate"],
        "posterior_feasible_rate": summary["posterior_feasible_rate"],
        "false_feasible_rate": summary["false_feasible_rate"],
    }
    for metric in (
        "simple_regret",
        "feasible_simple_regret",
        "constraint_violation",
        "true_chance_margin",
        "wall_time_sec",
        "initial_has_true_feasible_rate",
        "initial_true_feasible_count",
        "initial_true_feasible_rate",
        "initial_boundary_bracket_generated",
        "initial_mandatory_universal_generated",
        "adaptive_replication_selected_count",
        "adaptive_new_point_selected_count",
        "posterior_dominance_switch_count",
        "posterior_dominance_terminal_used_rate",
        "llm_prior_ok_count",
        "llm_prior_selected_count",
        "llm_prior_gate_mean",
        "pool_has_true_feasible_rate",
        "pool_missed_true_feasible_rate",
        "pool_has_true_safe_good_rate",
        "pool_selected_true_feasible_rate",
        "pool_best_true_feasible_posterior_margin",
        "pool_best_true_feasible_posterior_feasible_rate",
        "recommendation_has_true_feasible",
        "recommendation_missed_true_feasible",
        "recommendation_best_true_feasible_posterior_margin",
        "recommendation_best_true_feasible_posterior_feasible",
        "recommendation_best_true_feasible_decision_margin",
        "recommendation_best_true_feasible_decision_feasible",
        "recommendation_selected_decision_margin",
        "source_mean_prior_used",
        "source_mean_prior_n_feasible",
        "source_mean_prior_guard_used",
        "source_mean_prior_ranker_used",
        "source_mean_prior_guard_n_feasible",
        "calibrated_recommendation_rejected_by_observed",
        "observed_incumbent_used",
        "observed_incumbent_chance_margin",
        "certification_calibration_n_used",
        "certification_calibration_n_feasible",
        "recommendation_best_true_feasible_mu_con",
        "recommendation_best_true_feasible_epistemic_var",
        "recommendation_best_true_feasible_aleatoric_var",
        "recommendation_best_true_feasible_theory_margin",
        "recommendation_best_true_feasible_calibrated_margin",
        "recommendation_best_true_feasible_calibrated_rec_margin",
        "recommendation_best_true_feasible_calibrated_rec_objective",
        "recommendation_best_true_feasible_calibrated_rec_leverage",
        "recommendation_best_true_feasible_source_margin",
        "recommendation_selected_mu_con",
        "recommendation_selected_epistemic_var",
        "recommendation_selected_aleatoric_var",
        "recommendation_selected_theory_margin",
        "recommendation_selected_calibrated_margin",
        "recommendation_selected_calibrated_rec_margin",
        "recommendation_selected_calibrated_rec_objective",
        "recommendation_selected_calibrated_rec_leverage",
        "recommendation_selected_source_margin",
        "recommendation_calibration_n_feasible",
        "recommendation_calibration_sigma",
        "recommendation_calibration_selected_ridge",
        "recommendation_calibration_effective_rank",
        "recommendation_calibration_effective_rank_cap",
        "recommendation_calibration_rank_cap_satisfied_rate",
        "n_calibration_guarded",
        "n_calibration_certified_guarded",
        "n_calibration_feasible",
        "n_calibration_raw_feasible",
        "calibrated_constraint_margin",
        "calibrated_guarded_constraint_margin",
        "task_adaptive_violation_probability",
        "task_adaptive_expected_violation_loss",
        "task_adaptive_objective_loss",
        "task_adaptive_robust_component",
        "task_adaptive_empirical_component",
        "task_adaptive_total_loss",
        "task_adaptive_expected_empirical_trust",
        "task_adaptive_prequential_sigma",
        "task_adaptive_loo_sigma",
        "task_adaptive_conformal_sigma",
        "task_adaptive_empirical_hvd_variance",
        "calibration_selected_leverage",
        "calibration_selected_theory_margin",
        "calibration_min_leverage",
        "calibration_median_leverage",
        "calibration_min_theory_margin",
    ):
        for key, value in summary.get(metric, {}).items():
            row[f"{metric}_{key}"] = value
    return row


def run(args):
    heldouts = parse_csv(args.heldouts) or parse_csv(args.domains)
    lines = parse_csv(args.lines)
    seeds = parse_csv(args.seeds, int)
    if not seeds:
        seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    args_dict = vars(args).copy()
    if args.basis_pair_grid:
        basis_options = [
            (
                "same_raw_state",
                True,
                str(args.state_basis_mode),
                "",
            ),
            (
                "split_constraint_state",
                True,
                str(args.state_basis_mode),
                "state",
            ),
        ]
    elif args.basis_grid:
        basis_options = [
            ("raw", False, str(args.state_basis_mode), str(args.constraint_state_basis_mode)),
            ("state", True, str(args.state_basis_mode), str(args.constraint_state_basis_mode)),
        ]
    else:
        basis_options = [
            (
                "state" if bool(args.use_state_basis) else "raw",
                bool(args.use_state_basis),
                str(args.state_basis_mode),
                str(args.constraint_state_basis_mode),
            )
        ]
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else None
    resume_paths = []
    if checkpoint_path:
        resume_paths.append(checkpoint_path)
    for path in parse_csv(getattr(args, "resume_completed_from", "")):
        extra = Path(path)
        if extra not in resume_paths:
            resume_paths.append(extra)
    existing_rows = []
    completed = set()
    if args.resume_completed:
        for resume_path in resume_paths:
            if not resume_path.exists():
                continue
            for line in resume_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (
                    str(row["heldout"]),
                    str(row["line"]),
                    int(row["seed"]),
                    str(row.get(
                        "basis_label",
                        "state" if row.get("state_basis_enabled", True) else "raw",
                    )),
                )
                if key not in completed:
                    existing_rows.append(row)
                    completed.add(key)
            print(
                f"[lodo] resumed_rows={len(existing_rows)} "
                f"from {resume_path}",
                flush=True,
            )
    tasks = []
    shard_index = int(getattr(args, "task_shard_index", 0))
    num_shards = max(1, int(getattr(args, "task_num_shards", 1)))
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(
            f"--task_shard_index must be in [0,{num_shards}); got {shard_index}"
        )
    task_ordinal = 0
    for heldout in heldouts:
        for line in lines:
            for seed in seeds:
                for (
                    basis_label,
                    use_state_basis,
                    state_basis_mode,
                    constraint_state_basis_mode,
                ) in basis_options:
                    belongs_to_shard = (task_ordinal % num_shards) == shard_index
                    task_ordinal += 1
                    if not belongs_to_shard:
                        continue
                    key = (str(heldout), str(line), int(seed), str(basis_label))
                    if key in completed:
                        continue
                    task_args = dict(args_dict)
                    task_args["basis_label"] = str(basis_label)
                    task_args["use_state_basis"] = bool(use_state_basis)
                    task_args["state_basis_mode"] = str(state_basis_mode)
                    task_args["constraint_state_basis_mode"] = str(
                        constraint_state_basis_mode)
                    tasks.append({
                        "args": task_args,
                        "heldout": heldout,
                        "line": line,
                        "seed": seed,
                    })
    rows = list(existing_rows)
    total = len(tasks)
    print(
        f"[lodo] tasks={total} resumed={len(existing_rows)} jobs={args.jobs} "
        f"shard={shard_index}/{num_shards}",
        flush=True,
    )
    if checkpoint_path:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    def save_row(row):
        if not checkpoint_path:
            return
        with checkpoint_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(json_safe(row), sort_keys=True) + "\n")
            fh.flush()

    if args.jobs <= 1:
        for idx, task in enumerate(tasks):
            print(
                f"[{idx}/{total}] start line={task['line']} "
                f"heldout={task['heldout']} seed={task['seed']} "
                f"basis={task['args'].get('basis_label', '')}",
                flush=True,
            )
            row = run_one(task)
            rows.append(row)
            save_row(row)
            print(
                f"Step {idx + 1}/{total} [lodo-run] done "
                f"line={task['line']} heldout={task['heldout']} "
                f"seed={task['seed']} "
                f"basis={task['args'].get('basis_label', '')}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=int(args.jobs)) as executor:
            futures = {executor.submit(run_one, task): task for task in tasks}
            done = 0
            for future in as_completed(futures):
                task = futures[future]
                row = future.result()
                rows.append(row)
                save_row(row)
                done += 1
                print(
                    f"Step {done}/{total} [lodo-run] "
                    f"[{done}/{total}] done line={task['line']} "
                    f"heldout={task['heldout']} seed={task['seed']} "
                    f"basis={task['args'].get('basis_label', '')}",
                    flush=True,
                )
    summaries = summarize(rows)
    return {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": json_safe(args_dict),
        "rows": rows,
        "summary": summaries,
    }


def write_outputs(args, result):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.out_prefix or f"lodo_meta_prior_{time.strftime('%Y%m%d_%H%M%S')}"
    json_path = out_dir / f"{prefix}.json"
    rows_path = out_dir / f"{prefix}_rows.csv"
    summary_path = out_dir / f"{prefix}_summary.csv"
    json_path.write_text(json.dumps(json_safe(result), indent=2), encoding="utf-8")
    write_csv(rows_path, result["rows"])
    write_csv(summary_path, [flatten_summary(v) for v in result["summary"].values()])
    return {"json": str(json_path), "rows_csv": str(rows_path), "summary_csv": str(summary_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domains",
        default="FactorShockStatePolicyRZDT1,InventorySupplyChain,QueueResourceControl",
    )
    parser.add_argument("--heldouts", default="")
    parser.add_argument("--lines", default="strict,lodo,domain")
    parser.add_argument("--d", type=int, default=50)
    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=0.04)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--target_shared_shock_scale", type=float, default=1.0)
    parser.add_argument("--variance_audit_size", type=int, default=128)
    parser.add_argument("--weights", default="0.5,0.5")
    parser.add_argument("--N", type=int, default=30)
    parser.add_argument("--n0", type=int, default=8)
    parser.add_argument(
        "--initial_design",
        choices=("auto", "common_sobol"),
        default="auto",
    )
    parser.add_argument("--K1", type=int, default=25)
    parser.add_argument("--K2", type=int, default=0)
    parser.add_argument("--posterior_pool_size", type=int, default=300)
    parser.add_argument("--posterior_keep", type=int, default=15)
    parser.add_argument("--axis_candidate_count", type=int, default=-1)
    parser.add_argument("--structured_candidate_count", type=int, default=0)
    parser.add_argument("--state_candidate_count", type=int, default=12)
    parser.add_argument("--state_inverse_pool_size", type=int, default=300)
    parser.add_argument("--state_inverse_neighbors", type=int, default=1)
    parser.add_argument("--n_thr", type=int, default=5)
    parser.add_argument("--eval_pool_size", type=int, default=300)
    parser.add_argument("--variance_mode", default="factor")
    parser.add_argument(
        "--hvd_use_cumulative_provider",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--lambda_feas", type=float, default=0.25)
    parser.add_argument("--lambda_var", type=float, default=0.25)
    parser.add_argument("--lambda_mean", type=float, default=0.10)
    parser.add_argument("--lambda_constraint_epistemic", type=float, default=0.0)
    parser.add_argument("--lambda_coupling", type=float, default=0.05)
    parser.add_argument("--beta_g", type=float, default=2.0)
    parser.add_argument("--certification_mode", default="theory")
    parser.add_argument("--use_state_basis", dest="use_state_basis", action="store_true", default=True)
    parser.add_argument("--disable_state_basis", dest="use_state_basis", action="store_false")
    parser.add_argument(
        "--basis_grid",
        action="store_true",
        help="Run raw/state basis variants for the core transfer ablation table.",
    )
    parser.add_argument(
        "--basis_pair_grid",
        action="store_true",
        help=(
            "Run same raw+state mean basis versus split objective raw+state / "
            "constraint state basis."
        ),
    )
    parser.add_argument("--state_basis_mode", default="raw+state")
    parser.add_argument("--constraint_state_basis_mode", default="")
    parser.add_argument("--raw_basis_dim", type=int, default=32)
    parser.add_argument("--raw_projection_seed", type=int, default=314159)
    parser.add_argument("--numeric_backend", default="numpy")
    parser.add_argument("--numeric_backend_device", default="auto")
    parser.add_argument("--torch_dtype", default="float64")
    parser.add_argument("--torch_min_rows", type=int, default=128)
    parser.add_argument("--encoder_kind", default="synthetic")
    parser.add_argument("--encoder_latent_dim", type=int, default=8)
    parser.add_argument("--encoder_fit_pool_size", type=int, default=512)
    parser.add_argument("--lf_os_max_library_size", type=int, default=30)
    parser.add_argument("--lf_os_low_frequency_components", type=int, default=8)
    parser.add_argument("--lf_os_max_active", type=int, default=8)
    parser.add_argument("--lf_os_graph_neighbors", type=int, default=12)
    parser.add_argument("--lf_os_residual_floor_scale", type=float, default=0.05)
    parser.add_argument("--acquisition_mode", default="exact_mc")
    parser.add_argument(
        "--decision_backend",
        default="legacy",
        choices=(
            "legacy",
            "additive",
            "exact_kg",
            "n0_best",
            "random",
            "sobol",
            "sobol_new",
            "sobol_hvd_voi",
            "sobol_joint_voi",
            "sobol_exact_joint_voi",
            "certificate_depth_new",
            "risk_ts",
            "bayes_risk_ei",
            "constrained_ei",
            "transfer_utility",
        ),
    )
    parser.add_argument("--decision_risk_penalty", type=float, default=5.0)
    parser.add_argument(
        "--decision_aleatoric_mode",
        choices=("certification_upper", "posterior_central"),
        default="certification_upper",
    )
    parser.add_argument(
        "--decision_violation_loss_mode",
        choices=("positive_part", "failure_probability"),
        default="positive_part",
    )
    parser.add_argument(
        "--decision_ambiguity_mode",
        choices=("kl_robust", "posterior_nominal"),
        default="kl_robust",
    )
    parser.add_argument(
        "--decision_source_utility_weight", type=float, default=1.0)
    parser.add_argument(
        "--decision_backend_seed_offset", type=int, default=470003)
    parser.add_argument(
        "--decision_recommend_observed_only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--exact_kg_mc_samples", type=int, default=2)
    parser.add_argument("--exact_kg_jobs", type=int, default=1)
    parser.add_argument("--exact_kg_parallel_backend", default="thread")
    parser.add_argument("--exact_kg_sampling_mode", default="iid")
    parser.add_argument(
        "--exact_kg_clip_negative",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--exact_kg_use_score", action="store_true")
    parser.add_argument("--exact_kg_blend", type=float, default=0.0)
    parser.add_argument(
        "--exact_kg_terminal_mode", default="hard_certified")
    parser.add_argument(
        "--decision_contract_mode",
        choices=("legacy", "certified_lexicographic"),
        default="legacy",
    )
    parser.add_argument(
        "--implementation-contract-id", "--implementation_contract_id",
        dest="implementation_contract_id", default="unversioned")
    parser.add_argument(
        "--theory-contract-id", "--theory_contract_id",
        dest="theory_contract_id", default="unversioned")
    parser.add_argument(
        "--terminal_bayes_violation_penalty", type=float, default=5.0)
    parser.add_argument(
        "--terminal_frontier_candidate_count", type=int, default=0)
    parser.add_argument(
        "--tcb_v2_mode",
        choices=("off", "shadow", "frontier", "certified"),
        default="off",
    )
    parser.add_argument("--tcb_v2_frontier_count", type=int, default=1)
    parser.add_argument(
        "--tcb_v2_descriptor_mode",
        choices=(
            "raw", "learned_coordinate", "raw+learned_coordinate",
            "learned_risk", "raw+learned_risk",
            "provider_coordinate", "raw+provider_coordinate",
            "provider_risk", "raw+provider_risk",
        ),
        default="learned_risk",
    )
    parser.add_argument(
        "--finalist_terminal_value_mode",
        choices=("model_default", "certified_lexicographic"),
        default="model_default",
    )
    parser.add_argument("--task_posterior_mode", default="off")
    parser.add_argument(
        "--source_discrepancy_update",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--task_posterior_initial_design",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--task_posterior_boundary_bracket_fraction",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--task_posterior_mandatory_universal_count",
        type=int,
        default=0,
    )
    parser.add_argument("--task_posterior_pilot_count", type=int, default=-1)
    parser.add_argument("--task_posterior_temperature", type=float, default=0.5)
    parser.add_argument(
        "--task_posterior_temperature_decay", type=float, default=0.5)
    parser.add_argument(
        "--task_posterior_boundary_score_weight", type=float, default=0.25)
    parser.add_argument(
        "--task_posterior_objective_score_weight", type=float, default=0.25)
    parser.add_argument(
        "--task_posterior_constraint_score_weight", type=float, default=1.0)
    parser.add_argument(
        "--task_posterior_safe_generalized",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--task_posterior_safe_boundary_score_weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--task_posterior_safe_pairwise_score_weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--task_posterior_safe_pairwise_max_history",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--task_posterior_safe_pairwise_probability_floor",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--task_posterior_kl_radius_numerator", type=float, default=0.5)
    parser.add_argument(
        "--task_posterior_confidence_delta", type=float, default=0.05)
    parser.add_argument(
        "--task_posterior_max_kl_radius", type=float, default=4.0)
    parser.add_argument(
        "--task_posterior_robust_certificate_mode",
        choices=("separable", "joint_tangent"),
        default="separable",
    )
    parser.add_argument(
        "--certification_head_authority",
        choices=(
            "task_joint",
            "split_gpr_task_hvd",
            "split_gpr_cumulative_hvd",
        ),
        default="task_joint",
    )
    parser.add_argument(
        "--task_posterior_prior_protection_numerator",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--task_posterior_prior_protection_max", type=float, default=0.5)
    parser.add_argument(
        "--task_posterior_local_kernel_expert",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--task_posterior_candidate_count", type=int, default=0)
    parser.add_argument(
        "--task_posterior_recommendation_count", type=int, default=0)
    parser.add_argument(
        "--task_posterior_proposal_pool_size", type=int, default=1024)
    parser.add_argument(
        "--task_posterior_proposal_exploration", type=float, default=0.10)
    parser.add_argument(
        "--task_posterior_proposal_min_per_expert", type=int, default=2)
    parser.add_argument(
        "--task_posterior_sensitivity_mode", default="off")
    parser.add_argument(
        "--task_variance_posterior_mode",
        choices=("shared", "replication_only"),
        default="shared",
    )
    parser.add_argument(
        "--task_latent_inference_mode",
        choices=("shadow", "authoritative"),
        default="shadow",
    )
    parser.add_argument(
        "--task_latent_calibration_mode",
        choices=("source_profiles", "expert_ridge"),
        default="source_profiles",
    )
    parser.add_argument(
        "--tcb_v2_enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--tcb_v2_coordinate", default="boundary_latent")
    parser.add_argument("--tcb_v2_geometry", default="low_rank_psd")
    parser.add_argument("--tcb_v2_rank", type=int, default=2)
    parser.add_argument("--tcb_v2_ridge", type=float, default=1e-3)
    parser.add_argument(
        "--tcb_v2_domain_penalty", type=float, default=0.5)
    parser.add_argument(
        "--tcb_v2_boundary_temperature", type=float, default=1.0)
    parser.add_argument(
        "--tcb_v2_adaptation_ridge", type=float, default=5.0)
    parser.add_argument(
        "--tcb_v2_upper_alpha", type=float, default=0.01)
    parser.add_argument(
        "--tcb_v2_calibration_prior_df", type=float, default=2.0)
    parser.add_argument(
        "--tcb_v2_hierarchy_iterations", type=int, default=5)
    parser.add_argument(
        "--tcb_v2_effect_ridge", type=float, default=1.0)
    parser.add_argument(
        "--tcb_v2_rotation_mode", choices=("none", "planar"), default="none")
    parser.add_argument("--tcb_v2_rotation_ridge", type=float, default=5.0)
    parser.add_argument("--tcb_v2_target_residual_rank", type=int, default=0)
    parser.add_argument("--tcb_v2_residual_ridge", type=float, default=5.0)
    parser.add_argument("--constraint_uncertain_candidate_count", type=int, default=0)
    parser.add_argument("--constraint_uncertain_pool_size", type=int, default=300)
    parser.add_argument("--constraint_uncertain_state_pool_fraction", type=float, default=0.25)
    parser.add_argument("--constraint_uncertain_use_calibration", action="store_true")
    parser.add_argument("--constraint_epistemic_margin_softening", type=float, default=3.0)
    parser.add_argument("--replication_candidate_count", type=int, default=3)
    parser.add_argument("--replication_max_per_solution", type=int, default=5)
    parser.add_argument(
        "--evaluate_or_replicate_new_action_count", type=int, default=1)
    parser.add_argument(
        "--evaluate_or_replicate_new_action_policy",
        choices=(
            "canonical_sobol",
            "canonical_plus_posterior_risk",
            "canonical_plus_posterior_risk_certificate_coverage",
        ),
        default="canonical_sobol",
    )
    parser.add_argument(
        "--evaluate_or_replicate_baseline_new_action_count",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--policy_improvement_mode",
        choices=("off", "action_superset", "guarded_rollout", "joint"),
        default="off",
    )
    parser.add_argument(
        "--policy_improvement_mc_error_bound", type=float, default=0.0)
    parser.add_argument(
        "--policy_improvement_rollout_depth", type=int, default=1)
    parser.add_argument(
        "--policy_improvement_rollout_max_arms", type=int, default=4)
    parser.add_argument(
        "--policy_improvement_rollout_mc_samples", type=int, default=2)
    parser.add_argument(
        "--policy_improvement_rollout_mc_error_bound", type=float,
        default=0.0,
    )
    parser.add_argument("--replication_margin_softening", type=float, default=3.0)
    parser.add_argument(
        "--adaptive_replication_voi",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--posterior_dominance_enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--posterior_dominance_delta", type=float, default=0.05)
    parser.add_argument(
        "--posterior_dominance_min_mean_gain", type=float, default=0.0)
    parser.add_argument(
        "--posterior_dominance_initialization",
        choices=("risk", "certificate_lexicographic", "certified_only"),
        default="risk",
    )
    parser.add_argument("--certification_recheck_top_k", type=int, default=0)
    parser.add_argument(
        "--certification_recheck_min_replicates", type=int, default=3)
    parser.add_argument(
        "--certification_recheck_soft_margin_scale", type=float, default=2.0)
    parser.add_argument(
        "--certification_recheck_variance_prior_df", type=float, default=2.0)
    parser.add_argument("--finalist_replication_budget", type=int, default=0)
    parser.add_argument("--finalist_replication_count", type=int, default=2)
    parser.add_argument(
        "--finalist_observed_safety_count", type=int, default=1)
    parser.add_argument(
        "--finalist_replication_min_replicates", type=int, default=2)
    parser.add_argument(
        "--finalist_replication_delta", type=float, default=0.05)
    parser.add_argument(
        "--finalist_replication_variance_prior_df", type=float, default=2.0)
    parser.add_argument(
        "--finalist_replication_expert_stratified",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--finalist_replication_adaptive_race",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--finalist_replication_fixed_universe",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--finalist_replication_policy",
        choices=(
            "legacy",
            "commit_before_switch",
            "terminal_kg_1step",
            "terminal_kg_depth3",
        ),
        default="legacy",
    )
    parser.add_argument(
        "--finalist_empirical_override",
        choices=("legacy", "certified_only", "off"),
        default="legacy",
    )
    parser.add_argument("--finalist_terminal_max_arms", type=int, default=4)
    parser.add_argument("--finalist_terminal_mc_samples", type=int, default=2)
    parser.add_argument(
        "--finalist_frontier_policy",
        choices=("legacy", "coverage_reserved", "observed_safety_reserved"),
        default="legacy",
    )
    parser.add_argument(
        "--observed_incumbent_use_replicate_variance",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--safe_interior_candidate_count", type=int, default=0)
    parser.add_argument("--safe_interior_pool_size", type=int, default=300)
    parser.add_argument("--safe_interior_margin", type=float, default=0.0)
    parser.add_argument("--observed_neighbor_candidate_count", type=int, default=0)
    parser.add_argument("--observed_neighbor_radius", type=float, default=0.08)
    parser.add_argument("--observed_neighbor_safe_margin_scale", type=float, default=1.0)
    parser.add_argument("--recommendation_infeasible_penalty", type=float, default=5.0)
    parser.add_argument("--recommendation_infeasible_strategy", default="penalty")
    parser.add_argument("--recommend_observed_only", action="store_true")
    parser.add_argument("--recommendation_slack_initial", type=float, default=0.0)
    parser.add_argument("--recommendation_slack_decay", default="sqrt")
    parser.add_argument("--recommendation_calibration", action="store_true")
    parser.add_argument("--recommendation_calibration_scope", default="refinement")
    parser.add_argument(
        "--recommendation_calibration_max_effective_fraction",
        type=float,
        default=0.35,
    )
    parser.add_argument("--recommendation_calibration_min_obs", type=int, default=8)
    parser.add_argument("--recommendation_calibration_max_leverage", type=float, default=0.0)
    parser.add_argument(
        "--recommendation_calibration_max_theory_margin",
        type=float,
        default=0.0,
    )
    parser.add_argument("--recommendation_observed_fallback", action="store_true")
    parser.add_argument("--observed_incumbent_margin_scale", type=float, default=-0.5)
    parser.add_argument("--certification_calibration", action="store_true")
    parser.add_argument("--certification_calibration_min_obs", type=int, default=8)
    parser.add_argument("--certification_calibration_beta", type=float, default=2.0)
    parser.add_argument("--certification_calibration_policy", default="guarded")
    parser.add_argument("--certification_calibration_max_leverage", type=float, default=10.0)
    parser.add_argument("--certification_calibration_max_theory_margin", type=float, default=0.25)
    parser.add_argument("--certification_calibration_raise_delta", type=float, default=0.10)
    parser.add_argument("--calibration_standardize_features", action="store_true")
    parser.add_argument("--use_source_recommendation_slack", action="store_true")
    parser.add_argument("--source_mean_prior_fallback", action="store_true")
    parser.add_argument("--source_mean_prior_z", type=float, default=1.0)
    parser.add_argument("--source_mean_prior_margin_tol", type=float, default=0.0)
    parser.add_argument("--truth_pool_diagnostics", action="store_true")
    parser.add_argument("--truth_pool_good_regret", type=float, default=0.05)
    parser.add_argument("--truth_pool_max_candidates", type=int, default=0)
    parser.add_argument("--llm_prior_enabled", action="store_true")
    parser.add_argument(
        "--offline_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable external API/network-assisted priors for reproducible evaluation.",
    )
    parser.add_argument("--llm_prior_base_url", default="https://ruoli.dev")
    parser.add_argument("--llm_prior_model", default="gpt-5.4-mini")
    parser.add_argument("--llm_prior_api_key_env", default="SCOLHKG_LLM_API_KEY")
    parser.add_argument("--llm_prior_candidate_count", type=int, default=8)
    parser.add_argument("--llm_prior_inverse_pool_size", type=int, default=256)
    parser.add_argument("--llm_prior_interval", type=int, default=5)
    parser.add_argument("--llm_prior_min_obs", type=int, default=8)
    parser.add_argument("--llm_prior_timeout_sec", type=float, default=30.0)
    parser.add_argument("--llm_prior_gate_floor", type=float, default=0.05)
    parser.add_argument("--llm_prior_max_observations", type=int, default=24)
    parser.add_argument("--runtime_checkpoint_dir", default="")
    parser.add_argument("--runtime_checkpoint_resume", action="store_true")
    parser.add_argument(
        "--runtime_checkpoint_interval",
        type=int,
        default=0,
        help="Checkpoint every N stages; 0 disables runtime checkpoints.",
    )
    parser.add_argument("--evaluate_interval", type=int, default=5)
    parser.add_argument("--progress_logging", action="store_true")
    parser.add_argument("--progress_units_per_iteration", type=int, default=100)
    parser.add_argument("--progress_exact_updates", type=int, default=10)
    parser.add_argument("--source_records_per_domain", type=int, default=96)
    parser.add_argument("--meta_local_dim", type=int, default=3)
    parser.add_argument("--meta_shared_dim", type=int, default=3)
    parser.add_argument(
        "--meta_spectral_orthogonalization",
        choices=("symmetric", "ordered_cholesky", "none"),
        default="symmetric",
    )
    parser.add_argument(
        "--meta_ordered_cumulative_exposure",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--meta_ordered_exposure_max_frequency", type=int, default=8)
    parser.add_argument(
        "--meta_ordered_exposure_active_dim", type=int, default=2)
    parser.add_argument(
        "--meta_ordered_exposure_frequency_penalty", type=float, default=0.10)
    parser.add_argument(
        "--meta_ordered_exposure_basis_mode",
        choices=["full_quadratic", "diagonal_quadratic"],
        default="full_quadratic",
    )
    parser.add_argument(
        "--meta_ordered_exposure_orthogonal_coordinates",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--meta_ordered_exposure_adaptive_sparsity",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--meta_ordered_exposure_replace_local_kernel",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--meta_ordered_exposure_semiparametric_residual",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--meta_ordered_exposure_latent_structure_selection",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--meta_ordered_exposure_group_shared_shrinkage",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--meta_ordered_exposure_group_ridge_learning",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--meta_anchor_count", type=int, default=24)
    parser.add_argument("--meta_kmeans_iters", type=int, default=25)
    parser.add_argument("--meta_soft_temperature", type=float, default=0.75)
    parser.add_argument("--meta_ridge", type=float, default=1e-4)
    parser.add_argument("--meta_boundary_weight", type=float, default=1.0)
    parser.add_argument("--meta_boundary_temperature", type=float, default=1.0)
    parser.add_argument("--meta_variance_weight", type=float, default=0.5)
    parser.add_argument("--meta_feasible_penalty", type=float, default=6.0)
    parser.add_argument("--meta_feasible_bonus", type=float, default=0.15)
    parser.add_argument("--meta_elite_fraction", type=float, default=0.40)
    parser.add_argument("--meta_boundary_fraction", type=float, default=0.35)
    parser.add_argument("--meta_anchor_sampling_temperature", type=float, default=0.0)
    parser.add_argument("--meta_teacher_records_per_domain", type=int, default=96)
    parser.add_argument("--meta_teacher_weight", type=float, default=3.0)
    parser.add_argument("--meta_teacher_pool_size", type=int, default=2048)
    parser.add_argument("--meta_teacher_elite_fraction", type=float, default=0.50)
    parser.add_argument("--meta_teacher_boundary_fraction", type=float, default=0.35)
    parser.add_argument(
        "--meta_teacher_anchor_sampling_temperature",
        type=float,
        default=0.35,
    )
    parser.add_argument("--meta_hvd_noise_floor_scale", type=float, default=0.0)
    parser.add_argument("--meta_teacher_hvd_noise_floor_scale", type=float, default=1.0)
    parser.add_argument("--meta_universal_shape_count", type=int, default=64)
    parser.add_argument(
        "--meta_observable_mean_coordinate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use a source-learned observable eta coordinate for the constraint "
            "mean while retaining psi=(A,N) only for cumulative risk."
        ),
    )
    parser.add_argument(
        "--meta_observable_mean_ridges",
        default="0.01,0.1,1.0,10.0,100.0")
    parser.add_argument(
        "--meta_observable_mean_mode",
        choices=[
            "atoms", "aggregate", "latent", "consensus", "source_affine",
            "source_rank", "boundary_aligned",
        ],
        default="latent",
    )
    parser.add_argument(
        "--meta_observable_mean_latent_dim", type=int, default=2)
    parser.add_argument(
        "--meta_observable_mean_training_target",
        choices=["constraint_mean", "chance_margin"],
        default="constraint_mean",
    )
    parser.add_argument(
        "--meta_observable_mean_input_mode",
        choices=[
            "policy_profile",
            "source_learned_exposure",
            "observable_state_exposure",
            "provider_exposure",
        ],
        default="policy_profile",
    )
    parser.add_argument(
        "--meta_observable_mean_descriptor_mode",
        choices=[
            "ordered",
            "set_invariant",
            "role_aligned",
            "role_transport",
            "role_intervention_transport",
            "role_adaptive_ordered",
            "role_adaptive_set_invariant",
            "exchangeable_equivariant",
        ],
        default="ordered",
    )
    parser.add_argument(
        "--meta_observable_mean_feature_mode",
        choices=["linear", "diagonal_quadratic", "full_quadratic"],
        default="linear",
    )
    parser.add_argument(
        "--meta_observable_mean_latent_transform",
        choices=[
            "identity",
            "source_tanh",
            "source_support_clip",
            "source_support_residual",
        ],
        default="identity",
    )
    parser.add_argument(
        "--meta_observable_mean_target_residual_rank", type=int, default=0)
    parser.add_argument(
        "--meta_observable_mean_target_residual_prior_scale",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--meta_observable_mean_target_residual_pool_size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--meta_observable_mean_target_residual_rcond",
        type=float,
        default=1e-8,
    )
    parser.add_argument(
        "--meta_observable_mean_role_assignment_posterior",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--meta_observable_mean_role_assignment_prior",
        choices=[
            "uniform", "source_geometry", "source_geometry_boundary",
        ],
        default="uniform",
    )
    parser.add_argument(
        "--meta_observable_mean_role_assignment_prior_temperature_scale",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--meta_observable_mean_role_assignment_inactive_variance",
        type=float,
        default=1e-12,
    )
    parser.add_argument(
        "--meta_observable_variance_input_mode",
        choices=["legacy_policy_proxy", "observable_state_exposure"],
        default="legacy_policy_proxy",
        help=(
            "Fit cumulative factor-HVD coordinates from the same observable "
            "state/trajectory exposure used by the separate mean head."
        ),
    )
    parser.add_argument(
        "--source_constraint_mean_coefficient_prior",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Initialize the target constraint GPR from the source-learned "
            "eta coefficient law and condition it on charged target calls."
        ),
    )
    parser.add_argument(
        "--source_constraint_mean_hyperlaw_mode",
        choices=[
            "single_gaussian_draw",
            "shared_low_rank_discrepancy",
            "shared_low_rank_predictive",
            "grouped_task_discrepancy",
            "grouped_task_predictive",
            "grouped_task_deconvolved",
            "grouped_task_deconvolved_predictive",
        ],
        default="single_gaussian_draw",
        help="Choose the frozen source-to-target coefficient hyperlaw.",
    )
    parser.add_argument(
        "--source_constraint_mean_adaptation_mode",
        choices=[
            "frozen", "evidence_mixture", "sequential_evidence_mixture",
            "aggregate_mixture", "sequential_aggregate_mixture",
            "sequential_aggregate_hyperlaw",
            "support_adaptive_aggregate_mixture",
            "sequential_support_adaptive_aggregate_mixture",
        ],
        default="frozen",
        help=(
            "Use a frozen aggregate source coefficient law, a source-domain "
            "mixture, or an aggregate-source-versus-target-null mixture "
            "updated from charged target calls."
        ),
    )
    parser.add_argument(
        "--source_constraint_mean_deviation_mode",
        choices=["raw_independent", "latent_shared"],
        default="raw_independent",
        help=(
            "Represent source residual discrepancy as an independent raw-"
            "policy floor or as uncertainty in the transferable latent "
            "coefficient law plus a finite residual floor."
        ),
    )
    parser.add_argument(
        "--source_constraint_mean_misspecification_mode",
        choices=[
            "none",
            "predictive_scale",
            "predictive_scale_directional",
            "predictive_scale_upper_target",
            "predictive_scale_upper",
            "predictive_sandwich_hc3",
            "predictive_sandwich_hc3_task",
            "predictive_scale_sandwich_hc3",
            "predictive_scale_sandwich_hc3_task",
            "predictive_scale_sandwich_hc3_confidence",
            "predictive_scale_sandwich_hc3_task_confidence",
            "hierarchical_predictive_scale",
            "source_contrast",
        ],
        default="none",
        help=(
            "Conservatively inflate a source mean coefficient law when "
            "charged target residuals reject its predictive distribution."
        ),
    )
    parser.add_argument(
        "--source_constraint_mean_misspecification_prior_df",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--source_constraint_mean_misspecification_ridge",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--source_constraint_mean_misspecification_max_scale",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--source_constraint_mean_misspecification_delta",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--source_constraint_mean_confidence_mode",
        choices=["model", "source_bayes", "source_self_normalized"],
        default="model",
    )
    parser.add_argument(
        "--source_constraint_mean_confidence_delta",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--source_constraint_mean_contrast_scale", type=float, default=1.0)
    parser.add_argument(
        "--source_constraint_mean_role_epistemic_mode",
        choices=["none", "matching_loss", "matching_uncertainty"],
        default="none",
    )
    parser.add_argument(
        "--source_constraint_mean_null_weight", type=float, default=0.5)
    parser.add_argument(
        "--source_constraint_mean_null_geometry",
        choices=["isotropic", "target_pool"],
        default="isotropic",
    )
    parser.add_argument(
        "--source_constraint_mean_null_geometry_ridge",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--source_constraint_mean_evidence_temperature",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--source_constraint_mean_structure_score_mode",
        choices=[
            "marginal_likelihood",
            "loo_predictive",
            "geometry_conditional",
        ],
        default="marginal_likelihood",
    )
    parser.add_argument(
        "--source_constraint_mean_residual_rank_posterior",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--source_constraint_mean_residual_rank_prior",
        default="0.70,0.20,0.10",
    )
    parser.add_argument(
        "--source_constraint_mean_residual_rank_inactive_variance",
        type=float,
        default=1e-12,
    )
    parser.add_argument(
        "--boundary_coordinate_candidate_count", type=int, default=0)
    parser.add_argument(
        "--boundary_coordinate_pool_size", type=int, default=512)
    parser.add_argument(
        "--boundary_coordinate_safe_fraction", type=float, default=0.30)
    parser.add_argument(
        "--boundary_coordinate_boundary_fraction", type=float, default=0.40)
    parser.add_argument(
        "--boundary_coordinate_coverage_fraction", type=float, default=0.30)
    parser.add_argument(
        "--meta_source_observation_mode",
        choices=["analytic", "nominal", "replicated"],
        default="analytic",
        help=(
            "Source-data contract. 'analytic' reproduces privileged V32; "
            "'replicated' learns from ordinary simulator calls only."
        ),
    )
    parser.add_argument(
        "--meta_source_observation_replicates", type=int, default=1)
    parser.add_argument(
        "--meta_source_design_mode",
        choices=["random", "universal_mixture", "shared_uniform"],
        default="random",
    )
    parser.add_argument(
        "--meta_source_universal_fraction", type=float, default=0.75)
    parser.add_argument(
        "--meta_source_consensus_template_count",
        type=int,
        default=0,
        help=(
            "Number of shared universal profiles ranked by ordinary source "
            "chance-margin observations and replayed on the held-out target."
        ),
    )
    parser.add_argument(
        "--meta_component_stage",
        choices=["legacy_all", "coordinate", "spectral", "spectral_hvd"],
        default="spectral_hvd",
        help=(
            "Isolate LODO learning stages. 'spectral' enables only the frozen "
            "source-invariant low-frequency basis; 'spectral_hvd' additionally "
            "transfers a source-fitted cumulative-HVD prior in aligned risk "
            "coordinates."
        ),
    )
    parser.add_argument("--meta_spectral_active_dim", type=int, default=6)
    parser.add_argument("--meta_spectral_max_library_size", type=int, default=64)
    parser.add_argument(
        "--meta_spectral_low_frequency_components", type=int, default=8)
    parser.add_argument(
        "--meta_spectral_low_frequency_prior",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--meta_spectral_graph_neighbors", type=int, default=10)
    parser.add_argument("--meta_spectral_relevance_floor", type=float, default=0.05)
    parser.add_argument("--meta_spectral_gate_boundary_weight", type=float, default=2.0)
    parser.add_argument("--meta_spectral_gate_dangerous_weight", type=float, default=3.0)
    parser.add_argument(
        "--meta_spectral_gate_selection_tolerance", type=float, default=0.02)
    parser.add_argument(
        "--meta_spectral_gate_calibration_quantile", type=float, default=0.90)
    parser.add_argument(
        "--meta_spectral_frequency_adaptation",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--meta_spectral_frequency_cutoffs", default="3,5,8,12")
    parser.add_argument(
        "--meta_spectral_frequency_ridges", default="0.0001,0.01,1.0")
    parser.add_argument(
        "--meta_spectral_frequency_source_penalty", type=float, default=0.05)
    parser.add_argument(
        "--meta_spectral_frequency_temperature", type=float, default=0.5)
    parser.add_argument(
        "--meta_spectral_frequency_refit_interval", type=int, default=5)
    parser.add_argument(
        "--meta_spectral_risk_alignment",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--meta_spectral_alignment_active_dim", type=int, default=4)
    parser.add_argument(
        "--meta_spectral_alignment_subspace_dim", type=int, default=2)
    parser.add_argument(
        "--meta_spectral_alignment_domain_penalty", type=float, default=0.5)
    parser.add_argument(
        "--meta_spectral_alignment_source_procrustes",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--meta_spectral_alignment_target_ridge", type=float, default=5.0)
    parser.add_argument(
        "--meta_spectral_alignment_target_min_gain", type=float, default=0.02)
    parser.add_argument(
        "--meta_spectral_alignment_target_min_bins", type=int, default=3)
    parser.add_argument(
        "--meta_spectral_alignment_refit_interval", type=int, default=5)
    parser.add_argument(
        "--meta_spectral_alignment_source_episodes", type=int, default=0)
    parser.add_argument(
        "--meta_spectral_alignment_admission",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--meta_spectral_alignment_latent_proposals",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--meta_spectral_alignment_inverse_pool_size",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--meta_spectral_alignment_episode_pilot_size", type=int, default=10)
    parser.add_argument(
        "--meta_spectral_alignment_episode_evaluation_size",
        type=int,
        default=24,
    )
    parser.add_argument(
        "--meta_spectral_alignment_episode_ridge", type=float, default=0.1)
    parser.add_argument(
        "--meta_spectral_additive_adaptation",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--meta_spectral_additive_max_groups", type=int, default=8)
    parser.add_argument(
        "--meta_spectral_additive_target_max_groups", type=int, default=2)
    parser.add_argument(
        "--meta_spectral_additive_source_penalty", type=float, default=0.05)
    parser.add_argument(
        "--meta_spectral_additive_complexity_penalty", type=float, default=0.05)
    parser.add_argument(
        "--meta_spectral_additive_temperature", type=float, default=0.5)
    parser.add_argument(
        "--meta_spectral_additive_refit_interval", type=int, default=5)
    parser.add_argument(
        "--meta_spectral_additive_max_saturation_fraction",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--meta_spectral_coefficient_shrinkage",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--meta_spectral_shrinkage_strength", type=float, default=1.0)
    parser.add_argument(
        "--meta_spectral_shrinkage_floor", type=float, default=0.05)
    parser.add_argument(
        "--meta_spectral_adaptive_sparsity",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--meta_spectral_adaptive_min_pip", type=float, default=0.05)
    parser.add_argument("--meta_spectral_adaptive_max_pip", type=float, default=0.95)
    parser.add_argument(
        "--meta_spectral_adaptive_spike_ratio", type=float, default=0.05)
    parser.add_argument("--meta_spectral_adaptive_damping", type=float, default=0.5)
    parser.add_argument("--meta_spectral_adaptive_max_iter", type=int, default=40)
    parser.add_argument(
        "--meta_spectral_adaptive_tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--meta_spectral_adaptive_residual_floor_scale", type=float, default=0.05)
    parser.add_argument(
        "--meta_spectral_adaptive_gate_tolerance", type=float, default=0.05)
    parser.add_argument(
        "--meta_spectral_adaptive_multiplicity_correction",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--meta_spectral_adaptive_max_effective_fraction",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--meta_spectral_adaptive_saturation_fraction",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--meta_coordinate_mode",
        choices=["pca", "stable_supervised"],
        default="pca",
    )
    parser.add_argument("--meta_coordinate_relevance_floor", type=float, default=0.05)
    parser.add_argument("--meta_source_augments", type=int, default=1)
    parser.add_argument(
        "--meta_source_budget_mode",
        choices=["per_episode", "per_base_domain"],
        default="per_episode",
    )
    parser.add_argument(
        "--meta_source_geometry_shift_scale", type=float, default=0.0)
    parser.add_argument(
        "--meta_source_geometry_log_radius_jitter",
        type=float,
        default=0.0,
    )
    parser.add_argument("--meta_source_sigma_jitter", type=float, default=0.20)
    parser.add_argument("--meta_source_alpha_jitter", type=float, default=0.25)
    parser.add_argument("--meta_source_weight_jitter", type=float, default=0.05)
    parser.add_argument("--meta_seed", type=int, default=20260706)
    parser.add_argument(
        "--meta_source_seed_mode",
        choices=["frozen", "per_target"],
        default="frozen",
    )
    parser.add_argument("--meta_proposal_pool_size", type=int, default=512)
    parser.add_argument("--meta_refinement_count", type=int, default=96)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--checkpoint_path", default="")
    parser.add_argument("--resume_completed_from", default="")
    parser.add_argument("--resume_completed", action="store_true")
    parser.add_argument("--task_shard_index", type=int, default=0)
    parser.add_argument("--task_num_shards", type=int, default=1)
    parser.add_argument("--out_dir", default=str(ROOT / "profiles"))
    parser.add_argument("--out_prefix", default="")
    args = parser.parse_args()
    if args.N <= args.n0:
        raise ValueError("--N must be larger than --n0")
    if args.offline_only:
        os.environ["SCOLHKG_OFFLINE"] = "1"
        if args.llm_prior_enabled:
            raise ValueError(
                "--offline_only is incompatible with --llm_prior_enabled"
            )
    result = run(args)
    paths = write_outputs(args, result)
    print(json.dumps(json_safe({"paths": paths, "summary": result["summary"]}), indent=2))


if __name__ == "__main__":
    main()
