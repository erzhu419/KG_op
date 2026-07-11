#!/usr/bin/env python3
"""Submit LODO learned meta-prior SC-OLH-KG suites to CPU scheduler nodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

from scheduler_node_policy import allowed_node_flags, parse_cpu_nodes


DEFAULT_SCHEDULER = Path.home() / ".claude/skills/scheduler/scheduler.py"
DEFAULT_DEPLOY = Path.home() / "mine_code/KG_op_scheduler_deploy"
DEFAULT_SYNC_SCRIPT = Path(__file__).with_name("sync_scolhkg_scheduler_deploy.sh")
PYTHON = "/home/zhengliang01/scheduleurm_work/conda_envs/scomp-py310/bin/python"


LOSS_PRESETS = {
    "calibrated": {
        "meta_boundary_weight": 1.5,
        "meta_boundary_temperature": 1.0,
        "meta_variance_weight": 0.75,
        "meta_feasible_penalty": 7.0,
        "meta_feasible_bonus": 0.20,
        "meta_elite_fraction": 0.40,
        "meta_boundary_fraction": 0.35,
    },
    "strong_boundary": {
        "meta_boundary_weight": 3.0,
        "meta_boundary_temperature": 1.25,
        "meta_variance_weight": 1.0,
        "meta_feasible_penalty": 8.0,
        "meta_feasible_bonus": 0.15,
        "meta_elite_fraction": 0.30,
        "meta_boundary_fraction": 0.50,
    },
    "feasible_guard": {
        "meta_boundary_weight": 1.0,
        "meta_boundary_temperature": 0.75,
        "meta_variance_weight": 0.75,
        "meta_feasible_penalty": 12.0,
        "meta_feasible_bonus": 0.35,
        "meta_elite_fraction": 0.55,
        "meta_boundary_fraction": 0.25,
    },
}


def parse_csv(text, cast=str):
    return [cast(x.strip()) for x in str(text or "").split(",") if x.strip()]


def run_cmd(cmd, dry_run=False):
    print("+", " ".join(map(str, cmd)), flush=True)
    if dry_run:
        return ""
    return subprocess.check_output([str(c) for c in cmd], text=True)


def preset_flags(name):
    if name not in LOSS_PRESETS:
        raise ValueError(f"unknown loss preset {name!r}; choices={sorted(LOSS_PRESETS)}")
    return " ".join(f"--{key} {value}" for key, value in LOSS_PRESETS[name].items())


def checkpoint_root(args, run_id):
    return args.deploy / "SC-OLH-KG" / "checkpoints" / run_id / "lodo_meta"


def suite_command(args, run_id, N, preset, shard_index=0, num_shards=1):
    ckpt_root = checkpoint_root(args, run_id)
    base_prefix = f"lodo_meta_{run_id}_N{N}_{preset}"
    if int(num_shards) > 1:
        shard_suffix = f"_shard{int(shard_index):02d}of{int(num_shards):02d}"
    else:
        shard_suffix = ""
    prefix = base_prefix + shard_suffix
    base_checkpoint = ckpt_root / (base_prefix + ".jsonl")
    resume_completed_from = (
        f"--resume_completed_from {base_checkpoint} "
        if int(num_shards) > 1
        else ""
    )
    return (
        f"{PYTHON} performance/benchmark_lodo_meta_prior.py "
        f"--domains {args.domains} --heldouts {args.heldouts} --lines {args.lines} "
        f"--d {args.d} --L {args.L} --sigma {args.sigma} --alpha {args.alpha} "
        f"--weights {args.weights} --N {N} --n0 {args.n0} --K1 {args.K1} --K2 {args.K2} "
        f"--posterior_pool_size {args.posterior_pool_size} "
        f"--posterior_keep {args.posterior_keep} "
        f"--axis_candidate_count {args.axis_candidate_count} "
        f"--structured_candidate_count {args.structured_candidate_count} "
        f"--state_candidate_count {args.state_candidate_count} "
        f"--state_inverse_pool_size {args.state_inverse_pool_size} "
        f"--state_inverse_neighbors {args.state_inverse_neighbors} "
        f"--n_thr {args.n_thr} --eval_pool_size {args.eval_pool_size} "
        f"--variance_mode factor --lambda_feas {args.lambda_feas} "
        f"--lambda_var {args.lambda_var} --lambda_mean {args.lambda_mean} "
        f"--lambda_constraint_epistemic {args.lambda_constraint_epistemic} "
        f"--lambda_coupling {args.lambda_coupling} --beta_g {args.beta_g} "
        f"--certification_mode theory --use_state_basis "
        f"--state_basis_mode {args.state_basis_mode} "
        f"--constraint_state_basis_mode {args.constraint_state_basis_mode} "
        f"{'--basis_pair_grid ' if args.basis_pair_grid else ''}"
        f"--raw_basis_dim {args.raw_basis_dim} "
        f"--raw_projection_seed {args.raw_projection_seed} "
        f"--numeric_backend {args.numeric_backend} "
        f"--numeric_backend_device {args.numeric_backend_device} "
        f"--torch_dtype {args.torch_dtype} --torch_min_rows {args.torch_min_rows} "
        f"--acquisition_mode {args.acquisition_mode} "
        f"--exact_kg_mc_samples {args.exact_kg_mc_samples} "
        f"--exact_kg_jobs {args.exact_kg_jobs} "
        f"--exact_kg_parallel_backend {args.exact_kg_parallel_backend} "
        f"--exact_kg_sampling_mode {args.exact_kg_sampling_mode} "
        f"{'--exact_kg_clip_negative ' if args.exact_kg_clip_negative else '--no-exact_kg_clip_negative ' }"
        f"--exact_kg_blend {args.exact_kg_blend} "
        f"--exact_kg_terminal_mode {args.exact_kg_terminal_mode} "
        f"--terminal_bayes_violation_penalty "
        f"{args.terminal_bayes_violation_penalty} "
        f"--terminal_frontier_candidate_count "
        f"{args.terminal_frontier_candidate_count} "
        f"--task_posterior_mode {args.task_posterior_mode} "
        f"{'--task_posterior_initial_design' if args.task_posterior_initial_design else '--no-task_posterior_initial_design'} "
        f"--task_posterior_boundary_bracket_fraction "
        f"{args.task_posterior_boundary_bracket_fraction} "
        f"--task_posterior_mandatory_universal_count "
        f"{args.task_posterior_mandatory_universal_count} "
        f"--task_posterior_pilot_count {args.task_posterior_pilot_count} "
        f"--task_posterior_temperature {args.task_posterior_temperature} "
        f"--task_posterior_temperature_decay "
        f"{args.task_posterior_temperature_decay} "
        f"--task_posterior_boundary_score_weight "
        f"{args.task_posterior_boundary_score_weight} "
        f"--task_posterior_objective_score_weight "
        f"{args.task_posterior_objective_score_weight} "
        f"--task_posterior_constraint_score_weight "
        f"{args.task_posterior_constraint_score_weight} "
        f"{'--task_posterior_safe_generalized ' if args.task_posterior_safe_generalized else '--no-task_posterior_safe_generalized '}"
        f"--task_posterior_safe_boundary_score_weight "
        f"{args.task_posterior_safe_boundary_score_weight} "
        f"--task_posterior_safe_pairwise_score_weight "
        f"{args.task_posterior_safe_pairwise_score_weight} "
        f"--task_posterior_safe_pairwise_max_history "
        f"{args.task_posterior_safe_pairwise_max_history} "
        f"--task_posterior_safe_pairwise_probability_floor "
        f"{args.task_posterior_safe_pairwise_probability_floor} "
        f"--task_posterior_kl_radius_numerator "
        f"{args.task_posterior_kl_radius_numerator} "
        f"--task_posterior_confidence_delta "
        f"{args.task_posterior_confidence_delta} "
        f"--task_posterior_max_kl_radius "
        f"{args.task_posterior_max_kl_radius} "
        f"--task_posterior_prior_protection_numerator "
        f"{args.task_posterior_prior_protection_numerator} "
        f"--task_posterior_prior_protection_max "
        f"{args.task_posterior_prior_protection_max} "
        f"{'--task_posterior_local_kernel_expert ' if args.task_posterior_local_kernel_expert else '--no-task_posterior_local_kernel_expert '}"
        f"--task_posterior_candidate_count "
        f"{args.task_posterior_candidate_count} "
        f"--task_posterior_recommendation_count "
        f"{args.task_posterior_recommendation_count} "
        f"--task_posterior_proposal_pool_size "
        f"{args.task_posterior_proposal_pool_size} "
        f"--task_posterior_proposal_exploration "
        f"{args.task_posterior_proposal_exploration} "
        f"--task_posterior_proposal_min_per_expert "
        f"{args.task_posterior_proposal_min_per_expert} "
        f"--task_posterior_sensitivity_mode "
        f"{args.task_posterior_sensitivity_mode} "
        f"--constraint_uncertain_candidate_count {args.constraint_uncertain_candidate_count} "
        f"--constraint_uncertain_pool_size {args.constraint_uncertain_pool_size} "
        f"--constraint_uncertain_state_pool_fraction "
        f"{args.constraint_uncertain_state_pool_fraction} "
        f"--constraint_epistemic_margin_softening "
        f"{args.constraint_epistemic_margin_softening} "
        f"--replication_candidate_count {args.replication_candidate_count} "
        f"--replication_max_per_solution {args.replication_max_per_solution} "
        f"--replication_margin_softening {args.replication_margin_softening} "
        f"--certification_recheck_top_k {args.certification_recheck_top_k} "
        f"--certification_recheck_min_replicates "
        f"{args.certification_recheck_min_replicates} "
        f"--certification_recheck_soft_margin_scale "
        f"{args.certification_recheck_soft_margin_scale} "
        f"--certification_recheck_variance_prior_df "
        f"{args.certification_recheck_variance_prior_df} "
        f"--finalist_replication_budget {args.finalist_replication_budget} "
        f"--finalist_replication_count {args.finalist_replication_count} "
        f"--finalist_replication_min_replicates "
        f"{args.finalist_replication_min_replicates} "
        f"--finalist_replication_delta {args.finalist_replication_delta} "
        f"--finalist_replication_variance_prior_df "
        f"{args.finalist_replication_variance_prior_df} "
        f"{'--finalist_replication_expert_stratified ' if args.finalist_replication_expert_stratified else '--no-finalist_replication_expert_stratified '}"
        f"{'--finalist_replication_adaptive_race ' if args.finalist_replication_adaptive_race else '--no-finalist_replication_adaptive_race '}"
        f"{'--finalist_replication_fixed_universe ' if args.finalist_replication_fixed_universe else '--no-finalist_replication_fixed_universe '}"
        f"{'--observed_incumbent_use_replicate_variance ' if args.observed_incumbent_use_replicate_variance else '--no-observed_incumbent_use_replicate_variance '}"
        f"--safe_interior_candidate_count {args.safe_interior_candidate_count} "
        f"--safe_interior_pool_size {args.safe_interior_pool_size} "
        f"--safe_interior_margin {args.safe_interior_margin} "
        f"--observed_neighbor_candidate_count "
        f"{args.observed_neighbor_candidate_count} "
        f"--observed_neighbor_radius {args.observed_neighbor_radius} "
        f"--observed_neighbor_safe_margin_scale "
        f"{args.observed_neighbor_safe_margin_scale} "
        f"--recommendation_infeasible_penalty "
        f"{args.recommendation_infeasible_penalty} "
        f"--recommendation_infeasible_strategy "
        f"{args.recommendation_infeasible_strategy} "
        f"{'--recommend_observed_only ' if args.recommend_observed_only else ''}"
        f"{'--recommendation_calibration ' if args.recommendation_calibration else ''}"
        f"--recommendation_calibration_scope {args.recommendation_calibration_scope} "
        f"--recommendation_calibration_max_effective_fraction "
        f"{args.recommendation_calibration_max_effective_fraction} "
        f"--recommendation_calibration_min_obs {args.recommendation_calibration_min_obs} "
        f"--recommendation_calibration_max_theory_margin "
        f"{args.recommendation_calibration_max_theory_margin} "
        f"--recommendation_calibration_max_leverage "
        f"{args.recommendation_calibration_max_leverage} "
        f"--recommendation_slack_initial {args.recommendation_slack_initial} "
        f"--recommendation_slack_decay {args.recommendation_slack_decay} "
        f"{'--certification_calibration ' if args.certification_calibration else ''}"
        f"--certification_calibration_min_obs {args.certification_calibration_min_obs} "
        f"--certification_calibration_beta {args.certification_calibration_beta} "
        f"--certification_calibration_policy {args.certification_calibration_policy} "
        f"--certification_calibration_max_leverage "
        f"{args.certification_calibration_max_leverage} "
        f"--certification_calibration_max_theory_margin "
        f"{args.certification_calibration_max_theory_margin} "
        f"--certification_calibration_raise_delta "
        f"{args.certification_calibration_raise_delta} "
        f"{'--recommendation_observed_fallback ' if args.recommendation_observed_fallback else ''}"
        f"--observed_incumbent_margin_scale "
        f"{args.observed_incumbent_margin_scale} "
        f"{'--use_source_recommendation_slack ' if args.use_source_recommendation_slack else ''}"
        f"{'--source_mean_prior_fallback ' if args.source_mean_prior_fallback else ''}"
        f"--source_mean_prior_z {args.source_mean_prior_z} "
        f"--source_mean_prior_margin_tol {args.source_mean_prior_margin_tol} "
        f"{'--truth_pool_diagnostics ' if args.truth_pool_diagnostics else ''}"
        f"--truth_pool_max_candidates {args.truth_pool_max_candidates} "
        f"{'--offline_only' if args.offline_only else '--no-offline_only'} "
        f"--source_records_per_domain {args.source_records_per_domain} "
        f"--meta_local_dim {args.meta_local_dim} "
        f"--meta_shared_dim {args.meta_shared_dim} "
        f"{'--meta_ordered_cumulative_exposure ' if args.meta_ordered_cumulative_exposure else '--no-meta_ordered_cumulative_exposure '}"
        f"--meta_ordered_exposure_max_frequency "
        f"{args.meta_ordered_exposure_max_frequency} "
        f"--meta_ordered_exposure_active_dim "
        f"{args.meta_ordered_exposure_active_dim} "
        f"--meta_ordered_exposure_frequency_penalty "
        f"{args.meta_ordered_exposure_frequency_penalty} "
        f"--meta_ordered_exposure_basis_mode "
        f"{args.meta_ordered_exposure_basis_mode} "
        f"{'--meta_ordered_exposure_adaptive_sparsity ' if args.meta_ordered_exposure_adaptive_sparsity else '--no-meta_ordered_exposure_adaptive_sparsity '}"
        f"{'--meta_ordered_exposure_replace_local_kernel ' if args.meta_ordered_exposure_replace_local_kernel else '--no-meta_ordered_exposure_replace_local_kernel '}"
        f"{'--meta_ordered_exposure_semiparametric_residual ' if args.meta_ordered_exposure_semiparametric_residual else '--no-meta_ordered_exposure_semiparametric_residual '}"
        f"{'--meta_ordered_exposure_latent_structure_selection ' if args.meta_ordered_exposure_latent_structure_selection else '--no-meta_ordered_exposure_latent_structure_selection '}"
        f"{'--meta_ordered_exposure_group_shared_shrinkage ' if args.meta_ordered_exposure_group_shared_shrinkage else '--no-meta_ordered_exposure_group_shared_shrinkage '}"
        f"{'--meta_ordered_exposure_group_ridge_learning ' if args.meta_ordered_exposure_group_ridge_learning else '--no-meta_ordered_exposure_group_ridge_learning '}"
        f"--meta_anchor_count {args.meta_anchor_count} "
        f"--meta_kmeans_iters {args.meta_kmeans_iters} "
        f"--meta_soft_temperature {args.meta_soft_temperature} "
        f"--meta_ridge {args.meta_ridge} "
        f"{preset_flags(preset)} "
        f"--meta_anchor_sampling_temperature {args.meta_anchor_sampling_temperature} "
        f"--meta_teacher_records_per_domain {args.meta_teacher_records_per_domain} "
        f"--meta_teacher_weight {args.meta_teacher_weight} "
        f"--meta_teacher_pool_size {args.meta_teacher_pool_size} "
        f"--meta_teacher_elite_fraction {args.meta_teacher_elite_fraction} "
        f"--meta_teacher_boundary_fraction {args.meta_teacher_boundary_fraction} "
        f"--meta_teacher_anchor_sampling_temperature "
        f"{args.meta_teacher_anchor_sampling_temperature} "
        f"--meta_hvd_noise_floor_scale {args.meta_hvd_noise_floor_scale} "
        f"--meta_teacher_hvd_noise_floor_scale "
        f"{args.meta_teacher_hvd_noise_floor_scale} "
        f"--meta_universal_shape_count {args.meta_universal_shape_count} "
        f"--meta_component_stage {args.meta_component_stage} "
        f"--meta_spectral_active_dim {args.meta_spectral_active_dim} "
        f"--meta_spectral_max_library_size {args.meta_spectral_max_library_size} "
        f"--meta_spectral_low_frequency_components "
        f"{args.meta_spectral_low_frequency_components} "
        f"--meta_spectral_graph_neighbors {args.meta_spectral_graph_neighbors} "
        f"--meta_spectral_relevance_floor {args.meta_spectral_relevance_floor} "
        f"--meta_spectral_gate_boundary_weight "
        f"{args.meta_spectral_gate_boundary_weight} "
        f"--meta_spectral_gate_dangerous_weight "
        f"{args.meta_spectral_gate_dangerous_weight} "
        f"--meta_spectral_gate_selection_tolerance "
        f"{args.meta_spectral_gate_selection_tolerance} "
        f"--meta_spectral_gate_calibration_quantile "
        f"{args.meta_spectral_gate_calibration_quantile} "
        f"{'--meta_spectral_frequency_adaptation' if args.meta_spectral_frequency_adaptation else '--no-meta_spectral_frequency_adaptation'} "
        f"--meta_spectral_frequency_cutoffs "
        f"{args.meta_spectral_frequency_cutoffs} "
        f"--meta_spectral_frequency_ridges "
        f"{args.meta_spectral_frequency_ridges} "
        f"--meta_spectral_frequency_source_penalty "
        f"{args.meta_spectral_frequency_source_penalty} "
        f"--meta_spectral_frequency_temperature "
        f"{args.meta_spectral_frequency_temperature} "
        f"--meta_spectral_frequency_refit_interval "
        f"{args.meta_spectral_frequency_refit_interval} "
        f"{'--meta_spectral_risk_alignment' if args.meta_spectral_risk_alignment else '--no-meta_spectral_risk_alignment'} "
        f"--meta_spectral_alignment_active_dim {args.meta_spectral_alignment_active_dim} "
        f"--meta_spectral_alignment_subspace_dim {args.meta_spectral_alignment_subspace_dim} "
        f"--meta_spectral_alignment_domain_penalty {args.meta_spectral_alignment_domain_penalty} "
        f"{'--meta_spectral_alignment_source_procrustes' if args.meta_spectral_alignment_source_procrustes else '--no-meta_spectral_alignment_source_procrustes'} "
        f"--meta_spectral_alignment_target_ridge {args.meta_spectral_alignment_target_ridge} "
        f"--meta_spectral_alignment_target_min_gain {args.meta_spectral_alignment_target_min_gain} "
        f"--meta_spectral_alignment_target_min_bins {args.meta_spectral_alignment_target_min_bins} "
        f"--meta_spectral_alignment_refit_interval {args.meta_spectral_alignment_refit_interval} "
        f"--meta_spectral_alignment_source_episodes "
        f"{args.meta_spectral_alignment_source_episodes} "
        f"{'--meta_spectral_alignment_admission' if args.meta_spectral_alignment_admission else '--no-meta_spectral_alignment_admission'} "
        f"{'--meta_spectral_alignment_latent_proposals' if args.meta_spectral_alignment_latent_proposals else '--no-meta_spectral_alignment_latent_proposals'} "
        f"--meta_spectral_alignment_inverse_pool_size "
        f"{args.meta_spectral_alignment_inverse_pool_size} "
        f"--meta_spectral_alignment_episode_pilot_size "
        f"{args.meta_spectral_alignment_episode_pilot_size} "
        f"--meta_spectral_alignment_episode_evaluation_size "
        f"{args.meta_spectral_alignment_episode_evaluation_size} "
        f"--meta_spectral_alignment_episode_ridge "
        f"{args.meta_spectral_alignment_episode_ridge} "
        f"{'--meta_spectral_additive_adaptation' if args.meta_spectral_additive_adaptation else '--no-meta_spectral_additive_adaptation'} "
        f"--meta_spectral_additive_max_groups "
        f"{args.meta_spectral_additive_max_groups} "
        f"--meta_spectral_additive_target_max_groups "
        f"{args.meta_spectral_additive_target_max_groups} "
        f"--meta_spectral_additive_source_penalty "
        f"{args.meta_spectral_additive_source_penalty} "
        f"--meta_spectral_additive_complexity_penalty "
        f"{args.meta_spectral_additive_complexity_penalty} "
        f"--meta_spectral_additive_temperature "
        f"{args.meta_spectral_additive_temperature} "
        f"--meta_spectral_additive_refit_interval "
        f"{args.meta_spectral_additive_refit_interval} "
        f"--meta_spectral_additive_max_saturation_fraction "
        f"{args.meta_spectral_additive_max_saturation_fraction} "
        f"{'--meta_spectral_coefficient_shrinkage' if args.meta_spectral_coefficient_shrinkage else '--no-meta_spectral_coefficient_shrinkage'} "
        f"--meta_spectral_shrinkage_strength "
        f"{args.meta_spectral_shrinkage_strength} "
        f"--meta_spectral_shrinkage_floor {args.meta_spectral_shrinkage_floor} "
        f"{'--meta_spectral_adaptive_sparsity' if args.meta_spectral_adaptive_sparsity else '--no-meta_spectral_adaptive_sparsity'} "
        f"--meta_spectral_adaptive_min_pip {args.meta_spectral_adaptive_min_pip} "
        f"--meta_spectral_adaptive_max_pip {args.meta_spectral_adaptive_max_pip} "
        f"--meta_spectral_adaptive_spike_ratio {args.meta_spectral_adaptive_spike_ratio} "
        f"--meta_spectral_adaptive_damping {args.meta_spectral_adaptive_damping} "
        f"--meta_spectral_adaptive_max_iter {args.meta_spectral_adaptive_max_iter} "
        f"--meta_spectral_adaptive_tolerance {args.meta_spectral_adaptive_tolerance} "
        f"--meta_spectral_adaptive_residual_floor_scale "
        f"{args.meta_spectral_adaptive_residual_floor_scale} "
        f"--meta_spectral_adaptive_gate_tolerance "
        f"{args.meta_spectral_adaptive_gate_tolerance} "
        f"--meta_spectral_adaptive_multiplicity_correction "
        f"{args.meta_spectral_adaptive_multiplicity_correction} "
        f"--meta_spectral_adaptive_max_effective_fraction "
        f"{args.meta_spectral_adaptive_max_effective_fraction} "
        f"--meta_spectral_adaptive_saturation_fraction "
        f"{args.meta_spectral_adaptive_saturation_fraction} "
        f"--meta_coordinate_mode {args.meta_coordinate_mode} "
        f"--meta_coordinate_relevance_floor {args.meta_coordinate_relevance_floor} "
        f"--meta_source_augments {args.meta_source_augments} "
        f"--meta_source_sigma_jitter {args.meta_source_sigma_jitter} "
        f"--meta_source_alpha_jitter {args.meta_source_alpha_jitter} "
        f"--meta_source_weight_jitter {args.meta_source_weight_jitter} "
        f"--meta_seed {args.meta_seed} "
        f"--meta_source_seed_mode {args.meta_source_seed_mode} "
        f"--meta_proposal_pool_size {args.meta_proposal_pool_size} "
        f"--meta_refinement_count {args.meta_refinement_count} "
        f"--seed_start {args.seed_start} --n_seeds {args.n_seeds} "
        f"--jobs {args.jobs_per_suite} "
        f"--task_shard_index {int(shard_index)} "
        f"--task_num_shards {int(num_shards)} "
        f"--checkpoint_path {ckpt_root / (prefix + '.jsonl')} --resume_completed "
        f"--runtime_checkpoint_resume "
        f"--runtime_checkpoint_interval {args.runtime_checkpoint_interval} "
        f"{'--progress_logging ' if args.progress_logging else ''}"
        f"--progress_units_per_iteration {args.progress_units_per_iteration} "
        f"--progress_exact_updates {args.progress_exact_updates} "
        f"{resume_completed_from}"
        f"--out_prefix {prefix}"
    )


def exact_kg_diagnostic_command(
    args,
    run_id,
    N,
    preset,
    shard_index,
    num_shards,
):
    heldouts = parse_csv(args.heldouts)
    lines = parse_csv(args.lines)
    if len(heldouts) != 1 or len(lines) != 1:
        raise ValueError(
            "post-run exact-KG diagnostics require one heldout and one line"
        )
    if args.basis_pair_grid:
        raise ValueError(
            "post-run exact-KG diagnostics do not support --basis-pair-grid"
        )
    if int(num_shards) != int(args.n_seeds):
        raise ValueError(
            "post-run exact-KG diagnostics require num_shards == n_seeds "
            "so every shard contains exactly one seed"
        )
    seed = int(args.seed_start) + int(shard_index)
    basis_label = "state"
    safe_variant = f"{lines[0]}_{heldouts[0]}_{basis_label}_seed{seed}"
    safe_variant = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_"
        for ch in safe_variant
    )
    checkpoint = (
        checkpoint_root(args, run_id)
        / "runtime"
        / safe_variant
        / f"checkpoint_stage_{int(N) - 1:05d}.pkl"
    )
    output = (
        args.deploy
        / "SC-OLH-KG"
        / "profiles"
        / (
            f"exactkg_mcdiag_{run_id}_N{int(N)}_{preset}_"
            f"{heldouts[0]}_seed{seed}.json"
        )
    )
    return " ".join([
        PYTHON,
        "performance/diagnose_exact_kg_checkpoint.py",
        "--manifest", shlex.quote(str(args.exact_kg_diagnostic_manifest)),
        "--seed", str(seed),
        "--checkpoint", shlex.quote(str(checkpoint)),
        "--mc-samples", shlex.quote(str(args.exact_kg_diagnostic_mc_samples)),
        "--sampling-modes",
        shlex.quote(str(args.exact_kg_diagnostic_sampling_modes)),
        "--repeats", str(args.exact_kg_diagnostic_repeats),
        "--exact-jobs", str(args.exact_kg_jobs),
        "--parallel-backend", shlex.quote(str(args.exact_kg_parallel_backend)),
        "--out", shlex.quote(str(output)),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", default=str(DEFAULT_SCHEDULER))
    parser.add_argument("--deploy", type=Path, default=DEFAULT_DEPLOY)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--nodes", default="node001,node002,node003,node004,node005,node006")
    parser.add_argument("--N-values", default="40,80")
    parser.add_argument("--loss-presets", default="calibrated")
    parser.add_argument(
        "--domains",
        default="FactorShockStatePolicyRZDT1,InventorySupplyChain,QueueResourceControl",
    )
    parser.add_argument("--heldouts", default="FactorShockStatePolicyRZDT1,InventorySupplyChain,QueueResourceControl")
    parser.add_argument("--lines", default="strict,lodo,domain")
    parser.add_argument("--d", type=int, default=50)
    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=0.04)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--weights", default="0.5,0.5")
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--K1", type=int, default=20)
    parser.add_argument("--K2", type=int, default=0)
    parser.add_argument("--posterior-pool-size", type=int, default=300)
    parser.add_argument("--posterior-keep", type=int, default=15)
    parser.add_argument("--axis-candidate-count", type=int, default=-1)
    parser.add_argument("--structured-candidate-count", type=int, default=0)
    parser.add_argument("--state-candidate-count", type=int, default=24)
    parser.add_argument("--state-inverse-pool-size", type=int, default=600)
    parser.add_argument("--state-inverse-neighbors", type=int, default=2)
    parser.add_argument("--n-thr", type=int, default=5)
    parser.add_argument("--eval-pool-size", type=int, default=500)
    parser.add_argument("--lambda-feas", type=float, default=0.25)
    parser.add_argument("--lambda-var", type=float, default=0.25)
    parser.add_argument("--lambda-mean", type=float, default=0.10)
    parser.add_argument("--lambda-constraint-epistemic", type=float, default=0.20)
    parser.add_argument("--lambda-coupling", type=float, default=0.05)
    parser.add_argument("--beta-g", type=float, default=2.0)
    parser.add_argument("--state-basis-mode", default="raw+state")
    parser.add_argument("--constraint-state-basis-mode", default="state")
    parser.add_argument(
        "--basis-pair-grid",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--raw-basis-dim", type=int, default=32)
    parser.add_argument("--raw-projection-seed", type=int, default=314159)
    parser.add_argument("--numeric-backend", default="numpy")
    parser.add_argument("--numeric-backend-device", default="auto")
    parser.add_argument("--torch-dtype", default="float64")
    parser.add_argument("--torch-min-rows", type=int, default=128)
    parser.add_argument("--acquisition-mode", default="exact_mc")
    parser.add_argument("--exact-kg-mc-samples", type=int, default=2)
    parser.add_argument("--exact-kg-jobs", type=int, default=1)
    parser.add_argument("--exact-kg-parallel-backend", default="thread")
    parser.add_argument("--exact-kg-sampling-mode", default="iid")
    parser.add_argument(
        "--exact-kg-clip-negative",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--exact-kg-blend", type=float, default=0.0)
    parser.add_argument(
        "--exact-kg-terminal-mode", default="hard_certified")
    parser.add_argument(
        "--terminal-bayes-violation-penalty", type=float, default=5.0)
    parser.add_argument(
        "--terminal-frontier-candidate-count", type=int, default=0)
    parser.add_argument("--task-posterior-mode", default="off")
    parser.add_argument(
        "--task-posterior-initial-design",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--task-posterior-boundary-bracket-fraction",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--task-posterior-mandatory-universal-count",
        type=int,
        default=0,
    )
    parser.add_argument("--task-posterior-pilot-count", type=int, default=-1)
    parser.add_argument("--task-posterior-temperature", type=float, default=0.5)
    parser.add_argument(
        "--task-posterior-temperature-decay", type=float, default=0.5)
    parser.add_argument(
        "--task-posterior-boundary-score-weight", type=float, default=0.25)
    parser.add_argument(
        "--task-posterior-objective-score-weight", type=float, default=0.25)
    parser.add_argument(
        "--task-posterior-constraint-score-weight", type=float, default=1.0)
    parser.add_argument(
        "--task-posterior-safe-generalized",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--task-posterior-safe-boundary-score-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--task-posterior-safe-pairwise-score-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--task-posterior-safe-pairwise-max-history",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--task-posterior-safe-pairwise-probability-floor",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--task-posterior-kl-radius-numerator", type=float, default=0.5)
    parser.add_argument(
        "--task-posterior-confidence-delta", type=float, default=0.05)
    parser.add_argument(
        "--task-posterior-max-kl-radius", type=float, default=4.0)
    parser.add_argument(
        "--task-posterior-prior-protection-numerator",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--task-posterior-prior-protection-max", type=float, default=0.5)
    parser.add_argument(
        "--task-posterior-local-kernel-expert",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--task-posterior-candidate-count", type=int, default=0)
    parser.add_argument(
        "--task-posterior-recommendation-count", type=int, default=0)
    parser.add_argument(
        "--task-posterior-proposal-pool-size", type=int, default=1024)
    parser.add_argument(
        "--task-posterior-proposal-exploration", type=float, default=0.10)
    parser.add_argument(
        "--task-posterior-proposal-min-per-expert", type=int, default=2)
    parser.add_argument(
        "--task-posterior-sensitivity-mode", default="off")
    parser.add_argument("--constraint-uncertain-candidate-count", type=int, default=8)
    parser.add_argument("--constraint-uncertain-pool-size", type=int, default=180)
    parser.add_argument(
        "--constraint-uncertain-state-pool-fraction",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--constraint-epistemic-margin-softening",
        type=float,
        default=3.0,
    )
    parser.add_argument("--replication-candidate-count", type=int, default=3)
    parser.add_argument("--replication-max-per-solution", type=int, default=5)
    parser.add_argument("--replication-margin-softening", type=float, default=3.0)
    parser.add_argument("--certification-recheck-top-k", type=int, default=0)
    parser.add_argument(
        "--certification-recheck-min-replicates", type=int, default=3)
    parser.add_argument(
        "--certification-recheck-soft-margin-scale", type=float, default=2.0)
    parser.add_argument(
        "--certification-recheck-variance-prior-df", type=float, default=2.0)
    parser.add_argument("--finalist-replication-budget", type=int, default=0)
    parser.add_argument("--finalist-replication-count", type=int, default=2)
    parser.add_argument(
        "--finalist-replication-min-replicates", type=int, default=2)
    parser.add_argument(
        "--finalist-replication-delta", type=float, default=0.05)
    parser.add_argument(
        "--finalist-replication-variance-prior-df", type=float, default=2.0)
    parser.add_argument(
        "--finalist-replication-expert-stratified",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--finalist-replication-adaptive-race",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--finalist-replication-fixed-universe",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--observed-incumbent-use-replicate-variance",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--safe-interior-candidate-count", type=int, default=0)
    parser.add_argument("--safe-interior-pool-size", type=int, default=300)
    parser.add_argument("--safe-interior-margin", type=float, default=0.0)
    parser.add_argument("--observed-neighbor-candidate-count", type=int, default=0)
    parser.add_argument("--observed-neighbor-radius", type=float, default=0.08)
    parser.add_argument("--observed-neighbor-safe-margin-scale", type=float, default=1.0)
    parser.add_argument("--recommendation-infeasible-penalty", type=float, default=5.0)
    parser.add_argument("--recommendation-infeasible-strategy", default="penalty")
    parser.add_argument("--recommend-observed-only", action="store_true")
    parser.add_argument(
        "--recommendation-calibration",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--recommendation-calibration-scope", default="pool")
    parser.add_argument(
        "--recommendation-calibration-max-effective-fraction",
        type=float,
        default=0.35,
    )
    parser.add_argument("--recommendation-calibration-min-obs", type=int, default=8)
    parser.add_argument(
        "--recommendation-calibration-max-theory-margin",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--recommendation-calibration-max-leverage",
        type=float,
        default=20.0,
    )
    parser.add_argument("--recommendation-slack-initial", type=float, default=0.0)
    parser.add_argument("--recommendation-slack-decay", default="sqrt")
    parser.add_argument(
        "--certification-calibration",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--certification-calibration-min-obs", type=int, default=8)
    parser.add_argument("--certification-calibration-beta", type=float, default=2.0)
    parser.add_argument("--certification-calibration-policy", default="guarded")
    parser.add_argument("--certification-calibration-max-leverage", type=float, default=10.0)
    parser.add_argument("--certification-calibration-max-theory-margin", type=float, default=0.25)
    parser.add_argument("--certification-calibration-raise-delta", type=float, default=0.10)
    parser.add_argument(
        "--recommendation-observed-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--observed-incumbent-margin-scale", type=float, default=-0.5)
    parser.add_argument(
        "--use-source-recommendation-slack",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--source-mean-prior-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--source-mean-prior-z", type=float, default=0.5)
    parser.add_argument("--source-mean-prior-margin-tol", type=float, default=0.0)
    parser.add_argument(
        "--truth-pool-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--truth-pool-max-candidates", type=int, default=400)
    parser.add_argument(
        "--offline-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forbid external API/network-assisted priors in every submitted run.",
    )
    parser.add_argument("--source-records-per-domain", type=int, default=256)
    parser.add_argument("--meta-local-dim", type=int, default=3)
    parser.add_argument("--meta-shared-dim", type=int, default=3)
    parser.add_argument(
        "--meta-ordered-cumulative-exposure",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--meta-ordered-exposure-max-frequency", type=int, default=8)
    parser.add_argument(
        "--meta-ordered-exposure-active-dim", type=int, default=2)
    parser.add_argument(
        "--meta-ordered-exposure-frequency-penalty", type=float, default=0.10)
    parser.add_argument(
        "--meta-ordered-exposure-basis-mode",
        choices=["full_quadratic", "diagonal_quadratic"],
        default="full_quadratic",
    )
    parser.add_argument(
        "--meta-ordered-exposure-adaptive-sparsity",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--meta-ordered-exposure-replace-local-kernel",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--meta-ordered-exposure-semiparametric-residual",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--meta-ordered-exposure-latent-structure-selection",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--meta-ordered-exposure-group-shared-shrinkage",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--meta-ordered-exposure-group-ridge-learning",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--meta-anchor-count", type=int, default=32)
    parser.add_argument("--meta-kmeans-iters", type=int, default=35)
    parser.add_argument("--meta-soft-temperature", type=float, default=0.75)
    parser.add_argument("--meta-ridge", type=float, default=1e-4)
    parser.add_argument("--meta-anchor-sampling-temperature", type=float, default=0.0)
    parser.add_argument("--meta-teacher-records-per-domain", type=int, default=96)
    parser.add_argument("--meta-teacher-weight", type=float, default=3.0)
    parser.add_argument("--meta-teacher-pool-size", type=int, default=2048)
    parser.add_argument("--meta-teacher-elite-fraction", type=float, default=0.50)
    parser.add_argument("--meta-teacher-boundary-fraction", type=float, default=0.35)
    parser.add_argument(
        "--meta-teacher-anchor-sampling-temperature",
        type=float,
        default=0.35,
    )
    parser.add_argument("--meta-hvd-noise-floor-scale", type=float, default=0.0)
    parser.add_argument("--meta-teacher-hvd-noise-floor-scale", type=float, default=1.0)
    parser.add_argument("--meta-universal-shape-count", type=int, default=64)
    parser.add_argument(
        "--meta-component-stage",
        choices=["legacy_all", "coordinate", "spectral", "spectral_hvd"],
        default="spectral_hvd",
    )
    parser.add_argument("--meta-spectral-active-dim", type=int, default=6)
    parser.add_argument("--meta-spectral-max-library-size", type=int, default=64)
    parser.add_argument(
        "--meta-spectral-low-frequency-components", type=int, default=8)
    parser.add_argument("--meta-spectral-graph-neighbors", type=int, default=10)
    parser.add_argument("--meta-spectral-relevance-floor", type=float, default=0.05)
    parser.add_argument("--meta-spectral-gate-boundary-weight", type=float, default=2.0)
    parser.add_argument("--meta-spectral-gate-dangerous-weight", type=float, default=3.0)
    parser.add_argument(
        "--meta-spectral-gate-selection-tolerance", type=float, default=0.02)
    parser.add_argument(
        "--meta-spectral-gate-calibration-quantile", type=float, default=0.90)
    parser.add_argument(
        "--meta-spectral-frequency-adaptation",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--meta-spectral-frequency-cutoffs", default="3,5,8,12")
    parser.add_argument(
        "--meta-spectral-frequency-ridges", default="0.0001,0.01,1.0")
    parser.add_argument(
        "--meta-spectral-frequency-source-penalty", type=float, default=0.05)
    parser.add_argument(
        "--meta-spectral-frequency-temperature", type=float, default=0.5)
    parser.add_argument(
        "--meta-spectral-frequency-refit-interval", type=int, default=5)
    parser.add_argument(
        "--meta-spectral-risk-alignment",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--meta-spectral-alignment-active-dim", type=int, default=4)
    parser.add_argument(
        "--meta-spectral-alignment-subspace-dim", type=int, default=2)
    parser.add_argument(
        "--meta-spectral-alignment-domain-penalty", type=float, default=0.5)
    parser.add_argument(
        "--meta-spectral-alignment-source-procrustes",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--meta-spectral-alignment-target-ridge", type=float, default=5.0)
    parser.add_argument(
        "--meta-spectral-alignment-target-min-gain", type=float, default=0.02)
    parser.add_argument(
        "--meta-spectral-alignment-target-min-bins", type=int, default=3)
    parser.add_argument(
        "--meta-spectral-alignment-refit-interval", type=int, default=5)
    parser.add_argument(
        "--meta-spectral-alignment-source-episodes", type=int, default=0)
    parser.add_argument(
        "--meta-spectral-alignment-admission",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--meta-spectral-alignment-latent-proposals",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--meta-spectral-alignment-inverse-pool-size",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--meta-spectral-alignment-episode-pilot-size", type=int, default=10)
    parser.add_argument(
        "--meta-spectral-alignment-episode-evaluation-size",
        type=int,
        default=24,
    )
    parser.add_argument(
        "--meta-spectral-alignment-episode-ridge", type=float, default=0.1)
    parser.add_argument(
        "--meta-spectral-additive-adaptation",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--meta-spectral-additive-max-groups", type=int, default=8)
    parser.add_argument(
        "--meta-spectral-additive-target-max-groups", type=int, default=2)
    parser.add_argument(
        "--meta-spectral-additive-source-penalty", type=float, default=0.05)
    parser.add_argument(
        "--meta-spectral-additive-complexity-penalty", type=float, default=0.05)
    parser.add_argument(
        "--meta-spectral-additive-temperature", type=float, default=0.5)
    parser.add_argument(
        "--meta-spectral-additive-refit-interval", type=int, default=5)
    parser.add_argument(
        "--meta-spectral-additive-max-saturation-fraction",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--meta-spectral-coefficient-shrinkage",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--meta-spectral-shrinkage-strength", type=float, default=1.0)
    parser.add_argument(
        "--meta-spectral-shrinkage-floor", type=float, default=0.05)
    parser.add_argument(
        "--meta-spectral-adaptive-sparsity",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--meta-spectral-adaptive-min-pip", type=float, default=0.05)
    parser.add_argument("--meta-spectral-adaptive-max-pip", type=float, default=0.95)
    parser.add_argument(
        "--meta-spectral-adaptive-spike-ratio", type=float, default=0.05)
    parser.add_argument("--meta-spectral-adaptive-damping", type=float, default=0.5)
    parser.add_argument("--meta-spectral-adaptive-max-iter", type=int, default=40)
    parser.add_argument(
        "--meta-spectral-adaptive-tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--meta-spectral-adaptive-residual-floor-scale", type=float, default=0.05)
    parser.add_argument(
        "--meta-spectral-adaptive-gate-tolerance", type=float, default=0.05)
    parser.add_argument(
        "--meta-spectral-adaptive-multiplicity-correction",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--meta-spectral-adaptive-max-effective-fraction",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--meta-spectral-adaptive-saturation-fraction",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--meta-coordinate-mode",
        choices=["pca", "stable_supervised"],
        default="pca",
    )
    parser.add_argument("--meta-coordinate-relevance-floor", type=float, default=0.05)
    parser.add_argument("--meta-source-augments", type=int, default=1)
    parser.add_argument("--meta-source-sigma-jitter", type=float, default=0.20)
    parser.add_argument("--meta-source-alpha-jitter", type=float, default=0.25)
    parser.add_argument("--meta-source-weight-jitter", type=float, default=0.05)
    parser.add_argument("--meta-seed", type=int, default=20260706)
    parser.add_argument(
        "--meta-source-seed-mode",
        choices=["frozen", "per_target"],
        default="frozen",
    )
    parser.add_argument("--meta-proposal-pool-size", type=int, default=1024)
    parser.add_argument("--meta-refinement-count", type=int, default=192)
    parser.add_argument("--runtime-checkpoint-interval", type=int, default=1)
    parser.add_argument(
        "--progress-logging",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--progress-units-per-iteration", type=int, default=100)
    parser.add_argument("--progress-exact-updates", type=int, default=10)
    parser.add_argument(
        "--post-run-exact-kg-diagnostic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "After each one-seed shard, rescore the N-1 checkpoint with "
            "multiple exact-KG Monte Carlo schedules on the same node."
        ),
    )
    parser.add_argument(
        "--exact-kg-diagnostic-manifest",
        default="performance/manifests/v18b_exactkg_mcdiag.json",
    )
    parser.add_argument("--exact-kg-diagnostic-mc-samples", default="2,8")
    parser.add_argument(
        "--exact-kg-diagnostic-sampling-modes",
        default="iid,antithetic",
    )
    parser.add_argument("--exact-kg-diagnostic-repeats", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--jobs-per-suite", type=int, default=10)
    parser.add_argument("--num-shards-per-suite", type=int, default=1)
    parser.add_argument(
        "--shard-start",
        type=int,
        default=0,
        help="First shard index to submit, inclusive.",
    )
    parser.add_argument(
        "--shard-stop",
        type=int,
        default=-1,
        help="Shard index to stop before; -1 means num-shards-per-suite.",
    )
    parser.add_argument("--cpu", type=int, default=12)
    parser.add_argument("--ram-mb", type=int, default=32768)
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--sync-remote",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Synchronize SC-OLH-KG to the remote deploy before submission.",
    )
    parser.add_argument(
        "--bulk-submit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Submit the full shard group atomically through submit-jsonl.",
    )
    args = parser.parse_args()

    if args.sync_remote and not args.dry_run:
        run_cmd([DEFAULT_SYNC_SCRIPT])

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    cwd = args.deploy / "SC-OLH-KG"
    nodes = parse_cpu_nodes(args.nodes)
    task_ids = []
    num_shards = max(1, int(args.num_shards_per_suite))
    shard_start = max(0, int(args.shard_start))
    shard_stop = (
        num_shards
        if int(args.shard_stop) < 0
        else min(num_shards, int(args.shard_stop))
    )
    if shard_start >= shard_stop:
        raise ValueError(
            f"empty shard range [{shard_start},{shard_stop}) for "
            f"num_shards={num_shards}"
        )
    suites = [
        (N, preset, shard_index)
        for N in parse_csv(args.N_values, int)
        for preset in parse_csv(args.loss_presets)
        for shard_index in range(shard_start, shard_stop)
    ]
    bulk_specs = []
    for idx, (N, preset, shard_index) in enumerate(suites):
        node = nodes[idx % len(nodes)]
        command = suite_command(
            args,
            run_id,
            N,
            preset,
            shard_index=shard_index,
            num_shards=num_shards,
        )
        if args.post_run_exact_kg_diagnostic:
            diagnostic = exact_kg_diagnostic_command(
                args,
                run_id,
                N,
                preset,
                shard_index,
                num_shards,
            )
            payload_command = f"{command} && {diagnostic}"
        else:
            payload_command = command
        cmd = "; ".join([
            "export LC_ALL=C LANG=C",
            "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1",
            f"export SCOLHKG_OFFLINE={'1' if args.offline_only else '0'}",
            f"{payload_command} && echo DONE",
        ])
        description = (
                f"SC-OLH-KG LODO meta-prior N={N} {preset} {run_id} "
                f"shard={shard_index}/{num_shards}"
            )
        signature = (
                f"KG_op/scolhkg_lodo_meta/{run_id}/N{N}/{preset}/"
                f"shard{shard_index}of{num_shards}"
            )
        if args.bulk_submit:
            bulk_specs.append({
                "description": description,
                "cmd": cmd,
                "cwd": str(cwd),
                "signature": signature,
                "project": "KG-SYNTH",
                "vram": 0,
                "cpu": int(args.cpu),
                "ram_mb": int(args.ram_mb),
                "require_node": node,
                "allowed_nodes": list(nodes),
                # The benchmark owns resume semantics internally. Registering
                # the directory here protects it from launch rsync --delete.
                "ckpt_dir": str(checkpoint_root(args, run_id)),
                "allow_no_resume": True,
                "allow_shared_ckpt_dir": True,
                "allow_duplicate": True,
            })
        else:
            out = run_cmd([
                sys.executable, args.scheduler, "submit",
                "--description", description,
                "--cmd", cmd,
                "--cwd", str(cwd),
                "--signature", signature,
                "--project", "KG-SYNTH",
                "--vram", "0",
                "--cpu", str(args.cpu),
                "--ram-mb", str(args.ram_mb),
                "--ckpt-dir", str(checkpoint_root(args, run_id)),
                "--allow-no-resume",
                "--allow-shared-ckpt-dir",
                "--require-node", node,
                *allowed_node_flags(nodes),
                "--allow-duplicate",
            ], dry_run=args.dry_run)
            if out:
                print(out, end="")
                parts = out.split()
                if len(parts) > 1:
                    task_ids.append(parts[1])
    if args.bulk_submit and bulk_specs:
        command = [
            sys.executable,
            str(args.scheduler),
            "submit-jsonl",
            "--stdin",
            "--trusted",
            "--json",
            "--intent-label",
            f"scolhkg-lodo-{run_id}",
        ]
        print("+", " ".join(map(str, command)), flush=True)
        if args.dry_run:
            print(json.dumps(bulk_specs, indent=2))
        else:
            out = subprocess.check_output(
                command,
                input=json.dumps(bulk_specs),
                text=True,
            )
            print(out, end="")
            payload = json.loads(out)
            task_ids.extend([
                item["id"]
                for item in payload.get("submitted", [])
                if item.get("id")
            ])
    if args.dispatch:
        run_cmd([sys.executable, args.scheduler, "dispatch"], dry_run=args.dry_run)
    print({"run_id": run_id, "task_ids": task_ids, "suites": suites})


if __name__ == "__main__":
    main()
