"""Run one reproducible LODO shard by overriding a frozen manifest.

This keeps causal challengers tied to the exact baseline configuration.  Only
the explicitly exposed fields below may differ from the source manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.benchmark_lodo_meta_prior import run_one  # noqa: E402
from performance.benchmark_quality import json_safe  # noqa: E402
from performance.structural_ablation import (  # noqa: E402
    HVD_PROFILES,
    STRUCTURAL_PRIOR_PROFILES,
    apply_hvd_profile,
    apply_structural_prior_profile,
)
from core.designs import load_frozen_source_informed_design  # noqa: E402


def load_config(path):
    payload = json.loads(Path(path).read_text())
    if "config" not in payload:
        raise ValueError("LODO manifest must contain a config object")
    config = dict(payload["config"])
    config.setdefault("exact_kg_clip_negative", True)
    config.setdefault("exact_kg_sampling_mode", "iid")
    config.setdefault("task_posterior_safe_generalized", False)
    config.setdefault("source_discrepancy_update", True)
    config.setdefault("task_posterior_safe_boundary_score_weight", 1.0)
    config.setdefault("task_posterior_safe_pairwise_score_weight", 1.0)
    config.setdefault("task_posterior_safe_pairwise_max_history", 16)
    config.setdefault("task_posterior_safe_pairwise_probability_floor", 1e-6)
    config.setdefault("task_latent_inference_mode", "shadow")
    config.setdefault(
        "task_latent_calibration_mode", "source_profiles")
    config.setdefault("finalist_replication_budget", 0)
    config.setdefault("finalist_replication_count", 2)
    config.setdefault("finalist_observed_safety_count", 1)
    config.setdefault("finalist_replication_min_replicates", 2)
    config.setdefault("finalist_replication_delta", 0.05)
    config.setdefault("finalist_replication_variance_prior_df", 2.0)
    config.setdefault("finalist_replication_expert_stratified", False)
    config.setdefault("finalist_replication_adaptive_race", False)
    config.setdefault("finalist_replication_fixed_universe", False)
    config.setdefault("finalist_replication_policy", "legacy")
    config.setdefault("finalist_empirical_override", "legacy")
    config.setdefault("finalist_frontier_policy", "legacy")
    config.setdefault("finalist_terminal_max_arms", 4)
    config.setdefault("finalist_terminal_mc_samples", 2)
    config.setdefault("decision_contract_mode", "legacy")
    config.setdefault("tcb_v2_enabled", False)
    config.setdefault("tcb_v2_mode", "off")
    config.setdefault("tcb_v2_frontier_count", 1)
    config.setdefault("tcb_v2_descriptor_mode", "learned_risk")
    config.setdefault("finalist_terminal_value_mode", "model_default")
    config.setdefault("tcb_v2_coordinate", "boundary_latent")
    config.setdefault("tcb_v2_geometry", "low_rank_psd")
    config.setdefault("tcb_v2_rank", 2)
    config.setdefault("tcb_v2_ridge", 1e-3)
    config.setdefault("tcb_v2_domain_penalty", 0.5)
    config.setdefault("tcb_v2_boundary_temperature", 1.0)
    config.setdefault("tcb_v2_adaptation_ridge", 5.0)
    config.setdefault("tcb_v2_upper_alpha", 0.01)
    config.setdefault("tcb_v2_calibration_prior_df", 2.0)
    config.setdefault("tcb_v2_hierarchy_iterations", 5)
    config.setdefault("tcb_v2_effect_ridge", 1.0)
    config.setdefault("tcb_v2_rotation_mode", "none")
    config.setdefault("tcb_v2_rotation_ridge", 5.0)
    config.setdefault("tcb_v2_target_residual_rank", 0)
    config.setdefault("tcb_v2_residual_ridge", 5.0)
    config.setdefault("meta_observable_mean_coordinate", False)
    config.setdefault(
        "meta_observable_mean_ridges", "0.01,0.1,1.0,10.0,100.0")
    config.setdefault("meta_observable_mean_mode", "latent")
    config.setdefault("meta_observable_mean_latent_dim", 2)
    config.setdefault(
        "meta_observable_mean_training_target", "constraint_mean")
    config.setdefault("meta_source_observation_mode", "analytic")
    config.setdefault("meta_source_observation_replicates", 1)
    config.setdefault("meta_source_design_mode", "random")
    config.setdefault("meta_source_universal_fraction", 0.75)
    config.setdefault("initial_design", "auto")
    config.setdefault("decision_backend", "legacy")
    config.setdefault("decision_risk_penalty", 5.0)
    config.setdefault("decision_source_utility_weight", 1.0)
    config.setdefault("decision_backend_seed_offset", 470003)
    config.setdefault("decision_recommend_observed_only", True)
    return config


def atomic_write_json(path, payload):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--line", default="lodo_teacher")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--experiment-variant", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--runtime-checkpoint-dir", required=True)
    parser.add_argument("--d", type=int, default=None)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--n0", type=int, default=None)
    parser.add_argument(
        "--structural-prior-profile",
        choices=("inherit", *STRUCTURAL_PRIOR_PROFILES),
        default="inherit",
    )
    parser.add_argument(
        "--hvd-profile",
        choices=("inherit", *HVD_PROFILES),
        default="inherit",
    )
    parser.add_argument(
        "--ordered-cumulative-exposure",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--ordered-max-frequency", type=int, default=8)
    parser.add_argument("--ordered-active-dim", type=int, default=2)
    parser.add_argument("--ordered-frequency-penalty", type=float, default=0.10)
    parser.add_argument(
        "--spectral-orthogonalization",
        choices=("symmetric", "ordered_cholesky", "none"),
        default="symmetric",
    )
    parser.add_argument(
        "--ordered-basis-mode",
        choices=["full_quadratic", "diagonal_quadratic"],
        default="full_quadratic",
    )
    parser.add_argument(
        "--ordered-orthogonal-coordinates",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--ordered-adaptive-sparsity",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--ordered-replace-local-kernel",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--ordered-semiparametric-residual",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--ordered-latent-structure-selection",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--ordered-group-shared-shrinkage",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--ordered-group-ridge-learning",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--task-posterior-safe-generalized",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--source-discrepancy-update",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--task-posterior-safe-boundary-weight", type=float, default=1.0)
    parser.add_argument(
        "--task-posterior-safe-pairwise-weight", type=float, default=1.0)
    parser.add_argument(
        "--task-posterior-safe-pairwise-history", type=int, default=16)
    parser.add_argument(
        "--task-posterior-safe-pairwise-floor", type=float, default=1e-6)
    parser.add_argument(
        "--task-posterior-mandatory-universal-count", type=int, default=None)
    parser.add_argument(
        "--task-latent-inference-mode",
        choices=("shadow", "authoritative"),
        default="shadow",
    )
    parser.add_argument(
        "--task-latent-calibration-mode",
        choices=("source_profiles", "expert_ridge"),
        default="source_profiles",
    )
    parser.add_argument("--exact-sampling-mode", default="iid")
    parser.add_argument(
        "--decision-backend",
        choices=(
            "legacy", "additive", "exact_kg", "n0_best", "random",
            "sobol", "risk_ts", "bayes_risk_ei", "constrained_ei",
            "transfer_utility",
        ),
        default="legacy",
    )
    parser.add_argument("--decision-risk-penalty", type=float, default=5.0)
    parser.add_argument(
        "--decision-source-utility-weight", type=float, default=1.0)
    parser.add_argument(
        "--decision-recommend-observed-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--exact-terminal-mode",
        choices=(
            "hard_certified", "bayes_risk",
            "tcb_certified_lexicographic",
        ),
        default="hard_certified",
    )
    parser.add_argument(
        "--decision-contract-mode",
        choices=("legacy", "certified_lexicographic"),
        default="legacy",
    )
    parser.add_argument("--exact-mc-samples", type=int, default=2)
    parser.add_argument("--exact-jobs", type=int, default=32)
    parser.add_argument("--parallel-backend", default="process_fork")
    parser.add_argument(
        "--tcb-v2-mode",
        choices=("off", "shadow", "frontier", "certified"),
        default="off",
    )
    parser.add_argument("--tcb-v2-frontier-count", type=int, default=1)
    parser.add_argument(
        "--tcb-v2-descriptor-mode",
        choices=(
            "raw", "learned_coordinate", "raw+learned_coordinate",
            "learned_risk", "raw+learned_risk",
            "provider_coordinate", "raw+provider_coordinate",
            "provider_risk", "raw+provider_risk",
        ),
        default="learned_risk",
    )
    parser.add_argument("--tcb-v2-coordinate", default="boundary_latent")
    parser.add_argument("--tcb-v2-geometry", default="low_rank_psd")
    parser.add_argument("--tcb-v2-rank", type=int, default=2)
    parser.add_argument("--tcb-v2-ridge", type=float, default=1e-3)
    parser.add_argument("--tcb-v2-domain-penalty", type=float, default=0.5)
    parser.add_argument(
        "--tcb-v2-boundary-temperature", type=float, default=1.0)
    parser.add_argument(
        "--tcb-v2-adaptation-ridge", type=float, default=5.0)
    parser.add_argument("--tcb-v2-upper-alpha", type=float, default=0.01)
    parser.add_argument(
        "--tcb-v2-calibration-prior-df", type=float, default=2.0)
    parser.add_argument(
        "--tcb-v2-hierarchy-iterations", type=int, default=5)
    parser.add_argument("--tcb-v2-effect-ridge", type=float, default=1.0)
    parser.add_argument(
        "--tcb-v2-rotation-mode", choices=("none", "planar"), default="none")
    parser.add_argument("--tcb-v2-rotation-ridge", type=float, default=5.0)
    parser.add_argument("--tcb-v2-target-residual-rank", type=int, default=0)
    parser.add_argument("--tcb-v2-residual-ridge", type=float, default=5.0)
    parser.add_argument(
        "--observable-mean-coordinate",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--source-observation-mode",
        choices=("analytic", "nominal", "replicated"),
        default="analytic",
    )
    parser.add_argument("--source-observation-replicates", type=int, default=1)
    parser.add_argument(
        "--source-design-mode",
        choices=("random", "universal_mixture"),
        default="random",
    )
    parser.add_argument("--source-universal-fraction", type=float, default=0.75)
    parser.add_argument(
        "--source-consensus-template-count", type=int, default=0)
    parser.add_argument(
        "--initial-design",
        choices=("auto", "common_sobol", "source_informed"),
        default="auto",
    )
    parser.add_argument("--initial-design-file", default="")
    parser.add_argument(
        "--observable-mean-mode",
        choices=("atoms", "aggregate", "latent", "consensus"),
        default="latent",
    )
    parser.add_argument("--observable-mean-latent-dim", type=int, default=2)
    parser.add_argument(
        "--observable-mean-training-target",
        choices=("constraint_mean", "chance_margin"),
        default="constraint_mean",
    )
    parser.add_argument(
        "--finalist-terminal-value-mode",
        choices=("model_default", "certified_lexicographic"),
        default="model_default",
    )
    parser.add_argument("--finalist-replication-budget", type=int, default=0)
    parser.add_argument("--finalist-replication-count", type=int, default=2)
    parser.add_argument(
        "--finalist-observed-safety-count", type=int, default=1)
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
        "--finalist-replication-policy",
        choices=(
            "legacy",
            "commit_before_switch",
            "terminal_kg_1step",
            "terminal_kg_depth3",
        ),
        default="legacy",
    )
    parser.add_argument(
        "--finalist-empirical-override",
        choices=("legacy", "certified_only", "off"),
        default="legacy",
    )
    parser.add_argument(
        "--finalist-frontier-policy",
        choices=("legacy", "coverage_reserved", "observed_safety_reserved"),
        default="legacy",
    )
    parser.add_argument("--finalist-terminal-max-arms", type=int, default=4)
    parser.add_argument("--finalist-terminal-mc-samples", type=int, default=2)
    parser.add_argument("--certification-recheck-top-k", type=int, default=0)
    parser.add_argument(
        "--certification-recheck-min-replicates", type=int, default=3)
    parser.add_argument(
        "--certification-recheck-soft-margin-scale", type=float, default=2.0)
    args = parser.parse_args()

    config = load_config(args.manifest)
    for key, value in (("d", args.d), ("N", args.N), ("n0", args.n0)):
        if value is not None:
            config[key] = int(value)
    if int(config["n0"]) > int(config["N"]):
        raise ValueError("LODO shard requires n0 <= N")
    config.update({
        "experiment_variant": str(args.experiment_variant),
        "meta_ordered_cumulative_exposure": bool(
            args.ordered_cumulative_exposure),
        "meta_spectral_orthogonalization": str(
            args.spectral_orthogonalization),
        "meta_ordered_exposure_max_frequency": int(
            args.ordered_max_frequency),
        "meta_ordered_exposure_active_dim": int(args.ordered_active_dim),
        "meta_ordered_exposure_frequency_penalty": float(
            args.ordered_frequency_penalty),
        "meta_ordered_exposure_basis_mode": str(args.ordered_basis_mode),
        "meta_ordered_exposure_orthogonal_coordinates": bool(
            args.ordered_orthogonal_coordinates),
        "meta_ordered_exposure_adaptive_sparsity": bool(
            args.ordered_adaptive_sparsity),
        "meta_ordered_exposure_replace_local_kernel": bool(
            args.ordered_replace_local_kernel),
        "meta_ordered_exposure_semiparametric_residual": bool(
            args.ordered_semiparametric_residual),
        "meta_ordered_exposure_latent_structure_selection": bool(
            args.ordered_latent_structure_selection),
        "meta_ordered_exposure_group_shared_shrinkage": bool(
            args.ordered_group_shared_shrinkage),
        "meta_ordered_exposure_group_ridge_learning": bool(
            args.ordered_group_ridge_learning),
        "task_posterior_safe_generalized": bool(
            args.task_posterior_safe_generalized),
        "source_discrepancy_update": bool(
            args.source_discrepancy_update),
        "task_posterior_safe_boundary_score_weight": float(
            args.task_posterior_safe_boundary_weight),
        "task_posterior_safe_pairwise_score_weight": float(
            args.task_posterior_safe_pairwise_weight),
        "task_posterior_safe_pairwise_max_history": int(
            args.task_posterior_safe_pairwise_history),
        "task_posterior_safe_pairwise_probability_floor": float(
            args.task_posterior_safe_pairwise_floor),
        "task_latent_inference_mode": str(
            args.task_latent_inference_mode),
        "task_latent_calibration_mode": str(
            args.task_latent_calibration_mode),
        "exact_kg_sampling_mode": str(args.exact_sampling_mode),
        "decision_backend": str(args.decision_backend),
        "decision_risk_penalty": float(args.decision_risk_penalty),
        "decision_source_utility_weight": float(
            args.decision_source_utility_weight),
        "decision_recommend_observed_only": bool(
            args.decision_recommend_observed_only),
        "exact_kg_terminal_mode": str(args.exact_terminal_mode),
        "decision_contract_mode": str(args.decision_contract_mode),
        "exact_kg_mc_samples": int(args.exact_mc_samples),
        "exact_kg_jobs": int(args.exact_jobs),
        "exact_kg_parallel_backend": str(args.parallel_backend),
        "tcb_v2_enabled": bool(args.tcb_v2_mode != "off"),
        "tcb_v2_mode": str(args.tcb_v2_mode),
        "tcb_v2_frontier_count": int(args.tcb_v2_frontier_count),
        "tcb_v2_descriptor_mode": str(args.tcb_v2_descriptor_mode),
        "tcb_v2_coordinate": str(args.tcb_v2_coordinate),
        "tcb_v2_geometry": str(args.tcb_v2_geometry),
        "tcb_v2_rank": int(args.tcb_v2_rank),
        "tcb_v2_ridge": float(args.tcb_v2_ridge),
        "tcb_v2_domain_penalty": float(args.tcb_v2_domain_penalty),
        "tcb_v2_boundary_temperature": float(
            args.tcb_v2_boundary_temperature),
        "tcb_v2_adaptation_ridge": float(args.tcb_v2_adaptation_ridge),
        "tcb_v2_upper_alpha": float(args.tcb_v2_upper_alpha),
        "tcb_v2_calibration_prior_df": float(
            args.tcb_v2_calibration_prior_df),
        "tcb_v2_hierarchy_iterations": int(
            args.tcb_v2_hierarchy_iterations),
        "tcb_v2_effect_ridge": float(args.tcb_v2_effect_ridge),
        "tcb_v2_rotation_mode": str(args.tcb_v2_rotation_mode),
        "tcb_v2_rotation_ridge": float(args.tcb_v2_rotation_ridge),
        "tcb_v2_target_residual_rank": int(
            args.tcb_v2_target_residual_rank),
        "tcb_v2_residual_ridge": float(args.tcb_v2_residual_ridge),
        "meta_observable_mean_coordinate": bool(
            args.observable_mean_coordinate),
        "meta_observable_mean_mode": str(args.observable_mean_mode),
        "meta_observable_mean_latent_dim": int(
            args.observable_mean_latent_dim),
        "meta_observable_mean_training_target": str(
            args.observable_mean_training_target),
        "meta_source_observation_mode": str(
            args.source_observation_mode),
        "meta_source_observation_replicates": int(
            args.source_observation_replicates),
        "meta_source_design_mode": str(args.source_design_mode),
        "meta_source_universal_fraction": float(
            args.source_universal_fraction),
        "meta_source_consensus_template_count": int(
            args.source_consensus_template_count),
        "initial_design": str(args.initial_design),
        "finalist_terminal_value_mode": str(
            args.finalist_terminal_value_mode),
        "finalist_replication_budget": int(
            args.finalist_replication_budget),
        "finalist_replication_count": int(
            args.finalist_replication_count),
        "finalist_observed_safety_count": int(
            args.finalist_observed_safety_count),
        "finalist_replication_min_replicates": int(
            args.finalist_replication_min_replicates),
        "finalist_replication_delta": float(
            args.finalist_replication_delta),
        "finalist_replication_variance_prior_df": float(
            args.finalist_replication_variance_prior_df),
        "finalist_replication_expert_stratified": bool(
            args.finalist_replication_expert_stratified),
        "finalist_replication_adaptive_race": bool(
            args.finalist_replication_adaptive_race),
        "finalist_replication_fixed_universe": bool(
            args.finalist_replication_fixed_universe),
        "finalist_replication_policy": str(
            args.finalist_replication_policy),
        "finalist_empirical_override": str(
            args.finalist_empirical_override),
        "finalist_frontier_policy": str(
            args.finalist_frontier_policy),
        "finalist_terminal_max_arms": int(
            args.finalist_terminal_max_arms),
        "finalist_terminal_mc_samples": int(
            args.finalist_terminal_mc_samples),
        "certification_recheck_top_k": int(
            args.certification_recheck_top_k),
        "certification_recheck_min_replicates": int(
            args.certification_recheck_min_replicates),
        "certification_recheck_soft_margin_scale": float(
            args.certification_recheck_soft_margin_scale),
        "runtime_checkpoint_dir": str(args.runtime_checkpoint_dir),
        "runtime_checkpoint_resume": True,
        "runtime_checkpoint_interval": 1,
        "checkpoint_path": "",
        "resume_completed_from": "",
        "progress_logging": True,
        "progress_exact_updates": 10,
        "offline_only": True,
    })
    apply_structural_prior_profile(config, args.structural_prior_profile)
    apply_hvd_profile(config, args.hvd_profile)
    if args.task_posterior_mandatory_universal_count is not None:
        config["task_posterior_mandatory_universal_count"] = int(
            args.task_posterior_mandatory_universal_count)
    source_design_contract = None
    if args.initial_design == "source_informed":
        points, source_design_contract = load_frozen_source_informed_design(
            args.initial_design_file,
            heldout=args.heldout,
            seed=args.seed,
            n0=config["n0"],
            dimension=config["d"],
        )
        config.update({
            "initial_design_points": [list(point) for point in points],
            "initial_design_fingerprint": source_design_contract[
                "fingerprint"],
            "initial_design_source_archive_fingerprint": (
                source_design_contract["source_archive_fingerprint"]
            ),
        })
    task = {
        "args": config,
        "heldout": str(args.heldout),
        "line": str(args.line),
        "seed": int(args.seed),
    }
    row = run_one(task)
    row["experiment_variant"] = str(args.experiment_variant)
    payload = {
        "schema_version": 1,
        "source_manifest": str(args.manifest),
        "experiment_variant": str(args.experiment_variant),
        "causal_overrides": {
            "meta_ordered_cumulative_exposure": bool(
                args.ordered_cumulative_exposure),
            "structural_prior_profile": str(
                args.structural_prior_profile),
            "hvd_ablation_profile": str(args.hvd_profile),
            "meta_spectral_orthogonalization": str(
                args.spectral_orthogonalization),
            "meta_ordered_exposure_max_frequency": int(
                args.ordered_max_frequency),
            "meta_ordered_exposure_active_dim": int(args.ordered_active_dim),
            "meta_ordered_exposure_frequency_penalty": float(
                args.ordered_frequency_penalty),
            "meta_ordered_exposure_basis_mode": str(args.ordered_basis_mode),
            "meta_ordered_exposure_orthogonal_coordinates": bool(
                args.ordered_orthogonal_coordinates),
            "meta_ordered_exposure_adaptive_sparsity": bool(
                args.ordered_adaptive_sparsity),
            "meta_ordered_exposure_replace_local_kernel": bool(
                args.ordered_replace_local_kernel),
            "meta_ordered_exposure_semiparametric_residual": bool(
                args.ordered_semiparametric_residual),
            "meta_ordered_exposure_latent_structure_selection": bool(
                args.ordered_latent_structure_selection),
            "meta_ordered_exposure_group_shared_shrinkage": bool(
                args.ordered_group_shared_shrinkage),
            "meta_ordered_exposure_group_ridge_learning": bool(
                args.ordered_group_ridge_learning),
            "task_posterior_safe_generalized": bool(
                args.task_posterior_safe_generalized),
            "source_discrepancy_update": bool(
                args.source_discrepancy_update),
            "task_posterior_safe_boundary_score_weight": float(
                args.task_posterior_safe_boundary_weight),
            "task_posterior_safe_pairwise_score_weight": float(
                args.task_posterior_safe_pairwise_weight),
            "task_posterior_safe_pairwise_max_history": int(
                args.task_posterior_safe_pairwise_history),
            "task_posterior_safe_pairwise_probability_floor": float(
                args.task_posterior_safe_pairwise_floor),
            "task_latent_inference_mode": str(
                args.task_latent_inference_mode),
            "task_latent_calibration_mode": str(
                args.task_latent_calibration_mode),
            "exact_kg_sampling_mode": str(args.exact_sampling_mode),
            "decision_backend": str(args.decision_backend),
            "decision_risk_penalty": float(args.decision_risk_penalty),
            "decision_source_utility_weight": float(
                args.decision_source_utility_weight),
            "decision_recommend_observed_only": bool(
                args.decision_recommend_observed_only),
            "exact_kg_terminal_mode": str(args.exact_terminal_mode),
            "decision_contract_mode": str(args.decision_contract_mode),
            "exact_kg_mc_samples": int(args.exact_mc_samples),
            "exact_kg_jobs": int(args.exact_jobs),
            "exact_kg_parallel_backend": str(args.parallel_backend),
            "tcb_v2_enabled": bool(args.tcb_v2_mode != "off"),
            "tcb_v2_mode": str(args.tcb_v2_mode),
            "tcb_v2_frontier_count": int(args.tcb_v2_frontier_count),
            "tcb_v2_descriptor_mode": str(args.tcb_v2_descriptor_mode),
            "tcb_v2_coordinate": str(args.tcb_v2_coordinate),
            "tcb_v2_geometry": str(args.tcb_v2_geometry),
            "tcb_v2_rank": int(args.tcb_v2_rank),
            "tcb_v2_ridge": float(args.tcb_v2_ridge),
            "tcb_v2_domain_penalty": float(args.tcb_v2_domain_penalty),
            "tcb_v2_boundary_temperature": float(
                args.tcb_v2_boundary_temperature),
            "tcb_v2_adaptation_ridge": float(args.tcb_v2_adaptation_ridge),
            "tcb_v2_upper_alpha": float(args.tcb_v2_upper_alpha),
            "tcb_v2_calibration_prior_df": float(
                args.tcb_v2_calibration_prior_df),
            "tcb_v2_hierarchy_iterations": int(
                args.tcb_v2_hierarchy_iterations),
            "tcb_v2_effect_ridge": float(args.tcb_v2_effect_ridge),
            "tcb_v2_rotation_mode": str(args.tcb_v2_rotation_mode),
            "tcb_v2_rotation_ridge": float(args.tcb_v2_rotation_ridge),
            "tcb_v2_target_residual_rank": int(
                args.tcb_v2_target_residual_rank),
            "tcb_v2_residual_ridge": float(args.tcb_v2_residual_ridge),
            "meta_observable_mean_coordinate": bool(
                args.observable_mean_coordinate),
            "meta_observable_mean_mode": str(
                args.observable_mean_mode),
            "meta_observable_mean_latent_dim": int(
                args.observable_mean_latent_dim),
            "meta_observable_mean_training_target": str(
                args.observable_mean_training_target),
            "meta_source_observation_mode": str(
                args.source_observation_mode),
            "meta_source_observation_replicates": int(
                args.source_observation_replicates),
            "meta_source_design_mode": str(args.source_design_mode),
            "meta_source_universal_fraction": float(
                args.source_universal_fraction),
            "initial_design": str(args.initial_design),
            "initial_design_file": str(args.initial_design_file),
            "initial_design_fingerprint": (
                None if source_design_contract is None
                else source_design_contract["fingerprint"]
            ),
            "initial_design_source_archive_fingerprint": (
                None if source_design_contract is None
                else source_design_contract["source_archive_fingerprint"]
            ),
            "finalist_terminal_value_mode": str(
                args.finalist_terminal_value_mode),
            "finalist_replication_budget": int(
                args.finalist_replication_budget),
            "finalist_replication_count": int(
                args.finalist_replication_count),
            "finalist_observed_safety_count": int(
                args.finalist_observed_safety_count),
            "finalist_replication_min_replicates": int(
                args.finalist_replication_min_replicates),
            "finalist_replication_delta": float(
                args.finalist_replication_delta),
            "finalist_replication_variance_prior_df": float(
                args.finalist_replication_variance_prior_df),
            "finalist_replication_expert_stratified": bool(
                args.finalist_replication_expert_stratified),
            "finalist_replication_adaptive_race": bool(
                args.finalist_replication_adaptive_race),
            "finalist_replication_fixed_universe": bool(
                args.finalist_replication_fixed_universe),
            "finalist_replication_policy": str(
                args.finalist_replication_policy),
            "finalist_empirical_override": str(
                args.finalist_empirical_override),
            "finalist_frontier_policy": str(
                args.finalist_frontier_policy),
            "finalist_terminal_max_arms": int(
                args.finalist_terminal_max_arms),
            "finalist_terminal_mc_samples": int(
                args.finalist_terminal_mc_samples),
            "certification_recheck_top_k": int(
                args.certification_recheck_top_k),
            "certification_recheck_min_replicates": int(
                args.certification_recheck_min_replicates),
            "certification_recheck_soft_margin_scale": float(
                args.certification_recheck_soft_margin_scale),
        },
        "config": config,
        "rows": [row],
    }
    atomic_write_json(args.out, payload)
    print(json.dumps(json_safe(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
