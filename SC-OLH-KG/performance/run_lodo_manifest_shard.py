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
    config.setdefault("policy_improvement_score_normalization", "none")
    config.setdefault("policy_improvement_score_transform", "identity")
    config.setdefault("policy_improvement_guard_mode", "uniform_score")
    config.setdefault("policy_improvement_pairwise_prefix_samples", 32)
    config.setdefault("policy_improvement_pairwise_error_multiplier", 1.25)
    config.setdefault("policy_improvement_confirmation_samples", 4096)
    config.setdefault("policy_improvement_confirmation_batch_samples", 512)
    config.setdefault("policy_improvement_confirmation_delta", 0.05)
    config.setdefault("policy_improvement_confirmation_jobs", 0)
    config.setdefault("policy_improvement_confirmation_lambda_min", 0.001)
    config.setdefault("policy_improvement_confirmation_lambda_count", 24)
    config.setdefault("task_posterior_safe_generalized", False)
    config.setdefault("source_discrepancy_update", True)
    config.setdefault("task_posterior_safe_boundary_score_weight", 1.0)
    config.setdefault("task_posterior_safe_pairwise_score_weight", 1.0)
    config.setdefault("task_posterior_safe_pairwise_max_history", 16)
    config.setdefault("task_posterior_safe_pairwise_probability_floor", 1e-6)
    config.setdefault("task_latent_inference_mode", "shadow")
    config.setdefault("task_variance_posterior_mode", "shared")
    config.setdefault(
        "task_latent_calibration_mode", "source_profiles")
    config.setdefault("adaptive_replication_voi", False)
    config.setdefault("hvd_cumulative_transfer_mode", "scalar")
    config.setdefault("hvd_source_task_weight_mode", "independent")
    config.setdefault(
        "hvd_cumulative_target_evidence_mode", "replication_only")
    config.setdefault("hvd_singleton_evidence_mode", "in_sample_residual")
    config.setdefault(
        "task_posterior_robust_certificate_mode", "separable")
    config.setdefault("certification_head_authority", "task_joint")
    config.setdefault("posterior_dominance_enabled", False)
    config.setdefault("posterior_dominance_delta", 0.05)
    config.setdefault("posterior_dominance_min_mean_gain", 0.0)
    config.setdefault("posterior_dominance_initialization", "risk")
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
    config.setdefault("terminal_verification_budget", 0)
    config.setdefault("terminal_verification_delta", 0.05)
    config.setdefault("terminal_verification_mean_delta_fraction", 0.5)
    config.setdefault(
        "terminal_verification_method", "component_bonferroni")
    config.setdefault("terminal_verification_policy", "fixed_policy")
    config.setdefault("terminal_verification_shortlist_size", 1)
    config.setdefault("terminal_verification_fallback_budget", 0)
    config.setdefault(
        "terminal_verification_shortlist_mode", "posterior_ranked")
    config.setdefault(
        "terminal_objective_challenger_max_violation_probability", 0.5)
    config.setdefault("terminal_objective_incumbent_guard", False)
    config.setdefault("terminal_objective_comparison_budget", 0)
    config.setdefault("terminal_objective_comparison_delta", 0.05)
    config.setdefault(
        "terminal_safe_interior_candidate_scope", "initial")
    config.setdefault(
        "terminal_safe_interior_selection_mode", "diverse")
    config.setdefault(
        "terminal_safe_interior_probability_slack", 0.05)
    config.setdefault(
        "terminal_safe_interior_require_provider", False)
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
    config.setdefault(
        "meta_observable_mean_input_mode", "policy_profile")
    config.setdefault("meta_observable_mean_descriptor_mode", "ordered")
    config.setdefault("meta_observable_mean_feature_mode", "linear")
    config.setdefault("meta_observable_mean_latent_transform", "identity")
    config.setdefault("meta_observable_mean_target_residual_rank", 0)
    config.setdefault("meta_observable_mean_target_residual_prior_scale", 1.0)
    config.setdefault("meta_observable_mean_target_residual_pool_size", 128)
    config.setdefault("meta_observable_mean_target_residual_rcond", 1e-8)
    config.setdefault(
        "meta_observable_mean_role_assignment_posterior", False)
    config.setdefault(
        "meta_observable_mean_role_assignment_prior", "uniform")
    config.setdefault(
        "meta_observable_mean_role_assignment_prior_temperature_scale", 1.0)
    config.setdefault(
        "meta_observable_mean_role_assignment_inactive_variance", 1e-12)
    config.setdefault(
        "meta_observable_variance_input_mode", "legacy_policy_proxy")
    config.setdefault("source_constraint_mean_coefficient_prior", False)
    config.setdefault(
        "source_constraint_mean_hyperlaw_mode", "single_gaussian_draw")
    config.setdefault("source_constraint_mean_adaptation_mode", "frozen")
    config.setdefault(
        "source_constraint_mean_deviation_mode", "raw_independent")
    config.setdefault(
        "source_constraint_mean_misspecification_mode", "none")
    config.setdefault(
        "source_constraint_mean_misspecification_prior_df", 4.0)
    config.setdefault(
        "source_constraint_mean_misspecification_ridge", 1.0)
    config.setdefault(
        "source_constraint_mean_misspecification_max_scale", 100.0)
    config.setdefault(
        "source_constraint_mean_misspecification_delta", 0.05)
    config.setdefault(
        "source_constraint_mean_confidence_mode", "model")
    config.setdefault(
        "source_constraint_mean_confidence_delta", 0.05)
    config.setdefault("source_constraint_mean_contrast_scale", 1.0)
    config.setdefault(
        "source_constraint_mean_role_epistemic_mode", "none")
    config.setdefault("source_constraint_mean_null_weight", 0.5)
    config.setdefault("source_constraint_mean_null_geometry", "isotropic")
    config.setdefault("source_constraint_mean_null_geometry_ridge", 1e-3)
    config.setdefault("source_constraint_mean_evidence_temperature", 1.0)
    config.setdefault(
        "source_constraint_mean_structure_score_mode",
        "marginal_likelihood")
    config.setdefault(
        "source_constraint_mean_residual_rank_posterior", False)
    config.setdefault(
        "source_constraint_mean_residual_rank_prior", "0.70,0.20,0.10")
    config.setdefault(
        "source_constraint_mean_residual_rank_inactive_variance", 1e-12)
    config.setdefault("boundary_coordinate_candidate_count", 0)
    config.setdefault("boundary_coordinate_pool_size", 512)
    config.setdefault("boundary_coordinate_safe_fraction", 0.30)
    config.setdefault("boundary_coordinate_boundary_fraction", 0.40)
    config.setdefault("boundary_coordinate_coverage_fraction", 0.30)
    config.setdefault("truth_pool_diagnostics", False)
    config.setdefault("truth_pool_max_candidates", 0)
    config.setdefault("meta_source_observation_mode", "analytic")
    config.setdefault("meta_source_observation_replicates", 1)
    config.setdefault("meta_source_design_mode", "random")
    config.setdefault("meta_source_universal_fraction", 0.75)
    config.setdefault("meta_source_augments", 1)
    config.setdefault("meta_source_budget_mode", "per_episode")
    config.setdefault("meta_source_geometry_shift_scale", 0.0)
    config.setdefault(
        "meta_source_geometry_log_radius_jitter", 0.0)
    config.setdefault("meta_source_sigma_jitter", 0.20)
    config.setdefault("meta_source_alpha_jitter", 0.25)
    config.setdefault("meta_source_weight_jitter", 0.05)
    config.setdefault("source_records_per_domain", 96)
    config.setdefault("initial_design", "auto")
    config.setdefault(
        "initial_design_archive_match_mode", "exact")
    config.setdefault("target_shared_shock_scale", 1.0)
    config.setdefault("variance_audit_size", 128)
    config.setdefault("decision_backend", "legacy")
    config.setdefault("decision_terminal_rule", "bayes_risk")
    config.setdefault(
        "decision_terminal_maximum_violation_probability", 0.05)
    config.setdefault("decision_risk_penalty", 5.0)
    config.setdefault("decision_aleatoric_mode", "certification_upper")
    config.setdefault("decision_violation_loss_mode", "positive_part")
    config.setdefault("decision_ambiguity_mode", "kl_robust")
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
    parser.add_argument(
        "--runtime-checkpoint-interval",
        type=int,
        default=0,
        help="Checkpoint every N stages; 0 disables runtime checkpoints.",
    )
    parser.add_argument("--evaluate-interval", type=int, default=20)
    parser.add_argument("--d", type=int, default=None)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--n0", type=int, default=None)
    parser.add_argument(
        "--implementation-contract-id",
        default="unversioned",
    )
    parser.add_argument(
        "--theory-contract-id",
        default="unversioned",
    )
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
        "--task-posterior-robust-certificate-mode",
        choices=("separable", "joint_tangent"),
        default="separable",
    )
    parser.add_argument(
        "--certification-head-authority",
        choices=(
            "task_joint",
            "split_gpr_task_hvd",
            "split_gpr_cumulative_hvd",
        ),
        default="task_joint",
    )
    parser.add_argument(
        "--task-latent-inference-mode",
        choices=("shadow", "authoritative"),
        default="shadow",
    )
    parser.add_argument(
        "--task-variance-posterior-mode",
        choices=("shared", "replication_only"),
        default="shared",
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
            "sobol", "sobol_new", "sobol_hvd_voi", "sobol_joint_voi",
            "risk_ts", "constrained_ts",
            "sobol_exact_joint_voi",
            "certificate_depth_new",
            "bayes_risk_ei", "constrained_ei", "transfer_utility",
        ),
        default="legacy",
    )
    parser.add_argument("--decision-risk-penalty", type=float, default=5.0)
    parser.add_argument(
        "--decision-terminal-rule",
        choices=("bayes_risk", "feasible_first"),
        default="bayes_risk",
    )
    parser.add_argument(
        "--decision-terminal-maximum-violation-probability",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--decision-aleatoric-mode",
        choices=("certification_upper", "posterior_central"),
        default="certification_upper",
    )
    parser.add_argument(
        "--decision-violation-loss-mode",
        choices=("positive_part", "failure_probability"),
        default="positive_part",
    )
    parser.add_argument(
        "--decision-ambiguity-mode",
        choices=("kl_robust", "posterior_nominal"),
        default="kl_robust",
    )
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
            "bayes_risk_dominance",
            "certified_lexicographic",
            "tcb_certified_lexicographic",
        ),
        default="hard_certified",
    )
    parser.add_argument(
        "--adaptive-replication-voi",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--replication-candidate-count", type=int, default=None)
    parser.add_argument(
        "--replication-max-per-solution", type=int, default=None)
    parser.add_argument(
        "--evaluate-or-replicate-new-action-count", type=int, default=None)
    parser.add_argument(
        "--evaluate-or-replicate-new-action-policy",
        choices=(
            "canonical_sobol",
            "canonical_plus_posterior_risk",
            "canonical_plus_posterior_risk_certificate_coverage",
            "canonical_plus_posterior_pareto_support",
            "canonical_plus_posterior_guard_decomposition",
        ),
        default=None,
    )
    parser.add_argument(
        "--evaluate-or-replicate-baseline-new-action-count",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--policy-improvement-mode",
        choices=(
            "off", "action_superset", "guarded_rollout", "joint",
            "certificate_constrained",
        ),
        default=None,
    )
    parser.add_argument(
        "--policy-improvement-score-normalization",
        choices=("none", "current_terminal"),
        default=None,
    )
    parser.add_argument(
        "--policy-improvement-score-transform",
        choices=("identity", "bounded_current_gain"),
        default=None,
    )
    parser.add_argument(
        "--policy-improvement-guard-mode",
        choices=(
            "uniform_score",
            "paired_nested_difference",
            "paired_nested_absolute",
            "independent_confirmation",
        ),
        default=None,
    )
    parser.add_argument(
        "--policy-improvement-pairwise-prefix-samples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--policy-improvement-pairwise-error-multiplier",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--policy-improvement-confirmation-samples", type=int, default=None)
    parser.add_argument(
        "--policy-improvement-confirmation-batch-samples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--policy-improvement-confirmation-delta", type=float, default=None)
    parser.add_argument(
        "--policy-improvement-confirmation-jobs", type=int, default=None)
    parser.add_argument(
        "--policy-improvement-confirmation-lambda-min",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--policy-improvement-confirmation-lambda-count",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--policy-improvement-mc-error-bound", type=float, default=None)
    parser.add_argument(
        "--policy-improvement-certificate-mc-error-bound",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--policy-improvement-rollout-depth", type=int, default=None)
    parser.add_argument(
        "--policy-improvement-rollout-max-arms", type=int, default=None)
    parser.add_argument(
        "--policy-improvement-rollout-mc-samples", type=int, default=None)
    parser.add_argument(
        "--policy-improvement-rollout-mc-error-bound",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--safe-interior-candidate-count", type=int, default=None)
    parser.add_argument("--safe-interior-pool-size", type=int, default=None)
    parser.add_argument("--safe-interior-margin", type=float, default=None)
    parser.add_argument(
        "--posterior-dominance-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--posterior-dominance-delta", type=float, default=0.05)
    parser.add_argument(
        "--posterior-dominance-min-mean-gain", type=float, default=0.0)
    parser.add_argument(
        "--posterior-dominance-initialization",
        choices=("risk", "certificate_lexicographic", "certified_only"),
        default="risk",
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
        "--exact-clip-negative",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--exact-reuse-nested-prefix",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--exact-skip-redundant-primary-update",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--exact-chunk-schedule",
        choices=("legacy", "balanced_lcm"),
        default="balanced_lcm",
    )
    parser.add_argument(
        "--exact-max-chunks-per-candidate", type=int, default=8)
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
        choices=("random", "universal_mixture", "shared_uniform"),
        default="random",
    )
    parser.add_argument("--source-universal-fraction", type=float, default=0.75)
    parser.add_argument(
        "--source-consensus-template-count", type=int, default=0)
    parser.add_argument("--source-records-per-domain", type=int, default=None)
    parser.add_argument("--meta-source-augments", type=int, default=None)
    parser.add_argument(
        "--meta-source-budget-mode",
        choices=("per_episode", "per_base_domain"),
        default=None,
    )
    parser.add_argument(
        "--meta-source-geometry-shift-scale",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--meta-source-geometry-log-radius-jitter",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--meta-source-sigma-jitter", type=float, default=None)
    parser.add_argument(
        "--meta-source-alpha-jitter", type=float, default=None)
    parser.add_argument(
        "--meta-source-weight-jitter", type=float, default=None)
    parser.add_argument("--target-shared-shock-scale", type=float, default=1.0)
    parser.add_argument("--variance-audit-size", type=int, default=128)
    parser.add_argument(
        "--meta-source-d",
        type=int,
        default=None,
        help="Train the frozen source prior at this policy dimension.",
    )
    parser.add_argument(
        "--initial-design",
        choices=("auto", "common_sobol", "source_informed"),
        default="auto",
    )
    parser.add_argument("--initial-design-file", default="")
    parser.add_argument(
        "--initial-design-archive-match-mode",
        choices=("exact", "paired_frozen_control"),
        default="exact",
        help="Require one archive or freeze the proposal for a paired intervention.",
    )
    parser.add_argument(
        "--observable-mean-mode",
        choices=(
            "atoms", "aggregate", "latent", "consensus", "source_affine",
            "source_rank", "boundary_aligned",
        ),
        default="latent",
    )
    parser.add_argument("--observable-mean-latent-dim", type=int, default=2)
    parser.add_argument(
        "--observable-mean-training-target",
        choices=("constraint_mean", "chance_margin"),
        default="constraint_mean",
    )
    parser.add_argument(
        "--observable-mean-input-mode",
        choices=(
            "policy_profile",
            "source_learned_exposure",
            "observable_state_exposure",
            "provider_exposure",
        ),
        default="policy_profile",
    )
    parser.add_argument(
        "--observable-mean-descriptor-mode",
        choices=(
            "ordered",
            "set_invariant",
            "role_aligned",
            "role_transport",
            "role_intervention_transport",
            "role_adaptive_ordered",
            "role_adaptive_set_invariant",
            "exchangeable_equivariant",
        ),
        default="ordered",
    )
    parser.add_argument(
        "--observable-mean-feature-mode",
        choices=("linear", "diagonal_quadratic", "full_quadratic"),
        default="linear",
    )
    parser.add_argument(
        "--observable-mean-latent-transform",
        choices=(
            "identity",
            "source_tanh",
            "source_support_clip",
            "source_support_residual",
        ),
        default="identity",
    )
    parser.add_argument(
        "--observable-mean-target-residual-rank", type=int, default=0)
    parser.add_argument(
        "--observable-mean-target-residual-prior-scale",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--observable-mean-target-residual-pool-size", type=int, default=128)
    parser.add_argument(
        "--observable-mean-target-residual-rcond", type=float, default=1e-8)
    parser.add_argument(
        "--observable-mean-role-assignment-posterior",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--observable-mean-role-assignment-prior",
        choices=(
            "uniform", "source_geometry", "source_geometry_boundary",
        ),
        default="uniform",
    )
    parser.add_argument(
        "--observable-mean-role-assignment-prior-temperature-scale",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--observable-mean-role-assignment-inactive-variance",
        type=float,
        default=1e-12,
    )
    parser.add_argument(
        "--observable-variance-input-mode",
        choices=("legacy_policy_proxy", "observable_state_exposure"),
        default="legacy_policy_proxy",
    )
    parser.add_argument(
        "--source-constraint-mean-coefficient-prior",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--source-constraint-mean-hyperlaw-mode",
        choices=(
            "single_gaussian_draw",
            "shared_low_rank_discrepancy",
            "shared_low_rank_predictive",
            "grouped_task_discrepancy",
            "grouped_task_predictive",
            "grouped_task_deconvolved",
            "grouped_task_deconvolved_predictive",
        ),
        default="single_gaussian_draw",
    )
    parser.add_argument(
        "--source-constraint-mean-adaptation-mode",
        choices=(
            "frozen", "evidence_mixture", "sequential_evidence_mixture",
            "aggregate_mixture", "sequential_aggregate_mixture",
            "sequential_aggregate_hyperlaw",
            "support_adaptive_aggregate_mixture",
            "sequential_support_adaptive_aggregate_mixture",
        ),
        default="frozen",
    )
    parser.add_argument(
        "--source-constraint-mean-deviation-mode",
        choices=("raw_independent", "latent_shared"),
        default="raw_independent",
    )
    parser.add_argument(
        "--source-constraint-mean-misspecification-mode",
        choices=(
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
        ),
        default="none",
    )
    parser.add_argument(
        "--source-constraint-mean-misspecification-prior-df",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--source-constraint-mean-misspecification-ridge",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--source-constraint-mean-misspecification-max-scale",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--source-constraint-mean-misspecification-delta",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--source-constraint-mean-confidence-mode",
        choices=("model", "source_bayes", "source_self_normalized"),
        default="model",
    )
    parser.add_argument(
        "--source-constraint-mean-confidence-delta",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--source-constraint-mean-contrast-scale", type=float, default=1.0)
    parser.add_argument(
        "--source-constraint-mean-role-epistemic-mode",
        choices=("none", "matching_loss", "matching_uncertainty"),
        default="none",
    )
    parser.add_argument(
        "--hvd-source-task-weight-mode",
        choices=("independent", "constraint_mean"),
        default="independent",
    )
    parser.add_argument(
        "--hvd-cumulative-target-evidence-mode",
        choices=("replication_only", "prequential_upper"),
        default="replication_only",
    )
    parser.add_argument(
        "--hvd-singleton-evidence-mode",
        choices=("in_sample_residual", "source_prior"),
        default="in_sample_residual",
    )
    parser.add_argument(
        "--source-constraint-mean-null-weight", type=float, default=0.5)
    parser.add_argument(
        "--source-constraint-mean-null-geometry",
        choices=("isotropic", "target_pool"),
        default="isotropic",
    )
    parser.add_argument(
        "--source-constraint-mean-null-geometry-ridge",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--source-constraint-mean-evidence-temperature",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--source-constraint-mean-structure-score-mode",
        choices=(
            "marginal_likelihood",
            "loo_predictive",
            "geometry_conditional",
        ),
        default="marginal_likelihood",
    )
    parser.add_argument(
        "--source-constraint-mean-residual-rank-posterior",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--source-constraint-mean-residual-rank-prior",
        default="0.70,0.20,0.10",
    )
    parser.add_argument(
        "--source-constraint-mean-residual-rank-inactive-variance",
        type=float,
        default=1e-12,
    )
    parser.add_argument(
        "--boundary-coordinate-candidate-count", type=int, default=0)
    parser.add_argument(
        "--boundary-coordinate-pool-size", type=int, default=512)
    parser.add_argument(
        "--boundary-coordinate-safe-fraction", type=float, default=0.30)
    parser.add_argument(
        "--boundary-coordinate-boundary-fraction", type=float, default=0.40)
    parser.add_argument(
        "--boundary-coordinate-coverage-fraction", type=float, default=0.30)
    parser.add_argument(
        "--truth-pool-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Post-decision synthetic oracle audit; never changes selection.",
    )
    parser.add_argument(
        "--truth-pool-max-candidates",
        type=int,
        default=None,
        help="Optional post-run truth-audit cap; zero audits the whole pool.",
    )
    parser.add_argument(
        "--finalist-terminal-value-mode",
        choices=("model_default", "certified_lexicographic"),
        default="model_default",
    )
    parser.add_argument(
        "--terminal-frontier-candidate-count",
        type=int,
        default=None,
        help=(
            "Optional per-run override for the manifest's terminal frontier "
            "budget."
        ),
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
    parser.add_argument(
        "--terminal-verification-budget", type=int, default=None)
    parser.add_argument(
        "--terminal-verification-delta", type=float, default=None)
    parser.add_argument(
        "--terminal-verification-mean-delta-fraction",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--terminal-verification-method",
        choices=("component_bonferroni", "normal_quantile_tolerance"),
        default=None,
    )
    parser.add_argument(
        "--terminal-verification-policy",
        choices=("fixed_policy", "ordered_frozen_shortlist"),
        default=None,
    )
    parser.add_argument(
        "--terminal-verification-shortlist-size", type=int, default=None)
    parser.add_argument(
        "--terminal-verification-fallback-budget", type=int, default=None)
    parser.add_argument(
        "--terminal-verification-shortlist-mode",
        choices=(
            "posterior_ranked",
            "posterior_primary_safe_interior",
            "posterior_objective_challenger_then_safe",
        ),
        default=None,
    )
    parser.add_argument(
        "--terminal-objective-challenger-max-violation-probability",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--terminal-objective-incumbent-guard",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--terminal-objective-comparison-budget",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--terminal-objective-comparison-delta",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--terminal-safe-interior-candidate-scope",
        choices=("initial", "observed"),
        default=None,
    )
    parser.add_argument(
        "--terminal-safe-interior-selection-mode",
        choices=("diverse", "objective_ranked", "objective_safe_ranked"),
        default=None,
    )
    parser.add_argument(
        "--terminal-safe-interior-probability-slack",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--terminal-safe-interior-require-provider",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
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
    config["implementation_contract_id"] = str(
        args.implementation_contract_id)
    config["theory_contract_id"] = str(args.theory_contract_id)
    if int(config["n0"]) > int(config["N"]):
        raise ValueError("LODO shard requires n0 <= N")
    terminal_verification_budget = int(
        config["terminal_verification_budget"]
        if args.terminal_verification_budget is None
        else args.terminal_verification_budget
    )
    terminal_verification_delta = float(
        config["terminal_verification_delta"]
        if args.terminal_verification_delta is None
        else args.terminal_verification_delta
    )
    terminal_verification_mean_delta_fraction = float(
        config["terminal_verification_mean_delta_fraction"]
        if args.terminal_verification_mean_delta_fraction is None
        else args.terminal_verification_mean_delta_fraction
    )
    terminal_verification_method = str(
        config["terminal_verification_method"]
        if args.terminal_verification_method is None
        else args.terminal_verification_method
    )
    terminal_verification_policy = str(
        config["terminal_verification_policy"]
        if args.terminal_verification_policy is None
        else args.terminal_verification_policy
    )
    terminal_verification_shortlist_size = int(
        config["terminal_verification_shortlist_size"]
        if args.terminal_verification_shortlist_size is None
        else args.terminal_verification_shortlist_size
    )
    terminal_verification_fallback_budget = int(
        config["terminal_verification_fallback_budget"]
        if args.terminal_verification_fallback_budget is None
        else args.terminal_verification_fallback_budget
    )
    terminal_verification_shortlist_mode = str(
        config["terminal_verification_shortlist_mode"]
        if args.terminal_verification_shortlist_mode is None
        else args.terminal_verification_shortlist_mode
    )
    terminal_objective_challenger_max_violation_probability = float(
        config[
            "terminal_objective_challenger_max_violation_probability"
        ]
        if (
            args
            .terminal_objective_challenger_max_violation_probability
            is None
        )
        else (
            args
            .terminal_objective_challenger_max_violation_probability
        )
    )
    terminal_objective_incumbent_guard = bool(
        config["terminal_objective_incumbent_guard"]
        if args.terminal_objective_incumbent_guard is None
        else args.terminal_objective_incumbent_guard
    )
    terminal_objective_comparison_budget = int(
        config["terminal_objective_comparison_budget"]
        if args.terminal_objective_comparison_budget is None
        else args.terminal_objective_comparison_budget
    )
    terminal_objective_comparison_delta = float(
        config["terminal_objective_comparison_delta"]
        if args.terminal_objective_comparison_delta is None
        else args.terminal_objective_comparison_delta
    )
    terminal_safe_interior_candidate_scope = str(
        config["terminal_safe_interior_candidate_scope"]
        if args.terminal_safe_interior_candidate_scope is None
        else args.terminal_safe_interior_candidate_scope
    )
    terminal_safe_interior_selection_mode = str(
        config["terminal_safe_interior_selection_mode"]
        if args.terminal_safe_interior_selection_mode is None
        else args.terminal_safe_interior_selection_mode
    )
    terminal_safe_interior_probability_slack = float(
        config["terminal_safe_interior_probability_slack"]
        if args.terminal_safe_interior_probability_slack is None
        else args.terminal_safe_interior_probability_slack
    )
    terminal_safe_interior_require_provider = bool(
        config["terminal_safe_interior_require_provider"]
        if args.terminal_safe_interior_require_provider is None
        else args.terminal_safe_interior_require_provider
    )
    replication_candidate_count = int(
        config.get("replication_candidate_count", 0)
        if args.replication_candidate_count is None
        else args.replication_candidate_count
    )
    replication_max_per_solution = int(
        config.get("replication_max_per_solution", 5)
        if args.replication_max_per_solution is None
        else args.replication_max_per_solution
    )
    evaluate_or_replicate_new_action_count = int(
        config.get("evaluate_or_replicate_new_action_count", 1)
        if args.evaluate_or_replicate_new_action_count is None
        else args.evaluate_or_replicate_new_action_count
    )
    evaluate_or_replicate_new_action_policy = str(
        config.get(
            "evaluate_or_replicate_new_action_policy", "canonical_sobol")
        if args.evaluate_or_replicate_new_action_policy is None
        else args.evaluate_or_replicate_new_action_policy
    )
    evaluate_or_replicate_baseline_new_action_count = int(
        config.get("evaluate_or_replicate_baseline_new_action_count", 0)
        if args.evaluate_or_replicate_baseline_new_action_count is None
        else args.evaluate_or_replicate_baseline_new_action_count
    )
    policy_improvement_mode = str(
        config.get("policy_improvement_mode", "off")
        if args.policy_improvement_mode is None
        else args.policy_improvement_mode
    )
    policy_improvement_score_normalization = str(
        config.get("policy_improvement_score_normalization", "none")
        if args.policy_improvement_score_normalization is None
        else args.policy_improvement_score_normalization
    )
    policy_improvement_score_transform = str(
        config.get("policy_improvement_score_transform", "identity")
        if args.policy_improvement_score_transform is None
        else args.policy_improvement_score_transform
    )
    policy_improvement_guard_mode = str(
        config.get("policy_improvement_guard_mode", "uniform_score")
        if args.policy_improvement_guard_mode is None
        else args.policy_improvement_guard_mode
    )
    policy_improvement_pairwise_prefix_samples = int(
        config.get("policy_improvement_pairwise_prefix_samples", 32)
        if args.policy_improvement_pairwise_prefix_samples is None
        else args.policy_improvement_pairwise_prefix_samples
    )
    policy_improvement_pairwise_error_multiplier = float(
        config.get("policy_improvement_pairwise_error_multiplier", 1.25)
        if args.policy_improvement_pairwise_error_multiplier is None
        else args.policy_improvement_pairwise_error_multiplier
    )
    policy_improvement_confirmation_samples = int(
        config.get("policy_improvement_confirmation_samples", 4096)
        if args.policy_improvement_confirmation_samples is None
        else args.policy_improvement_confirmation_samples
    )
    policy_improvement_confirmation_batch_samples = int(
        config.get("policy_improvement_confirmation_batch_samples", 512)
        if args.policy_improvement_confirmation_batch_samples is None
        else args.policy_improvement_confirmation_batch_samples
    )
    policy_improvement_confirmation_delta = float(
        config.get("policy_improvement_confirmation_delta", 0.05)
        if args.policy_improvement_confirmation_delta is None
        else args.policy_improvement_confirmation_delta
    )
    policy_improvement_confirmation_jobs = int(
        config.get("policy_improvement_confirmation_jobs", 0)
        if args.policy_improvement_confirmation_jobs is None
        else args.policy_improvement_confirmation_jobs
    )
    policy_improvement_confirmation_lambda_min = float(
        config.get("policy_improvement_confirmation_lambda_min", 0.001)
        if args.policy_improvement_confirmation_lambda_min is None
        else args.policy_improvement_confirmation_lambda_min
    )
    policy_improvement_confirmation_lambda_count = int(
        config.get("policy_improvement_confirmation_lambda_count", 24)
        if args.policy_improvement_confirmation_lambda_count is None
        else args.policy_improvement_confirmation_lambda_count
    )
    policy_improvement_mc_error_bound = float(
        config.get("policy_improvement_mc_error_bound", 0.0)
        if args.policy_improvement_mc_error_bound is None
        else args.policy_improvement_mc_error_bound
    )
    policy_improvement_certificate_mc_error_bound = float(
        config.get("policy_improvement_certificate_mc_error_bound", 0.0)
        if args.policy_improvement_certificate_mc_error_bound is None
        else args.policy_improvement_certificate_mc_error_bound
    )
    policy_improvement_rollout_depth = int(
        config.get("policy_improvement_rollout_depth", 1)
        if args.policy_improvement_rollout_depth is None
        else args.policy_improvement_rollout_depth
    )
    policy_improvement_rollout_max_arms = int(
        config.get("policy_improvement_rollout_max_arms", 4)
        if args.policy_improvement_rollout_max_arms is None
        else args.policy_improvement_rollout_max_arms
    )
    policy_improvement_rollout_mc_samples = int(
        config.get("policy_improvement_rollout_mc_samples", 2)
        if args.policy_improvement_rollout_mc_samples is None
        else args.policy_improvement_rollout_mc_samples
    )
    policy_improvement_rollout_mc_error_bound = float(
        config.get("policy_improvement_rollout_mc_error_bound", 0.0)
        if args.policy_improvement_rollout_mc_error_bound is None
        else args.policy_improvement_rollout_mc_error_bound
    )
    safe_interior_candidate_count = int(
        config.get("safe_interior_candidate_count", 0)
        if args.safe_interior_candidate_count is None
        else args.safe_interior_candidate_count
    )
    safe_interior_pool_size = int(
        config.get("safe_interior_pool_size", 300)
        if args.safe_interior_pool_size is None
        else args.safe_interior_pool_size
    )
    safe_interior_margin = float(
        config.get("safe_interior_margin", 0.0)
        if args.safe_interior_margin is None
        else args.safe_interior_margin
    )
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
        "task_posterior_robust_certificate_mode": str(
            args.task_posterior_robust_certificate_mode),
        "certification_head_authority": str(
            args.certification_head_authority),
        "task_latent_inference_mode": str(
            args.task_latent_inference_mode),
        "task_variance_posterior_mode": str(
            args.task_variance_posterior_mode),
        "task_latent_calibration_mode": str(
            args.task_latent_calibration_mode),
        "exact_kg_sampling_mode": str(args.exact_sampling_mode),
        "decision_backend": str(args.decision_backend),
        "decision_terminal_rule": str(args.decision_terminal_rule),
        "decision_terminal_maximum_violation_probability": float(
            args.decision_terminal_maximum_violation_probability),
        "decision_risk_penalty": float(args.decision_risk_penalty),
        "decision_aleatoric_mode": str(args.decision_aleatoric_mode),
        "decision_violation_loss_mode": str(
            args.decision_violation_loss_mode),
        "decision_ambiguity_mode": str(args.decision_ambiguity_mode),
        "decision_source_utility_weight": float(
            args.decision_source_utility_weight),
        "decision_recommend_observed_only": bool(
            args.decision_recommend_observed_only),
        "exact_kg_terminal_mode": str(args.exact_terminal_mode),
        "adaptive_replication_voi": bool(args.adaptive_replication_voi),
        "replication_candidate_count": replication_candidate_count,
        "replication_max_per_solution": replication_max_per_solution,
        "evaluate_or_replicate_new_action_count": (
            evaluate_or_replicate_new_action_count),
        "evaluate_or_replicate_new_action_policy": (
            evaluate_or_replicate_new_action_policy),
        "evaluate_or_replicate_baseline_new_action_count": (
            evaluate_or_replicate_baseline_new_action_count),
        "policy_improvement_mode": policy_improvement_mode,
        "policy_improvement_score_normalization": (
            policy_improvement_score_normalization),
        "policy_improvement_score_transform": (
            policy_improvement_score_transform),
        "policy_improvement_guard_mode": policy_improvement_guard_mode,
        "policy_improvement_pairwise_prefix_samples": (
            policy_improvement_pairwise_prefix_samples),
        "policy_improvement_pairwise_error_multiplier": (
            policy_improvement_pairwise_error_multiplier),
        "policy_improvement_confirmation_samples": (
            policy_improvement_confirmation_samples),
        "policy_improvement_confirmation_batch_samples": (
            policy_improvement_confirmation_batch_samples),
        "policy_improvement_confirmation_delta": (
            policy_improvement_confirmation_delta),
        "policy_improvement_confirmation_jobs": (
            policy_improvement_confirmation_jobs),
        "policy_improvement_confirmation_lambda_min": (
            policy_improvement_confirmation_lambda_min),
        "policy_improvement_confirmation_lambda_count": (
            policy_improvement_confirmation_lambda_count),
        "policy_improvement_mc_error_bound": (
            policy_improvement_mc_error_bound),
        "policy_improvement_certificate_mc_error_bound": (
            policy_improvement_certificate_mc_error_bound),
        "policy_improvement_rollout_depth": (
            policy_improvement_rollout_depth),
        "policy_improvement_rollout_max_arms": (
            policy_improvement_rollout_max_arms),
        "policy_improvement_rollout_mc_samples": (
            policy_improvement_rollout_mc_samples),
        "policy_improvement_rollout_mc_error_bound": (
            policy_improvement_rollout_mc_error_bound),
        "safe_interior_candidate_count": safe_interior_candidate_count,
        "safe_interior_pool_size": safe_interior_pool_size,
        "safe_interior_margin": safe_interior_margin,
        "posterior_dominance_enabled": bool(
            args.posterior_dominance_enabled),
        "posterior_dominance_delta": float(
            args.posterior_dominance_delta),
        "posterior_dominance_min_mean_gain": float(
            args.posterior_dominance_min_mean_gain),
        "posterior_dominance_initialization": str(
            args.posterior_dominance_initialization),
        "decision_contract_mode": str(args.decision_contract_mode),
        "exact_kg_mc_samples": int(args.exact_mc_samples),
        "exact_kg_jobs": int(args.exact_jobs),
        "exact_kg_parallel_backend": str(args.parallel_backend),
        "exact_kg_clip_negative": bool(args.exact_clip_negative),
        "exact_kg_reuse_nested_prefix": bool(
            args.exact_reuse_nested_prefix),
        "exact_kg_skip_redundant_primary_update": bool(
            args.exact_skip_redundant_primary_update),
        "exact_kg_chunk_schedule": str(
            args.exact_chunk_schedule),
        "exact_kg_max_chunks_per_candidate": int(
            args.exact_max_chunks_per_candidate),
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
        "meta_observable_mean_input_mode": str(
            args.observable_mean_input_mode),
        "meta_observable_mean_descriptor_mode": str(
            args.observable_mean_descriptor_mode),
        "meta_observable_mean_feature_mode": str(
            args.observable_mean_feature_mode),
        "meta_observable_mean_latent_transform": str(
            args.observable_mean_latent_transform),
        "meta_observable_mean_target_residual_rank": int(
            args.observable_mean_target_residual_rank),
        "meta_observable_mean_target_residual_prior_scale": float(
            args.observable_mean_target_residual_prior_scale),
        "meta_observable_mean_target_residual_pool_size": int(
            args.observable_mean_target_residual_pool_size),
        "meta_observable_mean_target_residual_rcond": float(
            args.observable_mean_target_residual_rcond),
        "meta_observable_mean_role_assignment_posterior": bool(
            args.observable_mean_role_assignment_posterior),
        "meta_observable_mean_role_assignment_prior": str(
            args.observable_mean_role_assignment_prior),
        "meta_observable_mean_role_assignment_prior_temperature_scale": float(
            args.observable_mean_role_assignment_prior_temperature_scale),
        "meta_observable_mean_role_assignment_inactive_variance": float(
            args.observable_mean_role_assignment_inactive_variance),
        "meta_observable_variance_input_mode": str(
            args.observable_variance_input_mode),
        "source_constraint_mean_coefficient_prior": bool(
            args.source_constraint_mean_coefficient_prior),
        "source_constraint_mean_hyperlaw_mode": str(
            args.source_constraint_mean_hyperlaw_mode),
        "source_constraint_mean_adaptation_mode": str(
            args.source_constraint_mean_adaptation_mode),
        "source_constraint_mean_deviation_mode": str(
            args.source_constraint_mean_deviation_mode),
        "source_constraint_mean_misspecification_mode": str(
            args.source_constraint_mean_misspecification_mode),
        "source_constraint_mean_misspecification_prior_df": float(
            args.source_constraint_mean_misspecification_prior_df),
        "source_constraint_mean_misspecification_ridge": float(
            args.source_constraint_mean_misspecification_ridge),
        "source_constraint_mean_misspecification_max_scale": float(
            args.source_constraint_mean_misspecification_max_scale),
        "source_constraint_mean_misspecification_delta": float(
            args.source_constraint_mean_misspecification_delta),
        "source_constraint_mean_confidence_mode": str(
            args.source_constraint_mean_confidence_mode),
        "source_constraint_mean_confidence_delta": float(
            args.source_constraint_mean_confidence_delta),
        "source_constraint_mean_contrast_scale": float(
            args.source_constraint_mean_contrast_scale),
        "source_constraint_mean_role_epistemic_mode": str(
            args.source_constraint_mean_role_epistemic_mode),
        "hvd_source_task_weight_mode": str(
            args.hvd_source_task_weight_mode),
        "hvd_cumulative_target_evidence_mode": str(
            args.hvd_cumulative_target_evidence_mode),
        "hvd_singleton_evidence_mode": str(
            args.hvd_singleton_evidence_mode),
        "source_constraint_mean_null_weight": float(
            args.source_constraint_mean_null_weight),
        "source_constraint_mean_null_geometry": str(
            args.source_constraint_mean_null_geometry),
        "source_constraint_mean_null_geometry_ridge": float(
            args.source_constraint_mean_null_geometry_ridge),
        "source_constraint_mean_evidence_temperature": float(
            args.source_constraint_mean_evidence_temperature),
        "source_constraint_mean_structure_score_mode": str(
            args.source_constraint_mean_structure_score_mode),
        "source_constraint_mean_residual_rank_posterior": bool(
            args.source_constraint_mean_residual_rank_posterior),
        "source_constraint_mean_residual_rank_prior": str(
            args.source_constraint_mean_residual_rank_prior),
        "source_constraint_mean_residual_rank_inactive_variance": float(
            args.source_constraint_mean_residual_rank_inactive_variance),
        "boundary_coordinate_candidate_count": int(
            args.boundary_coordinate_candidate_count),
        "boundary_coordinate_pool_size": int(
            args.boundary_coordinate_pool_size),
        "boundary_coordinate_safe_fraction": float(
            args.boundary_coordinate_safe_fraction),
        "boundary_coordinate_boundary_fraction": float(
            args.boundary_coordinate_boundary_fraction),
        "boundary_coordinate_coverage_fraction": float(
            args.boundary_coordinate_coverage_fraction),
        "truth_pool_diagnostics": bool(
            config.get("truth_pool_diagnostics", False)
            if args.truth_pool_diagnostics is None
            else args.truth_pool_diagnostics
        ),
        "truth_pool_max_candidates": int(
            config.get("truth_pool_max_candidates", 0)
            if args.truth_pool_max_candidates is None
            else args.truth_pool_max_candidates
        ),
        "meta_source_observation_mode": str(
            args.source_observation_mode),
        "meta_source_observation_replicates": int(
            args.source_observation_replicates),
        "meta_source_design_mode": str(args.source_design_mode),
        "meta_source_universal_fraction": float(
            args.source_universal_fraction),
        "meta_source_consensus_template_count": int(
            args.source_consensus_template_count),
        "source_records_per_domain": int(
            config["source_records_per_domain"]
            if args.source_records_per_domain is None
            else args.source_records_per_domain
        ),
        "meta_source_augments": int(
            config["meta_source_augments"]
            if args.meta_source_augments is None
            else args.meta_source_augments
        ),
        "meta_source_budget_mode": str(
            config["meta_source_budget_mode"]
            if args.meta_source_budget_mode is None
            else args.meta_source_budget_mode
        ),
        "meta_source_geometry_shift_scale": float(
            config["meta_source_geometry_shift_scale"]
            if args.meta_source_geometry_shift_scale is None
            else args.meta_source_geometry_shift_scale
        ),
        "meta_source_geometry_log_radius_jitter": float(
            config["meta_source_geometry_log_radius_jitter"]
            if args.meta_source_geometry_log_radius_jitter is None
            else args.meta_source_geometry_log_radius_jitter
        ),
        "meta_source_sigma_jitter": float(
            config["meta_source_sigma_jitter"]
            if args.meta_source_sigma_jitter is None
            else args.meta_source_sigma_jitter
        ),
        "meta_source_alpha_jitter": float(
            config["meta_source_alpha_jitter"]
            if args.meta_source_alpha_jitter is None
            else args.meta_source_alpha_jitter
        ),
        "meta_source_weight_jitter": float(
            config["meta_source_weight_jitter"]
            if args.meta_source_weight_jitter is None
            else args.meta_source_weight_jitter
        ),
        "target_shared_shock_scale": float(
            args.target_shared_shock_scale),
        "variance_audit_size": int(args.variance_audit_size),
        "meta_source_dimension": int(
            config["d"] if args.meta_source_d is None else args.meta_source_d),
        "initial_design": str(args.initial_design),
        "initial_design_archive_match_mode": str(
            args.initial_design_archive_match_mode),
        "terminal_frontier_candidate_count": int(
            config.get("terminal_frontier_candidate_count", 0)
            if args.terminal_frontier_candidate_count is None
            else args.terminal_frontier_candidate_count
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
        "terminal_verification_budget": int(
            terminal_verification_budget),
        "terminal_verification_delta": float(
            terminal_verification_delta),
        "terminal_verification_mean_delta_fraction": float(
            terminal_verification_mean_delta_fraction),
        "terminal_verification_method": str(
            terminal_verification_method),
        "terminal_verification_policy": str(
            terminal_verification_policy),
        "terminal_verification_shortlist_size": int(
            terminal_verification_shortlist_size),
        "terminal_verification_fallback_budget": int(
            terminal_verification_fallback_budget),
        "terminal_verification_shortlist_mode": str(
            terminal_verification_shortlist_mode),
        "terminal_objective_challenger_max_violation_probability": float(
            terminal_objective_challenger_max_violation_probability),
        "terminal_objective_incumbent_guard": bool(
            terminal_objective_incumbent_guard),
        "terminal_objective_comparison_budget": int(
            terminal_objective_comparison_budget),
        "terminal_objective_comparison_delta": float(
            terminal_objective_comparison_delta),
        "terminal_safe_interior_candidate_scope": str(
            terminal_safe_interior_candidate_scope),
        "terminal_safe_interior_selection_mode": str(
            terminal_safe_interior_selection_mode),
        "terminal_safe_interior_probability_slack": float(
            terminal_safe_interior_probability_slack),
        "terminal_safe_interior_require_provider": bool(
            terminal_safe_interior_require_provider),
        "certification_recheck_top_k": int(
            args.certification_recheck_top_k),
        "certification_recheck_min_replicates": int(
            args.certification_recheck_min_replicates),
        "certification_recheck_soft_margin_scale": float(
            args.certification_recheck_soft_margin_scale),
        "runtime_checkpoint_dir": str(args.runtime_checkpoint_dir),
        "runtime_checkpoint_resume": True,
        "runtime_checkpoint_interval": int(args.runtime_checkpoint_interval),
        "evaluate_interval": max(0, int(args.evaluate_interval)),
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
            "initial_design_proposal_mode": source_design_contract[
                "proposal_mode"],
            "initial_design_structural_prior_profile": (
                source_design_contract["structural_prior_profile"]
            ),
            "initial_design_source_dimension": source_design_contract[
                "source_dimension"],
            "initial_design_target_dimension": source_design_contract[
                "target_dimension"],
        })
    task = {
        "args": config,
        "heldout": str(args.heldout),
        "line": str(args.line),
        "seed": int(args.seed),
    }
    row = run_one(task)
    row["experiment_variant"] = str(args.experiment_variant)
    row["proposal_mode"] = str(config.get(
        "initial_design_proposal_mode",
        "common_sobol" if args.initial_design == "common_sobol" else "auto",
    ))
    row["proposal_structural_prior_profile"] = str(config.get(
        "initial_design_structural_prior_profile", "none"))
    row["proposal_source_dimension"] = int(config.get(
        "initial_design_source_dimension", config["d"]))
    row["proposal_target_dimension"] = int(config.get(
        "initial_design_target_dimension", config["d"]))
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
            "task_posterior_robust_certificate_mode": str(
                args.task_posterior_robust_certificate_mode),
            "certification_head_authority": str(
                args.certification_head_authority),
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
            "task_variance_posterior_mode": str(
                args.task_variance_posterior_mode),
            "task_latent_calibration_mode": str(
                args.task_latent_calibration_mode),
            "exact_kg_sampling_mode": str(args.exact_sampling_mode),
            "decision_backend": str(args.decision_backend),
            "decision_terminal_rule": str(args.decision_terminal_rule),
            "decision_terminal_maximum_violation_probability": float(
                args.decision_terminal_maximum_violation_probability),
            "decision_risk_penalty": float(args.decision_risk_penalty),
            "decision_source_utility_weight": float(
                args.decision_source_utility_weight),
            "decision_recommend_observed_only": bool(
                args.decision_recommend_observed_only),
            "exact_kg_terminal_mode": str(args.exact_terminal_mode),
            "adaptive_replication_voi": bool(
                args.adaptive_replication_voi),
            "replication_candidate_count": replication_candidate_count,
            "replication_max_per_solution": replication_max_per_solution,
            "evaluate_or_replicate_new_action_count": (
                evaluate_or_replicate_new_action_count),
            "evaluate_or_replicate_new_action_policy": (
                evaluate_or_replicate_new_action_policy),
            "evaluate_or_replicate_baseline_new_action_count": (
                evaluate_or_replicate_baseline_new_action_count),
            "policy_improvement_mode": policy_improvement_mode,
            "policy_improvement_score_normalization": (
                policy_improvement_score_normalization),
            "policy_improvement_score_transform": (
                policy_improvement_score_transform),
            "policy_improvement_guard_mode": policy_improvement_guard_mode,
            "policy_improvement_pairwise_prefix_samples": (
                policy_improvement_pairwise_prefix_samples),
            "policy_improvement_pairwise_error_multiplier": (
                policy_improvement_pairwise_error_multiplier),
            "policy_improvement_confirmation_samples": (
                policy_improvement_confirmation_samples),
            "policy_improvement_confirmation_batch_samples": (
                policy_improvement_confirmation_batch_samples),
            "policy_improvement_confirmation_delta": (
                policy_improvement_confirmation_delta),
            "policy_improvement_confirmation_jobs": (
                policy_improvement_confirmation_jobs),
            "policy_improvement_confirmation_lambda_min": (
                policy_improvement_confirmation_lambda_min),
            "policy_improvement_confirmation_lambda_count": (
                policy_improvement_confirmation_lambda_count),
            "policy_improvement_mc_error_bound": (
                policy_improvement_mc_error_bound),
            "policy_improvement_certificate_mc_error_bound": (
                policy_improvement_certificate_mc_error_bound),
            "policy_improvement_rollout_depth": (
                policy_improvement_rollout_depth),
            "policy_improvement_rollout_max_arms": (
                policy_improvement_rollout_max_arms),
            "policy_improvement_rollout_mc_samples": (
                policy_improvement_rollout_mc_samples),
            "policy_improvement_rollout_mc_error_bound": (
                policy_improvement_rollout_mc_error_bound),
            "safe_interior_candidate_count": safe_interior_candidate_count,
            "safe_interior_pool_size": safe_interior_pool_size,
            "safe_interior_margin": safe_interior_margin,
            "posterior_dominance_enabled": bool(
                args.posterior_dominance_enabled),
            "posterior_dominance_delta": float(
                args.posterior_dominance_delta),
            "posterior_dominance_min_mean_gain": float(
                args.posterior_dominance_min_mean_gain),
            "posterior_dominance_initialization": str(
                args.posterior_dominance_initialization),
            "decision_contract_mode": str(args.decision_contract_mode),
            "exact_kg_mc_samples": int(args.exact_mc_samples),
            "exact_kg_jobs": int(args.exact_jobs),
            "exact_kg_parallel_backend": str(args.parallel_backend),
            "exact_kg_clip_negative": bool(args.exact_clip_negative),
            "exact_kg_reuse_nested_prefix": bool(
                args.exact_reuse_nested_prefix),
            "exact_kg_skip_redundant_primary_update": bool(
                args.exact_skip_redundant_primary_update),
            "exact_kg_chunk_schedule": str(
                args.exact_chunk_schedule),
            "exact_kg_max_chunks_per_candidate": int(
                args.exact_max_chunks_per_candidate),
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
            "meta_observable_mean_input_mode": str(
                args.observable_mean_input_mode),
            "meta_observable_mean_descriptor_mode": str(
                args.observable_mean_descriptor_mode),
            "meta_observable_mean_feature_mode": str(
                args.observable_mean_feature_mode),
            "meta_observable_mean_latent_transform": str(
                args.observable_mean_latent_transform),
            "meta_observable_mean_target_residual_rank": int(
                args.observable_mean_target_residual_rank),
            "meta_observable_mean_target_residual_prior_scale": float(
                args.observable_mean_target_residual_prior_scale),
            "meta_observable_mean_target_residual_pool_size": int(
                args.observable_mean_target_residual_pool_size),
            "meta_observable_mean_target_residual_rcond": float(
                args.observable_mean_target_residual_rcond),
            "meta_observable_mean_role_assignment_posterior": bool(
                args.observable_mean_role_assignment_posterior),
            "meta_observable_mean_role_assignment_prior": str(
                args.observable_mean_role_assignment_prior),
            "meta_observable_mean_role_assignment_prior_temperature_scale": (
                float(args.observable_mean_role_assignment_prior_temperature_scale)
            ),
            "meta_observable_mean_role_assignment_inactive_variance": float(
                args.observable_mean_role_assignment_inactive_variance),
            "meta_observable_variance_input_mode": str(
                args.observable_variance_input_mode),
            "source_constraint_mean_coefficient_prior": bool(
                args.source_constraint_mean_coefficient_prior),
            "source_constraint_mean_hyperlaw_mode": str(
                args.source_constraint_mean_hyperlaw_mode),
            "source_constraint_mean_adaptation_mode": str(
                args.source_constraint_mean_adaptation_mode),
            "source_constraint_mean_residual_rank_posterior": bool(
                args.source_constraint_mean_residual_rank_posterior),
            "source_constraint_mean_residual_rank_prior": str(
                args.source_constraint_mean_residual_rank_prior),
            "source_constraint_mean_residual_rank_inactive_variance": float(
                args.source_constraint_mean_residual_rank_inactive_variance),
            "source_constraint_mean_deviation_mode": str(
                args.source_constraint_mean_deviation_mode),
            "source_constraint_mean_misspecification_mode": str(
                args.source_constraint_mean_misspecification_mode),
            "source_constraint_mean_misspecification_prior_df": float(
                args.source_constraint_mean_misspecification_prior_df),
            "source_constraint_mean_misspecification_ridge": float(
                args.source_constraint_mean_misspecification_ridge),
            "source_constraint_mean_misspecification_max_scale": float(
                args.source_constraint_mean_misspecification_max_scale),
            "source_constraint_mean_misspecification_delta": float(
                args.source_constraint_mean_misspecification_delta),
            "source_constraint_mean_confidence_mode": str(
                args.source_constraint_mean_confidence_mode),
            "source_constraint_mean_confidence_delta": float(
                args.source_constraint_mean_confidence_delta),
            "source_constraint_mean_contrast_scale": float(
                args.source_constraint_mean_contrast_scale),
            "source_constraint_mean_role_epistemic_mode": str(
                args.source_constraint_mean_role_epistemic_mode),
            "source_constraint_mean_null_geometry": str(
                args.source_constraint_mean_null_geometry),
            "source_constraint_mean_null_geometry_ridge": float(
                args.source_constraint_mean_null_geometry_ridge),
            "boundary_coordinate_candidate_count": int(
                args.boundary_coordinate_candidate_count),
            "boundary_coordinate_pool_size": int(
                args.boundary_coordinate_pool_size),
            "boundary_coordinate_safe_fraction": float(
                args.boundary_coordinate_safe_fraction),
            "boundary_coordinate_boundary_fraction": float(
                args.boundary_coordinate_boundary_fraction),
            "boundary_coordinate_coverage_fraction": float(
                args.boundary_coordinate_coverage_fraction),
            "truth_pool_diagnostics": bool(
                config["truth_pool_diagnostics"]),
            "truth_pool_max_candidates": int(
                config["truth_pool_max_candidates"]),
            "hvd_source_task_weight_mode": str(
                args.hvd_source_task_weight_mode),
            "hvd_cumulative_target_evidence_mode": str(
                args.hvd_cumulative_target_evidence_mode),
            "hvd_singleton_evidence_mode": str(
                args.hvd_singleton_evidence_mode),
            "meta_source_observation_mode": str(
                args.source_observation_mode),
            "meta_source_observation_replicates": int(
                args.source_observation_replicates),
            "meta_source_design_mode": str(args.source_design_mode),
            "meta_source_universal_fraction": float(
                args.source_universal_fraction),
            "target_shared_shock_scale": float(
                args.target_shared_shock_scale),
            "variance_audit_size": int(args.variance_audit_size),
            "initial_design": str(args.initial_design),
            "initial_design_file": str(args.initial_design_file),
            "initial_design_archive_match_mode": str(
                args.initial_design_archive_match_mode),
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
            "terminal_verification_budget": int(
                terminal_verification_budget),
            "terminal_verification_delta": float(
                terminal_verification_delta),
            "terminal_verification_mean_delta_fraction": float(
                terminal_verification_mean_delta_fraction),
            "terminal_verification_method": str(
                terminal_verification_method),
            "terminal_verification_policy": str(
                terminal_verification_policy),
            "terminal_verification_shortlist_size": int(
                terminal_verification_shortlist_size),
            "terminal_verification_fallback_budget": int(
                terminal_verification_fallback_budget),
            "terminal_verification_shortlist_mode": str(
                terminal_verification_shortlist_mode),
            "terminal_objective_challenger_max_violation_probability": float(
                terminal_objective_challenger_max_violation_probability),
            "terminal_objective_incumbent_guard": bool(
                terminal_objective_incumbent_guard),
            "terminal_objective_comparison_budget": int(
                terminal_objective_comparison_budget),
            "terminal_objective_comparison_delta": float(
                terminal_objective_comparison_delta),
            "terminal_safe_interior_candidate_scope": str(
                terminal_safe_interior_candidate_scope),
            "terminal_safe_interior_selection_mode": str(
                terminal_safe_interior_selection_mode),
            "terminal_safe_interior_probability_slack": float(
                terminal_safe_interior_probability_slack),
            "terminal_safe_interior_require_provider": bool(
                terminal_safe_interior_require_provider),
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
