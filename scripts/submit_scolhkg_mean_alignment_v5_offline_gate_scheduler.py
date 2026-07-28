#!/usr/bin/env python3
"""Submit the V5 channel-role and mean-misspecification offline gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SYNC = ROOT / "scripts/sync_scolhkg_scheduler_deploy.sh"
DEFAULT_SCHEDULER = Path.home() / "mine_code/scheduleurm/skill/scheduler.py"
DEFAULT_DEPLOY = Path.home() / "mine_code/KG_op_scheduler_deploy"
REMOTE_ROOT = Path("/home/zhengliang01/scheduleurm_work/KG_op_scheduler_deploy")
REMOTE_PYTHON = Path(
    "/home/zhengliang01/scheduleurm_work/conda_envs/scomp-py310/bin/python")
CPU_NODES = tuple(f"node{index:03d}" for index in range(1, 7))
DEFAULT_SOURCE_RUN_ID = (
    "scolh_lowfreq_support_dimholdout_d1000_s5_20260716/"
    "proposals/risk_objective_atlas/low_frequency_only"
)
VARIANTS = {
    "v4_latent_control": {
        "descriptor": "ordered",
        "misspecification": "none",
    },
    "role_match_only": {
        "descriptor": "role_aligned",
        "misspecification": "none",
    },
    "misspec_scalar": {
        "descriptor": "ordered",
        "misspecification": "predictive_scale",
    },
    "misspec_directional": {
        "descriptor": "ordered",
        "misspecification": "predictive_scale_directional",
    },
    "v5_role_scalar": {
        "descriptor": "role_aligned",
        "misspecification": "predictive_scale",
    },
    "v5_role_directional": {
        "descriptor": "role_aligned",
        "misspecification": "predictive_scale_directional",
    },
}
SCENARIOS = (
    ("FactorShockStatePolicyRZDT1", 0.0),
    ("FactorShockStatePolicyRZDT1", 4.0),
    ("InventorySupplyChain", 1.0),
    ("QueueResourceControl", 1.0),
)


def _parse_csv(value):
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def _flags(profile, args):
    return [
        "--ordered-max-frequency", "8",
        "--ordered-active-dim", "2",
        "--ordered-frequency-penalty", "0.1",
        "--ordered-basis-mode", "diagonal_quadratic",
        "--ordered-orthogonal-coordinates",
        "--ordered-cumulative-exposure",
        "--source-observation-mode", "replicated",
        "--source-observation-replicates", "3",
        "--source-design-mode", "universal_mixture",
        "--source-universal-fraction", "1.0",
        "--source-consensus-template-count", "12",
        "--source-records-per-domain",
        str(profile.get(
            "source_records_per_domain",
            getattr(args, "source_records_per_domain", 96),
        )),
        "--initial-design-archive-match-mode",
        str(profile.get(
            "initial_design_archive_match_mode", "exact")),
        "--meta-source-augments",
        str(profile.get("source_augments", 1)),
        "--meta-source-budget-mode",
        str(profile.get("source_budget_mode", "per_episode")),
        "--meta-source-geometry-shift-scale",
        str(profile.get("source_geometry_shift_scale", 0.0)),
        "--meta-source-geometry-log-radius-jitter",
        str(profile.get("source_geometry_log_radius_jitter", 0.0)),
        "--meta-source-sigma-jitter",
        str(profile.get("source_sigma_jitter", 0.20)),
        "--meta-source-alpha-jitter",
        str(profile.get("source_alpha_jitter", 0.25)),
        "--meta-source-weight-jitter",
        str(profile.get("source_weight_jitter", 0.05)),
        "--observable-mean-coordinate",
        "--observable-mean-mode", "boundary_aligned",
        "--observable-mean-input-mode", "observable_state_exposure",
        "--observable-mean-descriptor-mode", profile["descriptor"],
        "--observable-mean-feature-mode", "linear",
        "--observable-mean-latent-transform",
        str(profile.get("latent_transform", "identity")),
        "--observable-mean-target-residual-rank",
        str(profile.get("target_residual_rank", 0)),
        "--observable-mean-target-residual-prior-scale",
        str(profile.get("target_residual_prior_scale", 1.0)),
        "--observable-mean-target-residual-pool-size",
        str(profile.get("target_residual_pool_size", 128)),
        "--observable-mean-target-residual-rcond",
        str(profile.get("target_residual_rcond", 1e-8)),
        (
            "--observable-mean-role-assignment-posterior"
            if profile.get("role_assignment_posterior", False)
            else "--no-observable-mean-role-assignment-posterior"
        ),
        "--observable-mean-role-assignment-prior",
        str(profile.get("role_assignment_prior", "uniform")),
        "--observable-mean-role-assignment-prior-temperature-scale",
        str(profile.get("role_assignment_prior_temperature_scale", 1.0)),
        "--observable-mean-role-assignment-inactive-variance",
        str(profile.get("role_assignment_inactive_variance", 1e-12)),
        "--observable-mean-latent-dim", str(args.rank),
        "--observable-mean-training-target", "chance_margin",
        "--observable-variance-input-mode", "observable_state_exposure",
        "--source-constraint-mean-coefficient-prior",
        "--source-constraint-mean-hyperlaw-mode",
        str(profile.get("hyperlaw_mode", "single_gaussian_draw")),
        "--source-constraint-mean-adaptation-mode",
        str(profile.get("adaptation", "sequential_evidence_mixture")),
        "--source-constraint-mean-deviation-mode", "latent_shared",
        "--source-constraint-mean-misspecification-mode",
        profile["misspecification"],
        "--source-constraint-mean-misspecification-prior-df",
        str(profile.get(
            "misspecification_prior_df",
            args.misspecification_prior_df,
        )),
        "--source-constraint-mean-misspecification-ridge",
        str(profile.get(
            "misspecification_ridge",
            args.misspecification_ridge,
        )),
        "--source-constraint-mean-misspecification-max-scale",
        str(profile.get(
            "misspecification_max_scale",
            args.misspecification_max_scale,
        )),
        "--source-constraint-mean-misspecification-delta",
        str(profile.get(
            "misspecification_delta",
            getattr(args, "misspecification_delta", 0.05),
        )),
        "--source-constraint-mean-confidence-mode",
        str(profile.get("confidence_mode", "model")),
        "--source-constraint-mean-confidence-delta",
        str(profile.get(
            "confidence_delta",
            getattr(args, "confidence_delta", 0.05),
        )),
        "--source-constraint-mean-contrast-scale",
        str(getattr(args, "contrast_scale", 1.0)),
        "--source-constraint-mean-null-weight",
        str(profile.get("null_weight", 0.5)),
        "--source-constraint-mean-role-epistemic-mode",
        str(profile.get("role_epistemic", "none")),
        "--source-constraint-mean-null-geometry",
        str(profile.get("null_geometry", "isotropic")),
        "--source-constraint-mean-null-geometry-ridge",
        str(getattr(args, "null_geometry_ridge", 1e-3)),
        "--source-constraint-mean-structure-score-mode",
        str(profile.get("structure_score", "marginal_likelihood")),
        "--source-constraint-mean-evidence-temperature",
        str(profile.get("evidence_temperature", 1.0)),
        (
            "--source-constraint-mean-residual-rank-posterior"
            if profile.get("residual_rank_posterior", False)
            else "--no-source-constraint-mean-residual-rank-posterior"
        ),
        "--source-constraint-mean-residual-rank-prior",
        str(profile.get("residual_rank_prior", "0.70,0.20,0.10")),
        "--source-constraint-mean-residual-rank-inactive-variance",
        str(profile.get("residual_rank_inactive_variance", 1e-12)),
        (
            "--source-discrepancy-update"
            if profile.get("source_discrepancy_update", True)
            else "--no-source-discrepancy-update"
        ),
        "--task-posterior-safe-generalized",
        "--task-posterior-safe-boundary-weight", "1.0",
        "--task-posterior-safe-pairwise-weight", "1.0",
        "--task-posterior-safe-pairwise-history", "16",
        "--task-posterior-safe-pairwise-floor", "1e-06",
        "--task-posterior-mandatory-universal-count", "10",
        "--task-posterior-robust-certificate-mode", "joint_tangent",
        "--certification-head-authority",
        str(profile.get(
            "certification_head_authority",
            getattr(args, "certification_head_authority", "task_joint"),
        )),
        "--task-latent-inference-mode", "shadow",
        "--task-latent-calibration-mode", "source_profiles",
        "--task-variance-posterior-mode",
        str(profile.get("variance_task_posterior", "shared")),
        "--hvd-source-task-weight-mode",
        str(profile.get("hvd_task_weight", "constraint_mean")),
        "--hvd-cumulative-target-evidence-mode",
        str(profile.get("hvd_target_evidence", "prequential_upper")),
        "--hvd-singleton-evidence-mode",
        str(profile.get("hvd_singleton_evidence", "in_sample_residual")),
        "--decision-backend",
        str(profile.get("decision_backend", "sobol_new")),
        "--decision-risk-penalty",
        str(profile.get("decision_risk_penalty", 5.0)),
        "--decision-aleatoric-mode",
        str(profile.get(
            "decision_aleatoric_mode", "certification_upper")),
        "--decision-violation-loss-mode",
        str(profile.get(
            "decision_violation_loss_mode", "positive_part")),
        "--decision-ambiguity-mode",
        str(profile.get("decision_ambiguity_mode", "kl_robust")),
        "--exact-terminal-mode",
        str(profile.get("exact_terminal_mode", "bayes_risk")),
        *(
            ["--exact-mc-samples", str(profile["exact_mc_samples"])]
            if "exact_mc_samples" in profile else []
        ),
        *(
            ["--exact-sampling-mode", str(profile["exact_sampling_mode"])]
            if "exact_sampling_mode" in profile else []
        ),
        *(
            ["--exact-jobs", str(profile["exact_jobs"])]
            if "exact_jobs" in profile else []
        ),
        *(
            ["--parallel-backend", str(profile["parallel_backend"])]
            if "parallel_backend" in profile else []
        ),
        (
            "--exact-clip-negative"
            if profile.get("exact_clip_negative", True)
            else "--no-exact-clip-negative"
        ),
        *(
            [
                "--terminal-frontier-candidate-count",
                str(profile["terminal_frontier_candidate_count"]),
            ]
            if "terminal_frontier_candidate_count" in profile
            else []
        ),
        *(
            [
                "--terminal-verification-budget",
                str(profile["terminal_verification_budget"]),
                "--terminal-verification-delta",
                str(profile.get("terminal_verification_delta", 0.05)),
                "--terminal-verification-mean-delta-fraction",
                str(profile.get(
                    "terminal_verification_mean_delta_fraction", 0.5)),
                "--terminal-verification-method",
                str(profile.get(
                    "terminal_verification_method",
                    "component_bonferroni")),
                "--terminal-verification-policy",
                str(profile.get(
                    "terminal_verification_policy", "fixed_policy")),
                "--terminal-verification-shortlist-size",
                str(profile.get(
                    "terminal_verification_shortlist_size", 1)),
                "--terminal-verification-fallback-budget",
                str(profile.get(
                    "terminal_verification_fallback_budget", 0)),
                "--terminal-verification-shortlist-mode",
                str(profile.get(
                    "terminal_verification_shortlist_mode",
                    "posterior_ranked")),
                "--terminal-objective-challenger-max-violation-probability",
                str(profile.get(
                    "terminal_objective_challenger_"
                    "max_violation_probability",
                    0.5,
                )),
                "--terminal-safe-interior-candidate-scope",
                str(profile.get(
                    "terminal_safe_interior_candidate_scope", "initial")),
                "--terminal-safe-interior-selection-mode",
                str(profile.get(
                    "terminal_safe_interior_selection_mode", "diverse")),
                "--terminal-safe-interior-probability-slack",
                str(profile.get(
                    "terminal_safe_interior_probability_slack", 0.05)),
                *(
                    ["--terminal-safe-interior-require-provider"]
                    if profile.get(
                        "terminal_safe_interior_require_provider", False)
                    else []
                ),
            ]
            if "terminal_verification_budget" in profile
            else []
        ),
        "--replication-candidate-count",
        str(profile.get("replication_candidate_count", 0)),
        "--replication-max-per-solution",
        str(profile.get("replication_max_per_solution", 5)),
        "--evaluate-or-replicate-new-action-count",
        str(profile.get("evaluate_or_replicate_new_action_count", 1)),
        "--evaluate-or-replicate-new-action-policy",
        str(profile.get(
            "evaluate_or_replicate_new_action_policy", "canonical_sobol")),
        "--evaluate-or-replicate-baseline-new-action-count",
        str(profile.get(
            "evaluate_or_replicate_baseline_new_action_count", 0)),
        "--policy-improvement-mode",
        str(profile.get("policy_improvement_mode", "off")),
        "--policy-improvement-score-normalization",
        str(profile.get(
            "policy_improvement_score_normalization", "none")),
        "--policy-improvement-score-transform",
        str(profile.get(
            "policy_improvement_score_transform", "identity")),
        "--policy-improvement-guard-mode",
        str(profile.get(
            "policy_improvement_guard_mode", "uniform_score")),
        "--policy-improvement-pairwise-prefix-samples",
        str(profile.get(
            "policy_improvement_pairwise_prefix_samples", 32)),
        "--policy-improvement-pairwise-error-multiplier",
        str(profile.get(
            "policy_improvement_pairwise_error_multiplier", 1.25)),
        "--policy-improvement-confirmation-samples",
        str(profile.get("policy_improvement_confirmation_samples", 4096)),
        "--policy-improvement-confirmation-batch-samples",
        str(profile.get(
            "policy_improvement_confirmation_batch_samples", 512)),
        "--policy-improvement-confirmation-delta",
        str(profile.get("policy_improvement_confirmation_delta", 0.05)),
        "--policy-improvement-confirmation-jobs",
        str(profile.get("policy_improvement_confirmation_jobs", 0)),
        "--policy-improvement-confirmation-lambda-min",
        str(profile.get(
            "policy_improvement_confirmation_lambda_min", 0.001)),
        "--policy-improvement-confirmation-lambda-count",
        str(profile.get(
            "policy_improvement_confirmation_lambda_count", 24)),
        "--policy-improvement-mc-error-bound",
        str(profile.get("policy_improvement_mc_error_bound", 0.0)),
        "--policy-improvement-certificate-mc-error-bound",
        str(profile.get(
            "policy_improvement_certificate_mc_error_bound", 0.0)),
        "--policy-improvement-rollout-depth",
        str(profile.get("policy_improvement_rollout_depth", 1)),
        "--policy-improvement-rollout-max-arms",
        str(profile.get("policy_improvement_rollout_max_arms", 4)),
        "--policy-improvement-rollout-mc-samples",
        str(profile.get("policy_improvement_rollout_mc_samples", 2)),
        "--policy-improvement-rollout-mc-error-bound",
        str(profile.get(
            "policy_improvement_rollout_mc_error_bound", 0.0)),
        (
            "--adaptive-replication-voi"
            if profile.get("adaptive_replication_voi", False)
            else "--no-adaptive-replication-voi"
        ),
        "--finalist-replication-budget", "0",
        "--finalist-empirical-override", "off",
        "--certification-recheck-top-k", "0",
        (
            "--posterior-dominance-enabled"
            if profile.get("posterior_dominance_enabled", False)
            else "--no-posterior-dominance-enabled"
        ),
        "--posterior-dominance-delta",
        str(profile.get(
            "posterior_dominance_delta",
            getattr(args, "posterior_dominance_delta", 0.05),
        )),
        "--posterior-dominance-min-mean-gain",
        str(profile.get(
            "posterior_dominance_min_mean_gain",
            getattr(args, "posterior_dominance_min_mean_gain", 0.0),
        )),
        "--posterior-dominance-initialization",
        str(profile.get(
            "posterior_dominance_initialization",
            getattr(args, "posterior_dominance_initialization", "risk"),
        )),
        "--decision-contract-mode",
        str(profile.get("decision_contract_mode", "legacy")),
        "--implementation-contract-id",
        str(profile.get(
            "implementation_contract_id",
            getattr(args, "implementation_contract_id", "unversioned"),
        )),
        "--theory-contract-id",
        str(profile.get(
            "theory_contract_id",
            getattr(args, "theory_contract_id", "unversioned"),
        )),
        "--finalist-terminal-value-mode",
        str(profile.get("finalist_terminal_value_mode", "model_default")),
        "--decision-recommend-observed-only",
        "--boundary-coordinate-candidate-count", "0",
        "--boundary-coordinate-pool-size", str(args.pool_size),
        "--truth-pool-diagnostics",
        "--truth-pool-max-candidates", "0",
        "--variance-audit-size", str(args.variance_audit_size),
    ]


def build_specs(args):
    nodes = _parse_csv(args.nodes)
    variants = _parse_csv(args.variants)
    variant_profiles = getattr(args, "variant_profiles", VARIANTS)
    if not nodes or any(node not in CPU_NODES for node in nodes):
        raise ValueError("nodes must be a nonempty subset of node001-node006")
    if not variants or any(
        variant not in variant_profiles for variant in variants
    ):
        raise ValueError("unknown V5 gate variant")
    sequential = bool(getattr(args, "sequential", False))
    if int(args.N) < int(args.n0):
        raise ValueError("V5 gate requires N >= n0")
    if not sequential and int(args.N) != int(args.n0):
        raise ValueError("offline V5 gate requires N == n0")
    if sequential and int(args.N) <= int(args.n0):
        raise ValueError("sequential V5 gate requires N > n0")
    stage_family = str(getattr(args, "stage_family", "mean_v5"))
    stage = (
        f"{stage_family}_sequential"
        if sequential
        else f"{stage_family}_offline"
    )
    gate_label = str(getattr(args, "gate_label", "Mean V5"))
    scenarios = tuple(getattr(args, "scenarios", SCENARIOS))

    local_project = Path(args.deploy) / "SC-OLH-KG"
    remote_project = REMOTE_ROOT / "SC-OLH-KG"
    try:
        relative_manifest = Path(args.manifest).relative_to(local_project)
        remote_manifest = remote_project / relative_manifest
    except ValueError:
        remote_manifest = Path(args.manifest)

    specs = []
    for variant in variants:
        profile = variant_profiles[variant]
        for heldout, shock_scale in scenarios:
            local_design = (
                local_project / "archives" / args.source_run_id / heldout
                / "source_initial_designs.json"
            )
            remote_design = (
                remote_project / "archives" / args.source_run_id / heldout
                / "source_initial_designs.json"
            )
            shock_label = f"{shock_scale:g}".replace(".", "p")
            for offset in range(int(args.n_seeds)):
                seed = int(args.seed_start) + offset
                cell = f"{variant}/shock{shock_label}"
                remote_result_dir = (
                    remote_project / "profiles" / args.run_id / cell
                    / heldout / f"seed{seed}"
                )
                local_result_dir = (
                    local_project / "profiles" / args.run_id / cell
                    / heldout / f"seed{seed}"
                )
                command = [
                    "env", "LC_ALL=C", "LANG=C", "SCOLHKG_OFFLINE=1",
                    "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
                    "OPENBLAS_NUM_THREADS=1", "PYTHONUNBUFFERED=1",
                    "PYTHONDONTWRITEBYTECODE=1", str(args.python),
                    "performance/run_lodo_manifest_shard.py",
                    "--manifest", str(remote_manifest),
                    "--heldout", heldout,
                    "--line", "lodo",
                    "--seed", str(seed),
                    "--experiment-variant", f"{stage}/{cell}",
                    "--out", str(remote_result_dir / "result.json"),
                    "--runtime-checkpoint-dir", "",
                    "--runtime-checkpoint-interval", "0",
                    "--evaluate-interval", "0",
                    "--d", str(args.d),
                    "--meta-source-d", str(args.source_d),
                    "--N", str(args.N),
                    "--n0", str(args.n0),
                    "--initial-design", "source_informed",
                    "--initial-design-file", str(remote_design),
                    "--structural-prior-profile",
                    str(profile.get("structural_prior_profile", "none")),
                    "--hvd-profile",
                    str(profile.get("hvd_profile", "factor_hierarchical")),
                    "--target-shared-shock-scale", str(shock_scale),
                    *_flags(profile, args),
                ]
                command_text = f"{shlex.join(command)} && echo DONE"
                wait_for_files = [str(local_design)]
                if bool(getattr(args, "remote_design_only", False)):
                    command_text = (
                        f"test -f {shlex.quote(str(remote_design))} && "
                        f"{command_text}"
                    )
                    wait_for_files = []
                specs.append({
                    "description": (
                        f"{gate_label} "
                        f"{'sequential' if sequential else 'offline'} "
                        f"{variant} {heldout} "
                        f"shock={shock_scale:g} seed={seed}"
                    ),
                    "cmd": command_text,
                    "cwd": str(local_project),
                    "signature": (
                        f"KG_op/{stage}/{args.run_id}/{cell}/"
                        f"{heldout}/seed{seed}"
                    ),
                    "project": "KG-SYNTH",
                    "vram": 0,
                    "cpu": int(args.cpu),
                    "ram_mb": int(args.ram_mb),
                    "allowed_nodes": list(nodes),
                    "wait_for_files": wait_for_files,
                    "result_dir": str(remote_result_dir),
                    "local_result_dir": str(local_result_dir),
                    "stage_excludes": ["checkpoints", "profiles", "results"],
                    "allow_duplicate": True,
                })
    return specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--deploy", type=Path, default=DEFAULT_DEPLOY)
    parser.add_argument("--python", type=Path, default=REMOTE_PYTHON)
    parser.add_argument("--manifest", type=Path, default=(
        DEFAULT_DEPLOY
        / "SC-OLH-KG/performance/manifests/v18b_exactkg_mcdiag.json"))
    parser.add_argument("--source-run-id", default=DEFAULT_SOURCE_RUN_ID)
    parser.add_argument("--remote-design-only", action="store_true")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--run-id", default=(
        "scolh_mean_alignment_v5_offline_s5_"
        f"{time.strftime('%Y%m%d_%H%M%S')}"))
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--nodes", default=",".join(CPU_NODES))
    parser.add_argument("--source-d", type=int, default=50)
    parser.add_argument("--d", type=int, default=1000)
    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--pool-size", type=int, default=512)
    parser.add_argument("--variance-audit-size", type=int, default=512)
    parser.add_argument("--source-records-per-domain", type=int, default=96)
    parser.add_argument("--misspecification-prior-df", type=float, default=4.0)
    parser.add_argument("--misspecification-ridge", type=float, default=1.0)
    parser.add_argument("--misspecification-max-scale", type=float, default=100.0)
    parser.add_argument("--contrast-scale", type=float, default=1.0)
    parser.add_argument(
        "--certification-head-authority",
        choices=(
            "task_joint",
            "split_gpr_task_hvd",
            "split_gpr_cumulative_hvd",
        ),
        default="task_joint",
    )
    parser.add_argument("--posterior-dominance-delta", type=float, default=0.05)
    parser.add_argument(
        "--posterior-dominance-min-mean-gain", type=float, default=0.0)
    parser.add_argument(
        "--posterior-dominance-initialization",
        choices=("risk", "certificate_lexicographic", "certified_only"),
        default="risk",
    )
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument("--ram-mb", type=int, default=4096)
    parser.add_argument("--no-sync", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sequential", action="store_true")
    args = parser.parse_args()
    specs = build_specs(args)
    if args.dry_run:
        print(json.dumps(specs, indent=2))
        return
    if not args.no_sync:
        subprocess.run([str(SYNC)], check=True, cwd=ROOT)
    payload = "\n".join(json.dumps(spec) for spec in specs) + "\n"
    subprocess.run(
        [
            sys.executable,
            str(args.scheduler),
            "submit-jsonl", "--stdin", "--trusted", "--json",
            "--intent-label", str(args.run_id),
        ],
        input=payload,
        text=True,
        check=True,
    )


if __name__ == "__main__":
    main()
