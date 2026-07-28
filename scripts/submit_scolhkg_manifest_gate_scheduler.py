#!/usr/bin/env python3
"""Submit one-seed LODO manifest gates as one atomic scheduler group."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time


DEFAULT_SCHEDULER = (
    Path.home() / "mine_code/scheduleurm/skill/scheduler.py"
)
DEFAULT_DEPLOY = Path.home() / "mine_code/KG_op_scheduler_deploy"
DEFAULT_SYNC_SCRIPT = Path(__file__).with_name(
    "sync_scolhkg_scheduler_deploy.sh"
)
DEFAULT_PYTHON = (
    "/home/zhengliang01/scheduleurm_work/conda_envs/"
    "scomp-py310/bin/python"
)
CPU_NODES = tuple(f"node{index:03d}" for index in range(1, 7))


def parse_csv(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def command_for(args, heldout, seed, result_file, checkpoint_dir):
    command = [
        "env",
        "LC_ALL=C",
        "LANG=C",
        "OMP_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "SCOLHKG_OFFLINE=1",
        "PYTHONUNBUFFERED=1",
        "PYTHONDONTWRITEBYTECODE=1",
        str(args.python),
        "performance/run_lodo_manifest_shard.py",
        "--manifest", str(args.manifest),
        "--heldout", str(heldout),
        "--line", str(args.line),
        "--seed", str(int(seed)),
        "--experiment-variant", str(getattr(args, "experiment_variant", "")),
        "--out", str(result_file),
        "--runtime-checkpoint-dir", str(checkpoint_dir),
        "--ordered-max-frequency", str(int(args.ordered_max_frequency)),
        "--spectral-orthogonalization",
        str(getattr(args, "spectral_orthogonalization", "symmetric")),
        "--ordered-active-dim", str(int(args.ordered_active_dim)),
        "--ordered-frequency-penalty",
        str(float(args.ordered_frequency_penalty)),
        "--ordered-basis-mode", str(args.ordered_basis_mode),
        (
            "--ordered-orthogonal-coordinates"
            if getattr(args, "ordered_orthogonal_coordinates", True)
            else "--no-ordered-orthogonal-coordinates"
        ),
        "--exact-sampling-mode", str(args.exact_sampling_mode),
        "--exact-terminal-mode",
        str(getattr(args, "exact_terminal_mode", "hard_certified")),
        "--exact-mc-samples", str(int(args.exact_mc_samples)),
        "--exact-jobs", str(int(args.exact_jobs)),
        "--parallel-backend", str(args.parallel_backend),
        "--tcb-v2-mode",
        str(getattr(args, "tcb_v2_mode", "off")),
        "--tcb-v2-frontier-count",
        str(int(getattr(args, "tcb_v2_frontier_count", 1))),
        "--tcb-v2-descriptor-mode",
        str(getattr(args, "tcb_v2_descriptor_mode", "learned_risk")),
        "--tcb-v2-coordinate",
        str(getattr(args, "tcb_v2_coordinate", "boundary_latent")),
        "--tcb-v2-geometry",
        str(getattr(args, "tcb_v2_geometry", "low_rank_psd")),
        "--tcb-v2-rank",
        str(int(getattr(args, "tcb_v2_rank", 2))),
        "--tcb-v2-ridge",
        str(float(getattr(args, "tcb_v2_ridge", 1e-3))),
        "--tcb-v2-domain-penalty",
        str(float(getattr(args, "tcb_v2_domain_penalty", 0.5))),
        "--tcb-v2-boundary-temperature",
        str(float(getattr(args, "tcb_v2_boundary_temperature", 1.0))),
        "--tcb-v2-adaptation-ridge",
        str(float(getattr(args, "tcb_v2_adaptation_ridge", 5.0))),
        "--tcb-v2-upper-alpha",
        str(float(getattr(args, "tcb_v2_upper_alpha", 0.01))),
        "--tcb-v2-calibration-prior-df",
        str(float(getattr(args, "tcb_v2_calibration_prior_df", 2.0))),
        "--tcb-v2-hierarchy-iterations",
        str(int(getattr(args, "tcb_v2_hierarchy_iterations", 5))),
        "--tcb-v2-effect-ridge",
        str(float(getattr(args, "tcb_v2_effect_ridge", 1.0))),
        "--tcb-v2-rotation-mode",
        str(getattr(args, "tcb_v2_rotation_mode", "none")),
        "--tcb-v2-rotation-ridge",
        str(float(getattr(args, "tcb_v2_rotation_ridge", 5.0))),
        "--tcb-v2-target-residual-rank",
        str(int(getattr(args, "tcb_v2_target_residual_rank", 0))),
        "--tcb-v2-residual-ridge",
        str(float(getattr(args, "tcb_v2_residual_ridge", 5.0))),
        "--source-observation-mode",
        str(getattr(args, "source_observation_mode", "analytic")),
        "--source-observation-replicates",
        str(int(getattr(args, "source_observation_replicates", 1))),
        "--source-design-mode",
        str(getattr(args, "source_design_mode", "random")),
        "--source-universal-fraction",
        str(float(getattr(args, "source_universal_fraction", 0.75))),
        "--source-consensus-template-count",
        str(int(getattr(args, "source_consensus_template_count", 0))),
        "--initial-design",
        str(getattr(args, "initial_design", "auto")),
        "--observable-mean-mode",
        str(getattr(args, "observable_mean_mode", "latent")),
        "--observable-mean-latent-dim",
        str(int(getattr(args, "observable_mean_latent_dim", 2))),
        "--observable-mean-training-target",
        str(getattr(
            args, "observable_mean_training_target", "constraint_mean")),
        "--finalist-terminal-value-mode",
        str(getattr(
            args, "finalist_terminal_value_mode", "model_default")),
        "--finalist-replication-budget",
        str(int(getattr(args, "finalist_replication_budget", 0))),
        "--finalist-replication-count",
        str(int(getattr(args, "finalist_replication_count", 2))),
        "--finalist-observed-safety-count",
        str(int(getattr(args, "finalist_observed_safety_count", 1))),
        "--finalist-replication-min-replicates",
        str(int(getattr(args, "finalist_replication_min_replicates", 2))),
        "--finalist-replication-delta",
        str(float(getattr(args, "finalist_replication_delta", 0.05))),
        "--finalist-replication-variance-prior-df",
        str(float(getattr(
            args, "finalist_replication_variance_prior_df", 2.0))),
        "--finalist-replication-policy",
        str(getattr(args, "finalist_replication_policy", "legacy")),
        "--finalist-empirical-override",
        str(getattr(args, "finalist_empirical_override", "legacy")),
        "--finalist-frontier-policy",
        str(getattr(args, "finalist_frontier_policy", "legacy")),
        "--finalist-terminal-max-arms",
        str(int(getattr(args, "finalist_terminal_max_arms", 4))),
        "--finalist-terminal-mc-samples",
        str(int(getattr(args, "finalist_terminal_mc_samples", 2))),
        "--decision-contract-mode",
        str(getattr(args, "decision_contract_mode", "legacy")),
        "--implementation-contract-id",
        str(getattr(args, "implementation_contract_id", "unversioned")),
        "--theory-contract-id",
        str(getattr(args, "theory_contract_id", "unversioned")),
        "--task-posterior-safe-boundary-weight",
        str(float(getattr(
            args, "task_posterior_safe_boundary_weight", 1.0))),
        "--task-posterior-safe-pairwise-weight",
        str(float(getattr(
            args, "task_posterior_safe_pairwise_weight", 1.0))),
        "--task-posterior-safe-pairwise-history",
        str(int(getattr(
            args, "task_posterior_safe_pairwise_history", 16))),
        "--task-posterior-safe-pairwise-floor",
        str(float(getattr(
            args, "task_posterior_safe_pairwise_floor", 1e-6))),
        "--task-latent-inference-mode",
        str(getattr(args, "task_latent_inference_mode", "shadow")),
        "--task-latent-calibration-mode",
        str(getattr(
            args, "task_latent_calibration_mode", "source_profiles")),
    ]
    command.append(
        "--finalist-replication-expert-stratified"
        if getattr(args, "finalist_replication_expert_stratified", False)
        else "--no-finalist-replication-expert-stratified"
    )
    command.append(
        "--finalist-replication-adaptive-race"
        if getattr(args, "finalist_replication_adaptive_race", False)
        else "--no-finalist-replication-adaptive-race"
    )
    command.append(
        "--finalist-replication-fixed-universe"
        if getattr(args, "finalist_replication_fixed_universe", False)
        else "--no-finalist-replication-fixed-universe"
    )
    if args.ordered_cumulative_exposure:
        command.append("--ordered-cumulative-exposure")
    if args.ordered_adaptive_sparsity:
        command.append("--ordered-adaptive-sparsity")
    if args.ordered_replace_local_kernel:
        command.append("--ordered-replace-local-kernel")
    if args.ordered_semiparametric_residual:
        command.append("--ordered-semiparametric-residual")
    if args.ordered_latent_structure_selection:
        command.append("--ordered-latent-structure-selection")
    if args.ordered_group_shared_shrinkage:
        command.append("--ordered-group-shared-shrinkage")
    if args.ordered_group_ridge_learning:
        command.append("--ordered-group-ridge-learning")
    command.append(
        "--task-posterior-safe-generalized"
        if getattr(args, "task_posterior_safe_generalized", False)
        else "--no-task-posterior-safe-generalized"
    )
    command.append(
        "--observable-mean-coordinate"
        if getattr(args, "observable_mean_coordinate", False)
        else "--no-observable-mean-coordinate"
    )
    mandatory_universal = getattr(
        args, "task_posterior_mandatory_universal_count", None)
    if mandatory_universal is not None:
        command.extend([
            "--task-posterior-mandatory-universal-count",
            str(int(mandatory_universal)),
        ])
    return f"{shlex.join(command)} && echo DONE"


def build_specs(args):
    heldouts = parse_csv(args.heldouts)
    nodes = parse_csv(args.nodes)
    if not heldouts:
        raise ValueError("at least one held-out domain is required")
    if not nodes or any(node not in CPU_NODES for node in nodes):
        raise ValueError("nodes must be a nonempty subset of node001-node006")
    if int(args.n_seeds) < 1:
        raise ValueError("n_seeds must be positive")

    deploy_project = Path(args.deploy) / "SC-OLH-KG"
    profile_root = deploy_project / "profiles" / str(args.run_id)
    checkpoint_root = deploy_project / "checkpoints" / str(args.run_id)
    specs = []
    task_index = 0
    for heldout in heldouts:
        for offset in range(int(args.n_seeds)):
            seed = int(args.seed_start) + offset
            node = nodes[task_index % len(nodes)]
            task_index += 1
            result_dir = profile_root / heldout / f"seed{seed}"
            checkpoint_dir = checkpoint_root / heldout / f"seed{seed}"
            result_file = result_dir / "result.json"
            command = command_for(
                args, heldout, seed, result_file, checkpoint_dir)
            specs.append({
                "description": (
                    f"SC-OLH-KG {args.run_id} Gate1 {heldout} seed={seed}"
                ),
                "cmd": command,
                "cwd": str(deploy_project),
                "signature": (
                    f"KG_op/scolhkg_manifest/{args.run_id}/"
                    f"{heldout}/seed{seed}"
                ),
                "project": "KG-SYNTH",
                "vram": 0,
                "cpu": int(args.cpu),
                "ram_mb": int(args.ram_mb),
                "require_node": node,
                "allowed_nodes": list(nodes),
                "ckpt_dir": str(checkpoint_dir),
                "ckpt_glob": "checkpoint_latest.pkl",
                "allow_initial_resume_scan_error": True,
                "allow_no_resume": True,
                "result_dir": str(result_dir),
                "local_result_dir": str(result_dir),
                "stage_excludes": ["checkpoints", "profiles", "results"],
                "allow_duplicate": bool(args.allow_duplicate),
            })
    return specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--deploy", type=Path, default=DEFAULT_DEPLOY)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            DEFAULT_DEPLOY
            / "SC-OLH-KG/performance/manifests/v18b_exactkg_mcdiag.json"
        ),
    )
    parser.add_argument(
        "--run-id",
        default=f"manifest_gate_{time.strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument(
        "--heldouts",
        default="FactorShockStatePolicyRZDT1,InventorySupplyChain",
    )
    parser.add_argument("--line", default="lodo_teacher")
    parser.add_argument("--experiment-variant", default="")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=7)
    parser.add_argument("--nodes", default=",".join(CPU_NODES))
    parser.add_argument("--cpu", type=int, default=33)
    parser.add_argument("--ram-mb", type=int, default=8192)
    parser.add_argument(
        "--ordered-cumulative-exposure",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--ordered-max-frequency", type=int, default=8)
    parser.add_argument(
        "--spectral-orthogonalization",
        choices=("symmetric", "ordered_cholesky", "none"),
        default="symmetric",
    )
    parser.add_argument("--ordered-active-dim", type=int, default=2)
    parser.add_argument("--ordered-frequency-penalty", type=float, default=0.10)
    parser.add_argument(
        "--ordered-basis-mode",
        choices=["full_quadratic", "diagonal_quadratic"],
        default="diagonal_quadratic",
    )
    parser.add_argument(
        "--ordered-orthogonal-coordinates",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--ordered-adaptive-sparsity",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--ordered-replace-local-kernel",
        action=argparse.BooleanOptionalAction,
        default=True,
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
    parser.add_argument(
        "--implementation-contract-id", default="unversioned")
    parser.add_argument(
        "--theory-contract-id", default="unversioned")
    parser.add_argument("--exact-mc-samples", type=int, default=2)
    parser.add_argument("--exact-jobs", type=int, default=32)
    parser.add_argument("--parallel-backend", default="process_fork")
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
    parser.add_argument("--source-consensus-template-count", type=int, default=0)
    parser.add_argument(
        "--initial-design",
        choices=("auto", "common_sobol"),
        default="auto",
    )
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
    parser.add_argument("--finalist-replication-budget", type=int, default=0)
    parser.add_argument(
        "--finalist-terminal-value-mode",
        choices=("model_default", "certified_lexicographic"),
        default="model_default",
    )
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
        "--finalist-frontier-policy",
        choices=("legacy", "coverage_reserved", "observed_safety_reserved"),
        default="legacy",
    )
    parser.add_argument(
        "--finalist-empirical-override",
        choices=("legacy", "certified_only", "off"),
        default="legacy",
    )
    parser.add_argument("--finalist-terminal-max-arms", type=int, default=4)
    parser.add_argument("--finalist-terminal-mc-samples", type=int, default=2)
    parser.add_argument("--allow-duplicate", action="store_true")
    parser.add_argument(
        "--sync-remote",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Synchronize SC-OLH-KG to the scheduler deploy before submission.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()

    if args.sync_remote and not args.dry_run:
        subprocess.check_call([str(DEFAULT_SYNC_SCRIPT)])

    specs = build_specs(args)
    if args.dry_run:
        print(json.dumps(specs, indent=2))
        return
    command = [
        sys.executable,
        str(args.scheduler),
        "submit-jsonl",
        "--stdin",
        "--trusted",
        "--json",
        "--intent-label",
        f"scolhkg-manifest-{args.run_id}",
    ]
    output = subprocess.check_output(
        command,
        input=json.dumps(specs),
        text=True,
    )
    print(output, end="" if output.endswith("\n") else "\n")
    submitted = json.loads(output).get("submitted", [])
    task_ids = [item["id"] for item in submitted if item.get("id")]
    if args.dispatch and task_ids:
        subprocess.check_call([
            sys.executable, str(args.scheduler), "dispatch",
        ])
    print(json.dumps({
        "run_id": args.run_id,
        "task_ids": task_ids,
        "n_tasks": len(task_ids),
    }))


if __name__ == "__main__":
    main()
