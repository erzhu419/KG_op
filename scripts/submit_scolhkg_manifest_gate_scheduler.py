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


DEFAULT_SCHEDULER = Path.home() / ".claude/skills/scheduler/scheduler.py"
DEFAULT_DEPLOY = Path.home() / "mine_code/KG_op_scheduler_deploy"
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
        "--out", str(result_file),
        "--runtime-checkpoint-dir", str(checkpoint_dir),
        "--ordered-max-frequency", str(int(args.ordered_max_frequency)),
        "--ordered-active-dim", str(int(args.ordered_active_dim)),
        "--ordered-frequency-penalty",
        str(float(args.ordered_frequency_penalty)),
        "--ordered-basis-mode", str(args.ordered_basis_mode),
        "--exact-sampling-mode", str(args.exact_sampling_mode),
        "--exact-mc-samples", str(int(args.exact_mc_samples)),
        "--exact-jobs", str(int(args.exact_jobs)),
        "--parallel-backend", str(args.parallel_backend),
        "--finalist-replication-budget",
        str(int(getattr(args, "finalist_replication_budget", 0))),
        "--finalist-replication-count",
        str(int(getattr(args, "finalist_replication_count", 2))),
        "--finalist-replication-min-replicates",
        str(int(getattr(args, "finalist_replication_min_replicates", 2))),
        "--finalist-replication-delta",
        str(float(getattr(args, "finalist_replication_delta", 0.05))),
        "--finalist-replication-variance-prior-df",
        str(float(getattr(
            args, "finalist_replication_variance_prior_df", 2.0))),
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
    return shlex.join(command)


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
    parser.add_argument("--ordered-active-dim", type=int, default=2)
    parser.add_argument("--ordered-frequency-penalty", type=float, default=0.10)
    parser.add_argument(
        "--ordered-basis-mode",
        choices=["full_quadratic", "diagonal_quadratic"],
        default="diagonal_quadratic",
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
    parser.add_argument("--exact-sampling-mode", default="iid")
    parser.add_argument("--exact-mc-samples", type=int, default=2)
    parser.add_argument("--exact-jobs", type=int, default=32)
    parser.add_argument("--parallel-backend", default="process_fork")
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
    parser.add_argument("--allow-duplicate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()

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
