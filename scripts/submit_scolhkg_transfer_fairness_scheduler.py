#!/usr/bin/env python3
"""Submit immutable-archive transfer CBO comparisons to node001-node006."""

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
SYNC_EXTERNAL = ROOT / "scripts/sync_scolhkg_transfer_repos.sh"
DEFAULT_SCHEDULER = Path.home() / "mine_code/scheduleurm/skill/scheduler.py"
DEFAULT_DEPLOY = Path.home() / "mine_code/KG_op_scheduler_deploy"
DEFAULT_PYTHON = (
    "/home/zhengliang01/scheduleurm_work/conda_envs/scomp-py310/bin/python"
)
REMOTE_ROOT = Path(
    "/home/zhengliang01/scheduleurm_work/KG_op_scheduler_deploy")
EXTERNAL_REPOS = REMOTE_ROOT / "external_repos"
TRANSFER_TORCH_OVERLAY = Path(
    "/home/zhengliang01/scheduleurm_work/python_pkgs/transfer_torch_py310"
)
TRANSFERGPBO_OVERLAY = Path(
    "/home/zhengliang01/scheduleurm_work/python_pkgs/transfergpbo_py310"
)
CPU_NODES = tuple(f"node{i:03d}" for i in range(1, 7))
ALL_METHODS = (
    "safe_fpacoh_cbo",
    "rgpe_cbo",
    "stacked_transfer_gp_cbo",
    "mtgp_cbo",
    "fsbo_cbo",
    "hyperbo_cbo",
    "metabo_cbo",
    "malibo_cbo",
)
CONFIGURED_OFFICIAL = {
    "safe_fpacoh_cbo",
    "rgpe_cbo",
    "stacked_transfer_gp_cbo",
    "mtgp_cbo",
    "fsbo_cbo",
    "hyperbo_cbo",
    "metabo_cbo",
    "malibo_cbo",
}


def parse_csv(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def archive_relative_path(run_id, heldout):
    return Path("archives") / run_id / heldout / f"heldout_{heldout}.json"


def design_relative_path(run_id, heldout):
    return Path("archives") / run_id / heldout / "source_initial_designs.json"


def build_specs(args):
    terminal_defaults = {
        "terminal_verification": True,
        "terminal_verification_primary_budget": 80,
        "terminal_verification_support_budget": 96,
        "terminal_verification_delta": 0.05,
        "terminal_verification_method": "normal_quantile_tolerance",
        "terminal_safe_interior_probability_slack": 0.05,
        "structural_prior_profile": "low_frequency_only",
        "proposal_mode": "risk_objective_atlas",
        "source_design_mode": "universal_mixture",
    }
    for name, value in terminal_defaults.items():
        if not hasattr(args, name):
            setattr(args, name, value)
    nodes = parse_csv(args.nodes)
    if not nodes or any(node not in CPU_NODES for node in nodes):
        raise ValueError("nodes must be a nonempty subset of node001-node006")
    methods = parse_csv(args.methods)
    unknown = sorted(set(methods) - set(ALL_METHODS))
    if unknown:
        raise ValueError(f"unknown transfer methods: {unknown}")
    if args.implementation == "official" and not args.allow_unconfigured:
        unsupported = sorted(set(methods) - CONFIGURED_OFFICIAL)
        if unsupported:
            raise ValueError(
                "official adapters are not configured for "
                + ",".join(unsupported)
            )
    heldouts = parse_csv(args.heldouts)
    deploy_project = Path(args.deploy) / "SC-OLH-KG"
    remote_project = REMOTE_ROOT / "SC-OLH-KG"
    try:
        manifest_relative = Path(args.manifest).relative_to(deploy_project)
        remote_manifest = remote_project / manifest_relative
    except ValueError:
        remote_manifest = Path(args.manifest)
    profile_root = deploy_project / "profiles" / args.run_id
    checkpoint_root = remote_project / "checkpoints" / args.run_id
    specs = []

    for index, heldout in enumerate(heldouts):
        relative = archive_relative_path(args.run_id, heldout)
        local_archive = deploy_project / relative
        remote_archive = remote_project / relative
        archive_dir = remote_archive.parent
        command = [
            "env", "LC_ALL=C", "LANG=C", "SCOLHKG_OFFLINE=1",
            "PYTHONUNBUFFERED=1", "PYTHONDONTWRITEBYTECODE=1",
            f"OMP_NUM_THREADS={int(args.cpu)}",
            f"MKL_NUM_THREADS={int(args.cpu)}",
            str(args.python),
            "performance/materialize_transfer_archives.py",
            "--manifest", str(remote_manifest),
            "--heldouts", heldout,
            "--out-dir", str(archive_dir),
            "--d", str(args.d),
        ]
        specs.append({
            "description": f"materialize transfer archive {heldout}",
            "cmd": f"{shlex.join(command)} && echo DONE",
            # scheduleurm stages from the submit host, so cwd must name the
            # local deploy tree. Absolute paths inside cmd remain remote.
            "cwd": str(deploy_project),
            "signature": f"KG_op/transfer_archive/{args.run_id}/{heldout}",
            "project": "KG-SYNTH",
            "vram": 0,
            "cpu": int(args.cpu),
            "ram_mb": int(args.ram_mb),
            "allowed_nodes": list(nodes),
            "result_dir": str(archive_dir),
            "local_result_dir": str(local_archive.parent),
            "stage_excludes": ["checkpoints", "profiles", "results"],
            "allow_duplicate": True,
        })

        if args.initial_design == "source_informed":
            design_relative = design_relative_path(args.run_id, heldout)
            local_design = deploy_project / design_relative
            remote_design = remote_project / design_relative
            design_command = [
                "env", "LC_ALL=C", "LANG=C", "SCOLHKG_OFFLINE=1",
                "PYTHONUNBUFFERED=1", "PYTHONDONTWRITEBYTECODE=1",
                f"OMP_NUM_THREADS={int(args.cpu)}",
                f"MKL_NUM_THREADS={int(args.cpu)}",
                str(args.python),
                "performance/materialize_source_initial_designs.py",
                "--manifest", str(remote_manifest),
                "--heldout", heldout,
                "--archive", str(remote_archive),
                "--out", str(remote_design),
                "--d", str(args.d),
                "--source-d", str(args.d),
                "--n0", str(args.n0),
                "--seed-start", str(args.seed_start),
                "--n-seeds", str(args.n_seeds),
                "--structural-prior-profile",
                str(args.structural_prior_profile),
                "--proposal-mode", str(args.proposal_mode),
                "--source-design-mode", str(args.source_design_mode),
            ]
            specs.append({
                "description": f"materialize source initial designs {heldout}",
                "cmd": f"{shlex.join(design_command)} && echo DONE",
                "cwd": str(deploy_project),
                "signature": (
                    f"KG_op/transfer_initial_design/{args.run_id}/{heldout}"
                ),
                "project": "KG-SYNTH",
                "vram": 0,
                "cpu": int(args.cpu),
                "ram_mb": int(args.ram_mb),
                "allowed_nodes": list(nodes),
                "wait_for_files": [str(local_archive)],
                "result_dir": str(remote_design.parent),
                "local_result_dir": str(local_design.parent),
                "stage_excludes": ["checkpoints", "profiles", "results"],
                "allow_duplicate": True,
            })

    index = 0
    for heldout in heldouts:
        relative = archive_relative_path(args.run_id, heldout)
        local_archive = deploy_project / relative
        remote_archive = remote_project / relative
        design_relative = design_relative_path(args.run_id, heldout)
        local_design = deploy_project / design_relative
        remote_design = remote_project / design_relative
        for method in methods:
            for offset in range(int(args.n_seeds)):
                seed = int(args.seed_start) + offset
                result_dir = (
                    profile_root / args.implementation / heldout / method
                    / f"seed{seed:04d}"
                )
                remote_result_dir = (
                    remote_project / "profiles" / args.run_id
                    / args.implementation / heldout / method
                    / f"seed{seed:04d}"
                )
                task_checkpoint_dir = (
                    checkpoint_root / args.implementation / heldout / method
                    / f"seed{seed:04d}"
                )
                command = [
                    "env", "LC_ALL=C", "LANG=C", "SCOLHKG_OFFLINE=1",
                    "PYTHONUNBUFFERED=1", "PYTHONDONTWRITEBYTECODE=1",
                    f"OMP_NUM_THREADS={int(args.cpu)}",
                    f"MKL_NUM_THREADS={int(args.cpu)}",
                    f"OPENBLAS_NUM_THREADS={int(args.cpu)}",
                    f"SCOLHKG_EXTERNAL_REPO_ROOT={EXTERNAL_REPOS}",
                    f"SCOLHKG_TRANSFERGPBO_OVERLAY={TRANSFERGPBO_OVERLAY}",
                    "SCOLHKG_HYPERBO_OVERLAY="
                    "/home/zhengliang01/scheduleurm_work/"
                    "python_pkgs/hyperbo_py310",
                    str(args.python),
                    "performance/benchmark_transfer_fairness.py",
                    "--method", method,
                    "--implementation", args.implementation,
                    "--heldout", heldout,
                    "--archive", str(remote_archive),
                    "--out", str(remote_result_dir / "result.json"),
                    "--checkpoint-dir", str(task_checkpoint_dir),
                    "--seed", str(seed),
                    "--d", str(args.d),
                    "--N", str(args.N),
                    "--n0", str(args.n0),
                    "--initial-design", args.initial_design,
                    "--source-train-steps", str(args.source_train_steps),
                    "--target-finetune-steps", str(args.target_finetune_steps),
                ]
                if args.terminal_verification:
                    command.extend([
                        "--terminal-verification",
                        "--terminal-verification-primary-budget",
                        str(args.terminal_verification_primary_budget),
                        "--terminal-verification-support-budget",
                        str(args.terminal_verification_support_budget),
                        "--terminal-verification-delta",
                        str(args.terminal_verification_delta),
                        "--terminal-verification-method",
                        str(args.terminal_verification_method),
                        "--terminal-safe-interior-probability-slack",
                        str(args.terminal_safe_interior_probability_slack),
                    ])
                else:
                    command.append("--no-terminal-verification")
                if args.initial_design == "source_informed":
                    command.extend([
                        "--initial-design-file", str(remote_design),
                    ])
                if args.implementation == "official":
                    command.insert(
                        5,
                        "PYTHONPATH="
                        f"{TRANSFER_TORCH_OVERLAY}:{TRANSFERGPBO_OVERLAY}",
                    )
                index += 1
                specs.append({
                    "description": (
                        f"transfer fairness {args.implementation} {heldout} "
                        f"{method} seed={seed}"
                    ),
                    "cmd": f"{shlex.join(command)} && echo DONE",
                    "cwd": str(deploy_project),
                    "signature": (
                        f"KG_op/transfer_fairness/{args.run_id}/"
                        f"{args.implementation}/{heldout}/{method}/"
                        f"seed{seed:04d}"
                    ),
                    "project": "KG-SYNTH",
                    "vram": 0,
                    "cpu": int(args.cpu),
                    "ram_mb": int(args.ram_mb),
                    "allowed_nodes": list(nodes),
                    "wait_for_files": [
                        str(local_archive),
                        *(
                            [str(local_design)]
                            if args.initial_design == "source_informed"
                            else []
                        ),
                    ],
                    "ckpt_dir": str(task_checkpoint_dir),
                    "ckpt_glob": "**/*.pkl",
                    "allow_initial_resume_scan_error": True,
                    "allow_no_resume": True,
                    "result_dir": str(remote_result_dir),
                    "local_result_dir": str(result_dir),
                    "stage_excludes": ["checkpoints", "profiles", "results"],
                    "allow_duplicate": True,
                })
    return specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--deploy", type=Path, default=DEFAULT_DEPLOY)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument(
        "--manifest", type=Path,
        default=DEFAULT_DEPLOY /
        "SC-OLH-KG/performance/manifests/v18b_exactkg_mcdiag.json",
    )
    parser.add_argument(
        "--run-id", default=f"transfer_fair_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--implementation", choices=("paper_core", "official"),
                        default="paper_core")
    parser.add_argument("--methods", default=",".join(ALL_METHODS))
    parser.add_argument("--allow-unconfigured", action="store_true")
    parser.add_argument("--heldouts", default=(
        "FactorShockStatePolicyRZDT1,InventorySupplyChain,QueueResourceControl"
    ))
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--d", type=int, default=50)
    parser.add_argument(
        "--initial-design",
        choices=(
            "common_sobol",
            "source_informed",
            "native_source_sequential",
        ),
        default="common_sobol",
    )
    parser.add_argument(
        "--structural-prior-profile",
        default="low_frequency_only",
    )
    parser.add_argument(
        "--proposal-mode",
        choices=(
            "rank_spanning",
            "risk_coordinate_atlas",
            "risk_objective_atlas",
        ),
        default="risk_objective_atlas",
    )
    parser.add_argument(
        "--source-design-mode",
        choices=("random", "universal_mixture", "shared_uniform"),
        default="universal_mixture",
    )
    parser.add_argument("--source-train-steps", type=int, default=0)
    parser.add_argument("--target-finetune-steps", type=int, default=100)
    parser.add_argument(
        "--terminal-verification",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--terminal-verification-primary-budget", type=int, default=80)
    parser.add_argument(
        "--terminal-verification-support-budget", type=int, default=96)
    parser.add_argument(
        "--terminal-verification-delta", type=float, default=0.05)
    parser.add_argument(
        "--terminal-verification-method",
        choices=("component_bonferroni", "normal_quantile_tolerance"),
        default="normal_quantile_tolerance",
    )
    parser.add_argument(
        "--terminal-safe-interior-probability-slack",
        type=float,
        default=0.05,
    )
    parser.add_argument("--nodes", default=",".join(CPU_NODES))
    parser.add_argument("--cpu", type=int, default=12)
    parser.add_argument("--ram-mb", type=int, default=16384)
    parser.add_argument("--sync-remote", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.sync_remote and not args.dry_run:
        subprocess.check_call([str(SYNC)])
        if args.implementation == "official":
            subprocess.check_call([str(SYNC_EXTERNAL)])
    specs = build_specs(args)
    if args.dry_run:
        print(json.dumps(specs, indent=2))
        return
    output = subprocess.check_output([
        sys.executable,
        str(args.scheduler),
        "submit-jsonl",
        "--stdin",
        "--trusted",
        "--json",
        "--intent-label",
        f"transfer-fairness-{args.run_id}",
    ], input=json.dumps(specs), text=True)
    print(output, end="" if output.endswith("\n") else "\n")
    if args.dispatch:
        subprocess.check_call([
            sys.executable, str(args.scheduler), "dispatch"
        ])
    print({"run_id": args.run_id, "task_count": len(specs)})


if __name__ == "__main__":
    main()
