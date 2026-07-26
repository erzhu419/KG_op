#!/usr/bin/env python3
"""Submit canonical SOTA fairness contracts to audited CPU/GPU nodes."""

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
DEFAULT_PYTHON = (
    "/home/zhengliang01/scheduleurm_work/conda_envs/scomp-py310/bin/python"
)
BOTORCH_OVERLAY = (
    "/home/zhengliang01/scheduleurm_work/python_pkgs/botorch_overlay_py310"
)
CPU_NODES = tuple(f"node{i:03d}" for i in range(1, 7))
GPU_NODES = ("jtl110gpu", "jtl110gpu2", "jtl311linux", "node007")
CUDA_METHODS = (
    "botorch_turbo",
    "botorch_scbo",
    "botorch_saasbo",
)


def parse_csv(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def build_specs(args):
    defaults = {
        "target_budget": 0,
        "d": 50,
        "n0": 10,
        "terminal_verification": True,
        "terminal_verification_primary_budget": 80,
        "terminal_verification_support_budget": 96,
        "terminal_verification_delta": 0.05,
        "terminal_verification_method": "normal_quantile_tolerance",
        "terminal_safe_interior_probability_slack": 0.05,
        "gpu_nodes": "node007",
        "gpu_methods": ",".join(CUDA_METHODS),
        "gpu_cpu": 12,
        "gpu_ram_mb": 32768,
        "gpu_vram_mb": 8192,
    }
    for name, value in defaults.items():
        if not hasattr(args, name):
            setattr(args, name, value)
    nodes = parse_csv(args.nodes)
    if not nodes or any(node not in CPU_NODES for node in nodes):
        raise ValueError("nodes must be a nonempty subset of node001-node006")
    gpu_nodes = parse_csv(args.gpu_nodes)
    if not gpu_nodes or any(node not in GPU_NODES for node in gpu_nodes):
        raise ValueError(
            "gpu-nodes must be a nonempty subset of "
            + ",".join(GPU_NODES)
        )
    protocols = parse_csv(args.protocols)
    methods = parse_csv(args.methods)
    gpu_methods = set(parse_csv(args.gpu_methods))
    unknown_gpu_methods = sorted(gpu_methods - set(CUDA_METHODS))
    if unknown_gpu_methods:
        raise ValueError(
            f"methods without an audited CUDA adapter: {unknown_gpu_methods}")
    heldouts = parse_csv(args.heldouts)
    deploy_project = Path(args.deploy) / "SC-OLH-KG"
    profile_root = deploy_project / "profiles" / args.run_id
    checkpoint_root = deploy_project / "checkpoints" / args.run_id
    specs = []
    cpu_index = 0
    gpu_index = 0
    for protocol in protocols:
        for heldout in heldouts:
            for method in methods:
                for offset in range(int(args.n_seeds)):
                    use_gpu = method in gpu_methods
                    task_cpu = (
                        int(args.gpu_cpu) if use_gpu else int(args.cpu))
                    task_ram_mb = (
                        int(args.gpu_ram_mb)
                        if use_gpu else int(args.ram_mb)
                    )
                    torch_device = "cuda" if use_gpu else "cpu"
                    seed = int(args.seed_start) + offset
                    result_dir = (
                        profile_root / protocol / heldout / method
                        / f"seed{seed:04d}"
                    )
                    checkpoint_dir = (
                        checkpoint_root / protocol / heldout / method
                        / f"seed{seed:04d}"
                    )
                    command = [
                        "env",
                        "LC_ALL=C",
                        "LANG=C",
                        "SCOLHKG_OFFLINE=1",
                        "PYTHONUNBUFFERED=1",
                        "PYTHONDONTWRITEBYTECODE=1",
                        f"OMP_NUM_THREADS={task_cpu}",
                        f"MKL_NUM_THREADS={task_cpu}",
                        f"OPENBLAS_NUM_THREADS={task_cpu}",
                        f"PYTHONPATH={BOTORCH_OVERLAY}",
                        str(args.python),
                        "performance/benchmark_sota_fairness.py",
                        "--protocol", protocol,
                        "--method", method,
                        "--heldout", heldout,
                        "--seed", str(seed),
                        "--manifest", str(args.manifest),
                        "--out", str(result_dir / "result.json"),
                        "--checkpoint-dir", str(checkpoint_dir),
                        "--target-budget", str(args.target_budget),
                        "--d", str(args.d),
                        "--n0", str(args.n0),
                        "--candidate-timeout-sec",
                        str(args.candidate_timeout_sec),
                        "--torch-device", torch_device,
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
                    if use_gpu:
                        node = gpu_nodes[gpu_index % len(gpu_nodes)]
                        allowed_nodes = list(gpu_nodes)
                        gpu_index += 1
                    else:
                        node = nodes[cpu_index % len(nodes)]
                        allowed_nodes = list(nodes)
                        cpu_index += 1
                    specs.append({
                        "description": (
                            f"canonical SOTA fairness {protocol} {heldout} "
                            f"{method} seed={seed}"
                        ),
                        "cmd": f"{shlex.join(command)} && echo DONE",
                        "cwd": str(deploy_project),
                        "signature": (
                            f"KG_op/sota_fairness/{args.run_id}/{protocol}/"
                            f"{heldout}/{method}/seed{seed:04d}"
                        ),
                        "project": "KG-SYNTH",
                        "vram": (
                            int(args.gpu_vram_mb) if use_gpu else 0),
                        "cpu": task_cpu,
                        "ram_mb": task_ram_mb,
                        "require_node": node,
                        "allowed_nodes": allowed_nodes,
                        "ckpt_dir": str(checkpoint_dir),
                        "ckpt_glob": "**/*.pkl",
                        "allow_initial_resume_scan_error": True,
                        "allow_no_resume": True,
                        "result_dir": str(result_dir),
                        "local_result_dir": str(result_dir),
                        "stage_excludes": ["checkpoints", "profiles", "results"],
                        "allow_duplicate": True,
                    })
    return specs


def build_dispatch_command(scheduler: Path, task_ids: list[str]) -> list[str]:
    command = [sys.executable, str(scheduler), "dispatch"]
    for task_id in task_ids:
        command.extend(["--task-id", task_id])
    return command


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
        "--run-id", default=f"sota_fair_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument(
        "--protocols",
        default=(
            "target_n13,shared_archive_n13,target_cost_matched_n397"
        ),
    )
    parser.add_argument(
        "--methods",
        default="botorch_turbo,botorch_scbo,botorch_saasbo",
    )
    parser.add_argument(
        "--heldouts",
        default=(
            "FactorShockStatePolicyRZDT1,InventorySupplyChain,"
            "QueueResourceControl"
        ),
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--target-budget", type=int, default=0)
    parser.add_argument("--d", type=int, default=50)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--nodes", default=",".join(CPU_NODES))
    parser.add_argument("--cpu", type=int, default=12)
    parser.add_argument("--ram-mb", type=int, default=8192)
    parser.add_argument("--gpu-nodes", default="node007")
    parser.add_argument(
        "--gpu-methods",
        default=",".join(CUDA_METHODS),
        help=(
            "BoTorch methods routed to CUDA nodes. Pass an empty value "
            "only for an explicit CPU deployment ablation."
        ),
    )
    parser.add_argument("--gpu-cpu", type=int, default=12)
    parser.add_argument("--gpu-ram-mb", type=int, default=32768)
    parser.add_argument("--gpu-vram-mb", type=int, default=8192)
    parser.add_argument("--candidate-timeout-sec", type=float, default=3600.0)
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
    parser.add_argument("--sync-remote", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.sync_remote and not args.dry_run:
        subprocess.check_call([str(SYNC)])
    specs = build_specs(args)
    if args.dry_run:
        print(json.dumps(specs, indent=2))
        return
    output = subprocess.check_output(
        [
            sys.executable,
            str(args.scheduler),
            "submit-jsonl",
            "--stdin",
            "--trusted",
            "--json",
            "--intent-label",
            f"sota-fairness-{args.run_id}",
        ],
        input=json.dumps(specs),
        text=True,
    )
    print(output, end="" if output.endswith("\n") else "\n")
    submitted = json.loads(output).get("submitted", [])
    task_ids = [item["id"] for item in submitted if item.get("id")]
    if args.dispatch and task_ids:
        subprocess.check_call(build_dispatch_command(args.scheduler, task_ids))
    print({
        "run_id": args.run_id,
        "task_count": len(task_ids),
        "task_ids": task_ids,
    })


if __name__ == "__main__":
    main()
