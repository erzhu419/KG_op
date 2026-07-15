#!/usr/bin/env python3
"""Submit canonical SOTA fairness contracts to node001-node006."""

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


def parse_csv(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def build_specs(args):
    nodes = parse_csv(args.nodes)
    if not nodes or any(node not in CPU_NODES for node in nodes):
        raise ValueError("nodes must be a nonempty subset of node001-node006")
    protocols = parse_csv(args.protocols)
    methods = parse_csv(args.methods)
    heldouts = parse_csv(args.heldouts)
    deploy_project = Path(args.deploy) / "SC-OLH-KG"
    profile_root = deploy_project / "profiles" / args.run_id
    checkpoint_root = deploy_project / "checkpoints" / args.run_id
    specs = []
    index = 0
    for protocol in protocols:
        for heldout in heldouts:
            for method in methods:
                for offset in range(int(args.n_seeds)):
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
                        f"OMP_NUM_THREADS={int(args.cpu)}",
                        f"MKL_NUM_THREADS={int(args.cpu)}",
                        f"OPENBLAS_NUM_THREADS={int(args.cpu)}",
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
                        "--candidate-timeout-sec",
                        str(args.candidate_timeout_sec),
                    ]
                    node = nodes[index % len(nodes)]
                    index += 1
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
                        "vram": 0,
                        "cpu": int(args.cpu),
                        "ram_mb": int(args.ram_mb),
                        "require_node": node,
                        "allowed_nodes": list(nodes),
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
        default="target_n20,shared_archive_n20,target_n404",
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
    parser.add_argument("--nodes", default=",".join(CPU_NODES))
    parser.add_argument("--cpu", type=int, default=12)
    parser.add_argument("--ram-mb", type=int, default=32768)
    parser.add_argument("--candidate-timeout-sec", type=float, default=3600.0)
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
    if args.dispatch:
        subprocess.check_call([
            sys.executable, str(args.scheduler), "dispatch",
        ])
    print({"run_id": args.run_id, "task_count": len(specs)})


if __name__ == "__main__":
    main()
