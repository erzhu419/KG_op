#!/usr/bin/env python3
"""Submit checkpoint-local, offline group-complexity diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


DEFAULT_SCHEDULER = Path.home() / ".claude/skills/scheduler/scheduler.py"
DEFAULT_DEPLOY = Path.home() / "mine_code/KG_op_scheduler_deploy"
DEFAULT_PYTHON = (
    "/home/zhengliang01/scheduleurm_work/conda_envs/"
    "scomp-py310/bin/python"
)
CPU_NODES = tuple(f"node{index:03d}" for index in range(1, 7))


def _domain_seed(signature):
    parts = str(signature).rstrip("/").split("/")
    if len(parts) < 2 or not parts[-1].startswith("seed"):
        raise ValueError(f"cannot parse heldout/seed from {signature!r}")
    return parts[-2], int(parts[-1][4:])


def build_specs(args, source_tasks):
    deploy_project = Path(args.deploy) / "SC-OLH-KG"
    specs = []
    seen = set()
    for task in sorted(source_tasks, key=lambda row: int(row["id"][1:])):
        if task.get("status") != "done":
            raise ValueError(f"source task {task['id']} is not done")
        node = str(task.get("node") or "")
        if node not in CPU_NODES:
            raise ValueError(f"source task {task['id']} is not on a CPU node")
        heldout, seed = _domain_seed(task["signature"])
        key = (heldout, seed)
        if key in seen:
            raise ValueError(f"duplicate source checkpoint {key}")
        seen.add(key)
        checkpoint = (
            deploy_project / "checkpoints" / args.source_run_id
            / heldout / f"seed{seed}" / "checkpoint_latest.pkl"
        )
        result_dir = (
            deploy_project / "profiles" / args.run_id
            / heldout / f"seed{seed}"
        )
        result_file = result_dir / "result.json"
        command = shlex.join([
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
            "performance/diagnose_group_complexity_checkpoint.py",
            "--manifest", str(args.manifest),
            "--heldout", heldout,
            "--seed", str(seed),
            "--checkpoint", str(checkpoint),
            "--pool-size", str(int(args.pool_size)),
            "--out", str(result_file),
        ])
        specs.append({
            "description": (
                f"SC-OLH-KG {args.run_id} offline complexity audit "
                f"{heldout} seed={seed}"
            ),
            "cmd": command,
            "cwd": str(deploy_project),
            "signature": (
                f"KG_op/scolhkg_complexity_audit/{args.run_id}/"
                f"{heldout}/seed{seed}"
            ),
            "project": "KG-SYNTH",
            "vram": 0,
            "cpu": int(args.cpu),
            "ram_mb": int(args.ram_mb),
            "require_node": node,
            "allowed_nodes": [node],
            "result_dir": str(result_dir),
            "local_result_dir": str(result_dir),
            "stage_excludes": ["checkpoints", "profiles", "results"],
            "allow_no_resume": True,
            "allow_duplicate": bool(args.allow_duplicate),
        })
    return specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--deploy", type=Path, default=DEFAULT_DEPLOY)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-task-ids", nargs="+", required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            DEFAULT_DEPLOY
            / "SC-OLH-KG/performance/manifests/v18b_exactkg_mcdiag.json"
        ),
    )
    parser.add_argument("--pool-size", type=int, default=512)
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--ram-mb", type=int, default=4096)
    parser.add_argument("--allow-duplicate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()

    status = subprocess.check_output([
        sys.executable,
        str(args.scheduler),
        "status",
        "--json",
        "--brief",
        "--ids",
        *args.source_task_ids,
    ], text=True)
    source_tasks = json.loads(status).get("tasks", [])
    if len(source_tasks) != len(args.source_task_ids):
        raise ValueError("not every source task was found in scheduler state")
    specs = build_specs(args, source_tasks)
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
            f"scolhkg-complexity-{args.run_id}",
        ],
        input=json.dumps(specs),
        text=True,
    )
    print(output, end="" if output.endswith("\n") else "\n")
    submitted = json.loads(output).get("submitted", [])
    task_ids = [row["id"] for row in submitted if row.get("id")]
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
