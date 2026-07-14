#!/usr/bin/env python3
"""Submit the frozen 4x4x3x2 transferable-boundary screen."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time


DEFAULT_SCHEDULER = Path.home() / "mine_code/scheduleurm/skill/scheduler.py"
DEFAULT_DEPLOY = Path.home() / "mine_code/KG_op_scheduler_deploy"
DEFAULT_PYTHON = (
    "/home/zhengliang01/scheduleurm_work/conda_envs/"
    "scomp-py310/bin/python"
)
CPU_NODES = tuple(f"node{index:03d}" for index in range(1, 7))
COORDINATES = (
    "explicit_stable",
    "learned_psi",
    "boundary_latent",
    "hybrid_explicit_latent",
)
GEOMETRIES = (
    "linear_monotone",
    "diagonal_psd",
    "low_rank_psd",
    "rbf",
)
ADAPTATIONS = ("frozen", "shift_scale", "orthogonal_shift")
RANKS = (2, 4)


def parse_csv(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def variant_id(coordinate, geometry, adaptation, rank):
    return f"{coordinate}__{geometry}__{adaptation}__r{int(rank)}"


def parse_task_indices(value, total):
    value = str(value or "").strip()
    if not value:
        return list(range(int(total)))
    selected = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = token.split("-", 1)
            selected.update(range(int(start), int(end) + 1))
        else:
            selected.add(int(token))
    if not selected or min(selected) < 0 or max(selected) >= int(total):
        raise ValueError(
            f"task indices must be within [0, {int(total) - 1}]")
    return sorted(selected)


def build_specs(args):
    nodes = parse_csv(args.nodes)
    if not nodes or any(node not in CPU_NODES for node in nodes):
        raise ValueError("nodes must be a nonempty subset of node001-node006")
    deploy_project = Path(args.deploy) / "SC-OLH-KG"
    root = deploy_project / "profiles" / str(args.run_id)
    specs = []
    combinations = itertools.product(
        COORDINATES, GEOMETRIES, ADAPTATIONS, RANKS)
    for index, (coordinate, geometry, adaptation, rank) in enumerate(combinations):
        name = variant_id(coordinate, geometry, adaptation, rank)
        result_dir = root / name
        result_file = result_dir / "result.json"
        command = [
            "env",
            "LC_ALL=C",
            "LANG=C",
            "OMP_NUM_THREADS=2",
            "MKL_NUM_THREADS=2",
            "OPENBLAS_NUM_THREADS=2",
            "SCOLHKG_OFFLINE=1",
            "PYTHONUNBUFFERED=1",
            "PYTHONDONTWRITEBYTECODE=1",
            str(args.python),
            "performance/benchmark_boundary_coordinate_screen.py",
            "--coordinate", coordinate,
            "--geometry", geometry,
            "--adaptation", adaptation,
            "--rank", str(int(rank)),
            "--records-per-domain", str(int(args.records_per_domain)),
            "--pilot-size", str(int(args.pilot_size)),
            "--evaluation-size", str(int(args.evaluation_size)),
            "--pool-multiplier", str(int(args.pool_multiplier)),
            "--jobs", str(int(args.jobs)),
            "--upper-alpha", str(float(getattr(
                args, "upper_alpha", 0.01))),
            "--calibration-prior-df", str(float(getattr(
                args, "calibration_prior_df", 1.0))),
            "--target-residual-rank", str(int(getattr(
                args, "target_residual_rank", 1))),
            "--pilot-selection-mode", str(getattr(
                args, "pilot_selection_mode", "random")),
            "--data-seed", str(int(args.data_seed)),
            "--out", str(result_file),
        ]
        specs.append({
            "description": f"SC-OLH-KG TCB screen {name}",
            "cmd": shlex.join(command),
            "cwd": str(deploy_project),
            "signature": f"KG_op/tcb_screen/{args.run_id}/{name}",
            "project": "KG-OFFLINE",
            "vram": 0,
            "cpu": int(args.cpu),
            "ram_mb": int(args.ram_mb),
            "require_node": nodes[index % len(nodes)],
            "allowed_nodes": list(nodes),
            "allow_no_resume": True,
            "result_dir": str(result_dir),
            "local_result_dir": str(result_dir),
            "stage_excludes": ["checkpoints", "profiles", "results"],
            "allow_duplicate": bool(args.allow_duplicate),
        })
    if len(specs) != 96:
        raise AssertionError(f"boundary screen must have 96 tasks, got {len(specs)}")
    indices = parse_task_indices(getattr(args, "task_indices", ""), 96)
    return [specs[index] for index in indices]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--deploy", type=Path, default=DEFAULT_DEPLOY)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument(
        "--run-id",
        default=f"tcb_screen_{time.strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--nodes", default=",".join(CPU_NODES))
    parser.add_argument("--cpu", type=int, default=12)
    parser.add_argument("--ram-mb", type=int, default=4096)
    parser.add_argument("--records-per-domain", type=int, default=96)
    parser.add_argument("--pilot-size", type=int, default=10)
    parser.add_argument("--evaluation-size", type=int, default=48)
    parser.add_argument("--pool-multiplier", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=5)
    parser.add_argument("--upper-alpha", type=float, default=0.01)
    parser.add_argument("--calibration-prior-df", type=float, default=1.0)
    parser.add_argument("--target-residual-rank", type=int, default=1)
    parser.add_argument(
        "--pilot-selection-mode",
        choices=(
            "random",
            "source_boundary_d_optimal",
            "source_boundary_robust_d_optimal",
        ),
        default="random",
    )
    parser.add_argument("--data-seed", type=int, default=33001)
    parser.add_argument(
        "--task-indices",
        default="",
        help="Stable zero-based indices/ranges, e.g. 0,12-15.",
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
        f"scolhkg-tcb-{args.run_id}",
        "--intent-ttl",
        "120",
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
