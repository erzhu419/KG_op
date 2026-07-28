#!/usr/bin/env python3
"""Submit shared-shock/replication HVD identifiability cells."""

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
    "/home/zhengliang01/scheduleurm_work/conda_envs/scomp-py310/bin/python"
)
CPU_NODES = tuple(f"node{index:03d}" for index in range(1, 7))
MODES = (
    "pooled",
    "class",
    "orthogonal_pointwise",
    "factor_pointwise",
    "factor_cumulative",
)


def _parse_csv(value, cast=str):
    return tuple(cast(item.strip()) for item in str(value).split(",") if item.strip())


def build_specs(args):
    nodes = _parse_csv(args.nodes)
    modes = _parse_csv(args.modes)
    shock_scales = _parse_csv(args.shock_scales, float)
    replications = _parse_csv(args.replications, int)
    if not nodes or any(node not in CPU_NODES for node in nodes):
        raise ValueError("nodes must be a nonempty subset of node001-node006")
    if not modes or any(mode not in MODES for mode in modes):
        raise ValueError("unknown HVD mode")
    if not shock_scales or any(value < 0.0 for value in shock_scales):
        raise ValueError("shock scales must be nonnegative")
    if not replications or any(value < 2 for value in replications):
        raise ValueError("replications must be at least two")

    local_project = Path(args.deploy) / "SC-OLH-KG"
    remote_project = REMOTE_ROOT / "SC-OLH-KG"
    specs = []
    for mode in modes:
        for shock_scale in shock_scales:
            shock_label = f"{shock_scale:g}".replace(".", "p")
            for replicates in replications:
                for offset in range(int(args.n_seeds)):
                    seed = int(args.seed_start) + offset
                    cell = (
                        f"{mode}/shock{shock_label}/rep{int(replicates)}/"
                        f"seed{seed}"
                    )
                    remote_result_dir = (
                        remote_project / "profiles" / args.run_id / cell
                    )
                    local_result_dir = (
                        local_project / "profiles" / args.run_id / cell
                    )
                    command = [
                        "env", "LC_ALL=C", "LANG=C", "SCOLHKG_OFFLINE=1",
                        "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
                        "OPENBLAS_NUM_THREADS=1", "PYTHONUNBUFFERED=1",
                        "PYTHONDONTWRITEBYTECODE=1", str(args.python),
                        "performance/benchmark_hvd_identifiability.py",
                        "--mode", mode,
                        "--shock-scale", str(shock_scale),
                        "--replicates", str(replicates),
                        "--seed", str(seed),
                        "--d", str(args.d),
                        "--n-train", str(args.n_train),
                        "--tau", str(args.tau),
                        "--activation-min-records", str(args.activation_min_records),
                        "--out", str(remote_result_dir / "result.json"),
                    ]
                    specs.append({
                        "description": f"HVD identifiability {cell}",
                        "cmd": f"{shlex.join(command)} && echo DONE",
                        "cwd": str(local_project),
                        "signature": f"KG_op/hvd_identifiability/{args.run_id}/{cell}",
                        "project": "KG-SYNTH",
                        "vram": 0,
                        "cpu": int(args.cpu),
                        "ram_mb": int(args.ram_mb),
                        "allowed_nodes": list(nodes),
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
    parser.add_argument("--nodes", default=",".join(CPU_NODES))
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--shock-scales", default="0,0.25,1,4")
    parser.add_argument("--replications", default="2,4,8,16")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--d", type=int, default=50)
    parser.add_argument("--n-train", type=int, default=32)
    parser.add_argument("--tau", type=float, default=0.25)
    parser.add_argument("--activation-min-records", type=int, default=16)
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument("--ram-mb", type=int, default=2048)
    parser.add_argument(
        "--run-id",
        default=f"scolh_hvd_identifiability_{time.strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--no-sync", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
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
            "submit-jsonl",
            "--stdin",
            "--trusted",
            "--json",
            "--intent-label",
            str(args.run_id),
        ],
        input=payload,
        text=True,
        check=True,
    )


if __name__ == "__main__":
    main()
