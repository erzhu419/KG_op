#!/usr/bin/env python3
"""Submit the frozen two-phase OPSD region-holdout confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SYNC = ROOT / "scripts/sync_scolhkg_scheduler_deploy.sh"
DEFAULT_SCHEDULER = Path.home() / "mine_code/scheduleurm/skill/scheduler.py"
DEFAULT_DEPLOY = Path.home() / "mine_code/KG_op_scheduler_deploy"
REMOTE_PYTHON = Path(
    "/home/zhengliang01/scheduleurm_work/conda_envs/"
    "scomp-py310/bin/python"
)
REGISTRATION = (
    ROOT
    / "SC-OLH-KG/performance/manifests/"
    / "or_review_confirmatory_execution_v1.json"
)
DATA_RELATIVE = Path("SC-OLH-KG/data/external/opsd_time_series_extended_v2.npz")
DESIGN_RELATIVE = Path("SC-OLH-KG/archives/or_review_energy_v2_designs")
RESULT_RELATIVE = Path("SC-OLH-KG/results/or_review_v1/energy_region_holdout")
CPU_NODES = tuple(f"node{index:03d}" for index in range(1, 7))
TARGET_CELL_COUNT = 450
TARGET_MARKET_COUNT = 18


def _partition(total, shards):
    base, remainder = divmod(int(total), int(shards))
    ranges = []
    start = 0
    for index in range(int(shards)):
        stop = start + base + int(index < remainder)
        ranges.append((start, stop))
        start = stop
    return tuple(ranges)


def load_registration(path=REGISTRATION):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("contract_id") != "or_review_confirmatory_execution_v1":
        raise ValueError("unexpected energy execution registration")
    return payload


def _common_env():
    return [
        "env",
        "LC_ALL=C",
        "LANG=C",
        "SCOLHKG_OFFLINE=1",
        "PYTHONUNBUFFERED=1",
        "PYTHONDONTWRITEBYTECODE=1",
        "OPENBLAS_NUM_THREADS=1",
        "OMP_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
    ]


def _stage_excludes():
    return [
        "SC-OLH-KG/results",
        "SC-OLH-KG/profiles",
        "SC-OLH-KG/checkpoints",
        "SC-OLH-KG/manuscript/review",
        "proof/.lake",
        "**/__pycache__",
    ]


def _design_paths(args):
    deploy = Path(args.deploy)
    return (
        deploy / DATA_RELATIVE,
        deploy / DESIGN_RELATIVE,
    )


def validate_frozen_designs(design_dir, freeze_commit):
    design_dir = Path(design_dir)
    summary = design_dir / "design_matrix.summary.json"
    if not summary.is_file():
        raise FileNotFoundError(
            f"energy target phase requires completed designs: {summary}")
    payload = json.loads(summary.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ValueError("energy design matrix is not complete")
    if payload.get("freeze_commit") != str(freeze_commit):
        raise ValueError("energy design freeze commit changed")
    if int(payload.get("market_count", 0)) != TARGET_MARKET_COUNT:
        raise ValueError("energy design market count changed")
    designs = tuple(design_dir.glob("source_atlas__target-*.json"))
    if len(designs) != TARGET_MARKET_COUNT:
        raise ValueError(
            f"expected {TARGET_MARKET_COUNT} energy designs, found "
            f"{len(designs)}"
        )
    return summary


def build_design_spec(args, registration):
    data_path, design_dir = _design_paths(args)
    freeze_commit = str(registration["method_freeze_commit"])
    command = [
        *_common_env(),
        str(args.python),
        "-u",
        "SC-OLH-KG/performance/run_external_energy_v2_matrix.py",
        "--phase",
        "designs",
        "--data",
        str(data_path),
        "--output-dir",
        str(design_dir),
        "--design-dir",
        str(design_dir),
        "--freeze-commit",
        freeze_commit,
        "--workers",
        str(args.design_workers),
    ]
    return {
        "description": "OR review OPSD V2 freeze 18 source designs",
        "cmd": f"{shlex.join(command)} && echo DONE",
        "cwd": str(args.deploy),
        "signature": "KG_op/or_review_v1_retry/energy/designs/all18",
        "project": "KG-SYNTH",
        "vram": 0,
        "cpu": int(args.design_workers),
        "ram_mb": int(args.design_ram_mb),
        "allowed_nodes": list(CPU_NODES),
        "allow_cpu_training": True,
        "cpu_training_justification": (
            "Frozen OPSD region-holdout source-design materialization."
        ),
        "result_dir": str(design_dir),
        "local_result_dir": str(design_dir),
        "stage_excludes": _stage_excludes(),
        "allow_duplicate": True,
    }


def build_target_specs(args, registration):
    data_path, design_dir = _design_paths(args)
    freeze_commit = str(registration["method_freeze_commit"])
    validate_frozen_designs(design_dir, freeze_commit)
    specs = []
    local_root = ROOT / RESULT_RELATIVE
    remote_root = Path(args.deploy) / RESULT_RELATIVE
    for shard_index, (start, stop) in enumerate(
        _partition(TARGET_CELL_COUNT, args.target_shards)
    ):
        remote_output = remote_root / f"shard_{shard_index}"
        local_output = local_root / f"shard_{shard_index}"
        command = [
            *_common_env(),
            str(args.python),
            "-u",
            "SC-OLH-KG/performance/run_external_energy_v2_matrix.py",
            "--phase",
            "targets",
            "--data",
            str(data_path),
            "--output-dir",
            str(remote_output),
            "--design-dir",
            str(design_dir),
            "--freeze-commit",
            freeze_commit,
            "--start",
            str(start),
            "--end",
            str(stop),
            "--workers",
            str(args.target_workers),
        ]
        specs.append({
            "description": (
                f"OR review OPSD V2 target shard {shard_index + 1}/"
                f"{args.target_shards} [{start},{stop})"
            ),
            "cmd": f"{shlex.join(command)} && echo DONE",
            "cwd": str(args.deploy),
            "signature": (
                f"KG_op/or_review_v1_retry/energy/targets/"
                f"shard{shard_index:02d}/{start}-{stop}"
            ),
            "project": "KG-SYNTH",
            "vram": 0,
            "cpu": int(args.target_workers),
            "ram_mb": int(args.target_ram_mb),
            "allowed_nodes": list(CPU_NODES),
            "preferred_node": CPU_NODES[shard_index % len(CPU_NODES)],
            "allow_cpu_training": True,
            "cpu_training_justification": (
                "Frozen non-SAAS OPSD region-holdout matrix with one BLAS "
                "thread per independent target cell."
            ),
            "wait_for_files": [str(
                design_dir / "design_matrix.summary.json")],
            "result_dir": str(remote_output),
            "local_result_dir": str(local_output),
            "stage_excludes": _stage_excludes(),
            "allow_duplicate": True,
        })
    return specs


def submit(args, specs):
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
            f"or-review-energy-v2-{args.phase}",
        ],
        input=payload,
        text=True,
        check=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("designs", "targets"), required=True)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--deploy", type=Path, default=DEFAULT_DEPLOY)
    parser.add_argument("--python", type=Path, default=REMOTE_PYTHON)
    parser.add_argument("--design-workers", type=int, default=18)
    parser.add_argument("--design-ram-mb", type=int, default=16384)
    parser.add_argument("--target-shards", type=int, default=6)
    parser.add_argument("--target-workers", type=int, default=75)
    parser.add_argument("--target-ram-mb", type=int, default=32768)
    parser.add_argument("--no-sync", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    registration = load_registration()
    specs = (
        [build_design_spec(args, registration)]
        if args.phase == "designs"
        else build_target_specs(args, registration)
    )
    if args.dry_run:
        print(json.dumps(specs, indent=2))
        return
    if not args.no_sync:
        subprocess.run([str(SYNC)], check=True, cwd=ROOT)
    submit(args, specs)


if __name__ == "__main__":
    main()
