#!/usr/bin/env python3
"""Submit the frozen OR-review profile matrices to the CPU cluster."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEDULER = Path.home() / "mine_code/scheduleurm/skill/scheduler.py"
DEFAULT_REGISTRATION = (
    ROOT
    / "SC-OLH-KG/performance/manifests/"
    / "or_review_confirmatory_execution_v1.json"
)
REMOTE_PYTHON = Path(
    "/home/zhengliang01/scheduleurm_work/conda_envs/"
    "scomp-py310/bin/python"
)
CPU_NODES = tuple(f"node{index:03d}" for index in range(1, 7))
MATRIX_SPECS = {
    "primary": (2880, "profile_primary", None),
    "sensitivity": (8640, "profile_sensitivity", "1000"),
    "schema_descriptor": (1280, "profile_schema_descriptor", "1000"),
    "equal_preverification_cost": (
        800,
        "profile_equal_preverification_cost",
        "1000",
    ),
}


def _csv(value):
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def _partition(total, shards):
    total = int(total)
    shards = int(shards)
    if total <= 0 or shards <= 0:
        raise ValueError("total and shards must be positive")
    base, remainder = divmod(total, shards)
    ranges = []
    start = 0
    for index in range(shards):
        stop = start + base + int(index < remainder)
        ranges.append((start, stop))
        start = stop
    return tuple(ranges)


def load_registration(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("contract_id") != "or_review_confirmatory_execution_v1":
        raise ValueError("unexpected confirmatory registration")
    if payload.get("status") != "frozen_not_yet_interpreted":
        raise ValueError("confirmatory registration is not frozen")
    return payload


def _stage_excludes():
    return [
        ".tmp",
        "markdown",
        "reference",
        "repo",
        "SC-OLH-KG/manuscript/review",
        "SC-OLH-KG/results",
        "SC-OLH-KG/profiles",
        "SC-OLH-KG/checkpoints",
        "SC-OLH-KG/data/external",
        "proof/.lake",
        "**/__pycache__",
    ]


def build_specs(args, registration):
    freeze_commit = str(registration["method_freeze_commit"])
    requested = _csv(args.matrices)
    unknown = set(requested) - set(MATRIX_SPECS)
    if unknown:
        raise ValueError(f"unknown matrices: {sorted(unknown)}")
    if not Path(args.python).is_absolute():
        raise ValueError("remote Python must be an absolute path")

    specs = []
    for matrix in requested:
        total, directory, dimensions = MATRIX_SPECS[matrix]
        for shard_index, (start, stop) in enumerate(
            _partition(total, args.shards)
        ):
            output_dir = (
                ROOT / "SC-OLH-KG/results/or_review_v1"
                / directory / f"shard_{shard_index}"
            )
            command = [
                "env",
                "LC_ALL=C",
                "LANG=C",
                "SCOLHKG_OFFLINE=1",
                "PYTHONUNBUFFERED=1",
                "PYTHONDONTWRITEBYTECODE=1",
                "OPENBLAS_NUM_THREADS=1",
                "OMP_NUM_THREADS=1",
                "MKL_NUM_THREADS=1",
                str(args.python),
                "-u",
                "SC-OLH-KG/performance/run_profile_stress_matrix.py",
                "--output-dir",
                str(output_dir),
                "--freeze-commit",
                freeze_commit,
                "--matrix",
                matrix,
                "--start",
                str(start),
                "--end",
                str(stop),
                "--workers",
                str(args.workers),
            ]
            if dimensions is not None:
                command.extend(["--dimensions", dimensions])
            specs.append({
                "description": (
                    f"OR review {matrix} shard {shard_index + 1}/"
                    f"{args.shards} [{start},{stop})"
                ),
                "cmd": f"{shlex.join(command)} && echo DONE",
                "cwd": str(ROOT),
                "signature": (
                    f"KG_op/or_review_v1_retry/{matrix}/"
                    f"shard{shard_index:02d}/{start}-{stop}"
                ),
                "project": "KG-SYNTH",
                "vram": 0,
                "cpu": int(args.workers),
                "ram_mb": int(args.ram_mb),
                "allowed_nodes": list(CPU_NODES),
                "preferred_node": CPU_NODES[shard_index % len(CPU_NODES)],
                "allow_cpu_training": True,
                "cpu_training_justification": (
                    "Frozen non-SAAS confirmatory profile stress matrix; "
                    "each shard is a flat process pool with one BLAS thread "
                    "per worker."
                ),
                "result_dir": str(output_dir),
                "local_result_dir": str(output_dir),
                "stage_excludes": _stage_excludes(),
                "allow_duplicate": True,
            })
    return specs


def build_preflight_spec(args, registration):
    freeze_commit = str(registration["method_freeze_commit"])
    output_dir = ROOT / "SC-OLH-KG/results/or_review_v1/preflight"
    command = [
        "env",
        "LC_ALL=C",
        "LANG=C",
        "SCOLHKG_OFFLINE=1",
        "PYTHONUNBUFFERED=1",
        "PYTHONDONTWRITEBYTECODE=1",
        "OPENBLAS_NUM_THREADS=1",
        "OMP_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
        str(args.python),
        "-u",
        "SC-OLH-KG/performance/run_profile_stress_matrix.py",
        "--output-dir",
        str(output_dir),
        "--freeze-commit",
        freeze_commit,
        "--matrix",
        "primary",
        "--start",
        "0",
        "--end",
        "1",
        "--workers",
        "1",
    ]
    return {
        "description": "OR review absolute-Python preflight",
        "cmd": f"{shlex.join(command)} && echo DONE",
        "cwd": str(ROOT),
        "signature": "KG_op/or_review_v1_retry/preflight/absolute-python",
        "project": "KG-SYNTH",
        "vram": 0,
        "cpu": 1,
        "ram_mb": 2048,
        "allowed_nodes": list(CPU_NODES),
        "allow_cpu_training": True,
        "cpu_training_justification": (
            "One-cell environment preflight for the frozen CPU matrix."
        ),
        "result_dir": str(output_dir),
        "local_result_dir": str(output_dir),
        "stage_excludes": _stage_excludes(),
        "allow_duplicate": True,
    }


def submit_specs(args, specs):
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
            str(args.intent_label),
        ],
        input=payload,
        text=True,
        check=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--registration", type=Path,
                        default=DEFAULT_REGISTRATION)
    parser.add_argument("--python", type=Path, default=REMOTE_PYTHON)
    parser.add_argument("--matrices", default=",".join(MATRIX_SPECS))
    parser.add_argument("--shards", type=int, default=6)
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--ram-mb", type=int, default=65536)
    parser.add_argument(
        "--intent-label", default="or_review_confirmatory_retry_v1")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    registration = load_registration(args.registration)
    specs = (
        [build_preflight_spec(args, registration)]
        if args.preflight_only
        else build_specs(args, registration)
    )
    if args.dry_run:
        print(json.dumps(specs, indent=2))
        return
    submit_specs(args, specs)


if __name__ == "__main__":
    main()
