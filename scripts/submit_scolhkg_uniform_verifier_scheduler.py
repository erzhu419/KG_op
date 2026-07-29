#!/usr/bin/env python3
"""Submit compact, uniform synthetic terminal-verification shards."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "SC-OLH-KG"))

from performance.materialize_uniform_verification_manifest import (  # noqa: E402
    _atomic_json,
    materialize,
)


SYNC = ROOT / "scripts/sync_scolhkg_scheduler_deploy.sh"
DEFAULT_SCHEDULER = Path.home() / "mine_code/scheduleurm/skill/scheduler.py"
DEFAULT_DEPLOY = Path.home() / "mine_code/KG_op_scheduler_deploy"
REMOTE_PYTHON = Path(
    "/home/zhengliang01/scheduleurm_work/conda_envs/scomp-py310/bin/python")
CPU_NODES = tuple(f"node{i:03d}" for i in range(1, 7))


def build_specs(args, manifest):
    project = Path(args.deploy) / "SC-OLH-KG"
    manifest_path = (
        project / "archives" / args.run_id
        / "uniform_verification_manifest.json"
    )
    out_root = project / "profiles" / args.run_id
    row_count = int(manifest["row_count"])
    shard_size = int(args.shard_size)
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    specs = []
    for shard, start in enumerate(range(0, row_count, shard_size)):
        end = min(row_count, start + shard_size)
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
            str(REMOTE_PYTHON),
            "performance/run_uniform_verification_shard.py",
            "--manifest", str(manifest_path),
            "--start", str(start),
            "--end", str(end),
            "--out-root", str(out_root),
            "--jobs", str(min(int(args.cpu), end - start)),
        ]
        specs.append({
            "description": (
                f"uniform verifier shard={shard} rows={start}:{end}"),
            "cmd": f"{shlex.join(command)} && echo DONE",
            "cwd": str(project),
            "signature": (
                f"KG_op/uniform_verifier/{args.run_id}/"
                f"shard{shard:04d}_{start}_{end}"
            ),
            "project": "KG-SYNTH",
            "vram": 0,
            "cpu": int(args.cpu),
            "ram_mb": int(args.ram_mb),
            "allowed_nodes": list(CPU_NODES),
            "wait_for_files": [str(manifest_path)],
            "result_dir": str(out_root),
            "local_result_dir": str(out_root),
            "stage_excludes": ["checkpoints", "profiles", "results"],
            "allow_duplicate": True,
        })
    expected = math.ceil(row_count / shard_size) if row_count else 0
    if len(specs) != expected:
        raise RuntimeError("uniform verifier shard count is inconsistent")
    return specs


def load_or_materialize_manifest(args):
    manifest_path = (
        Path(args.deploy) / "SC-OLH-KG" / "archives" / args.run_id
        / "uniform_verification_manifest.json"
    )
    if args.reuse_existing_manifest:
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"frozen uniform verifier manifest is missing: "
                f"{manifest_path}"
            )
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    if not args.audit or not args.selection:
        raise ValueError(
            "--audit and --selection are required unless "
            "--reuse-existing-manifest is set"
        )
    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    return materialize(
        audit,
        selections=args.selection,
        candidate_count=2,
        candidate_budget=args.candidate_budget,
        familywise_delta=args.familywise_delta,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--deploy", type=Path, default=DEFAULT_DEPLOY)
    parser.add_argument("--audit", default="")
    parser.add_argument("--selection", action="append", default=[])
    parser.add_argument(
        "--run-id",
        default=f"paper_uniform_verifier_{time.strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--candidate-budget", type=int, default=128)
    parser.add_argument("--familywise-delta", type=float, default=0.05)
    parser.add_argument("--expected-per-selection", type=int, default=60)
    parser.add_argument("--shard-size", type=int, default=8)
    parser.add_argument("--cpu", type=int, default=8)
    parser.add_argument("--ram-mb", type=int, default=8192)
    parser.add_argument(
        "--reuse-existing-manifest",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--registration-name",
        default="submission_manifest.json",
    )
    parser.add_argument(
        "--sync-remote",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = load_or_materialize_manifest(args)
    expected = int(args.expected_per_selection)
    if expected > 0 and any(
        int(count) != expected
        for count in manifest["selection_counts"].values()
    ):
        raise RuntimeError(
            "uniform verifier selection count does not match the "
            "preregistered expectation"
        )
    manifest_path = (
        Path(args.deploy) / "SC-OLH-KG" / "archives" / args.run_id
        / "uniform_verification_manifest.json"
    )
    specs = build_specs(args, manifest)
    if args.dry_run:
        print(json.dumps({
            "run_id": args.run_id,
            "manifest": manifest,
            "task_count": len(specs),
            "specs": specs,
        }, indent=2))
        return
    if not args.reuse_existing_manifest:
        _atomic_json(manifest_path, manifest)
    if args.sync_remote:
        subprocess.run([str(SYNC)], cwd=ROOT, check=True)
    output = subprocess.check_output(
        [
            sys.executable,
            str(args.scheduler),
            "submit-jsonl",
            "--stdin",
            "--trusted",
            "--json",
            "--intent-label",
            f"paper-uniform-verifier-{args.run_id}",
        ],
        input=json.dumps(specs),
        text=True,
    )
    response = json.loads(output)
    task_ids = [
        row["id"] for row in response.get("submitted", [])
        if row.get("id")
    ]
    registration = {
        "schema_version": 1,
        "run_id": args.run_id,
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "contract_id": manifest["contract_id"],
        "row_count": manifest["row_count"],
        "selection_counts": manifest["selection_counts"],
        "candidate_budgets": manifest["candidate_budgets"],
        "familywise_delta": manifest["familywise_delta"],
        "task_count": len(task_ids),
        "task_ids": task_ids,
        "checkpoint_or_model_artifacts_transferred": False,
        "reused_existing_manifest": bool(args.reuse_existing_manifest),
    }
    _atomic_json(
        Path(args.deploy) / "SC-OLH-KG" / "profiles" / args.run_id
        / args.registration_name,
        registration,
    )
    if args.dispatch and task_ids:
        command = [sys.executable, str(args.scheduler), "dispatch"]
        for task_id in task_ids:
            command.extend(["--task-id", task_id])
        subprocess.run(command, check=True)
    print(json.dumps(registration, indent=2))


if __name__ == "__main__":
    main()
