#!/usr/bin/env python3
"""Materialize the 20-seed frozen V51 source-informed designs."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


full_gate = _load(
    "mean_alignment_v51_full_submit",
    ROOT / "scripts/submit_scolhkg_mean_alignment_v51_full_gate_scheduler.py",
)
root_submit = full_gate._root_module()
CPU_NODES = full_gate.CPU_NODES
HELDOUTS = tuple(name for name, _ in full_gate.v51.v50.v49.v27.MEAN_SCENARIOS)
REMOTE_ROOT = Path("/home/zhengliang01/scheduleurm_work/KG_op_scheduler_deploy")
DEFAULT_ARCHIVE_RUN_ID = (
    "scolh_lowfreq_support_dimholdout_d1000_s5_20260716_source_d50"
)
DEFAULT_OUTPUT_SOURCE_RUN_ID = (
    "scolh_lowfreq_support_dimholdout_d1000_s20_20260721/"
    "proposals/risk_objective_atlas/low_frequency_only"
)


def _parse_csv(value):
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def build_specs(args):
    nodes = _parse_csv(args.nodes)
    heldouts = _parse_csv(args.heldouts)
    if not nodes or any(node not in CPU_NODES for node in nodes):
        raise ValueError("nodes must be a nonempty subset of node001-node006")
    if not heldouts or any(name not in HELDOUTS for name in heldouts):
        raise ValueError("unknown heldout domain")

    local_project = Path(args.deploy) / "SC-OLH-KG"
    remote_project = REMOTE_ROOT / "SC-OLH-KG"
    try:
        relative_manifest = Path(args.manifest).relative_to(local_project)
        remote_manifest = remote_project / relative_manifest
    except ValueError:
        remote_manifest = Path(args.manifest)

    specs = []
    for heldout in heldouts:
        archive = (
            remote_project / "archives" / args.archive_run_id / heldout
            / f"heldout_{heldout}.json"
        )
        relative_output = (
            Path("archives") / args.output_source_run_id / heldout
            / "source_initial_designs.json"
        )
        local_output = local_project / relative_output
        remote_output = remote_project / relative_output
        command = [
            "test", "-f", str(archive),
            "&&",
            "env", "LC_ALL=C", "LANG=C", "SCOLHKG_OFFLINE=1",
            f"OMP_NUM_THREADS={int(args.cpu)}",
            f"MKL_NUM_THREADS={int(args.cpu)}",
            "PYTHONUNBUFFERED=1", "PYTHONDONTWRITEBYTECODE=1",
            str(args.python),
            "performance/materialize_source_initial_designs.py",
            "--manifest", str(remote_manifest),
            "--heldout", heldout,
            "--archive", str(archive),
            "--out", str(remote_output),
            "--source-d", str(args.source_d),
            "--d", str(args.d),
            "--n0", str(args.n0),
            "--seed-start", str(args.seed_start),
            "--n-seeds", str(args.n_seeds),
            "--structural-prior-profile", "low_frequency_only",
            "--proposal-mode", "risk_objective_atlas",
            "--source-design-mode", "universal_mixture",
        ]
        # Keep the shell conjunction outside shlex's quoting of argv tokens.
        command_text = (
            f"test -f {shlex.quote(str(archive))} && "
            + shlex.join(command[4:])
            + " && echo DONE"
        )
        specs.append({
            "description": f"V51 20-seed frozen design {heldout}",
            "cmd": command_text,
            "cwd": str(local_project),
            "signature": (
                f"KG_op/mean_v51_s20_design/{args.run_id}/{heldout}"
            ),
            "project": "KG-SYNTH",
            "vram": 0,
            "cpu": int(args.cpu),
            "ram_mb": int(args.ram_mb),
            "allowed_nodes": list(nodes),
            "result_dir": str(remote_output.parent),
            "local_result_dir": str(local_output.parent),
            "stage_excludes": ["checkpoints", "profiles", "results"],
            "allow_duplicate": True,
        })
    return specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", type=Path,
                        default=root_submit.DEFAULT_SCHEDULER)
    parser.add_argument("--deploy", type=Path, default=root_submit.DEFAULT_DEPLOY)
    parser.add_argument("--python", type=Path, default=root_submit.REMOTE_PYTHON)
    parser.add_argument("--manifest", type=Path, default=(
        root_submit.DEFAULT_DEPLOY
        / "SC-OLH-KG/performance/manifests/v18b_exactkg_mcdiag.json"))
    parser.add_argument("--run-id", default=(
        "scolh_mean_alignment_v51_s20_designs_"
        f"{time.strftime('%Y%m%d_%H%M%S')}"))
    parser.add_argument("--archive-run-id", default=DEFAULT_ARCHIVE_RUN_ID)
    parser.add_argument("--output-source-run-id",
                        default=DEFAULT_OUTPUT_SOURCE_RUN_ID)
    parser.add_argument("--heldouts", default=",".join(HELDOUTS))
    parser.add_argument("--nodes", default=",".join(CPU_NODES))
    parser.add_argument("--source-d", type=int, default=50)
    parser.add_argument("--d", type=int, default=1000)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--cpu", type=int, default=12)
    parser.add_argument("--ram-mb", type=int, default=8192)
    parser.add_argument("--no-sync", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    specs = build_specs(args)
    if args.dry_run:
        print(json.dumps(specs, indent=2))
        return
    if not args.no_sync:
        subprocess.run([str(root_submit.SYNC)], check=True, cwd=ROOT)
    payload = "\n".join(json.dumps(spec) for spec in specs) + "\n"
    subprocess.run(
        [
            sys.executable,
            str(args.scheduler),
            "submit-jsonl", "--stdin", "--trusted", "--json",
            "--intent-label", str(args.run_id),
        ],
        input=payload,
        text=True,
        check=True,
    )


if __name__ == "__main__":
    main()
