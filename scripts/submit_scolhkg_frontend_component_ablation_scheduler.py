#!/usr/bin/env python3
"""Submit the causal universal/source/combined proposal decomposition."""

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
REMOTE_PYTHON = Path(
    "/home/zhengliang01/scheduleurm_work/conda_envs/scomp-py310/bin/python")
CPU_NODES = tuple(f"node{i:03d}" for i in range(1, 7))
DOMAINS = (
    "FactorShockStatePolicyRZDT1",
    "InventorySupplyChain",
    "QueueResourceControl",
)
COMPONENTS = (
    "universal_only",
    "source_templates_only",
    "combined",
)


def _parse_csv(value):
    return tuple(
        item.strip() for item in str(value).split(",") if item.strip())


def _terminal_flags():
    return [
        "--terminal-verification-primary-budget", "80",
        "--terminal-verification-support-budget", "128",
        "--terminal-verification-candidate-budgets", "80,128,128",
        "--terminal-verification-delta", "0.05",
        "--terminal-verification-method", "normal_quantile_tolerance",
        "--terminal-verification-shortlist-mode",
        "posterior_objective_challenger_then_safe",
        "--terminal-verification-shortlist-size", "3",
        "--terminal-objective-challenger-max-violation-probability", "0.5",
        "--terminal-objective-incumbent-guard",
        "--terminal-objective-comparison-budget", "8",
        "--terminal-objective-comparison-delta", str(0.05 / 3.0),
        "--terminal-safe-interior-probability-slack", "0.05",
    ]


def build_specs(args):
    project = Path(args.deploy) / "SC-OLH-KG"
    manifest = (
        project / "performance/manifests/v18b_exactkg_mcdiag.json")
    specs = []
    heldouts = _parse_csv(args.heldouts)
    components = _parse_csv(args.components)
    unknown = sorted(set(components) - set(COMPONENTS))
    if unknown:
        raise ValueError(f"unknown proposal components: {unknown}")

    for component in components:
        for heldout in heldouts:
            archive = (
                project / "archives" / args.archive_run_id / heldout
                / f"heldout_{heldout}.json"
            )
            design = (
                project / "archives" / args.run_id / component / heldout
                / "source_initial_designs.json"
            )
            materialize = [
                "env", "LC_ALL=C", "LANG=C", "SCOLHKG_OFFLINE=1",
                "PYTHONUNBUFFERED=1", "PYTHONDONTWRITEBYTECODE=1",
                f"OMP_NUM_THREADS={int(args.design_cpu)}",
                f"MKL_NUM_THREADS={int(args.design_cpu)}",
                f"OPENBLAS_NUM_THREADS={int(args.design_cpu)}",
                str(REMOTE_PYTHON),
                "performance/materialize_source_initial_designs.py",
                "--manifest", str(manifest),
                "--heldout", heldout,
                "--archive", str(archive),
                "--out", str(design),
                "--d", str(args.d),
                "--source-d", str(args.source_d),
                "--n0", str(args.n0),
                "--seed-start", str(args.seed_start),
                "--n-seeds", str(args.n_seeds),
                "--structural-prior-profile", "low_frequency_only",
                "--proposal-mode", "risk_objective_atlas",
                "--proposal-component-mode", component,
                "--source-design-mode", "universal_mixture",
            ]
            wait_for = [str(manifest)]
            if component != "universal_only":
                wait_for.append(str(archive))
            specs.append({
                "description": (
                    f"frontend component design {component} {heldout}"),
                "cmd": f"{shlex.join(materialize)} && echo DONE",
                "cwd": str(project),
                "signature": (
                    f"KG_op/frontend_component/{args.run_id}/design/"
                    f"{component}/{heldout}"
                ),
                "project": "KG-SYNTH",
                "vram": 0,
                "cpu": (
                    1
                    if component == "universal_only"
                    else int(args.design_cpu)
                ),
                "ram_mb": int(args.design_ram_mb),
                "allowed_nodes": list(CPU_NODES),
                "wait_for_files": wait_for,
                "result_dir": str(design.parent),
                "local_result_dir": str(design.parent),
                "stage_excludes": ["checkpoints", "profiles", "results"],
                "allow_duplicate": True,
            })

            for seed in range(
                int(args.seed_start),
                int(args.seed_start) + int(args.n_seeds),
            ):
                result_dir = (
                    project / "profiles" / args.run_id / component / heldout
                    / f"seed{seed:04d}"
                )
                source_calls = (
                    0
                    if component == "universal_only"
                    else int(args.offline_source_calls)
                )
                command = [
                    "env", "LC_ALL=C", "LANG=C", "SCOLHKG_OFFLINE=1",
                    "PYTHONUNBUFFERED=1", "PYTHONDONTWRITEBYTECODE=1",
                    "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
                    "OPENBLAS_NUM_THREADS=1",
                    str(REMOTE_PYTHON),
                    "performance/benchmark_frozen_proposal_only.py",
                    "--heldout", heldout,
                    "--seed", str(seed),
                    "--initial-design", "source_informed",
                    "--initial-design-file", str(design),
                    "--out", str(result_dir / "result.json"),
                    "--source-d", str(args.source_d),
                    "--d", str(args.d),
                    "--n0", str(args.n0),
                    "--offline-source-calls", str(source_calls),
                    *_terminal_flags(),
                ]
                specs.append({
                    "description": (
                        f"frontend component {component} {heldout} "
                        f"seed={seed}"
                    ),
                    "cmd": f"{shlex.join(command)} && echo DONE",
                    "cwd": str(project),
                    "signature": (
                        f"KG_op/frontend_component/{args.run_id}/result/"
                        f"{component}/{heldout}/seed{seed:04d}"
                    ),
                    "project": "KG-SYNTH",
                    "vram": 0,
                    "cpu": 1,
                    "ram_mb": int(args.run_ram_mb),
                    "allowed_nodes": list(CPU_NODES),
                    "wait_for_files": [str(design)],
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
    parser.add_argument(
        "--run-id",
        default=(
            "paper_frontend_component_ablation_d50_d1000_n10_"
            f"s80_99_{time.strftime('%Y%m%d_%H%M%S')}"
        ),
    )
    parser.add_argument(
        "--archive-run-id",
        default="transfer_source_informed_official_n20_s20_20260716",
    )
    parser.add_argument("--heldouts", default=",".join(DOMAINS))
    parser.add_argument("--components", default=",".join(COMPONENTS))
    parser.add_argument("--seed-start", type=int, default=80)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--source-d", type=int, default=50)
    parser.add_argument("--d", type=int, default=1000)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--offline-source-calls", type=int, default=384)
    parser.add_argument("--design-cpu", type=int, default=12)
    parser.add_argument("--design-ram-mb", type=int, default=16384)
    parser.add_argument("--run-ram-mb", type=int, default=4096)
    parser.add_argument(
        "--sync-remote",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    specs = build_specs(args)
    expected = (
        len(_parse_csv(args.components))
        * len(_parse_csv(args.heldouts))
        * (int(args.n_seeds) + 1)
    )
    if len(specs) != expected:
        raise RuntimeError(f"built {len(specs)} tasks, expected {expected}")
    if len({spec["signature"] for spec in specs}) != len(specs):
        raise RuntimeError("frontend-component task signatures are not unique")
    if args.dry_run:
        print(json.dumps({
            "run_id": args.run_id,
            "task_count": len(specs),
            "specs": specs,
        }, indent=2))
        return
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
            f"paper-frontend-component-{args.run_id}",
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
        "run_id": str(args.run_id),
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "task_count": int(len(task_ids)),
        "task_ids": task_ids,
        "contract": {
            "components": list(_parse_csv(args.components)),
            "heldout_domains": list(_parse_csv(args.heldouts)),
            "source_dimension": int(args.source_d),
            "target_dimension": int(args.d),
            "target_initial_calls": int(args.n0),
            "source_calls_by_component": {
                "universal_only": 0,
                "source_templates_only": int(args.offline_source_calls),
                "combined": int(args.offline_source_calls),
            },
            "target_oracle_used": False,
            "verifier": "v69_independent_three_policy_objective_guard",
            "allowed_nodes": list(CPU_NODES),
        },
        "checkpoint_or_model_artifacts_transferred": False,
    }
    manifest_path = (
        Path(args.deploy) / "SC-OLH-KG" / "profiles" / args.run_id
        / "submission_manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(registration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.dispatch and task_ids:
        command = [sys.executable, str(args.scheduler), "dispatch"]
        for task_id in task_ids:
            command.extend(["--task-id", task_id])
        subprocess.run(command, check=True)
    print(json.dumps(registration, indent=2))


if __name__ == "__main__":
    main()
