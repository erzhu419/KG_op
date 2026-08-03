#!/usr/bin/env python3
"""Submit the provider-cumulative HVD sequential gate to CPU nodes only."""

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
BOTORCH_OVERLAY = Path(
    "/home/zhengliang01/scheduleurm_work/python_pkgs/botorch_overlay_py310")
CPU_NODES = tuple(f"node{i:03d}" for i in range(1, 7))
DOMAINS = (
    "FactorShockStatePolicyRZDT1",
    "InventorySupplyChain",
    "QueueResourceControl",
)
MODES = ("pooled", "provider_cumulative_factor")


def parse_csv(value):
    return tuple(
        item.strip() for item in str(value).split(",") if item.strip())


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def terminal_flags():
    return [
        "--terminal-verification",
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


def validate_contract(args):
    deploy_project = Path(args.deploy) / "SC-OLH-KG"
    required_seeds = set(range(
        int(args.seed_start),
        int(args.seed_start) + int(args.n_seeds),
    ))
    domains = {}
    for heldout in parse_csv(args.heldouts):
        archive_path = (
            deploy_project / "archives" / args.archive_run_id / heldout
            / f"heldout_{heldout}.json"
        )
        design_path = (
            deploy_project / "archives" / args.design_run_id / heldout
            / "source_initial_designs.json"
        )
        archive = read_json(archive_path)
        design = read_json(design_path)
        if archive["fingerprint"] != design["source_archive_fingerprint"]:
            raise ValueError(f"{heldout} archive/design fingerprint mismatch")
        if int(design["source_dimension"]) != int(args.source_d):
            raise ValueError(f"{heldout} source dimension changed")
        if int(design["dimension"]) != int(args.d):
            raise ValueError(f"{heldout} target dimension changed")
        if int(design["n0"]) != int(args.n0):
            raise ValueError(f"{heldout} n0 changed")
        available = {int(seed) for seed in design["designs"]}
        if not required_seeds.issubset(available):
            raise ValueError(f"{heldout} frozen design lacks requested seeds")
        source_calls = sum(
            sum(len(row) for row in task["Y_replicates"])
            for task in archive["tasks"]
        )
        if int(source_calls) != int(args.offline_source_calls):
            raise ValueError(f"{heldout} source cost changed")
        domains[heldout] = {
            "source_archive_fingerprint": archive["fingerprint"],
            "initial_design_fingerprint": design.get("fingerprint"),
            "source_calls": int(source_calls),
            "source_dimension": int(design["source_dimension"]),
            "target_dimension": int(design["dimension"]),
            "n0": int(design["n0"]),
        }
    return domains


def build_specs(args):
    deploy_project = Path(args.deploy) / "SC-OLH-KG"
    manifest = deploy_project / "performance/manifests/v18b_exactkg_mcdiag.json"
    modes = parse_csv(args.modes)
    if not set(modes).issubset(set(MODES)):
        raise ValueError(f"modes must be drawn from {MODES}")
    specs = []
    for mode in modes:
        for heldout in parse_csv(args.heldouts):
            archive = (
                deploy_project / "archives" / args.archive_run_id / heldout
                / f"heldout_{heldout}.json"
            )
            design = (
                deploy_project / "archives" / args.design_run_id / heldout
                / "source_initial_designs.json"
            )
            for seed in range(
                int(args.seed_start),
                int(args.seed_start) + int(args.n_seeds),
            ):
                result_dir = (
                    deploy_project / "profiles" / args.run_id / mode
                    / heldout / f"seed{seed:04d}"
                )
                checkpoint_dir = (
                    deploy_project / "checkpoints" / args.run_id / mode
                    / heldout / f"seed{seed:04d}"
                )
                command = [
                    "env", "LC_ALL=C", "LANG=C", "SCOLHKG_OFFLINE=1",
                    "PYTHONUNBUFFERED=1", "PYTHONDONTWRITEBYTECODE=1",
                    f"OMP_NUM_THREADS={int(args.cpu)}",
                    f"MKL_NUM_THREADS={int(args.cpu)}",
                    f"OPENBLAS_NUM_THREADS={int(args.cpu)}",
                    "SCOLHKG_TORCH_DETERMINISTIC=1",
                    f"PYTHONPATH={BOTORCH_OVERLAY}",
                    str(REMOTE_PYTHON),
                    "performance/benchmark_sota_fairness.py",
                    "--protocol", "shared_archive_hvd_n13",
                    "--method", str(args.backend),
                    "--heldout", heldout,
                    "--seed", str(seed),
                    "--manifest", str(manifest),
                    "--out", str(result_dir / "result.json"),
                    "--checkpoint-dir", str(checkpoint_dir),
                    "--initial-design-file", str(design),
                    "--source-archive-file", str(archive),
                    "--aleatoric-head-mode", mode,
                    "--source-hvd-calibration-delta", "0.05",
                    "--source-hvd-calibration-quantile", "0.95",
                    "--aleatoric-audit-size", str(args.audit_size),
                    "--target-budget", str(args.N),
                    "--d", str(args.d),
                    "--n0", str(args.n0),
                    "--raw-samples", str(args.raw_samples),
                    "--num-restarts", str(args.num_restarts),
                    "--maxiter", str(args.maxiter),
                    "--ts-candidates", str(args.ts_candidates),
                    "--candidate-timeout-sec", str(args.candidate_timeout_sec),
                    "--torch-device", "cpu",
                    "--torch-deterministic",
                    *terminal_flags(),
                ]
                specs.append({
                    "description": (
                        f"provider HVD CPU gate {args.backend} {mode} "
                        f"{heldout} seed={seed}"
                    ),
                    "cmd": f"{shlex.join(command)} && echo DONE",
                    "cwd": str(deploy_project),
                    "signature": (
                        f"KG_op/provider_hvd_cpu_gate/{args.run_id}/"
                        f"{args.backend}/{mode}/{heldout}/seed{seed:04d}"
                    ),
                    "project": "KG-SYNTH",
                    "vram": 0,
                    "cpu": int(args.cpu),
                    "ram_mb": int(args.ram_mb),
                    "allowed_nodes": list(CPU_NODES),
                    "allow_cpu_training": True,
                    "cpu_training_justification": (
                        "Non-SAAS provider-HVD causal gate; CPU fleet is the "
                        "registered execution route and GPUs are excluded."
                    ),
                    "result_dir": str(result_dir),
                    "local_result_dir": str(result_dir),
                    "wait_for_files": [str(archive), str(design)],
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
            "provider_hvd_scbo_cpu_gate_d50_d1000_n13_s80_84_"
            f"{time.strftime('%Y%m%d_%H%M%S')}"),
    )
    parser.add_argument(
        "--archive-run-id",
        default="transfer_source_informed_official_n20_s20_20260716",
    )
    parser.add_argument(
        "--design-run-id",
        default="paper_certified_scolh_v64_cross_d50_d1000_s80_99_v1",
    )
    parser.add_argument("--heldouts", default=",".join(DOMAINS))
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument(
        "--backend",
        choices=("botorch_scbo", "botorch_turbo"),
        default="botorch_scbo",
    )
    parser.add_argument("--seed-start", type=int, default=80)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--source-d", type=int, default=50)
    parser.add_argument("--d", type=int, default=1000)
    parser.add_argument("--N", type=int, default=13)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--offline-source-calls", type=int, default=384)
    parser.add_argument("--audit-size", type=int, default=256)
    parser.add_argument("--raw-samples", type=int, default=1024)
    parser.add_argument("--num-restarts", type=int, default=10)
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--ts-candidates", type=int, default=2000)
    parser.add_argument("--candidate-timeout-sec", type=float, default=3600.0)
    parser.add_argument("--cpu", type=int, default=12)
    parser.add_argument("--ram-mb", type=int, default=24576)
    parser.add_argument(
        "--sync-remote",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    audit = validate_contract(args)
    specs = build_specs(args)
    expected = (
        len(parse_csv(args.heldouts))
        * len(parse_csv(args.modes))
        * int(args.n_seeds)
    )
    if len(specs) != expected:
        raise RuntimeError(f"built {len(specs)} tasks, expected {expected}")
    if len({spec["signature"] for spec in specs}) != len(specs):
        raise RuntimeError("provider HVD gate signatures are not unique")
    if args.dry_run:
        print(json.dumps({
            "task_count": len(specs),
            "contract_audit": audit,
            "specs": specs,
        }, indent=2, sort_keys=True))
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
            f"provider-hvd-cpu-gate-{args.run_id}",
        ],
        input=json.dumps(specs),
        text=True,
    )
    response = json.loads(output)
    task_ids = [
        row["id"] for row in response.get("submitted", []) if row.get("id")
    ]
    registration = {
        "schema_version": 1,
        "run_id": str(args.run_id),
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "task_count": int(len(specs)),
        "task_ids": task_ids,
        "contract_audit": audit,
        "contract": {
            "same_frozen_proposal": True,
            "same_source_archive": True,
            "same_cpu_backend": str(args.backend),
            "same_v69_independent_verifier": True,
            "only_changed_object": "source_aleatoric_head",
            "modes": list(parse_csv(args.modes)),
            "saas_used": False,
            "gpu_used": False,
            "allowed_nodes": list(CPU_NODES),
            "target_oracle_used_during_search": False,
            "post_run_variance_audit_used_for_selection": False,
            "checkpoint_results_synced_locally": False,
        },
    }
    registration_path = (
        Path(args.deploy) / "SC-OLH-KG" / "profiles" / args.run_id
        / "submission_manifest.json"
    )
    registration_path.parent.mkdir(parents=True, exist_ok=True)
    registration_path.write_text(
        json.dumps(registration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.dispatch and task_ids:
        subprocess.run(
            [
                sys.executable,
                str(args.scheduler),
                "dispatch",
                *sum(([
                    "--task-id", task_id
                ] for task_id in task_ids), []),
            ],
            check=True,
        )
    print(json.dumps(registration, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
