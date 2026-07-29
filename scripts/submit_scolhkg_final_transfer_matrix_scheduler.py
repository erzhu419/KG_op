#!/usr/bin/env python3
"""Submit all official transfer baselines under the final frozen front end."""

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
REMOTE_ROOT = Path(
    "/home/zhengliang01/scheduleurm_work/KG_op_scheduler_deploy")
EXTERNAL_REPOS = REMOTE_ROOT / "external_repos"
TRANSFER_TORCH_OVERLAY = Path(
    "/home/zhengliang01/scheduleurm_work/python_pkgs/transfer_torch_py310")
TRANSFERGPBO_OVERLAY = Path(
    "/home/zhengliang01/scheduleurm_work/python_pkgs/transfergpbo_py310")
CPU_NODES = tuple(f"node{i:03d}" for i in range(1, 7))
DOMAINS = (
    "FactorShockStatePolicyRZDT1",
    "InventorySupplyChain",
    "QueueResourceControl",
)
METHODS = (
    "safe_fpacoh_cbo",
    "rgpe_cbo",
    "stacked_transfer_gp_cbo",
    "mtgp_cbo",
    "fsbo_cbo",
    "hyperbo_cbo",
    "metabo_cbo",
    "malibo_cbo",
)


def _parse_csv(value):
    return tuple(
        item.strip() for item in str(value).split(",") if item.strip())


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _terminal_flags():
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
    seeds = set(range(
        int(args.seed_start),
        int(args.seed_start) + int(args.n_seeds),
    ))
    audit = {"domains": {}}
    for heldout in _parse_csv(args.heldouts):
        archive_path = (
            deploy_project / "archives" / args.archive_run_id / heldout
            / f"heldout_{heldout}.json"
        )
        design_path = (
            deploy_project / "archives" / args.design_run_id / heldout
            / "source_initial_designs.json"
        )
        archive = _read_json(archive_path)
        design = _read_json(design_path)
        if archive["fingerprint"] != design["source_archive_fingerprint"]:
            raise ValueError(f"{heldout} archive/design fingerprint mismatch")
        dimensions = {
            len(task["X"][0]) for task in archive["tasks"]
        }
        if dimensions != {int(args.source_d)}:
            raise ValueError(f"{heldout} archive source dimension changed")
        if int(design["source_dimension"]) != int(args.source_d):
            raise ValueError(f"{heldout} proposal source dimension changed")
        if int(design["dimension"]) != int(args.d):
            raise ValueError(f"{heldout} proposal target dimension changed")
        if str(design["proposal_mode"]) != "risk_objective_atlas":
            raise ValueError(f"{heldout} is not the final proposal")
        if int(design["n0"]) != int(args.n0):
            raise ValueError(f"{heldout} n0 changed")
        available = {int(seed) for seed in design["designs"]}
        if not seeds.issubset(available):
            raise ValueError(f"{heldout} proposal lacks required seeds")
        calls = sum(
            sum(len(row) for row in task["Y_replicates"])
            for task in archive["tasks"]
        )
        if int(calls) != int(args.offline_source_calls):
            raise ValueError(f"{heldout} source budget changed")
        audit["domains"][heldout] = {
            "archive_fingerprint": str(archive["fingerprint"]),
            "proposal_mode": str(design["proposal_mode"]),
            "source_dimension": int(args.source_d),
            "target_dimension": int(args.d),
            "source_calls": int(calls),
            "seed_count": int(len(seeds)),
        }
    return audit


def build_specs(args):
    deploy_project = Path(args.deploy) / "SC-OLH-KG"
    specs = []
    methods = _parse_csv(args.methods)
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"unknown transfer methods: {unknown}")
    for heldout in _parse_csv(args.heldouts):
        archive = (
            deploy_project / "archives" / args.archive_run_id / heldout
            / f"heldout_{heldout}.json"
        )
        design = (
            deploy_project / "archives" / args.design_run_id / heldout
            / "source_initial_designs.json"
        )
        for method in methods:
            for seed in range(
                int(args.seed_start),
                int(args.seed_start) + int(args.n_seeds),
            ):
                result_dir = (
                    deploy_project / "profiles" / args.run_id / "official"
                    / heldout / method / f"seed{seed:04d}"
                )
                checkpoint_dir = (
                    deploy_project / "checkpoints" / args.run_id / "official"
                    / heldout / method / f"seed{seed:04d}"
                )
                command = [
                    "env", "LC_ALL=C", "LANG=C", "SCOLHKG_OFFLINE=1",
                    "PYTHONUNBUFFERED=1", "PYTHONDONTWRITEBYTECODE=1",
                    f"OMP_NUM_THREADS={int(args.cpu)}",
                    f"MKL_NUM_THREADS={int(args.cpu)}",
                    f"OPENBLAS_NUM_THREADS={int(args.cpu)}",
                    f"PYTHONPATH={TRANSFER_TORCH_OVERLAY}:"
                    f"{TRANSFERGPBO_OVERLAY}",
                    f"SCOLHKG_EXTERNAL_REPO_ROOT={EXTERNAL_REPOS}",
                    f"SCOLHKG_TRANSFERGPBO_OVERLAY={TRANSFERGPBO_OVERLAY}",
                    "SCOLHKG_HYPERBO_OVERLAY="
                    "/home/zhengliang01/scheduleurm_work/"
                    "python_pkgs/hyperbo_py310",
                    str(REMOTE_PYTHON),
                    "performance/benchmark_transfer_fairness.py",
                    "--method", method,
                    "--implementation", "official",
                    "--heldout", heldout,
                    "--archive", str(archive),
                    "--initial-design", "source_informed",
                    "--initial-design-file", str(design),
                    "--out", str(result_dir / "result.json"),
                    "--checkpoint-dir", str(checkpoint_dir),
                    "--seed", str(seed),
                    "--d", str(args.d),
                    "--N", str(args.N),
                    "--n0", str(args.n0),
                    "--source-dimension-adapter",
                    "ordered_dct_quadratic",
                    "--source-coordinate-max-frequency", "8",
                    "--source-coordinate-frequency-penalty", "0.10",
                    "--source-train-steps", str(args.source_train_steps),
                    "--target-finetune-steps", str(
                        args.target_finetune_steps),
                    *_terminal_flags(),
                ]
                specs.append({
                    "description": (
                        f"final transfer matrix {method} {heldout} "
                        f"seed={seed}"
                    ),
                    "cmd": f"{shlex.join(command)} && echo DONE",
                    "cwd": str(deploy_project),
                    "signature": (
                        f"KG_op/final_transfer_matrix/{args.run_id}/"
                        f"{heldout}/{method}/seed{seed:04d}"
                    ),
                    "project": "KG-SYNTH",
                    "vram": 0,
                    "cpu": int(args.cpu),
                    "ram_mb": int(args.ram_mb),
                    "allowed_nodes": list(CPU_NODES),
                    "wait_for_files": [str(archive), str(design)],
                    "result_dir": str(result_dir),
                    "local_result_dir": str(result_dir),
                    "stage_excludes": [
                        "checkpoints", "profiles", "results"],
                    "allow_duplicate": True,
                })
    if bool(getattr(args, "skip_existing_success", False)):
        filtered = []
        for spec in specs:
            result_path = Path(spec["local_result_dir"]) / "result.json"
            if result_path.is_file():
                try:
                    if _read_json(result_path).get("status") == "ok":
                        continue
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
            filtered.append(spec)
        specs = filtered
    return specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--deploy", type=Path, default=DEFAULT_DEPLOY)
    parser.add_argument("--run-id", default=(
        "paper_final_transfer_risk_atlas_v69_d50_d1000_n13_s80_99_"
        f"{time.strftime('%Y%m%d_%H%M%S')}"))
    parser.add_argument(
        "--archive-run-id",
        default="transfer_source_informed_official_n20_s20_20260716",
    )
    parser.add_argument(
        "--design-run-id",
        default="paper_certified_scolh_v64_cross_d50_d1000_s80_99_v1",
    )
    parser.add_argument("--heldouts", default=",".join(DOMAINS))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--seed-start", type=int, default=80)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--source-d", type=int, default=50)
    parser.add_argument("--d", type=int, default=1000)
    parser.add_argument("--N", type=int, default=13)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--offline-source-calls", type=int, default=384)
    parser.add_argument("--source-train-steps", type=int, default=2048)
    parser.add_argument("--target-finetune-steps", type=int, default=100)
    parser.add_argument("--cpu", type=int, default=12)
    parser.add_argument("--ram-mb", type=int, default=16384)
    parser.add_argument(
        "--sync-remote",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--skip-existing-success",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    audit = validate_contract(args)
    specs = build_specs(args)
    expected = (
        len(_parse_csv(args.heldouts))
        * len(_parse_csv(args.methods))
        * int(args.n_seeds)
    )
    if not args.skip_existing_success and len(specs) != expected:
        raise RuntimeError(f"built {len(specs)} tasks, expected {expected}")
    if args.skip_existing_success and len(specs) > expected:
        raise RuntimeError("recovery task count exceeds the matrix")
    signatures = [spec["signature"] for spec in specs]
    if len(signatures) != len(set(signatures)):
        raise RuntimeError("final transfer signatures are not unique")
    if args.dry_run:
        print(json.dumps({
            "audit": audit,
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
            f"final-transfer-matrix-{args.run_id}",
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
        "task_count": int(len(specs)),
        "task_ids": task_ids,
        "audit": audit,
        "contract": {
            "source_calls": int(args.offline_source_calls),
            "source_dimension": int(args.source_d),
            "target_dimension": int(args.d),
            "target_search_calls": int(args.N),
            "n0": int(args.n0),
            "proposal": "frozen_risk_objective_atlas",
            "verifier": "v69_independent_three_policy_objective_guard",
            "target_oracle_used": False,
            "allowed_nodes": list(CPU_NODES),
        },
        "checkpoint_results_synced_locally": False,
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
                *sum((["--task-id", task_id] for task_id in task_ids), []),
            ],
            check=True,
        )
    print(json.dumps(registration, indent=2))


if __name__ == "__main__":
    main()

