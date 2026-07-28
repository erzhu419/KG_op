#!/usr/bin/env python3
"""Submit matched high-D transfer gates for SC-OLH-KG and official baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
TRANSFER_SUBMIT = ROOT / "scripts/submit_scolhkg_transfer_fairness_scheduler.py"
SC_SUBMIT = ROOT / "scripts/submit_scolhkg_source_informed_matched_scheduler.py"
SYNC = ROOT / "scripts/sync_scolhkg_scheduler_deploy.sh"
SYNC_EXTERNAL = ROOT / "scripts/sync_scolhkg_transfer_repos.sh"
DEFAULT_SCHEDULER = Path.home() / "mine_code/scheduleurm/skill/scheduler.py"
DEFAULT_DEPLOY = Path.home() / "mine_code/KG_op_scheduler_deploy"
CPU_NODES = tuple(f"node{index:03d}" for index in range(1, 7))


def parse_int_csv(value):
    return tuple(int(item.strip()) for item in str(value).split(",") if item.strip())


def run(command, *, dry_run=False):
    print("+", " ".join(map(str, command)), flush=True)
    if not dry_run:
        subprocess.run(list(map(str, command)), check=True, cwd=ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--deploy", type=Path, default=DEFAULT_DEPLOY)
    parser.add_argument(
        "--manifest", type=Path,
        default=(
            DEFAULT_DEPLOY
            / "SC-OLH-KG/performance/manifests/v18b_exactkg_mcdiag.json"
        ),
    )
    parser.add_argument(
        "--run-prefix",
        default=f"transfer_ratio_gate_{time.strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--dims", default="200,1000")
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--methods", default="")
    parser.add_argument("--nodes", default=",".join(CPU_NODES))
    parser.add_argument("--cpu", type=int, default=12)
    parser.add_argument("--ram-mb", type=int, default=32768)
    parser.add_argument("--exact-jobs", type=int, default=12)
    parser.add_argument("--no-sync", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dimensions = parse_int_csv(args.dims)
    if not dimensions or any(dimension < 1 for dimension in dimensions):
        raise ValueError("dims must contain positive integers")
    if not (1 <= int(args.n0) <= int(args.N)):
        raise ValueError("ratio gate requires 1 <= n0 <= N")

    if not args.no_sync:
        run([SYNC], dry_run=args.dry_run)
        run([SYNC_EXTERNAL], dry_run=args.dry_run)

    submitted_runs = []
    for dimension in dimensions:
        stem = f"{args.run_prefix}_d{dimension}_n{int(args.N)}"
        source_transfer = f"{stem}_transfer_source"
        common_transfer = f"{stem}_transfer_common"
        source_sc = f"{stem}_sc_source"
        common_sc = f"{stem}_sc_common"

        for regime, run_id in (
            ("source_informed", source_transfer),
            ("common_sobol", common_transfer),
        ):
            command = [
                sys.executable, TRANSFER_SUBMIT,
                "--scheduler", args.scheduler,
                "--deploy", args.deploy,
                "--manifest", args.manifest,
                "--run-id", run_id,
                "--implementation", "official",
                "--initial-design", regime,
                "--d", str(dimension),
                "--N", str(args.N),
                "--n0", str(args.n0),
                "--seed-start", str(args.seed_start),
                "--n-seeds", str(args.n_seeds),
                "--nodes", args.nodes,
                "--cpu", str(args.cpu),
                "--ram-mb", str(args.ram_mb),
                "--no-sync-remote",
            ]
            if args.methods:
                command.extend(["--methods", args.methods])
            if args.dry_run:
                command.append("--dry-run")
            run(command, dry_run=False)

        for regime, run_id in (
            ("source_informed", source_sc),
            ("common_sobol", common_sc),
        ):
            command = [
                sys.executable, SC_SUBMIT,
                "--scheduler", args.scheduler,
                "--deploy", args.deploy,
                "--manifest", args.manifest,
                "--source-run-id", source_transfer,
                "--run-id", run_id,
                "--initial-design", regime,
                "--d", str(dimension),
                "--N", str(args.N),
                "--n0", str(args.n0),
                "--seed-start", str(args.seed_start),
                "--n-seeds", str(args.n_seeds),
                "--nodes", args.nodes,
                "--cpu", str(args.cpu),
                "--exact-jobs", str(args.exact_jobs),
                "--ram-mb", str(args.ram_mb),
                "--no-sync",
            ]
            if args.dry_run:
                command.append("--dry-run")
            run(command, dry_run=False)

        submitted_runs.append({
            "dimension": int(dimension),
            "N": int(args.N),
            "n0": int(args.n0),
            "D_over_N": float(dimension) / float(args.N),
            "transfer_source": source_transfer,
            "transfer_common": common_transfer,
            "sc_source": source_sc,
            "sc_common": common_sc,
        })

    if args.dispatch and not args.dry_run:
        run([sys.executable, args.scheduler, "dispatch"])

    payload = {
        "schema_version": 1,
        "run_prefix": args.run_prefix,
        "source_domains_per_target": 2,
        "source_profiles_per_domain": 64,
        "source_replicates": 3,
        "source_simulator_calls_per_target": 384,
        "target_seeds": int(args.n_seeds),
        "initial_design_regimes": ["common_sobol", "source_informed"],
        "runs": submitted_runs,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
