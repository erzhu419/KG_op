#!/usr/bin/env python3
"""Submit the 20-seed prior and decision-contract ablation matrix."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SUBMIT = ROOT / "scripts/submit_scolhkg_manifest_gate_scheduler.py"
SYNC = ROOT / "scripts/sync_scolhkg_scheduler_deploy.sh"
DEFAULT_SCHEDULER = Path.home() / "mine_code/scheduleurm/skill/scheduler.py"
DEFAULT_DEPLOY = Path.home() / "mine_code/KG_op_scheduler_deploy"


VARIANTS = {
    "promoted_legacy": {},
    "closed_terminal_kg": {
        "decision_contract": "certified_lexicographic",
        "finalist_policy": "terminal_kg_1step",
        "empirical_override": "off",
        "terminal_value": "certified_lexicographic",
    },
    "closed_all_exact": {
        "decision_contract": "certified_lexicographic",
        "finalist_budget": 0,
        "finalist_policy": "terminal_kg_1step",
        "empirical_override": "off",
        "terminal_value": "certified_lexicographic",
    },
    "drop_low_frequency": {
        "decision_contract": "certified_lexicographic",
        "finalist_policy": "terminal_kg_1step",
        "empirical_override": "off",
        "terminal_value": "certified_lexicographic",
        # d=50 has DCT frequencies 1,...,49.  Keep the same active rank and
        # remove only the source preference for low-frequency coordinates.
        "max_frequency": 49,
        "frequency_penalty": 0.0,
    },
    "drop_orthogonality": {
        "decision_contract": "certified_lexicographic",
        "finalist_policy": "terminal_kg_1step",
        "empirical_override": "off",
        "terminal_value": "certified_lexicographic",
        "orthogonal": False,
        "spectral_orthogonalization": "none",
    },
    "drop_coefficient_sparsity": {
        "decision_contract": "certified_lexicographic",
        "finalist_policy": "terminal_kg_1step",
        "empirical_override": "off",
        "terminal_value": "certified_lexicographic",
        "adaptive_sparsity": False,
        "group_ridge": False,
    },
    "drop_additivity": {
        "decision_contract": "certified_lexicographic",
        "finalist_policy": "terminal_kg_1step",
        "empirical_override": "off",
        "terminal_value": "certified_lexicographic",
        "basis_mode": "full_quadratic",
    },
    "drop_all_four": {
        "decision_contract": "certified_lexicographic",
        "finalist_policy": "terminal_kg_1step",
        "empirical_override": "off",
        "terminal_value": "certified_lexicographic",
        "max_frequency": 49,
        "frequency_penalty": 0.0,
        "orthogonal": False,
        "spectral_orthogonalization": "none",
        "adaptive_sparsity": False,
        "group_ridge": False,
        "basis_mode": "full_quadratic",
    },
}


def _boolean_flag(name, enabled):
    return f"--{name}" if enabled else f"--no-{name}"


def command(args, run_id, name, overrides):
    values = {
        "decision_contract": "legacy",
        "finalist_budget": 3,
        "finalist_policy": "commit_before_switch",
        "empirical_override": "legacy",
        "terminal_value": "model_default",
        "max_frequency": 8,
        "frequency_penalty": 0.10,
        "orthogonal": True,
        "spectral_orthogonalization": "symmetric",
        "adaptive_sparsity": True,
        "group_ridge": True,
        "basis_mode": "diagonal_quadratic",
    }
    values.update(overrides)
    return [
        sys.executable,
        str(SUBMIT),
        "--scheduler", str(args.scheduler),
        "--deploy", str(args.deploy),
        "--manifest", str(args.manifest),
        "--run-id", f"{run_id}_{name}",
        "--experiment-variant", name,
        "--heldouts", args.heldouts,
        "--line", "lodo",
        "--seed-start", str(args.seed_start),
        "--n-seeds", str(args.n_seeds),
        "--nodes", args.nodes,
        "--cpu", str(args.cpu),
        "--ram-mb", str(args.ram_mb),
        "--ordered-cumulative-exposure",
        "--ordered-max-frequency", str(values["max_frequency"]),
        "--spectral-orthogonalization",
        values["spectral_orthogonalization"],
        "--ordered-active-dim", "2",
        "--ordered-frequency-penalty", str(values["frequency_penalty"]),
        "--ordered-basis-mode", values["basis_mode"],
        _boolean_flag("ordered-orthogonal-coordinates", values["orthogonal"]),
        _boolean_flag("ordered-adaptive-sparsity", values["adaptive_sparsity"]),
        "--ordered-replace-local-kernel",
        "--ordered-latent-structure-selection",
        "--no-ordered-group-shared-shrinkage",
        _boolean_flag("ordered-group-ridge-learning", values["group_ridge"]),
        "--task-posterior-safe-generalized",
        "--task-posterior-mandatory-universal-count", "10",
        "--task-latent-inference-mode", "shadow",
        "--task-latent-calibration-mode", "source_profiles",
        "--exact-sampling-mode", "iid",
        "--exact-terminal-mode", "hard_certified",
        "--exact-mc-samples", "2",
        "--exact-jobs", str(args.exact_jobs),
        "--parallel-backend", "process_fork",
        "--source-observation-mode", "replicated",
        "--source-observation-replicates", "3",
        "--source-design-mode", "universal_mixture",
        "--source-universal-fraction", "1.0",
        "--source-consensus-template-count", "12",
        "--finalist-replication-budget", str(values["finalist_budget"]),
        "--finalist-replication-count", "3",
        "--finalist-observed-safety-count", "2",
        "--finalist-replication-min-replicates", "2",
        "--finalist-replication-delta", "0.05",
        "--finalist-replication-variance-prior-df", "2.0",
        "--finalist-replication-expert-stratified",
        "--finalist-replication-adaptive-race",
        "--finalist-replication-fixed-universe",
        "--finalist-replication-policy", values["finalist_policy"],
        "--finalist-frontier-policy", "observed_safety_reserved",
        "--finalist-empirical-override", values["empirical_override"],
        "--finalist-terminal-value-mode", values["terminal_value"],
        "--decision-contract-mode", values["decision_contract"],
        "--allow-duplicate",
        "--no-sync-remote",
    ] + (["--dry-run"] if args.dry_run else [])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--deploy", type=Path, default=DEFAULT_DEPLOY)
    parser.add_argument(
        "--manifest", type=Path,
        default=DEFAULT_DEPLOY /
        "SC-OLH-KG/performance/manifests/v18b_exactkg_mcdiag.json",
    )
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--heldouts",
        default=(
            "FactorShockStatePolicyRZDT1,InventorySupplyChain,"
            "QueueResourceControl"
        ),
    )
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument(
        "--nodes", default=",".join(f"node{i:03d}" for i in range(1, 7)))
    parser.add_argument("--cpu", type=int, default=12)
    parser.add_argument("--ram-mb", type=int, default=32768)
    parser.add_argument("--exact-jobs", type=int, default=12)
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    names = [name.strip() for name in args.variants.split(",") if name.strip()]
    unknown = sorted(set(names) - set(VARIANTS))
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")
    run_id = args.run_id or f"prior_closure_{time.strftime('%Y%m%d_%H%M%S')}"
    if not args.dry_run:
        subprocess.check_call([str(SYNC)])
    for name in names:
        cmd = command(args, run_id, name, VARIANTS[name])
        print("+", " ".join(map(str, cmd)), flush=True)
        subprocess.check_call(cmd)
    if args.dispatch and not args.dry_run:
        subprocess.check_call([
            sys.executable, str(args.scheduler), "dispatch",
        ])
    print({
        "run_id": run_id,
        "variants": names,
        "task_count": len(names) * int(args.n_seeds) *
        len([x for x in args.heldouts.split(",") if x.strip()]),
    })


if __name__ == "__main__":
    main()
