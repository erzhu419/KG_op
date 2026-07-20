#!/usr/bin/env python3
"""Submit the V49 decision-HVD and terminal-loss factorial sentinel."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v27 = _load(
    "mean_alignment_v27_submit",
    ROOT / "scripts/submit_scolhkg_mean_alignment_v27_sequential_gate_scheduler.py",
)


def _profile(backend, aleatoric, loss):
    value = dict(v27.VARIANTS["exchangeable_aggregate_none"])
    exact = str(backend) == "sobol_exact_joint_voi"
    value.update({
        "decision_backend": str(backend),
        "decision_aleatoric_mode": str(aleatoric),
        "decision_violation_loss_mode": str(loss),
        "posterior_dominance_enabled": False,
        "adaptive_replication_voi": bool(exact),
        "replication_candidate_count": 4 if exact else 0,
        "replication_max_per_solution": 5,
        "exact_terminal_mode": "bayes_risk",
        "exact_clip_negative": False,
        "terminal_frontier_candidate_count": 0,
    })
    if exact:
        value.update({
            "exact_mc_samples": 2,
            "exact_sampling_mode": "antithetic",
            "exact_jobs": 12,
            "parallel_backend": "process_fork",
        })
    return value


VARIANTS = {
    "sobol_upper_positive": _profile(
        "sobol_new", "certification_upper", "positive_part"),
    "sobol_central_positive": _profile(
        "sobol_new", "posterior_central", "positive_part"),
    "sobol_upper_probability": _profile(
        "sobol_new", "certification_upper", "failure_probability"),
    "sobol_central_probability": _profile(
        "sobol_new", "posterior_central", "failure_probability"),
    "exact_upper_positive": _profile(
        "sobol_exact_joint_voi", "certification_upper", "positive_part"),
    "exact_central_positive": _profile(
        "sobol_exact_joint_voi", "posterior_central", "positive_part"),
    "exact_upper_probability": _profile(
        "sobol_exact_joint_voi", "certification_upper", "failure_probability"),
    "exact_central_probability": _profile(
        "sobol_exact_joint_voi", "posterior_central", "failure_probability"),
}
SCENARIO_SEEDS = {
    "FactorShockStatePolicyRZDT1": 0,
    "InventorySupplyChain": 0,
    "QueueResourceControl": 3,
}
CPU_NODES = v27.CPU_NODES


def _root_module():
    return v27._root_module()


def build_specs(args):
    source_records = int(getattr(args, "source_records_per_domain", 64))
    args.variant_profiles = {
        name: {
            **profile,
            "source_records_per_domain": source_records,
        }
        for name, profile in VARIANTS.items()
    }
    args.scenarios = v27.MEAN_SCENARIOS
    args.stage_family = "mean_v49"
    args.gate_label = "Mean V49"
    args.sequential = bool(int(args.N) > int(args.n0))
    args.seed_start = 0
    args.n_seeds = max(SCENARIO_SEEDS.values()) + 1
    specs = _root_module().build_specs(args)
    return [
        spec for spec in specs
        if any(
            f"/{scenario}/seed{seed}" in str(spec.get("signature", ""))
            for scenario, seed in SCENARIO_SEEDS.items()
        )
    ]


def main():
    parser = argparse.ArgumentParser()
    defaults = _root_module()
    parser.add_argument("--scheduler", type=Path, default=defaults.DEFAULT_SCHEDULER)
    parser.add_argument("--deploy", type=Path, default=defaults.DEFAULT_DEPLOY)
    parser.add_argument("--python", type=Path, default=defaults.REMOTE_PYTHON)
    parser.add_argument("--manifest", type=Path, default=(
        defaults.DEFAULT_DEPLOY
        / "SC-OLH-KG/performance/manifests/v18b_exactkg_mcdiag.json"))
    parser.add_argument("--source-run-id", default=defaults.DEFAULT_SOURCE_RUN_ID)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--run-id", default=(
        "scolh_mean_alignment_v49_decision_loss_"
        f"{time.strftime('%Y%m%d_%H%M%S')}"))
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--nodes", default=",".join(CPU_NODES))
    parser.add_argument("--source-d", type=int, default=50)
    parser.add_argument("--source-records-per-domain", type=int, default=64)
    parser.add_argument("--d", type=int, default=1000)
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--pool-size", type=int, default=512)
    parser.add_argument("--variance-audit-size", type=int, default=512)
    parser.add_argument("--misspecification-prior-df", type=float, default=4.0)
    parser.add_argument("--misspecification-ridge", type=float, default=1.0)
    parser.add_argument("--misspecification-max-scale", type=float, default=100.0)
    parser.add_argument("--misspecification-delta", type=float, default=0.05)
    parser.add_argument("--contrast-scale", type=float, default=1.0)
    parser.add_argument("--null-geometry-ridge", type=float, default=1e-3)
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
        subprocess.run([str(defaults.SYNC)], check=True, cwd=ROOT)
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
