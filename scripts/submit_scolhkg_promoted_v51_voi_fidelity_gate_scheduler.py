#!/usr/bin/env python3
"""Submit a common-state MC/shortlist fidelity audit for promoted V51."""

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


closure = _load(
    "promoted_v51_closure_submit",
    ROOT / "scripts/submit_scolhkg_promoted_v51_closure_gate_scheduler.py",
)
PROFILE = dict(closure.PROFILE)
CPU_NODES = closure.CPU_NODES
DEFAULT_SOURCE_RUN_ID = closure.DEFAULT_SOURCE_RUN_ID
DEFAULT_MC_SAMPLES = (2, 8, 32)
DEFAULT_SHORTLIST_SIZES = (1, 4, 8, 32)


def _parse_int_csv(value):
    values = tuple(
        int(item.strip()) for item in str(value).split(",") if item.strip()
    )
    if not values or any(value <= 0 for value in values):
        raise ValueError("integer lists must contain positive values")
    return values


def _variant_profiles(mc_samples, shortlist_sizes, source_records):
    profiles = {}
    for mc in mc_samples:
        for shortlist in shortlist_sizes:
            profiles[f"mc{mc}_k{shortlist}"] = {
                **PROFILE,
                "source_records_per_domain": int(source_records),
                "exact_mc_samples": int(mc),
                "exact_sampling_mode": "antithetic_nested",
                "evaluate_or_replicate_new_action_count": int(shortlist),
                "adaptive_replication_voi": True,
                "replication_candidate_count": 4,
                "replication_max_per_solution": 5,
            }
    return profiles


def build_specs(args):
    mc_samples = _parse_int_csv(args.mc_samples)
    shortlist_sizes = _parse_int_csv(args.shortlist_sizes)
    profiles = _variant_profiles(
        mc_samples, shortlist_sizes, args.source_records_per_domain)
    args.variant_profiles = profiles
    args.variants = ",".join(profiles)
    args.scenarios = closure.promoted.v51.v50.v49.v27.MEAN_SCENARIOS
    args.stage_family = "promoted_v51_voi_fidelity"
    args.gate_label = "Promoted V51 VOI fidelity"
    args.sequential = bool(int(args.N) > int(args.n0))
    if int(args.N) != int(args.n0) + 1:
        raise ValueError("VOI fidelity gate must charge exactly one online action")
    return closure._root_module().build_specs(args)


def main():
    defaults = closure._root_module()
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", type=Path, default=defaults.DEFAULT_SCHEDULER)
    parser.add_argument("--deploy", type=Path, default=defaults.DEFAULT_DEPLOY)
    parser.add_argument("--python", type=Path, default=defaults.REMOTE_PYTHON)
    parser.add_argument("--manifest", type=Path, default=(
        defaults.DEFAULT_DEPLOY
        / "SC-OLH-KG/performance/manifests/v18b_exactkg_mcdiag.json"))
    parser.add_argument("--source-run-id", default=DEFAULT_SOURCE_RUN_ID)
    parser.add_argument(
        "--local-design-prerequisite",
        action="store_false",
        dest="remote_design_only",
    )
    parser.set_defaults(remote_design_only=True)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--run-id", default=(
        "scolh_promoted_v51_voi_fidelity_s5_"
        f"{time.strftime('%Y%m%d_%H%M%S')}"))
    parser.add_argument("--nodes", default=",".join(CPU_NODES))
    parser.add_argument(
        "--mc-samples", default=",".join(map(str, DEFAULT_MC_SAMPLES)))
    parser.add_argument(
        "--shortlist-sizes",
        default=",".join(map(str, DEFAULT_SHORTLIST_SIZES)),
    )
    parser.add_argument("--source-d", type=int, default=50)
    parser.add_argument("--source-records-per-domain", type=int, default=64)
    parser.add_argument("--d", type=int, default=1000)
    parser.add_argument("--N", type=int, default=11)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--pool-size", type=int, default=512)
    parser.add_argument("--variance-audit-size", type=int, default=128)
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
