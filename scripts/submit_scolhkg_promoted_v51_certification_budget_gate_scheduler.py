#!/usr/bin/env python3
"""Submit the promoted-V51 certification nonvacuity budget screen."""

from __future__ import annotations

import argparse
import copy
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
DEFAULT_BUDGETS = (20, 40, 80)
DEFAULT_REPLICATION_CAPS = (5, 10, 20)


def _parse_int_csv(value):
    values = tuple(
        int(item.strip()) for item in str(value).split(",") if item.strip()
    )
    if not values or any(value <= 0 for value in values):
        raise ValueError("integer lists must contain positive values")
    return values


def _variant_profiles(replication_caps, source_records_per_domain):
    common = {
        **PROFILE,
        "source_records_per_domain": int(source_records_per_domain),
    }
    new_only = {
        **common,
        "decision_backend": "sobol_new",
        "adaptive_replication_voi": False,
        "replication_candidate_count": 0,
        "replication_max_per_solution": 0,
    }
    profiles = {"new_only": new_only}
    for cap in replication_caps:
        profiles[f"joint_cap{cap}"] = {
            **common,
            "decision_backend": "sobol_exact_joint_voi",
            "adaptive_replication_voi": True,
            "replication_candidate_count": 4,
            "replication_max_per_solution": int(cap),
        }
    return profiles


def build_specs(args):
    budgets = _parse_int_csv(args.budgets)
    caps = _parse_int_csv(args.replication_caps)
    if any(budget <= int(args.n0) for budget in budgets):
        raise ValueError("every certification budget must exceed n0")
    profiles = _variant_profiles(caps, args.source_records_per_domain)
    root_module = closure._root_module()
    specs = []
    for budget in budgets:
        child = copy.copy(args)
        child.N = int(budget)
        child.run_id = f"{args.run_id}_n{budget}"
        child.variant_profiles = profiles
        child.variants = ",".join(profiles)
        child.scenarios = closure.promoted.v51.v50.v49.v27.MEAN_SCENARIOS
        child.stage_family = "promoted_v51_certification_budget"
        child.gate_label = "Promoted V51 certification budget"
        child.sequential = True
        specs.extend(root_module.build_specs(child))
    return specs


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
        "scolh_promoted_v51_certification_budget_s5_"
        f"{time.strftime('%Y%m%d_%H%M%S')}"))
    parser.add_argument("--nodes", default=",".join(CPU_NODES))
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    parser.add_argument(
        "--replication-caps",
        default=",".join(map(str, DEFAULT_REPLICATION_CAPS)),
    )
    parser.add_argument("--source-d", type=int, default=50)
    parser.add_argument("--source-records-per-domain", type=int, default=64)
    parser.add_argument("--d", type=int, default=1000)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=5)
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
