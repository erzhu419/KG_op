#!/usr/bin/env python3
"""Submit a behavior-preserving finite-sample theory audit for promoted V51."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


closure = _load(
    "statistical_closure_v51",
    ROOT / "scripts/submit_scolhkg_promoted_v51_closure_gate_scheduler.py",
)
CPU_NODES = closure.CPU_NODES
DEFAULT_SOURCE_RUN_ID = closure.DEFAULT_SOURCE_RUN_ID
DEFAULT_REGISTRATION = (
    ROOT
    / "SC-OLH-KG/performance/manifests/statistical_closure_v2_audit_v1.json"
)
IMPLEMENTATION_CONTRACT_ID = "promoted_v51_observed_terminal_closure"
THEORY_CONTRACT_ID = "v51_statistical_closure_v2"
VARIANT = "statistical_closure_v2"
DEFAULT_RUN_ID = "scolh_statistical_closure_v2_audit_s5_20260721_01"


def load_registration(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("statistical closure audit schema_version must be 1")
    return payload


def validate_registration(registration):
    if registration.get("status") != "frozen":
        raise RuntimeError("statistical closure audit is not frozen")
    contracts = dict(registration.get("contracts") or {})
    if contracts.get("implementation_contract_id") != IMPLEMENTATION_CONTRACT_ID:
        raise ValueError("audit implementation contract is missing or changed")
    if contracts.get("theory_contract_id") != THEORY_CONTRACT_ID:
        raise ValueError("audit theory contract is missing or changed")
    expected = dict(registration.get("profile") or {})
    actual = {
        "exact_mc_samples": int(closure.PROFILE["exact_mc_samples"]),
        "exact_shortlist_size": int(
            closure.PROFILE["evaluate_or_replicate_new_action_count"]),
        "exact_sampling_mode": str(closure.PROFILE["exact_sampling_mode"]),
        "replication_cap": int(closure.PROFILE["replication_max_per_solution"]),
    }
    if expected != actual:
        raise ValueError("audit profile no longer matches promoted V51")
    return expected


def build_specs(args, registration):
    frozen_profile = validate_registration(registration)
    profile = {
        **closure.PROFILE,
        "source_records_per_domain": int(args.source_records_per_domain),
        "implementation_contract_id": IMPLEMENTATION_CONTRACT_ID,
        "theory_contract_id": THEORY_CONTRACT_ID,
    }
    args.implementation_contract_id = IMPLEMENTATION_CONTRACT_ID
    args.theory_contract_id = THEORY_CONTRACT_ID
    args.variant_profiles = {VARIANT: profile}
    args.variants = VARIANT
    args.scenarios = closure.promoted.v51.v50.v49.v27.MEAN_SCENARIOS
    args.stage_family = "statistical_closure_v2_audit"
    args.gate_label = "V51 statistical closure V2 audit"
    args.sequential = bool(int(args.N) > int(args.n0))
    specs = closure._root_module().build_specs(args)
    for spec in specs:
        spec["theory_audit_contract"] = {
            "implementation_contract_id": IMPLEMENTATION_CONTRACT_ID,
            "theory_contract_id": THEORY_CONTRACT_ID,
            **frozen_profile,
        }
    return specs


def main():
    defaults = closure._root_module()
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", type=Path, default=defaults.DEFAULT_SCHEDULER)
    parser.add_argument("--deploy", type=Path, default=defaults.DEFAULT_DEPLOY)
    parser.add_argument("--python", type=Path, default=defaults.REMOTE_PYTHON)
    parser.add_argument("--registration", type=Path,
                        default=DEFAULT_REGISTRATION)
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
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--nodes", default=",".join(CPU_NODES))
    parser.add_argument("--source-d", type=int, default=50)
    parser.add_argument("--source-records-per-domain", type=int, default=64)
    parser.add_argument("--d", type=int, default=1000)
    parser.add_argument("--N", type=int, default=20)
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
    registration = load_registration(args.registration)
    specs = build_specs(args, registration)
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
