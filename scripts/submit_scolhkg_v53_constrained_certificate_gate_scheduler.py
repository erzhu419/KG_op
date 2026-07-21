#!/usr/bin/env python3
"""Submit paired V53 MC-fidelity or constrained-certificate gates."""

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


closure = _load(
    "promoted_v51_closure_submit",
    ROOT / "scripts/submit_scolhkg_promoted_v51_closure_gate_scheduler.py",
)
CPU_NODES = closure.CPU_NODES
DEFAULT_SOURCE_RUN_ID = closure.DEFAULT_SOURCE_RUN_ID
CONTROL = "v51_control"
V52 = "v52_action_superset"
V53 = "v53_certificate_constrained"
FIDELITY = ("v53_mc8", "v53_mc32")
VARIANTS = (CONTROL, V52, V53, *FIDELITY)


def _command_option(cmd, option):
    tokens = shlex.split(str(cmd))
    try:
        index = tokens.index(str(option))
    except ValueError as exc:
        raise ValueError(f"V53 task is missing {option}") from exc
    if index + 1 >= len(tokens):
        raise ValueError(f"V53 task has no value for {option}")
    return tokens[index + 1]


def _local_deploy_path(args, remote_path):
    marker = "/SC-OLH-KG/"
    remote_path = str(remote_path)
    if marker not in remote_path:
        raise ValueError(
            "V53 initial-design path is outside the SC-OLH-KG deploy tree: "
            f"{remote_path}"
        )
    relative = remote_path.split(marker, 1)[1]
    return Path(args.deploy) / "SC-OLH-KG" / relative


def validate_frozen_design_seed_coverage(args, specs):
    """Fail locally when a frozen proposal does not contain requested seeds."""
    required = {}
    for spec in specs:
        remote_path = _command_option(spec["cmd"], "--initial-design-file")
        seed = int(_command_option(spec["cmd"], "--seed"))
        local_path = _local_deploy_path(args, remote_path)
        required.setdefault(local_path, set()).add(seed)

    coverage = {}
    for path, seeds in required.items():
        if not path.is_file():
            raise FileNotFoundError(
                "cannot validate V53 frozen-design seeds because the local "
                f"deploy mirror is missing {path}"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        available = {int(seed) for seed in payload.get("designs", {})}
        missing = sorted(seeds - available)
        if missing:
            available_text = (
                f"{min(available)}..{max(available)}" if available else "none"
            )
            raise ValueError(
                f"frozen design {path} is missing requested seeds {missing}; "
                f"available seeds: {available_text}"
            )
        coverage[str(path)] = sorted(seeds)
    return coverage


def _v53_profile(common, mc_samples):
    return {
        **common,
        "implementation_contract_id": "v53_constrained_certificate_deficit",
        "theory_contract_id": "v53_constrained_certificate_deficit_v1",
        "exact_mc_samples": int(mc_samples),
        "evaluate_or_replicate_new_action_count": 6,
        "evaluate_or_replicate_new_action_policy": (
            "canonical_plus_posterior_risk_certificate_coverage"),
        "policy_improvement_mode": "certificate_constrained",
    }


def variant_profiles(args):
    common = {
        **closure.PROFILE,
        "source_records_per_domain": int(args.source_records_per_domain),
        "exact_mc_samples": int(args.exact_mc_samples),
        "exact_sampling_mode": str(args.exact_sampling_mode),
        "evaluate_or_replicate_baseline_new_action_count": 4,
        "policy_improvement_mc_error_bound": float(args.risk_eta),
        "policy_improvement_certificate_mc_error_bound": float(
            args.certificate_eta),
        "policy_improvement_rollout_depth": 1,
        "policy_improvement_rollout_max_arms": 0,
        "policy_improvement_rollout_mc_samples": 0,
        "policy_improvement_rollout_mc_error_bound": 0.0,
    }
    return {
        CONTROL: {
            **common,
            "implementation_contract_id": (
                "promoted_v51_observed_terminal_closure"),
            "theory_contract_id": "v51_statistical_closure_v2",
            "evaluate_or_replicate_new_action_count": 4,
            "evaluate_or_replicate_new_action_policy": (
                "canonical_plus_posterior_risk"),
            "policy_improvement_mode": "off",
        },
        V52: {
            **common,
            "implementation_contract_id": "v52_safeguarded_policy_improvement",
            "theory_contract_id": "v52_safeguarded_closure_v1",
            "evaluate_or_replicate_new_action_count": 6,
            "evaluate_or_replicate_new_action_policy": (
                "canonical_plus_posterior_risk_certificate_coverage"),
            "policy_improvement_mode": "action_superset",
        },
        V53: _v53_profile(common, args.exact_mc_samples),
        "v53_mc8": _v53_profile(common, 8),
        "v53_mc32": _v53_profile(common, 32),
    }


def build_specs(args):
    profiles = variant_profiles(args)
    requested = [
        value.strip() for value in str(args.variants).split(",")
        if value.strip()
    ]
    unknown = sorted(set(requested) - set(profiles))
    if unknown:
        raise ValueError(f"unknown V53 variants: {unknown}")
    args.variant_profiles = {name: profiles[name] for name in requested}
    args.variants = ",".join(requested)
    args.scenarios = closure.promoted.v51.v50.v49.v27.MEAN_SCENARIOS
    args.stage_family = "v53_constrained_certificate_deficit"
    args.gate_label = "V53 constrained certificate-deficit policy"
    args.sequential = bool(int(args.N) > int(args.n0))
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
        "scolh_v53_constrained_certificate_s5_"
        f"{time.strftime('%Y%m%d_%H%M%S')}"))
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
    parser.add_argument(
        "--variants",
        default=",".join((CONTROL, V52, V53)),
    )
    parser.add_argument("--exact-mc-samples", type=int, default=8)
    parser.add_argument(
        "--exact-sampling-mode", default="antithetic_nested")
    parser.add_argument("--risk-eta", type=float, default=0.0)
    parser.add_argument("--certificate-eta", type=float, default=0.0)
    parser.add_argument("--cpu", type=int, default=12)
    parser.add_argument("--ram-mb", type=int, default=8192)
    parser.add_argument("--no-sync", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    specs = build_specs(args)
    if args.dry_run:
        print(json.dumps(specs, indent=2))
        return
    coverage = validate_frozen_design_seed_coverage(args, specs)
    print(json.dumps({"validated_frozen_design_seeds": coverage}, indent=2))
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
