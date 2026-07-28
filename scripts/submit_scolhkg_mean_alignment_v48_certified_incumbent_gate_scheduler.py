#!/usr/bin/env python3
"""Submit the V48 certified-only incumbent-preservation Queue gate."""

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


base = _load(
    "mean_alignment_v47_submit",
    ROOT
    / "scripts/submit_scolhkg_mean_alignment_v47_deconvolution_gate_scheduler.py",
)
v27 = _load(
    "mean_alignment_v27_submit",
    ROOT / "scripts/submit_scolhkg_mean_alignment_v27_sequential_gate_scheduler.py",
)


def _profile(source, *, enabled, initialization):
    value = dict(source)
    value.update({
        "posterior_dominance_enabled": bool(enabled),
        "posterior_dominance_initialization": str(initialization),
    })
    return value


V27 = v27.VARIANTS["exchangeable_aggregate_none"]
V47 = base.VARIANTS["v47_deconvolved_task_bayes"]
VARIANTS = {
    "v27_promoted_no_preservation": _profile(
        V27, enabled=False, initialization="risk"),
    "v27_certified_only": _profile(
        V27, enabled=True, initialization="certified_only"),
    "v47_risk_preservation": _profile(
        V47, enabled=True, initialization="risk"),
    "v47_no_preservation": _profile(
        V47, enabled=False, initialization="risk"),
    "v48_certified_only": _profile(
        V47, enabled=True, initialization="certified_only"),
}
SENTINEL_SCENARIOS = base.SENTINEL_SCENARIOS
FULL_SCENARIOS = base.FULL_SCENARIOS
CPU_NODES = base.CPU_NODES


def _root_module():
    return base._root_module()


def _parse_seeds(value):
    seeds = tuple(sorted(set(
        int(item.strip()) for item in str(value).split(",") if item.strip()
    )))
    if not seeds or seeds[0] < 0:
        raise ValueError("seeds must be a nonempty list of nonnegative integers")
    return seeds


def build_specs(args):
    seeds = _parse_seeds(getattr(args, "seeds", "1,3"))
    scope = str(getattr(args, "scope", "queue_sentinel")).strip().lower()
    if scope not in {"queue_sentinel", "full"}:
        raise ValueError("scope must be queue_sentinel or full")
    args.variant_profiles = VARIANTS
    args.scenarios = (
        SENTINEL_SCENARIOS if scope == "queue_sentinel" else FULL_SCENARIOS
    )
    args.stage_family = "mean_v48"
    args.gate_label = "Mean V48"
    args.sequential = bool(int(args.N) > int(args.n0))
    args.seed_start = min(seeds)
    args.n_seeds = max(seeds) - min(seeds) + 1
    specs = _root_module().build_specs(args)
    suffixes = tuple(f"/seed{seed}" for seed in seeds)
    return [
        spec for spec in specs
        if str(spec.get("signature", "")).endswith(suffixes)
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
        "scolh_mean_alignment_v48_certified_incumbent_"
        f"{time.strftime('%Y%m%d_%H%M%S')}"))
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--nodes", default=",".join(CPU_NODES))
    parser.add_argument(
        "--scope", choices=("queue_sentinel", "full"),
        default="queue_sentinel")
    parser.add_argument("--seeds", default="1,3")
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
    parser.add_argument("--confidence-delta", type=float, default=0.05)
    parser.add_argument("--contrast-scale", type=float, default=1.0)
    parser.add_argument("--null-geometry-ridge", type=float, default=1e-3)
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument("--ram-mb", type=int, default=4096)
    parser.add_argument("--no-sync", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for profile in VARIANTS.values():
        profile["source_records_per_domain"] = int(
            args.source_records_per_domain)
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
