#!/usr/bin/env python3
"""Submit the V43 finite-source predictive hyperlaw gate."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = (
    ROOT
    / "scripts/submit_scolhkg_mean_alignment_v42_hyperlaw_gate_scheduler.py"
)
SPEC = importlib.util.spec_from_file_location("mean_alignment_v42_submit", BASE_PATH)
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)


def _profile(hyperlaw_mode, confidence_mode):
    value = dict(base.VARIANTS["v35_model_confidence"])
    value.update({
        "hyperlaw_mode": str(hyperlaw_mode),
        "confidence_mode": str(confidence_mode),
        "confidence_delta": 0.05,
        "decision_backend": "sobol_new",
        "adaptive_replication_voi": False,
        "replication_candidate_count": 0,
    })
    return value


VARIANTS = {
    "v41_source_bayes": _profile("single_gaussian_draw", "source_bayes"),
    "v42_shared_low_rank_bayes": _profile(
        "shared_low_rank_discrepancy", "source_bayes"),
    "v43_predictive_low_rank_model": _profile(
        "shared_low_rank_predictive", "model"),
    "v43_predictive_low_rank_bayes": _profile(
        "shared_low_rank_predictive", "source_bayes"),
}
SENTINEL_SCENARIOS = (("QueueResourceControl", 1.0),)
FULL_SCENARIOS = base.MEAN_SCENARIOS
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
    args.stage_family = "mean_v43"
    args.gate_label = "Mean V43"
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
        "scolh_mean_alignment_v43_predictive_hyperlaw_"
        f"{time.strftime('%Y%m%d_%H%M%S')}"))
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--nodes", default=",".join(CPU_NODES))
    parser.add_argument(
        "--scope", choices=("queue_sentinel", "full"),
        default="queue_sentinel")
    parser.add_argument("--seeds", default="1,3")
    parser.add_argument("--source-d", type=int, default=50)
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
        profile["confidence_delta"] = float(args.confidence_delta)
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
