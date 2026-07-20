#!/usr/bin/env python3
"""Submit the V45 fixed-budget source-task episode gate."""

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
    / "scripts/submit_scolhkg_mean_alignment_v43_predictive_hyperlaw_gate_scheduler.py"
)
SPEC = importlib.util.spec_from_file_location("mean_alignment_v43_submit", BASE_PATH)
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)


def _profile(*, hyperlaw_mode, augments, geometry_shift, radius_jitter):
    value = dict(base.VARIANTS["v41_source_bayes"])
    value.update({
        "hyperlaw_mode": str(hyperlaw_mode),
        "confidence_mode": "source_bayes",
        "source_records_per_domain": 64,
        "source_augments": int(augments),
        "source_budget_mode": "per_base_domain",
        "source_geometry_shift_scale": float(geometry_shift),
        "source_geometry_log_radius_jitter": float(radius_jitter),
        "initial_design_archive_match_mode": (
            "exact" if int(augments) == 1 else "paired_frozen_control"
        ),
        # Isolate task geometry from the older nuisance-only augmentation.
        "source_sigma_jitter": 0.0,
        "source_alpha_jitter": 0.0,
        "source_weight_jitter": 0.0,
        "decision_backend": "sobol_new",
        "adaptive_replication_voi": False,
        "replication_candidate_count": 0,
    })
    return value


VARIANTS = {
    "v41_two_task_source_bayes": _profile(
        hyperlaw_mode="single_gaussian_draw",
        augments=1,
        geometry_shift=0.0,
        radius_jitter=0.0,
    ),
    "v45_episode_label_control": _profile(
        hyperlaw_mode="single_gaussian_draw",
        augments=4,
        geometry_shift=0.0,
        radius_jitter=0.0,
    ),
    "v45_geometry_source_bayes": _profile(
        hyperlaw_mode="single_gaussian_draw",
        augments=4,
        geometry_shift=0.075,
        radius_jitter=0.20,
    ),
    "v45_geometry_predictive_bayes": _profile(
        hyperlaw_mode="shared_low_rank_predictive",
        augments=4,
        geometry_shift=0.075,
        radius_jitter=0.20,
    ),
}
SENTINEL_SCENARIOS = (("QueueResourceControl", 1.0),)
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
    args.stage_family = "mean_v45"
    args.gate_label = "Mean V45"
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
        "scolh_mean_alignment_v45_source_episode_"
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
