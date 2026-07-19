#!/usr/bin/env python3
"""Submit the V21 cross-fitted role-assignment posterior gate."""

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
    ROOT / "scripts/submit_scolhkg_mean_alignment_v20_offline_gate_scheduler.py"
)
SPEC = importlib.util.spec_from_file_location("mean_alignment_v20_submit", BASE_PATH)
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)


def _loo(temperature):
    value = base._role_assignment()
    value.update({
        "structure_score": "loo_predictive",
        "evidence_temperature": float(temperature),
        "misspecification": "none",
    })
    return value


VARIANTS = {
    "v15_tanh_control": dict(base.VARIANTS["v15_tanh_control"]),
    "v20_assignment_marginal": base._role_assignment(),
    "assignment_loo_t05": _loo(0.5),
    "assignment_loo_t10": _loo(1.0),
    "assignment_loo_t20": _loo(2.0),
}
MEAN_SCENARIOS = base.MEAN_SCENARIOS
CPU_NODES = base.CPU_NODES


def _root_module():
    return base._root_module()


def build_specs(args):
    args.variant_profiles = VARIANTS
    args.scenarios = MEAN_SCENARIOS
    args.stage_family = "mean_v21"
    args.gate_label = "Mean V21"
    args.sequential = bool(int(args.N) > int(args.n0))
    return _root_module().build_specs(args)


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
        "scolh_mean_alignment_v21_offline_s5_"
        f"{time.strftime('%Y%m%d_%H%M%S')}"))
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--nodes", default=",".join(CPU_NODES))
    parser.add_argument("--source-d", type=int, default=50)
    parser.add_argument("--d", type=int, default=1000)
    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--pool-size", type=int, default=512)
    parser.add_argument("--variance-audit-size", type=int, default=512)
    parser.add_argument("--misspecification-prior-df", type=float, default=4.0)
    parser.add_argument("--misspecification-ridge", type=float, default=1.0)
    parser.add_argument("--misspecification-max-scale", type=float, default=100.0)
    parser.add_argument("--contrast-scale", type=float, default=1.0)
    parser.add_argument("--null-geometry-ridge", type=float, default=1e-3)
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument("--ram-mb", type=int, default=4096)
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
