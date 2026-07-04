"""Fresh-seed traffic trajectory encoder audit.

This runner deliberately refuses to fabricate a traffic result.  If the
fresh-seed trajectory log is absent it emits a `missing_data` result.  When a
log is present, it aggregates policy occupancy and risk exposures so the
traffic case can be wired into SC-OLH-KG without relying on saved checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from encoders.policy_state_encoder import TrafficTrajectoryEncoder  # noqa: E402
from benchmark_quality import json_safe  # noqa: E402


def run(args):
    status = TrafficTrajectoryEncoder.missing_data_status(args.trajectory_log)
    result = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "trajectory_log": args.trajectory_log,
        "status": status["status"],
    }
    if status["status"] == "missing_data":
        result.update(status)
        return result

    encoder = TrafficTrajectoryEncoder.from_csv(args.trajectory_log)
    policies = sorted(encoder.policy_features)
    rows = []
    for policy_id in policies:
        rows.append({
            "policy_id": policy_id,
            "features": encoder.features(policy_id).tolist(),
            "risk_exposure": encoder.risk_exposure(policy_id).tolist(),
            "shared_shock_exposure": encoder.shared_shock_exposure(policy_id).tolist(),
            "n_occupancy_cells": len(encoder.occupancy(policy_id)),
        })
    result.update({
        "status": "encoded",
        "n_policies": len(policies),
        "policies": rows,
    })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory_log", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--out_dir", default=str(ROOT / "profiles"))
    parser.add_argument("--out_prefix", default="")
    args = parser.parse_args()
    result = run(args)
    text = json.dumps(json_safe(result), indent=2)
    out_path = args.out
    if not out_path and args.out_prefix:
        out_path = str(Path(args.out_dir) / f"{args.out_prefix}.json")
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
