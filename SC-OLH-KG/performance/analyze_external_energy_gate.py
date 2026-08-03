#!/usr/bin/env python3
"""Aggregate the registered stochastic OPSD development gate."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import statistics


ARMS = ("frozen_proposal", "common_sobol")


def analyze(paths):
    rows = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    if any(row.get("status") != "ok" for row in rows):
        raise ValueError("energy gate contains incomplete rows")
    summaries = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row.get("arm") == arm]
        seeds = sorted({int(row["seed"]) for row in arm_rows})
        objective = [
            row["deployment_truth_audit"]["true_objective_mean"]
            for row in arm_rows
            if row.get("deployment_truth_audit") is not None
            and row["deployment_truth_audit"]["truly_chance_feasible"]
        ]
        summaries[arm] = {
            "row_count": int(len(arm_rows)),
            "seeds": seeds,
            "independently_certified_count": int(sum(
                row["independently_certified"] for row in arm_rows)),
            "false_certificate_count": int(sum(
                row["false_certificate"] for row in arm_rows)),
            "truly_feasible_deployment_count": int(sum(
                row.get("deployment_truth_audit") is not None
                and row["deployment_truth_audit"]["truly_chance_feasible"]
                for row in arm_rows
            )),
            "median_true_feasible_objective": (
                float(statistics.median(objective)) if objective else None
            ),
            "mean_verification_calls": (
                float(statistics.fmean(
                    row["verification_calls"] for row in arm_rows))
                if arm_rows else None
            ),
        }
    frozen = summaries["frozen_proposal"]
    common = summaries["common_sobol"]
    checks = {
        "complete_five_seed_pairing": bool(
            frozen["seeds"] == common["seeds"]
            and len(frozen["seeds"]) == 5
        ),
        "frozen_certifies_at_least_four_of_five": bool(
            frozen["independently_certified_count"] >= 4),
        "frozen_has_zero_false_certificates": bool(
            frozen["false_certificate_count"] == 0),
        "frozen_certification_not_worse_than_common": bool(
            frozen["independently_certified_count"]
            >= common["independently_certified_count"]),
    }
    return {
        "schema_version": 1,
        "contract_id": "opsd_energy_stochastic_online_development_gate_v1",
        "status": "pass" if all(checks.values()) else "fail",
        "development_only": True,
        "confirmatory_target_opened": False,
        "checks": checks,
        "arms": summaries,
        "next_action": (
            "freeze_and_open_gb_gbn_confirmatory_target"
            if all(checks.values())
            else "stop_without_opening_confirmatory_target"
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = analyze(sorted(glob.glob(args.inputs, recursive=True)))
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
