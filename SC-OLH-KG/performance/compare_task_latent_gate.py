"""Paired promotion gate for authoritative joint task-latent inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _rows(paths):
    rows = []
    for path in paths:
        payload = json.loads(Path(path).read_text())
        rows.extend(payload.get("rows", []))
    return rows


def _key(row):
    return str(row["heldout"]), int(row["seed"])


def _mean(values):
    values = [float(value) for value in values if value is not None]
    return None if not values else float(np.mean(values))


def _median(values):
    values = [float(value) for value in values if value is not None]
    return None if not values else float(np.median(values))


def _leq(challenger, baseline, tolerance):
    if baseline is None:
        return challenger is None
    if challenger is None:
        return False
    return float(challenger) <= float(baseline) + float(tolerance)


def compare(
    baseline_rows,
    challenger_rows,
    *,
    domains,
    tolerance=1e-12,
    required_calibration_mode=None,
):
    baseline = {_key(row): row for row in baseline_rows}
    challenger = {_key(row): row for row in challenger_rows}
    if len(baseline) != len(baseline_rows):
        raise ValueError("baseline contains duplicate domain/seed rows")
    if len(challenger) != len(challenger_rows):
        raise ValueError("challenger contains duplicate domain/seed rows")
    requested = set(map(str, domains))
    baseline = {
        key: row for key, row in baseline.items() if key[0] in requested
    }
    challenger = {
        key: row for key, row in challenger.items() if key[0] in requested
    }
    if set(baseline) != set(challenger):
        missing = sorted(set(baseline) - set(challenger))
        extra = sorted(set(challenger) - set(baseline))
        raise ValueError(
            f"paired rows disagree: missing={missing}, extra={extra}")
    if {key[0] for key in baseline} != requested:
        raise ValueError("one or more requested domains are missing")

    summaries = []
    domain_passes = []
    for domain in domains:
        keys = sorted(key for key in baseline if key[0] == domain)
        base = [baseline[key] for key in keys]
        chall = [challenger[key] for key in keys]
        base_feasible = int(sum(bool(row.get("true_feasible")) for row in base))
        chall_feasible = int(sum(bool(row.get("true_feasible")) for row in chall))
        base_false = int(sum(bool(row.get("false_feasible")) for row in base))
        chall_false = int(sum(bool(row.get("false_feasible")) for row in chall))
        base_regret = _median(
            row.get("feasible_simple_regret") for row in base)
        chall_regret = _median(
            row.get("feasible_simple_regret") for row in chall)
        base_violation = _mean(row.get("constraint_violation") for row in base)
        chall_violation = _mean(
            row.get("constraint_violation") for row in chall)
        authoritative = [bool(
            (row.get("task_posterior") or {}).get(
                "task_latent_authoritative", False)
            or row.get("task_latent_inference_mode") == "authoritative"
        ) for row in chall]
        calibration_mode_recorded = [
            (
                required_calibration_mode is None
                or row.get("task_latent_calibration_mode")
                    == required_calibration_mode
                or (row.get("task_posterior") or {}).get(
                    "task_latent_calibration_mode")
                    == required_calibration_mode
            )
            for row in chall
        ]
        checks = {
            "authoritative_recorded": bool(all(authoritative)),
            "calibration_mode_recorded": bool(all(
                calibration_mode_recorded)),
            "false_feasible_nonincreasing": chall_false <= base_false,
            "true_feasible_nondecreasing": chall_feasible >= base_feasible,
            "median_feasible_regret_nonincreasing": _leq(
                chall_regret, base_regret, tolerance),
            "mean_violation_nonincreasing": _leq(
                chall_violation, base_violation, tolerance),
        }
        passed = bool(all(checks.values()))
        domain_passes.append(passed)
        summaries.append({
            "heldout": str(domain),
            "n_seeds": len(keys),
            "seeds": [key[1] for key in keys],
            "baseline_true_feasible": base_feasible,
            "challenger_true_feasible": chall_feasible,
            "baseline_false_feasible": base_false,
            "challenger_false_feasible": chall_false,
            "baseline_median_feasible_regret": base_regret,
            "challenger_median_feasible_regret": chall_regret,
            "baseline_mean_violation": base_violation,
            "challenger_mean_violation": chall_violation,
            "checks": checks,
            "passed": passed,
        })
    return {
        "schema_version": 1,
        "gate": "paired_authoritative_task_latent",
        "domains": list(map(str, domains)),
        "n_pairs": len(baseline),
        "passed": bool(all(domain_passes)),
        "summaries": summaries,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", nargs="+", required=True)
    parser.add_argument("--challenger", nargs="+", required=True)
    parser.add_argument(
        "--domains",
        default=(
            "FactorShockStatePolicyRZDT1,InventorySupplyChain,"
            "QueueResourceControl"
        ),
    )
    parser.add_argument("--tolerance", type=float, default=1e-12)
    parser.add_argument(
        "--required-calibration-mode",
        choices=("source_profiles", "expert_ridge"),
    )
    parser.add_argument("--out")
    args = parser.parse_args()
    result = compare(
        _rows(args.baseline),
        _rows(args.challenger),
        domains=[item.strip() for item in args.domains.split(",") if item.strip()],
        tolerance=args.tolerance,
        required_calibration_mode=args.required_calibration_mode,
    )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
