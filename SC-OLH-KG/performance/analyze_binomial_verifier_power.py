#!/usr/bin/env python3
"""Exact validity and nonvacuity table for the all-success verifier."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.terminal_verification import (  # noqa: E402
    exact_all_success_power,
    minimum_all_success_binomial_budget,
)


def power_table(
    *,
    required_probability=0.95,
    familywise_delta=0.05,
    shortlist_size=3,
    budgets=(40, 60, 80, 96, 128, 160, 200),
    true_probabilities=(0.95, 0.96, 0.975, 0.99, 0.995, 1.0),
):
    shortlist_size = int(shortlist_size)
    if shortlist_size < 1:
        raise ValueError("shortlist_size must be positive")
    candidate_delta = float(familywise_delta) / shortlist_size
    minimum_budget = minimum_all_success_binomial_budget(
        required_probability, candidate_delta)
    rows = []
    for budget in map(int, budgets):
        valid = bool(
            float(required_probability) ** budget <= candidate_delta)
        for probability in map(float, true_probabilities):
            rows.append({
                "verification_budget": budget,
                "true_feasibility_probability": probability,
                "candidate_delta": candidate_delta,
                "required_feasibility_probability": float(
                    required_probability),
                "candidate_validity_enabled": valid,
                "certification_probability": (
                    exact_all_success_power(probability, budget)
                    if valid else 0.0
                ),
                "all_success_probability": exact_all_success_power(
                    probability, budget),
                "minimum_valid_budget": minimum_budget,
            })
    return {
        "schema_version": 1,
        "contract_id": "exact_all_success_binomial_power_v1",
        "method": "exact_binomial_all_success",
        "candidate_validity_statement": (
            "For any unsafe p <= p_required, false certification probability "
            "is at most p_required^n <= candidate_delta."
        ),
        "familywise_contract": "bonferroni_over_frozen_shortlist",
        "required_feasibility_probability": float(required_probability),
        "familywise_delta": float(familywise_delta),
        "shortlist_size": shortlist_size,
        "candidate_delta": candidate_delta,
        "minimum_valid_budget": minimum_budget,
        "warning": (
            "For a fixed all-success rule, power at p<1 decreases as n grows. "
            "The rule is chosen for transparent exact validity, not optimal "
            "sample efficiency."
        ),
        "rows": rows,
    }


def _csv_floats(value):
    return tuple(float(item.strip()) for item in str(value).split(",") if item.strip())


def _csv_ints(value):
    return tuple(int(item.strip()) for item in str(value).split(",") if item.strip())


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--csv")
    parser.add_argument("--required-probability", type=float, default=0.95)
    parser.add_argument("--familywise-delta", type=float, default=0.05)
    parser.add_argument("--shortlist-size", type=int, default=3)
    parser.add_argument("--budgets", default="40,60,80,96,128,160,200")
    parser.add_argument(
        "--true-probabilities", default="0.95,0.96,0.975,0.99,0.995,1.0")
    args = parser.parse_args()
    payload = power_table(
        required_probability=args.required_probability,
        familywise_delta=args.familywise_delta,
        shortlist_size=args.shortlist_size,
        budgets=_csv_ints(args.budgets),
        true_probabilities=_csv_floats(args.true_probabilities),
    )
    _atomic_json(args.out, payload)
    if args.csv:
        _write_csv(args.csv, payload["rows"])
    print(json.dumps({
        "status": "ok",
        "out": str(args.out),
        "minimum_valid_budget": payload["minimum_valid_budget"],
        "row_count": len(payload["rows"]),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
