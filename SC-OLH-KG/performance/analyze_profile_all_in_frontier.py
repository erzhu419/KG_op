#!/usr/bin/env python3
"""All-in cost audit for source-funded versus target-only profile designs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CONTRACT_ID = "randomized_ordered_profile_stress_v2"
SOURCE_CONFIGURATION = "source384-target10"
TARGET_CONFIGURATION = "target394"


def _success(row):
    return bool(
        row["contains_true_feasible"]
        and row["independently_certified"]
        and not row["false_certificate"]
    )


def _epsilon_success(row):
    return bool(_success(row) and row["feasible_and_epsilon_optimal_005"])


def analyze(paths):
    rows = []
    failures = []
    seen = set()
    for path in map(Path, paths):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(f"{path}: unreadable: {exc}")
            continue
        if row.get("contract_id") != CONTRACT_ID or row.get("status") != "ok":
            failures.append(f"{path}: wrong contract or status")
            continue
        configuration = row.get("matrix_configuration_id")
        if configuration not in {SOURCE_CONFIGURATION, TARGET_CONFIGURATION}:
            continue
        key = (
            row["regime"], int(row["target_seed"]), row["arm"],
            int(row["nominal_dimension"]), configuration,
        )
        if key in seen:
            failures.append(f"{path}: duplicate cell {key}")
            continue
        seen.add(key)
        rows.append(row)

    source = {
        (row["regime"], int(row["target_seed"]), int(row["nominal_dimension"])): row
        for row in rows
        if row["arm"] == "source_atlas"
        and row["matrix_configuration_id"] == SOURCE_CONFIGURATION
    }
    controls = sorted({
        row["arm"] for row in rows
        if row["matrix_configuration_id"] == TARGET_CONFIGURATION
    })
    paired = []
    for control in controls:
        comparator = {
            (
                row["regime"], int(row["target_seed"]),
                int(row["nominal_dimension"]),
            ): row
            for row in rows
            if row["arm"] == control
            and row["matrix_configuration_id"] == TARGET_CONFIGURATION
        }
        for key in sorted(set(source) & set(comparator)):
            first = source[key]
            second = comparator[key]
            first_cap = int(first["all_in_budget_cap_unamortized"])
            second_cap = int(second["all_in_budget_cap_unamortized"])
            paired.append({
                "regime": key[0],
                "target_seed": key[1],
                "nominal_dimension": key[2],
                "control": control,
                "equal_maximum_all_in_budget": bool(first_cap == second_cap),
                "source_maximum_all_in_budget": first_cap,
                "control_maximum_all_in_budget": second_cap,
                "source_success": _success(first),
                "control_success": _success(second),
                "source_epsilon_success": _epsilon_success(first),
                "control_epsilon_success": _epsilon_success(second),
                "source_actual_all_in_calls": int(
                    first["all_in_calls_unamortized"]),
                "control_actual_all_in_calls": int(
                    second["all_in_calls_unamortized"]),
                "source_penalized_loss": float(first["penalized_loss"]),
                "control_penalized_loss": float(second["penalized_loss"]),
            })

    summaries = []
    for control in controls:
        group = [row for row in paired if row["control"] == control]
        if not group:
            continue
        break_even = []
        for row in group:
            source_row = source[(
                row["regime"], row["target_seed"], row["nominal_dimension"])]
            source_operating = (
                int(source_row["target_search_calls"])
                + int(source_row["verification_calls"])
            )
            control_operating = row["control_actual_all_in_calls"]
            denominator = control_operating - source_operating
            if denominator > 0:
                break_even.append(float(source_row["source_calls"] / denominator))
        summaries.append({
            "control": control,
            "paired_independent_task_count": len(group),
            "all_pairs_have_equal_maximum_all_in_budget": bool(all(
                row["equal_maximum_all_in_budget"] for row in group)),
            "source_certified_success_count": int(sum(
                row["source_success"] for row in group)),
            "control_certified_success_count": int(sum(
                row["control_success"] for row in group)),
            "source_epsilon_success_count": int(sum(
                row["source_epsilon_success"] for row in group)),
            "control_epsilon_success_count": int(sum(
                row["control_epsilon_success"] for row in group)),
            "median_source_actual_all_in_calls": float(np.median([
                row["source_actual_all_in_calls"] for row in group])),
            "median_control_actual_all_in_calls": float(np.median([
                row["control_actual_all_in_calls"] for row in group])),
            "median_source_minus_control_penalized_loss": float(np.median([
                row["source_penalized_loss"] - row["control_penalized_loss"]
                for row in group
            ])),
            "median_archive_break_even_target_count": (
                None if not break_even else float(np.median(break_even))),
            "break_even_definition": (
                "source_calls/(target_only_actual_calls-"
                "source_target_search_and_verification_calls)"
            ),
        })
    return {
        "schema_version": 1,
        "contract_id": "profile_design_all_in_frontier_v1",
        "status": "complete" if paired and not failures else "incomplete",
        "fixed_budget_definition": (
            "source+target_search+maximum_frozen_shortlist_verification_calls"
        ),
        "primary_unit": "independent_target_task",
        "row_count": len(rows),
        "paired_row_count": len(paired),
        "summaries": summaries,
        "paired_rows": paired,
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = analyze(args.paths)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    print(json.dumps({
        "status": payload["status"],
        "paired_row_count": payload["paired_row_count"],
        "failure_count": len(payload["failures"]),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
