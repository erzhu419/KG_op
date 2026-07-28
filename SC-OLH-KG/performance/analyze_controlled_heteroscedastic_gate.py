#!/usr/bin/env python3
"""Aggregate the controlled heteroscedastic optimization/certificate gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics


def _median(values):
    finite = [
        float(value) for value in values
        if value is not None
    ]
    return None if not finite else float(statistics.median(finite))


def _load_rows(root):
    rows = []
    for path in sorted(Path(root).rglob("result.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("experiment") != "controlled_heteroscedastic_optimum":
            continue
        payload["_path"] = str(path)
        rows.append(payload)
    return rows


def _group_summary(rows):
    groups = {}
    for row in rows:
        key = (
            row["scenario"],
            row["variance_mode"],
            row["backend"],
            row.get("information_contract", {}).get(
                "terminal_safe_interior_selection_mode",
                row.get("independent_terminal_certificate", {}).get(
                    "safe_interior_selection_mode", "diverse"),
            ),
        )
        groups.setdefault(key, []).append(row)
    output = []
    for key, members in sorted(groups.items()):
        primary = [
            bool(row["paired_deployment_effect"]["primary_true_feasible"])
            for row in members
        ]
        deployment = [
            bool(row["paired_deployment_effect"]["deployment_true_feasible"])
            for row in members
        ]
        terminal = [
            bool(row["independent_terminal_certificate"]["certified"])
            for row in members
        ]
        posterior = [
            not bool(row["posterior_certificate"].get(
                "posterior_certificate_vacuous", True))
            for row in members
        ]
        output.append({
            "scenario": key[0],
            "variance_mode": key[1],
            "backend": key[2],
            "terminal_safe_interior_selection": key[3],
            "completed": int(len(members)),
            "primary_true_feasible": int(sum(primary)),
            "deployment_true_feasible": int(sum(deployment)),
            "terminal_certified": int(sum(terminal)),
            "posterior_nonvacuous": int(sum(posterior)),
            "posterior_certified_points": int(sum(
                int(row["posterior_certificate"].get(
                    "posterior_certified_count", 0))
                for row in members
            )),
            "posterior_false_certificates": int(sum(
                int(row["posterior_certificate"].get(
                    "false_certificate_count", 0))
                for row in members
            )),
            "terminal_false_certificates": int(sum(
                bool(row["independent_terminal_certificate"].get(
                    "false_certificate", False))
                for row in members
            )),
            "deployment_rescues": int(sum(
                bool(row["paired_deployment_effect"]["feasibility_rescue"])
                for row in members
            )),
            "deployment_losses": int(sum(
                bool(row["paired_deployment_effect"]["feasibility_loss"])
                for row in members
            )),
            "strict_objective_wins": int(sum(
                bool(row["paired_deployment_effect"]["strict_objective_win"])
                for row in members
            )),
            "strict_objective_losses": int(sum(
                bool(row["paired_deployment_effect"]["strict_objective_loss"])
                for row in members
            )),
            "oracle_hits_at_0_01": int(sum(
                bool(row["best_evaluated_truth"]["oracle_hit_at_0_01"])
                for row in members
            )),
            "median_primary_feasible_regret": _median([
                row["paired_deployment_effect"][
                    "primary_feasible_regret"]
                for row in members
            ]),
            "median_deployment_feasible_regret": _median([
                row["paired_deployment_effect"][
                    "deployment_feasible_regret"]
                for row in members
            ]),
            "median_best_evaluated_feasible_regret": _median([
                row["best_evaluated_truth"][
                    "best_evaluated_feasible_regret"]
                for row in members
            ]),
            "median_log_variance_rmse": _median([
                row["variance_audit"]["log_variance_rmse"]
                for row in members
            ]),
            "median_upper_variance_coverage": _median([
                row["variance_audit"]["upper_variance_coverage"]
                for row in members
            ]),
            "median_wall_time_sec": _median([
                row["wall_time_sec"] for row in members
            ]),
        })
    return output


def analyze(root):
    root = Path(root)
    rows = _load_rows(root)
    manifest_path = root / "submission_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file() else {}
    )
    planned = int(manifest.get("task_count", 0))
    paired = [row["paired_deployment_effect"] for row in rows]
    terminal = [
        row["independent_terminal_certificate"] for row in rows]
    evaluated = [row["best_evaluated_truth"] for row in rows]
    posterior = [row["posterior_certificate"] for row in rows]
    return {
        "schema_version": 1,
        "experiment": "controlled_heteroscedastic_optimum_gate",
        "run_id": manifest.get("run_id", root.name),
        "planned": planned,
        "completed": int(len(rows)),
        "complete": bool(planned > 0 and len(rows) == planned),
        "optimization_support": {
            "evaluated_true_feasible_runs": int(sum(
                bool(row.get("found_true_feasible", False))
                for row in evaluated
            )),
            "primary_true_feasible_runs": int(sum(
                bool(row.get("primary_true_feasible", False))
                for row in paired
            )),
            "deployment_true_feasible_runs": int(sum(
                bool(row.get("deployment_true_feasible", False))
                for row in paired
            )),
            "found_but_not_primary_runs": int(sum(
                bool(found.get("found_true_feasible", False))
                and not bool(effect.get("primary_true_feasible", False))
                for found, effect in zip(evaluated, paired)
            )),
        },
        "posterior_certificate_audit": {
            "nonvacuous_runs": int(sum(
                not bool(row.get(
                    "posterior_certificate_vacuous", True))
                for row in posterior
            )),
            "certified_points": int(sum(
                int(row.get("posterior_certified_count", 0))
                for row in posterior
            )),
            "certified_true_feasible_points": int(sum(
                int(row.get("certified_true_feasible_count", 0))
                for row in posterior
            )),
            "false_certified_points": int(sum(
                int(row.get("false_certificate_count", 0))
                for row in posterior
            )),
        },
        "paired_certificate_effect": {
            "recommendation_changes": int(sum(
                bool(row["recommendation_changed"]) for row in paired)),
            "feasibility_rescues": int(sum(
                bool(row["feasibility_rescue"]) for row in paired)),
            "feasibility_losses": int(sum(
                bool(row["feasibility_loss"]) for row in paired)),
            "strict_objective_wins": int(sum(
                bool(row["strict_objective_win"]) for row in paired)),
            "strict_objective_losses": int(sum(
                bool(row["strict_objective_loss"]) for row in paired)),
            "terminal_certificates": int(sum(
                bool(row["certified"]) for row in terminal)),
            "terminal_false_certificates": int(sum(
                bool(row["false_certificate"]) for row in terminal)),
        },
        "groups": _group_summary(rows),
    }


def _write_group_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    summary = analyze(args.root)
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_group_csv(args.out / "grouped_summary.csv", summary["groups"])
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
