#!/usr/bin/env python3
"""Analyze the V5 misspecification sequential promotion gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


VARIANTS = ("v4_latent_control", "misspec_scalar")
SCENARIOS = (
    ("FactorShockStatePolicyRZDT1", 0.0),
    ("FactorShockStatePolicyRZDT1", 4.0),
    ("InventorySupplyChain", 1.0),
    ("QueueResourceControl", 1.0),
)


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value == value and abs(value) != float("inf") else None


def _median(values):
    values = [value for value in (_number(item) for item in values)
              if value is not None]
    return None if not values else float(median(values))


def _variant(payload, row):
    label = str(row.get("experiment_variant") or payload.get(
        "experiment_variant", ""))
    for variant in VARIANTS:
        if f"/{variant}/" in f"/{label}/" or label.endswith(f"/{variant}"):
            return variant
    return None


def load_rows(root):
    rows = []
    for path in sorted(Path(root).rglob("result.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for raw in payload.get("rows", []):
            row = dict(raw)
            variant = _variant(payload, row)
            if variant is not None:
                row["gate_variant"] = variant
                rows.append(row)
    return rows


def _scenario(row):
    return (
        str(row.get("heldout")),
        float(row.get("target_shared_shock_scale", 1.0)),
    )


def _cell(rows, variant, scenario, expected_seeds):
    group = [
        row for row in rows
        if row["gate_variant"] == variant and _scenario(row) == scenario
    ]
    seeds = {int(row["seed"]) for row in group if row.get("seed") is not None}
    raw = [row.get("boundary_raw_pool_truth_diagnostics") or {}
           for row in group]
    feasible_regrets = [
        row.get("feasible_simple_regret") for row in group
        if bool(row.get("true_feasible", False))
    ]
    return {
        "variant": variant,
        "domain": scenario[0],
        "shock_scale": scenario[1],
        "n": int(len(group)),
        "complete": bool(
            len(group) == int(expected_seeds)
            and len(seeds) == int(expected_seeds)),
        "oracle_free": bool(group and all(bool(
            (row.get("audit") or {}).get(
                "admissible_oracle_free_transfer", False))
            for row in group)),
        "true_feasible_count": int(sum(bool(
            row.get("true_feasible", False)) for row in group)),
        "median_feasible_regret": _median(feasible_regrets),
        "initial_true_feasible_count": int(sum(bool(
            row.get("initial_has_true_feasible", False)) for row in group)),
        "adaptive_rescue_count": int(sum(bool(
            row.get("adaptive_rescue", False)) for row in group)),
        "adaptive_loss_count": int(sum(bool(
            row.get("adaptive_loss", False)) for row in group)),
        "adaptive_improvement_count": int(sum(bool(
            row.get("adaptive_improves_initial_best", False))
            for row in group)),
        "median_adaptive_regret_change": _median([
            row.get("adaptive_regret_change") for row in group]),
        "posterior_feasible_count": int(sum(bool(
            row.get("posterior_feasible", False)) for row in group)),
        "posterior_vacuous_count": int(sum(bool(
            row.get("posterior_certificate_vacuous", False))
            for row in group)),
        "evaluated_false_certificate_count": int(sum(
            int(row.get("false_certificate_count", 0) or 0)
            for row in group)),
        "audit_pool_false_certificate_count": int(sum(
            int(value.get("boundary_raw_pool_false_certified_count", 0) or 0)
            for value in raw)),
        "median_mean_abs_error": _median([
            value.get("boundary_raw_pool_constraint_mean_median_abs_error")
            for value in raw]),
        "median_mean_rank_correlation": _median([
            value.get("boundary_raw_pool_constraint_mean_rank_correlation")
            for value in raw]),
        "median_variance_log_rmse": _median([
            row.get("variance_log_rmse") for row in group]),
        "initial_design_fingerprints": sorted(set(str(
            (row.get("task_initial_design") or {}).get(
                "fingerprint", "missing"))
            for row in group)),
        "source_archive_fingerprints": sorted(set(str(
            (row.get("source_target_adaptation_contract") or {}).get(
                "source_archive_fingerprint", "missing"))
            for row in group)),
    }


def summarize(rows, expected_seeds=5):
    cells = {
        (variant, *scenario): _cell(rows, variant, scenario, expected_seeds)
        for variant in VARIANTS for scenario in SCENARIOS
    }
    challenger = [cells[("misspec_scalar", *scenario)]
                  for scenario in SCENARIOS]
    control = [cells[("v4_latent_control", *scenario)]
               for scenario in SCENARIOS]
    paired_initial_design = all(
        left["initial_design_fingerprints"]
        == right["initial_design_fingerprints"]
        and left["source_archive_fingerprints"]
        == right["source_archive_fingerprints"]
        for left, right in zip(challenger, control)
    )
    regret_nonworse = 0
    mean_nonworse = 0
    variance_nonworse = 0
    for left, right in zip(challenger, control):
        if (left["median_feasible_regret"] is not None
                and right["median_feasible_regret"] is not None):
            regret_nonworse += int(
                left["median_feasible_regret"]
                <= right["median_feasible_regret"] + 1e-12)
        if (left["median_mean_abs_error"] is not None
                and right["median_mean_abs_error"] is not None):
            mean_nonworse += int(
                left["median_mean_abs_error"]
                <= 1.10 * right["median_mean_abs_error"] + 0.01)
        if (left["median_variance_log_rmse"] is not None
                and right["median_variance_log_rmse"] is not None):
            variance_nonworse += int(
                left["median_variance_log_rmse"]
                <= 1.10 * right["median_variance_log_rmse"])
    challenger_feasible = sum(value["true_feasible_count"]
                              for value in challenger)
    control_feasible = sum(value["true_feasible_count"] for value in control)
    challenger_loss = sum(value["adaptive_loss_count"]
                          for value in challenger)
    control_loss = sum(value["adaptive_loss_count"] for value in control)
    challenger_false = sum(value["evaluated_false_certificate_count"]
                           for value in challenger)
    control_false = sum(value["evaluated_false_certificate_count"]
                        for value in control)
    challenger_audit_false = sum(
        value["audit_pool_false_certificate_count"] for value in challenger)
    control_audit_false = sum(
        value["audit_pool_false_certificate_count"] for value in control)
    checks = {
        "all_cells_complete": all(
            value["complete"] for value in cells.values()),
        "oracle_free": all(value["oracle_free"] for value in cells.values()),
        "paired_initial_design_and_archive": bool(paired_initial_design),
        "true_feasible_count_nonworse": bool(
            challenger_feasible >= control_feasible),
        "adaptive_loss_nonworse": bool(challenger_loss <= control_loss),
        "feasible_regret_nonworse_at_least_3_of_4": bool(
            regret_nonworse >= 3),
        "mean_mae_nonworse_at_least_3_of_4": bool(mean_nonworse >= 3),
        "variance_rmse_nonworse_at_least_3_of_4": bool(
            variance_nonworse >= 3),
        "evaluated_false_certification_nonworse": bool(
            challenger_false <= control_false),
        "audit_pool_false_certification_strictly_lower": bool(
            challenger_audit_false < control_audit_false),
    }
    return {
        "schema_version": 1,
        "row_count": int(len(rows)),
        "expected_row_count": int(
            len(VARIANTS) * len(SCENARIOS) * expected_seeds),
        "cells": list(cells.values()),
        "totals": {
            "challenger_true_feasible": int(challenger_feasible),
            "control_true_feasible": int(control_feasible),
            "challenger_adaptive_loss": int(challenger_loss),
            "control_adaptive_loss": int(control_loss),
            "challenger_evaluated_false_certificates": int(challenger_false),
            "control_evaluated_false_certificates": int(control_false),
            "challenger_audit_pool_false_certificates": int(
                challenger_audit_false),
            "control_audit_pool_false_certificates": int(control_audit_false),
        },
        "checks": checks,
        "promote_misspec_scalar": bool(all(checks.values())),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-seeds", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    result = summarize(load_rows(args.root), args.expected_seeds)
    output = args.out or args.root / "mean_alignment_v5_sequential_gate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "promote_misspec_scalar": result["promote_misspec_scalar"],
        "totals": result["totals"],
        "checks": result["checks"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
