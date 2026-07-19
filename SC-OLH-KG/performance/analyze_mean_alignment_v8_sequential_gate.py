#!/usr/bin/env python3
"""Analyze the paired V8 support-adaptive sequential gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

try:
    from .analyze_mean_alignment_v8_gate import (
        SCENARIOS,
        _cell as _offline_cell,
        _scenario,
        load_rows,
    )
except ImportError:  # Direct script execution.
    from analyze_mean_alignment_v8_gate import (
        SCENARIOS,
        _cell as _offline_cell,
        _scenario,
        load_rows,
    )


VARIANTS = ("v4_latent_control", "adaptive_role_ordered")


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


def _cell(rows, variant, scenario, expected_seeds):
    value = _offline_cell(rows, variant, scenario, expected_seeds)
    group = [
        row for row in rows
        if row["gate_variant"] == variant and _scenario(row) == scenario
    ]
    value.update({
        "initial_true_feasible_count": int(sum(bool(
            row.get("initial_has_true_feasible", False)) for row in group)),
        "initial_total_true_feasible_points": int(sum(int(
            row.get("initial_true_feasible_count", 0) or 0)
            for row in group)),
        "median_initial_best_feasible_regret": _median([
            row.get("initial_best_feasible_regret") for row in group]),
        "adaptive_loss_count": int(sum(bool(
            row.get("adaptive_loss", False)) for row in group)),
        "adaptive_rescue_count": int(sum(bool(
            row.get("adaptive_rescue", False)) for row in group)),
        "adaptive_improvement_count": int(sum(bool(
            row.get("adaptive_improves_initial_best", False))
            for row in group)),
        "median_adaptive_regret_change": _median([
            row.get("adaptive_regret_change") for row in group]),
        "posterior_vacuous_count": int(sum(bool(
            row.get("posterior_certificate_vacuous", False))
            for row in group)),
        "posterior_certified_evaluated_count": int(sum(int(
            row.get("posterior_certified_evaluated_count", 0) or 0)
            for row in group)),
        "evaluated_false_certificate_count": int(sum(int(
            row.get("false_certificate_count", 0) or 0)
            for row in group)),
    })
    return value


def _count_nonworse(challenger, control, key, *, multiplier=1.0,
                    offset=0.0):
    count = 0
    for left, right in zip(challenger, control):
        if left[key] is not None and right[key] is not None:
            count += int(left[key] <= multiplier * right[key] + offset)
    return count


def summarize(rows, expected_seeds=5, *, N=20, n0=10):
    cells = {
        (variant, *scenario): _cell(rows, variant, scenario, expected_seeds)
        for variant in VARIANTS for scenario in SCENARIOS
    }
    grouped = {
        variant: [cells[(variant, *scenario)] for scenario in SCENARIOS]
        for variant in VARIANTS
    }
    control = grouped["v4_latent_control"]
    challenger = grouped["adaptive_role_ordered"]
    all_cells = list(cells.values())
    paired = all(
        left["initial_design_fingerprints"]
        == right["initial_design_fingerprints"]
        and left["source_archive_fingerprints"]
        == right["source_archive_fingerprints"]
        for left, right in zip(challenger, control)
    )

    totals = {}
    for variant, values in grouped.items():
        totals[variant] = {
            "initial_has_true_feasible": int(sum(
                value["initial_true_feasible_count"] for value in values)),
            "final_true_feasible": int(sum(
                value["true_feasible_count"] for value in values)),
            "adaptive_rescue": int(sum(
                value["adaptive_rescue_count"] for value in values)),
            "adaptive_loss": int(sum(
                value["adaptive_loss_count"] for value in values)),
            "adaptive_improvement": int(sum(
                value["adaptive_improvement_count"] for value in values)),
            "posterior_vacuous": int(sum(
                value["posterior_vacuous_count"] for value in values)),
            "posterior_certified_evaluated": int(sum(
                value["posterior_certified_evaluated_count"]
                for value in values)),
            "evaluated_false_certificates": int(sum(
                value["evaluated_false_certificate_count"]
                for value in values)),
            "audit_pool_false_certificates": int(sum(
                value["audit_pool_false_certificate_count"]
                for value in values)),
        }

    control_total = totals["v4_latent_control"]
    challenger_total = totals["adaptive_role_ordered"]
    recommendation_strictly_improves = bool(
        challenger_total["final_true_feasible"]
        > control_total["final_true_feasible"]
        or challenger_total["adaptive_loss"]
        < control_total["adaptive_loss"]
        or challenger_total["adaptive_rescue"]
        > control_total["adaptive_rescue"]
        or challenger_total["adaptive_improvement"]
        > control_total["adaptive_improvement"]
    )
    selector_checks = {
        "selection_is_outcome_free": all(
            value["selection_present"] and value["selection_outcome_free"]
            for value in challenger),
        "unsupported_factor_uses_ordered_fallback": all(
            value["cardinality_support"] == [False]
            and value["selected_coordinates"] == ["ordered"]
            for value in challenger[:2]),
        "supported_inventory_queue_use_roles": all(
            value["cardinality_support"] == [True]
            and value["selected_coordinates"] == ["role_aligned"]
            for value in challenger[2:]),
    }
    checks = {
        "all_cells_complete": all(value["complete"] for value in all_cells),
        "oracle_free": all(value["oracle_free"] for value in all_cells),
        "paired_initial_design_and_archive": bool(paired),
        "online_budget_is_positive": int(N) > int(n0),
        **selector_checks,
        "recommendation_quality_strictly_improves": (
            recommendation_strictly_improves),
        "final_true_feasible_nonworse": bool(
            challenger_total["final_true_feasible"]
            >= control_total["final_true_feasible"]),
        "adaptive_loss_nonworse": bool(
            challenger_total["adaptive_loss"]
            <= control_total["adaptive_loss"]),
        "evaluated_false_certification_nonworse": bool(
            challenger_total["evaluated_false_certificates"]
            <= control_total["evaluated_false_certificates"]),
        "audit_pool_false_certification_strictly_lower": bool(
            challenger_total["audit_pool_false_certificates"]
            < control_total["audit_pool_false_certificates"]),
        "posterior_vacuity_nonworse": bool(
            challenger_total["posterior_vacuous"]
            <= control_total["posterior_vacuous"]),
        "mean_mae_nonworse_at_least_3_of_4": bool(_count_nonworse(
            challenger, control, "median_mean_abs_error",
            multiplier=1.10, offset=0.01) >= 3),
        "variance_rmse_nonworse_at_least_3_of_4": bool(_count_nonworse(
            challenger, control, "median_variance_log_rmse",
            multiplier=1.10) >= 3),
        "feasible_regret_nonworse_at_least_3_of_4": bool(_count_nonworse(
            challenger, control, "median_feasible_regret") >= 3),
    }
    return {
        "schema_version": 1,
        "row_count": int(len(rows)),
        "expected_row_count": int(
            len(VARIANTS) * len(SCENARIOS) * expected_seeds),
        "N": int(N),
        "n0": int(n0),
        "cells": all_cells,
        "totals": totals,
        "checks": checks,
        "promote_adaptive_role_ordered": bool(all(checks.values())),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-seeds", type=int, default=5)
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    result = summarize(
        load_rows(args.root), args.expected_seeds, N=args.N, n0=args.n0)
    output = args.out or args.root / "mean_alignment_v8_sequential_gate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "promote_adaptive_role_ordered": result[
            "promote_adaptive_role_ordered"],
        "totals": result["totals"],
        "checks": result["checks"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
