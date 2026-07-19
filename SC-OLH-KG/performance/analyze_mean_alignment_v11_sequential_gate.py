#!/usr/bin/env python3
"""Analyze the paired V11 support-adaptive sequential gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .analyze_mean_alignment_v11_gate import (
        CHALLENGER,
        SCENARIOS,
        VARIANTS,
        _cell as _offline_cell,
        _scenario,
        load_rows,
    )
    from .analyze_mean_alignment_v9_gate import _median
except ImportError:  # Direct script execution.
    from analyze_mean_alignment_v11_gate import (
        CHALLENGER,
        SCENARIOS,
        VARIANTS,
        _cell as _offline_cell,
        _scenario,
        load_rows,
    )
    from analyze_mean_alignment_v9_gate import _median


def _cell(rows, variant, scenario, expected_seeds, target_count):
    value = _offline_cell(
        rows, variant, scenario, expected_seeds, target_count)
    group = [
        row for row in rows
        if row["gate_variant"] == variant and _scenario(row) == scenario
    ]
    value.update({
        "initial_has_true_feasible_count": int(sum(bool(
            row.get("initial_has_true_feasible", False)) for row in group)),
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


def summarize(rows, expected_seeds=5, *, N=20, n0=10):
    cells = {
        (variant, *scenario): _cell(
            rows, variant, scenario, expected_seeds, N)
        for variant in VARIANTS for scenario in SCENARIOS
    }
    grouped = {
        variant: [cells[(variant, *scenario)] for scenario in SCENARIOS]
        for variant in VARIANTS
    }
    all_cells = list(cells.values())
    totals = {}
    for variant, values in grouped.items():
        totals[variant] = {
            "initial_has_true_feasible": int(sum(
                item["initial_has_true_feasible_count"] for item in values)),
            "final_true_feasible": int(sum(
                item["true_feasible_count"] for item in values)),
            "adaptive_rescue": int(sum(
                item["adaptive_rescue_count"] for item in values)),
            "adaptive_loss": int(sum(
                item["adaptive_loss_count"] for item in values)),
            "adaptive_improvement": int(sum(
                item["adaptive_improvement_count"] for item in values)),
            "posterior_vacuous": int(sum(
                item["posterior_vacuous_count"] for item in values)),
            "posterior_certified_evaluated": int(sum(
                item["posterior_certified_evaluated_count"]
                for item in values)),
            "evaluated_false_certificates": int(sum(
                item["evaluated_false_certificate_count"]
                for item in values)),
            "audit_pool_false_certificates": int(sum(
                item["audit_pool_false_certificate_count"]
                for item in values)),
        }
    v4 = totals["v4_mixture_control"]
    v8 = totals["v8_mixture_control"]
    challenger = totals[CHALLENGER]
    strict_gain = bool(
        challenger["final_true_feasible"]
        > max(v4["final_true_feasible"], v8["final_true_feasible"])
        or challenger["adaptive_loss"]
        < min(v4["adaptive_loss"], v8["adaptive_loss"])
        or challenger["adaptive_rescue"]
        > max(v4["adaptive_rescue"], v8["adaptive_rescue"])
        or challenger["adaptive_improvement"]
        > max(v4["adaptive_improvement"], v8["adaptive_improvement"])
    )
    paired = all(
        len({tuple(item["initial_design_fingerprints"])
             for item in scenario_cells}) == 1
        and len({tuple(item["source_archive_fingerprints"])
                for item in scenario_cells}) == 1
        for scenario_cells in zip(*(grouped[variant] for variant in VARIANTS))
    )
    values = grouped[CHALLENGER]
    controls = [grouped["v4_mixture_control"], grouped["v8_mixture_control"]]
    regret_nonworse = all(sum(
        left["median_feasible_regret"] is not None
        and right["median_feasible_regret"] is not None
        and left["median_feasible_regret"]
        <= right["median_feasible_regret"] + 1e-12
        for left, right in zip(values, control)
    ) >= 3 for control in controls)
    checks = {
        "all_cells_complete": all(item["complete"] for item in all_cells),
        "oracle_free": all(item["oracle_free"] for item in all_cells),
        "paired_initial_design_and_archive": bool(paired),
        "online_budget_is_positive": int(N) > int(n0),
        "support_adaptive_contract_retained": all(
            item["support_adaptive_requested"]
            and item["support_selection_outcome_free"]
            and item["target_posterior_updated"]
            and item["posterior_weights_valid"]
            for item in values),
        "recommendation_quality_strictly_improves": strict_gain,
        "final_true_feasible_nonworse_than_both": bool(
            challenger["final_true_feasible"] >= max(
                v4["final_true_feasible"], v8["final_true_feasible"])),
        "adaptive_loss_nonworse_than_both": bool(
            challenger["adaptive_loss"]
            <= min(v4["adaptive_loss"], v8["adaptive_loss"])),
        "evaluated_false_certification_nonworse_than_both": bool(
            challenger["evaluated_false_certificates"] <= min(
                v4["evaluated_false_certificates"],
                v8["evaluated_false_certificates"])),
        "audit_false_certification_nonworse_than_v8": bool(
            challenger["audit_pool_false_certificates"]
            <= v8["audit_pool_false_certificates"]),
        "posterior_vacuity_nonworse_than_both": bool(
            challenger["posterior_vacuous"]
            <= min(v4["posterior_vacuous"], v8["posterior_vacuous"])),
        "feasible_regret_nonworse_at_least_3_of_4": bool(regret_nonworse),
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
        "promote_support_adaptive_aggregate": bool(all(checks.values())),
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
    output = args.out or args.root / "mean_alignment_v11_sequential_gate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "promote_support_adaptive_aggregate": result[
            "promote_support_adaptive_aggregate"],
        "totals": result["totals"],
        "checks": result["checks"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
