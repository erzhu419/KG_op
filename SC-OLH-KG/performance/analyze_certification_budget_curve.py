#!/usr/bin/env python3
"""Analyze the fixed-coordinate N=40/80 certification budget curve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .analyze_hvd_decision_gate import (
        _median,
        load_rows,
        paired_factor_effect,
        summarize_cell,
    )
except ImportError:  # Direct script execution.
    from analyze_hvd_decision_gate import (
        _median,
        load_rows,
        paired_factor_effect,
        summarize_cell,
    )


def _final_margin(row):
    decomposition = row.get("certification_margin_decomposition") or {}
    minimum = decomposition.get("minimum_margin") or {}
    return (minimum.get("final_certificate") or {}).get("margin")


def _parse_root(value):
    label, separator, raw_path = str(value).partition("=")
    if not separator:
        raise ValueError("budget roots must use N=PATH")
    budget = int(label)
    if budget <= 0 or not raw_path:
        raise ValueError("budget roots require a positive N and nonempty path")
    return budget, Path(raw_path)


def _cell_summary(rows):
    result = summarize_cell(rows)
    result.update({
        "median_final_certificate_margin": _median(
            _final_margin(row) for row in rows),
        "median_pool_min_true_margin": _median(
            (row.get("truth_pool_diagnostics") or {}).get(
                "mean_pool_min_true_margin")
            for row in rows),
        "median_pool_min_posterior_margin": _median(
            (row.get("truth_pool_diagnostics") or {}).get(
                "mean_pool_min_posterior_margin")
            for row in rows),
    })
    return result


def _budget_effect(rows, control_budget, challenger_budget, action):
    index = {}
    for row in rows:
        if row["action"] != action:
            continue
        key = (row["heldout"], row["shock_label"], int(row["seed"]))
        index.setdefault(key, {})[int(row["budget"])] = row
    pairs = [
        item for item in index.values()
        if control_budget in item and challenger_budget in item
    ]
    certificate_delta = []
    final_margin_delta = []
    feasibility_wins = feasibility_losses = 0
    regret_wins = regret_losses = 0
    for pair in pairs:
        control = pair[control_budget]
        challenger = pair[challenger_budget]
        control_feasible = bool(control.get("true_feasible", False))
        challenger_feasible = bool(challenger.get("true_feasible", False))
        feasibility_wins += int(challenger_feasible and not control_feasible)
        feasibility_losses += int(control_feasible and not challenger_feasible)
        if control_feasible and challenger_feasible:
            first = control.get("feasible_simple_regret")
            second = challenger.get("feasible_simple_regret")
            if first is not None and second is not None:
                regret_wins += int(float(second) < float(first) - 1e-12)
                regret_losses += int(float(first) < float(second) - 1e-12)
        certificate_delta.append(
            int(not bool(challenger.get(
                "posterior_certificate_vacuous", True)))
            - int(not bool(control.get(
                "posterior_certificate_vacuous", True)))
        )
        first_margin = _final_margin(control)
        second_margin = _final_margin(challenger)
        if first_margin is not None and second_margin is not None:
            final_margin_delta.append(
                float(second_margin) - float(first_margin))
    return {
        "control_budget": int(control_budget),
        "challenger_budget": int(challenger_budget),
        "action": str(action),
        "pair_count": int(len(pairs)),
        "certificate_nonvacuous_net": int(sum(certificate_delta)),
        "median_final_certificate_margin_delta": _median(final_margin_delta),
        "feasibility_wins": int(feasibility_wins),
        "feasibility_losses": int(feasibility_losses),
        "conditional_regret_wins": int(regret_wins),
        "conditional_regret_losses": int(regret_losses),
    }


def analyze(roots, expected_count=80):
    rows = []
    errors = []
    root_map = {}
    for budget, root in roots:
        root_map[str(budget)] = str(root)
        loaded, load_errors = load_rows(root)
        for row in loaded:
            recorded_budget = row.get("N")
            if recorded_budget is not None and int(recorded_budget) != budget:
                errors.append({
                    "path": row.get("result_path"),
                    "error": (
                        f"registered budget {budget} disagrees with "
                        f"result N={recorded_budget}"),
                })
                continue
            row["budget"] = int(budget)
            rows.append(row)
        errors.extend(load_errors)

    budgets = sorted({int(row["budget"]) for row in rows})
    grouped = {}
    for row in rows:
        key = (
            int(row["budget"]), row["heldout"], row["shock_label"],
            row["action"],
        )
        grouped.setdefault(key, []).append(row)
    cells = {
        "/".join(map(str, key)): _cell_summary(items)
        for key, items in sorted(grouped.items())
    }

    action_effects = {}
    action_factor = {"action": ("new_only", "joint_voi")}
    for budget in budgets:
        action_effects[str(budget)] = paired_factor_effect(
            [row for row in rows if row["budget"] == budget],
            "action",
            action_factor,
        )

    budget_effects = {}
    if len(budgets) >= 2:
        first, last = budgets[0], budgets[-1]
        budget_effects = {
            action: _budget_effect(rows, first, last, action)
            for action in ("new_only", "joint_voi")
        }

    primary_budget = max(budgets) if budgets else 0
    primary = [
        row for row in rows
        if row["budget"] == primary_budget and row["action"] == "joint_voi"
    ]
    primary_nonvacuous = sum(
        not bool(row.get("posterior_certificate_vacuous", True))
        for row in primary)
    primary_false = sum(
        int(row.get("false_certificate_count") or 0) for row in primary)
    primary_gain = sum(
        bool(row.get("adaptive_rescue", False))
        or bool(row.get("adaptive_improves_initial_best", False))
        for row in primary)
    primary_loss = sum(
        bool(row.get("adaptive_loss", False)) for row in primary)
    action_effect = action_effects.get(str(primary_budget), {})
    budget_effect = budget_effects.get("joint_voi", {})
    margin_delta = budget_effect.get("median_final_certificate_margin_delta")
    criteria = {
        "complete_expected_matrix": bool(
            not errors and len(rows) == int(expected_count)),
        "primary_has_nonvacuous_certificate": bool(primary_nonvacuous > 0),
        "primary_has_zero_false_certificates": bool(primary_false == 0),
        "primary_has_no_net_adaptive_loss": bool(primary_gain >= primary_loss),
        "joint_voi_improves_certificate_count_at_primary_budget": bool(
            action_effect.get("certificate_nonvacuous_net", 0) > 0),
        "certificate_count_is_monotone_in_budget": bool(
            budget_effect.get("certificate_nonvacuous_net", -1) >= 0),
        "certificate_margin_improves_in_budget": bool(
            margin_delta is not None and margin_delta < 0.0),
    }
    return {
        "schema_version": 1,
        "roots": root_map,
        "expected_count": int(expected_count),
        "parsed_count": int(len(rows)),
        "errors": errors,
        "budgets": budgets,
        "cells": cells,
        "action_effects": action_effects,
        "budget_effects": budget_effects,
        "gate": {
            "promote_joint_voi": bool(all(criteria.values())),
            "coordinate_information_sufficient_at_primary_budget": bool(
                primary_nonvacuous > 0 and primary_false == 0),
            "primary_budget": int(primary_budget),
            "primary_count": int(len(primary)),
            "primary_nonvacuous_count": int(primary_nonvacuous),
            "primary_false_certificate_count": int(primary_false),
            "primary_online_gain_count": int(primary_gain),
            "primary_adaptive_loss_count": int(primary_loss),
            "criteria": criteria,
        },
    }


def markdown_report(result):
    gate = result["gate"]
    lines = [
        "# Certification budget curve",
        "",
        f"- Parsed: {result['parsed_count']}/{result['expected_count']}",
        f"- Budgets: {result['budgets']}",
        f"- Promote joint VOI: `{gate['promote_joint_voi']}`",
        "- Coordinate information sufficient at largest budget: "
        f"`{gate['coordinate_information_sufficient_at_primary_budget']}`",
        f"- Largest-budget certificates / false certificates: "
        f"{gate['primary_nonvacuous_count']} / "
        f"{gate['primary_false_certificate_count']}",
        f"- Largest-budget online gains / losses: "
        f"{gate['primary_online_gain_count']} / "
        f"{gate['primary_adaptive_loss_count']}",
        "",
        "## Action effects",
        "",
        "| N | pairs | feasible +/- | regret +/- | certificate net | log-RMSE delta |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for budget, effect in result["action_effects"].items():
        lines.append(
            f"| {budget} | {effect['pair_count']} | "
            f"{effect['feasibility_wins']}/{effect['feasibility_losses']} | "
            f"{effect['conditional_regret_wins']}/"
            f"{effect['conditional_regret_losses']} | "
            f"{effect['certificate_nonvacuous_net']} | "
            f"{effect['median_variance_log_rmse_delta']} |"
        )
    lines.extend([
        "",
        "## Budget effects",
        "",
        "| action | pairs | feasible +/- | regret +/- | certificate net | final-margin delta |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for action, effect in result["budget_effects"].items():
        lines.append(
            f"| {action} | {effect['pair_count']} | "
            f"{effect['feasibility_wins']}/{effect['feasibility_losses']} | "
            f"{effect['conditional_regret_wins']}/"
            f"{effect['conditional_regret_losses']} | "
            f"{effect['certificate_nonvacuous_net']} | "
            f"{effect['median_final_certificate_margin_delta']} |"
        )
    lines.extend(["", "## Criteria", ""])
    for key, value in gate["criteria"].items():
        lines.append(f"- [{'x' if value else ' '}] {key}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", action="append", required=True,
        help="registered budget root in N=PATH form; repeat for each budget",
    )
    parser.add_argument("--expected-count", type=int, default=80)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()
    roots = [_parse_root(value) for value in args.root]
    result = analyze(roots, expected_count=args.expected_count)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(markdown_report(result), encoding="utf-8")
    print(json.dumps({
        "parsed_count": result["parsed_count"],
        "promote_joint_voi": result["gate"]["promote_joint_voi"],
        "out_json": str(args.out_json),
        "out_md": str(args.out_md),
    }))


if __name__ == "__main__":
    main()
