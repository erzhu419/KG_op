#!/usr/bin/env python3
"""Analyze the V40 robust-certificate lexicographic terminal gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from . import analyze_mean_alignment_v39_gate as base
    from . import analyze_mean_alignment_v34_gate as common
except ImportError:  # Direct script execution.
    import analyze_mean_alignment_v39_gate as base
    import analyze_mean_alignment_v34_gate as common


VARIANTS = (
    "v35_sobol_new",
    "v39_signed_bayes_mc4",
    "v40_robust_lex_mc2",
    "v40_robust_lex_mc4",
)
ROBUST_VARIANTS = VARIANTS[2:]


def load_rows(root):
    rows = []
    for path in sorted(Path(root).rglob("result.json")):
        payload = json.loads(path.read_text())
        experiment = str(payload.get("experiment_variant", ""))
        variant = next(
            (name for name in VARIANTS if f"/{name}/" in f"/{experiment}/"),
            None,
        )
        if variant is None:
            continue
        for raw in payload.get("rows", []):
            row = dict(raw)
            row["gate_variant"] = variant
            rows.append(row)
    return rows


def _robust_contract(rows, variant, mc_samples):
    selected = [row for row in rows if row.get("gate_variant") == variant]
    return bool(
        selected
        and all(
            str(row.get("decision_contract_mode", ""))
            == "certified_lexicographic"
            and str(row.get("decision_backend_terminal_rule", ""))
            == "robust_certified_lexicographic"
            and not bool(row.get("posterior_dominance_enabled", True))
            and bool((row.get("finalist_replication") or {}).get(
                "mathematically_closed", False))
            and int((row.get("exact_kg_diagnostics") or {}).get(
                "mc_samples", -1)) == int(mc_samples)
            and bool((row.get("exact_kg_diagnostics") or {}).get(
                "ranking_uses_signed_values", False))
            and not bool((row.get("adaptive_replication_voi") or {}).get(
                "target_oracle_used", True))
            for row in selected
        )
    )


def summarize(rows, expected_seeds=1):
    present = tuple(
        variant for variant in VARIANTS
        if any(row.get("gate_variant") == variant for row in rows)
    )
    required = {"v35_sobol_new", "v39_signed_bayes_mc4"}
    if not required.issubset(present):
        raise ValueError("V40 analysis requires V35 and signed Bayes controls")
    summaries = {
        variant: base._summary(rows, variant, expected_seeds)
        for variant in present
    }
    for variant in present:
        selected = [
            row for row in rows if row.get("gate_variant") == variant]
        margins = [
            float(row["decision_backend_terminal_margin"])
            for row in selected
            if row.get("decision_backend_terminal_margin") is not None
        ]
        summaries[variant]["mean_terminal_robust_margin"] = (
            sum(margins) / len(margins) if margins else None)
    paired = base.base.base._paired_initial_contract(
        rows, present, expected_seeds)
    control = summaries["v35_sobol_new"]
    bayes = summaries["v39_signed_bayes_mc4"]
    contracts = {
        "v40_robust_lex_mc2": _robust_contract(
            rows, "v40_robust_lex_mc2", 2),
        "v40_robust_lex_mc4": _robust_contract(
            rows, "v40_robust_lex_mc4", 4),
    }
    expected = len(common.MEAN_SCENARIOS) * int(expected_seeds)
    action_shift = {}
    diagnostic = []
    promotion = []
    for variant in ROBUST_VARIANTS:
        if variant not in summaries:
            continue
        value = summaries[variant]
        action_shift[variant] = {
            "replication_delta_vs_bayes": int(
                value["selected_replications"]
                - bayes["selected_replications"]),
            "new_point_delta_vs_bayes": int(
                value["selected_new_points"]
                - bayes["selected_new_points"]),
            "fewer_replications_than_bayes": bool(
                value["selected_replications"]
                < bayes["selected_replications"]),
            "more_new_points_than_bayes": bool(
                value["selected_new_points"]
                > bayes["selected_new_points"]),
        }
        no_false = bool(
            value["audit_pool_false_certificates"] == 0
            and value["evaluated_false_certificates"] == 0
        )
        no_regression = bool(
            value["true_feasible"] >= control["true_feasible"]
            and value["adaptive_losses"] <= control["adaptive_losses"]
        )
        aligned = bool(
            action_shift[variant]["fewer_replications_than_bayes"]
            and action_shift[variant]["more_new_points_than_bayes"]
        )
        if all((value["complete"], paired, contracts[variant], no_false,
                no_regression, aligned)):
            diagnostic.append(variant)
        nonvacuous = bool(
            value["audit_pool_true_certificates"] > 0
            or value["evaluated_true_certificates"] > 0
            or value["adaptive_improvements"]
            > control["adaptive_improvements"]
        )
        if all((value["complete"], paired, contracts[variant], no_false,
                aligned, value["true_feasible"] == expected,
                value["adaptive_losses"] == 0, nonvacuous)):
            promotion.append(variant)
    return {
        "paired_initial_design_and_archive": bool(paired),
        "robust_terminal_contract": contracts,
        "variant_summaries": summaries,
        "robust_action_shift": action_shift,
        "diagnostic_eligible": diagnostic,
        "promotion_eligible": promotion,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-seeds", type=int, default=1)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = summarize(load_rows(args.root), args.expected_seeds)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
