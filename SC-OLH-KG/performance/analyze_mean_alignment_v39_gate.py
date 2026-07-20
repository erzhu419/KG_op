#!/usr/bin/env python3
"""Analyze the V39 signed common-random exact-VOI gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from . import analyze_mean_alignment_v38_gate as base
    from . import analyze_mean_alignment_v34_gate as common
except ImportError:  # Direct script execution.
    import analyze_mean_alignment_v38_gate as base
    import analyze_mean_alignment_v34_gate as common


VARIANTS = (
    "v35_sobol_new",
    "v38_clipped_mc2",
    "v39_signed_mc2",
    "v39_signed_mc4",
)
SIGNED_VARIANTS = VARIANTS[2:]


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


def _signed_contract(rows, variant, mc_samples):
    selected = [row for row in rows if row.get("gate_variant") == variant]
    return bool(
        selected
        and all(
            str((row.get("decision_backend_contract") or {}).get(
                "backend", "")) == "sobol_exact_joint_voi"
            and bool((row.get("adaptive_replication_voi") or {}).get(
                "exact_refit_action_value", False))
            and not bool((row.get("exact_kg_diagnostics") or {}).get(
                "clip_negative", True))
            and bool((row.get("exact_kg_diagnostics") or {}).get(
                "ranking_uses_signed_values", False))
            and int((row.get("exact_kg_diagnostics") or {}).get(
                "n_iterations", 0)) > 0
            and int((row.get("exact_kg_diagnostics") or {}).get(
                "mc_samples", -1)) == int(mc_samples)
            and not bool((row.get("adaptive_replication_voi") or {}).get(
                "target_oracle_used", True))
            for row in selected
        )
    )


def _summary(rows, variant, expected_seeds):
    result = base.base._variant_summary(rows, variant, expected_seeds)
    selected = [row for row in rows if row.get("gate_variant") == variant]
    regrets = np.asarray([
        (
            np.nan
            if row.get("feasible_simple_regret") is None
            else float(row.get("feasible_simple_regret"))
        )
        for row in selected
    ], dtype=float)
    regrets = regrets[np.isfinite(regrets)]
    diagnostics = [dict(row.get("exact_kg_diagnostics") or {})
                   for row in selected]
    result.update({
        "median_feasible_regret": (
            float(np.median(regrets)) if len(regrets) else None),
        "mean_wall_time_sec": (
            float(np.mean([float(row.get("wall_time_sec", 0.0))
                           for row in selected]))
            if selected else None),
        "selected_nonpositive": int(sum(int(
            diag.get("selected_nonpositive_count", 0) or 0)
            for diag in diagnostics)),
        "selected_nonpositive_replications": int(sum(int(
            diag.get("selected_nonpositive_replication_count", 0) or 0)
            for diag in diagnostics)),
        "selected_clipped_replications": int(sum(int(
            diag.get("selected_clipped_replication_count", 0) or 0)
            for diag in diagnostics)),
    })
    return result


def summarize(rows, expected_seeds=1):
    present = tuple(
        variant for variant in VARIANTS
        if any(row.get("gate_variant") == variant for row in rows)
    )
    required = {"v35_sobol_new", "v38_clipped_mc2"}
    if not required.issubset(present):
        raise ValueError("V39 analysis requires V35 and clipped V38 controls")
    summaries = {
        variant: _summary(rows, variant, expected_seeds)
        for variant in present
    }
    paired = base.base._paired_initial_contract(
        rows, present, expected_seeds)
    control = summaries["v35_sobol_new"]
    clipped = summaries["v38_clipped_mc2"]
    contracts = {
        "v39_signed_mc2": _signed_contract(
            rows, "v39_signed_mc2", 2),
        "v39_signed_mc4": _signed_contract(
            rows, "v39_signed_mc4", 4),
    }
    expected = len(common.MEAN_SCENARIOS) * int(expected_seeds)
    diagnostic = []
    promotion = []
    action_shift = {}
    for variant in SIGNED_VARIANTS:
        if variant not in summaries:
            continue
        value = summaries[variant]
        action_shift[variant] = {
            "replication_delta_vs_clipped": int(
                value["selected_replications"]
                - clipped["selected_replications"]),
            "new_point_delta_vs_clipped": int(
                value["selected_new_points"]
                - clipped["selected_new_points"]),
            "fewer_replications_than_clipped": bool(
                value["selected_replications"]
                < clipped["selected_replications"]),
            "more_new_points_than_clipped": bool(
                value["selected_new_points"]
                > clipped["selected_new_points"]),
        }
        no_false = bool(
            value["audit_pool_false_certificates"] == 0
            and value["evaluated_false_certificates"] == 0
        )
        no_regression = bool(
            value["true_feasible"] >= control["true_feasible"]
            and value["adaptive_losses"] <= control["adaptive_losses"]
        )
        corrected = bool(
            action_shift[variant]["fewer_replications_than_clipped"]
            and action_shift[variant]["more_new_points_than_clipped"]
            and value["selected_clipped_replications"] == 0
        )
        if all((value["complete"], paired, contracts[variant], no_false,
                no_regression, corrected)):
            diagnostic.append(variant)
        nonvacuous = bool(
            value["audit_pool_true_certificates"] > 0
            or value["evaluated_true_certificates"] > 0
            or value["adaptive_improvements"]
            > control["adaptive_improvements"]
        )
        if all((value["complete"], paired, contracts[variant], no_false,
                corrected, value["true_feasible"] == expected,
                value["adaptive_losses"] == 0, nonvacuous)):
            promotion.append(variant)
    return {
        "paired_initial_design_and_archive": bool(paired),
        "signed_exact_contract": contracts,
        "variant_summaries": summaries,
        "signed_action_shift": action_shift,
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
