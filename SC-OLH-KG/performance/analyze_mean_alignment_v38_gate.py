#!/usr/bin/env python3
"""Analyze the V38 exact-refit evaluate-or-replicate gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from . import analyze_mean_alignment_v36_gate as base
except ImportError:  # Direct script execution.
    import analyze_mean_alignment_v36_gate as base


VARIANTS = (
    "v35_sobol_new",
    "v37_cluster_rep4",
    "v38_exact_rep4",
    "v38_exact_rep8",
)
EXACT_VARIANTS = VARIANTS[2:]


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


def _source_prior(row):
    numerics = list(row.get("gpr_numerics") or [])
    return (
        dict(numerics[1].get("source_parametric_prior") or {})
        if len(numerics) > 1 else {}
    )


def _cluster_contract(rows, variant):
    selected = [row for row in rows if row.get("gate_variant") == variant]
    priors = [_source_prior(row) for row in selected]
    return bool(
        selected
        and all(
            prior.get("source_mean_sandwich_clustered_replicates", False)
            and not prior.get("target_oracle_used_for_misspecification", True)
            for prior in priors
        )
    )


def _exact_contract(rows, variant):
    selected = [row for row in rows if row.get("gate_variant") == variant]
    return bool(
        selected
        and all(
            str((row.get("decision_backend_contract") or {}).get(
                "backend", "")) == "sobol_exact_joint_voi"
            and bool((row.get("adaptive_replication_voi") or {}).get(
                "exact_refit_action_value", False))
            and bool((row.get("adaptive_replication_voi") or {}).get(
                "unified_exact_voi", False))
            and int((row.get("exact_kg_diagnostics") or {}).get(
                "n_iterations", 0)) > 0
            and not bool((row.get("adaptive_replication_voi") or {}).get(
                "target_oracle_used", True))
            for row in selected
        )
    )


def summarize(rows, expected_seeds=5):
    present = tuple(
        variant for variant in VARIANTS
        if any(row.get("gate_variant") == variant for row in rows)
    )
    required = {"v35_sobol_new", "v37_cluster_rep4"}
    if not required.issubset(present):
        raise ValueError("V38 analysis requires paired V35 and V37 controls")
    summaries = {
        variant: base._variant_summary(rows, variant, expected_seeds)
        for variant in present
    }
    paired = base._paired_initial_contract(rows, present, expected_seeds)
    cluster_contract = {
        variant: _cluster_contract(rows, variant)
        for variant in present if variant != "v35_sobol_new"
    }
    exact_contract = {
        variant: _exact_contract(rows, variant)
        for variant in EXACT_VARIANTS if variant in present
    }
    approximate = summaries["v37_cluster_rep4"]
    control = summaries["v35_sobol_new"]
    action_shift = {}
    diagnostic = []
    promotion = []
    expected = len(base.base.base.MEAN_SCENARIOS) * int(expected_seeds)
    for variant in EXACT_VARIANTS:
        if variant not in summaries:
            continue
        value = summaries[variant]
        action_shift[variant] = {
            "fewer_replications_than_v37": bool(
                value["selected_replications"]
                < approximate["selected_replications"]),
            "more_new_points_than_v37": bool(
                value["selected_new_points"]
                > approximate["selected_new_points"]),
            "replication_delta": int(
                value["selected_replications"]
                - approximate["selected_replications"]),
            "new_point_delta": int(
                value["selected_new_points"]
                - approximate["selected_new_points"]),
        }
        no_false = bool(
            value["audit_pool_false_certificates"] == 0
            and value["evaluated_false_certificates"] == 0
        )
        no_regression = bool(
            value["true_feasible"] >= control["true_feasible"]
            and value["adaptive_losses"] <= control["adaptive_losses"]
        )
        exact_mechanism = bool(
            exact_contract.get(variant, False)
            and cluster_contract.get(variant, False)
            and value["oracle_free_action_contract"]
        )
        corrected_action_value = bool(
            action_shift[variant]["fewer_replications_than_v37"]
            and action_shift[variant]["more_new_points_than_v37"]
        )
        if all((value["complete"], paired, no_false, no_regression,
                exact_mechanism, corrected_action_value)):
            diagnostic.append(variant)
        nonvacuous = bool(
            value["audit_pool_true_certificates"] > 0
            or value["evaluated_true_certificates"] > 0
            or value["adaptive_improvements"] > control[
                "adaptive_improvements"]
        )
        if all((value["complete"], paired, no_false, exact_mechanism,
                corrected_action_value, value["true_feasible"] == expected,
                value["adaptive_losses"] == 0, nonvacuous)):
            promotion.append(variant)
    return {
        "paired_initial_design_and_archive": bool(paired),
        "clustered_hc3_contract": cluster_contract,
        "exact_refit_contract": exact_contract,
        "variant_summaries": summaries,
        "exact_refit_action_shift": action_shift,
        "diagnostic_eligible": diagnostic,
        "promotion_eligible": promotion,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-seeds", type=int, default=5)
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
