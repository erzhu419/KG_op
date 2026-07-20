#!/usr/bin/env python3
"""Analyze the V36 evaluate-or-replicate information gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from . import analyze_mean_alignment_v35_gate as base
except ImportError:  # Direct script execution.
    import analyze_mean_alignment_v35_gate as base


VARIANTS = (
    "v35_sobol_new",
    "v36_joint_new_only",
    "v36_joint_rep4",
    "v36_joint_rep8",
)
REPLICATION_VARIANTS = VARIANTS[2:]


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


def _integer(row, key):
    value = row.get(key)
    if value is None:
        value = (row.get("adaptive_replication_voi") or {}).get(key)
    return int(value or 0)


def _oracle_free_action_contract(row):
    action = dict(row.get("adaptive_replication_voi") or {})
    backend = dict(row.get("decision_backend_contract") or {})
    return bool(
        not action.get("target_oracle_used", True)
        and not backend.get("target_oracle_used", True)
        and backend.get("online_updates_use_budgeted_target_observations_only")
    )


def _variant_summary(rows, variant, expected_seeds):
    selected = [row for row in rows if row.get("gate_variant") == variant]
    expected = len(base.base.MEAN_SCENARIOS) * int(expected_seeds)
    audits = [
        dict(row.get("boundary_raw_pool_truth_diagnostics") or {})
        for row in selected
    ]
    certificate_audits = [
        dict(row.get("certificate_outcome_audit") or {})
        for row in selected
    ]
    replication_count = sum(
        _integer(row, "selected_replication_count")
        or _integer(row, "adaptive_replication_selected_count")
        for row in selected
    )
    new_point_count = sum(
        _integer(row, "selected_new_point_count")
        or _integer(row, "adaptive_new_point_selected_count")
        for row in selected
    )
    return {
        "count": int(len(selected)),
        "complete": bool(len(selected) == expected),
        "true_feasible": int(sum(bool(
            row.get("true_feasible", False)) for row in selected)),
        "adaptive_losses": int(sum(bool(
            row.get("adaptive_loss", False)) for row in selected)),
        "adaptive_improvements": int(sum(bool(
            row.get("adaptive_improves_initial_best", False))
            for row in selected)),
        "audit_pool_true_certificates": int(sum(int(
            audit.get("boundary_raw_pool_true_certified_count", 0) or 0)
            for audit in audits)),
        "audit_pool_false_certificates": int(sum(int(
            audit.get("boundary_raw_pool_false_certified_count", 0) or 0)
            for audit in audits)),
        "evaluated_true_certificates": int(sum(int(
            audit.get("certified_true_feasible_count", 0) or 0)
            for audit in certificate_audits)),
        "evaluated_false_certificates": int(sum(int(
            audit.get("false_certificate_count", 0) or 0)
            for audit in certificate_audits)),
        "selected_replications": int(replication_count),
        "selected_new_points": int(new_point_count),
        "oracle_free_action_contract": bool(
            selected and all(_oracle_free_action_contract(row)
                             for row in selected)),
        "replication_enabled": bool(selected and all(bool(
            (row.get("adaptive_replication_voi") or {}).get("enabled", False)
        ) for row in selected)),
    }


def _paired_initial_contract(rows, variants, expected_seeds):
    for scenario in base.base.MEAN_SCENARIOS:
        for seed in range(int(expected_seeds)):
            group = [
                row for row in rows
                if base.base._scenario(row) == scenario
                and int(row.get("seed", -1)) == seed
                and row.get("gate_variant") in variants
            ]
            if len(group) != len(variants):
                return False
            initial = {
                str((row.get("source_target_adaptation_contract") or {}).get(
                    "target_initial_design_fingerprint", "missing"))
                for row in group
            }
            archive = {
                str((row.get("source_target_adaptation_contract") or {}).get(
                    "source_archive_fingerprint", "missing"))
                for row in group
            }
            if len(initial) != 1 or "missing" in initial:
                return False
            if len(archive) != 1 or "missing" in archive:
                return False
    return True


def summarize(rows, expected_seeds=5):
    present = tuple(
        variant for variant in VARIANTS
        if any(row.get("gate_variant") == variant for row in rows)
    )
    if "v35_sobol_new" not in present:
        raise ValueError("V36 analysis requires v35_sobol_new")
    summaries = {
        variant: _variant_summary(rows, variant, expected_seeds)
        for variant in present
    }
    paired = _paired_initial_contract(rows, present, expected_seeds)
    control = summaries["v35_sobol_new"]
    new_only = summaries.get("v36_joint_new_only")
    new_only_action_contract = bool(
        new_only is not None
        and not new_only["replication_enabled"]
        and new_only["selected_replications"] == 0
    )
    diagnostic = []
    promotion = []
    expected = len(base.base.MEAN_SCENARIOS) * int(expected_seeds)
    for variant in REPLICATION_VARIANTS:
        if variant not in summaries:
            continue
        value = summaries[variant]
        no_false = bool(
            value["audit_pool_false_certificates"] == 0
            and value["evaluated_false_certificates"] == 0
        )
        no_feasibility_regression = bool(
            value["true_feasible"] >= control["true_feasible"]
            and value["adaptive_losses"] <= control["adaptive_losses"]
        )
        information_gain = bool(
            value["audit_pool_true_certificates"]
            > control["audit_pool_true_certificates"]
            or value["evaluated_true_certificates"]
            > control["evaluated_true_certificates"]
            or value["true_feasible"] > control["true_feasible"]
            or value["adaptive_losses"] < control["adaptive_losses"]
        )
        mechanism = bool(
            value["replication_enabled"]
            and value["selected_replications"] > 0
            and value["oracle_free_action_contract"]
        )
        if all((value["complete"], paired, new_only_action_contract, no_false,
                no_feasibility_regression, mechanism, information_gain)):
            diagnostic.append(variant)
        nonvacuous = bool(
            value["audit_pool_true_certificates"] > 0
            or value["evaluated_true_certificates"] > 0
        )
        if all((value["complete"], paired, new_only_action_contract,
                no_false, mechanism,
                value["true_feasible"] == expected,
                value["adaptive_losses"] == 0,
                nonvacuous, information_gain)):
            promotion.append(variant)
    return {
        "paired_initial_design_and_archive": bool(paired),
        "new_only_action_contract": new_only_action_contract,
        "variant_summaries": summaries,
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
