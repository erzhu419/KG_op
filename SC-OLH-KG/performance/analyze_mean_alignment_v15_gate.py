#!/usr/bin/env python3
"""Analyze V15 independent mean/HVD task-posterior gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .analyze_mean_alignment_v8_gate import SCENARIOS, _scenario
    from .analyze_mean_alignment_v12_gate import _cell as _v12_cell
    from .analyze_mean_alignment_v14_gate import _variance_head_signature
except ImportError:  # Direct script execution.
    from analyze_mean_alignment_v8_gate import SCENARIOS, _scenario
    from analyze_mean_alignment_v12_gate import _cell as _v12_cell
    from analyze_mean_alignment_v14_gate import _variance_head_signature


VARIANTS = (
    "v4_shared_control",
    "v8_shared_control",
    "bounded_shared_control",
    "isolated_linear",
    "isolated_linear_contrast",
    "isolated_tanh",
    "isolated_tanh_contrast",
)
ISOLATED_VARIANTS = set(VARIANTS[3:])
CHALLENGERS = VARIANTS[3:]
CONTRAST_VARIANTS = {
    "isolated_linear_contrast",
    "isolated_tanh_contrast",
}


def load_rows(root):
    rows = []
    for path in sorted(Path(root).rglob("result.json")):
        payload = json.loads(path.read_text())
        for row in payload.get("rows", []):
            item = dict(row)
            variant = str(payload.get("experiment_variant", ""))
            item["gate_variant"] = next(
                (name for name in VARIANTS if f"/{name}/" in f"/{variant}/"),
                "missing",
            )
            rows.append(item)
    return [row for row in rows if row["gate_variant"] in VARIANTS]


def _cell(rows, variant, scenario, expected_seeds):
    value = _v12_cell(rows, variant, scenario, expected_seeds)
    group = [
        row for row in rows
        if row["gate_variant"] == variant and _scenario(row) == scenario
    ]
    task = [dict(row.get("task_posterior") or {}) for row in group]
    variance_posteriors = [
        dict(item.get("variance_structure_posterior") or {})
        for item in task
    ]
    audits = [dict(row.get("variance_calibration_audit") or {})
              for row in group]
    value.update({
        "variance_task_posterior_modes": sorted(set(str(
            item.get("variance_structure_posterior_mode", "missing"))
            for item in task)),
        "variance_task_posterior_statuses": sorted(set(str(
            item.get("status", "missing")) for item in variance_posteriors)),
        "variance_task_evidence_counts": sorted(set(int(
            item.get("evidence_count", -1)) for item in variance_posteriors)),
        "variance_task_effective_dofs": sorted(set(int(
            item.get("effective_dof", -1)) for item in variance_posteriors)),
        "variance_task_uses_target_mean": sorted(set(bool(
            item.get("target_mean_used", True)) for item in variance_posteriors)),
        "variance_posterior_sources": sorted(set(str(
            item.get("posterior_source", "missing")) for item in audits)),
    })
    return value


def _row_key(row):
    return (*_scenario(row), int(row["seed"]))


def _numeric_equal(left, right, key, tolerance=1e-12):
    a = left.get(key)
    b = right.get(key)
    if a is None or b is None:
        return a is None and b is None
    return bool(np.isclose(float(a), float(b), rtol=tolerance, atol=tolerance))


def _variance_head_exact(rows, variant):
    reference = {
        _row_key(row): row for row in rows
        if row["gate_variant"] == "isolated_linear"
    }
    candidate = {
        _row_key(row): row for row in rows
        if row["gate_variant"] == variant
    }
    if not reference or set(reference) != set(candidate):
        return False
    metrics = (
        "variance_log_rmse",
        "certified_variance_log_rmse",
        "median_predicted_true_variance_ratio",
        "median_certified_true_variance_ratio",
        "variance_upper_coverage",
    )
    return all(
        all(_numeric_equal(reference[key], candidate[key], metric)
            for metric in metrics)
        and _variance_head_signature(reference[key])
        == _variance_head_signature(candidate[key])
        for key in reference
    )


def _count_lower_nonworse(left, right, key, multiplier=1.0, offset=0.0):
    return sum(
        item[key] is not None and reference[key] is not None
        and item[key] <= multiplier * reference[key] + offset
        for item, reference in zip(left, right)
    )


def _count_higher_nonworse(left, right, key, offset=0.02):
    return sum(
        item[key] is not None and reference[key] is not None
        and item[key] + offset >= reference[key]
        for item, reference in zip(left, right)
    )


def summarize(rows, expected_seeds=5):
    cells = {
        (variant, *scenario): _cell(
            rows, variant, scenario, expected_seeds)
        for variant in VARIANTS for scenario in SCENARIOS
    }
    grouped = {
        variant: [cells[(variant, *scenario)] for scenario in SCENARIOS]
        for variant in VARIANTS
    }
    paired = all(
        len({tuple(cell["initial_design_fingerprints"])
             for cell in scenario_cells}) == 1
        and len({tuple(cell["source_archive_fingerprints"])
                 for cell in scenario_cells}) == 1
        for scenario_cells in zip(*(grouped[value] for value in VARIANTS))
    )
    totals = {
        variant: {
            "true_feasible": int(sum(
                item["true_feasible_count"] for item in values)),
            "audit_pool_false_certificates": int(sum(
                item["audit_pool_false_certificate_count"]
                for item in values)),
        }
        for variant, values in grouped.items()
    }
    global_checks = {
        "all_cells_complete": all(
            item["complete"] for item in cells.values()),
        "oracle_free": all(item["oracle_free"] for item in cells.values()),
        "paired_initial_design_and_archive": bool(paired),
    }
    v8 = grouped["v8_shared_control"]
    bounded = grouped["bounded_shared_control"]
    checks = {}
    for variant in CHALLENGERS:
        values = grouped[variant]
        isolation = all(
            item["variance_task_posterior_modes"] == ["replication_only"]
            and item["variance_task_posterior_statuses"]
            == ["frozen_source_prior"]
            and item["variance_task_evidence_counts"] == [0]
            and item["variance_task_effective_dofs"] == [0]
            and item["variance_task_uses_target_mean"] == [False]
            and item["variance_posterior_sources"]
            == ["replication_variance_task_posterior_hvd_mixture"]
            for item in values
        )
        contrast_contract = (
            all(
                item["contrast_uncertainty_monotone"]
                and item["contrast_rank_bounded"]
                and item["contrast_source_only"]
                for item in values
            )
            if variant in CONTRAST_VARIANTS
            else True
        )
        checks[variant] = {
            **global_checks,
            "independent_variance_task_posterior_contract": bool(isolation),
            "source_mean_misspecification_contract": bool(contrast_contract),
            "variance_head_exactly_invariant_to_mean_coordinate": bool(
                _variance_head_exact(rows, variant)),
            "true_feasible_nonworse_than_v8": bool(
                totals[variant]["true_feasible"]
                >= totals["v8_shared_control"]["true_feasible"]),
            "false_certification_nonworse_than_bounded": bool(
                totals[variant]["audit_pool_false_certificates"]
                <= totals["bounded_shared_control"][
                    "audit_pool_false_certificates"]),
            "variance_rmse_nonworse_than_v8_at_least_3_of_4": bool(
                _count_lower_nonworse(
                    values, v8, "median_variance_log_rmse",
                    multiplier=1.10) >= 3),
            "mean_mae_nonworse_than_v8_at_least_3_of_4": bool(
                _count_lower_nonworse(
                    values, v8, "median_mean_abs_error",
                    multiplier=1.05, offset=0.005) >= 3),
            "mean_rank_nonworse_than_v8_at_least_3_of_4": bool(
                _count_higher_nonworse(
                    values, v8,
                    "median_constraint_mean_rank_correlation") >= 3),
            "feasible_regret_nonworse_at_least_3_of_4": bool(
                _count_lower_nonworse(
                    values, v8, "median_feasible_regret") >= 3),
            "strict_false_certification_gain_over_v8": bool(
                totals[variant]["audit_pool_false_certificates"]
                < totals["v8_shared_control"][
                    "audit_pool_false_certificates"]),
        }
    eligible = [
        variant for variant in CHALLENGERS
        if all(checks[variant].values())
    ]
    return {
        "schema_version": 1,
        "row_count": int(len(rows)),
        "expected_row_count": int(
            len(VARIANTS) * len(SCENARIOS) * expected_seeds),
        "cells": list(cells.values()),
        "totals": totals,
        "global_checks": global_checks,
        "variant_checks": checks,
        "sequential_gate_eligible": eligible,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-seeds", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    result = summarize(load_rows(args.root), args.expected_seeds)
    output = args.out or args.root / "mean_alignment_v15_gate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "sequential_gate_eligible": result["sequential_gate_eligible"],
        "totals": result["totals"],
        "variant_checks": result["variant_checks"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
