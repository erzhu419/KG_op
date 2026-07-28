#!/usr/bin/env python3
"""Analyze the V41 source-conditioned confidence-sequence gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from . import analyze_mean_alignment_v39_gate as base
    from . import analyze_mean_alignment_v34_gate as common
except ImportError:  # Direct script execution.
    import analyze_mean_alignment_v39_gate as base
    import analyze_mean_alignment_v34_gate as common


VARIANTS = (
    "v35_model_confidence",
    "v41_source_bayes",
    "v41_source_self_normalized",
)
SOURCE_VARIANTS = VARIANTS[1:]
CONFIDENCE_MODES = {
    "v35_model_confidence": "model",
    "v41_source_bayes": "source_bayes",
    "v41_source_self_normalized": "source_self_normalized",
}


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


def _finite_median(values):
    finite = np.asarray([
        float(value) for value in values
        if value is not None and np.isfinite(float(value))
    ], dtype=float)
    return float(np.median(finite)) if len(finite) else None


def _confidence_contract(rows, variant):
    expected_mode = CONFIDENCE_MODES[variant]
    selected = [row for row in rows if row.get("gate_variant") == variant]
    if not selected:
        return False
    for row in selected:
        if str(row.get(
            "source_constraint_mean_confidence_mode", "missing"
        )) != expected_mode:
            return False
        diagnostics = dict(row.get("source_conditioned_confidence") or {})
        if diagnostics.get("mode") != expected_mode:
            return False
        if diagnostics.get("status") != "active":
            return False
        if diagnostics.get("target_oracle_used", True):
            return False
        if expected_mode == "model":
            continue
        finite_positive = (
            "effective_beta",
            "effective_rank",
            "total_radius_median",
        )
        if any(
            not np.isfinite(float(diagnostics.get(key, np.nan)))
            or float(diagnostics.get(key, np.nan)) <= 0.0
            for key in finite_positive
        ):
            return False
        finite_nonnegative = (
            "information_gain",
            "transfer_residual_floor",
            "transfer_radius",
            "coefficient_radius_median",
        )
        if any(
            not np.isfinite(float(diagnostics.get(key, np.nan)))
            or float(diagnostics.get(key, np.nan)) < 0.0
            for key in finite_nonnegative
        ):
            return False
        ratio = float(diagnostics.get(
            "equivalent_to_model_variance_median_ratio", np.nan))
        if not np.isfinite(ratio) or ratio <= 0.0:
            return False
        if diagnostics.get(
            "solution_specific_deviation_double_counted", True
        ):
            return False
        if not diagnostics.get(
            "transfer_guard_can_only_increase_radius", False
        ):
            return False
        if int(diagnostics.get("source_domain_count", 0)) < 1:
            return False
        if int(diagnostics.get("target_count", -1)) < 0:
            return False
    return True


def _paired_online_contract(rows, variants, expected_seeds):
    """Require identical target observations under the Sobol diagnostic arm."""

    for scenario in common.MEAN_SCENARIOS:
        for seed in range(int(expected_seeds)):
            group = [
                row for row in rows
                if common._scenario(row) == scenario
                and int(row.get("seed", -1)) == seed
                and row.get("gate_variant") in variants
            ]
            if len(group) != len(variants):
                return False
            if any(row.get("decision_backend") != "sobol_new" for row in group):
                return False
            designs = {
                str(row.get("target_design_fingerprint", "missing"))
                for row in group
            }
            actions = {
                str(row.get(
                    "online_action_sequence_fingerprint", "missing"))
                for row in group
            }
            if len(designs) != 1 or "missing" in designs:
                return False
            if len(actions) != 1 or "missing" in actions:
                return False
            traces = [list(row.get("online_action_trace") or []) for row in group]
            expected_online = max(
                int(group[0].get("N", 0)) - int(group[0].get("n0", 0)), 0)
            if not traces or expected_online <= 0:
                return False
            if any(len(trace) != expected_online for trace in traces):
                return False
            reference = traces[0]
            for trace in traces:
                if any(
                    left.get("x_fingerprint") != right.get("x_fingerprint")
                    or left.get("observed_response")
                    != right.get("observed_response")
                    or right.get("candidate_source") != "sobol_continuation"
                    for left, right in zip(reference, trace)
                ):
                    return False
            if any(
                bool(row.get("online_action_trace_target_oracle_used", True))
                for row in group
            ):
                return False
    return True


def _summary(rows, variant, expected_seeds):
    result = base._summary(rows, variant, expected_seeds)
    selected = [row for row in rows if row.get("gate_variant") == variant]
    confidence = [
        dict(row.get("source_conditioned_confidence") or {})
        for row in selected
    ]
    raw = [
        dict(row.get("boundary_raw_pool_truth_diagnostics") or {})
        for row in selected
    ]
    result.update({
        "median_best_feasible_epistemic_radius": _finite_median([
            item.get("boundary_raw_pool_best_feasible_epistemic_radius")
            for item in raw
        ]),
        "median_effective_beta": _finite_median([
            item.get("effective_beta") for item in confidence
        ]),
        "median_information_gain": _finite_median([
            item.get("information_gain") for item in confidence
        ]),
        "median_effective_rank": _finite_median([
            item.get("effective_rank") for item in confidence
        ]),
        "median_coefficient_radius": _finite_median([
            item.get("coefficient_radius_median") for item in confidence
        ]),
        "median_transfer_radius": _finite_median([
            item.get("transfer_radius") for item in confidence
        ]),
        "median_total_radius": _finite_median([
            item.get("total_radius_median") for item in confidence
        ]),
        "median_equivalent_to_model_variance_ratio": _finite_median([
            item.get("equivalent_to_model_variance_median_ratio")
            for item in confidence
        ]),
    })
    return result


def summarize(rows, expected_seeds=5):
    present = tuple(
        variant for variant in VARIANTS
        if any(row.get("gate_variant") == variant for row in rows)
    )
    if "v35_model_confidence" not in present:
        raise ValueError("V41 analysis requires the V35 model-confidence arm")
    summaries = {
        variant: _summary(rows, variant, expected_seeds)
        for variant in present
    }
    paired_initial = base.base.base._paired_initial_contract(
        rows, present, expected_seeds)
    paired_online = _paired_online_contract(rows, present, expected_seeds)
    contracts = {
        variant: _confidence_contract(rows, variant)
        for variant in present
    }
    control = summaries["v35_model_confidence"]
    diagnostic = []
    expected = len(common.MEAN_SCENARIOS) * int(expected_seeds)
    comparisons = {}
    for variant in SOURCE_VARIANTS:
        if variant not in summaries:
            continue
        value = summaries[variant]
        source_radius = value["median_best_feasible_epistemic_radius"]
        control_radius = control["median_best_feasible_epistemic_radius"]
        radius_nonworse = bool(
            source_radius is not None
            and control_radius is not None
            and source_radius <= control_radius + 1e-12
        )
        no_false = bool(
            value["audit_pool_false_certificates"] == 0
            and value["evaluated_false_certificates"] == 0
        )
        no_regression = bool(
            value["true_feasible"] >= control["true_feasible"]
            and value["adaptive_losses"] <= control["adaptive_losses"]
        )
        nonvacuous = bool(
            value["audit_pool_true_certificates"] > 0
            or value["evaluated_true_certificates"] > 0
            or value["adaptive_improvements"]
            > control["adaptive_improvements"]
        )
        comparisons[variant] = {
            "epistemic_radius_nonworse_than_model": radius_nonworse,
            "true_feasible_nonworse_than_model": bool(
                value["true_feasible"] >= control["true_feasible"]),
            "adaptive_losses_nonworse_than_model": bool(
                value["adaptive_losses"] <= control["adaptive_losses"]),
            "nonvacuous_certificate_or_strict_improvement": nonvacuous,
        }
        if all((
            value["complete"], paired_initial, paired_online,
            contracts[variant], no_false, no_regression, radius_nonworse,
        )):
            diagnostic.append(variant)

    challenger = "v41_source_self_normalized"
    promotion = []
    if challenger in summaries:
        value = summaries[challenger]
        checks = comparisons[challenger]
        no_false = bool(
            value["audit_pool_false_certificates"] == 0
            and value["evaluated_false_certificates"] == 0
        )
        if all((
            value["complete"], paired_initial, paired_online,
            contracts[challenger], no_false,
            value["true_feasible"] == expected,
            value["adaptive_losses"] == 0,
            checks["epistemic_radius_nonworse_than_model"],
            checks["nonvacuous_certificate_or_strict_improvement"],
        )):
            promotion.append(challenger)
    return {
        "paired_initial_design_and_archive": bool(paired_initial),
        "paired_sobol_actions_and_responses": bool(paired_online),
        "source_confidence_contract": contracts,
        "variant_summaries": summaries,
        "comparisons_to_model_confidence": comparisons,
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
