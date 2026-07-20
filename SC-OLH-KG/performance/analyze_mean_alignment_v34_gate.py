#!/usr/bin/env python3
"""Analyze the V34 robust empirical-Bayes mean-posterior gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .analyze_mean_alignment_v16_gate import (
        _constraint_prior,
        _count_higher_nonworse,
        _count_lower_nonworse,
    )
    from .analyze_mean_alignment_v20_gate import MEAN_SCENARIOS
    from .analyze_mean_alignment_v28_gate import _cell as _v28_cell
    from .analyze_mean_alignment_v29_gate import _dominance_record_contract
    from .analyze_mean_alignment_v8_gate import _scenario
except ImportError:  # Direct script execution.
    from analyze_mean_alignment_v16_gate import (
        _constraint_prior,
        _count_higher_nonworse,
        _count_lower_nonworse,
    )
    from analyze_mean_alignment_v20_gate import MEAN_SCENARIOS
    from analyze_mean_alignment_v28_gate import _cell as _v28_cell
    from analyze_mean_alignment_v29_gate import _dominance_record_contract
    from analyze_mean_alignment_v8_gate import _scenario


VARIANTS = (
    "v29_scale_control",
    "v31_hc3_control",
    "v34_scale_hc3",
    "v34_scale_hc3_task",
)
CHALLENGERS = VARIANTS[2:]
MISSPECIFICATION_MODES = {
    "v29_scale_control": "predictive_scale",
    "v31_hc3_control": "predictive_sandwich_hc3",
    "v34_scale_hc3": "predictive_scale_sandwich_hc3",
    "v34_scale_hc3_task": "predictive_scale_sandwich_hc3_task",
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


def _robust_posterior_contract(prior, expected_mode):
    mode = str(prior.get("source_mean_misspecification_mode", "missing"))
    if mode != expected_mode or prior.get("target_oracle_used", True):
        return False
    if prior.get("target_oracle_used_for_misspecification", True):
        return False
    if not prior.get("source_mean_misspecification_applied", False):
        return False
    scale = float(prior.get("source_mean_misspecification_scale", np.nan))
    if not np.isfinite(scale) or scale < 1.0:
        return False
    expected_scaled = expected_mode.startswith("predictive_scale_sandwich")
    expected_sandwich = "sandwich_hc3" in expected_mode
    if bool(prior.get(
        "source_mean_prior_scaled_before_conditioning", False
    )) != expected_scaled:
        return False
    if bool(prior.get("source_mean_sandwich_applied", False)) != expected_sandwich:
        return False
    if expected_scaled and prior.get(
        "source_mean_misspecification_application"
    ) != "source_prior_scale_then_posterior_sandwich":
        return False
    if expected_sandwich:
        if not prior.get("source_mean_posterior_mean_preserved", False):
            return False
        if float(prior.get(
            "source_mean_sandwich_covariance_trace", -1.0)) < 0.0:
            return False
        if float(prior.get(
            "source_mean_posterior_covariance_trace_after", -np.inf
        )) + 1e-12 < float(prior.get(
            "source_mean_posterior_covariance_trace_before", np.inf)):
            return False
        expected_authority = (
            "confidence_only"
            if expected_mode.endswith("_confidence")
            else "joint_predictive"
        )
        if prior.get(
            "source_mean_sandwich_decision_authority"
        ) != expected_authority:
            return False
        if expected_authority == "confidence_only" and not prior.get(
            "decision_covariance_available", False
        ):
            return False
    return bool(prior.get(
        "misspecification_uncertainty_can_only_increase", False))


def _cell(
    rows,
    variant,
    scenario,
    expected_seeds,
    misspecification_modes=MISSPECIFICATION_MODES,
):
    value = _v28_cell(rows, variant, scenario, expected_seeds)
    group = [
        row for row in rows
        if row["gate_variant"] == variant and _scenario(row) == scenario
    ]
    priors = [_constraint_prior(row) for row in group]
    expected_mode = misspecification_modes[variant]
    value.update({
        "posterior_dominance_contract": bool(group) and all(
            _dominance_record_contract(row, True) for row in group),
        "robust_posterior_contract": bool(priors) and all(
            _robust_posterior_contract(prior, expected_mode)
            for prior in priors),
        "misspecification_modes": sorted(set(str(
            prior.get("source_mean_misspecification_mode", "missing"))
            for prior in priors)),
        "prior_scaled_before_conditioning_count": int(sum(bool(
            prior.get("source_mean_prior_scaled_before_conditioning", False))
            for prior in priors)),
        "sandwich_applied_count": int(sum(bool(
            prior.get("source_mean_sandwich_applied", False))
            for prior in priors)),
        "posterior_dominance_switch_count": int(sum(int(
            row.get("posterior_dominance_switch_count", 0) or 0)
            for row in group)),
    })
    return value


def summarize(
    rows,
    expected_seeds=5,
    *,
    variants=VARIANTS,
    challengers=CHALLENGERS,
    misspecification_modes=MISSPECIFICATION_MODES,
    complexity_order=None,
):
    cells = {
        (variant, *scenario): _cell(
            rows, variant, scenario, expected_seeds,
            misspecification_modes=misspecification_modes)
        for variant in variants for scenario in MEAN_SCENARIOS
    }
    grouped = {
        variant: [cells[(variant, *scenario)] for scenario in MEAN_SCENARIOS]
        for variant in variants
    }
    paired = all(
        len({tuple(cell["initial_design_fingerprints"])
             for cell in scenario_cells}) == 1
        and len({tuple(cell["source_archive_fingerprints"])
                 for cell in scenario_cells}) == 1
        for scenario_cells in zip(*(grouped[value] for value in variants))
    )
    totals = {
        variant: {
            "true_feasible": int(sum(
                item["true_feasible_count"] for item in values)),
            "adaptive_losses": int(sum(
                item["adaptive_loss_count"] for item in values)),
            "audit_pool_true_certificates": int(sum(
                item["audit_pool_true_certificate_count"]
                for item in values)),
            "audit_pool_false_certificates": int(sum(
                item["audit_pool_false_certificate_count"]
                for item in values)),
            "oracle_both_certificates": int(sum(
                item["oracle_both_certified_total"] for item in values)),
            "posterior_dominance_switches": int(sum(
                item["posterior_dominance_switch_count"]
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
    calibration_control = grouped["v29_scale_control"]
    checks = {}
    for variant in challengers:
        values = grouped[variant]
        checks[variant] = {
            **global_checks,
            "single_empirical_bayes_hyperlaw_contract": bool(all(
                item["single_hyperlaw_contract"]
                and item["posterior_target_data_used"]
                for item in values)),
            "independent_cumulative_hvd_contract": bool(all(
                item["hvd_source_task_weight_modes"] == ["independent"]
                and item["variance_task_posterior_modes"]
                == ["replication_only"]
                and item["certification_head_authorities"]
                == ["split_gpr_cumulative_hvd"]
                for item in values)),
            "robust_mean_posterior_contract": bool(all(
                item["robust_posterior_contract"] for item in values)),
            "posterior_dominance_contract": bool(all(
                item["posterior_dominance_contract"] for item in values)),
            "all_initially_safe_incumbents_preserved": bool(
                totals[variant]["true_feasible"]
                == len(MEAN_SCENARIOS) * expected_seeds
                and totals[variant]["adaptive_losses"] == 0),
            "audit_pool_false_certificates_eliminated": bool(
                totals[variant]["audit_pool_false_certificates"] == 0),
            "nonvacuous_true_certificate_coverage": bool(
                totals[variant]["audit_pool_true_certificates"] > 0),
            "mean_rank_preserved_at_least_2_of_3": bool(
                _count_higher_nonworse(
                    values, calibration_control,
                    "median_constraint_mean_rank_correlation") >= 2),
            "mean_mae_preserved_at_least_2_of_3": bool(
                _count_lower_nonworse(
                    values, calibration_control, "median_mean_abs_error",
                    multiplier=1.02, offset=0.002) >= 2),
        }
    decision_expansion_eligible = [
        variant for variant in challengers
        if all(value for key, value in checks[variant].items()
               if key != "nonvacuous_true_certificate_coverage")
    ]
    eligible = [
        variant for variant in challengers
        if all(checks[variant].values())
    ]
    if complexity_order is None:
        complexity_order = {
            variant: index for index, variant in enumerate(challengers)
        }
    recommended = (
        min(
            eligible,
            key=lambda variant: (
                -totals[variant]["audit_pool_true_certificates"],
                complexity_order[variant],
            ),
        )
        if eligible else None
    )
    return {
        "totals": totals,
        "variant_checks": checks,
        "decision_expansion_eligible": decision_expansion_eligible,
        "promotion_eligible": eligible,
        "post_gate_baseline_recommendation": recommended,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-seeds", type=int, default=5)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    summary = summarize(load_rows(args.root), args.expected_seeds)
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
