#!/usr/bin/env python3
"""Analyze the V42 shared-low-rank source hyperlaw gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from . import analyze_mean_alignment_v41_gate as base
    from . import analyze_mean_alignment_v34_gate as common
except ImportError:  # Direct script execution.
    import analyze_mean_alignment_v41_gate as base
    import analyze_mean_alignment_v34_gate as common


VARIANTS = (
    "v35_model_confidence",
    "v41_source_bayes",
    "v42_shared_low_rank_model",
    "v42_shared_low_rank_bayes",
)
CONFIDENCE_MODES = {
    "v35_model_confidence": "model",
    "v41_source_bayes": "source_bayes",
    "v42_shared_low_rank_model": "model",
    "v42_shared_low_rank_bayes": "source_bayes",
}
HYPERLAW_MODES = {
    "v35_model_confidence": "single_gaussian_draw",
    "v41_source_bayes": "single_gaussian_draw",
    "v42_shared_low_rank_model": "shared_low_rank_discrepancy",
    "v42_shared_low_rank_bayes": "shared_low_rank_discrepancy",
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


def _prior_diagnostics(row):
    numerics = list(row.get("gpr_numerics") or [])
    if len(numerics) < 2:
        return {}
    return dict(numerics[1].get("source_parametric_prior") or {})


def _confidence_contract(rows, variant):
    expected = CONFIDENCE_MODES[variant]
    selected = [row for row in rows if row.get("gate_variant") == variant]
    if not selected:
        return False
    for row in selected:
        if row.get("source_constraint_mean_confidence_mode") != expected:
            return False
        diagnostics = dict(row.get("source_conditioned_confidence") or {})
        if diagnostics.get("mode") != expected:
            return False
        if diagnostics.get("status") != "active":
            return False
        if diagnostics.get("target_oracle_used", True):
            return False
        if expected == "model":
            continue
        for key in ("effective_beta", "effective_rank", "total_radius_median"):
            value = float(diagnostics.get(key, np.nan))
            if not np.isfinite(value) or value <= 0.0:
                return False
        if diagnostics.get("solution_specific_deviation_double_counted", True):
            return False
        if not diagnostics.get("transfer_guard_can_only_increase_radius", False):
            return False
    return True


def _hyperlaw_contract(rows, variant):
    expected = HYPERLAW_MODES[variant]
    selected = [row for row in rows if row.get("gate_variant") == variant]
    if not selected:
        return False
    for row in selected:
        if row.get("source_constraint_mean_hyperlaw_mode") != expected:
            return False
        diagnostics = _prior_diagnostics(row)
        if diagnostics.get("configured_hyperlaw_mode") != expected:
            return False
        if diagnostics.get("target_oracle_used", True):
            return False
        low_rank = expected == "shared_low_rank_discrepancy"
        if bool(diagnostics.get("shared_low_rank_prior_selected")) != low_rank:
            return False
        if not low_rank:
            if diagnostics.get("target_task_law") != "single_gaussian_draw":
                return False
            continue
        if diagnostics.get("target_task_law") != (
            "shared_mean_plus_low_rank_domain_discrepancy"
        ):
            return False
        if not diagnostics.get(
            "source_estimation_covariance_enters_shared_mean_only", False
        ):
            return False
        if diagnostics.get("within_source_estimation_as_target_variation", True):
            return False
        if not diagnostics.get("channel_role_covariance_retained", False):
            return False
        rank = int(diagnostics.get("domain_discrepancy_rank", -1))
        upper = int(diagnostics.get("domain_discrepancy_rank_upper_bound", -1))
        if rank < 0 or upper < 0 or rank > upper:
            return False
        for key in (
            "shared_mean_covariance_trace",
            "channel_role_covariance_trace",
            "between_domain_covariance_trace",
            "prior_covariance_trace",
        ):
            value = float(diagnostics.get(key, np.nan))
            if not np.isfinite(value) or value < 0.0:
                return False
    return True


def _summary(rows, variant, expected_seeds):
    result = base._summary(rows, variant, expected_seeds)
    selected = [row for row in rows if row.get("gate_variant") == variant]
    priors = [_prior_diagnostics(row) for row in selected]

    def median(key):
        values = np.asarray([
            float(item[key]) for item in priors
            if item.get(key) is not None and np.isfinite(float(item[key]))
        ], dtype=float)
        return float(np.median(values)) if len(values) else None

    result.update({
        "median_shared_mean_covariance_trace": median(
            "shared_mean_covariance_trace"),
        "median_channel_role_covariance_trace": median(
            "channel_role_covariance_trace"),
        "median_between_domain_covariance_trace": median(
            "between_domain_covariance_trace"),
        "median_hyperlaw_prior_covariance_trace": median(
            "prior_covariance_trace"),
    })
    return result


def summarize(rows, expected_seeds=5):
    present = tuple(
        variant for variant in VARIANTS
        if any(row.get("gate_variant") == variant for row in rows)
    )
    if "v35_model_confidence" not in present:
        raise ValueError("V42 analysis requires the V35 control arm")
    summaries = {
        variant: _summary(rows, variant, expected_seeds)
        for variant in present
    }
    paired_initial = base.base.base.base._paired_initial_contract(
        rows, present, expected_seeds)
    paired_online = base._paired_online_contract(
        rows, present, expected_seeds)
    confidence = {
        variant: _confidence_contract(rows, variant) for variant in present
    }
    hyperlaw = {
        variant: _hyperlaw_contract(rows, variant) for variant in present
    }
    control = summaries["v35_model_confidence"]
    legacy_bayes = summaries.get("v41_source_bayes", control)
    expected = len(common.MEAN_SCENARIOS) * int(expected_seeds)
    challenger = "v42_shared_low_rank_bayes"
    promotion = []
    comparisons = {}
    if challenger in summaries:
        value = summaries[challenger]
        false_certificates = (
            value["audit_pool_false_certificates"]
            + value["evaluated_false_certificates"])
        nonvacuous = bool(
            value["audit_pool_true_certificates"] > 0
            or value["evaluated_true_certificates"] > 0
            or value["adaptive_improvements"] > 0
        )
        strict = bool(
            value["true_feasible"] > max(
                control["true_feasible"], legacy_bayes["true_feasible"])
            or value["adaptive_improvements"] > max(
                control["adaptive_improvements"],
                legacy_bayes["adaptive_improvements"])
            or value["audit_pool_true_certificates"] > max(
                control["audit_pool_true_certificates"],
                legacy_bayes["audit_pool_true_certificates"])
            or value["evaluated_true_certificates"] > max(
                control["evaluated_true_certificates"],
                legacy_bayes["evaluated_true_certificates"])
        )
        comparisons[challenger] = {
            "strictly_better_than_v35_and_v41": strict,
            "nonvacuous_certificate_or_strict_improvement": nonvacuous,
            "zero_false_certificates": bool(false_certificates == 0),
            "zero_adaptive_losses": bool(value["adaptive_losses"] == 0),
            "all_targets_true_feasible": bool(value["true_feasible"] == expected),
        }
        if all((
            value["complete"], paired_initial, paired_online,
            confidence[challenger], hyperlaw[challenger],
            false_certificates == 0,
            value["adaptive_losses"] == 0,
            value["true_feasible"] == expected,
            nonvacuous,
            strict,
        )):
            promotion.append(challenger)
    return {
        "paired_initial_design_and_archive": bool(paired_initial),
        "paired_sobol_actions_and_responses": bool(paired_online),
        "source_confidence_contract": confidence,
        "source_hyperlaw_contract": hyperlaw,
        "variant_summaries": summaries,
        "comparisons": comparisons,
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
