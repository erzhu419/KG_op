#!/usr/bin/env python3
"""Analyze the V43 finite-source predictive hyperlaw sentinel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from . import analyze_mean_alignment_v42_gate as base
except ImportError:  # Direct script execution.
    import analyze_mean_alignment_v42_gate as base


VARIANTS = (
    "v41_source_bayes",
    "v42_shared_low_rank_bayes",
    "v43_predictive_low_rank_model",
    "v43_predictive_low_rank_bayes",
)
PREDICTIVE_VARIANTS = VARIANTS[2:]
SENTINEL_SEEDS = (1, 3)
QUEUE_SCENARIO = ("QueueResourceControl", 1.0)
CONFIDENCE_MODES = {
    "v41_source_bayes": "source_bayes",
    "v42_shared_low_rank_bayes": "source_bayes",
    "v43_predictive_low_rank_model": "model",
    "v43_predictive_low_rank_bayes": "source_bayes",
}
HYPERLAW_MODES = {
    "v41_source_bayes": "single_gaussian_draw",
    "v42_shared_low_rank_bayes": "shared_low_rank_discrepancy",
    "v43_predictive_low_rank_model": "shared_low_rank_predictive",
    "v43_predictive_low_rank_bayes": "shared_low_rank_predictive",
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


def _scenario(row):
    return base.common._scenario(row)


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
    return True


def _hyperlaw_contract(rows, variant):
    expected = HYPERLAW_MODES[variant]
    selected = [row for row in rows if row.get("gate_variant") == variant]
    if not selected:
        return False
    for row in selected:
        if row.get("source_constraint_mean_hyperlaw_mode") != expected:
            return False
        diagnostics = base._prior_diagnostics(row)
        if diagnostics.get("configured_hyperlaw_mode") != expected:
            return False
        if diagnostics.get("target_oracle_used", True):
            return False
        predictive = expected == "shared_low_rank_predictive"
        if bool(diagnostics.get(
            "finite_source_predictive_prior_selected", False
        )) != predictive:
            return False
        if not predictive:
            continue
        if diagnostics.get("target_task_law") != (
            "shared_mean_plus_finite_source_predictive_discrepancy"
        ):
            return False
        if not diagnostics.get("finite_source_predictive_correction", False):
            return False
        concentration = float(diagnostics.get(
            "weighted_source_concentration", np.nan))
        multiplier = float(diagnostics.get(
            "finite_source_predictive_multiplier", np.nan))
        if not (np.isfinite(concentration) and 0.0 < concentration < 1.0):
            return False
        expected_multiplier = (1.0 + concentration) / (1.0 - concentration)
        if not np.isclose(multiplier, expected_multiplier, rtol=1e-10):
            return False
        if multiplier < 1.0:
            return False
        population = float(diagnostics.get(
            "between_domain_population_covariance_trace", np.nan))
        predictive_trace = float(diagnostics.get(
            "between_domain_predictive_covariance_trace", np.nan))
        if not (
            np.isfinite(population) and population >= 0.0
            and np.isfinite(predictive_trace) and predictive_trace >= 0.0
            and np.isclose(
                predictive_trace, multiplier * population,
                rtol=1e-8, atol=1e-10)
        ):
            return False
        rank = int(diagnostics.get("predictive_discrepancy_rank", -1))
        upper = int(diagnostics.get("domain_discrepancy_rank_upper_bound", -1))
        if rank < 0 or upper < 0 or rank > upper:
            return False
        if not diagnostics.get(
            "source_estimation_covariance_enters_shared_mean_only", False
        ):
            return False
        if diagnostics.get("within_source_estimation_as_target_variation", True):
            return False
    return True


def _variant_summary(rows, variant, expected_count):
    selected = [row for row in rows if row.get("gate_variant") == variant]
    audits = [
        dict(row.get("boundary_raw_pool_truth_diagnostics") or {})
        for row in selected
    ]
    certificate_audits = [
        dict(row.get("certificate_outcome_audit") or {})
        for row in selected
    ]
    regrets = np.asarray([
        (
            np.nan
            if row.get("feasible_simple_regret") is None
            else float(row["feasible_simple_regret"])
        )
        for row in selected
    ], dtype=float)
    regrets = regrets[np.isfinite(regrets)]
    return {
        "count": int(len(selected)),
        "complete": bool(len(selected) == int(expected_count)),
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
        "median_feasible_regret": (
            float(np.median(regrets)) if len(regrets) else None),
    }


def _paired_fingerprint(row, field):
    value = row.get(field)
    if value is not None:
        return str(value)
    contract = dict(row.get("source_target_adaptation_contract") or {})
    nested_field = {
        "source_archive_fingerprint": "source_archive_fingerprint",
        "target_design_fingerprint": "target_initial_design_fingerprint",
    }.get(field, field)
    return str(contract.get(nested_field, "missing"))


def _paired_contract(rows, seeds):
    for seed in seeds:
        group = [
            row for row in rows
            if _scenario(row) == QUEUE_SCENARIO
            and int(row.get("seed", -1)) == int(seed)
            and row.get("gate_variant") in VARIANTS
        ]
        if len(group) != len(VARIANTS):
            return False
        if any(row.get("decision_backend") != "sobol_new" for row in group):
            return False
        for field in (
            "source_archive_fingerprint",
            "target_design_fingerprint",
            "online_action_sequence_fingerprint",
        ):
            values = {_paired_fingerprint(row, field) for row in group}
            if len(values) != 1 or "missing" in values:
                return False
        traces = [list(row.get("online_action_trace") or []) for row in group]
        if not traces:
            return False
        reference = traces[0]
        for trace in traces:
            if len(trace) != len(reference):
                return False
            if any(
                left.get("x_fingerprint") != right.get("x_fingerprint")
                or left.get("observed_response") != right.get("observed_response")
                or right.get("candidate_source") != "sobol_continuation"
                for left, right in zip(reference, trace)
            ):
                return False
        if any(bool(row.get(
            "online_action_trace_target_oracle_used", True
        )) for row in group):
            return False
    return True


def summarize(rows, seeds=SENTINEL_SEEDS):
    seeds = tuple(int(seed) for seed in seeds)
    selected = [
        row for row in rows
        if _scenario(row) == QUEUE_SCENARIO
        and int(row.get("seed", -1)) in seeds
    ]
    summaries = {
        variant: _variant_summary(selected, variant, len(seeds))
        for variant in VARIANTS
    }
    paired = _paired_contract(selected, seeds)
    confidence = {
        variant: _confidence_contract(selected, variant)
        for variant in VARIANTS
    }
    hyperlaw = {
        variant: _hyperlaw_contract(selected, variant)
        for variant in VARIANTS
    }
    challenger = summaries["v43_predictive_low_rank_bayes"]
    legacy = summaries["v41_source_bayes"]
    population = summaries["v42_shared_low_rank_bayes"]
    false_certificates = (
        challenger["audit_pool_false_certificates"]
        + challenger["evaluated_false_certificates"])
    strict = bool(
        challenger["true_feasible"] > max(
            legacy["true_feasible"], population["true_feasible"])
        or challenger["adaptive_losses"] < min(
            legacy["adaptive_losses"], population["adaptive_losses"])
        or challenger["adaptive_improvements"] > max(
            legacy["adaptive_improvements"],
            population["adaptive_improvements"])
    )
    promotion = bool(all((
        challenger["complete"],
        paired,
        confidence["v43_predictive_low_rank_bayes"],
        hyperlaw["v43_predictive_low_rank_bayes"],
        challenger["true_feasible"] == len(seeds),
        challenger["adaptive_losses"] == 0,
        false_certificates == 0,
        strict,
    )))
    return {
        "scope": "queue_failure_sentinel",
        "seeds": list(seeds),
        "paired_initial_design_archive_actions_and_responses": bool(paired),
        "source_confidence_contract": confidence,
        "source_hyperlaw_contract": hyperlaw,
        "variant_summaries": summaries,
        "challenger_strictly_improves_v41_and_v42": strict,
        "promotion_eligible": [
            "v43_predictive_low_rank_bayes"] if promotion else [],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--seeds", default="1,3")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    seeds = tuple(
        int(value.strip()) for value in args.seeds.split(",") if value.strip())
    result = summarize(load_rows(args.root), seeds)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
