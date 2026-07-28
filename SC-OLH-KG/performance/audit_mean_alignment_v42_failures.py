#!/usr/bin/env python3
"""Produce a compact paired failure audit for the V42 hyperlaw gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from . import analyze_mean_alignment_v42_gate as gate
    from .analyze_mean_alignment_v34_gate import _scenario
except ImportError:  # Direct script execution.
    import analyze_mean_alignment_v42_gate as gate
    from analyze_mean_alignment_v34_gate import _scenario


def _number(row, key):
    value = row.get(key)
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _finite_median(values):
    finite = np.asarray([
        float(value) for value in values
        if value is not None and np.isfinite(float(value))
    ], dtype=float)
    return float(np.median(finite)) if len(finite) else None


def _point_summary(point):
    if point is None:
        return None
    values = np.asarray(point, dtype=np.int64).reshape(-1)
    return {
        "fingerprint": hashlib.sha256(values.tobytes()).hexdigest()[:16],
        "dimension": int(len(values)),
        "head": values[:8].tolist(),
        "minimum": int(np.min(values)) if len(values) else None,
        "maximum": int(np.max(values)) if len(values) else None,
        "mean": float(np.mean(values)) if len(values) else None,
        "nonzero_count": int(np.count_nonzero(values)),
    }


def _record(row):
    prior = gate._prior_diagnostics(row)
    raw = dict(row.get("boundary_raw_pool_truth_diagnostics") or {})
    dominance = dict(row.get("posterior_dominance") or {})
    history = list(dominance.get("history") or [])
    return {
        "variant": row.get("gate_variant"),
        "true_feasible": bool(row.get("true_feasible", False)),
        "adaptive_loss": bool(row.get("adaptive_loss", False)),
        "adaptive_improvement": bool(
            row.get("adaptive_improves_initial_best", False)),
        "initial_has_true_feasible": bool(
            row.get("initial_has_true_feasible", False)),
        "x_recommended": _point_summary(row.get("x_recommended")),
        "posterior_dominance_terminal_used": bool(
            row.get("posterior_dominance_terminal_used", False)),
        "posterior_dominance_switch_count": int(
            row.get("posterior_dominance_switch_count", 0) or 0),
        "posterior_dominance_incumbent": _point_summary(
            dominance.get("incumbent")),
        "posterior_dominance_history_length": len(history),
        "posterior_dominance_history_reasons": [
            item.get("reason") for item in history
        ],
        "observed_incumbent_chance_margin": _number(
            row, "observed_incumbent_chance_margin"),
        "decision_backend_terminal_margin": _number(
            row, "decision_backend_terminal_margin"),
        "calibrated_recommendation_reason": row.get(
            "calibrated_recommendation_reason"),
        "true_chance_margin": _number(row, "true_chance_margin"),
        "posterior_theory_margin": _number(
            row, "posterior_theory_chance_margin"),
        "posterior_chance_margin": _number(row, "posterior_chance_margin"),
        "selected_mu_constraint": _number(
            row, "recommendation_selected_mu_con"),
        "selected_epistemic_variance": _number(
            row, "recommendation_selected_epistemic_var"),
        "selected_aleatoric_variance": _number(
            row, "recommendation_selected_aleatoric_var"),
        "best_feasible_mu_constraint": _number(
            row, "recommendation_best_true_feasible_mu_con"),
        "best_feasible_theory_margin": _number(
            row, "recommendation_best_true_feasible_theory_margin"),
        "best_feasible_epistemic_radius": _number(
            raw, "boundary_raw_pool_best_feasible_epistemic_radius"),
        "best_feasible_true_margin": _number(
            raw, "boundary_raw_pool_best_feasible_true_margin"),
        "feasible_simple_regret": _number(row, "feasible_simple_regret"),
        "initial_best_feasible_regret": _number(
            row, "initial_best_feasible_regret"),
        "source_misspecification_scale": _number(
            prior, "source_mean_misspecification_scale"),
        "prior_covariance_trace": _number(prior, "prior_covariance_trace"),
        "decision_covariance_trace": _number(
            prior, "decision_covariance_trace"),
        "robust_covariance_trace": _number(
            prior, "robust_covariance_trace"),
        "shared_mean_covariance_trace": _number(
            prior, "shared_mean_covariance_trace"),
        "channel_role_covariance_trace": _number(
            prior, "channel_role_covariance_trace"),
        "between_domain_covariance_trace": _number(
            prior, "between_domain_covariance_trace"),
    }


def summarize(rows, expected_seeds=5):
    paired = []
    per_domain = {}
    for scenario in gate.common.MEAN_SCENARIOS:
        scenario_name = str(scenario[0])
        per_domain[scenario_name] = {}
        for variant in gate.VARIANTS:
            selected = [
                row for row in rows
                if row.get("gate_variant") == variant
                and _scenario(row) == scenario
            ]
            per_domain[scenario_name][variant] = {
                "count": len(selected),
                "true_feasible": int(sum(bool(
                    row.get("true_feasible", False)) for row in selected)),
                "adaptive_losses": int(sum(bool(
                    row.get("adaptive_loss", False)) for row in selected)),
                "median_true_margin": _finite_median([
                    row.get("true_chance_margin") for row in selected]),
                "median_theory_margin": _finite_median([
                    row.get("posterior_theory_chance_margin")
                    for row in selected
                ]),
            }
        for seed in range(int(expected_seeds)):
            group = {
                row.get("gate_variant"): row for row in rows
                if _scenario(row) == scenario
                and int(row.get("seed", -1)) == seed
            }
            if not all(variant in group for variant in gate.VARIANTS):
                continue
            records = {
                variant: _record(group[variant]) for variant in gate.VARIANTS
            }
            legacy = records["v41_source_bayes"]
            challenger = records["v42_shared_low_rank_bayes"]
            if (
                legacy["true_feasible"] != challenger["true_feasible"]
                or legacy["adaptive_loss"] != challenger["adaptive_loss"]
                or legacy["x_recommended"] != challenger["x_recommended"]
                or any(record["adaptive_loss"] for record in records.values())
            ):
                paired.append({
                    "scenario": scenario_name,
                    "seed": seed,
                    "arms": records,
                })
    return {
        "per_domain": per_domain,
        "paired_recommendation_changes": paired,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-seeds", type=int, default=5)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = summarize(gate.load_rows(args.root), args.expected_seeds)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
