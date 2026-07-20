#!/usr/bin/env python3
"""Analyze the V47 random-effects deconvolution Queue sentinel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from . import analyze_mean_alignment_v46_gate as base
except ImportError:  # Direct script execution.
    import analyze_mean_alignment_v46_gate as base


VARIANTS = (
    "v41_two_task_source_bayes",
    "v45_geometry_source_bayes",
    "v46_grouped_task_bayes",
    "v47_deconvolved_task_bayes",
    "v47_deconvolved_task_predictive_bayes",
)
SENTINEL_SEEDS = (1, 3)
QUEUE_SCENARIO = base.QUEUE_SCENARIO
EXPECTED_HYPERLAW = {
    "v41_two_task_source_bayes": "single_gaussian_draw",
    "v45_geometry_source_bayes": "single_gaussian_draw",
    "v46_grouped_task_bayes": "grouped_task_discrepancy",
    "v47_deconvolved_task_bayes": "grouped_task_deconvolved",
    "v47_deconvolved_task_predictive_bayes": (
        "grouped_task_deconvolved_predictive"),
}
V47_VARIANTS = set(VARIANTS[-2:])
V43_ANALYZER = base.base.base
V42_ANALYZER = V43_ANALYZER.base


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


def _episode_contract(rows, variant):
    if variant not in V47_VARIANTS:
        return base._episode_contract(rows, variant)
    selected = [
        dict(row, gate_variant="v46_grouped_task_bayes")
        for row in rows if row.get("gate_variant") == variant
    ]
    return base._episode_contract(selected, "v46_grouped_task_bayes")


def _hyperlaw_contract(rows, variant):
    if variant not in V47_VARIANTS:
        return base._hyperlaw_contract(rows, variant)
    selected = [row for row in rows if row.get("gate_variant") == variant]
    if not selected:
        return False
    expected = EXPECTED_HYPERLAW[variant]
    predictive = expected.endswith("_predictive")
    for row in selected:
        if row.get("source_constraint_mean_hyperlaw_mode") != expected:
            return False
        diagnostics = V42_ANALYZER._prior_diagnostics(row)
        if diagnostics.get("configured_hyperlaw_mode") != expected:
            return False
        if diagnostics.get("target_oracle_used", True):
            return False
        if not diagnostics.get("grouped_task_prior_selected", False):
            return False
        if not diagnostics.get(
            "random_effects_deconvolution_selected", False):
            return False
        if not diagnostics.get("random_effects_deconvolution", False):
            return False
        if not diagnostics.get(
            "source_estimation_covariance_used_for_deconvolution", False
        ):
            return False
        if int(diagnostics.get("source_base_domain_count", -1)) != 2:
            return False
        if diagnostics.get("source_episode_counts_by_base_domain") != {
            "FactorShockStatePolicyRZDT1": 4,
            "InventorySupplyChain": 4,
        }:
            return False
        for rank_name, upper_name in (
            (
                "between_base_discrepancy_rank",
                "between_base_discrepancy_rank_upper_bound",
            ),
            (
                "within_base_task_discrepancy_rank",
                "within_base_task_discrepancy_rank_upper_bound",
            ),
            (
                "combined_discrepancy_rank",
                "combined_discrepancy_rank_upper_bound",
            ),
        ):
            rank = int(diagnostics.get(rank_name, -1))
            upper = int(diagnostics.get(upper_name, -1))
            if rank < 0 or upper < 0 or rank > upper:
                return False
        for prefix, corrected_key in (
            ("channel_role", "channel_role_covariance_trace"),
            ("between_base", "between_base_domain_covariance_trace"),
            ("within_base", "within_base_task_covariance_trace"),
        ):
            observed = float(diagnostics.get(
                f"{prefix}_observed_covariance_trace", np.nan))
            noise = float(diagnostics.get(
                f"{prefix}_estimation_noise_trace", np.nan))
            corrected = float(diagnostics.get(corrected_key, np.nan))
            if not all(np.isfinite(value) for value in (
                observed, noise, corrected
            )):
                return False
            if observed < 0.0 or noise <= 0.0 or corrected < 0.0:
                return False
            if corrected > observed + 1e-10:
                return False
        if bool(diagnostics.get(
            "finite_source_predictive_prior_selected", False
        )) != predictive:
            return False
        expected_law = (
            "shared_base_mean_plus_deconvolved_predictive_discrepancy"
            if predictive else
            "shared_base_mean_plus_deconvolved_task_discrepancy"
        )
        if diagnostics.get("target_task_law") != expected_law:
            return False
    return True


def _paired_target_contract(rows, seeds):
    for seed in seeds:
        group = [
            row for row in rows
            if base._scenario(row) == QUEUE_SCENARIO
            and int(row.get("seed", -1)) == int(seed)
            and row.get("gate_variant") in VARIANTS
        ]
        if len(group) != len(VARIANTS):
            return False
        if any(row.get("decision_backend") != "sobol_new" for row in group):
            return False
        proposal_archives = {
            str((row.get("initial_design_archive_contract") or {}).get(
                "proposal_archive_fingerprint", ""))
            for row in group
        }
        if len(proposal_archives) != 1 or "" in proposal_archives:
            return False
        for field in (
            "target_design_fingerprint",
            "online_action_sequence_fingerprint",
        ):
            values = {V43_ANALYZER._paired_fingerprint(
                row, field) for row in group}
            if len(values) != 1 or "missing" in values:
                return False
        traces = [list(row.get("online_action_trace") or []) for row in group]
        reference = traces[0] if traces else []
        if not reference:
            return False
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


def _deconvolution_summary(rows, variant):
    selected = [row for row in rows if row.get("gate_variant") == variant]
    priors = [V42_ANALYZER._prior_diagnostics(row) for row in selected]
    result = {}
    for prefix, corrected_key in (
        ("channel_role", "channel_role_covariance_trace"),
        ("between_base", "between_base_domain_covariance_trace"),
        ("within_base", "within_base_task_covariance_trace"),
    ):
        result[f"median_{prefix}_observed_covariance_trace"] = (
            base._finite_median([
                prior.get(f"{prefix}_observed_covariance_trace", np.nan)
                for prior in priors
            ])
        )
        result[f"median_{prefix}_estimation_noise_trace"] = (
            base._finite_median([
                prior.get(f"{prefix}_estimation_noise_trace", np.nan)
                for prior in priors
            ])
        )
        result[f"median_{prefix}_corrected_covariance_trace"] = (
            base._finite_median([
                prior.get(corrected_key, np.nan) for prior in priors
            ])
        )
    return result


def summarize(rows, seeds=SENTINEL_SEEDS):
    seeds = tuple(int(seed) for seed in seeds)
    selected = [
        row for row in rows
        if base._scenario(row) == QUEUE_SCENARIO
        and int(row.get("seed", -1)) in seeds
    ]
    summaries = {
        variant: V43_ANALYZER._variant_summary(
            selected, variant, len(seeds))
        for variant in VARIANTS
    }
    mechanisms = {
        variant: base._mechanism_summary(selected, variant)
        for variant in VARIANTS
    }
    for variant in V47_VARIANTS:
        mechanisms[variant].update(
            _deconvolution_summary(selected, variant))
    episodes = {
        variant: _episode_contract(selected, variant)
        for variant in VARIANTS
    }
    hyperlaws = {
        variant: _hyperlaw_contract(selected, variant)
        for variant in VARIANTS
    }
    paired = _paired_target_contract(selected, seeds)
    challenger_name = "v47_deconvolved_task_bayes"
    challenger = summaries[challenger_name]
    controls = [summaries[name] for name in VARIANTS[:3]]
    false_certificates = (
        challenger["audit_pool_false_certificates"]
        + challenger["evaluated_false_certificates"])
    strict = bool(
        challenger["true_feasible"] > max(
            control["true_feasible"] for control in controls)
        or challenger["adaptive_losses"] < min(
            control["adaptive_losses"] for control in controls)
        or challenger["adaptive_improvements"] > max(
            control["adaptive_improvements"] for control in controls)
    )
    radius = mechanisms[challenger_name][
        "median_best_feasible_epistemic_radius"]
    control_radius = mechanisms["v46_grouped_task_bayes"][
        "median_best_feasible_epistemic_radius"]
    radius_contracts = bool(
        radius is not None and control_radius is not None
        and float(radius) < float(control_radius) - 1e-12
    )
    promotion = bool(all((
        challenger["complete"],
        paired,
        all(episodes.values()),
        all(hyperlaws.values()),
        radius_contracts,
        challenger["true_feasible"] == len(seeds),
        challenger["adaptive_losses"] == 0,
        false_certificates == 0,
        strict,
    )))
    return {
        "scope": "queue_failure_sentinel",
        "seeds": list(seeds),
        "fixed_source_call_budget_per_variant": 384,
        "paired_initial_design_actions_and_target_responses": bool(paired),
        "source_episode_contract": episodes,
        "source_hyperlaw_contract": hyperlaws,
        "variant_summaries": summaries,
        "mechanism_summaries": mechanisms,
        "deconvolution_contracts_epistemic_radius": radius_contracts,
        "deconvolved_population_strictly_improves_controls": strict,
        "promotion_eligible": [challenger_name] if promotion else [],
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
