#!/usr/bin/env python3
"""Analyze the V46 grouped source-task hyperlaw Queue sentinel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from . import analyze_mean_alignment_v45_gate as base
except ImportError:  # Direct script execution.
    import analyze_mean_alignment_v45_gate as base


VARIANTS = (
    "v41_two_task_source_bayes",
    "v45_geometry_source_bayes",
    "v46_grouped_task_bayes",
    "v46_grouped_task_predictive_bayes",
)
SENTINEL_SEEDS = (1, 3)
QUEUE_SCENARIO = ("QueueResourceControl", 1.0)
EXPECTED_AUGMENTS = {
    "v41_two_task_source_bayes": 1,
    "v45_geometry_source_bayes": 4,
    "v46_grouped_task_bayes": 4,
    "v46_grouped_task_predictive_bayes": 4,
}
EXPECTED_HYPERLAW = {
    "v41_two_task_source_bayes": "single_gaussian_draw",
    "v45_geometry_source_bayes": "single_gaussian_draw",
    "v46_grouped_task_bayes": "grouped_task_discrepancy",
    "v46_grouped_task_predictive_bayes": "grouped_task_predictive",
}
GEOMETRY_VARIANTS = set(VARIANTS[1:])


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
    return base._scenario(row)


def _training(row):
    return dict((row.get("meta_prior") or {}).get("training") or {})


def _episode_contract(rows, variant):
    selected = [row for row in rows if row.get("gate_variant") == variant]
    if not selected:
        return False
    augments = EXPECTED_AUGMENTS[variant]
    geometry_expected = variant in GEOMETRY_VARIANTS
    expected_tasks = 2 * augments
    expected_records = 2 * 64
    expected_calls = expected_records * 3
    expected_archive_mode = (
        "exact" if variant == "v41_two_task_source_bayes"
        else "paired_frozen_control"
    )
    archive_fingerprints = set()
    episode_specs_by_seed = []
    for row in selected:
        training = _training(row)
        archive = dict(row.get("initial_design_archive_contract") or {})
        if archive.get("mode") != expected_archive_mode:
            return False
        if archive.get("target_data_used", True):
            return False
        if archive.get("target_oracle_used", True):
            return False
        if variant == "v41_two_task_source_bayes":
            if not archive.get("matches", False):
                return False
        else:
            if archive.get("matches", True):
                return False
            if not archive.get("proposal_frozen_across_arms", False):
                return False
        if training.get("source_seed_mode") != "frozen":
            return False
        if training.get("target_seed_used_for_source_training", True):
            return False
        if training.get("source_episode_target_data_used", True):
            return False
        if training.get("source_episode_target_oracle_used", True):
            return False
        if int(training.get("source_base_domain_count", -1)) != 2:
            return False
        if int(training.get("source_episode_count_per_base_domain", -1)) != augments:
            return False
        if training.get("source_episode_budget_mode") != "per_base_domain":
            return False
        if not training.get("source_episode_cost_matched", False):
            return False
        if int(training.get("source_episode_record_budget", -1)) != expected_records:
            return False
        if int(training.get("source_task_count", -1)) != expected_tasks:
            return False
        if int(training.get("source_archive_simulator_calls", -1)) != expected_calls:
            return False
        adaptation = dict(row.get("source_target_adaptation_contract") or {})
        if int(adaptation.get("source_simulator_calls", -1)) != expected_calls:
            return False
        if adaptation.get("source_oracle_aided", True):
            return False
        specs = list(training.get("source_episode_specs") or [])
        if len(specs) != expected_tasks:
            return False
        expected_per_episode = 64 // augments
        if any(
            int(spec.get("record_count", -1)) != expected_per_episode
            for spec in specs
        ):
            return False
        nontrivial = any(
            np.linalg.norm(np.asarray(
                spec.get("task_geometry_shift", [0.0, 0.0, 0.0]),
                dtype=float,
            )) > 1e-12
            or abs(float(
                spec.get("task_geometry_radius_scale", 1.0)) - 1.0) > 1e-12
            for spec in specs
        )
        if bool(nontrivial) != geometry_expected:
            return False
        archive_fingerprints.add(str(training.get("source_archive_fingerprint")))
        episode_specs_by_seed.append(specs)
    return bool(
        len(archive_fingerprints) == 1
        and all(specs == episode_specs_by_seed[0] for specs in episode_specs_by_seed)
    )


def _hyperlaw_contract(rows, variant):
    selected = [row for row in rows if row.get("gate_variant") == variant]
    if not selected:
        return False
    expected = EXPECTED_HYPERLAW[variant]
    grouped = expected.startswith("grouped_task_")
    predictive = expected == "grouped_task_predictive"
    for row in selected:
        if row.get("source_constraint_mean_hyperlaw_mode") != expected:
            return False
        diagnostics = base.base.base._prior_diagnostics(row)
        if diagnostics.get("configured_hyperlaw_mode") != expected:
            return False
        if diagnostics.get("target_oracle_used", True):
            return False
        if bool(diagnostics.get(
            "grouped_task_prior_selected", False)) != grouped:
            return False
        if not grouped:
            continue
        if int(diagnostics.get("source_base_domain_count", -1)) != 2:
            return False
        if diagnostics.get("source_episode_counts_by_base_domain") != {
            "FactorShockStatePolicyRZDT1": 4,
            "InventorySupplyChain": 4,
        }:
            return False
        if not diagnostics.get(
            "source_estimation_covariance_enters_shared_mean_only", False
        ):
            return False
        if diagnostics.get("within_source_estimation_as_target_variation", True):
            return False
        if not diagnostics.get(
            "within_base_task_covariance_included", False):
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
        if predictive:
            if not diagnostics.get(
                "finite_source_predictive_correction", False):
                return False
            if not diagnostics.get(
                "finite_source_predictive_prior_selected", False):
                return False
        elif diagnostics.get("finite_source_predictive_correction", True):
            return False
    return True


def _paired_target_contract(rows, seeds):
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
            values = {base.base._paired_fingerprint(row, field) for row in group}
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


def _finite_median(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if len(finite) else None


def _mechanism_summary(rows, variant):
    selected = [row for row in rows if row.get("gate_variant") == variant]
    priors = [base.base.base._prior_diagnostics(row) for row in selected]
    audits = [
        dict(row.get("boundary_raw_pool_truth_diagnostics") or {})
        for row in selected
    ]
    return {
        "median_prior_covariance_trace": _finite_median([
            prior.get("prior_covariance_trace", np.nan) for prior in priors
        ]),
        "median_shared_mean_covariance_trace": _finite_median([
            prior.get("shared_mean_covariance_trace", np.nan)
            for prior in priors
        ]),
        "median_between_base_covariance_trace": _finite_median([
            prior.get("between_base_domain_covariance_trace", np.nan)
            for prior in priors
        ]),
        "median_within_base_covariance_trace": _finite_median([
            prior.get("within_base_task_covariance_trace", np.nan)
            for prior in priors
        ]),
        "raw_pool_true_feasible_count": int(sum(int(
            audit.get("boundary_raw_pool_true_feasible_count", 0) or 0)
            for audit in audits)),
        "median_best_feasible_epistemic_radius": _finite_median([
            audit.get("boundary_raw_pool_best_feasible_epistemic_radius", np.nan)
            for audit in audits
        ]),
        "median_best_feasible_true_margin": _finite_median([
            audit.get("boundary_raw_pool_best_feasible_true_margin", np.nan)
            for audit in audits
        ]),
    }


def summarize(rows, seeds=SENTINEL_SEEDS):
    seeds = tuple(int(seed) for seed in seeds)
    selected = [
        row for row in rows
        if _scenario(row) == QUEUE_SCENARIO
        and int(row.get("seed", -1)) in seeds
    ]
    summaries = {
        variant: base.base._variant_summary(selected, variant, len(seeds))
        for variant in VARIANTS
    }
    mechanisms = {
        variant: _mechanism_summary(selected, variant)
        for variant in VARIANTS
    }
    episodes = {
        variant: _episode_contract(selected, variant)
        for variant in VARIANTS
    }
    hyperlaws = {
        variant: _hyperlaw_contract(selected, variant)
        for variant in VARIANTS
    }
    paired = _paired_target_contract(selected, seeds)
    challenger_name = "v46_grouped_task_bayes"
    challenger = summaries[challenger_name]
    controls = [
        summaries["v41_two_task_source_bayes"],
        summaries["v45_geometry_source_bayes"],
    ]
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
    promotion = bool(all((
        challenger["complete"],
        paired,
        all(episodes.values()),
        all(hyperlaws.values()),
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
        "grouped_population_strictly_improves_controls": strict,
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
