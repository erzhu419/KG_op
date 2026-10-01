#!/usr/bin/env python3
"""Post-review controls on the existing primary task seeds.

Rules are in manifests/submission_revision_controls_20260907.json. The original
target evaluator is reused by replacing only its two design-selection functions
inside a process-local context. Existing source and legacy outcomes are read.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.stats import norm

from core.profile_atlas import (
    AtlasMember, AtlasSelection, SourceScoredProfileAtlas,
    farthest_first_indices, percentile_ranks, profile_cosine_coordinate,
    regular_profile_nodes,
)
import performance.benchmark_profile_stress_suite as benchmark
from performance.run_profile_stress_matrix import build_primary_cells

PROTOCOL_PATH = ROOT / "performance/manifests/submission_revision_controls_20260907.json"
OUTPUT_PATH = ROOT / "paper_artifacts/submission_revision_20260907"
NEW_ARMS = ("standardized_generic_dct_maximin", "source_greedy_portfolio")
REFERENCE_ARMS = ("source_atlas", "generic_dct_maximin")


def standardized_generic_dct_maximin(profiles, count, **coordinate_kwargs):
    """Outcome-free medoid/farthest-first in source-identical coordinates."""
    coordinates = np.vstack([
        profile_cosine_coordinate(profile, **coordinate_kwargs)
        for profile in profiles
    ])
    scale = np.std(coordinates, axis=0)
    scale = np.where(scale > 1e-10, scale, 1.0)
    standardized = (coordinates - np.mean(coordinates, axis=0, keepdims=True)) / scale[None, :]
    distances = np.linalg.norm(
        standardized[:, None, :] - standardized[None, :, :], axis=2)
    first = int(np.argmin(np.mean(distances, axis=1)))
    selected = farthest_first_indices(standardized, int(count), initial_index=first)
    return selected, {
        "contract_id": "standardized_generic_dct_maximin_supplement_v1",
        "source_outcomes_used": False,
        "target_outcomes_used": False,
        "target_oracle_used": False,
        "selected_indices": list(selected),
        "first_center_rule": "standardized_finite_library_medoid",
        "remaining_center_rule": "gonzalez_farthest_first",
        "coordinate_dimension": int(standardized.shape[1]),
    }


def greedy_portfolio_indices(scores, count):
    """Select a portfolio from task-by-profile scores, with specified ties."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or not 1 <= int(count) <= scores.shape[1]:
        raise ValueError("portfolio requires task-by-profile scores and a valid count")
    overall = np.mean(scores, axis=0)
    incumbent = np.full(scores.shape[0], np.inf)
    selected = []
    objective_path = []
    for _ in range(int(count)):
        candidates = [j for j in range(scores.shape[1]) if j not in selected]
        choice = min(candidates, key=lambda j: (
            float(np.mean(np.minimum(incumbent, scores[:, j]))),
            float(overall[j]), j,
        ))
        selected.append(choice)
        incumbent = np.minimum(incumbent, scores[:, choice])
        objective_path.append(float(np.mean(incumbent)))
    return tuple(selected), objective_path


def source_greedy_portfolio(records, count, variance_floor=1e-6):
    """Use the same source statistics as the atlas, with uniform source weights.

    Returns an AtlasSelection for direct use by the synthetic or Energy runner.
    This API takes source records only and never accepts target observations.
    """
    records = tuple(records)
    by_pair = {(str(r.task_id), str(r.profile_id)): r for r in records}
    tasks = sorted({str(r.task_id) for r in records})
    profiles = sorted({str(r.profile_id) for r in records})
    if not tasks or len(by_pair) != len(records) or len(records) != len(tasks) * len(profiles):
        raise ValueError("portfolio requires one record per shared source task/profile pair")
    safety = np.empty((len(tasks), len(profiles)))
    objective = np.empty_like(safety)
    margins = np.empty_like(safety)
    for ti, task in enumerate(tasks):
        source_objectives = []
        for pi, profile in enumerate(profiles):
            record = by_pair[(task, profile)]
            samples = np.asarray(record.constraint_samples, dtype=float)
            variance = float(np.var(samples, ddof=1)) if len(samples) > 1 else 0.0
            margins[ti, pi] = (
                float(np.mean(samples))
                + float(norm.ppf(1.0 - record.alpha)) * math.sqrt(max(variance, variance_floor))
                - record.tau
            )
            source_objectives.append(float(np.mean(record.objective_samples)))
        safety[ti] = percentile_ranks(margins[ti])
        objective[ti] = percentile_ranks(source_objectives)
    scores = 0.5 * safety + 0.5 * objective
    selected, objective_path = greedy_portfolio_indices(scores, count)
    members = []
    for order, index in enumerate(selected, 1):
        record = by_pair[(tasks[0], profiles[index])]
        members.append(AtlasMember(
            profile_id=profiles[index],
            profile=tuple(record.profile),
            nodes=tuple(record.nodes) if record.nodes else tuple(regular_profile_nodes(len(record.profile))),
            safety_rank=float(np.mean(safety[:, index])),
            objective_rank=float(np.mean(objective[:, index])),
            robust_source_feasible=bool(np.all(margins[:, index] <= 0)),
            selected_order=order,
        ))
    return AtlasSelection(tuple(members), {
        "contract_id": "source_greedy_portfolio_supplement_v1",
        "source_outcomes_used": True,
        "target_outcomes_used": False,
        "target_oracle_used": False,
        "source_task_count": len(tasks),
        "source_task_weights": {task: 1.0 / len(tasks) for task in tasks},
        "selected_profile_ids": [member.profile_id for member in members],
        "source_portfolio_objective_path": objective_path,
        "single_profile_score": "0.5*safety_percentile_rank+0.5*objective_percentile_rank",
        "selection_rule": "greedy_mean_across_source_tasks_of_best_selected_profile_score",
    })


class _PortfolioSelector(SourceScoredProfileAtlas):
    def fit(self, records, *, target_descriptor=None):
        if target_descriptor is not None:
            raise ValueError("supplemental source portfolio uses uniform source weights")
        self.selection = source_greedy_portfolio(records, self.config.n0, self.config.variance_floor)
        return self


def compact_row(result, *, arm=None, provenance):
    keys = (
        "regime", "target_seed", "design_seed", "nominal_dimension", "effective_rank",
        "initial_design_contains_true_feasible", "independently_certified",
        "false_certificate", "penalized_loss", "source_calls", "target_search_calls",
        "verification_calls", "all_in_calls_unamortized", "selected_profile_ids",
    )
    row = {key: result[key] for key in keys}
    truth = result.get("deployed_truth")
    row.update({
        "arm": arm or result["arm"],
        "deployed_true_feasible": bool(truth is not None and truth["feasible"]),
        "certified_true_feasible": bool(result["independently_certified"] and not result["false_certificate"] and truth is not None and truth["feasible"]),
        "all_in_calls_amortized": result["source_calls"] / 20.0 + result["target_search_calls"] + result["verification_calls"],
        "provenance": provenance,
    })
    return row


def run_control(cell):
    arm = cell["arm"]
    kwargs = {key: cell[key] for key in ("regime", "target_seed", "design_seed", "dimension", "N", "schema_mode", "descriptor_mode")}
    if arm == NEW_ARMS[0]:
        with patch.object(benchmark, "generic_dct_maximin", standardized_generic_dct_maximin):
            result = benchmark.run_task(arm="generic_dct_maximin", **kwargs)
    elif arm == NEW_ARMS[1]:
        with patch.object(benchmark, "SourceScoredProfileAtlas", _PortfolioSelector):
            result = benchmark.run_task(arm="source_atlas", **kwargs)
    else:
        raise ValueError(f"unknown supplemental arm: {arm}")
    row = compact_row(result, arm=arm, provenance="new_post_review_evaluation")
    row["frontend_diagnostics"] = result["frontend_diagnostics"]
    return row


def row_key(row):
    return row["arm"], int(row["nominal_dimension"]), row["regime"], int(row["target_seed"])


def summarize(rows):
    count = len(rows)
    certified = sum(row["certified_true_feasible"] for row in rows)
    covered = sum(row["initial_design_contains_true_feasible"] for row in rows)
    return {
        "arm": rows[0]["arm"],
        "independent_task_count": count,
        "N": 10, "n0": 10,
        "true_feasible_coverage_count": covered,
        "initial_design_true_feasible_coverage_count": covered,
        "initial_design_true_feasible_coverage_rate": covered / count,
        "certified_true_feasible_deployment_count": certified,
        "certified_true_feasible_deployment_rate": certified / count,
        "independently_certified_count": sum(row["independently_certified"] for row in rows),
        "false_certificate_count": sum(row["false_certificate"] for row in rows),
        "mean_penalized_loss": float(np.mean([row["penalized_loss"] for row in rows])),
        "mean_source_calls": float(np.mean([row["source_calls"] for row in rows])),
        "mean_target_search_calls": float(np.mean([row["target_search_calls"] for row in rows])),
        "mean_verification_calls": float(np.mean([row["verification_calls"] for row in rows])),
        "mean_all_in_calls_unamortized": float(np.mean([row["all_in_calls_unamortized"] for row in rows])),
        "mean_all_in_calls_amortized": float(np.mean([row["all_in_calls_amortized"] for row in rows])),
    }


def _crossing(source_difference, recurrent_difference):
    """Integers M>=1 with difference source_difference/M+recurrent_difference<=0."""
    a, b = float(source_difference), float(recurrent_difference)
    if b < 0:
        start = max(1, math.ceil(a / -b))
        return {"first_M": start, "last_M": None, "holds_for_all_M_from": start}
    if b == 0:
        return {"first_M": 1 if a <= 0 else None, "last_M": None, "holds_for_all_M_from": 1 if a <= 0 else None}
    last = math.floor(-a / b)
    return {"first_M": 1 if last >= 1 else None, "last_M": last if last >= 1 else None, "holds_for_all_M_from": None}


def analyze_rows(rows, *, protocol):
    by_key = {row_key(row): row for row in rows}
    if len(by_key) != len(rows):
        raise RuntimeError("duplicate supplemental/reference scalar cells")
    expected = build_primary_cells(freeze_commit=protocol["historical_seed_freeze_commit"], arms=REFERENCE_ARMS + NEW_ARMS)
    expected_keys = {(c["arm"], c["dimension"], c["regime"], c["target_seed"]) for c in expected}
    if set(by_key) != expected_keys:
        raise RuntimeError("incomplete or wrong primary task matrix")
    for cell in expected:
        row = by_key[(cell["arm"], cell["dimension"], cell["regime"], cell["target_seed"])]
        if row["design_seed"] != cell["design_seed"]:
            raise RuntimeError("paired algorithm seed differs from frozen task mapping")
    summaries = []
    dimensions = protocol["matrix"]["dimensions"]
    regimes = protocol["matrix"]["regimes"]
    arms = REFERENCE_ARMS + NEW_ARMS
    dimension_summaries = []
    for dimension in dimensions:
        for arm in arms:
            selected = [r for r in rows if r["nominal_dimension"] == dimension and r["arm"] == arm]
            dimension_summaries.append({"nominal_dimension": dimension, **summarize(selected)})
            for regime in regimes:
                group = [r for r in selected if r["regime"] == regime]
                summaries.append({"nominal_dimension": dimension, "regime": regime, "effective_rank": group[0]["effective_rank"], **summarize(group)})
    # Detect stale or mismatched local source/legacy evidence before publishing comparisons.
    compact = json.loads((ROOT / "paper_artifacts/or_review/randomized_profile_primary.json").read_text())["aggregate_analysis"]["summaries"]
    new_summaries = {(r["arm"], r["nominal_dimension"], r["regime"]): r for r in summaries}
    verified_groups = 0
    fields = ("independent_task_count", "true_feasible_coverage_count", "initial_design_true_feasible_coverage_count", "certified_true_feasible_deployment_count", "false_certificate_count", "mean_penalized_loss", "mean_verification_calls", "mean_all_in_calls_unamortized", "mean_all_in_calls_amortized")
    for reference in compact:
        if reference["arm"] not in REFERENCE_ARMS:
            continue
        actual = new_summaries[(reference["arm"], reference["nominal_dimension"], reference["regime"])]
        for field in fields:
            if not math.isclose(actual[field], reference[field], rel_tol=1e-12, abs_tol=1e-12):
                raise RuntimeError(f"local raw evidence differs from current compact summary: {reference['arm']} {reference['nominal_dimension']} {reference['regime']} {field}")
        verified_groups += 1
    paired = []
    costs = []
    endpoint_fields = ("certified_true_feasible", "initial_design_contains_true_feasible", "penalized_loss", "verification_calls")
    for dimension in dimensions:
        for control in (REFERENCE_ARMS[1],) + NEW_ARMS:
            differences = []
            for regime in regimes:
                seeds = sorted(r["target_seed"] for r in rows if r["arm"] == "source_atlas" and r["nominal_dimension"] == dimension and r["regime"] == regime)
                differences.append([[float(by_key[("source_atlas", dimension, regime, seed)][field]) - float(by_key[(control, dimension, regime, seed)][field]) for field in endpoint_fields] for seed in seeds])
            delta = np.asarray(differences)
            rng = np.random.default_rng(protocol["analysis"]["bootstrap_seed"])
            indices = rng.integers(0, 20, size=(10000, len(regimes), 20))
            boot = delta[np.arange(len(regimes))[None, :, None], indices].mean(axis=2)
            for regime_index in [None] + list(range(len(regimes))):
                values = delta if regime_index is None else delta[regime_index]
                samples = boot.mean(axis=1) if regime_index is None else boot[:, regime_index]
                endpoint_summary = {}
                for endpoint_index, field in enumerate(endpoint_fields):
                    field_values = values[..., endpoint_index]
                    endpoint_summary[field] = {
                        "source_minus_control_mean": float(np.mean(field_values)),
                        "paired_bootstrap_95ci": [float(v) for v in np.quantile(samples[:, endpoint_index], [0.025, 0.975])],
                    }
                    if endpoint_index < 2:
                        endpoint_summary[field].update({"source_only_count": int(np.sum(field_values == 1)), "control_only_count": int(np.sum(field_values == -1)), "tie_count": int(np.sum(field_values == 0))})
                paired.append({"nominal_dimension": dimension, "regime": "all_fixed_regimes" if regime_index is None else regimes[regime_index], "source_arm": "source_atlas", "control_arm": control, "task_count": 160 if regime_index is None else 20, "bootstrap_replicates": 10000, "endpoints": endpoint_summary})
            groups = {r["arm"]: r for r in dimension_summaries if r["nominal_dimension"] == dimension}
            source, comparison = groups["source_atlas"], groups[control]
            recurrent = lambda r: r["mean_target_search_calls"] + r["mean_verification_calls"]
            probability = lambda r: r["certified_true_feasible_deployment_rate"]
            cost_row = {"nominal_dimension": dimension, "control_arm": control, "source_call_break_even_targets": _crossing(source["mean_source_calls"] - comparison["mean_source_calls"], recurrent(source) - recurrent(comparison)), "by_M": {}}
            if probability(source) > 0 and probability(comparison) > 0:
                cost_row["source_calls_per_certificate_break_even_targets"] = _crossing(source["mean_source_calls"] / probability(source) - comparison["mean_source_calls"] / probability(comparison), recurrent(source) / probability(source) - recurrent(comparison) / probability(comparison))
            for m in (1, 20, 100):
                cost_row["by_M"][str(m)] = {}
                for arm, group in (("source_atlas", source), (control, comparison)):
                    total = group["mean_source_calls"] / m + recurrent(group)
                    cost_row["by_M"][str(m)][arm] = {"mean_all_in_calls": total, "calls_per_true_certificate": total / probability(group) if probability(group) > 0 else None}
            costs.append(cost_row)
    return {"schema_version": 1, "status": "complete", "interpretation": protocol["interpretation"], "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)), "row_count": len(rows), "independent_latent_task_count": 160, "raw_reference_validation": {"matching_source_legacy_regime_dimension_summaries": verified_groups, "fields": list(fields), "exact_task_and_design_seed_pairing": True}, "summaries": summaries, "dimension_summaries": dimension_summaries, "paired_comparisons": paired, "cost_comparisons": costs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL_PATH.read_text())
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    if args.analyze_only:
        payload = json.loads((OUTPUT_PATH / "rows.json").read_text())
        rows = payload["rows"]
    else:
        start = time.monotonic()
        started_at = datetime.now(timezone.utc).isoformat()
        rows = []
        raw_dir = ROOT / "results/or_review_v2_evalpatch4/profile_primary"
        for arm in REFERENCE_ARMS:
            for path in sorted(raw_dir.glob(f"**/*__{arm}__*.json")):
                result = json.loads(path.read_text())
                if result["nominal_dimension"] in protocol["matrix"]["dimensions"]:
                    rows.append(compact_row(result, provenance=str(path.relative_to(ROOT))))
        # These prior diagnostic outcomes are reused without another simulation.
        diagnostic_path = ROOT / protocol[NEW_ARMS[0]]["previous_d1000_results"]
        diagnostic = json.loads(diagnostic_path.read_text())
        rank_by_task = {(r["regime"], r["target_seed"]): r["effective_rank"] for r in rows if r["nominal_dimension"] == 1000 and r["arm"] == "source_atlas"}
        for result in diagnostic["rows"]:
            normalized = {**result, "nominal_dimension": 1000, "effective_rank": rank_by_task[(result["regime"], result["target_seed"])], "arm": NEW_ARMS[0]}
            rows.append(compact_row(normalized, provenance=str(diagnostic_path.relative_to(ROOT))))
        cells = build_primary_cells(freeze_commit=protocol["historical_seed_freeze_commit"], arms=NEW_ARMS)
        cells = [cell for cell in cells if not (cell["arm"] == NEW_ARMS[0] and cell["dimension"] == 1000)]
        print(f"reference_rows={len(rows)-160} reused_diagnostic_rows=160 new_cells={len(cells)} workers={args.workers}", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(run_control, cell) for cell in cells]
            for done, future in enumerate(as_completed(futures), 1):
                rows.append(future.result())
                if done == 1 or done % 40 == 0 or done == len(cells):
                    print(f"completed={done}/{len(cells)} elapsed_seconds={time.monotonic()-start:.1f}", flush=True)
        rows.sort(key=row_key)
        payload = {"schema_version": 1, "interpretation": protocol["interpretation"], "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)), "started_at_utc": started_at, "finished_at_utc": datetime.now(timezone.utc).isoformat(), "new_evaluated_cell_count": len(cells), "reused_diagnostic_cell_count": 160, "read_original_cell_count": len(rows)-len(cells)-160, "rows": rows}
        (OUTPUT_PATH / "rows.json").write_text(json.dumps(payload, indent=2) + "\n")
    analysis = analyze_rows(rows, protocol=protocol)
    (OUTPUT_PATH / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    for row in analysis["dimension_summaries"]:
        print(json.dumps(row), flush=True)
    print(f"SUPPLEMENT_COMPLETE rows={len(rows)} output={OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
