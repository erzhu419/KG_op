#!/usr/bin/env python3
"""Analyze the oracle-free replicated HVD identifiability gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


MODES = ("pooled", "factor_cumulative")
SHOCK_SCALES = (0.0, 4.0)
REPLICATIONS = (2, 4, 8)


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value == value and abs(value) != float("inf") else None


def _median(values):
    values = [item for item in (_finite(value) for value in values)
              if item is not None]
    return None if not values else float(median(values))


def load_rows(root):
    rows = []
    for path in sorted(Path(root).rglob("result.json")):
        payload = json.loads(path.read_text())
        if payload.get("experiment") != "hvd_identifiability":
            continue
        if str(payload.get("mode")) not in MODES:
            continue
        row = dict(payload)
        row["result_path"] = str(path)
        rows.append(row)
    return rows


def _cell(items):
    return {
        "count": int(len(items)),
        "median_log_variance_rmse": _median(
            item.get("log_variance_rmse") for item in items),
        "median_variance_spearman": _median(
            item.get("variance_spearman") for item in items),
        "median_shared_risk_spearman": _median(
            item.get("shared_risk_spearman") for item in items),
        "median_predicted_variance": _median(
            item.get("median_predicted_variance") for item in items),
        "median_true_variance": _median(
            item.get("median_true_variance") for item in items),
        "median_fitted_shared_risk": _median(
            item.get("median_fitted_shared_risk") for item in items),
        "false_feasible_count": int(sum(
            int(item.get("false_feasible_count", 0)) for item in items)),
        "nonvacuous_count": int(sum(
            bool(item.get("certificate_nonvacuous", False)) for item in items)),
        "median_certificate_precision": _median(
            item.get("certificate_precision") for item in items),
        "median_certificate_recall": _median(
            item.get("certificate_recall") for item in items),
    }


def summarize(rows, expected_seeds=5):
    grouped = {
        (mode, shock, replication): [
            row for row in rows
            if str(row.get("mode")) == mode
            and float(row.get("shared_shock_scale")) == shock
            and int(row.get("replicates_per_policy")) == replication
        ]
        for mode in MODES
        for shock in SHOCK_SCALES
        for replication in REPLICATIONS
    }
    cells = {
        f"{mode}/shock{shock:g}/rep{replication}": _cell(items)
        for (mode, shock, replication), items in grouped.items()
    }
    index = {
        (
            str(row["mode"]),
            float(row["shared_shock_scale"]),
            int(row["replicates_per_policy"]),
            int(row["seed"]),
        ): row
        for row in rows
    }
    complete_keys = {
        (mode, shock, replication, seed)
        for mode in MODES
        for shock in SHOCK_SCALES
        for replication in REPLICATIONS
        for seed in range(expected_seeds)
    }
    information_contract = bool(rows) and all(
        not row.get("information_contract", {}).get(
            "oracle_used_for_fit", True)
        and row.get("information_contract", {}).get(
            "oracle_used_for_post_run_audit", False)
        and not row.get("information_contract", {}).get(
            "true_constraint_mean_used_for_fit", True)
        and row.get("information_contract", {}).get("fit_inputs")
        == "ordinary_replicate_sample_mean_and_sample_variance"
        for row in rows
    )

    scale_order = []
    factor_minus_pooled_rmse = []
    factor_minus_pooled_rank = []
    factor_minus_pooled_false = []
    factor_variance_ranks = []
    for replication in REPLICATIONS:
        for seed in range(expected_seeds):
            weak = index.get(("factor_cumulative", 0.0, replication, seed))
            strong = index.get(("factor_cumulative", 4.0, replication, seed))
            if weak is not None and strong is not None:
                scale_order.append(
                    float(strong["median_predicted_variance"])
                    > float(weak["median_predicted_variance"]))
            for shock in SHOCK_SCALES:
                pooled = index.get(("pooled", shock, replication, seed))
                factor = index.get(
                    ("factor_cumulative", shock, replication, seed))
                if pooled is None or factor is None:
                    continue
                factor_minus_pooled_rmse.append(
                    float(factor["log_variance_rmse"])
                    - float(pooled["log_variance_rmse"]))
                if (
                    factor.get("variance_spearman") is not None
                    and pooled.get("variance_spearman") is not None
                ):
                    factor_minus_pooled_rank.append(
                        float(factor["variance_spearman"])
                        - float(pooled["variance_spearman"]))
                if factor.get("variance_spearman") is not None:
                    factor_variance_ranks.append(
                        float(factor["variance_spearman"]))
                factor_minus_pooled_false.append(
                    int(factor["false_feasible_count"])
                    - int(pooled["false_feasible_count"]))

    factor_replication_rmse = []
    factor_replication_false = []
    for shock in SHOCK_SCALES:
        for seed in range(expected_seeds):
            low = index.get(("factor_cumulative", shock, 2, seed))
            high = index.get(("factor_cumulative", shock, 8, seed))
            if low is None or high is None:
                continue
            factor_replication_rmse.append(
                float(high["log_variance_rmse"])
                - float(low["log_variance_rmse"]))
            factor_replication_false.append(
                int(high["false_feasible_count"])
                - int(low["false_feasible_count"]))

    diagnostics = {
        "scale4_predicted_variance_exceeds_scale0_fraction": (
            None if not scale_order else float(sum(scale_order) / len(scale_order))),
        "factor_minus_pooled_median_log_rmse": _median(
            factor_minus_pooled_rmse),
        "factor_minus_pooled_median_variance_rank": _median(
            factor_minus_pooled_rank),
        "factor_median_variance_rank": _median(factor_variance_ranks),
        "factor_positive_variance_rank_fraction": (
            None if not factor_variance_ranks else float(sum(
                rank > 0.0 for rank in factor_variance_ranks)
                / len(factor_variance_ranks))),
        "paired_variance_rank_count": int(len(factor_minus_pooled_rank)),
        "factor_minus_pooled_false_feasible_net": int(sum(
            factor_minus_pooled_false)),
        "factor_rep8_minus_rep2_median_log_rmse": _median(
            factor_replication_rmse),
        "factor_rep8_minus_rep2_false_feasible_net": int(sum(
            factor_replication_false)),
    }
    criteria = {
        "complete": set(index) == complete_keys,
        "ordinary_replicated_fit_only": information_contract,
        "factor_recovers_shock_scale_ordering": bool(
            diagnostics[
                "scale4_predicted_variance_exceeds_scale0_fraction"] is not None
            and diagnostics[
                "scale4_predicted_variance_exceeds_scale0_fraction"] >= 0.8),
        "factor_rmse_strictly_better_than_pooled": bool(
            diagnostics["factor_minus_pooled_median_log_rmse"] is not None
            and diagnostics["factor_minus_pooled_median_log_rmse"] < 0.0),
        # Pooled variance is constant over candidates, so its Spearman rank is
        # mathematically undefined. Use the paired difference when available;
        # otherwise require factor-HVD itself to recover a positive ordering in
        # at least 80% of the independently seeded cells.
        "factor_rank_strictly_better_than_pooled": bool(
            (
                diagnostics[
                    "factor_minus_pooled_median_variance_rank"] is not None
                and diagnostics[
                    "factor_minus_pooled_median_variance_rank"] > 0.0
            )
            or (
                diagnostics["paired_variance_rank_count"] == 0
                and diagnostics["factor_median_variance_rank"] is not None
                and diagnostics["factor_median_variance_rank"] > 0.0
                and diagnostics[
                    "factor_positive_variance_rank_fraction"] is not None
                and diagnostics[
                    "factor_positive_variance_rank_fraction"] >= 0.8
            )),
        "factor_false_feasible_nonworse_than_pooled": bool(
            diagnostics["factor_minus_pooled_false_feasible_net"] <= 0),
        "more_replication_improves_factor_rmse": bool(
            diagnostics["factor_rep8_minus_rep2_median_log_rmse"] is not None
            and diagnostics["factor_rep8_minus_rep2_median_log_rmse"] < 0.0),
        "more_replication_does_not_add_false_feasible": bool(
            diagnostics["factor_rep8_minus_rep2_false_feasible_net"] <= 0),
    }
    return {
        "schema_version": 1,
        "row_count": int(len(rows)),
        "expected_row_count": int(
            len(MODES) * len(SHOCK_SCALES) * len(REPLICATIONS)
            * expected_seeds),
        "cells": cells,
        "diagnostics": diagnostics,
        "criteria": criteria,
        "gate_pass": bool(all(criteria.values())),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-seeds", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    result = summarize(load_rows(args.root), args.expected_seeds)
    output = args.out or args.root / "hvd_replication_identifiability_gate.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "gate_pass": result["gate_pass"],
        "criteria": result["criteria"],
        "diagnostics": result["diagnostics"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
