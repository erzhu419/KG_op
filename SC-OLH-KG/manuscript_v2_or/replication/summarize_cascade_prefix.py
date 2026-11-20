"""Summarize compact receipts for the frozen cascade query experiment."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import median

from scipy.stats import spearmanr


def summarize(rows: list[dict]) -> dict:
    result = {}
    for family in sorted({r["family"] for r in rows}):
        group = [row for row in rows if row["family"] == family]
        stable = [
            row for row in group
            if row["top"]["full_9"] == row["top"]["full_17"]
            and row["top"]["single_9"] == row["top"]["single_17"]
        ]
        disagrees = lambda row: row["top"]["single_17"] != row["top"]["full_17"]
        regret = [row["full_value_gap_from_restricted_single"] for row in group]
        backup = [row["baseline_backup_value"] for row in group]
        corr = float(spearmanr(backup, regret).statistic) if len(set(regret)) > 1 else float("nan")
        mid = median(backup)
        result[family] = {
            "states": len(group),
            "query_count_min_max": [min(row["candidate_count"] for row in group),
                                    max(row["candidate_count"] for row in group)],
            "audit_status_quo_classes": dict(Counter(row["audit_status_quo_class"] for row in group)),
            "anchor_risk_min_max": [min(row["anchor_p"] for row in group), max(row["anchor_p"] for row in group)],
            "disagreement": sum(map(disagrees, group)),
            "actual_implementation_disagreement": sum(
                row["saved_query"] != row["top"]["full_17"] for row in group
            ),
            "full_order_agrees": sum(row["top"]["full_9"] == row["top"]["full_17"] for row in group),
            "single_order_agrees": sum(row["top"]["single_9"] == row["top"]["single_17"] for row in group),
            "stable_states": len(stable),
            "stable_disagreement": sum(map(disagrees, stable)),
            "median_rank_correlation": median(row["rank_correlation_restricted"] for row in group),
            "median_regret": median(regret),
            "max_regret": max(regret),
            "median_backup_value": mid,
            "rank_correlation_of_backup_and_regret": corr if math.isfinite(corr) else None,
            "disagreement_by_backup_value": {
                side: {"n": len(selected), "disagreement": sum(map(disagrees, selected))}
                for side, selected in (("at_or_below_median", [r for r in group if r["baseline_backup_value"] <= mid]),
                                       ("above_median", [r for r in group if r["baseline_backup_value"] > mid]))
            },
            "total_queries": sum(row["candidate_count"] for row in group),
            "total_retained_queries": sum(row["retained_queries"] for row in group),
            "states_retaining_full_best": sum(row["retains_full_best"] for row in group),
            "states_certifying_a_unique_query": sum(row["retained_queries"] == 1 for row in group),
            "retain_min_max": [min(row["retained_queries"] for row in group),
                               max(row["retained_queries"] for row in group)],
            "largest_value_outside_frechet_interval": max(row["value_outside_frechet_interval"] for row in group),
            "max_quadrature_value_difference": max(row["quadrature_full_values_max_difference"] for row in group),
            "median_kernel_cpu_seconds": {
                key: median(row["timing"][key] for row in group)
                for key in group[0]["timing"]
            },
        }
    return result


def summarize_refined(rows: list[dict]) -> dict:
    summary = {}
    for family in sorted({row["family"] for row in rows}):
        group = [r for r in rows if r["family"] == family]
        stable = [r for r in group if r["single_24_and_48_agree"] and r["full_numerical_top_agreement"]]
        only = [r for r in group if r["retained_queries"] == 1]
        agrees = lambda r: r["first_slot_top"] == r["full_top_mean32768"]
        regret = [r["fine_mean_full_value_gap_from_restricted_single"] for r in group]
        backup = [r["baseline_backup_value_refined"] for r in group]
        corr = float(spearmanr(backup, regret).statistic) if len(set(regret)) > 1 else float("nan")
        mid = median(backup)
        summary[family] = {
            "states": len(group),
            "stable_marginal_24_48": sum(r["single_24_and_48_agree"] for r in group),
            "stable_full_four_rules": sum(r["full_numerical_top_agreement"] for r in group),
            "stable_both_count": len(stable),
            "stable_both_disagreement": sum(not agrees(r) for r in stable),
            "numeric_mean_disagreement": sum(not agrees(r) for r in group),
            "numeric_mean_disagreement_actual_first": sum(
                r["actual_first_slot_top"] != r["full_top_mean32768"] for r in group
            ),
            "strict_exclusion_actual_first": sum(not r["actual_single_retained"] for r in group),
            "strict_exclusion_matching_first_both_orders": sum(
                r["single_24_and_48_agree"] and not r["single_retained"] for r in group
            ),
            "strict_unique_full_count": len(only),
            "strict_unique_full_equals_actual_first": sum(r["actual_single_retained"] for r in only),
            "strict_unique_full_equals_matched_first_both_orders": sum(
                r["single_24_and_48_agree"] and r["single_retained"] for r in only
            ),
            "actual_queries_evaluated": sum(r["evaluated_queries"] for r in group),
            "median_regret": median(regret),
            "max_regret": max(regret),
            "median_backup_value": mid,
            "rank_correlation_of_backup_and_regret": corr if math.isfinite(corr) else None,
            "numeric_disagreement_by_backup_value": {
                side: {"n": len(selected), "disagreement": sum(not agrees(r) for r in selected)}
                for side, selected in (("at_or_below_median", [r for r in group if r["baseline_backup_value_refined"] <= mid]),
                                       ("above_median", [r for r in group if r["baseline_backup_value_refined"] > mid]))
            },
            "max_full_fine_scramble_difference": max(r["largest_abs_fine_scramble_difference"] for r in group),
            "worst_fine_interval_overshoot": max(r["worst_fine_interval_overshoot"] for r in group),
            "median_screen_and_first_value_cpu_seconds": median(
                r["screen_and_first_value_cpu_seconds"] for r in group
            ),
            "median_cpu_seconds": median(r["cpu_seconds"] for r in group),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.receipts.read_text(encoding="utf-8"))
    readout = summarize(data["scoring"]["receipts"])
    if "refinement" in data:
        for family, row in summarize_refined(data["refinement"]["receipts"]).items():
            readout[family]["refinement"] = row
    if args.output:
        args.output.write_text(json.dumps(readout, indent=2) + "\n", encoding="utf-8")
    else:
        print(json.dumps(readout, indent=2))


if __name__ == "__main__":
    main()
