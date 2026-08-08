#!/usr/bin/env python3
"""Market- and region-level analysis for the OPSD V2 benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from performance.statistical_inference import (
    apply_holm_family,
    bootstrap_mean_ci,
)


CONTRACT_ID = "opsd_region_heldout_profile_design_v2"
ACCEPTED_CONTRACT_IDS = {
    CONTRACT_ID,
    "opsd_region_heldout_functional_scbo_v1",
}
CONTROLS = (
    "generic_dct_maximin",
    "random_low_frequency",
    "natural_constant_grid",
    "raw_sobol",
    "target_only_dct_space_scbo",
)


def _receipt(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _successful(row):
    return bool(row["independently_certified"] and not row["false_certificate"])


def _paired_seed_outcome(first, second):
    first_success = _successful(first)
    second_success = _successful(second)
    if first_success != second_success:
        return 1 if first_success else -1
    if not first_success:
        return 0
    first_objective = float(first["objective_if_certified"])
    second_objective = float(second["objective_if_certified"])
    if first_objective < second_objective - 1e-12:
        return 1
    if first_objective > second_objective + 1e-12:
        return -1
    return 0


def analyze(
    paths,
    *,
    accepted_contract_ids=ACCEPTED_CONTRACT_IDS,
    controls=CONTROLS,
    analysis_contract_id="opsd_region_heldout_profile_design_analysis_v2",
):
    accepted_contract_ids = set(accepted_contract_ids)
    controls = tuple(str(value) for value in controls)
    rows = []
    failures = []
    keys = set()
    for path in map(Path, paths):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(f"{path}: unreadable: {exc}")
            continue
        if (
            row.get("contract_id") not in accepted_contract_ids
            or row.get("status") != "ok"
        ):
            failures.append(f"{path}: wrong contract or status")
            continue
        key = (row["target_market"], int(row["target_seed"]), row["arm"])
        if key in keys:
            failures.append(f"{path}: duplicate cell {key}")
            continue
        keys.add(key)
        row["_path"] = str(path)
        row["_sha256"] = _receipt(path)
        rows.append(row)

    market_groups = {}
    for row in rows:
        key = (row["target_region"], row["target_market"], row["arm"])
        market_groups.setdefault(key, []).append(row)
    market_summaries = []
    for (region, market, arm), group in sorted(market_groups.items()):
        objectives = [
            float(row["objective_if_certified"])
            for row in group if _successful(row)
        ]
        wall_times = [
            float(row["wall_time_sec"])
            for row in group if row.get("wall_time_sec") is not None
        ]
        market_summaries.append({
            "target_region": region,
            "target_market": market,
            "arm": arm,
            "algorithmic_seed_count": len(group),
            "certified_safe_count": int(sum(_successful(row) for row in group)),
            "false_certificate_count": int(sum(
                bool(row["false_certificate"]) for row in group)),
            "median_objective_if_certified": (
                None if not objectives else float(np.median(objectives))),
            "mean_verification_calls": float(np.mean([
                row["verification_calls"] for row in group])),
            "mean_all_in_calls_unamortized": float(np.mean([
                row["all_in_calls_unamortized"] for row in group])),
            "mean_all_in_budget_cap_unamortized": float(np.mean([
                row["all_in_budget_cap_unamortized"] for row in group])),
            "mean_all_in_calls_amortized": float(np.mean([
                row["all_in_calls_amortized"] for row in group])),
            "mean_all_in_budget_cap_amortized": float(np.mean([
                row["all_in_budget_cap_amortized"] for row in group])),
            "median_wall_time_sec": (
                None if not wall_times else float(np.median(wall_times))),
            "mean_wall_time_sec": (
                None if not wall_times else float(np.mean(wall_times))),
        })

    paired = []
    for market in sorted({row["target_market"] for row in rows}):
        region = next(
            row["target_region"] for row in rows
            if row["target_market"] == market)
        source = {
            int(row["target_seed"]): row for row in rows
            if row["target_market"] == market and row["arm"] == "source_atlas"
        }
        for control in controls:
            comparator = {
                int(row["target_seed"]): row for row in rows
                if row["target_market"] == market and row["arm"] == control
            }
            common = sorted(set(source) & set(comparator))
            if not common:
                continue
            outcomes = [
                _paired_seed_outcome(source[seed], comparator[seed])
                for seed in common
            ]
            break_even = []
            for seed in common:
                source_row = source[seed]
                control_row = comparator[seed]
                source_operating = (
                    int(source_row["target_search_calls"])
                    + int(source_row["verification_calls"])
                )
                denominator = (
                    int(control_row["all_in_calls_unamortized"])
                    - source_operating
                )
                if denominator > 0:
                    break_even.append(float(
                        source_row["source_calls"] / denominator))
            wins = int(sum(value > 0 for value in outcomes))
            losses = int(sum(value < 0 for value in outcomes))
            paired.append({
                "target_region": region,
                "target_market": market,
                "first": "source_atlas",
                "second": control,
                "paired_algorithmic_seed_count": len(common),
                "first_wins": wins,
                "first_losses": losses,
                "ties": int(sum(value == 0 for value in outcomes)),
                "algorithmic_repeatability_sign_pvalue": (
                    1.0 if wins + losses == 0 else float(binomtest(
                        wins, wins + losses, p=0.5,
                        alternative="greater").pvalue)
                ),
                "inference_family_id": (
                    "energy_market_algorithmic_repeatability"),
                "task_population_inference_claimed": False,
                "median_archive_break_even_target_count": (
                    None if not break_even else float(np.median(break_even))),
            })
    apply_holm_family(
        paired,
        pvalue_field="algorithmic_repeatability_sign_pvalue",
        family_field="inference_family_id",
    )

    region_summaries = []
    for region in sorted({row["target_region"] for row in market_summaries}):
        for arm in sorted({row["arm"] for row in market_summaries}):
            group = [
                row for row in market_summaries
                if row["target_region"] == region and row["arm"] == arm
            ]
            if not group:
                continue
            region_summaries.append({
                "target_region": region,
                "arm": arm,
                "independent_market_count": len(group),
                "mean_market_certified_safe_rate": float(np.mean([
                    row["certified_safe_count"] / row["algorithmic_seed_count"]
                    for row in group
                ])),
                "total_false_certificates": int(sum(
                    row["false_certificate_count"] for row in group)),
                "mean_market_all_in_calls_amortized": float(np.mean([
                    row["mean_all_in_calls_amortized"] for row in group
                ])),
            })

    region_direction = []
    for control in controls:
        wins = losses = ties = 0
        compared_regions = 0
        rate_differences = []
        for region in sorted({row["target_region"] for row in market_summaries}):
            source = [
                row for row in market_summaries
                if row["target_region"] == region and row["arm"] == "source_atlas"
            ]
            comparator = [
                row for row in market_summaries
                if row["target_region"] == region and row["arm"] == control
            ]
            if not source or not comparator:
                continue
            compared_regions += 1
            source_rate = float(np.mean([
                row["certified_safe_count"] / row["algorithmic_seed_count"]
                for row in source
            ]))
            control_rate = float(np.mean([
                row["certified_safe_count"] / row["algorithmic_seed_count"]
                for row in comparator
            ]))
            rate_differences.append(source_rate - control_rate)
            if source_rate > control_rate + 1e-12:
                wins += 1
            elif source_rate < control_rate - 1e-12:
                losses += 1
            else:
                ties += 1
        if compared_regions == 0:
            continue
        region_direction.append({
            "first": "source_atlas",
            "second": control,
            "region_count": compared_regions,
            "first_wins": wins,
            "first_losses": losses,
            "ties": ties,
            "one_sided_region_sign_pvalue": (
                1.0 if wins + losses == 0 else float(binomtest(
                    wins, wins + losses, p=0.5,
                    alternative="greater").pvalue)
            ),
            "mean_source_minus_control_region_safe_rate": (
                float(np.mean(rate_differences))
            ),
            "mean_source_minus_control_region_safe_rate_bootstrap_95ci": (
                bootstrap_mean_ci(
                    rate_differences,
                    seed=20260808 + len(region_direction),
                )
            ),
            "inference_family_id": "energy_region_primary_controls",
            "warning": (
                "Only five geographic regions are available and markets share "
                "calendar data; this is a conservative descriptive audit, not "
                "a broad task-population claim."
            ),
        })
    apply_holm_family(
        region_direction,
        pvalue_field="one_sided_region_sign_pvalue",
        family_field="inference_family_id",
    )

    compact_rows = [{
        "target_region": row["target_region"],
        "target_market": row["target_market"],
        "target_seed": int(row["target_seed"]),
        "arm": row["arm"],
        "independently_certified": bool(row["independently_certified"]),
        "false_certificate": bool(row["false_certificate"]),
        "objective_if_certified": row["objective_if_certified"],
        "source_calls": int(row["source_calls"]),
        "target_search_calls": int(row["target_search_calls"]),
        "verification_calls": int(row["verification_calls"]),
        "all_in_calls_unamortized": int(row["all_in_calls_unamortized"]),
        "all_in_budget_cap_unamortized": int(
            row["all_in_budget_cap_unamortized"]),
        "all_in_calls_amortized": float(row["all_in_calls_amortized"]),
        "all_in_budget_cap_amortized": float(
            row["all_in_budget_cap_amortized"]),
        "wall_time_sec": (
            None if row.get("wall_time_sec") is None
            else float(row["wall_time_sec"])),
        "raw_result": Path(row["_path"]).name,
        "raw_sha256": row["_sha256"],
    } for row in rows]
    return {
        "schema_version": 1,
        "contract_id": str(analysis_contract_id),
        "status": "complete" if rows and not failures else "incomplete",
        "primary_generalization_unit": "geographic_region",
        "secondary_task_unit": "market",
        "algorithmic_repeatability_unit": "seed_within_market",
        "market_count": len({row["target_market"] for row in rows}),
        "region_count": len({row["target_region"] for row in rows}),
        "row_count": len(rows),
        "market_summaries": market_summaries,
        "region_summaries": region_summaries,
        "paired_algorithmic_repeatability": paired,
        "region_level_directional_audit": region_direction,
        "compact_rows": compact_rows,
        "failures": failures,
    }


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["status"]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows or [{"status": "empty"}])
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--csv")
    args = parser.parse_args()
    payload = analyze(args.paths)
    _atomic_json(args.out, payload)
    if args.csv:
        _write_csv(args.csv, payload["compact_rows"])
    print(json.dumps({
        "status": payload["status"],
        "row_count": payload["row_count"],
        "market_count": payload["market_count"],
        "region_count": payload["region_count"],
        "failure_count": len(payload["failures"]),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
