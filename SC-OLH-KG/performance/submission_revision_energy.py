#!/usr/bin/env python3
"""Evaluate supplemental design controls on the unchanged Energy V3 protocol."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance import benchmark_external_energy_v3 as energy
from performance.submission_revision_controls import (
    standardized_generic_dct_maximin, source_greedy_portfolio,
)

NEW_ARMS = ("standardized_generic_dct_maximin", "source_greedy_portfolio")
FIELDS = (
    "target_market", "target_region", "target_seed", "design_seed", "arm",
    "year", "nominal_dimension", "n0", "N", "source_calls",
    "target_search_calls", "verification_calls", "all_in_calls_unamortized",
    "all_in_calls_amortized", "independently_certified", "false_certificate",
    "objective_if_certified", "selected_profile_ids", "certificate_scope",
)


def evaluate_market(market, data_path, *, target_seeds=range(5)):
    # Both selections are fixed before any target simulations in this worker.
    library = energy.generate_structural_profile_library(
        64, dimension=128, seed=20261799, maximum_frequency=40)
    indices, standard_diagnostics = standardized_generic_dct_maximin(
        [p.values for p in library], 10, nodes=library[0].nodes,
        max_frequency=8, frequency_penalty=0.25,
        include_diagonal_quadratic=True)
    standard_profiles = tuple(library[i] for i in indices)
    records, source_markets = energy.build_source_archive(
        data_path, target_market=market, library=library, seed=20260808 + 1237)
    greedy = source_greedy_portfolio(records, 10)
    lookup = {p.profile_id: p for p in library}
    greedy_profiles = tuple(lookup[m.profile_id] for m in greedy.members)
    designs = {
        NEW_ARMS[0]: (standard_profiles, standard_diagnostics),
        NEW_ARMS[1]: (greedy_profiles, greedy.diagnostics),
    }
    rows = []
    original_selector = energy._structural_initial_profiles
    try:
        for arm in NEW_ARMS:
            profiles, diagnostics = designs[arm]
            energy._structural_initial_profiles = (
                lambda _arm, _library, _atlas, _n0, _seed,
                profiles=profiles, diagnostics=diagnostics: (profiles, diagnostics))
            for seed in target_seeds:
                result = energy.run_task(
                    data_path=data_path, target_market=market, target_seed=seed,
                    arm="generic_dct_maximin")
                row = {key: result[key] for key in FIELDS}
                row["arm"] = arm
                source_calls = 0 if arm == NEW_ARMS[0] else len(source_markets) * 64 * 3
                row["source_calls"] = source_calls
                row["all_in_calls_unamortized"] += source_calls
                row["all_in_calls_amortized"] += source_calls / 20
                row["source_markets"] = [] if source_calls == 0 else list(source_markets)
                row["provenance"] = "post_review_supplemental"
                rows.append(row)
    finally:
        energy._structural_initial_profiles = original_selector
    return rows


def success(row):
    return bool(row["independently_certified"] and not row["false_certificate"])


def summarize(rows):
    summaries = []
    for arm in sorted({r["arm"] for r in rows}):
        for market in energy.TARGET_MARKETS:
            selected = [r for r in rows if r["arm"] == arm and r["target_market"] == market]
            if not selected:
                continue
            objectives = [r["objective_if_certified"] for r in selected
                          if success(r) and r["objective_if_certified"] is not None]
            summaries.append({
                "arm": arm, "target_market": market,
                "target_region": selected[0]["target_region"],
                "algorithmic_seed_count": len(selected),
                "certified_safe_count": sum(success(r) for r in selected),
                "false_certificate_count": sum(r["false_certificate"] for r in selected),
                "median_objective_if_certified": float(np.median(objectives)) if objectives else None,
                "mean_verification_calls": (float(np.mean([r["verification_calls"] for r in selected]))
                    if all(r["verification_calls"] is not None for r in selected) else None),
                "mean_all_in_calls_unamortized": (float(np.mean([r["all_in_calls_unamortized"] for r in selected]))
                    if all(r["all_in_calls_unamortized"] is not None for r in selected) else None),
            })
    source = {(r["target_market"], r["target_seed"]): r for r in rows if r["arm"] == "source_atlas"}
    pairs = []
    for arm in NEW_ARMS:
        control = {(r["target_market"], r["target_seed"]): r for r in rows if r["arm"] == arm}
        if set(source) != set(control):
            raise ValueError("supplemental Energy comparisons require the same market-seed cells")
        for key in source:
            if source[key]["design_seed"] != control[key]["design_seed"]:
                raise ValueError("Energy search seed mismatch")
        regions = sorted({r["target_region"] for r in source.values()})
        differences = [float(np.mean([
            int(success(source[k])) - int(success(control[k])) for k in source
            if source[k]["target_region"] == region])) for region in regions]
        rng = np.random.default_rng(20260907)
        boot = np.mean(rng.choice(differences, size=(10000, len(regions))), axis=1)
        pairs.append({
            "comparator": arm, "regions": regions,
            "region_differences": differences,
            "mean_region_difference": float(np.mean(differences)),
            "descriptive_region_bootstrap_95CI": np.quantile(boot, [.025, .975]).tolist(),
            "region_wins": sum(v > 0 for v in differences),
            "region_losses": sum(v < 0 for v in differences),
            "region_ties": sum(v == 0 for v in differences),
        })
    return {
        "status": "complete", "contract_id": "submission_revision_energy_controls_20260907",
        "study_role": "post-review supplement on previously used markets and seeds",
        "market_summaries": summaries, "paired_comparisons": pairs,
        "original_source_and_generic_counts": {
            arm: sum(success(r) for r in rows if r["arm"] == arm)
            for arm in ("source_atlas", "generic_dct_maximin")},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data/external/opsd_time_series_extended_v2.npz")
    parser.add_argument("--results", type=Path, default=ROOT / "results/or_review_energy_forecast_indexed_v3")
    parser.add_argument("--out", type=Path, default=ROOT / "paper_artifacts/submission_revision_20260907")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if args.analyze_only:
        rows = json.loads((args.out / "energy_rows.json").read_text())["rows"]
        (args.out / "energy_analysis.json").write_text(json.dumps(summarize(rows), indent=2) + "\n")
        return
    if energy.CONTRACT_ID != "opsd_forecast_indexed_region_holdout_v3":
        raise ValueError("Historical V3 physical execution requires the September 7 source attachment; use submission_revision_energy_v5.py for current V5 runs. --analyze-only replays the stored historical outcomes.")
    if not args.data.is_file():
        raise FileNotFoundError("The local extended OPSD archive is required; this runner never downloads data")
    old = []
    for path in sorted(args.results.glob("**/cell*.json")):
        row = json.loads(path.read_text())
        if row.get("arm") in {"source_atlas", "generic_dct_maximin"}:
            old.append({**{k: row[k] for k in FIELDS}, "provenance": "original_frozen_result"})
    for arm, expected in (("source_atlas", 60), ("generic_dct_maximin", 70)):
        selected = [r for r in old if r["arm"] == arm]
        if len(selected) != 90 or sum(success(r) for r in selected) != expected:
            raise ValueError("local Energy records do not reproduce the published original counts")
    new = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(evaluate_market, market, str(args.data)): market
                for market in energy.TARGET_MARKETS}
        for job in as_completed(jobs):
            new.extend(job.result())
            print(f"Energy supplemental cells {len(new)}/180", flush=True)
    rows = sorted(old + new, key=lambda r: (r["arm"], r["target_market"], r["target_seed"]))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "energy_rows.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    analysis = summarize(rows)
    (args.out / "energy_analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    print(json.dumps({arm: sum(success(r) for r in rows if r["arm"] == arm)
                      for arm in sorted({r["arm"] for r in rows})}), flush=True)


if __name__ == "__main__":
    main()
