#!/usr/bin/env python3
"""Corrected finite-horizon Energy study with one archive per held-out region."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from core.profile_atlas import SourceProfileRecord
from performance import benchmark_external_energy_v3 as energy
from performance.audit_external_energy_temporal_blocks import audit_frozen_policy
from performance.submission_revision_controls import standardized_generic_dct_maximin, source_greedy_portfolio
from performance.submission_revision_energy import FIELDS, success, summarize

PROTOCOL = ROOT / "performance/manifests/submission_revision_energy_v5_20260930.json"
OUTPUT = ROOT / "paper_artifacts/submission_revision_energy_v5_20260930"
DATA = ROOT / "data/external/opsd_time_series_extended_v2.npz"


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def freeze_archives(data, out):
    library = energy.generate_structural_profile_library(64, dimension=128, seed=20261799, maximum_frequency=40)
    regions = {}
    for market in energy.TARGET_MARKETS:
        region = energy.market_region(market)
        if region not in regions:
            start = time.perf_counter()
            records, source_markets = energy.build_source_archive(data, target_market=market, library=library, seed=20260808 + 1237)
            regions[region] = (records, source_markets)
            write(out / "energy_archives" / (region + ".json"), {
                "excluded_region": region, "source_markets": list(source_markets),
                "source_calls": len(source_markets) * 64 * 3,
                "build_wall_time_sec": time.perf_counter() - start,
                "initial_soc_fraction": 0.5, "records": [asdict(r) for r in records],
            })
        records, source_markets = regions[region]
        with patch.object(energy, "build_source_archive", lambda *a, **k: (records, source_markets)):
            design = energy.materialize_source_atlas(data_path=data, target_market=market)
        write(out / "energy_designs" / (market + ".json"), design)
    print("ENERGY_ARCHIVES_FROZEN regions=5 markets=18", flush=True)


def evaluate_market(market, data, out, arms):
    start = time.perf_counter()
    region = energy.market_region(market)
    archive = json.loads((out / "energy_archives" / (region + ".json")).read_text())
    records = tuple(SourceProfileRecord(**r) for r in archive["records"])
    library = energy.generate_structural_profile_library(64, dimension=128, seed=20261799, maximum_frequency=40)
    indices, diagnostics = standardized_generic_dct_maximin([p.values for p in library], 10, nodes=library[0].nodes, max_frequency=8, frequency_penalty=0.25, include_diagonal_quadratic=True)
    portfolio = source_greedy_portfolio(records, 10)
    lookup = {p.profile_id: p for p in library}
    designs = {
        "standardized_generic_dct_maximin": (tuple(library[i] for i in indices), diagnostics),
        "source_greedy_portfolio": (tuple(lookup[m.profile_id] for m in portfolio.members), portfolio.diagnostics),
    }
    target = energy.OPSDForecastIndexedStorageProblem(data, market=market)
    rows = []
    for arm in arms:
        for seed in range(5):
            try:
                kwargs = dict(data_path=data, target_market=market, target_seed=seed, arm=arm)
                if arm == "source_atlas":
                    kwargs["design_path"] = out / "energy_designs" / (market + ".json")
                if arm in designs:
                    profiles, diag = designs[arm]
                    kwargs["arm"] = "generic_dct_maximin"
                    with patch.object(energy, "_structural_initial_profiles", lambda *a: (profiles, diag)):
                        result = energy.run_task(**kwargs)
                else:
                    result = energy.run_task(**kwargs)
                row = {k: result[k] for k in FIELDS}
                row.update(arm=arm, status="ok", wall_time_sec=result["wall_time_sec"], information_contract=result["information_contract"])
                if arm == "source_greedy_portfolio":
                    row["source_calls"] = archive["source_calls"]
                    row["all_in_calls_unamortized"] += archive["source_calls"]
                    row["all_in_calls_amortized"] += archive["source_calls"] / 20
                row["source_markets"] = archive["source_markets"] if row["source_calls"] else []
                if row["independently_certified"]:
                    rank = int(result["verification"]["selected_shortlist_rank"])
                    point = result["shortlist"][rank - 1]["point"]
                    row["temporal_audit"] = audit_frozen_policy(target, point)
                    starts = target.split_window_starts("verification")
                    sampled = starts[np.linspace(0, len(starts)-1, 512).round().astype(int)]
                    _, state = target._evaluate_start_batch(point, sampled, return_diagnostics=True)
                    row["mean_terminal_energy"] = float(np.mean(state["terminal_energy"]))
                row["verification"] = result["verification"]
                row["functional_coordinate_contract"] = result.get("functional_coordinate_contract")
            except Exception as error:
                # A backend failure is an unsuccessful task, with unknown executed cost.
                row = dict(arm=arm, status="error", target_market=market, target_region=region,
                           target_seed=seed, design_seed=4300000+seed, independently_certified=False,
                           false_certificate=False, objective_if_certified=None,
                           verification_calls=None, all_in_calls_unamortized=None,
                           error_type=type(error).__name__, error=str(error))
            rows.append(row)
    write(out / "energy_markets" / (market + ".json"), {"rows": rows, "wall_time_sec": time.perf_counter()-start})
    return rows


def analyze(rows, out):
    if any(r["status"] == "ok" and r.get("information_contract", {}).get("hourly_power_budget") != "shared across all reserve-adjustment and balancing flows" for r in rows):
        raise ValueError("V5 analysis requires shared-power outcomes; historical V4 replay uses its original source attachment")
    analysis = summarize(rows)
    analysis["contract_id"] = energy.CONTRACT_ID
    analysis["study_role"] = "corrected model on previously used data; not independent external confirmation"
    analysis["algorithmic_failure_count"] = sum(r["status"] == "error" for r in rows)
    analysis["cost_observations_complete"] = analysis["algorithmic_failure_count"] == 0
    analysis["arm_totals"] = {arm: {"task_count": sum(r["arm"] == arm for r in rows), "certified_safe": sum(success(r) for r in rows if r["arm"] == arm), "false_certificates": sum(r["false_certificate"] for r in rows if r["arm"] == arm)} for arm in sorted({r["arm"] for r in rows})}
    differences = []
    for region in sorted({r["target_region"] for r in rows}):
        selected = [r for r in rows if r["target_region"] == region]
        differences.append(float(np.mean([success(r) for r in selected if r["arm"] == "source_atlas"])) - float(np.mean([success(r) for r in selected if r["arm"] == "generic_dct_maximin"])))
    boot = np.random.default_rng(20260907).choice(differences, size=(10000, len(differences))).mean(axis=1)
    analysis["paired_comparisons"].append(dict(comparator="generic_dct_maximin", region_differences=differences, mean_region_difference=float(np.mean(differences)), descriptive_region_bootstrap_95CI=np.quantile(boot,[.025,.975]).tolist()))
    temporal = []
    reuse = []
    for arm in sorted({r["arm"] for r in rows}):
        certified = [r for r in rows if r["arm"] == arm and r["independently_certified"]]
        block = sum(r["temporal_audit"]["minimum_chronological_block_feasibility_probability"] >= .95 for r in certified)
        disjoint = sum(r["temporal_audit"]["nonoverlapping_summary"]["feasibility_probability"] >= .95 for r in certified)
        joint = sum(r["temporal_audit"]["minimum_chronological_block_feasibility_probability"] >= .95 and r["temporal_audit"]["nonoverlapping_summary"]["feasibility_probability"] >= .95 for r in certified)
        temporal.append(dict(arm=arm, certified_count=len(certified), block_stable_count=block, nonoverlap_stable_count=disjoint, joint_stable_count=joint))
        for region in sorted({r["target_region"] for r in rows}):
            selected = [r for r in rows if r["arm"] == arm and r["target_region"] == region]
            if any(r["status"] == "error" for r in selected):
                continue
            archive = json.loads((out / "energy_archives" / (region + ".json")).read_text())
            markets = sorted({r["target_market"] for r in selected})
            source_calls = archive["source_calls"] if arm in {"source_atlas", "source_greedy_portfolio"} else 0
            cumulative_calls = source_calls + sum(r["target_search_calls"] + r["verification_calls"] for r in selected) / 5
            certificates = sum(success(r) for r in selected) / 5
            reuse.append(dict(arm=arm, excluded_region=region, actual_deployment_markets=markets, deployment_count=len(markets), source_calls_once=source_calls, expected_certificates=certificates, cumulative_calls=cumulative_calls, calls_per_certificate=cumulative_calls/certificates if certificates else None))
    analysis["temporal_summaries"] = temporal
    analysis["actual_archive_reuse"] = reuse
    return analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    out = args.out
    if args.analyze_only:
        rows = json.loads((out / "energy_rows.json").read_text())["rows"]
    else:
        if not (out / "energy_designs" / (energy.TARGET_MARKETS[-1]+".json")).exists():
            freeze_archives(str(DATA), out)
        for market in energy.TARGET_MARKETS:
            if json.loads((out / "energy_designs" / (market + ".json")).read_text())["contract_id"] != energy.DESIGN_CONTRACT_ID:
                raise ValueError("design belongs to an earlier physical model; use a new --out directory")
        rows = []
        pending = []
        for market in energy.TARGET_MARKETS:
            path = out / "energy_markets" / (market + ".json")
            if path.exists():
                rows.extend(json.loads(path.read_text())["rows"])
            else:
                pending.append(market)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = [pool.submit(evaluate_market, market, str(DATA), out, protocol["energy"]["arms"]) for market in pending]
            for job in as_completed(jobs):
                rows.extend(job.result())
                print(f"ENERGY_V5_COMPLETED {len(rows)}/720", flush=True)
        rows.sort(key=lambda r: (r["arm"], r["target_market"], r["target_seed"]))
        write(out / "energy_rows.json", {"protocol": str(PROTOCOL.relative_to(ROOT)), "rows": rows})
    if len(rows) != 720:
        raise ValueError(f"expected 720 corrected Energy cells, got {len(rows)}")
    analysis = analyze(rows, out)
    write(out / "energy_analysis.json", analysis)
    print(json.dumps(analysis["arm_totals"]), flush=True)


if __name__ == "__main__":
    main()
