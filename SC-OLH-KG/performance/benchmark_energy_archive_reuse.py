#!/usr/bin/env python3
"""Frozen retrospective test of same-market historical archive reuse."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from core.designs import next_sobol_integer_candidate
from core.profile_atlas import SourceProfileRecord
from core.terminal_verification import TERMINAL_BINOMIAL_VERIFICATION_STREAM_TAG, exact_binomial_lower
from performance.benchmark_external_energy_v3 import _profile_point, _structural_initial_profiles, market_region
from performance.benchmark_profile_stress_suite import _select_shortlist
from performance.submission_revision_attribution import selections
from problems.energy_archive_reuse import TemporalArchiveReuseStorageProblem, month_period, sampled_window_starts
from problems.randomized_profiles import generate_structural_profile_library

PROTOCOL = ROOT / "performance/manifests/energy_archive_reuse_v2_20260930.json"
OUTPUT = ROOT / "paper_artifacts/energy_archive_reuse_v2_20260930"
DATA = ROOT / "data/external/opsd_time_series_extended_v2.npz"


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def freeze_archives(protocol, out):
    library = generate_structural_profile_library(64, dimension=128, seed=20261799, maximum_frequency=40)
    for market_index, market in enumerate(protocol["markets"]):
        path = out / "archives" / (market + ".json")
        if path.exists():
            if json.loads(path.read_text()).get("contract_id") != protocol["contract_id"]:
                raise ValueError("archive belongs to an earlier physical model; use a new --out directory")
            continue
        rows = []
        for period_index, period in enumerate(protocol["archive"]["source_periods"]):
            source = TemporalArchiveReuseStorageProblem(DATA, market=market,
                       search_period=period, verification_period=period)
            for profile_index, profile in enumerate(library):
                seeds = [[protocol["archive"]["seed"], market_index, period_index, profile_index, r, 3109]
                         for r in range(3)]
                starts = sampled_window_starts(source, "search", seeds)
                samples = source.evaluate_window_starts(_profile_point(source, profile), starts)
                rows.append(SourceProfileRecord(
                    task_id=f"{market}:2017:H{period_index + 1}", profile_id=profile.profile_id,
                    profile=profile.values, nodes=profile.nodes,
                    objective_samples=tuple(samples[:, 0]), constraint_samples=tuple(samples[:, 1]),
                    alpha=.05, tau=0., descriptor=()))
        chosen = selections(rows)
        design_ids = {arm: [m.profile_id for m in chosen[arm].members]
                      for arm in protocol["target"]["arms"] if arm != "generic_dct_maximin"}
        generic, _ = _structural_initial_profiles("generic_dct_maximin", library, None, 10, 0)
        design_ids["generic_dct_maximin"] = [p.profile_id for p in generic]
        write(path, dict(contract_id=protocol["contract_id"], market=market, source_calls=384,
                        source_periods=protocol["archive"]["source_periods"],
                        designs=design_ids, records=[asdict(r) for r in rows],
                        target_outcomes_used=False, normalization_fit_period=["2017-01-01T00", "2018-01-01T00"]))
        print(f"ARCHIVE_FROZEN {market_index + 1}/{len(protocol['markets'])} {market}", flush=True)


def verify_batch(problem, shortlist, design_seed):
    """Execute the original fixed-budget Bernoulli law in physical batches.

    Each of the 80 seeds still executes one physical window. No early stopping
    inside a candidate, no sample reuse, and no change to shortlist ordering.
    """
    attempts = []
    deployed = None
    for candidate_index, row in enumerate(shortlist):
        seeds = [[design_seed + 71003, candidate_index, replication,
                  TERMINAL_BINOMIAL_VERIFICATION_STREAM_TAG] for replication in range(80)]
        starts = sampled_window_starts(problem, "verification", seeds)
        values = problem.evaluate_window_starts(row["point"], starts)
        successes = int(np.sum(values[:, 1] <= problem.tau))
        certified = bool(successes == 80 and .95 ** 80 <= .05 / 3)
        attempts.append(dict(candidate_index=candidate_index, successes=successes,
                             trials=80, certified=certified,
                             exact_lower_confidence_bound=exact_binomial_lower(successes, 80, .05 / 3),
                             objective_sample_mean=float(np.mean(values[:, 0]))))
        if certified:
            deployed = tuple(row["point"])
            break
    return deployed, dict(certified=deployed is not None, verification_budget=80 * len(attempts),
                          attempts=attempts, selected_shortlist_rank=None if deployed is None else len(attempts),
                          familywise_delta=.05, candidate_budgets=[80, 80, 80],
                          shortlist_frozen_before_verification=True)


def evaluate_task(problem, points, design_seed):
    records = []
    observed = list(points)
    for index, point in enumerate(points):
        rng = np.random.default_rng(np.random.SeedSequence([design_seed, index, 3211]))
        records.append(dict(point=point, observation=problem.simulate(point, rng), evaluation_index=index))
    while len(records) < 13:
        point = next_sobol_integer_candidate(problem, design_seed, observed=observed, seed_offset=330107)
        index = len(records)
        rng = np.random.default_rng(np.random.SeedSequence([design_seed, index, 3211]))
        records.append(dict(point=point, observation=problem.simulate(point, rng), evaluation_index=index))
        observed.append(point)
    shortlist = _select_shortlist(records, problem.tau, size=3)
    deployed, verification = verify_batch(problem, shortlist, design_seed)
    return deployed, verification


def evaluate_market(protocol, market, out):
    started = time.perf_counter()
    market_index = protocol["markets"].index(market)
    archive = json.loads((out / "archives" / (market + ".json")).read_text())
    library = {p.profile_id: p for p in generate_structural_profile_library(64, dimension=128, seed=20261799, maximum_frequency=40)}
    rows = []
    diagnostic_windows = 0
    for month in protocol["target"]["months"]:
        path = out / "months" / f"{market}_{month:02d}.json"
        if path.exists():
            payload = json.loads(path.read_text())
            rows.extend(payload["rows"])
            diagnostic_windows += payload["postdecision_diagnostic_window_evaluations"]
            continue
        problem = TemporalArchiveReuseStorageProblem(DATA, market=market,
                   search_period=month_period(2018, month), verification_period=month_period(2019, month))
        points = {arm: tuple(_profile_point(problem, library[pid]) for pid in ids)
                  for arm, ids in archive["designs"].items()}
        pending = []
        for seed in protocol["target"]["algorithm_seeds"]:
            design_seed = 9030000 + 10000 * market_index + 100 * month + seed
            for arm in protocol["target"]["arms"]:
                deployed, verification = evaluate_task(problem, points[arm], design_seed)
                pending.append((dict(target_market=market, target_region=market_region(market), month=month,
                    algorithm_seed=seed, design_seed=design_seed, arm=arm, status="ok",
                    target_search_calls=13, verification_calls=verification["verification_budget"],
                    independently_certified=verification["certified"], verification=verification,
                    selected_profile_ids=archive["designs"][arm]), deployed))
        # Enumerate truth only after every decision in this market/month has
        # been frozen and tested. Repeated policies share this diagnostic work.
        truth = {}
        all_points = set(p for arm_points in points.values() for p in arm_points)
        all_points.update(deployed for _, deployed in pending if deployed is not None)
        starts = problem.split_window_starts("verification")
        for point in all_points:
            values = problem.evaluate_window_starts(point, starts)
            truth[point] = dict(probability=float(np.mean(values[:, 1] <= problem.tau)), objective=float(np.mean(values[:, 0])))
        month_rows = []
        for row, deployed in pending:
            safe = bool(deployed is not None and truth[deployed]["probability"] >= .95)
            row.update(certified_true_feasible=safe, false_certificate=bool(deployed is not None and not safe),
                       objective_if_certified=None if deployed is None else truth[deployed]["objective"],
                       deployment_probability=None if deployed is None else truth[deployed]["probability"],
                       initial_feasible_coverage=any(truth[p]["probability"] >= .95 for p in points[row["arm"]]),
                       first_center_true_feasible=truth[points[row["arm"]][0]]["probability"] >= .95,
                       target_truth_used_for_selection=False)
            month_rows.append(row)
        count = len(all_points) * len(starts)
        diagnostic_windows += count
        payload = dict(rows=month_rows, information_contract=problem.information_contract(),
                       postdecision_diagnostic_window_evaluations=count)
        write(path, payload)
        rows.extend(month_rows)
        print(f"MONTH_COMPLETED {market} {month:02d} rows={len(month_rows)}", flush=True)
    payload = dict(rows=rows, postdecision_diagnostic_window_evaluations=diagnostic_windows,
                   wall_time_sec=time.perf_counter() - started)
    write(out / "markets" / (market + ".json"), payload)
    return payload


def analyze(rows, protocol):
    expected = {(market, month, seed, arm) for market in protocol["markets"]
                for month in protocol["target"]["months"] for seed in protocol["target"]["algorithm_seeds"]
                for arm in protocol["target"]["arms"]}
    keys = [(r["target_market"], r["month"], r["algorithm_seed"], r["arm"]) for r in rows]
    if set(keys) != expected or len(keys) != len(expected):
        raise ValueError("complete paired market/month/seed/arm matrix required")
    summaries = []
    market_summaries = []
    for arm in protocol["target"]["arms"]:
        arm_rows = [r for r in rows if r["arm"] == arm]
        source_once = 0 if arm in {"standardized_generic_dct_maximin", "generic_dct_maximin"} else 384
        for market in protocol["markets"]:
            selected = [r for r in arm_rows if r["target_market"] == market]
            certificates = sum(r["certified_true_feasible"] for r in selected) / 5
            cumulative = source_once + sum(r["target_search_calls"] + r["verification_calls"] for r in selected) / 5
            market_summaries.append(dict(arm=arm, market=market, region=market_region(market),
                actual_monthly_tasks=12, source_calls_once=source_once, expected_true_certificates=certificates,
                cumulative_calls=cumulative, calls_per_true_certificate=cumulative / certificates if certificates else None,
                certification_rate=certificates / 12))
        costs = [r for r in market_summaries if r["arm"] == arm]
        certificates = sum(r["expected_true_certificates"] for r in costs)
        cumulative = sum(r["cumulative_calls"] for r in costs)
        objectives = [r["objective_if_certified"] for r in arm_rows if r["certified_true_feasible"]]
        summaries.append(dict(arm=arm, row_count=len(arm_rows),
            actual_deployment_tasks=216, certified_true_feasible_count=sum(r["certified_true_feasible"] for r in arm_rows),
            certification_rate=certificates / 216,
            initial_feasible_coverage_count=sum(r["initial_feasible_coverage"] for r in arm_rows) / 5,
            first_center_true_feasible_count=sum(r["first_center_true_feasible"] for r in arm_rows) / 5,
            false_certificate_count=sum(r["false_certificate"] for r in arm_rows),
            source_calls_once=sum(r["source_calls_once"] for r in costs), cumulative_calls=cumulative,
            expected_true_certificates=certificates, calls_per_true_certificate=cumulative / certificates if certificates else None,
            certified_objective_mean=float(np.mean(objectives)) if objectives else None))
    contrasts = []
    for comparator in protocol["target"]["arms"]:
        if comparator == "source_best_z":
            continue
        region_values = []
        regions = sorted({r["region"] for r in market_summaries})
        for region in regions:
            rates = {arm: np.mean([r["certification_rate"] for r in market_summaries if r["region"] == region and r["arm"] == arm])
                     for arm in ("source_best_z", comparator)}
            region_values.append(float(rates["source_best_z"] - rates[comparator]))
        boot = np.random.default_rng(202609302).choice(region_values, size=(10000, len(regions))).mean(axis=1)
        contrasts.append(dict(comparator=comparator, regions=regions, region_differences=region_values,
            mean_region_difference=float(np.mean(region_values)), descriptive_region_bootstrap_95ci=np.quantile(boot, [.025, .975]).tolist()))
    return dict(status="complete", contract_id=protocol["contract_id"], row_count=len(rows),
                actual_deployment_tasks=216, evidence_role=protocol["registration"],
                summaries=summaries, market_summaries=market_summaries, paired_contrasts=contrasts,
                diagnostic_truth_used_for_selection=False,
                certified_objectives_are_conditional_on_each_methods_success=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("freeze", "targets", "analyze"), required=True)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    if args.phase == "freeze":
        freeze_archives(protocol, args.out)
        return
    if args.phase == "targets":
        # All eighteen designs must exist before target access starts.
        for market in protocol["markets"]:
            if not (args.out / "archives" / (market + ".json")).is_file():
                raise ValueError("run --phase freeze before target simulations")
            if json.loads((args.out / "archives" / (market + ".json")).read_text()).get("contract_id") != protocol["contract_id"]:
                raise ValueError("target simulation requires an archive from the same physical model")
        payloads = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = [pool.submit(evaluate_market, protocol, market, args.out) for market in protocol["markets"]]
            for job in as_completed(jobs):
                payloads.append(job.result())
        rows = [r for p in payloads for r in p["rows"]]
        rows.sort(key=lambda r: (r["target_market"], r["month"], r["algorithm_seed"], r["arm"]))
        write(args.out / "rows.json", dict(protocol=PROTOCOL.relative_to(ROOT).as_posix(), rows=rows,
            postdecision_diagnostic_window_evaluations=sum(p["postdecision_diagnostic_window_evaluations"] for p in payloads)))
    else:
        rows = json.loads((args.out / "rows.json").read_text())["rows"]
    analysis = analyze(rows, protocol)
    write(args.out / "analysis.json", analysis)
    print(json.dumps(analysis["summaries"]), flush=True)


if __name__ == "__main__":
    main()
