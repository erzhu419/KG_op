#!/usr/bin/env python3
"""Analyze the preregistered paired GB_GBN confirmatory experiment."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import statistics
import sys

from scipy.stats import beta, binomtest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.benchmark_external_energy_confirmation import (
    CONFIRMATORY_CONTRACT_ID,
    CONFIRMATORY_MARKET,
    CONFIRMATORY_SEEDS,
    CONFIRMATORY_YEAR,
)


ARMS = ("frozen_proposal", "common_sobol")


def _proportion_interval(successes, trials, delta=0.05):
    successes = int(successes)
    trials = int(trials)
    lower = 0.0 if successes == 0 else float(
        beta.ppf(delta / 2.0, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(
        beta.ppf(1.0 - delta / 2.0, successes + 1, trials - successes))
    return [lower, upper]


def _safe_objective(row):
    truth = row.get("deployment_truth_audit")
    if (
        not row.get("independently_certified")
        or row.get("false_certificate")
        or truth is None
        or not truth.get("truly_chance_feasible")
    ):
        return None
    return float(truth["true_objective_mean"])


def analyze(paths):
    rows = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    expected_seeds = list(CONFIRMATORY_SEEDS)
    index = {}
    for row in rows:
        if row.get("status") != "ok":
            raise ValueError("confirmation contains an incomplete row")
        if row.get("contract_id") != CONFIRMATORY_CONTRACT_ID:
            raise ValueError("confirmation contract mismatch")
        if row.get("market") != CONFIRMATORY_MARKET or int(
                row.get("year", -1)) != CONFIRMATORY_YEAR:
            raise ValueError("confirmation target mismatch")
        key = (str(row.get("arm")), int(row.get("seed", -1)))
        if key in index:
            raise ValueError(f"duplicate confirmatory row: {key}")
        index[key] = row

    summaries = {}
    for arm in ARMS:
        arm_rows = [index[(arm, seed)] for seed in expected_seeds
                    if (arm, seed) in index]
        objectives = [value for value in map(_safe_objective, arm_rows)
                      if value is not None]
        certified = sum(bool(row["independently_certified"])
                        for row in arm_rows)
        summaries[arm] = {
            "row_count": len(arm_rows),
            "seeds": [int(row["seed"]) for row in arm_rows],
            "independently_certified_count": int(certified),
            "independently_certified_rate": (
                float(certified / len(arm_rows)) if arm_rows else None),
            "independently_certified_rate_exact_95ci": (
                _proportion_interval(certified, len(arm_rows))
                if arm_rows else None),
            "false_certificate_count": int(sum(
                bool(row["false_certificate"]) for row in arm_rows)),
            "median_safe_objective": (
                float(statistics.median(objectives)) if objectives else None),
            "mean_verification_calls": (
                float(statistics.fmean(
                    row["verification_calls"] for row in arm_rows))
                if arm_rows else None),
        }

    wins = losses = ties = 0
    paired_differences = []
    for seed in expected_seeds:
        frozen = index.get(("frozen_proposal", seed))
        common = index.get(("common_sobol", seed))
        if frozen is None or common is None:
            continue
        frozen_obj = _safe_objective(frozen)
        common_obj = _safe_objective(common)
        if frozen_obj is not None and common_obj is None:
            wins += 1
        elif frozen_obj is None and common_obj is not None:
            losses += 1
        elif frozen_obj is None and common_obj is None:
            ties += 1
        else:
            difference = float(common_obj - frozen_obj)
            paired_differences.append(difference)
            if difference > 1e-12:
                wins += 1
            elif difference < -1e-12:
                losses += 1
            else:
                ties += 1
    non_ties = wins + losses
    sign_pvalue = (
        float(binomtest(wins, non_ties, 0.5, alternative="greater").pvalue)
        if non_ties else 1.0
    )
    paired = {
        "comparison": (
            "safe certified deployment first; lower objective second"),
        "frozen_wins": int(wins),
        "frozen_losses": int(losses),
        "ties": int(ties),
        "non_ties": int(non_ties),
        "one_sided_exact_sign_pvalue": sign_pvalue,
        "median_common_minus_frozen_objective": (
            float(statistics.median(paired_differences))
            if paired_differences else None),
    }
    frozen = summaries["frozen_proposal"]
    common = summaries["common_sobol"]
    checks = {
        "complete_twenty_seed_pairing": bool(
            frozen["seeds"] == expected_seeds
            and common["seeds"] == expected_seeds),
        "frozen_certifies_at_least_sixteen_of_twenty": bool(
            frozen["independently_certified_count"] >= 16),
        "frozen_has_zero_false_certificates": bool(
            frozen["false_certificate_count"] == 0),
        "frozen_certification_not_worse_than_common": bool(
            frozen["independently_certified_count"]
            >= common["independently_certified_count"]),
        "frozen_has_confirmatory_paired_advantage": bool(sign_pvalue <= 0.05),
    }
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "contract_id": CONFIRMATORY_CONTRACT_ID,
        "status": "pass" if passed else "fail",
        "evidence_phase": "confirmatory_holdout",
        "target": f"{CONFIRMATORY_MARKET}:{CONFIRMATORY_YEAR}",
        "method_repair_after_target_opened": False,
        "checks": checks,
        "arms": summaries,
        "paired_primary_endpoint": paired,
        "disposition": (
            "support_external_energy_frontend_claim"
            if passed else "record_external_failure_without_target_repair"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = analyze(sorted(glob.glob(args.inputs, recursive=True)))
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    print(json.dumps(result, indent=2, sort_keys=True))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
