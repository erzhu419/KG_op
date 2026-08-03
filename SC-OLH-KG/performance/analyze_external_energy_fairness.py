#!/usr/bin/env python3
"""Aggregate the post-confirmatory OPSD fairness controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


ARMS = (
    "frozen_proposal_n13",
    "low_frequency_grid_n13",
    "common_sobol_n397",
)
SEEDS = tuple(range(100, 120))


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _objective(row):
    truth = row.get("deployment_truth_audit") or {}
    value = truth.get("true_objective_mean")
    return None if value is None else float(value)


def _bootstrap_median(values, *, seed=20260803, repetitions=20_000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(int(seed))
    samples = rng.choice(
        values, size=(int(repetitions), len(values)), replace=True)
    medians = np.median(samples, axis=1)
    return [float(value) for value in np.quantile(medians, [0.025, 0.975])]


def analyze(paths):
    rows = []
    for path in map(Path, paths):
        row = json.loads(path.read_text(encoding="utf-8"))
        row["_path"] = str(path)
        row["_sha256"] = _sha256(path)
        rows.append(row)
    by_arm = {arm: {} for arm in ARMS}
    for row in rows:
        arm = row.get("fairness_arm")
        if arm is None and row.get("contract_id") == (
                "opsd_energy_gb_gbn_confirmatory_v1"):
            arm = "frozen_proposal_n13"
        if arm not in by_arm:
            continue
        seed = int(row["seed"])
        if seed in by_arm[arm]:
            raise ValueError(f"duplicate {arm}/seed={seed}")
        by_arm[arm][seed] = row

    failures = []
    for arm in ARMS:
        missing = sorted(set(SEEDS) - set(by_arm[arm]))
        extra = sorted(set(by_arm[arm]) - set(SEEDS))
        if missing:
            failures.append(f"{arm} missing seeds {missing}")
        if extra:
            failures.append(f"{arm} has unexpected seeds {extra}")
        for seed, row in by_arm[arm].items():
            if row.get("status") != "ok":
                failures.append(f"{arm}/seed={seed} status is not ok")
            if _objective(row) is None:
                failures.append(f"{arm}/seed={seed} lacks deployment truth")
    if failures:
        return {
            "schema_version": 1,
            "contract_id": "opsd_energy_postconfirmatory_fairness_v1",
            "status": "incomplete",
            "failures": failures,
        }

    summaries = {}
    for arm in ARMS:
        arm_rows = [by_arm[arm][seed] for seed in SEEDS]
        values = np.asarray([_objective(row) for row in arm_rows])
        summaries[arm] = {
            "seed_count": len(arm_rows),
            "independently_certified_count": int(sum(
                bool(row["independently_certified"]) for row in arm_rows)),
            "false_certificate_count": int(sum(
                bool(row["false_certificate"]) for row in arm_rows)),
            "median_safe_objective": float(np.median(values)),
            "median_safe_objective_bootstrap_95ci": _bootstrap_median(values),
            "minimum_safe_objective": float(np.min(values)),
            "maximum_safe_objective": float(np.max(values)),
            "mean_verification_calls": float(np.mean([
                row["verification_calls"] for row in arm_rows
            ])),
        }

    comparisons = {}
    frozen = np.asarray([
        _objective(by_arm["frozen_proposal_n13"][seed]) for seed in SEEDS
    ])
    for arm in ARMS[1:]:
        control = np.asarray([_objective(by_arm[arm][seed]) for seed in SEEDS])
        difference = frozen - control
        wins = int(np.sum(difference < -1e-12))
        losses = int(np.sum(difference > 1e-12))
        ties = int(len(difference) - wins - losses)
        non_ties = wins + losses
        comparisons[f"frozen_vs_{arm}"] = {
            "difference": "frozen_minus_control_for_minimization",
            "frozen_wins": wins,
            "frozen_losses": losses,
            "ties": ties,
            "two_sided_exact_sign_pvalue": (
                1.0 if non_ties == 0 else float(binomtest(
                    wins, non_ties, p=0.5, alternative="two-sided").pvalue)
            ),
            "one_sided_frozen_better_exact_sign_pvalue": (
                1.0 if non_ties == 0 else float(binomtest(
                    wins, non_ties, p=0.5, alternative="greater").pvalue)
            ),
            "median_paired_objective_difference": float(
                np.median(difference)),
            "paired_median_difference_bootstrap_95ci": _bootstrap_median(
                difference, seed=20260804),
        }

    grid_comparison = comparisons[
        "frozen_vs_low_frequency_grid_n13"]
    total_cost_comparison = comparisons[
        "frozen_vs_common_sobol_n397"]
    compact_rows = []
    for arm in ARMS:
        for seed in SEEDS:
            row = by_arm[arm][seed]
            truth = row["deployment_truth_audit"]
            compact_rows.append({
                "arm": arm,
                "seed": int(seed),
                "status": str(row["status"]),
                "target_search_calls": int(row["target_search_calls"]),
                "verification_calls": int(row["verification_calls"]),
                "independently_certified": bool(
                    row["independently_certified"]),
                "false_certificate": bool(row["false_certificate"]),
                "truly_chance_feasible": bool(
                    truth["truly_chance_feasible"]),
                "true_objective_mean": float(
                    truth["true_objective_mean"]),
                "raw_result_basename": Path(row["_path"]).name,
                "raw_result_sha256": str(row["_sha256"]),
            })
    return {
        "schema_version": 1,
        "contract_id": "opsd_energy_postconfirmatory_fairness_v1",
        "status": "complete",
        "evidence_phase": "post_confirmatory_fairness_audit",
        "confirmatory_claim_eligible": False,
        "target": "GB_GBN:2018",
        "seeds": list(SEEDS),
        "summaries": summaries,
        "paired_comparisons": comparisons,
        "decision": {
            "source_atlas_superior_to_natural_low_frequency_control": bool(
                grid_comparison["frozen_wins"]
                > grid_comparison["frozen_losses"]
                and grid_comparison[
                    "one_sided_frozen_better_exact_sign_pvalue"] < 0.05
            ),
            "source_atlas_superior_to_target_only_sobol_at_equal_source_plus_search_cost": bool(
                total_cost_comparison["frozen_wins"]
                > total_cost_comparison["frozen_losses"]
                and total_cost_comparison[
                    "one_sided_frozen_better_exact_sign_pvalue"] < 0.05
            ),
            "external_energy_claim": (
                "dimension-equivariant low-frequency structural transfer and "
                "total-cost advantage over unstructured Sobol; no superiority "
                "claim over the natural constant-policy grid"
            ),
            "method_repair_allowed": False,
        },
        "compact_rows": compact_rows,
        "raw_result_receipts": sorted(
            {Path(row["_path"]).name: row["_sha256"]
             for row in rows}.items()),
        "failures": [],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = analyze(args.paths)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    print(json.dumps({
        "status": payload["status"],
        "out": str(path),
        "decision": payload.get("decision"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
