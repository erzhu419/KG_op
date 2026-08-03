#!/usr/bin/env python3
"""Post-freeze truth audit of external OPSD initial-design coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.designs import (  # noqa: E402
    common_sobol_integer_design,
    load_frozen_source_informed_design,
)
from performance.benchmark_quality import json_safe  # noqa: E402
from performance.materialize_external_energy_design import target_task_id  # noqa: E402
from problems.energy_reliability import OPSDStorageReliabilityProblem  # noqa: E402


AUDIT_CONTRACT_ID = "opsd_energy_frozen_frontend_coverage_development_v1"


def _audit_points(problem, points, *, maximum_windows=None):
    rows = []
    for index, point in enumerate(points):
        values = problem.split_population(
            point, "audit", maximum_windows=maximum_windows)
        probability = float(np.mean(values[:, 1] <= problem.tau))
        rows.append({
            "index": int(index),
            "objective_mean": float(np.mean(values[:, 0])),
            "constraint_mean": float(np.mean(values[:, 1])),
            "feasible_probability": probability,
            "chance_feasible": bool(probability >= 1.0 - problem.alpha),
            "window_count": int(len(values)),
        })
    feasible = [row for row in rows if row["chance_feasible"]]
    return {
        "point_count": int(len(rows)),
        "chance_feasible_count": int(len(feasible)),
        "contains_chance_feasible": bool(feasible),
        "best_feasible_objective": (
            min(row["objective_mean"] for row in feasible)
            if feasible else None
        ),
        "rows": rows,
    }


def audit_frontend(
    design_path,
    data_path,
    *,
    market="DK_2",
    year=2018,
    maximum_windows=None,
):
    payload = json.loads(Path(design_path).read_text(encoding="utf-8"))
    dimension = int(payload["dimension"])
    n0 = int(payload["n0"])
    problem = OPSDStorageReliabilityProblem(
        data_path, market=market, year=year, d=dimension)
    expected_target = target_task_id(market, year)
    if payload.get("heldout_target_domain") != expected_target:
        raise ValueError("energy design and audit target disagree")
    rows = []
    frozen_cache = {}
    for seed_text in sorted(payload["designs"], key=int):
        seed = int(seed_text)
        frozen, contract = load_frozen_source_informed_design(
            design_path,
            heldout=expected_target,
            seed=seed,
            n0=n0,
            dimension=dimension,
        )
        sobol = common_sobol_integer_design(problem, n0, seed)
        fingerprint = contract["fingerprint"]
        if fingerprint not in frozen_cache:
            frozen_cache[fingerprint] = _audit_points(
                problem, frozen, maximum_windows=maximum_windows)
        rows.append({
            "seed": seed,
            "frozen": frozen_cache[fingerprint],
            "common_sobol": _audit_points(
                problem, sobol, maximum_windows=maximum_windows),
            "frozen_design_contract": contract,
        })

    def summarize(key):
        arm = [row[key] for row in rows]
        feasible_objectives = [
            row["best_feasible_objective"] for row in arm
            if row["best_feasible_objective"] is not None
        ]
        return {
            "seed_count": int(len(arm)),
            "seeds_with_chance_feasible": int(sum(
                row["contains_chance_feasible"] for row in arm)),
            "median_best_feasible_objective": (
                float(statistics.median(feasible_objectives))
                if feasible_objectives else None
            ),
        }

    frozen_summary = summarize("frozen")
    sobol_summary = summarize("common_sobol")
    unique_frozen_designs = int(len(frozen_cache))
    checks = {
        "all_registered_seeds_present": bool(
            len(rows) == int(payload["n_seeds"])),
        "target_outcome_free_materialization": bool(
            payload["target_labels_used"] is False
            and payload["target_oracle_used"] is False
            and payload["target_simulator_calls_during_materialization"] == 0
            and payload.get("target_actual_error_used_during_materialization")
            is False
            and payload.get("data_contract", {}).get("actual_target_error_read")
            is False
            and payload.get("data_contract", {}).get(
                "problem_information", {}).get("outcome_access_enabled") is False
        ),
        "deterministic_frozen_atlas_contains_chance_feasible": bool(
            any(row["contains_chance_feasible"] for row in frozen_cache.values())
        ),
        "frozen_not_worse_than_common_sobol_coverage": bool(
            frozen_summary["seeds_with_chance_feasible"]
            >= sobol_summary["seeds_with_chance_feasible"]
        ),
    }
    return {
        "schema_version": 1,
        "contract_id": AUDIT_CONTRACT_ID,
        "status": "pass" if all(checks.values()) else "fail",
        "development_only": True,
        "confirmatory_target_opened": False,
        "market": market,
        "year": int(year),
        "dimension": dimension,
        "n0": n0,
        "unique_frozen_design_count": unique_frozen_designs,
        "frozen_atlas_is_deterministic_across_seeds": bool(
            unique_frozen_designs == 1),
        "seed_interpretation": (
            "frozen truth coverage is one deterministic-atlas result; seeds "
            "become independent only in the charged stochastic online gate"
        ),
        "checks": checks,
        "frozen_summary": frozen_summary,
        "common_sobol_summary": sobol_summary,
        "seed_rows": rows,
    }


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--market", default="DK_2")
    parser.add_argument("--year", type=int, default=2018)
    parser.add_argument("--maximum-windows", type=int, default=0)
    args = parser.parse_args()
    result = audit_frontend(
        args.design,
        args.data,
        market=args.market,
        year=args.year,
        maximum_windows=(
            args.maximum_windows if args.maximum_windows > 0 else None),
    )
    _atomic_json(args.out, result)
    print(json.dumps({
        "status": result["status"],
        "out": str(args.out),
        "frozen": result["frozen_summary"],
        "common_sobol": result["common_sobol_summary"],
        "checks": result["checks"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
