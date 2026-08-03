#!/usr/bin/env python3
"""Audit whether the registered OPSD development domain is nonvacuous."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.benchmark_quality import json_safe  # noqa: E402
from problems.energy_reliability import (  # noqa: E402
    OPSDStorageReliabilityProblem,
    StoragePhysics,
)


AUDIT_CONTRACT_ID = "opsd_energy_certifiability_dk2_development_v1"
CLEAR_INFEASIBILITY_MARGIN = 0.05


def _constant(problem, level):
    return tuple([int(round(float(level) * problem.L))] * problem.d)


def _profile(problem, kind, low, high, cycles=1):
    position = (np.arange(problem.d, dtype=float) + 0.5) / problem.d
    if kind == "ramp_up":
        values = low + (high - low) * position
    elif kind == "ramp_down":
        values = high - (high - low) * position
    elif kind == "sinusoid":
        center = 0.5 * (low + high)
        amplitude = 0.5 * (high - low)
        values = center + amplitude * np.sin(2.0 * np.pi * cycles * position)
    elif kind == "two_block":
        values = np.where(position < 0.5, low, high)
    else:
        raise ValueError(f"unknown registered profile {kind!r}")
    return tuple(np.clip(np.rint(values * problem.L), 0, problem.L).astype(int))


def registered_policy_library(problem):
    rows = [
        (f"constant_{level:03d}", _constant(problem, level / 100.0))
        for level in range(0, 101, 10)
    ]
    rows.extend([
        ("ramp_up_10_90", _profile(problem, "ramp_up", 0.10, 0.90)),
        ("ramp_down_10_90", _profile(problem, "ramp_down", 0.10, 0.90)),
        ("sinusoid_20_80_f1", _profile(
            problem, "sinusoid", 0.20, 0.80, cycles=1)),
        ("sinusoid_20_80_f4", _profile(
            problem, "sinusoid", 0.20, 0.80, cycles=4)),
        ("two_block_20_80", _profile(problem, "two_block", 0.20, 0.80)),
        ("two_block_80_20", _profile(problem, "two_block", 0.80, 0.20)),
    ])
    return rows


def audit_problem(problem, *, maximum_windows=None):
    rows = []
    for label, point in registered_policy_library(problem):
        values = problem.split_population(
            point, "audit", maximum_windows=maximum_windows)
        feasible = values[:, 1] <= problem.tau
        rows.append({
            "policy": label,
            "objective_mean": float(np.mean(values[:, 0])),
            "objective_std": float(np.std(values[:, 0], ddof=1)),
            "constraint_mean": float(np.mean(values[:, 1])),
            "constraint_q95": float(np.quantile(values[:, 1], 0.95)),
            "feasible_probability": float(np.mean(feasible)),
            "window_count": int(len(values)),
        })
    target_probability = 1.0 - float(problem.alpha)
    feasible_rows = [
        row for row in rows
        if row["feasible_probability"] >= target_probability
    ]
    infeasible_rows = [
        row for row in rows
        if row["feasible_probability"] < target_probability
    ]
    clearly_infeasible_rows = [
        row for row in rows
        if row["feasible_probability"]
        <= target_probability - CLEAR_INFEASIBILITY_MARGIN
    ]
    objective_spread = (
        max(row["objective_mean"] for row in feasible_rows)
        - min(row["objective_mean"] for row in feasible_rows)
        if feasible_rows else 0.0
    )
    checks = {
        "contains_chance_feasible_policy": bool(feasible_rows),
        "contains_chance_infeasible_policy": bool(infeasible_rows),
        "contains_clearly_infeasible_policy": bool(clearly_infeasible_rows),
        "feasible_objective_spread_positive": bool(objective_spread > 1e-4),
        "chronological_splits_disjoint": bool(
            problem._starts["search"][-1] + problem.d - 1
            < problem._starts["audit"][0]
            and problem._starts["audit"][-1] + problem.d - 1
            < problem._starts["verification"][0]
        ),
        "observable_coordinate_target_outcome_free": bool(
            not problem.information_contract()[
                "actual_target_error_used_by_observable_coordinate"]
        ),
    }
    return {
        "schema_version": 1,
        "contract_id": AUDIT_CONTRACT_ID,
        "status": "pass" if all(checks.values()) else "fail",
        "development_only": True,
        "admissible_for_confirmatory_method_selection": True,
        "problem": problem.problem_name,
        "market": problem.market,
        "year": problem.year,
        "dimension": problem.d,
        "alpha": problem.alpha,
        "target_probability": target_probability,
        "clear_infeasibility_margin": CLEAR_INFEASIBILITY_MARGIN,
        "physics": vars(problem.physics),
        "information_contract": problem.information_contract(),
        "checks": checks,
        "feasible_policy_count": int(len(feasible_rows)),
        "chance_infeasible_policy_count": int(len(infeasible_rows)),
        "clearly_infeasible_policy_count": int(len(clearly_infeasible_rows)),
        "feasible_objective_spread": float(objective_spread),
        "best_registered_feasible_policy": (
            min(feasible_rows, key=lambda row: row["objective_mean"])
            if feasible_rows else None
        ),
        "policy_rows": rows,
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
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--market", default="DK_2")
    parser.add_argument("--year", type=int, default=2018)
    parser.add_argument("--d", type=int, default=1000)
    parser.add_argument("--maximum-windows", type=int, default=0)
    parser.add_argument("--energy-capacity", type=float, default=0.40)
    parser.add_argument("--power-capacity", type=float, default=0.40)
    parser.add_argument(
        "--maximum-unserved-fraction", type=float, default=0.11)
    args = parser.parse_args()
    problem = OPSDStorageReliabilityProblem(
        args.data,
        market=args.market,
        year=args.year,
        d=args.d,
        physics=StoragePhysics(
            energy_capacity=args.energy_capacity,
            power_capacity=args.power_capacity,
            maximum_unserved_fraction=args.maximum_unserved_fraction,
        ),
    )
    result = audit_problem(
        problem,
        maximum_windows=(
            int(args.maximum_windows) if args.maximum_windows > 0 else None),
    )
    _atomic_json(args.out, result)
    print(json.dumps({
        "status": result["status"],
        "out": str(args.out),
        "feasible_policy_count": result["feasible_policy_count"],
        "clearly_infeasible_policy_count": result[
            "clearly_infeasible_policy_count"],
        "best_registered_feasible_policy": result[
            "best_registered_feasible_policy"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
