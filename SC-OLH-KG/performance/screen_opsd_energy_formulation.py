#!/usr/bin/env python3
"""Reproduce the outcome-independent OPSD development formulation screen."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.audit_opsd_energy_certifiability import (  # noqa: E402
    audit_problem,
)
from performance.benchmark_quality import json_safe  # noqa: E402
from problems.energy_reliability import (  # noqa: E402
    OPSDStorageReliabilityProblem,
    StoragePhysics,
)


KAPPA_GRID = (0.07, 0.09, 0.11, 0.13)
ENERGY_GRID = (0.20, 0.30, 0.35, 0.40, 0.45, 0.50)
POWER_GRID = (0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50)
REGISTERED_SELECTION = (0.11, 0.40, 0.40)


def _evaluate(args):
    data, market, year, dimension, windows, kappa, energy, power = args
    problem = OPSDStorageReliabilityProblem(
        data,
        market=market,
        year=year,
        d=dimension,
        physics=StoragePhysics(
            energy_capacity=energy,
            power_capacity=power,
            maximum_unserved_fraction=kappa,
        ),
    )
    result = audit_problem(problem, maximum_windows=windows)
    probabilities = [
        row["feasible_probability"] for row in result["policy_rows"]
    ]
    return {
        "kappa": float(kappa),
        "energy_capacity": float(energy),
        "power_capacity": float(power),
        "status": result["status"],
        "feasible_policy_count": result["feasible_policy_count"],
        "chance_infeasible_policy_count": result[
            "chance_infeasible_policy_count"],
        "clearly_infeasible_policy_count": result[
            "clearly_infeasible_policy_count"],
        "feasible_objective_spread": result["feasible_objective_spread"],
        "minimum_feasible_probability": float(min(probabilities)),
        "maximum_feasible_probability": float(max(probabilities)),
    }


def select_configuration(rows):
    passing = [row for row in rows if row["status"] == "pass"]
    if not passing:
        return None
    return min(
        passing,
        key=lambda row: (
            row["kappa"],
            row["energy_capacity"],
            row["power_capacity"],
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--market", default="DK_2")
    parser.add_argument("--year", type=int, default=2018)
    parser.add_argument("--d", type=int, default=1000)
    parser.add_argument("--maximum-windows", type=int, default=512)
    parser.add_argument("--jobs", type=int, default=12)
    args = parser.parse_args()

    configurations = [
        (
            args.data,
            args.market,
            int(args.year),
            int(args.d),
            int(args.maximum_windows),
            kappa,
            energy,
            power,
        )
        for kappa, energy, power in product(
            KAPPA_GRID, ENERGY_GRID, POWER_GRID)
    ]
    with ProcessPoolExecutor(max_workers=max(1, int(args.jobs))) as executor:
        rows = list(executor.map(_evaluate, configurations))
    rows.sort(key=lambda row: (
        row["kappa"], row["energy_capacity"], row["power_capacity"]))
    selected = select_configuration(rows)
    selected_tuple = None if selected is None else (
        selected["kappa"],
        selected["energy_capacity"],
        selected["power_capacity"],
    )
    status = "pass" if selected_tuple == REGISTERED_SELECTION else "fail"
    payload = {
        "schema_version": 1,
        "contract_id": "opsd_energy_formulation_screen_v1",
        "status": status,
        "development_only": True,
        "optimizer_executed": False,
        "target_policy_outcomes_used_for_training": False,
        "selection_order": [
            "minimum_kappa",
            "minimum_energy_capacity",
            "minimum_power_capacity",
        ],
        "registered_selection": {
            "kappa": REGISTERED_SELECTION[0],
            "energy_capacity": REGISTERED_SELECTION[1],
            "power_capacity": REGISTERED_SELECTION[2],
        },
        "selected": selected,
        "maximum_windows": int(args.maximum_windows),
        "rows": rows,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps({
        "status": status,
        "selected": selected,
        "out": str(output),
    }, indent=2, sort_keys=True))
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
