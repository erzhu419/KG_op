#!/usr/bin/env python3
"""Run one shard of the uniform synthetic terminal verifier."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import norm

from core.designs import integer_design_fingerprint
from core.terminal_verification import verify_frozen_shortlist
from performance.benchmark_lodo_meta_prior import build_scalarized_problem


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _truth_audit(problem, point):
    point = tuple(int(value) for value in point)
    true_objective = float(problem.true_objective(point))
    true_constraint_mean = float(problem.true_constraint_mean(point))
    true_sigma = np.asarray(problem.true_sigma(point), dtype=float)
    z = float(norm.ppf(1.0 - float(problem.alpha)))
    true_margin = float(
        true_constraint_mean
        + z * float(true_sigma[1])
        - float(problem.tau)
    )
    true_best_x, true_best_objective = problem.true_best_feasible()
    regret = (
        float("nan")
        if not math.isfinite(float(true_best_objective))
        else true_objective - float(true_best_objective)
    )
    return {
        "true_objective": true_objective,
        "true_constraint_mean": true_constraint_mean,
        "true_constraint_sigma": float(true_sigma[1]),
        "true_chance_margin": true_margin,
        "true_feasible": bool(true_margin <= 0.0),
        "true_best_x": (
            None
            if true_best_x is None
            else list(map(int, true_best_x))
        ),
        "true_best_objective": float(true_best_objective),
        "simple_regret": float(regret),
        "feasible_regret": (
            max(0.0, float(regret)) if true_margin <= 0.0 else None
        ),
        "constraint_violation": max(0.0, true_margin),
    }


def verify_row(row, contract):
    problem = build_scalarized_problem(
        row["domain"],
        int(row["target_dimension"]),
        100,
        0.04,
        0.05,
        (0.5, 0.5),
    )
    frozen = [
        {
            **item,
            "point_fingerprint": integer_design_fingerprint([
                item["point"]]),
            "shortlist_frozen_before_verification": True,
        }
        for item in row["shortlist"]
    ]
    deployed, verification = verify_frozen_shortlist(
        problem,
        frozen,
        seed=int(row["seed"]) + 74000000,
        search_evaluation_count=int(row["target_search_calls"]),
        candidate_budgets=tuple(map(
            int, contract["candidate_budgets"])),
        familywise_delta=float(contract["familywise_delta"]),
        method=str(contract["method"]),
        shortlist_mode=str(contract["shortlist_mode"]),
    )
    truth = _truth_audit(problem, deployed)
    result = {
        "schema_version": 1,
        "status": "ok",
        "method": row["uniform_method_identity"],
        "source_method_identity": row["source_method_identity"],
        "source_track_id": row["source_track_id"],
        "problem": row["domain"],
        "d": int(row["target_dimension"]),
        "seed": int(row["seed"]),
        "source_calls": int(row["source_calls"]),
        "n_search_simulations": int(row["target_search_calls"]),
        "n_verification_simulations": int(
            verification["verification_budget"]),
        "n_target_simulations_total": int(
            row["target_search_calls"]
            + verification["verification_budget"]
        ),
        "optimization_calls_excluding_verification": int(
            row["optimization_calls_excluding_verification"]),
        "source_archive_fingerprint": row[
            "source_archive_fingerprint"],
        "initial_design_fingerprint": row[
            "initial_design_fingerprint"],
        "source_result_sha256": row["source_result_sha256"],
        "uniform_verifier_contract_id": contract["contract_id"],
        "x_recommended": list(map(int, deployed)),
        "terminal_verification": verification,
        "terminal_verification_truth_audit": {
            **truth,
            "used_for_selection": False,
            "computed_after_verification": True,
        },
        **truth,
    }
    return result


def _output_path(root, row):
    return (
        Path(root)
        / row["source_method_identity"].replace("/", "_")
        / row["domain"]
        / f"seed{int(row['seed']):04d}"
        / "result.json"
    )


def _run_one(item):
    row, contract, out_root = item
    output = _output_path(out_root, row)
    result = verify_row(row, contract)
    _atomic_json(output, {"status": "ok", "result": result})
    return str(output)


def run_shard(manifest, *, start, end, out_root, jobs=1):
    rows = list(manifest["rows"])[int(start):int(end)]
    contract = {
        key: manifest[key]
        for key in (
            "contract_id",
            "candidate_budgets",
            "familywise_delta",
            "method",
            "shortlist_mode",
        )
    }
    work = [(row, contract, str(out_root)) for row in rows]
    if int(jobs) <= 1:
        outputs = [_run_one(item) for item in work]
    else:
        with ProcessPoolExecutor(max_workers=int(jobs)) as executor:
            outputs = list(executor.map(_run_one, work))
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    outputs = run_shard(
        manifest,
        start=args.start,
        end=args.end,
        out_root=args.out_root,
        jobs=args.jobs,
    )
    print(json.dumps({
        "status": "ok",
        "start": int(args.start),
        "end": int(args.end),
        "completed": len(outputs),
        "outputs": outputs,
    }, indent=2))


if __name__ == "__main__":
    main()
