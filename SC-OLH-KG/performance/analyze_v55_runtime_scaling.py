#!/usr/bin/env python3
"""Audit exact-V55 numerical equivalence and CPU-worker scaling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SCORE_FIELDS = (
    "exact_kg_raw_scores_active",
    "exact_kg_policy_scores_active",
    "certificate_deficit_raw_scores_active",
    "certificate_deficit_policy_scores_active",
    "pairwise_prefix_risk_policy_scores_active",
    "pairwise_prefix_certificate_policy_scores_active",
)


def load_run(cores, root):
    records = {}
    for path in sorted(Path(root).rglob("result.json")):
        payload = json.loads(path.read_text())
        if not payload.get("rows"):
            continue
        row = payload["rows"][0]
        trace = list(row.get("online_action_trace") or [])
        if len(trace) != 1:
            raise ValueError(f"runtime probe must have one action: {path}")
        heldout = str(row["heldout"])
        records[heldout] = {
            "path": str(path),
            "exact_jobs": int(payload["config"]["exact_kg_jobs"]),
            "algorithm_time_sec": float(row["algorithm_time_sec"]),
            "kg_time_sec": float(
                row["stage_times"]["t_kg_compute"]["total"]),
            "selected_fingerprint": str(trace[0]["x_fingerprint"]),
            "active_fingerprints": list(
                trace[0]["exact_kg_active_action_fingerprints"]),
            "scores": {
                field: np.asarray(trace[0][field], dtype=float)
                for field in SCORE_FIELDS
            },
        }
    if not records:
        raise FileNotFoundError(f"no result.json files under {root}")
    if any(record["exact_jobs"] != int(cores) for record in records.values()):
        raise ValueError(f"declared cores and exact jobs disagree under {root}")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", action="append", required=True,
        help="worker-count=result-root (repeat for each scaling point)",
    )
    parser.add_argument("--reference-cores", type=int, default=12)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    runs = {}
    for item in args.run:
        raw_cores, raw_root = item.split("=", 1)
        cores = int(raw_cores)
        if cores in runs:
            raise ValueError(f"duplicate worker count {cores}")
        runs[cores] = load_run(cores, Path(raw_root))
    reference_cores = int(args.reference_cores)
    if reference_cores not in runs:
        raise ValueError("reference worker count is missing")
    reference = runs[reference_cores]
    domains = sorted(reference)

    cells = []
    all_equivalent = True
    for cores, records in sorted(runs.items()):
        if sorted(records) != domains:
            raise ValueError(f"domain mismatch for {cores} workers")
        for domain in domains:
            base = reference[domain]
            record = records[domain]
            fingerprint_equal = (
                record["selected_fingerprint"]
                == base["selected_fingerprint"]
                and record["active_fingerprints"]
                == base["active_fingerprints"]
            )
            max_abs_error = 0.0
            arrays_equal = True
            for field in SCORE_FIELDS:
                lhs = record["scores"][field]
                rhs = base["scores"][field]
                if lhs.shape != rhs.shape:
                    arrays_equal = False
                    max_abs_error = float("inf")
                    break
                error = float(np.max(np.abs(lhs - rhs))) if lhs.size else 0.0
                max_abs_error = max(max_abs_error, error)
                arrays_equal = arrays_equal and bool(np.allclose(
                    lhs, rhs, rtol=0.0, atol=float(args.atol),
                    equal_nan=True,
                ))
            equivalent = bool(fingerprint_equal and arrays_equal)
            all_equivalent = all_equivalent and equivalent
            speedup = (
                base["kg_time_sec"] / record["kg_time_sec"]
                if record["kg_time_sec"] > 0.0 else float("inf")
            )
            cells.append({
                "cores": int(cores),
                "domain": domain,
                "kg_time_sec": record["kg_time_sec"],
                "algorithm_time_sec": record["algorithm_time_sec"],
                "speedup_vs_reference": float(speedup),
                "parallel_efficiency_vs_reference": float(
                    speedup * reference_cores / cores),
                "selected_and_action_set_equal": fingerprint_equal,
                "score_arrays_equal": arrays_equal,
                "max_abs_score_error": max_abs_error,
                "equivalent": equivalent,
                "path": record["path"],
            })

    aggregate = {}
    for cores in sorted(runs):
        subset = [row for row in cells if row["cores"] == cores]
        aggregate[str(cores)] = {
            "median_kg_time_sec": float(np.median([
                row["kg_time_sec"] for row in subset])),
            "median_speedup_vs_reference": float(np.median([
                row["speedup_vs_reference"] for row in subset])),
            "median_parallel_efficiency_vs_reference": float(np.median([
                row["parallel_efficiency_vs_reference"]
                for row in subset
            ])),
            "all_equivalent": bool(all(row["equivalent"] for row in subset)),
        }
    report = {
        "status": "pass" if all_equivalent else "fail",
        "reference_cores": reference_cores,
        "absolute_tolerance": float(args.atol),
        "all_numerically_equivalent": bool(all_equivalent),
        "aggregate": aggregate,
        "cells": cells,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    raise SystemExit(0 if all_equivalent else 1)


if __name__ == "__main__":
    main()
