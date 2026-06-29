"""Forensic summaries of historical RZDT result directories."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
CANDIDATE_DIRS = [
    PROJECT / "GPR_KG_Code" / "results" / "d5_v2",
    PROJECT / "GPR_KG_Code" / "results" / "rzdt5rr",
    PROJECT / "GPR_KG_Code" / "results" / "instrumented",
    PROJECT / "GPR_KG_Code" / "results" / "phaseB",
    PROJECT / "server311_pilot_results",
    PROJECT / "server311_targeted_results",
    PROJECT / "cpu2_updated_results_20260518_120624",
    PROJECT / "server311_checkpointed_full_20260519",
    PROJECT / "server311_nv_full_20260520",
]


def load_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def collect_results(root: Path):
    rows = []
    if not root.exists():
        return rows
    for path in root.rglob("*.json"):
        if path.name in {
            "summary.json",
            "summary_all.json",
            "summary_by_problem.json",
            "summary_by_problem_method.json",
            "run_config.json",
            "run_meta.json",
            "recommendation_summary.json",
        }:
            continue
        data = load_json(path)
        if not isinstance(data, dict):
            continue
        if not {"hv_final", "igd_final", "cvr_final"}.issubset(data):
            continue
        method = data.get("method")
        problem = data.get("problem")
        if not method or not problem:
            parts = path.parts
            for p in parts:
                if p in {"RZDT1", "RZDT2", "RZDT5", "RZDT5_R", "RZDT5_RR"}:
                    problem = problem or p
            stem = path.stem
            for m in ("GPR_KG_nV", "GPR_KG", "cEHVI", "cParEGO", "NSGA-II-K", "NSGA-II-D", "RS"):
                if m in stem:
                    method = method or m.replace("_", "-")
        if not method or not problem:
            continue
        rows.append({
            "root": str(root.relative_to(PROJECT)),
            "path": str(path.relative_to(PROJECT)),
            "problem": str(problem),
            "method": str(method).replace("GPR_KG", "GPR-KG"),
            "hv": float(data["hv_final"]),
            "igd": float(data["igd_final"]),
            "cvr": float(data["cvr_final"]),
            "nd": float(data.get("n_pareto_solutions", math.nan)),
            "time": float(data.get("total_time_sec", data.get("wall_time_sec", math.nan))),
            "seed": data.get("seed"),
            "rep": data.get("rep"),
        })
    return rows


def mean(vals):
    arr = np.array(vals, dtype=float)
    return float(np.mean(arr)) if len(arr) else math.nan


def se(vals):
    arr = np.array(vals, dtype=float)
    if len(arr) <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / math.sqrt(len(arr)))


def main():
    all_rows = []
    for root in CANDIDATE_DIRS:
        all_rows.extend(collect_results(root))

    groups = {}
    for row in all_rows:
        key = (row["root"], row["problem"], row["method"])
        groups.setdefault(key, []).append(row)

    records = []
    for (root, problem, method), rows in sorted(groups.items()):
        records.append({
            "root": root,
            "problem": problem,
            "method": method,
            "n": len(rows),
            "hv_mean": mean([r["hv"] for r in rows]),
            "hv_se": se([r["hv"] for r in rows]),
            "igd_mean": mean([r["igd"] for r in rows]),
            "igd_se": se([r["igd"] for r in rows]),
            "cvr_mean": mean([r["cvr"] for r in rows]),
            "cvr_se": se([r["cvr"] for r in rows]),
            "nd_mean": mean([r["nd"] for r in rows if not math.isnan(r["nd"])]),
            "time_mean": mean([r["time"] for r in rows if not math.isnan(r["time"])]),
        })

    out = PROJECT / "GPR_KG_Code" / "results" / "audit" / "forensics_rzdt_versions_20260521.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump({"n_rows": len(all_rows), "summary": records}, f, indent=2)

    print(f"rows={len(all_rows)} groups={len(records)} wrote={out}")
    for rec in records:
        if rec["problem"] in {"RZDT1", "RZDT2", "RZDT5_RR"}:
            print(
                f"{rec['root']:<42s} {rec['problem']:<8s} {rec['method']:<10s} "
                f"n={rec['n']:2d} HV={rec['hv_mean']:.3f} "
                f"IGD={rec['igd_mean']:.3f} CVR={rec['cvr_mean']:.3f} "
                f"ND={rec['nd_mean']:.1f}"
            )


if __name__ == "__main__":
    main()
