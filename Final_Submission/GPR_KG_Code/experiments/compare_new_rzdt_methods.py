"""Build a compact comparison table for the latest RZDT reruns."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "GPR_KG_Code"
OUT_DIR = ROOT / "results" / "audit"

KG_DIR = PROJECT / "server311_checkpointed_full_20260519"
NV_DIR = PROJECT / "server311_nv_full_20260520"
BASELINE_DIR = PROJECT / "server311_structured_baselines_full_20260521_cmd"
PROBLEMS = ("RZDT1", "RZDT2", "RZDT5_RR")
METHOD_ORDER = ("GPR-KG", "GPR-KG-nV", "cEHVI", "cParEGO", "NSGA-II-K")


def mean_se(vals):
    arr = np.array(vals, dtype=float)
    return (
        float(np.mean(arr)),
        float(np.std(arr, ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0,
    )


def load_runs():
    rows = []
    for problem in PROBLEMS:
        for method, root in (("GPR-KG", KG_DIR), ("GPR-KG-nV", NV_DIR)):
            for path in sorted((root / problem).glob(f"{method}_rep*/result.json")):
                with path.open("r", encoding="utf-8") as f:
                    r = json.load(f)
                rows.append({
                    "problem": problem,
                    "method": method,
                    "hv": float(r["hv_final"]),
                    "igd": float(r["igd_final"]),
                    "cvr": float(r["cvr_final"]),
                    "nd": float(r["n_pareto_solutions"]),
                    "tpos": float(r.get("n_true_pareto_hits", 0.0)),
                    "time": float(r.get("total_time_sec", r.get("wall_time_sec", 0.0))),
                })
        for method in ("cEHVI", "cParEGO", "NSGA-II-K"):
            for path in sorted((BASELINE_DIR / problem / method).glob("rep*/result.json")):
                with path.open("r", encoding="utf-8") as f:
                    r = json.load(f)
                rows.append({
                    "problem": problem,
                    "method": method,
                    "hv": float(r["hv_final"]),
                    "igd": float(r["igd_final"]),
                    "cvr": float(r["cvr_final"]),
                    "nd": float(r["n_pareto_solutions"]),
                    "tpos": float(r.get("n_true_pareto_hits", 0.0)),
                    "time": float(r.get("wall_time_sec", r.get("total_time_sec", 0.0))),
                })
    return rows


def main():
    rows = load_runs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for problem in PROBLEMS:
        for method in METHOD_ORDER:
            group = [r for r in rows if r["problem"] == problem and r["method"] == method]
            rec = {"problem": problem, "method": method, "n": len(group)}
            for metric in ("hv", "igd", "cvr", "nd", "tpos", "time"):
                m, se = mean_se([g[metric] for g in group])
                rec[f"{metric}_mean"] = m
                rec[f"{metric}_se"] = se
            summary.append(rec)

    path = OUT_DIR / "latest_rzdt_method_comparison_20260521.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print(f"wrote {path}")
    for rec in summary:
        print(
            f"{rec['problem']:8s} {rec['method']:9s} "
            f"HV={rec['hv_mean']:.3f} IGD={rec['igd_mean']:.3f} "
            f"CVR={rec['cvr_mean']:.3f} ND={rec['nd_mean']:.1f} "
            f"TPOS={rec['tpos_mean']:.1f}"
        )


if __name__ == "__main__":
    main()
