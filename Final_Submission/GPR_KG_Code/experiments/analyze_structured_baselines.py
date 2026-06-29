"""Analyze structured-baseline reruns against manuscript baseline values."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


RESULT_DIR = PROJECT / "server311_structured_baselines_full_20260521_cmd"
OUT_DIR = ROOT / "results" / "audit"
METHODS = ("cEHVI", "cParEGO", "NSGA-II-K")
PROBLEMS = ("RZDT1", "RZDT2", "RZDT5_RR")

OLD_BASELINES = {
    ("RZDT1", "cEHVI"): dict(hv=0.426, igd=0.838, cvr=0.519, nd=5.1, tpos=0.0, time=73.5),
    ("RZDT1", "cParEGO"): dict(hv=0.225, igd=1.003, cvr=0.380, nd=5.1, tpos=0.0, time=1.1),
    ("RZDT1", "NSGA-II-K"): dict(hv=0.282, igd=0.879, cvr=0.400, nd=5.2, tpos=0.0, time=9.0),
    ("RZDT2", "cEHVI"): dict(hv=0.073, igd=1.462, cvr=0.302, nd=3.8, tpos=0.0, time=90.4),
    ("RZDT2", "cParEGO"): dict(hv=0.000, igd=1.761, cvr=0.290, nd=4.1, tpos=0.0, time=4.8),
    ("RZDT2", "NSGA-II-K"): dict(hv=0.036, igd=1.770, cvr=0.183, nd=3.8, tpos=0.0, time=29.8),
    ("RZDT5_RR", "cEHVI"): dict(hv=2.087, igd=0.061, cvr=0.022, nd=11.2, tpos=0.0, time=997.4),
    ("RZDT5_RR", "cParEGO"): dict(hv=2.080, igd=0.068, cvr=0.013, nd=12.7, tpos=0.0, time=2.9),
    ("RZDT5_RR", "NSGA-II-K"): dict(hv=2.077, igd=0.066, cvr=0.019, nd=11.3, tpos=0.0, time=57.5),
}


def load_runs():
    rows = []
    missing = []
    for problem in PROBLEMS:
        for method in METHODS:
            for rep in range(10):
                path = RESULT_DIR / problem / method / f"rep{rep:02d}" / "result.json"
                if not path.exists():
                    missing.append(str(path))
                    continue
                with path.open("r", encoding="utf-8") as f:
                    r = json.load(f)
                rows.append({
                    "problem": problem,
                    "method": method,
                    "rep": rep,
                    "seed": r.get("seed"),
                    "hv": float(r["hv_final"]),
                    "igd": float(r["igd_final"]),
                    "cvr": float(r["cvr_final"]),
                    "nd": float(r["n_pareto_solutions"]),
                    "tpos": float(r.get("n_true_pareto_hits", math.nan)),
                    "time": float(r.get("wall_time_sec", r.get("total_time_sec", 0.0))),
                    "n_simulations": int(r.get("n_simulations", 0)),
                    "n_initial": len(r.get("initial_samples", [])),
                    "n_observations": len(r.get("observation_history", [])),
                })
    return rows, missing


def mean_se(values):
    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr))
    se = float(np.std(arr, ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return mean, se


def main():
    rows, missing = load_runs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    grouped = {}
    for row in rows:
        grouped.setdefault((row["problem"], row["method"]), []).append(row)

    summary_rows = []
    for key, group in sorted(grouped.items()):
        problem, method = key
        old = OLD_BASELINES[key]
        rec = {"problem": problem, "method": method, "n": len(group)}
        for metric in ("hv", "igd", "cvr", "nd", "tpos", "time"):
            m, se = mean_se([g[metric] for g in group])
            rec[f"new_{metric}_mean"] = m
            rec[f"new_{metric}_se"] = se
            rec[f"old_{metric}"] = old[metric]
            rec[f"delta_{metric}"] = m - old[metric]
        rec["n_bad_budget"] = sum(
            1 for g in group if g["n_simulations"] != 150
            or g["n_initial"] != 30 or g["n_observations"] != 150)
        summary_rows.append(rec)

    raw_path = OUT_DIR / "structured_baseline_raw_runs_20260521.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = OUT_DIR / "structured_baseline_vs_old_20260521.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    payload = {
        "result_dir": str(RESULT_DIR),
        "n_runs": len(rows),
        "missing": missing,
        "summary": summary_rows,
    }
    json_path = OUT_DIR / "structured_baseline_vs_old_20260521.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"runs={len(rows)} missing={len(missing)}")
    print(f"wrote {raw_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {json_path}")
    for rec in summary_rows:
        print(
            f"{rec['problem']:8s} {rec['method']:9s} "
            f"HV {rec['new_hv_mean']:.3f} vs {rec['old_hv']:.3f} "
            f"IGD {rec['new_igd_mean']:.3f} vs {rec['old_igd']:.3f} "
            f"CVR {rec['new_cvr_mean']:.3f} vs {rec['old_cvr']:.3f} "
            f"ND {rec['new_nd_mean']:.1f} vs {rec['old_nd']:.1f} "
            f"budget_issues={rec['n_bad_budget']}"
        )


if __name__ == "__main__":
    main()
