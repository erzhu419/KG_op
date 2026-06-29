"""Analyze unified variance-study results.

The script is intentionally file-based: after server results are pulled back,
rerun this analyzer locally on the run directory to regenerate all summary
tables and oracle-gap diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = [
    "hv_final",
    "igd_final",
    "cvr_final",
    "n_pareto_solutions",
    "total_time_sec",
]


def load_results(run_dir: Path) -> list[dict]:
    records = []
    for path in sorted(run_dir.glob("*/*/result.json")):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        data["result_path"] = str(path)
        records.append(data)
    return records


def row_from_result(data: dict) -> dict:
    return {
        "problem": data["problem"],
        "method": data["method"],
        "method_base": data.get("method_base", data["method"]),
        "variance_mode": data.get("variance_mode", ""),
        "rep": int(data["rep"]),
        "seed": int(data["seed"]),
        "hv_final": float(data["hv_final"]),
        "igd_final": float(data["igd_final"]),
        "cvr_final": float(data["cvr_final"]),
        "n_pareto_solutions": int(data["n_pareto_solutions"]),
        "n_simulations": int(data["n_simulations"]),
        "total_time_sec": float(data["total_time_sec"]),
        "completed": bool(data.get("completed", False)),
        "result_path": data["result_path"],
    }


def write_csv(rows: list[dict], path: Path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["problem"], row["method"]), []).append(row)

    summary_rows = []
    for (problem, method), group in sorted(grouped.items()):
        out = {"problem": problem, "method": method, "n": len(group)}
        for metric in METRICS:
            vals = np.array([float(r[metric]) for r in group], dtype=float)
            out[f"{metric}_mean"] = float(np.mean(vals))
            out[f"{metric}_std"] = (
                float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
            out[f"{metric}_se"] = (
                out[f"{metric}_std"] / float(np.sqrt(len(vals)))
                if len(vals) > 0 else 0.0)
        summary_rows.append(out)
    return summary_rows


def _gain_ratio(num: float, den: float):
    if abs(den) < 1e-12:
        return None
    return float(num / den)


def oracle_gap_rows(rows: list[dict]) -> list[dict]:
    """Compute paired oracle-gap diagnostics by problem and replication."""
    by_key: dict[tuple[str, int, str], dict] = {}
    for row in rows:
        by_key[(row["problem"], int(row["rep"]), row["method"])] = row

    problems = sorted({r["problem"] for r in rows})
    reps = sorted({int(r["rep"]) for r in rows})
    out_rows = []
    for problem in problems:
        for rep in reps:
            pooled = by_key.get((problem, rep, "GPR-KG-pooled-pre"))
            vepm = by_key.get((problem, rep, "GPR-KG"))
            oracle = by_key.get((problem, rep, "GPR-KG-oracleV"))
            if pooled is None or vepm is None or oracle is None:
                continue

            hv_oracle_gain = oracle["hv_final"] - pooled["hv_final"]
            hv_vepm_gain = vepm["hv_final"] - pooled["hv_final"]
            igd_oracle_gain = pooled["igd_final"] - oracle["igd_final"]
            igd_vepm_gain = pooled["igd_final"] - vepm["igd_final"]
            cvr_oracle_gain = pooled["cvr_final"] - oracle["cvr_final"]
            cvr_vepm_gain = pooled["cvr_final"] - vepm["cvr_final"]

            out_rows.append({
                "problem": problem,
                "rep": rep,
                "hv_oracle_gain": float(hv_oracle_gain),
                "hv_vepm_gain": float(hv_vepm_gain),
                "hv_captured_oracle_ratio": _gain_ratio(
                    hv_vepm_gain, hv_oracle_gain),
                "hv_unrealized_gap": float(
                    oracle["hv_final"] - vepm["hv_final"]),
                "igd_oracle_gain": float(igd_oracle_gain),
                "igd_vepm_gain": float(igd_vepm_gain),
                "igd_captured_oracle_ratio": _gain_ratio(
                    igd_vepm_gain, igd_oracle_gain),
                "igd_unrealized_gap": float(
                    vepm["igd_final"] - oracle["igd_final"]),
                "cvr_oracle_gain": float(cvr_oracle_gain),
                "cvr_vepm_gain": float(cvr_vepm_gain),
                "cvr_captured_oracle_ratio": _gain_ratio(
                    cvr_vepm_gain, cvr_oracle_gain),
                "cvr_unrealized_gap": float(
                    vepm["cvr_final"] - oracle["cvr_final"]),
            })
    return out_rows


def summarize_oracle_gap(gap_rows: list[dict]) -> list[dict]:
    if not gap_rows:
        return []
    metrics = [k for k in gap_rows[0] if k not in ("problem", "rep")]
    grouped: dict[str, list[dict]] = {}
    for row in gap_rows:
        grouped.setdefault(row["problem"], []).append(row)
    out_rows = []
    for problem, group in sorted(grouped.items()):
        out = {"problem": problem, "n": len(group)}
        for metric in metrics:
            vals = np.array([
                float(r[metric]) for r in group if r[metric] is not None
            ], dtype=float)
            if len(vals) == 0:
                out[f"{metric}_mean"] = ""
                out[f"{metric}_std"] = ""
                out[f"{metric}_se"] = ""
                continue
            out[f"{metric}_mean"] = float(np.mean(vals))
            out[f"{metric}_std"] = (
                float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
            out[f"{metric}_se"] = (
                out[f"{metric}_std"] / float(np.sqrt(len(vals)))
                if len(vals) > 0 else 0.0)
        out_rows.append(out)
    return out_rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze a variance-study run directory.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--out_dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_results(run_dir)
    rows = [row_from_result(r) for r in records]
    write_csv(rows, out_dir / "runs.csv")
    summary_rows = summarize(rows)
    write_csv(summary_rows, out_dir / "summary_by_problem_method.csv")
    gap_rows = oracle_gap_rows(rows)
    write_csv(gap_rows, out_dir / "oracle_gap.csv")
    gap_summary = summarize_oracle_gap(gap_rows)
    write_csv(gap_summary, out_dir / "oracle_gap_summary.csv")

    print(f"[analyze] loaded {len(rows)} result files from {run_dir}")
    print(f"[analyze] wrote tables to {out_dir}")
    for row in gap_summary:
        hv = row.get("hv_oracle_gain_mean", "")
        cap = row.get("hv_captured_oracle_ratio_mean", "")
        print(f"[oracle-gap] {row['problem']} hv_oracle_gain={hv} "
              f"hv_capture={cap}")


if __name__ == "__main__":
    main()
