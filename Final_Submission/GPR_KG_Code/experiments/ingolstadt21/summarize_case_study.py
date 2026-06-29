"""Aggregate RESCO ingolstadt21 case-study logs for manuscript tables.

The script is intentionally post-hoc: it does not call SUMO and only reads the
completed optimization logs under ``results/ingolstadt21``.  It is meant to
make the manuscript's case-study diagnostics reproducible from saved JSON
artifacts rather than from hand calculations.

Outputs
-------
results/ingolstadt21/case_study_aggregate_summary.json
results/ingolstadt21/case_study_aggregate_table.tex

Usage
-----
python -m experiments.ingolstadt21.summarize_case_study
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / "results" / "ingolstadt21"

REF_POINT = (1.5, 1.5)


@dataclass(frozen=True)
class RunRecord:
    method: str
    partition: str
    seed: int
    path: Path
    hv_final: float
    hv_peak: float
    pareto_final_size: int
    n_feasible_observed: int | None
    best_f1_feasible: float | None
    best_f2_feasible: float | None
    best_f3_observed: float | None


def _canonical_run_id(path: Path, summary: dict[str, Any]) -> tuple[str, str, int]:
    """Return (method, partition, seed), including legacy seed-100 folders."""
    name = path.parent.name
    if name == "GPR_KG_run":
        return "GPR-KG", "binary_bin", 100
    if name == "GPR_KG_nV_run":
        return "GPR-KG-nV", "binary_bin", 100

    seed = int(summary.get("seed", -1))
    method = summary.get("method", "")
    partition = summary.get("partition_method", "")

    m = re.match(r"(?P<safe>.+)_(?P<partition>binary_bin|aggregate|medoid_K)_seed(?P<seed>\d+)$", name)
    if m:
        safe = m.group("safe")
        method = safe.replace("_", "-")
        if method == "GPR-KG-nV":
            method = "GPR-KG-nV"
        elif method == "GPR-KG":
            method = "GPR-KG"
        partition = m.group("partition")
        seed = int(m.group("seed"))

    return method, partition, seed


def _is_pareto(points: list[tuple[float, float]]) -> list[bool]:
    keep = [True] * len(points)
    for i, p in enumerate(points):
        if not keep[i]:
            continue
        for j, q in enumerate(points):
            if i == j:
                continue
            if q[0] <= p[0] and q[1] <= p[1] and (q[0] < p[0] or q[1] < p[1]):
                keep[i] = False
                break
    return keep


def _observed_y_from_snapshots(run_dir: Path) -> tuple[list[list[float]], int | None]:
    snap = run_dir / "snapshots.jsonl"
    if not snap.exists():
        return [], None
    rows: list[list[float]] = []
    last_pareto_size: int | None = None
    with snap.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("pareto_set_size") is not None:
                last_pareto_size = int(rec["pareto_set_size"])
            if "Y_observed" in rec:
                y = rec["Y_observed"]
                if isinstance(y, list) and len(y) == 3 and not isinstance(y[0], list):
                    rows.append([float(v) for v in y])
                elif isinstance(y, list):
                    rows.extend([[float(v) for v in yy] for yy in y])
    return rows, last_pareto_size


def _pareto_size_from_y(y_obs: list[list[float]], tau: float) -> int:
    pts = [(float(y[0]), float(y[1])) for y in y_obs if float(y[2]) <= tau]
    if not pts:
        return 0
    return sum(_is_pareto(pts))


def _read_record(summary_path: Path) -> RunRecord | None:
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    method, partition, seed = _canonical_run_id(summary_path, summary)
    if method not in {"GPR-KG", "GPR-KG-nV"}:
        return None
    if partition not in {"binary_bin", "aggregate", "medoid_K"}:
        return None

    hv_hist = summary.get("hv_history", [])
    hv_vals = [float(v[1]) for v in hv_hist]
    hv_final = hv_vals[-1] if hv_vals else 0.0
    hv_peak = max(hv_vals) if hv_vals else 0.0

    tau = float(summary.get("tau", 1.0))
    y_obs, last_pareto_size = _observed_y_from_snapshots(summary_path.parent)
    feasible = [y for y in y_obs if float(y[2]) <= tau]

    final_true = summary.get("final_true_objs", [])
    final_pareto = summary.get("final_pareto_set", [])
    if "final_pareto_set" in summary:
        pareto_final_size = len(final_pareto)
    elif last_pareto_size is not None and not final_true:
        pareto_final_size = last_pareto_size
    elif final_true:
        pareto_final_size = len(final_true)
    else:
        pareto_final_size = _pareto_size_from_y(y_obs, tau)

    return RunRecord(
        method=method,
        partition=partition,
        seed=seed,
        path=summary_path,
        hv_final=hv_final,
        hv_peak=hv_peak,
        pareto_final_size=pareto_final_size,
        n_feasible_observed=len(feasible) if y_obs else None,
        best_f1_feasible=min((float(y[0]) for y in feasible), default=None),
        best_f2_feasible=min((float(y[1]) for y in feasible), default=None),
        best_f3_observed=min((float(y[2]) for y in y_obs), default=None),
    )


def _se(vals: list[float]) -> float:
    if len(vals) <= 1:
        return 0.0
    return stdev(vals) / math.sqrt(len(vals))


def _summarize(records: list[RunRecord]) -> dict[str, Any]:
    groups: dict[str, list[RunRecord]] = {}
    for rec in records:
        key = f"{rec.method}|{rec.partition}"
        groups.setdefault(key, []).append(rec)

    pooled_key = "GPR-KG-nV|binary_bin"
    pooled = groups.get(pooled_key, [])
    pooled_peak = [r.hv_peak for r in pooled]

    out: dict[str, Any] = {}
    for key, rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: r.seed)
        hv_peak = [r.hv_peak for r in rows]
        hv_final = [r.hv_final for r in rows]
        pareto = [float(r.pareto_final_size) for r in rows]
        feasible_counts = [
            float(r.n_feasible_observed)
            for r in rows
            if r.n_feasible_observed is not None
        ]

        delta_vs_pooled = None
        z_vs_pooled = None
        if pooled_peak and key != pooled_key:
            delta_vs_pooled = mean(hv_peak) - mean(pooled_peak)
            denom = math.sqrt(_se(hv_peak) ** 2 + _se(pooled_peak) ** 2)
            z_vs_pooled = delta_vs_pooled / denom if denom > 0 else None

        out[key] = {
            "n": len(rows),
            "seeds": [r.seed for r in rows],
            "hv_peak_mean": mean(hv_peak),
            "hv_peak_se": _se(hv_peak),
            "hv_final_mean": mean(hv_final),
            "hv_final_se": _se(hv_final),
            "pareto_final_mean": mean(pareto),
            "pareto_final_se": _se(pareto),
            "n_feasible_observed_mean": mean(feasible_counts) if feasible_counts else None,
            "delta_hv_peak_vs_pooled": delta_vs_pooled,
            "z_hv_peak_vs_pooled": z_vs_pooled,
            "best_f1_feasible_min": min(
                (r.best_f1_feasible for r in rows if r.best_f1_feasible is not None),
                default=None,
            ),
            "best_f2_feasible_min": min(
                (r.best_f2_feasible for r in rows if r.best_f2_feasible is not None),
                default=None,
            ),
            "best_f3_observed_min": min(
                (r.best_f3_observed for r in rows if r.best_f3_observed is not None),
                default=None,
            ),
        }
    return out


def _format_pm(mean_value: float, se_value: float) -> str:
    return f"{mean_value:.4f} $\\pm$ {se_value:.4f}"


def _write_latex(summary: dict[str, Any], out_path: Path) -> None:
    label = {
        "GPR-KG|binary_bin": "$\\phi_{\\mathrm{Zheng}}$ VEPM",
        "GPR-KG|aggregate": "$\\phi_{\\mathrm{agg}}$ VEPM",
        "GPR-KG|medoid_K": "Medoid-$K$ VEPM",
        "GPR-KG-nV|binary_bin": "Pooled (GPR-KG-nV)",
    }
    order = [
        "GPR-KG|binary_bin",
        "GPR-KG|aggregate",
        "GPR-KG|medoid_K",
        "GPR-KG-nV|binary_bin",
    ]
    lines = [
        "% Auto-generated by experiments.ingolstadt21.summarize_case_study.py",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Variance estimator & HV-peak & HV-final & \\# final Pareto \\\\",
        "\\midrule",
    ]
    for key in order:
        if key not in summary:
            continue
        s = summary[key]
        lines.append(
            f"{label[key]} & "
            f"{_format_pm(s['hv_peak_mean'], s['hv_peak_se'])} & "
            f"{_format_pm(s['hv_final_mean'], s['hv_final_se'])} & "
            f"{s['pareto_final_mean']:.1f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    records: list[RunRecord] = []
    for path in RESULTS_DIR.glob("*/summary.json"):
        rec = _read_record(path)
        if rec is not None:
            records.append(rec)

    if not records:
        raise SystemExit(f"No ingolstadt21 summary files found under {RESULTS_DIR}")

    summary = _summarize(records)
    payload = {
        "source": "experiments.ingolstadt21.summarize_case_study",
        "results_dir": str(RESULTS_DIR),
        "n_runs": len(records),
        "groups": summary,
    }

    out_json = RESULTS_DIR / "case_study_aggregate_summary.json"
    out_tex = RESULTS_DIR / "case_study_aggregate_table.tex"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_latex(summary, out_tex)

    print(f"Wrote {out_json}")
    print(f"Wrote {out_tex}")
    print()
    for key, s in summary.items():
        z = s["z_hv_peak_vs_pooled"]
        z_text = "NA" if z is None else f"{z:.2f}"
        print(
            f"{key:24s} n={s['n']:2d} "
            f"HVpeak={s['hv_peak_mean']:.4f}+/-{s['hv_peak_se']:.4f} "
            f"HVfinal={s['hv_final_mean']:.4f}+/-{s['hv_final_se']:.4f} "
            f"z_vs_pooled={z_text}"
        )


if __name__ == "__main__":
    main()
