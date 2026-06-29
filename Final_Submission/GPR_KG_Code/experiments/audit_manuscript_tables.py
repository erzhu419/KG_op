"""Audit manuscript tables against saved numerical-result files.

The script is intentionally read-only with respect to experiment results.  It
recomputes the aggregate quantities used by the manuscript tables from the
saved JSON/CSV logs and writes a compact audit trail under
``GPR_KG_Code/results/audit``.

Usage:
    python GPR_KG_Code/experiments/audit_manuscript_tables.py
"""

from __future__ import annotations

import csv
import glob
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from gpr_kg import RZDT1, RZDT2, RZDT5_RR  # noqa: E402


OUT_DIR = ROOT / "results" / "audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROBLEMS = {
    "RZDT1": {
        "factory": lambda: RZDT1(d=5, sigma=0.04, heteroscedastic=True, alpha=0.05),
        "hv_star": 1.635,
        "tpos_star": 21,
        "baseline_dir": ROOT / "results" / "d5_v2" / "RZDT1",
    },
    "RZDT2": {
        "factory": lambda: RZDT2(d=5, sigma=0.04, heteroscedastic=True, alpha=0.05),
        "hv_star": 1.403,
        "tpos_star": 34,
        "baseline_dir": ROOT / "results" / "d5_v2" / "RZDT2",
    },
    "RZDT5_RR": {
        "factory": lambda: RZDT5_RR(d=5, sigma=0.04, heteroscedastic=True, alpha=0.05),
        "hv_star": 2.183,
        "tpos_star": 46,
        "baseline_dir": ROOT / "results" / "rzdt5rr" / "RZDT5_RR",
    },
}

BASELINE_METHODS = ["cEHVI", "cParEGO", "NSGA-II-K", "NSGA-II-D", "RS"]
SAFE = {m: m.replace("-", "_") for m in BASELINE_METHODS}

CHECKPOINTED = {
    "GPR-KG": REPO / "server311_checkpointed_full_20260519",
    "GPR-KG-nV": REPO / "server311_nv_full_20260520",
}

MANUSCRIPT_MAIN = {
    ("RZDT1", "GPR-KG"): dict(hv=1.332, igd=0.201, cvr=0.325, nd=3.3, infeas=1.2, tpos=1.8, time=382.5),
    ("RZDT1", "GPR-KG-nV"): dict(hv=1.268, igd=0.225, cvr=0.340, nd=3.6, infeas=1.5, tpos=1.7, time=528.6),
    ("RZDT1", "cEHVI"): dict(hv=0.426, igd=0.838, cvr=0.519, nd=5.1, infeas=2.6, tpos=0.0, time=73.5),
    ("RZDT1", "cParEGO"): dict(hv=0.225, igd=1.003, cvr=0.380, nd=5.1, infeas=2.0, tpos=0.0, time=1.1),
    ("RZDT1", "NSGA-II-K"): dict(hv=0.282, igd=0.879, cvr=0.400, nd=5.2, infeas=2.2, tpos=0.0, time=9.0),
    ("RZDT1", "NSGA-II-D"): dict(hv=1.644, igd=0.081, cvr=0.525, nd=10.2, infeas=5.3, tpos=0.0, time=3.3),
    ("RZDT1", "RS"): dict(hv=0.345, igd=0.867, cvr=0.493, nd=5.1, infeas=2.6, tpos=0.0, time=0.0),
    ("RZDT2", "GPR-KG"): dict(hv=1.289, igd=0.115, cvr=0.100, nd=2.6, infeas=0.3, tpos=2.2, time=376.3),
    ("RZDT2", "GPR-KG-nV"): dict(hv=1.287, igd=0.104, cvr=0.075, nd=3.2, infeas=0.3, tpos=2.8, time=481.0),
    ("RZDT2", "cEHVI"): dict(hv=0.073, igd=1.462, cvr=0.302, nd=3.8, infeas=1.2, tpos=0.0, time=90.4),
    ("RZDT2", "cParEGO"): dict(hv=0.000, igd=1.761, cvr=0.290, nd=4.1, infeas=1.2, tpos=0.0, time=4.8),
    ("RZDT2", "NSGA-II-K"): dict(hv=0.036, igd=1.770, cvr=0.183, nd=3.8, infeas=0.8, tpos=0.0, time=29.8),
    ("RZDT2", "NSGA-II-D"): dict(hv=0.896, igd=0.410, cvr=0.157, nd=5.4, infeas=1.0, tpos=0.0, time=5.0),
    ("RZDT2", "RS"): dict(hv=0.011, igd=1.558, cvr=0.195, nd=3.5, infeas=0.8, tpos=0.0, time=0.0),
    ("RZDT5_RR", "GPR-KG"): dict(hv=2.104, igd=0.054, cvr=0.008, nd=12.8, infeas=0.1, tpos=0.1, time=840.6),
    ("RZDT5_RR", "GPR-KG-nV"): dict(hv=2.094, igd=0.065, cvr=0.014, nd=12.7, infeas=0.2, tpos=0.1, time=728.7),
    ("RZDT5_RR", "cEHVI"): dict(hv=2.087, igd=0.061, cvr=0.022, nd=11.2, infeas=0.2, tpos=0.0, time=997.4),
    ("RZDT5_RR", "cParEGO"): dict(hv=2.080, igd=0.068, cvr=0.013, nd=12.7, infeas=0.2, tpos=0.0, time=2.9),
    ("RZDT5_RR", "NSGA-II-K"): dict(hv=2.077, igd=0.066, cvr=0.019, nd=11.3, infeas=0.2, tpos=0.0, time=57.5),
    ("RZDT5_RR", "NSGA-II-D"): dict(hv=2.156, igd=0.040, cvr=0.005, nd=14.3, infeas=0.1, tpos=0.0, time=10.7),
    ("RZDT5_RR", "RS"): dict(hv=2.087, igd=0.064, cvr=0.033, nd=10.8, infeas=0.3, tpos=0.0, time=0.2),
}

MANUSCRIPT_FINAL = {
    ("RZDT1", "GPR-KG", "recommended"): dict(hv=1.391, igd=0.150, cvr=0.196, nd=6.0, tpos=2.3),
    ("RZDT1", "GPR-KG-nV", "recommended"): dict(hv=1.326, igd=0.219, cvr=0.254, nd=7.1, tpos=2.1),
    ("RZDT2", "GPR-KG", "recommended"): dict(hv=1.272, igd=0.106, cvr=0.020, nd=5.2, tpos=3.4),
    ("RZDT2", "GPR-KG-nV", "recommended"): dict(hv=1.243, igd=0.149, cvr=0.080, nd=5.4, tpos=3.5),
    ("RZDT5_RR", "GPR-KG", "recommended"): dict(hv=2.113, igd=0.050, cvr=0.009, nd=20.2, tpos=0.2),
    ("RZDT5_RR", "GPR-KG-nV", "recommended"): dict(hv=2.102, igd=0.060, cvr=0.030, nd=17.9, tpos=0.1),
}

MANUSCRIPT_ABLATION = {
    "default": dict(hv=1.353, igd=0.242, cvr=0.000, time=195.2),
    "vepm_off": dict(hv=1.283, igd=0.262, cvr=0.000, time=129.2),
    "cand_30": dict(hv=1.258, igd=0.289, cvr=0.000, time=44.4),
    "cand_150": dict(hv=1.337, igd=0.237, cvr=0.000, time=787.9),
}


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def stat(vals: list[float]) -> dict[str, float]:
    vals = [float(v) for v in vals if v is not None and not math.isnan(float(v))]
    n = len(vals)
    if n == 0:
        return {"mean": math.nan, "se": math.nan, "n": 0}
    se = stdev(vals) / math.sqrt(n) if n > 1 else 0.0
    return {"mean": mean(vals), "se": se, "n": n}


def make_problem(problem: str):
    prob = PROBLEMS[problem]["factory"]()
    prob.tau = 0.0
    return prob


def tpos_set(prob) -> set[tuple[int, ...]]:
    lo, hi = prob.int_bounds()
    sols = set()
    for x1 in range(int(lo[0]), int(hi[0]) + 1):
        x = tuple([x1] + [0] * (prob.d - 1))
        if prob.is_truly_feasible(x):
            sols.add(x)
    return sols


def count_infeas_tpos(problem: str, sols: list[list[int]]) -> tuple[int, int]:
    prob = make_problem(problem)
    tpos = tpos_set(prob)
    infeas = 0
    hits = 0
    for x in sols:
        xt = tuple(int(v) for v in x)
        infeas += int(not prob.is_truly_feasible(xt))
        hits += int(xt in tpos)
    return infeas, hits


def load_checkpointed_rep_files(root: Path, problem: str, method: str) -> list[Path]:
    return sorted((root / problem).glob(f"{method}_rep*/result.json"))


def checkpointed_metrics(problem: str, method: str) -> dict[str, dict[str, float]]:
    root = CHECKPOINTED[method]
    files = load_checkpointed_rep_files(root, problem, method)
    reps = []
    for path in files:
        data = load_json(path)
        infeas, tpos = count_infeas_tpos(problem, data.get("pareto_solutions", []))
        reps.append(
            {
                "hv": data["hv_final"],
                "igd": data["igd_final"],
                "cvr": data["cvr_final"],
                "nd": data["n_pareto_solutions"],
                "infeas": infeas,
                "tpos": tpos,
                "time": data["total_time_sec"],
            }
        )
    return {k: stat([r[k] for r in reps]) for k in ["hv", "igd", "cvr", "nd", "infeas", "tpos", "time"]}


def baseline_metrics(problem: str, method: str) -> dict[str, dict[str, float]]:
    base = PROBLEMS[problem]["baseline_dir"]
    files = sorted(base.glob(f"{SAFE[method]}_rep*.json"))
    reps = []
    for path in files:
        data = load_json(path)
        sols = data.get("pareto_solutions", [])
        infeas, tpos = count_infeas_tpos(problem, sols)
        reps.append(
            {
                "hv": data["hv_final"],
                "igd": data["igd_final"],
                "cvr": data["cvr_final"],
                "nd": data.get("n_pareto_solutions", len(sols)),
                "infeas": infeas,
                "tpos": tpos,
                "time": data.get("wall_time_sec", data.get("total_time_sec", 0.0)),
            }
        )
    return {k: stat([r[k] for r in reps]) for k in ["hv", "igd", "cvr", "nd", "infeas", "tpos", "time"]}


def final_recommendation_metrics(problem: str, method: str) -> dict[str, dict[str, float]]:
    root = CHECKPOINTED[method] / "postprocessing_compact" / "recommendation_summary.json"
    data = load_json(root)
    entry = data[problem]["generic"]["1.25"]
    mapping = {
        "hv": "hv_final",
        "igd": "igd_final",
        "cvr": "cvr_final",
        "nd": "n_pareto_solutions",
        "tpos": "tpos_hits",
    }
    return {k: entry[v] for k, v in mapping.items()}


def add_compare_rows(rows: list[dict[str, Any]], table: str, problem: str, method: str,
                     computed: dict[str, Any], manuscript: dict[str, float],
                     metrics: list[str]) -> None:
    for metric in metrics:
        comp = computed[metric]["mean"] if "mean" in computed[metric] else computed[metric]
        comp_se = computed[metric].get("se") if isinstance(computed[metric], dict) else None
        man = manuscript[metric]
        rounded = round(comp, 3 if metric not in {"nd", "infeas", "tpos", "time"} else 1)
        tol = 0.0005 if metric not in {"nd", "infeas", "tpos", "time"} else 0.05
        rows.append(
            {
                "table": table,
                "problem": problem,
                "method": method,
                "metric": metric,
                "manuscript": man,
                "computed": comp,
                "computed_se": comp_se,
                "rounded_for_table": rounded,
                "abs_delta_after_rounding": abs(float(man) - float(rounded)),
                "status": "OK" if abs(float(man) - float(rounded)) <= tol else "CHECK",
            }
        )


def audit_main_tables(rows: list[dict[str, Any]]) -> dict[str, Any]:
    source = {}
    for problem in PROBLEMS:
        source[problem] = {}
        for method in ["GPR-KG", "GPR-KG-nV"]:
            computed = checkpointed_metrics(problem, method)
            source[problem][method] = computed
            add_compare_rows(rows, "main_sample_tables", problem, method, computed,
                             MANUSCRIPT_MAIN[(problem, method)],
                             ["hv", "igd", "cvr", "nd", "infeas", "tpos", "time"])
        for method in BASELINE_METHODS:
            computed = baseline_metrics(problem, method)
            source[problem][method] = computed
            add_compare_rows(rows, "main_sample_tables", problem, method, computed,
                             MANUSCRIPT_MAIN[(problem, method)],
                             ["hv", "igd", "cvr", "nd", "infeas", "tpos", "time"])
    return source


def audit_final_recommendation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    source = {}
    for problem in PROBLEMS:
        source[problem] = {}
        for method in ["GPR-KG", "GPR-KG-nV"]:
            computed = final_recommendation_metrics(problem, method)
            source[problem][method] = computed
            add_compare_rows(rows, "final_recommendation", problem, method, computed,
                             MANUSCRIPT_FINAL[(problem, method, "recommended")],
                             ["hv", "igd", "cvr", "nd", "tpos"])
    return source


def audit_ablation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    source = load_json(ROOT / "results" / "sec66" / "summary.json")
    mapping = {"hv": "hv_mean", "igd": "igd_mean", "cvr": "cvr_mean", "time": "time_mean"}
    for variant, man in MANUSCRIPT_ABLATION.items():
        comp = {k: {"mean": source[variant][v], "se": source[variant].get(v.replace("_mean", "_se"))}
                for k, v in mapping.items()}
        add_compare_rows(rows, "ablation", "RZDT1", variant, comp, man, ["hv", "igd", "cvr", "time"])
    return source


def audit_case_study() -> dict[str, Any]:
    return {
        "hetero": load_json(ROOT / "results" / "ingolstadt21" / "hetero_test.json"),
        "case_summary": load_json(ROOT / "results" / "ingolstadt21" / "case_study_aggregate_summary.json"),
        "vepm_convergence": load_json(ROOT / "results" / "ingolstadt21" / "vepm_convergence_summary.json"),
    }


def write_csv_rows(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows: list[dict[str, Any]] = []
    audit = {
        "sources": {
            "GPR-KG": str(CHECKPOINTED["GPR-KG"]),
            "GPR-KG-nV": str(CHECKPOINTED["GPR-KG-nV"]),
            "baselines_RZDT1_RZDT2": str(ROOT / "results" / "d5_v2"),
            "baselines_RZDT5_RR": str(ROOT / "results" / "rzdt5rr"),
            "ablation": str(ROOT / "results" / "sec66" / "summary.json"),
            "case_study": str(ROOT / "results" / "ingolstadt21"),
        },
        "main_tables": audit_main_tables(rows),
        "final_recommendation": audit_final_recommendation(rows),
        "ablation": audit_ablation(rows),
        "case_study_sources": audit_case_study(),
    }

    csv_path = OUT_DIR / "manuscript_table_audit.csv"
    json_path = OUT_DIR / "manuscript_table_audit.json"
    write_csv_rows(rows, csv_path)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    n_check = sum(1 for r in rows if r["status"] == "CHECK")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Audited {len(rows)} manuscript values; CHECK rows: {n_check}")
    if n_check:
        print("Rows requiring attention:")
        for r in rows:
            if r["status"] == "CHECK":
                print(r)


if __name__ == "__main__":
    main()
