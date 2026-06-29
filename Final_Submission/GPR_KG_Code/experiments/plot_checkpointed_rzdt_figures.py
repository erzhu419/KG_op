"""Regenerate RZDT figures from the latest checkpointed full runs.

This script reads the two theory-aligned run directories:

* server311_checkpointed_full_20260519      (GPR-KG)
* server311_nv_full_20260520               (GPR-KG-nV)

and overwrites the manuscript figure files under GPR_KG_Code/results/figures.
The diagnostics use only quantities saved by run_rzdt_checkpointed.py:
final sample/recommendation sets, HV histories, iteration logs, and selected
candidate posterior summaries.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpr_kg import RZDT1, RZDT2, RZDT5_RR, compute_hypervolume_2d  # noqa: E402


PROBLEMS = ["RZDT1", "RZDT2", "RZDT5_RR"]
METHODS = {
    "GPR-KG": {
        "root": PROJECT / "server311_checkpointed_full_20260519",
        "color": "#d62728",
        "marker": "o",
    },
    "GPR-KG-nV": {
        "root": PROJECT / "server311_nv_full_20260520",
        "color": "#1f77b4",
        "marker": "s",
    },
}
LABELS = ["(a) RZDT1", "(b) RZDT2", r"(c) RZDT5\_RR"]
FIXED_PDF_DATE = datetime(2026, 5, 21, tzinfo=timezone.utc)


def make_problem(name: str):
    kwargs = dict(d=5, L=100, sigma=0.04, heteroscedastic=True, alpha=0.05)
    if name == "RZDT1":
        problem = RZDT1(**kwargs)
    elif name == "RZDT2":
        problem = RZDT2(**kwargs)
    elif name == "RZDT5_RR":
        problem = RZDT5_RR(**kwargs)
    else:
        raise ValueError(name)
    problem.tau = 0.0
    return problem


def run_dirs(root: Path, problem: str, method: str) -> list[Path]:
    return sorted((root / problem).glob(f"{method}_rep*"))


def load_result(run_dir: Path) -> dict:
    with (run_dir / "result.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def load_results(root: Path, problem: str, method: str) -> list[dict]:
    results = []
    for rd in run_dirs(root, problem, method):
        result_file = rd / "result.json"
        if result_file.exists():
            results.append(load_result(rd))
    return results


def save_fig(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        kwargs = {"bbox_inches": "tight"}
        if ext == "pdf":
            kwargs["metadata"] = {
                "Creator": "plot_checkpointed_rzdt_figures.py",
                "CreationDate": FIXED_PDF_DATE,
                "ModDate": FIXED_PDF_DATE,
            }
        if ext == "png":
            kwargs["dpi"] = 300
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, **kwargs)
        print(f"saved {path}")
    plt.close(fig)


def mean_se_on_grid(series: Iterable[tuple[np.ndarray, np.ndarray]]):
    series = [(np.asarray(x, float), np.asarray(y, float)) for x, y in series if len(x)]
    if not series:
        return None
    grid = series[0][0]
    y = np.vstack([np.interp(grid, x, v) for x, v in series])
    mean = y.mean(axis=0)
    se = y.std(axis=0, ddof=1) / math.sqrt(y.shape[0]) if y.shape[0] > 1 else np.zeros_like(mean)
    return grid, mean, se


def pareto_segments(problem):
    lo, hi = problem.int_bounds()
    feasible_x1 = [
        x1 for x1 in range(lo[0], hi[0] + 1)
        if problem.is_truly_feasible(tuple([x1] + [0] * (problem.d - 1)))
    ]
    if not feasible_x1:
        return []
    segments = [[feasible_x1[0]]]
    for x1 in feasible_x1[1:]:
        if x1 == segments[-1][-1] + 1:
            segments[-1].append(x1)
        else:
            segments.append([x1])
    out = []
    for seg in segments:
        vals = np.array([problem.true_objectives(tuple([x1] + [0] * (problem.d - 1)))[:2] for x1 in seg])
        out.append(vals)
    return out


def load_recommended_points(root: Path, problem: str, key: str = "generic_k1.25") -> np.ndarray:
    pts = []
    details = root / "postprocessing_compact" / "details"
    for detail_file in sorted(details.glob(f"{problem}_rep*.json")):
        with detail_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        rec = data["recommendations"][key]
        pts.extend(rec.get("pareto_objectives_true", []))
    return np.asarray(pts, dtype=float) if pts else np.empty((0, 2))


def figure_recommendations(out_dir: Path):
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.2), sharex=False, sharey=False)

    for col, problem_name in enumerate(PROBLEMS):
        problem = make_problem(problem_name)
        tpos = problem.true_pareto_front()
        for row, (method, info) in enumerate(METHODS.items()):
            ax = axes[row, col]
            for seg in pareto_segments(problem):
                ax.plot(seg[:, 0], seg[:, 1], color="black", lw=1.1, alpha=0.75)
            if len(tpos):
                ax.scatter(tpos[:, 0], tpos[:, 1], s=12, color="black", alpha=0.75)

            results = load_results(info["root"], problem_name, method)
            sample = []
            for result in results:
                sample.extend(result.get("pareto_objectives_true", []))
            sample = np.asarray(sample, dtype=float) if sample else np.empty((0, 2))
            recommended = load_recommended_points(info["root"], problem_name)

            if len(sample):
                ax.scatter(
                    sample[:, 0],
                    sample[:, 1],
                    s=24,
                    marker="o",
                    facecolors="none",
                    edgecolors=info["color"],
                    linewidths=0.9,
                    alpha=0.75,
                    label="sample",
                )
            if len(recommended):
                ax.scatter(
                    recommended[:, 0],
                    recommended[:, 1],
                    s=30,
                    marker="^",
                    color="#d97706",
                    alpha=0.65,
                    label="recommended",
                )
            ax.set_title(f"{LABELS[col]}  {method}" if row == 0 else method, fontsize=10)
            ax.set_xlabel(r"$f^1$")
            if col == 0:
                ax.set_ylabel(r"$f^2$")
            ax.grid(True, color="#dddddd", linewidth=0.6, alpha=0.8)
            if problem_name == "RZDT5_RR":
                ax.set_xlim(-0.02, 0.52)
            else:
                ax.set_xlim(-0.03, 1.03)
            ax.set_ylim(-0.04, max(1.05, float(np.nanmax(tpos[:, 1])) + 0.08 if len(tpos) else 1.05))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, out_dir, "fig_rzdt_gprkg_recommendation")


def figure_sample_ablation(out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for idx, problem_name in enumerate(PROBLEMS):
        ax = axes[idx]
        problem = make_problem(problem_name)
        tpos = problem.true_pareto_front()
        for seg in pareto_segments(problem):
            ax.plot(seg[:, 0], seg[:, 1], color="black", lw=1.1, alpha=0.75)
        if len(tpos):
            ax.scatter(tpos[:, 0], tpos[:, 1], s=13, color="black", alpha=0.75, label="TPOS" if idx == 0 else None)

        for method, info in METHODS.items():
            pts = []
            for result in load_results(info["root"], problem_name, method):
                pts.extend(result.get("pareto_objectives_true", []))
            pts = np.asarray(pts, dtype=float) if pts else np.empty((0, 2))
            if len(pts):
                ax.scatter(
                    pts[:, 0],
                    pts[:, 1],
                    s=27,
                    marker=info["marker"],
                    color=info["color"],
                    alpha=0.55,
                    edgecolors="none",
                    label=method if idx == 0 else None,
                )

        ax.set_title(LABELS[idx], fontsize=10)
        ax.set_xlabel(r"$f^1$")
        if idx == 0:
            ax.set_ylabel(r"$f^2$")
        ax.grid(alpha=0.3)
        if problem_name == "RZDT5_RR":
            ax.set_xlim(-0.02, 0.52)
        else:
            ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.04, max(1.05, float(np.nanmax(tpos[:, 1])) + 0.08 if len(tpos) else 1.05))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=8, bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout(rect=[0, 0, 0.88, 1])
    save_fig(fig, out_dir, "fig5_vepm_ablation")


def figure_hv(out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for idx, problem_name in enumerate(PROBLEMS):
        ax = axes[idx]
        problem = make_problem(problem_name)
        hv_star = compute_hypervolume_2d(problem.true_pareto_front(), problem.ref_point)
        for method, info in METHODS.items():
            results = load_results(info["root"], problem_name, method)
            series = []
            for result in results:
                hist = result.get("hv_history", [])
                if hist:
                    series.append((np.array([h[0] for h in hist]), np.array([h[1] for h in hist])))
            stats = mean_se_on_grid(series)
            if stats is None:
                continue
            grid, mean, se = stats
            ax.plot(grid, mean, color=info["color"], marker=info["marker"], ms=4.2, lw=1.3, label=method)
            ax.fill_between(grid, mean - se, mean + se, color=info["color"], alpha=0.14)
        ax.axhline(hv_star, color="black", ls="--", lw=0.9, label="HV$^*$" if idx == 0 else None)
        ax.set_title(LABELS[idx], fontsize=10)
        ax.set_xlabel("# simulations")
        if idx == 0:
            ax.set_ylabel("HV")
        ax.grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=8, bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout(rect=[0, 0, 0.88, 1])
    save_fig(fig, out_dir, "fig2_hv_convergence")


def figure_selected_rmse(out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for idx, problem_name in enumerate(PROBLEMS):
        ax = axes[idx]
        problem = make_problem(problem_name)
        for method, info in METHODS.items():
            series = []
            for result in load_results(info["root"], problem_name, method):
                xs, ys = [], []
                for log in result.get("iteration_log", []):
                    mu = np.asarray(log.get("mu_before_update", []), dtype=float)
                    x = tuple(int(v) for v in log.get("x_selected", []))
                    if len(mu) != 3 or len(x) != problem.d:
                        continue
                    true = np.asarray(problem.true_objectives(x), dtype=float)
                    ys.append(float(np.sqrt(np.mean((mu - true) ** 2))))
                    xs.append(int(log["stage"]))
                series.append((np.asarray(xs), np.asarray(ys)))
            stats = mean_se_on_grid(series)
            if stats is None:
                continue
            grid, mean, se = stats
            ax.plot(grid, mean, color=info["color"], marker=info["marker"], ms=3.5, lw=1.2, label=method)
            ax.fill_between(grid, mean - se, mean + se, color=info["color"], alpha=0.14)
        ax.set_title(LABELS[idx], fontsize=10)
        ax.set_xlabel("# simulations")
        if idx == 0:
            ax.set_ylabel(r"selected-point RMSE")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=8, bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout(rect=[0, 0, 0.88, 1])
    save_fig(fig, out_dir, "fig3_rmse")


def figure_infeasible(out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for idx, problem_name in enumerate(PROBLEMS):
        ax = axes[idx]
        problem = make_problem(problem_name)
        for method, info in METHODS.items():
            series = []
            for result in load_results(info["root"], problem_name, method):
                xs, ys = [], []
                for log in result.get("iteration_log", []):
                    pf = log.get("pareto_front")
                    if not pf:
                        continue
                    n_bad = 0
                    for x in pf:
                        xt = tuple(int(v) for v in x)
                        n_bad += int(not problem.is_truly_feasible(xt))
                    xs.append(int(log["stage"]))
                    ys.append(float(n_bad))
                series.append((np.asarray(xs), np.asarray(ys)))
            stats = mean_se_on_grid(series)
            if stats is None:
                continue
            grid, mean, se = stats
            ax.plot(grid, mean, color=info["color"], marker=info["marker"], ms=4.2, lw=1.3, label=method)
            ax.fill_between(grid, mean - se, mean + se, color=info["color"], alpha=0.14)
        ax.set_title(LABELS[idx], fontsize=10)
        ax.set_xlabel("# simulations")
        if idx == 0:
            ax.set_ylabel("# infeasible in posterior PF")
        ax.set_ylim(bottom=-0.1)
        ax.grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=8, bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout(rect=[0, 0, 0.88, 1])
    save_fig(fig, out_dir, "fig4_infeas")


def figure_variance_error(out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for idx, problem_name in enumerate(PROBLEMS):
        ax = axes[idx]
        problem = make_problem(problem_name)
        for method, info in METHODS.items():
            series = []
            for result in load_results(info["root"], problem_name, method):
                xs, ys = [], []
                for log in result.get("iteration_log", []):
                    sigma2 = log.get("sigma2_before_update")
                    x = tuple(int(v) for v in log.get("x_selected", []))
                    if not sigma2 or len(x) != problem.d:
                        continue
                    est = max(float(sigma2[2]), 1e-12)
                    true = max(float(problem.true_sigma(x)[2] ** 2), 1e-12)
                    ys.append(abs(math.log(est / true)))
                    xs.append(int(log["stage"]))
                series.append((np.asarray(xs), np.asarray(ys)))
            stats = mean_se_on_grid(series)
            if stats is None:
                continue
            grid, mean, se = stats
            ax.plot(grid, mean, color=info["color"], marker=info["marker"], ms=3.5, lw=1.2, label=method)
            ax.fill_between(grid, mean - se, mean + se, color=info["color"], alpha=0.14)
        ax.set_title(LABELS[idx], fontsize=10)
        ax.set_xlabel("# simulations")
        if idx == 0:
            ax.set_ylabel(r"$|\log(\hat\sigma_3^2/\sigma_3^2)|$")
        ax.grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=8, bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout(rect=[0, 0, 0.88, 1])
    save_fig(fig, out_dir, "fig5a_variance")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=ROOT / "results" / "figures")
    args = parser.parse_args()

    for method, info in METHODS.items():
        if not info["root"].exists():
            raise FileNotFoundError(f"Missing {method} result root: {info['root']}")

    figure_recommendations(args.out_dir)
    figure_sample_ablation(args.out_dir)
    figure_hv(args.out_dir)
    figure_selected_rmse(args.out_dir)
    figure_infeasible(args.out_dir)
    figure_variance_error(args.out_dir)


if __name__ == "__main__":
    main()
