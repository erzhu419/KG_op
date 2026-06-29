"""Plot checkpointed GPR-KG sample and final-recommendation outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpr_kg import RZDT1, RZDT2, RZDT5_RR  # noqa: E402


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


def load_points(results_dir: Path, problem: str, output_key: str):
    pts = []
    if output_key == "sample":
        for result_file in sorted((results_dir / problem).glob("GPR-KG_rep*/result.json")):
            with result_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            pts.extend(data.get("pareto_objectives_true", []))
    else:
        details_dir = results_dir / "postprocessing_compact" / "details"
        for detail_file in sorted(details_dir.glob(f"{problem}_rep*.json")):
            with detail_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            rec = data["recommendations"][output_key]
            pts.extend(rec.get("pareto_objectives_true", []))
    return np.asarray(pts, dtype=float) if pts else np.empty((0, 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument("--out_pdf", type=Path, required=True)
    parser.add_argument("--out_png", type=Path, default=None)
    parser.add_argument("--recommendation_key", default="generic_k1.25")
    args = parser.parse_args()

    problems = ["RZDT1", "RZDT2", "RZDT5_RR"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4), sharex=False, sharey=False)

    for ax, name in zip(axes, problems):
        problem = make_problem(name)
        curve_x, curve_y = problem.true_pareto_curve()
        tpos = problem.true_pareto_front()
        sample = load_points(args.results_dir, name, "sample")
        recommended = load_points(args.results_dir, name, args.recommendation_key)

        ax.plot(curve_x, curve_y, color="black", lw=1.4, label="True PF")
        if len(tpos):
            ax.scatter(tpos[:, 0], tpos[:, 1], s=14, color="black", alpha=0.8, label="TPOS")
        if len(sample):
            ax.scatter(
                sample[:, 0],
                sample[:, 1],
                s=26,
                marker="o",
                facecolors="none",
                edgecolors="#2b6cb0",
                linewidths=1.0,
                alpha=0.8,
                label="Sample Pareto",
            )
        if len(recommended):
            ax.scatter(
                recommended[:, 0],
                recommended[:, 1],
                s=34,
                marker="^",
                color="#d97706",
                alpha=0.72,
                label="Recommended",
            )

        ax.set_title(name)
        ax.set_xlabel(r"$f^1$")
        ax.grid(True, color="#dddddd", linewidth=0.6, alpha=0.8)
        if name == "RZDT5_RR":
            ax.set_xlim(-0.015, 0.52)
        else:
            ax.set_xlim(-0.03, 1.03)
        ymax = max(1.05, float(np.nanmax(curve_y)) + 0.05)
        ax.set_ylim(-0.04, ymax)

    axes[0].set_ylabel(r"$f^2$")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_pdf, bbox_inches="tight")
    if args.out_png is not None:
        args.out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out_png, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
