#!/usr/bin/env python3
"""Render publication tables and figures from compact aggregate CSV files.

This script never reads runtime checkpoints, pickle files, model weights, or
raw scheduler profiles.  Its only inputs are the CSV files emitted by
``aggregate_completed_matrix.py``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import statistics
from typing import Iterable

import numpy as np


METHOD_LABELS = {
    "frozen_proposal": "Frozen proposal",
    "proposal_sobol": "Proposal + Sobol",
    "promoted_joint_voi": "SC-OLH joint VOI",
    "new_point_only": "SC-OLH, new only",
    "pooled_variance": "Pooled variance",
    "frozen_source_discrepancy": "Frozen discrepancy",
    "observed_terminal_closure": "SC-OLH (promoted)",
    "promoted": "SC-OLH (promoted)",
    "new_only": "SC-OLH, new only",
    "n0_best": "Frozen proposal",
    "sobol_new": "Proposal + Sobol",
    "pooled": "Pooled variance",
    "no_discrepancy": "No discrepancy update",
    "botorch_turbo": "TuRBO",
    "botorch_scbo": "SCBO",
    "botorch_saasbo": "SAASBO",
    "safe_fpacoh_cbo": "Safe F-PACOH",
    "rgpe_cbo": "RGPE-CBO",
    "fsbo_cbo": "FSBO-CBO",
    "malibo_cbo": "MALIBO-CBO",
}

METHOD_COLORS = {
    "frozen_proposal": "#666666",
    "proposal_sobol": "#009E73",
    "promoted_joint_voi": "#0072B2",
    "new_point_only": "#56B4E9",
    "pooled_variance": "#E69F00",
    "frozen_source_discrepancy": "#CC79A7",
    "observed_terminal_closure": "#0072B2",
    "promoted": "#0072B2",
    "new_only": "#56B4E9",
    "n0_best": "#666666",
    "sobol_new": "#009E73",
    "pooled": "#E69F00",
    "no_discrepancy": "#CC79A7",
    "botorch_turbo": "#D55E00",
    "botorch_scbo": "#F0E442",
    "botorch_saasbo": "#000000",
    "safe_fpacoh_cbo": "#8C564B",
    "rgpe_cbo": "#9467BD",
    "fsbo_cbo": "#17BECF",
    "malibo_cbo": "#BCBD22",
}


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value):
    number = _number(value)
    return None if number is None else int(number)


def _boolean(value):
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def read_csv(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bootstrap_interval(
    values: Iterable[float],
    *,
    statistic=np.median,
    samples: int = 4000,
    seed: int = 20260721,
) -> tuple[float | None, float | None]:
    values = np.asarray([
        value for value in (_number(item) for item in values)
        if value is not None
    ], dtype=float)
    if len(values) == 0:
        return None, None
    if len(values) == 1:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(int(samples), len(values)))
    resampled = values[indices]
    estimates = np.asarray([statistic(row) for row in resampled], dtype=float)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def _method(row: dict) -> str:
    return str(row.get("method") or row.get("variant") or "unknown")


def _label(method: str) -> str:
    return METHOD_LABELS.get(method, method.replace("_", " "))


def _latex(value: str) -> str:
    return str(value).replace("_", "\\_").replace("%", "\\%")


def _group_rows(rows: list[dict]) -> dict[tuple, list[dict]]:
    groups = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("domain") or "unknown"),
            _integer(row.get("d")),
            _integer(row.get("N")),
            str(row.get("initial_design") or "unknown"),
            _method(row),
        )
        groups[key].append(row)
    return groups


def _cell_metrics(items: list[dict]) -> dict:
    feasible = [row for row in items if _boolean(row.get("true_feasible"))]
    regrets = [
        value for value in (_number(row.get("feasible_regret")) for row in feasible)
        if value is not None
    ]
    rate = len(feasible) / len(items) if items else None
    rate_ci = bootstrap_interval(
        [_boolean(row.get("true_feasible")) for row in items],
        statistic=np.mean,
    )
    regret_ci = bootstrap_interval(regrets)
    certified = sum(_integer(row.get("posterior_certified_count")) or 0
                    for row in items)
    evaluated = sum(_integer(row.get("evaluated_point_count")) or 0
                    for row in items)
    return {
        "n": len(items),
        "feasible_count": len(feasible),
        "feasible_rate": rate,
        "feasible_rate_ci": rate_ci,
        "median_regret": statistics.median(regrets) if regrets else None,
        "median_regret_ci": regret_ci,
        "false_certificates": sum(
            _integer(row.get("false_certificate_count")) or 0
            for row in items),
        "certificate_coverage": certified / evaluated if evaluated else None,
        "median_wall_time": (
            statistics.median(values)
            if (values := [
                value for value in (
                    _number(row.get("wall_time_sec")) for row in items)
                if value is not None
            ]) else None
        ),
    }


def write_main_table(rows: list[dict], path: Path) -> None:
    groups = _group_rows(rows)
    lines = [
        "\\begin{tabular}{lllrrrrr}",
        "\\toprule",
        "Domain & $d/N$ & Method & Feasible & Regret & Cert. cov. & False cert. & Time (s) \\\\",
        "\\midrule",
    ]
    previous_domain = None
    for key in sorted(groups, key=lambda item: (
        item[0], item[1] or -1, item[2] or -1, _label(item[4]))):
        domain, dimension, budget, _initial, method = key
        metrics = _cell_metrics(groups[key])
        if previous_domain is not None and previous_domain != domain:
            lines.append("\\midrule")
        previous_domain = domain
        ratio = (
            "--" if dimension is None or not budget
            else f"{dimension / budget:.1f}"
        )
        regret = (
            "--" if metrics["median_regret"] is None
            else f"{metrics['median_regret']:.4f}"
        )
        coverage = (
            "--" if metrics["certificate_coverage"] is None
            else f"{100 * metrics['certificate_coverage']:.1f}\\%"
        )
        runtime = (
            "--" if metrics["median_wall_time"] is None
            else f"{metrics['median_wall_time']:.1f}"
        )
        lines.append(
            f"{_latex(domain)} & {ratio} & {_latex(_label(method))} & "
            f"{metrics['feasible_count']}/{metrics['n']} & {regret} & "
            f"{coverage} & {metrics['false_certificates']} & {runtime} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_frontier_table(rows: list[dict], path: Path) -> None:
    groups = _group_rows(rows)
    lines = [
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        "Method & Domain & $d$ & $N$ & $d/N$ & Feasible & Regret & 95\\% CI \\\\",
        "\\midrule",
    ]
    for key in sorted(groups, key=lambda item: (
        _label(item[4]), item[0], item[1] or -1, item[2] or -1)):
        domain, dimension, budget, _initial, method = key
        metrics = _cell_metrics(groups[key])
        median = metrics["median_regret"]
        low, high = metrics["median_regret_ci"]
        ci = "--" if low is None else f"[{low:.4f}, {high:.4f}]"
        lines.append(
            f"{_latex(_label(method))} & {_latex(domain)} & {dimension} & "
            f"{budget} & {dimension / budget:.1f} & "
            f"{metrics['feasible_count']}/{metrics['n']} & "
            f"{'--' if median is None else f'{median:.4f}'} & {ci} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _paired_comparisons(rows: list[dict], primary: str) -> list[dict]:
    by_cell = defaultdict(dict)
    for row in rows:
        key = (
            str(row.get("domain")), _integer(row.get("d")),
            _integer(row.get("N")), _integer(row.get("seed")),
            str(row.get("initial_design")),
        )
        by_cell[key][_method(row)] = row
    methods = sorted({_method(row) for row in rows if _method(row) != primary})
    out = []
    for method in methods:
        pairs = [
            (cell[primary], cell[method])
            for cell in by_cell.values()
            if primary in cell and method in cell
        ]
        wins = losses = ties = 0
        regret_deltas = []
        for left, right in pairs:
            left_feasible = _boolean(left.get("true_feasible")) is True
            right_feasible = _boolean(right.get("true_feasible")) is True
            if left_feasible and not right_feasible:
                wins += 1
            elif right_feasible and not left_feasible:
                losses += 1
            elif not left_feasible and not right_feasible:
                ties += 1
            else:
                left_regret = _number(left.get("feasible_regret"))
                right_regret = _number(right.get("feasible_regret"))
                if left_regret is None or right_regret is None:
                    ties += 1
                else:
                    delta = left_regret - right_regret
                    regret_deltas.append(delta)
                    if delta < -1e-12:
                        wins += 1
                    elif delta > 1e-12:
                        losses += 1
                    else:
                        ties += 1
        p_value = None
        if regret_deltas and any(abs(value) > 1e-12 for value in regret_deltas):
            try:
                from scipy.stats import wilcoxon
                p_value = float(wilcoxon(
                    regret_deltas, alternative="two-sided",
                    zero_method="wilcox").pvalue)
            except (ImportError, ValueError):
                p_value = None
        out.append({
            "primary": primary,
            "comparator": method,
            "paired_n": len(pairs),
            "lexicographic_wins": wins,
            "lexicographic_losses": losses,
            "lexicographic_ties": ties,
            "median_regret_delta_when_both_feasible": (
                statistics.median(regret_deltas) if regret_deltas else None),
            "wilcoxon_p": p_value,
        })
    valid = sorted(
        ((index, row["wilcoxon_p"]) for index, row in enumerate(out)
         if row["wilcoxon_p"] is not None),
        key=lambda item: item[1],
    )
    running = 0.0
    m = len(valid)
    for rank, (index, p_value) in enumerate(valid):
        adjusted = min(1.0, (m - rank) * p_value)
        running = max(running, adjusted)
        out[index]["wilcoxon_holm_p"] = running
    return out


def _plot_style():
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
    })


def _save(fig, stem: Path):
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")


def plot_frontier(rows: list[dict], stem: Path) -> bool:
    import matplotlib.pyplot as plt

    groups = _group_rows(rows)
    cells = []
    for key, items in groups.items():
        domain, dimension, budget, _initial, method = key
        if dimension is None or not budget:
            continue
        cells.append((domain, dimension / budget, method, _cell_metrics(items)))
    if not cells:
        return False
    domains = sorted({cell[0] for cell in cells})
    fig, axes = plt.subplots(2, len(domains), figsize=(4.1 * len(domains), 6),
                             squeeze=False, sharex="col")
    for column, domain in enumerate(domains):
        domain_cells = [cell for cell in cells if cell[0] == domain]
        methods = sorted({cell[2] for cell in domain_cells}, key=_label)
        for method in methods:
            subset = sorted(
                (cell for cell in domain_cells if cell[2] == method),
                key=lambda cell: cell[1],
            )
            x = [cell[1] for cell in subset]
            feasible = [cell[3]["feasible_rate"] for cell in subset]
            regret = [cell[3]["median_regret"] for cell in subset]
            color = METHOD_COLORS.get(method, "#444444")
            axes[0, column].plot(x, feasible, marker="o", lw=1.8,
                                 color=color, label=_label(method))
            finite = [(a, b) for a, b in zip(x, regret) if b is not None]
            if finite:
                axes[1, column].plot(
                    [item[0] for item in finite],
                    [item[1] for item in finite],
                    marker="o", lw=1.8, color=color, label=_label(method),
                )
        axes[0, column].set_title(domain.replace("StatePolicyRZDT1", ""))
        axes[0, column].set_ylim(-0.03, 1.03)
        axes[0, column].grid(alpha=0.25)
        axes[1, column].set_xscale("log")
        axes[1, column].set_yscale("symlog", linthresh=1e-3)
        axes[1, column].grid(alpha=0.25, which="both")
        axes[1, column].set_xlabel("Dimension / target calls")
    axes[0, 0].set_ylabel("True-feasible rate")
    axes[1, 0].set_ylabel("Median regret | feasible")
    axes[0, -1].legend(loc="lower left", frameon=False)
    _save(fig, stem)
    plt.close(fig)
    return True


def plot_convergence(traces: list[dict], stem: Path) -> bool:
    import matplotlib.pyplot as plt

    usable = [
        row for row in traces
        if _integer(row.get("target_call")) is not None
    ]
    if not usable:
        return False
    domains = sorted({str(row.get("domain")) for row in usable})
    fig, axes = plt.subplots(2, len(domains), figsize=(4.1 * len(domains), 6),
                             squeeze=False, sharex="col")
    for column, domain in enumerate(domains):
        domain_rows = [row for row in usable if str(row.get("domain")) == domain]
        methods = sorted({_method(row) for row in domain_rows}, key=_label)
        for method in methods:
            method_rows = [row for row in domain_rows if _method(row) == method]
            calls = sorted({_integer(row.get("target_call")) for row in method_rows})
            rates = []
            medians = []
            lows = []
            highs = []
            for call in calls:
                at_call = [row for row in method_rows
                           if _integer(row.get("target_call")) == call]
                regrets = [
                    value for value in (
                        _number(row.get("incumbent_feasible_regret_post_run"))
                        for row in at_call
                    ) if value is not None
                ]
                rates.append(len(regrets) / len(at_call) if at_call else np.nan)
                medians.append(statistics.median(regrets) if regrets else np.nan)
                low, high = bootstrap_interval(regrets, samples=2000,
                                               seed=20260721 + int(call))
                lows.append(np.nan if low is None else low)
                highs.append(np.nan if high is None else high)
            color = METHOD_COLORS.get(method, "#444444")
            axes[0, column].plot(calls, rates, lw=1.8, color=color,
                                 label=_label(method))
            axes[1, column].plot(calls, medians, lw=1.8, color=color,
                                 label=_label(method))
            axes[1, column].fill_between(calls, lows, highs, color=color,
                                         alpha=0.13, linewidth=0)
        axes[0, column].set_title(domain.replace("StatePolicyRZDT1", ""))
        axes[0, column].set_ylim(-0.03, 1.03)
        axes[0, column].grid(alpha=0.25)
        axes[1, column].grid(alpha=0.25)
        axes[1, column].set_xlabel("Charged target calls")
    axes[0, 0].set_ylabel("Incumbent feasible rate")
    axes[1, 0].set_ylabel("Median incumbent regret | feasible")
    axes[0, -1].legend(loc="lower left", frameon=False)
    _save(fig, stem)
    plt.close(fig)
    return True


def plot_certification(rows: list[dict], stem: Path) -> bool:
    import matplotlib.pyplot as plt

    groups = _group_rows(rows)
    cells = []
    for key, items in groups.items():
        domain, _dimension, budget, _initial, method = key
        if budget is None:
            continue
        metrics = _cell_metrics(items)
        if metrics["certificate_coverage"] is not None:
            cells.append((domain, budget, method, metrics))
    if not cells:
        return False
    domains = sorted({cell[0] for cell in cells})
    fig, axes = plt.subplots(1, len(domains), figsize=(4.1 * len(domains), 3.3),
                             squeeze=False, sharey=True)
    for column, domain in enumerate(domains):
        ax = axes[0, column]
        domain_cells = [cell for cell in cells if cell[0] == domain]
        for method in sorted({cell[2] for cell in domain_cells}, key=_label):
            subset = sorted((cell for cell in domain_cells if cell[2] == method),
                            key=lambda cell: cell[1])
            ax.plot([cell[1] for cell in subset],
                    [cell[3]["certificate_coverage"] for cell in subset],
                    marker="o", lw=1.8,
                    color=METHOD_COLORS.get(method, "#444444"),
                    label=_label(method))
        ax.set_title(domain.replace("StatePolicyRZDT1", ""))
        ax.set_xlabel("Charged target calls")
        ax.grid(alpha=0.25)
    axes[0, 0].set_ylabel("Certified / evaluated points")
    axes[0, -1].legend(loc="upper left", frameon=False)
    _save(fig, stem)
    plt.close(fig)
    return True


def render(rows_path: Path, summary_path: Path, traces_path: Path,
           out_dir: Path, primary_method: str, no_plots: bool = False) -> dict:
    rows = read_csv(rows_path)
    summaries = read_csv(summary_path)
    traces = read_csv(traces_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_main_table(rows, out_dir / "table_main.tex")
    write_frontier_table(rows, out_dir / "table_frontier.tex")
    comparisons = _paired_comparisons(rows, primary_method)
    (out_dir / "paired_statistics.json").write_text(
        json.dumps(comparisons, indent=2) + "\n", encoding="utf-8")
    figures = {}
    if not no_plots:
        _plot_style()
        figures = {
            "dimension_budget_frontier": plot_frontier(
                rows, out_dir / "fig_dimension_budget_frontier"),
            "target_convergence": plot_convergence(
                traces, out_dir / "fig_target_convergence"),
            "certification_budget": plot_certification(
                rows, out_dir / "fig_certification_budget"),
        }
    manifest = {
        "schema_version": 1,
        "inputs": {
            "rows": {"path": str(rows_path), "sha256": _sha256(rows_path)},
            "summary": {
                "path": str(summary_path), "sha256": _sha256(summary_path)},
            "traces": {
                "path": str(traces_path), "sha256": _sha256(traces_path)},
        },
        "row_count": len(rows),
        "summary_count": len(summaries),
        "trace_count": len(traces),
        "primary_method": primary_method,
        "figures": figures,
        "contracts": {
            "reads_compact_csv_only": True,
            "reads_checkpoints": False,
            "reads_pickle_or_model_weights": False,
            "conditional_regret_always_accompanied_by_feasibility": True,
            "post_run_truth_not_used_for_decisions": all(
                _boolean(row.get("target_oracle_used_for_decision")) is not True
                for row in traces
            ),
        },
    }
    (out_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--primary-method", default="promoted_joint_voi")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    manifest = render(
        args.rows, args.summary, args.traces, args.out_dir,
        args.primary_method, no_plots=args.no_plots)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
