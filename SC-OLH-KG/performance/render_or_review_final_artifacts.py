#!/usr/bin/env python3
"""Render the final review-remediated OR tables and figures.

Every quantitative input is a compact artifact registered by
``final_evidence_registry_v1.json``.  The figures make five distinct claims:
the method separates source design, target search, and verification; source
scoring helps on the registered randomized task law but not every regime;
source cost must be amortized; the verifier is valid but can be low power; and
source-data quantity is not monotonically beneficial.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE = PROJECT_ROOT / "paper_artifacts/or_review"
DEFAULT_MANUSCRIPT = PROJECT_ROOT / "manuscript"

ARM_LABELS = {
    "source_atlas": "Source-scored atlas",
    "generic_dct_maximin": "Generic DCT maximin",
    "random_low_frequency": "Random low frequency",
    "natural_blockwise": "Natural blockwise",
    "natural_constant_grid": "Natural constant grid",
    "raw_sobol": "Raw Sobol",
    "target_only_dct_space_scbo": "Target-only functional SCBO",
}

METHOD_LABELS = {
    "safe_fpacoh_cbo": "Safe F-PACOH",
    "rgpe_cbo": "RGPE",
    "stacked_transfer_gp_cbo": "Stacked GP",
    "mtgp_cbo": "MTGP",
    "fsbo_cbo": "FSBO",
    "hyperbo_cbo": "HyperBO",
    "metabo_cbo": "MetaBO",
    "malibo_cbo": "MALIBO",
}

REGIME_LABELS = {
    "aligned_low_frequency": "Aligned low frequency",
    "growing_effective_rank": "Growing rank",
    "frequency_support_shift": "Frequency shift",
    "coordinate_permutation": "Permutation",
    "irregular_grid": "Irregular grid",
    "piecewise_smooth": "Piecewise smooth",
    "sparse_high_frequency": "Sparse high frequency",
    "misspecified_target": "Misspecified target",
}

SENSITIVITY_LABELS = {
    "active_rank": "Effective rank",
    "alpha": "Chance level $\\alpha$",
    "safe_mass": "Calibrated safe mass",
    "n0": "Initial-design size $n_0$",
    "source_task_count": "Source tasks",
    "source_profiles_per_task": "Profiles per source task",
    "source_replications_per_profile": "Source replications",
    "atlas_max_frequency": "Retained frequency $K$",
    "atlas_frequency_penalty": "Frequency penalty $\\kappa$",
    "atlas_safety_metric_weight": "Safety-rank weight",
    "atlas_objective_metric_weight": "Objective-rank weight",
    "atlas_first_center_safety_weight": "First-center safety weight",
}

SENSITIVITY_BASELINES = {
    "alpha": 0.05,
    "safe_mass": 0.08,
    "n0": 10,
    "source_task_count": 2,
    "source_profiles_per_task": 64,
    "source_replications_per_profile": 3,
    "atlas_max_frequency": 8,
    "atlas_frequency_penalty": 0.25,
    "atlas_safety_metric_weight": 1.0,
    "atlas_objective_metric_weight": 1.0,
    "atlas_first_center_safety_weight": 0.5,
}

COLORS = {
    "source": "#2878B5",
    "generic": "#8A8F98",
    "natural": "#E09F3E",
    "sobol": "#6C757D",
    "certified": "#2A9D8F",
    "warning": "#C44E52",
    "light": "#DCE6EF",
    "ink": "#263238",
}


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _analysis(evidence, name):
    path = Path(evidence) / f"{name}.json"
    payload = _read_json(path)
    if payload.get("status") not in {
        "complete", "complete_with_algorithmic_failures"
    }:
        raise RuntimeError(f"analysis {name} is not complete")
    return payload["aggregate_analysis"], path


def _latex(value):
    return (
        str(value)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def _write(path, lines):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _weighted(rows, field):
    total = sum(int(row["independent_task_count"]) for row in rows)
    return sum(
        float(row[field]) * int(row["independent_task_count"])
        for row in rows
    ) / total


def _aggregate_by_arm(rows):
    output = []
    for arm in sorted({row["arm"] for row in rows}):
        selected = [row for row in rows if row["arm"] == arm]
        count = sum(int(row["independent_task_count"]) for row in selected)
        output.append({
            "arm": arm,
            "task_count": count,
            "feasible_rate": sum(
                int(row["true_feasible_coverage_count"]) for row in selected
            ) / count,
            "certified_rate": sum(
                int(row["certified_true_feasible_deployment_count"])
                for row in selected
            ) / count,
            "false_certificates": sum(
                int(row["false_certificate_count"]) for row in selected
            ),
            "penalized_loss": _weighted(selected, "mean_penalized_loss"),
            "verification_calls": _weighted(selected, "mean_verification_calls"),
            "all_in_calls": _weighted(
                selected, "mean_all_in_calls_unamortized"
            ),
            "amortized_calls": _weighted(
                selected, "mean_all_in_calls_amortized"
            ),
            "target_calls": int(selected[0]["N"]),
        })
    return output


def _energy_by_arm(rows):
    output = []
    for arm in sorted({row["arm"] for row in rows}):
        selected = [row for row in rows if row["arm"] == arm]
        count = sum(int(row["algorithmic_seed_count"]) for row in selected)
        objectives = [
            float(row["median_objective_if_certified"])
            for row in selected
            if row["median_objective_if_certified"] is not None
        ]
        output.append({
            "arm": arm,
            "count": count,
            "certified": sum(int(row["certified_safe_count"]) for row in selected),
            "false": sum(int(row["false_certificate_count"]) for row in selected),
            "median_objective": float(np.median(objectives)),
            "mean_verification": float(np.mean([
                row["mean_verification_calls"] for row in selected
            ])),
        })
    return output


def _write_primary_table(analysis, path):
    rows = _aggregate_by_arm([
        row for row in analysis["summaries"]
        if row["arm"] != "oracle_library_upper_bound"
    ])
    lines = [
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Initial design & Tasks & Feasible (\\%) & Certified (\\%) & "
        "False cert. & Penalized loss \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_latex(ARM_LABELS[row['arm']])} & {row['task_count']} & "
            f"{100 * row['feasible_rate']:.1f} & "
            f"{100 * row['certified_rate']:.1f} & "
            f"{row['false_certificates']} & {row['penalized_loss']:.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    _write(path, lines)


def _write_regime_table(analysis, path):
    summaries = analysis["summaries"]
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Regime & Source feasible & Source certified & Best source-free "
        "certified & Difference \\\\",
        "\\midrule",
    ]
    for regime in REGIME_LABELS:
        selected = [row for row in summaries if row["regime"] == regime]
        by_arm = {row["arm"]: [] for row in selected}
        for row in selected:
            by_arm[row["arm"]].append(row)
        source = _aggregate_by_arm(by_arm["source_atlas"])[0]
        controls = []
        for arm, rows in by_arm.items():
            if arm not in {"source_atlas", "oracle_library_upper_bound"}:
                controls.append(_aggregate_by_arm(rows)[0])
        best = max(controls, key=lambda row: row["certified_rate"])
        difference = source["certified_rate"] - best["certified_rate"]
        lines.append(
            f"{_latex(REGIME_LABELS[regime])} & "
            f"{100 * source['feasible_rate']:.1f} & "
            f"{100 * source['certified_rate']:.1f} & "
            f"{_latex(ARM_LABELS[best['arm']])}: "
            f"{100 * best['certified_rate']:.1f} & "
            f"{100 * difference:+.1f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    _write(path, lines)


def _write_schema_table(analysis, path):
    rows = analysis["summaries"]
    groups = []
    for schema in ("declared", "schema_blind"):
        for descriptor in ("domain_blind", "conditioned"):
            selected = [
                row for row in rows
                if row["arm"] == "source_atlas"
                and row["schema_mode"] == schema
                and row["descriptor_mode"] == descriptor
            ]
            aggregate = _aggregate_by_arm(selected)[0]
            groups.append((schema, descriptor, aggregate))
    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Schema & Source weighting & Tasks & Feasible (\\%) & Certified "
        "(\\%) & Loss \\\\",
        "\\midrule",
    ]
    for schema, descriptor, row in groups:
        lines.append(
            f"{_latex(schema.replace('_', ' '))} & "
            f"{_latex(descriptor.replace('_', ' '))} & "
            f"{row['task_count']} & {100 * row['feasible_rate']:.1f} & "
            f"{100 * row['certified_rate']:.1f} & "
            f"{row['penalized_loss']:.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    _write(path, lines)


def _compact_number(value):
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.3g}".lstrip("0")


def _write_sensitivity_table(analysis, path):
    rows = [
        row for row in analysis["one_factor_sensitivity_curves"]
        if row["arm"] == "source_atlas"
    ]
    baseline = next(
        row for row in rows if row["sensitivity_axis"] == "baseline"
    )
    lines = [
        "\\begin{tabular}{llll}",
        "\\toprule",
        "Factor & Tested levels & Feasible (\\%) & Certified (\\%) \\\\",
        "\\midrule",
    ]
    for axis, label in SENSITIVITY_LABELS.items():
        selected = [row for row in rows if row["sensitivity_axis"] == axis]
        points = []
        if axis in SENSITIVITY_BASELINES:
            points.append((
                SENSITIVITY_BASELINES[axis],
                baseline["mean_task_feasible_rate"],
                baseline["mean_task_certificate_rate"],
            ))
        points.extend((
            row["sensitivity_value"],
            row["mean_task_feasible_rate"],
            row["mean_task_certificate_rate"],
        ) for row in selected)
        points = sorted(points, key=lambda point: float(point[0]))
        levels = "/".join(_compact_number(point[0]) for point in points)
        feasible = "/".join(f"{100 * point[1]:.1f}" for point in points)
        certified = "/".join(f"{100 * point[2]:.1f}" for point in points)
        lines.append(
            f"{label} & {levels} & {feasible} & {certified} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    _write(path, lines)


def _write_equal_cost_table(analysis, functional, path):
    rows = _aggregate_by_arm(analysis["summaries"])
    functional_rows = functional["summaries"]
    functional_count = sum(
        int(row["independent_task_count"]) for row in functional_rows
    )
    functional_row = {
        "label": "Target-only functional SCBO",
        "source": 0,
        "target": 394,
        "certified_rate": sum(
            int(row["certified_true_feasible_deployment_count"])
            for row in functional_rows
        ) / functional_count,
        "false": sum(
            int(row["false_certificate_count"]) for row in functional_rows
        ),
        "failure": int(functional["algorithmic_failure_count"]),
        "all_in": float(np.mean([
            row["mean_all_in_calls"] for row in functional_rows
            if row["mean_all_in_calls"] is not None
        ])),
    }
    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Method & Source & Target & Certified (\\%) & False & Algo. fail. "
        "& Mean all-in \\\\",
        "\\midrule",
    ]
    for row in rows:
        source = 384 if row["arm"] == "source_atlas" else 0
        lines.append(
            f"{_latex(ARM_LABELS[row['arm']])} & {source} & "
            f"{row['target_calls']} & {100 * row['certified_rate']:.1f} & "
            f"{row['false_certificates']} & 0 & {row['all_in_calls']:.1f} \\\\"
        )
    lines.append(
        f"{functional_row['label']} & 0 & 394 & "
        f"{100 * functional_row['certified_rate']:.1f} & "
        f"{functional_row['false']} & {functional_row['failure']} & "
        f"{functional_row['all_in']:.1f} \\\\"
    )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    _write(path, lines)


def _write_energy_table(analysis, path):
    rows = _energy_by_arm(analysis["market_summaries"])
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Design/backend & Certified & False & Median objective & Mean verify \\\\",
        "\\midrule",
    ]
    order = [
        "source_atlas", "generic_dct_maximin", "random_low_frequency",
        "natural_constant_grid", "raw_sobol", "target_only_dct_space_scbo",
    ]
    indexed = {row["arm"]: row for row in rows}
    for arm in order:
        row = indexed[arm]
        lines.append(
            f"{_latex(ARM_LABELS[arm])} & {row['certified']}/{row['count']} & "
            f"{row['false']} & {row['median_objective']:.4f} & "
            f"{row['mean_verification']:.1f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    _write(path, lines)


def _write_native_transfer_table(analysis, path):
    lines = [
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Native method & FactorShock & Inventory & Queue & Total & False \\\\",
        "\\midrule",
    ]
    for row in analysis["method_summaries"]:
        domains = row["domain_counts"]
        lines.append(
            f"{_latex(METHOD_LABELS[row['method']])} & "
            f"{domains['FactorShockStatePolicyRZDT1']}/20 & "
            f"{domains['InventorySupplyChain']}/20 & "
            f"{domains['QueueResourceControl']}/20 & "
            f"{row['certified_safe_count']}/60 & "
            f"{row['false_certificate_count']} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    _write(path, lines)


def _write_verifier_table(analysis, path):
    probabilities = {0.95, 0.975, 0.99, 0.995, 1.0}
    rows = [
        row for row in analysis["rows"]
        if row["verification_budget"] in {80, 160}
        and row["true_feasibility_probability"] in probabilities
    ]
    lines = [
        "\\begin{tabular}{rrrrrr}",
        "\\toprule",
        "$n$ & $p$ & All-success power & CP threshold & Allowed failures & "
        "CP power \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['verification_budget']} & "
            f"{row['true_feasibility_probability']:.3f} & "
            f"{row['certification_probability']:.3f} & "
            f"{row['clopper_pearson_success_threshold']} & "
            f"{row['clopper_pearson_allowed_failures']} & "
            f"{row['clopper_pearson_certification_probability']:.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    _write(path, lines)


def _write_temporal_table(analysis, path):
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Matrix & Originally certified & Block stable & Nonoverlap stable & "
        "Joint stable \\\\",
        "\\midrule",
    ]
    labels = {
        "energy_v2": "Energy V2",
        "energy_functional_v2": "Functional V2",
        "energy_v3": "Energy V3",
    }
    for row in analysis["matrix_summaries"]:
        denominator = int(row["originally_certified_count"])
        lines.append(
            f"{labels[row['matrix']]} & {denominator} & "
            f"{row['chronological_block_stable_count']}/{denominator} & "
            f"{row['nonoverlap_stable_count']}/{denominator} & "
            f"{row['joint_descriptive_stability_count']}/{denominator} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    _write(path, lines)


def _plot_style():
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.frameon": False,
        "legend.fontsize": 6.5,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })


def _save(fig, stem):
    stem = Path(stem)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    svg = stem.with_suffix(".svg")
    fig.savefig(svg, bbox_inches="tight")
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines())
        + "\n",
        encoding="utf-8",
    )
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")


def _plot_method(stem):
    """Schematic-led figure: one source archive, three charged stages."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(7.2, 2.25))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (0.02, "Source archive\nreplicated outcomes", COLORS["light"]),
        (0.22, "Source ranks +\nprofile coordinate", "#CFE8E1"),
        (0.42, "10-point maximin\ninitial design", "#BFD7EA"),
        (0.62, "Replaceable\ntarget backend", "#E7E7E7"),
        (0.82, "Independent\nterminal verifier", "#F2D6D3"),
    ]
    width = 0.16
    for x, label, color in boxes:
        patch = FancyBboxPatch(
            (x, 0.40), width, 0.32,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            linewidth=0.8, edgecolor=COLORS["ink"], facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x + width / 2, 0.56, label, ha="center", va="center")
    for x in (0.18, 0.38, 0.58, 0.78):
        ax.add_patch(FancyArrowPatch(
            (x, 0.56), (x + 0.035, 0.56), arrowstyle="-|>",
            mutation_scale=9, linewidth=0.8, color=COLORS["ink"],
        ))
    ax.text(0.10, 0.28, "S source calls", ha="center", color=COLORS["ink"])
    ax.text(0.50, 0.28, "N target search calls", ha="center", color=COLORS["ink"])
    ax.text(0.90, 0.28, "V fresh verification calls", ha="center", color=COLORS["ink"])
    ax.plot([0.42, 0.42], [0.39, 0.20], color=COLORS["ink"], lw=0.7)
    ax.plot([0.82, 0.82], [0.39, 0.20], color=COLORS["ink"], lw=0.7)
    ax.annotate(
        "target responses begin", xy=(0.42, 0.17), xytext=(0.42, 0.07),
        ha="center", arrowprops={"arrowstyle": "-", "lw": 0.6},
    )
    ax.annotate(
        "verification never feeds search", xy=(0.82, 0.17),
        xytext=(0.82, 0.07), ha="center",
        arrowprops={"arrowstyle": "-", "lw": 0.6},
    )
    _save(fig, stem)
    plt.close(fig)


def _plot_primary(analysis, stem):
    """Quantitative comparison: aggregate coverage and certification."""
    import matplotlib.pyplot as plt

    rows = _aggregate_by_arm([
        row for row in analysis["summaries"]
        if row["arm"] != "oracle_library_upper_bound"
    ])
    order = [
        "source_atlas", "natural_blockwise", "raw_sobol",
        "random_low_frequency", "generic_dct_maximin",
    ]
    indexed = {row["arm"]: row for row in rows}
    labels = [ARM_LABELS[arm] for arm in order]
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(5.8, 2.7))
    ax.barh(
        y + 0.17, [100 * indexed[a]["feasible_rate"] for a in order],
        height=0.32, color=COLORS["light"], edgecolor=COLORS["source"],
        linewidth=0.7, label="True-feasible design",
    )
    ax.barh(
        y - 0.17, [100 * indexed[a]["certified_rate"] for a in order],
        height=0.32, color=COLORS["certified"], label="Certified deployment",
    )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Rate across 480 independent target tasks (%)")
    ax.legend(loc="lower right")
    ax.grid(axis="x", color="#E5E5E5", lw=0.5)
    _save(fig, stem)
    plt.close(fig)


def _plot_regime(analysis, stem):
    """Scope heatmap: source-minus-generic certification by regime and d."""
    import matplotlib.pyplot as plt

    summaries = analysis["summaries"]
    dimensions = [200, 1000, 10000]
    regimes = list(REGIME_LABELS)
    values = np.zeros((len(regimes), len(dimensions)))
    for i, regime in enumerate(regimes):
        for j, dimension in enumerate(dimensions):
            pair = [
                row for row in summaries
                if row["regime"] == regime
                and row["nominal_dimension"] == dimension
                and row["arm"] in {"source_atlas", "generic_dct_maximin"}
            ]
            rates = {
                row["arm"]: row["certified_true_feasible_deployment_rate"]
                for row in pair
            }
            values[i, j] = rates["source_atlas"] - rates["generic_dct_maximin"]
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    image = ax.imshow(values, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(3), ["d=200", "d=1,000", "d=10,000"])
    ax.set_yticks(np.arange(len(regimes)), [REGIME_LABELS[r] for r in regimes])
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{100 * values[i, j]:+.0f}", ha="center", va="center",
                    color="white" if abs(values[i, j]) > 0.45 else COLORS["ink"])
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    colorbar.set_label("Source - generic DCT certified rate")
    _save(fig, stem)
    plt.close(fig)


def _plot_cost(primary, equal_cost, stem):
    """Cost decomposition: offline archive is visible and amortized separately."""
    import matplotlib.pyplot as plt

    primary_rows = {row["arm"]: row for row in _aggregate_by_arm(
        primary["summaries"]
    )}
    equal_rows = {row["arm"]: row for row in _aggregate_by_arm(
        equal_cost["summaries"]
    )}
    labels = ["Source atlas\nN=10", "Generic DCT\nN=10", "Raw Sobol\nN=394"]
    source = np.array([384.0, 0.0, 0.0])
    search = np.array([10.0, 10.0, 394.0])
    verify = np.array([
        primary_rows["source_atlas"]["verification_calls"],
        primary_rows["generic_dct_maximin"]["verification_calls"],
        equal_rows["raw_sobol"]["verification_calls"],
    ])
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    ax.bar(x, source, color=COLORS["source"], label="Source archive")
    ax.bar(x, search, bottom=source, color=COLORS["generic"], label="Target search")
    ax.bar(x, verify, bottom=source + search, color=COLORS["certified"],
           label="Independent verification")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean simulator calls per target")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.grid(axis="y", color="#E5E5E5", lw=0.5)
    _save(fig, stem)
    plt.close(fig)


def _plot_verifier(analysis, stem):
    """Power audit: validity does not imply useful near-boundary power."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    for budget, color in ((80, COLORS["source"]), (160, COLORS["warning"])):
        rows = [
            row for row in analysis["rows"]
            if row["verification_budget"] == budget
        ]
        x = [row["true_feasibility_probability"] for row in rows]
        axes[0].plot(x, [row["certification_probability"] for row in rows],
                     marker="o", ms=3, color=color, label=f"n={budget}")
        axes[1].plot(
            x, [row["clopper_pearson_certification_probability"] for row in rows],
            marker="o", ms=3, color=color, label=f"n={budget}",
        )
    for ax, title in zip(axes, ("Preregistered all-success", "Exact CP sensitivity")):
        ax.axvline(0.95, color="#777777", lw=0.7, ls="--")
        ax.set_title(title)
        ax.set_xlabel("True feasibility probability")
        ax.set_ylim(0, 1.02)
        ax.grid(color="#E5E5E5", lw=0.5)
    axes[0].set_ylabel("Certification probability")
    axes[1].legend(loc="upper left")
    _save(fig, stem)
    plt.close(fig)


def _plot_source_budget(analysis, stem):
    """Sensitivity: more source calls are not uniformly better."""
    import matplotlib.pyplot as plt

    baseline = next(
        row for row in analysis["one_factor_sensitivity_curves"]
        if row["arm"] == "source_atlas" and row["sensitivity_axis"] == "baseline"
    )
    axes = {
        "source_task_count": (lambda value: int(value) * 64 * 3, "Source tasks"),
        "source_profiles_per_task": (lambda value: 2 * int(value) * 3, "Profiles/task"),
        "source_replications_per_profile": (
            lambda value: 2 * 64 * int(value), "Replications/profile"
        ),
    }
    fig, ax = plt.subplots(figsize=(4.7, 2.7))
    markers = ("o", "s", "^")
    for marker, (axis_name, (calls, label)) in zip(markers, axes.items()):
        rows = [
            row for row in analysis["one_factor_sensitivity_curves"]
            if row["arm"] == "source_atlas"
            and row["sensitivity_axis"] == axis_name
        ]
        base_value = {
            "source_task_count": 2,
            "source_profiles_per_task": 64,
            "source_replications_per_profile": 3,
        }[axis_name]
        points = [(calls(base_value), baseline["mean_task_certificate_rate"])]
        points.extend((calls(row["sensitivity_value"]), row["mean_task_certificate_rate"])
                      for row in rows)
        points = sorted(set(points))
        ax.plot([x for x, _ in points], [y for _, y in points], marker=marker,
                ms=4, lw=1.0, label=label)
    ax.axvline(384, color="#777777", lw=0.7, ls="--")
    ax.set_xlabel("Source simulator calls")
    ax.set_ylabel("Mean task certified rate")
    ax.set_ylim(0, 0.55)
    ax.grid(color="#E5E5E5", lw=0.5)
    ax.legend()
    _save(fig, stem)
    plt.close(fig)


def render(evidence, manuscript, *, skip_figures=False):
    evidence = Path(evidence)
    manuscript = Path(manuscript)
    registry = _read_json(evidence / "final_evidence_registry_v1.json")
    if registry.get("publication_ready") is not True:
        raise RuntimeError("final evidence registry is not publication ready")

    names = [
        "randomized_profile_primary",
        "randomized_profile_sensitivity",
        "randomized_profile_schema_descriptor",
        "randomized_profile_equal_preverification",
        "energy_v3",
        "energy_temporal",
        "native_transfer",
        "functional_equal_preverification",
        "verifier_power",
    ]
    loaded = {}
    inputs = [evidence / "final_evidence_registry_v1.json"]
    for name in names:
        loaded[name], path = _analysis(evidence, name)
        inputs.append(path)

    tables = manuscript / "tables"
    figures = manuscript / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    outputs = []

    table_calls = {
        "review_stress_primary.tex": lambda path: _write_primary_table(
            loaded["randomized_profile_primary"], path),
        "review_stress_regimes.tex": lambda path: _write_regime_table(
            loaded["randomized_profile_primary"], path),
        "review_schema_controls.tex": lambda path: _write_schema_table(
            loaded["randomized_profile_schema_descriptor"], path),
        "review_sensitivity.tex": lambda path: _write_sensitivity_table(
            loaded["randomized_profile_sensitivity"], path),
        "review_equal_cost.tex": lambda path: _write_equal_cost_table(
            loaded["randomized_profile_equal_preverification"],
            loaded["functional_equal_preverification"], path),
        "review_energy_v3.tex": lambda path: _write_energy_table(
            loaded["energy_v3"], path),
        "review_native_transfer.tex": lambda path: _write_native_transfer_table(
            loaded["native_transfer"], path),
        "review_verifier_power.tex": lambda path: _write_verifier_table(
            loaded["verifier_power"], path),
        "review_temporal.tex": lambda path: _write_temporal_table(
            loaded["energy_temporal"], path),
    }
    for name, writer in table_calls.items():
        path = tables / name
        writer(path)
        outputs.append(path)

    if not skip_figures:
        _plot_style()
        plots = {
            "review_method": lambda stem: _plot_method(stem),
            "review_primary": lambda stem: _plot_primary(
                loaded["randomized_profile_primary"], stem),
            "review_regime": lambda stem: _plot_regime(
                loaded["randomized_profile_primary"], stem),
            "review_cost": lambda stem: _plot_cost(
                loaded["randomized_profile_primary"],
                loaded["randomized_profile_equal_preverification"], stem),
            "review_verifier_power": lambda stem: _plot_verifier(
                loaded["verifier_power"], stem),
            "review_source_budget": lambda stem: _plot_source_budget(
                loaded["randomized_profile_sensitivity"], stem),
        }
        for name, writer in plots.items():
            stem = figures / name
            writer(stem)
            outputs.extend(stem.with_suffix(suffix) for suffix in (".pdf", ".svg", ".png"))

    manifest = {
        "schema_version": 1,
        "contract_id": "or_review_final_manuscript_render_v1",
        "status": "complete",
        "contracts": {
            "reads_compact_audited_artifacts_only": True,
            "evidence_frozen_before_manuscript_render": True,
            "output_hashes_cover_all_generated_tables_and_figures": True,
        },
        "evidence_registry_sha256": _sha256(
            evidence / "final_evidence_registry_v1.json"
        ),
        "inputs": [
            {"path": path.name, "sha256": _sha256(path)} for path in inputs
        ],
        "outputs": [
            {
                "path": path.relative_to(manuscript).as_posix(),
                "sha256": _sha256(path),
            }
            for path in outputs
        ],
    }
    manifest_path = manuscript / "review_artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", default=str(DEFAULT_EVIDENCE))
    parser.add_argument("--manuscript", default=str(DEFAULT_MANUSCRIPT))
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()
    manifest = render(
        args.evidence, args.manuscript, skip_figures=args.skip_figures
    )
    print(json.dumps({
        "status": manifest["status"],
        "input_count": len(manifest["inputs"]),
        "output_count": len(manifest["outputs"]),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
