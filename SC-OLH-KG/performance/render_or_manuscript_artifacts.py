#!/usr/bin/env python3
"""Render the frozen Operations Research manuscript tables and figures.

Only compact, audited JSON/CSV artifacts are read.  Runtime checkpoints,
pickles, policy vectors, model weights, and scheduler profiles are outside the
input contract.
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

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT = PROJECT_ROOT / "paper_artifacts/paper_result_audit_v1.json"
DEFAULT_SUMMARY = (
    PROJECT_ROOT / "paper_artifacts/paper_result_audit_summary_v1.csv"
)
DEFAULT_DIMENSION = (
    PROJECT_ROOT / "paper_artifacts/final_dimension_budget_evidence_v1.json"
)
DEFAULT_COVERAGE = (
    PROJECT_ROOT / "paper_artifacts/source_target_coverage/aggregate.json"
)
DEFAULT_ENERGY = (
    PROJECT_ROOT
    / "performance/manifests/external_energy_postconfirmatory_fairness_result_v1.json"
)
DEFAULT_CONVERGENCE = (
    PROJECT_ROOT / "paper_artifacts/paper_search_convergence_v1.csv"
)


METHOD_LABELS = {
    "canonical_saasbo_every_iteration": "Canonical SAASBO",
    "frozen_crossdim_proposal_only": "Proposal only",
    "common_sobol_proposal_only": "Proposal only",
    "stacked_transfer_gp_cbo:official_transfergpbo_code": "Stacked GP",
    "scolh:v69_feasible_first_verified_initial_incumbent": "SC-V69",
    "safe_fpacoh_cbo:official_code_with_compatibility_shims": "Safe F-PACOH",
    "rgpe_cbo:official_transfergpbo_code": "RGPE",
    "mtgp_cbo:official_transfergpbo_code": "Transfer MTGP",
    "fsbo_cbo:official_code_adapted_to_scalar_cbo": "FSBO",
    "hyperbo_cbo:official_hyperbo_code_with_gfile_shim": "HyperBO",
    "metabo_cbo:official_neuralaf_ppo_fixed_archive_extension": "MetaBO",
    "malibo_cbo:official_metablor_core_adapted_to_cbo": "MALIBO",
    "frozen_universal_proposal_only": "Universal support",
    "frozen_source_templates_proposal_only": "Source templates",
    "botorch_scbo:canonical_scbo_constrained_ts+hvd:pooled": "Pooled",
    "botorch_scbo:canonical_scbo_constrained_ts+hvd:provider_cumulative_factor": (
        "Cumulative factor-HVD"
    ),
    "uniform_verified::botorch_turbo:canonical_turbo1_ts": "Target-only TuRBO",
    "uniform_verified::botorch_scbo:canonical_scbo_constrained_ts": (
        "Target-only SCBO"
    ),
    "uniform_verified::saasbo_periodic_capped": "Target-only SAASBO",
    "uniform_verified::canonical_saasbo_every_iteration": (
        "Atlas + canonical SAASBO"
    ),
}

DOMAIN_LABELS = {
    "FactorShockStatePolicyRZDT1": "FactorShock",
    "InventorySupplyChain": "Inventory",
    "QueueResourceControl": "Queue",
}

COLORS = {
    "frozen": "#0072B2",
    "sobol": "#999999",
    "universal": "#E69F00",
    "source": "#56B4E9",
    "combined": "#009E73",
    "pooled": "#999999",
    "hvd": "#CC79A7",
    "d1000": "#0072B2",
    "d10000": "#D55E00",
}


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _label_method(method: str) -> str:
    return METHOD_LABELS.get(method, method.replace("_", " "))


def _label_domain(domain: str) -> str:
    return DOMAIN_LABELS.get(domain, domain)


def _latex(text: str) -> str:
    return (
        str(text)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def _records(
    audit: dict,
    *,
    track: str,
    method: str | None = None,
    domain: str | None = None,
) -> list[dict]:
    rows = [
        row
        for row in audit["records"]
        if row.get("track_id") == track and row.get("status") == "ok"
    ]
    if method is not None:
        rows = [row for row in rows if row.get("method_identity") == method]
    if domain is not None:
        rows = [row for row in rows if row.get("domain") == domain]
    return rows


def _metrics(rows: list[dict]) -> dict:
    feasible = [row for row in rows if row.get("true_feasible") is True]
    certified = [row for row in rows if row.get("terminal_certified") is True]
    regrets = [
        float(row["feasible_regret"])
        for row in feasible
        if _number(row.get("feasible_regret")) is not None
    ]
    verification = [
        float(row["target_verification_calls"])
        for row in rows
        if _number(row.get("target_verification_calls")) is not None
    ]
    return {
        "n": len(rows),
        "feasible": len(feasible),
        "certified": len(certified),
        "false": sum(row.get("false_certificate") is True for row in rows),
        "median_regret": statistics.median(regrets) if regrets else None,
        "mean_verification": statistics.mean(verification) if verification else None,
        "source_calls": sorted({row.get("source_calls") for row in rows}),
        "target_search_calls": sorted(
            {row.get("target_search_calls") for row in rows}
        ),
    }


def _plot_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
        }
    )


def _save(fig, stem: Path) -> None:
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    svg_path = stem.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight")
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")


def _write_frontend_backend_table(audit: dict, path: Path) -> None:
    domains = list(DOMAIN_LABELS)
    arms = [
        (
            "Frozen atlas",
            "final_frozen_source_frontend_backend_d1000_n13",
            "frozen_crossdim_proposal_only",
        ),
        (
            "Frozen atlas",
            "final_frozen_source_frontend_backend_d1000_n13",
            "stacked_transfer_gp_cbo:official_transfergpbo_code",
        ),
        (
            "Frozen atlas",
            "final_frozen_source_frontend_backend_d1000_n13",
            "canonical_saasbo_every_iteration",
        ),
        (
            "Common Sobol",
            "final_frozen_sobol_frontend_control_d1000_n13",
            "common_sobol_proposal_only",
        ),
        (
            "Common Sobol",
            "final_frozen_sobol_frontend_control_d1000_n13",
            "stacked_transfer_gp_cbo:official_transfergpbo_code",
        ),
        (
            "Common Sobol",
            "final_frozen_sobol_frontend_control_d1000_n13",
            "canonical_saasbo_every_iteration",
        ),
    ]
    lines = [
        "\\begin{tabular}{lllrrrr}",
        "\\toprule",
        "Domain & Frontend & Backend & Feasible & Certified & False & Med. regret \\\\",
        "\\midrule",
    ]
    for domain in domains:
        for frontend, track, method in arms:
            metrics = _metrics(
                _records(audit, track=track, method=method, domain=domain)
            )
            regret = (
                "--"
                if metrics["median_regret"] is None
                else f"{metrics['median_regret']:.4f}"
            )
            lines.append(
                f"{_latex(_label_domain(domain))} & {_latex(frontend)} & "
                f"{_latex(_label_method(method))} & "
                f"{metrics['feasible']}/{metrics['n']} & "
                f"{metrics['certified']}/{metrics['n']} & "
                f"{metrics['false']} & {regret} \\\\"
            )
        if domain != domains[-1]:
            lines.append("\\midrule")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_component_table(audit: dict, path: Path) -> None:
    track = "frontend_component_causal_ablation_d1000_n10"
    methods = [
        "frozen_universal_proposal_only",
        "frozen_source_templates_proposal_only",
        "frozen_crossdim_proposal_only",
    ]
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Frontend component & Feasible & Certified & False & Med. regret \\\\",
        "\\midrule",
    ]
    for method in methods:
        metrics = _metrics(_records(audit, track=track, method=method))
        regret = (
            "--"
            if metrics["median_regret"] is None
            else f"{metrics['median_regret']:.4f}"
        )
        label = (
            "Combined maximin atlas"
            if method == "frozen_crossdim_proposal_only"
            else _label_method(method)
        )
        lines.append(
            f"{_latex(label)} & {metrics['feasible']}/{metrics['n']} & "
            f"{metrics['certified']}/{metrics['n']} & {metrics['false']} & "
            f"{regret} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_transfer_table(audit: dict, path: Path) -> None:
    transfer_track = "final_archive_fair_transfer_matrix_d1000_n13"
    final_track = "final_frozen_source_frontend_backend_d1000_n13"
    methods = [
        "canonical_saasbo_every_iteration",
        "safe_fpacoh_cbo:official_code_with_compatibility_shims",
        "rgpe_cbo:official_transfergpbo_code",
        "stacked_transfer_gp_cbo:official_transfergpbo_code",
        "mtgp_cbo:official_transfergpbo_code",
        "fsbo_cbo:official_code_adapted_to_scalar_cbo",
        "hyperbo_cbo:official_hyperbo_code_with_gfile_shim",
        "metabo_cbo:official_neuralaf_ppo_fixed_archive_extension",
        "malibo_cbo:official_metablor_core_adapted_to_cbo",
    ]
    lines = [
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Method & Feasible & Certified & False & Med. regret & Mean verify \\\\",
        "\\midrule",
    ]
    for method in methods:
        track = final_track if method == "canonical_saasbo_every_iteration" else transfer_track
        metrics = _metrics(_records(audit, track=track, method=method))
        regret = (
            "--"
            if metrics["median_regret"] is None
            else f"{metrics['median_regret']:.4f}"
        )
        verification = (
            "--"
            if metrics["mean_verification"] is None
            else f"{metrics['mean_verification']:.1f}"
        )
        label = (
            "Atlas + canonical SAASBO"
            if method == "canonical_saasbo_every_iteration"
            else _label_method(method)
        )
        lines.append(
            f"{_latex(label)} & {metrics['feasible']}/{metrics['n']} & "
            f"{metrics['certified']}/{metrics['n']} & {metrics['false']} & "
            f"{regret} & {verification} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_total_cost_table(audit: dict, path: Path) -> None:
    track = "uniform_external_total_cost_d1000_n397"
    methods = [
        "uniform_verified::canonical_saasbo_every_iteration",
        "uniform_verified::botorch_turbo:canonical_turbo1_ts",
        "uniform_verified::botorch_scbo:canonical_scbo_constrained_ts",
        "uniform_verified::saasbo_periodic_capped",
    ]
    lines = [
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Method & Source & Search & Feasible & Certified & False \\\\",
        "\\midrule",
    ]
    for method in methods:
        metrics = _metrics(_records(audit, track=track, method=method))
        source = metrics["source_calls"][0] if metrics["source_calls"] else "--"
        search = (
            metrics["target_search_calls"][0]
            if metrics["target_search_calls"]
            else "--"
        )
        lines.append(
            f"{_latex(_label_method(method))} & {source} & {search} & "
            f"{metrics['feasible']}/{metrics['n']} & "
            f"{metrics['certified']}/{metrics['n']} & {metrics['false']} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_dimension_table(dimension: dict, path: Path) -> None:
    order = [
        "d1000_N10",
        "d1000_N13",
        "d10000_N10",
        "d10000_N13",
        "d10000_N20",
        "d10000_N40",
    ]
    lines = [
        "\\begin{tabular}{rrrrrrrr}",
        "\\toprule",
        "$d$ & Search & $d/N$ & Source & Feasible & Certified & False & Med. regret \\\\",
        "\\midrule",
    ]
    for key in order:
        cell = dimension["cells"][key]
        lines.append(
            f"{cell['dimension']} & {cell['target_search_calls']} & "
            f"{cell['dimension_over_target_search_calls']:.1f} & "
            f"{cell['source_calls'][0]} & "
            f"{cell['true_feasible_count']}/{cell['result_count']} & "
            f"{cell['certified_count']}/{cell['result_count']} & "
            f"{cell['false_certificate_count']} & "
            f"{cell['median_feasible_regret']:.4f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_hvd_table(summary_rows: list[dict], path: Path) -> None:
    rows = [
        row
        for row in summary_rows
        if row.get("track_id") == "source_hvd_causal_gate_d1000_n13"
    ]
    lines = [
        "\\begin{tabular}{lllrrrr}",
        "\\toprule",
        "Domain & Variance model & Feasible & Log-RMSE & Shape corr. & Mean verify \\\\",
        "\\midrule",
    ]
    for domain in DOMAIN_LABELS:
        for row in rows:
            if row.get("domain") != domain:
                continue
            lines.append(
                f"{_latex(_label_domain(domain))} & "
                f"{_latex(_label_method(row['method_identity']))} & "
                f"{row['true_feasible_count']}/{row['successful_rows']} & "
                f"{float(row['mean_aleatoric_log_variance_rmse']):.3f} & "
                f"{float(row['mean_aleatoric_variance_shape_correlation']):.3f} & "
                f"{float(row['mean_target_verification_calls']):.1f} \\\\"
            )
        if domain != list(DOMAIN_LABELS)[-1]:
            lines.append("\\midrule")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_energy_table(energy: dict, path: Path) -> None:
    labels = {
        "frozen_proposal_n13": "Frozen atlas, $N=13$",
        "low_frequency_grid_n13": "Natural low-frequency grid, $N=13$",
        "common_sobol_n397": "Common Sobol, $N=397$",
    }
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Arm & Certified & False & Median objective & Mean verify \\\\",
        "\\midrule",
    ]
    for arm in ["frozen_proposal_n13", "low_frequency_grid_n13", "common_sobol_n397"]:
        cell = energy["summaries"][arm]
        lines.append(
            f"{labels[arm]} & {cell['independently_certified_count']}/"
            f"{cell['seed_count']} & {cell['false_certificate_count']} & "
            f"{cell['median_safe_objective']:.4f} & "
            f"{cell['mean_verification_calls']:.1f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_coverage_table(coverage: dict, path: Path) -> None:
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Domain & $r_{cover}$ & Support shift & Safe radius & Slack \\\\",
        "\\midrule",
    ]
    for row in coverage["rows"]:
        lines.append(
            f"{_latex(_label_domain(row['domain']))} & "
            f"{row['source_support_atlas_cover_radius']:.4f} & "
            f"{row['source_support_shift']:.4f} & "
            f"{row['finite_library_safe_radius']:.4f} & "
            f"{row['finite_library_coverage_slack']:.4f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_frontend_backend(audit: dict, stem: Path) -> None:
    import matplotlib.pyplot as plt

    backends = [
        ("Proposal only", "frozen_crossdim_proposal_only", "common_sobol_proposal_only"),
        (
            "Stacked GP",
            "stacked_transfer_gp_cbo:official_transfergpbo_code",
            "stacked_transfer_gp_cbo:official_transfergpbo_code",
        ),
        ("SAASBO", "canonical_saasbo_every_iteration", "canonical_saasbo_every_iteration"),
    ]
    domains = list(DOMAIN_LABELS)
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.1), sharey=True)
    x = np.arange(len(backends), dtype=float)
    width = 0.34
    for axis, domain in zip(axes, domains):
        frozen = []
        sobol = []
        for _, frozen_method, sobol_method in backends:
            frozen_rows = _records(
                audit,
                track="final_frozen_source_frontend_backend_d1000_n13",
                method=frozen_method,
                domain=domain,
            )
            sobol_rows = _records(
                audit,
                track="final_frozen_sobol_frontend_control_d1000_n13",
                method=sobol_method,
                domain=domain,
            )
            frozen.append(_metrics(frozen_rows)["certified"] / len(frozen_rows))
            sobol.append(_metrics(sobol_rows)["certified"] / len(sobol_rows))
        axis.bar(x - width / 2, frozen, width, color=COLORS["frozen"], label="Frozen atlas")
        axis.bar(x + width / 2, sobol, width, color=COLORS["sobol"], label="Common Sobol")
        axis.set_xticks(x, [item[0] for item in backends], rotation=20, ha="right")
        axis.set_title(_label_domain(domain))
        axis.set_ylim(0, 1.05)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Independently certified rate")
    axes[-1].legend(loc="upper right")
    _save(fig, stem)
    plt.close(fig)


def _plot_component_ablation(audit: dict, stem: Path) -> None:
    import matplotlib.pyplot as plt

    track = "frontend_component_causal_ablation_d1000_n10"
    methods = [
        ("Universal", "frozen_universal_proposal_only", COLORS["universal"]),
        ("Source templates", "frozen_source_templates_proposal_only", COLORS["source"]),
        ("Combined atlas", "frozen_crossdim_proposal_only", COLORS["combined"]),
    ]
    domains = list(DOMAIN_LABELS)
    x = np.arange(len(domains), dtype=float)
    width = 0.24
    fig, axis = plt.subplots(figsize=(6.8, 3.3))
    for index, (label, method, color) in enumerate(methods):
        values = []
        for domain in domains:
            rows = _records(audit, track=track, method=method, domain=domain)
            values.append(_metrics(rows)["certified"] / len(rows))
        axis.bar(x + (index - 1) * width, values, width, color=color, label=label)
    axis.set_xticks(x, [_label_domain(domain) for domain in domains])
    axis.set_ylabel("Independently certified rate")
    axis.set_ylim(0, 1.05)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper center", ncol=3)
    _save(fig, stem)
    plt.close(fig)


def _plot_dimension_budget(dimension: dict, stem: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    for dimension_value, keys, color, label in [
        (1000, ["d1000_N10", "d1000_N13"], COLORS["d1000"], "$d=1{,}000$"),
        (
            10000,
            ["d10000_N10", "d10000_N13", "d10000_N20", "d10000_N40"],
            COLORS["d10000"],
            "$d=10{,}000$",
        ),
    ]:
        cells = [dimension["cells"][key] for key in keys]
        budgets = [cell["target_search_calls"] for cell in cells]
        regrets = [cell["median_feasible_regret"] for cell in cells]
        verification = [cell["mean_target_verification_calls"] for cell in cells]
        axes[0].plot(budgets, regrets, marker="o", lw=1.8, color=color, label=label)
        axes[1].plot(budgets, verification, marker="o", lw=1.8, color=color, label=label)
    axes[0].set_xlabel("Target search calls")
    axes[0].set_ylabel("Median feasible regret")
    axes[1].set_xlabel("Target search calls")
    axes[1].set_ylabel("Mean verification calls")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
    _save(fig, stem)
    plt.close(fig)


def _plot_hvd(summary_rows: list[dict], stem: Path) -> None:
    import matplotlib.pyplot as plt

    rows = [
        row
        for row in summary_rows
        if row.get("track_id") == "source_hvd_causal_gate_d1000_n13"
    ]
    domains = list(DOMAIN_LABELS)
    methods = [
        ("Pooled", "botorch_scbo:canonical_scbo_constrained_ts+hvd:pooled", COLORS["pooled"]),
        (
            "Cumulative factor-HVD",
            "botorch_scbo:canonical_scbo_constrained_ts+hvd:provider_cumulative_factor",
            COLORS["hvd"],
        ),
    ]
    x = np.arange(len(domains), dtype=float)
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    for index, (label, method, color) in enumerate(methods):
        selected = {
            row["domain"]: row for row in rows if row["method_identity"] == method
        }
        rmse = [float(selected[domain]["mean_aleatoric_log_variance_rmse"]) for domain in domains]
        corr = [float(selected[domain]["mean_aleatoric_variance_shape_correlation"]) for domain in domains]
        offset = x + (index - 0.5) * width
        axes[0].bar(offset, rmse, width, color=color, label=label)
        axes[1].bar(offset, corr, width, color=color, label=label)
    labels = [_label_domain(domain) for domain in domains]
    for axis in axes:
        axis.set_xticks(x, labels, rotation=15, ha="right")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean log-variance RMSE")
    axes[1].set_ylabel("Mean variance-shape correlation")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(loc="lower right")
    _save(fig, stem)
    plt.close(fig)


def _plot_energy(energy: dict, stem: Path) -> None:
    import matplotlib.pyplot as plt

    by_arm = defaultdict(dict)
    for row in energy["compact_rows"]:
        by_arm[row["arm"]][int(row["seed"])] = float(row["true_objective_mean"])
    frozen = by_arm["frozen_proposal_n13"]
    grid = by_arm["low_frequency_grid_n13"]
    sobol = by_arm["common_sobol_n397"]
    seeds = sorted(frozen)
    differences = [
        ("Natural low-frequency grid", [frozen[seed] - grid[seed] for seed in seeds]),
        ("Equal-total-cost Sobol", [frozen[seed] - sobol[seed] for seed in seeds]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.3))
    arm_order = [
        ("Frozen\natlas", frozen, COLORS["frozen"]),
        ("Natural\ngrid", grid, COLORS["combined"]),
        ("Sobol\n$N=397$", sobol, COLORS["sobol"]),
    ]
    rng = np.random.default_rng(20260803)
    for index, (label, values, color) in enumerate(arm_order):
        vector = np.asarray([values[seed] for seed in seeds])
        jitter = rng.uniform(-0.06, 0.06, size=len(vector))
        axes[0].scatter(np.full(len(vector), index) + jitter, vector, s=18, color=color, alpha=0.75)
        axes[0].plot([index - 0.18, index + 0.18], [np.median(vector)] * 2, color="black", lw=2)
    axes[0].set_xticks(range(3), [item[0] for item in arm_order])
    axes[0].set_ylabel("Safe objective (lower is better)")
    axes[0].grid(axis="y", alpha=0.25)
    for index, (label, vector) in enumerate(differences):
        jitter = rng.uniform(-0.06, 0.06, size=len(vector))
        axes[1].scatter(np.full(len(vector), index) + jitter, vector, s=18, alpha=0.75, color="#444444")
        axes[1].plot([index - 0.18, index + 0.18], [np.median(vector)] * 2, color="black", lw=2)
    axes[1].axhline(0, color="#D55E00", ls="--", lw=1)
    axes[1].set_xticks(range(2), [item[0] for item in differences], rotation=12, ha="right")
    axes[1].set_ylabel("Frozen atlas minus control")
    axes[1].grid(axis="y", alpha=0.25)
    _save(fig, stem)
    plt.close(fig)


def _plot_convergence(convergence_rows: list[dict], stem: Path) -> None:
    import matplotlib.pyplot as plt

    rows = [
        row
        for row in convergence_rows
        if row.get("track_id") == "final_frozen_source_frontend_backend_d1000_n13"
        and row.get("method_identity") == "canonical_saasbo_every_iteration"
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.0), sharey=False)
    for axis, domain in zip(axes, DOMAIN_LABELS):
        subset = [row for row in rows if row.get("domain") == domain]
        calls = sorted({int(row["target_call"]) for row in subset})
        medians = []
        lower = []
        upper = []
        for call in calls:
            values = np.asarray(
                [
                    float(row["incumbent_feasible_regret_post_run"])
                    for row in subset
                    if int(row["target_call"]) == call
                    and _number(row.get("incumbent_feasible_regret_post_run")) is not None
                ]
            )
            if values.size:
                medians.append(float(np.median(values)))
                lower.append(float(np.quantile(values, 0.25)))
                upper.append(float(np.quantile(values, 0.75)))
            else:
                medians.append(np.nan)
                lower.append(np.nan)
                upper.append(np.nan)
        axis.plot(calls, medians, color=COLORS["frozen"], lw=1.8)
        axis.fill_between(calls, lower, upper, color=COLORS["frozen"], alpha=0.18, linewidth=0)
        axis.axvline(10, color="#777777", ls="--", lw=1)
        axis.set_title(_label_domain(domain))
        axis.set_xlabel("Target search calls")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Incumbent feasible regret")
    _save(fig, stem)
    plt.close(fig)


def render(
    *,
    audit_path: Path,
    summary_path: Path,
    dimension_path: Path,
    coverage_path: Path,
    energy_path: Path,
    convergence_path: Path,
    output_dir: Path,
    no_plots: bool = False,
) -> dict:
    audit = _read_json(audit_path)
    summary_rows = _read_csv(summary_path)
    dimension = _read_json(dimension_path)
    coverage = _read_json(coverage_path)
    energy = _read_json(energy_path)
    convergence_rows = _read_csv(convergence_path)

    if audit.get("status") != "pass":
        raise ValueError("paper result audit is not passed")
    if dimension.get("status") != "complete":
        raise ValueError("dimension evidence is not complete")
    if not str(coverage.get("status", "")).startswith("complete"):
        raise ValueError("coverage evidence is not complete")
    if energy.get("status") != "complete":
        raise ValueError("external energy fairness audit is not complete")

    tables = output_dir / "tables"
    figures = output_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    _write_frontend_backend_table(audit, tables / "frontend_backend.tex")
    _write_component_table(audit, tables / "frontend_components.tex")
    _write_transfer_table(audit, tables / "transfer_baselines.tex")
    _write_total_cost_table(audit, tables / "total_cost_controls.tex")
    _write_dimension_table(dimension, tables / "dimension_budget.tex")
    _write_hvd_table(summary_rows, tables / "hvd_diagnostic.tex")
    _write_energy_table(energy, tables / "external_energy.tex")
    _write_coverage_table(coverage, tables / "coverage_audit.tex")

    if not no_plots:
        _plot_style()
        _plot_frontend_backend(audit, figures / "frontend_backend")
        _plot_component_ablation(audit, figures / "frontend_components")
        _plot_dimension_budget(dimension, figures / "dimension_budget")
        _plot_hvd(summary_rows, figures / "hvd_diagnostic")
        _plot_energy(energy, figures / "external_energy")
        _plot_convergence(convergence_rows, figures / "target_convergence")

    inputs = [
        audit_path,
        summary_path,
        dimension_path,
        coverage_path,
        energy_path,
        convergence_path,
    ]
    outputs = sorted(
        [path for path in tables.iterdir() if path.is_file()]
        + [path for path in figures.iterdir() if path.is_file()]
    )
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "contract_id": "or_manuscript_artifacts_from_audited_compact_data_v1",
        "inputs": [
            {"path": str(path), "sha256": _sha256(path)} for path in inputs
        ],
        "outputs": [
            {
                "path": str(path.relative_to(output_dir)),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in outputs
        ],
        "contracts": {
            "reads_compact_audited_artifacts_only": True,
            "reads_checkpoints": False,
            "reads_pickles": False,
            "reads_model_weights": False,
            "exports_policy_vectors": False,
            "source_search_verification_costs_kept_separate": True,
            "hvd_rendered_as_secondary_diagnostic": True,
            "external_structured_control_negative_result_retained": True,
        },
    }
    (output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--dimension", type=Path, default=DEFAULT_DIMENSION)
    parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--energy", type=Path, default=DEFAULT_ENERGY)
    parser.add_argument("--convergence", type=Path, default=DEFAULT_CONVERGENCE)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "manuscript"
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    manifest = render(
        audit_path=args.audit,
        summary_path=args.summary,
        dimension_path=args.dimension,
        coverage_path=args.coverage,
        energy_path=args.energy,
        convergence_path=args.convergence,
        output_dir=args.output_dir,
        no_plots=args.no_plots,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
