#!/usr/bin/env python3
"""Stage U4 Publication Table & Vector Figure Generator for INFORMS *Operations Research*.

Reads strictly from sealed Stage P0, Stage P1, and Stage U3 CSV ledgers and JSON
diagnostics (`v2_or_p1_diagnostics_v1`, `v2_or_p1_mech_v1`, `v2_or_u3_confirm_v1`,
`v2_or_u3_messy_archive_v1`, `v2_or_u3_live_sumo_v1`, `v2_or_p0_reaudit_v1`).
Adds corrected KG and operating-policy tables from the independent panel readouts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t
from scipy.stats import wilcoxon

SC_ROOT = Path(__file__).resolve().parents[1]
ART_DIR = SC_ROOT / "paper_artifacts"
MS_DIR = SC_ROOT / "manuscript_v2_or"


FAMILY_DISPLAY = {
    "Inventory_CrossHorizon": "Inventory DES ($M_{\\mathrm{tgt}}\\!=\\!16,\\, d_t\\!\\in\\!\\{192,240\\}$)",
    "Queueing_CrossHorizon": "Queueing DES ($M_{\\mathrm{tgt}}\\!=\\!16,\\, d_t\\!\\in\\!\\{192,240\\}$)",
    "OPSD_EuroGrid": "OPSD Grid ($M_{\\mathrm{tgt}}\\!=\\!14,\\, d_t\\!=\\!336$)",
    "Traffic_Ingolstadt": "Traffic DES ($M_{\\mathrm{tgt}}\\!=\\!16,\\, d_t\\!=\\!44$)",
}

SHORT_FAM = {
    "Inventory_CrossHorizon": "Inventory",
    "Queueing_CrossHorizon": "Queueing",
    "OPSD_EuroGrid": "OPSD Grid",
    "Traffic_Ingolstadt": "Traffic-21",
    "Inventory_CrossHorizon_Messy": "Inventory",
    "Queueing_CrossHorizon_Messy": "Queueing",
    "OPSD_EuroGrid_Messy": "OPSD Grid",
    "Traffic_Ingolstadt_Messy": "Traffic-21",
}

PLOT_TITLES = {
    "Inventory_CrossHorizon": "(a) Dual-Sourcing Inventory DES ($M_{\\mathrm{tgt}}=16$)",
    "Queueing_CrossHorizon": "(b) Nonstationary Queueing DES ($M_{\\mathrm{tgt}}=16$)",
    "OPSD_EuroGrid": "(c) ENTSO-E Grid Storage ($M_{\\mathrm{tgt}}=14$, 24h Buffer)",
    "Traffic_Ingolstadt": "(d) Ingolstadt-21 Signal DES ($M_{\\mathrm{tgt}}=16$, Regime Shift)",
}


def _load_ledger(campaign: str) -> list[dict[str, str]]:
    with (ART_DIR / campaign / "per_run_ledger.csv").open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _holm_adjust(p_map: dict[str, float]) -> dict[str, float]:
    items = sorted(p_map.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, float] = {}
    rmax = 0.0
    for idx, (k, p_raw) in enumerate(items):
        rmax = max(rmax, min(1.0, float(p_raw) * float(m - idx)))
        out[k] = rmax
    return out


def _fmt_mean_se(mean: float, se: float) -> str:
    if abs(mean) < 0.15:
        return f"${mean:.4f} \\pm {se:.4f}$"
    return f"${mean:.3f} \\pm {se:.3f}$"


def _fmt_p(p: float | None, *, exploratory: bool = False) -> str:
    if p is None:
        return "---"
    if p < 1e-4:
        value = "<10^{-4}"
    elif p < 1e-3:
        value = f"{p:.4f}"
    else:
        value = f"{p:.3f}"
    mark = "{}^{\\dagger}" if exploratory else ""
    return f"${value}{mark}$"


def _disp_method(m: str) -> str:
    if m == "GP-FTA-KG":
        return "\\textbf{GP-FTA-KG}"
    if m == "GP-FTA-TS":
        return "\\textbf{GP-FTA-TS}"
    return f"\\texttt{{{m}}}"


def _audit_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    certified = [row for row in rows if row["certified"] == "True"]
    same_law = [
        float(row["verif_law_mc_viol_prob"])
        for row in certified
        if row.get("verif_law_mc_viol_prob", "")
    ]
    classes = [row["safety_audit_class"] for row in certified]
    classified = {
        "audit_safe": classes.count("confidently_safe"),
        "audit_ambig": classes.count("boundary_ambiguous"),
        "audit_unsafe": classes.count("confidently_unsafe"),
    }
    return {
        "n_runs": len(rows),
        "n_certified": len(certified),
        "n_abstained": len(rows) - len(certified),
        "same_law_scored": len(same_law),
        "same_law_exceed": sum(p > 0.05 for p in same_law),
        "audit_unclassified": len(certified) - sum(classified.values()),
        **classified,
    }


def _certified_count(stats: dict[str, Any]) -> str:
    return f"{stats['n_certified']}/{stats['n_runs']}"


def _format_audit_classes(stats: dict[str, Any]) -> str:
    if stats["n_certified"] == stats["audit_unclassified"]:
        return "---"
    return f"{stats['audit_safe']}/{stats['audit_ambig']}/{stats['audit_unsafe']}"


def _task_agg(rows: list[dict[str, str]], family: str) -> dict[str, Any]:
    frows = [r for r in rows if r["family"] == family]
    tasks = sorted({r["task_id"] for r in frows})
    budgets = sorted({int(r["n_online_steps"]) for r in frows})
    methods = sorted({r["method_name"] for r in frows})

    stop_rows = [r for r in frows if r["method_name"] == "STOP"]
    out: dict[int, dict[str, dict[str, Any]]] = {}

    for b in budgets:
        out[b] = {}
        for m in methods:
            mrows = stop_rows if m == "STOP" else [r for r in frows if r["method_name"] == m and int(r["n_online_steps"]) == b]
            if not mrows:
                continue
            loss_vec = np.array([np.mean([float(x["status_quo_loss"]) for x in mrows if x["task_id"] == t]) for t in tasks])
            fr10_vec = np.array([np.mean([float(x["fixed_reject_loss_10pct"]) for x in mrows if x["task_id"] == t]) for t in tasks])
            cert_vec = np.array([np.mean([1.0 if x["certified"] == "True" else 0.0 for x in mrows if x["task_id"] == t]) for t in tasks])
            call_vec = np.array([np.mean([float(x["total_target_calls"]) for x in mrows if x["task_id"] == t]) for t in tasks])
            prim_vec = np.array([np.mean([float(x["primary_slot0_mc_obj"]) for x in mrows if x["task_id"] == t]) for t in tasks])
            anc_vec = np.array([np.mean([float(x["anchor_mc_obj"]) for x in mrows if x["task_id"] == t]) for t in tasks])
            out[b][m] = {
                "tasks": tasks,
                "loss_vec": loss_vec,
                "mean_loss": float(np.mean(loss_vec)),
                "se_loss": float(np.std(loss_vec, ddof=1) / np.sqrt(len(tasks))),
                "fr10_vec": fr10_vec,
                "mean_fr10": float(np.mean(fr10_vec)),
                "se_fr10": float(np.std(fr10_vec, ddof=1) / np.sqrt(len(tasks))),
                "cert_rate": float(np.mean(cert_vec)),
                "call_vec": call_vec,
                "mean_calls": float(np.mean(call_vec)),
                "se_calls": float(np.std(call_vec, ddof=1) / np.sqrt(len(tasks))),
                "mean_prim": float(np.mean(prim_vec)),
                "mean_anc": float(np.mean(anc_vec)),
                "anc_vec": anc_vec,
                **_audit_counts(mrows),
                "mean_wall": float(np.mean([float(x["online_wall_time_sec"]) for x in mrows])),
                "mean_pre": float(np.mean([float(x["source_pretrain_wall_sec"]) for x in mrows])),
            }
    return {"tasks": tasks, "budgets": budgets, "by_budget": out}


def build_table_main(rows_u3: list[dict[str, str]]) -> str:
    fams = ["Inventory_CrossHorizon", "Queueing_CrossHorizon", "OPSD_EuroGrid", "Traffic_Ingolstadt"]
    ext_baselines = ["SCBO-on-Psi", "RGPE-on-Psi", "SCBO", "MAP-SAAS-CBO", "HetGP-KG", "RGPE", "HyperBO-EB", "Target-SPSA"]
    controls = ["GP-FTA-TS", "PublicPsi-TargetGP-TS", "SCBO-on-Psi", "RGPE-on-Psi"]
    nat_holm_set = ext_baselines[2:]
    display_order = ["GP-FTA-KG", *controls, *nat_holm_set, "STOP"]

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Frozen U3 shared-panel comparisons across four operational families (14 or 16 tasks, two paired streams per task). Reported means, observed search-plus-verification calls, and task comparisons are for $N=60$ except the $N=15$ loss column and the separately executed zero-step STOP. $\\mathcal{L}_{10\\%}$ applies an illustrative delay cost on abstention.}\\label{tab:main_confirm}",
        "\\scriptsize",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llccccccc}",
        "\\toprule",
        "\\textbf{Family} & \\textbf{Method} & \\textbf{$N=15$ $\\mathcal{L}_{\\mathrm{SQ}}$} & \\textbf{$N=60$ $\\mathcal{L}_{\\mathrm{SQ}}$} & \\textbf{$N=60$ $\\mathcal{L}_{10\\%}$} & \\textbf{Cert./runs} & \\textbf{Actual calls} & \\textbf{KG gain (W/L/T)} & \\textbf{$p$} \\\\",
        "\\midrule",
    ]

    for f_idx, fam in enumerate(fams):
        agg = _task_agg(rows_u3, fam)["by_budget"]
        b15, b60 = agg[15], agg[60]
        ref60 = b60["GP-FTA-KG"]["loss_vec"]
        fmt = ".4f" if fam.startswith("OPSD") else ".3f"

        raw_p_nat: dict[str, float] = {}
        comp_info: dict[str, tuple[float, int, int, int, float]] = {}
        for m in display_order[1:]:
            diff = b60[m]["loss_vec"] - ref60
            nz = diff[np.abs(diff) > 1e-12]
            p1 = float(wilcoxon(nz, alternative="greater").pvalue) if len(nz) >= 3 else 1.0
            if m in nat_holm_set:
                raw_p_nat[m] = p1
            w = int(np.sum(diff > 1e-9))
            l = int(np.sum(diff < -1e-9))
            t = len(diff) - w - l
            rpct = 100.0 * float(np.mean(diff / np.maximum(np.abs(b60[m]["loss_vec"]), 1e-8)))
            comp_info[m] = (rpct, w, l, t, p1)
        holm_nat = _holm_adjust(raw_p_nat)

        min_sq60 = min(b60[m]["mean_loss"] for m in display_order)
        min_fr60 = min(b60[m]["mean_fr10"] for m in display_order)

        for r_i, m in enumerate(display_order):
            fam_col = f"\\multirow{{{len(display_order)}}}{{*}}{{{FAMILY_DISPLAY[fam]}}}" if r_i == 0 else ""
            m_disp = _disp_method(m)
            c15 = _fmt_mean_se(b15[m]["mean_loss"], b15[m]["se_loss"])
            c60 = _fmt_mean_se(b60[m]["mean_loss"], b60[m]["se_loss"])
            fr60_val = b60[m]["mean_fr10"]
            fr60 = f"$\\mathbf{{{fr60_val:{fmt}}}}$" if abs(fr60_val - min_fr60) < 1e-12 else f"${fr60_val:{fmt}}$"
            if abs(b60[m]["mean_loss"] - min_sq60) < 1e-12:
                c60 = f"$\\mathbf{{{c60[1:-1]}}}$"
            cert_str = _certified_count(b60[m])
            call_str = f"{b60[m]['mean_calls']:.1f}"

            if m == "GP-FTA-KG":
                gain_str, p_str = "---", "---"
            elif m in nat_holm_set:
                rpct, w, l, t, _ = comp_info[m]
                sgn = "+" if rpct >= 0 else ""
                gain_str = f"${sgn}{rpct:.1f}\\%$ ({w}/{l}/{t})"
                p_str = _fmt_p(holm_nat[m])
            else:
                rpct, w, l, t, p_raw = comp_info[m]
                sgn = "+" if rpct >= 0 else ""
                gain_str = f"${sgn}{rpct:.1f}\\%$ ({w}/{l}/{t})"
                p_str = _fmt_p(p_raw, exploratory=True)
            lines.append(f"{fam_col} & {m_disp} & {c15} & {c60} & {fr60} & {cert_str} & {call_str} & {gain_str} & {p_str} \\\\")
        if f_idx < len(fams) - 1:
            lines.append("\\midrule")

    lines.extend([
        "\\bottomrule",
        "\\multicolumn{9}{l}{\\scriptsize One-sided paired Wilcoxon $p$, Holm-adjusted across the six raw-policy comparators within each family. $^{\\dagger}$Exploratory unadjusted comparison.}\\\\",
        "\\end{tabular}%",
        "}",
        "\\end{table}",
    ])
    return "\n".join(lines)


def build_table_ablations(rows_u3: list[dict[str, str]]) -> str:
    fams = ["Inventory_CrossHorizon", "Queueing_CrossHorizon", "OPSD_EuroGrid", "Traffic_Ingolstadt"]
    ctrl_arms = [
        "GP-FTA-KG",
        "GP-FTA-TS",
        "RGPE-on-Psi",
        "SCBO-on-Psi",
        "PublicPsi-TargetGP-TS",
        "WarmStart-PatternSearch",
        "Untempered-PriorResid",
        "Frozen-AFTER",
        "STOP",
    ]
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Frozen U3 controls sharing the public representation and unclipped grid sensitivity. Two-sided $p$ values compare $N=60$ status-quo loss with GP-FTA-KG and are unadjusted. STOP is the same zero-step run throughout.}\\label{tab:ablations}",
        "\\scriptsize",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llccccccc}",
        "\\toprule",
        "\\textbf{Family} & \\textbf{Mechanism / Same-$\\psi$ Control} & \\textbf{$N=8$} & \\textbf{$N=15$} & \\textbf{$N=30$} & \\textbf{$N=60$ $\\mathcal{L}_{\\mathrm{SQ}}$} & \\textbf{$N=60$ $\\mathcal{L}_{10\\%}$} & \\textbf{$N=60$ Cert.\\%} & \\textbf{Two-Sided $p$} \\\\",
        "\\midrule",
    ]

    for fam in fams:
        agg = _task_agg(rows_u3, fam)["by_budget"]
        fmt = ".4f" if fam.startswith("OPSD") else ".3f"
        mins = {b: min(agg[b][m]["mean_loss"] for m in ctrl_arms) for b in (8, 15, 30, 60)}
        min_fr = min(agg[60][m]["mean_fr10"] for m in ctrl_arms)
        ref60 = agg[60]["GP-FTA-KG"]["loss_vec"]

        for r_i, m in enumerate(ctrl_arms):
            fam_col = f"\\multirow{{{len(ctrl_arms)}}}{{*}}{{{SHORT_FAM[fam]}}}" if r_i == 0 else ""
            m_disp = _disp_method(m)
            vals = []
            for b in (8, 15, 30, 60):
                v = agg[b][m]["mean_loss"]
                s = f"$\\mathbf{{{v:{fmt}}}}$" if abs(v - mins[b]) < 1e-12 else f"${v:{fmt}}$"
                vals.append(s)
            frv = agg[60][m]["mean_fr10"]
            s_fr = f"$\\mathbf{{{frv:{fmt}}}}$" if abs(frv - min_fr) < 1e-12 else f"${frv:{fmt}}$"
            cert60 = f"{100.0 * agg[60][m]['cert_rate']:.1f}\\%"
            if m == "GP-FTA-KG":
                p2_str = "---"
            else:
                d = agg[60][m]["loss_vec"] - ref60
                nz = d[np.abs(d) > 1e-12]
                p2 = float(wilcoxon(nz, alternative="two-sided").pvalue) if len(nz) >= 3 else 1.0
                p2_str = _fmt_p(p2)
            lines.append(f"{fam_col} & {m_disp} & {vals[0]} & {vals[1]} & {vals[2]} & {vals[3]} & {s_fr} & {cert60} & {p2_str} \\\\")
        lines.append("\\midrule")

    # Panel B: Unclamped OPSD raw-tail sensitivity
    u_agg = _task_agg(rows_u3, "OPSD_EuroGrid_Unclamped")["by_budget"]
    u_arms = ["GP-FTA-KG", "GP-FTA-TS", "RGPE-on-Psi", "SCBO-on-Psi", "PublicPsi-TargetGP-TS", "WarmStart-PatternSearch", "SCBO", "RGPE", "STOP"]
    min15 = min(u_agg[15][m]["mean_loss"] for m in u_arms)
    min60 = min(u_agg[60][m]["mean_loss"] for m in u_arms)
    min_fr_u = min(u_agg[60][m]["mean_fr10"] for m in u_arms)
    ref_u60 = u_agg[60]["GP-FTA-KG"]["loss_vec"]
    for r_i, m in enumerate(u_arms):
        fam_col = f"\\multirow{{{len(u_arms)}}}{{*}}{{OPSD Raw Tails}}" if r_i == 0 else ""
        m_disp = _disp_method(m)
        v15 = u_agg[15][m]["mean_loss"]
        v60 = u_agg[60][m]["mean_loss"]
        fr60 = u_agg[60][m]["mean_fr10"]
        s15 = f"$\\mathbf{{{v15:.4f}}}$" if abs(v15 - min15) < 1e-12 else f"${v15:.4f}$"
        s60 = f"$\\mathbf{{{v60:.4f}}}$" if abs(v60 - min60) < 1e-12 else f"${v60:.4f}$"
        sfr = f"$\\mathbf{{{fr60:.4f}}}$" if abs(fr60 - min_fr_u) < 1e-12 else f"${fr60:.4f}$"
        cert60 = f"{100.0 * u_agg[60][m]['cert_rate']:.1f}\\%"
        if m == "GP-FTA-KG":
            p2_str = "---"
        else:
            d = u_agg[60][m]["loss_vec"] - ref_u60
            nz = d[np.abs(d) > 1e-12]
            p2 = float(wilcoxon(nz, alternative="two-sided").pvalue) if len(nz) >= 3 else 1.0
            p2_str = _fmt_p(p2)
        lines.append(f"{fam_col} & {m_disp} & --- & {s15} & --- & {s60} & {sfr} & {cert60} & {p2_str} \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"])
    return "\n".join(lines)


def build_table_messy_archive(rows_messy: list[dict[str, str]]) -> str:
    fams = [
        "Inventory_CrossHorizon_Messy",
        "Queueing_CrossHorizon_Messy",
        "OPSD_EuroGrid_Messy",
        "Traffic_Ingolstadt_Messy",
    ]
    arms = [
        "GP-FTA-KG",
        "GP-FTA-TS",
        "GP-FTA-NoDebias",
        "RGPE-on-Psi",
        "WarmStart-PatternSearch",
        "PublicPsi-TargetGP-TS",
        "RGPE",
        "HyperBO-EB",
        "STOP",
    ]
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Heterogeneous simulated source archives on the shared U3 target panel ($2{,}108$ runs). Logs combine single replications, unequal budgets, incomplete solver coverage, and heterogeneous historical risk tolerances. Two-sided $p$ values compare status-quo loss with frozen GP-FTA-KG and are unadjusted. Online wall time is for $N=60$.}\\label{tab:messy_archive}",
        "\\scriptsize",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llcccccc}",
        "\\toprule",
        "\\textbf{Family} & \\textbf{Method} & \\textbf{$N=15$ $\\mathcal{L}_{\\mathrm{SQ}}$} & \\textbf{$N=60$ $\\mathcal{L}_{\\mathrm{SQ}}$} & \\textbf{$N=60$ $\\mathcal{L}_{10\\%}$} & \\textbf{$N=60$ Cert.\\%} & \\textbf{Two-Sided $p$ ($N=60$)} & \\textbf{Online Wall (s)} \\\\",
        "\\midrule",
    ]
    for f_idx, fam in enumerate(fams):
        agg = _task_agg(rows_messy, fam)["by_budget"]
        fmt = ".4f" if "OPSD" in fam else ".3f"
        min15 = min(agg[15][m]["mean_loss"] for m in arms)
        min60 = min(agg[60][m]["mean_loss"] for m in arms)
        minfr = min(agg[60][m]["mean_fr10"] for m in arms)
        ref60 = agg[60]["GP-FTA-KG"]["loss_vec"]

        for r_i, m in enumerate(arms):
            fam_col = f"\\multirow{{{len(arms)}}}{{*}}{{{SHORT_FAM[fam]}}}" if r_i == 0 else ""
            m_disp = _disp_method(m)
            v15, v60, fr60 = agg[15][m]["mean_loss"], agg[60][m]["mean_loss"], agg[60][m]["mean_fr10"]
            s15 = f"$\\mathbf{{{v15:{fmt}}}}$" if abs(v15 - min15) < 1e-12 else f"${v15:{fmt}}$"
            s60 = f"$\\mathbf{{{v60:{fmt}}}}$" if abs(v60 - min60) < 1e-12 else f"${v60:{fmt}}$"
            sfr = f"$\\mathbf{{{fr60:{fmt}}}}$" if abs(fr60 - minfr) < 1e-12 else f"${fr60:{fmt}}$"
            cert = f"{100.0 * agg[60][m]['cert_rate']:.1f}\\%"
            wall = f"{agg[60][m]['mean_wall']:.1f}"
            if m == "GP-FTA-KG":
                p_str = "---"
            else:
                d = agg[60][m]["loss_vec"] - ref60
                nz = d[np.abs(d) > 1e-12]
                p2 = float(wilcoxon(nz, alternative="two-sided").pvalue) if len(nz) >= 3 else 1.0
                p_str = _fmt_p(p2)
            lines.append(f"{fam_col} & {m_disp} & {s15} & {s60} & {sfr} & {cert} & {p_str} & {wall} \\\\")
        if f_idx < len(fams) - 1:
            lines.append("\\midrule")

    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"])
    return "\n".join(lines)


def build_table_live_sumo(rows_sumo_u3: list[dict[str, str]], rows_sumo_p0: list[dict[str, str]]) -> str:
    agg_u3 = _task_agg(rows_sumo_u3, "Ingolstadt21_LiveSUMO_Calibrated")["by_budget"][15]
    u3_order = sorted(agg_u3.keys(), key=lambda k: agg_u3[k]["mean_loss"])
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Microscopic \\texttt{Ingolstadt21} SUMO evaluation ($N=15$). The calibrated U3 panel has eight threshold and weight configurations of one network and route, with two simulation streams per configuration. The 40 uncalibrated stress runs were retained from the earlier experiment.}\\label{tab:live_sumo}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{4.6pt}",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "\\textbf{Regime} & \\textbf{Method} & \\textbf{Status-Quo Loss} & \\textbf{Challenger $J_t^{\\ddagger}$} & \\textbf{Cert./runs} & \\textbf{S/A/U$^*$} & \\textbf{Actual Calls} \\\\",
        "\\midrule",
    ]
    min_loss = min(agg_u3[m]["mean_loss"] for m in u3_order)
    for r_i, m in enumerate(u3_order):
        md = agg_u3[m]
        reg_col = f"\\multirow{{{len(u3_order)}}}{{*}}{{Calibrated ($144$ runs)}}" if r_i == 0 else ""
        m_disp = _disp_method(m)
        l_str = f"${md['mean_loss']:.4f} \\pm {md['se_loss']:.4f}$"
        if abs(md["mean_loss"] - min_loss) < 1e-12:
            l_str = f"$\\mathbf{{{md['mean_loss']:.4f}}} \\pm {md['se_loss']:.4f}$"
        lines.append(
            f"{reg_col} & {m_disp} & {l_str} & ${md['mean_prim']:.4f}$ & "
            f"{_certified_count(md)} & {_format_audit_classes(md)} & "
            f"{md['mean_calls']:.1f} \\\\"
        )

    lines.append("\\midrule")
    p0_by_m: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows_sumo_p0:
        p0_by_m[r["method_name"]].append(r)
    p0_order = ["TLS-OLH-KG", "SCBO", "RGPE", "Target-SPSA", "STOP"]
    for r_i, m in enumerate(p0_order):
        mrows = p0_by_m[m]
        reg_col = f"\\multirow{{{len(p0_order)}}}{{*}}{{Stress Test ($40$ runs)}}" if r_i == 0 else ""
        m_disp = f"\\texttt{{{m}}} (Legacy)" if m == "TLS-OLH-KG" else f"\\texttt{{{m}}}"
        sq = float(np.mean([float(x["status_quo_loss"]) for x in mrows]))
        pr = float(np.mean([float(x["uncertified_selected_candidate_obj"]) for x in mrows]))
        counts = _audit_counts(mrows)
        calls = float(np.mean([float(x["total_target_calls"]) for x in mrows]))
        lines.append(f"{reg_col} & {m_disp} & ${sq:.4f}$ & ${pr:.4f}$ & {_certified_count(counts)} & --- & {calls:.1f} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\par\\smallskip\\parbox{\\textwidth}{\\scriptsize $^*$S/A/U counts certified deployments classified as confidently safe, boundary-ambiguous, or confidently unsafe by an independent audit. A dash indicates no audited deployment. $^{\\ddagger}$Calibrated runs use the frozen primary. Stress runs use the saved uncertified candidate.}",
        "\\end{table}",
    ])
    return "\n".join(lines)


def build_table_verification_audit(
    rows_u3: list[dict[str, str]],
    rows_messy: list[dict[str, str]],
    rows_sumo_u3: list[dict[str, str]],
    rows_sumo_p0: list[dict[str, str]],
) -> str:
    fta = {"GP-FTA-KG", "GP-FTA-TS"}
    rows_opsd = [row for row in rows_u3 if row["family"].startswith("OPSD")]
    rows_opsd_fta = [row for row in rows_opsd if row["method_name"] in fta]
    strata = [
        ("Main and raw-tail suite, all methods", rows_u3),
        ("Grid, both variants, all methods", rows_opsd),
        ("Grid, clipped, GP-FTA pair", [row for row in rows_opsd_fta if row["family"] == "OPSD_EuroGrid"]),
        ("Grid, unclipped, GP-FTA pair", [row for row in rows_opsd_fta if row["family"] == "OPSD_EuroGrid_Unclamped"]),
        ("Heterogeneous archive, all methods", rows_messy),
        ("Heterogeneous archive, grid only", [row for row in rows_messy if row["family"].startswith("OPSD")]),
        ("Calibrated SUMO, all methods", rows_sumo_u3),
        ("Calibrated SUMO, GP-FTA-KG", [row for row in rows_sumo_u3 if row["method_name"] == "GP-FTA-KG"]),
        ("Uncalibrated SUMO stress runs", rows_sumo_p0),
    ]
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Verification coverage and independent audit outcomes from the sealed ledgers. Rows can overlap. Same-law exceedances use the Monte Carlo point estimate and count only certified runs with a recorded verification-law recheck. S/A/U and missing are defined over certified deployments. S/A/U denote confident safety, boundary ambiguity and confident unsafety based on two-sided 95\\% Clopper--Pearson intervals. A dash indicates that the endpoint has no observations.}\\label{tab:verification_audit}",
        "\\scriptsize",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "\\textbf{Panel and subset} & \\textbf{Cert./runs} & \\textbf{Abstained} & \\textbf{Same-law exceed./scored} & \\textbf{Independent S/A/U} & \\textbf{Unclassified cert.} \\\\",
        "\\midrule",
    ]
    for title, rows in strata:
        stats = _audit_counts(rows)
        scored = stats["same_law_scored"]
        same_law = f"{stats['same_law_exceed']}/{scored}" if scored else "---"
        lines.append(
            f"{title} & {_certified_count(stats)} & {stats['n_abstained']} & "
            f"{same_law} & {_format_audit_classes(stats)} & {stats['audit_unclassified']} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"])
    return "\n".join(lines)


def build_table_independent_panel(readout: dict[str, Any]) -> str:
    families = ("Inventory_CrossHorizon", "Queueing_CrossHorizon", "Traffic_Ingolstadt")
    methods = ("GP-FTA-KG", "GP-FTA-TS", "PublicPsi-TargetGP-TS")
    class_order = ("confidently_safe", "boundary_ambiguous", "confidently_unsafe")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Corrected KG and matched controls on the independent target panel at $N=60$. "
        r"Each family contains 24 new tasks with two paired evaluation streams per task. "
        r"Calls are mean realized search and terminal verification replications per run.}\label{tab:independent_panel}",
        r"\scriptsize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrccccc}",
        r"\toprule",
        r"\textbf{Family} & \textbf{Method} & \textbf{$\mathcal{L}_{\mathrm{SQ}}$} & \textbf{Calls} & "
        r"\textbf{Cert./runs} & \textbf{S/A/U} & \textbf{Exceed./audited} & "
        r"\textbf{Control $-$ KG [95\% CI]} & \textbf{Holm $p$} \\",
    ]
    for family in families:
        lines.append(r"\midrule")
        group = readout["families"][family]
        for index, method in enumerate(methods):
            arm = group["arm_results"][method]
            audited = arm["audited_deployments"]
            classes = arm["safety_audit_classes"]
            safety = "/".join(str(classes.get(category, 0)) for category in class_order) if audited else "---"
            exceed = f"{arm['audit_viol_prob_point_estimates_over_threshold']}/{audited}" if audited else "---"
            if method == methods[0]:
                gain, p_value = "---", "---"
            else:
                paired = group["paired_loss"][method]
                lower, upper = paired["ci95"]
                gain = rf"${paired['comparator_minus_kg_mean']:.4f}\ [{lower:.4f},\,{upper:.4f}]$"
                p_value = _fmt_p(paired["paired_t_p_holm_six_tests"])
            family_col = rf"\multirow{{3}}{{*}}{{{SHORT_FAM[family]}}}" if index == 0 else ""
            lines.append(
                f"{family_col} & {_disp_method(method)} & ${arm['mean_status_quo_loss']:.4f}$ & "
                f"{arm['mean_total_target_calls']:.1f} & {arm['certified']}/{arm['runs']} & "
                rf"{safety} & {exceed} & {gain} & {p_value} \\"
            )
    lines.extend([
        r"\bottomrule", r"\end{tabular}%", "}",
        r"\par\smallskip\parbox{\textwidth}{\scriptsize "
        r"S/A/U are counts of audited certified deployments classified as confidently safe, boundary-ambiguous, "
        r"or confidently unsafe using two-sided 95\% Clopper--Pearson intervals. Exceedances count audited "
        r"certified deployments with violation point estimates above 5\%. Positive loss differences favor KG. "
        r"The paired $t$ intervals and two-sided tests average the two streams within each task. "
        r"Holm adjusts all six tests together.}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def build_table_operating_policy(readout: dict[str, Any]) -> str:
    methods = ("GP-FTA-KG", "GP-FTA-TS", "PublicPsi-TargetGP-TS")
    classes = ("confidently_safe", "boundary_ambiguous", "confidently_unsafe")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Audited risk of each operating policy on the independent panel. "
        r"A run deploys its certified nominee or retains its status-quo policy on abstention. "
        r"Classifications use two-sided 95\% Clopper--Pearson intervals relative to the 5\% threshold. "
        r"An audit of the status-quo policy is reused across methods and streams within a task.}\label{tab:operating_policy}",
        r"\scriptsize",
        r"\begin{tabular}{llccrrr}",
        r"\toprule",
        r"\textbf{Family} & \textbf{Method} & \textbf{New certificates} & \textbf{Status quo} & "
        r"\textbf{Safe} & \textbf{Ambiguous} & \textbf{Unsafe} \\",
    ]
    for family in ("Inventory_CrossHorizon", "Queueing_CrossHorizon", "Traffic_Ingolstadt"):
        lines.append(r"\midrule")
        for index, method in enumerate(methods):
            arm = readout["families"][family]["arms"][method]
            family_label = "Traffic DES" if family == "Traffic_Ingolstadt" else SHORT_FAM[family]
            family_col = rf"\multirow{{3}}{{*}}{{{family_label}}}" if index == 0 else ""
            safe, ambiguous, unsafe = (arm["operating_policy_classes"].get(label, 0) for label in classes)
            lines.append(
                f"{family_col} & {_disp_method(method)} & {arm['certified']}/{arm['runs']} & "
                rf"{arm['retained_anchor']} & {safe} & {ambiguous} & {unsafe} \\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)



def build_referee_tables(readout: dict[str, Any]) -> tuple[str, str]:
    names = {
        "TargetWeak-GateKG": "Weak",
        "HistoryDiag-GateKG": "Diagonal",
        "HistoryFull-GateKG": "Full",
        "HistoryFull-FeasibleKG": "Feasible KG",
        "InitialRelease-GateKG": "Early release",
    }
    policy = [
        r"\begin{table}[htbp]\centering",
        r"\caption{Matched runs on new queueing and traffic tasks. Operating loss and audit classes",
        r"include the status quo on abstention. Each row contains 48 runs in 24 paired tasks.",
        r"Early release is an exact replay of the selected weak or full path.}\label{tab:matched_policy}",
        r"\small\begin{tabular}{llrcccc}\toprule",
        r"Family & Prior / rule & Loss & Terminal calls & Certified & Safe / unclear / unsafe & Violation rate \\",
    ]
    comparisons = [
        r"\begin{table}[htbp]\centering",
        r"\caption{Paired operating loss, averaged within each task. Positive differences favor the latter arm.",
        r"Intervals are individual 95\% paired $t$ intervals. Holm adjustment uses all ten tests.}\label{tab:matched_contrasts}",
        r"\small\begin{tabular}{llccc}\toprule",
        r"Family & Former $\to$ latter & Difference [interval] & Wins / losses / ties & Holm $p$ \\",
    ]
    for family, values in readout["families"].items():
        policy.append(r"\midrule")
        comparisons.append(r"\midrule")
        short_family = "Queueing" if family.startswith("Queueing") else "Traffic"
        for j, (arm, row) in enumerate(values["arms"].items()):
            counts = "/".join(str(row["operating_audits"].get(key, 0)) for key in
                              ("confidently_safe", "boundary_ambiguous", "confidently_unsafe"))
            fam = short_family if j == 0 else ""
            policy.append(
                f"{fam} & {names[arm]} & {row['mean_operating_loss']:.4f} & "
                f"{row['mean_terminal_calls']:.1f} & {row['certified']}/{row['runs']} & {counts} & "
                rf"{100*row['mean_operating_violation_rate']:.2f}\% \\"
            )
        for j, (label, row) in enumerate(values["comparisons"].items()):
            before, after = label.split(" to ")
            lo, hi = row["ci95"]
            wins = "/".join(map(str, row["latter_wins_losses_ties"]))
            fam = short_family if j == 0 else ""
            comparisons.append(
                rf"{fam} & {names[before]} $\to$ {names[after]} & "
                rf"${row['improvement']:.4f}\ [{lo:.4f},\,{hi:.4f}]$ & {wins} & "
                rf"{_fmt_p(row['p_holm'])} \\"
            )
    end = [r"\bottomrule\end{tabular}", r"\end{table}"]
    return "\n".join(policy + end) + "\n", "\n".join(comparisons + end) + "\n"


def build_referee_variances(readout: dict[str, Any]) -> str:
    lines = [
        r"\begin{table}[htbp]\centering",
        r"\caption{Mean squared variance error against independent audits, by family and role.}\label{tab:matched_variances}",
        r"\small\begin{tabular}{llrcccc}\toprule",
        r"& & & \multicolumn{2}{c}{Objective variance} & \multicolumn{2}{c}{Constraint variance} \\",
        r"Family & Policy role & Runs & Fixed batch & Pooled & Fixed batch & Pooled \\",
    ]
    for family, values in readout["families"].items():
        lines.append(r"\midrule")
        short_family = "Queueing" if family.startswith("Queueing") else "Traffic"
        paid_arms = [arm for name, arm in values["arms"].items() if name != "InitialRelease-GateKG"]
        for role, label in (("primary", "Primary"), ("deployed", "Certified")):
            roles = [arm["variance_diagnostics"][role] for arm in paid_arms]
            n = sum(row["n"] for row in roles)
            cells = []
            for head in ("obj_var", "con_var"):
                for estimator in ("fixed_batch", "pooled"):
                    mse = sum(row["n"] * row["mean_squared_errors"][estimator][head] for row in roles if row["n"]) / n
                    base, exponent = f"{mse:.3e}".split("e")
                    cells.append(rf"${base}\times 10^{{{int(exponent)}}}$")
            family_cell = short_family if role == "primary" else ""
            lines.append(" & ".join([family_cell, label, str(n), *cells]) + r" \\")
    lines.extend([r"\bottomrule\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def build_terminal_replay_table(readout: dict[str, Any]) -> str:
    lines = [
        r"\begin{table}[htbp]\centering",
        r"\caption{First-slot and full-rule operating loss, certification counts out of 48",
        r"and terminal calls on identical realized search paths. First-slot replay",
        r"uses 112 calls.}\label{tab:terminal_replay}",
        r"\small\begin{tabular}{llrrccr}\toprule",
        r"& & \multicolumn{2}{c}{Mean loss} & \multicolumn{2}{c}{Certified} & Mean calls \\",
        r"Family & Arm & Slot 0 & Full rule & Slot 0 & Full rule & Full rule \\",
    ]
    labels = {"TargetWeak-GateKG": "Weak", "HistoryDiag-GateKG": "Diagonal",
              "HistoryFull-GateKG": "Full (gate power)", "HistoryFull-FeasibleKG": "Full (feasibility)"}
    for family, values in readout["families"].items():
        lines.append(r"\midrule")
        family_name = "Queueing" if family.startswith("Queueing") else "Traffic"
        for name, label in labels.items():
            arm = values["arms"][name]
            first = arm["first_slot"]
            full = arm["full_terminal"]
            lines.append(
                f"{family_name} & {label} & {first['mean_loss']:.5f} & {full['mean_loss']:.5f} & "
                rf"{first['certified']} & {full['certified']} & {full['mean_terminal_calls']:.2f} \\"
            )
            family_name = ""
    lines.extend([r"\bottomrule\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def build_query_screen_table(readout: dict[str, Any]) -> str:
    queue = readout["Queueing_CrossHorizon"]
    traffic = readout["Traffic_Ingolstadt"]
    rules = [
        ("Retained queries", "total_retained_queries", "total_queries", False),
        ("Unique full-rule choice", "strict_unique_full_count", "states", True),
        ("Unique choice agrees with matched first slot", "strict_unique_full_equals_matched_first_both_orders", "strict_unique_full_count", True),
        ("Original scorer's choice excluded", "strict_exclusion_actual_first", "states", True),
        ("Matched first-slot choice excluded", "strict_exclusion_matching_first_both_orders", "stable_marginal_24_48", True),
        ("Different choices among numerically stable states", "stable_both_disagreement", "stable_both_count", True),
    ]
    lines = [
        r"\begin{table}[htbp]\centering",
        r"\caption{Query screen and numerical comparisons in the 24 matched first-query states.",
        r"The matched first-slot exclusions use agreement under both marginal rules.}",
        r"\label{tab:query_screen}",
        r"\small\begin{tabular}{lrr}\toprule",
        r"Measure & Queueing & Traffic \\",
        r"\midrule",
    ]
    for label, key, denominator, refined in rules:
        values = [family["refinement"] if refined else family for family in (queue, traffic)]
        cells = [f"{row[key]}/{row[denominator]}" for row in values]
        lines.append(" & ".join([label, *cells]) + r" \\")
    lines.extend([r"\bottomrule\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def _compute_honest_amortization(rows_u3: list[dict[str, str]], fam: str, b_src: int = 1428) -> dict[str, Any]:
    agg = _task_agg(rows_u3, fam)["by_budget"]
    fta15 = agg[15]["GP-FTA-KG"]
    scbo60 = agg[60]["SCBO"]

    diff_calls = scbo60["call_vec"] - fta15["call_vec"]
    n_t = len(diff_calls)
    mean_diff = float(np.mean(diff_calls))
    se_diff = float(np.std(diff_calls, ddof=1) / np.sqrt(n_t))
    t_crit = float(student_t.ppf(0.975, df=n_t - 1))
    ci_lo = mean_diff - t_crit * se_diff
    ci_hi = mean_diff + t_crit * se_diff

    assert fta15["mean_loss"] < scbo60["mean_loss"], f"Quality dominance violated on {fam}"
    assert mean_diff > 0.0 and ci_lo > 0.0, f"Positive call saving violated on {fam}: {mean_diff}, {ci_lo}"

    m_break_point = math.ceil(b_src / mean_diff)
    m_break_ci95 = math.ceil(b_src / ci_lo)
    return {
        "B_src": b_src,
        "fta15_loss": fta15["mean_loss"],
        "scbo60_loss": scbo60["mean_loss"],
        "fta15_calls": fta15["mean_calls"],
        "scbo60_calls": scbo60["mean_calls"],
        "mean_saving": mean_diff,
        "se_saving": se_diff,
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "m_break_sunk": 1,
        "m_break_point": m_break_point,
        "m_break_ci95_upper": m_break_ci95,
    }


def build_table_amortization(rows_u3: list[dict[str, str]]) -> str:
    fams = ["Inventory_CrossHorizon", "Queueing_CrossHorizon", "OPSD_EuroGrid", "Traffic_Ingolstadt"]
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Illustrative simulation-call amortization against raw-policy SCBO. The table compares observed losses and search-plus-verification calls, charging $1{,}428$ source calls per family in the full-cost scenario. Intervals quantify uncertainty in the paired mean call savings.}\\label{tab:amortization}",
        "\\scriptsize",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "\\textbf{Family} & \\textbf{GP-FTA ($N\\!=\\!15$)} & \\textbf{\\texttt{SCBO} ($N\\!=\\!60$)} & \\textbf{$B_{\\mathrm{FTA},15}$} & \\textbf{$B_{\\mathrm{SCBO},60}$} & \\textbf{Call Saving $\\Delta B$ ($95\\%$ CI)} & \\textbf{$M_{\\mathrm{break}}^{\\mathrm{sunk}}$} & \\textbf{$M_{\\mathrm{break}}^{\\mathrm{full}}$ ($95\\%$ UB)} \\\\",
        "\\midrule",
    ]
    for fam in fams:
        am = _compute_honest_amortization(rows_u3, fam)
        fmt = ".4f" if fam.startswith("OPSD") else ".3f"
        lines.append(
            f"{SHORT_FAM[fam]} & $\\mathbf{{{am['fta15_loss']:{fmt}}}}$ & ${am['scbo60_loss']:{fmt}}$ & "
            f"{am['fta15_calls']:.1f} & {am['scbo60_calls']:.1f} & "
            f"$+{am['mean_saving']:.1f}$ $[+{am['ci95_lo']:.1f}, +{am['ci95_hi']:.1f}]$ & "
            f"$1$ & $\\mathbf{{{am['m_break_point']}}}$ ($\\le {am['m_break_ci95_upper']}$) \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"])
    return "\n".join(lines)


def build_figures(
    fig_dir: Path,
    rows_u3: list[dict[str, str]],
    rows_messy: list[dict[str, str]],
    rows_p1_mech: list[dict[str, str]],
    p1_diag: dict[str, Any],
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8.8,
        "axes.titlesize": 9.3,
        "axes.labelsize": 8.8,
        "legend.fontsize": 7.6,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "pdf.fonttype": 42,
    })

    fams = ["Inventory_CrossHorizon", "Queueing_CrossHorizon", "OPSD_EuroGrid", "Traffic_Ingolstadt"]

    # --- Figure 1: 2x2 Task-Level Status-Quo Loss Scaling Including Same-Psi Controls ---
    palette = [
        ("GP-FTA-KG", "GP-FTA-KG (Ours)", "#0072B2", "o", "-", 2.2),
        ("GP-FTA-TS", "GP-FTA-TS (Ours)", "#56B4E9", "v", "-", 1.8),
        ("RGPE-on-Psi", "RGPE-on-Psi", "#D55E00", "s", "--", 1.5),
        ("SCBO-on-Psi", "SCBO-on-Psi", "#CC79A7", "P", ":", 1.5),
        ("PublicPsi-TargetGP-TS", "PublicPsi-TargetGP-TS", "#009E73", "D", "-.", 1.5),
        ("SCBO", "SCBO (Raw Space)", "#E69F00", "^", "--", 1.4),
        ("STOP", "STOP (N=0)", "#555555", "x", ":", 1.2),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 5.9))
    for idx, fam in enumerate(fams):
        ax = axes.flatten()[idx]
        agg = _task_agg(rows_u3, fam)["by_budget"]
        xs = [8, 15, 30, 60]
        for m_key, m_lbl, col, mk, ls, lw in palette:
            ys = [agg[x][m_key]["mean_loss"] for x in xs]
            ses = [agg[x][m_key]["se_loss"] for x in xs]
            ax.errorbar(
                xs, ys, yerr=ses, label=m_lbl, color=col, marker=mk,
                linestyle=ls, linewidth=lw, markersize=4.6, capsize=2.2,
            )
        ax.set_title(PLOT_TITLES[fam], fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xlabel("Online Target Step Budget $N$")
        ax.set_ylabel("Status-Quo Decision Loss ($\\downarrow$)")
        ax.grid(True, linestyle=":", alpha=0.5)
    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=True, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(fig_dir / "fig1_horizon_scaling.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(fig_dir / "fig1_horizon_scaling.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: Genuine Out-of-Sample Variance Calibration & Unified-Budget P1 Ladder ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    vc_map = {
        (d["family"], float(d["shift_scale"]), int(d["n_reps_train"])): d["estimators"]
        for d in p1_diag["variance_calibration"]
    }
    fam_keys_diag = [("inventory", "Inventory"), ("queueing", "Queueing"), ("opsd", "OPSD"), ("traffic", "Traffic-21")]
    x_pos = np.arange(len(fam_keys_diag))
    rmse_naive = [vc_map[(fk, 1.3, 4)]["naive_prior_residual_sq"]["log_var_rmse"] for fk, _ in fam_keys_diag]
    rmse_within = [vc_map[(fk, 1.3, 4)]["within_ssq"]["log_var_rmse"] for fk, _ in fam_keys_diag]

    w = 0.35
    ax1.bar(x_pos - w / 2, rmse_naive, width=w, color="#D55E00", alpha=0.88, label="Raw Prior-Residual NNLS")
    ax1.bar(x_pos + w / 2, rmse_within, width=w, color="#0072B2", alpha=0.92, label="Within-Policy $S^2/(r-1)$ NNLS")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([lbl for _, lbl in fam_keys_diag])
    ax1.set_ylim(0.0, max(rmse_naive) * 1.24)
    ax1.set_ylabel("Out-of-Sample $\\log v_t(x)$ RMSE ($\\downarrow$)")
    ax1.set_title("(a) Genuine Variance Calibration ($1.3\\times$ Shift, $r=4$)", fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.grid(True, axis="y", linestyle=":", alpha=0.5)

    ladder = [
        ("B_Add_SourceWarmStart", "B: +WarmStart Init"),
        ("C_Add_SourceBasis_WeakPrior", "C: +Orth. Basis (Weak)"),
        ("D_Add_FuncMetaPrior", "D: +Func. Prior $(\\mu_0,\\Sigma_0)$"),
        ("Dhvd_Add_SourceNoisePrior", "Dhvd: +HVD Prior $\\gamma_0$"),
        ("E_Add_TemplateProposals_TS", "E: +Templates (Gate-TS)"),
        ("F2_Swap_GatePowerVKG", "F2: Gate-Power V-KG"),
        ("G_WarmStart_PatternSearch", "G: Pattern Search"),
    ]
    agg_inv_p1 = _task_agg(rows_p1_mech, "Inventory_P1_Dev")["by_budget"]
    agg_que_p1 = _task_agg(rows_p1_mech, "Queueing_P1_Dev")["by_budget"]

    def _pooled_pct_vs_A(budget: int, arm_key: str) -> tuple[float, float]:
        rels = []
        for ag in (agg_inv_p1, agg_que_p1):
            a_vec = ag[budget]["A_PublicPsi_TargetGP_TS"]["loss_vec"]
            m_vec = ag[budget][arm_key]["loss_vec"]
            rels.append(100.0 * (a_vec - m_vec) / a_vec)
        all_r = np.concatenate(rels)
        return float(np.mean(all_r)), float(np.std(all_r, ddof=1) / np.sqrt(len(all_r)))

    y_idx = np.arange(len(ladder))
    vals15, ses15, vals60, ses60 = [], [], [], []
    for ak, _ in ladder:
        m15, s15 = _pooled_pct_vs_A(15, ak)
        m60, s60 = _pooled_pct_vs_A(60, ak)
        vals15.append(m15)
        ses15.append(s15)
        vals60.append(m60)
        ses60.append(s60)

    bh = 0.36
    ax2.barh(y_idx - bh / 2, vals15, height=bh, xerr=ses15, color="#56B4E9", alpha=0.92, capsize=2.0, label="Unified $N=15$")
    ax2.barh(y_idx + bh / 2, vals60, height=bh, xerr=ses60, color="#0072B2", alpha=0.92, capsize=2.0, label="Unified $N=60$")
    ax2.axvline(0.0, color="#333333", linewidth=1.0)
    ax2.set_yticks(y_idx)
    ax2.set_yticklabels([lbl for _, lbl in ladder])
    ax2.invert_yaxis()
    ax2.set_xlim(-6.2, 8.2)
    ax2.set_xlabel("Gain vs. Cold $\\psi$-GP Arm A (%, Mean $\\pm$ SE, 32 Dev Tasks)")
    ax2.set_title("(b) P1 Sequential Mechanism Ladder (Unified Budgets)", fontweight="bold")
    ax2.legend(loc="lower right")
    ax2.grid(True, axis="x", linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_tempering_diagnostics.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(fig_dir / "fig2_tempering_diagnostics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 3: Certification Reliability vs Gain Frontier & Messy-Archive Robustness ---
    fig, (ax_f, ax_m) = plt.subplots(1, 2, figsize=(9.2, 3.5))

    fam_aggs = {fam: _task_agg(rows_u3, fam)["by_budget"] for fam in fams}
    scatter_arms = [
        ("GP-FTA-KG", "GP-FTA-KG (Ours)", "#0072B2", "o"),
        ("GP-FTA-TS", "GP-FTA-TS (Ours)", "#56B4E9", "v"),
        ("PublicPsi-TargetGP-TS", "PublicPsi-TargetGP-TS", "#009E73", "D"),
        ("RGPE-on-Psi", "RGPE-on-Psi", "#D55E00", "s"),
        ("SCBO-on-Psi", "SCBO-on-Psi", "#CC79A7", "P"),
        ("WarmStart-PatternSearch", "WarmStart-PatternSearch", "#882255", "h"),
        ("SCBO", "SCBO (Raw Space)", "#E69F00", "^"),
        ("RGPE", "RGPE (Warped)", "#994F00", "<"),
    ]
    for m_key, m_lbl, col, mk in scatter_arms:
        cert60 = float(np.mean([100.0 * fam_aggs[f][60][m_key]["cert_rate"] for f in fams]))
        red60 = float(np.mean([
            float(np.mean(100.0 * (fam_aggs[f][60][m_key]["anc_vec"] - fam_aggs[f][60][m_key]["fr10_vec"]) / fam_aggs[f][60][m_key]["anc_vec"]))
            for f in fams
        ]))
        ax_f.scatter([cert60], [red60], color=col, marker=mk, s=72, zorder=4, label=m_lbl)

    ax_f.axhline(0.0, color="#555555", linestyle=":", linewidth=1.0)
    ax_f.set_xlabel("Mean Terminal Gate Certification Rate at $N=60$ (%, $\\uparrow$)")
    ax_f.set_ylabel("Mean Net Gain on $\\mathcal{L}_{10\\%}$ vs. Status Quo (%, $\\uparrow$)")
    ax_f.set_title("(a) Safety Certification vs. Net Saving ($N=60$)", fontweight="bold")
    ax_f.legend(loc="lower right", fontsize=6.8, ncol=1, framealpha=0.92)
    ax_f.grid(True, linestyle=":", alpha=0.5)

    messy_fams = [
        ("Inventory_CrossHorizon_Messy", "Inventory"),
        ("Queueing_CrossHorizon_Messy", "Queueing"),
        ("OPSD_EuroGrid_Messy", "OPSD"),
        ("Traffic_Ingolstadt_Messy", "Traffic-21"),
    ]
    m_aggs = {mf: _task_agg(rows_messy, mf)["by_budget"][60] for mf, _ in messy_fams}
    comp_bars = [
        ("GP-FTA-KG", "GP-FTA-KG (Ours)", "#0072B2"),
        ("GP-FTA-TS", "GP-FTA-TS (Ours)", "#56B4E9"),
        ("RGPE-on-Psi", "RGPE-on-Psi", "#D55E00"),
        ("WarmStart-PatternSearch", "WarmStart-PS", "#CC79A7"),
    ]
    x_m = np.arange(len(messy_fams))
    bw = 0.19
    for idx_b, (ak, albl, col) in enumerate(comp_bars):
        gains = [
            float(np.mean(100.0 * (m_aggs[mf][ak]["anc_vec"] - m_aggs[mf][ak]["fr10_vec"]) / m_aggs[mf][ak]["anc_vec"]))
            for mf, _ in messy_fams
        ]
        ax_m.bar(x_m + (idx_b - 1.5) * bw, gains, width=bw, color=col, alpha=0.90, label=albl)
    ax_m.axhline(0.0, color="#333333", linewidth=1.0)
    ax_m.set_xticks(x_m)
    ax_m.set_xticklabels([lbl for _, lbl in messy_fams])
    ax_m.set_ylabel("Net Gain on $\\mathcal{L}_{10\\%}$ vs. Status Quo (%, $\\uparrow$)")
    ax_m.set_title("(b) Messy Enterprise Archive ($r_a\\!=\\!1$, $\\alpha_m \\neq \\alpha_t$)", fontweight="bold")
    ax_m.legend(loc="best", fontsize=7.2)
    ax_m.grid(True, axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(fig_dir / "fig3_amortization_mbreak.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(fig_dir / "fig3_amortization_mbreak.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=MS_DIR, help="Directory for regenerated tables and figures.")
    args = parser.parse_args()
    tab_dir = args.output_dir / "tables"
    tab_dir.mkdir(parents=True, exist_ok=True)
    rows_u3 = _load_ledger("v2_or_u3_confirm_v1")
    rows_messy = _load_ledger("v2_or_u3_messy_archive_v1")
    rows_sumo_u3 = _load_ledger("v2_or_u3_live_sumo_v1")
    rows_p1_mech = _load_ledger("v2_or_p1_mech_v1")
    with (ART_DIR / "v2_or_p0_reaudit_v1/v2_or_live_sumo_v1/per_run_ledger.csv").open("r", encoding="utf-8") as fh:
        rows_sumo_p0 = list(csv.DictReader(fh))
    p1_diag = json.loads((ART_DIR / "v2_or_p1_diagnostics_v1/p1_diagnostics_summary.json").read_text(encoding="utf-8"))

    (tab_dir / "table_main.tex").write_text(build_table_main(rows_u3), encoding="utf-8")
    (tab_dir / "table_ablations.tex").write_text(build_table_ablations(rows_u3), encoding="utf-8")
    (tab_dir / "table_messy_archive.tex").write_text(build_table_messy_archive(rows_messy), encoding="utf-8")
    (tab_dir / "table_live_sumo.tex").write_text(build_table_live_sumo(rows_sumo_u3, rows_sumo_p0), encoding="utf-8")
    (tab_dir / "table_amortization.tex").write_text(build_table_amortization(rows_u3), encoding="utf-8")
    (tab_dir / "table_verification_audit.tex").write_text(
        build_table_verification_audit(rows_u3, rows_messy, rows_sumo_u3, rows_sumo_p0), encoding="utf-8"
    )
    independent_path = ART_DIR / "v2_or_kg_quadrature_confirm_v1" / "independent_panel_readout.json"
    if independent_path.is_file():
        independent_readout = json.loads(independent_path.read_text(encoding="utf-8"))
        (tab_dir / "table_independent_panel.tex").write_text(
            build_table_independent_panel(independent_readout), encoding="utf-8"
        )
    operating_path = independent_path.with_name("operating_policy_readout.json")
    if operating_path.is_file():
        operating_readout = json.loads(operating_path.read_text(encoding="utf-8"))
        (tab_dir / "table_operating_policy.tex").write_text(
            build_table_operating_policy(operating_readout), encoding="utf-8"
        )
    referee = ART_DIR / "v2_or_gpfta_referee_ablation_v1/referee_ablation_readout.json"
    if referee.is_file():
        readout = json.loads(referee.read_text(encoding="utf-8"))
        policy, comparisons = build_referee_tables(readout)
        (tab_dir / "table_matched_policy.tex").write_text(policy, encoding="utf-8")
        (tab_dir / "table_matched_contrasts.tex").write_text(comparisons, encoding="utf-8")
        (tab_dir / "table_matched_variances.tex").write_text(build_referee_variances(readout), encoding="utf-8")
    replay = referee.with_name("terminal_replay_readout.json")
    if replay.is_file():
        replay_table = build_terminal_replay_table(json.loads(replay.read_text(encoding="utf-8")))
        (tab_dir / "table_terminal_replay.tex").write_text(replay_table, encoding="utf-8")
    cascade = referee.parent / "cascade_query_v1/readout.json"
    if cascade.is_file():
        query_readout = json.loads(cascade.read_text(encoding="utf-8"))
        if "refinement" in query_readout["Queueing_CrossHorizon"]:
            (tab_dir / "table_query_screen.tex").write_text(build_query_screen_table(query_readout), encoding="utf-8")
    build_figures(args.output_dir / "figures", rows_u3, rows_messy, rows_p1_mech, p1_diag)
    print(f"Regenerated sealed-ledger tables and figures in {args.output_dir}.")


if __name__ == "__main__":
    main()
