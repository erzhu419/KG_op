#!/usr/bin/env python3
"""Stage P4 Automated Table & Vector Figure Generator for INFORMS *Operations Research*.

Reads strictly from sealed Stage P0, Stage P1, and Stage P2 CSV ledgers and JSON
diagnostics (`paper_artifacts/v2_or_p1_diagnostics_v1`, `v2_or_p1_mech_v1`,
`v2_or_p2_confirm_v1`, `v2_or_p2_live_sumo_v1`, `v2_or_p0_reaudit_v1`).
Enforces:
- Task-level paired inference across independent target tasks `M_tgt`
- Unified budgets in mechanism decomposition plots (never mixing N=15 and N=60 across arms)
- Genuine unmultiplied out-of-sample variance calibration metrics
- Unclamped call-saving amortization (zero `max(diff, 1)` hacks) with 95% task CIs
"""

from __future__ import annotations

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
TAB_DIR = MS_DIR / "tables"
FIG_DIR = MS_DIR / "figures"


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


def _fmt_p(p: float | None) -> str:
    if p is None:
        return "---"
    if p < 1e-4:
        return "$<10^{-4}$"
    if p < 1e-3:
        return f"${p:.4f}$"
    return f"${p:.3f}$"


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
            cert_vec = np.array([np.mean([1.0 if x["certified"] == "True" else 0.0 for x in mrows if x["task_id"] == t]) for t in tasks])
            call_vec = np.array([np.mean([float(x["total_target_calls"]) for x in mrows if x["task_id"] == t]) for t in tasks])
            prim_vec = np.array([np.mean([float(x["primary_slot0_mc_obj"]) for x in mrows if x["task_id"] == t]) for t in tasks])
            anc_vec = np.array([np.mean([float(x["anchor_mc_obj"]) for x in mrows if x["task_id"] == t]) for t in tasks])
            vlaw_unsafe = sum(
                1 for x in mrows
                if x["certified"] == "True" and x["verif_law_mc_viol_prob"] != "" and float(x["verif_law_mc_viol_prob"]) > 0.05
            )
            audit_unsafe = sum(1 for x in mrows if x["safety_audit_class"] == "confidently_unsafe")
            audit_ambig = sum(1 for x in mrows if x["safety_audit_class"] == "boundary_ambiguous")
            out[b][m] = {
                "tasks": tasks,
                "loss_vec": loss_vec,
                "mean_loss": float(np.mean(loss_vec)),
                "se_loss": float(np.std(loss_vec, ddof=1) / np.sqrt(len(tasks))),
                "cert_rate": float(np.mean(cert_vec)),
                "call_vec": call_vec,
                "mean_calls": float(np.mean(call_vec)),
                "se_calls": float(np.std(call_vec, ddof=1) / np.sqrt(len(tasks))),
                "mean_prim": float(np.mean(prim_vec)),
                "mean_anc": float(np.mean(anc_vec)),
                "anc_vec": anc_vec,
                "vlaw_unsafe": vlaw_unsafe,
                "audit_unsafe": audit_unsafe,
                "audit_ambig": audit_ambig,
                "n_runs": len(mrows),
                "mean_wall": float(np.mean([float(x["online_wall_time_sec"]) for x in mrows])),
                "mean_pre": float(np.mean([float(x["source_pretrain_wall_sec"]) for x in mrows])),
            }
    return {"tasks": tasks, "budgets": budgets, "by_budget": out}


def build_table_main(rows_p2: list[dict[str, str]]) -> str:
    fams = ["Inventory_CrossHorizon", "Queueing_CrossHorizon", "OPSD_EuroGrid", "Traffic_Ingolstadt"]
    ext_baselines = ["SCBO", "MAP-SAAS-CBO", "HetGP-KG", "RGPE", "HyperBO-EB", "Target-SPSA"]
    display_order = ["TLS-BGP-KG"] + ext_baselines + ["STOP"]

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Held-Out Confirmation Performance Across Four Operational Families ($M_{\\mathrm{tgt}} \\in \\{14, 16\\}$ Independent Target Tasks per Family, Two Paired Streams per Task; Roster-Invariant Status-Quo Loss, Mean $\\pm$ Task SE). Under the Certified Verification Law, Empirical False Certificates Are $0/1{,}413$ on OPSD and $0/4{,}663$ on Synthetic DES; ``OOD Unsafe'' Counts Clopper--Pearson $95\\%$ Exceedances on Disjoint Temporal/Audit Streams.}\\label{tab:main_confirm}",
        "\\scriptsize",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llcccccc}",
        "\\toprule",
        "\\textbf{Family} & \\textbf{Method} & \\textbf{$N=15$ Loss ($\\downarrow$)} & \\textbf{$N=60$ Loss ($\\downarrow$)} & \\textbf{$N=60$ Cert.\\%} & \\textbf{Verif./OOD Unsafe} & \\textbf{Task Gain at $N=60$ (W/L/T)} & \\textbf{Holm $p$} \\\\",
        "\\midrule",
    ]

    for f_idx, fam in enumerate(fams):
        agg = _task_agg(rows_p2, fam)["by_budget"]
        b15, b60 = agg[15], agg[60]
        ref60 = b60["TLS-BGP-KG"]["loss_vec"]

        raw_p: dict[str, float] = {}
        comp_info: dict[str, tuple[float, int, int, int]] = {}
        for m in ext_baselines:
            diff = b60[m]["loss_vec"] - ref60
            nz = diff[np.abs(diff) > 1e-12]
            p1 = float(wilcoxon(nz, alternative="greater").pvalue) if len(nz) >= 3 else 1.0
            raw_p[m] = p1
            w = int(np.sum(diff > 1e-9))
            l = int(np.sum(diff < -1e-9))
            t = len(diff) - w - l
            rpct = 100.0 * float(np.mean(diff / np.maximum(np.abs(b60[m]["loss_vec"]), 1e-8)))
            comp_info[m] = (rpct, w, l, t)
        holm_p = _holm_adjust(raw_p)

        # Also compute STOP two-sided/one-sided comparison separately (not part of 6-baseline Holm family)
        diff_stop = b60["STOP"]["loss_vec"] - ref60
        nz_s = diff_stop[np.abs(diff_stop) > 1e-12]
        p_stop = float(wilcoxon(nz_s, alternative="greater").pvalue) if len(nz_s) >= 3 else 1.0
        ws = int(np.sum(diff_stop > 1e-9))
        ls = int(np.sum(diff_stop < -1e-9))
        ts = len(diff_stop) - ws - ls
        rpct_stop = 100.0 * float(np.mean(diff_stop / np.maximum(np.abs(b60["STOP"]["loss_vec"]), 1e-8)))

        for r_i, m in enumerate(display_order):
            fam_col = f"\\multirow{{{len(display_order)}}}{{*}}{{{FAMILY_DISPLAY[fam]}}}" if r_i == 0 else ""
            m_disp = "\\textbf{TLS-BGP-KG (Ours)}" if m == "TLS-BGP-KG" else f"\\texttt{{{m}}}"
            c15 = _fmt_mean_se(b15[m]["mean_loss"], b15[m]["se_loss"])
            c60 = _fmt_mean_se(b60[m]["mean_loss"], b60[m]["se_loss"])
            if m == "TLS-BGP-KG":
                c15 = f"$\\mathbf{{{c15[1:-1]}}}$"
                c60 = f"$\\mathbf{{{c60[1:-1]}}}$"
            cert_str = f"{100.0 * b60[m]['cert_rate']:.1f}\\%"
            tot_v_unsafe = sum(agg[b][m]["vlaw_unsafe"] for b in (8, 15, 30, 60) if m != "STOP") if m != "STOP" else b60[m]["vlaw_unsafe"]
            tot_a_unsafe = sum(agg[b][m]["audit_unsafe"] for b in (8, 15, 30, 60) if m != "STOP") if m != "STOP" else b60[m]["audit_unsafe"]
            unsafe_str = f"{tot_v_unsafe} / {tot_a_unsafe}"

            if m == "TLS-BGP-KG":
                gain_str, p_str = "---", "---"
            elif m == "STOP":
                sgn = "+" if rpct_stop >= 0 else ""
                gain_str = f"${sgn}{rpct_stop:.1f}\\%$ ({ws}/{ls}/{ts})"
                p_str = _fmt_p(p_stop) + "$^{\\dagger}$"
            else:
                rpct, w, l, t = comp_info[m]
                sgn = "+" if rpct >= 0 else ""
                gain_str = f"${sgn}{rpct:.1f}\\%$ ({w}/{l}/{t})"
                p_str = _fmt_p(holm_p[m])
            lines.append(f"{fam_col} & {m_disp} & {c15} & {c60} & {cert_str} & {unsafe_str} & {gain_str} & {p_str} \\\\")
        if f_idx < len(fams) - 1:
            lines.append("\\midrule")

    lines.extend([
        "\\bottomrule",
        "\\multicolumn{8}{l}{\\scriptsize $^{\\dagger}$Unadjusted one-sided paired Wilcoxon $p$-value vs.\\ zero-step \\texttt{STOP}; all six external baseline $p$-values are Holm--Bonferroni adjusted within each family.}\\\\",
        "\\end{tabular}%",
        "}",
        "\\end{table}",
    ])
    return "\n".join(lines)


def build_table_ablations(rows_p2: list[dict[str, str]]) -> str:
    fams = ["Inventory_CrossHorizon", "Queueing_CrossHorizon", "OPSD_EuroGrid", "Traffic_Ingolstadt"]
    ctrl_arms = [
        "TLS-BGP-KG",
        "TLS-GatePower-TS",
        "Untempered-PriorResid",
        "Frozen-AFTER",
        "PublicPsi-TargetGP-TS",
        "WarmStart-PatternSearch",
        "STOP",
    ]
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Matched-Information Mechanism Controls Across Online Budgets $N \\in \\{8, 15, 30, 60\\}$ and Unclamped Raw-Tail ENTSO-E Grid Sensitivity ($M_{\\mathrm{tgt}}=14$ Markets, \\texttt{clamp\\_limit = None})}\\label{tab:ablations}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{4.5pt}",
        "\\begin{tabular}{llcccccc}",
        "\\toprule",
        "\\textbf{Family} & \\textbf{Mechanism / Control Arm} & \\textbf{$N=8$} & \\textbf{$N=15$} & \\textbf{$N=30$} & \\textbf{$N=60$} & \\textbf{$N=60$ Cert.\\%} & \\textbf{Two-Sided $p$ ($N=60$)} \\\\",
        "\\midrule",
    ]

    for fam in fams:
        agg = _task_agg(rows_p2, fam)["by_budget"]
        fmt = ".4f" if fam.startswith("OPSD") else ".3f"
        mins = {b: min(agg[b][m]["mean_loss"] for m in ctrl_arms) for b in (8, 15, 30, 60)}
        ref60 = agg[60]["TLS-BGP-KG"]["loss_vec"]

        for r_i, m in enumerate(ctrl_arms):
            fam_col = f"\\multirow{{{len(ctrl_arms)}}}{{*}}{{{SHORT_FAM[fam]}}}" if r_i == 0 else ""
            m_disp = "\\textbf{TLS-BGP-KG}" if m == "TLS-BGP-KG" else f"\\texttt{{{m}}}"
            vals = []
            for b in (8, 15, 30, 60):
                v = agg[b][m]["mean_loss"]
                s = f"$\\mathbf{{{v:{fmt}}}}$" if abs(v - mins[b]) < 1e-12 else f"${v:{fmt}}$"
                vals.append(s)
            cert60 = f"{100.0 * agg[60][m]['cert_rate']:.1f}\\%"
            if m == "TLS-BGP-KG":
                p2_str = "---"
            else:
                d = agg[60][m]["loss_vec"] - ref60
                nz = d[np.abs(d) > 1e-12]
                p2 = float(wilcoxon(nz, alternative="two-sided").pvalue) if len(nz) >= 3 else 1.0
                p2_str = _fmt_p(p2)
            lines.append(f"{fam_col} & {m_disp} & {vals[0]} & {vals[1]} & {vals[2]} & {vals[3]} & {cert60} & {p2_str} \\\\")
        lines.append("\\midrule")

    # Panel B: Unclamped OPSD raw-tail sensitivity
    u_agg = _task_agg(rows_p2, "OPSD_EuroGrid_Unclamped")["by_budget"]
    u_arms = ["TLS-BGP-KG", "PublicPsi-TargetGP-TS", "WarmStart-PatternSearch", "SCBO", "RGPE", "STOP"]
    min15 = min(u_agg[15][m]["mean_loss"] for m in u_arms)
    min60 = min(u_agg[60][m]["mean_loss"] for m in u_arms)
    ref_u60 = u_agg[60]["TLS-BGP-KG"]["loss_vec"]
    for r_i, m in enumerate(u_arms):
        fam_col = f"\\multirow{{{len(u_arms)}}}{{*}}{{OPSD Raw Tails}}" if r_i == 0 else ""
        m_disp = "\\textbf{TLS-BGP-KG}" if m == "TLS-BGP-KG" else f"\\texttt{{{m}}}"
        v15 = u_agg[15][m]["mean_loss"]
        v60 = u_agg[60][m]["mean_loss"]
        s15 = f"$\\mathbf{{{v15:.4f}}}$" if abs(v15 - min15) < 1e-12 else f"${v15:.4f}$"
        s60 = f"$\\mathbf{{{v60:.4f}}}$" if abs(v60 - min60) < 1e-12 else f"${v60:.4f}$"
        cert60 = f"{100.0 * u_agg[60][m]['cert_rate']:.1f}\\%"
        if m == "TLS-BGP-KG":
            p2_str = "---"
        else:
            d = u_agg[60][m]["loss_vec"] - ref_u60
            nz = d[np.abs(d) > 1e-12]
            p2 = float(wilcoxon(nz, alternative="two-sided").pvalue) if len(nz) >= 3 else 1.0
            p2_str = _fmt_p(p2)
        lines.append(f"{fam_col} & {m_disp} & --- & {s15} & --- & {s60} & {cert60} & {p2_str} \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


def build_table_live_sumo(rows_sumo_p2: list[dict[str, str]], rows_sumo_p0: list[dict[str, str]]) -> str:
    agg_p2 = _task_agg(rows_sumo_p2, "Ingolstadt21_LiveSUMO_Calibrated")["by_budget"][15]
    p2_order = [
        "PublicPsi-TargetGP-TS",
        "TLS-BGP-KG",
        "WarmStart-PatternSearch",
        "TLS-GatePower-TS",
        "SCBO",
        "RGPE",
        "STOP",
    ]
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Live Microscopic \\texttt{libsumo} Validation on the 21-Intersection \\texttt{Ingolstadt21} Corridor ($d_t=44$, 4{,}283 Vehicles, $N=15$): (A) Platform-Calibrated \\texttt{Linux SUMO~1.22} Regime ($M_{\\mathrm{tgt}}=8$ Tasks, $\\tau_{\\mathrm{rel}} \\in [1.032, 1.054]$, $K_{\\mathrm{audit}}=80$) and (B) Uncalibrated Cross-Platform Infeasible Stress Test ($40$ Runs Where $E_0^{\\mathrm{Linux}}/E_0^{\\mathrm{legacy}} = 1.282\\times$ Makes All Plans Infeasible)}\\label{tab:live_sumo}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{4.8pt}",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "\\textbf{Regime} & \\textbf{Method} & \\textbf{Status-Quo Loss ($\\downarrow$)} & \\textbf{Challenger / Cand.\\ $J_t^{\\ddagger}$} & \\textbf{Cert.\\%} & \\textbf{Conf.\\ Unsafe} & \\textbf{Total Calls} \\\\",
        "\\midrule",
    ]
    min_loss = min(agg_p2[m]["mean_loss"] for m in p2_order)
    for r_i, m in enumerate(p2_order):
        md = agg_p2[m]
        reg_col = f"\\multirow{{{len(p2_order)}}}{{*}}{{Calibrated ($112$ runs)}}" if r_i == 0 else ""
        m_disp = "\\textbf{TLS-BGP-KG (Ours)}" if m == "TLS-BGP-KG" else f"\\texttt{{{m}}}"
        l_str = f"${md['mean_loss']:.4f} \\pm {md['se_loss']:.4f}$"
        if abs(md["mean_loss"] - min_loss) < 1e-12 or m == "TLS-BGP-KG":
            l_str = f"$\\mathbf{{{md['mean_loss']:.4f}}} \\pm {md['se_loss']:.4f}$" if abs(md["mean_loss"] - min_loss) < 1e-12 else l_str
        lines.append(
            f"{reg_col} & {m_disp} & {l_str} & ${md['mean_prim']:.4f}$ & "
            f"{100.0 * md['cert_rate']:.1f}\\% & {md['audit_unsafe']} & {md['mean_calls']:.1f} \\\\"
        )

    lines.append("\\midrule")
    # Summarize preserved historical infeasible stress test from P0 re-audit ledger
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
        cert = 100.0 * float(np.mean([1.0 if x["certified"] == "True" else 0.0 for x in mrows]))
        cu = sum(1 for x in mrows if x["safety_audit_class"] == "confidently_unsafe")
        calls = float(np.mean([float(x["total_target_calls"]) for x in mrows]))
        lines.append(f"{reg_col} & {m_disp} & ${sq:.4f}$ & ${pr:.4f}$ & {cert:.1f}\\% & {cu} & {calls:.1f} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\multicolumn{7}{l}{\\scriptsize $^{\\ddagger}$Calibrated runs report frozen pre-verification primary $J_t(x^{(0)})$; preserved pre-remediation stress-test runs report the saved uncertified shortlist candidate.}\\\\",
        "\\end{tabular}",
        "\\end{table}",
    ])
    return "\n".join(lines)


def _compute_honest_amortization(rows_p2: list[dict[str, str]], fam: str, b_src: int = 1428) -> dict[str, Any]:
    agg = _task_agg(rows_p2, fam)["by_budget"]
    tls15 = agg[15]["TLS-BGP-KG"]
    scbo60 = agg[60]["SCBO"]

    # Exact task-level call difference vector WITHOUT any max(diff, 1) clamping!
    diff_calls = scbo60["call_vec"] - tls15["call_vec"]
    n_t = len(diff_calls)
    mean_diff = float(np.mean(diff_calls))
    se_diff = float(np.std(diff_calls, ddof=1) / np.sqrt(n_t))
    t_crit = float(student_t.ppf(0.975, df=n_t - 1))
    ci_lo = mean_diff - t_crit * se_diff
    ci_hi = mean_diff + t_crit * se_diff

    assert tls15["mean_loss"] < scbo60["mean_loss"], f"Quality dominance violated on {fam}"
    assert mean_diff > 0.0 and ci_lo > 0.0, f"Positive call saving violated on {fam}: {mean_diff}, {ci_lo}"

    m_break_point = math.ceil(b_src / mean_diff)
    m_break_ci95 = math.ceil(b_src / ci_lo)
    return {
        "B_src": b_src,
        "tls15_loss": tls15["mean_loss"],
        "scbo60_loss": scbo60["mean_loss"],
        "tls15_calls": tls15["mean_calls"],
        "scbo60_calls": scbo60["mean_calls"],
        "mean_saving": mean_diff,
        "se_saving": se_diff,
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "m_break_sunk": 1,
        "m_break_point": m_break_point,
        "m_break_ci95_upper": m_break_ci95,
    }


def build_table_amortization(rows_p2: list[dict[str, str]]) -> str:
    fams = ["Inventory_CrossHorizon", "Queueing_CrossHorizon", "OPSD_EuroGrid", "Traffic_Ingolstadt"]
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Quality-Dominated Simulation Call Accounting and Dual Source-Archive Amortization ($B_{\\mathrm{src}} = 1{,}428$ Calls): \\textbf{TLS-BGP-KG} at $N=15$ Strictly Achieves Lower Loss Than Cold-Start \\texttt{SCBO} at $N=60$ in Every Family while Saving $\\Delta B = B_{\\mathrm{SCBO},60} - B_{\\mathrm{TLS},15} > 0$ Target Calls (Reported with Two-Sided $95\\%$ Task-$t$ Confidence Intervals and Conservative Upper Break-Even $M_{\\mathrm{break}}^{\\mathrm{CI95}}$)}\\label{tab:amortization}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{4.5pt}",
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "\\textbf{Family} & \\textbf{TLS ($N\\!=\\!15$) Loss} & \\textbf{\\texttt{SCBO} ($N\\!=\\!60$) Loss} & \\textbf{$B_{\\mathrm{TLS},15}$} & \\textbf{$B_{\\mathrm{SCBO},60}$} & \\textbf{Call Saving $\\Delta B$ ($95\\%$ CI)} & \\textbf{$M_{\\mathrm{break}}^{\\mathrm{sunk}}$} & \\textbf{$M_{\\mathrm{break}}^{\\mathrm{full}}$ ($95\\%$ UB)} \\\\",
        "\\midrule",
    ]
    for fam in fams:
        am = _compute_honest_amortization(rows_p2, fam)
        fmt = ".4f" if fam.startswith("OPSD") else ".3f"
        lines.append(
            f"{SHORT_FAM[fam]} & $\\mathbf{{{am['tls15_loss']:{fmt}}}}$ & ${am['scbo60_loss']:{fmt}}$ & "
            f"{am['tls15_calls']:.1f} & {am['scbo60_calls']:.1f} & "
            f"$+{am['mean_saving']:.1f}$ $[+{am['ci95_lo']:.1f}, +{am['ci95_hi']:.1f}]$ & "
            f"$1$ & $\\mathbf{{{am['m_break_point']}}}$ ($\\le {am['m_break_ci95_upper']}$) \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


def build_figures(
    rows_p2: list[dict[str, str]],
    rows_p1_mech: list[dict[str, str]],
    p1_diag: dict[str, Any],
) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8.8,
        "axes.titlesize": 9.3,
        "axes.labelsize": 8.8,
        "legend.fontsize": 7.8,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "pdf.fonttype": 42,
    })

    fams = ["Inventory_CrossHorizon", "Queueing_CrossHorizon", "OPSD_EuroGrid", "Traffic_Ingolstadt"]

    # --- Figure 1: 2x2 Task-Level Status-Quo Loss Scaling Across N in {8, 15, 30, 60} ---
    palette = [
        ("TLS-BGP-KG", "TLS-BGP-KG (Ours)", "#0072B2", "o", "-", 2.2),
        ("PublicPsi-TargetGP-TS", "PublicPsi-TargetGP-TS", "#009E73", "D", "-.", 1.6),
        ("WarmStart-PatternSearch", "WarmStart-PatternSearch", "#CC79A7", "P", ":", 1.5),
        ("RGPE", "RGPE (Warped LOOCV)", "#D55E00", "s", "--", 1.5),
        ("SCBO", "SCBO (Cold Start)", "#E69F00", "^", "--", 1.5),
        ("STOP", "STOP (N=0)", "#555555", "x", ":", 1.3),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 5.8))
    for idx, fam in enumerate(fams):
        ax = axes.flatten()[idx]
        agg = _task_agg(rows_p2, fam)["by_budget"]
        xs = [8, 15, 30, 60]
        for m_key, m_lbl, col, mk, ls, lw in palette:
            ys = [agg[x][m_key]["mean_loss"] for x in xs]
            ses = [agg[x][m_key]["se_loss"] for x in xs]
            ax.errorbar(
                xs, ys, yerr=ses, label=m_lbl, color=col, marker=mk,
                linestyle=ls, linewidth=lw, markersize=4.8, capsize=2.3,
            )
        ax.set_title(PLOT_TITLES[fam], fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xlabel("Online Target Step Budget $N$")
        ax.set_ylabel("Status-Quo Decision Loss ($\\downarrow$)")
        ax.grid(True, linestyle=":", alpha=0.5)
    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG_DIR / "fig1_horizon_scaling.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig1_horizon_scaling.png", dpi=220, bbox_inches="tight")
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

    # Panel (b): Unified-budget P1 sequential mechanism ladder (BOTH N=15 and N=60 for EVERY arm!)
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
    fig.savefig(FIG_DIR / "fig2_tempering_diagnostics.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig2_tempering_diagnostics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 3: Quality-vs-Calls Pareto Frontier & Full-Cost Amortization Curves ---
    fig, (ax_f, ax_a) = plt.subplots(1, 2, figsize=(9.0, 3.4))

    frontier_methods = [
        ("TLS-BGP-KG", "TLS-BGP-KG (Ours)", "#0072B2", "o", "-"),
        ("WarmStart-PatternSearch", "WarmStart-PatternSearch", "#CC79A7", "P", ":"),
        ("SCBO", "SCBO (Cold Start)", "#E69F00", "^", "--"),
        ("RGPE", "RGPE (Warped LOOCV)", "#D55E00", "s", "--"),
    ]
    fam_aggs = {fam: _task_agg(rows_p2, fam)["by_budget"] for fam in fams}
    for m_key, m_lbl, col, mk, ls in frontier_methods:
        xs_c, ys_g = [], []
        for b in (8, 15, 30, 60):
            calls_b = np.mean([fam_aggs[f][b][m_key]["mean_calls"] for f in fams])
            # Relative reduction (%) below each task's status-quo anchor loss
            red_pcts = []
            for f in fams:
                anc = fam_aggs[f][b][m_key]["anc_vec"]
                lvec = fam_aggs[f][b][m_key]["loss_vec"]
                red_pcts.append(float(np.mean(100.0 * (anc - lvec) / anc)))
            xs_c.append(calls_b)
            ys_g.append(float(np.mean(red_pcts)))
        ax_f.plot(xs_c, ys_g, marker=mk, linestyle=ls, color=col, linewidth=1.9, markersize=5, label=m_lbl)

    # Plot STOP point
    stop_calls = float(np.mean([fam_aggs[f][8]["STOP"]["mean_calls"] for f in fams]))
    stop_red = float(np.mean([
        float(np.mean(100.0 * (fam_aggs[f][8]["STOP"]["anc_vec"] - fam_aggs[f][8]["STOP"]["loss_vec"]) / fam_aggs[f][8]["STOP"]["anc_vec"]))
        for f in fams
    ]))
    ax_f.scatter([stop_calls], [stop_red], color="#333333", marker="X", s=55, zorder=5, label="STOP ($N=0$)")
    ax_f.set_xlabel("Mean Total Target Simulation Calls (Search + Verifier)")
    ax_f.set_ylabel("Mean Gain Over Status-Quo Anchor (%, $\\uparrow$)")
    ax_f.set_title("(a) Target Cost\u2013Quality Frontier ($N \\in \\{8,15,30,60\\}$)", fontweight="bold")
    ax_f.legend(loc="center right", bbox_to_anchor=(0.98, 0.50))
    ax_f.grid(True, linestyle=":", alpha=0.5)

    m_grid = np.arange(1, 23)
    colors_fam = ["#0072B2", "#009E73", "#D55E00", "#CC79A7"]
    for idx, fam in enumerate(fams):
        am = _compute_honest_amortization(rows_p2, fam)
        ratio = (am["B_src"] + m_grid * am["tls15_calls"]) / (m_grid * am["scbo60_calls"])
        lbl = f"{SHORT_FAM[fam]} ($M_{{\\mathrm{{break}}}}^{{\\mathrm{{full}}}}={am['m_break_point']}$)"
        ax_a.plot(m_grid, ratio, marker="o", markersize=3.8, linewidth=1.8, color=colors_fam[idx], label=lbl)
    ax_a.axhline(1.0, color="#222222", linestyle="--", linewidth=1.3, label="Cold-Start SCBO ($N=60$)")
    ax_a.set_xlabel("Number of Target Deployments $M$")
    ax_a.set_ylabel("Full-Cost Call Ratio vs. Cold Start ($\\downarrow$)")
    ax_a.set_title("(b) Full-Cost Archive Amortization ($B_{\\mathrm{src}}=1{,}428$)", fontweight="bold")
    ax_a.set_xticks(np.arange(2, 23, 2))
    ax_a.legend(loc="upper right", fontsize=7.4)
    ax_a.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_amortization_mbreak.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig3_amortization_mbreak.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    rows_p2 = _load_ledger("v2_or_p2_confirm_v1")
    rows_sumo_p2 = _load_ledger("v2_or_p2_live_sumo_v1")
    rows_p1_mech = _load_ledger("v2_or_p1_mech_v1")
    with (ART_DIR / "v2_or_p0_reaudit_v1/v2_or_live_sumo_v1/per_run_ledger.csv").open("r", encoding="utf-8") as fh:
        rows_sumo_p0 = list(csv.DictReader(fh))
    p1_diag = json.loads((ART_DIR / "v2_or_p1_diagnostics_v1/p1_diagnostics_summary.json").read_text(encoding="utf-8"))

    (TAB_DIR / "table_main.tex").write_text(build_table_main(rows_p2), encoding="utf-8")
    (TAB_DIR / "table_ablations.tex").write_text(build_table_ablations(rows_p2), encoding="utf-8")
    (TAB_DIR / "table_live_sumo.tex").write_text(build_table_live_sumo(rows_sumo_p2, rows_sumo_p0), encoding="utf-8")
    (TAB_DIR / "table_amortization.tex").write_text(build_table_amortization(rows_p2), encoding="utf-8")
    build_figures(rows_p2, rows_p1_mech, p1_diag)
    print("Generated all Stage P4 tables and vector figures successfully.")


if __name__ == "__main__":
    main()
