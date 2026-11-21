#!/usr/bin/env python3
"""Deterministic Publication Table & Vector Figure Generator for *Operations Research*.

Reads strictly from sealed CSV ledgers and JSON readouts in `SC-OLH-KG/paper_artifacts/`
(requiring zero raw simulation working directories). Regenerates all main-body and
Electronic Companion tables and vector figures byte-reproducibly.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
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
    "Traffic_Ingolstadt": "(d) Ingolstadt-21 Signal DES ($M_{\\mathrm{tgt}}=16$, Perturbed Drift)",
}

METHOD_RENAMES = {
    "Frozen-AFTER": "Init-Frozen-Transfer",
    "TLS-OLH-KG": "Gaussian-Margin-KG",
}


def _clean_zero(val: float, decimals: int = 4) -> float:
    return 0.0 if abs(val) < 0.5 * (10.0 ** (-decimals)) else float(val)


def _fmt_num(val: float, decimals: int = 4, signed: bool = False) -> str:
    v = _clean_zero(val, decimals)
    if signed:
        if v == 0.0:
            return f"{0.0:.{decimals}f}"
        return f"{v:+.{decimals}f}"
    return f"{v:.{decimals}f}"


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
        return f"${_fmt_num(mean, 4)} \\pm {_fmt_num(se, 4)}$"
    return f"${_fmt_num(mean, 3)} \\pm {_fmt_num(se, 3)}$"


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
    m_clean = METHOD_RENAMES.get(m, m)
    if m_clean in {"GP-FTA-ECKG", "GP-FTA-KG", "GP-FTA-TS"}:
        return f"\\textbf{{{m_clean}}}"
    return f"\\texttt{{{m_clean}}}"


def _audit_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    certified = [row for row in rows if row["certified"] == "True"]
    same_law = [
        float(row["verif_law_mc_viol_prob"])
        for row in certified
        if row.get("verif_law_mc_viol_prob", "")
    ]
    audit = [
        float(row["deployed_mc_viol_prob"]) for row in certified
        if row.get("deployed_mc_viol_prob", "")
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
        "audit_scored": len(audit),
        "audit_exceed": sum(p > 0.05 for p in audit),
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


# ==============================================================================
# MAIN-BODY TABLES (Tables 1-7)
# ==============================================================================

def _mean_online_wall_by_method(
    rows_prosp: list[dict[str, str]],
    rows_mp_opsd: list[dict[str, str]],
) -> dict[tuple[str, str], float]:
    acc: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows_prosp:
        acc[(r["family"], r["method_name"])].append(float(r["online_wall_time_sec"]))
    for r in rows_mp_opsd:
        if r["family"] == "OPSD_EuroGrid":
            acc[(r["family"], r["method_name"])].append(float(r["online_wall_time_sec"]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def build_table_prospective_main(
    prosp: dict[str, Any],
    mp_opsd: dict[str, Any],
    wall_map: dict[tuple[str, str], float],
) -> str:
    fams = ("Inventory_CrossHorizon", "Queueing_CrossHorizon", "OPSD_EuroGrid", "Traffic_Ingolstadt")
    methods = (
        "GP-FTA-ECKG",
        "GP-FTA-SingleSlotKG",
        "GP-FTA-FeasibleKG",
        "GP-FTA-LOO-ECKG",
        "PublicPsi-TargetGP-TS",
    )
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Five-arm performance, simulation calls, online wall-clock time, and certification outcomes across 72 DES tasks and 14 previously studied ENTSO-E grid markets with 336-dimensional policies ($860$ runs at $N=60$, two paired evaluation streams per task, zero new source calls). Online Time averages per-run sequential search and clean-up wall-clock seconds on single-thread CPU workers. Cert.\ S/A/U classifies certified deployments on held-out audit draws ($0/125$ verification-pool exceedances across all five OPSD arms; paired contrasts appear in Table~\ref{tab:prospective_contrasts}).}\label{tab:prospective_main}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{5.5pt}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"\textbf{Family} & \textbf{Method} & \textbf{$\mathcal{L}_{\mathrm{SQ}}$ (Mean $\pm$ SE)} & \textbf{Calls} & \textbf{Online Time (s)} & \textbf{Cert./runs} & \textbf{Cert.\ S/A/U} \\",
    ]
    for fam in fams:
        lines.append(r"\midrule")
        is_opsd = fam == "OPSD_EuroGrid"
        fd = mp_opsd["families"][fam] if is_opsd else prosp["families"][fam]
        min_l = min(fd["methods"][m]["status_quo_loss_task_mean"] for m in methods)
        fam_disp = "OPSD-336" if is_opsd else SHORT_FAM[fam]
        for idx, m in enumerate(methods):
            md = fd["methods"][m]
            f_col = rf"\multirow{{5}}{{*}}{{{fam_disp}}}" if idx == 0 else ""
            l_val = md["status_quo_loss_task_mean"]
            l_se = md["status_quo_loss_task_se"]
            l_str = (
                f"$\\mathbf{{{_fmt_num(l_val, 4)}}} \\pm {_fmt_num(l_se, 4)}$"
                if abs(l_val - min_l) < 1e-12
                else f"${_fmt_num(l_val, 4)} \\pm {_fmt_num(l_se, 4)}$"
            )
            ds = md["deployed_safety_counts"]
            sau = f"{ds['safe']}/{ds['borderline']}/{ds['unsafe']}"
            w_sec = wall_map[(fam, m)]
            lines.append(
                f"{f_col} & {_disp_method(m)} & {l_str} & {md['mean_total_target_calls']:.1f} & "
                f"{w_sec:.1f} & {md['n_certified']}/{md['n_runs']} & {sau} \\\\"
            )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines) + "\n"


def build_table_prospective_contrasts(prosp: dict[str, Any], mp_opsd: dict[str, Any]) -> str:
    fams = ("Inventory_CrossHorizon", "Queueing_CrossHorizon", "OPSD_EuroGrid", "Traffic_Ingolstadt")
    comp_only = (
        "GP-FTA-SingleSlotKG",
        "GP-FTA-FeasibleKG",
        "GP-FTA-LOO-ECKG",
        "PublicPsi-TargetGP-TS",
    )
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Task-paired status-quo decision-loss contrasts against \textbf{GP-FTA-ECKG} for the five-arm study in Table~\ref{tab:prospective_main}. Positive differences (`Arm $-$ ECKG') favor \textbf{GP-FTA-ECKG}. Primary $p$-values use two-sided paired-$t$ tests on stream-averaged task means ($n=24$ per DES family, $n=14$ for OPSD-336, and $n=72$ standardized by family anchor SD in the pooled DES block) with four-comparator Holm adjustment within each block.}\label{tab:prospective_contrasts}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4.2pt}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"\textbf{Domain / Panel} & \textbf{Comparator} & \textbf{Arm $-$ ECKG [95\% CI]} & \textbf{W/L/T} & \textbf{Two-sided $p$} & \textbf{Holm $p$} \\",
    ]
    for fam in fams:
        lines.append(r"\midrule")
        is_opsd = fam == "OPSD_EuroGrid"
        fd = mp_opsd["families"][fam] if is_opsd else prosp["families"][fam]
        n_t = 14 if is_opsd else 24
        fam_disp = f"OPSD-336 ($n={n_t}$)" if is_opsd else f"{SHORT_FAM[fam]} ($n={n_t}$)"
        for idx, m in enumerate(comp_only):
            f_col = rf"\multirow{{4}}{{*}}{{{fam_disp}}}" if idx == 0 else ""
            pr = fd["opsd_336_primary_contrasts"][m] if is_opsd else fd["paired_vs_ref"][m]["status_quo_loss"]
            lo, hi = pr["ci95"]
            diff_str = rf"${_fmt_num(pr['mean_diff'], 4, signed=True)}\ [{_fmt_num(lo, 4)},\,{_fmt_num(hi, 4)}]$"
            w, l, t = pr["task_wlt"]
            lines.append(
                f"{f_col} & {_disp_method(m)} & {diff_str} & {w}/{l}/{t} & "
                f"{_fmt_p(pr['paired_t_two_sided_p'])} & {_fmt_p(pr['holm_two_sided_t_p'])} \\\\"
            )

    lines.append(r"\midrule")
    pool = prosp["pooled_72_task_standardized_vs_ref"]
    for idx, m in enumerate(comp_only):
        f_col = r"\multirow{4}{*}{Pooled DES ($72$)}" if idx == 0 else ""
        pr = pool[m]
        lo, hi = pr["ci95"]
        diff_str = rf"${_fmt_num(pr['mean_diff'], 4, signed=True)}\ [{_fmt_num(lo, 4)},\,{_fmt_num(hi, 4)}]$"
        w, l, t = pr["task_wlt"]
        lines.append(
            f"{f_col} & {_disp_method(m)} & {diff_str} & {w}/{l}/{t} & "
            f"{_fmt_p(pr['paired_t_two_sided_p'])} & {_fmt_p(pr['holm_two_sided_t_p'])} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines) + "\n"


def build_table_surge_compliance(prosp: dict[str, Any], mp_opsd: dict[str, Any]) -> str:
    fams = ("Inventory_CrossHorizon", "Queueing_CrossHorizon", "OPSD_EuroGrid", "Traffic_Ingolstadt")
    methods = (
        "GP-FTA-ECKG",
        "GP-FTA-SingleSlotKG",
        "GP-FTA-FeasibleKG",
        "GP-FTA-LOO-ECKG",
        "PublicPsi-TargetGP-TS",
    )
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Operating-policy risk, cascade guard/rescue activity, and posterior shrinkage. Operating S/A/U counts safe, borderline, and unsafe policies under independent audit intervals, including retained anchors ($63.49\%$ mean anchor violation in traffic). Guard/Rescue counts Slot-1 objective overrides and Slot-1/Slot-2 fallback rescues. For grid, Same-Law gives verification-pool point exceedances above $5\%$ (parentheses give mean risk); Electronic Companion Table~\ref*{tab:verification_audit} reports interleaved audit-pool counts.}\label{tab:surge_compliance}",
        r"\scriptsize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llcccccccc}",
        r"\toprule",
        r"\textbf{Family (Anc.\ Viol.)} & \textbf{Method} & \textbf{Oper.\ S/A/U} & \textbf{Oper.\ $p_t(X_t^{\mathrm{op}})$} & \textbf{Guard/Resc.} & \textbf{$\mathcal{L}_{10\%}$} & \textbf{$\mathcal{L}_{\mathrm{comp}}$} & \textbf{Same-Law / Init.\ $\hat{\lambda}_g$} & \textbf{Final $\hat{\lambda}_g^{\mathrm{SURE}}$} & \textbf{Revisit \%} \\",
    ]
    for fam in fams:
        lines.append(r"\midrule")
        is_opsd = fam == "OPSD_EuroGrid"
        fd = mp_opsd["families"][fam] if is_opsd else prosp["families"][fam]
        if is_opsd:
            fam_lbl = "OPSD-336 (0.0\\%)"
        else:
            anc_v = 100.0 * fd["anchor_task_mean_viol_prob"]
            fam_lbl = f"{SHORT_FAM[fam]} ({anc_v:.1f}\\%)"
        for idx, m in enumerate(methods):
            md = fd["methods"][m]
            f_col = rf"\multirow{{5}}{{*}}{{{fam_lbl}}}" if idx == 0 else ""
            os_c = md["operating_safety_counts"]
            op_sau = f"{os_c['safe']}/{os_c['borderline']}/{os_c['unsafe']}"
            op_v = f"{100.0 * md['operating_viol_prob_task_mean']:.2f}\\%"
            gr = f"{md['guard_overrides']}/{md['backup_rescues']}"
            fr10 = f"${_fmt_num(md['fixed_reject_loss_10pct_task_mean'], 4)}$"
            comp = f"${_fmt_num(md['compliance_penalized_loss_task_mean'], 4)}$"
            if is_opsd:
                l0 = f"{md['verif_law_exceed_alpha_count']}/{md['verif_law_scored_count']} ({100.0*md['verif_law_viol_prob_mean']:.2f}\\%)"
                if m in {"GP-FTA-ECKG", "GP-FTA-SingleSlotKG", "GP-FTA-FeasibleKG"}:
                    lf = f"{md['final_sure_lambda_con_mean']:.3f}"
                elif m == "GP-FTA-LOO-ECKG":
                    lf = "LOO"
                else:
                    lf = "---"
            else:
                st = md["sure_telemetry"]
                if m in {"GP-FTA-ECKG", "GP-FTA-SingleSlotKG", "GP-FTA-FeasibleKG"}:
                    l0 = f"{st['init_lambda_con_mean']:.3f}"
                    lf = f"{st['final_lambda_con_mean']:.3f}"
                elif m == "GP-FTA-LOO-ECKG":
                    l0 = "---"
                    lf = f"$\\beta_g={st['final_beta_con_mean']:.2f}$"
                else:
                    l0, lf = "---", "---"
            rev_pct = f"{100.0 * (md['mean_replication_queries'] / 60.0):.1f}\\%"
            lines.append(
                f"{f_col} & {_disp_method(m)} & {op_sau} & {op_v} & {gr} & {fr10} & {comp} & {l0} & {lf} & {rev_pct} \\\\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def build_table_metaprior_decomposition(mp_opsd: dict[str, Any], mp_prop: dict[str, Any]) -> str:
    fams = ("Inventory_CrossHorizon", "Queueing_CrossHorizon", "OPSD_EuroGrid", "Traffic_Ingolstadt")
    arms = ("ColdPsi-TargetGP-ECKG", "ColdInit-SourceBasisWeak-ECKG", "ColdInit-MetaPrior-SURE-ECKG", "GP-FTA-ECKG")
    contrasts = ("pure_metaprior_vs_cold_psi", "coef_coupling_vs_basis_weak", "warmstart_increment_over_cold_metaprior")
    models = ("TargetGP-RawPsi", "MetaPrior-Scrambled", "MetaPrior-DiagSURE", "MetaPrior-FullSURE")
    lines = [
        r"\begin{table}[htbp]\centering\scriptsize",
        r"\caption{Closed-loop source-prior comparisons and held-out predictions after eight shared observations. In Panel A positive differences favor ColdMeta over ColdPsi or BasisWeak, and FullWarm over ColdMeta. Parentheses give the Holm-adjusted paired \(p\) for the two prespecified cold contrasts in each family and the exploratory unadjusted \(p\) for the warm contrast. The warm configuration changes initialization, subsequent proposals and source variance regularization. Panel B reports objective rank correlation. The last three columns give standardized RMS changes in summary, objective and constraint for the selected raw-policy perturbations.}\label{tab:metaprior_decomp}",
        r"\resizebox{\textwidth}{!}{\begin{tabular}{lrrrrrrr}\toprule",
        r"\multicolumn{8}{l}{\textbf{A. Operating loss and task-paired contrasts}} \\",
        r"Domain & ColdPsi & BasisWeak & ColdMeta & FullWarm & ColdPsi $-$ ColdMeta & BasisWeak $-$ ColdMeta & ColdMeta $-$ FullWarm \\",
        r"\midrule",
    ]
    prediction_rows = []
    for family in fams:
        group = mp_opsd["families"][family]
        pred = mp_prop["families"][family]
        label = "OPSD-336" if family == "OPSD_EuroGrid" else SHORT_FAM[family]
        cells = [f"{label} ({pred['n_tasks']})"]
        for arm in arms:
            cells.append(f"${_fmt_num(group['methods'][arm]['status_quo_loss_task_mean'], 4)}$")
        for contrast in contrasts:
            stats = group["metaprior_ladder_contrasts"][contrast]
            pvalue = stats["paired_t_two_sided_p"] if contrast == contrasts[-1] else stats["holm_two_sided_t_p"]
            cells.append(f"${_fmt_num(stats['mean_diff'], 5, signed=True)}$ ({_fmt_p(pvalue)})")
        lines.append(" & ".join(cells) + r" \\")
        cells = [label] + [f"${pred['models'][model]['spearman_obj']:.3f}$" for model in models]
        fiber = pred["psi_fiber_audit"]
        for name in ("mean_rel_psi_rms_dist", "fiber_obj_rms_over_sd", "fiber_con_rms_over_sd"):
            cells.append(f"${fiber[name]:.2f}$")
        prediction_rows.append(" & ".join(cells) + r" \\")
    lines.extend([
        r"\bottomrule\end{tabular}}\par\smallskip",
        r"\begin{tabular}{lrrrrrrr}\toprule",
        r"\multicolumn{8}{l}{\textbf{B. Held-out policy ranks and perturbation sizes}} \\",
        r"Domain & Raw & Scrambled & Diagonal & Full & $\psi$ RMS & Objective RMS & Constraint RMS \\",
        r"\midrule",
        *prediction_rows,
        r"\bottomrule\end{tabular}\end{table}",
    ])
    return "\n".join(lines) + "\n"


def build_table_direct_propagation(dprop_exp: dict[str, Any], dprop_conf: dict[str, Any]) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Direct single-probe cross-policy information propagation ($N_0=8 \to 9$ policies) on 32 disjoint held-out policies per task, holding public $\psi(x)$, source-learned basis $\tilde{\phi}(\psi(x))$, target-only within-policy HVD, and probe observations identical across models. Zero-mean kernel models (`Full', `Diag', `Iso', `Perm.') set active source means to zero ($\mu_{0,\mathrm{act}}\equiv 0$) and match active-subspace trace so pre-data shape NRMSE equals $1.000$ identically. Positive differences (`Comparator $-$ Structured') indicate lower post-probe demeaned shape NRMSE for the structured model; parentheses give four-way Holm-adjusted paired-$t$ $p$-values within each domain-head block (`+/$-$/0' counts tasks where the ninth policy reduces/increases/ties `Full' error). Panel~A evaluates the existing 86-task panel; Panel~B executes the pre-registered one-shot confirmation on 72 unseen DES tasks.}\label{tab:direct_prop}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.8pt}",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"\textbf{Domain ($n$)} & \textbf{Head} & \textbf{Full After} & \textbf{+/$-$/0} & \textbf{Iso $-$ Full} & \textbf{Diag $-$ Full} & \textbf{Perm.\ $-$ Full} & \textbf{Online Weak $-$ Full} \\",
        r"\midrule",
        r"\multicolumn{8}{l}{\textbf{Panel A: Exploratory 86-Task Four-Domain Panel (\texttt{exploratory\_86}, $8 \to 9$ transition)}} \\",
        r"\midrule",
    ]

    def _emit_panel(readout: dict[str, Any], fam_list: tuple[tuple[str, str], ...]) -> None:
        for idx_f, (fk, flbl) in enumerate(fam_list):
            fb = readout["families"][fk]["regime_8to9"]
            nt = readout["families"][fk]["n_tasks"]
            for h_i, (hk, hlbl) in enumerate((("obj", "Obj."), ("con", "Con."))):
                d_col = rf"\multirow{{2}}{{*}}{{{flbl} ({nt})}}" if h_i == 0 else ""
                m_full = fb["models"]["Kernel-FullSource"]
                after_s = f"${m_full[f'after_nrmse_{hk}']:.3f}$"
                p, ng, z = m_full[f"pos_neg_zero_tasks_{hk}"]
                pnz_s = f"{p}/{ng}/{z}"
                kc = fb["contrasts"][f"kernel_post_accuracy_{hk}"]
                oc = fb["contrasts"][f"online_post_accuracy_{hk}"]
                c_iso = kc["Kernel_Iso_minus_Full_AfterErr"]
                c_diag = kc["Kernel_Diag_minus_Full_AfterErr"]
                c_perm = kc["Kernel_Permuted_minus_Full_AfterErr"]
                c_onl = oc["Online_FullSURE_vs_Weak_AfterErr"]
                cells = [
                    d_col,
                    hlbl,
                    after_s,
                    pnz_s,
                    f"${_fmt_num(c_iso['mean_diff'], 3, signed=True)}$ ({_fmt_p(c_iso['holm_two_sided_t_p'])})",
                    f"${_fmt_num(c_diag['mean_diff'], 3, signed=True)}$ ({_fmt_p(c_diag['holm_two_sided_t_p'])})",
                    f"${_fmt_num(c_perm['mean_diff'], 3, signed=True)}$ ({_fmt_p(c_perm['holm_two_sided_t_p'])})",
                    f"${_fmt_num(c_onl['mean_diff'], 3, signed=True)}$ ({_fmt_p(c_onl['holm_two_sided_t_p'])})",
                ]
                lines.append(" & ".join(cells) + r" \\")
            if idx_f < len(fam_list) - 1:
                lines.append(r"\cmidrule(lr){1-8}")

    _emit_panel(
        dprop_exp,
        (
            ("Inventory_CrossHorizon", "Inventory"),
            ("Queueing_CrossHorizon", "Queueing"),
            ("OPSD_EuroGrid", "OPSD-336"),
            ("Traffic_Ingolstadt", "Traffic-21"),
            ("Pooled_DES_72", "Pooled DES"),
        ),
    )
    lines.extend([
        r"\midrule",
        r"\multicolumn{8}{l}{\textbf{Panel B: Pre-Registered Confirmatory 72-Task Unseen DES Panel (\texttt{confirm\_fresh\_72}, $8 \to 9$ transition)}} \\",
        r"\midrule",
    ])
    _emit_panel(
        dprop_conf,
        (
            ("Inventory_CrossHorizon", "Inventory"),
            ("Queueing_CrossHorizon", "Queueing"),
            ("Traffic_Ingolstadt", "Traffic-21"),
            ("Pooled_DES_72", "Pooled DES"),
        ),
    )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def build_table_prefix_census(prefix: dict[str, Any], ref_val: dict[str, Any], mc_cei: dict[str, Any]) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Menu diagnostics, 24-scramble single-nomination reference values, and \texttt{UB-cEI} comparisons. Panel~A profiles 96 diagnostic states (screen cap 24). Panels~B and~C evaluate 24 production-stream states ($256\times 1{,}024$ Sobol nodes across $24$ independent scrambles per state under the frozen working model). Panel~B reports mean single-nomination gains ($\times 10^{-3}$), wins or ties of \textbf{EC-KG} against \texttt{UB-cEI} \citep[Section~5]{ungredda2024bayesian}, ratio of group means (`Mean Ratio'), and unweighted average of state-wise ratios (`State Ratio'; $^{\ast}$5/6 non-saturated states in late traffic, excluding one near-zero state with $\Delta^{\mathrm{term}}_{\max}=3.1\times 10^{-10}$). Across all 24 states, \textbf{EC-KG} beats or ties \texttt{SingleSlotKG} in $22/24$, \texttt{FeasibleKG} in $21/24$, \texttt{UB-cEI} in $22/24$, and \texttt{Gate-cEI} in $23/24$ ($0.06$--$0.07$s fast score vs.\ $12.6$--$18.8$s all-160 reference; $3/24$ states have negative single-nomination gain). Panel~C reports Stage~A screen membership, missed revisit vs.\ new-candidate winners, all-160 Spearman correlation, menu MAE, screen loss, and query loss.}\label{tab:prefix_census}",
        r"\scriptsize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llcccccccc}",
        r"\toprule",
        r"\multicolumn{10}{l}{\textbf{Panel A: 96-State Menu-Proxy Mechanism Diagnostic ($n=24$ tasks per family, up to $24$ of $160$ candidates)}} \\",
        r"\textbf{Stage} & \textbf{Family} & \textbf{$|\mathcal{D}_n|$} & \textbf{Diff.\ vs.\ Slot-0} & \textbf{Diff.\ vs.\ Feas.} & \textbf{EC-KG Revisit} & \textbf{$\widehat{\mathrm{VOI}}^{\mathrm{EC}}$ ($\times 10^{-3}$)} & \textbf{Refine / Admit} & \textbf{Slot-0 ($\times 10^{-3}$)} & \textbf{Wall Mean / P95 (s)} \\",
        r"\midrule",
    ]
    for st_key, st_lbl in (("step_0", "Initial ($n=0$)"), ("step_59", "Near-term.\ ($n=59$)")):
        for fam, fam_lbl in (("Queueing_CrossHorizon", "Queueing"), ("Traffic_Ingolstadt", "Traffic-21")):
            row = prefix["families"][fam][st_key]
            n = row["n_states"]
            d_sz = f"{row['mean_unique_evaluated']:.1f}"
            d_s0 = f"{row['disagreed_with_single_slot_count']}/{n} ({100.0 * row['disagreed_with_single_slot_rate']:.1f}\\%)"
            d_fs = f"{row['disagreed_with_feasible_kg_count']}/{n} ({100.0 * row['disagreed_with_feasible_kg_rate']:.1f}\\%)"
            rev = f"{row['eckg_revisit_count']}/{n} ({100.0 * row['eckg_revisit_rate']:.1f}\\%)"
            v_tot = 1000.0 * row["mean_eckg_voi_total"]
            v_ref = 1000.0 * row["mean_eckg_voi_refine"]
            sh_ref = 100.0 * v_ref / max(v_tot, 1e-12)
            sh_adm = 100.0 - sh_ref
            v_s0 = 1000.0 * row["mean_single_voi_under_eckg"]
            w_str = f"{row['mean_step_wall_ms_eckg']/1000.0:.2f} / {row['p95_step_wall_ms_eckg']/1000.0:.2f}"
            lines.append(
                f"{st_lbl} & {fam_lbl} & {d_sz} & {d_s0} & {d_fs} & {rev} & "
                f"${v_tot:.3f}$ & {sh_ref:.1f}\\% / {sh_adm:.1f}\\% & ${v_s0:.3f}$ & {w_str} \\\\"
            )

    lines.extend([
        r"\midrule",
        r"\multicolumn{10}{l}{\textbf{B. 24-Scramble Single-Nomination Reference Gains ($\times 10^{-3}$), \texttt{UB-cEI} Comparison, and Ratio Metrics}} \\",
        r"\textbf{Stage} & \textbf{Family} & \textbf{All-160} & \textbf{EC-KG} & \textbf{Slot-0} & \textbf{Feas.} & \textbf{UB-cEI} & \textbf{EC $\ge$ cEI} & \textbf{Mean Ratio} & \textbf{State Ratio} \\",
        r"\midrule",
    ])
    reference_rows = []
    for stage, label in (("step_0", "Initial"), ("step_59", "Step 59")):
        for family in ("Queueing_CrossHorizon", "Traffic_Ingolstadt"):
            row = ref_val["families"][family][stage]
            mrow = mc_cei["families"][family][stage]
            n = row["n_states"]
            cells = [label, SHORT_FAM[family]]
            gains = [
                f"{mrow['opt_term_mean_x1e3']:.3f}",
                f"{mrow['ec_term_mean_x1e3']:.3f}",
                f"{mrow['single_term_mean_x1e3']:.3f}",
                f"{mrow['feas_term_mean_x1e3']:.3f}",
                f"{mrow['ub_cei_term_mean_x1e3']:.3f}",
            ]
            win_cei = f"{mrow['ec_beats_or_ties_ub_cei_count']}/{n}"
            r_mean = f"{100.0 * mrow['ratio_of_means_24scr']:.1f}\\%"
            star = "$^{\\ast}$" if mrow["nonsaturated_states_count"] < n else ""
            r_state = f"{100.0 * mrow['mean_of_state_ratios_24scr']:.1f}\\%{star}"
            lines.append(" & ".join(cells + gains + [win_cei, r_mean, r_state]) + r" \\")
            hits = [f"{round(n * row[f'stage_a_top16_contains_{m}_opt_rate'])}/{n}" for m in ("menu", "term")]
            miss_split = f"{mrow['stage_a_missed_revisit_count']}/{mrow['stage_a_missed_new_count']}"
            nrho = row["quadrature_spearman_defined_states"]
            rho = f"{row['quadrature_spearman_mean']:.3f} ({nrho}/{n})" if nrho else "---"
            losses = [f"{row[k]:.3f}" for k in (
                "mean_abs_fast_minus_reference_menu_x1e3", "reference_nomination_screen_loss_mean_x1e3",
                "reference_nomination_query_loss_mean_x1e3",
            )]
            reference_rows.append(
                " & ".join(cells + [f"{row['mean_screened_candidates']:.1f}"] + hits + [miss_split, rho] + losses) + r" \\"
            )
    lines.extend([
        r"\midrule",
        r"\multicolumn{10}{l}{\textbf{C. Logged Stage A Membership, Missed-Winner Anatomy, and All-Candidate Finite-Grid Error}} \\",
        r"\textbf{Stage} & \textbf{Family} & \textbf{Screen} & \textbf{Menu Hits} & \textbf{Nom.\ Hits} & \textbf{Miss (Rev/New)} & \textbf{Spearman (Def.)} & \textbf{MAE ($\times 10^{-3}$)} & \textbf{Screen Loss} & \textbf{Query Loss} \\",
        r"\midrule", *reference_rows,
    ])
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def build_referee_tables(readout: dict[str, Any], replay_readout: dict[str, Any]) -> tuple[str, str]:
    names = {
        "TargetWeak-GateKG": r"\texttt{TargetWeak-GateKG}",
        "HistoryDiag-GateKG": r"\texttt{HistoryDiag-GateKG}",
        "HistoryFull-GateKG": r"\texttt{HistoryFull-GateKG}",
        "HistoryFull-FeasibleKG": r"\texttt{HistoryFull-FeasibleKG}",
        "InitialRelease-GateKG": r"\texttt{InitialRelease-GateKG}",
    }
    policy = [
        r"\begin{table}[htbp]\centering",
        r"\caption{Matched coefficient-prior and single-slot acquisition comparisons on 48 unseen queueing and traffic tasks ($384$ paid runs plus exact initialization-release replay, cold-DCT initialization, shared local proposals). Slot-0 Loss evaluates deploying the primary nominee alone at $K_0=112$ on the identical search paths.}\label{tab:matched_policy}",
        r"\scriptsize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llccccccc}\toprule",
        r"\textbf{Family} & \textbf{Matched Arm} & \textbf{Slot-0 $\mathcal{L}_{\mathrm{SQ}}$} & \textbf{Cascade $\mathcal{L}_{\mathrm{SQ}}$} & \textbf{Verif.\ Calls} & \textbf{Cert./runs} & \textbf{Oper.\ S/A/U} & \textbf{Oper.\ $p_t(X_t^{\mathrm{op}})$} & \textbf{Revisit \%} \\",
    ]
    comparisons = [
        r"\begin{table}[htbp]\centering",
        r"\caption{Paired task-level operating-loss contrasts on the matched 48-task panel ($n=24$ tasks per family). Positive differences favor the latter arm; Holm adjustment spans all ten contrasts.}\label{tab:matched_contrasts}",
        r"\scriptsize\begin{tabular}{llccc}\toprule",
        r"\textbf{Family} & \textbf{Contrast (Former $\to$ Latter)} & \textbf{Former $-$ Latter [95\% CI]} & \textbf{W/L/T} & \textbf{Holm $p$} \\",
    ]
    for family, values in readout["families"].items():
        policy.append(r"\midrule")
        comparisons.append(r"\midrule")
        short_family = SHORT_FAM[family]
        rep_arms = replay_readout["families"][family]["arms"]
        for j, (arm, row) in enumerate(values["arms"].items()):
            counts = "/".join(str(row["operating_audits"].get(key, 0)) for key in
                              ("confidently_safe", "boundary_ambiguous", "confidently_unsafe"))
            fam = rf"\multirow{{5}}{{*}}{{{short_family}}}" if j == 0 else ""
            s0_loss = f"${_fmt_num(rep_arms[arm]['first_slot']['mean_loss'], 4)}$" if arm in rep_arms else "---"
            rev_sh = f"{100.0 * row['revisit_share']:.1f}\\%" if "revisit_share" in row else "---"
            policy.append(
                f"{fam} & {names[arm]} & {s0_loss} & ${_fmt_num(row['mean_operating_loss'], 4)}$ & "
                f"{row['mean_terminal_calls']:.1f} & {row['certified']}/{row['runs']} & {counts} & "
                rf"{100*row['mean_operating_violation_rate']:.2f}\% & {rev_sh} \\"
            )
        for j, (label, row) in enumerate(values["comparisons"].items()):
            before, after = label.split(" to ")
            lo, hi = row["ci95"]
            wins = "/".join(map(str, row["latter_wins_losses_ties"]))
            fam = rf"\multirow{{5}}{{*}}{{{short_family}}}" if j == 0 else ""
            comparisons.append(
                rf"{fam} & {names[before]} $\to$ {names[after]} & "
                rf"${_fmt_num(row['improvement'], 4, signed=True)}\ [{_fmt_num(lo, 4)},\,{_fmt_num(hi, 4)}]$ & {wins} & "
                rf"{_fmt_p(row['p_holm'])} \\"
            )
    policy.extend([r"\bottomrule\end{tabular}%", r"}", r"\end{table}"])
    comparisons.extend([r"\bottomrule\end{tabular}", r"\end{table}"])
    return "\n".join(policy) + "\n", "\n".join(comparisons) + "\n"


def build_table_broad_summary(
    rows_u3: list[dict[str, str]],
    rows_messy: list[dict[str, str]],
    rows_sumo_u3: list[dict[str, str]],
) -> str:
    """Compact main-body overview across all four operational domains, raw-tail OPSD, messy logs, and live SUMO."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Cross-domain summary of the 9,796-run four-family benchmark, ENTSO-E grid raw-tail sensitivity, heterogeneous single-replication enterprise archive ($r_a=1$, $\alpha_m \neq \alpha_t$), and calibrated microscopic \texttt{Ingolstadt21} SUMO suite (complete per-method tables in Sections~\ref*{ec:shared_panel}--\ref*{ec:archived_sensitivity}).}\label{tab:broad_summary}",
        r"\scriptsize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"\textbf{Suite \& Domain} & \textbf{Tasks ($d_s \to d_t$)} & \textbf{GP-FTA-KG $\mathcal{L}_{\mathrm{SQ}}$} & \textbf{PublicPsi $\mathcal{L}_{\mathrm{SQ}}$} & \textbf{Best Raw-Space BO} & \textbf{RGPE (Warped)} & \textbf{GP-FTA Cert.} & \textbf{Raw-BO Wins} \\",
        r"\midrule",
    ]
    suite_specs = [
        ("Standard Archive ($N=60$)", "Inventory_CrossHorizon", rows_u3, 60, "16 ($72\\!\\to\\!240$)"),
        ("Standard Archive ($N=60$)", "Queueing_CrossHorizon", rows_u3, 60, "16 ($72\\!\\to\\!240$)"),
        ("Standard Archive ($N=60$)", "OPSD_EuroGrid", rows_u3, 60, "14 ($168\\!\\to\\!336$)"),
        ("Standard Archive ($N=60$)", "Traffic_Ingolstadt", rows_u3, 60, "16 ($22\\!\\to\\!44$)"),
        ("OPSD Raw Tails ($N=60$)", "OPSD_EuroGrid_Unclamped", rows_u3, 60, "14 ($168\\!\\to\\!336$)"),
        ("Messy Archive ($N=60$)", "Inventory_CrossHorizon_Messy", rows_messy, 60, "16 ($r_a\\!\\ge\\!1$)"),
        ("Messy Archive ($N=60$)", "Queueing_CrossHorizon_Messy", rows_messy, 60, "16 ($r_a\\!\\ge\\!1$)"),
        ("Messy Archive ($N=60$)", "OPSD_EuroGrid_Messy", rows_messy, 60, "14 ($r_a\\!\\ge\\!1$)"),
        ("Messy Archive ($N=60$)", "Traffic_Ingolstadt_Messy", rows_messy, 60, "16 ($r_a\\!\\ge\\!1$)"),
    ]
    raw_cands = ["SCBO", "MAP-SAAS-CBO", "HetGP-KG", "HyperBO-EB", "Target-SPSA"]
    for suite_lbl, fam, rset, bud, dim_str in suite_specs:
        agg = _task_agg(rset, fam)["by_budget"][bud]
        dec = 4 if "OPSD" in fam else 3
        v_fta = agg["GP-FTA-KG"]["mean_loss"]
        v_psi = agg["PublicPsi-TargetGP-TS"]["mean_loss"]
        avail_raw = [m for m in raw_cands if m in agg]
        best_raw_m = min(avail_raw, key=lambda m: agg[m]["mean_loss"]) if avail_raw else None
        v_raw_s = f"${_fmt_num(agg[best_raw_m]['mean_loss'], dec)}$ (\\texttt{{{best_raw_m}}})" if best_raw_m else "---"
        v_rgpe = f"${_fmt_num(agg['RGPE']['mean_loss'], dec)}$" if "RGPE" in agg else "---"
        cert_s = _certified_count(agg["GP-FTA-KG"])
        n_raw_beaten = sum(1 for m in avail_raw + (["RGPE"] if "RGPE" in agg else []) if v_fta < agg[m]["mean_loss"])
        n_raw_tot = len(avail_raw) + (1 if "RGPE" in agg else 0)
        fam_clean = SHORT_FAM.get(fam, "OPSD Raw")
        lines.append(
            f"{suite_lbl}: {fam_clean} & {dim_str} & ${_fmt_num(v_fta, dec)}$ & ${_fmt_num(v_psi, dec)}$ & "
            f"{v_raw_s} & {v_rgpe} & {cert_s} & {n_raw_beaten}/{n_raw_tot} \\\\"
        )

    # Calibrated Live SUMO row
    s_agg = _task_agg(rows_sumo_u3, "Ingolstadt21_LiveSUMO_Calibrated")["by_budget"][15]
    avail_s_raw = [m for m in raw_cands if m in s_agg]
    best_s_raw = min(avail_s_raw, key=lambda m: s_agg[m]["mean_loss"])
    n_s_win = sum(1 for m in avail_s_raw + ["RGPE"] if s_agg["GP-FTA-KG"]["mean_loss"] < s_agg[m]["mean_loss"])
    lines.append(
        f"Live SUMO ($N=15$): Ingolstadt-21 & 8 configs ($d_t\\!=\\!44$) & "
        f"$\\mathbf{{{_fmt_num(s_agg['GP-FTA-KG']['mean_loss'], 4)}}}$ & ${_fmt_num(s_agg['PublicPsi-TargetGP-TS']['mean_loss'], 4)}$ & "
        f"${_fmt_num(s_agg[best_s_raw]['mean_loss'], 4)}$ (\\texttt{{{best_s_raw}}}) & ${_fmt_num(s_agg['RGPE']['mean_loss'], 4)}$ & "
        f"{_certified_count(s_agg['GP-FTA-KG'])} & {n_s_win}/{len(avail_s_raw)+1} \\\\"
    )

    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"])
    return "\n".join(lines) + "\n"


# ==============================================================================
# ELECTRONIC COMPANION TABLES
# ==============================================================================

def build_table_main(rows_u3: list[dict[str, str]]) -> str:
    fams = ["Inventory_CrossHorizon", "Queueing_CrossHorizon", "OPSD_EuroGrid", "Traffic_Ingolstadt"]
    ext_baselines = ["SCBO-on-Psi", "RGPE-on-Psi", "SCBO", "MAP-SAAS-CBO", "HetGP-KG", "RGPE", "HyperBO-EB", "Target-SPSA"]
    controls = ["GP-FTA-TS", "PublicPsi-TargetGP-TS", "SCBO-on-Psi", "RGPE-on-Psi"]
    nat_holm_set = ext_baselines[2:]
    display_order = ["GP-FTA-KG", *controls, *nat_holm_set, "STOP"]

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Shared-panel four-family comparisons across 14 or 16 tasks per family (two paired streams per task, frozen runtime \\texttt{5965482dba30}). Reported means, observed search-plus-verification calls, and task comparisons are at $N=60$ except the $N=15$ column and zero-step \\texttt{STOP}.}\\label{tab:main_confirm}",
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
        dec = 4 if fam.startswith("OPSD") else 3

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
            fr60 = f"$\\mathbf{{{_fmt_num(fr60_val, dec)}}}$" if abs(fr60_val - min_fr60) < 1e-12 else f"${_fmt_num(fr60_val, dec)}$"
            if abs(b60[m]["mean_loss"] - min_sq60) < 1e-12:
                c60 = f"$\\mathbf{{{c60[1:-1]}}}$"
            cert_str = _certified_count(b60[m])
            call_str = f"{b60[m]['mean_calls']:.1f}"

            if m == "GP-FTA-KG":
                gain_str, p_str = "---", "---"
            elif m in nat_holm_set:
                rpct, w, l, t, _ = comp_info[m]
                gain_str = f"${_fmt_num(rpct, 1, signed=True)}\\%$ ({w}/{l}/{t})"
                p_str = _fmt_p(holm_nat[m])
            else:
                rpct, w, l, t, p_raw = comp_info[m]
                gain_str = f"${_fmt_num(rpct, 1, signed=True)}\\%$ ({w}/{l}/{t})"
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
        "\\caption{Same-$\\psi$ controls, mechanism ablations (including \\texttt{Untempered-PriorResid} and \\texttt{Init-Frozen-Transfer}, which freezes GP kernel hyperparameters after paid initialization), and unclipped ENTSO-E OPSD raw-tail sensitivity. Two-sided Wilcoxon $p$ values compare $N=60$ status-quo loss against \\textbf{GP-FTA-KG}.}\\label{tab:ablations}",
        "\\scriptsize",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llccccccc}",
        "\\toprule",
        "\\textbf{Family} & \\textbf{Mechanism / Same-$\\psi$ Control} & \\textbf{$N=8$} & \\textbf{$N=15$} & \\textbf{$N=30$} & \\textbf{$N=60$ $\\mathcal{L}_{\\mathrm{SQ}}$} & \\textbf{$N=60$ $\\mathcal{L}_{10\\%}$} & \\textbf{$N=60$ Cert.\\%} & \\textbf{Two-sided $p$} \\\\",
        "\\midrule",
    ]

    for fam in fams:
        agg = _task_agg(rows_u3, fam)["by_budget"]
        dec = 4 if fam.startswith("OPSD") else 3
        mins = {b: min(agg[b][m]["mean_loss"] for m in ctrl_arms) for b in (8, 15, 30, 60)}
        min_fr = min(agg[60][m]["mean_fr10"] for m in ctrl_arms)
        ref60 = agg[60]["GP-FTA-KG"]["loss_vec"]

        for r_i, m in enumerate(ctrl_arms):
            fam_col = f"\\multirow{{{len(ctrl_arms)}}}{{*}}{{{SHORT_FAM[fam]}}}" if r_i == 0 else ""
            m_disp = _disp_method(m)
            vals = []
            for b in (8, 15, 30, 60):
                v = agg[b][m]["mean_loss"]
                s = f"$\\mathbf{{{_fmt_num(v, dec)}}}$" if abs(v - mins[b]) < 1e-12 else f"${_fmt_num(v, dec)}$"
                vals.append(s)
            frv = agg[60][m]["mean_fr10"]
            s_fr = f"$\\mathbf{{{_fmt_num(frv, dec)}}}$" if abs(frv - min_fr) < 1e-12 else f"${_fmt_num(frv, dec)}$"
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
        s15 = f"$\\mathbf{{{_fmt_num(v15, 4)}}}$" if abs(v15 - min15) < 1e-12 else f"${_fmt_num(v15, 4)}$"
        s60 = f"$\\mathbf{{{_fmt_num(v60, 4)}}}$" if abs(v60 - min60) < 1e-12 else f"${_fmt_num(v60, 4)}$"
        sfr = f"$\\mathbf{{{_fmt_num(fr60, 4)}}}$" if abs(fr60 - min_fr_u) < 1e-12 else f"${_fmt_num(fr60, 4)}$"
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
        "\\caption{Heterogeneous enterprise source archives ($2{,}108$ runs). Source logs mix single-replication records ($r_a=1$), $3.7\\times$ unequal task budgets, incomplete solver subsets, and mismatched historical risk tolerances ($\\alpha_m \\in [0.02, 0.10]$).}\\label{tab:messy_archive}",
        "\\scriptsize",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llcccccc}",
        "\\toprule",
        "\\textbf{Family} & \\textbf{Method} & \\textbf{$N=15$ $\\mathcal{L}_{\\mathrm{SQ}}$} & \\textbf{$N=60$ $\\mathcal{L}_{\\mathrm{SQ}}$} & \\textbf{$N=60$ $\\mathcal{L}_{10\\%}$} & \\textbf{$N=60$ Cert.\\%} & \\textbf{Two-sided $p$ ($N=60$)} & \\textbf{Online Wall (s)} \\\\",
        "\\midrule",
    ]
    for f_idx, fam in enumerate(fams):
        agg = _task_agg(rows_messy, fam)["by_budget"]
        dec = 4 if "OPSD" in fam else 3
        min15 = min(agg[15][m]["mean_loss"] for m in arms)
        min60 = min(agg[60][m]["mean_loss"] for m in arms)
        minfr = min(agg[60][m]["mean_fr10"] for m in arms)
        ref60 = agg[60]["GP-FTA-KG"]["loss_vec"]

        for r_i, m in enumerate(arms):
            fam_col = f"\\multirow{{{len(arms)}}}{{*}}{{{SHORT_FAM[fam]}}}" if r_i == 0 else ""
            m_disp = _disp_method(m)
            v15, v60, fr60 = agg[15][m]["mean_loss"], agg[60][m]["mean_loss"], agg[60][m]["mean_fr10"]
            s15 = f"$\\mathbf{{{_fmt_num(v15, dec)}}}$" if abs(v15 - min15) < 1e-12 else f"${_fmt_num(v15, dec)}$"
            s60 = f"$\\mathbf{{{_fmt_num(v60, dec)}}}$" if abs(v60 - min60) < 1e-12 else f"${_fmt_num(v60, dec)}$"
            sfr = f"$\\mathbf{{{_fmt_num(fr60, dec)}}}$" if abs(fr60 - minfr) < 1e-12 else f"${_fmt_num(fr60, dec)}$"
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
        "\\caption{Microscopic \\texttt{Ingolstadt21} SUMO evaluation ($N=15$). The calibrated panel evaluates eight threshold and weight configurations (two streams each); all 40 uncalibrated stress runs ended in abstention (\\texttt{Gaussian-Margin-KG} denotes the predecessor $z_{1-\\alpha}$ Gaussian-margin lookahead).}\\label{tab:live_sumo}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{4.6pt}",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "\\textbf{Regime} & \\textbf{Method} & \\textbf{Status-Quo Loss} & \\textbf{Challenger $J_t^{\\ddagger}$} & \\textbf{Cert./runs} & \\textbf{Cert.\ S/A/U} & \\textbf{Actual Calls} \\\\",
        "\\midrule",
    ]
    min_loss = min(agg_u3[m]["mean_loss"] for m in u3_order)
    for r_i, m in enumerate(u3_order):
        md = agg_u3[m]
        reg_col = f"\\multirow{{{len(u3_order)}}}{{*}}{{Calibrated ($144$ runs)}}" if r_i == 0 else ""
        m_disp = _disp_method(m)
        l_str = f"${_fmt_num(md['mean_loss'], 4)} \\pm {_fmt_num(md['se_loss'], 4)}$"
        if abs(md["mean_loss"] - min_loss) < 1e-12:
            l_str = f"$\\mathbf{{{_fmt_num(md['mean_loss'], 4)}}} \\pm {_fmt_num(md['se_loss'], 4)}$"
        lines.append(
            f"{reg_col} & {m_disp} & {l_str} & ${_fmt_num(md['mean_prim'], 4)}$ & "
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
        m_disp = _disp_method(m)
        sq = float(np.mean([float(x["status_quo_loss"]) for x in mrows]))
        pr = float(np.mean([float(x["uncertified_selected_candidate_obj"]) for x in mrows]))
        counts = _audit_counts(mrows)
        calls = float(np.mean([float(x["total_target_calls"]) for x in mrows]))
        lines.append(f"{reg_col} & {m_disp} & ${_fmt_num(sq, 4)}$ & ${_fmt_num(pr, 4)}$ & {_certified_count(counts)} & --- & {calls:.1f} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\par\\smallskip\\parbox{\\textwidth}{\\scriptsize $^{\\ddagger}$Calibrated runs report the frozen Slot-0 primary objective; uncalibrated stress runs report the best uncertified candidate (never deployed under strict abstention).}",
        "\\end{table}",
    ])
    return "\n".join(lines)


def build_table_verification_audit(
    rows_prosp: list[dict[str, str]],
    rows_mp_opsd: list[dict[str, str]],
    rows_u3: list[dict[str, str]],
    rows_messy: list[dict[str, str]],
    rows_sumo_u3: list[dict[str, str]],
    rows_sumo_p0: list[dict[str, str]],
) -> str:
    fta = {"GP-FTA-KG", "GP-FTA-TS"}
    rows_opsd = [row for row in rows_u3 if row["family"].startswith("OPSD")]
    rows_opsd_fta = [row for row in rows_opsd if row["method_name"] in fta]
    rows_mp_grid = [row for row in rows_mp_opsd if row["family"] == "OPSD_EuroGrid"]
    strata = [
        ("Prospective 720-run DES panel, all 5 arms", rows_prosp),
        ("Prospective 720-run DES panel, GP-FTA-ECKG", [row for row in rows_prosp if row["method_name"] == "GP-FTA-ECKG"]),
        ("Prospective 336-dim OPSD Grid, all 8 arms", rows_mp_grid),
        ("Prospective 336-dim OPSD Grid, GP-FTA-ECKG", [row for row in rows_mp_grid if row["method_name"] == "GP-FTA-ECKG"]),
        ("Cold-init meta-prior DES ladder, all 3 arms", [row for row in rows_mp_opsd if row["family"] != "OPSD_EuroGrid"]),
        ("Shared 4-family \\& raw-tail suite, all methods", rows_u3),
        ("Shared OPSD Grid (clipped + raw), all methods", rows_opsd),
        ("Shared OPSD Grid (clipped + raw), GP-FTA-KG/TS", rows_opsd_fta),
        ("Heterogeneous enterprise archive, all methods", rows_messy),
        ("Calibrated live SUMO, all methods", rows_sumo_u3),
        ("Uncalibrated live SUMO stress runs", rows_sumo_p0),
    ]
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Verification and audit outcomes among certified policies. The two exceedance columns count point estimates above $5\\%$, with their respective scored denominators. Grid verification-pool rechecks use 8,000 draws per deployment. S/A/U classifies the audit-pool 95\\% Clopper--Pearson interval as below, overlapping or above $5\\%$. Grid block pools span the same historical period.}\\label{tab:verification_audit}",
        "\\scriptsize",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "\\textbf{Cohort and Subset} & \\textbf{Cert./runs} & \\textbf{Abstained} & \\textbf{Verif.\\ pool $>$5\\%} & \\textbf{Audit pool $>$5\\%} & \\textbf{Cert.\\ S/A/U} & \\textbf{Unclassified} \\\\",
        "\\midrule",
    ]
    for title, rows in strata:
        stats = _audit_counts(rows)
        scored = stats["same_law_scored"]
        same_law = f"{stats['same_law_exceed']}/{scored}" if scored else "---"
        audit = f"{stats['audit_exceed']}/{stats['audit_scored']}" if stats["audit_scored"] else "---"
        lines.append(
            f"{title} & {_certified_count(stats)} & {stats['n_abstained']} & "
            f"{same_law} & {audit} & {_format_audit_classes(stats)} & {stats['audit_unclassified']} \\\\"
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
        r"\caption{Archived 432-run panel evaluating quadrature-aligned single-slot \textbf{GP-FTA-KG} (\texttt{LOO-PLL} posterior) at $N=60$ across 72 tasks (two streams per task).}\label{tab:independent_panel}",
        r"\scriptsize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrccccc}",
        r"\toprule",
        r"\textbf{Family} & \textbf{Method} & \textbf{$\mathcal{L}_{\mathrm{SQ}}$} & \textbf{Calls} & "
        r"\textbf{Cert./runs} & \textbf{Cert.\ S/A/U} & \textbf{Exceed./audited} & "
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
                gain = rf"${_fmt_num(paired['comparator_minus_kg_mean'], 4, signed=True)}\ [{_fmt_num(lower, 4)},\,{_fmt_num(upper, 4)}]$"
                p_value = _fmt_p(paired["paired_t_p_holm_six_tests"])
            family_col = rf"\multirow{{3}}{{*}}{{{SHORT_FAM[family]}}}" if index == 0 else ""
            lines.append(
                f"{family_col} & {_disp_method(method)} & ${_fmt_num(arm['mean_status_quo_loss'], 4)}$ & "
                f"{arm['mean_total_target_calls']:.2f} & {arm['certified']}/{arm['runs']} & "
                rf"{safety} & {exceed} & {gain} & {p_value} \\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}%", "}", r"\end{table}"])
    return "\n".join(lines)


def build_table_operating_policy(readout: dict[str, Any]) -> str:
    methods = ("GP-FTA-KG", "GP-FTA-TS", "PublicPsi-TargetGP-TS")
    classes = ("confidently_safe", "boundary_ambiguous", "confidently_unsafe")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Operating-policy risk classifications on the archived 432-run single-slot KG panel.}\label{tab:operating_policy}",
        r"\scriptsize",
        r"\begin{tabular}{llccrrr}",
        r"\toprule",
        r"\textbf{Family} & \textbf{Method} & \textbf{Cert./runs} & \textbf{Retained SQ} & "
        r"\textbf{Safe} & \textbf{Ambiguous} & \textbf{Unsafe} \\",
    ]
    for family in ("Inventory_CrossHorizon", "Queueing_CrossHorizon", "Traffic_Ingolstadt"):
        lines.append(r"\midrule")
        for index, method in enumerate(methods):
            arm = readout["families"][family]["arms"][method]
            family_col = rf"\multirow{{3}}{{*}}{{{SHORT_FAM[family]}}}" if index == 0 else ""
            safe, ambiguous, unsafe = (arm["operating_policy_classes"].get(label, 0) for label in classes)
            lines.append(
                f"{family_col} & {_disp_method(method)} & {arm['certified']}/{arm['runs']} & "
                rf"{arm['retained_anchor']} & {safe} & {ambiguous} & {unsafe} \\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def build_referee_variances(readout: dict[str, Any]) -> str:
    lines = [
        r"\begin{table}[htbp]\centering",
        r"\caption{Mean squared variance estimation error against 2,000-replication independent audits on the matched 48-task panel (fixed initial-batch within-SSQ vs.\ pooled revisit within-SSQ).}\label{tab:matched_variances}",
        r"\scriptsize\begin{tabular}{llrcccc}\toprule",
        r"& & & \multicolumn{2}{c}{\textbf{Objective variance MSE}} & \multicolumn{2}{c}{\textbf{Constraint variance MSE}} \\",
        r"\textbf{Family} & \textbf{Policy role} & \textbf{Runs} & \textbf{Fixed batch} & \textbf{Pooled} & \textbf{Fixed batch} & \textbf{Pooled} \\",
    ]
    for family, values in readout["families"].items():
        lines.append(r"\midrule")
        short_family = SHORT_FAM[family]
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
        r"\caption{Counterfactual Slot-0-only ($K_0=112$) versus full three-slot guarded cascade verification on identical realized search paths ($48$ runs per arm).}\label{tab:terminal_replay}",
        r"\scriptsize\begin{tabular}{llrrccr}\toprule",
        r"& & \multicolumn{2}{c}{\textbf{Mean $\mathcal{L}_{\mathrm{SQ}}$}} & \multicolumn{2}{c}{\textbf{Certified / 48}} & \textbf{Verif.\ calls} \\",
        r"\textbf{Family} & \textbf{Matched Arm} & \textbf{Slot 0} & \textbf{Full cascade} & \textbf{Slot 0} & \textbf{Full cascade} & \textbf{Full cascade} \\",
    ]
    labels = {
        "TargetWeak-GateKG": r"\texttt{TargetWeak-GateKG}",
        "HistoryDiag-GateKG": r"\texttt{HistoryDiag-GateKG}",
        "HistoryFull-GateKG": r"\texttt{HistoryFull-GateKG}",
        "HistoryFull-FeasibleKG": r"\texttt{HistoryFull-FeasibleKG}",
    }
    for family, values in readout["families"].items():
        lines.append(r"\midrule")
        family_name = SHORT_FAM[family]
        for name, label in labels.items():
            arm = values["arms"][name]
            first = arm["first_slot"]
            full = arm["full_terminal"]
            lines.append(
                f"{family_name} & {label} & ${_fmt_num(first['mean_loss'], 5)}$ & ${_fmt_num(full['mean_loss'], 5)}$ & "
                rf"{first['certified']}/48 & {full['certified']}/48 & {full['mean_terminal_calls']:.2f} \\"
            )
            family_name = ""
    lines.extend([r"\bottomrule\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def build_table_amortization(rows_u3: list[dict[str, str]], b_src: int = 1428) -> str:
    """Honest dual comparator amortization: against BOTH raw-space SCBO (N=60) and PublicPsi-TargetGP-TS (N=60)."""
    fams = ["Inventory_CrossHorizon", "Queueing_CrossHorizon", "OPSD_EuroGrid", "Traffic_Ingolstadt"]
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Source-archive simulation-call amortization ($B_{\\mathrm{src}}=1{,}428$ calls per family) comparing low-budget \\textbf{GP-FTA-KG} ($N=15$) against both raw-policy \\texttt{SCBO} ($N=60$) and same-representation \\texttt{PublicPsi-TargetGP-TS} ($N=60$). Full-cost break-even $M_{\\mathrm{break}}^{\\mathrm{full}}$ is reported only when \\textbf{GP-FTA-KG} ($N=15$) attains equal or lower mean $\\mathcal{L}_{\\mathrm{SQ}}$ than the $N=60$ cold-start comparator (`Not dom.' indicates the $N=60$ cold-start representation control achieves lower loss).}\\label{tab:amortization}",
        "\\scriptsize",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llcccccc}",
        "\\toprule",
        "\\textbf{Comparator ($N=60$)} & \\textbf{Family} & \\textbf{GP-FTA ($N\\!=\\!15$)} & \\textbf{Cold ($N\\!=\\!60$)} & \\textbf{$B_{\\mathrm{FTA},15}$ / $B_{\\mathrm{cold},60}$} & \\textbf{Call Saving $\\Delta B$ [95\\% CI]} & \\textbf{$M_{\\mathrm{break}}^{\\mathrm{sunk}}$} & \\textbf{$M_{\\mathrm{break}}^{\\mathrm{full}}$ (95\\% UB)} \\\\",
        "\\midrule",
    ]
    for c_idx, comp in enumerate(("SCBO", "PublicPsi-TargetGP-TS")):
        for r_i, fam in enumerate(fams):
            agg = _task_agg(rows_u3, fam)["by_budget"]
            fta15 = agg[15]["GP-FTA-KG"]
            cold60 = agg[60][comp]
            dec = 4 if fam.startswith("OPSD") else 3
            diff_calls = cold60["call_vec"] - fta15["call_vec"]
            n_t = len(diff_calls)
            m_sav = float(np.mean(diff_calls))
            se_sav = float(np.std(diff_calls, ddof=1) / np.sqrt(n_t))
            t_crit = float(student_t.ppf(0.975, df=n_t - 1))
            ci_lo = m_sav - t_crit * se_sav
            ci_hi = m_sav + t_crit * se_sav

            c_col = f"\\multirow{{4}}{{*}}{{\\texttt{{{comp}}}}}" if r_i == 0 else ""
            qual_dom = fta15["mean_loss"] <= cold60["mean_loss"]
            if qual_dom and m_sav > 0:
                mb_pt = math.ceil(b_src / m_sav)
                mb_ub = f"$\\le {math.ceil(b_src / ci_lo)}$" if ci_lo > 0 else "$\\infty$"
                sunk_s = "$1$"
                full_s = f"$\\mathbf{{{mb_pt}}}$ ({mb_ub})"
            else:
                sunk_s = "Not dom."
                full_s = "Not dom."

            lines.append(
                f"{c_col} & {SHORT_FAM[fam]} & ${_fmt_num(fta15['mean_loss'], dec)}$ & ${_fmt_num(cold60['mean_loss'], dec)}$ & "
                f"{fta15['mean_calls']:.1f} / {cold60['mean_calls']:.1f} & "
                f"${_fmt_num(m_sav, 1, signed=True)}$ $[{_fmt_num(ci_lo, 1, signed=True)}, {_fmt_num(ci_hi, 1, signed=True)}]$ & "
                f"{sunk_s} & {full_s} \\\\"
            )
        if c_idx == 0:
            lines.append("\\midrule")
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"])
    return "\n".join(lines) + "\n"


# ==============================================================================
# PUBLICATION VECTOR FIGURES (2 Main Body + 1 Electronic Companion)
# ==============================================================================

def build_figures(
    fig_dir: Path,
    prosp: dict[str, Any],
    mp_opsd: dict[str, Any],
    mc_cei: dict[str, Any],
    rows_u3: list[dict[str, str]],
    rows_messy: list[dict[str, str]],
    p1_diag: dict[str, Any],
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8.6,
        "axes.titlesize": 9.2,
        "axes.labelsize": 8.6,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "pdf.fonttype": 42,
    })

    # --------------------------------------------------------------------------
    # Figure 1 (Main Body): Prospective 4-Domain Controlled Contrasts + 24-Scramble Single-Nomination Reference Gain
    # --------------------------------------------------------------------------
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(9.6, 3.9), gridspec_kw={"width_ratios": [1.15, 1.05]})

    groups = [
        ("Inventory\n($n=24$)", "Inventory_CrossHorizon"),
        ("Queueing\n($n=24$)", "Queueing_CrossHorizon"),
        ("OPSD-336\n($n=14$)", "OPSD_EuroGrid"),
        ("Traffic-21\n($n=24$)", "Traffic_Ingolstadt"),
        ("Pooled DES\n($n=72$)", "POOLED"),
    ]
    comp_specs = [
        ("GP-FTA-SingleSlotKG", "GP-FTA-SingleSlotKG", "#0072B2", "o", -0.22),
        ("GP-FTA-FeasibleKG", "GP-FTA-FeasibleKG", "#D55E00", "s", -0.07),
        ("GP-FTA-LOO-ECKG", "GP-FTA-LOO-ECKG", "#CC79A7", "^", +0.08),
        ("PublicPsi-TargetGP-TS", "PublicPsi-TargetGP-TS", "#009E73", "D", +0.23),
    ]

    y_base = np.arange(len(groups))
    ax_l.axvline(0.0, color="#333333", linestyle="--", linewidth=1.0, zorder=1)
    for m_key, m_lbl, col, mk, y_off in comp_specs:
        means, lo_err, hi_err = [], [], []
        for _, g_key in groups:
            if g_key == "POOLED":
                st = prosp["pooled_72_task_standardized_vs_ref"][m_key]
                scale = 1.0
            elif g_key == "OPSD_EuroGrid":
                fd = mp_opsd["families"][g_key]
                st = fd["opsd_336_primary_contrasts"][m_key]
                scale = max(fd["methods"]["GP-FTA-ECKG"]["status_quo_loss_task_se"] * math.sqrt(14.0), 1e-6)
            else:
                fd = prosp["families"][g_key]
                st = fd["paired_vs_ref"][m_key]["status_quo_loss"]
                scale = max(fd["methods"]["GP-FTA-ECKG"]["status_quo_loss_task_se"] * math.sqrt(24.0), 1e-6)
            m_v = st["mean_diff"] / scale
            c_lo, c_hi = st["ci95"][0] / scale, st["ci95"][1] / scale
            means.append(m_v)
            lo_err.append(m_v - c_lo)
            hi_err.append(c_hi - m_v)
        ax_l.errorbar(
            means, y_base + y_off, xerr=[lo_err, hi_err],
            fmt=mk, color=col, markersize=4.8, linewidth=1.3, capsize=2.2, label=m_lbl, zorder=3,
        )

    ax_l.set_yticks(y_base)
    ax_l.set_yticklabels([g[0] for g in groups])
    ax_l.invert_yaxis()
    ax_l.set_xlim(-0.62, 0.88)
    ax_l.set_xlabel("Standardized Paired Loss Reduction vs. GP-FTA-ECKG (95% CI, $>0$ favors ECKG)")
    ax_l.set_title("(a) Prospective 4-Domain Controlled Contrasts", fontweight="bold")
    ax_l.legend(loc="lower right", fontsize=6.7, framealpha=0.95)
    ax_l.grid(True, axis="x", linestyle=":", alpha=0.55)

    # Panel (b): Ratio of Mean 24-Scramble 256x1024 Single-Nomination Gain to All-160 Optimum (incl. UB-cEI)
    bars_x = np.arange(4)
    bar_labels = ["Queueing\nStep 0", "Queueing\nStep 59", "Traffic-21\nStep 0", "Traffic-21\nStep 59"]
    keys_b = [
        ("Queueing_CrossHorizon", "step_0"),
        ("Queueing_CrossHorizon", "step_59"),
        ("Traffic_Ingolstadt", "step_0"),
        ("Traffic_Ingolstadt", "step_59"),
    ]
    pct_eckg, pct_single, pct_feas, pct_ubcei = [], [], [], []
    for fk, sk in keys_b:
        mv = mc_cei["families"][fk][sk]
        denom = max(mv["opt_term_mean_x1e3"], 1e-9)
        pct_eckg.append(100.0 * mv["ec_term_mean_x1e3"] / denom)
        pct_single.append(100.0 * mv["single_term_mean_x1e3"] / denom)
        pct_feas.append(100.0 * mv["feas_term_mean_x1e3"] / denom)
        pct_ubcei.append(100.0 * mv["ub_cei_term_mean_x1e3"] / denom)

    bw = 0.19
    ax_r.axhline(100.0, color="#333333", linestyle="--", linewidth=0.95, label="All-160 Reference Optimum (100%)")
    ax_r.bar(bars_x - 1.5 * bw, pct_eckg, width=bw, color="#0072B2", label="GP-FTA-ECKG (screen cap 16)")
    ax_r.bar(bars_x - 0.5 * bw, pct_single, width=bw, color="#56B4E9", label="GP-FTA-SingleSlotKG")
    ax_r.bar(bars_x + 0.5 * bw, pct_feas, width=bw, color="#D55E00", label="GP-FTA-FeasibleKG")
    ax_r.bar(bars_x + 1.5 * bw, pct_ubcei, width=bw, color="#009E73", label="UB-cEI (Ungredda & Branke 2024)")
    ax_r.set_xticks(bars_x)
    ax_r.set_xticklabels(bar_labels)
    ax_r.set_ylim(0.0, 152.0)
    ax_r.set_yticks([0, 20, 40, 60, 80, 100])
    ax_r.set_ylabel("Ratio of mean gain to all-160 optimum (%)")
    ax_r.set_title("(b) 24-Scramble Single-Nomination Gain ($256\\times 1024$)", fontweight="bold")
    ax_r.legend(loc="upper center", fontsize=6.2, frameon=True, framealpha=0.95)
    ax_r.grid(True, axis="y", linestyle=":", alpha=0.55)

    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_prospective_eckg.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(fig_dir / "fig1_prospective_eckg.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Figure 2 (Main Body): 2x2 Broad 4-Family Scaling Curves (Zero "(Ours)")
    # --------------------------------------------------------------------------
    fams = ["Inventory_CrossHorizon", "Queueing_CrossHorizon", "OPSD_EuroGrid", "Traffic_Ingolstadt"]
    palette = [
        ("GP-FTA-KG", "GP-FTA-KG", "#0072B2", "o", "-", 2.1),
        ("GP-FTA-TS", "GP-FTA-TS", "#56B4E9", "v", "-", 1.7),
        ("PublicPsi-TargetGP-TS", "PublicPsi-TargetGP-TS", "#009E73", "D", "-.", 1.5),
        ("RGPE-on-Psi", "RGPE-on-Psi", "#D55E00", "s", "--", 1.4),
        ("SCBO-on-Psi", "SCBO-on-Psi", "#CC79A7", "P", ":", 1.4),
        ("SCBO", "SCBO (Raw Space)", "#E69F00", "^", "--", 1.3),
        ("STOP", "STOP (N=0)", "#555555", "x", ":", 1.1),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 5.8))
    for idx, fam in enumerate(fams):
        ax = axes.flatten()[idx]
        agg = _task_agg(rows_u3, fam)["by_budget"]
        xs = [8, 15, 30, 60]
        for m_key, m_lbl, col, mk, ls, lw in palette:
            ys = [agg[x][m_key]["mean_loss"] for x in xs]
            ses = [agg[x][m_key]["se_loss"] for x in xs]
            ax.errorbar(
                xs, ys, yerr=ses, label=m_lbl, color=col, marker=mk,
                linestyle=ls, linewidth=lw, markersize=4.4, capsize=2.0,
            )
        ax.set_title(PLOT_TITLES[fam], fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xlabel("Online Target Step Budget $N$")
        ax.set_ylabel("Status-Quo Decision Loss $\\mathcal{L}_{\\mathrm{SQ}}$ ($\\downarrow$)")
        ax.grid(True, linestyle=":", alpha=0.5)
    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=True, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(fig_dir / "fig2_horizon_scaling.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(fig_dir / "fig2_horizon_scaling.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Figure EC.1 (Electronic Companion): Variance Calibration & Complete 6-Bar Messy Archive
    # --------------------------------------------------------------------------
    fig, (ax1, ax_m) = plt.subplots(1, 2, figsize=(9.6, 3.8), gridspec_kw={"width_ratios": [0.88, 1.22]})
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
    ax1.set_ylim(0.0, max(rmse_naive) * 1.25)
    ax1.set_ylabel("Out-of-Sample $\\log v_t(x)$ RMSE ($\\downarrow$)")
    ax1.set_title("(a) Heteroscedastic Calibration ($1.3\\times$ Shift)", fontweight="bold")
    ax1.legend(loc="upper right", fontsize=7.2)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.5)

    messy_fams = [
        ("Inventory_CrossHorizon_Messy", "Inventory"),
        ("Queueing_CrossHorizon_Messy", "Queueing"),
        ("OPSD_EuroGrid_Messy", "OPSD Grid"),
        ("Traffic_Ingolstadt_Messy", "Traffic-21"),
    ]
    m_aggs = {mf: _task_agg(rows_messy, mf)["by_budget"][60] for mf, _ in messy_fams}
    # Includes BOTH PublicPsi-TargetGP-TS and GP-FTA-NoDebias (zero omitted controls, zero "(Ours)")
    comp_bars = [
        ("GP-FTA-KG", "GP-FTA-KG", "#0072B2"),
        ("GP-FTA-TS", "GP-FTA-TS", "#56B4E9"),
        ("GP-FTA-NoDebias", "GP-FTA-NoDebias", "#332288"),
        ("PublicPsi-TargetGP-TS", "PublicPsi-TargetGP-TS", "#009E73"),
        ("RGPE-on-Psi", "RGPE-on-Psi", "#D55E00"),
        ("WarmStart-PatternSearch", "WarmStart-PS", "#CC79A7"),
    ]
    x_m = np.arange(len(messy_fams))
    bw = 0.132
    all_gains = []
    for idx_b, (ak, albl, col) in enumerate(comp_bars):
        gains = [
            float(np.mean(100.0 * (m_aggs[mf][ak]["anc_vec"] - m_aggs[mf][ak]["fr10_vec"]) / m_aggs[mf][ak]["anc_vec"]))
            for mf, _ in messy_fams
        ]
        all_gains.extend(gains)
        ax_m.bar(x_m + (idx_b - 2.5) * bw, gains, width=bw, color=col, alpha=0.92, label=albl)
    ax_m.axhline(0.0, color="#333333", linewidth=1.0)
    ax_m.set_xticks(x_m)
    ax_m.set_xticklabels([lbl for _, lbl in messy_fams])
    ax_m.set_ylim(min(min(all_gains) * 1.18, -2.0), max(all_gains) * 1.36)
    ax_m.set_ylabel("Net Gain on $\\mathcal{L}_{10\\%}$ vs. Status Quo (%, $\\uparrow$)")
    ax_m.set_title("(b) Heterogeneous Archive ($r_a\\geq 1$, $\\alpha_m \\neq \\alpha_t$)", fontweight="bold")
    ax_m.legend(loc="upper center", ncol=3, fontsize=6.7, framealpha=0.94)
    ax_m.grid(True, axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(fig_dir / "fig3_diagnostics_and_messy.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(fig_dir / "fig3_diagnostics_and_messy.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=MS_DIR, help="Directory for regenerated tables and figures.")
    args = parser.parse_args()
    tab_dir = args.output_dir / "tables"
    tab_dir.mkdir(parents=True, exist_ok=True)

    rows_prosp = _load_ledger("v2_or_eckg_sure_confirm_v1")
    prosp_readout = json.loads((ART_DIR / "v2_or_eckg_sure_confirm_v1/prospective_panel_readout.json").read_text(encoding="utf-8"))
    prefix_readout = json.loads((ART_DIR / "v2_or_eckg_sure_confirm_v1/eckg_prefix_audit_readout.json").read_text(encoding="utf-8"))
    ref_val_readout = json.loads((ART_DIR / "v2_or_eckg_sure_confirm_v1/eckg_reference_validation_corrected.json").read_text(encoding="utf-8"))
    mc_cei_readout = json.loads((ART_DIR / "v2_or_eckg_sure_confirm_v1/eckg_multiscramble_cei_readout.json").read_text(encoding="utf-8"))

    rows_mp_opsd = _load_ledger("v2_or_metaprior_opsd_confirm_v1")
    mp_opsd_readout = json.loads((ART_DIR / "v2_or_metaprior_opsd_confirm_v1/metaprior_opsd_panel_readout.json").read_text(encoding="utf-8"))
    mp_prop_readout = json.loads((ART_DIR / "v2_or_metaprior_opsd_confirm_v1/metaprior_propagation_readout.json").read_text(encoding="utf-8"))
    dprop_exp_readout = json.loads((ART_DIR / "v2_or_metaprior_opsd_confirm_v1/direct_propagation_exploratory_86_readout.json").read_text(encoding="utf-8"))
    dprop_conf_readout = json.loads((ART_DIR / "v2_or_metaprior_opsd_confirm_v1/direct_propagation_confirm_fresh_72_readout.json").read_text(encoding="utf-8"))

    rows_u3 = _load_ledger("v2_or_u3_confirm_v1")
    rows_messy = _load_ledger("v2_or_u3_messy_archive_v1")
    rows_sumo_u3 = _load_ledger("v2_or_u3_live_sumo_v1")
    with (ART_DIR / "v2_or_p0_reaudit_v1/v2_or_live_sumo_v1/per_run_ledger.csv").open("r", encoding="utf-8") as fh:
        rows_sumo_p0 = list(csv.DictReader(fh))
    p1_diag = json.loads((ART_DIR / "v2_or_p1_diagnostics_v1/p1_diagnostics_summary.json").read_text(encoding="utf-8"))

    wall_map = _mean_online_wall_by_method(rows_prosp, rows_mp_opsd)

    # Main-body tables (7 tables)
    (tab_dir / "table_prospective_main.tex").write_text(
        build_table_prospective_main(prosp_readout, mp_opsd_readout, wall_map), encoding="utf-8"
    )
    (tab_dir / "table_prospective_contrasts.tex").write_text(
        build_table_prospective_contrasts(prosp_readout, mp_opsd_readout), encoding="utf-8"
    )
    (tab_dir / "table_surge_compliance.tex").write_text(build_table_surge_compliance(prosp_readout, mp_opsd_readout), encoding="utf-8")
    (tab_dir / "table_metaprior_decomposition.tex").write_text(build_table_metaprior_decomposition(mp_opsd_readout, mp_prop_readout), encoding="utf-8")
    (tab_dir / "table_direct_propagation.tex").write_text(
        build_table_direct_propagation(dprop_exp_readout, dprop_conf_readout), encoding="utf-8"
    )
    (tab_dir / "table_prefix_census.tex").write_text(
        build_table_prefix_census(prefix_readout, ref_val_readout, mc_cei_readout), encoding="utf-8"
    )
    (tab_dir / "table_broad_summary.tex").write_text(
        build_table_broad_summary(rows_u3, rows_messy, rows_sumo_u3), encoding="utf-8"
    )

    referee_readout = json.loads((ART_DIR / "v2_or_gpfta_referee_ablation_v1/referee_ablation_readout.json").read_text(encoding="utf-8"))
    replay_readout = json.loads((ART_DIR / "v2_or_gpfta_referee_ablation_v1/terminal_replay_readout.json").read_text(encoding="utf-8"))
    policy_tab, contrasts_tab = build_referee_tables(referee_readout, replay_readout)
    (tab_dir / "table_matched_policy.tex").write_text(policy_tab, encoding="utf-8")
    (tab_dir / "table_matched_contrasts.tex").write_text(contrasts_tab, encoding="utf-8")
    (tab_dir / "table_matched_variances.tex").write_text(build_referee_variances(referee_readout), encoding="utf-8")
    (tab_dir / "table_terminal_replay.tex").write_text(build_terminal_replay_table(replay_readout), encoding="utf-8")

    # Electronic Companion tables (12 tables total)
    (tab_dir / "table_main.tex").write_text(build_table_main(rows_u3), encoding="utf-8")
    (tab_dir / "table_ablations.tex").write_text(build_table_ablations(rows_u3), encoding="utf-8")
    (tab_dir / "table_messy_archive.tex").write_text(build_table_messy_archive(rows_messy), encoding="utf-8")
    (tab_dir / "table_live_sumo.tex").write_text(build_table_live_sumo(rows_sumo_u3, rows_sumo_p0), encoding="utf-8")
    (tab_dir / "table_amortization.tex").write_text(build_table_amortization(rows_u3), encoding="utf-8")
    (tab_dir / "table_verification_audit.tex").write_text(
        build_table_verification_audit(rows_prosp, rows_mp_opsd, rows_u3, rows_messy, rows_sumo_u3, rows_sumo_p0), encoding="utf-8"
    )

    independent_readout = json.loads((ART_DIR / "v2_or_kg_quadrature_confirm_v1/independent_panel_readout.json").read_text(encoding="utf-8"))
    operating_readout = json.loads((ART_DIR / "v2_or_kg_quadrature_confirm_v1/operating_policy_readout.json").read_text(encoding="utf-8"))
    (tab_dir / "table_independent_panel.tex").write_text(build_table_independent_panel(independent_readout), encoding="utf-8")
    (tab_dir / "table_operating_policy.tex").write_text(build_table_operating_policy(operating_readout), encoding="utf-8")

    build_figures(args.output_dir / "figures", prosp_readout, mp_opsd_readout, mc_cei_readout, rows_u3, rows_messy, p1_diag)
    print(f"Regenerated 19 sealed-ledger tables and 3 vector figures in {args.output_dir}.")


if __name__ == "__main__":
    main()
