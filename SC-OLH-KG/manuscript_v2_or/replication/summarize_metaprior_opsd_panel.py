#!/usr/bin/env python3
"""Regenerate `metaprior_opsd_panel_readout.json` strictly from sealed CSV ledgers.

Requires zero cluster working files (`items/*.json`): reads `per_run_ledger.csv`,
`per_slot_certificates.csv`, and `compact_summary.json` from `v2_or_metaprior_opsd_confirm_v1`
and the sealed campaign manifest for audit replication counts. It pairs the three DES families with sealed
`GP-FTA-ECKG` runs from `v2_or_eckg_sure_confirm_v1` on identical `(target_task_idx, rep_seed)`.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t, ttest_1samp, wilcoxon


def _holm_adjust(p_map: dict[str, float]) -> dict[str, float]:
    items = sorted(p_map.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, float] = {}
    running_max = 0.0
    for idx, (k, p_raw) in enumerate(items):
        adj = min(1.0, float(p_raw) * float(m - idx))
        running_max = max(running_max, adj)
        out[k] = running_max
    return out


def _paired_stats(diff_pos_favors_ref: np.ndarray) -> dict[str, Any]:
    n = int(diff_pos_favors_ref.size)
    mean_d = float(np.mean(diff_pos_favors_ref))
    sd_d = float(np.std(diff_pos_favors_ref, ddof=1)) if n > 1 else 0.0
    se_d = sd_d / np.sqrt(n) if n > 1 else 0.0
    t_crit = float(student_t.ppf(0.975, df=max(n - 1, 1))) if n > 1 else 0.0
    ci95 = [mean_d - t_crit * se_d, mean_d + t_crit * se_d]

    if n < 2 or sd_d < 1e-14:
        p_t_two = 1.0
    else:
        p_t_two = float(ttest_1samp(diff_pos_favors_ref, popmean=0.0, alternative="two-sided").pvalue)

    nz = diff_pos_favors_ref[np.abs(diff_pos_favors_ref) > 1e-12]
    if nz.size == 0:
        p_w_two = 1.0
    else:
        try:
            p_w_two = float(wilcoxon(nz, alternative="two-sided", zero_method="wilcox", method="auto").pvalue)
        except Exception:
            p_w_two = 1.0

    wlt = [
        int(np.sum(diff_pos_favors_ref > 1e-9)),
        int(np.sum(diff_pos_favors_ref < -1e-9)),
        int(np.sum(np.abs(diff_pos_favors_ref) <= 1e-9)),
    ]
    return {
        "n_tasks": n,
        "mean_diff": mean_d,
        "se_diff": se_d,
        "ci95": ci95,
        "task_wlt": wlt,
        "paired_t_two_sided_p": p_t_two,
        "wilcoxon_two_sided_p": p_w_two,
    }


def _norm_safety_bucket(cls_name: str) -> str:
    if cls_name in {"confidently_safe", "confirmed_safe"}:
        return "safe"
    if cls_name in {"boundary_ambiguous", "audit_ci_straddles_alpha"}:
        return "borderline"
    if cls_name in {"confidently_unsafe", "confirmed_violation"}:
        return "unsafe"
    return "abstained"


def _load_rows(campaign_dir: Path, filter_method: str | None = None, alpha: float = 0.05) -> list[dict[str, Any]]:
    ledger_path = campaign_dir / "per_run_ledger.csv"
    cert_path = campaign_dir / "per_slot_certificates.csv"
    with ledger_path.open("r", encoding="utf-8", newline="") as fh:
        raw_rows = list(csv.DictReader(fh))

    cert_by_job: dict[str, list[dict[str, str]]] = {}
    if cert_path.is_file():
        with cert_path.open("r", encoding="utf-8", newline="") as fh:
            for cr in csv.DictReader(fh):
                cert_by_job.setdefault(cr["job_id"], []).append(cr)

    out: list[dict[str, Any]] = []
    for r in raw_rows:
        if filter_method is not None and r["method_name"] != filter_method:
            continue
        cert = r["certified"].strip().lower() == "true"
        anc_obj = float(r["anchor_mc_obj"])
        sq_loss = float(r["status_quo_loss"])
        fr_loss = float(r["fixed_reject_loss_10pct"])
        dep_viol = float(r["deployed_mc_viol_prob"]) if r.get("deployed_mc_viol_prob") not in ("", None) else None
        ver_viol = float(r["verif_law_mc_viol_prob"]) if r.get("verif_law_mc_viol_prob") not in ("", None) else None
        anc_viol = float(r["anchor_mc_viol_prob"]) if r.get("anchor_mc_viol_prob") not in ("", None) else 0.0
        op_viol = dep_viol if (cert and dep_viol is not None) else anc_viol

        dep_cls = r["safety_audit_class"]
        anc_cls = r.get("anchor_safety_audit_class") or "confidently_safe"
        op_cls = dep_cls if cert else anc_cls
        comp_loss = sq_loss + 2.0 * abs(anc_obj) * max(op_viol - alpha, 0.0)
        mech = json.loads(r["mechanism_receipt_json"]) if r.get("mechanism_receipt_json") else {}

        slots = cert_by_job.get(r["job_id"], [])
        slot0_passed = any(int(s["slot_index"]) == 0 and s["certified"].strip().lower() == "true" for s in slots)
        non_primary_deployed = bool(cert and r["selected_role"] != "posterior_primary")
        out.append({
            "job_id": r["job_id"],
            "family": r["family"],
            "target_task_idx": int(r["target_task_idx"]),
            "rep_seed": int(r["rep_seed"]),
            "method_name": r["method_name"],
            "certified": cert,
            "selected_role": r["selected_role"],
            "guard_override": bool(non_primary_deployed and slot0_passed),
            "backup_rescue": bool(non_primary_deployed and not slot0_passed),
            "anchor_obj": anc_obj,
            "anchor_viol": anc_viol,
            "anchor_cls": anc_cls,
            "sq_loss": sq_loss,
            "fr10_loss": fr_loss,
            "comp_loss": comp_loss,
            "dep_viol": dep_viol,
            "ver_viol": ver_viol,
            "op_viol": op_viol,
            "dep_cls": dep_cls,
            "op_cls": op_cls,
            "n_rep_queries": int(r.get("n_replication_queries") or 0),
            "total_target_calls": int(r["total_target_calls"]),
            "audit_calls_used": int(r["audit_calls_used"]),
            "verification_law_audit_calls_used": int(r["verification_law_audit_calls_used"]),
            "online_wall_sec": float(r["online_wall_time_sec"]),
            "mech": mech,
        })
    return out


def reconcile_audits(campaign_dir: Path, compact: dict, rows: list[dict]) -> tuple[dict, dict]:
    """Recover the split of completed research calls in the sealed 656-run cohort."""
    manifest = json.loads((campaign_dir / "sealed_manifest.json").read_bytes())
    grid_families = {
        f["name"]: f["audit_replications"] for f in manifest["families"] if f["kind"] == "opsd"
    }
    grid_rows = [r for r in rows if r["family"] in grid_families]
    if any(r["certified"] != (r["ver_viol"] is not None) for r in grid_rows):
        raise ValueError("Missing or unexpected same-law grid recheck")
    same_law = sum(grid_families[r["family"]] for r in grid_rows if r["ver_viol"] is not None)
    # The frozen OPSD sampler booked every research draw against the policy audit counter.
    total_policy = sum(r["audit_calls_used"] + r["verification_law_audit_calls_used"] for r in rows)
    accounting = dict(compact["call_accounting"])
    if total_policy != accounting["policy_audit_calls_used"] + accounting["verification_law_audit_calls_used"]:
        raise ValueError("Completed research draws disagree between the ledger and the compact summary")
    accounting["policy_audit_calls_used"] = total_policy - same_law
    accounting["verification_law_audit_calls_used"] = same_law

    # Successful OPSD audit calls bypassed attempted_calls in this execution. Keep attempts,
    # CPU times and any failure charges in the original receipts; these are successful calls.
    receipts = compact["stage_attempt_fees"]
    anchor_attempted = receipts["audit_anchors"]["attempted_calls_recorded"].get("audit", 0)
    policy_attempted = sum(receipts["audit_and_summarize"]["attempted_calls_recorded"].values())
    grid_tasks = {(r["family"], r["target_task_idx"]) for r in grid_rows}
    grid_anchor = sum(grid_families[family] for family, _ in grid_tasks)
    grid_policy = sum(r["audit_calls_used"] + r["verification_law_audit_calls_used"] for r in grid_rows)
    reconciliation = {
        "completed_policy_audit_calls_both_pools": total_policy,
        "grid_certified_rechecks": sum(r["ver_viol"] is not None for r in grid_rows),
        "grid_draws_per_certified_recheck_by_family": grid_families,
        "original_attempt_meter_anchor_audit_draws": anchor_attempted,
        "completed_grid_anchor_draws_absent_from_attempt_meter": grid_anchor,
        "original_attempt_meter_policy_audit_draws": policy_attempted,
        "completed_grid_policy_draws_absent_from_attempt_meter_both_pools": grid_policy,
        "research_draws_including_anchors_and_both_policy_pools": accounting["anchor_audit_calls_used"] + total_policy,
    }
    return accounting, reconciliation


def build_readout(
    campaign_dir: Path,
    prospective_dir: Path,
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    compact = json.loads((campaign_dir / "compact_summary.json").read_text(encoding="utf-8"))
    rows_new = _load_rows(campaign_dir, alpha=alpha)
    previous_rows = _load_rows(prospective_dir, alpha=alpha)
    rows_warm = [r for r in previous_rows if r["method_name"] == "GP-FTA-ECKG"]
    run_keys = {
        (r["family"], r["target_task_idx"], r["rep_seed"], r["method_name"])
        for panel in (previous_rows, rows_new) for r in panel
    }
    if len(run_keys) != len(previous_rows) + len(rows_new):
        raise ValueError("Overlapping task-stream-method runs in the two sealed cohorts")
    accounting, reconciliation = reconcile_audits(campaign_dir, compact, rows_new)

    # Combine rows: OPSD_EuroGrid already includes GP-FTA-ECKG in campaign_dir;
    # synthetic families pair with sealed GP-FTA-ECKG from prospective_dir.
    existing_keys = {(r["family"], r["target_task_idx"], r["rep_seed"], r["method_name"]) for r in rows_new}
    all_rows = list(rows_new)
    for rw in rows_warm:
        key = (rw["family"], rw["target_task_idx"], rw["rep_seed"], rw["method_name"])
        if key not in existing_keys:
            all_rows.append(rw)

    families = sorted({r["family"] for r in all_rows})
    metaprior_ladder_arms = (
        "ColdPsi-TargetGP-ECKG",
        "ColdInit-SourceBasisWeak-ECKG",
        "ColdInit-MetaPrior-SURE-ECKG",
        "GP-FTA-ECKG",
    )
    opsd_confirm_comparators = (
        "GP-FTA-SingleSlotKG",
        "GP-FTA-FeasibleKG",
        "GP-FTA-LOO-ECKG",
        "PublicPsi-TargetGP-TS",
    )

    out_fams: dict[str, Any] = {}
    for fam in families:
        f_rows = [r for r in all_rows if r["family"] == fam]
        methods = sorted({r["method_name"] for r in f_rows})
        task_sq: dict[str, dict[int, float]] = {}
        m_summary: dict[str, Any] = {}

        for m in methods:
            sub = [r for r in f_rows if r["method_name"] == m]
            by_t_sq: dict[int, list[float]] = {}
            by_t_fr: dict[int, list[float]] = {}
            by_t_cp: dict[int, list[float]] = {}
            by_t_ov: dict[int, list[float]] = {}
            for r in sub:
                ti = r["target_task_idx"]
                by_t_sq.setdefault(ti, []).append(r["sq_loss"])
                by_t_fr.setdefault(ti, []).append(r["fr10_loss"])
                by_t_cp.setdefault(ti, []).append(r["comp_loss"])
                by_t_ov.setdefault(ti, []).append(r["op_viol"])
            t_sq = {ti: float(np.mean(v)) for ti, v in sorted(by_t_sq.items())}
            task_sq[m] = t_sq
            sq_arr = np.array(list(t_sq.values()), dtype=float)
            fr_arr = np.array([float(np.mean(v)) for _, v in sorted(by_t_fr.items())], dtype=float)
            cp_arr = np.array([float(np.mean(v)) for _, v in sorted(by_t_cp.items())], dtype=float)
            ov_arr = np.array([float(np.mean(v)) for _, v in sorted(by_t_ov.items())], dtype=float)

            dep_b = Counter(_norm_safety_bucket(r["dep_cls"]) for r in sub)
            op_b = Counter(_norm_safety_bucket(r["op_cls"]) for r in sub)
            dep_viols = [r["dep_viol"] for r in sub if r["certified"] and r["dep_viol"] is not None]
            ver_viols = [r["ver_viol"] for r in sub if r["certified"] and r["ver_viol"] is not None]
            sure_con = [r["mech"]["final_sure_lambda_con"] for r in sub if "final_sure_lambda_con" in r["mech"]]
            rel_flags = [1.0 if r["mech"].get("final_source_released") else 0.0 for r in sub if "final_source_released" in r["mech"]]

            m_summary[m] = {
                "n_runs": len(sub),
                "n_tasks": len(sq_arr),
                "n_certified": int(sum(1 for r in sub if r["certified"])),
                "certified_rate": float(np.mean([1.0 if r["certified"] else 0.0 for r in sub])),
                "guard_overrides": int(sum(1 for r in sub if r["guard_override"])),
                "backup_rescues": int(sum(1 for r in sub if r["backup_rescue"])),
                "status_quo_loss_task_mean": float(np.mean(sq_arr)),
                "status_quo_loss_task_se": float(np.std(sq_arr, ddof=1) / np.sqrt(len(sq_arr))),
                "fixed_reject_loss_10pct_task_mean": float(np.mean(fr_arr)),
                "compliance_penalized_loss_task_mean": float(np.mean(cp_arr)),
                "operating_viol_prob_task_mean": float(np.mean(ov_arr)),
                "deployed_viol_prob_mean": float(np.mean(dep_viols)) if dep_viols else None,
                "deployed_audit_exceed_alpha_count": int(sum(1 for v in dep_viols if v > alpha)),
                "verif_law_viol_prob_mean": float(np.mean(ver_viols)) if ver_viols else None,
                "verif_law_exceed_alpha_count": int(sum(1 for v in ver_viols if v > alpha)) if ver_viols else None,
                "verif_law_scored_count": len(ver_viols),
                "deployed_safety_counts": {
                    "safe": int(dep_b["safe"]),
                    "borderline": int(dep_b["borderline"]),
                    "unsafe": int(dep_b["unsafe"]),
                    "abstained": int(dep_b["abstained"]),
                },
                "operating_safety_counts": {
                    "safe": int(op_b["safe"]),
                    "borderline": int(op_b["borderline"]),
                    "unsafe": int(op_b["unsafe"]),
                },
                "mean_total_target_calls": float(np.mean([r["total_target_calls"] for r in sub])),
                "mean_replication_queries": float(np.mean([r["n_rep_queries"] for r in sub])),
                "final_sure_lambda_con_mean": float(np.mean(sure_con)) if sure_con else None,
                "source_released_rate": float(np.mean(rel_flags)) if rel_flags else None,
            }

        # Pre-registered Cold-Init Meta-Prior Decomposition (Holm across the 2 cold-init comparators)
        ladder_contrasts: dict[str, Any] = {}
        if all(a in task_sq for a in metaprior_ladder_arms):
            t_keys = sorted(task_sq["ColdInit-MetaPrior-SURE-ECKG"])
            c_vs_raw = _paired_stats(np.array([
                task_sq["ColdPsi-TargetGP-ECKG"][k] - task_sq["ColdInit-MetaPrior-SURE-ECKG"][k] for k in t_keys
            ]))
            c_vs_weak = _paired_stats(np.array([
                task_sq["ColdInit-SourceBasisWeak-ECKG"][k] - task_sq["ColdInit-MetaPrior-SURE-ECKG"][k] for k in t_keys
            ]))
            holm_cold = _holm_adjust({
                "meta_prior_vs_cold_psi": c_vs_raw["paired_t_two_sided_p"],
                "meta_prior_vs_basis_weak": c_vs_weak["paired_t_two_sided_p"],
            })
            c_vs_raw["holm_two_sided_t_p"] = holm_cold["meta_prior_vs_cold_psi"]
            c_vs_weak["holm_two_sided_t_p"] = holm_cold["meta_prior_vs_basis_weak"]

            c_basis_vs_raw = _paired_stats(np.array([
                task_sq["ColdPsi-TargetGP-ECKG"][k] - task_sq["ColdInit-SourceBasisWeak-ECKG"][k] for k in t_keys
            ]))
            c_warm_vs_cold_meta = _paired_stats(np.array([
                task_sq["ColdInit-MetaPrior-SURE-ECKG"][k] - task_sq["GP-FTA-ECKG"][k] for k in t_keys
            ]))
            c_full_vs_cold_psi = _paired_stats(np.array([
                task_sq["ColdPsi-TargetGP-ECKG"][k] - task_sq["GP-FTA-ECKG"][k] for k in t_keys
            ]))
            ladder_contrasts = {
                "pure_metaprior_vs_cold_psi": c_vs_raw,
                "coef_coupling_vs_basis_weak": c_vs_weak,
                "spectral_basis_vs_cold_psi": c_basis_vs_raw,
                "warmstart_increment_over_cold_metaprior": c_warm_vs_cold_meta,
                "full_gpfta_vs_cold_psi_same_eckg": c_full_vs_cold_psi,
            }

        # OPSD 336-dim 4-comparator confirmation (Holm across the 4 primary comparators)
        opsd_contrasts: dict[str, Any] = {}
        if fam == "OPSD_EuroGrid" and "GP-FTA-ECKG" in task_sq:
            t_keys = sorted(task_sq["GP-FTA-ECKG"])
            raw_p_map: dict[str, float] = {}
            for comp in opsd_confirm_comparators:
                if comp in task_sq:
                    st = _paired_stats(np.array([task_sq[comp][k] - task_sq["GP-FTA-ECKG"][k] for k in t_keys]))
                    opsd_contrasts[comp] = st
                    raw_p_map[comp] = st["paired_t_two_sided_p"]
            for k, hp in _holm_adjust(raw_p_map).items():
                opsd_contrasts[k]["holm_two_sided_t_p"] = hp

        out_fams[fam] = {
            "methods": m_summary,
            "metaprior_ladder_contrasts": ladder_contrasts,
            "opsd_336_primary_contrasts": opsd_contrasts,
        }

    return {
        "campaign_name": compact["campaign_name"],
        "execution_code_sha256": compact["execution_code_sha256"],
        "manifest_sha256": compact["manifest_sha256"],
        "per_run_ledger_csv_sha256": compact["per_run_ledger_csv_sha256"],
        "per_slot_certificates_csv_sha256": compact["per_slot_certificates_csv_sha256"],
        "call_accounting": accounting,
        "audit_reconciliation": reconciliation,
        "unique_runs_across_both_cohorts": len(run_keys),
        "total_new_cells": len(rows_new),
        "families": out_fams,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-dir",
        type=Path,
        default=Path("SC-OLH-KG/paper_artifacts/v2_or_metaprior_opsd_confirm_v1"),
    )
    parser.add_argument(
        "--prospective-dir",
        type=Path,
        default=Path("SC-OLH-KG/paper_artifacts/v2_or_eckg_sure_confirm_v1"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    out = build_readout(args.campaign_dir.resolve(), args.prospective_dir.resolve())
    out_path = (args.output or (args.campaign_dir.resolve() / "metaprior_opsd_panel_readout.json")).resolve()
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(out_path), "total_new_cells": out["total_new_cells"]}))


if __name__ == "__main__":
    main()
