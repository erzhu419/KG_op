"""Ledger-only prospective readout generator for `v2_or_eckg_sure_confirm_v1`.

Computes 100% of task-level and run-level statistics, paired 95% confidence
intervals, two-sided and one-sided paired tests, Holm familywise adjustments,
certified vs operating-policy safety classifications, compliance-penalized
operating losses, and SURE / Eligible-Cascade KG mechanism telemetry directly
from `per_run_ledger.csv` and `compact_summary.json` (no raw cluster `items/`
required).
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t, ttest_1samp, wilcoxon


SC_ROOT = Path(__file__).resolve().parents[2]


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
        p_t_one = 1.0
    else:
        p_t_two = float(ttest_1samp(diff_pos_favors_ref, popmean=0.0, alternative="two-sided").pvalue)
        p_t_one = float(ttest_1samp(diff_pos_favors_ref, popmean=0.0, alternative="greater").pvalue)

    nz = diff_pos_favors_ref[np.abs(diff_pos_favors_ref) > 1e-12]
    if nz.size == 0:
        p_w_two = 1.0
        p_w_one = 1.0
    else:
        try:
            p_w_two = float(wilcoxon(nz, alternative="two-sided", zero_method="wilcox", method="auto").pvalue)
            p_w_one = float(wilcoxon(nz, alternative="greater", zero_method="wilcox", method="auto").pvalue)
        except Exception:
            p_w_two = 1.0
            p_w_one = 1.0

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
        "paired_t_one_sided_p": p_t_one,
        "wilcoxon_two_sided_p": p_w_two,
        "wilcoxon_one_sided_p": p_w_one,
    }


def _norm_safety_bucket(cls_name: str) -> str:
    if cls_name == "confirmed_safe":
        return "safe"
    if cls_name == "audit_ci_straddles_alpha":
        return "borderline"
    if cls_name == "confirmed_violation":
        return "unsafe"
    return "abstained"


def build_readout(campaign_dir: Path, *, ref_arm: str = "GP-FTA-ECKG", alpha: float = 0.05) -> dict[str, Any]:
    ledger_path = campaign_dir / "per_run_ledger.csv"
    compact_path = campaign_dir / "compact_summary.json"
    compact = json.loads(compact_path.read_text(encoding="utf-8"))

    with ledger_path.open("r", encoding="utf-8", newline="") as fh:
        raw_rows = list(csv.DictReader(fh))

    rows: list[dict[str, Any]] = []
    for r in raw_rows:
        cert = r["certified"].strip().lower() == "true"
        anc_obj = float(r["anchor_mc_obj"])
        sq_loss = float(r["status_quo_loss"])
        fr_loss = float(r["fixed_reject_loss_10pct"])
        dep_viol = float(r["deployed_mc_viol_prob"]) if r.get("deployed_mc_viol_prob") not in ("", None) else None
        anc_viol = float(r["anchor_mc_viol_prob"]) if r.get("anchor_mc_viol_prob") not in ("", None) else 0.0
        op_viol = dep_viol if (cert and dep_viol is not None) else anc_viol

        dep_cls = r["safety_audit_class"]
        anc_cls = r.get("anchor_safety_audit_class") or "confirmed_safe"
        op_cls = dep_cls if cert else anc_cls

        # Compliance-penalized operating loss: rewards restoring feasibility under surge shift
        comp_loss = sq_loss + 2.0 * abs(anc_obj) * max(op_viol - alpha, 0.0)
        mech = json.loads(r["mechanism_receipt_json"]) if r.get("mechanism_receipt_json") else {}

        rows.append({
            "family": r["family"],
            "target_task_idx": int(r["target_task_idx"]),
            "rep_seed": int(r["rep_seed"]),
            "method_name": r["method_name"],
            "certified": cert,
            "selected_role": r["selected_role"],
            "anchor_obj": anc_obj,
            "anchor_viol": anc_viol,
            "anchor_cls": anc_cls,
            "sq_loss": sq_loss,
            "fr10_loss": fr_loss,
            "comp_loss": comp_loss,
            "dep_viol": dep_viol,
            "op_viol": op_viol,
            "dep_cls": dep_cls,
            "op_cls": op_cls,
            "n_rep_queries": int(r.get("n_replication_queries") or 0),
            "online_wall_sec": float(r["online_wall_time_sec"]),
            "total_target_calls": int(r["total_target_calls"]),
            "mech": mech,
        })

    families = sorted({r["family"] for r in rows})
    out_families: dict[str, Any] = {}
    pooled_z_diffs: dict[str, list[float]] = {}

    for fam in families:
        f_rows = [r for r in rows if r["family"] == fam]
        methods = sorted({r["method_name"] for r in f_rows})
        m_summary: dict[str, Any] = {}
        task_sq: dict[str, dict[int, float]] = {}
        task_fr: dict[str, dict[int, float]] = {}
        task_comp: dict[str, dict[int, float]] = {}
        task_op_viol: dict[str, dict[int, float]] = {}

        # Status-quo anchor baseline reference for this family
        anc_by_task = {}
        anc_viol_by_task = {}
        anc_cls_counts: Counter[str] = Counter()
        seen_task_seed: set[tuple[int, int]] = set()
        for r in f_rows:
            ts = (r["target_task_idx"], r["rep_seed"])
            if ts not in seen_task_seed:
                seen_task_seed.add(ts)
                anc_by_task[r["target_task_idx"]] = r["anchor_obj"]
                anc_viol_by_task[r["target_task_idx"]] = r["anchor_viol"]
                anc_cls_counts[_norm_safety_bucket(r["anchor_cls"])] += 1

        for m in methods:
            sub = [r for r in f_rows if r["method_name"] == m]
            by_t_sq: dict[int, list[float]] = {}
            by_t_fr: dict[int, list[float]] = {}
            by_t_comp: dict[int, list[float]] = {}
            by_t_opv: dict[int, list[float]] = {}
            for r in sub:
                ti = r["target_task_idx"]
                by_t_sq.setdefault(ti, []).append(r["sq_loss"])
                by_t_fr.setdefault(ti, []).append(r["fr10_loss"])
                by_t_comp.setdefault(ti, []).append(r["comp_loss"])
                by_t_opv.setdefault(ti, []).append(r["op_viol"])

            t_sq = {ti: float(np.mean(v)) for ti, v in sorted(by_t_sq.items())}
            t_fr = {ti: float(np.mean(v)) for ti, v in sorted(by_t_fr.items())}
            t_cp = {ti: float(np.mean(v)) for ti, v in sorted(by_t_comp.items())}
            t_ov = {ti: float(np.mean(v)) for ti, v in sorted(by_t_opv.items())}
            task_sq[m] = t_sq
            task_fr[m] = t_fr
            task_comp[m] = t_cp
            task_op_viol[m] = t_ov

            sq_arr = np.array(list(t_sq.values()), dtype=float)
            fr_arr = np.array(list(t_fr.values()), dtype=float)
            cp_arr = np.array(list(t_cp.values()), dtype=float)
            ov_arr = np.array(list(t_ov.values()), dtype=float)

            dep_buckets = Counter(_norm_safety_bucket(r["dep_cls"]) for r in sub)
            op_buckets = Counter(_norm_safety_bucket(r["op_cls"]) for r in sub)
            dep_viols = [r["dep_viol"] for r in sub if r["certified"] and r["dep_viol"] is not None]

            sure_l_obj = [r["mech"]["final_sure_lambda_obj"] for r in sub if "final_sure_lambda_obj" in r["mech"]]
            sure_l_con = [r["mech"]["final_sure_lambda_con"] for r in sub if "final_sure_lambda_con" in r["mech"]]
            sure_init_obj = [r["mech"]["init_sure_lambda_obj"] for r in sub if "init_sure_lambda_obj" in r["mech"]]
            sure_init_con = [r["mech"]["init_sure_lambda_con"] for r in sub if "init_sure_lambda_con" in r["mech"]]
            rel_flags = [1.0 if r["mech"].get("final_source_released") else 0.0 for r in sub if "final_source_released" in r["mech"]]
            betas_con = [r["mech"]["final_beta_con"] for r in sub if "final_beta_con" in r["mech"]]

            eckg_blocks = [r["mech"]["eckg_summary"] for r in sub if "eckg_summary" in r["mech"]]
            eckg_agg = None
            if eckg_blocks:
                tot_steps = sum(int(b["n_steps"]) for b in eckg_blocks)
                eckg_agg = {
                    "total_online_steps": tot_steps,
                    "disagreed_with_single_slot_rate": float(
                        sum(int(b["disagreed_with_single_slot_steps"]) for b in eckg_blocks) / max(tot_steps, 1)
                    ),
                    "disagreed_with_feasible_kg_rate": float(
                        sum(int(b["disagreed_with_feasible_kg_steps"]) for b in eckg_blocks) / max(tot_steps, 1)
                    ),
                    "chosen_revisit_rate": float(
                        sum(int(b["chosen_revisit_steps"]) for b in eckg_blocks) / max(tot_steps, 1)
                    ),
                    "mean_voi_total": float(np.mean([b["mean_voi_total"] for b in eckg_blocks])),
                    "mean_voi_refine": float(np.mean([b["mean_voi_refine"] for b in eckg_blocks])),
                    "mean_voi_admit": float(np.mean([b["mean_voi_admit"] for b in eckg_blocks])),
                    "mean_step_wall_ms": float(np.mean([b["mean_step_wall_ms"] for b in eckg_blocks])),
                }

            m_summary[m] = {
                "n_runs": len(sub),
                "n_tasks": len(sq_arr),
                "certified_rate": float(np.mean([1.0 if r["certified"] else 0.0 for r in sub])),
                "abstention_rate": float(np.mean([0.0 if r["certified"] else 1.0 for r in sub])),
                "selected_role_breakdown": dict(Counter(r["selected_role"] for r in sub)),
                "status_quo_loss_task_mean": float(np.mean(sq_arr)),
                "status_quo_loss_task_se": float(np.std(sq_arr, ddof=1) / np.sqrt(len(sq_arr))),
                "fixed_reject_loss_10pct_task_mean": float(np.mean(fr_arr)),
                "fixed_reject_loss_10pct_task_se": float(np.std(fr_arr, ddof=1) / np.sqrt(len(fr_arr))),
                "compliance_penalized_loss_task_mean": float(np.mean(cp_arr)),
                "compliance_penalized_loss_task_se": float(np.std(cp_arr, ddof=1) / np.sqrt(len(cp_arr))),
                "deployed_viol_prob_mean": float(np.mean(dep_viols)) if dep_viols else None,
                "operating_viol_prob_task_mean": float(np.mean(ov_arr)),
                "deployed_safety_counts": {
                    "safe": int(dep_buckets["safe"]),
                    "borderline": int(dep_buckets["borderline"]),
                    "unsafe": int(dep_buckets["unsafe"]),
                    "abstained": int(dep_buckets["abstained"]),
                },
                "operating_safety_counts": {
                    "safe": int(op_buckets["safe"]),
                    "borderline": int(op_buckets["borderline"]),
                    "unsafe": int(op_buckets["unsafe"]),
                },
                "mean_replication_queries": float(np.mean([r["n_rep_queries"] for r in sub])),
                "mean_total_target_calls": float(np.mean([r["total_target_calls"] for r in sub])),
                "mean_online_wall_sec": float(np.mean([r["online_wall_sec"] for r in sub])),
                "sure_telemetry": {
                    "init_lambda_obj_mean": float(np.mean(sure_init_obj)) if sure_init_obj else None,
                    "init_lambda_con_mean": float(np.mean(sure_init_con)) if sure_init_con else None,
                    "final_lambda_obj_mean": float(np.mean(sure_l_obj)) if sure_l_obj else None,
                    "final_lambda_con_mean": float(np.mean(sure_l_con)) if sure_l_con else None,
                    "source_released_rate": float(np.mean(rel_flags)) if rel_flags else None,
                    "final_beta_con_mean": float(np.mean(betas_con)) if betas_con else None,
                },
                "eckg_telemetry": eckg_agg,
            }

        paired_vs_ref: dict[str, Any] = {}
        if ref_arm in methods:
            p_two_sq: dict[str, float] = {}
            p_one_sq: dict[str, float] = {}
            p_two_cp: dict[str, float] = {}
            # Anchor SD across tasks for cross-family standardized effect pooling
            anc_vals = np.array([anc_by_task[k] for k in sorted(anc_by_task)], dtype=float)
            fam_scale = max(float(np.std(anc_vals, ddof=1)), 1e-6)

            for other in methods:
                if other == ref_arm:
                    continue
                t_keys = sorted(set(task_sq[ref_arm]) & set(task_sq[other]))
                d_sq = np.array([task_sq[other][k] - task_sq[ref_arm][k] for k in t_keys], dtype=float)
                d_fr = np.array([task_fr[other][k] - task_fr[ref_arm][k] for k in t_keys], dtype=float)
                d_cp = np.array([task_comp[other][k] - task_comp[ref_arm][k] for k in t_keys], dtype=float)
                d_ov = np.array([task_op_viol[other][k] - task_op_viol[ref_arm][k] for k in t_keys], dtype=float)

                st_sq = _paired_stats(d_sq)
                st_fr = _paired_stats(d_fr)
                st_cp = _paired_stats(d_cp)
                st_ov = _paired_stats(d_ov)
                p_two_sq[other] = st_sq["paired_t_two_sided_p"]
                p_one_sq[other] = st_sq["wilcoxon_one_sided_p"]
                p_two_cp[other] = st_cp["paired_t_two_sided_p"]

                pooled_z_diffs.setdefault(other, []).extend((d_sq / fam_scale).tolist())
                paired_vs_ref[other] = {
                    "status_quo_loss": st_sq,
                    "fixed_reject_loss_10pct": st_fr,
                    "compliance_penalized_loss": st_cp,
                    "operating_viol_reduction": st_ov,
                }

            for k, hp in _holm_adjust(p_two_sq).items():
                paired_vs_ref[k]["status_quo_loss"]["holm_two_sided_t_p"] = hp
            for k, hp in _holm_adjust(p_one_sq).items():
                paired_vs_ref[k]["status_quo_loss"]["holm_one_sided_wilcoxon_p"] = hp
            for k, hp in _holm_adjust(p_two_cp).items():
                paired_vs_ref[k]["compliance_penalized_loss"]["holm_two_sided_t_p"] = hp

        out_families[fam] = {
            "anchor_task_mean_obj": float(np.mean(list(anc_by_task.values()))),
            "anchor_task_mean_viol_prob": float(np.mean(list(anc_viol_by_task.values()))),
            "anchor_safety_counts": {
                "safe": int(anc_cls_counts["safe"]),
                "borderline": int(anc_cls_counts["borderline"]),
                "unsafe": int(anc_cls_counts["unsafe"]),
            },
            "methods": m_summary,
            "paired_vs_ref": paired_vs_ref,
        }

    pooled_72: dict[str, Any] = {}
    p_pool_two: dict[str, float] = {}
    for other, z_list in sorted(pooled_z_diffs.items()):
        z_arr = np.asarray(z_list, dtype=float)
        st_z = _paired_stats(z_arr)
        p_pool_two[other] = st_z["paired_t_two_sided_p"]
        pooled_72[other] = st_z
    for k, hp in _holm_adjust(p_pool_two).items():
        pooled_72[k]["holm_two_sided_t_p"] = hp

    return {
        "campaign_name": compact["campaign_name"],
        "reference_arm": ref_arm,
        "execution_code_sha256": compact["execution_code_sha256"],
        "manifest_sha256": compact["manifest_sha256"],
        "per_run_ledger_csv_sha256": compact["per_run_ledger_csv_sha256"],
        "per_slot_certificates_csv_sha256": compact["per_slot_certificates_csv_sha256"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "total_cells": len(rows),
        "families": out_families,
        "pooled_72_task_standardized_vs_ref": pooled_72,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-dir",
        type=Path,
        default=SC_ROOT / "paper_artifacts/v2_or_eckg_sure_confirm_v1",
    )
    parser.add_argument("--ref-arm", default="GP-FTA-ECKG")
    args = parser.parse_args()

    out = build_readout(args.campaign_dir.resolve(), ref_arm=args.ref_arm)
    out_path = args.campaign_dir.resolve() / "prospective_eckg_readout.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "readout": str(out_path), "total_cells": out["total_cells"]}))


if __name__ == "__main__":
    main()
