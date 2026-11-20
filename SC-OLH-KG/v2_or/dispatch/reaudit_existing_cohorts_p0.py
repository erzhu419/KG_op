#!/usr/bin/env python3
"""P0-1..P0-3 Non-Destructive Re-Aggregation & Ledger Recovery for Existing Cohorts.

Reads the preserved per-run JSON items of `v2_or_pilot_v1`, `v2_or_confirm_v1`, and
`v2_or_live_sumo_v1` on the cluster WITHOUT modifying any historical file, and builds:
1. A complete per-run CSV ledger explicitly separating:
   - `decision_state`: `"certified_deployment"` vs `"abstention"`
   - `certified_obj`: audited objective of the deployed policy when `certified == True` (`None` if abstained)
   - `uncertified_candidate_obj`: audited objective of the uncertified candidate (`None` if certified)
   - `status_quo_loss`: `certified_obj` if certified else `anchor_true_objective` (strictly roster-invariant!)
   - `fixed_reject_loss_10pct`: `certified_obj` if certified else `1.10 * anchor_true_objective` (strictly roster-invariant!)
   - Exact Clopper-Pearson 95% confidence interval `[viol_prob_ci95_lo, viol_prob_ci95_hi]` on audit violations,
     classifying certified runs into `"confidently_safe"`, `"boundary_ambiguous"`, or `"confidently_unsafe"`.
   - Explicit provenance flags (`pre_verify_primary_policy_saved = False`, `slot_certificates_saved = False`,
     `execution_code_hash = "unrecorded_pre_d1a13a40"`).
2. Roster-invariant statistical summaries at BOTH:
   - **Task level** (`M_tgt` independent tasks, averaging across replication seeds within each task first:
     task-level paired mean diff, task-level SE, task-level W/L/T, exact Wilcoxon signed-rank p across tasks), and
   - **Run level** (`M_tgt * S` fixed-benchmark seed pairs).
3. Automated consistency verification:
   - Asserts `STOP` metrics are bit-for-bit identical across `N = 15, 30, 60`;
   - Asserts removing arbitrary comparison methods changes zero values for remaining methods.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import beta as beta_dist, ttest_1samp, wilcoxon

SC_ROOT = Path(__file__).resolve().parents[2]


def clopper_pearson_ci(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    alpha_tail = (1.0 - conf) / 2.0
    lo = 0.0 if k <= 0 else float(beta_dist.ppf(alpha_tail, k, n - k + 1))
    hi = 1.0 if k >= n else float(beta_dist.ppf(1.0 - alpha_tail, k + 1, n - k))
    return lo, hi


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


def _one_sided_wilcoxon(diffs_pos_is_better: np.ndarray) -> float:
    nz = diffs_pos_is_better[np.abs(diffs_pos_is_better) > 1e-12]
    if nz.size == 0:
        return 1.0
    try:
        res = wilcoxon(nz, alternative="greater", zero_method="wilcox", method="auto")
        return float(res.pvalue)
    except Exception:
        return 1.0


def _one_sided_paired_t(diffs_pos_is_better: np.ndarray) -> float:
    if diffs_pos_is_better.size < 2 or float(np.std(diffs_pos_is_better, ddof=1)) < 1e-14:
        return 1.0
    res = ttest_1samp(diffs_pos_is_better, popmean=0.0, alternative="greater")
    return float(res.pvalue)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_ledger_rows(manifest: dict[str, Any], campaign_dir: Path) -> list[dict[str, Any]]:
    audit_items_dir = campaign_dir / "stages" / "audit_and_summarize" / "items"
    if not audit_items_dir.is_dir():
        raise FileNotFoundError(f"Missing audit items dir: {audit_items_dir}")

    fam_alpha = {f["name"]: float(f.get("alpha", 0.05)) for f in manifest["families"]}
    files = sorted(audit_items_dir.glob("*.json"))
    ledger: list[dict[str, Any]] = []

    for fpath in files:
        raw = json.loads(fpath.read_text(encoding="utf-8"))
        fam = str(raw["family"])
        alpha = fam_alpha.get(fam, 0.05)
        cert = bool(raw["certified"])
        aud = raw["audit"]
        n_aud = int(aud.get("n_audit_samples", 4000))
        v_prob = float(aud["true_violation_prob"])
        k_viol = int(round(v_prob * n_aud))
        ci_lo, ci_hi = clopper_pearson_ci(k_viol, n_aud, conf=0.95)

        if not cert:
            safety_audit_class = "abstained_not_deployed"
        elif ci_hi <= alpha:
            safety_audit_class = "confidently_safe"
        elif ci_lo > alpha:
            safety_audit_class = "confidently_unsafe"
        else:
            safety_audit_class = "boundary_ambiguous"

        t_obj = float(aud["true_objective"])
        anc_obj = float(raw["anchor_true_objective"])
        anc_feas = bool(raw["anchor_is_truly_feasible"])

        # Roster-invariant decision losses:
        # 1) status_quo_loss: deploy certified policy if certified, else keep domain operational anchor
        # 2) fixed_reject_loss_10pct: pre-specified fixed 10% abstinence penalty above domain operational anchor
        status_quo_loss = t_obj if cert else anc_obj
        fixed_reject_loss = t_obj if cert else (1.10 * anc_obj)

        ledger.append({
            "campaign": manifest["campaign_name"],
            "job_id": raw["job_id"],
            "family": fam,
            "target_task_idx": int(raw["target_task_idx"]),
            "task_id": raw["task_id"],
            "policy_dim": int(raw["policy_dim"]),
            "rep_seed": int(raw["rep_seed"]),
            "seed": int(raw["seed"]),
            "method_name": raw["method_name"],
            "n0_init_points": int(raw["n0_init_points"]),
            "n_online_steps": int(raw["n_online_steps"]),
            "search_calls_used": int(raw["search_calls_used"]),
            "verification_calls_used": int(raw["verification_calls_used"]),
            "total_target_calls": int(raw["total_target_calls"]),
            "wall_time_sec": float(raw["wall_time_sec"]),
            "decision_state": "certified_deployment" if cert else "abstention",
            "certified": cert,
            "selected_role": str(raw["selected_role"]),
            "certified_deployed_obj": t_obj if cert else None,
            "uncertified_selected_candidate_obj": None if cert else t_obj,
            "anchor_true_objective": anc_obj,
            "anchor_is_truly_feasible": anc_feas,
            "status_quo_loss": status_quo_loss,
            "fixed_reject_loss_10pct": fixed_reject_loss,
            "audit_mc_samples": n_aud,
            "audit_mc_violation_count": k_viol,
            "audit_mc_violation_prob": v_prob,
            "audit_viol_ci95_lo": ci_lo,
            "audit_viol_ci95_hi": ci_hi,
            "audit_point_exceeds_alpha": bool(cert and (v_prob > alpha)),
            "safety_audit_class": safety_audit_class,
            "pre_verify_primary_policy_saved": bool("primary_policy" in raw),
            "slot_certificates_saved": bool("slot_certificates" in raw),
            "execution_code_hash": str(raw.get("execution_code_hash", "unrecorded_pre_d1a13a40")),
        })
    return ledger


def summarize_roster_invariant(
    manifest: dict[str, Any],
    campaign_dir: Path,
    ledger: list[dict[str, Any]],
    *,
    method_filter: set[str] | None = None,
) -> dict[str, Any]:
    rows = [r for r in ledger if (method_filter is None or r["method_name"] in method_filter)]
    stop_lookup = {
        (r["family"], r["target_task_idx"], r["rep_seed"]): r
        for r in rows
        if r["method_name"] == "STOP"
    }

    src_comp_path = campaign_dir / "stages" / "source_archive" / "completion.json"
    prior_diag = json.loads(src_comp_path.read_text(encoding="utf-8")).get("prior_diagnostics", {}) if src_comp_path.is_file() else {}

    families_out: dict[str, Any] = {}
    for fam in manifest["families"]:
        fam_name = fam["name"]
        if not fam.get("online_step_budgets"):
            continue
        src_fam = fam.get("reuse_archive_from_family", fam_name)
        src_calls = int(prior_diag.get(src_fam, {}).get("source_simulation_calls", 0))
        fam_entry: dict[str, Any] = {"source_simulation_calls": src_calls, "budgets": {}}

        for b_steps in fam["online_step_budgets"]:
            b_rows = [r for r in rows if r["family"] == fam_name and int(r["n_online_steps"]) == int(b_steps)]
            if not any(r["method_name"] == "STOP" for r in b_rows):
                for (fn, _tidx, _rseed), s_row in stop_lookup.items():
                    if fn == fam_name:
                        b_rows.append(s_row)

            methods = sorted({r["method_name"] for r in b_rows})
            method_stats: dict[str, Any] = {}
            run_sq_vecs: dict[str, dict[tuple[int, int], float]] = {}
            task_sq_vecs: dict[str, dict[int, float]] = {}

            for m_name in methods:
                m_sub = [r for r in b_rows if r["method_name"] == m_name]
                cert_objs = [float(r["certified_deployed_obj"]) for r in m_sub if r["certified"]]
                sq_losses = [float(r["status_quo_loss"]) for r in m_sub]
                fr_losses = [float(r["fixed_reject_loss_10pct"]) for r in m_sub]
                cert_flags = [1.0 if r["certified"] else 0.0 for r in m_sub]

                # Task-level aggregation (averaging over seeds within each independent task first)
                by_task_sq: dict[int, list[float]] = {}
                by_task_fr: dict[int, list[float]] = {}
                by_task_cert: dict[int, list[float]] = {}
                for r in m_sub:
                    ti = int(r["target_task_idx"])
                    by_task_sq.setdefault(ti, []).append(float(r["status_quo_loss"]))
                    by_task_fr.setdefault(ti, []).append(float(r["fixed_reject_loss_10pct"]))
                    by_task_cert.setdefault(ti, []).append(1.0 if r["certified"] else 0.0)

                task_means_sq = {ti: float(np.mean(vals)) for ti, vals in sorted(by_task_sq.items())}
                task_means_fr = [float(np.mean(vals)) for _, vals in sorted(by_task_fr.items())]
                task_means_cert = [float(np.mean(vals)) for _, vals in sorted(by_task_cert.items())]
                t_sq_arr = np.asarray(list(task_means_sq.values()), dtype=float)

                role_counts = dict(Counter(r["selected_role"] for r in m_sub))
                safety_counts = dict(Counter(r["safety_audit_class"] for r in m_sub))

                run_sq_vecs[m_name] = {(int(r["target_task_idx"]), int(r["rep_seed"])): float(r["status_quo_loss"]) for r in m_sub}
                task_sq_vecs[m_name] = task_means_sq

                method_stats[m_name] = {
                    "n_runs": len(m_sub),
                    "n_independent_tasks": len(task_means_sq),
                    "certified_rate": float(np.mean(cert_flags)),
                    "abstention_rate": float(1.0 - np.mean(cert_flags)),
                    "n_certified_runs": len(cert_objs),
                    "conditional_certified_obj_mean": float(np.mean(cert_objs)) if cert_objs else None,
                    "conditional_certified_obj_se": (
                        float(np.std(cert_objs, ddof=1) / np.sqrt(len(cert_objs))) if len(cert_objs) > 1 else (0.0 if len(cert_objs) == 1 else None)
                    ),
                    # Strictly roster-invariant composite decision losses:
                    "status_quo_loss_run_mean": float(np.mean(sq_losses)),
                    "status_quo_loss_run_se": float(np.std(sq_losses, ddof=1) / np.sqrt(len(sq_losses))) if len(sq_losses) > 1 else 0.0,
                    "status_quo_loss_task_mean": float(np.mean(t_sq_arr)),
                    "status_quo_loss_task_se": float(np.std(t_sq_arr, ddof=1) / np.sqrt(len(t_sq_arr))) if len(t_sq_arr) > 1 else 0.0,
                    "fixed_reject_loss_10pct_task_mean": float(np.mean(task_means_fr)),
                    "task_certified_rate_mean": float(np.mean(task_means_cert)),
                    "audit_point_fc_count": int(sum(1 for r in m_sub if r["audit_point_exceeds_alpha"])),
                    "safety_audit_breakdown": safety_counts,
                    "selected_role_breakdown": role_counts,
                    "mean_search_calls": float(np.mean([r["search_calls_used"] for r in m_sub])),
                    "mean_verification_calls": float(np.mean([r["verification_calls_used"] for r in m_sub])),
                    "mean_total_target_calls": float(np.mean([r["total_target_calls"] for r in m_sub])),
                    "mean_wall_sec": float(np.mean([r["wall_time_sec"] for r in m_sub])),
                }

            # Compute both TASK-LEVEL and RUN-LEVEL paired comparisons against TLS-OLH-KG
            paired_task_vs_tls: dict[str, Any] = {}
            paired_run_vs_tls: dict[str, Any] = {}
            if "TLS-OLH-KG" in methods:
                tls_t = task_sq_vecs["TLS-OLH-KG"]
                tls_r = run_sq_vecs["TLS-OLH-KG"]
                raw_p_task: dict[str, float] = {}
                raw_p_run: dict[str, float] = {}

                for m_name in methods:
                    if m_name == "TLS-OLH-KG":
                        continue
                    # Task-level (M_tgt independent tasks)
                    t_keys = sorted(set(tls_t.keys()) & set(task_sq_vecs[m_name].keys()))
                    d_task = np.array([task_sq_vecs[m_name][k] - tls_t[k] for k in t_keys], dtype=float)
                    b_task = np.array([task_sq_vecs[m_name][k] for k in t_keys], dtype=float)
                    pw_t = _one_sided_wilcoxon(d_task)
                    pt_t = _one_sided_paired_t(d_task)
                    raw_p_task[m_name] = pw_t
                    paired_task_vs_tls[m_name] = {
                        "n_tasks": len(t_keys),
                        "mean_diff_status_quo_loss": float(np.mean(d_task)),
                        "se_diff_status_quo_loss": float(np.std(d_task, ddof=1) / np.sqrt(len(d_task))) if len(d_task) > 1 else 0.0,
                        "mean_rel_improvement_pct": float(100.0 * np.mean(d_task / np.maximum(np.abs(b_task), 1e-8))),
                        "task_wins_losses_ties": [
                            int(np.sum(d_task > 1e-9)),
                            int(np.sum(d_task < -1e-9)),
                            int(np.sum(np.abs(d_task) <= 1e-9)),
                        ],
                        "wilcoxon_one_sided_p": pw_t,
                        "paired_t_one_sided_p": pt_t,
                    }

                    # Run-level (fixed-benchmark M_tgt * S pairs)
                    r_keys = sorted(set(tls_r.keys()) & set(run_sq_vecs[m_name].keys()))
                    d_run = np.array([run_sq_vecs[m_name][k] - tls_r[k] for k in r_keys], dtype=float)
                    b_run = np.array([run_sq_vecs[m_name][k] for k in r_keys], dtype=float)
                    pw_r = _one_sided_wilcoxon(d_run)
                    raw_p_run[m_name] = pw_r
                    paired_run_vs_tls[m_name] = {
                        "n_pairs": len(r_keys),
                        "mean_diff_status_quo_loss": float(np.mean(d_run)),
                        "mean_rel_improvement_pct": float(100.0 * np.mean(d_run / np.maximum(np.abs(b_run), 1e-8))),
                        "run_wins_losses_ties": [
                            int(np.sum(d_run > 1e-9)),
                            int(np.sum(d_run < -1e-9)),
                            int(np.sum(np.abs(d_run) <= 1e-9)),
                        ],
                        "wilcoxon_one_sided_p": pw_r,
                    }

                holm_t = _holm_adjust(raw_p_task)
                holm_r = _holm_adjust(raw_p_run)
                for m_name, hp in holm_t.items():
                    paired_task_vs_tls[m_name]["holm_adjusted_p"] = hp
                for m_name, hp in holm_r.items():
                    paired_run_vs_tls[m_name]["holm_adjusted_p"] = hp

            fam_entry["budgets"][str(b_steps)] = {
                "methods": method_stats,
                "paired_task_level_vs_tls": paired_task_vs_tls,
                "paired_run_level_vs_tls": paired_run_vs_tls,
            }
        families_out[fam_name] = fam_entry

    return {
        "campaign_name": manifest["campaign_name"],
        "reaudit_version": "p0_roster_invariant_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "historical_execution_code_hash": "unrecorded_pre_d1a13a40",
        "total_unique_runs": len(rows),
        "total_certified_deployments": int(sum(1 for r in rows if r["certified"])),
        "total_abstentions": int(sum(1 for r in rows if not r["certified"])),
        "safety_audit_summary": dict(Counter(r["safety_audit_class"] for r in rows)),
        "families": families_out,
    }


def verify_p0_invariants(manifest: dict[str, Any], campaign_dir: Path, ledger: list[dict[str, Any]], full_summary: dict[str, Any]) -> dict[str, bool]:
    # 1. STOP metrics must be bit-for-bit identical across all online_step_budgets within each family
    for fam_name, fam_data in full_summary["families"].items():
        b_keys = sorted(fam_data["budgets"].keys(), key=lambda x: int(x))
        if len(b_keys) >= 2 and "STOP" in fam_data["budgets"][b_keys[0]]["methods"]:
            ref_stop = fam_data["budgets"][b_keys[0]]["methods"]["STOP"]
            for bk in b_keys[1:]:
                cur_stop = fam_data["budgets"][bk]["methods"]["STOP"]
                for key in (
                    "certified_rate",
                    "conditional_certified_obj_mean",
                    "status_quo_loss_run_mean",
                    "status_quo_loss_task_mean",
                    "fixed_reject_loss_10pct_task_mean",
                ):
                    if ref_stop[key] != cur_stop[key]:
                        raise AssertionError(f"STOP invariant violated on {fam_name} budget {bk} field {key}: {ref_stop[key]} != {cur_stop[key]}")

    # 2. Roster invariance: dropping SCBO / Target-CMAES / Untempered-FULL must NOT alter TLS-OLH-KG or STOP metrics!
    subset = {r["method_name"] for r in ledger} - {"SCBO", "Target-CMAES", "Untempered-FULL"}
    sub_summary = summarize_roster_invariant(manifest, campaign_dir, ledger, method_filter=subset)
    for fam_name, fam_data in sub_summary["families"].items():
        for bk, b_data in fam_data["budgets"].items():
            for m_name, m_stats in b_data["methods"].items():
                orig_stats = full_summary["families"][fam_name]["budgets"][bk]["methods"][m_name]
                for key in (
                    "certified_rate",
                    "conditional_certified_obj_mean",
                    "status_quo_loss_run_mean",
                    "status_quo_loss_task_mean",
                    "fixed_reject_loss_10pct_task_mean",
                ):
                    if m_stats[key] != orig_stats[key]:
                        raise AssertionError(f"Roster invariance violated on {fam_name} {bk} {m_name} {key}")

    return {
        "stop_identical_across_budgets": True,
        "roster_invariant_across_arm_subsets": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()

    out_root = SC_ROOT / args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    master_index: dict[str, Any] = {"generated_utc": datetime.now(timezone.utc).isoformat(), "cohorts": {}}
    for m_rel in args.manifests:
        m_path = SC_ROOT / m_rel
        manifest = json.loads(m_path.read_text(encoding="utf-8"))
        c_name = manifest["campaign_name"]
        c_dir = SC_ROOT / manifest["output_dir"]

        ledger = build_ledger_rows(manifest, c_dir)
        summary = summarize_roster_invariant(manifest, c_dir, ledger)
        checks = verify_p0_invariants(manifest, c_dir, ledger, summary)
        summary["p0_invariant_checks"] = checks
        summary["manifest_sha256"] = _sha256_file(m_path)

        cohort_out = out_root / c_name
        cohort_out.mkdir(parents=True, exist_ok=True)
        csv_path = cohort_out / "per_run_ledger.csv"
        if ledger:
            with csv_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(ledger[0].keys()))
                writer.writeheader()
                writer.writerows(ledger)

        sum_path = cohort_out / "p0_roster_invariant_summary.json"
        sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        master_index["cohorts"][c_name] = {
            "total_unique_runs": summary["total_unique_runs"],
            "total_certified_deployments": summary["total_certified_deployments"],
            "safety_audit_summary": summary["safety_audit_summary"],
            "p0_invariant_checks": checks,
            "ledger_csv_sha256": _sha256_file(csv_path),
            "summary_json_sha256": _sha256_file(sum_path),
        }

    (out_root / "p0_reaudit_index.json").write_text(json.dumps(master_index, indent=2), encoding="utf-8")
    print(json.dumps(master_index, indent=2))


if __name__ == "__main__":
    main()
