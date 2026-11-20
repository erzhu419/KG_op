#!/usr/bin/env python3
"""Cluster diagnostic: Meta-prior cross-policy propagation, variance buffer calibration,
psi-fiber near-collision sensitivity, and paired shift_scale=1.0 vs 1.3 anchor audit.

Lives in `scripts/` so `SC-OLH-KG/v2_or` runtime fingerprint remains frozen.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import shlex
import socket
import subprocess
import sys
import time
from typing import Any

import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
SC_ROOT = REPO_ROOT / "SC-OLH-KG"
if str(SC_ROOT) not in sys.path:
    sys.path.insert(0, str(SC_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from scheduler_node_policy import DEFAULT_CPU_NODES, parse_cpu_nodes
from v2_or.acquisition.verification_kg import (
    build_cold_dct_initial_points,
    generate_dynamic_candidate_pool,
)
from v2_or.contracts import SufficientStatsRecord
from v2_or.dispatch.campaign_runner import build_family_tasks, compute_v2_or_code_sha256
from v2_or.models import TemperedHierarchicalHVDModel
from v2_or.verification.gated_verifier_v2 import (
    exact_binomial_critical_k,
    posterior_binomial_gate_power,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _make_scrambled_prior(prior: Any, seed: int) -> Any:
    rng = np.random.default_rng(seed)
    p_scr = copy.deepcopy(prior)
    k_act = p_scr.mu_0_obj_.size - 1
    if k_act > 1:
        perm_f = rng.permutation(k_act)
        signs_f = rng.choice([-1.0, 1.0], size=k_act)
        perm_g = rng.permutation(k_act)
        signs_g = rng.choice([-1.0, 1.0], size=k_act)
        p_scr.mu_0_obj_[1:] = signs_f * p_scr.mu_0_obj_[1:][perm_f]
        p_scr.mu_0_con_[1:] = signs_g * p_scr.mu_0_con_[1:][perm_g]
    p_scr.cov_0_obj_ = np.diag(np.diag(p_scr.cov_0_obj_))
    p_scr.cov_0_con_ = np.diag(np.diag(p_scr.cov_0_con_))
    return p_scr


def _make_diag_prior(prior: Any) -> Any:
    p_diag = copy.deepcopy(prior)
    p_diag.cov_0_obj_ = np.diag(np.diag(p_diag.cov_0_obj_))
    p_diag.cov_0_con_ = np.diag(np.diag(p_diag.cov_0_con_))
    return p_diag


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    val = float(spearmanr(a, b).statistic)
    return 0.0 if np.isnan(val) else val


def _run_one_task(manifest: dict[str, Any], out_dir: Path, jb: dict[str, Any]) -> dict[str, Any]:
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    ipath = out_dir / "items" / f"{jb['job_id']}.json"
    if ipath.is_file():
        return {"job_id": jb["job_id"], "status": "cached"}

    t0 = time.perf_counter()
    fam_name = jb["family"]
    t_idx = int(jb["target_task_idx"])
    fam_spec = next(f for f in manifest["families"] if f["name"] == fam_name)
    tasks = build_family_tasks(fam_spec, "target")
    prob = tasks[t_idx]

    src_campaign = SC_ROOT / "paper_artifacts" / manifest["reuse_source_campaign"]
    src_fam = fam_spec.get("reuse_archive_from_family", fam_name)
    with (src_campaign / "source_bundles" / src_fam / "prior_default.pkl").open("rb") as fh:
        prior = pickle.load(fh)

    base_seed = 7700000 + 1009 * t_idx + sum(ord(c) for c in fam_name)
    rng_init = np.random.default_rng(base_seed)
    rng_test = np.random.default_rng(base_seed + 111)

    n0 = 8
    b_reps = 2
    init_pts = build_cold_dct_initial_points(prob, n0, rng_init)
    init_recs: list[SufficientStatsRecord] = []
    for x0 in init_pts:
        exp0 = prob.nominal_exposure(x0)
        samp0 = prob.simulate_replications(x0, b_reps, rng_init, split="search")
        init_recs.append(SufficientStatsRecord.from_samples(x0, samp0, exp0))

    # Build 6 models fitted to the EXACT SAME 8 cold-DCT target observations
    models: dict[str, TemperedHierarchicalHVDModel] = {
        "TargetGP-RawPsi": TemperedHierarchicalHVDModel(
            None, mode="target_only", use_source_basis=False, use_source_coef_prior=False, use_source_hvd_prior=False
        ),
        "SourceBasis-WeakIso": TemperedHierarchicalHVDModel(
            prior, mode="source_basis_weak_prior", use_source_basis=True, use_source_coef_prior=False,
            use_source_hvd_prior=False, use_empirical_r_perp=False
        ),
        "MetaPrior-Scrambled": TemperedHierarchicalHVDModel(
            _make_scrambled_prior(prior, base_seed + 333), mode="tempered_sure_release",
            use_source_basis=True, use_source_coef_prior=True, use_source_hvd_prior=False
        ),
        "MetaPrior-DiagSURE": TemperedHierarchicalHVDModel(
            _make_diag_prior(prior), mode="tempered_sure_release",
            use_source_basis=True, use_source_coef_prior=True, use_source_hvd_prior=False
        ),
        "MetaPrior-NoSURE": TemperedHierarchicalHVDModel(
            prior, mode="untempered_fixed_prior", variance_estimator="within_ssq",
            use_source_basis=True, use_source_coef_prior=True, use_source_hvd_prior=False
        ),
        "MetaPrior-FullSURE": TemperedHierarchicalHVDModel(
            prior, mode="tempered_sure_release",
            use_source_basis=True, use_source_coef_prior=True, use_source_hvd_prior=False
        ),
    }
    for mdl in models.values():
        for rec in init_recs:
            mdl.observe(rec)

    # Sample 32 held-out unobserved DCT operational policies
    cands_pool, exps_pool = generate_dynamic_candidate_pool(
        prob, models["TargetGP-RawPsi"], None, rng_test, pool_size=96, enable_source_templates=False
    )
    obs_keys = {models["TargetGP-RawPsi"]._policy_key(r.x_norm) for r in init_recs}
    held_cands: list[np.ndarray] = []
    held_exps: list[Any] = []
    for c_x, e_x in zip(cands_pool, exps_pool):
        if models["TargetGP-RawPsi"]._policy_key(c_x) not in obs_keys:
            held_cands.append(c_x)
            held_exps.append(e_x)
        if len(held_cands) >= 32:
            break

    n_mc_truth = 400
    true_f = np.empty(len(held_cands), dtype=float)
    true_g = np.empty(len(held_cands), dtype=float)
    true_v_g = np.empty(len(held_cands), dtype=float)
    true_p_viol = np.empty(len(held_cands), dtype=float)
    for idx_h, x_h in enumerate(held_cands):
        aud = prob.audit_ground_truth(x_h, n_eval=n_mc_truth, seed=base_seed + 5000 + idx_h, split="audit")
        true_f[idx_h] = float(aud["mc_objective_mean"])
        true_g[idx_h] = float(aud["mc_constraint_mean"])
        true_v_g[idx_h] = float(aud["mc_constraint_std"]) ** 2
        true_p_viol[idx_h] = float(aud["mc_violation_prob"])

    true_feas = (true_p_viol <= prob.alpha).astype(float)
    f_sd = max(float(np.std(true_f, ddof=1)), 1e-8)
    g_sd = max(float(np.std(true_g, ddof=1)), 1e-8)

    model_metrics: dict[str, Any] = {}
    k0_crit = max(exact_binomial_critical_k(112, float(prob.alpha), 0.025), 0)
    for m_name, mdl in models.items():
        pr = mdl.predict_joint(held_cands, held_exps)
        # Demeaned shape RMSE + raw RMSE + Spearman rank correlation
        df_c = (pr.mean_obj - np.mean(pr.mean_obj)) - (true_f - np.mean(true_f))
        dg_c = (pr.mean_con - np.mean(pr.mean_con)) - (true_g - np.mean(true_g))
        shape_nrmse_f = float(np.sqrt(np.mean(df_c ** 2)) / f_sd)
        shape_nrmse_g = float(np.sqrt(np.mean(dg_c ** 2)) / g_sd)
        raw_nrmse_f = float(np.sqrt(np.mean((pr.mean_obj - true_f) ** 2)) / f_sd)
        raw_nrmse_g = float(np.sqrt(np.mean((pr.mean_con - true_g) ** 2)) / g_sd)
        rho_f = _safe_spearman(pr.mean_obj, true_f)
        rho_g = _safe_spearman(pr.mean_con, true_g)

        p_gate0 = posterior_binomial_gate_power(
            pr.mean_con, pr.std_con, pr.ucb_aleatoric_var_con,
            n_samples=112, alpha=float(prob.alpha), delta_slot=0.025, r_perp_con=pr.r_perp_con,
        )
        brier_feas = float(np.mean((p_gate0 - true_feas) ** 2))
        buf_cov = float(np.mean(pr.ucb_aleatoric_var_con >= true_v_g))
        raw_var_cov = float(np.mean(pr.aleatoric_var_con >= true_v_g))

        model_metrics[m_name] = {
            "shape_nrmse_obj": shape_nrmse_f,
            "shape_nrmse_con": shape_nrmse_g,
            "raw_nrmse_obj": raw_nrmse_f,
            "raw_nrmse_con": raw_nrmse_g,
            "spearman_obj": rho_f,
            "spearman_con": rho_g,
            "feas_brier": brier_feas,
            "var_buffer_coverage": buf_cov,
            "var_raw_coverage": raw_var_cov,
            "sure_lambda_obj": float(mdl.sure_lambda_obj_),
            "sure_lambda_con": float(mdl.sure_lambda_con_),
        }

    # 2. Psi-fiber near-collision sensitivity audit (12 pairs)
    psi_held = np.array([e.psi for e in held_exps], dtype=float)
    psi_scale = np.maximum(np.std(psi_held, axis=0, ddof=1), 1e-4)
    d_dim = int(prob.d)
    alt_sign = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(d_dim)], dtype=float)

    fiber_x_rms: list[float] = []
    fiber_psi_rel: list[float] = []
    fiber_df: list[float] = []
    fiber_dg: list[float] = []
    for k_idx in range(min(12, len(held_cands))):
        x_base = held_cands[k_idx]
        psi_base = held_exps[k_idx].psi
        best_pert = None
        best_psi_d = float("inf")
        for _ in range(24):
            env_w = rng_test.normal(0.0, 1.0, size=d_dim)
            hf = alt_sign * (env_w - np.mean(env_w))
            hf = hf / max(float(np.sqrt(np.mean(hf ** 2))), 1e-8) * 0.08
            x_cand = np.clip(x_base + hf, 0.0, 1.0)
            x_dist = float(np.sqrt(np.mean((x_cand - x_base) ** 2)))
            if x_dist < 0.04:
                continue
            exp_c = prob.nominal_exposure(x_cand)
            p_dist = float(np.sqrt(np.mean(((exp_c.psi - psi_base) / psi_scale) ** 2)))
            if p_dist < best_psi_d:
                best_psi_d = p_dist
                best_pert = (x_cand, x_dist)
        if best_pert is not None:
            x_alt, x_d = best_pert
            aud_b = prob.audit_ground_truth(x_base, n_eval=300, seed=base_seed + 8000 + k_idx, split="audit")
            aud_a = prob.audit_ground_truth(x_alt, n_eval=300, seed=base_seed + 8000 + k_idx, split="audit")
            fiber_x_rms.append(x_d)
            fiber_psi_rel.append(best_psi_d)
            fiber_df.append(abs(float(aud_a["mc_objective_mean"]) - float(aud_b["mc_objective_mean"])))
            fiber_dg.append(abs(float(aud_a["mc_constraint_mean"]) - float(aud_b["mc_constraint_mean"])))

    fiber_summary = {
        "n_pairs": len(fiber_df),
        "mean_raw_policy_rms_dist": float(np.mean(fiber_x_rms)) if fiber_x_rms else 0.0,
        "mean_rel_psi_rms_dist": float(np.mean(fiber_psi_rel)) if fiber_psi_rel else 0.0,
        "fiber_obj_rms_over_sd": float(np.sqrt(np.mean(np.square(fiber_df))) / f_sd) if fiber_df else 0.0,
        "fiber_con_rms_over_sd": float(np.sqrt(np.mean(np.square(fiber_dg))) / g_sd) if fiber_dg else 0.0,
    }

    # 3. Paired shift_scale = 1.0 vs 1.3 anchor compliance check (synthetic families)
    anchor_shift_pair = None
    if fam_spec["kind"] in {"inventory", "queueing", "traffic_des"}:
        fam_s10 = copy.deepcopy(fam_spec)
        fam_s10["target_config"]["shift_scale"] = 1.0
        prob_s10 = build_family_tasks(fam_s10, "target")[t_idx]
        anc_13 = prob.audit_ground_truth(prob.domain_heuristic_anchor(), n_eval=2000, seed=base_seed + 9500, split="audit")
        anc_10 = prob_s10.audit_ground_truth(prob_s10.domain_heuristic_anchor(), n_eval=2000, seed=base_seed + 9500, split="audit")
        anchor_shift_pair = {
            "viol_prob_shift_1_0": float(anc_10["mc_violation_prob"]),
            "safety_class_shift_1_0": anc_10["safety_audit_class"],
            "viol_prob_shift_1_3": float(anc_13["mc_violation_prob"]),
            "safety_class_shift_1_3": anc_13["safety_audit_class"],
            "obj_shift_1_0": float(anc_10["mc_objective_mean"]),
            "obj_shift_1_3": float(anc_13["mc_objective_mean"]),
        }

    payload = {
        "job_id": jb["job_id"],
        "family": fam_name,
        "target_task_idx": t_idx,
        "task_id": prob.task_id,
        "policy_dim": int(prob.d),
        "wall_sec": float(time.perf_counter() - t0),
        "models": model_metrics,
        "psi_fiber_audit": fiber_summary,
        "anchor_shift_pair": anchor_shift_pair,
    }
    _write_json(ipath, payload)
    return {"job_id": jb["job_id"], "status": "ok"}


def describe(manifest_path: Path, out_dir: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs = []
    for fam in manifest["families"]:
        t_conf = fam["target_config"]
        n_t = len(t_conf["markets"]) if "markets" in t_conf else int(t_conf["n_tasks"])
        for ti in range(n_t):
            jobs.append({
                "job_id": f"mpprop_{fam['name']}_t{ti:02d}",
                "family": fam["name"],
                "target_task_idx": ti,
            })
    desc = {
        "diagnostic_name": "v2_or_metaprior_propagation_audit_v1",
        "manifest_relpath": str(manifest_path.resolve().relative_to(SC_ROOT)),
        "execution_code_sha256": compute_v2_or_code_sha256(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_jobs": len(jobs),
        "jobs": jobs,
    }
    _write_json(out_dir / "jobs.json", desc)
    return desc


def run_worker(manifest_path: Path, out_dir: Path, start: int, end: int, workers: int) -> None:
    node = socket.gethostname().split(".")[0]
    portable_ok = os.environ.get("GPFTA_PORTABLE_REVIEW") == "1" or (REPO_ROOT / ".gpfta_portable_review").is_file()
    if not portable_ok and node not in DEFAULT_CPU_NODES:
        raise RuntimeError(f"Diagnostic worker must run on node001-node006, got {node!r}")
    desc = json.loads((out_dir / "jobs.json").read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    slice_jobs = desc["jobs"][start:end]
    n_proc = max(1, min(int(workers), len(slice_jobs)))
    failures = []
    if n_proc == 1:
        for jb in slice_jobs:
            res = _run_one_task(manifest, out_dir, jb)
            if res["status"] not in {"ok", "cached"}:
                failures.append(res)
    else:
        with ProcessPoolExecutor(max_workers=n_proc) as pool:
            futs = [pool.submit(_run_one_task, manifest, out_dir, jb) for jb in slice_jobs]
            for f in as_completed(futs):
                res = f.result()
                if res["status"] not in {"ok", "cached"}:
                    failures.append(res)
    if failures:
        raise RuntimeError(f"{len(failures)} propagation audit jobs failed: {failures[:2]}")


def summarize(out_dir: Path, summary_path: Path) -> dict[str, Any]:
    desc = json.loads((out_dir / "jobs.json").read_text(encoding="utf-8"))
    items = [json.loads((out_dir / "items" / f"{jb['job_id']}.json").read_text(encoding="utf-8")) for jb in desc["jobs"]]
    fams = sorted({it["family"] for it in items})
    fam_out: dict[str, Any] = {}
    for fam in fams:
        sub = [it for it in items if it["family"] == fam]
        m_names = list(sub[0]["models"].keys())
        m_agg = {}
        for mn in m_names:
            vals = [it["models"][mn] for it in sub]
            m_agg[mn] = {
                k: float(np.mean([v[k] for v in vals]))
                for k in vals[0].keys()
            }
            for k in ("spearman_obj", "spearman_con", "shape_nrmse_obj", "shape_nrmse_con"):
                m_agg[mn][f"{k}_se"] = float(np.std([v[k] for v in vals], ddof=1) / np.sqrt(len(vals)))
        fib = [it["psi_fiber_audit"] for it in sub]
        anc_pairs = [it["anchor_shift_pair"] for it in sub if it["anchor_shift_pair"] is not None]
        anc_summary = None
        if anc_pairs:
            anc_summary = {
                "n_tasks": len(anc_pairs),
                "mean_viol_shift_1_0": float(np.mean([a["viol_prob_shift_1_0"] for a in anc_pairs])),
                "mean_viol_shift_1_3": float(np.mean([a["viol_prob_shift_1_3"] for a in anc_pairs])),
                "unsafe_count_shift_1_0": int(sum(1 for a in anc_pairs if a["safety_class_shift_1_0"] == "confidently_unsafe")),
                "unsafe_count_shift_1_3": int(sum(1 for a in anc_pairs if a["safety_class_shift_1_3"] == "confidently_unsafe")),
                "safe_count_shift_1_0": int(sum(1 for a in anc_pairs if a["safety_class_shift_1_0"] == "confidently_safe")),
                "safe_count_shift_1_3": int(sum(1 for a in anc_pairs if a["safety_class_shift_1_3"] == "confidently_safe")),
            }
        fam_out[fam] = {
            "n_tasks": len(sub),
            "policy_dim_max": int(max(it["policy_dim"] for it in sub)),
            "models": m_agg,
            "psi_fiber_audit": {
                "mean_raw_policy_rms_dist": float(np.mean([x["mean_raw_policy_rms_dist"] for x in fib])),
                "mean_rel_psi_rms_dist": float(np.mean([x["mean_rel_psi_rms_dist"] for x in fib])),
                "fiber_obj_rms_over_sd": float(np.mean([x["fiber_obj_rms_over_sd"] for x in fib])),
                "fiber_con_rms_over_sd": float(np.mean([x["fiber_con_rms_over_sd"] for x in fib])),
            },
            "anchor_shift_1_0_vs_1_3": anc_summary,
        }
    readout = {
        "diagnostic_name": desc["diagnostic_name"],
        "execution_code_sha256": desc["execution_code_sha256"],
        "total_tasks_audited": len(items),
        "families": fam_out,
    }
    _write_json(summary_path, readout)
    return readout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("describe", "worker", "summarize", "submit"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", type=Path, default=SC_ROOT / "v2_or/manifests/v2_or_metaprior_opsd_confirm_v1.json")
        p.add_argument("--out-dir", type=Path, default=SC_ROOT / "paper_artifacts/v2_or_metaprior_opsd_confirm_v1/propagation_audit")
        if name == "worker":
            p.add_argument("--start", type=int, required=True)
            p.add_argument("--end", type=int, required=True)
            p.add_argument("--workers", type=int, default=1)
        elif name == "summarize":
            p.add_argument("--summary-path", type=Path, default=SC_ROOT / "paper_artifacts/v2_or_metaprior_opsd_confirm_v1/metaprior_propagation_readout.json")
        elif name == "submit":
            p.add_argument("--scheduler", type=Path, default=Path(os.environ.get("GPFTA_CLUSTER_SCHEDULER", "scheduler.py")))
            p.add_argument("--remote-cwd", default=os.environ.get("GPFTA_REMOTE_CWD", "."))
            p.add_argument("--remote-python", default=os.environ.get("GPFTA_REMOTE_PYTHON", sys.executable))
            p.add_argument("--ram-mb", type=int, default=16384)
    args = parser.parse_args()

    if args.action == "describe":
        d = describe(args.manifest, args.out_dir)
        print(json.dumps({"n_jobs": d["n_jobs"]}))
    elif args.action == "worker":
        run_worker(args.manifest, args.out_dir, args.start, args.end, args.workers)
    elif args.action == "summarize":
        r = summarize(args.out_dir, args.summary_path)
        print(json.dumps({"status": "ok", "total_tasks_audited": r["total_tasks_audited"]}))
    elif args.action == "submit":
        desc = json.loads((args.out_dir / "jobs.json").read_text(encoding="utf-8"))
        nodes = parse_cpu_nodes(",".join(DEFAULT_CPU_NODES))
        rel_man = args.manifest.resolve().relative_to(SC_ROOT)
        rel_out = args.out_dir.resolve().relative_to(SC_ROOT)
        cmd = shlex.join([
            "env", "OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
            "PYTHONPATH=.:../scripts",
            args.remote_python, "-u", "../scripts/run_metaprior_propagation_audit.py",
            "worker", "--manifest", str(rel_man), "--out-dir", str(rel_out),
            "--start", "{start}", "--end", "{end}", "--workers", "{workers}",
        ])
        s_cmd = [
            sys.executable, str(args.scheduler), "submit-cpu-batch",
            "--items", str(desc["n_jobs"]),
            "--cmd-template", cmd,
            "--cwd", args.remote_cwd,
            "--nodes", ",".join(nodes),
            "--signature", "gp_fta_v2/metaprior_prop_audit/{start}-{end}",
            "--description", "v2_or metaprior propagation audit {node} [{start},{end})",
            "--ram-mb", str(args.ram_mb),
            "--project", "gp_fta_v2",
            "--skip-launch-staging", "--env-spec", "none", "--json",
        ]
        cp = subprocess.run(s_cmd, text=True, capture_output=True)
        print(cp.stdout, end="")
        print(cp.stderr, end="", file=sys.stderr)
        raise SystemExit(cp.returncode)


if __name__ == "__main__":
    main()
