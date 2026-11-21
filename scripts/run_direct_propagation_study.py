#!/usr/bin/env python3
"""Cluster study: Direct before/after cross-policy information propagation audit.

Directly answers whether observing a pre-specified policy `x_probe` improves predictions on
a disjoint set of unobserved policies `{x'}` through the learned functional structure, while
holding initial observations, public representation `psi`, learned basis `tilde_phi`,
target-only HVD noise estimates, and candidate pools 100% identical.

Supports two strictly separated panels:
  1. `exploratory_86`: the existing 86 tasks across all four domains (Inventory 24,
     Queueing 24, OPSD 14, Traffic 24).
  2. `confirm_fresh_72`: 72 newly generated, never-before-seen target tasks (24 each in
     Inventory, Queueing, and Traffic using disjoint task seed offset +1,000,000).
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
import subprocess
import sys
import time
from typing import Any

import numpy as np
from scipy.stats import spearmanr, t as student_t, wilcoxon

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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _stabilized_cov(cov_raw: np.ndarray, m_src: int = 6) -> np.ndarray:
    p = int(cov_raw.shape[0])
    w = float(p) / float(m_src + p)
    return (1.0 - w) * cov_raw + w * np.diag(np.diag(cov_raw))


def _derangement_perm(k: int, rng: np.random.Generator) -> np.ndarray:
    if k <= 1:
        return np.arange(k, dtype=int)
    for _ in range(64):
        perm = rng.permutation(k)
        if not np.any(perm == np.arange(k)):
            return perm
    return np.roll(np.arange(k, dtype=int), 1)


def _build_prior_variants(prior: Any, seed: int) -> dict[str, Any]:
    """Construct prior copies preserving identical learned basis `tilde_phi` while varying
    only `(mu_0, Sigma_0)` under exact trace and marginal-scale controls."""
    rng = np.random.default_rng(int(seed))
    cov_f_stab = _stabilized_cov(prior.cov_0_obj_, m_src=6)
    cov_g_stab = _stabilized_cov(prior.cov_0_con_, m_src=6)
    k_act = int(prior.mu_0_obj_.size - 1)
    perm_f = _derangement_perm(k_act, rng)
    perm_g = _derangement_perm(k_act, rng)
    signs_f = rng.choice([-1.0, 1.0], size=k_act)
    signs_g = rng.choice([-1.0, 1.0], size=k_act)

    def _apply_signed_perm(mu: np.ndarray, cov: np.ndarray, perm: np.ndarray, signs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        P = np.eye(mu.size, dtype=float)
        P_act = np.zeros((k_act, k_act), dtype=float)
        for i in range(k_act):
            P_act[i, perm[i]] = signs[i]
        P[1:, 1:] = P_act
        return P @ mu, P @ cov @ P.T

    # 1. Zero-mean trace-matched kernel variants (pure kernel propagation, zero mu_0 head-start!)
    def _zero_mean_prior(cov_f_mat: np.ndarray, cov_g_mat: np.ndarray) -> Any:
        p_cp = copy.deepcopy(prior)
        p_cp.mu_0_obj_ = np.zeros_like(prior.mu_0_obj_)
        p_cp.mu_0_con_ = np.zeros_like(prior.mu_0_con_)
        p_cp.cov_0_obj_ = cov_f_mat.copy()
        p_cp.cov_0_con_ = cov_g_mat.copy()
        return p_cp

    tr_f_iso = float(np.trace(cov_f_stab[1:, 1:])) / max(k_act, 1)
    tr_g_iso = float(np.trace(cov_g_stab[1:, 1:])) / max(k_act, 1)
    iso_f = np.diag(np.r_[cov_f_stab[0, 0], np.full(k_act, tr_f_iso)])
    iso_g = np.diag(np.r_[cov_g_stab[0, 0], np.full(k_act, tr_g_iso)])

    _, perm_cov_f = _apply_signed_perm(prior.mu_0_obj_, cov_f_stab, perm_f, signs_f)
    _, perm_cov_g = _apply_signed_perm(prior.mu_0_con_, cov_g_stab, perm_g, signs_g)

    # 2. Online SURE variants (Weak, DiagSURE, FullSURE, ScalePreservingPermSURE)
    p_diag_online = copy.deepcopy(prior)
    p_diag_online.cov_0_obj_ = np.diag(np.diag(prior.cov_0_obj_))
    p_diag_online.cov_0_con_ = np.diag(np.diag(prior.cov_0_con_))

    p_perm_online = copy.deepcopy(prior)
    p_perm_online.mu_0_obj_, p_perm_online.cov_0_obj_ = _apply_signed_perm(prior.mu_0_obj_, prior.cov_0_obj_, perm_f, signs_f)
    p_perm_online.mu_0_con_, p_perm_online.cov_0_con_ = _apply_signed_perm(prior.mu_0_con_, prior.cov_0_con_, perm_g, signs_g)

    return {
        "zm_iso": _zero_mean_prior(iso_f, iso_g),
        "zm_diag": _zero_mean_prior(np.diag(np.diag(cov_f_stab)), np.diag(np.diag(cov_g_stab))),
        "zm_full": _zero_mean_prior(cov_f_stab, cov_g_stab),
        "zm_perm": _zero_mean_prior(perm_cov_f, perm_cov_g),
        "online_diag": p_diag_online,
        "online_full": prior,
        "online_perm": p_perm_online,
    }


def _select_deterministic_probes_and_heldout(
    prior: Any,
    init_recs: list[SufficientStatsRecord],
    cands_pool: list[np.ndarray],
    exps_pool: list[Any],
    *,
    n_probes: int = 4,
    n_heldout: int = 32,
) -> tuple[list[int], list[int]]:
    """Select `n_probes` probe queries and `n_heldout` evaluation policies deterministically
    using ONLY public active-basis geometry `tilde_phi(psi(x))` (zero truth or Sigma_0 peeking)."""
    obs_keys = {TemperedHierarchicalHVDModel._policy_key(r.x_norm) for r in init_recs}
    unseen_idx: list[int] = []
    seen_local: set[bytes] = set()
    for idx, x in enumerate(cands_pool):
        k = TemperperedKey = TemperedHierarchicalHVDModel._policy_key(x)
        if k not in obs_keys and k not in seen_local:
            seen_local.add(k)
            unseen_idx.append(idx)
        if len(unseen_idx) >= n_probes + n_heldout + 12:
            break

    phi_act, _, _, _, _ = prior.transform_exposures([exps_pool[i] for i in unseen_idx])
    feat = phi_act[:, 1:]  # active modes (M_u, 8)
    norms = np.maximum(np.linalg.norm(feat, axis=1, keepdims=True), 1e-8)
    unit_feat = feat / norms
    sim_sq = (unit_feat @ unit_feat.T) ** 2

    chosen_local: list[int] = []
    for _ in range(n_probes):
        best_u = -1
        best_score = -1.0
        for u in range(len(unseen_idx)):
            if u in chosen_local:
                continue
            # Maximize mean squared cosine alignment with remaining unseen pool while avoiding near-duplicate probes
            if chosen_local:
                max_corr_prev = float(np.max(sim_sq[u, chosen_local]))
                if max_corr_prev > 0.85:
                    continue
            sc = float(np.mean(sim_sq[u]))
            if sc > best_score:
                best_score = sc
                best_u = u
        if best_u < 0:
            best_u = next(u for u in range(len(unseen_idx)) if u not in chosen_local)
        chosen_local.append(best_u)

    probe_pool_idx = [unseen_idx[u] for u in chosen_local]
    held_pool_idx = [unseen_idx[u] for u in range(len(unseen_idx)) if u not in set(chosen_local)][:n_heldout]
    return probe_pool_idx, held_pool_idx


def _eval_heldout_metrics(
    pred: Any,
    true_f: np.ndarray,
    true_g: np.ndarray,
    f_sd: float,
    g_sd: float,
) -> dict[str, float]:
    """Compute demeaned shape NRMSE, demeaned Gaussian predictive log-score, and Spearman rho."""
    out: dict[str, float] = {}
    for head, mu, sd_pred, y_true, scale in (
        ("obj", pred.mean_obj, pred.std_obj, true_f, f_sd),
        ("con", pred.mean_con, pred.std_con, true_g, g_sd),
    ):
        err_c = (mu - np.mean(mu)) - (y_true - np.mean(y_true))
        nrmse = float(np.sqrt(np.mean(err_c ** 2)) / scale)
        var_s = np.maximum((sd_pred / scale) ** 2, 1e-4)
        err_s = err_c / scale
        log_score = float(np.mean(-0.5 * (err_s ** 2 / var_s + np.log(2.0 * np.pi * var_s))))
        rho = 0.0 if (np.std(mu) < 1e-12 or np.std(y_true) < 1e-12) else float(spearmanr(mu, y_true).statistic)
        if np.isnan(rho):
            rho = 0.0
        out[f"nrmse_{head}"] = nrmse
        out[f"logscore_{head}"] = log_score
        out[f"spearman_{head}"] = rho
        out[f"err_sq_vec_{head}"] = err_s ** 2  # temporary array removed after step pairing
    return out


def _make_model_suite(priors: dict[str, Any], base_recs: list[SufficientStatsRecord]) -> dict[str, TemperedHierarchicalHVDModel]:
    """Instantiate both pure-kernel fixed-scale models and online SURE models on identical `base_recs`."""
    suite: dict[str, TemperedHierarchicalHVDModel] = {}
    for label, p_key in (
        ("Kernel-IsoBasis", "zm_iso"),
        ("Kernel-DiagSource", "zm_diag"),
        ("Kernel-FullSource", "zm_full"),
        ("Kernel-PermutedScale", "zm_perm"),
    ):
        p_inst = copy.deepcopy(priors[p_key])
        lvl_f = float(np.mean([r.mean_obj for r in base_recs]))
        lvl_g = float(np.mean([r.mean_con for r in base_recs]))
        p_inst.mu_0_obj_[0] = lvl_f
        p_inst.mu_0_con_[0] = lvl_g
        m = TemperedHierarchicalHVDModel(
            p_inst,
            mode="matched_coefficients",
            coefficient_prior="historical_full",
            use_source_basis=True,
            use_source_hvd_prior=False,
            use_empirical_r_perp=False,
        )
        m.freeze_matched_scales(base_recs)
        # Keep auxiliary isotropic basis shift negligible so propagation acts strictly via the trace-matched kernel
        m.tau_shift_obj_ = 1e-6 * m.tau_shift_obj_
        m.tau_shift_con_ = 1e-6 * m.tau_shift_con_
        for r in base_recs:
            m.observe(r)
        suite[label] = m

    online_specs = [
        ("Online-SameBasisWeak", priors["online_full"], "source_basis_weak_prior", False, False),
        ("Online-DiagSURE", priors["online_diag"], "tempered_sure_release", True, True),
        ("Online-FullSURE", priors["online_full"], "tempered_sure_release", True, True),
        ("Online-PermutedScaleSURE", priors["online_perm"], "tempered_sure_release", True, True),
    ]
    for label, p_obj, mode_str, use_coef, use_rp in online_specs:
        m = TemperedHierarchicalHVDModel(
            p_obj,
            mode=mode_str,
            use_source_basis=True,
            use_source_coef_prior=use_coef,
            use_source_hvd_prior=False,
            use_empirical_r_perp=use_rp,
        )
        for r in base_recs:
            m.observe(r)
        suite[label] = m
    return suite


def _evaluate_regime_transitions(
    priors: dict[str, Any],
    base_recs: list[SufficientStatsRecord],
    probe_recs: list[SufficientStatsRecord],
    held_cands: list[np.ndarray],
    held_exps: list[Any],
    true_f: np.ndarray,
    true_g: np.ndarray,
    f_sd: float,
    g_sd: float,
) -> dict[str, Any]:
    """Measure exact before-update vs after-update prediction changes averaged across `probe_recs`."""
    base_suite = _make_model_suite(priors, base_recs)
    base_eval = {
        name: _eval_heldout_metrics(mdl.predict_joint(held_cands, held_exps), true_f, true_g, f_sd, g_sd)
        for name, mdl in base_suite.items()
    }

    acc: dict[str, dict[str, list[float]]] = {
        name: {
            "before_nrmse_obj": [], "after_nrmse_obj": [], "delta_nrmse_obj": [],
            "before_nrmse_con": [], "after_nrmse_con": [], "delta_nrmse_con": [],
            "delta_logscore_obj": [], "delta_logscore_con": [],
            "before_spearman_obj": [], "after_spearman_obj": [], "delta_spearman_obj": [],
            "before_spearman_con": [], "after_spearman_con": [], "delta_spearman_con": [],
            "pointwise_improve_rate_obj": [], "pointwise_improve_rate_con": [],
            "after_sure_lambda_obj": [], "after_sure_lambda_con": [],
        }
        for name in base_suite
    }

    for p_rec in probe_recs:
        # Clone exact pre-update models and ingest single probe observation `p_rec`
        for name, mdl_base in base_suite.items():
            mdl_post = copy.deepcopy(mdl_base)
            mdl_post.observe(p_rec)
            b_m = base_eval[name]
            a_m = _eval_heldout_metrics(mdl_post.predict_joint(held_cands, held_exps), true_f, true_g, f_sd, g_sd)
            d_acc = acc[name]
            for h in ("obj", "con"):
                d_acc[f"before_nrmse_{h}"].append(b_m[f"nrmse_{h}"])
                d_acc[f"after_nrmse_{h}"].append(a_m[f"nrmse_{h}"])
                # Positive delta_nrmse means observing x_probe REDUCED error on unobserved x'!
                d_acc[f"delta_nrmse_{h}"].append(b_m[f"nrmse_{h}"] - a_m[f"nrmse_{h}"])
                d_acc[f"delta_logscore_{h}"].append(a_m[f"logscore_{h}"] - b_m[f"logscore_{h}"])
                d_acc[f"before_spearman_{h}"].append(b_m[f"spearman_{h}"])
                d_acc[f"after_spearman_{h}"].append(a_m[f"spearman_{h}"])
                d_acc[f"delta_spearman_{h}"].append(a_m[f"spearman_{h}"] - b_m[f"spearman_{h}"])
                d_acc[f"pointwise_improve_rate_{h}"].append(
                    float(np.mean(a_m[f"err_sq_vec_{h}"] < b_m[f"err_sq_vec_{h}"]))
                )
            d_acc["after_sure_lambda_obj"].append(float(mdl_post.sure_lambda_obj_))
            d_acc["after_sure_lambda_con"].append(float(mdl_post.sure_lambda_con_))

    return {
        name: {k: float(np.mean(vals)) for k, vals in m_dict.items()}
        for name, m_dict in acc.items()
    }


def _build_panel_spec(manifest: dict[str, Any], panel: str) -> list[dict[str, Any]]:
    fams = copy.deepcopy(manifest["families"])
    if panel == "exploratory_86":
        return fams
    if panel == "confirm_fresh_72":
        out = []
        for f in fams:
            if f["kind"] not in {"inventory", "queueing", "traffic_des"}:
                continue
            f["target_config"]["seed"] = int(f["target_config"]["seed"]) + 1_000_000
            out.append(f)
        return out
    raise ValueError(f"Unknown panel {panel!r}")


def _run_one_task(manifest: dict[str, Any], panel: str, out_dir: Path, jb: dict[str, Any]) -> dict[str, Any]:
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    ipath = out_dir / "items" / f"{jb['job_id']}.json"
    if ipath.is_file():
        return {"job_id": jb["job_id"], "status": "cached"}

    t0_cpu = time.process_time()
    t0_wall = time.perf_counter()
    fam_name = jb["family"]
    t_idx = int(jb["target_task_idx"])
    fams = _build_panel_spec(manifest, panel)
    fam_spec = next(f for f in fams if f["name"] == fam_name)
    prob = build_family_tasks(fam_spec, "target")[t_idx]

    src_campaign = SC_ROOT / "paper_artifacts" / manifest["reuse_source_campaign"]
    src_fam = fam_spec.get("reuse_archive_from_family", fam_name)
    with (src_campaign / "source_bundles" / src_fam / "prior_default.pkl").open("rb") as fh:
        prior = pickle.load(fh)

    panel_seed_shift = 0 if panel == "exploratory_86" else 20_000_000
    base_seed = 8_800_000 + panel_seed_shift + 1_009 * t_idx + sum(ord(c) for c in fam_name)
    rng_init = np.random.default_rng(base_seed)
    rng_pool = np.random.default_rng(base_seed + 211)
    rng_probe = np.random.default_rng(base_seed + 433)

    init_pts = build_cold_dct_initial_points(prob, 8, rng_init)
    init_recs: list[SufficientStatsRecord] = []
    for x0 in init_pts:
        exp0 = prob.nominal_exposure(x0)
        samp0 = prob.simulate_replications(x0, 2, rng_init, split="search")
        init_recs.append(SufficientStatsRecord.from_samples(x0, samp0, exp0))

    ref_mdl = TemperedHierarchicalHVDModel(
        None, mode="target_only", use_source_basis=False, use_source_coef_prior=False, use_source_hvd_prior=False
    )
    for rec in init_recs:
        ref_mdl.observe(rec)

    cands_pool, exps_pool = generate_dynamic_candidate_pool(
        prob, ref_mdl, None, rng_pool, pool_size=128, enable_source_templates=False
    )
    probe_idx, held_idx = _select_deterministic_probes_and_heldout(
        prior, init_recs, cands_pool, exps_pool, n_probes=4, n_heldout=32
    )
    probe_recs: list[SufficientStatsRecord] = []
    for p_i in probe_idx:
        xp = cands_pool[p_i]
        ep = exps_pool[p_i]
        sp = prob.simulate_replications(xp, 4, rng_probe, split="search")
        probe_recs.append(SufficientStatsRecord.from_samples(xp, sp, ep))

    held_cands = [cands_pool[i] for i in held_idx]
    held_exps = [exps_pool[i] for i in held_idx]
    true_f = np.empty(len(held_cands), dtype=float)
    true_g = np.empty(len(held_cands), dtype=float)
    for idx_h, x_h in enumerate(held_cands):
        aud = prob.audit_ground_truth(x_h, n_eval=400, seed=base_seed + 6000 + idx_h, split="audit")
        true_f[idx_h] = float(aud["mc_objective_mean"])
        true_g[idx_h] = float(aud["mc_constraint_mean"])
    f_sd = max(float(np.std(true_f, ddof=1)), 1e-8)
    g_sd = max(float(np.std(true_g, ddof=1)), 1e-8)

    priors = _build_prior_variants(prior, seed=base_seed + 777)
    # Regime 1: Scarce observations (2 -> 3 policies: anchor + 1 cold DCT -> + probe)
    reg_2to3 = _evaluate_regime_transitions(
        priors, init_recs[:2], probe_recs, held_cands, held_exps, true_f, true_g, f_sd, g_sd
    )
    # Regime 2: Paid cold initialization (8 -> 9 policies: 8 cold DCT -> + probe)
    reg_8to9 = _evaluate_regime_transitions(
        priors, init_recs, probe_recs, held_cands, held_exps, true_f, true_g, f_sd, g_sd
    )

    payload = {
        "job_id": jb["job_id"],
        "panel": panel,
        "family": fam_name,
        "target_task_idx": t_idx,
        "task_id": prob.task_id,
        "policy_dim": int(prob.d),
        "diagnostic_search_calls": int(prob.search_calls),
        "diagnostic_verification_calls": int(prob.verification_calls),
        "diagnostic_audit_calls": int(prob.audit_calls),
        "worker_cpu_seconds": float(time.process_time() - t0_cpu),
        "worker_wall_seconds": float(time.perf_counter() - t0_wall),
        "regime_2to3": reg_2to3,
        "regime_8to9": reg_8to9,
    }
    _write_json(ipath, payload)
    return {"job_id": jb["job_id"], "status": "ok"}


def _paired_contrast(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, Any]:
    """Paired difference `lhs - rhs` across tasks (> 0 favors `lhs`)."""
    diff = lhs - rhs
    n = int(diff.size)
    m = float(np.mean(diff))
    se = float(np.std(diff, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    t_crit = float(student_t.ppf(0.975, df=max(n - 1, 1))) if n > 1 else 0.0
    if se > 1e-15 and n > 1:
        t_stat = m / se
        p_t = float(2.0 * student_t.sf(abs(t_stat), df=n - 1))
    else:
        p_t = 1.0
    nz = diff[np.abs(diff) > 1e-12]
    p_w = float(wilcoxon(nz).pvalue) if nz.size >= 4 else 1.0
    wins = int(np.sum(diff > 1e-12))
    losses = int(np.sum(diff < -1e-12))
    ties = n - wins - losses
    return {
        "n_tasks": n,
        "mean_diff": m,
        "se_diff": se,
        "ci95": [m - t_crit * se, m + t_crit * se],
        "task_wlt": [wins, losses, ties],
        "paired_t_two_sided_p": p_t,
        "wilcoxon_two_sided_p": p_w,
    }


def _holm_adjust(p_vals: list[float]) -> list[float]:
    m = len(p_vals)
    order = np.argsort(p_vals)
    adj = np.empty(m, dtype=float)
    run_max = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, float(p_vals[idx]) * float(m - rank))
        run_max = max(run_max, val)
        adj[idx] = run_max
    return adj.tolist()


def describe(manifest_path: Path, panel: str, out_dir: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fams = _build_panel_spec(manifest, panel)
    jobs = []
    for fam in fams:
        t_conf = fam["target_config"]
        n_t = len(t_conf["markets"]) if "markets" in t_conf else int(t_conf["n_tasks"])
        for ti in range(n_t):
            jobs.append({
                "job_id": f"dprop_{panel}_{fam['name']}_t{ti:02d}",
                "family": fam["name"],
                "target_task_idx": ti,
            })
    desc = {
        "diagnostic_name": f"v2_or_direct_propagation_{panel}_v1",
        "panel": panel,
        "manifest_relpath": str(manifest_path.resolve().relative_to(SC_ROOT)),
        "execution_code_sha256": compute_v2_or_code_sha256(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_jobs": len(jobs),
        "jobs": jobs,
    }
    _write_json(out_dir / "jobs.json", desc)
    return desc


def _summarize_item_group(f_items: list[dict[str, Any]]) -> dict[str, Any]:
    fam_block: dict[str, Any] = {"n_tasks": len(f_items)}
    for reg_key in ("regime_2to3", "regime_8to9"):
        m_names = list(f_items[0][reg_key].keys())
        models_agg: dict[str, Any] = {}
        for mn in m_names:
            keys = list(f_items[0][reg_key][mn].keys())
            models_agg[mn] = {}
            for k in keys:
                arr = np.array([it[reg_key][mn][k] for it in f_items], dtype=float)
                models_agg[mn][k] = float(np.mean(arr))
                models_agg[mn][f"{k}_se"] = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
            for h in ("obj", "con"):
                dn = np.array([it[reg_key][mn][f"delta_nrmse_{h}"] for it in f_items], dtype=float)
                models_agg[mn][f"pos_neg_zero_tasks_{h}"] = [
                    int(np.sum(dn > 1e-4)),
                    int(np.sum(dn < -1e-4)),
                    int(np.sum(np.abs(dn) <= 1e-4)),
                ]

        # Paired contrasts on single-step NRMSE reduction (delta_nrmse_obj & delta_nrmse_con)
        # AND post-update NRMSE advantage (comparator after_nrmse minus structured after_nrmse)
        contrasts: dict[str, Any] = {}
        for h in ("obj", "con"):
            k_pairs = [
                ("Kernel_Full_vs_Iso", "Kernel-FullSource", "Kernel-IsoBasis"),
                ("Kernel_Diag_vs_Iso", "Kernel-DiagSource", "Kernel-IsoBasis"),
                ("Kernel_Full_vs_Diag", "Kernel-FullSource", "Kernel-DiagSource"),
                ("Kernel_Full_vs_Permuted", "Kernel-FullSource", "Kernel-PermutedScale"),
            ]
            c_del = {}
            p_list = []
            for cname, m_lhs, m_rhs in k_pairs:
                a_lhs = np.array([it[reg_key][m_lhs][f"delta_nrmse_{h}"] for it in f_items], dtype=float)
                a_rhs = np.array([it[reg_key][m_rhs][f"delta_nrmse_{h}"] for it in f_items], dtype=float)
                res = _paired_contrast(a_lhs, a_rhs)
                c_del[cname] = res
                p_list.append(res["paired_t_two_sided_p"])
            for (cname, _, _), hp in zip(k_pairs, _holm_adjust(p_list)):
                c_del[cname]["holm_two_sided_t_p"] = hp

            o_pairs = [
                ("Online_FullSURE_vs_Weak_AfterErr", "Online-SameBasisWeak", "Online-FullSURE"),
                ("Online_DiagSURE_vs_Weak_AfterErr", "Online-SameBasisWeak", "Online-DiagSURE"),
                ("Online_FullSURE_vs_DiagSURE_AfterErr", "Online-DiagSURE", "Online-FullSURE"),
                ("Online_FullSURE_vs_PermutedSURE_AfterErr", "Online-PermutedScaleSURE", "Online-FullSURE"),
            ]
            o_post = {}
            op_list = []
            for cname, m_bad, m_good in o_pairs:
                # Positive means m_good has LOWER post-update NRMSE than m_bad!
                a_bad = np.array([it[reg_key][m_bad][f"after_nrmse_{h}"] for it in f_items], dtype=float)
                a_good = np.array([it[reg_key][m_good][f"after_nrmse_{h}"] for it in f_items], dtype=float)
                res = _paired_contrast(a_bad, a_good)
                o_post[cname] = res
                op_list.append(res["paired_t_two_sided_p"])
            for (cname, _, _), hp in zip(o_pairs, _holm_adjust(op_list)):
                o_post[cname]["holm_two_sided_t_p"] = hp

            k_post_pairs = [
                ("Kernel_Iso_minus_Full_AfterErr", "Kernel-IsoBasis", "Kernel-FullSource"),
                ("Kernel_Iso_minus_Diag_AfterErr", "Kernel-IsoBasis", "Kernel-DiagSource"),
                ("Kernel_Diag_minus_Full_AfterErr", "Kernel-DiagSource", "Kernel-FullSource"),
                ("Kernel_Permuted_minus_Full_AfterErr", "Kernel-PermutedScale", "Kernel-FullSource"),
            ]
            c_kpost = {}
            kp_list = []
            for cname, m_bad, m_good in k_post_pairs:
                a_bad = np.array([it[reg_key][m_bad][f"after_nrmse_{h}"] for it in f_items], dtype=float)
                a_good = np.array([it[reg_key][m_good][f"after_nrmse_{h}"] for it in f_items], dtype=float)
                res = _paired_contrast(a_bad, a_good)
                c_kpost[cname] = res
                kp_list.append(res["paired_t_two_sided_p"])
            for (cname, _, _), hp in zip(k_post_pairs, _holm_adjust(kp_list)):
                c_kpost[cname]["holm_two_sided_t_p"] = hp

            contrasts[f"kernel_step_gain_{h}"] = c_del
            contrasts[f"kernel_post_accuracy_{h}"] = c_kpost
            contrasts[f"online_post_accuracy_{h}"] = o_post

        fam_block[reg_key] = {
            "models": models_agg,
            "contrasts": contrasts,
        }
    return fam_block


def _stabilize_floats(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _stabilize_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_stabilize_floats(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return 0.0 if abs(v) == 0.0 else float(f"{v:.12g}")
    return obj


def summarize(out_dir: Path, summary_path: Path) -> dict[str, Any]:
    desc = json.loads((out_dir / "jobs.json").read_text(encoding="utf-8"))
    items = [json.loads((out_dir / "items" / f"{jb['job_id']}.json").read_text(encoding="utf-8")) for jb in desc["jobs"]]
    fams_present = sorted({it["family"] for it in items})

    fam_summaries: dict[str, Any] = {}
    for fam in fams_present:
        f_items = [it for it in items if it["family"] == fam]
        fam_summaries[fam] = _summarize_item_group(f_items)

    des_items = [it for it in items if it["family"] != "OPSD_EuroGrid"]
    if des_items:
        fam_summaries["Pooled_DES_72"] = _summarize_item_group(des_items)

    fees = {
        k: sum(it[k] for it in items)
        for k in ("diagnostic_search_calls", "diagnostic_verification_calls", "diagnostic_audit_calls", "worker_cpu_seconds", "worker_wall_seconds")
    }
    summary = _stabilize_floats({
        "diagnostic_name": desc["diagnostic_name"],
        "panel": desc["panel"],
        "execution_code_sha256": desc["execution_code_sha256"],
        "total_tasks": len(items),
        "diagnostic_fees": fees,
        "families": fam_summaries,
    })
    _write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("describe", "worker", "summarize", "submit"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", type=Path, default=SC_ROOT / "v2_or/manifests/v2_or_metaprior_opsd_confirm_v1.json")
        p.add_argument("--panel", choices=["exploratory_86", "confirm_fresh_72"], default="exploratory_86")
        p.add_argument("--out-dir", type=Path, default=None)
        if name == "worker":
            p.add_argument("--start", type=int, required=True)
            p.add_argument("--end", type=int, required=True)
            p.add_argument("--workers", type=int, default=1)
        elif name == "summarize":
            p.add_argument("--summary-path", type=Path, default=None)
        elif name == "submit":
            p.add_argument("--scheduler", type=Path, default=Path(os.environ.get("GPFTA_CLUSTER_SCHEDULER", "scheduler.py")))
            p.add_argument("--remote-cwd", default=os.environ.get("GPFTA_REMOTE_CWD", "."))
            p.add_argument("--remote-python", default=sys.executable)
            p.add_argument("--nodes", default=",".join(DEFAULT_CPU_NODES))
            p.add_argument("--ram-mb", type=int, default=14336)
    args = parser.parse_args()

    out_dir = args.out_dir or (SC_ROOT / f"paper_artifacts/v2_or_metaprior_opsd_confirm_v1/direct_propagation_{args.panel}")

    if args.cmd == "describe":
        res = describe(args.manifest, args.panel, out_dir)
        print(json.dumps({"panel": args.panel, "n_jobs": res["n_jobs"]}))
    elif args.cmd == "worker":
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        desc = json.loads((out_dir / "jobs.json").read_text(encoding="utf-8"))
        sub_jobs = desc["jobs"][args.start:args.end]
        w = max(1, min(int(args.workers), len(sub_jobs)))
        if w == 1:
            for jb in sub_jobs:
                print(json.dumps(_run_one_task(manifest, args.panel, out_dir, jb)), flush=True)
        else:
            with ProcessPoolExecutor(max_workers=w) as ex:
                futs = [ex.submit(_run_one_task, manifest, args.panel, out_dir, jb) for jb in sub_jobs]
                for f in as_completed(futs):
                    print(json.dumps(f.result()), flush=True)
    elif args.cmd == "summarize":
        s_path = args.summary_path or (SC_ROOT / f"paper_artifacts/v2_or_metaprior_opsd_confirm_v1/direct_propagation_{args.panel}_readout.json")
        s = summarize(out_dir, s_path)
        print(json.dumps({"summary_path": str(s_path), "total_tasks": s["total_tasks"]}))
    elif args.cmd == "submit":
        nodes = parse_cpu_nodes(args.nodes)
        desc = json.loads((out_dir / "jobs.json").read_text(encoding="utf-8"))
        cmd = shlex.join([
            "env", "OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
            "PYTHONPATH=.:../scripts",
            args.remote_python, "-u", "../scripts/run_direct_propagation_study.py",
            "worker", "--manifest", str(args.manifest.relative_to(SC_ROOT)),
            "--panel", args.panel,
            "--out-dir", str(out_dir.relative_to(SC_ROOT)),
            "--start", "{start}", "--end", "{end}", "--workers", "{workers}",
        ])
        s_cmd = [
            sys.executable, str(args.scheduler), "submit-cpu-batch",
            "--items", str(desc["n_jobs"]), "--cmd-template", cmd, "--cwd", args.remote_cwd,
            "--nodes", ",".join(nodes), "--signature", f"gp_fta_v2/dprop_{args.panel}/{{start}}-{{end}}",
            "--description", f"v2_or direct prop {args.panel} {{node}} [{{start}},{{end}})",
            "--ram-mb", str(args.ram_mb), "--project", "gp_fta_v2",
            "--no-auto-requeue", "--skip-launch-staging", "--env-spec", "none", "--json",
        ]
        cp = subprocess.run(s_cmd, text=True, capture_output=True, check=True)
        print(cp.stdout, end="")


if __name__ == "__main__":
    main()
