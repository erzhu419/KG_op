#!/usr/bin/env python3
"""Cluster study R7-1: External frontier baselines, contract relaxation, verification
gate sensitivity, and high-budget reference oracle on the active 86-task panel.

Strictly preserves sealed `SC-OLH-KG/v2_or` code (`fe8df9098424...`) and all 1,376
previously completed runs while enforcing a strict two-stage barrier:
  Stage 1 (`search_and_verify`, `truth_used = False`):
      1. Closed-loop arms across all 86 tasks x 2 streams (172 cells/arm):
         - `SCBO-3Slot`
         - `RGPE-on-Psi-3Slot`
         - `WarmStart-MTGP-KG-3Slot`
         - `GP-FTA-LatentNEI-3Slot`
         - `GP-FTA-AllowUnvisited`
         - `GP-FTA-DirectMaxU`
         - `GP-FTA-LongSearch-k0`
      2. Terminal gate sensitivity & sequential curtailment across all 172 cells:
         - `Gate-k0` `(72, 82, 90)`
         - `Gate-k1-Default` `(112, 128, 140)` + `Gate-k1-Curtailed`
         - `Gate-k2` `(142, 155, 165)`
         - `Gate-k3` `(173, 187, 198)`
  Stage 2 (`audit_and_oracle`, `truth_used = True`):
      Runs only after Stage 1 completes. Audits frozen deployed policies on the exact
      task `audit_seed` (2,000 DES / 8,000 OPSD draws) and computes the 86-task
      high-budget offline `Ref-Oracle` feasible benchmark (`p_audit <= 0.05` and `<= 0.02`).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr, ndtri
from scipy.stats import binom, norm, qmc, t as student_t, wilcoxon

REPO_ROOT = Path(__file__).resolve().parents[1]
SC_ROOT = REPO_ROOT / "SC-OLH-KG"
if str(SC_ROOT) not in sys.path:
    sys.path.insert(0, str(SC_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from v2_or.acquisition.verification_kg import (
    SequentialRunResult,
    build_balanced_warmstart_points,
    generate_dynamic_candidate_pool,
    score_gate_power_vkg,
)
from v2_or.baselines.sota_baselines import (
    RawInputExactGP,
    _archive_row_noise_vars,
    pretrain_rgpe_psi_source_ensemble,
    run_rgpe_on_psi_baseline,
    run_scbo_baseline,
)
from v2_or.contracts import BaseDynamicORProblem, SufficientStatsRecord
from v2_or.dispatch.campaign_runner import build_family_tasks, compute_v2_or_code_sha256
from v2_or.legacy_solvers import load_legacy_archive_json
from v2_or.models.tempered_hvd_posterior import (
    JointPosteriorPrediction,
    TemperedHierarchicalHVDModel,
    _safe_cholesky,
)
from v2_or.priors.debiased_spectral_prior import (
    DebiasedSpectralMetaPrior,
    dct_profile_to_policy,
    sample_smooth_dct_delta,
)
from v2_or.verification.gated_verifier_v2 import (
    GateCertificate,
    TerminalDeploymentReceipt,
    build_feasible_aware_shortlist,
    exact_binomial_critical_k,
    exact_binomial_gate,
    gaussian_noncentral_t_gate,
    posterior_binomial_gate_power,
    verify_and_deploy_shortlist,
)

OUT_ROOT = SC_ROOT / "paper_artifacts" / "v2_or_rev7_or_panel_v1" / "closedloop_study"
PROSP_DIR = SC_ROOT / "paper_artifacts" / "v2_or_eckg_sure_confirm_v1"
MP_DIR = SC_ROOT / "paper_artifacts" / "v2_or_metaprior_opsd_confirm_v1"
U3_BUNDLES = SC_ROOT / "paper_artifacts" / "v2_or_u3_confirm_v1" / "source_bundles"

CLOSEDLOOP_ARMS = (
    "SCBO-3Slot",
    "RGPE-on-Psi-3Slot",
    "WarmStart-MTGP-KG-3Slot",
    "GP-FTA-LatentNEI-3Slot",
    "GP-FTA-AllowUnvisited",
    "GP-FTA-DirectMaxU",
    "GP-FTA-LongSearch-k0",
)

GATE_SPECS = {
    "Gate-k0": (72, 82, 90),
    "Gate-k1-Default": (112, 128, 140),
    "Gate-k2": (142, 155, 165),
    "Gate-k3": (173, 187, 198),
}


def _stab_float(val: float) -> float:
    return float(f"{float(val):.12g}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_active_family_specs() -> list[dict[str, Any]]:
    p_man = PROSP_DIR / "sealed_manifest.json"
    if not p_man.is_file():
        p_man = SC_ROOT / "v2_or" / "manifests" / "v2_or_eckg_sure_confirm_v1.json"
    m_man = MP_DIR / "sealed_manifest.json"
    if not m_man.is_file():
        m_man = SC_ROOT / "v2_or" / "manifests" / "v2_or_metaprior_opsd_confirm_v1.json"
    m_prosp = _read_json(p_man)
    m_mp = _read_json(m_man)
    fams = list(m_prosp["families"])
    for f in m_mp["families"]:
        if f["name"] == "OPSD_EuroGrid":
            fams.append(f)
            break
    return fams


def _eval_cascade_rosters_fast(
    sq_ref: float,
    r_mu_f: np.ndarray,
    r_cov_f: np.ndarray,
    r_av_f: np.ndarray,
    r_mu_g: np.ndarray,
    r_cov_g: np.ndarray,
    r_sig_g: np.ndarray,
    n_slots: np.ndarray,
    has_guard: np.ndarray,
    z_in: np.ndarray,
    slot_budgets: tuple[int, int, int] = (112, 128, 140),
    alpha: float = 0.05,
    familywise_delta: float = 0.05,
) -> np.ndarray:
    """Evaluate working-model 3-slot cascade saving U(S) over candidate triples."""
    total = int(r_mu_f.shape[0])
    if total == 0:
        return np.zeros(0, dtype=float)
    d_alloc = (0.50 * familywise_delta, 0.30 * familywise_delta, 0.20 * familywise_delta)
    k_crits = [max(exact_binomial_critical_k(int(slot_budgets[j]), alpha, d_alloc[j]), 0) for j in range(3)]
    sb_arr = np.asarray(slot_budgets, dtype=float)

    c0 = sq_ref - r_mu_f[:, 0]
    c2 = np.where(n_slots >= 2, sq_ref - r_mu_f[:, 1], 0.0)
    c3 = np.where(n_slots >= 3, sq_ref - r_mu_f[:, 2], 0.0)

    diff = r_mu_f[:, 0] - r_mu_f[:, 1]
    v_diff = np.maximum(r_cov_f[:, 0, 0] + r_cov_f[:, 1, 1] - 2.0 * r_cov_f[:, 0, 1], 0.0)
    s_obs = np.sqrt(np.maximum(v_diff + r_av_f[:, 0] / sb_arr[0] + r_av_f[:, 1] / sb_arr[1], 1e-16))
    z_d = diff / s_obs
    guard_corr = diff * ndtr(z_d) + (v_diff / s_obs) * (np.exp(-0.5 * z_d * z_d) / np.sqrt(2.0 * np.pi))
    c1 = np.where(has_guard & (n_slots >= 2), guard_corr, 0.0)

    cov_sym = 0.5 * (r_cov_g + np.swapaxes(r_cov_g, 1, 2)) + 1e-9 * np.eye(3, dtype=float)[None, :, :]
    try:
        L_g = np.linalg.cholesky(cov_sym)
    except np.linalg.LinAlgError:
        evals, evecs = np.linalg.eigh(cov_sym)
        cov_rep = np.einsum("bij,bj,bkj->bik", evecs, np.maximum(evals, 1e-9), evecs) + 1e-9 * np.eye(3)[None, :, :]
        L_g = np.linalg.cholesky(cov_rep)

    g_draws = r_mu_g[:, None, :] + np.einsum("mk,bjk->bmj", z_in, L_g)
    p_viol = np.clip(ndtr(g_draws / np.maximum(r_sig_g[:, None, :], 1e-8)), 0.0, 1.0)
    a = np.empty_like(p_viol)
    for j in range(3):
        K_j = int(slot_budgets[j])
        kc = int(k_crits[j])
        pv = p_viol[:, :, j]
        qv = 1.0 - pv
        if kc == 0:
            a[:, :, j] = qv ** K_j
        elif kc == 1:
            a[:, :, j] = (qv ** (K_j - 1)) * (1.0 + (K_j - 1) * pv)
        else:
            a[:, :, j] = binom.cdf(kc, K_j, pv)
    a[:, :, 1] = np.where((n_slots >= 2)[:, None], a[:, :, 1], 0.0)
    a[:, :, 2] = np.where((n_slots >= 3)[:, None], a[:, :, 2], 0.0)

    a0, a1, a2 = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    m0 = np.mean(a0, axis=1)
    m1 = np.mean(a0 * a1, axis=1)
    m2 = np.mean((1.0 - a0) * a1, axis=1)
    m3 = np.mean((1.0 - a0) * (1.0 - a1) * a2, axis=1)
    return c0 * m0 + c1 * m1 + c2 * m2 + c3 * m3


def build_direct_max_u_shortlist(
    cands: Sequence[np.ndarray],
    pred: JointPosteriorPrediction,
    eval_recs: Sequence[SufficientStatsRecord],
    *,
    alpha: float = 0.05,
    sq_ref: float,
    z_in: np.ndarray,
    slot_budgets: tuple[int, int, int] = (112, 128, 140),
) -> tuple[list[tuple[str, np.ndarray]], float]:
    """Parameter-free shortlist nominator directly maximizing working-model cascade utility U(S)."""
    n_c = len(cands)
    if n_c == 1:
        return [("posterior_primary", np.asarray(cands[0], dtype=float).copy())], 0.0

    mf = pred.mean_obj
    mg = pred.mean_con
    sg = pred.std_con
    uv = np.maximum(pred.ucb_aleatoric_var_con, 1e-8)
    rp = np.maximum(pred.r_perp_con, 0.0)
    p_g0 = posterior_binomial_gate_power(mg, sg, uv, n_samples=slot_budgets[0], alpha=alpha, delta_slot=0.025, r_perp_con=rp)
    p_g2 = posterior_binomial_gate_power(mg, sg, uv, n_samples=slot_budgets[2], alpha=alpha, delta_slot=0.010, r_perp_con=rp)

    # Slot 1 empirical incumbent without arbitrary 0.55 threshold: best zero-violation record by blended cost, else min margin
    zero_v = [
        (0.5 * float(r.mean_obj) + 0.5 * float(mf[i]), float(r.mean_con), i)
        for i, r in enumerate(eval_recs[:n_c])
        if r.violation_count == 0
    ]
    if zero_v:
        zero_v.sort()
        i1 = int(zero_v[0][2])
    else:
        margins = [
            (float(r.mean_con) + float(norm.ppf(1.0 - alpha)) * np.sqrt(max(uv[i], 1e-8)), i)
            for i, r in enumerate(eval_recs[:n_c])
        ]
        margins.sort()
        i1 = int(margins[0][1])

    u1_single = np.maximum(sq_ref - mf, 0.0) * p_g0
    top0 = list(dict.fromkeys([int(i) for i in np.argsort(-u1_single)[: min(10, n_c)]] + [int(np.argmax(p_g0))]))
    top2 = list(dict.fromkeys([int(i) for i in np.argsort(-p_g2)[: min(6, n_c)]]))

    triples: list[list[int]] = []
    roles_list: list[list[str]] = []
    for i0 in top0:
        for i2 in top2:
            seq: list[int] = [i0]
            rls: list[str] = ["posterior_primary"]
            seen = {np.round(cands[i0], 5).tobytes()}
            k1 = np.round(cands[i1], 5).tobytes()
            if k1 not in seen:
                seq.append(i1)
                rls.append("empirical_feasible_incumbent")
                seen.add(k1)
            k2 = np.round(cands[i2], 5).tobytes()
            if k2 not in seen and len(seq) < 3:
                seq.append(i2)
                rls.append("conservative_posterior_interior")
            triples.append(seq)
            roles_list.append(rls)

    B = len(triples)
    r_mf = np.zeros((B, 3), dtype=float)
    r_cf = np.zeros((B, 3, 3), dtype=float)
    r_avf = np.full((B, 3), 1e-6, dtype=float)
    r_mg = np.zeros((B, 3), dtype=float)
    r_cg = np.eye(3, dtype=float)[None, :, :] * 1e-6
    r_cg = np.repeat(r_cg, B, axis=0)
    r_sg = np.ones((B, 3), dtype=float)
    ns_arr = np.zeros(B, dtype=int)
    hg_arr = np.zeros(B, dtype=bool)
    cov_g_tot = pred.cov_con + np.diag(0.25 * rp)

    for b_i, (seq, rls) in enumerate(zip(triples, roles_list)):
        L = len(seq)
        ns_arr[b_i] = L
        hg_arr[b_i] = L >= 2 and rls[1] == "empirical_feasible_incumbent"
        idx_a = np.asarray(seq, dtype=int)
        r_mf[b_i, :L] = mf[idx_a]
        r_cf[b_i, :L, :L] = pred.cov_obj[np.ix_(idx_a, idx_a)]
        r_avf[b_i, :L] = np.maximum(pred.aleatoric_var_obj[idx_a], 1e-8)
        r_mg[b_i, :L] = mg[idx_a]
        r_cg[b_i, :L, :L] = cov_g_tot[np.ix_(idx_a, idx_a)]
        r_sg[b_i, :L] = np.sqrt(uv[idx_a])

    u_vals = _eval_cascade_rosters_fast(
        sq_ref, r_mf, r_cf, r_avf, r_mg, r_cg, r_sg, ns_arr, hg_arr, z_in,
        slot_budgets=slot_budgets, alpha=alpha,
    )
    best_b = int(np.argmax(u_vals))
    shortlist = [(r, np.asarray(cands[idx], dtype=float).copy()) for r, idx in zip(roles_list[best_b], triples[best_b])]
    return shortlist, float(u_vals[best_b])


def score_latent_nei(
    pred: JointPosteriorPrediction,
    model: TemperedHierarchicalHVDModel,
    cands: Sequence[np.ndarray],
    exps: Sequence[Any],
    rng: np.random.Generator,
    *,
    alpha: float = 0.05,
    sq_ref: float,
    n_mc: int = 32,
) -> int:
    """Faithful latent-integrated Noisy Expected Improvement (Ungredda & Branke 2024, Sec. 5; Letham et al. 2019).

    Integrates joint posterior realizations over evaluated designs D_n and candidates C_n,
    recomputing the feasible sampled incumbent on each draw.
    """
    eval_recs = model.unique_records
    n_eval = len(eval_recs)
    all_x = [r.x_norm for r in eval_recs] + list(cands)
    all_e = [r.exposure for r in eval_recs] + list(exps)
    pred_all = model.predict_joint(all_x, all_e)
    n_tot = len(all_x)

    L_f = _safe_cholesky(pred_all.cov_obj)
    cov_g = pred_all.cov_con + np.diag(0.25 * np.maximum(pred_all.r_perp_con, 0.0))
    L_g = _safe_cholesky(cov_g)

    half_m = max(n_mc // 2, 4)
    z_raw = rng.standard_normal(size=(n_tot, half_m))
    z_f = np.hstack([z_raw, -z_raw])
    z_raw_g = rng.standard_normal(size=(n_tot, half_m))
    z_g = np.hstack([z_raw_g, -z_raw_g])

    f_draws = pred_all.mean_obj[:, None] + L_f @ z_f
    g_draws = pred_all.mean_con[:, None] + L_g @ z_g
    feas_lim = float(norm.ppf(alpha)) * np.sqrt(np.maximum(pred_all.ucb_aleatoric_var_con, 1e-8))[:, None]
    is_feas = g_draws <= feas_lim

    eval_f = f_draws[:n_eval, :]
    eval_feas = is_feas[:n_eval, :]
    cand_f = f_draws[n_eval:, :]
    cand_feas = is_feas[n_eval:, :]

    inc_per_draw = np.where(eval_feas, eval_f, np.inf)
    b_draws = np.min(inc_per_draw, axis=0)
    b_draws = np.where(np.isfinite(b_draws), b_draws, float(sq_ref))

    imp = np.maximum(b_draws[None, :] - cand_f, 0.0) * cand_feas.astype(float)
    nei = np.mean(imp, axis=1)
    if float(np.max(nei)) > 1e-12:
        return int(np.argmax(nei))
    # Fallback when all draws are infeasible: maximize posterior binomial gate power (standard cNEI fallback)
    p_gate = posterior_binomial_gate_power(
        pred.mean_con, pred.std_con, np.maximum(pred.ucb_aleatoric_var_con, 1e-8),
        n_samples=112, alpha=alpha, delta_slot=0.025, r_perp_con=pred.r_perp_con,
    )
    return int(np.argmax(p_gate))


def pretrain_mtgp_source_psi(records: Sequence[ Any ]) -> tuple[RawInputExactGP, RawInputExactGP, np.ndarray, np.ndarray, float]:
    """Fit source-mean GP on standardized psi across historical archive for WarmStart-MTGP-KG."""
    t0 = time.perf_counter()
    psi_all = np.array([r.psi for r in records], dtype=float)
    psi_mean = np.mean(psi_all, axis=0)
    psi_std = np.maximum(np.std(psi_all, axis=0, ddof=1), 1e-4)

    # Aggregate duplicate psi vectors across source archive
    by_key: dict[bytes, list[Any]] = {}
    for r in records:
        k = np.round((np.asarray(r.psi, dtype=float) - psi_mean) / psi_std, 4).tobytes()
        by_key.setdefault(k, []).append(r)

    rows = [ grp[0] for grp in by_key.values() ]
    if len(rows) > 360:
        idx = np.linspace(0, len(rows) - 1, 360, dtype=int)
        rows = [rows[int(i)] for i in idx]

    Psi_s = np.clip((np.array([r.psi for r in rows], dtype=float) - psi_mean[None, :]) / psi_std[None, :], -4.0, 4.0)
    yfs = np.array([r.mean_obj for r in rows], dtype=float)
    ygs = np.array([r.mean_con for r in rows], dtype=float)
    vfs, vgs = _archive_row_noise_vars(rows)
    gp_f = RawInputExactGP().fit(Psi_s, yfs, vfs)
    gp_g = RawInputExactGP().fit(Psi_s, ygs, vgs)
    return gp_f, gp_g, psi_mean, psi_std, float(time.perf_counter() - t0)


def run_warmstart_mtgp_kg(
    problem: BaseDynamicORProblem,
    prior: DebiasedSpectralMetaPrior,
    mtgp_bundle: tuple[RawInputExactGP, RawInputExactGP, np.ndarray, np.ndarray, float],
    *,
    n_online_steps: int = 60,
    n0_init: int = 8,
    batch_reps: int = 2,
    seed: int = 1001,
    pool_size: int = 160,
    gate_mode: str = "auto",
) -> SequentialRunResult:
    """Poloczek-Wang-Frazier (2017) style source-mean + target-discrepancy GP on psi with gate-power KG."""
    src_gp_f, src_gp_g, psi_mean, psi_std, _ = mtgp_bundle
    t0 = time.perf_counter()
    rng_init = np.random.default_rng(seed)
    rng_search = np.random.default_rng(seed + 100_003)
    verif_seed = seed + 200_007
    calls_start = problem.search_calls

    init_pts = build_balanced_warmstart_points(problem, prior, n0_init, rng_init)
    obs_Helper = TemperedHierarchicalHVDModel(prior, mode="tempered_sure_release")
    for x0 in init_pts:
        e0 = problem.nominal_exposure(x0)
        s0 = problem.simulate_replications(x0, batch_reps, rng_init, split="search")
        obs_Helper.observe(SufficientStatsRecord.from_samples(x0, s0, e0))

    sq_est_obj = float(obs_Helper.unique_records[0].mean_obj)
    disc_f = RawInputExactGP()
    disc_g = RawInputExactGP()
    gp_logvf = RawInputExactGP()
    gp_logvg = RawInputExactGP()

    def _norm_psi(exps: Sequence[Any]) -> np.ndarray:
        raw = np.array([e.psi for e in exps], dtype=float)
        return np.clip((raw - psi_mean[None, :]) / psi_std[None, :], -4.0, 4.0)

    def _fit_mtgp() -> None:
        recs = obs_Helper.unique_records
        Psi_o = _norm_psi([r.exposure for r in recs])
        yf = np.array([r.mean_obj for r in recs], dtype=float)
        yg = np.array([r.mean_con for r in recs], dtype=float)
        s2f = np.array([max(r.ssq_obj / max(r.n_reps - 1, 1), 1e-5) for r in recs], dtype=float)
        s2g = np.array([max(r.ssq_con / max(r.n_reps - 1, 1), 1e-5) for r in recs], dtype=float)
        vf = np.array([s2f[i] / max(r.n_reps, 1) for i, r in enumerate(recs)], dtype=float)
        vg = np.array([s2g[i] / max(r.n_reps, 1) for i, r in enumerate(recs)], dtype=float)
        ms_f, _ = src_gp_f.predict(Psi_o)
        ms_g, _ = src_gp_g.predict(Psi_o)
        disc_f.fit(Psi_o, yf - ms_f, vf)
        disc_g.fit(Psi_o, yg - ms_g, vg)
        gp_logvf.fit(Psi_o, np.log(s2f), np.full(len(recs), 0.5))
        gp_logvg.fit(Psi_o, np.log(s2g), np.full(len(recs), 0.5))

    def _predict_mtgp(exps: Sequence[Any], *, return_cov: bool = True) -> JointPosteriorPrediction:
        Psi_c = _norm_psi(exps)
        ms_f, vs_f = src_gp_f.predict(Psi_c)
        ms_g, vs_g = src_gp_g.predict(Psi_c)
        if return_cov:
            md_f, cov_df = disc_f.predict(Psi_c, return_cov=True)
            md_g, cov_dg = disc_g.predict(Psi_c, return_cov=True)
            cov_f = cov_df + np.diag(0.20 * np.maximum(vs_f ** 2, 0.0))
            cov_g = cov_dg + np.diag(0.20 * np.maximum(vs_g ** 2, 0.0))
            sf = np.sqrt(np.maximum(np.diag(cov_f), 1e-10))
            sg = np.sqrt(np.maximum(np.diag(cov_g), 1e-10))
        else:
            md_f, sd_f = disc_f.predict(Psi_c)
            md_g, sd_g = disc_g.predict(Psi_c)
            sf = np.sqrt(sd_f ** 2 + 0.20 * (vs_f ** 2))
            sg = np.sqrt(sd_g ** 2 + 0.20 * (vs_g ** 2))
            cov_f = np.diag(sf ** 2)
            cov_g = np.diag(sg ** 2)
        mlvf, _ = gp_logvf.predict(Psi_c)
        mlvg, slvg = gp_logvg.predict(Psi_c)
        av_f = np.exp(np.clip(mlvf, -12.0, 4.0))
        av_g = np.exp(np.clip(mlvg, -12.0, 4.0))
        ucb_g = np.exp(np.clip(mlvg + 0.5 * slvg, -12.0, 4.0))
        return JointPosteriorPrediction(
            mean_obj=ms_f + md_f,
            cov_obj=cov_f,
            std_obj=sf,
            mean_con=ms_g + md_g,
            cov_con=cov_g,
            std_con=sg,
            aleatoric_var_obj=av_f,
            aleatoric_var_con=av_g,
            ucb_aleatoric_var_con=ucb_g,
            r_perp_con=np.full(len(exps), 1e-4),
        )

    n_rep_q = 0
    for _ in range(int(n_online_steps)):
        _fit_mtgp()
        cands, exps = generate_dynamic_candidate_pool(
            problem, obs_Helper, prior, rng_search, pool_size=pool_size, enable_source_templates=True
        )
        pred = _predict_mtgp(exps, return_cov=True)
        sq_ref = 0.5 * sq_est_obj + 0.5 * float(pred.mean_obj[0])
        scores = score_gate_power_vkg(pred, alpha=problem.alpha, batch_reps=batch_reps, status_quo_obj=sq_ref)
        idx = int(np.argmax(scores))
        xn = cands[idx]
        if obs_Helper._policy_key(xn) in obs_Helper._records_by_key:
            n_rep_q += 1
        samp = problem.simulate_replications(xn, batch_reps, rng_search, split="search")
        obs_Helper.observe(SufficientStatsRecord.from_samples(xn, samp, exps[idx]))

    _fit_mtgp()
    eval_recs = obs_Helper.unique_records
    term_cands = [r.x_norm.copy() for r in eval_recs]
    term_pred = _predict_mtgp([r.exposure for r in eval_recs], return_cov=False)
    sq_ref_term = 0.5 * sq_est_obj + 0.5 * float(term_pred.mean_obj[0])
    shortlist = build_feasible_aware_shortlist(
        candidate_pool=term_cands,
        mean_obj=term_pred.mean_obj,
        mean_con=term_pred.mean_con,
        std_con=term_pred.std_con,
        ucb_aleat_var_con=term_pred.ucb_aleatoric_var_con,
        r_perp_con=term_pred.r_perp_con,
        initial_records=eval_recs,
        alpha=problem.alpha,
        nomination_mode="cascade_bellman_3slot",
        reference_obj=sq_ref_term,
    )
    receipt = verify_and_deploy_shortlist(problem, shortlist, seed=verif_seed, gate_mode=gate_mode)
    search_used = int(problem.search_calls - calls_start)
    return SequentialRunResult(
        method_name="WarmStart-MTGP-KG-3Slot",
        task_id=problem.task_id,
        seed=seed,
        n0_init_points=n0_init,
        n_online_steps=n_online_steps,
        search_calls_used=search_used,
        verification_calls_used=receipt.verification_calls_used,
        total_target_calls=search_used + receipt.verification_calls_used,
        wall_time_sec=time.perf_counter() - t0,
        deployed_policy=receipt.deployed_policy,
        certified=receipt.certified,
        selected_role=receipt.selected_role,
        posterior_beta_con=1.0,
        posterior_tau_shift_con=0.0,
        untempered_inflation_factor=1.0,
        n_unique_policies_evaluated=len(eval_recs),
        n_replication_queries=n_rep_q,
        receipt=receipt,
    )


def run_gpfta_variant(
    problem: BaseDynamicORProblem,
    prior: DebiasedSpectralMetaPrior,
    *,
    variant: str,
    seed: int,
    n0_init: int = 8,
    n_online_steps: int = 60,
    batch_reps: int = 2,
    pool_size: int = 160,
    gate_mode: str = "auto",
) -> tuple[SequentialRunResult, dict[str, Any]]:
    """Run GP-FTA closed-loop variants (`GP-FTA-LatentNEI-3Slot`, `GP-FTA-AllowUnvisited`,
    `GP-FTA-DirectMaxU`, `GP-FTA-LongSearch-k0`, or `GP-FTA-GateSensitivity`)."""
    t0 = time.perf_counter()
    rng_init = np.random.default_rng(seed)
    rng_search = np.random.default_rng(seed + 100_003)
    verif_seed = seed + 200_007
    calls_start = problem.search_calls

    steps_eff = 100 if variant == "GP-FTA-LongSearch-k0" else int(n_online_steps)
    slot_budgets = (72, 82, 90) if variant == "GP-FTA-LongSearch-k0" else (112, 128, 140)

    model = TemperedHierarchicalHVDModel(
        prior,
        mode="tempered_sure_release",
        use_source_basis=True,
        use_source_coef_prior=True,
        use_source_hvd_prior=True,
        n0_freeze=n0_init,
    )
    init_pts = build_balanced_warmstart_points(problem, prior, n0_init, rng_init)
    for x0 in init_pts:
        e0 = problem.nominal_exposure(x0)
        s0 = problem.simulate_replications(x0, batch_reps, rng_init, split="search")
        model.observe(SufficientStatsRecord.from_samples(x0, s0, e0))

    sq_est_obj = float(model.unique_records[0].mean_obj)
    n_rep_q = 0

    eng_in = qmc.Sobol(d=3, scramble=True, seed=seed + 71)
    u_in = np.clip(eng_in.random_base2(5), 1e-6, 1.0 - 1e-6)
    z_in64 = np.vstack([ndtri(u_in), -ndtri(u_in)])  # (64, 3)

    from v2_or.acquisition.verification_kg import score_eligible_cascade_kg

    for step_i in range(steps_eff):
        enable_tpl = not bool(getattr(model, "is_source_released_", False))
        cands, exps = generate_dynamic_candidate_pool(
            problem, model, prior, rng_search, pool_size=pool_size, enable_source_templates=enable_tpl
        )
        pred = model.predict_joint(cands, exps)
        sq_ref = 0.5 * sq_est_obj + 0.5 * float(pred.mean_obj[0])

        if variant == "GP-FTA-LatentNEI-3Slot":
            chosen_idx = score_latent_nei(pred, model, cands, exps, rng_search, alpha=problem.alpha, sq_ref=sq_ref, n_mc=32)
        elif variant == "GP-FTA-AllowUnvisited":
            scores = score_gate_power_vkg(pred, alpha=problem.alpha, batch_reps=batch_reps, status_quo_obj=sq_ref)
            chosen_idx = int(np.argmax(scores))
        elif variant == "GP-FTA-DirectMaxU":
            # Stage A screen (16 contenders) + Stage B menu lookahead using parameter-free DirectMaxU nominator
            _, diag = score_eligible_cascade_kg(
                pred, model, cands, exps, alpha=problem.alpha, batch_reps=batch_reps,
                status_quo_obj=sq_ref, nomination_mode="cascade_bellman_3slot", return_diagnostics=True,
            )
            # Re-score top contenders using DirectMaxU shortlist utility
            p_g0 = posterior_binomial_gate_power(
                pred.mean_con, pred.std_con, np.maximum(pred.ucb_aleatoric_var_con, 1e-8),
                n_samples=112, alpha=problem.alpha, delta_slot=0.025, r_perp_con=pred.r_perp_con,
            )
            u_marg = np.maximum(sq_ref - pred.mean_obj, 0.0) * p_g0
            contenders = list(dict.fromkeys([int(diag["chosen_query"])] + [int(i) for i in np.argsort(-u_marg)[:11]]))[:12]
            eval_recs_step = model.unique_records
            eval_x = [r.x_norm for r in eval_recs_step]
            eval_e = [r.exposure for r in eval_recs_step]
            pred_eval = model.predict_joint(eval_x, eval_e)
            _, u_base_curr = build_direct_max_u_shortlist(eval_x, pred_eval, eval_recs_step, alpha=problem.alpha, sq_ref=sq_ref, z_in=z_in64)
            best_q = contenders[0]
            best_voi = -1.0
            z_fan = np.array([[-1.35, -0.45, 0.45, 1.35], [-0.90, 0.90, -0.90, 0.90]], dtype=float)
            for q_idx in contenders:
                xq, eq = cands[q_idx], exps[q_idx]
                is_rev = model._policy_key(xq) in model._records_by_key
                # 4 antithetic rank-1 fantasy draws
                pool_x = list(eval_x) if is_rev else list(eval_x) + [xq]
                pool_e = list(eval_e) if is_rev else list(eval_e) + [eq]
                pr_p = model.predict_joint(pool_x, pool_e)
                q_pos = next((i for i, z in enumerate(pool_x) if model._policy_key(z) == model._policy_key(xq)), len(pool_x) - 1)
                sf_tot = np.sqrt(max(pr_p.cov_obj[q_pos, q_pos] + pr_p.aleatoric_var_obj[q_pos] / batch_reps, 1e-12))
                sg_tot = np.sqrt(max(pr_p.cov_con[q_pos, q_pos] + pr_p.aleatoric_var_con[q_pos] / batch_reps, 1e-12))
                kf = pr_p.cov_obj[:, q_pos] / sf_tot
                kg = pr_p.cov_con[:, q_pos] / sg_tot
                cov_f_next = pr_p.cov_obj - np.outer(kf, kf)
                cov_g_next = pr_p.cov_con - np.outer(kg, kg)
                sf_next = np.sqrt(np.maximum(np.diag(cov_f_next), 1e-12))
                sg_next = np.sqrt(np.maximum(np.diag(cov_g_next), 1e-12))
                voi_acc = 0.0
                for d_i in range(4):
                    mf_d = pr_p.mean_obj + kf * z_fan[0, d_i]
                    mg_d = pr_p.mean_con + kg * z_fan[1, d_i]
                    pr_d = JointPosteriorPrediction(
                        mean_obj=mf_d, cov_obj=cov_f_next, std_obj=sf_next,
                        mean_con=mg_d, cov_con=cov_g_next, std_con=sg_next,
                        aleatoric_var_obj=pr_p.aleatoric_var_obj,
                        aleatoric_var_con=pr_p.aleatoric_var_con,
                        ucb_aleatoric_var_con=pr_p.ucb_aleatoric_var_con,
                        r_perp_con=pr_p.r_perp_con,
                    )
                    _, u_admit = build_direct_max_u_shortlist(pool_x, pr_d, eval_recs_step, alpha=problem.alpha, sq_ref=sq_ref, z_in=z_in64)
                    voi_acc += max(u_admit, max(u_base_curr, 0.0)) - max(u_base_curr, 0.0)
                if voi_acc > best_voi:
                    best_voi = voi_acc
                    best_q = q_idx
            chosen_idx = best_q
        else:
            # GP-FTA-LongSearch-k0 or GP-FTA-GateSensitivity (exact EC-KG trajectory!)
            scores, diag = score_eligible_cascade_kg(
                pred, model, cands, exps, alpha=problem.alpha, batch_reps=batch_reps,
                status_quo_obj=sq_ref, nomination_mode="cascade_bellman_3slot", return_diagnostics=True,
            )
            chosen_idx = int(diag["chosen_query"])

        xn = cands[chosen_idx]
        if model._policy_key(xn) in model._records_by_key:
            n_rep_q += 1
        samp = problem.simulate_replications(xn, batch_reps, rng_search, split="search")
        model.observe(SufficientStatsRecord.from_samples(xn, samp, exps[chosen_idx]))

    eval_recs = model.unique_records
    eval_keys = {model._policy_key(r.x_norm) for r in eval_recs}
    term_cands = [r.x_norm.copy() for r in eval_recs]
    term_exps = [r.exposure for r in eval_recs]
    term_pred = model.predict_joint(term_cands, term_exps)
    sq_ref_term = 0.5 * sq_est_obj + 0.5 * float(term_pred.mean_obj[0])

    extra_info: dict[str, Any] = {}
    if variant == "GP-FTA-GateSensitivity":
        # Evaluate all 4 exact binomial gate levels + sequential curtailment on this exact frozen state!
        gate_results: dict[str, Any] = {}
        for g_name, g_budgets in GATE_SPECS.items():
            sl_g = build_feasible_aware_shortlist(
                candidate_pool=term_cands,
                mean_obj=term_pred.mean_obj,
                mean_con=term_pred.mean_con,
                std_con=term_pred.std_con,
                ucb_aleat_var_con=term_pred.ucb_aleatoric_var_con,
                r_perp_con=term_pred.r_perp_con,
                initial_records=eval_recs,
                alpha=problem.alpha,
                nomination_mode="cascade_bellman_3slot",
                reference_obj=sq_ref_term,
                slot_budgets=g_budgets,
            )
            # Run verification with sample-level curtailment tracking
            rcpt_g, curtailed_calls = _verify_shortlist_with_curtailment(
                problem, sl_g, seed=verif_seed, slot_budgets=g_budgets, gate_mode=gate_mode
            )
            gate_results[g_name] = {
                "slot_budgets": list(g_budgets),
                "certified": bool(rcpt_g.certified),
                "selected_slot": rcpt_g.selected_slot,
                "selected_role": rcpt_g.selected_role,
                "fixed_verification_calls": int(rcpt_g.verification_calls_used),
                "curtailed_verification_calls": int(curtailed_calls),
                "deployed_policy": None if rcpt_g.deployed_policy is None else [float(v) for v in rcpt_g.deployed_policy],
                "primary_policy": [float(v) for v in rcpt_g.primary_policy],
            }
        # Dummy receipt from Default
        sl_def = build_feasible_aware_shortlist(
            candidate_pool=term_cands, mean_obj=term_pred.mean_obj, mean_con=term_pred.mean_con,
            std_con=term_pred.std_con, ucb_aleat_var_con=term_pred.ucb_aleatoric_var_con,
            r_perp_con=term_pred.r_perp_con, initial_records=eval_recs, alpha=problem.alpha,
            nomination_mode="cascade_bellman_3slot", reference_obj=sq_ref_term,
        )
        rcpt_def, _ = _verify_shortlist_with_curtailment(problem, sl_def, seed=verif_seed, slot_budgets=(112, 128, 140), gate_mode=gate_mode)
        extra_info["gate_designs"] = gate_results
        res_out = SequentialRunResult(
            method_name=variant, task_id=problem.task_id, seed=seed, n0_init_points=n0_init,
            n_online_steps=steps_eff, search_calls_used=int(n0_init * batch_reps + steps_eff * batch_reps),
            verification_calls_used=rcpt_def.verification_calls_used,
            total_target_calls=int(n0_init * batch_reps + steps_eff * batch_reps) + rcpt_def.verification_calls_used,
            wall_time_sec=time.perf_counter() - t0, deployed_policy=rcpt_def.deployed_policy,
            certified=rcpt_def.certified, selected_role=rcpt_def.selected_role,
            posterior_beta_con=float(model.beta_con_), posterior_tau_shift_con=0.0,
            untempered_inflation_factor=1.0, n_unique_policies_evaluated=len(eval_recs),
            n_replication_queries=n_rep_q, receipt=rcpt_def,
        )
        return res_out, extra_info

    if variant == "GP-FTA-AllowUnvisited":
        unvis_cands, unvis_exps = generate_dynamic_candidate_pool(
            problem, model, prior, rng_search, pool_size=256, enable_source_templates=True
        )
        dedup_unvis_x = []
        dedup_unvis_e = []
        seen_u = set(eval_keys)
        for ux, ue in zip(unvis_cands, unvis_exps):
            uk = model._policy_key(ux)
            if uk not in seen_u:
                seen_u.add(uk)
                dedup_unvis_x.append(ux)
                dedup_unvis_e.append(ue)
        comb_cands = term_cands + dedup_unvis_x
        comb_exps = term_exps + dedup_unvis_e
        comb_pred = model.predict_joint(comb_cands, comb_exps)
        shortlist = build_feasible_aware_shortlist(
            candidate_pool=comb_cands,
            mean_obj=comb_pred.mean_obj,
            mean_con=comb_pred.mean_con,
            std_con=comb_pred.std_con,
            ucb_aleat_var_con=comb_pred.ucb_aleatoric_var_con,
            r_perp_con=comb_pred.r_perp_con,
            initial_records=eval_recs,
            alpha=problem.alpha,
            nomination_mode="cascade_bellman_3slot",
            reference_obj=sq_ref_term,
        )
        extra_info["slot0_is_unvisited"] = bool(model._policy_key(shortlist[0][1]) not in eval_keys)
        extra_info["any_slot_unvisited"] = bool(any(model._policy_key(sp) not in eval_keys for _, sp in shortlist))
    elif variant == "GP-FTA-DirectMaxU":
        shortlist, u_val = build_direct_max_u_shortlist(
            term_cands, term_pred, eval_recs, alpha=problem.alpha, sq_ref=sq_ref_term, z_in=z_in64
        )
        extra_info["direct_max_u_expected_saving"] = float(u_val)
    else:
        shortlist = build_feasible_aware_shortlist(
            candidate_pool=term_cands,
            mean_obj=term_pred.mean_obj,
            mean_con=term_pred.mean_con,
            std_con=term_pred.std_con,
            ucb_aleat_var_con=term_pred.ucb_aleatoric_var_con,
            r_perp_con=term_pred.r_perp_con,
            initial_records=eval_recs,
            alpha=problem.alpha,
            nomination_mode="cascade_bellman_3slot",
            reference_obj=sq_ref_term,
            slot_budgets=slot_budgets,
        )

    receipt, curtailed_calls = _verify_shortlist_with_curtailment(
        problem, shortlist, seed=verif_seed, slot_budgets=slot_budgets, gate_mode=gate_mode
    )
    extra_info["curtailed_verification_calls"] = int(curtailed_calls)
    if variant == "GP-FTA-LongSearch-k0":
        verif_used = int(curtailed_calls)
    else:
        verif_used = int(receipt.verification_calls_used)
    search_used = int(problem.search_calls - calls_start)
    res_out = SequentialRunResult(
        method_name=variant,
        task_id=problem.task_id,
        seed=seed,
        n0_init_points=n0_init,
        n_online_steps=steps_eff,
        search_calls_used=search_used,
        verification_calls_used=verif_used,
        total_target_calls=search_used + verif_used,
        wall_time_sec=time.perf_counter() - t0,
        deployed_policy=receipt.deployed_policy,
        certified=receipt.certified,
        selected_role=receipt.selected_role,
        posterior_beta_con=float(model.beta_con_),
        posterior_tau_shift_con=0.0,
        untempered_inflation_factor=1.0,
        n_unique_policies_evaluated=len(eval_recs),
        n_replication_queries=n_rep_q,
        receipt=receipt,
    )
    return res_out, extra_info


def _verify_shortlist_with_curtailment(
    problem: BaseDynamicORProblem,
    shortlist: Sequence[tuple[str, np.ndarray]],
    *,
    seed: int,
    slot_budgets: tuple[int, int, int] = (112, 128, 140),
    familywise_delta: float = 0.05,
    gate_mode: str = "auto",
) -> tuple[TerminalDeploymentReceipt, int]:
    """Run exact verification while also recording exact sequential binomial curtailment calls."""
    rng = np.random.default_rng(seed)
    use_mode = ("gaussian_noncentral_t" if problem.is_gaussian_constraint_noise else "exact_binomial") if gate_mode == "auto" else str(gate_mode)
    n_slots = min(len(shortlist), 3)
    if n_slots == 1:
        deltas = (float(familywise_delta),)
    elif n_slots == 2:
        deltas = (0.625 * float(familywise_delta), 0.375 * float(familywise_delta))
    else:
        deltas = (0.50 * float(familywise_delta), 0.30 * float(familywise_delta), 0.20 * float(familywise_delta))
    used_budgets = tuple(int(slot_budgets[min(i, len(slot_budgets) - 1)]) for i in range(n_slots))
    primary_x = np.asarray(shortlist[0][1], dtype=float).copy()
    shortlist_pols = tuple(np.asarray(x, dtype=float).copy() for _, x in shortlist[:n_slots])
    shortlist_rls = tuple(str(r) for r, _ in shortlist[:n_slots])

    certs: list[GateCertificate] = []
    slot_curtailed: list[int] = []
    fixed_calls = 0

    def _curtail_count(con_vec: np.ndarray, k_crit: int) -> int:
        K = int(con_vec.size)
        if k_crit < 0 or K == 0:
            return 0
        viols = (con_vec > 0.0).astype(int)
        v_acc = 0
        for m_idx in range(1, K + 1):
            v_acc += int(viols[m_idx - 1])
            if v_acc >= k_crit + 1:
                return m_idx
            if v_acc + (K - m_idx) <= k_crit:
                return m_idx
        return K

    def _run_slot(idx: int) -> GateCertificate:
        nonlocal fixed_calls
        role, x_pol = shortlist[idx]
        k_bud = used_budgets[idx]
        d_alloc = float(deltas[idx])
        samples = problem.simulate_replications(x_pol, k_bud, rng, split="verification")
        fixed_calls += k_bud
        if use_mode == "exact_binomial":
            ok, stat_val, viol_r, v_obs, k_crit = exact_binomial_gate(samples[:, 1], alpha=problem.alpha, delta_slot=d_alloc)
            slot_curtailed.append(_curtail_count(samples[:, 1], int(k_crit)))
        else:
            ok, stat_val, viol_r, v_obs = gaussian_noncentral_t_gate(samples[:, 1], alpha=problem.alpha, delta_slot=d_alloc)
            k_crit = None
            slot_curtailed.append(k_bud)
        cert = GateCertificate(
            slot_index=idx, role=role, certified=ok, method=use_mode, n_samples=k_bud,
            delta_allocated=d_alloc, critical_k=k_crit, observed_violations=v_obs,
            p_value_or_margin=stat_val, observed_violation_rate=viol_r,
            sample_mean_obj=float(np.mean(samples[:, 0])), sample_mean_con=float(np.mean(samples[:, 1])),
        )
        certs.append(cert)
        return cert

    def _make_rcpt(chosen_idx: int | None) -> TerminalDeploymentReceipt:
        if chosen_idx is not None:
            return TerminalDeploymentReceipt(
                decision_state="certified_deployment", deployed_policy=shortlist_pols[chosen_idx].copy(),
                selected_uncertified_candidate=None, primary_policy=primary_x,
                shortlist_policies=shortlist_pols, certified=True, selected_slot=chosen_idx,
                selected_role=shortlist_rls[chosen_idx], verification_calls_used=fixed_calls,
                familywise_delta=float(familywise_delta), slot_budgets_configured=used_budgets,
                slot_deltas_configured=deltas, certificates=tuple(certs), shortlist_roles=shortlist_rls,
            )
        best_uncert = min(certs, key=lambda c: (c.observed_violation_rate, c.sample_mean_obj))
        return TerminalDeploymentReceipt(
            decision_state="abstention", deployed_policy=None,
            selected_uncertified_candidate=shortlist_pols[best_uncert.slot_index].copy(),
            primary_policy=primary_x, shortlist_policies=shortlist_pols, certified=False,
            selected_slot=None, selected_role="abstained_uncertified", verification_calls_used=fixed_calls,
            familywise_delta=float(familywise_delta), slot_budgets_configured=used_budgets,
            slot_deltas_configured=deltas, certificates=tuple(certs), shortlist_roles=shortlist_rls,
        )

    has_guard = n_slots >= 2 and shortlist_rls[1] in {"empirical_feasible_incumbent", "safe_initial_incumbent"}
    c0 = _run_slot(0)
    if c0.certified and not has_guard:
        return _make_rcpt(0), int(sum(slot_curtailed))
    if has_guard:
        c1 = _run_slot(1)
        if c0.certified and c1.certified:
            # Complete full blocks on Slots 0 & 1 when both pass so guard mean comparison is sample-path identical
            return _make_rcpt(0 if c0.sample_mean_obj <= c1.sample_mean_obj else 1), int(used_budgets[0] + used_budgets[1])
        if c0.certified:
            return _make_rcpt(0), int(sum(slot_curtailed))
        if c1.certified:
            return _make_rcpt(1), int(sum(slot_curtailed))
        start_rem = 2
    else:
        start_rem = 1
    for idx in range(start_rem, n_slots):
        ci = _run_slot(idx)
        if ci.certified:
            return _make_rcpt(idx), int(sum(slot_curtailed))
    return _make_rcpt(None), int(sum(slot_curtailed))


def _run_stage1_item(job: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Execute one Stage-1 item (`truth_used = False`): either a closed-loop arm or GateSensitivity."""
    item_path = out_dir / "stage1" / "items" / f"{job['job_id']}.json"
    if item_path.is_file():
        return {"job_id": job["job_id"], "status": "cached"}

    fams = {f["name"]: f for f in _load_active_family_specs()}
    fam_spec = fams[job["family"]]
    prob = build_family_tasks(fam_spec, "target")[int(job["target_task_idx"])]
    src_fam = fam_spec.get("reuse_archive_from_family", job["family"])
    bundle_dir = U3_BUNDLES / src_fam
    prior_path = bundle_dir / "prior_default.pkl"
    archive_path = bundle_dir / "archive.json"

    arm_name = job["arm_name"]
    seed = int(job["seed"])
    n0 = int(job["n0_init"])
    n_steps = int(job["n_online_steps"])
    b_reps = int(job["batch_reps"])
    p_size = int(job["pool_size"])
    gate_mode = fam_spec.get("gate_mode", "auto")
    src_pretrain_sec = 0.0
    extra_info: dict[str, Any] = {}

    if arm_name == "SCBO-3Slot":
        res = run_scbo_baseline(
            prob, n_online_steps=n_steps, n0_init=n0, batch_reps=b_reps,
            seed=seed, pool_size=p_size, gate_mode=gate_mode,
        )
    elif arm_name == "RGPE-on-Psi-3Slot":
        records, _ = load_legacy_archive_json(archive_path)
        rgpe_cache = bundle_dir / "rgpe_psi_pretrain.pkl"
        psi_bundle = pickle.loads(rgpe_cache.read_bytes()) if rgpe_cache.is_file() else pretrain_rgpe_psi_source_ensemble(records)
        src_pretrain_sec = float(psi_bundle[6])
        res = run_rgpe_on_psi_baseline(
            prob, records, n_online_steps=n_steps, n0_init=n0, batch_reps=b_reps,
            seed=seed, pool_size=p_size, gate_mode=gate_mode, pretrained_bundle=psi_bundle,
        )
    elif arm_name == "WarmStart-MTGP-KG-3Slot":
        prior = pickle.loads(prior_path.read_bytes())
        mtgp_cache = out_dir / "cache" / f"mtgp_psi_{src_fam}.pkl"
        if mtgp_cache.is_file():
            mtgp_bundle = pickle.loads(mtgp_cache.read_bytes())
        else:
            records, _ = load_legacy_archive_json(archive_path)
            mtgp_bundle = pretrain_mtgp_source_psi(records)
            mtgp_cache.parent.mkdir(parents=True, exist_ok=True)
            tmp_c = mtgp_cache.with_suffix(f".{os.getpid()}.tmp")
            tmp_c.write_bytes(pickle.dumps(mtgp_bundle))
            tmp_c.replace(mtgp_cache)
        src_pretrain_sec = float(mtgp_bundle[4])
        res = run_warmstart_mtgp_kg(
            prob, prior, mtgp_bundle, n_online_steps=n_steps, n0_init=n0,
            batch_reps=b_reps, seed=seed, pool_size=p_size, gate_mode=gate_mode,
        )
    elif arm_name in {
        "GP-FTA-LatentNEI-3Slot",
        "GP-FTA-AllowUnvisited",
        "GP-FTA-DirectMaxU",
        "GP-FTA-LongSearch-k0",
        "GP-FTA-GateSensitivity",
    }:
        prior = pickle.loads(prior_path.read_bytes())
        res, extra_info = run_gpfta_variant(
            prob, prior, variant=arm_name, seed=seed, n0_init=n0,
            n_online_steps=n_steps, batch_reps=b_reps, pool_size=p_size, gate_mode=gate_mode,
        )
    else:
        raise ValueError(f"Unknown arm {arm_name!r}")

    rcpt = res.receipt
    payload = {
        "job_id": job["job_id"],
        "family": job["family"],
        "target_task_idx": int(job["target_task_idx"]),
        "task_id": prob.task_id,
        "policy_dim": int(prob.d),
        "rep_seed": int(job["rep_seed"]),
        "seed": seed,
        "method_name": arm_name,
        "search_calls_used": int(res.search_calls_used),
        "verification_calls_used": int(res.verification_calls_used),
        "total_target_calls": int(res.total_target_calls),
        "online_wall_time_sec": float(res.wall_time_sec),
        "source_pretrain_wall_sec": float(src_pretrain_sec),
        "decision_state": rcpt.decision_state,
        "certified": bool(res.certified),
        "selected_slot": rcpt.selected_slot,
        "selected_role": res.selected_role,
        "n_unique_policies_evaluated": int(res.n_unique_policies_evaluated),
        "n_replication_queries": int(res.n_replication_queries),
        "deployed_policy": None if res.deployed_policy is None else [float(v) for v in res.deployed_policy],
        "primary_policy": [float(v) for v in rcpt.primary_policy],
        "extra_info": extra_info,
        "execution_code_sha256": compute_v2_or_code_sha256(),
        "truth_used": False,
    }
    _write_json(item_path, payload)
    return {"job_id": job["job_id"], "status": "ok"}


def _compute_task_oracle(fam_spec: dict[str, Any], task_idx: int, out_dir: Path) -> dict[str, Any]:
    """Offline high-budget feasible reference benchmark (`Ref-Oracle`, Stage 2 only)."""
    oracle_path = out_dir / "stage2" / "oracles" / f"{fam_spec['name']}_t{task_idx:02d}.json"
    if oracle_path.is_file():
        return _read_json(oracle_path)

    prob = build_family_tasks(fam_spec, "target")[int(task_idx)]
    src_fam = fam_spec.get("reuse_archive_from_family", fam_spec["name"])
    prior = pickle.loads((U3_BUNDLES / src_fam / "prior_default.pkl").read_bytes())
    n_audit = int(fam_spec.get("audit_replications", 2000))
    audit_seed = int(fam_spec.get("audit_seed_base", 2056112001)) + 997 * int(task_idx)

    rng = np.random.default_rng(909_001 + 10_007 * int(task_idx))
    anchor = prob.domain_heuristic_anchor()
    n_ch = int(prob.num_control_channels)

    cands: list[np.ndarray] = [anchor.copy()]
    cands.extend(prior.warmstart_candidates(prob.d, 24, rng, anchor=anchor, num_channels=n_ch))
    sobol = qmc.Sobol(d=prob.d, scramble=True, seed=int(77_001 + task_idx))
    u_sob = sobol.random_base2(7)  # 128 space-filling points
    for row in u_sob:
        cands.append(np.clip(row, 0.0, 1.0))
    for i in range(200):
        rad = (0.06, 0.14, 0.25, 0.38)[i % 4]
        base = cands[i % min(len(cands), 32)]
        delta = sample_smooth_dct_delta(rng, prob.d, radius=rad, k_modes=12, num_channels=n_ch)
        cands.append(np.clip(base + delta, 0.0, 1.0))
    while len(cands) < 400:
        prof = rng.normal(0.0, 0.28 / np.arange(1, 13), size=12)
        cands.append(dct_profile_to_policy(prof, prob.d, num_channels=n_ch))
    cands = cands[:400]

    # Fast 12-replication multi-fidelity screen + local DCT polish around top 8 feasible contenders
    screen_rows: list[tuple[int, float, float, np.ndarray]] = []
    for x in cands:
        s = prob.simulate_replications(x, 12, rng, split="search")
        v_rate = float(np.mean(s[:, 1] > 0.0))
        m_obj = float(np.mean(s[:, 0]))
        m_con = float(np.mean(s[:, 1]))
        bucket = 0 if v_rate <= 0.09 else (1 if m_con <= 0.0 else 2)
        screen_rows.append((bucket, m_obj if bucket == 0 else m_con, v_rate, x))
    screen_rows.sort(key=lambda t: (t[0], t[1]))

    polished: list[np.ndarray] = [anchor.copy()] + [t[3] for t in screen_rows[:16]]
    for seed_x in [t[3] for t in screen_rows[:6]]:
        best_x = seed_x.copy()
        s_b = prob.simulate_replications(best_x, 16, rng, split="search")
        best_key = (0 if np.mean(s_b[:, 1] > 0.0) <= 0.065 else 1, float(np.mean(s_b[:, 0])), float(np.mean(s_b[:, 1])))
        for step_r in (0.08, 0.04, 0.02):
            for _ in range(8):
                cand_x = np.clip(best_x + sample_smooth_dct_delta(rng, prob.d, radius=step_r, k_modes=10, num_channels=n_ch), 0.0, 1.0)
                s_c = prob.simulate_replications(cand_x, 16, rng, split="search")
                ck = (0 if np.mean(s_c[:, 1] > 0.0) <= 0.065 else 1, float(np.mean(s_c[:, 0])), float(np.mean(s_c[:, 1])))
                if ck < best_key:
                    best_key = ck
                    best_x = cand_x
        polished.append(best_x)

    # Deduplicate and audit on exact `audit_seed`
    uniq_pols: list[np.ndarray] = []
    seen = set()
    for p_arr in polished:
        kb = np.round(p_arr, 6).tobytes()
        if kb not in seen:
            seen.add(kb)
            uniq_pols.append(p_arr)

    anchor_aud = prob.audit_ground_truth(anchor, n_eval=n_audit, seed=audit_seed, split="audit")
    aud_list = []
    for p_arr in uniq_pols[:24]:
        if np.allclose(p_arr, anchor, atol=1e-10, rtol=0.0):
            aud = dict(anchor_aud)
        else:
            aud = prob.audit_ground_truth(p_arr, n_eval=n_audit, seed=audit_seed, split="audit")
        aud_list.append(aud)

    feas_05 = [float(a["mc_objective_mean"]) for a in aud_list if float(a["mc_violation_prob"]) <= 0.05]
    feas_02 = [float(a["mc_objective_mean"]) for a in aud_list if float(a["mc_violation_prob"]) <= 0.02]
    anc_obj = float(anchor_aud["mc_objective_mean"])
    anc_viol = float(anchor_aud["mc_violation_prob"])
    min_viol_any = min(aud_list, key=lambda a: (float(a["mc_violation_prob"]), float(a["mc_objective_mean"])))

    res = {
        "family": fam_spec["name"],
        "target_task_idx": int(task_idx),
        "task_id": prob.task_id,
        "audit_seed": int(audit_seed),
        "n_audit": int(n_audit),
        "anchor_obj": anc_obj,
        "anchor_viol_prob": anc_viol,
        "oracle_feas05_obj": float(min(feas_05)) if feas_05 else float(min_viol_any["mc_objective_mean"]),
        "oracle_feas05_found": bool(len(feas_05) > 0),
        "oracle_cert02_obj": float(min(feas_02)) if feas_02 else (float(min(feas_05)) if feas_05 else float(min_viol_any["mc_objective_mean"])),
        "oracle_cert02_found": bool(len(feas_02) > 0),
        "best_viol_prob_found": float(min_viol_any["mc_violation_prob"]),
        "truth_used": True,
    }
    _write_json(oracle_path, res)
    return res


def _run_stage2_item(job: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Execute one Stage-2 item (`truth_used = True`): either task oracle or deployed-policy audit."""
    if job["kind"] == "oracle":
        fams = {f["name"]: f for f in _load_active_family_specs()}
        _compute_task_oracle(fams[job["family"]], int(job["target_task_idx"]), out_dir)
        return {"job_id": job["job_id"], "status": "ok"}

    aud_path = out_dir / "stage2" / "items" / f"{job['job_id']}.json"
    if aud_path.is_file():
        return {"job_id": job["job_id"], "status": "cached"}

    sv_item = _read_json(out_dir / "stage1" / "items" / f"{job['sv_job_id']}.json")
    fams = {f["name"]: f for f in _load_active_family_specs()}
    fam_spec = fams[sv_item["family"]]
    t_idx = int(sv_item["target_task_idx"])
    prob = build_family_tasks(fam_spec, "target")[t_idx]
    n_audit = int(fam_spec.get("audit_replications", 2000))
    audit_seed = int(fam_spec.get("audit_seed_base", 2056112001)) + 997 * t_idx
    x_anc = prob.domain_heuristic_anchor()
    anchor_aud = prob.audit_ground_truth(x_anc, n_eval=n_audit, seed=audit_seed, split="audit")
    anc_obj = float(anchor_aud["mc_objective_mean"])
    anc_viol = float(anchor_aud["mc_violation_prob"])

    def _audit_pol(pol_list: list[float] | None) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        if pol_list is None:
            return None, None
        x_d = np.asarray(pol_list, dtype=float)
        if np.allclose(x_d, x_anc, atol=1e-10, rtol=0.0):
            da = dict(anchor_aud)
        else:
            da = prob.audit_ground_truth(x_d, n_eval=n_audit, seed=audit_seed, split="audit")
        va = None
        if fam_spec["kind"] == "opsd":
            va = prob.audit_ground_truth(x_d, n_eval=n_audit, seed=audit_seed + 500_009, split="verification")
        return da, va

    if sv_item["method_name"] == "GP-FTA-GateSensitivity":
        gate_audits: dict[str, Any] = {}
        for g_name, g_rec in sv_item["extra_info"]["gate_designs"].items():
            da, va = _audit_pol(g_rec["deployed_policy"] if g_rec["certified"] else None)
            dep_obj = float(da["mc_objective_mean"]) if da is not None else None
            dep_viol = float(da["mc_violation_prob"]) if da is not None else None
            op_viol = dep_viol if dep_viol is not None else anc_viol
            sq_loss = dep_obj if dep_obj is not None else anc_obj
            gate_audits[g_name] = {
                "certified": bool(g_rec["certified"]),
                "selected_slot": g_rec["selected_slot"],
                "selected_role": g_rec["selected_role"],
                "fixed_verification_calls": int(g_rec["fixed_verification_calls"]),
                "curtailed_verification_calls": int(g_rec["curtailed_verification_calls"]),
                "status_quo_loss": sq_loss,
                "operating_viol_prob": op_viol,
                "deployed_viol_prob": dep_viol,
                "deployed_viol_ucb95": float(da["mc_viol_ci95_hi"]) if da is not None else None,
                "deployed_viol_lcb95": float(da["mc_viol_ci95_lo"]) if da is not None else None,
                "verif_pool_viol_prob": float(va["mc_violation_prob"]) if va is not None else None,
            }
        out_payload = {
            "job_id": job["job_id"],
            "sv_job_id": job["sv_job_id"],
            "family": sv_item["family"],
            "target_task_idx": t_idx,
            "task_id": sv_item["task_id"],
            "rep_seed": int(sv_item["rep_seed"]),
            "method_name": "GP-FTA-GateSensitivity",
            "anchor_obj": anc_obj,
            "anchor_viol_prob": anc_viol,
            "gate_audits": gate_audits,
            "truth_used": True,
        }
        _write_json(aud_path, out_payload)
        return {"job_id": job["job_id"], "status": "ok"}

    da, va = _audit_pol(sv_item["deployed_policy"] if sv_item["certified"] else None)
    dep_obj = float(da["mc_objective_mean"]) if da is not None else None
    dep_viol = float(da["mc_violation_prob"]) if da is not None else None
    sq_loss = dep_obj if dep_obj is not None else anc_obj
    op_viol = dep_viol if dep_viol is not None else anc_viol

    out_payload = {
        "job_id": job["job_id"],
        "sv_job_id": job["sv_job_id"],
        "family": sv_item["family"],
        "target_task_idx": t_idx,
        "task_id": sv_item["task_id"],
        "policy_dim": int(sv_item["policy_dim"]),
        "rep_seed": int(sv_item["rep_seed"]),
        "method_name": sv_item["method_name"],
        "search_calls_used": int(sv_item["search_calls_used"]),
        "verification_calls_used": int(sv_item["verification_calls_used"]),
        "total_target_calls": int(sv_item["total_target_calls"]),
        "online_wall_time_sec": float(sv_item["online_wall_time_sec"]),
        "source_pretrain_wall_sec": float(sv_item["source_pretrain_wall_sec"]),
        "certified": bool(sv_item["certified"]),
        "selected_slot": sv_item["selected_slot"],
        "selected_role": sv_item["selected_role"],
        "anchor_obj": anc_obj,
        "anchor_viol_prob": anc_viol,
        "deployed_obj": dep_obj,
        "deployed_viol_prob": dep_viol,
        "deployed_viol_lcb95": float(da["mc_viol_ci95_lo"]) if da is not None else None,
        "deployed_viol_ucb95": float(da["mc_viol_ci95_hi"]) if da is not None else None,
        "verif_pool_viol_prob": float(va["mc_violation_prob"]) if va is not None else None,
        "status_quo_loss": sq_loss,
        "operating_viol_prob": op_viol,
        "extra_info": sv_item.get("extra_info", {}),
        "truth_used": True,
    }
    _write_json(aud_path, out_payload)
    return {"job_id": job["job_id"], "status": "ok"}


def describe_stage(stage: str, out_dir: Path) -> dict[str, Any]:
    fams = _load_active_family_specs()
    jobs: list[dict[str, Any]] = []
    if stage == "stage1":
        all_arms = list(CLOSEDLOOP_ARMS) + ["GP-FTA-GateSensitivity"]
        for fam in fams:
            fam_name = fam["name"]
            t_conf = fam["target_config"]
            n_tgt = len(t_conf["markets"]) if "markets" in t_conf else int(t_conf["n_tasks"])
            rep_seeds = [int(s) for s in fam["target_eval_seeds"]]
            for t_idx in range(n_tgt):
                for r_seed in rep_seeds:
                    cell_seed = r_seed + 10_007 * t_idx
                    for arm_name in all_arms:
                        jobs.append({
                            "job_id": f"r7s1_{fam_name}_t{t_idx:02d}_s{r_seed}_{arm_name}",
                            "family": fam_name,
                            "target_task_idx": t_idx,
                            "rep_seed": r_seed,
                            "seed": cell_seed,
                            "n0_init": int(fam.get("n0_init", 8)),
                            "n_online_steps": 60,
                            "batch_reps": int(fam.get("batch_reps", 2)),
                            "pool_size": int(fam.get("pool_size", 160)),
                            "arm_name": arm_name,
                        })
    elif stage == "stage2":
        s1_desc = _read_json(out_dir / "stage1" / "jobs.json")
        missing = [j["job_id"] for j in s1_desc["jobs"] if not (out_dir / "stage1" / "items" / f"{j['job_id']}.json").is_file()]
        if missing:
            raise RuntimeError(f"Stage 1 incomplete: {len(missing)} items missing (e.g. {missing[0]})")
        for fam in fams:
            fam_name = fam["name"]
            t_conf = fam["target_config"]
            n_tgt = len(t_conf["markets"]) if "markets" in t_conf else int(t_conf["n_tasks"])
            for t_idx in range(n_tgt):
                jobs.append({
                    "job_id": f"r7orc_{fam_name}_t{t_idx:02d}",
                    "kind": "oracle",
                    "family": fam_name,
                    "target_task_idx": t_idx,
                })
        for j in s1_desc["jobs"]:
            jobs.append({
                "job_id": f"r7aud_{j['job_id']}",
                "kind": "audit",
                "sv_job_id": j["job_id"],
            })
    else:
        raise ValueError(f"Unknown stage {stage!r}")

    desc = {
        "stage": stage,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "execution_code_sha256": compute_v2_or_code_sha256(),
        "n_jobs": len(jobs),
        "jobs": jobs,
    }
    _write_json(out_dir / stage / "jobs.json", desc)
    return desc


def run_worker(stage: str, out_dir: Path, start: int, end: int, workers: int) -> None:
    desc = _read_json(out_dir / stage / "jobs.json")
    sub_jobs = desc["jobs"][int(start): int(end)]
    fn = _run_stage1_item if stage == "stage1" else _run_stage2_item
    n_w = max(1, min(int(workers), len(sub_jobs))) if sub_jobs else 1
    done = 0
    if n_w == 1:
        for jb in sub_jobs:
            fn(jb, out_dir)
            done += 1
    else:
        with ProcessPoolExecutor(max_workers=n_w) as ex:
            futs = [ex.submit(fn, jb, out_dir) for jb in sub_jobs]
            for f in as_completed(futs):
                f.result()
                done += 1
    print(json.dumps({"stage": stage, "start": start, "end": end, "completed": done}))


def _holm_adjust(p_vals: list[float]) -> list[float]:
    m = len(p_vals)
    order = np.argsort(p_vals)
    adj = np.empty(m, dtype=float)
    run_max = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, float(p_vals[idx]) * float(m - rank))
        run_max = max(run_max, val)
        adj[idx] = run_max
    return [float(v) for v in adj]


def _paired_stats(diffs: np.ndarray) -> dict[str, Any]:
    d = np.asarray(diffs, dtype=float)
    n = int(d.size)
    mean_d = float(np.mean(d))
    sd_d = float(np.std(d, ddof=1)) if n > 1 else 0.0
    se_d = sd_d / np.sqrt(max(n, 1))
    if se_d > 1e-15 and n > 1:
        t_stat = mean_d / se_d
        p_t = float(2.0 * student_t.sf(abs(t_stat), df=n - 1))
        t_crit = float(student_t.ppf(0.975, df=n - 1))
    else:
        p_t = 1.0
        t_crit = 0.0
    nz = d[np.abs(d) > 1e-12]
    if nz.size >= 5:
        try:
            p_w = float(wilcoxon(nz).pvalue)
        except Exception:
            p_w = 1.0
    else:
        p_w = 1.0
    wins = int(np.sum(d > 1e-12))
    losses = int(np.sum(d < -1e-12))
    ties = int(n - wins - losses)
    return {
        "n_tasks": n,
        "mean_diff": _stab_float(mean_d),
        "se_diff": _stab_float(se_d),
        "ci95_lo": _stab_float(mean_d - t_crit * se_d),
        "ci95_hi": _stab_float(mean_d + t_crit * se_d),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "p_two_sided": _stab_float(p_t),
        "p_wilcoxon": _stab_float(p_w),
    }


def _load_sealed_reference_runs() -> list[dict[str, Any]]:
    """Load sealed GP-FTA-ECKG, SingleSlotKG, FeasibleKG, PublicPsi-TargetGP-TS, and ColdPsi runs."""
    rows: list[dict[str, Any]] = []
    for csv_path, keep_methods in (
        (PROSP_DIR / "per_run_ledger.csv", {
            "GP-FTA-ECKG", "GP-FTA-SingleSlotKG", "GP-FTA-FeasibleKG",
            "GP-FTA-LOO-ECKG", "PublicPsi-TargetGP-TS",
        }),
        (MP_DIR / "per_run_ledger.csv", {
            "GP-FTA-ECKG", "GP-FTA-SingleSlotKG", "GP-FTA-FeasibleKG",
            "GP-FTA-LOO-ECKG", "PublicPsi-TargetGP-TS",
            "ColdPsi-TargetGP-ECKG", "ColdInit-MetaPrior-SURE-ECKG",
        }),
    ):
        with csv_path.open("r", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["method_name"] not in keep_methods:
                    continue
                if csv_path.parent == MP_DIR and r["family"] != "OPSD_EuroGrid" and r["method_name"] not in {
                    "ColdPsi-TargetGP-ECKG", "ColdInit-MetaPrior-SURE-ECKG"
                }:
                    continue
                cert = r["certified"].strip().lower() == "true"
                anc_obj = float(r["anchor_mc_obj"])
                anc_viol = float(r["anchor_mc_viol_prob"])
                dep_obj = float(r["certified_deployed_mc_obj"]) if (cert and r.get("certified_deployed_mc_obj")) else None
                dep_viol = float(r["deployed_mc_viol_prob"]) if (cert and r["deployed_mc_viol_prob"]) else None
                rows.append({
                    "family": r["family"],
                    "target_task_idx": int(r["target_task_idx"]),
                    "rep_seed": int(r["rep_seed"]),
                    "method_name": r["method_name"],
                    "search_calls_used": int(float(r["search_calls_used"])),
                    "verification_calls_used": int(float(r["verification_calls_used"])),
                    "total_target_calls": int(float(r["total_target_calls"])),
                    "online_wall_time_sec": float(r["online_wall_time_sec"]),
                    "certified": cert,
                    "anchor_obj": anc_obj,
                    "anchor_viol_prob": anc_viol,
                    "deployed_obj": dep_obj,
                    "deployed_viol_prob": dep_viol,
                    "deployed_viol_ucb95": float(r["deployed_mc_viol_ci95_hi"]) if (cert and r.get("deployed_mc_viol_ci95_hi")) else None,
                    "status_quo_loss": float(r["status_quo_loss"]),
                    "operating_viol_prob": dep_viol if dep_viol is not None else anc_viol,
                })
    return rows


def _stabilize_floats(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _stabilize_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_stabilize_floats(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return 0.0 if abs(v) == 0.0 else float(f"{v:.12g}")
    return obj


def summarize(out_dir: Path, summary_path: Path, ledger_csv_path: Path) -> dict[str, Any]:
    compact_inputs_path = summary_path.parent / "rev7_closedloop_compact_inputs.json"
    if (out_dir / "stage2" / "jobs.json").is_file():
        s2_desc = _read_json(out_dir / "stage2" / "jobs.json")
        exec_sha = s2_desc["execution_code_sha256"]
        oracles_list: list[dict[str, Any]] = []
        new_runs: list[dict[str, Any]] = []
        gate_sens_items: list[dict[str, Any]] = []

        for jb in s2_desc["jobs"]:
            if jb["kind"] == "oracle":
                orc = _read_json(out_dir / "stage2" / "oracles" / f"{jb['family']}_t{int(jb['target_task_idx']):02d}.json")
                oracles_list.append(orc)
            else:
                item = _read_json(out_dir / "stage2" / "items" / f"{jb['job_id']}.json")
                if item["method_name"] == "GP-FTA-GateSensitivity":
                    gate_sens_items.append(item)
                else:
                    new_runs.append(item)

        ledger_csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_cols = [
            "family", "target_task_idx", "rep_seed", "method_name",
            "search_calls_used", "verification_calls_used", "total_target_calls",
            "online_wall_time_sec", "certified", "selected_role",
            "anchor_obj", "anchor_viol_prob", "deployed_obj", "deployed_viol_prob",
            "deployed_viol_ucb95", "verif_pool_viol_prob", "status_quo_loss", "operating_viol_prob",
            "slot0_is_unvisited",
        ]
        with ledger_csv_path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=csv_cols)
            w.writeheader()
            for r in sorted(new_runs, key=lambda z: (z["family"], z["target_task_idx"], z["rep_seed"], z["method_name"])):
                row_d = {k: r.get(k) for k in csv_cols if k != "slot0_is_unvisited"}
                row_d["slot0_is_unvisited"] = bool(r.get("extra_info", {}).get("slot0_is_unvisited", False))
                w.writerow(row_d)

        _write_json(compact_inputs_path, _stabilize_floats({
            "execution_code_sha256": exec_sha,
            "oracles": sorted(oracles_list, key=lambda z: (z["family"], int(z["target_task_idx"]))),
            "gate_sensitivity": sorted(gate_sens_items, key=lambda z: (z["family"], int(z["target_task_idx"]), int(z["rep_seed"]))),
        }))
    else:
        for cand_p in (
            out_dir / "rev7_closedloop_compact_inputs.json",
            summary_path.parent / "rev7_closedloop_compact_inputs.json",
            SC_ROOT / "paper_artifacts" / "v2_or_rev7_or_panel_v1" / "rev7_closedloop_compact_inputs.json",
        ):
            if cand_p.is_file():
                compact_inputs_path = cand_p
                break
        if not ledger_csv_path.is_file() and (out_dir / "per_run_ledger.csv").is_file():
            ledger_csv_path = out_dir / "per_run_ledger.csv"
        comp_in = _read_json(compact_inputs_path)
        exec_sha = comp_in["execution_code_sha256"]
        oracles_list = comp_in["oracles"]
        gate_sens_items = comp_in["gate_sensitivity"]
        new_runs = []
        with ledger_csv_path.open("r", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                cert = r["certified"].strip().lower() == "true"
                new_runs.append({
                    "family": r["family"],
                    "target_task_idx": int(r["target_task_idx"]),
                    "rep_seed": int(r["rep_seed"]),
                    "method_name": r["method_name"],
                    "search_calls_used": int(float(r["search_calls_used"])),
                    "verification_calls_used": int(float(r["verification_calls_used"])),
                    "total_target_calls": int(float(r["total_target_calls"])),
                    "online_wall_time_sec": float(r["online_wall_time_sec"]),
                    "certified": cert,
                    "selected_role": r["selected_role"],
                    "anchor_obj": float(r["anchor_obj"]),
                    "anchor_viol_prob": float(r["anchor_viol_prob"]),
                    "deployed_obj": float(r["deployed_obj"]) if r["deployed_obj"] else None,
                    "deployed_viol_prob": float(r["deployed_viol_prob"]) if r["deployed_viol_prob"] else None,
                    "deployed_viol_ucb95": float(r["deployed_viol_ucb95"]) if r["deployed_viol_ucb95"] else None,
                    "verif_pool_viol_prob": float(r["verif_pool_viol_prob"]) if r["verif_pool_viol_prob"] else None,
                    "status_quo_loss": float(r["status_quo_loss"]),
                    "operating_viol_prob": float(r["operating_viol_prob"]),
                    "extra_info": {"slot0_is_unvisited": r["slot0_is_unvisited"].strip().lower() == "true"},
                })

    oracles = {(o["family"], int(o["target_task_idx"])): o for o in oracles_list}

    sealed_rows = _load_sealed_reference_runs()
    all_rows = sealed_rows + new_runs

    # Task-level aggregation per (family, method_name, target_task_idx)
    by_fam_method_task: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for r in all_rows:
        by_fam_method_task.setdefault((r["family"], r["method_name"], int(r["target_task_idx"])), []).append(r)

    families = ["Inventory_CrossHorizon", "Queueing_CrossHorizon", "OPSD_EuroGrid", "Traffic_Ingolstadt"]
    des_families = ["Inventory_CrossHorizon", "Queueing_CrossHorizon", "Traffic_Ingolstadt"]
    all_methods = [
        "GP-FTA-ECKG",
        "SCBO-3Slot",
        "GP-FTA-LatentNEI-3Slot",
        "RGPE-on-Psi-3Slot",
        "WarmStart-MTGP-KG-3Slot",
        "GP-FTA-AllowUnvisited",
        "GP-FTA-DirectMaxU",
        "GP-FTA-LongSearch-k0",
        "GP-FTA-SingleSlotKG",
        "GP-FTA-FeasibleKG",
        "PublicPsi-TargetGP-TS",
    ]

    # Oracle summary & feasible gap closure per family/method
    oracle_summary: dict[str, Any] = {}
    for fam in families:
        fam_orcs = [v for (f, _), v in sorted(oracles.items()) if f == fam]
        anc_m = float(np.mean([o["anchor_obj"] for o in fam_orcs]))
        orc05_m = float(np.mean([o["oracle_feas05_obj"] for o in fam_orcs]))
        orc02_m = float(np.mean([o["oracle_cert02_obj"] for o in fam_orcs]))
        oracle_summary[fam] = {
            "n_tasks": len(fam_orcs),
            "anchor_mean_obj": _stab_float(anc_m),
            "oracle_feas05_mean_obj": _stab_float(orc05_m),
            "oracle_cert02_mean_obj": _stab_float(orc02_m),
            "mean_achievable_feas05_gap": _stab_float(max(anc_m - orc05_m, 0.0)),
            "mean_achievable_cert02_gap": _stab_float(max(anc_m - orc02_m, 0.0)),
            "feas05_found_count": int(sum(1 for o in fam_orcs if o["oracle_feas05_found"])),
        }

    domain_tables: dict[str, Any] = {}
    for fam in families:
        n_tgt = 14 if fam == "OPSD_EuroGrid" else 24
        fam_block: dict[str, Any] = {}
        for m in all_methods:
            t_losses = []
            t_op_viol = []
            t_calls = []
            t_verif = []
            t_wall = []
            cert_cnt = 0
            tot_runs = 0
            exceed_ucb = 0
            unvis_s0_cnt = 0
            for ti in range(n_tgt):
                rlist = by_fam_method_task.get((fam, m, ti), [])
                if not rlist:
                    continue
                t_losses.append(float(np.mean([x["status_quo_loss"] for x in rlist])))
                t_op_viol.append(float(np.mean([x["operating_viol_prob"] for x in rlist])))
                t_calls.append(float(np.mean([x["total_target_calls"] for x in rlist])))
                t_verif.append(float(np.mean([x["verification_calls_used"] for x in rlist])))
                t_wall.append(float(np.mean([x["online_wall_time_sec"] for x in rlist])))
                for x in rlist:
                    tot_runs += 1
                    if x["certified"]:
                        cert_cnt += 1
                        if x.get("deployed_viol_ucb95") is not None and float(x["deployed_viol_ucb95"]) > 0.05 and float(x.get("deployed_viol_prob") or 0.0) > 0.05:
                            exceed_ucb += 1
                    if x.get("extra_info", {}).get("slot0_is_unvisited"):
                        unvis_s0_cnt += 1
            if not t_losses:
                continue
            arr_l = np.asarray(t_losses, dtype=float)
            anc_arr = np.asarray([oracles[(fam, ti)]["anchor_obj"] for ti in range(n_tgt)], dtype=float)
            orc_arr = np.asarray([oracles[(fam, ti)]["oracle_feas05_obj"] for ti in range(n_tgt)], dtype=float)
            denom_gap = float(np.mean(np.maximum(anc_arr - orc_arr, 1e-8)))
            closed_pct = 100.0 * float(np.mean(anc_arr - arr_l)) / denom_gap if fam != "Traffic_Ingolstadt" else (
                100.0 * float(np.mean(anc_arr - arr_l)) / max(float(np.mean(anc_arr - orc_arr)), 1e-6)
            )
            # L_comp across penalty weights {1, 2, 5, 10}
            lcomp_by_w = {}
            for w_pen in (1.0, 2.0, 5.0, 10.0):
                lc_vals = []
                for ti in range(n_tgt):
                    rlist = by_fam_method_task[(fam, m, ti)]
                    a_abs = abs(oracles[(fam, ti)]["anchor_obj"])
                    lc_vals.append(float(np.mean([
                        x["status_quo_loss"] + w_pen * a_abs * max(float(x["operating_viol_prob"]) - 0.05, 0.0)
                        for x in rlist
                    ])))
                lcomp_by_w[str(int(w_pen))] = _stab_float(float(np.mean(lc_vals)))

            fam_block[m] = {
                "n_tasks": len(t_losses),
                "n_runs": tot_runs,
                "mean_sq_loss": _stab_float(float(np.mean(arr_l))),
                "se_sq_loss": _stab_float(float(np.std(arr_l, ddof=1) / np.sqrt(len(arr_l)))),
                "mean_oracle_gap": _stab_float(float(np.mean(arr_l - orc_arr))),
                "feasible_gap_closed_pct": _stab_float(closed_pct),
                "cert_count": cert_cnt,
                "cert_rate": _stab_float(cert_cnt / max(tot_runs, 1)),
                "audit_point_exceed_count": exceed_ucb,
                "mean_operating_viol_prob": _stab_float(float(np.mean(t_op_viol))),
                "mean_total_calls": _stab_float(float(np.mean(t_calls))),
                "mean_verif_calls": _stab_float(float(np.mean(t_verif))),
                "mean_wall_sec": _stab_float(float(np.mean(t_wall))),
                "unvisited_slot0_runs": unvis_s0_cnt,
                "lcomp_by_penalty_weight": lcomp_by_w,
            }
        domain_tables[fam] = fam_block

    # Paired contrasts vs GP-FTA-ECKG: Step 1 (4 external frontier baselines with Holm), Step 2 (contract/budget arms)
    frontier_4 = ["SCBO-3Slot", "GP-FTA-LatentNEI-3Slot", "RGPE-on-Psi-3Slot", "WarmStart-MTGP-KG-3Slot"]
    contract_3 = ["GP-FTA-AllowUnvisited", "GP-FTA-DirectMaxU", "GP-FTA-LongSearch-k0"]

    contrasts: dict[str, Any] = {}
    for fam in families:
        n_tgt = 14 if fam == "OPSD_EuroGrid" else 24
        ref_vec = np.array([np.mean([x["status_quo_loss"] for x in by_fam_method_task[(fam, "GP-FTA-ECKG", ti)]]) for ti in range(n_tgt)])
        f_stats = {}
        raw_ps = []
        for comp in frontier_4:
            c_vec = np.array([np.mean([x["status_quo_loss"] for x in by_fam_method_task[(fam, comp, ti)]]) for ti in range(n_tgt)])
            st = _paired_stats(c_vec - ref_vec)
            f_stats[comp] = st
            raw_ps.append(st["p_two_sided"])
        for comp, hp in zip(frontier_4, _holm_adjust(raw_ps)):
            f_stats[comp]["p_holm_4frontier"] = _stab_float(hp)

        c_stats = {}
        raw_ps_c = []
        for comp in contract_3:
            c_vec = np.array([np.mean([x["status_quo_loss"] for x in by_fam_method_task[(fam, comp, ti)]]) for ti in range(n_tgt)])
            st = _paired_stats(c_vec - ref_vec)
            c_stats[comp] = st
            raw_ps_c.append(st["p_two_sided"])
        for comp, hp in zip(contract_3, _holm_adjust(raw_ps_c)):
            c_stats[comp]["p_holm_3contract"] = _stab_float(hp)
        contrasts[fam] = {"frontier_vs_eckg": f_stats, "contract_vs_eckg": c_stats}

    # Pooled DES (n = 72) under BOTH anchor-SD standardization AND EC-KG-SD standardization!
    pooled_contrasts: dict[str, Any] = {}
    for scale_mode in ("anchor_sd", "eckg_sd"):
        f_pool = {}
        p_list = []
        for comp in frontier_4:
            diffs_all = []
            for fam in des_families:
                anc_sd = float(np.std([oracles[(fam, ti)]["anchor_obj"] for ti in range(24)], ddof=1))
                ref_vec = np.array([np.mean([x["status_quo_loss"] for x in by_fam_method_task[(fam, "GP-FTA-ECKG", ti)]]) for ti in range(24)])
                eckg_sd = float(np.std(ref_vec, ddof=1))
                denom = anc_sd if scale_mode == "anchor_sd" else eckg_sd
                c_vec = np.array([np.mean([x["status_quo_loss"] for x in by_fam_method_task[(fam, comp, ti)]]) for ti in range(24)])
                diffs_all.append((c_vec - ref_vec) / max(denom, 1e-8))
            st = _paired_stats(np.concatenate(diffs_all))
            f_pool[comp] = st
            p_list.append(st["p_two_sided"])
        for comp, hp in zip(frontier_4, _holm_adjust(p_list)):
            f_pool[comp]["p_holm_4frontier"] = _stab_float(hp)

        c_pool = {}
        p_list_c = []
        for comp in contract_3 + ["GP-FTA-FeasibleKG", "GP-FTA-SingleSlotKG", "PublicPsi-TargetGP-TS"]:
            diffs_all = []
            for fam in des_families:
                anc_sd = float(np.std([oracles[(fam, ti)]["anchor_obj"] for ti in range(24)], ddof=1))
                ref_vec = np.array([np.mean([x["status_quo_loss"] for x in by_fam_method_task[(fam, "GP-FTA-ECKG", ti)]]) for ti in range(24)])
                eckg_sd = float(np.std(ref_vec, ddof=1))
                denom = anc_sd if scale_mode == "anchor_sd" else eckg_sd
                c_vec = np.array([np.mean([x["status_quo_loss"] for x in by_fam_method_task[(fam, comp, ti)]]) for ti in range(24)])
                diffs_all.append((c_vec - ref_vec) / max(denom, 1e-8))
            st = _paired_stats(np.concatenate(diffs_all))
            c_pool[comp] = st
            if comp in contract_3:
                p_list_c.append(st["p_two_sided"])
        for comp, hp in zip(contract_3, _holm_adjust(p_list_c)):
            c_pool[comp]["p_holm_3contract"] = _stab_float(hp)
        pooled_contrasts[scale_mode] = {"frontier_vs_eckg": f_pool, "contract_and_core_vs_eckg": c_pool}

    # Gate sensitivity & curtailment summary across all 4 families
    gate_summary: dict[str, Any] = {}
    for fam in families:
        f_items = [it for it in gate_sens_items if it["family"] == fam]
        n_tgt = 14 if fam == "OPSD_EuroGrid" else 24
        g_fam: dict[str, Any] = {}
        for g_name in GATE_SPECS:
            by_t: dict[int, list[dict[str, Any]]] = {}
            cert_c = 0
            false_cert_c = 0
            for it in f_items:
                gr = it["gate_audits"][g_name]
                by_t.setdefault(int(it["target_task_idx"]), []).append(gr)
                if gr["certified"]:
                    cert_c += 1
                    v_check = gr["verif_pool_viol_prob"] if fam == "OPSD_EuroGrid" else gr["deployed_viol_prob"]
                    if v_check is not None and float(v_check) > 0.05:
                        false_cert_c += 1
            t_loss = [float(np.mean([z["status_quo_loss"] for z in by_t[ti]])) for ti in range(n_tgt)]
            t_opv = [float(np.mean([z["operating_viol_prob"] for z in by_t[ti]])) for ti in range(n_tgt)]
            t_fix = [float(np.mean([z["fixed_verification_calls"] for z in by_t[ti]])) for ti in range(n_tgt)]
            t_curt = [float(np.mean([z["curtailed_verification_calls"] for z in by_t[ti]])) for ti in range(n_tgt)]
            g_fam[g_name] = {
                "slot_budgets": list(GATE_SPECS[g_name]),
                "cert_count": cert_c,
                "n_runs": len(f_items),
                "cert_rate": _stab_float(cert_c / max(len(f_items), 1)),
                "verif_law_point_exceed_count": false_cert_c,
                "mean_sq_loss": _stab_float(float(np.mean(t_loss))),
                "mean_operating_viol_prob": _stab_float(float(np.mean(t_opv))),
                "mean_fixed_verif_calls": _stab_float(float(np.mean(t_fix))),
                "mean_curtailed_verif_calls": _stab_float(float(np.mean(t_curt))),
                "curtailment_call_saving": _stab_float(float(np.mean(np.asarray(t_fix) - np.asarray(t_curt)))),
            }
        gate_summary[fam] = g_fam

    readout = _stabilize_floats({
        "protocol_id": "PREREGISTRATION_REV7_OR_PANEL_20261121",
        "execution_code_sha256": exec_sha,
        "n_new_closedloop_runs": len(new_runs),
        "n_gate_sensitivity_replays": len(gate_sens_items),
        "n_oracle_tasks": len(oracles),
        "oracle_summary": oracle_summary,
        "domain_tables": domain_tables,
        "domain_contrasts": contrasts,
        "pooled_des_72_contrasts": pooled_contrasts,
        "gate_design_and_curtailment": gate_summary,
    })
    _write_json(summary_path, readout)
    return readout


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_desc = sub.add_parser("describe")
    p_desc.add_argument("--stage", choices=["stage1", "stage2"], required=True)
    p_desc.add_argument("--out-dir", type=Path, default=OUT_ROOT)

    p_work = sub.add_parser("worker")
    p_work.add_argument("--stage", choices=["stage1", "stage2"], required=True)
    p_work.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    p_work.add_argument("--start", type=int, required=True)
    p_work.add_argument("--end", type=int, required=True)
    p_work.add_argument("--workers", type=int, default=1)

    p_sum = sub.add_parser("summarize")
    p_sum.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    p_sum.add_argument(
        "--summary-path",
        type=Path,
        default=SC_ROOT / "paper_artifacts" / "v2_or_rev7_or_panel_v1" / "rev7_closedloop_readout.json",
    )
    p_sum.add_argument(
        "--ledger-csv",
        type=Path,
        default=SC_ROOT / "paper_artifacts" / "v2_or_rev7_or_panel_v1" / "per_run_ledger.csv",
    )

    args = p.parse_args()
    if args.cmd == "describe":
        d = describe_stage(args.stage, args.out_dir)
        print(json.dumps({"stage": d["stage"], "n_jobs": d["n_jobs"]}))
    elif args.cmd == "worker":
        run_worker(args.stage, args.out_dir, args.start, args.end, args.workers)
    elif args.cmd == "summarize":
        ro = summarize(args.out_dir, args.summary_path, args.ledger_csv)
        print(json.dumps({"n_new_runs": ro["n_new_closedloop_runs"], "n_oracles": ro["n_oracle_tasks"]}))


if __name__ == "__main__":
    main()
