#!/usr/bin/env python3
"""Cluster diagnostic: Independent high-precision all-160 reference validation of EC-KG
on exact realized closed-loop trajectories (`step_0` and `step_59`).

Evaluates ALL 160 candidates (no Stage-A pre-screen) with independent scrambled Sobol
quadrature (`256` antithetic outer fantasies x `1024` inner draws) under BOTH:
  1. Nested menu-max proxy VOI (`VOI_menu_ref`), and
  2. Exact actual terminal single-nomination rule gain (`Delta_term_ref`, no menu retention).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from scipy.special import ndtr, ndtri
from scipy.stats import qmc, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
SC_ROOT = REPO_ROOT / "SC-OLH-KG"
if str(SC_ROOT) not in sys.path:
    sys.path.insert(0, str(SC_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from scheduler_node_policy import DEFAULT_CPU_NODES, parse_cpu_nodes
from v2_or.acquisition.verification_kg import (
    _SLOT_SIZES,
    build_balanced_warmstart_points,
    generate_dynamic_candidate_pool,
    score_chance_feasibility_kg,
    score_eligible_cascade_kg,
    score_gate_power_vkg,
)
from v2_or.contracts import SufficientStatsRecord
from v2_or.dispatch.campaign_runner import build_family_tasks, compute_v2_or_code_sha256
from v2_or.models import TemperedHierarchicalHVDModel
from v2_or.verification.gated_verifier_v2 import (
    build_feasible_aware_shortlist,
    exact_binomial_critical_k,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _make_ref_qmc(seed: int, m_out_pow2: int = 7, m_in_pow2: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Build independent antithetic 256-point outer 4D Sobol and 1024-point inner 3D Sobol grids."""
    eng_out = qmc.Sobol(d=4, scramble=True, seed=seed)
    u_out = np.clip(eng_out.random_base2(m=m_out_pow2), 1e-6, 1.0 - 1e-6)
    z_half = ndtri(u_out)
    z_out = np.vstack([z_half, -z_half])  # (256, 4)

    eng_in = qmc.Sobol(d=3, scramble=True, seed=seed + 17)
    u_in = np.clip(eng_in.random_base2(m=m_in_pow2 - 1), 1e-6, 1.0 - 1e-6)
    z_in_half = ndtri(u_in)
    z_in = np.vstack([z_in_half, -z_in_half])  # (1024, 3)
    return z_out, z_in


def _eval_rosters_ref(
    mu_f_sq: np.ndarray,
    r_mu_f: np.ndarray,
    r_cov_f: np.ndarray,
    r_av_f: np.ndarray,
    r_mu_g: np.ndarray,
    r_cov_g: np.ndarray,
    r_sig_g: np.ndarray,
    n_slots: np.ndarray,
    has_guard: np.ndarray,
    z_inner_1024: np.ndarray,
) -> np.ndarray:
    """Evaluate expected 3-slot cascade net saving in chunks using 1024 inner draws."""
    total = int(mu_f_sq.size)
    if total == 0:
        return np.zeros(0, dtype=float)
    out = np.empty(total, dtype=float)
    chunk = 512
    nm1 = (_SLOT_SIZES - 1)[None, None, :]
    for s0 in range(0, total, chunk):
        s1 = min(s0 + chunk, total)
        mf_sq = mu_f_sq[s0:s1]
        mf = r_mu_f[s0:s1]
        cf = r_cov_f[s0:s1]
        avf = r_av_f[s0:s1]
        mg = r_mu_g[s0:s1]
        cg = r_cov_g[s0:s1]
        sg = r_sig_g[s0:s1]
        ns = n_slots[s0:s1]
        hg = has_guard[s0:s1]

        c0 = mf_sq - mf[:, 0]
        c2 = np.where(ns >= 2, mf_sq - mf[:, 1], 0.0)
        c3 = np.where(ns >= 3, mf_sq - mf[:, 2], 0.0)

        diff = mf[:, 0] - mf[:, 1]
        v_diff = np.maximum(cf[:, 0, 0] + cf[:, 1, 1] - 2.0 * cf[:, 0, 1], 0.0)
        s_obs = np.sqrt(np.maximum(v_diff + avf[:, 0] / float(_SLOT_SIZES[0]) + avf[:, 1] / float(_SLOT_SIZES[1]), 1e-16))
        z_d = diff / s_obs
        guard_corr = diff * ndtr(z_d) + (v_diff / s_obs) * (np.exp(-0.5 * z_d * z_d) / np.sqrt(2.0 * np.pi))
        c1 = np.where(hg & (ns >= 2), guard_corr, 0.0)

        cov_sym = 0.5 * (cg + np.swapaxes(cg, 1, 2)) + 1e-10 * np.eye(3, dtype=float)[None, :, :]
        try:
            L_g = np.linalg.cholesky(cov_sym)
        except np.linalg.LinAlgError:
            evals, evecs = np.linalg.eigh(cov_sym)
            cov_rep = np.einsum("bij,bj,bkj->bik", evecs, np.maximum(evals, 1e-9), evecs) + 1e-9 * np.eye(3)[None, :, :]
            L_g = np.linalg.cholesky(cov_rep)

        g_draws = mg[:, None, :] + np.einsum("mk,bjk->bmj", z_inner_1024, L_g)
        p_viol = ndtr(g_draws / np.maximum(sg[:, None, :], 1e-8))
        a = ((1.0 - p_viol) ** nm1) * (1.0 + nm1 * p_viol)
        a[:, :, 1] = np.where((ns >= 2)[:, None], a[:, :, 1], 0.0)
        a[:, :, 2] = np.where((ns >= 3)[:, None], a[:, :, 2], 0.0)

        a0, a1, a2 = a[:, :, 0], a[:, :, 1], a[:, :, 2]
        m0 = np.mean(a0, axis=1)
        m1 = np.mean(a0 * a1, axis=1)
        m2 = np.mean((1.0 - a0) * a1, axis=1)
        m3 = np.mean((1.0 - a0) * (1.0 - a1) * a2, axis=1)
        out[s0:s1] = c0 * m0 + c1 * m1 + c2 * m2 + c3 * m3
    return out


def _evaluate_frozen_state_all160(
    pred: Any,
    model: TemperedHierarchicalHVDModel,
    cands: list[np.ndarray],
    exps: list[Any],
    *,
    alpha: float,
    batch_reps: int,
    sq_ref: float,
    ref_seed: int,
) -> dict[str, Any]:
    # 1. Production rules (top_k=16 fast ECKG, SingleSlotKG, FeasibleKG)
    t0 = time.perf_counter()
    s_fast16, diag16 = score_eligible_cascade_kg(
        pred, model, cands, exps, alpha=alpha, batch_reps=batch_reps,
        status_quo_obj=sq_ref, nomination_mode="cascade_bellman_3slot",
        top_k_contenders=16, return_diagnostics=True,
    )
    fast16_ms = 1000.0 * (time.perf_counter() - t0)

    # Also evaluate fast 32x128 across ALL 160 candidates (`top_k_contenders=len(cands)`) to isolate screening vs quadrature!
    s_fast_all160, _ = score_eligible_cascade_kg(
        pred, model, cands, exps, alpha=alpha, batch_reps=batch_reps,
        status_quo_obj=sq_ref, nomination_mode="cascade_bellman_3slot",
        top_k_contenders=len(cands), return_diagnostics=True,
    )

    s_single = score_gate_power_vkg(pred, alpha=alpha, batch_reps=batch_reps, status_quo_obj=sq_ref)
    s_feas = score_chance_feasibility_kg(pred, alpha=alpha, batch_reps=batch_reps, status_quo_obj=sq_ref)

    q_eckg16 = int(diag16["chosen_query"])
    q_single = int(np.argmax(s_single))
    q_feas = int(np.argmax(s_feas))

    # 2. Independent 256 x 1024 all-160 evaluation (both menu-max and actual single-nomination rule)
    z_out_256, z_in_1024 = _make_ref_qmc(ref_seed, m_out_pow2=7, m_in_pow2=10)
    n_outer = int(z_out_256.shape[0])
    m_cand = len(cands)

    obs_recs = model.unique_records
    key_to_cand = {model._policy_key(x): idx for idx, x in enumerate(cands)}
    elig_base_idx = []
    elig_base_recs = []
    for rec in obs_recs:
        k = model._policy_key(rec.x_norm)
        if k in key_to_cand:
            elig_base_idx.append(key_to_cand[k])
            elig_base_recs.append(rec)

    base_arr = np.asarray(elig_base_idx, dtype=int)
    m_obs = int(base_arr.size)
    base_pos_by_cand = {int(c_i): pos for pos, c_i in enumerate(elig_base_idx)}
    sq_cand_idx = int(base_arr[0])
    obs_anchor_val = float(elig_base_recs[0].mean_obj)

    av_f = np.maximum(pred.aleatoric_var_obj, 1e-10)
    av_g = np.maximum(pred.aleatoric_var_con, 1e-10)
    av_ucb_g = np.maximum(pred.ucb_aleatoric_var_con, 1e-8)
    sig_ucb_g = np.sqrt(av_ucb_g)
    rp_g = np.maximum(pred.r_perp_con, 0.0)
    cov_g_eff = pred.cov_con + np.diag(0.25 * rp_g)
    std_g_base = np.sqrt(np.maximum(np.diag(pred.cov_con), 0.0))

    base_cands = [cands[i] for i in elig_base_idx]
    curr_sl = build_feasible_aware_shortlist(
        candidate_pool=base_cands,
        mean_obj=pred.mean_obj[base_arr],
        mean_con=pred.mean_con[base_arr],
        std_con=std_g_base[base_arr],
        ucb_aleat_var_con=av_ucb_g[base_arr],
        r_perp_con=rp_g[base_arr],
        initial_records=elig_base_recs,
        alpha=alpha,
        nomination_mode="cascade_bellman_3slot",
        reference_obj=sq_ref,
        familywise_delta=0.05,
    )
    base_round_map: dict[bytes, int] = {}
    for pos, x_b in enumerate(base_cands):
        rk = np.round(x_b, 5).tobytes()
        if rk not in base_round_map:
            base_round_map[rk] = int(elig_base_idx[pos])
    curr_roster_cands = tuple(base_round_map[np.round(x, 5).tobytes()] for _, x in curr_sl)
    curr_guard = len(curr_sl) >= 2 and curr_sl[1][0] in {"empirical_feasible_incumbent", "safe_initial_incumbent"}
    curr_pad = np.array([
        curr_roster_cands[0],
        curr_roster_cands[1] if len(curr_roster_cands) >= 2 else curr_roster_cands[0],
        curr_roster_cands[2] if len(curr_roster_cands) >= 3 else curr_roster_cands[0],
    ], dtype=int)
    curr_len = len(curr_roster_cands)

    u_curr_ref = float(_eval_rosters_ref(
        np.array([pred.mean_obj[sq_cand_idx]]),
        pred.mean_obj[curr_pad][None, :],
        pred.cov_obj[np.ix_(curr_pad, curr_pad)][None, :, :],
        av_f[curr_pad][None, :],
        pred.mean_con[curr_pad][None, :],
        cov_g_eff[np.ix_(curr_pad, curr_pad)][None, :, :],
        sig_ucb_g[curr_pad][None, :],
        np.array([curr_len], dtype=int),
        np.array([curr_guard], dtype=bool),
        z_in_1024,
    )[0])
    curr_active = bool(u_curr_ref > 0.0)

    denom_f = np.sqrt(np.maximum(np.diag(pred.cov_obj) + av_f / float(batch_reps), 1e-10))
    denom_g = np.sqrt(np.maximum(np.diag(pred.cov_con) + av_g / float(batch_reps), 1e-10))
    tilde_f_all = pred.cov_obj / denom_f[:, None]
    tilde_g_all = pred.cov_con / denom_g[:, None]

    eval_sq_list: list[float] = []
    eval_mu_f: list[np.ndarray] = []
    eval_cov_f: list[np.ndarray] = []
    eval_av_f: list[np.ndarray] = []
    eval_mu_g: list[np.ndarray] = []
    eval_cov_g: list[np.ndarray] = []
    eval_sig_g: list[np.ndarray] = []
    eval_nslots: list[int] = []
    eval_guard: list[bool] = []
    # ptr roles: 0 = S_curr (always evaluated for actual terminal rule), 1 = post-query D_n, 2 = post-query D_n U {q}
    slot_ptr = np.full((m_cand, n_outer, 3), -1, dtype=int)

    t_ref0 = time.perf_counter()
    curr_local_pad = np.array([base_pos_by_cand[int(c)] for c in curr_pad], dtype=int)

    for q in range(m_cand):
        is_revisit = q in base_pos_by_cand
        elig_idx_q = base_arr if is_revisit else np.append(base_arr, q)
        elig_cands_q = base_cands if is_revisit else [*base_cands, cands[q]]
        tf_q = tilde_f_all[q, elig_idx_q]
        tg_q = tilde_g_all[q, elig_idx_q]
        if is_revisit and np.max(np.abs(tf_q)) < 1e-14 and np.max(np.abs(tg_q)) < 1e-14:
            continue

        cov_f_post = pred.cov_obj[np.ix_(elig_idx_q, elig_idx_q)] - np.outer(tf_q, tf_q)
        cov_g_post_raw = pred.cov_con[np.ix_(elig_idx_q, elig_idx_q)] - np.outer(tg_q, tg_q)
        std_g_post = np.sqrt(np.maximum(np.diag(cov_g_post_raw), 0.0))
        cov_g_post_eff = cov_g_post_raw + np.diag(0.25 * rp_g[elig_idx_q])

        lat_v = np.array([max(float(pred.cov_obj[q, q]), 0.0), max(float(pred.cov_con[q, q]), 0.0)])
        obs_v = np.array([float(av_f[q]), float(av_g[q])])
        means_q = np.array([float(pred.mean_obj[q]), float(pred.mean_con[q])])
        centers = means_q[None, :] + z_out_256[:, :2] * np.sqrt(lat_v + obs_v / float(batch_reps))[None, :]
        contrasts = z_out_256[:, 2:] * np.sqrt(obs_v / 2.0)[None, :]

        round_to_local_d: dict[bytes, int] = {}
        for loc_i in range(m_obs):
            rk = np.round(elig_cands_q[loc_i], 5).tobytes()
            if rk not in round_to_local_d:
                round_to_local_d[rk] = loc_i
        round_to_local_full = dict(round_to_local_d)
        if not is_revisit:
            rk_q = np.round(cands[q], 5).tobytes()
            if rk_q not in round_to_local_full:
                round_to_local_full[rk_q] = m_obs

        for s in range(n_outer):
            mf_s = pred.mean_obj[elig_idx_q] + float(z_out_256[s, 0]) * tf_q
            mg_s = pred.mean_con[elig_idx_q] + float(z_out_256[s, 1]) * tg_q
            mf_sq_s = float(mf_s[0])
            ref_s = 0.5 * (obs_anchor_val + mf_sq_s)

            samps_s = np.stack([centers[s] + contrasts[s], centers[s] - contrasts[s]], axis=0)
            rec_new = SufficientStatsRecord.from_samples(cands[q], samps_s, exps[q])

            if is_revisit:
                pos_q = base_pos_by_cand[q]
                recs_d = list(elig_base_recs)
                recs_d[pos_q] = recs_d[pos_q].merge(rec_new)
                recs_full = recs_d
            else:
                recs_d = elig_base_recs
                recs_full = [*elig_base_recs, rec_new]

            sl_d = build_feasible_aware_shortlist(
                candidate_pool=base_cands, mean_obj=mf_s[:m_obs], mean_con=mg_s[:m_obs],
                std_con=std_g_post[:m_obs], ucb_aleat_var_con=av_ucb_g[base_arr], r_perp_con=rp_g[base_arr],
                initial_records=recs_d, alpha=alpha, nomination_mode="cascade_bellman_3slot",
                reference_obj=ref_s, familywise_delta=0.05,
            )
            loc_d = tuple(round_to_local_d[np.round(x, 5).tobytes()] for _, x in sl_d)
            grd_d = len(sl_d) >= 2 and sl_d[1][0] in {"empirical_feasible_incumbent", "safe_initial_incumbent"}

            if is_revisit:
                loc_full, grd_full = loc_d, grd_d
            else:
                sl_full = build_feasible_aware_shortlist(
                    candidate_pool=elig_cands_q, mean_obj=mf_s, mean_con=mg_s,
                    std_con=std_g_post, ucb_aleat_var_con=av_ucb_g[elig_idx_q], r_perp_con=rp_g[elig_idx_q],
                    initial_records=recs_full, alpha=alpha, nomination_mode="cascade_bellman_3slot",
                    reference_obj=ref_s, familywise_delta=0.05,
                )
                loc_full = tuple(round_to_local_full[np.round(x, 5).tobytes()] for _, x in sl_full)
                grd_full = len(sl_full) >= 2 and sl_full[1][0] in {"empirical_feasible_incumbent", "safe_initial_incumbent"}

            local_cache: dict[tuple[tuple[int, ...], bool], int] = {}

            def _reg(r_loc: tuple[int, ...], g_fl: bool) -> int:
                key_r = (r_loc, bool(g_fl))
                if key_r in local_cache:
                    return local_cache[key_r]
                pad_loc = np.array([
                    r_loc[0],
                    r_loc[1] if len(r_loc) >= 2 else r_loc[0],
                    r_loc[2] if len(r_loc) >= 3 else r_loc[0],
                ], dtype=int)
                glob_pad = elig_idx_q[pad_loc]
                idx_b = len(eval_sq_list)
                eval_sq_list.append(mf_sq_s)
                eval_mu_f.append(mf_s[pad_loc])
                eval_cov_f.append(cov_f_post[np.ix_(pad_loc, pad_loc)])
                eval_av_f.append(av_f[glob_pad])
                eval_mu_g.append(mg_s[pad_loc])
                eval_cov_g.append(cov_g_post_eff[np.ix_(pad_loc, pad_loc)])
                eval_sig_g.append(sig_ucb_g[glob_pad])
                eval_nslots.append(len(r_loc))
                eval_guard.append(bool(g_fl))
                local_cache[key_r] = idx_b
                return idx_b

            curr_loc_tup = tuple(int(v) for v in curr_local_pad[:curr_len])
            slot_ptr[q, s, 0] = _reg(curr_loc_tup, curr_guard)
            slot_ptr[q, s, 1] = _reg(loc_d, grd_d)
            slot_ptr[q, s, 2] = slot_ptr[q, s, 1] if is_revisit else _reg(loc_full, grd_full)

    u_batch = _eval_rosters_ref(
        np.asarray(eval_sq_list, dtype=float),
        np.asarray(eval_mu_f, dtype=float),
        np.asarray(eval_cov_f, dtype=float),
        np.asarray(eval_av_f, dtype=float),
        np.asarray(eval_mu_g, dtype=float),
        np.asarray(eval_cov_g, dtype=float),
        np.asarray(eval_sig_g, dtype=float),
        np.asarray(eval_nslots, dtype=int),
        np.asarray(eval_guard, dtype=bool),
        z_in_1024,
    )
    ref_wall_s = float(time.perf_counter() - t_ref0)

    voi_menu_ref = np.zeros(m_cand, dtype=float)
    delta_term_ref = np.zeros(m_cand, dtype=float)

    for q in range(m_cand):
        ptrs = slot_ptr[q]
        if np.all(ptrs == -1):
            continue
        u_curr_s = u_batch[ptrs[:, 0]]
        u_ref_star = u_curr_s if curr_active else np.zeros_like(u_curr_s)
        u_d = u_batch[ptrs[:, 1]]
        u_full = u_batch[ptrs[:, 2]]

        m0_max = np.maximum(0.0, np.maximum(u_ref_star, u_d))
        m1_max = np.maximum(m0_max, u_full)
        voi_menu_ref[q] = float(np.mean(np.maximum(m1_max - u_ref_star, 0.0)))
        # Exact actual terminal rule: single nomination `u_full` vs current nomination `u_curr_s`
        delta_term_ref[q] = float(np.mean(u_full - u_curr_s))

    q_menu_opt = int(np.argmax(voi_menu_ref))
    q_term_opt = int(np.argmax(delta_term_ref))
    contenders_16_mask = s_fast16 > 0.0

    # Spearman rank correlation across all 160 candidates between fast-all160 (32x128) and ref-all160 (256x1024)
    rho_quad = float(spearmanr(s_fast_all160, voi_menu_ref).statistic) if np.std(voi_menu_ref) > 1e-12 else 1.0
    rho_menu_vs_term = float(spearmanr(voi_menu_ref, delta_term_ref).statistic) if (np.std(voi_menu_ref) > 1e-12 and np.std(delta_term_ref) > 1e-12) else 1.0

    max_menu = max(float(voi_menu_ref[q_menu_opt]), 1e-12)
    max_term = float(delta_term_ref[q_term_opt])

    return {
        "n_unique_evaluated": len(obs_recs),
        "fast16_wall_ms": fast16_ms,
        "ref_all160_wall_sec": ref_wall_s,
        "q_eckg16": q_eckg16,
        "q_single": q_single,
        "q_feas": q_feas,
        "q_menu_ref_opt": q_menu_opt,
        "q_term_ref_opt": q_term_opt,
        "menu_opt_in_stage_a_top16": bool(contenders_16_mask[q_menu_opt]) if np.any(contenders_16_mask) else True,
        "term_opt_in_stage_a_top16": bool(contenders_16_mask[q_term_opt]) if np.any(contenders_16_mask) else True,
        "quadrature_spearman_32x128_vs_256x1024": 0.0 if np.isnan(rho_quad) else rho_quad,
        "menu_vs_actual_terminal_spearman": 0.0 if np.isnan(rho_menu_vs_term) else rho_menu_vs_term,
        "menu_ref_max": float(voi_menu_ref[q_menu_opt]),
        "menu_ref_at_eckg16": float(voi_menu_ref[q_eckg16]),
        "menu_ref_at_single": float(voi_menu_ref[q_single]),
        "menu_ref_at_feas": float(voi_menu_ref[q_feas]),
        "menu_ref_rel_regret_eckg16": float((voi_menu_ref[q_menu_opt] - voi_menu_ref[q_eckg16]) / max_menu),
        "actual_term_ref_max": max_term,
        "actual_term_ref_at_eckg16": float(delta_term_ref[q_eckg16]),
        "actual_term_ref_at_single": float(delta_term_ref[q_single]),
        "actual_term_ref_at_feas": float(delta_term_ref[q_feas]),
    }


def _run_one_task(manifest: dict[str, Any], out_dir: Path, jb: dict[str, Any]) -> dict[str, Any]:
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    ipath = out_dir / "items" / f"{jb['job_id']}.json"
    if ipath.is_file():
        return {"job_id": jb["job_id"], "status": "cached"}

    fam_name = jb["family"]
    t_idx = int(jb["target_task_idx"])
    fam_spec = next(f for f in manifest["families"] if f["name"] == fam_name)
    prob = build_family_tasks(fam_spec, "target")[t_idx]

    src_campaign = SC_ROOT / "paper_artifacts" / manifest["reuse_source_campaign"]
    src_fam = fam_spec.get("reuse_archive_from_family", fam_name)
    with (src_campaign / "source_bundles" / src_fam / "prior_default.pkl").open("rb") as fh:
        prior = pickle.load(fh)

    # Exact closed-loop stream-1 seeds from v2_or_eckg_sure_confirm_v1
    rep_seed = int(fam_spec["target_eval_seeds"][0])
    cell_seed = rep_seed + 10_007 * t_idx
    rng_init = np.random.default_rng(cell_seed)
    rng_search = np.random.default_rng(cell_seed + 100_003)

    n0 = int(fam_spec.get("n0_init", 8))
    b_reps = int(fam_spec.get("batch_reps", 2))
    p_size = int(fam_spec.get("pool_size", 160))

    model = TemperedHierarchicalHVDModel(
        prior, mode="tempered_sure_release",
        use_source_basis=True, use_source_coef_prior=True, use_source_hvd_prior=True,
        n0_freeze=n0,
    )
    init_pts = build_balanced_warmstart_points(prob, prior, n0, rng_init)
    init_recs: list[SufficientStatsRecord] = []
    for x0 in init_pts:
        exp0 = prob.nominal_exposure(x0)
        samp0 = prob.simulate_replications(x0, b_reps, rng_init, split="search")
        rec0 = SufficientStatsRecord.from_samples(x0, samp0, exp0)
        init_recs.append(rec0)
        model.observe(rec0)
    sq_est_obj = float(init_recs[0].mean_obj)

    snap_step0 = None
    snap_step59 = None
    for step_idx in range(60):
        enable_tpl = bool(not model.is_source_released_)
        cands, exps = generate_dynamic_candidate_pool(
            prob, model, prior, rng_search, pool_size=p_size, enable_source_templates=enable_tpl,
        )
        pred = model.predict_joint(cands, exps)
        sq_ref = 0.5 * sq_est_obj + 0.5 * float(pred.mean_obj[0])

        if step_idx == 0:
            snap_step0 = _evaluate_frozen_state_all160(
                pred, model, cands, exps, alpha=prob.alpha, batch_reps=b_reps,
                sq_ref=sq_ref, ref_seed=910000 + 101 * t_idx,
            )
        elif step_idx == 59:
            snap_step59 = _evaluate_frozen_state_all160(
                pred, model, cands, exps, alpha=prob.alpha, batch_reps=b_reps,
                sq_ref=sq_ref, ref_seed=920000 + 101 * t_idx,
            )
            break

        _, step_diag = score_eligible_cascade_kg(
            pred, model, cands, exps, alpha=prob.alpha, batch_reps=b_reps,
            status_quo_obj=sq_ref, nomination_mode="cascade_bellman_3slot",
            return_diagnostics=True,
        )
        chosen_idx = int(step_diag["chosen_query"])
        samp = prob.simulate_replications(cands[chosen_idx], b_reps, rng_search, split="search")
        model.observe(SufficientStatsRecord.from_samples(cands[chosen_idx], samp, exps[chosen_idx]))

    payload = {
        "job_id": jb["job_id"],
        "family": fam_name,
        "target_task_idx": t_idx,
        "task_id": prob.task_id,
        "cell_seed": cell_seed,
        "step_0": snap_step0,
        "step_59": snap_step59,
    }
    _write_json(ipath, payload)
    return {"job_id": jb["job_id"], "status": "ok"}


def describe(manifest_path: Path, out_dir: Path) -> dict[str, Any]:
    jobs = []
    for fam_name in ("Queueing_CrossHorizon", "Traffic_Ingolstadt"):
        for ti in range(6):
            jobs.append({
                "job_id": f"eckgref_{fam_name}_t{ti:02d}",
                "family": fam_name,
                "target_task_idx": ti,
            })
    desc = {
        "diagnostic_name": "v2_or_eckg_reference_validation_v1",
        "manifest_relpath": str(manifest_path.resolve().relative_to(SC_ROOT)),
        "execution_code_sha256": compute_v2_or_code_sha256(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_jobs": len(jobs),
        "n_states": 2 * len(jobs),
        "jobs": jobs,
    }
    _write_json(out_dir / "jobs.json", desc)
    return desc


def run_worker(manifest_path: Path, out_dir: Path, start: int, end: int, workers: int) -> None:
    node = socket.gethostname().split(".")[0]
    portable_ok = os.environ.get("GPFTA_PORTABLE_REVIEW") == "1" or (REPO_ROOT / ".gpfta_portable_review").is_file()
    if not portable_ok and node not in DEFAULT_CPU_NODES:
        raise RuntimeError(f"Reference validation worker must run on node001-node006, got {node!r}")
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
        raise RuntimeError(f"{len(failures)} reference validation jobs failed: {failures[:2]}")


def summarize(out_dir: Path, summary_path: Path) -> dict[str, Any]:
    desc = json.loads((out_dir / "jobs.json").read_text(encoding="utf-8"))
    items = [json.loads((out_dir / "items" / f"{jb['job_id']}.json").read_text(encoding="utf-8")) for jb in desc["jobs"]]
    fam_out: dict[str, Any] = {}
    for fam in ("Queueing_CrossHorizon", "Traffic_Ingolstadt"):
        sub = [it for it in items if it["family"] == fam]
        stage_agg = {}
        for st_key in ("step_0", "step_59"):
            st = [it[st_key] for it in sub]
            stage_agg[st_key] = {
                "n_states": len(st),
                "mean_n_unique_evaluated": float(np.mean([x["n_unique_evaluated"] for x in st])),
                "fast16_mean_wall_ms": float(np.mean([x["fast16_wall_ms"] for x in st])),
                "ref_all160_mean_wall_sec": float(np.mean([x["ref_all160_wall_sec"] for x in st])),
                "stage_a_top16_contains_menu_opt_rate": float(np.mean([1.0 if x["menu_opt_in_stage_a_top16"] else 0.0 for x in st])),
                "stage_a_top16_contains_term_opt_rate": float(np.mean([1.0 if x["term_opt_in_stage_a_top16"] else 0.0 for x in st])),
                "exact_match_eckg16_vs_menu_opt_rate": float(np.mean([1.0 if x["q_eckg16"] == x["q_menu_ref_opt"] else 0.0 for x in st])),
                "quadrature_spearman_mean": float(np.mean([x["quadrature_spearman_32x128_vs_256x1024"] for x in st])),
                "menu_vs_actual_terminal_spearman_mean": float(np.mean([x["menu_vs_actual_terminal_spearman"] for x in st])),
                "menu_ref_rel_regret_eckg16_mean": float(np.mean([x["menu_ref_rel_regret_eckg16"] for x in st])),
                "menu_ref_max_mean_x1e3": 1000.0 * float(np.mean([x["menu_ref_max"] for x in st])),
                "menu_ref_eckg16_mean_x1e3": 1000.0 * float(np.mean([x["menu_ref_at_eckg16"] for x in st])),
                "menu_ref_single_mean_x1e3": 1000.0 * float(np.mean([x["menu_ref_at_single"] for x in st])),
                "menu_ref_feas_mean_x1e3": 1000.0 * float(np.mean([x["menu_ref_at_feas"] for x in st])),
                "actual_term_ref_max_mean_x1e3": 1000.0 * float(np.mean([x["actual_term_ref_max"] for x in st])),
                "actual_term_ref_eckg16_mean_x1e3": 1000.0 * float(np.mean([x["actual_term_ref_at_eckg16"] for x in st])),
                "actual_term_ref_single_mean_x1e3": 1000.0 * float(np.mean([x["actual_term_ref_at_single"] for x in st])),
                "actual_term_ref_feas_mean_x1e3": 1000.0 * float(np.mean([x["actual_term_ref_at_feas"] for x in st])),
                "actual_term_eckg16_beats_or_ties_single_count": int(
                    sum(1 for x in st if x["actual_term_ref_at_eckg16"] >= x["actual_term_ref_at_single"] - 1e-12)
                ),
                "actual_term_eckg16_beats_or_ties_feas_count": int(
                    sum(1 for x in st if x["actual_term_ref_at_eckg16"] >= x["actual_term_ref_at_feas"] - 1e-12)
                ),
            }
        fam_out[fam] = stage_agg
    readout = {
        "diagnostic_name": desc["diagnostic_name"],
        "execution_code_sha256": desc["execution_code_sha256"],
        "outer_sobol_points": 256,
        "inner_sobol_points": 1024,
        "candidates_scored_per_state": 160,
        "total_states": 2 * len(items),
        "families": fam_out,
    }
    _write_json(summary_path, readout)
    return readout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("describe", "worker", "summarize", "submit"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", type=Path, default=SC_ROOT / "v2_or/manifests/v2_or_eckg_sure_confirm_v1.json")
        p.add_argument("--out-dir", type=Path, default=SC_ROOT / "paper_artifacts/v2_or_eckg_sure_confirm_v1/eckg_ref_val")
        if name == "worker":
            p.add_argument("--start", type=int, required=True)
            p.add_argument("--end", type=int, required=True)
            p.add_argument("--workers", type=int, default=1)
        elif name == "summarize":
            p.add_argument("--summary-path", type=Path, default=SC_ROOT / "paper_artifacts/v2_or_eckg_sure_confirm_v1/eckg_reference_validation_readout.json")
        elif name == "submit":
            p.add_argument("--scheduler", type=Path, default=Path(os.environ.get("GPFTA_CLUSTER_SCHEDULER", "scheduler.py")))
            p.add_argument("--remote-cwd", default=os.environ.get("GPFTA_REMOTE_CWD", "."))
            p.add_argument("--remote-python", default=os.environ.get("GPFTA_REMOTE_PYTHON", sys.executable))
            p.add_argument("--ram-mb", type=int, default=16384)
    args = parser.parse_args()

    if args.action == "describe":
        d = describe(args.manifest, args.out_dir)
        print(json.dumps({"n_jobs": d["n_jobs"], "n_states": d["n_states"]}))
    elif args.action == "worker":
        run_worker(args.manifest, args.out_dir, args.start, args.end, args.workers)
    elif args.action == "summarize":
        r = summarize(args.out_dir, args.summary_path)
        print(json.dumps({"status": "ok", "total_states": r["total_states"]}))
    elif args.action == "submit":
        desc = json.loads((args.out_dir / "jobs.json").read_text(encoding="utf-8"))
        nodes = parse_cpu_nodes(",".join(DEFAULT_CPU_NODES))
        rel_man = args.manifest.resolve().relative_to(SC_ROOT)
        rel_out = args.out_dir.resolve().relative_to(SC_ROOT)
        cmd = shlex.join([
            "env", "OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
            "PYTHONPATH=.:../scripts",
            args.remote_python, "-u", "../scripts/run_eckg_reference_validation.py",
            "worker", "--manifest", str(rel_man), "--out-dir", str(rel_out),
            "--start", "{start}", "--end", "{end}", "--workers", "{workers}",
        ])
        s_cmd = [
            sys.executable, str(args.scheduler), "submit-cpu-batch",
            "--items", str(desc["n_jobs"]),
            "--cmd-template", cmd,
            "--cwd", args.remote_cwd,
            "--nodes", ",".join(nodes),
            "--signature", "gp_fta_v2/eckg_ref_val/{start}-{end}",
            "--description", "v2_or eckg all160 reference validation {node} [{start},{end})",
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
