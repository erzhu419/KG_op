#!/usr/bin/env python3
"""Cluster diagnostic: Multi-scramble QMC confidence intervals on frozen search states,
Stage-A screening miss anatomy, and Ungredda & Branke (2024, Sec. 5) cEI comparison.

Evaluates 24 independent scrambled 256 x 1,024 Sobol replications across key contenders
(`q_eckg16`, `q_single`, `q_feas`, `q_ub_cei`, `q_gate_cei`, `q_menu_opt`, `q_term_opt`,
`q_screen_term_opt`) on all 24 realized closed-loop states (`Queueing` and `Traffic`,
`t00`..`t05`, `step_0` and `step_59`), plus archive/target clipping trigger rates.
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
import subprocess
import sys
import time
from typing import Any

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import norm, qmc, t as student_t

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
    score_constrained_ei,
    score_eligible_cascade_kg,
    score_gate_power_vkg,
)
from v2_or.contracts import SufficientStatsRecord
from v2_or.dispatch.campaign_runner import build_family_tasks, compute_v2_or_code_sha256
from v2_or.models import TemperedHierarchicalHVDModel
from v2_or.priors.debiased_spectral_prior import _build_raw_dictionary
from v2_or.verification.gated_verifier_v2 import (
    build_feasible_aware_shortlist,
    exact_binomial_critical_k,
    posterior_binomial_gate_power,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _make_ref_qmc(seed: int, m_out_pow2: int = 7, m_in_pow2: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Build independent antithetic 256-point outer 4D Sobol and 1024-point inner 3D Sobol grids."""
    eng_out = qmc.Sobol(d=4, scramble=True, seed=int(seed))
    u_out = np.clip(eng_out.random_base2(m=m_out_pow2), 1e-6, 1.0 - 1e-6)
    z_out = np.vstack([ndtri(u_out), -ndtri(u_out)])  # (256, 4)

    eng_in = qmc.Sobol(d=3, scramble=True, seed=int(seed) + 17)
    u_in = np.clip(eng_in.random_base2(m=m_in_pow2 - 1), 1e-6, 1.0 - 1e-6)
    z_in = np.vstack([ndtri(u_in), -ndtri(u_in)])  # (1024, 3)
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
    """Evaluate expected 3-slot cascade net saving in chunks using 1,024 inner draws."""
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
        s_obs = np.sqrt(
            np.maximum(v_diff + avf[:, 0] / float(_SLOT_SIZES[0]) + avf[:, 1] / float(_SLOT_SIZES[1]), 1e-16)
        )
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


def _score_ub_chance_cei(
    pred: Any,
    elig_base_idx: list[int],
    *,
    alpha: float = 0.05,
    sq_ref: float,
) -> np.ndarray:
    """Ungredda & Branke (2024, Sec. 5) final-step constrained EI (`EI * P_feas`)
    benchmarked against the best already-sampled feasible solution in `D_n`."""
    av_ucb = np.maximum(pred.ucb_aleatoric_var_con, 1e-8)
    s_con = np.sqrt(np.maximum(pred.std_con ** 2 + 0.25 * np.maximum(pred.r_perp_con, 0.0), 1e-12))
    feas_lim = float(norm.ppf(alpha)) * np.sqrt(av_ucb)
    p_feas = ndtr((feas_lim - pred.mean_con) / s_con)

    base_arr = np.asarray(elig_base_idx, dtype=int)
    feas_sampled = base_arr[p_feas[base_arr] >= 0.50]
    if feas_sampled.size > 0:
        f_best = float(np.min(pred.mean_obj[feas_sampled]))
    else:
        f_best = float(sq_ref)
    s_f = np.maximum(pred.std_obj, 1e-6)
    u = (f_best - pred.mean_obj) / s_f
    ei = s_f * (u * ndtr(u) + norm.pdf(u))
    return ei * p_feas


def _score_subset_single_scramble(
    pred: Any,
    model: TemperedHierarchicalHVDModel,
    cands: list[np.ndarray],
    exps: list[Any],
    key_cands: list[int],
    *,
    alpha: float,
    batch_reps: int,
    sq_ref: float,
    scramble_seed: int,
) -> tuple[dict[int, float], dict[int, float], dict[int, dict[str, float]]]:
    """Evaluate `key_cands` on one independent 256 x 1,024 scramble, returning menu VOI,
    single-nomination gain, and slot-change frequencies relative to current shortlist."""
    z_out_256, z_in_1024 = _make_ref_qmc(scramble_seed, m_out_pow2=7, m_in_pow2=10)
    n_outer = int(z_out_256.shape[0])

    obs_recs = model.unique_records
    key_to_cand = {model._policy_key(x): idx for idx, x in enumerate(cands)}
    elig_base_idx: list[int] = []
    elig_base_recs: list[SufficientStatsRecord] = []
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
    curr_local_pad = np.array([base_pos_by_cand[int(c)] for c in curr_pad], dtype=int)
    curr_local_tup = tuple(int(v) for v in curr_local_pad[:curr_len])

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
    slot_ptr = np.full((len(key_cands), n_outer, 3), -1, dtype=int)
    slot_change_counts: dict[int, dict[str, float]] = {}

    for k_pos, q in enumerate(key_cands):
        is_revisit = q in base_pos_by_cand
        elig_idx_q = base_arr if is_revisit else np.append(base_arr, q)
        elig_cands_q = base_cands if is_revisit else [*base_cands, cands[q]]
        tf_q = tilde_f_all[q, elig_idx_q]
        tg_q = tilde_g_all[q, elig_idx_q]
        if is_revisit and np.max(np.abs(tf_q)) < 1e-14 and np.max(np.abs(tg_q)) < 1e-14:
            slot_change_counts[q] = {"slot0_chg": 0.0, "slot1_chg": 0.0, "slot2_chg": 0.0}
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

        c_s0 = c_s1 = c_s2 = 0
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
                candidate_pool=base_cands,
                mean_obj=mf_s[:m_obs],
                mean_con=mg_s[:m_obs],
                std_con=std_g_post[:m_obs],
                ucb_aleat_var_con=av_ucb_g[base_arr],
                r_perp_con=rp_g[base_arr],
                initial_records=recs_d,
                alpha=alpha,
                nomination_mode="cascade_bellman_3slot",
                reference_obj=ref_s,
                familywise_delta=0.05,
            )
            loc_d = tuple(round_to_local_d[np.round(x, 5).tobytes()] for _, x in sl_d)
            grd_d = len(sl_d) >= 2 and sl_d[1][0] in {"empirical_feasible_incumbent", "safe_initial_incumbent"}

            if is_revisit:
                loc_full, grd_full = loc_d, grd_d
            else:
                sl_full = build_feasible_aware_shortlist(
                    candidate_pool=elig_cands_q,
                    mean_obj=mf_s,
                    mean_con=mg_s,
                    std_con=std_g_post,
                    ucb_aleat_var_con=av_ucb_g[elig_idx_q],
                    r_perp_con=rp_g[elig_idx_q],
                    initial_records=recs_full,
                    alpha=alpha,
                    nomination_mode="cascade_bellman_3slot",
                    reference_obj=ref_s,
                    familywise_delta=0.05,
                )
                loc_full = tuple(round_to_local_full[np.round(x, 5).tobytes()] for _, x in sl_full)
                grd_full = len(sl_full) >= 2 and sl_full[1][0] in {"empirical_feasible_incumbent", "safe_initial_incumbent"}

            if loc_full[0] != curr_local_tup[0]:
                c_s0 += 1
            if (len(loc_full) >= 2) != (len(curr_local_tup) >= 2) or (
                len(loc_full) >= 2 and len(curr_local_tup) >= 2 and loc_full[1] != curr_local_tup[1]
            ):
                c_s1 += 1
            if (len(loc_full) >= 3) != (len(curr_local_tup) >= 3) or (
                len(loc_full) >= 3 and len(curr_local_tup) >= 3 and loc_full[2] != curr_local_tup[2]
            ):
                c_s2 += 1

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

            slot_ptr[k_pos, s, 0] = _reg(curr_local_tup, curr_guard)
            slot_ptr[k_pos, s, 1] = _reg(loc_d, grd_d)
            slot_ptr[k_pos, s, 2] = slot_ptr[k_pos, s, 1] if is_revisit else _reg(loc_full, grd_full)

        slot_change_counts[q] = {
            "slot0_chg": c_s0 / float(n_outer),
            "slot1_chg": c_s1 / float(n_outer),
            "slot2_chg": c_s2 / float(n_outer),
        }

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
    ) if eval_sq_list else np.zeros(0, dtype=float)

    menu_out: dict[int, float] = {}
    term_out: dict[int, float] = {}
    for k_pos, q in enumerate(key_cands):
        ptrs = slot_ptr[k_pos]
        if np.all(ptrs < 0):
            menu_out[q] = 0.0
            term_out[q] = 0.0
            continue
        u_curr_s = np.where(ptrs[:, 0] >= 0, u_batch[np.maximum(ptrs[:, 0], 0)], 0.0)
        u_ref_star = u_curr_s if curr_active else np.zeros_like(u_curr_s)
        u_d = np.where(ptrs[:, 1] >= 0, u_batch[np.maximum(ptrs[:, 1], 0)], 0.0)
        u_full = np.where(ptrs[:, 2] >= 0, u_batch[np.maximum(ptrs[:, 2], 0)], 0.0)
        m0_max = np.maximum(0.0, np.maximum(u_ref_star, u_d))
        m1_max = np.maximum(m0_max, u_full)
        menu_out[q] = float(np.mean(np.maximum(m1_max - u_ref_star, 0.0)))
        term_out[q] = float(np.mean(u_full - u_curr_s))
    return menu_out, term_out, slot_change_counts


def _stage_a_anatomy(
    pred: Any,
    model: TemperedHierarchicalHVDModel,
    cands: list[np.ndarray],
    saved_state: dict[str, Any],
    *,
    alpha: float,
    batch_reps: int,
    sq_ref: float,
) -> dict[str, Any]:
    """Inspect Stage-A ranking of `q_term_opt` within revisits vs new candidates."""
    obs_recs = model.unique_records
    key_to_cand = {model._policy_key(x): idx for idx, x in enumerate(cands)}
    elig_base_idx = [key_to_cand[model._policy_key(r.x_norm)] for r in obs_recs if model._policy_key(r.x_norm) in key_to_cand]
    base_set = set(elig_base_idx)

    # Recompute Stage-A proxy score vector exactly as in verification_kg.py
    base_arr = np.asarray(elig_base_idx, dtype=int)
    av_f = np.maximum(pred.aleatoric_var_obj, 1e-10)
    av_g = np.maximum(pred.aleatoric_var_con, 1e-10)
    av_ucb_g = np.maximum(pred.ucb_aleatoric_var_con, 1e-8)
    sig_ucb_g = np.sqrt(av_ucb_g)
    rp_g = np.maximum(pred.r_perp_con, 0.0)
    cov_g_eff = pred.cov_con + np.diag(0.25 * rp_g)

    p0_all = posterior_binomial_gate_power(
        pred.mean_con, pred.std_con, av_ucb_g, n_samples=112, alpha=float(alpha), delta_slot=0.025, r_perp_con=rp_g
    )
    p2_all = posterior_binomial_gate_power(
        pred.mean_con, pred.std_con, av_ucb_g, n_samples=140, alpha=float(alpha), delta_slot=0.010, r_perp_con=rp_g
    )
    # Use saved fast_menu_scores & reference_nomination_scores
    term_vec = np.asarray(saved_state["reference_nomination_scores"], dtype=float)
    menu_vec = np.asarray(saved_state["reference_menu_scores"], dtype=float)
    screen = list(saved_state["screened_candidate_ids"])
    q_term_opt = int(np.argmax(term_vec))
    q_menu_opt = int(np.argmax(menu_vec))
    q_screen_term_opt = int(screen[int(np.argmax(term_vec[screen]))])

    is_revisit = bool(q_term_opt in base_set)
    return {
        "q_term_opt": q_term_opt,
        "q_menu_opt": q_menu_opt,
        "q_screen_term_opt": q_screen_term_opt,
        "term_opt_is_revisit": is_revisit,
        "term_opt_in_screen16": bool(q_term_opt in screen),
        "term_opt_p0_gate": float(p0_all[q_term_opt]),
        "term_opt_p2_gate": float(p2_all[q_term_opt]),
        "term_opt_mean_obj": float(pred.mean_obj[q_term_opt]),
        "sq_ref_obj": float(sq_ref),
    }


def _evaluate_state_multiscramble(
    pred: Any,
    model: TemperedHierarchicalHVDModel,
    cands: list[np.ndarray],
    exps: list[Any],
    saved_state: dict[str, Any],
    *,
    alpha: float,
    batch_reps: int,
    sq_ref: float,
    state_uid: int,
    n_scrambles: int = 24,
) -> dict[str, Any]:
    obs_recs = model.unique_records
    key_to_cand = {model._policy_key(x): idx for idx, x in enumerate(cands)}
    elig_base_idx = [key_to_cand[model._policy_key(r.x_norm)] for r in obs_recs if model._policy_key(r.x_norm) in key_to_cand]

    _, diag16 = score_eligible_cascade_kg(
        pred, model, cands, exps, alpha=alpha, batch_reps=batch_reps,
        status_quo_obj=sq_ref, nomination_mode="cascade_bellman_3slot",
        top_k_contenders=16, return_diagnostics=True,
    )
    s_single = score_gate_power_vkg(pred, alpha=alpha, batch_reps=batch_reps, status_quo_obj=sq_ref)
    s_feas = score_chance_feasibility_kg(pred, alpha=alpha, batch_reps=batch_reps, status_quo_obj=sq_ref)
    s_ub_cei = _score_ub_chance_cei(pred, elig_base_idx, alpha=alpha, sq_ref=sq_ref)
    s_gate_cei = score_constrained_ei(pred, alpha=alpha)

    q_eckg16 = int(diag16["chosen_query"])
    q_single = int(np.argmax(s_single))
    q_feas = int(np.argmax(s_feas))
    q_ub_cei = int(np.argmax(s_ub_cei))
    q_gate_cei = int(np.argmax(s_gate_cei))

    if (q_eckg16, q_single, q_feas) != (saved_state["q_eckg16"], saved_state["q_single"], saved_state["q_feas"]):
        raise RuntimeError("Reconstructed state diverged from saved eckg_reference_states.json")

    anatomy = _stage_a_anatomy(pred, model, cands, saved_state, alpha=alpha, batch_reps=batch_reps, sq_ref=sq_ref)
    q_menu_opt = anatomy["q_menu_opt"]
    q_term_opt = anatomy["q_term_opt"]
    q_screen_term_opt = anatomy["q_screen_term_opt"]

    key_cands = sorted({
        q_eckg16, q_single, q_feas, q_ub_cei, q_gate_cei, q_menu_opt, q_term_opt, q_screen_term_opt
    })

    menu_draws = {q: [] for q in key_cands}
    term_draws = {q: [] for q in key_cands}
    slot_chg_acc = {q: {"slot0_chg": [], "slot1_chg": [], "slot2_chg": []} for q in key_cands}

    for r_i in range(int(n_scrambles)):
        sc_seed = 9_500_000 + 10_007 * int(state_uid) + 131 * r_i
        m_d, t_d, chg_d = _score_subset_single_scramble(
            pred, model, cands, exps, key_cands,
            alpha=alpha, batch_reps=batch_reps, sq_ref=sq_ref, scramble_seed=sc_seed,
        )
        for q in key_cands:
            menu_draws[q].append(m_d[q])
            term_draws[q].append(t_d[q])
            for sk in ("slot0_chg", "slot1_chg", "slot2_chg"):
                slot_chg_acc[q][sk].append(chg_d[q][sk])

    # Target candidate pool clipping trigger rates
    psi_mat = np.array([e.psi for e in exps], dtype=float)
    psi_std_raw = (psi_mat - model.prior.psi_mean_[None, :]) / model.prior.psi_std_[None, :]
    clip4_cand_rate = float(np.mean(np.abs(psi_std_raw) > 4.0))
    raw_dict = _build_raw_dictionary(psi_std_raw)
    dict_lp_c = raw_dict[:, model.prior.lowpass_mask_] - model.prior.dict_center_[None, :]
    orth_unclipped = dict_lp_c @ model.prior.whitening_mat_.T
    clip5_cand_rate = float(np.mean(np.abs(orth_unclipped) > 5.0))

    return {
        "n_scrambles": int(n_scrambles),
        "queries": {
            "eckg16": q_eckg16,
            "single": q_single,
            "feas": q_feas,
            "ub_cei": q_ub_cei,
            "gate_cei": q_gate_cei,
            "menu_opt_ref1": q_menu_opt,
            "term_opt_ref1": q_term_opt,
            "screen_term_opt_ref1": q_screen_term_opt,
        },
        "query_is_revisit": {
            name: bool(q in set(elig_base_idx))
            for name, q in (
                ("eckg16", q_eckg16),
                ("single", q_single),
                ("feas", q_feas),
                ("ub_cei", q_ub_cei),
                ("gate_cei", q_gate_cei),
                ("term_opt_ref1", q_term_opt),
            )
        },
        "stage_a_anatomy": anatomy,
        "key_candidates": key_cands,
        "menu_voi_scrambles": {str(q): menu_draws[q] for q in key_cands},
        "term_gain_scrambles": {str(q): term_draws[q] for q in key_cands},
        "slot_change_rates": {
            str(q): {sk: float(np.mean(slot_chg_acc[q][sk])) for sk in ("slot0_chg", "slot1_chg", "slot2_chg")}
            for q in key_cands
        },
        "target_pool_clip4_coord_rate": clip4_cand_rate,
        "target_pool_clip5_coord_rate": clip5_cand_rate,
    }


def _audit_source_archive_clipping(src_campaign: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for fam in ("Inventory_CrossHorizon", "Queueing_CrossHorizon", "OPSD_EuroGrid", "Traffic_Ingolstadt"):
        bdir = src_campaign / "source_bundles" / fam
        p_path = bdir / "prior_default.pkl"
        a_path = bdir / "archive.json"
        if not (p_path.is_file() and a_path.is_file()):
            continue
        with p_path.open("rb") as fh:
            prior = pickle.load(fh)
        arch = json.loads(a_path.read_text(encoding="utf-8"))
        recs = arch["records"] if isinstance(arch, dict) and "records" in arch else arch
        psi_mat = np.array([r["psi"] for r in recs], dtype=float)
        psi_std_raw = (psi_mat - prior.psi_mean_[None, :]) / prior.psi_std_[None, :]
        raw_dict = _build_raw_dictionary(psi_std_raw)
        dict_lp_c = raw_dict[:, prior.lowpass_mask_] - prior.dict_center_[None, :]
        orth_unclipped = dict_lp_c @ prior.whitening_mat_.T
        out[fam] = {
            "n_records": int(psi_mat.shape[0]),
            "source_psi_clip4_coord_rate": float(np.mean(np.abs(psi_std_raw) > 4.0)),
            "source_psi_clip4_any_record_rate": float(np.mean(np.any(np.abs(psi_std_raw) > 4.0, axis=1))),
            "source_orth_would_clip5_coord_rate": float(np.mean(np.abs(orth_unclipped) > 5.0)),
            "source_orth_would_clip5_any_record_rate": float(np.mean(np.any(np.abs(orth_unclipped) > 5.0, axis=1))),
        }
    return out


def _run_one_job(manifest: dict[str, Any], saved_states_by_key: dict[tuple[str, int], dict], out_dir: Path, jb: dict[str, Any], n_scrambles: int) -> dict[str, Any]:
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
    fam_spec = next(f for f in manifest["families"] if f["name"] == fam_name)
    prob = build_family_tasks(fam_spec, "target")[t_idx]

    src_campaign = SC_ROOT / "paper_artifacts" / manifest["reuse_source_campaign"]
    src_fam = fam_spec.get("reuse_archive_from_family", fam_name)
    with (src_campaign / "source_bundles" / src_fam / "prior_default.pkl").open("rb") as fh:
        prior = pickle.load(fh)

    saved_task = saved_states_by_key[(fam_name, t_idx)]
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

    fam_offset = 0 if fam_name == "Queueing_CrossHorizon" else 100
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
            snap_step0 = _evaluate_state_multiscramble(
                pred, model, cands, exps, saved_task["step_0"],
                alpha=prob.alpha, batch_reps=b_reps, sq_ref=sq_ref,
                state_uid=fam_offset + 2 * t_idx, n_scrambles=n_scrambles,
            )
        elif step_idx == 59:
            snap_step59 = _evaluate_state_multiscramble(
                pred, model, cands, exps, saved_task["step_59"],
                alpha=prob.alpha, batch_reps=b_reps, sq_ref=sq_ref,
                state_uid=fam_offset + 2 * t_idx + 1, n_scrambles=n_scrambles,
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

    payload: dict[str, Any] = {
        "job_id": jb["job_id"],
        "family": fam_name,
        "target_task_idx": t_idx,
        "task_id": prob.task_id,
        "cell_seed": cell_seed,
        "diagnostic_search_calls": int(prob.search_calls),
        "diagnostic_verification_calls": int(prob.verification_calls),
        "diagnostic_audit_calls": int(prob.audit_calls),
        "worker_cpu_seconds": float(time.process_time() - t0_cpu),
        "worker_wall_seconds": float(time.perf_counter() - t0_wall),
        "step_0": snap_step0,
        "step_59": snap_step59,
    }
    if jb["job_id"] == "mc_cei_Queueing_CrossHorizon_t00":
        payload["source_archive_clipping"] = _audit_source_archive_clipping(src_campaign)
    _write_json(ipath, payload)
    return {"job_id": jb["job_id"], "status": "ok"}


def _ci95(arr: np.ndarray) -> tuple[float, float, float, float]:
    n = int(arr.size)
    m = float(np.mean(arr))
    se = float(np.std(arr, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    t_crit = float(student_t.ppf(0.975, df=max(n - 1, 1))) if n > 1 else 0.0
    return m, se, m - t_crit * se, m + t_crit * se


def describe(manifest_path: Path, out_dir: Path, n_scrambles: int = 24) -> dict[str, Any]:
    jobs = []
    for fam_name in ("Queueing_CrossHorizon", "Traffic_Ingolstadt"):
        for ti in range(6):
            jobs.append({
                "job_id": f"mc_cei_{fam_name}_t{ti:02d}",
                "family": fam_name,
                "target_task_idx": ti,
            })
    desc = {
        "diagnostic_name": "v2_or_eckg_multiscramble_cei_audit_v1",
        "manifest_relpath": str(manifest_path.resolve().relative_to(SC_ROOT)),
        "execution_code_sha256": compute_v2_or_code_sha256(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_scrambles_per_state": int(n_scrambles),
        "outer_sobol_points": 256,
        "inner_sobol_points": 1024,
        "n_jobs": len(jobs),
        "n_states": 2 * len(jobs),
        "jobs": jobs,
    }
    _write_json(out_dir / "jobs.json", desc)
    return desc


def _stabilize_floats(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _stabilize_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_stabilize_floats(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return 0.0 if abs(v) == 0.0 else float(f"{v:.12g}")
    return obj


def summarize(out_dir: Path, states_path: Path, summary_path: Path) -> dict[str, Any]:
    desc = json.loads((out_dir / "jobs.json").read_text(encoding="utf-8"))
    saved_bundle = json.loads(states_path.read_text(encoding="utf-8"))
    saved_map = {(t["family"], int(t["target_task_idx"])): t for t in saved_bundle["tasks"]}

    items = [json.loads((out_dir / "items" / f"{jb['job_id']}.json").read_text(encoding="utf-8")) for jb in desc["jobs"]]
    src_clip = next((it["source_archive_clipping"] for it in items if "source_archive_clipping" in it), {})

    families_out: dict[str, Any] = {}
    state_table: list[dict[str, Any]] = []
    neg_ec_count_24 = 0

    for fam_name in ("Queueing_CrossHorizon", "Traffic_Ingolstadt"):
        families_out[fam_name] = {}
        fam_items = [it for it in items if it["family"] == fam_name]
        for stg in ("step_0", "step_59"):
            ratios_single_ref = []
            ratios_mc_mean = []
            ec_means = []
            opt_means = []
            ub_cei_means = []
            gate_cei_means = []
            single_means = []
            feas_means = []
            ec_ge_ub_cei = 0
            ec_ge_gate_cei = 0
            ec_diff_ub_cei = 0
            revisit_misses = 0
            new_misses = 0
            clip4_rates = []
            clip5_rates = []

            for it in fam_items:
                st = it[stg]
                sv = saved_map[(fam_name, int(it["target_task_idx"]))][stg]
                qmap = st["queries"]
                t_draws = {int(k): np.asarray(v, dtype=float) for k, v in st["term_gain_scrambles"].items()}
                m_draws = {int(k): np.asarray(v, dtype=float) for k, v in st["menu_voi_scrambles"].items()}

                q_ec = int(qmap["eckg16"])
                q_topt = int(qmap["term_opt_ref1"])
                q_mopt = int(qmap["menu_opt_ref1"])
                q_sopt = int(qmap["screen_term_opt_ref1"])
                q_ub = int(qmap["ub_cei"])
                q_gcei = int(qmap["gate_cei"])
                q_sg = int(qmap["single"])
                q_fs = int(qmap["feas"])

                ec_m, ec_se, ec_lo, ec_hi = _ci95(t_draws[q_ec])
                opt_m, opt_se, opt_lo, opt_hi = _ci95(t_draws[q_topt])
                gap_m, gap_se, gap_lo, gap_hi = _ci95(t_draws[q_topt] - t_draws[q_ec])
                scr_gap_m, scr_gap_se, scr_gap_lo, scr_gap_hi = _ci95(t_draws[q_topt] - t_draws[q_sopt])
                ub_m, _, _, _ = _ci95(t_draws[q_ub])
                gcei_m, _, _, _ = _ci95(t_draws[q_gcei])
                sg_m, _, _, _ = _ci95(t_draws[q_sg])
                fs_m, _, _, _ = _ci95(t_draws[q_fs])
                ec_menu_m, ec_menu_se, ec_menu_lo, ec_menu_hi = _ci95(m_draws[q_ec])
                topt_menu_m, _, _, _ = _ci95(m_draws[q_topt])

                if ec_m < -1e-9:
                    neg_ec_count_24 += 1
                if ec_m >= ub_m - 1e-12:
                    ec_ge_ub_cei += 1
                if ec_m >= gcei_m - 1e-12:
                    ec_ge_gate_cei += 1
                if q_ec != q_ub:
                    ec_diff_ub_cei += 1

                anat = st["stage_a_anatomy"]
                if not anat["term_opt_in_screen16"]:
                    if anat["term_opt_is_revisit"]:
                        revisit_misses += 1
                    else:
                        new_misses += 1

                ref1_term = np.asarray(sv["reference_nomination_scores"], dtype=float)
                r1_max = float(np.max(ref1_term))
                r1_ec = float(ref1_term[q_ec])
                is_saturated = bool(opt_m < 1e-7 and abs(ec_m) < 1e-6)
                s_ratio_r1 = None if r1_max < 1e-7 else (r1_ec / r1_max)
                s_ratio_24 = None if is_saturated else (ec_m / max(opt_m, 1e-12))
                if s_ratio_r1 is not None:
                    ratios_single_ref.append(s_ratio_r1)
                if s_ratio_24 is not None:
                    ratios_mc_mean.append(s_ratio_24)

                ec_means.append(ec_m)
                opt_means.append(opt_m)
                ub_cei_means.append(ub_m)
                gate_cei_means.append(gcei_m)
                single_means.append(sg_m)
                feas_means.append(fs_m)
                clip4_rates.append(st["target_pool_clip4_coord_rate"])
                clip5_rates.append(st["target_pool_clip5_coord_rate"])

                state_table.append({
                    "family": fam_name,
                    "task_idx": int(it["target_task_idx"]),
                    "stage": stg,
                    "queries": qmap,
                    "query_is_revisit": st["query_is_revisit"],
                    "term_opt_in_screen16": anat["term_opt_in_screen16"],
                    "term_opt_is_revisit": anat["term_opt_is_revisit"],
                    "term_opt_slot_changes": st["slot_change_rates"][str(q_topt)],
                    "ec_term_x1e3": {"mean": 1e3 * ec_m, "se": 1e3 * ec_se, "ci95": [1e3 * ec_lo, 1e3 * ec_hi]},
                    "ec_menu_x1e3": {"mean": 1e3 * ec_menu_m, "se": 1e3 * ec_menu_se, "ci95": [1e3 * ec_menu_lo, 1e3 * ec_menu_hi]},
                    "opt_term_x1e3": {"mean": 1e3 * opt_m, "se": 1e3 * opt_se, "ci95": [1e3 * opt_lo, 1e3 * opt_hi]},
                    "opt_menu_x1e3_mean": 1e3 * topt_menu_m,
                    "query_loss_opt_minus_ec_x1e3": {"mean": 1e3 * gap_m, "se": 1e3 * gap_se, "ci95": [1e3 * gap_lo, 1e3 * gap_hi]},
                    "screen_loss_opt_minus_screen_x1e3": {"mean": 1e3 * scr_gap_m, "se": 1e3 * scr_gap_se, "ci95": [1e3 * scr_gap_lo, 1e3 * scr_gap_hi]},
                    "ub_cei_term_x1e3_mean": 1e3 * ub_m,
                    "gate_cei_term_x1e3_mean": 1e3 * gcei_m,
                    "single_term_x1e3_mean": 1e3 * sg_m,
                    "feas_term_x1e3_mean": 1e3 * fs_m,
                    "is_saturated_near_zero": is_saturated,
                    "state_ratio_ref1": s_ratio_r1,
                    "state_ratio_24scr": s_ratio_24,
                })

            sum_opt = float(np.mean(opt_means))
            families_out[fam_name][stg] = {
                "n_states": len(fam_items),
                "nonsaturated_states_count": len(ratios_mc_mean),
                "ratio_of_means_24scr": float(np.mean(ec_means) / sum_opt) if sum_opt > 1e-12 else 1.0,
                "mean_of_state_ratios_ref1": float(np.mean(ratios_single_ref)),
                "mean_of_state_ratios_24scr": float(np.mean(ratios_mc_mean)),
                "state_ratios_ref1": [round(float(x), 4) for x in ratios_single_ref],
                "state_ratios_24scr": [round(float(x), 4) for x in ratios_mc_mean],
                "ec_term_mean_x1e3": 1e3 * float(np.mean(ec_means)),
                "opt_term_mean_x1e3": 1e3 * sum_opt,
                "ub_cei_term_mean_x1e3": 1e3 * float(np.mean(ub_cei_means)),
                "gate_cei_term_mean_x1e3": 1e3 * float(np.mean(gate_cei_means)),
                "single_term_mean_x1e3": 1e3 * float(np.mean(single_means)),
                "feas_term_mean_x1e3": 1e3 * float(np.mean(feas_means)),
                "ec_beats_or_ties_ub_cei_count": ec_ge_ub_cei,
                "ec_beats_or_ties_gate_cei_count": ec_ge_gate_cei,
                "ec_differs_from_ub_cei_count": ec_diff_ub_cei,
                "stage_a_missed_revisit_count": revisit_misses,
                "stage_a_missed_new_count": new_misses,
                "target_pool_clip4_coord_rate_mean": float(np.mean(clip4_rates)),
                "target_pool_clip5_coord_rate_mean": float(np.mean(clip5_rates)),
            }

    fees = {
        k: sum(it[k] for it in items)
        for k in ("diagnostic_search_calls", "diagnostic_verification_calls", "diagnostic_audit_calls", "worker_cpu_seconds", "worker_wall_seconds")
    }
    summary = _stabilize_floats({
        "diagnostic_name": "v2_or_eckg_multiscramble_cei_audit_v1",
        "execution_code_sha256": desc["execution_code_sha256"],
        "n_scrambles_per_state": desc["n_scrambles_per_state"],
        "outer_sobol_points": 256,
        "inner_sobol_points": 1024,
        "total_states": 24,
        "negative_ec_term_states_24scr_count": neg_ec_count_24,
        "diagnostic_fees": fees,
        "source_archive_clipping": src_clip,
        "families": families_out,
        "states": state_table,
    })
    _write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("describe", "worker", "summarize", "submit"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", type=Path, default=SC_ROOT / "v2_or/manifests/v2_or_eckg_sure_confirm_v1.json")
        p.add_argument("--out-dir", type=Path, default=SC_ROOT / "paper_artifacts/v2_or_eckg_sure_confirm_v1/eckg_multiscramble_cei")
        p.add_argument("--states-path", type=Path, default=SC_ROOT / "paper_artifacts/v2_or_eckg_sure_confirm_v1/eckg_reference_states.json")
        p.add_argument("--n-scrambles", type=int, default=24)
        if name == "worker":
            p.add_argument("--start", type=int, required=True)
            p.add_argument("--end", type=int, required=True)
            p.add_argument("--workers", type=int, default=1)
        elif name == "summarize":
            p.add_argument("--summary-path", type=Path, default=SC_ROOT / "paper_artifacts/v2_or_eckg_sure_confirm_v1/eckg_multiscramble_cei_readout.json")
        elif name == "submit":
            p.add_argument("--scheduler", type=Path, default=Path(os.environ.get("GPFTA_CLUSTER_SCHEDULER", "scheduler.py")))
            p.add_argument("--remote-cwd", default=os.environ.get("GPFTA_REMOTE_CWD", "."))
            p.add_argument("--remote-python", default=sys.executable)
            p.add_argument("--nodes", default=",".join(DEFAULT_CPU_NODES))
            p.add_argument("--ram-mb", type=int, default=16384)
    args = parser.parse_args()

    if args.cmd == "describe":
        res = describe(args.manifest, args.out_dir, args.n_scrambles)
        print(json.dumps({"n_jobs": res["n_jobs"], "n_states": res["n_states"]}))
    elif args.cmd == "worker":
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        saved_bundle = json.loads(args.states_path.read_text(encoding="utf-8"))
        saved_map = {(t["family"], int(t["target_task_idx"])): t for t in saved_bundle["tasks"]}
        desc = json.loads((args.out_dir / "jobs.json").read_text(encoding="utf-8"))
        sub_jobs = desc["jobs"][args.start:args.end]
        w = max(1, min(int(args.workers), len(sub_jobs)))
        if w == 1:
            for jb in sub_jobs:
                print(json.dumps(_run_one_job(manifest, saved_map, args.out_dir, jb, args.n_scrambles)), flush=True)
        else:
            with ProcessPoolExecutor(max_workers=w) as ex:
                futs = [ex.submit(_run_one_job, manifest, saved_map, args.out_dir, jb, args.n_scrambles) for jb in sub_jobs]
                for f in as_completed(futs):
                    print(json.dumps(f.result()), flush=True)
    elif args.cmd == "summarize":
        s = summarize(args.out_dir, args.states_path, args.summary_path)
        print(json.dumps({"summary_path": str(args.summary_path), "total_states": s["total_states"]}))
    elif args.cmd == "submit":
        nodes = parse_cpu_nodes(args.nodes)
        desc = json.loads((args.out_dir / "jobs.json").read_text(encoding="utf-8"))
        cmd = shlex.join([
            "env", "OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
            "PYTHONPATH=.:../scripts",
            args.remote_python, "-u", "../scripts/run_eckg_multiscramble_cei_audit.py",
            "worker", "--manifest", str(args.manifest.relative_to(SC_ROOT)),
            "--out-dir", str(args.out_dir.relative_to(SC_ROOT)),
            "--states-path", str(args.states_path.relative_to(SC_ROOT)),
            "--n-scrambles", str(args.n_scrambles),
            "--start", "{start}", "--end", "{end}", "--workers", "{workers}",
        ])
        s_cmd = [
            sys.executable, str(args.scheduler), "submit-cpu-batch",
            "--items", str(desc["n_jobs"]), "--cmd-template", cmd, "--cwd", args.remote_cwd,
            "--nodes", ",".join(nodes), "--signature", "gp_fta_v2/eckg_mc_cei/{start}-{end}",
            "--description", "v2_or eckg multiscramble cei {node} [{start},{end})",
            "--ram-mb", str(args.ram_mb), "--project", "gp_fta_v2",
            "--no-auto-requeue", "--skip-launch-staging", "--env-spec", "none", "--json",
        ]
        cp = subprocess.run(s_cmd, text=True, capture_output=True, check=True)
        print(cp.stdout, end="")


if __name__ == "__main__":
    main()
