#!/usr/bin/env python3
"""Cluster study R8: Definitive single-factor counterfactuals, streaming sequential
curtailment, unbiased hypergeometric gate-power calibration, faithful ranking-weighted
RGPE, forecast-fed generic-psi isolation, and exact-omega horizon-balanced M_src scaling.

Preserves sealed `SC-OLH-KG/v2_or` (`fe8df9098424...`) and all prior cohorts untouched,
enforcing strict stage barriers:
  - `prepare_priors`: Reuses existing Stage-0 source trajectories to fit horizon-balanced
    `M_src = 4` (2 short + 2 medium source tasks), horizon-stratified `M_src = 3` (2 + 1),
    and forecast-fed generic-psi priors (`truth_used = False`).
  - `stage1` (`truth_used = False`, 172 items across 86 active tasks x 2 streams):
      * 172 `frozen_term_replays` items: on the exact frozen terminal search states `F_N`
        of `GP-FTA-ECKG`, evaluates (i) pure single-factor `TermOnly-AllowUnvisited`
        (keeping online `score_eligible_cascade_kg` search identical and relaxing ONLY
        terminal shortlist eligibility to `D_N U C_N^unvis`), (ii) `TermOnly-UnthreshMaxU`
        (combinatorial maximization of exact posterior `U_N(S)` over all ordered triples of
        top-14 evaluated policies across guarded/unguarded modes with zero 0.55 thresholds,
        zero 0.5/0.5 blending, and zero heuristic Slot-1 fixing), and (iii) live streaming
        replication-by-replication curtailment (`Streaming-Curtailed-k1` vs coupled
        `FixedLoop-k1`) measuring actual simulator calls and live verification wall-clock.
  - `stage2` (`truth_used = True`, 244 items, launched strictly after `stage1`):
      * 172 post-hoc ground-truth audits of `stage1` items on exact task `audit_seed`.
      * 72 `confirm72_definitive` items on the 72 unseen DES tasks computing exact unbiased
        hypergeometric U-statistic `C_hat_U` gate-power MSE & differences alongside 400-draw
        plug-in RMSE, genuine 64-bootstrap pairwise ranking-loss `RGPE-BootstrapRank-on-Psi`,
        forecast-fed HVD-controlled `Psi-GenericPhysOnly`, and exact `omega_M = p/(M+p)`
        scaling across horizon-balanced `M_src in {4, 6, 12}` and `M_src = 3`.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Sequence

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import binom, norm, qmc, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
SC_ROOT = REPO_ROOT / "SC-OLH-KG"
if str(SC_ROOT) not in sys.path:
    sys.path.insert(0, str(SC_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_direct_propagation_study import (
    _derangement_perm,
    _select_deterministic_probes_and_heldout,
    _stabilized_cov,
)
from run_rev7_closedloop_study import (
    _eval_cascade_rosters_fast,
    _holm_adjust,
    _load_active_family_specs,
    _load_sealed_reference_runs,
    _paired_stats,
    _stab_float,
    _verify_shortlist_with_curtailment,
)
from run_rev7_prior_and_simopt_study import (
    DES_FAMILIES,
    _corrupt_forecast_exposure,
    _drop_vol_exposure,
    _fit_ablr_source_prior,
    _load_des_fam_specs,
    _predict_ablr,
    compute_generic_blackbox_exposure,
)
from v2_or.acquisition.verification_kg import (
    build_balanced_warmstart_points,
    build_cold_dct_initial_points,
    generate_dynamic_candidate_pool,
    score_eligible_cascade_kg,
)
from v2_or.baselines.sota_baselines import (
    RawInputExactGP,
    pretrain_rgpe_psi_source_ensemble,
)
from v2_or.contracts import (
    BaseDynamicORProblem,
    LegacyTrajectoryRecord,
    NominalTrajectoryExposure,
    SufficientStatsRecord,
)
from v2_or.dispatch.campaign_runner import build_family_tasks, compute_v2_or_code_sha256
from v2_or.legacy_solvers import load_legacy_archive_json
from v2_or.models.tempered_hvd_posterior import (
    JointPosteriorPrediction,
    TemperedHierarchicalHVDModel,
)
from v2_or.priors.debiased_spectral_prior import DebiasedSpectralMetaPrior
from v2_or.verification.gated_verifier_v2 import (
    GateCertificate,
    TerminalDeploymentReceipt,
    build_feasible_aware_shortlist,
    exact_binomial_critical_k,
    exact_binomial_gate,
    exact_binomial_pass_power,
    gaussian_noncentral_t_gate,
    posterior_binomial_gate_power,
)

OUT_ROOT = SC_ROOT / "paper_artifacts" / "v2_or_rev8_definitive_v1"
REV7_P2_ROOT = SC_ROOT / "paper_artifacts" / "v2_or_rev7_or_panel_v1" / "prior_and_simopt"
U3_BUNDLES = SC_ROOT / "paper_artifacts" / "v2_or_u3_confirm_v1" / "source_bundles"
COMB_400_112 = math.comb(400, 112)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _task_public_forecast_vector(prob: BaseDynamicORProblem) -> np.ndarray:
    """Extract the public load/arrival forecast vector matching `x_norm` window count."""
    from v2_or.simulators.inventory_des import DynamicInventoryNetworkDES
    from v2_or.simulators.queueing_des import DynamicQueueingStaffingDES
    from v2_or.simulators.traffic_signal_des import IngolstadtSignalNetworkDES

    if isinstance(prob, DynamicInventoryNetworkDES):
        return np.asarray(prob.demand_forecast, dtype=float)
    if isinstance(prob, DynamicQueueingStaffingDES):
        return np.asarray(prob.arrival_forecast, dtype=float)
    if isinstance(prob, IngolstadtSignalNetworkDES):
        # Mean public arrival rate per controlled green phase (length d_k)
        return np.mean(np.asarray(prob.public_arrivals, dtype=float), axis=0)
    return np.ones(max(prob.d // max(int(prob.num_control_channels), 1), 1), dtype=float)


def _unbiased_c112_1_from_counts(v_counts: np.ndarray, n_aud: int = 400) -> np.ndarray:
    """Exact minimum-variance unbiased hypergeometric U-statistic for BinomCDF(1; 112, p)."""
    denom = float(math.comb(n_aud, 112))
    out = np.empty(v_counts.size, dtype=float)
    for idx, v_raw in enumerate(v_counts):
        v = int(round(float(v_raw)))
        rem = n_aud - v
        t0 = math.comb(rem, 112) if rem >= 112 else 0
        t1 = v * math.comb(rem, 111) if rem >= 111 else 0
        out[idx] = float(t0 + t1) / denom
    return out


def _build_prior_variants_exact_m(prior: DebiasedSpectralMetaPrior, seed: int, m_src: int) -> dict[str, Any]:
    """Construct zero-mean trace-matched kernel variants using exact `omega_M = p / (M + p)`."""
    rng = np.random.default_rng(int(seed))
    cov_f_stab = _stabilized_cov(prior.cov_0_obj_, m_src=int(m_src))
    cov_g_stab = _stabilized_cov(prior.cov_0_con_, m_src=int(m_src))
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

    def _zero_mean_prior(cov_f_mat: np.ndarray, cov_g_mat: np.ndarray) -> DebiasedSpectralMetaPrior:
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

    p_diag_online = copy.deepcopy(prior)
    p_diag_online.cov_0_obj_ = np.diag(np.diag(prior.cov_0_obj_))
    p_diag_online.cov_0_con_ = np.diag(np.diag(prior.cov_0_con_))

    return {
        "zm_iso": _zero_mean_prior(iso_f, iso_g),
        "zm_diag": _zero_mean_prior(np.diag(np.diag(cov_f_stab)), np.diag(np.diag(cov_g_stab))),
        "zm_full": _zero_mean_prior(cov_f_stab, cov_g_stab),
        "zm_perm": _zero_mean_prior(perm_cov_f, perm_cov_g),
        "online_diag": p_diag_online,
        "online_full": prior,
    }


def build_unthresholded_triple_max_u_shortlist(
    cands: Sequence[np.ndarray],
    pred: JointPosteriorPrediction,
    eval_recs: Sequence[SufficientStatsRecord],
    *,
    alpha: float = 0.05,
    sq_ref: float,
    z_in128: np.ndarray,
    slot_budgets: tuple[int, int, int] = (112, 128, 140),
) -> tuple[list[tuple[str, np.ndarray]], float, int]:
    """Unthresholded 3-slot cascade utility maximizer over all distinct ordered triples of
    top evaluated candidates in `D_N` (both guarded and unguarded modes, zero 0.55 gate
    thresholds, zero 0.5/0.5 blending, zero heuristic Slot-1 fixing)."""
    n_c = len(cands)
    if n_c == 1:
        return [("posterior_primary", np.asarray(cands[0], dtype=float).copy())], 0.0, 1

    mf = pred.mean_obj
    mg = pred.mean_con
    sg = pred.std_con
    uv = np.maximum(pred.ucb_aleatoric_var_con, 1e-8)
    rp = np.maximum(pred.r_perp_con, 0.0)
    p_g0 = posterior_binomial_gate_power(mg, sg, uv, n_samples=slot_budgets[0], alpha=alpha, delta_slot=0.025, r_perp_con=rp)
    p_g1 = posterior_binomial_gate_power(mg, sg, uv, n_samples=slot_budgets[1], alpha=alpha, delta_slot=0.015, r_perp_con=rp)
    p_g2 = posterior_binomial_gate_power(mg, sg, uv, n_samples=slot_budgets[2], alpha=alpha, delta_slot=0.010, r_perp_con=rp)

    u0 = np.maximum(sq_ref - mf, 0.0) * p_g0
    u1 = np.maximum(sq_ref - mf, 0.0) * p_g1
    pool_idx: list[int] = []
    seen_k: set[bytes] = set()
    rank_union = (
        list(np.argsort(-u0)[:10])
        + list(np.argsort(-u1)[:6])
        + list(np.argsort(-p_g2)[:6])
        + [0]
    )
    for idx_raw in rank_union:
        ii = int(idx_raw)
        if 0 <= ii < n_c:
            kb = np.round(cands[ii], 5).tobytes()
            if kb not in seen_k:
                seen_k.add(kb)
                pool_idx.append(ii)
            if len(pool_idx) >= 14:
                break

    if len(pool_idx) < 3:
        for ii in range(n_c):
            kb = np.round(cands[ii], 5).tobytes()
            if kb not in seen_k:
                seen_k.add(kb)
                pool_idx.append(ii)
            if len(pool_idx) >= min(3, n_c):
                break

    triples: list[list[int]] = []
    guard_flags: list[bool] = []
    if len(pool_idx) == 1:
        triples.append([pool_idx[0]])
        guard_flags.append(False)
    elif len(pool_idx) == 2:
        for a in pool_idx:
            for b in pool_idx:
                if a != b:
                    for gf in (True, False):
                        triples.append([a, b])
                        guard_flags.append(gf)
    else:
        for a in pool_idx:
            for b in pool_idx:
                if b == a:
                    continue
                for c in pool_idx:
                    if c == a or c == b:
                        continue
                    triples.append([a, b, c])
                    guard_flags.append(True)
                    triples.append([a, b, c])
                    guard_flags.append(False)

    B = len(triples)
    r_mf = np.zeros((B, 3), dtype=float)
    r_cf = np.zeros((B, 3, 3), dtype=float)
    r_avf = np.full((B, 3), 1e-6, dtype=float)
    r_mg = np.zeros((B, 3), dtype=float)
    r_cg = np.repeat(np.eye(3, dtype=float)[None, :, :] * 1e-6, B, axis=0)
    r_sg = np.ones((B, 3), dtype=float)
    ns_arr = np.zeros(B, dtype=int)
    hg_arr = np.asarray(guard_flags, dtype=bool)
    cov_g_tot = pred.cov_con + np.diag(0.25 * rp)

    for b_i, seq in enumerate(triples):
        L = len(seq)
        ns_arr[b_i] = L
        idx_a = np.asarray(seq, dtype=int)
        r_mf[b_i, :L] = mf[idx_a]
        r_cf[b_i, :L, :L] = pred.cov_obj[np.ix_(idx_a, idx_a)]
        r_avf[b_i, :L] = np.maximum(pred.aleatoric_var_obj[idx_a], 1e-8)
        r_mg[b_i, :L] = mg[idx_a]
        r_cg[b_i, :L, :L] = cov_g_tot[np.ix_(idx_a, idx_a)]
        r_sg[b_i, :L] = np.sqrt(uv[idx_a])

    u_vals = _eval_cascade_rosters_fast(
        sq_ref, r_mf, r_cf, r_avf, r_mg, r_cg, r_sg, ns_arr, hg_arr, z_in128,
        slot_budgets=slot_budgets, alpha=alpha,
    )
    best_b = int(np.argmax(u_vals))
    best_seq = triples[best_b]
    best_guard = bool(guard_flags[best_b])
    roles = ["posterior_primary"]
    if len(best_seq) >= 2:
        roles.append("empirical_feasible_incumbent" if best_guard else "ungated_backup_slot1")
    if len(best_seq) >= 3:
        roles.append("conservative_posterior_interior")
    shortlist = [(r, np.asarray(cands[idx], dtype=float).copy()) for r, idx in zip(roles, best_seq)]
    return shortlist, float(u_vals[best_b]), B


def _verify_coupled_streaming_vs_fixed(
    problem: BaseDynamicORProblem,
    shortlist: Sequence[tuple[str, np.ndarray]],
    *,
    seed: int,
    slot_budgets: tuple[int, int, int] = (112, 128, 140),
    familywise_delta: float = 0.05,
    gate_mode: str = "auto",
) -> dict[str, Any]:
    """Execute both live replication-by-replication fixed-block verification and live
    streaming curtailed verification on coupled per-slot RNG streams, confirming identical
    deployed decisions on every run while measuring actual live verification wall-clock."""
    use_mode = ("gaussian_noncentral_t" if problem.is_gaussian_constraint_noise else "exact_binomial") if gate_mode == "auto" else str(gate_mode)
    n_slots = min(len(shortlist), 3)
    if n_slots == 1:
        deltas = (float(familywise_delta),)
    elif n_slots == 2:
        deltas = (0.625 * float(familywise_delta), 0.375 * float(familywise_delta))
    else:
        deltas = (0.50 * float(familywise_delta), 0.30 * float(familywise_delta), 0.20 * float(familywise_delta))
    used_budgets = tuple(int(slot_budgets[min(i, len(slot_budgets) - 1)]) for i in range(n_slots))
    shortlist_rls = tuple(str(r) for r, _ in shortlist[:n_slots])
    has_guard = n_slots >= 2 and shortlist_rls[1] in {"empirical_feasible_incumbent", "safe_initial_incumbent"}
    k_crits = tuple(
        exact_binomial_critical_k(used_budgets[i], float(problem.alpha), float(deltas[i]))
        if use_mode == "exact_binomial" else -1
        for i in range(n_slots)
    )

    # 1. Fixed-loop draw-by-draw verification
    t_fix0 = time.perf_counter()
    fix_calls = 0

    def _run_fixed_slot(idx: int) -> tuple[bool, float, np.ndarray]:
        nonlocal fix_calls
        rng_s = np.random.default_rng(int(seed) + 907 * idx)
        K = used_budgets[idx]
        x_pol = shortlist[idx][1]
        rows = [problem.simulate_replications(x_pol, 1, rng_s, split="verification")[0] for _ in range(K)]
        fix_calls += K
        arr = np.asarray(rows, dtype=float)
        if use_mode == "exact_binomial":
            ok, _, _, _, _ = exact_binomial_gate(arr[:, 1], alpha=problem.alpha, delta_slot=float(deltas[idx]))
        else:
            ok, _, _, _ = gaussian_noncentral_t_gate(arr[:, 1], alpha=problem.alpha, delta_slot=float(deltas[idx]))
        return bool(ok), float(np.mean(arr[:, 0])), arr

    f_ok0, f_m0, _ = _run_fixed_slot(0)
    if f_ok0 and not has_guard:
        fix_sel = 0
    elif has_guard:
        f_ok1, f_m1, _ = _run_fixed_slot(1)
        if f_ok0 and f_ok1:
            fix_sel = 0 if f_m0 <= f_m1 else 1
        elif f_ok0:
            fix_sel = 0
        elif f_ok1:
            fix_sel = 1
        else:
            fix_sel = None
            for idx in range(2, n_slots):
                ok_i, _, _ = _run_fixed_slot(idx)
                if ok_i:
                    fix_sel = idx
                    break
    else:
        fix_sel = None
        for idx in range(1, n_slots):
            ok_i, _, _ = _run_fixed_slot(idx)
            if ok_i:
                fix_sel = idx
                break
    fix_wall = time.perf_counter() - t_fix0

    # 2. Live streaming curtailed draw-by-draw verification on identical coupled slot seeds!
    t_str0 = time.perf_counter()
    str_calls = 0

    def _run_stream_slot(idx: int) -> tuple[bool, list[np.ndarray], np.random.Generator]:
        nonlocal str_calls
        rng_s = np.random.default_rng(int(seed) + 907 * idx)
        K = used_budgets[idx]
        kc = k_crits[idx]
        x_pol = shortlist[idx][1]
        rows: list[np.ndarray] = []
        v_acc = 0
        if use_mode != "exact_binomial":
            for _ in range(K):
                rows.append(problem.simulate_replications(x_pol, 1, rng_s, split="verification")[0])
                str_calls += 1
            arr = np.asarray(rows, dtype=float)
            ok, _, _, _ = gaussian_noncentral_t_gate(arr[:, 1], alpha=problem.alpha, delta_slot=float(deltas[idx]))
            return bool(ok), rows, rng_s

        for m_idx in range(1, K + 1):
            dr = problem.simulate_replications(x_pol, 1, rng_s, split="verification")[0]
            rows.append(dr)
            str_calls += 1
            if float(dr[1]) > 0.0:
                v_acc += 1
            if v_acc >= kc + 1:
                return False, rows, rng_s
            if v_acc + (K - m_idx) <= kc:
                return True, rows, rng_s
        return bool(v_acc <= kc), rows, rng_s

    s_ok0, s_rows0, s_rng0 = _run_stream_slot(0)
    if s_ok0 and not has_guard:
        str_sel = 0
    elif has_guard:
        s_ok1, s_rows1, s_rng1 = _run_stream_slot(1)
        if s_ok0 and s_ok1:
            # Complete full blocks on Slots 0 and 1 so objective guard comparison is sample-path identical!
            while len(s_rows0) < used_budgets[0]:
                s_rows0.append(problem.simulate_replications(shortlist[0][1], 1, s_rng0, split="verification")[0])
                str_calls += 1
            while len(s_rows1) < used_budgets[1]:
                s_rows1.append(problem.simulate_replications(shortlist[1][1], 1, s_rng1, split="verification")[0])
                str_calls += 1
            m0_full = float(np.mean([r[0] for r in s_rows0]))
            m1_full = float(np.mean([r[0] for r in s_rows1]))
            str_sel = 0 if m0_full <= m1_full else 1
        elif s_ok0:
            str_sel = 0
        elif s_ok1:
            str_sel = 1
        else:
            str_sel = None
            for idx in range(2, n_slots):
                ok_i, _, _ = _run_stream_slot(idx)
                if ok_i:
                    str_sel = idx
                    break
    else:
        str_sel = None
        for idx in range(1, n_slots):
            ok_i, _, _ = _run_stream_slot(idx)
            if ok_i:
                str_sel = idx
                break
    str_wall = time.perf_counter() - t_str0

    assert fix_sel == str_sel, f"Streaming curtailment mismatch: {fix_sel} vs {str_sel}"
    dep_pol = None if str_sel is None else [float(v) for v in shortlist[str_sel][1]]
    return {
        "selected_slot": str_sel,
        "certified": str_sel is not None,
        "deployed_policy": dep_pol,
        "fixed_loop_calls": int(fix_calls),
        "streaming_curtailed_calls": int(str_calls),
        "fixed_loop_wall_sec": float(fix_wall),
        "streaming_curtailed_wall_sec": float(str_wall),
        "exact_decision_match": True,
    }


def prepare_priors(out_dir: Path) -> dict[str, Any]:
    """Fit horizon-balanced M_src=4, horizon-stratified M_src=3, and forecast-fed generic priors."""
    s0_desc = _read_json(REV7_P2_ROOT / "stage0" / "jobs.json")
    by_fam_ext: dict[str, list[LegacyTrajectoryRecord]] = {}
    for jb in s0_desc["jobs"]:
        if jb["family"] not in DES_FAMILIES:
            continue
        it = _read_json(REV7_P2_ROOT / "stage0" / "items" / f"{jb['job_id']}.json")
        for rd in it["records"]:
            by_fam_ext.setdefault(jb["family"], []).append(LegacyTrajectoryRecord(
                task_id=rd["task_id"],
                solver_name=rd["solver_name"],
                step_index=int(rd["step_index"]),
                policy_dim=int(rd["policy_dim"]),
                x_norm=tuple(float(v) for v in rd["x_norm"]),
                n_reps=int(rd["n_reps"]),
                mean_obj=float(rd["mean_obj"]),
                ssq_obj=float(rd["ssq_obj"]),
                mean_con=float(rd["mean_con"]),
                ssq_con=float(rd["ssq_con"]),
                violation_count=int(rd["violation_count"]),
                psi=tuple(float(v) for v in rd["psi"]),
                A_local=tuple(float(v) for v in rd["A_local"]),
                N_shared=tuple(float(v) for v in rd["N_shared"]),
            ))

    bundles_out = out_dir / "bundles"
    bundles_out.mkdir(parents=True, exist_ok=True)
    fam_specs = _load_des_fam_specs()
    summary_info: dict[str, Any] = {}

    for fam in DES_FAMILIES:
        recs_6, _ = load_legacy_archive_json(U3_BUNDLES / fam / "archive.json")
        recs_12 = list(recs_6) + by_fam_ext.get(fam, [])

        # Group the 6 baseline source tasks by policy_dim so M_src=4 is strictly 2+2 horizon-balanced!
        dim_by_tid: dict[str, int] = {}
        for r in recs_6:
            dim_by_tid[r.task_id] = int(r.policy_dim)
        dims_sorted = sorted(set(dim_by_tid.values()))
        by_dim = {dm: sorted([t for t, d in dim_by_tid.items() if d == dm]) for dm in dims_sorted}
        if len(dims_sorted) >= 2:
            tids_m4_bal = set(by_dim[dims_sorted[0]][:2] + by_dim[dims_sorted[-1]][:2])
            tids_m3_strat = set(by_dim[dims_sorted[0]][:2] + by_dim[dims_sorted[-1]][:1])
        else:
            all_t6 = sorted(dim_by_tid.keys())
            tids_m4_bal = set(all_t6[:4])
            tids_m3_strat = set(all_t6[:3])
        tids_m3_orig = set(sorted(dim_by_tid.keys())[:3])

        recs_m4_bal = [r for r in recs_6 if r.task_id in tids_m4_bal]
        recs_m3_strat = [r for r in recs_6 if r.task_id in tids_m3_strat]
        recs_m3_orig = [r for r in recs_6 if r.task_id in tids_m3_orig]

        fam_b = bundles_out / fam
        fam_b.mkdir(parents=True, exist_ok=True)

        (fam_b / "prior_m3_orig.pkl").write_bytes(pickle.dumps(DebiasedSpectralMetaPrior().fit(recs_m3_orig)))
        (fam_b / "prior_m3_strat.pkl").write_bytes(pickle.dumps(DebiasedSpectralMetaPrior().fit(recs_m3_strat)))
        (fam_b / "prior_m4_bal.pkl").write_bytes(pickle.dumps(DebiasedSpectralMetaPrior().fit(recs_m4_bal)))
        (fam_b / "prior_m6.pkl").write_bytes((U3_BUNDLES / fam / "prior_default.pkl").read_bytes())
        (fam_b / "prior_m12.pkl").write_bytes((REV7_P2_ROOT / "bundles" / fam / "prior_m12.pkl").read_bytes())

        for fn in ("prior_no_debias.pkl", "prior_no_lowpass.pkl", "prior_no_ortho.pkl", "prior_no_sparse.pkl", "prior_dropvol7d.pkl", "ablr_m6.pkl", "mtgp_m6.pkl", "rgpe_m6.pkl"):
            (fam_b / fn).write_bytes((REV7_P2_ROOT / "bundles" / fam / fn).read_bytes())

        # Fit forecast-fed generic-psi priors (both PhysOnly keeping A_local/N_shared, and ForecastFull)
        src_probs = {p.task_id: p for p in build_family_tasks(fam_specs[fam], "source")}
        n_ch = 2 if fam in {"Inventory_CrossHorizon", "Queueing_CrossHorizon"} else 1
        recs_gen_phys = []
        recs_gen_full = []
        for r in recs_6:
            sp = src_probs.get(r.task_id)
            fc_vec = _task_public_forecast_vector(sp) if sp is not None else None
            eg = compute_generic_blackbox_exposure(np.asarray(r.x_norm, dtype=float), num_channels=n_ch, pub_load_profile=fc_vec)
            recs_gen_phys.append(LegacyTrajectoryRecord(**{
                **asdict(r),
                "psi": tuple(float(v) for v in eg.psi),
            }))
            recs_gen_full.append(LegacyTrajectoryRecord(**{
                **asdict(r),
                "psi": tuple(float(v) for v in eg.psi),
                "A_local": tuple(float(v) for v in eg.A_local),
                "N_shared": tuple(float(v) for v in eg.N_shared),
            }))
        (fam_b / "prior_generic_phys_only.pkl").write_bytes(pickle.dumps(DebiasedSpectralMetaPrior().fit(recs_gen_phys)))
        (fam_b / "prior_generic_forecast_full.pkl").write_bytes(pickle.dumps(DebiasedSpectralMetaPrior().fit(recs_gen_full)))

        summary_info[fam] = {
            "m4_bal_tasks": sorted(tids_m4_bal),
            "m3_strat_tasks": sorted(tids_m3_strat),
            "n_m12_records": len(recs_12),
        }

    done = {"completed_utc": datetime.now(timezone.utc).isoformat(), "families": summary_info}
    _write_json(out_dir / "priors_ready.json", done)
    return done


def describe_stage(stage: str, out_dir: Path) -> dict[str, Any]:
    exec_sha = compute_v2_or_code_sha256()
    fams = _load_active_family_specs()
    jobs: list[dict[str, Any]] = []

    if stage == "stage1":
        for fam in fams:
            fname = fam["name"]
            t_conf = fam["target_config"]
            n_tgt = len(t_conf["markets"]) if "markets" in t_conf else int(t_conf["n_tasks"])
            rep_seeds = [int(s) for s in fam["target_eval_seeds"]]
            for t_idx in range(n_tgt):
                for rep_seed in rep_seeds:
                    seed = int(rep_seed) + 10_007 * t_idx
                    jobs.append({
                        "job_id": f"r8_s1_term__{fname}__t{t_idx:02d}__r{rep_seed}",
                        "kind": "frozen_term_replays",
                        "family": fname,
                        "target_task_idx": t_idx,
                        "rep_seed": int(rep_seed),
                        "seed": seed,
                        "n0_init": int(fam.get("n0_init", 8)),
                        "n_online_steps": 60,
                        "batch_reps": int(fam.get("batch_reps", 2)),
                        "pool_size": int(fam.get("pool_size", 160)),
                    })
    elif stage == "stage2":
        s1_desc = _read_json(out_dir / "stage1" / "jobs.json")
        for fj in s1_desc["jobs"]:
            fname = fj["family"]
            t_idx = int(fj["target_task_idx"])
            rep_seed = int(fj["rep_seed"])
            jobs.append({
                "job_id": f"r8_s2_audit__{fname}__t{t_idx:02d}__r{rep_seed}",
                "kind": "cell_audit",
                "family": fname,
                "target_task_idx": t_idx,
                "rep_seed": rep_seed,
                "replay_job_id": fj["job_id"],
            })
        for fam in DES_FAMILIES:
            for t_idx in range(24):
                jobs.append({
                    "job_id": f"r8_s2_conf72__{fam}__t{t_idx:02d}",
                    "kind": "confirm72_definitive",
                    "family": fam,
                    "target_task_idx": t_idx,
                })
    else:
        raise ValueError(stage)

    desc = {
        "stage": stage,
        "execution_code_sha256": exec_sha,
        "n_jobs": len(jobs),
        "jobs": jobs,
    }
    _write_json(out_dir / stage / "jobs.json", desc)
    return desc


def _replay_eckg_search_state(
    prob: BaseDynamicORProblem,
    prior: DebiasedSpectralMetaPrior,
    *,
    seed: int,
    n0_init: int,
    n_steps: int,
    batch_reps: int,
    pool_size: int,
) -> tuple[TemperedHierarchicalHVDModel, np.random.Generator, float, int, int]:
    """Reconstruct the exact 60-step `GP-FTA-ECKG` search trajectory and terminal posterior."""
    rng_init = np.random.default_rng(seed)
    rng_search = np.random.default_rng(seed + 100_003)
    calls_start = prob.search_calls

    model = TemperedHierarchicalHVDModel(
        prior,
        mode="tempered_sure_release",
        use_source_basis=True,
        use_source_coef_prior=True,
        use_source_hvd_prior=True,
        n0_freeze=n0_init,
    )
    init_pts = build_balanced_warmstart_points(prob, prior, n0_init, rng_init)
    for x0 in init_pts:
        e0 = prob.nominal_exposure(x0)
        s0 = prob.simulate_replications(x0, batch_reps, rng_init, split="search")
        model.observe(SufficientStatsRecord.from_samples(x0, s0, e0))

    sq_est_obj = float(model.unique_records[0].mean_obj)
    n_rep_q = 0
    for _ in range(int(n_steps)):
        enable_tpl = not bool(getattr(model, "is_source_released_", False))
        cands, exps = generate_dynamic_candidate_pool(
            prob, model, prior, rng_search, pool_size=pool_size, enable_source_templates=enable_tpl
        )
        pred = model.predict_joint(cands, exps)
        sq_ref = 0.5 * sq_est_obj + 0.5 * float(pred.mean_obj[0])
        _, diag = score_eligible_cascade_kg(
            pred, model, cands, exps, alpha=prob.alpha, batch_reps=batch_reps,
            status_quo_obj=sq_ref, nomination_mode="cascade_bellman_3slot", return_diagnostics=True,
        )
        chosen_idx = int(diag["chosen_query"])
        xn = cands[chosen_idx]
        if model._policy_key(xn) in model._records_by_key:
            n_rep_q += 1
        samp = prob.simulate_replications(xn, batch_reps, rng_search, split="search")
        model.observe(SufficientStatsRecord.from_samples(xn, samp, exps[chosen_idx]))

    search_used = int(prob.search_calls - calls_start)
    return model, rng_search, sq_est_obj, n_rep_q, search_used


def _run_stage1_item(job: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    item_path = out_dir / "stage1" / "items" / f"{job['job_id']}.json"
    if item_path.is_file():
        return {"job_id": job["job_id"], "status": "cached"}

    fams = {f["name"]: f for f in _load_active_family_specs()}
    fam_spec = fams[job["family"]]
    prob = build_family_tasks(fam_spec, "target")[int(job["target_task_idx"])]
    src_fam = fam_spec.get("reuse_archive_from_family", job["family"])
    prior = pickle.loads((U3_BUNDLES / src_fam / "prior_default.pkl").read_bytes())
    gate_mode = fam_spec.get("gate_mode", "auto")
    seed = int(job["seed"])
    verif_seed = seed + 200_007

    t0 = time.perf_counter()
    model, rng_search, sq_est_obj, n_rep_q, search_used = _replay_eckg_search_state(
        prob, prior, seed=seed, n0_init=int(job["n0_init"]),
        n_steps=int(job["n_online_steps"]), batch_reps=int(job["batch_reps"]),
        pool_size=int(job["pool_size"]),
    )
    eval_recs = model.unique_records
    eval_keys = {model._policy_key(r.x_norm) for r in eval_recs}
    term_cands = [r.x_norm.copy() for r in eval_recs]
    term_exps = [r.exposure for r in eval_recs]
    term_pred = model.predict_joint(term_cands, term_exps)
    sq_ref_term = 0.5 * sq_est_obj + 0.5 * float(term_pred.mean_obj[0])

    # `frozen_term_replays`: pure single-factor terminal evaluations on frozen F_N of GP-FTA-ECKG
    # 1. Default GP-FTA-ECKG shortlist for live streaming curtailment timing
    sl_def = build_feasible_aware_shortlist(
        candidate_pool=term_cands,
        mean_obj=term_pred.mean_obj,
        mean_con=term_pred.mean_con,
        std_con=term_pred.std_con,
        ucb_aleat_var_con=term_pred.ucb_aleatoric_var_con,
        r_perp_con=term_pred.r_perp_con,
        initial_records=eval_recs,
        alpha=prob.alpha,
        nomination_mode="cascade_bellman_3slot",
        reference_obj=sq_ref_term,
    )
    streaming_res = _verify_coupled_streaming_vs_fixed(
        prob, sl_def, seed=verif_seed, slot_budgets=(112, 128, 140), gate_mode=gate_mode
    )

    # 2. TermOnly-AllowUnvisited: identical 60-step EC-KG search state F_N, relaxing ONLY terminal eligibility!
    unvis_cands, unvis_exps = generate_dynamic_candidate_pool(
        prob, model, prior, rng_search, pool_size=256, enable_source_templates=True
    )
    dedup_x = []
    dedup_e = []
    seen_u = set(eval_keys)
    for ux, ue in zip(unvis_cands, unvis_exps):
        uk = model._policy_key(ux)
        if uk not in seen_u:
            seen_u.add(uk)
            dedup_x.append(ux)
            dedup_e.append(ue)
    comb_cands = term_cands + dedup_x
    comb_exps = term_exps + dedup_e
    comb_pred = model.predict_joint(comb_cands, comb_exps)
    sl_unvis = build_feasible_aware_shortlist(
        candidate_pool=comb_cands,
        mean_obj=comb_pred.mean_obj,
        mean_con=comb_pred.mean_con,
        std_con=comb_pred.std_con,
        ucb_aleat_var_con=comb_pred.ucb_aleatoric_var_con,
        r_perp_con=comb_pred.r_perp_con,
        initial_records=eval_recs,
        alpha=prob.alpha,
        nomination_mode="cascade_bellman_3slot",
        reference_obj=sq_ref_term,
    )
    rcpt_unvis, curt_unvis = _verify_shortlist_with_curtailment(
        prob, sl_unvis, seed=verif_seed, slot_budgets=(112, 128, 140), gate_mode=gate_mode
    )

    # 3. TermOnly-UnthreshMaxU: unthresholded 3-slot posterior cascade utility maximization on D_N
    eng_in = qmc.Sobol(d=3, scramble=True, seed=seed + 71)
    u_in = np.clip(eng_in.random_base2(6), 1e-6, 1.0 - 1e-6)
    z_in128 = np.vstack([ndtri(u_in), -ndtri(u_in)])  # (128, 3)
    t_maxu0 = time.perf_counter()
    sl_maxu, u_max_val, n_triples = build_unthresholded_triple_max_u_shortlist(
        term_cands, term_pred, eval_recs, alpha=prob.alpha, sq_ref=sq_ref_term, z_in128=z_in128
    )
    maxu_nom_wall = time.perf_counter() - t_maxu0
    rcpt_maxu, curt_maxu = _verify_shortlist_with_curtailment(
        prob, sl_maxu, seed=verif_seed, slot_budgets=(112, 128, 140), gate_mode=gate_mode
    )
    wall_tot = time.perf_counter() - t0

    payload = {
        "job_id": job["job_id"],
        "kind": "frozen_term_replays",
        "family": job["family"],
        "target_task_idx": int(job["target_task_idx"]),
        "rep_seed": int(job["rep_seed"]),
        "seed": seed,
        "search_calls_used": search_used,
        "replay_wall_sec": float(wall_tot),
        "term_only_allow_unvisited": {
            "certified": bool(rcpt_unvis.certified),
            "selected_slot": rcpt_unvis.selected_slot,
            "selected_role": rcpt_unvis.selected_role,
            "verification_calls_used": int(rcpt_unvis.verification_calls_used),
            "curtailed_verification_calls": int(curt_unvis),
            "slot0_is_unvisited": bool(model._policy_key(sl_unvis[0][1]) not in eval_keys),
            "any_slot_unvisited": bool(any(model._policy_key(sp) not in eval_keys for _, sp in sl_unvis)),
            "deployed_policy": None if rcpt_unvis.deployed_policy is None else [float(v) for v in rcpt_unvis.deployed_policy],
        },
        "term_only_unthresh_maxu": {
            "n_triples_scored": int(n_triples),
            "nomination_wall_sec": float(maxu_nom_wall),
            "expected_cascade_saving": float(u_max_val),
            "same_slot0_as_default": bool(model._policy_key(sl_maxu[0][1]) == model._policy_key(sl_def[0][1])),
            "certified": bool(rcpt_maxu.certified),
            "selected_slot": rcpt_maxu.selected_slot,
            "selected_role": rcpt_maxu.selected_role,
            "verification_calls_used": int(rcpt_maxu.verification_calls_used),
            "curtailed_verification_calls": int(curt_maxu),
            "deployed_policy": None if rcpt_maxu.deployed_policy is None else [float(v) for v in rcpt_maxu.deployed_policy],
        },
        "streaming_curtailment_k1": streaming_res,
        "truth_used": False,
    }
    _write_json(item_path, payload)
    return {"job_id": job["job_id"], "status": "ok"}


def _eval_heldout_metrics_definitive(
    mean_f: np.ndarray,
    std_f: np.ndarray,
    mean_g: np.ndarray,
    std_g: np.ndarray,
    ucb_var_g: np.ndarray,
    r_perp_g: np.ndarray,
    sq_ref: float,
    true_obj: np.ndarray,
    true_con: np.ndarray,
    true_p_viol_plug: np.ndarray,
    true_v_counts: np.ndarray,
    anchor_true_obj: float,
    alpha: float = 0.05,
) -> dict[str, float | None]:
    pred_gate = posterior_binomial_gate_power(
        mean_g, std_g, np.maximum(ucb_var_g, 1e-8),
        n_samples=112, alpha=alpha, delta_slot=0.025, r_perp_con=np.maximum(r_perp_g, 0.0),
    )
    plug_gate = exact_binomial_pass_power(true_p_viol_plug, 112, 1)
    unb_gate_cu = _unbiased_c112_1_from_counts(true_v_counts, n_aud=400)

    gate_rmse_plugin = float(np.sqrt(np.mean((pred_gate - plug_gate) ** 2)))
    # MSE against exact unbiased U-statistic C_hat_U: paired differences have EXACT ZERO finite-audit bias!
    gate_mse_unbiased = float(np.mean((pred_gate - unb_gate_cu) ** 2))

    sig_tot = np.sqrt(np.maximum(std_g ** 2 + 0.25 * np.maximum(r_perp_g, 0.0), 1e-12))
    thr_g = float(norm.ppf(alpha)) * np.sqrt(np.maximum(ucb_var_g, 1e-8))
    p_feas_pred = ndtr((thr_g - mean_g) / sig_tot)
    y_feas = (true_p_viol_plug <= alpha).astype(float)
    feas_brier = float(np.mean((p_feas_pred - y_feas) ** 2))

    n_pos = int(np.sum(y_feas > 0.5))
    n_neg = int(y_feas.size - n_pos)
    if n_pos > 0 and n_neg > 0:
        pos_s = p_feas_pred[y_feas > 0.5][:, None]
        neg_s = p_feas_pred[y_feas < 0.5][None, :]
        auroc = float(np.mean((pos_s > neg_s) + 0.5 * (pos_s == neg_s)))
    else:
        auroc = None

    pred_u = np.maximum(sq_ref - mean_f, 0.0) * pred_gate
    sel_i = int(np.argmax(pred_u))
    # Also evaluate unbiased true Slot-0 gain using C_hat_U!
    sel_gain_plugin = float(max(anchor_true_obj - float(true_obj[sel_i]), 0.0) * float(plug_gate[sel_i]))
    sel_gain_unbiased = float(max(anchor_true_obj - float(true_obj[sel_i]), 0.0) * float(unb_gate_cu[sel_i]))

    sd_f_true = max(float(np.std(true_obj, ddof=1)), 1e-6)
    sd_g_true = max(float(np.std(true_con, ddof=1)), 1e-6)
    obj_nrmse = float(np.sqrt(np.mean(((mean_f - np.mean(mean_f)) - (true_obj - np.mean(true_obj))) ** 2)) / sd_f_true)
    con_nrmse = float(np.sqrt(np.mean(((mean_g - np.mean(mean_g)) - (true_con - np.mean(true_con))) ** 2)) / sd_g_true)

    return {
        "gate_mse_unbiased_x1e3": 1e3 * gate_mse_unbiased,
        "gate_power_rmse_plugin": gate_rmse_plugin,
        "feas_brier": feas_brier,
        "feas_auroc": auroc,
        "sel_slot0_gain_unbiased": sel_gain_unbiased,
        "sel_slot0_gain_plugin": sel_gain_plugin,
        "con_nrmse": con_nrmse,
        "obj_nrmse": obj_nrmse,
    }


def _compute_bootstrap_rgpe_weights(
    models: list[RawInputExactGP],
    Psi_obs: np.ndarray,
    y_obs: np.ndarray,
    tgt_gp: RawInputExactGP,
    rng: np.random.Generator,
    n_mc: int = 64,
) -> np.ndarray:
    """Faithful Feurer et al. (2018) pairwise ranking-loss bootstrap weights + LOO-CV."""
    if not models or len(y_obs) < 3:
        return np.r_[np.zeros(len(models)), 1.0]
    src_ms = [m.predict(Psi_obs) for m in models]
    t_loo_m, t_loo_s = tgt_gp.loocv_predict()
    all_means = np.asarray([ms[0] for ms in src_ms] + [t_loo_m], dtype=float)
    all_sds = np.asarray([ms[1] for ms in src_ms] + [t_loo_s], dtype=float)

    wins = np.zeros(len(models) + 1, dtype=float)
    n_obs = len(y_obs)
    for _ in range(n_mc):
        b_idx = rng.integers(0, n_obs, size=n_obs)
        yb = y_obs[b_idx]
        diff_y = yb[:, None] < yb[None, :]
        draws = all_means[:, b_idx] + all_sds[:, b_idx] * rng.normal(size=(len(models) + 1, n_obs))
        losses = np.empty(len(models) + 1, dtype=float)
        for m_i in range(len(models) + 1):
            diff_p = draws[m_i, :, None] < draws[m_i, None, :]
            losses[m_i] = float(np.mean(diff_y ^ diff_p))
        best_l = float(np.min(losses))
        tied = np.where(np.abs(losses - best_l) <= 1e-12)[0]
        wins[tied] += 1.0 / float(tied.size)
    w = wins / float(n_mc)
    keep = w >= 0.5 * max(float(w[-1]), 0.05)
    keep[-1] = True
    w = np.where(keep, w, 0.0)
    s_w = float(np.sum(w))
    return w / s_w if s_w > 1e-12 else np.r_[np.zeros(len(models)), 1.0]


def _eval_confirm72_definitive_item(job: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    aud_path = out_dir / "stage2" / "items" / f"{job['job_id']}.json"
    if aud_path.is_file():
        return {"job_id": job["job_id"], "status": "cached"}

    fam_name = job["family"]
    t_idx = int(job["target_task_idx"])
    fam_spec = copy.deepcopy(_load_des_fam_specs()[fam_name])
    fam_spec["target_config"]["seed"] = int(fam_spec["target_config"]["seed"]) + 1_000_000
    n_ch = 2 if fam_name in {"Inventory_CrossHorizon", "Queueing_CrossHorizon"} else 1
    prob = build_family_tasks(fam_spec, "target")[t_idx]
    fc_pub_vec = _task_public_forecast_vector(prob)

    fam_b = out_dir / "bundles" / fam_name
    p_m3_orig = pickle.loads((fam_b / "prior_m3_orig.pkl").read_bytes())
    p_m3_strat = pickle.loads((fam_b / "prior_m3_strat.pkl").read_bytes())
    p_m4_bal = pickle.loads((fam_b / "prior_m4_bal.pkl").read_bytes())
    p_m6 = pickle.loads((fam_b / "prior_m6.pkl").read_bytes())
    p_m12 = pickle.loads((fam_b / "prior_m12.pkl").read_bytes())
    p_nodeb = pickle.loads((fam_b / "prior_no_debias.pkl").read_bytes())
    p_nolp = pickle.loads((fam_b / "prior_no_lowpass.pkl").read_bytes())
    p_noorth = pickle.loads((fam_b / "prior_no_ortho.pkl").read_bytes())
    p_nosprs = pickle.loads((fam_b / "prior_no_sparse.pkl").read_bytes())
    p_drop7 = pickle.loads((fam_b / "prior_dropvol7d.pkl").read_bytes())
    p_gen_phys = pickle.loads((fam_b / "prior_generic_phys_only.pkl").read_bytes())
    p_gen_full = pickle.loads((fam_b / "prior_generic_forecast_full.pkl").read_bytes())
    ablr_b = pickle.loads((fam_b / "ablr_m6.pkl").read_bytes())
    mtgp_b = pickle.loads((fam_b / "mtgp_m6.pkl").read_bytes())
    rgpe_b = pickle.loads((fam_b / "rgpe_m6.pkl").read_bytes())

    seed_base = 3_100_001 + 10_007 * t_idx
    rng_init = np.random.default_rng(seed_base)
    rng_pool = np.random.default_rng(seed_base + 77)
    rng_probe = np.random.default_rng(seed_base + 199)
    rng_truth = np.random.default_rng(seed_base + 500_011)

    init_pts = build_cold_dct_initial_points(prob, 8, rng_init)
    init_recs = []
    ref_mod = TemperedHierarchicalHVDModel(
        p_m6, mode="tempered_sure_release",
        use_source_basis=True, use_source_coef_prior=False, use_source_hvd_prior=False,
    )
    for x0 in init_pts:
        e0 = prob.nominal_exposure(x0)
        s0 = prob.simulate_replications(x0, 2, rng_init, split="search")
        rc = SufficientStatsRecord.from_samples(x0, s0, e0)
        init_recs.append(rc)
        ref_mod.observe(rc)

    cands_pool, exps_pool = generate_dynamic_candidate_pool(
        prob, ref_mod, p_m6, rng_pool, pool_size=160, enable_source_templates=False
    )
    probe_idx, held_idx = _select_deterministic_probes_and_heldout(
        p_m6, init_recs, cands_pool, exps_pool, n_probes=4, n_heldout=32
    )
    probe_recs = [
        SufficientStatsRecord.from_samples(
            cands_pool[pi], prob.simulate_replications(cands_pool[pi], 2, rng_probe, split="search"), exps_pool[pi]
        )
        for pi in probe_idx
    ]
    held_x = [cands_pool[hi] for hi in held_idx]
    held_e = [exps_pool[hi] for hi in held_idx]

    s_anc = prob.simulate_replications(prob.domain_heuristic_anchor(), 400, rng_truth, split="audit")
    anc_true_obj = float(np.mean(s_anc[:, 0]))
    true_obj = np.empty(len(held_x), dtype=float)
    true_con = np.empty(len(held_x), dtype=float)
    true_pv = np.empty(len(held_x), dtype=float)
    true_vcnt = np.empty(len(held_x), dtype=float)
    for i, hx in enumerate(held_x):
        sh = prob.simulate_replications(hx, 400, rng_truth, split="audit")
        true_obj[i] = float(np.mean(sh[:, 0]))
        true_con[i] = float(np.mean(sh[:, 1]))
        vc = int(np.sum(sh[:, 1] > 0.0))
        true_vcnt[i] = float(vc)
        true_pv[i] = float(vc) / 400.0

    sq_ref = float(init_recs[0].mean_obj)

    def _score_gpfta(pr_obj: DebiasedSpectralMetaPrior, mode_str: str, use_coef: bool, exp_fn=None) -> dict[str, float | None]:
        agg: dict[str, list[float]] = {}
        for prb in probe_recs:
            recs9 = list(init_recs) + [prb]
            if exp_fn is not None:
                recs9 = [SufficientStatsRecord(
                    x_norm=r.x_norm, n_reps=r.n_reps, mean_obj=r.mean_obj, ssq_obj=r.ssq_obj,
                    mean_con=r.mean_con, ssq_con=r.ssq_con, violation_count=r.violation_count,
                    exposure=exp_fn(r.x_norm, r.exposure),
                ) for r in recs9]
                he_eval = [exp_fn(hx, he) for hx, he in zip(held_x, held_e)]
            else:
                he_eval = held_e

            if mode_str == "matched_coefficients":
                p_inst = copy.deepcopy(pr_obj)
                p_inst.mu_0_obj_[0] = float(np.mean([r.mean_obj for r in recs9[:8]]))
                p_inst.mu_0_con_[0] = float(np.mean([r.mean_con for r in recs9[:8]]))
                m = TemperedHierarchicalHVDModel(
                    p_inst, mode="matched_coefficients", coefficient_prior="historical_full",
                    use_source_basis=True, use_source_hvd_prior=False, use_empirical_r_perp=False,
                )
                m.freeze_matched_scales(recs9[:8])
                m.tau_shift_obj_ = 1e-6 * m.tau_shift_obj_
                m.tau_shift_con_ = 1e-6 * m.tau_shift_con_
            else:
                m = TemperedHierarchicalHVDModel(
                    pr_obj, mode=mode_str, use_source_basis=True, use_source_coef_prior=use_coef,
                    use_source_hvd_prior=False, use_empirical_r_perp=use_coef, n0_freeze=8,
                )
            for r in recs9:
                m.observe(r)
            pd = m.predict_joint(held_x, he_eval)
            met = _eval_heldout_metrics_definitive(
                pd.mean_obj, pd.std_obj, pd.mean_con, pd.std_con,
                pd.ucb_aleatoric_var_con, pd.r_perp_con, sq_ref,
                true_obj, true_con, true_pv, true_vcnt, anc_true_obj,
            )
            for k, v in met.items():
                if v is not None:
                    agg.setdefault(k, []).append(float(v))
        return {k: (float(np.mean(vals)) if vals else None) for k, vals in agg.items()}

    def _score_cold_psi() -> dict[str, float | None]:
        agg: dict[str, list[float]] = {}
        for prb in probe_recs:
            m = TemperedHierarchicalHVDModel(
                None, mode="target_only",
                use_source_basis=False, use_source_coef_prior=False, use_source_hvd_prior=False,
            )
            for r in list(init_recs) + [prb]:
                m.observe(r)
            pd = m.predict_joint(held_x, held_e)
            met = _eval_heldout_metrics_definitive(
                pd.mean_obj, pd.std_obj, pd.mean_con, pd.std_con,
                pd.ucb_aleatoric_var_con, pd.r_perp_con, sq_ref,
                true_obj, true_con, true_pv, true_vcnt, anc_true_obj,
            )
            for k, v in met.items():
                if v is not None:
                    agg.setdefault(k, []).append(float(v))
        return {k: (float(np.mean(vals)) if vals else None) for k, vals in agg.items()}

    def _score_external_meta(kind_ext: str) -> dict[str, float | None]:
        agg: dict[str, list[float]] = {}
        rng_rgpe = np.random.default_rng(seed_base + 4242)
        for prb in probe_recs:
            recs9 = list(init_recs) + [prb]
            m_hvd = TemperedHierarchicalHVDModel(
                None, mode="target_only", use_source_basis=False, use_source_coef_prior=False, use_source_hvd_prior=False,
            )
            for r in recs9:
                m_hvd.observe(r)
            pd_hvd = m_hvd.predict_joint(held_x, held_e)

            if kind_ext == "EB-LinearDict":
                mf, sf, mg, sg = _predict_ablr(ablr_b, recs9, held_e, pd_hvd.aleatoric_var_obj, pd_hvd.aleatoric_var_con)
            elif kind_ext == "PooledSourceResGP":
                src_f, src_g, pm, ps, _ = mtgp_b
                Po = np.clip((np.array([r.exposure.psi for r in recs9]) - pm) / ps, -4.0, 4.0)
                Pc = np.clip((np.array([e.psi for e in held_e]) - pm) / ps, -4.0, 4.0)
                yf = np.array([r.mean_obj for r in recs9])
                yg = np.array([r.mean_con for r in recs9])
                vf = np.array([max(r.ssq_obj / max(r.n_reps - 1, 1), 1e-5) / max(r.n_reps, 1) for r in recs9])
                vg = np.array([max(r.ssq_con / max(r.n_reps - 1, 1), 1e-5) / max(r.n_reps, 1) for r in recs9])
                msf_o, _ = src_f.predict(Po)
                msg_o, _ = src_g.predict(Po)
                df_gp = RawInputExactGP().fit(Po, yf - msf_o, vf)
                dg_gp = RawInputExactGP().fit(Po, yg - msg_o, vg)
                msf_c, ssf_c = src_f.predict(Pc)
                msg_c, ssg_c = src_g.predict(Pc)
                mdf_c, sdf_c = df_gp.predict(Pc)
                mdg_c, sdg_c = dg_gp.predict(Pc)
                mf, sf = msf_c + mdf_c, np.sqrt(sdf_c ** 2 + 0.20 * (ssf_c ** 2))
                mg, sg = msg_c + mdg_c, np.sqrt(sdg_c ** 2 + 0.20 * (ssg_c ** 2))
            elif kind_ext in {"RGPE-BootstrapRank", "Fixed50-Ensemble"}:
                sm_f, sm_g, pm, ps, _, _, _ = rgpe_b
                Po = np.clip((np.array([r.exposure.psi for r in recs9]) - pm) / ps, -4.0, 4.0)
                Pc = np.clip((np.array([e.psi for e in held_e]) - pm) / ps, -4.0, 4.0)
                yf = np.array([r.mean_obj for r in recs9])
                yg = np.array([r.mean_con for r in recs9])
                vf = np.array([max(r.ssq_obj / max(r.n_reps - 1, 1), 1e-5) / max(r.n_reps, 1) for r in recs9])
                vg = np.array([max(r.ssq_con / max(r.n_reps - 1, 1), 1e-5) / max(r.n_reps, 1) for r in recs9])
                tf_gp = RawInputExactGP().fit(Po, yf, vf)
                tg_gp = RawInputExactGP().fit(Po, yg, vg)
                if kind_ext == "RGPE-BootstrapRank":
                    wf = _compute_bootstrap_rgpe_weights(sm_f, Po, yf, tf_gp, rng_rgpe, n_mc=64)
                    wg = _compute_bootstrap_rgpe_weights(sm_g, Po, yg, tg_gp, rng_rgpe, n_mc=64)
                else:
                    n_s = max(len(sm_f), 1)
                    wf = np.r_[np.full(len(sm_f), 0.5 / n_s), 0.5]
                    wg = np.r_[np.full(len(sm_g), 0.5 / n_s), 0.5]

                preds_f = [m.predict(Pc) for m in sm_f] + [tf_gp.predict(Pc)]
                preds_g = [m.predict(Pc) for m in sm_g] + [tg_gp.predict(Pc)]
                mf = sum(w_i * pr[0] for w_i, pr in zip(wf, preds_f))
                sf = np.sqrt(np.maximum(sum((w_i ** 2) * (pr[1] ** 2) for w_i, pr in zip(wf, preds_f)), 1e-12))
                mg = sum(w_i * pr[0] for w_i, pr in zip(wg, preds_g))
                sg = np.sqrt(np.maximum(sum((w_i ** 2) * (pr[1] ** 2) for w_i, pr in zip(wg, preds_g)), 1e-12))
            else:
                raise ValueError(kind_ext)

            met = _eval_heldout_metrics_definitive(
                mf, sf, mg, sg, pd_hvd.ucb_aleatoric_var_con, np.zeros_like(mg),
                sq_ref, true_obj, true_con, true_pv, true_vcnt, anc_true_obj,
            )
            for k, v in met.items():
                if v is not None:
                    agg.setdefault(k, []).append(float(v))
        return {k: (float(np.mean(vals)) if vals else None) for k, vals in agg.items()}

    # Exact omega_M = p / (M + p) variants!
    v_m3_orig = _build_prior_variants_exact_m(p_m3_orig, seed=seed_base + 31, m_src=3)
    v_m3_strat = _build_prior_variants_exact_m(p_m3_strat, seed=seed_base + 32, m_src=3)
    v_m4_bal = _build_prior_variants_exact_m(p_m4_bal, seed=seed_base + 33, m_src=4)
    v_m6 = _build_prior_variants_exact_m(p_m6, seed=seed_base + 34, m_src=6)
    v_m12 = _build_prior_variants_exact_m(p_m12, seed=seed_base + 35, m_src=12)

    rng_fc = np.random.default_rng(seed_base + 919)
    fc_mult = np.clip(rng_fc.normal(1.0, 0.15, size=512), 0.55, 1.45)

    def _gen_phys_only_fn(x: np.ndarray, e_orig: NominalTrajectoryExposure) -> NominalTrajectoryExposure:
        eg = compute_generic_blackbox_exposure(x, num_channels=n_ch, pub_load_profile=fc_pub_vec)
        return NominalTrajectoryExposure(psi=eg.psi, A_local=e_orig.A_local, N_shared=e_orig.N_shared)

    def _gen_full_fn(x: np.ndarray, _e_orig: NominalTrajectoryExposure) -> NominalTrajectoryExposure:
        return compute_generic_blackbox_exposure(x, num_channels=n_ch, pub_load_profile=fc_pub_vec)

    models_res = {
        "Online-FullSURE": _score_gpfta(p_m6, "tempered_sure_release", True),
        "Online-DiagSURE": _score_gpfta(v_m6["online_diag"], "tempered_sure_release", True),
        "Online-SameBasisWeak": _score_gpfta(p_m6, "source_basis_weak_prior", False),
        "ColdPsi-GP": _score_cold_psi(),
        "EB-LinearDict-on-Psi": _score_external_meta("EB-LinearDict"),
        "PooledSourceResGP-on-Psi": _score_external_meta("PooledSourceResGP"),
        "RGPE-BootstrapRank-on-Psi": _score_external_meta("RGPE-BootstrapRank"),
        "Fixed50-Ensemble-on-Psi": _score_external_meta("Fixed50-Ensemble"),
        "Comp-NoLowPass": _score_gpfta(p_nolp, "tempered_sure_release", True),
        "Comp-NoDebias": _score_gpfta(p_nodeb, "tempered_sure_release", True),
        "Comp-NoOrtho": _score_gpfta(p_noorth, "tempered_sure_release", True),
        "Comp-NoSparse": _score_gpfta(p_nosprs, "tempered_sure_release", True),
        "Psi-DropVol7D": _score_gpfta(p_drop7, "tempered_sure_release", True, exp_fn=lambda _x, e: _drop_vol_exposure(e)),
        "Psi-GenericPhysOnly": _score_gpfta(p_gen_phys, "tempered_sure_release", True, exp_fn=_gen_phys_only_fn),
        "Psi-GenericForecastFull": _score_gpfta(p_gen_full, "tempered_sure_release", True, exp_fn=_gen_full_fn),
        "ForecastNoise15-FullSURE": _score_gpfta(p_m6, "tempered_sure_release", True, exp_fn=lambda x, _e: _corrupt_forecast_exposure(prob, x, fc_mult)),
    }

    msrc_res = {
        "M3_Orig_OnlineFull": _score_gpfta(p_m3_orig, "tempered_sure_release", True),
        "M3_Orig_KernelFull": _score_gpfta(v_m3_orig["zm_full"], "matched_coefficients", True),
        "M3_Orig_KernelDiag": _score_gpfta(v_m3_orig["zm_diag"], "matched_coefficients", True),
        "M3_Strat_OnlineFull": _score_gpfta(p_m3_strat, "tempered_sure_release", True),
        "M3_Strat_KernelFull": _score_gpfta(v_m3_strat["zm_full"], "matched_coefficients", True),
        "M3_Strat_KernelDiag": _score_gpfta(v_m3_strat["zm_diag"], "matched_coefficients", True),
        "M4_Bal_OnlineFull": _score_gpfta(p_m4_bal, "tempered_sure_release", True),
        "M4_Bal_KernelFull": _score_gpfta(v_m4_bal["zm_full"], "matched_coefficients", True),
        "M4_Bal_KernelDiag": _score_gpfta(v_m4_bal["zm_diag"], "matched_coefficients", True),
        "M4_Bal_KernelIso": _score_gpfta(v_m4_bal["zm_iso"], "matched_coefficients", True),
        "M6_OnlineFull": models_res["Online-FullSURE"],
        "M6_KernelFull": _score_gpfta(v_m6["zm_full"], "matched_coefficients", True),
        "M6_KernelDiag": _score_gpfta(v_m6["zm_diag"], "matched_coefficients", True),
        "M6_KernelIso": _score_gpfta(v_m6["zm_iso"], "matched_coefficients", True),
        "M12_OnlineFull": _score_gpfta(p_m12, "tempered_sure_release", True),
        "M12_KernelFull": _score_gpfta(v_m12["zm_full"], "matched_coefficients", True),
        "M12_KernelDiag": _score_gpfta(v_m12["zm_diag"], "matched_coefficients", True),
        "M12_KernelIso": _score_gpfta(v_m12["zm_iso"], "matched_coefficients", True),
    }

    out = {
        "job_id": job["job_id"],
        "kind": "confirm72_definitive",
        "family": fam_name,
        "target_task_idx": t_idx,
        "models": models_res,
        "msrc_exact_omega": msrc_res,
        "truth_used": True,
    }
    _write_json(aud_path, out)
    return {"job_id": job["job_id"], "status": "ok"}


def _run_stage2_item(job: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    if job["kind"] == "confirm72_definitive":
        return _eval_confirm72_definitive_item(job, out_dir)

    aud_path = out_dir / "stage2" / "items" / f"{job['job_id']}.json"
    if aud_path.is_file():
        return {"job_id": job["job_id"], "status": "cached"}

    fams = {f["name"]: f for f in _load_active_family_specs()}
    fam_spec = fams[job["family"]]
    t_idx = int(job["target_task_idx"])
    prob = build_family_tasks(fam_spec, "target")[t_idx]
    audit_seed = int(fam_spec.get("audit_seed_base", 2056112001)) + 997 * t_idx
    n_eval = int(fam_spec.get("audit_replications", 2000))

    x_anc = prob.domain_heuristic_anchor()
    anc_aud = prob.audit_ground_truth(x_anc, n_eval=n_eval, seed=audit_seed, split="audit")
    anc_obj = float(anc_aud["mc_objective_mean"])
    anc_viol = float(anc_aud["mc_violation_prob"])

    def _audit_pol(pol_list: list[float] | None) -> dict[str, Any]:
        if pol_list is None:
            return {
                "certified": False,
                "status_quo_loss": anc_obj,
                "operating_viol_prob": anc_viol,
                "deployed_viol_ucb95": None,
                "verif_pool_viol_prob": None,
            }
        xp = np.asarray(pol_list, dtype=float)
        if np.allclose(xp, x_anc, atol=1e-10, rtol=0.0):
            da = dict(anc_aud)
        else:
            da = prob.audit_ground_truth(xp, n_eval=n_eval, seed=audit_seed, split="audit")
        vp = None
        if fam_spec.get("kind") == "opsd" or job["family"] == "OPSD_EuroGrid":
            va = prob.audit_ground_truth(xp, n_eval=n_eval, seed=audit_seed + 500_009, split="verification")
            vp = float(va["mc_violation_prob"])
        return {
            "certified": True,
            "status_quo_loss": float(da["mc_objective_mean"]),
            "operating_viol_prob": float(da["mc_violation_prob"]),
            "deployed_viol_ucb95": float(da["mc_viol_ci95_hi"]),
            "verif_pool_viol_prob": vp,
        }

    rep_s1 = _read_json(out_dir / "stage1" / "items" / f"{job['replay_job_id']}.json")

    unv_info = rep_s1["term_only_allow_unvisited"]
    unv_aud = _audit_pol(unv_info["deployed_policy"])
    maxu_info = rep_s1["term_only_unthresh_maxu"]
    maxu_aud = _audit_pol(maxu_info["deployed_policy"])
    strm_info = rep_s1["streaming_curtailment_k1"]
    strm_aud = _audit_pol(strm_info["deployed_policy"])

    out = {
        "job_id": job["job_id"],
        "kind": "cell_audit",
        "family": job["family"],
        "target_task_idx": t_idx,
        "rep_seed": int(job["rep_seed"]),
        "anchor_obj": anc_obj,
        "anchor_viol_prob": anc_viol,
        "term_only_allow_unvisited": {
            "search_calls_used": int(rep_s1["search_calls_used"]),
            "verification_calls_used": int(unv_info["verification_calls_used"]),
            "total_target_calls": int(rep_s1["search_calls_used"]) + int(unv_info["verification_calls_used"]),
            "slot0_is_unvisited": bool(unv_info["slot0_is_unvisited"]),
            "any_slot_unvisited": bool(unv_info["any_slot_unvisited"]),
            **unv_aud,
        },
        "term_only_unthresh_maxu": {
            "search_calls_used": int(rep_s1["search_calls_used"]),
            "verification_calls_used": int(maxu_info["verification_calls_used"]),
            "total_target_calls": int(rep_s1["search_calls_used"]) + int(maxu_info["verification_calls_used"]),
            "nomination_wall_sec": float(maxu_info["nomination_wall_sec"]),
            "n_triples_scored": int(maxu_info["n_triples_scored"]),
            "same_slot0_as_default": bool(maxu_info["same_slot0_as_default"]),
            **maxu_aud,
        },
        "streaming_curtailment_k1": {
            **{k: v for k, v in strm_info.items() if k != "deployed_policy"},
            **strm_aud,
        },
        "truth_used": True,
    }
    _write_json(aud_path, out)
    return {"job_id": job["job_id"], "status": "ok"}


def run_worker(stage: str, out_dir: Path, start: int, end: int, workers: int) -> None:
    desc = _read_json(out_dir / stage / "jobs.json")
    sub = desc["jobs"][start:end]
    fn = _run_stage1_item if stage == "stage1" else _run_stage2_item
    if workers <= 1 or len(sub) <= 1:
        for jb in sub:
            r = fn(jb, out_dir)
            print(json.dumps(r), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(fn, jb, out_dir) for jb in sub]
            for f in as_completed(futs):
                print(json.dumps(f.result()), flush=True)


def summarize(out_dir: Path, summary_path: Path) -> dict[str, Any]:
    compact_path = summary_path.parent / "rev8_definitive_compact_inputs.json"
    if (out_dir / "stage2" / "jobs.json").is_file():
        s2_desc = _read_json(out_dir / "stage2" / "jobs.json")
        exec_sha = s2_desc["execution_code_sha256"]
        cell_audits = [
            _read_json(out_dir / "stage2" / "items" / f"{jb['job_id']}.json")
            for jb in s2_desc["jobs"] if jb["kind"] == "cell_audit"
        ]
        c72_items = [
            _read_json(out_dir / "stage2" / "items" / f"{jb['job_id']}.json")
            for jb in s2_desc["jobs"] if jb["kind"] == "confirm72_definitive"
        ]
        cell_audits.sort(key=lambda z: (z["family"], int(z["target_task_idx"]), int(z["rep_seed"])))
        c72_items.sort(key=lambda z: (z["family"], int(z["target_task_idx"])))
        # Save full-precision floats in compact_inputs so zero-sim summarize + cmp is 100% byte-identical!
        _write_json(compact_path, {
            "execution_code_sha256": exec_sha,
            "cell_audits": cell_audits,
            "confirm72_definitive": c72_items,
        })
    else:
        for cand_p in (
            out_dir / "rev8_definitive_compact_inputs.json",
            summary_path.parent / "rev8_definitive_compact_inputs.json",
            OUT_ROOT / "rev8_definitive_compact_inputs.json",
        ):
            if cand_p.is_file():
                compact_path = cand_p
                break
        comp = _read_json(compact_path)
        exec_sha = comp["execution_code_sha256"]
        cell_audits = comp["cell_audits"]
        c72_items = comp["confirm72_definitive"]

    # 1. Summarize single-factor ECKG-AllowUnvisited, TermOnly-UnthreshMaxU, and Streaming Curtailment
    sealed_ref = [r for r in _load_sealed_reference_runs() if r["method_name"] == "GP-FTA-ECKG"]
    ref_by_ft: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for r in sealed_ref:
        ref_by_ft.setdefault((r["family"], int(r["target_task_idx"])), []).append(r)

    cells_by_ft: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for c in cell_audits:
        cells_by_ft.setdefault((c["family"], int(c["target_task_idx"])), []).append(c)

    families = ["Inventory_CrossHorizon", "Queueing_CrossHorizon", "OPSD_EuroGrid", "Traffic_Ingolstadt"]
    single_factor_by_domain: dict[str, Any] = {}
    streaming_by_domain: dict[str, Any] = {}

    for fam in families:
        n_tgt = 14 if fam == "OPSD_EuroGrid" else 24
        ref_vec = np.array([np.mean([x["status_quo_loss"] for x in ref_by_ft[(fam, ti)]]) for ti in range(n_tgt)])
        unv_vec = np.array([np.mean([x["term_only_allow_unvisited"]["status_quo_loss"] for x in cells_by_ft[(fam, ti)]]) for ti in range(n_tgt)])
        mxu_vec = np.array([np.mean([x["term_only_unthresh_maxu"]["status_quo_loss"] for x in cells_by_ft[(fam, ti)]]) for ti in range(n_tgt)])

        all_cells_f = [x for ti in range(n_tgt) for x in cells_by_ft[(fam, ti)]]
        st_unv = _paired_stats(unv_vec - ref_vec)
        st_mxu = _paired_stats(mxu_vec - ref_vec)

        single_factor_by_domain[fam] = {
            "TermOnly-AllowUnvisited": {
                "mean_sq_loss": _stab_float(float(np.mean(unv_vec))),
                "mean_total_calls": _stab_float(float(np.mean([x["term_only_allow_unvisited"]["total_target_calls"] for x in all_cells_f]))),
                "mean_verif_calls": _stab_float(float(np.mean([x["term_only_allow_unvisited"]["verification_calls_used"] for x in all_cells_f]))),
                "cert_count": int(sum(1 for x in all_cells_f if x["term_only_allow_unvisited"]["certified"])),
                "n_runs": len(all_cells_f),
                "unvisited_slot0_runs": int(sum(1 for x in all_cells_f if x["term_only_allow_unvisited"]["slot0_is_unvisited"])),
                "unvisited_any_slot_runs": int(sum(1 for x in all_cells_f if x["term_only_allow_unvisited"]["any_slot_unvisited"])),
                "mean_operating_viol_prob": _stab_float(float(np.mean([x["term_only_allow_unvisited"]["operating_viol_prob"] for x in all_cells_f]))),
                "contrast_vs_eckg": st_unv,
            },
            "TermOnly-UnthreshMaxU": {
                "mean_sq_loss": _stab_float(float(np.mean(mxu_vec))),
                "mean_total_calls": _stab_float(float(np.mean([x["term_only_unthresh_maxu"]["total_target_calls"] for x in all_cells_f]))),
                "mean_verif_calls": _stab_float(float(np.mean([x["term_only_unthresh_maxu"]["verification_calls_used"] for x in all_cells_f]))),
                "mean_nomination_wall_sec": _stab_float(float(np.mean([x["term_only_unthresh_maxu"]["nomination_wall_sec"] for x in all_cells_f]))),
                "mean_triples_scored": _stab_float(float(np.mean([x["term_only_unthresh_maxu"]["n_triples_scored"] for x in all_cells_f]))),
                "same_slot0_runs": int(sum(1 for x in all_cells_f if x["term_only_unthresh_maxu"]["same_slot0_as_default"])),
                "cert_count": int(sum(1 for x in all_cells_f if x["term_only_unthresh_maxu"]["certified"])),
                "n_runs": len(all_cells_f),
                "mean_operating_viol_prob": _stab_float(float(np.mean([x["term_only_unthresh_maxu"]["operating_viol_prob"] for x in all_cells_f]))),
                "contrast_vs_eckg": st_mxu,
            },
        }

        fix_c = float(np.mean([x["streaming_curtailment_k1"]["fixed_loop_calls"] for x in all_cells_f]))
        cur_c = float(np.mean([x["streaming_curtailment_k1"]["streaming_curtailed_calls"] for x in all_cells_f]))
        fix_w = float(np.mean([x["streaming_curtailment_k1"]["fixed_loop_wall_sec"] for x in all_cells_f]))
        cur_w = float(np.mean([x["streaming_curtailment_k1"]["streaming_curtailed_wall_sec"] for x in all_cells_f]))
        streaming_by_domain[fam] = {
            "n_runs": len(all_cells_f),
            "exact_decision_match_runs": int(sum(1 for x in all_cells_f if x["streaming_curtailment_k1"]["exact_decision_match"])),
            "fixed_verif_calls": _stab_float(fix_c),
            "streaming_curtailed_calls": _stab_float(cur_c),
            "call_saving_pct": _stab_float(100.0 * (fix_c - cur_c) / max(fix_c, 1e-6)),
            "fixed_verif_wall_sec": _stab_float(fix_w),
            "streaming_curtailed_wall_sec": _stab_float(cur_w),
            "verif_wall_saving_pct": _stab_float(100.0 * (fix_w - cur_w) / max(fix_w, 1e-9)),
        }

    # Pooled DES (n=72) standardized by anchor SD
    pooled_sf: dict[str, Any] = {}
    for arm_key, sub_key in (("TermOnly-AllowUnvisited", "term_only_allow_unvisited"), ("TermOnly-UnthreshMaxU", "term_only_unthresh_maxu")):
        diffs_all = []
        for fam in DES_FAMILIES:
            anc_sd = float(np.std([cells_by_ft[(fam, ti)][0]["anchor_obj"] for ti in range(24)], ddof=1))
            ref_v = np.array([np.mean([x["status_quo_loss"] for x in ref_by_ft[(fam, ti)]]) for ti in range(24)])
            arm_v = np.array([np.mean([x[sub_key]["status_quo_loss"] for x in cells_by_ft[(fam, ti)]]) for ti in range(24)])
            diffs_all.append((arm_v - ref_v) / max(anc_sd, 1e-8))
        pooled_sf[arm_key] = _paired_stats(np.concatenate(diffs_all))

    # 2. Summarize confirm72_definitive (unbiased gate MSE, faithful RGPE, forecast-fed generic psi, exact-omega M_src)
    model_names = list(c72_items[0]["models"].keys())
    met_keys = (
        "gate_mse_unbiased_x1e3",
        "gate_power_rmse_plugin",
        "feas_brier",
        "feas_auroc",
        "sel_slot0_gain_unbiased",
        "sel_slot0_gain_plugin",
        "con_nrmse",
        "obj_nrmse",
    )
    pooled_models: dict[str, Any] = {}
    for m in model_names:
        m_rec: dict[str, Any] = {}
        for mk in met_keys:
            vals = [float(it["models"][m][mk]) for it in c72_items if it["models"][m].get(mk) is not None]
            m_rec[mk] = _stab_float(float(np.mean(vals))) if vals else None
            m_rec[f"{mk}_se"] = _stab_float(float(np.std(vals, ddof=1) / math.sqrt(len(vals)))) if len(vals) > 1 else None
            if mk == "feas_auroc":
                m_rec["feas_auroc_defined_tasks"] = len(vals)
        pooled_models[m] = m_rec

    meta_comps = [
        "ColdPsi-GP",
        "Online-SameBasisWeak",
        "EB-LinearDict-on-Psi",
        "PooledSourceResGP-on-Psi",
        "RGPE-BootstrapRank-on-Psi",
        "Fixed50-Ensemble-on-Psi",
    ]
    meta_primary5 = [
        "ColdPsi-GP",
        "Online-SameBasisWeak",
        "EB-LinearDict-on-Psi",
        "PooledSourceResGP-on-Psi",
        "RGPE-BootstrapRank-on-Psi",
    ]
    calib_contrasts: dict[str, Any] = {}
    for mk in ("gate_mse_unbiased_x1e3", "gate_power_rmse_plugin", "feas_brier", "con_nrmse", "sel_slot0_gain_unbiased", "feas_auroc"):
        blk: dict[str, Any] = {}
        for c in meta_comps:
            pairs = [
                (float(it["models"][c][mk]), float(it["models"]["Online-FullSURE"][mk]))
                for it in c72_items
                if it["models"][c].get(mk) is not None and it["models"]["Online-FullSURE"].get(mk) is not None
            ]
            diffs = np.array([a - b for a, b in pairs], dtype=float)
            blk[c] = _paired_stats(diffs)
        ps_m5 = [blk[c]["p_two_sided"] for c in meta_primary5]
        for c, hp in zip(meta_primary5, _holm_adjust(ps_m5)):
            blk[c]["p_holm_5meta"] = _stab_float(hp)

        # Separate 4-way Holm for the 4 implementation components + separate Holm for psi/forecast ablations
        comp_4 = ["Comp-NoLowPass", "Comp-NoDebias", "Comp-NoOrtho", "Comp-NoSparse"]
        psi_4 = ["Psi-DropVol7D", "Psi-GenericPhysOnly", "Psi-GenericForecastFull", "ForecastNoise15-FullSURE"]
        for grp, h_name in ((comp_4, "p_holm_4comp"), (psi_4, "p_holm_4psi")):
            ps_g = []
            for c in grp:
                pairs = [
                    (float(it["models"][c][mk]), float(it["models"]["Online-FullSURE"][mk]))
                    for it in c72_items
                    if it["models"][c].get(mk) is not None and it["models"]["Online-FullSURE"].get(mk) is not None
                ]
                st = _paired_stats(np.array([a - b for a, b in pairs], dtype=float))
                blk[c] = st
                ps_g.append(st["p_two_sided"])
            for c, hp in zip(grp, _holm_adjust(ps_g)):
                blk[c][h_name] = _stab_float(hp)
        calib_contrasts[mk] = blk

    # Exact-omega M_src scaling summary
    msrc_keys = list(c72_items[0]["msrc_exact_omega"].keys())
    msrc_levels: dict[str, Any] = {}
    for mk_s in msrc_keys:
        msrc_levels[mk_s] = {
            met: _stab_float(float(np.mean([
                it["msrc_exact_omega"][mk_s].get(met)
                for it in c72_items if it["msrc_exact_omega"][mk_s].get(met) is not None
            ])))
            for met in ("con_nrmse", "obj_nrmse", "gate_mse_unbiased_x1e3", "gate_power_rmse_plugin")
        }

    msrc_contrasts: dict[str, Any] = {}
    for tag in ("M3_Orig", "M3_Strat", "M4_Bal", "M6", "M12"):
        df_con = np.array([
            float(it["msrc_exact_omega"][f"{tag}_KernelDiag"]["con_nrmse"])
            - float(it["msrc_exact_omega"][f"{tag}_KernelFull"]["con_nrmse"])
            for it in c72_items
        ])
        df_obj = np.array([
            float(it["msrc_exact_omega"][f"{tag}_KernelDiag"]["obj_nrmse"])
            - float(it["msrc_exact_omega"][f"{tag}_KernelFull"]["obj_nrmse"])
            for it in c72_items
        ])
        cold_vs_onl_mse = np.array([
            float(it["models"]["ColdPsi-GP"]["gate_mse_unbiased_x1e3"])
            - float(it["msrc_exact_omega"][f"{tag}_OnlineFull"]["gate_mse_unbiased_x1e3"])
            for it in c72_items
        ])
        rec_c: dict[str, Any] = {
            "diag_minus_full_con_nrmse": _paired_stats(df_con),
            "diag_minus_full_obj_nrmse": _paired_stats(df_obj),
            "cold_minus_online_unbiased_gate_mse_x1e3": _paired_stats(cold_vs_onl_mse),
        }
        if f"{tag}_KernelIso" in msrc_levels:
            iso_full_con = np.array([
                float(it["msrc_exact_omega"][f"{tag}_KernelIso"]["con_nrmse"])
                - float(it["msrc_exact_omega"][f"{tag}_KernelFull"]["con_nrmse"])
                for it in c72_items
            ])
            rec_c["iso_minus_full_con_nrmse"] = _paired_stats(iso_full_con)
        msrc_contrasts[tag] = rec_c

    def _stabilize_rec(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _stabilize_rec(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_stabilize_rec(v) for v in obj]
        if isinstance(obj, (float, np.floating)):
            v = float(obj)
            return 0.0 if abs(v) == 0.0 else float(f"{v:.12g}")
        return obj

    readout = _stabilize_rec({
        "protocol_id": "PREREGISTRATION_REV8_DEFINITIVE_AUDIT_20261121",
        "execution_code_sha256": exec_sha,
        "single_factor_counterfactuals": {
            "by_domain": single_factor_by_domain,
            "pooled_des_72_anchor_sd": pooled_sf,
        },
        "live_streaming_curtailment_k1": streaming_by_domain,
        "confirm72_definitive_calibration": {
            "pooled_means": pooled_models,
            "contrasts_vs_online_full_sure": calib_contrasts,
        },
        "msrc_exact_omega_scaling": {
            "levels": msrc_levels,
            "contrasts": msrc_contrasts,
        },
    })
    _write_json(summary_path, readout)
    return readout


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("prepare_priors").add_argument("--out-dir", type=Path, default=OUT_ROOT)

    p_d = sub.add_parser("describe")
    p_d.add_argument("--stage", choices=["stage1", "stage2"], required=True)
    p_d.add_argument("--out-dir", type=Path, default=OUT_ROOT)

    p_w = sub.add_parser("worker")
    p_w.add_argument("--stage", choices=["stage1", "stage2"], required=True)
    p_w.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    p_w.add_argument("--start", type=int, required=True)
    p_w.add_argument("--end", type=int, required=True)
    p_w.add_argument("--workers", type=int, default=1)

    p_s = sub.add_parser("summarize")
    p_s.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    p_s.add_argument("--summary-path", type=Path, default=OUT_ROOT / "rev8_definitive_readout.json")

    args = p.parse_args()
    if args.cmd == "prepare_priors":
        print(json.dumps(prepare_priors(args.out_dir)))
    elif args.cmd == "describe":
        d = describe_stage(args.stage, args.out_dir)
        print(json.dumps({"stage": d["stage"], "n_jobs": d["n_jobs"]}))
    elif args.cmd == "worker":
        run_worker(args.stage, args.out_dir, args.start, args.end, args.workers)
    elif args.cmd == "summarize":
        ro = summarize(args.out_dir, args.summary_path)
        print(json.dumps({
            "pooled_single_factor": {k: v["mean_diff"] for k, v in ro["single_factor_counterfactuals"]["pooled_des_72_anchor_sd"].items()},
            "unbiased_gate_mse_fullsure": ro["confirm72_definitive_calibration"]["pooled_means"]["Online-FullSURE"]["gate_mse_unbiased_x1e3"],
        }))


if __name__ == "__main__":
    main()
