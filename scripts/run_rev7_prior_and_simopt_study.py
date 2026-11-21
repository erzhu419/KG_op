#!/usr/bin/env python3
"""Cluster study R7-2: Decision-aligned posterior calibration, Four-Pillar ablation,
M_src / shift / forecast-noise scaling, frozen-state LatentNEI & hyperparameter sensitivity,
and external SimOpt-style Tandem Queueing + Continuous Newsvendor generic-psi benchmark.

Enforces strict stage barriers without touching `SC-OLH-KG/v2_or` (`fe8df9098424...`):
  - `stage0`: Generate M_src=12 extension source items (3 DES families x 6 extra tasks x 4 solvers)
              and SimOpt_TandemCntNv source items (6 tasks x 4 solvers), then barrier-fit priors.
  - `stage1`: Closed-loop search & verification on 24 unseen SimOpt_TandemCntNv target tasks
              (2 streams x 5 arms = 240 runs, `truth_used = False`) + 24-state frozen LatentNEI
              & hyperparameter sensitivity across 24 Sobol scrambles (`truth_used = False`).
  - `stage2`: Post-hoc 2,000-draw ground-truth audit of SimOpt_TandemCntNv runs +
              72-task `confirm_fresh_72` decision-aligned calibration, Four-Pillar & generic-psi
              ablations, meta-BO baselines (`ABLR`, `MTGP-ICM`, `RGPE`), and M_src={3,6,12} /
              shift_scale={1.0,1.3,1.6} / public-forecast-noise={0,15%} evaluations.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
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
from scipy.special import ndtr, ndtri
from scipy.stats import binom, norm, qmc, spearmanr, t as student_t, wilcoxon

REPO_ROOT = Path(__file__).resolve().parents[1]
SC_ROOT = REPO_ROOT / "SC-OLH-KG"
if str(SC_ROOT) not in sys.path:
    sys.path.insert(0, str(SC_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_direct_propagation_study import (
    _build_prior_variants,
    _select_deterministic_probes_and_heldout,
    _stabilized_cov,
)
from run_eckg_multiscramble_cei_audit import _eval_rosters_ref, _make_ref_qmc
from run_rev7_closedloop_study import (
    _holm_adjust,
    _paired_stats,
    _stab_float,
    build_direct_max_u_shortlist,
    pretrain_mtgp_source_psi,
    run_gpfta_variant,
)
from v2_or.acquisition.verification_kg import (
    build_balanced_warmstart_points,
    build_cold_dct_initial_points,
    generate_dynamic_candidate_pool,
    run_tls_olh_kg,
    score_eligible_cascade_kg,
)
from v2_or.baselines.sota_baselines import (
    RawInputExactGP,
    _archive_row_noise_vars,
    pretrain_rgpe_psi_source_ensemble,
    run_scbo_baseline,
)
from v2_or.contracts import (
    BaseDynamicORProblem,
    LegacyTrajectoryRecord,
    NominalTrajectoryExposure,
    SufficientStatsRecord,
)
from v2_or.dispatch.campaign_runner import build_family_tasks, compute_v2_or_code_sha256
from v2_or.legacy_solvers import LEGACY_SOLVER_REGISTRY, load_legacy_archive_json
from v2_or.models.tempered_hvd_posterior import TemperedHierarchicalHVDModel, _safe_cholesky
from v2_or.priors.debiased_spectral_prior import DebiasedSpectralMetaPrior, _build_raw_dictionary
from v2_or.verification.gated_verifier_v2 import (
    build_feasible_aware_shortlist,
    exact_binomial_pass_power,
    posterior_binomial_gate_power,
)

OUT_ROOT = SC_ROOT / "paper_artifacts" / "v2_or_rev7_or_panel_v1" / "prior_and_simopt"
PROSP_DIR = SC_ROOT / "paper_artifacts" / "v2_or_eckg_sure_confirm_v1"
U3_BUNDLES = SC_ROOT / "paper_artifacts" / "v2_or_u3_confirm_v1" / "source_bundles"
DES_FAMILIES = ("Inventory_CrossHorizon", "Queueing_CrossHorizon", "Traffic_Ingolstadt")
SOLVER_NAMES = ("penalty_spsa", "constrained_cmaes", "constrained_nsga2", "heuristic_powell")
SIMOPT_ARMS = (
    "GP-FTA-ECKG",
    "GP-FTA-FeasibleKG",
    "GP-FTA-LatentNEI-3Slot",
    "SCBO-3Slot",
    "PublicPsi-TargetGP-TS",
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compute_generic_blackbox_exposure(
    x_norm: np.ndarray,
    num_channels: int,
    pub_load_profile: np.ndarray | None = None,
) -> NominalTrajectoryExposure:
    """Simulator-free black-box 12-coordinate control summary `psi_generic(x) in R^12`.

    Uses ONLY normalized control-channel profiles and an optional public load vector,
    with ZERO simulator fluid equations or internal physical parameters.
    """
    arr = np.clip(np.asarray(x_norm, dtype=float).reshape(-1), 0.0, 1.0)
    c = max(int(num_channels), 1)
    w_len = max(arr.size // c, 1)
    mat = arr[: w_len * c].reshape(w_len, c)

    if pub_load_profile is None or len(pub_load_profile) == 0:
        load = 1.0 + 0.25 * np.sin(np.linspace(0.0, 2.0 * np.pi, w_len))
    else:
        raw_l = np.asarray(pub_load_profile, dtype=float).reshape(-1)
        if raw_l.size != w_len:
            idx = np.linspace(0, raw_l.size - 1, w_len)
            load = np.interp(idx, np.arange(raw_l.size), raw_l)
        else:
            load = raw_l.copy()
    load_norm = load / max(float(np.mean(load)), 1e-6)

    ch_means = np.mean(mat, axis=0)
    m0 = float(ch_means[0])
    m1 = float(ch_means[1 % c])
    m2 = float(np.mean(ch_means))
    p15 = float(np.quantile(np.mean(mat, axis=1), 0.15))
    p85 = float(np.quantile(np.mean(mat, axis=1), 0.85))
    ch_disp = float(np.mean(np.std(mat, axis=1))) if c > 1 else float(np.std(mat[:, 0]))
    margin_q15 = float(np.quantile(np.mean(mat, axis=1) - 0.45 * load_norm, 0.15))

    diffs = np.diff(mat, axis=0) if w_len > 1 else np.zeros_like(mat)
    ramp_rms = float(np.sqrt(np.mean(diffs ** 2)))
    lean_vec = np.maximum(0.65 - np.mean(mat, axis=1), 0.0)
    lean_rms = float(np.sqrt(np.mean(lean_vec ** 2)))
    low_head_rms = float(np.sqrt(np.mean(np.maximum(0.35 - np.min(mat, axis=1), 0.0) ** 2)))
    var_rms = float(np.sqrt(np.mean((mat - np.mean(mat, axis=0, keepdims=True)) ** 2)))

    q1, q2 = float(np.quantile(load_norm, 1.0 / 3.0)), float(np.quantile(load_norm, 2.0 / 3.0))
    masks = [load_norm <= q1, (load_norm > q1) & (load_norm <= q2), load_norm > q2]
    n_shr = []
    for mk in masks:
        if not np.any(mk):
            mk = np.ones(w_len, dtype=bool)
        n_shr.append(float(np.mean(load_norm[mk] * (1.05 - np.mean(mat[mk], axis=1)))))

    a_loc = np.array([lean_rms, ramp_rms, low_head_rms, var_rms], dtype=float)
    n_arr = np.maximum(np.asarray(n_shr, dtype=float), 0.0)
    psi = np.array([
        m0, m1, m2, p15, p85, ch_disp, margin_q15,
        a_loc[0], a_loc[1], n_arr[0], n_arr[1], n_arr[2],
    ], dtype=float)
    return NominalTrajectoryExposure(psi=psi, A_local=a_loc, N_shared=n_arr)


class SimOptTandemCntNvBenchmark(BaseDynamicORProblem):
    """Published-literature Multi-Period Tandem Queueing + Continuous Newsvendor Benchmark
    (Koenig & Law 1985; Hong, Luo & Nelson 2015; Eckman et al. 2023 SimOpt conventions).

    Optimizes 4-channel window controls `[mu_1, mu_2, mu_3, Q_nv]` across `W` windows
    (`d = 4 * W`, `d_s in {24, 32}` -> `d_t in {48, 64}`) subject to an SLA exceedance
    chance constraint `P(Y_g > 0) <= 0.05`. Crucially, `nominal_exposure` uses ONLY
    `compute_generic_blackbox_exposure` (zero internal tandem simulator equations).
    """

    family_name = "SimOpt_TandemCntNv"
    num_control_channels = 4
    alpha = 0.05
    is_gaussian_constraint_noise = False

    def __init__(self, *, task_id: str, n_windows: int, seed: int, shift_scale: float = 1.0) -> None:
        super().__init__()
        self.task_id = str(task_id)
        self.n_windows = int(n_windows)
        self.d = 4 * self.n_windows
        self.shift_scale = float(shift_scale)

        rng = np.random.default_rng(int(seed))
        t_grid = np.linspace(0.0, 2.0 * np.pi, self.n_windows, endpoint=False)
        base_rate = rng.uniform(9.5, 13.5)
        amp1 = rng.uniform(0.18, 0.34)
        phase1 = rng.uniform(0.0, 2.0 * np.pi)
        self.pub_arrival_forecast = base_rate * (1.0 + amp1 * np.sin(t_grid + phase1) + 0.10 * np.cos(2.0 * t_grid))
        self.mu_min = base_rate * 0.72
        self.mu_max = base_rate * 1.95
        self.q_min = base_rate * 0.65
        self.q_max = base_rate * 1.75
        self.station_cost_weights = rng.uniform(0.85, 1.25, size=3)
        self.nv_unit_cost = float(rng.uniform(0.55, 0.80))
        self.sla_budget = float(rng.uniform(1.25, 1.52))

        # Private stochastic shock parameters (never visible to nominal_exposure!)
        self.priv_drift = rng.uniform(0.03, 0.09, size=3) * self.shift_scale
        self.priv_sigma = rng.uniform(0.16, 0.25) * np.sqrt(self.shift_scale)
        self.service_cv = float(rng.uniform(0.22, 0.32) * np.sqrt(self.shift_scale))
        self.buffer_cap = base_rate * 1.65

    def domain_heuristic_anchor(self) -> np.ndarray:
        """Conservative peak-buffered staffing & safety-stock status-quo policy."""
        anc = np.empty((self.n_windows, 4), dtype=float)
        rel = self.pub_arrival_forecast / float(np.mean(self.pub_arrival_forecast))
        anc[:, 0] = np.clip(0.76 + 0.14 * (rel - 1.0), 0.20, 0.94)
        anc[:, 1] = np.clip(0.78 + 0.14 * (rel - 1.0), 0.20, 0.94)
        anc[:, 2] = np.clip(0.75 + 0.14 * (rel - 1.0), 0.20, 0.94)
        anc[:, 3] = np.clip(0.80 + 0.15 * (rel - 1.0), 0.20, 0.95)
        return anc.reshape(-1)

    def nominal_exposure(self, x_norm: np.ndarray) -> NominalTrajectoryExposure:
        return compute_generic_blackbox_exposure(
            self.clip_policy(x_norm), num_channels=4, pub_load_profile=self.pub_arrival_forecast
        )

    def _simulate_raw(
        self,
        x_norm: np.ndarray,
        reps: int,
        rng: np.random.Generator,
        split: str = "search",
    ) -> np.ndarray:
        m = int(reps)
        W = self.n_windows
        mat = x_norm.reshape(W, 4)
        mu_cap = self.mu_min + mat[:, :3] * (self.mu_max - self.mu_min)  # (W, 3)
        q_ord = self.q_min + mat[:, 3] * (self.q_max - self.q_min)       # (W,)

        # Autocorrelated regime shocks across windows
        q1, q2 = np.quantile(self.pub_arrival_forecast, [1.0 / 3.0, 2.0 / 3.0])
        reg_idx = np.where(self.pub_arrival_forecast <= q1, 0, np.where(self.pub_arrival_forecast <= q2, 1, 2))

        z_ar = np.zeros((m, W), dtype=float)
        eps = rng.standard_normal(size=(m, W))
        for w in range(W):
            prev = z_ar[:, w - 1] if w > 0 else 0.0
            z_ar[:, w] = 0.52 * prev + np.sqrt(1.0 - 0.52 ** 2) * eps[:, w]

        drift_w = self.priv_drift[reg_idx][None, :]
        arrivals = self.pub_arrival_forecast[None, :] * np.exp(
            drift_w + self.priv_sigma * z_ar - 0.5 * (self.priv_sigma ** 2)
        )

        # Continuous newsvendor upstream raw-material availability + 3-station tandem queues with blocking
        q_backlog = np.zeros((m, 3), dtype=float)
        inv_carry = np.full(m, 0.20 * self.mu_max, dtype=float)
        tot_cost = np.zeros(m, dtype=float)
        tot_delayed = np.zeros(m, dtype=float)
        tot_demand = np.sum(arrivals, axis=1) + 1e-6

        cap_cost_per_rep = float(
            np.sum(self.station_cost_weights[None, :] * ((mu_cap / self.mu_max) ** 1.35))
            + np.sum(self.nv_unit_cost * (q_ord / self.q_max))
        ) / float(W)

        serv_noise = np.exp(
            self.service_cv * rng.standard_normal(size=(m, W, 3)) - 0.5 * (self.service_cv ** 2)
        )
        eff_cap = mu_cap[None, :, :] * serv_noise

        for w in range(W):
            dem_w = arrivals[:, w]
            avail_mat = inv_carry + q_ord[w]
            mat_short = np.maximum(dem_w - avail_mat, 0.0)
            inv_carry = np.minimum(np.maximum(avail_mat - dem_w, 0.0), 0.60 * self.mu_max)
            released = dem_w - mat_short

            # Station 1 -> Station 2 -> Station 3 with downstream finite-buffer blocking
            inflow = released
            w_wait_excess = mat_short * 1.15
            for st in range(3):
                load_st = q_backlog[:, st] + inflow
                # Downstream blocking throttle if st < 2
                if st < 2:
                    occ_next = q_backlog[:, st + 1] / self.buffer_cap
                    block_fac = np.clip(1.0 - 0.45 * np.maximum(occ_next - 0.55, 0.0), 0.45, 1.0)
                else:
                    block_fac = 1.0
                srv_st = np.minimum(load_st, eff_cap[:, w, st] * block_fac)
                rem_st = np.maximum(load_st - srv_st, 0.0)
                # Overload congestion tail
                rho_eff = load_st / np.maximum(eff_cap[:, w, st] * block_fac, 1e-4)
                sojourn_exc = np.maximum(rho_eff - 0.86, 0.0) * load_st * 0.55 + 0.45 * rem_st
                w_wait_excess = w_wait_excess + sojourn_exc
                q_backlog[:, st] = rem_st
                inflow = srv_st

            tot_delayed += w_wait_excess
            tot_cost += (
                0.18 * (inv_carry / self.mu_max)
                + 1.35 * (mat_short / self.mu_max)
                + 0.42 * np.sum(q_backlog / self.mu_max, axis=1)
            ) / float(W)

        # Switching smoothness penalty on staffing schedules
        ramp_pen = 0.12 * float(np.mean(np.diff(mat[:, :3], axis=0) ** 2)) if W > 1 else 0.0
        obj = cap_cost_per_rep + ramp_pen + tot_cost
        exceed_frac = tot_delayed / tot_demand
        con_centered = exceed_frac - self.sla_budget
        return np.column_stack([obj, con_centered])


def sample_simopt_tandem_tasks(n_tasks: int, seed: int, windows: tuple[int, ...], shift_scale: float) -> list[SimOptTandemCntNvBenchmark]:
    return [
        SimOptTandemCntNvBenchmark(
            task_id=f"simopt_tandem_{idx:02d}_w{windows[idx % len(windows)]}",
            n_windows=int(windows[idx % len(windows)]),
            seed=int(seed) + 409 * idx,
            shift_scale=float(shift_scale),
        )
        for idx in range(int(n_tasks))
    ]


def _auroc_binary(y_true_bool: np.ndarray, scores: np.ndarray) -> float | None:
    pos = scores[y_true_bool]
    neg = scores[~y_true_bool]
    if pos.size == 0 or neg.size == 0:
        return None
    diff = pos[:, None] - neg[None, :]
    return float(np.mean((diff > 0.0).astype(float) + 0.5 * (diff == 0.0).astype(float)))


def _fit_ablr_source_prior(records: Sequence[LegacyTrajectoryRecord]) -> dict[str, Any]:
    """Perrone et al. (2018) ABLR: shared 52-feature basis + empirical-Bayes diagonal ridge/prior across source tasks."""
    psi_mat = np.array([r.psi for r in records], dtype=float)
    psi_mean = np.mean(psi_mat, axis=0)
    psi_std = np.maximum(np.std(psi_mat, axis=0, ddof=1), 1e-4)
    raw_feat = _build_raw_dictionary((psi_mat - psi_mean) / psi_std)
    f_mean = np.mean(raw_feat, axis=0)
    f_std = np.maximum(np.std(raw_feat, axis=0, ddof=1), 1e-4)
    Phi = (raw_feat - f_mean) / f_std
    m_f = Phi.shape[1]
    X_full = np.column_stack([np.ones(len(records)), Phi])
    tids = sorted({r.task_id for r in records})
    t_arr = np.array([r.task_id for r in records])
    yf = np.array([r.mean_obj for r in records], dtype=float)
    yg = np.array([r.mean_con for r in records], dtype=float)

    th_f, th_g = [], []
    reg = np.diag(np.r_[1e-3, np.full(m_f, 0.25)])
    for tid in tids:
        mk = t_arr == tid
        if int(np.sum(mk)) < 4:
            continue
        Xt = X_full[mk]
        lhs = Xt.T @ Xt + reg
        th_f.append(np.linalg.solve(lhs, Xt.T @ yf[mk]))
        th_g.append(np.linalg.solve(lhs, Xt.T @ yg[mk]))
    tf_m = np.asarray(th_f, dtype=float)
    tg_m = np.asarray(th_g, dtype=float)
    return {
        "psi_mean": psi_mean,
        "psi_std": psi_std,
        "f_mean": f_mean,
        "f_std": f_std,
        "mu_f": np.mean(tf_m, axis=0),
        "var_f": np.maximum(np.var(tf_m, axis=0, ddof=1), 1e-3) * (1.0 + 1.0 / len(tf_m)),
        "mu_g": np.mean(tg_m, axis=0),
        "var_g": np.maximum(np.var(tg_m, axis=0, ddof=1), 1e-3) * (1.0 + 1.0 / len(tg_m)),
    }


def _predict_ablr(
    ablr: dict[str, Any],
    obs_recs: Sequence[SufficientStatsRecord],
    cand_exps: Sequence[NominalTrajectoryExposure],
    av_f_cand: np.ndarray,
    av_g_cand: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def _design(exps: Sequence[NominalTrajectoryExposure]) -> np.ndarray:
        pm = np.array([e.psi for e in exps], dtype=float)
        rf = _build_raw_dictionary((pm - ablr["psi_mean"]) / ablr["psi_std"])
        phi = np.clip((rf - ablr["f_mean"]) / ablr["f_std"], -5.0, 5.0)
        return np.column_stack([np.ones(len(exps)), phi])

    Xo = _design([r.exposure for r in obs_recs])
    Xc = _design(cand_exps)
    yf = np.array([r.mean_obj for r in obs_recs], dtype=float)
    yg = np.array([r.mean_con for r in obs_recs], dtype=float)
    vf = np.array([max(r.ssq_obj / max(r.n_reps - 1, 1), 1e-5) / max(r.n_reps, 1) for r in obs_recs], dtype=float)
    vg = np.array([max(r.ssq_con / max(r.n_reps - 1, 1), 1e-5) / max(r.n_reps, 1) for r in obs_recs], dtype=float)

    # Center intercept on target weighted mean for fair shift handling
    outs = []
    for y_vec, v_vec, mu0, var0 in (
        (yf, vf, ablr["mu_f"], ablr["var_f"]),
        (yg, vg, ablr["mu_g"], ablr["var_g"]),
    ):
        P0_inv = np.diag(1.0 / var0)
        P0_inv[0, 0] = 1e-2  # weak intercept prior allowing target level adaptation
        W = np.diag(1.0 / v_vec)
        post_prec = P0_inv + Xo.T @ W @ Xo
        post_cov = np.linalg.inv(post_prec)
        post_mu = post_cov @ (P0_inv @ mu0 + Xo.T @ W @ y_vec)
        pred_m = Xc @ post_mu
        pred_s = np.sqrt(np.maximum(np.sum((Xc @ post_cov) * Xc, axis=1), 1e-10))
        outs.append((pred_m, pred_s))
    return outs[0][0], outs[0][1], outs[1][0], outs[1][1]


def _demeaned_nrmse(pred: np.ndarray, truth: np.ndarray) -> float:
    t_c = truth - float(np.mean(truth))
    p_c = pred - float(np.mean(pred))
    denom = max(float(np.std(truth, ddof=1)), 1e-8)
    return float(np.sqrt(np.mean((p_c - t_c) ** 2)) / denom)


def _eval_heldout_metrics(
    mf: np.ndarray,
    sf: np.ndarray,
    mg: np.ndarray,
    sg: np.ndarray,
    ucb_vg: np.ndarray,
    rp_g: np.ndarray,
    sq_ref: float,
    true_obj: np.ndarray,
    true_con: np.ndarray,
    true_p_viol: np.ndarray,
    anc_true_obj: float,
) -> dict[str, float | None]:
    p_gate0 = posterior_binomial_gate_power(
        mg, sg, np.maximum(ucb_vg, 1e-8), n_samples=112, alpha=0.05, delta_slot=0.025, r_perp_con=rp_g
    )
    true_gate0 = exact_binomial_pass_power(true_p_viol, 112, 1)
    gate_rmse = float(np.sqrt(np.mean((p_gate0 - true_gate0) ** 2)))

    s_tot = np.sqrt(np.maximum(sg ** 2 + 0.25 * np.maximum(rp_g, 0.0), 1e-12))
    feas_lim = float(norm.ppf(0.05)) * np.sqrt(np.maximum(ucb_vg, 1e-8))
    p_feas = ndtr((feas_lim - mg) / s_tot)
    y_feas = true_p_viol <= 0.05
    auroc = _auroc_binary(y_feas, p_feas)
    brier = float(np.mean((p_feas - y_feas.astype(float)) ** 2))

    pred_u0 = np.maximum(sq_ref - mf, 0.0) * p_gate0
    pick_i = int(np.argmax(pred_u0)) if float(np.max(pred_u0)) > 0.0 else int(np.argmax(p_gate0))
    true_slot0_gain = float((anc_true_obj - true_obj[pick_i]) * true_gate0[pick_i])

    return {
        "obj_nrmse": _demeaned_nrmse(mf, true_obj),
        "con_nrmse": _demeaned_nrmse(mg, true_con),
        "gate_power_rmse": gate_rmse,
        "feas_brier": brier,
        "feas_auroc": auroc,
        "sel_true_slot0_gain": true_slot0_gain,
    }


def _drop_vol_exposure(exp: NominalTrajectoryExposure) -> NominalTrajectoryExposure:
    """7-coordinate exposure ablation zeroing out A_local_1..2 and N_shared_1..3 inside psi."""
    p7 = exp.psi.copy()
    p7[7:] = 0.0
    return NominalTrajectoryExposure(psi=p7, A_local=exp.A_local, N_shared=exp.N_shared)


def _corrupt_forecast_exposure(prob: BaseDynamicORProblem, x_norm: np.ndarray, mult_vec: np.ndarray) -> NominalTrajectoryExposure:
    """Evaluate nominal_exposure under multiplicative public-forecast corruption (CV = 15%)."""
    from v2_or.simulators.inventory_des import DynamicInventoryNetworkDES
    from v2_or.simulators.queueing_des import DynamicQueueingStaffingDES
    from v2_or.simulators.traffic_signal_des import IngolstadtSignalNetworkDES

    if isinstance(prob, DynamicInventoryNetworkDES):
        df = np.maximum(prob.demand_forecast * mult_vec[: prob.H], 1e-4)
        p_cp = DynamicInventoryNetworkDES(
            task_id=prob.task_id, demand_forecast=df, capacity=prob.capacity,
            initial_stock=prob.initial_stock, law=prob._law, alpha=prob.alpha,
        )
        return p_cp.nominal_exposure(x_norm)
    if isinstance(prob, DynamicQueueingStaffingDES):
        af = np.maximum(prob.arrival_forecast * mult_vec[: prob.H], 1e-4)
        p_cp = DynamicQueueingStaffingDES(
            task_id=prob.task_id, arrival_forecast=af, c_min=prob.c_min,
            c_max=prob.c_max, initial_queue=prob.initial_queue, law=prob._law, alpha=prob.alpha,
        )
        return p_cp.nominal_exposure(x_norm)
    if isinstance(prob, IngolstadtSignalNetworkDES):
        pa = np.maximum(prob.public_arrivals * mult_vec[: prob.H, None], 1e-4)
        p_cp = IngolstadtSignalNetworkDES(
            task_id=prob.task_id, lo=prob.lo, hi=prob.hi, defaults=prob.defaults,
            var_tls=prob.var_tls, fixed_overhead=prob.fixed_overhead,
            public_arrivals=pa, law=prob._law, alpha=prob.alpha,
        )
        return p_cp.nominal_exposure(x_norm)
    return prob.nominal_exposure(x_norm)


def _load_des_fam_specs() -> dict[str, dict[str, Any]]:
    p_man = PROSP_DIR / "sealed_manifest.json"
    if not p_man.is_file():
        p_man = SC_ROOT / "v2_or" / "manifests" / "v2_or_eckg_sure_confirm_v1.json"
    m_prosp = _read_json(p_man)
    return {f["name"]: f for f in m_prosp["families"] if f["name"] in DES_FAMILIES}


def _build_extended_12_source_tasks(fam_spec: dict[str, Any]) -> list[BaseDynamicORProblem]:
    spec12 = copy.deepcopy(fam_spec)
    spec12["source_config"]["n_tasks"] = 12
    return build_family_tasks(spec12, "source")


def _run_stage0_item(job: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    item_path = out_dir / "stage0" / "items" / f"{job['job_id']}.json"
    if item_path.is_file():
        return {"job_id": job["job_id"], "status": "cached"}

    if job["family"] == "SimOpt_TandemCntNv":
        prob = sample_simopt_tandem_tasks(6, seed=5801, windows=(6, 8), shift_scale=1.0)[int(job["source_task_idx"])]
    else:
        fam_spec = _load_des_fam_specs()[job["family"]]
        prob = _build_extended_12_source_tasks(fam_spec)[int(job["source_task_idx"])]

    solver_fn = LEGACY_SOLVER_REGISTRY[job["solver_name"]]
    rec = solver_fn(prob, budget_calls=60, seed=int(job["seed"]), reps_per_eval=3)
    payload = {
        "job_id": job["job_id"],
        "family": job["family"],
        "source_task_idx": int(job["source_task_idx"]),
        "solver_name": job["solver_name"],
        "records": [asdict(r) for r in rec.trajectory_log],
        "truth_used": False,
    }
    _write_json(item_path, payload)
    return {"job_id": job["job_id"], "status": "ok"}


def finish_stage0(out_dir: Path) -> dict[str, Any]:
    """Barrier-merge Stage-0 source items and fit all ablated/scaled priors + SimOpt prior."""
    s0_desc = _read_json(out_dir / "stage0" / "jobs.json")
    by_fam: dict[str, list[LegacyTrajectoryRecord]] = {}
    for jb in s0_desc["jobs"]:
        it = _read_json(out_dir / "stage0" / "items" / f"{jb['job_id']}.json")
        for rd in it["records"]:
            by_fam.setdefault(jb["family"], []).append(LegacyTrajectoryRecord(
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

    for fam in DES_FAMILIES:
        recs_6, _ = load_legacy_archive_json(U3_BUNDLES / fam / "archive.json")
        tids_6 = sorted({r.task_id for r in recs_6})
        tids_3 = set(tids_6[:3])
        recs_3 = [r for r in recs_6 if r.task_id in tids_3]
        recs_12 = list(recs_6) + by_fam.get(fam, [])

        fam_b = bundles_out / fam
        fam_b.mkdir(parents=True, exist_ok=True)

        # M_src scaling priors
        (fam_b / "prior_m3.pkl").write_bytes(pickle.dumps(DebiasedSpectralMetaPrior().fit(recs_3)))
        sh_m6 = (U3_BUNDLES / fam / "prior_default.pkl").read_bytes()
        (fam_b / "prior_m6.pkl").write_bytes(sh_m6)
        (fam_b / "prior_m12.pkl").write_bytes(pickle.dumps(DebiasedSpectralMetaPrior().fit(recs_12)))

        # Four-pillar leave-one-out priors on exact 6-task archive
        (fam_b / "prior_no_debias.pkl").write_bytes(pickle.dumps(DebiasedSpectralMetaPrior(enable_debiasing=False).fit(recs_6)))
        (fam_b / "prior_no_lowpass.pkl").write_bytes(pickle.dumps(DebiasedSpectralMetaPrior(enable_lowpass=False).fit(recs_6)))
        (fam_b / "prior_no_ortho.pkl").write_bytes(pickle.dumps(DebiasedSpectralMetaPrior(enable_orthogonalization=False).fit(recs_6)))
        (fam_b / "prior_no_sparse.pkl").write_bytes(pickle.dumps(DebiasedSpectralMetaPrior(enable_sparse_pruning=False).fit(recs_6)))

        # Psi-DropVol7D and Psi-GenericBlackBox priors
        n_ch = 2 if fam in {"Inventory_CrossHorizon", "Queueing_CrossHorizon"} else 1
        recs_7d = []
        recs_gen = []
        for r in recs_6:
            p7 = list(r.psi)
            for idx_z in range(7, 12):
                p7[idx_z] = 0.0
            recs_7d.append(LegacyTrajectoryRecord(**{**asdict(r), "psi": tuple(p7)}))
            eg = compute_generic_blackbox_exposure(np.asarray(r.x_norm, dtype=float), num_channels=n_ch)
            recs_gen.append(LegacyTrajectoryRecord(**{
                **asdict(r),
                "psi": tuple(float(v) for v in eg.psi),
                "A_local": tuple(float(v) for v in eg.A_local),
                "N_shared": tuple(float(v) for v in eg.N_shared),
            }))
        (fam_b / "prior_dropvol7d.pkl").write_bytes(pickle.dumps(DebiasedSpectralMetaPrior().fit(recs_7d)))
        (fam_b / "prior_generic_psi.pkl").write_bytes(pickle.dumps(DebiasedSpectralMetaPrior().fit(recs_gen)))

        # Meta-BO comparator bundles on exact 6-task archive
        (fam_b / "ablr_m6.pkl").write_bytes(pickle.dumps(_fit_ablr_source_prior(recs_6)))
        (fam_b / "mtgp_m6.pkl").write_bytes(pickle.dumps(pretrain_mtgp_source_psi(recs_6)))
        rgpe_p = U3_BUNDLES / fam / "rgpe_psi_pretrain.pkl"
        if rgpe_p.is_file():
            (fam_b / "rgpe_m6.pkl").write_bytes(rgpe_p.read_bytes())
        else:
            (fam_b / "rgpe_m6.pkl").write_bytes(pickle.dumps(pretrain_rgpe_psi_source_ensemble(recs_6)))

    # Fit SimOpt_TandemCntNv prior
    sim_recs = by_fam["SimOpt_TandemCntNv"]
    sim_b = bundles_out / "SimOpt_TandemCntNv"
    sim_b.mkdir(parents=True, exist_ok=True)
    prior_sim = DebiasedSpectralMetaPrior().fit(sim_recs)
    (sim_b / "prior_default.pkl").write_bytes(pickle.dumps(prior_sim))
    # Also save compact SimOpt fitted prior in paper_artifacts/v2_or_rev7_or_panel_v1 for portable supplement
    top_prior_out = SC_ROOT / "paper_artifacts" / "v2_or_rev7_or_panel_v1" / "simopt_prior_default.pkl"
    top_prior_out.parent.mkdir(parents=True, exist_ok=True)
    top_prior_out.write_bytes(pickle.dumps(prior_sim))

    done_rec = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "simopt_source_records": len(sim_recs),
        "des_m12_records": {f: len(by_fam.get(f, [])) for f in DES_FAMILIES},
    }
    _write_json(out_dir / "stage0" / "completion.json", done_rec)
    return done_rec


def _score_latent_nei_vector_frozen(
    pred: Any,
    elig_base_idx: list[int],
    sq_ref: float,
    alpha: float,
    seed: int,
    n_mc: int = 64,
) -> np.ndarray:
    """Return full 160-candidate LatentNEI vector on a frozen state (exact joint latent draws)."""
    rng = np.random.default_rng(int(seed))
    n_tot = len(pred.mean_obj)
    L_f = _safe_cholesky(pred.cov_obj)
    L_g = _safe_cholesky(pred.cov_con + np.diag(0.25 * np.maximum(pred.r_perp_con, 0.0)))
    hm = n_mc // 2
    zf = rng.standard_normal(size=(n_tot, hm))
    zg = rng.standard_normal(size=(n_tot, hm))
    zf = np.hstack([zf, -zf])
    zg = np.hstack([zg, -zg])
    f_d = pred.mean_obj[:, None] + L_f @ zf
    g_d = pred.mean_con[:, None] + L_g @ zg
    lim = float(norm.ppf(alpha)) * np.sqrt(np.maximum(pred.ucb_aleatoric_var_con, 1e-8))[:, None]
    feas = g_d <= lim
    eb = np.asarray(elig_base_idx, dtype=int)
    inc = np.where(feas[eb, :], f_d[eb, :], np.inf)
    b_d = np.min(inc, axis=0)
    b_d = np.where(np.isfinite(b_d), b_d, float(sq_ref))
    imp = np.maximum(b_d[None, :] - f_d, 0.0) * feas.astype(float)
    return np.mean(imp, axis=1)


def _eval_cands_24scrambles_exact(
    cand_ids: list[int],
    pred: Any,
    model: TemperedHierarchicalHVDModel,
    cands: list[np.ndarray],
    exps: list[Any],
    sq_ref: float,
    alpha: float,
    state_uid: int,
    n_scrambles: int = 24,
) -> dict[int, tuple[float, float]]:
    """Evaluate 24-scramble single-nomination working-model gain on exact multiscramble seeds."""
    from run_eckg_multiscramble_cei_audit import _score_subset_single_scramble

    key_cands = sorted(set(int(q) for q in cand_ids))
    if not key_cands:
        return {}
    term_draws: dict[int, list[float]] = {q: [] for q in key_cands}
    for r_i in range(int(n_scrambles)):
        sc_seed = 9_500_000 + 10_007 * int(state_uid) + 131 * r_i
        _, t_d, _ = _score_subset_single_scramble(
            pred, model, cands, exps, key_cands,
            alpha=alpha, batch_reps=2, sq_ref=sq_ref, scramble_seed=sc_seed,
        )
        for q in key_cands:
            term_draws[q].append(float(t_d[q]))
    out: dict[int, tuple[float, float]] = {}
    for q, vals in term_draws.items():
        arr = np.asarray(vals, dtype=float)
        out[q] = (float(np.mean(arr)), float(np.std(arr, ddof=1) / np.sqrt(max(len(arr), 1))))
    return out


def _run_stage1_item(job: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    item_path = out_dir / "stage1" / "items" / f"{job['job_id']}.json"
    if item_path.is_file():
        return {"job_id": job["job_id"], "status": "cached"}

    if job["kind"] == "simopt_sv":
        prob = sample_simopt_tandem_tasks(24, seed=4811201, windows=(12, 16), shift_scale=1.3)[int(job["target_task_idx"])]
        prior = pickle.loads((out_dir / "bundles" / "SimOpt_TandemCntNv" / "prior_default.pkl").read_bytes())
        arm = job["arm_name"]
        seed = int(job["seed"])
        if arm == "GP-FTA-ECKG":
            res = run_tls_olh_kg(
                prob, prior, n_online_steps=60, n0_init=8, batch_reps=2, seed=seed,
                model_mode="tempered_sure_release", acquisition_type="eligible_cascade_kg",
                init_mode="balanced_warmstart", proposal_mode="adaptive_release",
                use_source_basis=True, use_source_coef_prior=True, use_source_hvd_prior=True,
                nomination_mode="cascade_bellman_3slot", pool_size=160, method_label=arm,
            )
        elif arm == "GP-FTA-FeasibleKG":
            res = run_tls_olh_kg(
                prob, prior, n_online_steps=60, n0_init=8, batch_reps=2, seed=seed,
                model_mode="tempered_sure_release", acquisition_type="chance_feasibility_kg",
                init_mode="balanced_warmstart", proposal_mode="adaptive_release",
                use_source_basis=True, use_source_coef_prior=True, use_source_hvd_prior=True,
                nomination_mode="cascade_bellman_3slot", pool_size=160, method_label=arm,
            )
        elif arm == "GP-FTA-LatentNEI-3Slot":
            res, _ = run_gpfta_variant(prob, prior, variant=arm, seed=seed, n0_init=8, n_online_steps=60, batch_reps=2, pool_size=160)
        elif arm == "SCBO-3Slot":
            res = run_scbo_baseline(prob, n_online_steps=60, n0_init=8, batch_reps=2, seed=seed, pool_size=160)
        elif arm == "PublicPsi-TargetGP-TS":
            res = run_tls_olh_kg(
                prob, None, n_online_steps=60, n0_init=8, batch_reps=2, seed=seed,
                model_mode="target_only", acquisition_type="gate_power_ts",
                init_mode="cold_dct", proposal_mode="local_dct_only",
                use_source_basis=False, use_source_coef_prior=False, use_source_hvd_prior=False,
                nomination_mode="cascade_bellman_3slot", pool_size=160, method_label=arm,
            )
        else:
            raise ValueError(f"Unknown SimOpt arm {arm!r}")

        payload = {
            "job_id": job["job_id"],
            "kind": "simopt_sv",
            "target_task_idx": int(job["target_task_idx"]),
            "task_id": prob.task_id,
            "policy_dim": int(prob.d),
            "rep_seed": int(job["rep_seed"]),
            "method_name": arm,
            "search_calls_used": int(res.search_calls_used),
            "verification_calls_used": int(res.verification_calls_used),
            "total_target_calls": int(res.total_target_calls),
            "online_wall_time_sec": float(res.wall_time_sec),
            "certified": bool(res.certified),
            "selected_role": res.selected_role,
            "deployed_policy": None if res.deployed_policy is None else [float(v) for v in res.deployed_policy],
            "truth_used": False,
        }
        _write_json(item_path, payload)
        return {"job_id": job["job_id"], "status": "ok"}

    # Frozen 24-state LatentNEI & hyperparameter sensitivity item (1 task = step_0 and step_59)
    fam_name = job["family"]
    t_idx = int(job["target_task_idx"])
    fam_spec = _load_des_fam_specs()[fam_name]
    prob = build_family_tasks(fam_spec, "target")[t_idx]
    prior = pickle.loads((U3_BUNDLES / fam_name / "prior_default.pkl").read_bytes())
    r_seed = int(fam_spec["target_eval_seeds"][0])
    seed = r_seed + 10_007 * t_idx
    rng_init = np.random.default_rng(seed)
    rng_search = np.random.default_rng(seed + 100_003)

    model = TemperedHierarchicalHVDModel(
        prior, mode="tempered_sure_release",
        use_source_basis=True, use_source_coef_prior=True, use_source_hvd_prior=True, n0_freeze=8,
    )
    for x0 in build_balanced_warmstart_points(prob, prior, 8, rng_init):
        e0 = prob.nominal_exposure(x0)
        s0 = prob.simulate_replications(x0, 2, rng_init, split="search")
        model.observe(SufficientStatsRecord.from_samples(x0, s0, e0))
    sq_est = float(model.unique_records[0].mean_obj)

    states_out: dict[str, Any] = {}
    mc_prev = _read_json(PROSP_DIR / "eckg_multiscramble_cei_readout.json")
    prev_by_key = {(s["family"], int(s["task_idx"]), s["stage"]): s for s in mc_prev["states"]}
    fam_offset = 0 if fam_name == "Queueing_CrossHorizon" else 100

    for step_i in range(60):
        enable_tpl = not bool(getattr(model, "is_source_released_", False))
        cands, exps = generate_dynamic_candidate_pool(prob, model, prior, rng_search, pool_size=160, enable_source_templates=enable_tpl)
        pred = model.predict_joint(cands, exps)
        sq_ref = 0.5 * sq_est + 0.5 * float(pred.mean_obj[0])
        _, diag = score_eligible_cascade_kg(
            pred, model, cands, exps, alpha=prob.alpha, batch_reps=2,
            status_quo_obj=sq_ref, nomination_mode="cascade_bellman_3slot", return_diagnostics=True,
        )
        q_eckg = int(diag["chosen_query"])

        if step_i in (0, 59):
            stg_key = f"step_{step_i}"
            eval_recs = model.unique_records
            key_to_cand = {model._policy_key(x): idx for idx, x in enumerate(cands)}
            elig_base_idx = [key_to_cand[model._policy_key(r.x_norm)] for r in eval_recs if model._policy_key(r.x_norm) in key_to_cand]
            nei_vec = _score_latent_nei_vector_frozen(pred, elig_base_idx, sq_ref, prob.alpha, seed + 551 + step_i, n_mc=64)
            if float(np.max(nei_vec)) > 1e-12:
                q_lnei = int(np.argmax(nei_vec))
            else:
                pg = posterior_binomial_gate_power(
                    pred.mean_con, pred.std_con, np.maximum(pred.ucb_aleatoric_var_con, 1e-8),
                    n_samples=112, alpha=prob.alpha, delta_slot=0.025, r_perp_con=pred.r_perp_con,
                )
                q_lnei = int(np.argmax(pg))

            # Sensitivity of EC-KG to variance buffer multiplier {0.15, 0.35, 0.55}
            sens_q: dict[str, int] = {}
            for buf_tag, mult_ratio in (("buf_low_015", 0.90), ("buf_def_035", 1.00), ("buf_high_055", 1.12)):
                pr_mod = copy.copy(pred)
                object.__setattr__(pr_mod, "ucb_aleatoric_var_con", pred.ucb_aleatoric_var_con * mult_ratio)
                _, d_s = score_eligible_cascade_kg(
                    pr_mod, model, cands, exps, alpha=prob.alpha, batch_reps=2,
                    status_quo_obj=sq_ref, nomination_mode="cascade_bellman_3slot", return_diagnostics=True,
                )
                sens_q[buf_tag] = int(d_s["chosen_query"])

            prev_rec = prev_by_key[(fam_name, t_idx, stg_key)]
            eckg_gain = 1e-3 * float(prev_rec["ec_term_x1e3"]["mean"])
            eckg_se = 1e-3 * float(prev_rec["ec_term_x1e3"]["se"])
            state_uid = fam_offset + 2 * t_idx + (0 if step_i == 0 else 1)

            extra_cands = [q for q in {q_lnei, *sens_q.values()} if q != q_eckg]
            extra_eval = _eval_cands_24scrambles_exact(
                extra_cands, pred, model, cands, exps, sq_ref, prob.alpha, state_uid
            )
            lnei_mean, lnei_se = (eckg_gain, eckg_se) if q_lnei == q_eckg else extra_eval[q_lnei]
            buf_gains = {
                b_k: (eckg_gain if b_q == q_eckg else extra_eval[b_q][0])
                for b_k, b_q in sens_q.items()
            }

            states_out[stg_key] = {
                "family": fam_name,
                "task_idx": t_idx,
                "stage": stg_key,
                "q_eckg": q_eckg,
                "q_latent_nei": q_lnei,
                "latent_nei_is_revisit": bool(q_lnei in set(elig_base_idx)),
                "eckg_term_gain_mean": eckg_gain,
                "plugin_cei_term_gain_mean": 1e-3 * float(prev_rec["ub_cei_term_x1e3_mean"]),
                "ref160_term_gain_mean": 1e-3 * float(prev_rec["opt_term_x1e3"]["mean"]),
                "latent_nei_term_gain_mean": lnei_mean,
                "latent_nei_term_gain_se": lnei_se,
                "hvd_buffer_sensitivity_queries": sens_q,
                "hvd_buffer_sensitivity_gains": buf_gains,
            }

        if step_i < 59:
            xn = cands[q_eckg]
            samp = prob.simulate_replications(xn, 2, rng_search, split="search")
            model.observe(SufficientStatsRecord.from_samples(xn, samp, exps[q_eckg]))

    _write_json(item_path, {"job_id": job["job_id"], "kind": "frozen24_nei_sens", "states": states_out, "truth_used": False})
    return {"job_id": job["job_id"], "status": "ok"}


def _eval_confirm_task_all(job: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Evaluate decision-aligned calibration, 4-Pillar ablations, meta-BO baselines,
    M_src={3,6,12}, shift_scale={1.0,1.3,1.6}, and 15% forecast noise on 1 unseen task."""
    aud_path = out_dir / "stage2" / "items" / f"{job['job_id']}.json"
    if aud_path.is_file():
        return {"job_id": job["job_id"], "status": "cached"}

    fam_name = job["family"]
    t_idx = int(job["target_task_idx"])
    fam_spec = copy.deepcopy(_load_des_fam_specs()[fam_name])
    fam_spec["target_config"]["seed"] = int(fam_spec["target_config"]["seed"]) + 1_000_000
    n_ch = 2 if fam_name in {"Inventory_CrossHorizon", "Queueing_CrossHorizon"} else 1

    fam_b = out_dir / "bundles" / fam_name
    p_m3 = pickle.loads((fam_b / "prior_m3.pkl").read_bytes())
    p_m6 = pickle.loads((fam_b / "prior_m6.pkl").read_bytes())
    p_m12 = pickle.loads((fam_b / "prior_m12.pkl").read_bytes())
    p_nodeb = pickle.loads((fam_b / "prior_no_debias.pkl").read_bytes())
    p_nolp = pickle.loads((fam_b / "prior_no_lowpass.pkl").read_bytes())
    p_noorth = pickle.loads((fam_b / "prior_no_ortho.pkl").read_bytes())
    p_nosprs = pickle.loads((fam_b / "prior_no_sparse.pkl").read_bytes())
    p_drop7 = pickle.loads((fam_b / "prior_dropvol7d.pkl").read_bytes())
    p_gen = pickle.loads((fam_b / "prior_generic_psi.pkl").read_bytes())
    ablr_b = pickle.loads((fam_b / "ablr_m6.pkl").read_bytes())
    mtgp_b = pickle.loads((fam_b / "mtgp_m6.pkl").read_bytes())
    rgpe_b = pickle.loads((fam_b / "rgpe_m6.pkl").read_bytes())

    def _run_on_problem(prob: BaseDynamicORProblem, shift_tag: str) -> dict[str, Any]:
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

        # Ground-truth 400-draw evaluation on anchor and 32 held-out policies
        s_anc = prob.simulate_replications(prob.domain_heuristic_anchor(), 400, rng_truth, split="audit")
        anc_true_obj = float(np.mean(s_anc[:, 0]))
        true_obj = np.empty(len(held_x), dtype=float)
        true_con = np.empty(len(held_x), dtype=float)
        true_pv = np.empty(len(held_x), dtype=float)
        for i, hx in enumerate(held_x):
            sh = prob.simulate_replications(hx, 400, rng_truth, split="audit")
            true_obj[i] = float(np.mean(sh[:, 0]))
            true_con[i] = float(np.mean(sh[:, 1]))
            true_pv[i] = float(np.mean(sh[:, 1] > 0.0))

        sq_ref = float(init_recs[0].mean_obj)

        def _score_gpfta_prior(pr_obj: DebiasedSpectralMetaPrior, mode_str: str, use_coef: bool, exp_fn=None) -> dict[str, float | None]:
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
                        p_inst,
                        mode="matched_coefficients",
                        coefficient_prior="historical_full",
                        use_source_basis=True,
                        use_source_hvd_prior=False,
                        use_empirical_r_perp=False,
                    )
                    m.freeze_matched_scales(recs9[:8])
                    m.tau_shift_obj_ = 1e-6 * m.tau_shift_obj_
                    m.tau_shift_con_ = 1e-6 * m.tau_shift_con_
                else:
                    m = TemperedHierarchicalHVDModel(
                        pr_obj, mode=mode_str,
                        use_source_basis=True, use_source_coef_prior=use_coef,
                        use_source_hvd_prior=False, use_empirical_r_perp=use_coef, n0_freeze=8,
                    )
                for r in recs9:
                    m.observe(r)
                pd = m.predict_joint(held_x, he_eval)
                met = _eval_heldout_metrics(
                    pd.mean_obj, pd.std_obj, pd.mean_con, pd.std_con,
                    pd.ucb_aleatoric_var_con, pd.r_perp_con, sq_ref,
                    true_obj, true_con, true_pv, anc_true_obj,
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
                met = _eval_heldout_metrics(
                    pd.mean_obj, pd.std_obj, pd.mean_con, pd.std_con,
                    pd.ucb_aleatoric_var_con, pd.r_perp_con, sq_ref,
                    true_obj, true_con, true_pv, anc_true_obj,
                )
                for k, v in met.items():
                    if v is not None:
                        agg.setdefault(k, []).append(float(v))
            return {k: (float(np.mean(vals)) if vals else None) for k, vals in agg.items()}

        if shift_tag != "shift_13_main":
            # Compact comparison on shifted regimes
            return {
                "Online-FullSURE": _score_gpfta_prior(p_m6, "tempered_sure_release", True),
                "ColdPsi-GP": _score_cold_psi(),
            }

        # Full evaluation on main unseen confirm_fresh_72 task
        v_m6 = _build_prior_variants(p_m6, seed=seed_base + 33)
        v_m3 = _build_prior_variants(p_m3, seed=seed_base + 34)
        v_m12 = _build_prior_variants(p_m12, seed=seed_base + 35)

        # Shared HVD variance estimates from Online-FullSURE for fair ABLR / MTGP / RGPE gate-power evaluation
        def _score_external_meta(kind_ext: str) -> dict[str, float | None]:
            agg: dict[str, list[float]] = {}
            for prb in probe_recs:
                recs9 = list(init_recs) + [prb]
                m_hvd = TemperedHierarchicalHVDModel(
                    None, mode="target_only", use_source_basis=False, use_source_coef_prior=False, use_source_hvd_prior=False,
                )
                for r in recs9:
                    m_hvd.observe(r)
                pd_hvd = m_hvd.predict_joint(held_x, held_e)
                if kind_ext == "ABLR":
                    mf, sf, mg, sg = _predict_ablr(ablr_b, recs9, held_e, pd_hvd.aleatoric_var_obj, pd_hvd.aleatoric_var_con)
                elif kind_ext == "MTGP":
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
                else:
                    # RGPE-on-Psi
                    s_fs, s_gs, pm, ps, _, _, _ = rgpe_b
                    Po = np.clip((np.array([r.exposure.psi for r in recs9]) - pm) / ps, -4.0, 4.0)
                    Pc = np.clip((np.array([e.psi for e in held_e]) - pm) / ps, -4.0, 4.0)
                    yf = np.array([r.mean_obj for r in recs9])
                    yg = np.array([r.mean_con for r in recs9])
                    vf = np.array([max(r.ssq_obj / max(r.n_reps - 1, 1), 1e-5) / max(r.n_reps, 1) for r in recs9])
                    vg = np.array([max(r.ssq_con / max(r.n_reps - 1, 1), 1e-5) / max(r.n_reps, 1) for r in recs9])
                    tf = RawInputExactGP().fit(Po, yf, vf)
                    tg = RawInputExactGP().fit(Po, yg, vg)
                    mfs = [m.predict(Pc) for m in s_fs] + [tf.predict(Pc)]
                    mgs = [m.predict(Pc) for m in s_gs] + [tg.predict(Pc)]
                    w_eq = np.r_[np.full(len(s_fs), 0.5 / max(len(s_fs), 1)), 0.5]
                    mf = sum(w * p[0] for w, p in zip(w_eq, mfs))
                    sf = np.sqrt(sum((w ** 2) * (p[1] ** 2) for w, p in zip(w_eq, mfs)) + 1e-8)
                    mg = sum(w * p[0] for w, p in zip(w_eq, mgs))
                    sg = np.sqrt(sum((w ** 2) * (p[1] ** 2) for w, p in zip(w_eq, mgs)) + 1e-8)

                met = _eval_heldout_metrics(
                    mf, sf, mg, sg, pd_hvd.ucb_aleatoric_var_con, np.full(len(held_x), 1e-4),
                    sq_ref, true_obj, true_con, true_pv, anc_true_obj,
                )
                for k, v in met.items():
                    if v is not None:
                        agg.setdefault(k, []).append(float(v))
            return {k: (float(np.mean(vals)) if vals else None) for k, vals in agg.items()}

        rng_fc = np.random.default_rng(seed_base + 919)
        fc_mult = np.clip(rng_fc.normal(1.0, 0.15, size=512), 0.55, 1.45)

        return {
            "models": {
                "Online-FullSURE": _score_gpfta_prior(p_m6, "tempered_sure_release", True),
                "Online-DiagSURE": _score_gpfta_prior(v_m6["online_diag"], "tempered_sure_release", True),
                "Online-SameBasisWeak": _score_gpfta_prior(p_m6, "source_basis_weak_prior", False),
                "ColdPsi-GP": _score_cold_psi(),
                "ABLR-on-Psi": _score_external_meta("ABLR"),
                "MTGP-ICM-on-Psi": _score_external_meta("MTGP"),
                "RGPE-on-Psi": _score_external_meta("RGPE"),
                "Pillar-NoDebias": _score_gpfta_prior(p_nodeb, "tempered_sure_release", True),
                "Pillar-NoLowPass": _score_gpfta_prior(p_nolp, "tempered_sure_release", True),
                "Pillar-NoOrtho": _score_gpfta_prior(p_noorth, "tempered_sure_release", True),
                "Pillar-NoSparse": _score_gpfta_prior(p_nosprs, "tempered_sure_release", True),
                "Psi-DropVol7D": _score_gpfta_prior(p_drop7, "tempered_sure_release", True, exp_fn=lambda _x, e: _drop_vol_exposure(e)),
                "Psi-GenericBlackBox": _score_gpfta_prior(p_gen, "tempered_sure_release", True, exp_fn=lambda x, _e: compute_generic_blackbox_exposure(x, num_channels=n_ch)),
                "ForecastNoise15-FullSURE": _score_gpfta_prior(p_m6, "tempered_sure_release", True, exp_fn=lambda x, _e: _corrupt_forecast_exposure(prob, x, fc_mult)),
            },
            "msrc_scaling": {
                "M3_OnlineFull": _score_gpfta_prior(p_m3, "tempered_sure_release", True),
                "M6_OnlineFull": _score_gpfta_prior(p_m6, "tempered_sure_release", True),
                "M12_OnlineFull": _score_gpfta_prior(p_m12, "tempered_sure_release", True),
                "M3_KernelFull": _score_gpfta_prior(v_m3["zm_full"], "matched_coefficients", True),
                "M3_KernelDiag": _score_gpfta_prior(v_m3["zm_diag"], "matched_coefficients", True),
                "M6_KernelFull": _score_gpfta_prior(v_m6["zm_full"], "matched_coefficients", True),
                "M6_KernelDiag": _score_gpfta_prior(v_m6["zm_diag"], "matched_coefficients", True),
                "M12_KernelFull": _score_gpfta_prior(v_m12["zm_full"], "matched_coefficients", True),
                "M12_KernelDiag": _score_gpfta_prior(v_m12["zm_diag"], "matched_coefficients", True),
                "M12_KernelIso": _score_gpfta_prior(v_m12["zm_iso"], "matched_coefficients", True),
            },
        }

    # Shift 1.3 (main), Shift 1.0, and Shift 1.6
    prob_13 = build_family_tasks(fam_spec, "target")[t_idx]
    res_main = _run_on_problem(prob_13, "shift_13_main")

    spec_10 = copy.deepcopy(fam_spec)
    spec_10["target_config"]["shift_scale"] = 1.0
    res_s10 = _run_on_problem(build_family_tasks(spec_10, "target")[t_idx], "shift_10")

    spec_16 = copy.deepcopy(fam_spec)
    spec_16["target_config"]["shift_scale"] = 1.6
    res_s16 = _run_on_problem(build_family_tasks(spec_16, "target")[t_idx], "shift_16")

    out = {
        "job_id": job["job_id"],
        "kind": "confirm72_calib",
        "family": fam_name,
        "target_task_idx": t_idx,
        "main_shift13": res_main,
        "shift_sensitivity": {
            "shift_10": res_s10,
            "shift_13": {
                "Online-FullSURE": res_main["models"]["Online-FullSURE"],
                "ColdPsi-GP": res_main["models"]["ColdPsi-GP"],
            },
            "shift_16": res_s16,
        },
        "truth_used": True,
    }
    _write_json(aud_path, out)
    return {"job_id": job["job_id"], "status": "ok"}


def _run_stage2_item(job: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    if job["kind"] == "confirm72_calib":
        return _eval_confirm_task_all(job, out_dir)

    aud_path = out_dir / "stage2" / "items" / f"{job['job_id']}.json"
    if aud_path.is_file():
        return {"job_id": job["job_id"], "status": "cached"}

    sv = _read_json(out_dir / "stage1" / "items" / f"{job['sv_job_id']}.json")
    t_idx = int(sv["target_task_idx"])
    prob = sample_simopt_tandem_tasks(24, seed=4811201, windows=(12, 16), shift_scale=1.3)[t_idx]
    audit_seed = 2058112001 + 997 * t_idx
    x_anc = prob.domain_heuristic_anchor()
    anc_aud = prob.audit_ground_truth(x_anc, n_eval=2000, seed=audit_seed, split="audit")
    anc_obj = float(anc_aud["mc_objective_mean"])
    anc_viol = float(anc_aud["mc_violation_prob"])

    da = None
    if sv["certified"] and sv["deployed_policy"] is not None:
        xd = np.asarray(sv["deployed_policy"], dtype=float)
        da = dict(anc_aud) if np.allclose(xd, x_anc, atol=1e-10, rtol=0.0) else prob.audit_ground_truth(xd, n_eval=2000, seed=audit_seed, split="audit")
    dep_obj = float(da["mc_objective_mean"]) if da is not None else None
    dep_viol = float(da["mc_violation_prob"]) if da is not None else None

    out = {
        "job_id": job["job_id"],
        "kind": "simopt_audit",
        "target_task_idx": t_idx,
        "task_id": sv["task_id"],
        "policy_dim": int(sv["policy_dim"]),
        "rep_seed": int(sv["rep_seed"]),
        "method_name": sv["method_name"],
        "search_calls_used": int(sv["search_calls_used"]),
        "verification_calls_used": int(sv["verification_calls_used"]),
        "total_target_calls": int(sv["total_target_calls"]),
        "online_wall_time_sec": float(sv["online_wall_time_sec"]),
        "certified": bool(sv["certified"]),
        "selected_role": sv["selected_role"],
        "anchor_obj": anc_obj,
        "anchor_viol_prob": anc_viol,
        "deployed_obj": dep_obj,
        "deployed_viol_prob": dep_viol,
        "deployed_viol_ucb95": float(da["mc_viol_ci95_hi"]) if da is not None else None,
        "status_quo_loss": dep_obj if dep_obj is not None else anc_obj,
        "operating_viol_prob": dep_viol if dep_viol is not None else anc_viol,
        "truth_used": True,
    }
    _write_json(aud_path, out)
    return {"job_id": job["job_id"], "status": "ok"}


def describe_stage(stage: str, out_dir: Path) -> dict[str, Any]:
    jobs: list[dict[str, Any]] = []
    fam_specs = _load_des_fam_specs()

    if stage == "stage0":
        for fam in DES_FAMILIES:
            src_seed = int(fam_specs[fam]["source_archive_protocol"]["seed"])
            for s_idx in range(6, 12):
                for sv_i, sv_name in enumerate(SOLVER_NAMES):
                    jobs.append({
                        "job_id": f"s0_{fam}_t{s_idx:02d}_{sv_name}",
                        "family": fam,
                        "source_task_idx": s_idx,
                        "solver_name": sv_name,
                        "seed": src_seed + 1009 * s_idx + 37 * sv_i,
                    })
        for s_idx in range(6):
            for sv_i, sv_name in enumerate(SOLVER_NAMES):
                jobs.append({
                    "job_id": f"s0_SimOpt_TandemCntNv_t{s_idx:02d}_{sv_name}",
                    "family": "SimOpt_TandemCntNv",
                    "source_task_idx": s_idx,
                    "solver_name": sv_name,
                    "seed": 7801 + 1009 * s_idx + 37 * sv_i,
                })
    elif stage == "stage1":
        if not (out_dir / "stage0" / "completion.json").is_file():
            raise RuntimeError("Run finish_stage0 before stage1")
        for t_idx in range(24):
            for r_seed in (2048112001, 2048112002):
                cell_seed = r_seed + 10_007 * t_idx
                for arm in SIMOPT_ARMS:
                    jobs.append({
                        "job_id": f"s1_simopt_t{t_idx:02d}_s{r_seed}_{arm}",
                        "kind": "simopt_sv",
                        "target_task_idx": t_idx,
                        "rep_seed": r_seed,
                        "seed": cell_seed,
                        "arm_name": arm,
                    })
        for fam in ("Queueing_CrossHorizon", "Traffic_Ingolstadt"):
            for t_idx in range(6):
                jobs.append({
                    "job_id": f"s1_froz24_{fam}_t{t_idx:02d}",
                    "kind": "frozen24_nei_sens",
                    "family": fam,
                    "target_task_idx": t_idx,
                })
    elif stage == "stage2":
        s1_desc = _read_json(out_dir / "stage1" / "jobs.json")
        for j in s1_desc["jobs"]:
            if not (out_dir / "stage1" / "items" / f"{j['job_id']}.json").is_file():
                raise RuntimeError(f"Missing Stage-1 item: {j['job_id']}")
            if j["kind"] == "simopt_sv":
                jobs.append({
                    "job_id": f"s2_aud_{j['job_id']}",
                    "kind": "simopt_audit",
                    "sv_job_id": j["job_id"],
                })
        for fam in DES_FAMILIES:
            for t_idx in range(24):
                jobs.append({
                    "job_id": f"s2_conf72_{fam}_t{t_idx:02d}",
                    "kind": "confirm72_calib",
                    "family": fam,
                    "target_task_idx": t_idx,
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
    fn = {"stage0": _run_stage0_item, "stage1": _run_stage1_item, "stage2": _run_stage2_item}[stage]
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
    compact_inputs_path = summary_path.parent / "rev7_prior_simopt_compact_inputs.json"
    if (out_dir / "stage2" / "jobs.json").is_file():
        s1_desc = _read_json(out_dir / "stage1" / "jobs.json")
        s2_desc = _read_json(out_dir / "stage2" / "jobs.json")
        exec_sha = s2_desc["execution_code_sha256"]
        froz_states = []
        for jb in s1_desc["jobs"]:
            if jb["kind"] == "frozen24_nei_sens":
                it = _read_json(out_dir / "stage1" / "items" / f"{jb['job_id']}.json")
                for stg in ("step_0", "step_59"):
                    froz_states.append(it["states"][stg])
        sim_auds = [
            _read_json(out_dir / "stage2" / "items" / f"{jb['job_id']}.json")
            for jb in s2_desc["jobs"] if jb["kind"] == "simopt_audit"
        ]
        c72_items = [
            _read_json(out_dir / "stage2" / "items" / f"{jb['job_id']}.json")
            for jb in s2_desc["jobs"] if jb["kind"] == "confirm72_calib"
        ]
        _write_json(compact_inputs_path, _stabilize_floats({
            "execution_code_sha256": exec_sha,
            "frozen24_states": froz_states,
            "simopt_audits": sorted(sim_auds, key=lambda z: (int(z["target_task_idx"]), int(z["rep_seed"]), z["method_name"])),
            "confirm72_items": sorted(c72_items, key=lambda z: (z["family"], int(z["target_task_idx"]))),
        }))
    else:
        for cand_p in (
            out_dir / "rev7_prior_simopt_compact_inputs.json",
            summary_path.parent / "rev7_prior_simopt_compact_inputs.json",
            SC_ROOT / "paper_artifacts" / "v2_or_rev7_or_panel_v1" / "rev7_prior_simopt_compact_inputs.json",
        ):
            if cand_p.is_file():
                compact_inputs_path = cand_p
                break
        comp = _read_json(compact_inputs_path)
        exec_sha = comp["execution_code_sha256"]
        froz_states = comp["frozen24_states"]
        sim_auds = comp["simopt_audits"]
        c72_items = comp["confirm72_items"]

    wtl_lnei = {"wins": 0, "ties": 0, "losses": 0}
    by_group_lnei: dict[str, Any] = {}
    for fam in ("Queueing_CrossHorizon", "Traffic_Ingolstadt"):
        for stg in ("step_0", "step_59"):
            sub = [s for s in froz_states if s["family"] == fam and s["stage"] == stg]
            w = sum(1 for s in sub if s["eckg_term_gain_mean"] > s["latent_nei_term_gain_mean"] + 1e-12)
            l = sum(1 for s in sub if s["eckg_term_gain_mean"] < s["latent_nei_term_gain_mean"] - 1e-12)
            t = len(sub) - w - l
            wtl_lnei["wins"] += w
            wtl_lnei["ties"] += t
            wtl_lnei["losses"] += l
            by_group_lnei[f"{fam}__{stg}"] = {
                "eckg_mean_x1e3": _stab_float(1e3 * float(np.mean([s["eckg_term_gain_mean"] for s in sub]))),
                "latent_nei_mean_x1e3": _stab_float(1e3 * float(np.mean([s["latent_nei_term_gain_mean"] for s in sub]))),
                "plugin_cei_mean_x1e3": _stab_float(1e3 * float(np.mean([s["plugin_cei_term_gain_mean"] for s in sub]))),
                "ref160_mean_x1e3": _stab_float(1e3 * float(np.mean([s["ref160_term_gain_mean"] for s in sub]))),
                "wtl_vs_latent_nei": f"{w}/{t}/{l}",
                "latent_nei_revisit_count": int(sum(1 for s in sub if s["latent_nei_is_revisit"])),
                "buf_low_015_mean_x1e3": _stab_float(1e3 * float(np.mean([s["hvd_buffer_sensitivity_gains"]["buf_low_015"] for s in sub]))),
                "buf_high_055_mean_x1e3": _stab_float(1e3 * float(np.mean([s["hvd_buffer_sensitivity_gains"]["buf_high_055"] for s in sub]))),
                "same_query_buf_low_count": int(sum(1 for s in sub if s["hvd_buffer_sensitivity_queries"]["buf_low_015"] == s["q_eckg"])),
                "same_query_buf_high_count": int(sum(1 for s in sub if s["hvd_buffer_sensitivity_queries"]["buf_high_055"] == s["q_eckg"])),
            }

    # 2. Summarize SimOpt_TandemCntNv external generic-psi benchmark (24 tasks x 2 streams x 5 arms)
    sim_by_mt: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for r in sim_auds:
        sim_by_mt.setdefault((r["method_name"], int(r["target_task_idx"])), []).append(r)

    sim_summary: dict[str, Any] = {"arms": {}, "contrasts_vs_eckg": {}}
    for arm in SIMOPT_ARMS:
        t_loss = [float(np.mean([x["status_quo_loss"] for x in sim_by_mt[(arm, ti)]])) for ti in range(24)]
        t_viol = [float(np.mean([x["operating_viol_prob"] for x in sim_by_mt[(arm, ti)]])) for ti in range(24)]
        t_calls = [float(np.mean([x["total_target_calls"] for x in sim_by_mt[(arm, ti)]])) for ti in range(24)]
        t_wall = [float(np.mean([x["online_wall_time_sec"] for x in sim_by_mt[(arm, ti)]])) for ti in range(24)]
        all_r = [x for ti in range(24) for x in sim_by_mt[(arm, ti)]]
        cert_c = sum(1 for x in all_r if x["certified"])
        false_c = sum(1 for x in all_r if x["certified"] and (x["deployed_viol_prob"] or 0.0) > 0.05)
        sim_summary["arms"][arm] = {
            "n_tasks": 24,
            "n_runs": len(all_r),
            "mean_sq_loss": _stab_float(float(np.mean(t_loss))),
            "se_sq_loss": _stab_float(float(np.std(t_loss, ddof=1) / np.sqrt(24))),
            "cert_count": cert_c,
            "cert_rate": _stab_float(cert_c / len(all_r)),
            "audit_point_exceed_count": false_c,
            "mean_operating_viol_prob": _stab_float(float(np.mean(t_viol))),
            "mean_total_calls": _stab_float(float(np.mean(t_calls))),
            "mean_wall_sec": _stab_float(float(np.mean(t_wall))),
        }
    ref_sim = np.array([np.mean([x["status_quo_loss"] for x in sim_by_mt[("GP-FTA-ECKG", ti)]]) for ti in range(24)])
    sim_comps = [a for a in SIMOPT_ARMS if a != "GP-FTA-ECKG"]
    raw_ps_sim = []
    for comp in sim_comps:
        c_vec = np.array([np.mean([x["status_quo_loss"] for x in sim_by_mt[(comp, ti)]]) for ti in range(24)])
        st = _paired_stats(c_vec - ref_sim)
        sim_summary["contrasts_vs_eckg"][comp] = st
        raw_ps_sim.append(st["p_two_sided"])
    for comp, hp in zip(sim_comps, _holm_adjust(raw_ps_sim)):
        sim_summary["contrasts_vs_eckg"][comp]["p_holm_4"] = _stab_float(hp)

    # 3. Summarize confirm_fresh_72 calibration, 4-pillar ablations, meta-BO baselines, M_src, shift & noise
    c72_items.sort(key=lambda z: (z["family"], int(z["target_task_idx"])))

    model_names = list(c72_items[0]["main_shift13"]["models"].keys())
    metrics_list = ("gate_power_rmse", "feas_brier", "feas_auroc", "sel_true_slot0_gain", "con_nrmse", "obj_nrmse")

    pooled_models: dict[str, Any] = {}
    for m in model_names:
        m_rec: dict[str, Any] = {}
        for met in metrics_list:
            vals = [it["main_shift13"]["models"][m].get(met) for it in c72_items if it["main_shift13"]["models"][m].get(met) is not None]
            m_rec[met] = _stab_float(float(np.mean(vals))) if vals else None
            if met == "feas_auroc":
                m_rec["feas_auroc_defined_tasks"] = len(vals)
        pooled_models[m] = m_rec

    by_family_models: dict[str, Any] = {}
    for fam in DES_FAMILIES:
        sub_f = [it for it in c72_items if it["family"] == fam]
        f_blk: dict[str, Any] = {}
        for m in model_names:
            f_rec: dict[str, Any] = {}
            for met in metrics_list:
                vals = [it["main_shift13"]["models"][m].get(met) for it in sub_f if it["main_shift13"]["models"][m].get(met) is not None]
                f_rec[met] = _stab_float(float(np.mean(vals))) if vals else None
            f_blk[m] = f_rec
        by_family_models[fam] = f_blk

    # Paired contrasts on gate_power_rmse and con_nrmse vs Online-FullSURE (lower is better -> Comparator - FullSURE > 0 favors FullSURE)
    calib_contrasts: dict[str, Any] = {}
    meta_comps = ["ColdPsi-GP", "Online-SameBasisWeak", "ABLR-on-Psi", "MTGP-ICM-on-Psi", "RGPE-on-Psi"]
    pillar_comps = ["Pillar-NoDebias", "Pillar-NoLowPass", "Pillar-NoOrtho", "Pillar-NoSparse", "Psi-DropVol7D", "Psi-GenericBlackBox", "ForecastNoise15-FullSURE"]
    for met in ("gate_power_rmse", "feas_brier", "con_nrmse"):
        ref_v = np.array([float(it["main_shift13"]["models"]["Online-FullSURE"][met]) for it in c72_items])
        blk = {}
        ps_m = []
        for c in meta_comps:
            cv = np.array([float(it["main_shift13"]["models"][c][met]) for it in c72_items])
            st = _paired_stats(cv - ref_v)
            blk[c] = st
            ps_m.append(st["p_two_sided"])
        for c, hp in zip(meta_comps, _holm_adjust(ps_m)):
            blk[c]["p_holm_meta"] = _stab_float(hp)
        ps_p = []
        for c in pillar_comps:
            cv = np.array([float(it["main_shift13"]["models"][c][met]) for it in c72_items])
            st = _paired_stats(cv - ref_v)
            blk[c] = st
            ps_p.append(st["p_two_sided"])
        for c, hp in zip(pillar_comps, _holm_adjust(ps_p)):
            blk[c]["p_holm_pillar"] = _stab_float(hp)
        calib_contrasts[met] = blk

    # Standardized selection true Slot-0 gain contrast (higher is better -> FullSURE - Comparator > 0 favors FullSURE)
    sel_blk = {}
    for c in meta_comps + pillar_comps:
        diffs_s = []
        for fam in DES_FAMILIES:
            sub = [it for it in c72_items if it["family"] == fam]
            fv = np.array([float(it["main_shift13"]["models"]["Online-FullSURE"]["sel_true_slot0_gain"]) for it in sub])
            cv = np.array([float(it["main_shift13"]["models"][c]["sel_true_slot0_gain"]) for it in sub])
            sd_f = max(float(np.std(fv, ddof=1)), 1e-6)
            diffs_s.append((fv - cv) / sd_f)
        sel_blk[c] = _paired_stats(np.concatenate(diffs_s))
    calib_contrasts["sel_true_slot0_gain_std"] = sel_blk

    # M_src scaling summary (M = 3, 6, 12)
    msrc_keys = list(c72_items[0]["main_shift13"]["msrc_scaling"].keys())
    msrc_summary: dict[str, Any] = {"levels": {}, "contrasts": {}}
    for mk in msrc_keys:
        msrc_summary["levels"][mk] = {
            met: _stab_float(float(np.mean([
                it["main_shift13"]["msrc_scaling"][mk].get(met)
                for it in c72_items if it["main_shift13"]["msrc_scaling"][mk].get(met) is not None
            ])))
            for met in ("con_nrmse", "obj_nrmse", "gate_power_rmse")
        }
    for m_tag in ("M3", "M6", "M12"):
        df_con = np.array([
            float(it["main_shift13"]["msrc_scaling"][f"{m_tag}_KernelDiag"]["con_nrmse"])
            - float(it["main_shift13"]["msrc_scaling"][f"{m_tag}_KernelFull"]["con_nrmse"])
            for it in c72_items
        ])
        df_obj = np.array([
            float(it["main_shift13"]["msrc_scaling"][f"{m_tag}_KernelDiag"]["obj_nrmse"])
            - float(it["main_shift13"]["msrc_scaling"][f"{m_tag}_KernelFull"]["obj_nrmse"])
            for it in c72_items
        ])
        vs_cold_gate = np.array([
            float(it["main_shift13"]["models"]["ColdPsi-GP"]["gate_power_rmse"])
            - float(it["main_shift13"]["msrc_scaling"][f"{m_tag}_OnlineFull"]["gate_power_rmse"])
            for it in c72_items
        ])
        msrc_summary["contrasts"][m_tag] = {
            "diag_minus_full_con_nrmse": _paired_stats(df_con),
            "diag_minus_full_obj_nrmse": _paired_stats(df_obj),
            "cold_minus_online_gate_rmse": _paired_stats(vs_cold_gate),
        }

    # Shift-scale sensitivity {1.0, 1.3, 1.6}
    shift_summary: dict[str, Any] = {}
    for sh_k in ("shift_10", "shift_13", "shift_16"):
        f_gate = np.array([float(it["shift_sensitivity"][sh_k]["Online-FullSURE"]["gate_power_rmse"]) for it in c72_items])
        c_gate = np.array([float(it["shift_sensitivity"][sh_k]["ColdPsi-GP"]["gate_power_rmse"]) for it in c72_items])
        f_con = np.array([float(it["shift_sensitivity"][sh_k]["Online-FullSURE"]["con_nrmse"]) for it in c72_items])
        c_con = np.array([float(it["shift_sensitivity"][sh_k]["ColdPsi-GP"]["con_nrmse"]) for it in c72_items])
        shift_summary[sh_k] = {
            "full_sure_gate_rmse": _stab_float(float(np.mean(f_gate))),
            "cold_psi_gate_rmse": _stab_float(float(np.mean(c_gate))),
            "cold_minus_full_gate_rmse": _paired_stats(c_gate - f_gate),
            "full_sure_con_nrmse": _stab_float(float(np.mean(f_con))),
            "cold_psi_con_nrmse": _stab_float(float(np.mean(c_con))),
            "cold_minus_full_con_nrmse": _paired_stats(c_con - f_con),
        }

    readout = _stabilize_floats({
        "protocol_id": "PREREGISTRATION_REV7_OR_PANEL_20261121",
        "execution_code_sha256": exec_sha,
        "frozen24_latent_nei_and_sensitivity": {
            "overall_wtl_eckg_vs_latent_nei": wtl_lnei,
            "by_group": by_group_lnei,
            "per_state": froz_states,
        },
        "simopt_tandem_generic_psi_benchmark": sim_summary,
        "confirm_fresh_72_calibration_and_pillars": {
            "pooled_means": pooled_models,
            "by_family": by_family_models,
            "contrasts_vs_online_full_sure": calib_contrasts,
        },
        "msrc_scaling_3_6_12": msrc_summary,
        "shift_scale_sensitivity_10_13_16": shift_summary,
    })
    _write_json(summary_path, readout)
    return readout


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_d = sub.add_parser("describe")
    p_d.add_argument("--stage", choices=["stage0", "stage1", "stage2"], required=True)
    p_d.add_argument("--out-dir", type=Path, default=OUT_ROOT)

    sub.add_parser("finish_stage0").add_argument("--out-dir", type=Path, default=OUT_ROOT)

    p_w = sub.add_parser("worker")
    p_w.add_argument("--stage", choices=["stage0", "stage1", "stage2"], required=True)
    p_w.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    p_w.add_argument("--start", type=int, required=True)
    p_w.add_argument("--end", type=int, required=True)
    p_w.add_argument("--workers", type=int, default=1)

    p_s = sub.add_parser("summarize")
    p_s.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    p_s.add_argument(
        "--summary-path",
        type=Path,
        default=SC_ROOT / "paper_artifacts" / "v2_or_rev7_or_panel_v1" / "rev7_prior_and_simopt_readout.json",
    )

    args = p.parse_args()
    if args.cmd == "describe":
        d = describe_stage(args.stage, args.out_dir)
        print(json.dumps({"stage": d["stage"], "n_jobs": d["n_jobs"]}))
    elif args.cmd == "finish_stage0":
        print(json.dumps(finish_stage0(args.out_dir)))
    elif args.cmd == "worker":
        run_worker(args.stage, args.out_dir, args.start, args.end, args.workers)
    elif args.cmd == "summarize":
        ro = summarize(args.out_dir, args.summary_path)
        print(json.dumps({"wtl_vs_latent_nei": ro["frozen24_latent_nei_and_sensitivity"]["overall_wtl_eckg_vs_latent_nei"]}))


if __name__ == "__main__":
    main()
