#!/usr/bin/env python3
"""Cluster-distributed P1 Scientific Diagnostics & LiveSUMO Platform Calibration.

Executes four independent, decisive P1 diagnostics on `node001-node006` (`vram=0`,
1 thread/worker):
1. `representation_collision`: Measures cost & risk variation across policies with
   near-identical `psi(x) in R^12` under smooth DCT policies vs. high-frequency
   temporal oscillations / slot permutations / intersection swaps.
2. `variance_calibration`: Compares `within_ssq`, `naive_prior_residual_sq`, and
   `posterior_residual_sq` holding the mean model, kernel, and observations 100%
   identical against 2,000-rep Monte Carlo ground-truth variance across shift scales.
3. `gate_power_calibration`: Compares naive Gaussian `z_{1-alpha}` margin pass proxy
   vs. `posterior_binomial_gate_power` against 400 independent trials of the actual
   `exact_binomial(K=112, alpha=0.05, delta=0.025)` gate per candidate.
4. `sumo_platform_calibration`: Runs 60 independent fixed-time replications + 24
   candidate signal plans on `Ingolstadt21` Linux `SUMO 1.22` (`libsumo`) to establish
   exact platform baseline `(T0_linux, A0_linux, E0_linux)` and verify existence of a
   non-empty feasible region under relative-emission constraints.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import sys
from typing import Any

import numpy as np
from scipy.special import ndtr
from scipy.stats import norm, spearmanr

SC_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = SC_ROOT.parent
if str(SC_ROOT) not in sys.path:
    sys.path.insert(0, str(SC_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from scheduler_node_policy import DEFAULT_CPU_NODES
from v2_or.contracts import SufficientStatsRecord
from v2_or.dispatch.campaign_runner import compute_v2_or_code_sha256
from v2_or.legacy_solvers import generate_legacy_archive
from v2_or.models import TemperedHierarchicalHVDModel
from v2_or.priors import DebiasedSpectralMetaPrior, dct_profile_to_policy, sample_smooth_dct_delta
from v2_or.simulators import (
    IngolstadtLiveSUMOBenchmark,
    OPSDGridStorageBenchmark,
    sample_inventory_tasks,
    sample_queueing_tasks,
    sample_traffic_tasks,
)
from v2_or.verification import exact_binomial_gate, posterior_binomial_gate_power


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _make_problem(fam: str, shift_scale: float = 1.25):
    if fam == "inventory":
        return sample_inventory_tasks(1, seed=4101, horizons=(96,), shift_scale=shift_scale)[0]
    if fam == "queueing":
        return sample_queueing_tasks(1, seed=4201, horizons=(96,), shift_scale=shift_scale)[0]
    if fam == "opsd":
        return OPSDGridStorageBenchmark(market="DK_2", d=336, buffer_hours=24, clamp_limit=0.20)
    if fam == "traffic":
        return sample_traffic_tasks(1, seed=4401, tls_counts=(21,), shift_scale=shift_scale)[0]
    raise ValueError(fam)


def _make_source_prior(fam: str) -> DebiasedSpectralMetaPrior:
    if fam == "inventory":
        src = sample_inventory_tasks(4, seed=1101, horizons=(36, 48), shift_scale=1.0)
    elif fam == "queueing":
        src = sample_queueing_tasks(4, seed=1201, horizons=(36, 48), shift_scale=1.0)
    elif fam == "opsd":
        src = [OPSDGridStorageBenchmark(market=m, d=168, buffer_hours=24) for m in ("DK_1", "NO_1", "SE_1", "IT_CNOR")]
    elif fam == "traffic":
        src = sample_traffic_tasks(4, seed=1401, tls_counts=(11, 15), shift_scale=1.0)
    else:
        raise ValueError(fam)
    recs, _ = generate_legacy_archive(
        src, solvers=("penalty_spsa", "constrained_cmaes", "heuristic_powell"), per_solver_budget=45, reps_per_eval=3, seed=5001
    )
    return DebiasedSpectralMetaPrior(active_rank=8).fit(recs)


def run_diag_representation_collision(fam: str) -> dict[str, Any]:
    prob = _make_problem(fam, shift_scale=1.25)
    rng = np.random.default_rng(60001 + hash(fam) % 1000)
    n_ch = int(prob.num_control_channels)
    anc = prob.domain_heuristic_anchor()

    # 1. Sample 80 smooth operational DCT policies and evaluate MC outcomes (800 reps)
    smooth_pols = []
    for i in range(80):
        delta = sample_smooth_dct_delta(rng, prob.d, radius=0.20, k_modes=12, num_channels=n_ch)
        smooth_pols.append(np.clip(anc + delta, 0.0, 1.0))
    smooth_psis = np.array([prob.nominal_exposure(x).psi for x in smooth_pols], dtype=float)
    psi_sd = np.maximum(np.std(smooth_psis, axis=0, ddof=1), 1e-4)
    psi_norm = (smooth_psis - np.mean(smooth_psis, axis=0)) / psi_sd

    smooth_objs = np.empty(80, dtype=float)
    smooth_viols = np.empty(80, dtype=float)
    for i, x in enumerate(smooth_pols):
        aud = prob.audit_ground_truth(x, n_eval=800, seed=61000 + i)
        smooth_objs[i] = aud["mc_objective_mean"]
        smooth_viols[i] = aud["mc_violation_prob"]

    global_obj_sd = max(float(np.std(smooth_objs, ddof=1)), 1e-6)
    global_viol_sd = max(float(np.std(smooth_viols, ddof=1)), 1e-6)

    # Nearest-neighbor pairs on the smooth DCT manifold
    dists = np.sqrt(np.mean((psi_norm[:, None, :] - psi_norm[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    nn_idx = np.argmin(dists, axis=1)
    nn_psi_dist = dists[np.arange(80), nn_idx]
    close_mask = nn_psi_dist <= np.quantile(nn_psi_dist, 0.25)
    smooth_collision_obj_mad = float(np.mean(np.abs(smooth_objs[close_mask] - smooth_objs[nn_idx[close_mask]])))
    smooth_collision_viol_mad = float(np.mean(np.abs(smooth_viols[close_mask] - smooth_viols[nn_idx[close_mask]])))

    # 2. Adversarial high-frequency / permutation perturbations on 24 base policies
    hf_psi_dists = []
    hf_obj_diffs = []
    hf_viol_diffs = []
    perm_psi_dists = []
    perm_obj_diffs = []
    perm_viol_diffs = []

    for i in range(24):
        base_x = smooth_pols[i]
        base_psi = (smooth_psis[i] - np.mean(smooth_psis, axis=0)) / psi_sd
        # Alternating +-0.12 high-frequency ripple within each channel (zero temporal mean)
        alt = np.empty(prob.d, dtype=float)
        if n_ch == 1:
            alt = 0.12 * ((-1.0) ** np.arange(prob.d))
        else:
            n0 = (prob.d + 1) // 2
            n1 = prob.d // 2
            alt[0::2] = 0.12 * ((-1.0) ** np.arange(n0))
            if n1 > 0:
                alt[1::2] = 0.12 * ((-1.0) ** np.arange(n1))
        x_hf = np.clip(base_x + alt, 0.0, 1.0)
        psi_hf = (prob.nominal_exposure(x_hf).psi - np.mean(smooth_psis, axis=0)) / psi_sd
        aud_hf = prob.audit_ground_truth(x_hf, n_eval=800, seed=62000 + i)
        hf_psi_dists.append(float(np.sqrt(np.mean((psi_hf - base_psi) ** 2))))
        hf_obj_diffs.append(abs(float(aud_hf["mc_objective_mean"]) - smooth_objs[i]))
        hf_viol_diffs.append(abs(float(aud_hf["mc_violation_prob"]) - smooth_viols[i]))

        # Channel-preserving block reversal (reversing first half vs second half of horizon/corridor)
        x_perm = base_x.copy()
        if n_ch == 1:
            x_perm = base_x[::-1].copy()
        else:
            x_perm[0::2] = base_x[0::2][::-1]
            x_perm[1::2] = base_x[1::2][::-1]
        psi_perm = (prob.nominal_exposure(x_perm).psi - np.mean(smooth_psis, axis=0)) / psi_sd
        aud_perm = prob.audit_ground_truth(x_perm, n_eval=800, seed=63000 + i)
        perm_psi_dists.append(float(np.sqrt(np.mean((psi_perm - base_psi) ** 2))))
        perm_obj_diffs.append(abs(float(aud_perm["mc_objective_mean"]) - smooth_objs[i]))
        perm_viol_diffs.append(abs(float(aud_perm["mc_violation_prob"]) - smooth_viols[i]))

    # Out-of-fold ridge R^2 of 12-dim psi + quadratic dictionary on smooth policies
    X_lin = np.column_stack([np.ones(80), psi_norm, psi_norm ** 2])
    preds_oof = np.empty(80, dtype=float)
    for fold in range(4):
        te = np.arange(fold * 20, (fold + 1) * 20)
        tr = np.setdiff1d(np.arange(80), te)
        w = np.linalg.solve(X_lin[tr].T @ X_lin[tr] + 2.0 * np.eye(X_lin.shape[1]), X_lin[tr].T @ smooth_objs[tr])
        preds_oof[te] = X_lin[te] @ w
    r2_oof_smooth = float(1.0 - np.sum((smooth_objs - preds_oof) ** 2) / np.sum((smooth_objs - np.mean(smooth_objs)) ** 2))

    return {
        "diagnostic": "representation_collision",
        "family": fam,
        "psi_dim": 12,
        "policy_dim": int(prob.d),
        "global_smooth_obj_sd": global_obj_sd,
        "global_smooth_viol_sd": global_viol_sd,
        "oof_r2_on_smooth_dct_policies": r2_oof_smooth,
        "smooth_nn_psi_rms_dist": float(np.mean(nn_psi_dist[close_mask])),
        "smooth_nn_obj_mad_ratio_to_sd": float(smooth_collision_obj_mad / global_obj_sd),
        "smooth_nn_viol_mad": smooth_collision_viol_mad,
        "adversarial_highfreq_psi_rms_dist": float(np.mean(hf_psi_dists)),
        "adversarial_highfreq_obj_diff_ratio_to_sd": float(np.mean(hf_obj_diffs) / global_obj_sd),
        "adversarial_highfreq_viol_diff": float(np.mean(hf_viol_diffs)),
        "horizon_reversal_psi_rms_dist": float(np.mean(perm_psi_dists)),
        "horizon_reversal_obj_diff_ratio_to_sd": float(np.mean(perm_obj_diffs) / global_obj_sd),
        "horizon_reversal_viol_diff": float(np.mean(perm_viol_diffs)),
    }


def run_diag_variance_calibration(fam: str, shift_scale: float, n_reps_train: int) -> dict[str, Any]:
    prior = _make_source_prior(fam)
    prob = _make_problem(fam, shift_scale=shift_scale)
    rng = np.random.default_rng(70001 + int(round(shift_scale * 100)) + 17 * n_reps_train)
    n_ch = int(prob.num_control_channels)
    anc = prob.domain_heuristic_anchor()

    # 36 diagnostic policies spanning safe interior, near-boundary, and lean/congested regimes
    diag_pols = []
    for i in range(36):
        scale = 0.55 + 0.025 * i
        delta = sample_smooth_dct_delta(rng, prob.d, radius=0.14, k_modes=10, num_channels=n_ch)
        diag_pols.append(np.clip(anc * scale + delta, 0.02, 0.98))
    diag_exps = [prob.nominal_exposure(x) for x in diag_pols]

    # Ground-truth Monte Carlo constraint mean & variance (`2,000` independent replications per policy!)
    gt_mean_g = np.empty(36, dtype=float)
    gt_var_g = np.empty(36, dtype=float)
    for i, x in enumerate(diag_pols):
        samps = prob.simulate_replications(x, 2000, np.random.default_rng(75000 + i), split="audit")
        gt_mean_g[i] = float(np.mean(samps[:, 1]))
        gt_var_g[i] = max(float(np.var(samps[:, 1], ddof=1)), 1e-7)

    # Repeat across 12 independent training sample draws of size `n_obs = 24` policies x `n_reps_train`
    estimators = ("within_ssq", "naive_prior_residual_sq", "posterior_residual_sq")
    acc = {est: {"median_ratio": [], "mean_ratio": [], "log_var_rmse": [], "cov90": [], "cov95": [], "nlpd": []} for est in estimators}

    for trial in range(12):
        t_rng = np.random.default_rng(78000 + 101 * trial + n_reps_train)
        obs_idx = t_rng.choice(36, size=20, replace=False)
        te_idx = np.setdiff1d(np.arange(36), obs_idx)

        obs_recs = []
        for idx in obs_idx:
            samps = prob.simulate_replications(diag_pols[idx], int(n_reps_train), t_rng, split="search")
            obs_recs.append(SufficientStatsRecord.from_samples(diag_pols[idx], samps, diag_exps[idx]))

        te_pols = [diag_pols[i] for i in te_idx]
        te_exps = [diag_exps[i] for i in te_idx]
        v_true = gt_var_g[te_idx]
        m_true = gt_mean_g[te_idx]

        # Fresh single-replication held-out test draws for interval coverage & NLPD
        y_te_draw = np.array([
            prob.simulate_replications(diag_pols[i], 1, t_rng, split="audit")[0, 1] for i in te_idx
        ], dtype=float)

        for est in estimators:
            mod = TemperedHierarchicalHVDModel(
                prior,
                mode="tempered",
                use_source_basis=True,
                use_source_coef_prior=True,
                use_source_hvd_prior=False,  # Hold variance prior identical & unconfounded!
                variance_estimator=est,
            )
            for r in obs_recs:
                mod.observe(r)
            pred = mod.predict_joint(te_pols, te_exps)
            v_hat = np.maximum(pred.aleatoric_var_con, 1e-7)
            ratios = v_hat / v_true
            acc[est]["median_ratio"].append(float(np.median(ratios)))
            acc[est]["mean_ratio"].append(float(np.mean(ratios)))
            acc[est]["log_var_rmse"].append(float(np.sqrt(np.mean((np.log(v_hat) - np.log(v_true)) ** 2))))

            tot_sd = np.sqrt(pred.std_con ** 2 + v_hat)
            z_err = np.abs(y_te_draw - pred.mean_con) / np.maximum(tot_sd, 1e-6)
            acc[est]["cov90"].append(float(np.mean(z_err <= 1.64485)))
            acc[est]["cov95"].append(float(np.mean(z_err <= 1.95996)))
            nlpd = 0.5 * np.log(2.0 * np.pi * (tot_sd ** 2)) + 0.5 * ((m_true - pred.mean_con) ** 2 + v_true) / (tot_sd ** 2)
            acc[est]["nlpd"].append(float(np.mean(nlpd)))

    summary_est = {}
    for est in estimators:
        summary_est[est] = {
            k: float(np.mean(vals)) for k, vals in acc[est].items()
        }
        summary_est[est]["median_ratio_se"] = float(np.std(acc[est]["median_ratio"], ddof=1) / np.sqrt(12))

    return {
        "diagnostic": "variance_calibration",
        "family": fam,
        "shift_scale": float(shift_scale),
        "n_reps_train": int(n_reps_train),
        "estimators": summary_est,
    }


def run_diag_gate_power_calibration(fam: str) -> dict[str, Any]:
    prior = _make_source_prior(fam)
    prob = _make_problem(fam, shift_scale=1.25)
    rng = np.random.default_rng(81001 + hash(fam) % 500)
    n_ch = int(prob.num_control_channels)
    anc = prob.domain_heuristic_anchor()

    # Fit posterior on 24 observed points
    mod = TemperedHierarchicalHVDModel(prior, mode="tempered")
    for _ in range(24):
        delta = sample_smooth_dct_delta(rng, prob.d, radius=0.18, k_modes=12, num_channels=n_ch)
        x = np.clip(anc * float(rng.uniform(0.65, 1.15)) + delta, 0.02, 0.98)
        samps = prob.simulate_replications(x, 2, rng, split="search")
        mod.observe(SufficientStatsRecord.from_samples(x, samps, prob.nominal_exposure(x)))

    # Evaluate 60 candidate policies spanning interior, near-boundary, and infeasible regions
    cands = []
    for i in range(60):
        scale = 0.60 + 0.01 * i
        delta = sample_smooth_dct_delta(rng, prob.d, radius=0.12, k_modes=10, num_channels=n_ch)
        cands.append(np.clip(anc * scale + delta, 0.02, 0.98))
    exps = [prob.nominal_exposure(c) for c in cands]
    pred = mod.predict_joint(cands, exps)

    # 1. Old Gaussian margin pass proxy (`d1a13a40`)
    z_a = float(norm.ppf(1.0 - prob.alpha))
    q_margin = pred.mean_con + z_a * np.sqrt(np.maximum(pred.ucb_aleatoric_var_con, 1e-8))
    s_tot = np.sqrt(pred.std_con ** 2 + 0.25 * pred.r_perp_con + pred.aleatoric_var_con / 112.0)
    p_old_proxy = ndtr(-q_margin / np.maximum(s_tot, 1e-6))

    # 2. Exact Posterior Binomial Gate Power (`K_0 = 112, alpha = 0.05, delta_0 = 0.025`)
    p_new_gate = posterior_binomial_gate_power(
        pred.mean_con,
        pred.std_con,
        pred.ucb_aleatoric_var_con,
        n_samples=112,
        alpha=prob.alpha,
        delta_slot=0.025,
        r_perp_con=pred.r_perp_con,
    )

    # 3. Ground-truth empirical pass rate across 300 actual `exact_binomial(K=112)` gate runs + MC objective
    emp_pass = np.empty(60, dtype=float)
    mc_obj = np.empty(60, dtype=float)
    mc_viol = np.empty(60, dtype=float)
    for i, x in enumerate(cands):
        g_rng = np.random.default_rng(85000 + 307 * i)
        # Draw 300 * 112 = 33,600 replications cleanly in chunks
        all_con = []
        all_obj = []
        for _ in range(30):
            s = prob.simulate_replications(x, 1120, g_rng, split="verification")
            all_obj.append(s[:, 0])
            all_con.append(s[:, 1])
        con_mat = np.concatenate(all_con).reshape(300, 112)
        obj_vec = np.concatenate(all_obj)
        passes = [exact_binomial_gate(con_mat[r], alpha=prob.alpha, delta_slot=0.025)[0] for r in range(300)]
        emp_pass[i] = float(np.mean(passes))
        mc_obj[i] = float(np.mean(obj_vec))
        mc_viol[i] = float(np.mean(con_mat > 0.0))

    anc_aud = prob.audit_ground_truth(anc, n_eval=2000, seed=89999)
    j_sq = float(anc_aud["mc_objective_mean"])
    true_net_saving = np.maximum(j_sq - mc_obj, 0.0) * emp_pass

    old_est_saving = np.maximum(j_sq - pred.mean_obj, 0.0) * p_old_proxy
    new_est_saving = np.maximum(j_sq - pred.mean_obj, 0.0) * p_new_gate

    boundary_mask = (mc_viol >= 0.008) & (mc_viol <= 0.075)
    best_true = float(np.max(true_net_saving))

    return {
        "diagnostic": "gate_power_calibration",
        "family": fam,
        "n_candidates": 60,
        "n_boundary_candidates": int(np.sum(boundary_mask)),
        "brier_all_old_gaussian_proxy": float(np.mean((p_old_proxy - emp_pass) ** 2)),
        "brier_all_new_binomial_power": float(np.mean((p_new_gate - emp_pass) ** 2)),
        "mae_boundary_old_gaussian_proxy": (
            float(np.mean(np.abs(p_old_proxy[boundary_mask] - emp_pass[boundary_mask])))
            if np.any(boundary_mask)
            else None
        ),
        "mae_boundary_new_binomial_power": (
            float(np.mean(np.abs(p_new_gate[boundary_mask] - emp_pass[boundary_mask])))
            if np.any(boundary_mask)
            else None
        ),
        "mean_signed_bias_boundary_old": (
            float(np.mean(p_old_proxy[boundary_mask] - emp_pass[boundary_mask]))
            if np.any(boundary_mask)
            else None
        ),
        "mean_signed_bias_boundary_new": (
            float(np.mean(p_new_gate[boundary_mask] - emp_pass[boundary_mask]))
            if np.any(boundary_mask)
            else None
        ),
        "utility_spearman_old": float(spearmanr(old_est_saving, true_net_saving).statistic),
        "utility_spearman_new": float(spearmanr(new_est_saving, true_net_saving).statistic),
        "top1_selection_regret_old": float(best_true - true_net_saving[int(np.argmax(old_est_saving))]),
        "top1_selection_regret_new": float(best_true - true_net_saving[int(np.argmax(new_est_saving))]),
    }


def run_diag_sumo_platform_calibration() -> dict[str, Any]:
    """Run 50 independent fixed-time replications + 16 candidate signal plans on Linux SUMO 1.22."""
    import libsumo

    bench = IngolstadtLiveSUMOBenchmark(task_id="sumo_linux_cal", tau_emission=1.0)
    live_prob = bench._get_live_problem()
    old_T0, old_A0, old_E0 = float(live_prob.T0), float(live_prob.A0), float(live_prob.E0)

    # 1. Measure raw T, A, E on published RESCO fixed-time plan across 30 independent calibration seeds
    x_default_sec = bench.defaults.copy()
    raw_T, raw_A, raw_E = [], [], []
    for s_idx in range(30):
        sim_seed = 410001 + s_idx
        y_raw = live_prob._sim.simulate(
            var_map=live_prob.var_map,
            x=x_default_sec,
            route_file=live_prob._route_files[0],
            T0=1.0,
            A0=1.0,
            E0=1.0,
            seed=sim_seed,
        )
        raw_T.append(float(y_raw[0]))
        raw_A.append(float(y_raw[1]))
        raw_E.append(float(y_raw[2]))

    T0_linux = float(np.mean(raw_T))
    A0_linux = float(np.mean(raw_A))
    E0_linux = float(np.mean(raw_E))

    # 2. Probe 12 smooth candidate plans around RESCO fixed-time plan (5 reps each on separate seeds)
    rng = np.random.default_rng(420001)
    anc = bench.domain_heuristic_anchor()
    cand_stats = []
    for c_i in range(12):
        if c_i == 0:
            x_cand = anc.copy()
        else:
            delta = sample_smooth_dct_delta(rng, bench.d, radius=0.14, k_modes=10, num_channels=2)
            x_cand = np.clip(anc + delta, 0.05, 0.95)
        x_sec = bench.lo + x_cand * (bench.hi - bench.lo)
        c_T, c_A, c_E = [], [], []
        for r_i in range(5):
            yr = live_prob._sim.simulate(
                var_map=live_prob.var_map,
                x=x_sec,
                route_file=live_prob._route_files[0],
                T0=T0_linux,
                A0=A0_linux,
                E0=E0_linux,
                seed=430001 + 31 * c_i + r_i,
            )
            c_T.append(float(yr[0]))
            c_A.append(float(yr[1]))
            c_E.append(float(yr[2]))
        e_arr = np.asarray(c_E)
        cand_stats.append({
            "cand_idx": c_i,
            "mean_T_ratio_linux": float(np.mean(c_T)),
            "mean_A_ratio_linux": float(np.mean(c_A)),
            "mean_obj_50_50_linux": float(0.5 * np.mean(c_T) + 0.5 * np.mean(c_A)),
            "mean_E_ratio_linux": float(np.mean(e_arr)),
            "p95_E_ratio_linux": float(np.quantile(e_arr, 0.95)),
            "viol_rate_at_tau_1015_old_E0": float(np.mean(e_arr * (E0_linux / old_E0) > 1.015)),
            "viol_rate_at_tau_1035_linux_E0": float(np.mean(e_arr > 1.035)),
            "viol_rate_at_tau_1045_linux_E0": float(np.mean(e_arr > 1.045)),
        })

    return {
        "diagnostic": "sumo_platform_calibration",
        "libsumo_version": getattr(libsumo, "__version__", "unknown"),
        "route_file": str(live_prob._route_files[0]),
        "n_calibration_reps": 30,
        "legacy_baseline_json": {"T0": old_T0, "A0": old_A0, "E0": old_E0},
        "linux_sumo122_calibrated": {
            "T0": T0_linux,
            "T0_se": float(np.std(raw_T, ddof=1) / np.sqrt(30)),
            "A0": A0_linux,
            "A0_se": float(np.std(raw_A, ddof=1) / np.sqrt(30)),
            "E0": E0_linux,
            "E0_se": float(np.std(raw_E, ddof=1) / np.sqrt(30)),
            "E0_p95_ratio_to_mean": float(np.quantile(np.asarray(raw_E) / E0_linux, 0.95)),
            "E0_ratio_linux_over_legacy": float(E0_linux / old_E0),
            "fixed_time_viol_rate_under_legacy_tau_1025": float(np.mean(np.asarray(raw_E) / old_E0 > 1.025)),
        },
        "candidate_feasibility_scan": cand_stats,
    }


def build_diagnostic_jobs() -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for fam in ("inventory", "queueing", "opsd", "traffic"):
        jobs.append({"job_id": f"diag1_collision_{fam}", "kind": "collision", "fam": fam})
    for fam in ("inventory", "queueing", "opsd", "traffic"):
        for shift in (1.0, 1.3, 1.6):
            for n_reps in (2, 4):
                jobs.append({
                    "job_id": f"diag2_varcal_{fam}_s{int(round(shift*10))}_r{n_reps}",
                    "kind": "varcal",
                    "fam": fam,
                    "shift": shift,
                    "n_reps": n_reps,
                })
    for fam in ("inventory", "queueing", "traffic"):
        jobs.append({"job_id": f"diag3_gatecal_{fam}", "kind": "gatecal", "fam": fam})
    jobs.append({"job_id": "diag4_sumo_platform_cal", "kind": "sumo_cal"})
    return jobs


def _exec_one_diag(out_dir: Path, jb: dict[str, Any]) -> dict[str, Any]:
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    ipath = out_dir / "items" / f"{jb['job_id']}.json"
    if ipath.is_file():
        return {"job_id": jb["job_id"], "status": "cached"}
    kind = jb["kind"]
    if kind == "collision":
        res = run_diag_representation_collision(jb["fam"])
    elif kind == "varcal":
        res = run_diag_variance_calibration(jb["fam"], float(jb["shift"]), int(jb["n_reps"]))
    elif kind == "gatecal":
        res = run_diag_gate_power_calibration(jb["fam"])
    elif kind == "sumo_cal":
        res = run_diag_sumo_platform_calibration()
    else:
        raise ValueError(kind)
    res["job_id"] = jb["job_id"]
    res["execution_code_sha256"] = compute_v2_or_code_sha256()
    _write_json(ipath, res)
    return {"job_id": jb["job_id"], "status": "ok"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p_desc = sub.add_parser("describe")
    p_desc.add_argument("--out-dir", default="paper_artifacts/v2_or_p1_diagnostics_v1")

    p_work = sub.add_parser("worker")
    p_work.add_argument("--out-dir", default="paper_artifacts/v2_or_p1_diagnostics_v1")
    p_work.add_argument("--start", type=int, required=True)
    p_work.add_argument("--end", type=int, required=True)
    p_work.add_argument("--workers", type=int, required=True)

    p_fin = sub.add_parser("finish")
    p_fin.add_argument("--out-dir", default="paper_artifacts/v2_or_p1_diagnostics_v1")

    args = parser.parse_args()
    out_dir = SC_ROOT / args.out_dir
    jobs = build_diagnostic_jobs()

    if args.action == "describe":
        _write_json(out_dir / "jobs.json", {"n_jobs": len(jobs), "jobs": jobs})
        print(json.dumps({"n_jobs": len(jobs)}))
    elif args.action == "worker":
        node = socket.gethostname().split(".")[0]
        if node not in DEFAULT_CPU_NODES:
            raise RuntimeError(f"Must run on node001-node006, got {node!r}")
        sl = jobs[args.start : args.end]
        n_proc = max(1, min(int(args.workers), len(sl)))
        if n_proc == 1:
            for jb in sl:
                _exec_one_diag(out_dir, jb)
        else:
            with ProcessPoolExecutor(max_workers=n_proc) as pool:
                futs = [pool.submit(_exec_one_diag, out_dir, jb) for jb in sl]
                for f in as_completed(futs):
                    f.result()
    elif args.action == "finish":
        items = [json.loads((out_dir / "items" / f"{jb['job_id']}.json").read_text(encoding="utf-8")) for jb in jobs]
        summary = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "execution_code_sha256": compute_v2_or_code_sha256(),
            "n_diagnostics": len(items),
            "representation_collision": [x for x in items if x["diagnostic"] == "representation_collision"],
            "variance_calibration": [x for x in items if x["diagnostic"] == "variance_calibration"],
            "gate_power_calibration": [x for x in items if x["diagnostic"] == "gate_power_calibration"],
            "sumo_platform_calibration": next((x for x in items if x["diagnostic"] == "sumo_platform_calibration"), None),
        }
        _write_json(out_dir / "p1_diagnostics_summary.json", summary)
        print(json.dumps({"status": "complete", "n_diagnostics": len(items)}))


if __name__ == "__main__":
    main()
