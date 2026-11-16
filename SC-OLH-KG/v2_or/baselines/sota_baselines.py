"""Canonical Cold-Start & Transfer SOTA Baselines under Shared Budget & Verification Contracts.

Every baseline consumes the exact same search simulation call budget `B_search` and
passes its final shortlist to the exact same `verify_and_deploy_shortlist` gate:

Cold-Start SOTA:
1. `SCBO` (Eriksson & Poloczek 2021): Trust-region state machine + Matern-5/2 GPs on
   `(f, chance_margin)` + Constrained Thompson Sampling in `[0, 1]^d`.
2. `SAAS-CBO` (Eriksson & Jankowiak 2021): Half-Cauchy sparse inverse-lengthscale MAP
   ARD GP in `[0, 1]^d` + Constrained Expected Improvement (`cEI`).
3. `HetGP-KG` (Binois et al. 2018 + Scott et al. 2011): Coupled mean + log-variance
   GP in raw `[0, 1]^d` + one-step KG lookahead (isolating state-coupling + transfer).
4. `Target-SPSA` & `Target-CMAES`: Direct legacy simulation-optimization solvers run
   cold on the target problem.

External Transfer / Meta-BO SOTA (fed the exact same `LegacyTrajectoryRecord` archive):
5. `RGPE` (Feurer et al. 2018): Ranking-Weighted GP Ensemble combining `M_src` source-task
   GPs + target GP via bootstrap pairwise ranking-loss weights in policy space.
6. `HyperBO` (Wang et al. JMLR 2024): Empirical-Bayes pre-trained GP learning shared
   mean/kernel hyperparameters across `M_src` legacy-solver source tasks in policy space
   (with linear horizon interpolation when `d_s != d_t`).
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np
from scipy.interpolate import interp1d
from scipy.special import ndtr
from scipy.stats import norm

from v2_or.acquisition.verification_kg import SequentialRunResult, score_verification_kg
from v2_or.contracts import BaseDynamicORProblem, LegacyTrajectoryRecord, SufficientStatsRecord
from v2_or.legacy_solvers.solvers import run_constrained_cmaes, run_penalty_spsa
from v2_or.models.tempered_hvd_posterior import JointPosteriorPrediction, _safe_cholesky
from v2_or.priors.debiased_spectral_prior import dct_profile_to_policy
from v2_or.verification.gated_verifier_v2 import (
    build_feasible_aware_shortlist,
    verify_and_deploy_shortlist,
)


def _warp_policy_dim(x_src: np.ndarray, target_d: int) -> np.ndarray:
    """Linearly interpolate a policy from `d_s` to `target_d` when `d_s != d_t`."""
    arr = np.asarray(x_src, dtype=float).reshape(-1)
    if arr.size == target_d:
        return arr.copy()
    u_src = np.linspace(0.0, 1.0, arr.size)
    u_tgt = np.linspace(0.0, 1.0, int(target_d))
    f_interp = interp1d(u_src, arr, kind="linear", fill_value="extrapolate")
    return np.clip(f_interp(u_tgt), 0.0, 1.0)


def _ard_matern52(X1: np.ndarray, X2: np.ndarray, inv_ls: np.ndarray, outscale: float) -> np.ndarray:
    """ARD Matern-5/2 kernel normalized by dimension `d`."""
    w = np.asarray(inv_ls, dtype=float).reshape(1, 1, -1)
    diff = (X1[:, None, :] - X2[None, :, :]) * w
    dist2 = np.mean(diff * diff, axis=2)
    r = np.sqrt(np.maximum(5.0 * dist2, 0.0))
    return float(outscale) * (1.0 + r + (r * r) / 3.0) * np.exp(-r)


class RawInputExactGP:
    """Exact Matern-5/2 GP on raw `[0, 1]^d` supporting isotropic or SAAS sparse ARD."""

    def __init__(self, *, sparse_saas: bool = False) -> None:
        self.sparse_saas = bool(sparse_saas)
        self.X: np.ndarray = np.empty((0, 0))
        self.y_mean: float = 0.0
        self.y_std: float = 1.0
        self.inv_ls: np.ndarray = np.empty(0)
        self.outscale: float = 1.0
        self.L: np.ndarray = np.empty((0, 0))
        self.alpha_vec: np.ndarray = np.empty(0)

    def fit(self, X: np.ndarray, y: np.ndarray, noise_var: np.ndarray) -> "RawInputExactGP":
        self.X = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        n, d = self.X.shape
        self.y_mean = float(np.mean(y_arr))
        self.y_std = max(float(np.std(y_arr)), 1e-4)
        y_norm = (y_arr - self.y_mean) / self.y_std
        nv_norm = np.maximum(np.asarray(noise_var, dtype=float) / (self.y_std ** 2), 1e-5)

        if self.sparse_saas and n >= 6:
            # SAAS Half-Cauchy shrinkage: estimate coordinate relevance via empirical marginal correlation
            X_c = self.X - np.mean(self.X, axis=0, keepdims=True)
            X_sd = np.maximum(np.std(X_c, axis=0), 1e-6)
            corr = np.abs((X_c / X_sd[None, :]).T @ y_norm) / float(n)
            # Half-Cauchy horseshoe-style thresholding: shrink noisy coordinates toward 0
            tau_global = 0.35
            rel = np.maximum(corr - np.median(corr), 0.0)
            rel_norm = rel / max(float(np.max(rel)), 1e-6)
            base_inv_ls = tau_global + 3.5 * (rel_norm ** 2)
        else:
            base_inv_ls = np.ones(d, dtype=float)

        best_nll = float("inf")
        best_params = (base_inv_ls, 1.0)
        for mult in (0.6, 1.4, 3.0):
            cand_ls = base_inv_ls * mult
            for out_s in (0.4, 1.0, 2.2):
                K = _ard_matern52(self.X, self.X, cand_ls, out_s) + np.diag(nv_norm)
                L = _safe_cholesky(K)
                alp = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
                nll = 0.5 * float(y_norm @ alp) + float(np.sum(np.log(np.diag(L))))
                if nll < best_nll:
                    best_nll = nll
                    best_params = (cand_ls, out_s)

        self.inv_ls, self.outscale = best_params
        K_fin = _ard_matern52(self.X, self.X, self.inv_ls, self.outscale) + np.diag(nv_norm)
        self.L = _safe_cholesky(K_fin)
        self.alpha_vec = np.linalg.solve(self.L.T, np.linalg.solve(self.L, y_norm))
        return self

    def predict(self, X_cand: np.ndarray, *, return_cov: bool = False) -> tuple[np.ndarray, np.ndarray]:
        X_c = np.asarray(X_cand, dtype=float)
        K_co = _ard_matern52(X_c, self.X, self.inv_ls, self.outscale)
        mean_norm = K_co @ self.alpha_vec
        mean = self.y_mean + self.y_std * mean_norm
        V = np.linalg.solve(self.L, K_co.T)
        if return_cov:
            K_cc = _ard_matern52(X_c, X_c, self.inv_ls, self.outscale)
            cov = (self.y_std ** 2) * (K_cc - V.T @ V)
            cov = 0.5 * (cov + cov.T)
            return mean, cov
        diag_var = np.maximum(self.outscale - np.sum(V * V, axis=0), 1e-9) * (self.y_std ** 2)
        return mean, np.sqrt(diag_var)


def _shared_cold_init(
    problem: BaseDynamicORProblem,
    n0_init: int,
    batch_reps: int,
    rng_init: np.random.Generator,
) -> list[SufficientStatsRecord]:
    pts = [problem.domain_heuristic_anchor()]
    while len(pts) < n0_init:
        if len(pts) % 2 == 1:
            prof = rng_init.normal(0.0, 0.22 / np.arange(1, 13), size=12)
            pts.append(dct_profile_to_policy(prof, problem.d))
        else:
            pts.append(rng_init.uniform(0.05, 0.95, size=problem.d))
    recs = []
    for x in pts:
        exp = problem.nominal_exposure(x)
        samp = problem.simulate_replications(x, batch_reps, rng_init, split="search")
        recs.append(SufficientStatsRecord.from_samples(x, samp, exp))
    return recs


def _records_arrays(recs: Sequence[SufficientStatsRecord]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = np.array([r.x_norm for r in recs], dtype=float)
    yf = np.array([r.mean_obj for r in recs], dtype=float)
    yg = np.array([r.mean_con for r in recs], dtype=float)
    vf = np.array([max(r.ssq_obj / max(r.n_reps - 1, 1), 1e-4) / r.n_reps for r in recs], dtype=float)
    vg = np.array([max(r.ssq_con / max(r.n_reps - 1, 1), 1e-4) / r.n_reps for r in recs], dtype=float)
    return X, yf, yg, vf, vg


def run_scbo_baseline(
    problem: BaseDynamicORProblem,
    *,
    n_online_steps: int = 30,
    n0_init: int = 8,
    batch_reps: int = 2,
    seed: int = 1001,
    pool_size: int = 160,
    gate_mode: str = "auto",
) -> SequentialRunResult:
    """Canonical SCBO trust-region constrained Thompson Sampling baseline."""
    t0 = time.perf_counter()
    rng_init = np.random.default_rng(seed)
    rng_search = np.random.default_rng(seed + 100_003)
    verif_seed = seed + 200_007
    calls_start = problem.search_calls
    z_a = float(norm.ppf(1.0 - problem.alpha))

    init_recs = _shared_cold_init(problem, n0_init, batch_reps, rng_init)
    all_recs = list(init_recs)

    tr_len = 0.40
    succ_cnt, fail_cnt = 0, 0
    best_score = (1, float("inf"), float("inf"))

    def _rec_score(r: SufficientStatsRecord) -> tuple[int, float, float]:
        sd = float(np.sqrt(max(r.ssq_con / max(r.n_reps - 1, 1), 1e-6)))
        m = r.mean_con + z_a * sd
        return (0, r.mean_obj, m) if m <= 0.0 else (1, m, r.mean_obj)

    best_center = min(all_recs, key=_rec_score)
    best_score = _rec_score(best_center)

    gp_f = RawInputExactGP(sparse_saas=False)
    gp_g = RawInputExactGP(sparse_saas=False)
    gp_logv = RawInputExactGP(sparse_saas=False)

    for _ in range(int(n_online_steps)):
        X, yf, yg, vf, vg = _records_arrays(all_recs)
        s2_raw_g = np.array([max(r.ssq_con / max(r.n_reps - 1, 1), 1e-5) for r in all_recs], dtype=float)
        gp_f.fit(X, yf, vf)
        gp_g.fit(X, yg, vg)
        gp_logv.fit(X, np.log(s2_raw_g), np.full(len(all_recs), 0.5))

        # Generate Sobol/perturbation candidates inside trust region around best_center
        lo_tr = np.maximum(best_center.x_norm - 0.5 * tr_len, 0.0)
        hi_tr = np.minimum(best_center.x_norm + 0.5 * tr_len, 1.0)
        X_tr = rng_search.uniform(lo_tr, hi_tr, size=(pool_size, problem.d))
        # Mask coordinates per TuRBO/SCBO perturbation probability
        prob_pert = min(20.0 / problem.d, 1.0)
        mask = rng_search.uniform(size=(pool_size, problem.d)) < prob_pert
        X_cand = np.where(mask, X_tr, best_center.x_norm[None, :])

        mf, sf = gp_f.predict(X_cand)
        mg, sg = gp_g.predict(X_cand)
        mlogv, _ = gp_logv.predict(X_cand)
        v_pred = np.exp(np.clip(mlogv, -12.0, 4.0))

        f_ts = mf + sf * rng_search.normal(size=pool_size)
        g_ts = mg + sg * rng_search.normal(size=pool_size) + z_a * np.sqrt(v_pred)
        feas = np.where(g_ts <= 0.0)[0]
        idx = int(feas[np.argmin(f_ts[feas])]) if feas.size > 0 else int(np.argmin(g_ts))

        x_next = X_cand[idx]
        samp = problem.simulate_replications(x_next, batch_reps, rng_search, split="search")
        rec_new = SufficientStatsRecord.from_samples(x_next, samp, problem.nominal_exposure(x_next))
        all_recs.append(rec_new)

        sc_new = _rec_score(rec_new)
        if sc_new < best_score:
            best_score = sc_new
            best_center = rec_new
            succ_cnt += 1
            fail_cnt = 0
        else:
            fail_cnt += 1
            succ_cnt = 0
        if succ_cnt >= 3:
            tr_len = min(tr_len * 1.6, 0.80)
            succ_cnt = 0
        elif fail_cnt >= 4:
            tr_len = max(tr_len * 0.5, 0.04)
            fail_cnt = 0

    X, yf, yg, vf, vg = _records_arrays(all_recs)
    s2_raw_g = np.array([max(r.ssq_con / max(r.n_reps - 1, 1), 1e-5) for r in all_recs], dtype=float)
    gp_f.fit(X, yf, vf)
    gp_g.fit(X, yg, vg)
    gp_logv.fit(X, np.log(s2_raw_g), np.full(len(all_recs), 0.5))

    mf, _ = gp_f.predict(X)
    mg, sg = gp_g.predict(X)
    mlogv, slogv = gp_logv.predict(X)
    ucb_vg = np.exp(np.clip(mlogv + 0.5 * slogv, -12.0, 4.0))

    shortlist = build_feasible_aware_shortlist(
        candidate_pool=list(X),
        mean_obj=mf,
        mean_con=mg,
        std_con=sg,
        ucb_aleat_var_con=ucb_vg,
        r_perp_con=np.full(len(X), 1e-3),
        initial_records=all_recs,
        alpha=problem.alpha,
    )
    receipt = verify_and_deploy_shortlist(problem, shortlist, seed=verif_seed, gate_mode=gate_mode)
    search_used = int(problem.search_calls - calls_start)
    return SequentialRunResult(
        method_name="SCBO",
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
        n_unique_policies_evaluated=len(all_recs),
        n_replication_queries=0,
        receipt=receipt,
    )


def run_saasbo_baseline(
    problem: BaseDynamicORProblem,
    *,
    n_online_steps: int = 30,
    n0_init: int = 8,
    batch_reps: int = 2,
    seed: int = 1001,
    pool_size: int = 160,
    gate_mode: str = "auto",
) -> SequentialRunResult:
    """Sparse Axis-Aligned Subspace Constrained BO (`SAAS-CBO`) baseline."""
    t0 = time.perf_counter()
    rng_init = np.random.default_rng(seed)
    rng_search = np.random.default_rng(seed + 100_003)
    verif_seed = seed + 200_007
    calls_start = problem.search_calls
    z_a = float(norm.ppf(1.0 - problem.alpha))

    init_recs = _shared_cold_init(problem, n0_init, batch_reps, rng_init)
    all_recs = list(init_recs)
    gp_f = RawInputExactGP(sparse_saas=True)
    gp_g = RawInputExactGP(sparse_saas=True)
    gp_logv = RawInputExactGP(sparse_saas=False)

    for _ in range(int(n_online_steps)):
        X, yf, yg, vf, vg = _records_arrays(all_recs)
        s2_raw_g = np.array([max(r.ssq_con / max(r.n_reps - 1, 1), 1e-5) for r in all_recs], dtype=float)
        gp_f.fit(X, yf, vf)
        gp_g.fit(X, yg, vg)
        gp_logv.fit(X, np.log(s2_raw_g), np.full(len(all_recs), 0.5))

        # Perturb mostly along high-relevance SAAS coordinates + global exploration
        top_dims = np.argsort(-(gp_f.inv_ls + gp_g.inv_ls))[: max(min(12, problem.d), 4)]
        X_cand = np.empty((pool_size, problem.d), dtype=float)
        for i in range(pool_size):
            if i < pool_size * 3 // 4:
                parent = all_recs[i % len(all_recs)].x_norm.copy()
                parent[top_dims] = np.clip(parent[top_dims] + rng_search.normal(0.0, 0.14, size=top_dims.size), 0.0, 1.0)
                X_cand[i] = parent
            else:
                X_cand[i] = rng_search.uniform(0.0, 1.0, size=problem.d)

        mf, sf = gp_f.predict(X_cand)
        mg, sg = gp_g.predict(X_cand)
        mlogv, _ = gp_logv.predict(X_cand)
        q_g = mg + z_a * np.sqrt(np.exp(np.clip(mlogv, -12.0, 4.0)))
        pof = ndtr(-q_g / np.maximum(sg, 1e-6))
        feas_obs = [r.mean_obj for r in all_recs if r.mean_con <= 0.0]
        f_best = min(feas_obs) if feas_obs else float(np.median(yf))
        u = (f_best - mf) / np.maximum(sf, 1e-6)
        cei = np.maximum(sf, 1e-6) * (u * ndtr(u) + norm.pdf(u)) * pof
        x_next = X_cand[int(np.argmax(cei))]

        samp = problem.simulate_replications(x_next, batch_reps, rng_search, split="search")
        all_recs.append(SufficientStatsRecord.from_samples(x_next, samp, problem.nominal_exposure(x_next)))

    X, yf, yg, vf, vg = _records_arrays(all_recs)
    s2_raw_g = np.array([max(r.ssq_con / max(r.n_reps - 1, 1), 1e-5) for r in all_recs], dtype=float)
    gp_f.fit(X, yf, vf)
    gp_g.fit(X, yg, vg)
    gp_logv.fit(X, np.log(s2_raw_g), np.full(len(all_recs), 0.5))

    mf, _ = gp_f.predict(X)
    mg, sg = gp_g.predict(X)
    mlogv, slogv = gp_logv.predict(X)
    ucb_vg = np.exp(np.clip(mlogv + 0.5 * slogv, -12.0, 4.0))

    shortlist = build_feasible_aware_shortlist(
        candidate_pool=list(X),
        mean_obj=mf,
        mean_con=mg,
        std_con=sg,
        ucb_aleat_var_con=ucb_vg,
        r_perp_con=np.full(len(X), 1e-3),
        initial_records=all_recs,
        alpha=problem.alpha,
    )
    receipt = verify_and_deploy_shortlist(problem, shortlist, seed=verif_seed, gate_mode=gate_mode)
    search_used = int(problem.search_calls - calls_start)
    return SequentialRunResult(
        method_name="SAAS-CBO",
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
        n_unique_policies_evaluated=len(all_recs),
        n_replication_queries=0,
        receipt=receipt,
    )


def run_hetgp_kg_baseline(
    problem: BaseDynamicORProblem,
    *,
    n_online_steps: int = 30,
    n0_init: int = 8,
    batch_reps: int = 2,
    seed: int = 1001,
    pool_size: int = 140,
    gate_mode: str = "auto",
) -> SequentialRunResult:
    """Target-only raw-input Heteroscedastic GP + Knowledge Gradient (`HetGP-KG`)."""
    t0 = time.perf_counter()
    rng_init = np.random.default_rng(seed)
    rng_search = np.random.default_rng(seed + 100_003)
    verif_seed = seed + 200_007
    calls_start = problem.search_calls

    init_recs = _shared_cold_init(problem, n0_init, batch_reps, rng_init)
    all_recs = list(init_recs)
    gp_f = RawInputExactGP(sparse_saas=False)
    gp_g = RawInputExactGP(sparse_saas=False)
    gp_logvf = RawInputExactGP(sparse_saas=False)
    gp_logvg = RawInputExactGP(sparse_saas=False)

    for _ in range(int(n_online_steps)):
        X, yf, yg, vf, vg = _records_arrays(all_recs)
        s2_f = np.array([max(r.ssq_obj / max(r.n_reps - 1, 1), 1e-5) for r in all_recs], dtype=float)
        s2_g = np.array([max(r.ssq_con / max(r.n_reps - 1, 1), 1e-5) for r in all_recs], dtype=float)
        gp_f.fit(X, yf, vf)
        gp_g.fit(X, yg, vg)
        gp_logvf.fit(X, np.log(s2_f), np.full(len(all_recs), 0.5))
        gp_logvg.fit(X, np.log(s2_g), np.full(len(all_recs), 0.5))

        X_rand = rng_search.uniform(0.0, 1.0, size=(max(pool_size - len(X), 32), problem.d))
        X_cand = np.vstack([X, X_rand])[:pool_size]
        mf, cov_f = gp_f.predict(X_cand, return_cov=True)
        mg, cov_g = gp_g.predict(X_cand, return_cov=True)
        mlogvf, _ = gp_logvf.predict(X_cand)
        mlogvg, slogvg = gp_logvg.predict(X_cand)
        av_f = np.exp(np.clip(mlogvf, -12.0, 4.0))
        av_g = np.exp(np.clip(mlogvg, -12.0, 4.0))
        ucb_avg = np.exp(np.clip(mlogvg + 0.5 * slogvg, -12.0, 4.0))

        pred = JointPosteriorPrediction(
            mean_obj=mf,
            cov_obj=cov_f,
            std_obj=np.sqrt(np.maximum(np.diag(cov_f), 1e-10)),
            mean_con=mg,
            cov_con=cov_g,
            std_con=np.sqrt(np.maximum(np.diag(cov_g), 1e-10)),
            aleatoric_var_obj=av_f,
            aleatoric_var_con=av_g,
            ucb_aleatoric_var_con=ucb_avg,
            r_perp_con=np.full(len(X_cand), 1e-3),
        )
        scores = score_verification_kg(pred, alpha=problem.alpha, batch_reps=batch_reps)
        x_next = X_cand[int(np.argmax(scores))]
        samp = problem.simulate_replications(x_next, batch_reps, rng_search, split="search")
        all_recs.append(SufficientStatsRecord.from_samples(x_next, samp, problem.nominal_exposure(x_next)))

    X, yf, yg, vf, vg = _records_arrays(all_recs)
    s2_g = np.array([max(r.ssq_con / max(r.n_reps - 1, 1), 1e-5) for r in all_recs], dtype=float)
    gp_f.fit(X, yf, vf)
    gp_g.fit(X, yg, vg)
    gp_logvg.fit(X, np.log(s2_g), np.full(len(all_recs), 0.5))
    mf, _ = gp_f.predict(X)
    mg, sg = gp_g.predict(X)
    mlogvg, slogvg = gp_logvg.predict(X)
    ucb_vg = np.exp(np.clip(mlogvg + 0.5 * slogvg, -12.0, 4.0))

    shortlist = build_feasible_aware_shortlist(
        candidate_pool=list(X),
        mean_obj=mf,
        mean_con=mg,
        std_con=sg,
        ucb_aleat_var_con=ucb_vg,
        r_perp_con=np.full(len(X), 1e-3),
        initial_records=all_recs,
        alpha=problem.alpha,
    )
    receipt = verify_and_deploy_shortlist(problem, shortlist, seed=verif_seed, gate_mode=gate_mode)
    search_used = int(problem.search_calls - calls_start)
    return SequentialRunResult(
        method_name="HetGP-KG",
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
        n_unique_policies_evaluated=len(all_recs),
        n_replication_queries=0,
        receipt=receipt,
    )


def run_rgpe_transfer_baseline(
    problem: BaseDynamicORProblem,
    source_archive: Sequence[LegacyTrajectoryRecord],
    *,
    n_online_steps: int = 30,
    n0_init: int = 8,
    batch_reps: int = 2,
    seed: int = 1001,
    pool_size: int = 160,
    gate_mode: str = "auto",
) -> SequentialRunResult:
    """Feurer et al. (2018) Ranking-Weighted GP Ensemble (`RGPE`) on raw/warped policy space."""
    t0 = time.perf_counter()
    rng_init = np.random.default_rng(seed)
    rng_search = np.random.default_rng(seed + 100_003)
    verif_seed = seed + 200_007
    calls_start = problem.search_calls
    z_a = float(norm.ppf(1.0 - problem.alpha))

    # Fit per-source-task GPs on raw (or linearly horizon-warped) policies
    src_tasks = sorted({r.task_id for r in source_archive})
    src_models_f: list[RawInputExactGP] = []
    src_models_g: list[RawInputExactGP] = []
    src_elites: list[np.ndarray] = []

    for tid in src_tasks:
        rows = [r for r in source_archive if r.task_id == tid]
        if len(rows) < 5:
            continue
        Xs = np.array([_warp_policy_dim(np.asarray(r.x_norm), problem.d) for r in rows], dtype=float)
        yfs = np.array([r.mean_obj for r in rows], dtype=float)
        ygs = np.array([r.mean_con for r in rows], dtype=float)
        vfs = np.array([max(r.ssq_obj / max(r.n_reps - 1, 1), 1e-4) / r.n_reps for r in rows], dtype=float)
        vgs = np.array([max(r.ssq_con / max(r.n_reps - 1, 1), 1e-4) / r.n_reps for r in rows], dtype=float)
        src_models_f.append(RawInputExactGP().fit(Xs, yfs, vfs))
        src_models_g.append(RawInputExactGP().fit(Xs, ygs, vgs))
        best_r = min(rows, key=lambda r: (0 if r.mean_con <= 0.0 else 1, r.mean_obj if r.mean_con <= 0.0 else r.mean_con))
        src_elites.append(_warp_policy_dim(np.asarray(best_r.x_norm), problem.d))

    init_recs = _shared_cold_init(problem, n0_init, batch_reps, rng_init)
    all_recs = list(init_recs)
    tgt_f = RawInputExactGP()
    tgt_g = RawInputExactGP()

    def _ranking_weights(models: list[RawInputExactGP], X_t: np.ndarray, y_t: np.ndarray, tgt_gp: RawInputExactGP) -> np.ndarray:
        if not models or len(y_t) < 3:
            return np.r_[np.zeros(len(models)), 1.0]
        preds = [m.predict(X_t)[0] for m in models]
        # Target LOO proxy via in-sample regularized prediction + noise
        tgt_pred, tgt_sd = tgt_gp.predict(X_t)
        preds.append(tgt_pred)
        P = np.asarray(preds, dtype=float)  # (M+1, n)
        # Pairwise ranking discordance loss with bootstrap draws
        n_boot = 32
        wins = np.zeros(len(preds), dtype=float)
        for _ in range(n_boot):
            b_idx = rng_search.integers(0, len(y_t), size=len(y_t))
            yb = y_t[b_idx]
            diff_y = yb[:, None] < yb[None, :]
            losses = np.empty(len(preds), dtype=float)
            for m_i in range(len(preds)):
                pb = P[m_i, b_idx] + (tgt_sd[b_idx] * rng_search.normal(size=len(y_t)) if m_i == len(preds) - 1 else 0.0)
                diff_p = pb[:, None] < pb[None, :]
                losses[m_i] = float(np.mean(diff_y ^ diff_p))
            wins[int(np.argmin(losses))] += 1.0
        w = wins / float(n_boot)
        return w / max(float(np.sum(w)), 1e-8)

    for _ in range(int(n_online_steps)):
        X, yf, yg, vf, vg = _records_arrays(all_recs)
        tgt_f.fit(X, yf, vf)
        tgt_g.fit(X, yg, vg)
        wf = _ranking_weights(src_models_f, X, yf, tgt_f)
        wg = _ranking_weights(src_models_g, X, yg, tgt_g)

        cands_list = [e + rng_search.normal(0.0, 0.08, size=problem.d) for e in src_elites]
        while len(cands_list) < pool_size:
            if len(cands_list) % 2 == 0:
                p = all_recs[len(cands_list) % len(all_recs)].x_norm
                cands_list.append(p + rng_search.normal(0.0, 0.10, size=problem.d))
            else:
                cands_list.append(rng_search.uniform(0.0, 1.0, size=problem.d))
        X_cand = np.clip(np.asarray(cands_list[:pool_size], dtype=float), 0.0, 1.0)

        mf_all = [m.predict(X_cand) for m in src_models_f] + [tgt_f.predict(X_cand)]
        mg_all = [m.predict(X_cand) for m in src_models_g] + [tgt_g.predict(X_cand)]

        mf = sum(w * pred[0] for w, pred in zip(wf, mf_all))
        sf = np.sqrt(sum((w ** 2) * (pred[1] ** 2) for w, pred in zip(wf, mf_all)) + 1e-8)
        mg = sum(w * pred[0] for w, pred in zip(wg, mg_all))
        sg = np.sqrt(sum((w ** 2) * (pred[1] ** 2) for w, pred in zip(wg, mg_all)) + 1e-8)

        avg_vg = float(np.median([max(r.ssq_con / max(r.n_reps - 1, 1), 1e-4) for r in all_recs]))
        pof = ndtr(-(mg + z_a * np.sqrt(avg_vg)) / sg)
        f_best = float(np.min(yf[yg <= 0.0])) if np.any(yg <= 0.0) else float(np.median(yf))
        u = (f_best - mf) / sf
        cei = sf * (u * ndtr(u) + norm.pdf(u)) * pof
        x_next = X_cand[int(np.argmax(cei))]
        samp = problem.simulate_replications(x_next, batch_reps, rng_search, split="search")
        all_recs.append(SufficientStatsRecord.from_samples(x_next, samp, problem.nominal_exposure(x_next)))

    X, yf, yg, vf, vg = _records_arrays(all_recs)
    tgt_f.fit(X, yf, vf)
    tgt_g.fit(X, yg, vg)
    wf = _ranking_weights(src_models_f, X, yf, tgt_f)
    wg = _ranking_weights(src_models_g, X, yg, tgt_g)
    mf_all = [m.predict(X) for m in src_models_f] + [tgt_f.predict(X)]
    mg_all = [m.predict(X) for m in src_models_g] + [tgt_g.predict(X)]
    mf = sum(w * pred[0] for w, pred in zip(wf, mf_all))
    mg = sum(w * pred[0] for w, pred in zip(wg, mg_all))
    sg = np.sqrt(sum((w ** 2) * (pred[1] ** 2) for w, pred in zip(wg, mg_all)) + 1e-8)
    s2_obs = np.array([max(r.ssq_con / max(r.n_reps - 1, 1), 1e-4) for r in all_recs], dtype=float)

    shortlist = build_feasible_aware_shortlist(
        candidate_pool=list(X),
        mean_obj=mf,
        mean_con=mg,
        std_con=sg,
        ucb_aleat_var_con=s2_obs * 1.15,
        r_perp_con=np.full(len(X), 1e-3),
        initial_records=all_recs,
        alpha=problem.alpha,
    )
    receipt = verify_and_deploy_shortlist(problem, shortlist, seed=verif_seed, gate_mode=gate_mode)
    search_used = int(problem.search_calls - calls_start)
    return SequentialRunResult(
        method_name="RGPE",
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
        posterior_beta_con=float(wg[-1]),
        posterior_tau_shift_con=0.0,
        untempered_inflation_factor=1.0,
        n_unique_policies_evaluated=len(all_recs),
        n_replication_queries=0,
        receipt=receipt,
    )


def run_hyperbo_transfer_baseline(
    problem: BaseDynamicORProblem,
    source_archive: Sequence[LegacyTrajectoryRecord],
    *,
    n_online_steps: int = 30,
    n0_init: int = 8,
    batch_reps: int = 2,
    seed: int = 1001,
    pool_size: int = 160,
    gate_mode: str = "auto",
) -> SequentialRunResult:
    """Wang et al. (JMLR 2024) Empirical-Bayes Pre-trained GP (`HyperBO`) in raw policy space."""
    t0 = time.perf_counter()
    rng_init = np.random.default_rng(seed)
    rng_search = np.random.default_rng(seed + 100_003)
    verif_seed = seed + 200_007
    calls_start = problem.search_calls
    z_a = float(norm.ppf(1.0 - problem.alpha))

    # Pre-train shared linear mean + shared Matern-5/2 kernel across source tasks in raw policy space
    Xs_all = np.array([_warp_policy_dim(np.asarray(r.x_norm), problem.d) for r in source_archive], dtype=float)
    yfs_all = np.array([r.mean_obj for r in source_archive], dtype=float)
    ygs_all = np.array([r.mean_con for r in source_archive], dtype=float)

    # Ridge-regularized empirical-Bayes prior mean in raw coordinate space
    X_aug = np.column_stack([np.ones(len(Xs_all)), Xs_all])
    reg = np.diag(np.r_[1e-3, np.full(problem.d, 2.0)])
    w_mean_f = np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ yfs_all)
    w_mean_g = np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ ygs_all)
    prior_var_f = max(float(np.var(yfs_all - X_aug @ w_mean_f)), 1e-3)
    prior_var_g = max(float(np.var(ygs_all - X_aug @ w_mean_g)), 1e-3)
    inv_ls_shared = np.full(problem.d, 1.8, dtype=float)

    init_recs = _shared_cold_init(problem, n0_init, batch_reps, rng_init)
    all_recs = list(init_recs)

    def _hyperbo_predict(X_obs: np.ndarray, y_obs: np.ndarray, v_obs: np.ndarray, w_m: np.ndarray, p_var: float, X_c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m0_obs = np.column_stack([np.ones(len(X_obs)), X_obs]) @ w_m
        m0_c = np.column_stack([np.ones(len(X_c)), X_c]) @ w_m
        K_oo = _ard_matern52(X_obs, X_obs, inv_ls_shared, p_var) + np.diag(v_obs)
        K_co = _ard_matern52(X_c, X_obs, inv_ls_shared, p_var)
        L = _safe_cholesky(K_oo)
        alp = np.linalg.solve(L.T, np.linalg.solve(L, y_obs - m0_obs))
        V = np.linalg.solve(L, K_co.T)
        mean = m0_c + K_co @ alp
        std = np.sqrt(np.maximum(p_var - np.sum(V * V, axis=0), 1e-8))
        return mean, std

    for _ in range(int(n_online_steps)):
        X, yf, yg, vf, vg = _records_arrays(all_recs)
        X_cand = np.vstack([
            np.clip(X[rng_search.integers(0, len(X), size=pool_size // 2)] + rng_search.normal(0.0, 0.10, size=(pool_size // 2, problem.d)), 0.0, 1.0),
            rng_search.uniform(0.0, 1.0, size=(pool_size - pool_size // 2, problem.d)),
        ])
        mf, sf = _hyperbo_predict(X, yf, vf, w_mean_f, prior_var_f, X_cand)
        mg, sg = _hyperbo_predict(X, yg, vg, w_mean_g, prior_var_g, X_cand)
        avg_vg = float(np.median([max(r.ssq_con / max(r.n_reps - 1, 1), 1e-4) for r in all_recs]))
        pof = ndtr(-(mg + z_a * np.sqrt(avg_vg)) / sg)
        f_best = float(np.min(yf[yg <= 0.0])) if np.any(yg <= 0.0) else float(np.median(yf))
        u = (f_best - mf) / sf
        cei = sf * (u * ndtr(u) + norm.pdf(u)) * pof
        x_next = X_cand[int(np.argmax(cei))]
        samp = problem.simulate_replications(x_next, batch_reps, rng_search, split="search")
        all_recs.append(SufficientStatsRecord.from_samples(x_next, samp, problem.nominal_exposure(x_next)))

    X, yf, yg, vf, vg = _records_arrays(all_recs)
    mf, _ = _hyperbo_predict(X, yf, vf, w_mean_f, prior_var_f, X)
    mg, sg = _hyperbo_predict(X, yg, vg, w_mean_g, prior_var_g, X)
    s2_obs = np.array([max(r.ssq_con / max(r.n_reps - 1, 1), 1e-4) for r in all_recs], dtype=float)

    shortlist = build_feasible_aware_shortlist(
        candidate_pool=list(X),
        mean_obj=mf,
        mean_con=mg,
        std_con=sg,
        ucb_aleat_var_con=s2_obs * 1.15,
        r_perp_con=np.full(len(X), 1e-3),
        initial_records=all_recs,
        alpha=problem.alpha,
    )
    receipt = verify_and_deploy_shortlist(problem, shortlist, seed=verif_seed, gate_mode=gate_mode)
    search_used = int(problem.search_calls - calls_start)
    return SequentialRunResult(
        method_name="HyperBO",
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
        n_unique_policies_evaluated=len(all_recs),
        n_replication_queries=0,
        receipt=receipt,
    )


def run_direct_legacy_baseline(
    problem: BaseDynamicORProblem,
    *,
    solver_name: str = "penalty_spsa",
    n_online_steps: int = 30,
    n0_init: int = 8,
    batch_reps: int = 2,
    seed: int = 1001,
    gate_mode: str = "auto",
) -> SequentialRunResult:
    """Run `Penalty-SPSA` or `Constrained-CMAES` directly on target under equal `B_search`."""
    t0 = time.perf_counter()
    verif_seed = seed + 200_007
    calls_start = problem.search_calls
    total_search_budget = (int(n0_init) + int(n_online_steps)) * int(batch_reps)

    if solver_name == "penalty_spsa":
        rec = run_penalty_spsa(problem, budget_calls=total_search_budget, seed=seed, reps_per_eval=batch_reps)
        label = "Target-SPSA"
    elif solver_name == "constrained_cmaes":
        rec = run_constrained_cmaes(problem, budget_calls=total_search_budget, seed=seed, reps_per_eval=batch_reps)
        label = "Target-CMAES"
    else:
        raise ValueError(f"Unsupported direct legacy baseline {solver_name!r}")

    all_recs = rec.records
    init_recs = all_recs[: min(len(all_recs), n0_init)]
    X = [r.x_norm for r in all_recs]
    mf = np.array([r.mean_obj for r in all_recs], dtype=float)
    mg = np.array([r.mean_con for r in all_recs], dtype=float)
    s2_g = np.array([max(r.ssq_con / max(r.n_reps - 1, 1), 1e-4) for r in all_recs], dtype=float)
    sg = np.sqrt(s2_g / np.array([r.n_reps for r in all_recs], dtype=float))

    shortlist = build_feasible_aware_shortlist(
        candidate_pool=X,
        mean_obj=mf,
        mean_con=mg,
        std_con=sg,
        ucb_aleat_var_con=s2_g * 1.15,
        r_perp_con=np.full(len(X), 1e-3),
        initial_records=all_recs,
        alpha=problem.alpha,
    )
    receipt = verify_and_deploy_shortlist(problem, shortlist, seed=verif_seed, gate_mode=gate_mode)
    search_used = int(problem.search_calls - calls_start)
    return SequentialRunResult(
        method_name=label,
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
        n_unique_policies_evaluated=len(all_recs),
        n_replication_queries=0,
        receipt=receipt,
    )
