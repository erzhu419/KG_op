"""Faithful External SOTA & Simple Strong OR Baselines Under Shared Contracts.

Implements accurately named external baselines and simple strong OR controls under
identical target simulation call budgets and identical terminal certification:
1. `SCBO`: Trust-region Constrained BO with Sobol candidates, MLE Matérn-5/2 GPs,
   and FULL joint-covariance Cholesky posterior Thompson Sampling.
2. `MAP-SAAS-CBO`: Eriksson & Jankowiak (2021) hierarchical Half-Cauchy sparse ARD GP
   fitted via L-BFGS-B with exact analytical log-posterior gradients + gate-aware cEI.
3. `RGPE`: Feurer et al. (2018) / BoTorch Ranking-Weighted GP Ensemble using channel-
   preserving cross-horizon warping, exact analytic Rasmussen-Williams GP LOOCV target
   ranking loss, and joint posterior ensemble Thompson Sampling (with offline source GP
   fitting metered separately from online target wall-clock time).
4. `HyperBO-EB`: Wang et al. (JMLR 2024) Empirical-Bayes pre-trained GP maximizing the
   sum of exact GP marginal log-likelihoods across separate source-task datasets via
   L-BFGS-B (with offline pre-training metered separately from online target time).
5. `Custom-HetGP-VKG`: Two-stage raw-coordinate heteroscedastic GP paired with one-step
   lookahead (explicitly named as a custom raw-space heteroscedastic control).
6. `WarmStart-PatternSearch` & `WarmStart-SPSA` (`Arm G`): Simple strong OR heuristics
   sharing the exact same source warm-start candidates and channel-aware DCT low-frequency
   directions, using ZERO GP surrogate.
7. `Target-SPSA` & `Target-SepCMA`: Direct target derivative-free solvers.
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr
from scipy.stats import norm, qmc

from v2_or.acquisition.verification_kg import (
    SequentialRunResult,
    build_cold_dct_initial_points,
    score_gate_power_vkg,
)
from v2_or.contracts import (
    BaseDynamicORProblem,
    LegacyTrajectoryRecord,
    SufficientStatsRecord,
    warp_multichannel_policy,
)
from v2_or.legacy_solvers.solvers import (
    _deb_feasible_key,
    run_constrained_cmaes,
    run_penalty_spsa,
)
from v2_or.models.tempered_hvd_posterior import JointPosteriorPrediction, _safe_cholesky
from v2_or.priors.debiased_spectral_prior import (
    DebiasedSpectralMetaPrior,
    dct_profile_to_policy,
    policy_to_dct_profile,
    sample_smooth_dct_delta,
)
from v2_or.verification.gated_verifier_v2 import (
    build_feasible_aware_shortlist,
    exact_binomial_critical_k,
    exact_binomial_pass_power,
    posterior_binomial_gate_power,
    verify_and_deploy_shortlist,
)


def _warp_policy_dim(x_src: np.ndarray, target_d: int, num_channels: int = 1) -> np.ndarray:
    """Channel-preserving linear policy interpolation from `d_s` to `target_d`."""
    return warp_multichannel_policy(x_src, target_d, num_channels=num_channels)


def _ard_matern52(X1: np.ndarray, X2: np.ndarray, inv_ls: np.ndarray, outscale: float) -> np.ndarray:
    """ARD Matern-5/2 kernel normalized by dimension `d`."""
    w = np.asarray(inv_ls, dtype=float).reshape(1, 1, -1)
    diff = (X1[:, None, :] - X2[None, :, :]) * w
    dist2 = np.mean(diff * diff, axis=2)
    r = np.sqrt(np.maximum(5.0 * dist2, 0.0))
    return float(outscale) * (1.0 + r + (r * r) / 3.0) * np.exp(-r)


class RawInputExactGP:
    """Exact Matérn-5/2 GP supporting isotropic MLE or analytical-gradient MAP-SAAS."""

    def __init__(self, *, map_saas: bool = False, saas_alpha: float = 0.1) -> None:
        self.map_saas = bool(map_saas)
        self.saas_alpha = float(saas_alpha)
        self.X: np.ndarray = np.empty((0, 0))
        self.y_norm: np.ndarray = np.empty(0)
        self.nv_norm: np.ndarray = np.empty(0)
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
        self.y_norm = (y_arr - self.y_mean) / self.y_std
        self.nv_norm = np.maximum(np.asarray(noise_var, dtype=float) / (self.y_std ** 2), 1e-5)

        if self.map_saas and n >= 5:
            self._fit_map_saas()
        else:
            self._fit_isotropic_lbfgs()

        K_fin = _ard_matern52(self.X, self.X, self.inv_ls, self.outscale) + np.diag(self.nv_norm)
        self.L = _safe_cholesky(K_fin)
        self.alpha_vec = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y_norm))
        return self

    def _fit_isotropic_lbfgs(self) -> None:
        n, d = self.X.shape
        diff0 = self.X[:, None, :] - self.X[None, :, :]
        base_dist = np.sqrt(np.maximum(5.0 * np.mean(diff0 * diff0, axis=2), 0.0))

        def _nll_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
            log_inv_ls, log_os = float(theta[0]), float(theta[1])
            inv_ls = np.exp(log_inv_ls)
            os_val = np.exp(log_os)
            r = base_dist * inv_ls
            exp_mr = np.exp(-r)
            K_base = (1.0 + r + (r * r) / 3.0) * exp_mr
            K = os_val * K_base + np.diag(self.nv_norm)
            L = _safe_cholesky(K)
            K_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(n)))
            alp = K_inv @ self.y_norm
            W = K_inv - np.outer(alp, alp)  # d(NLL)/dK = 0.5 * W

            # dK/d(r) = -os * r * (1 + r) / 3 * exp(-r), dr/d(log_inv_ls) = r
            dK_dlog_ls = -os_val * (r * r) * (1.0 + r) / 3.0 * exp_mr
            dK_dlog_os = os_val * K_base
            nll = 0.5 * float(self.y_norm @ alp) + float(np.sum(np.log(np.diag(L))))
            # Mild log-normal hyperprior preventing boundary degeneracy
            nll += 0.05 * (log_inv_ls ** 2 + log_os ** 2)
            g_ls = 0.5 * float(np.sum(W * dK_dlog_ls)) + 0.10 * log_inv_ls
            g_os = 0.5 * float(np.sum(W * dK_dlog_os)) + 0.10 * log_os
            return nll, np.array([g_ls, g_os], dtype=float)

        res = minimize(
            _nll_and_grad,
            x0=np.array([0.2, 0.0], dtype=float),
            method="L-BFGS-B",
            jac=True,
            bounds=[(-2.5, 2.5), (-2.0, 2.0)],
            options={"maxiter": 25, "ftol": 1e-5},
        )
        self.inv_ls = np.full(d, float(np.exp(res.x[0])), dtype=float)
        self.outscale = float(np.exp(res.x[1]))

    def _fit_map_saas(self) -> None:
        """Exact MAP optimization of Eriksson & Jankowiak (2021) Half-Cauchy SAAS model.

        Model on log-transformed positive parameters `(u_j = log rho_j, v = log tau, s = log outscale)`
        where `rho_j = inv_ls_j^2 ~ HalfCauchy(tau)` and `tau ~ HalfCauchy(saas_alpha)`:
            log p(tau) + log |d tau / dv| = -log(saas_alpha^2 + tau^2) + v + const
            log p(rho_j | tau) + log |d rho_j / du_j| = log(tau) - log(tau^2 + rho_j^2) + u_j + const
        """
        n, d = self.X.shape
        sq_diff = (self.X[:, None, :] - self.X[None, :, :]) ** 2 / float(max(d, 1))
        alpha2 = self.saas_alpha ** 2

        def _neg_log_post_and_grad(params: np.ndarray) -> tuple[float, np.ndarray]:
            u = params[:d]              # log(rho_j) where rho_j = inv_ls_j^2
            v = float(params[d])        # log(tau)
            s = float(params[d + 1])    # log(outscale)

            rho = np.exp(u)
            tau = float(np.exp(v))
            tau2 = tau * tau
            os_val = float(np.exp(s))

            dist2 = np.tensordot(sq_diff, rho, axes=([2], [0]))
            r = np.sqrt(np.maximum(5.0 * dist2, 1e-14))
            exp_mr = np.exp(-r)
            K_base = (1.0 + r + (r * r) / 3.0) * exp_mr
            K = os_val * K_base + np.diag(self.nv_norm)
            L = _safe_cholesky(K)
            K_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(n)))
            alp = K_inv @ self.y_norm
            W = K_inv - np.outer(alp, alp)

            nll = 0.5 * float(self.y_norm @ alp) + float(np.sum(np.log(np.diag(L))))
            # Exact negative log Half-Cauchy hierarchical prior (with log-space Jacobians)
            rho2 = rho * rho
            nlp_prior = (
                np.log(alpha2 + tau2) - v
                + float(np.sum(np.log(tau2 + rho2) - v - u))
                + 0.10 * (s * s)
            )

            # dK / d(dist2) = os * dK_base/dr * dr/d(dist2)
            # Since dK_base/dr = -r*(1+r)/3 * exp(-r) and dr/d(dist2) = 5 / (2*r),
            # the `r` cancels cleanly at r -> 0: dK/d(dist2) = -(5/6) * os * (1 + r) * exp(-r)!
            factor = -(5.0 / 6.0) * os_val * (1.0 + r) * exp_mr
            # Grad of NLL wrt rho_j: 0.5 * sum_{i,k} W_{ik} * factor_{ik} * sq_diff_{ikj}
            WF = 0.5 * W * factor
            g_nll_rho = np.tensordot(WF, sq_diff, axes=([0, 1], [0, 1]))
            g_u = g_nll_rho * rho + (2.0 * rho2 / (tau2 + rho2) - 1.0)
            g_v = (
                (2.0 * tau2 / (alpha2 + tau2) - 1.0)
                + float(np.sum(2.0 * tau2 / (tau2 + rho2) - 1.0))
            )
            g_s = 0.5 * float(np.sum(W * (os_val * K_base))) + 0.20 * s
            grad = np.empty(d + 2, dtype=float)
            grad[:d] = g_u
            grad[d] = g_v
            grad[d + 1] = g_s
            return float(nll + nlp_prior), grad

        x0 = np.r_[np.full(d, -1.5), np.log(self.saas_alpha), 0.0]
        bnds = [(-8.0, 4.0)] * d + [(-6.0, 2.0), (-2.5, 2.5)]
        res = minimize(
            _neg_log_post_and_grad,
            x0=x0,
            method="L-BFGS-B",
            jac=True,
            bounds=bnds,
            options={"maxiter": 30, "ftol": 1e-5},
        )
        self.inv_ls = np.exp(0.5 * res.x[:d])
        self.outscale = float(np.exp(res.x[d + 1]))

    def loocv_predict(self) -> tuple[np.ndarray, np.ndarray]:
        """Exact analytic Rasmussen-Williams (Eq. 5.12) Leave-One-Out predictive `(mean, std)`."""
        n = self.X.shape[0]
        K_inv = np.linalg.solve(self.L.T, np.linalg.solve(self.L, np.eye(n)))
        diag_inv = np.maximum(np.diag(K_inv), 1e-9)
        loo_mean_norm = self.y_norm - (self.alpha_vec / diag_inv)
        loo_var_norm = 1.0 / diag_inv
        return (
            self.y_mean + self.y_std * loo_mean_norm,
            self.y_std * np.sqrt(np.maximum(loo_var_norm, 1e-9)),
        )

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

    def sample_posterior(self, X_cand: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Draw a genuine joint posterior function realization over `X_cand`."""
        mean, cov = self.predict(X_cand, return_cov=True)
        L_post = _safe_cholesky(cov)
        return mean + L_post @ rng.normal(size=mean.size)


def _shared_cold_init(
    problem: BaseDynamicORProblem,
    n0_init: int,
    batch_reps: int,
    rng_init: np.random.Generator,
) -> list[SufficientStatsRecord]:
    pts = build_cold_dct_initial_points(problem, n0_init, rng_init)
    recs = []
    for x in pts:
        exp = problem.nominal_exposure(x)
        samp = problem.simulate_replications(x, batch_reps, rng_init, split="search")
        recs.append(SufficientStatsRecord.from_samples(x, samp, exp))
    return recs


def _records_arrays(
    recs: Sequence[SufficientStatsRecord],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    """Faithful SCBO baseline with Sobol trust region and JOINT GP Thompson Sampling."""
    t0 = time.perf_counter()
    rng_init = np.random.default_rng(seed)
    rng_search = np.random.default_rng(seed + 100_003)
    verif_seed = seed + 200_007
    calls_start = problem.search_calls
    z_a = float(norm.ppf(1.0 - problem.alpha))

    init_recs = _shared_cold_init(problem, n0_init, batch_reps, rng_init)
    all_recs = list(init_recs)
    sq_est_obj = float(init_recs[0].mean_obj)

    tr_len = 0.40
    succ_cnt, fail_cnt = 0, 0

    def _rec_score(r: SufficientStatsRecord) -> tuple[int, float, float]:
        sd = float(np.sqrt(max(r.ssq_con / max(r.n_reps - 1, 1), 1e-6)))
        m = r.mean_con + z_a * sd
        return (0, r.mean_obj, m) if m <= 0.0 else (1, m, r.mean_obj)

    best_center = min(all_recs, key=_rec_score)
    best_score = _rec_score(best_center)

    gp_f = RawInputExactGP(map_saas=False)
    gp_g = RawInputExactGP(map_saas=False)
    gp_logv = RawInputExactGP(map_saas=False)

    for step_i in range(int(n_online_steps)):
        X, yf, yg, vf, vg = _records_arrays(all_recs)
        s2_raw_g = np.array([max(r.ssq_con / max(r.n_reps - 1, 1), 1e-5) for r in all_recs], dtype=float)
        gp_f.fit(X, yf, vf)
        gp_g.fit(X, yg, vg)
        gp_logv.fit(X, np.log(s2_raw_g), np.full(len(all_recs), 0.5))

        lo_tr = np.maximum(best_center.x_norm - 0.5 * tr_len, 0.0)
        hi_tr = np.minimum(best_center.x_norm + 0.5 * tr_len, 1.0)
        sobol = qmc.Sobol(d=problem.d, scramble=True, seed=int(seed + 31 * step_i))
        m_pow2 = int(np.ceil(np.log2(max(int(pool_size), 2))))
        u_sob = sobol.random_base2(m_pow2)[: int(pool_size)]
        X_tr = lo_tr[None, :] + u_sob * (hi_tr - lo_tr)[None, :]
        prob_pert = min(20.0 / problem.d, 1.0)
        mask = rng_search.uniform(size=(pool_size, problem.d)) < prob_pert
        no_pert = ~np.any(mask, axis=1)
        if np.any(no_pert):
            rand_cols = rng_search.integers(0, problem.d, size=int(np.sum(no_pert)))
            mask[no_pert, rand_cols] = True
        X_cand = np.where(mask, X_tr, best_center.x_norm[None, :])

        # Joint-posterior GP function draws (fixing independent marginal draw approximation)
        f_ts = gp_f.sample_posterior(X_cand, rng_search)
        g_lat_ts = gp_g.sample_posterior(X_cand, rng_search)
        mlogv, _ = gp_logv.predict(X_cand)
        v_pred = np.exp(np.clip(mlogv, -12.0, 4.0))
        g_ts = g_lat_ts + z_a * np.sqrt(v_pred)

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
        r_perp_con=np.full(len(X), 1e-4),
        initial_records=all_recs,
        alpha=problem.alpha,
        reference_obj=sq_est_obj,
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
    """Analytical-gradient Half-Cauchy Hierarchical Sparse ARD MAP BO (`MAP-SAAS-CBO`)."""
    t0 = time.perf_counter()
    rng_init = np.random.default_rng(seed)
    rng_search = np.random.default_rng(seed + 100_003)
    verif_seed = seed + 200_007
    calls_start = problem.search_calls

    init_recs = _shared_cold_init(problem, n0_init, batch_reps, rng_init)
    all_recs = list(init_recs)
    sq_est_obj = float(init_recs[0].mean_obj)
    gp_f = RawInputExactGP(map_saas=True, saas_alpha=0.1)
    gp_g = RawInputExactGP(map_saas=True, saas_alpha=0.1)
    gp_logv = RawInputExactGP(map_saas=False)

    for _ in range(int(n_online_steps)):
        X, yf, yg, vf, vg = _records_arrays(all_recs)
        s2_raw_g = np.array([max(r.ssq_con / max(r.n_reps - 1, 1), 1e-5) for r in all_recs], dtype=float)
        gp_f.fit(X, yf, vf)
        gp_g.fit(X, yg, vg)
        gp_logv.fit(X, np.log(s2_raw_g), np.full(len(all_recs), 0.5))

        top_dims = np.argsort(-(gp_f.inv_ls + gp_g.inv_ls))[: max(min(16, problem.d), 4)]
        X_cand = np.empty((pool_size, problem.d), dtype=float)
        for i in range(pool_size):
            if i < pool_size * 3 // 4:
                parent = all_recs[i % len(all_recs)].x_norm.copy()
                parent[top_dims] = np.clip(
                    parent[top_dims] + rng_search.normal(0.0, 0.14, size=top_dims.size), 0.0, 1.0
                )
                X_cand[i] = parent
            else:
                X_cand[i] = rng_search.uniform(0.0, 1.0, size=problem.d)

        mf, sf = gp_f.predict(X_cand)
        mg, sg = gp_g.predict(X_cand)
        mlogv, _ = gp_logv.predict(X_cand)
        av_pred = np.exp(np.clip(mlogv, -12.0, 4.0))
        p_gate = posterior_binomial_gate_power(mg, sg, av_pred, n_samples=112, alpha=problem.alpha, delta_slot=0.025)

        feas_obs = [r.mean_obj for r in all_recs if r.violation_count == 0 and r.mean_con <= 0.0]
        f_best = min(feas_obs) if feas_obs else float(np.median(yf))
        u = (f_best - mf) / np.maximum(sf, 1e-6)
        cei = np.maximum(sf, 1e-6) * (u * ndtr(u) + norm.pdf(u)) * p_gate
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
        r_perp_con=np.full(len(X), 1e-4),
        initial_records=all_recs,
        alpha=problem.alpha,
        reference_obj=sq_est_obj,
    )
    receipt = verify_and_deploy_shortlist(problem, shortlist, seed=verif_seed, gate_mode=gate_mode)
    search_used = int(problem.search_calls - calls_start)
    return SequentialRunResult(
        method_name="MAP-SAAS-CBO",
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
    """Two-stage raw-input heteroscedastic GP + Gate-Power Lookahead (`Custom-HetGP-VKG`)."""
    t0 = time.perf_counter()
    rng_init = np.random.default_rng(seed)
    rng_search = np.random.default_rng(seed + 100_003)
    verif_seed = seed + 200_007
    calls_start = problem.search_calls

    init_recs = _shared_cold_init(problem, n0_init, batch_reps, rng_init)
    all_recs = list(init_recs)
    sq_est_obj = float(init_recs[0].mean_obj)
    gp_f = RawInputExactGP(map_saas=False)
    gp_g = RawInputExactGP(map_saas=False)
    gp_logvf = RawInputExactGP(map_saas=False)
    gp_logvg = RawInputExactGP(map_saas=False)

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
            r_perp_con=np.full(len(X_cand), 1e-4),
        )
        scores = score_gate_power_vkg(pred, alpha=problem.alpha, batch_reps=batch_reps, status_quo_obj=sq_est_obj)
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
        r_perp_con=np.full(len(X), 1e-4),
        initial_records=all_recs,
        alpha=problem.alpha,
        reference_obj=sq_est_obj,
    )
    receipt = verify_and_deploy_shortlist(problem, shortlist, seed=verif_seed, gate_mode=gate_mode)
    search_used = int(problem.search_calls - calls_start)
    return SequentialRunResult(
        method_name="Custom-HetGP-VKG",
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


def _archive_row_noise_vars(rows: Sequence[LegacyTrajectoryRecord]) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-evaluation noise variances, imputing `n_reps == 1` rows cleanly from multi-rep dispersion."""
    multi_f = [r.ssq_obj / float(r.n_reps - 1) for r in rows if r.n_reps > 1 and r.ssq_obj > 0.0]
    multi_g = [r.ssq_con / float(r.n_reps - 1) for r in rows if r.n_reps > 1 and r.ssq_con > 0.0]
    yfs = np.array([r.mean_obj for r in rows], dtype=float)
    ygs = np.array([r.mean_con for r in rows], dtype=float)
    imp_f = float(np.median(multi_f)) if multi_f else max(0.05 * float(np.var(yfs)), 1e-4)
    imp_g = float(np.median(multi_g)) if multi_g else max(0.05 * float(np.var(ygs)), 1e-4)
    vfs = np.array(
        [
            (max(r.ssq_obj / float(r.n_reps - 1), 1e-5) if r.n_reps > 1 else imp_f) / float(max(r.n_reps, 1))
            for r in rows
        ],
        dtype=float,
    )
    vgs = np.array(
        [
            (max(r.ssq_con / float(r.n_reps - 1), 1e-5) if r.n_reps > 1 else imp_g) / float(max(r.n_reps, 1))
            for r in rows
        ],
        dtype=float,
    )
    return vfs, vgs


def pretrain_rgpe_source_ensemble(
    problem: BaseDynamicORProblem,
    source_archive: Sequence[LegacyTrajectoryRecord],
) -> tuple[list[RawInputExactGP], list[RawInputExactGP], list[np.ndarray], float]:
    """Offline source GP ensemble fitting for `RGPE` (metered separately from online search!)."""
    t_pre = time.perf_counter()
    src_tasks = sorted({r.task_id for r in source_archive})
    src_models_f: list[RawInputExactGP] = []
    src_models_g: list[RawInputExactGP] = []
    src_elites: list[np.ndarray] = []

    for tid in src_tasks:
        rows = [r for r in source_archive if r.task_id == tid]
        if len(rows) < 5:
            continue
        Xs = np.array([problem.warp_source_policy(np.asarray(r.x_norm)) for r in rows], dtype=float)
        yfs = np.array([r.mean_obj for r in rows], dtype=float)
        ygs = np.array([r.mean_con for r in rows], dtype=float)
        vfs, vgs = _archive_row_noise_vars(rows)
        src_models_f.append(RawInputExactGP().fit(Xs, yfs, vfs))
        src_models_g.append(RawInputExactGP().fit(Xs, ygs, vgs))
        best_r = min(
            rows,
            key=lambda r: (
                0 if (r.mean_con <= 0.0 and r.violation_count == 0) else 1,
                r.mean_obj if (r.mean_con <= 0.0 and r.violation_count == 0) else r.mean_con,
            ),
        )
        src_elites.append(problem.warp_source_policy(np.asarray(best_r.x_norm)))

    return src_models_f, src_models_g, src_elites, float(time.perf_counter() - t_pre)


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
    pretrained_bundle: tuple[list[RawInputExactGP], list[RawInputExactGP], list[np.ndarray], float] | None = None,
) -> SequentialRunResult:
    """Faithful RGPE (Feurer et al. 2018 / BoTorch) with channel-aware warping, exact analytic LOOCV, & joint TS."""
    if pretrained_bundle is None:
        src_models_f, src_models_g, src_elites, _ = pretrain_rgpe_source_ensemble(problem, source_archive)
    else:
        src_models_f, src_models_g, src_elites, _ = pretrained_bundle

    # Online wall-clock clock starts AFTER offline source GP fitting (matching TLS-OLH-KG!)
    t0 = time.perf_counter()
    rng_init = np.random.default_rng(seed)
    rng_search = np.random.default_rng(seed + 100_003)
    verif_seed = seed + 200_007
    calls_start = problem.search_calls

    init_recs = _shared_cold_init(problem, n0_init, batch_reps, rng_init)
    all_recs = list(init_recs)
    sq_est_obj = float(init_recs[0].mean_obj)
    tgt_f = RawInputExactGP()
    tgt_g = RawInputExactGP()

    def _ranking_weights(
        models: list[RawInputExactGP],
        X_t: np.ndarray,
        y_t: np.ndarray,
        tgt_gp: RawInputExactGP,
    ) -> np.ndarray:
        if not models or len(y_t) < 3:
            return np.r_[np.zeros(len(models)), 1.0]
        # Posterior sampling ranking loss on source GPs + EXACT analytic LOOCV posterior on target GP!
        src_means_sds = [m.predict(X_t) for m in models]
        tgt_loo_mean, tgt_loo_sd = tgt_gp.loocv_predict()
        all_means = np.asarray([ms[0] for ms in src_means_sds] + [tgt_loo_mean], dtype=float)
        all_sds = np.asarray([ms[1] for ms in src_means_sds] + [tgt_loo_sd], dtype=float)

        n_mc = 64
        wins = np.zeros(len(models) + 1, dtype=float)
        n_obs = len(y_t)
        for _ in range(n_mc):
            b_idx = rng_search.integers(0, n_obs, size=n_obs)
            yb = y_t[b_idx]
            diff_y = yb[:, None] < yb[None, :]
            draws = all_means[:, b_idx] + all_sds[:, b_idx] * rng_search.normal(size=(len(models) + 1, n_obs))
            losses = np.empty(len(models) + 1, dtype=float)
            for m_i in range(len(models) + 1):
                diff_p = draws[m_i, :, None] < draws[m_i, None, :]
                losses[m_i] = float(np.mean(diff_y ^ diff_p))
            best_loss = float(np.min(losses))
            tied = np.where(np.abs(losses - best_loss) <= 1e-12)[0]
            wins[tied] += 1.0 / float(tied.size)
        w = wins / float(n_mc)
        # Drop diluted source models per Feurer et al. (2018) Section 3.3
        tgt_w = w[-1]
        keep = w >= 0.5 * max(tgt_w, 0.05)
        keep[-1] = True
        w = np.where(keep, w, 0.0)
        return w / max(float(np.sum(w)), 1e-8)

    k_crit = max(exact_binomial_critical_k(112, problem.alpha, 0.025), 0)

    for _ in range(int(n_online_steps)):
        X, yf, yg, vf, vg = _records_arrays(all_recs)
        tgt_f.fit(X, yf, vf)
        tgt_g.fit(X, yg, vg)
        wf = _ranking_weights(src_models_f, X, yf, tgt_f)
        wg = _ranking_weights(src_models_g, X, yg, tgt_g)

        cands_list = [np.clip(e + rng_search.normal(0.0, 0.08, size=problem.d), 0.0, 1.0) for e in src_elites]
        while len(cands_list) < pool_size:
            if len(cands_list) % 2 == 0:
                p = all_recs[len(cands_list) % len(all_recs)].x_norm
                cands_list.append(np.clip(p + rng_search.normal(0.0, 0.10, size=problem.d), 0.0, 1.0))
            else:
                cands_list.append(rng_search.uniform(0.0, 1.0, size=problem.d))
        X_cand = np.asarray(cands_list[:pool_size], dtype=float)

        # Joint posterior sample path from active ensemble components
        f_ens = np.zeros(len(X_cand), dtype=float)
        for w_i, mod_f in zip(wf, src_models_f + [tgt_f]):
            if w_i > 1e-6:
                f_ens += float(w_i) * mod_f.sample_posterior(X_cand, rng_search)
        g_ens = np.zeros(len(X_cand), dtype=float)
        for w_i, mod_g in zip(wg, src_models_g + [tgt_g]):
            if w_i > 1e-6:
                g_ens += float(w_i) * mod_g.sample_posterior(X_cand, rng_search)

        avg_vg = float(np.median([max(r.ssq_con / max(r.n_reps - 1, 1), 1e-4) for r in all_recs]))
        p_draw = ndtr(g_ens / np.sqrt(avg_vg))
        c_pow = exact_binomial_pass_power(p_draw, 112, k_crit)
        cert_idx = np.where(c_pow >= 0.50)[0]
        if cert_idx.size > 0:
            util = np.maximum(sq_est_obj - f_ens[cert_idx], 1e-4) * c_pow[cert_idx]
            chosen = int(cert_idx[int(np.argmax(util))])
        else:
            chosen = int(np.argmax(c_pow - 0.02 * f_ens))

        x_next = X_cand[chosen]
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
        r_perp_con=np.full(len(X), 1e-4),
        initial_records=all_recs,
        alpha=problem.alpha,
        reference_obj=sq_est_obj,
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


def pretrain_hyperbo_eb_params(
    problem: BaseDynamicORProblem,
    source_archive: Sequence[LegacyTrajectoryRecord],
) -> tuple[dict[str, tuple[np.ndarray, float, float]], float]:
    """Multi-task Empirical-Bayes GP pre-training across separate source tasks (Wang et al. 2024)."""
    t_pre = time.perf_counter()
    src_tasks = sorted({r.task_id for r in source_archive})
    datasets: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for tid in src_tasks:
        rows = [r for r in source_archive if r.task_id == tid]
        if len(rows) < 5:
            continue
        Xs = np.array([problem.warp_source_policy(np.asarray(r.x_norm)) for r in rows], dtype=float)
        yfs = np.array([r.mean_obj for r in rows], dtype=float)
        ygs = np.array([r.mean_con for r in rows], dtype=float)
        vfs, vgs = _archive_row_noise_vars(rows)
        datasets.append((Xs, yfs, vfs, ygs, vgs))

    n_ch = int(problem.num_control_channels)

    def _channel_summary_features(X: np.ndarray) -> np.ndarray:
        # Channel-level affine mean features `[1, mean_ch0, mean_ch1/std]` shared across tasks
        if n_ch == 1 or X.shape[1] == 1:
            return np.column_stack([np.ones(len(X)), np.mean(X, axis=1), np.std(X, axis=1)])
        return np.column_stack([np.ones(len(X)), np.mean(X[:, 0::2], axis=1), np.mean(X[:, 1::2], axis=1)])

    def _fit_shared_head(is_obj: bool) -> tuple[np.ndarray, float, float]:
        # Step 1: GLS shared mean across tasks; Step 2: L-BFGS-B exact multi-task GP MLL over (inv_ls, outscale)
        H_all = np.vstack([_channel_summary_features(ds[0]) for ds in datasets])
        y_all = np.concatenate([ds[1] if is_obj else ds[3] for ds in datasets])
        w_m = np.linalg.solve(H_all.T @ H_all + 1e-2 * np.eye(H_all.shape[1]), H_all.T @ y_all)

        task_dists = []
        task_resids = []
        task_nvars = []
        for Xs, yfs, vfs, ygs, vgs in datasets:
            yv = yfs if is_obj else ygs
            nv = vfs if is_obj else vgs
            diff = Xs[:, None, :] - Xs[None, :, :]
            task_dists.append(np.sqrt(np.maximum(5.0 * np.mean(diff * diff, axis=2), 0.0)))
            task_resids.append(yv - (_channel_summary_features(Xs) @ w_m))
            task_nvars.append(nv)

        def _mt_nll_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
            inv_ls = float(np.exp(theta[0]))
            os_val = float(np.exp(theta[1]))
            tot_nll = 0.05 * float(theta[0] ** 2 + theta[1] ** 2)
            g_ls = 0.10 * float(theta[0])
            g_os = 0.10 * float(theta[1])
            for bd, res_s, nv_s in zip(task_dists, task_resids, task_nvars):
                n_s = res_s.size
                r = bd * inv_ls
                exp_mr = np.exp(-r)
                Kb = (1.0 + r + (r * r) / 3.0) * exp_mr
                K = os_val * Kb + np.diag(nv_s)
                L = _safe_cholesky(K)
                K_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(n_s)))
                alp = K_inv @ res_s
                W = K_inv - np.outer(alp, alp)
                tot_nll += 0.5 * float(res_s @ alp) + float(np.sum(np.log(np.diag(L))))
                dK_ls = -os_val * (r * r) * (1.0 + r) / 3.0 * exp_mr
                g_ls += 0.5 * float(np.sum(W * dK_ls))
                g_os += 0.5 * float(np.sum(W * (os_val * Kb)))
            return tot_nll, np.array([g_ls, g_os], dtype=float)

        init_os = np.log(max(float(np.var(np.concatenate(task_resids))), 1e-3))
        opt = minimize(
            _mt_nll_and_grad,
            x0=np.array([0.3, np.clip(init_os, -4.0, 2.0)], dtype=float),
            method="L-BFGS-B",
            jac=True,
            bounds=[(-2.5, 2.5), (-5.0, 3.0)],
            options={"maxiter": 30, "ftol": 1e-5},
        )
        return w_m, float(np.exp(opt.x[0])), float(np.exp(opt.x[1]))

    params = {
        "obj": _fit_shared_head(True),
        "con": _fit_shared_head(False),
    }
    return params, float(time.perf_counter() - t_pre)


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
    pretrained_params: tuple[dict[str, tuple[np.ndarray, float, float]], float] | None = None,
) -> SequentialRunResult:
    """Empirical-Bayes Multi-Task Pre-Trained GP (`HyperBO-EB`, Wang et al. JMLR 2024)."""
    if pretrained_params is None:
        eb_params, _ = pretrain_hyperbo_eb_params(problem, source_archive)
    else:
        eb_params, _ = pretrained_params

    t0 = time.perf_counter()
    rng_init = np.random.default_rng(seed)
    rng_search = np.random.default_rng(seed + 100_003)
    verif_seed = seed + 200_007
    calls_start = problem.search_calls
    n_ch = int(problem.num_control_channels)

    def _chan_feats(X: np.ndarray) -> np.ndarray:
        if n_ch == 1 or X.shape[1] == 1:
            return np.column_stack([np.ones(len(X)), np.mean(X, axis=1), np.std(X, axis=1)])
        return np.column_stack([np.ones(len(X)), np.mean(X[:, 0::2], axis=1), np.mean(X[:, 1::2], axis=1)])

    w_f, inv_ls_f, os_f = eb_params["obj"]
    w_g, inv_ls_g, os_g = eb_params["con"]
    ls_vec_f = np.full(problem.d, inv_ls_f, dtype=float)
    ls_vec_g = np.full(problem.d, inv_ls_g, dtype=float)

    init_recs = _shared_cold_init(problem, n0_init, batch_reps, rng_init)
    all_recs = list(init_recs)
    sq_est_obj = float(init_recs[0].mean_obj)

    def _eb_predict(
        X_obs: np.ndarray,
        y_obs: np.ndarray,
        v_obs: np.ndarray,
        w_m: np.ndarray,
        ls_vec: np.ndarray,
        p_var: float,
        X_c: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        m0_obs = _chan_feats(X_obs) @ w_m
        m0_c = _chan_feats(X_c) @ w_m
        K_oo = _ard_matern52(X_obs, X_obs, ls_vec, p_var) + np.diag(v_obs)
        K_co = _ard_matern52(X_c, X_obs, ls_vec, p_var)
        L = _safe_cholesky(K_oo)
        alp = np.linalg.solve(L.T, np.linalg.solve(L, y_obs - m0_obs))
        V = np.linalg.solve(L, K_co.T)
        mean = m0_c + K_co @ alp
        std = np.sqrt(np.maximum(p_var - np.sum(V * V, axis=0), 1e-8))
        return mean, std

    for _ in range(int(n_online_steps)):
        X, yf, yg, vf, vg = _records_arrays(all_recs)
        X_cand = np.vstack([
            np.clip(
                X[rng_search.integers(0, len(X), size=pool_size // 2)]
                + rng_search.normal(0.0, 0.10, size=(pool_size // 2, problem.d)),
                0.0,
                1.0,
            ),
            rng_search.uniform(0.0, 1.0, size=(pool_size - pool_size // 2, problem.d)),
        ])
        mf, sf = _eb_predict(X, yf, vf, w_f, ls_vec_f, os_f, X_cand)
        mg, sg = _eb_predict(X, yg, vg, w_g, ls_vec_g, os_g, X_cand)
        avg_vg = float(np.median([max(r.ssq_con / max(r.n_reps - 1, 1), 1e-4) for r in all_recs]))
        p_gate = posterior_binomial_gate_power(
            mg, sg, np.full_like(mg, avg_vg), n_samples=112, alpha=problem.alpha, delta_slot=0.025
        )
        f_best = float(np.min(yf[yg <= 0.0])) if np.any(yg <= 0.0) else float(np.median(yf))
        u = (f_best - mf) / sf
        cei = sf * (u * ndtr(u) + norm.pdf(u)) * p_gate
        x_next = X_cand[int(np.argmax(cei))]
        samp = problem.simulate_replications(x_next, batch_reps, rng_search, split="search")
        all_recs.append(SufficientStatsRecord.from_samples(x_next, samp, problem.nominal_exposure(x_next)))

    X, yf, yg, vf, vg = _records_arrays(all_recs)
    mf, _ = _eb_predict(X, yf, vf, w_f, ls_vec_f, os_f, X)
    mg, sg = _eb_predict(X, yg, vg, w_g, ls_vec_g, os_g, X)
    s2_obs = np.array([max(r.ssq_con / max(r.n_reps - 1, 1), 1e-4) for r in all_recs], dtype=float)

    shortlist = build_feasible_aware_shortlist(
        candidate_pool=list(X),
        mean_obj=mf,
        mean_con=mg,
        std_con=sg,
        ucb_aleat_var_con=s2_obs * 1.15,
        r_perp_con=np.full(len(X), 1e-4),
        initial_records=all_recs,
        alpha=problem.alpha,
        reference_obj=sq_est_obj,
    )
    receipt = verify_and_deploy_shortlist(problem, shortlist, seed=verif_seed, gate_mode=gate_mode)
    search_used = int(problem.search_calls - calls_start)
    return SequentialRunResult(
        method_name="HyperBO-EB",
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


def run_warmstart_pattern_search_baseline(
    problem: BaseDynamicORProblem,
    prior: DebiasedSpectralMetaPrior,
    *,
    n_online_steps: int = 30,
    n0_init: int = 8,
    batch_reps: int = 2,
    seed: int = 1001,
    gate_mode: str = "auto",
) -> SequentialRunResult:
    """Arm G: Simple strong OR control (`WarmStart-PatternSearch`, ZERO GP surrogate!).

    Shares the EXACT same source DCT warm-start initial candidates and channel-aware
    low-frequency DCT adjustment directions as our method, isolating whether the
    functional GP posterior adds decision value beyond good warm-starts + smooth search.
    """
    t0 = time.perf_counter()
    rng_init = np.random.default_rng(seed)
    rng_search = np.random.default_rng(seed + 100_003)
    verif_seed = seed + 200_007
    calls_start = problem.search_calls
    n_ch = int(problem.num_control_channels)

    init_pts = prior.warmstart_candidates(
        problem.d,
        n0_init,
        rng_init,
        anchor=problem.domain_heuristic_anchor(),
        num_channels=n_ch,
    )
    all_recs: list[SufficientStatsRecord] = []
    for x0 in init_pts:
        samp = problem.simulate_replications(x0, batch_reps, rng_init, split="search")
        all_recs.append(SufficientStatsRecord.from_samples(x0, samp, problem.nominal_exposure(x0)))
    sq_est_obj = float(all_recs[0].mean_obj)

    # Rank by gate-calibrated empirical margin (`z_pow = 2.25` for K=112, k*=1)
    def _gate_emp_key(r: SufficientStatsRecord) -> tuple[int, float, float]:
        sd = float(np.sqrt(max(r.ssq_con / max(r.n_reps - 1, 1), 1e-6)))
        m = r.mean_con + 2.25 * sd
        return (0, float(r.mean_obj), m) if (m <= 0.0 and r.violation_count == 0) else (1, m, float(r.mean_obj))

    best_rec = min(all_recs, key=_gate_emp_key)
    step_radius = 0.12

    for step_i in range(int(n_online_steps)):
        delta = sample_smooth_dct_delta(rng_search, problem.d, radius=step_radius, k_modes=12, num_channels=n_ch)
        sign = 1.0 if (step_i % 2 == 0) else -1.0
        x_try = np.clip(best_rec.x_norm + sign * delta, 0.0, 1.0)
        samp = problem.simulate_replications(x_try, batch_reps, rng_search, split="search")
        rec_try = SufficientStatsRecord.from_samples(x_try, samp, problem.nominal_exposure(x_try))
        all_recs.append(rec_try)
        if _gate_emp_key(rec_try) < _gate_emp_key(best_rec):
            best_rec = rec_try
            step_radius = min(step_radius * 1.15, 0.20)
        else:
            step_radius = max(step_radius * 0.90, 0.025)

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
        r_perp_con=np.full(len(X), 1e-4),
        initial_records=all_recs,
        alpha=problem.alpha,
        reference_obj=sq_est_obj,
    )
    receipt = verify_and_deploy_shortlist(problem, shortlist, seed=verif_seed, gate_mode=gate_mode)
    search_used = int(problem.search_calls - calls_start)
    return SequentialRunResult(
        method_name="WarmStart-PatternSearch",
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
    """Run `Penalty-SPSA` or `Sep-Constrained-CMA` directly on target under equal `B_search`."""
    t0 = time.perf_counter()
    verif_seed = seed + 200_007
    calls_start = problem.search_calls
    total_search_budget = (int(n0_init) + int(n_online_steps)) * int(batch_reps)

    if solver_name == "penalty_spsa":
        rec = run_penalty_spsa(problem, budget_calls=total_search_budget, seed=seed, reps_per_eval=batch_reps)
        label = "Target-SPSA"
    elif solver_name in {"constrained_cmaes", "sep_cmaes"}:
        rec = run_constrained_cmaes(problem, budget_calls=total_search_budget, seed=seed, reps_per_eval=batch_reps)
        label = "Target-SepCMA"
    else:
        raise ValueError(f"Unsupported direct legacy baseline {solver_name!r}")

    all_recs = rec.records
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
        r_perp_con=np.full(len(X), 1e-4),
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


def run_scbo_on_psi_baseline(
    problem: BaseDynamicORProblem,
    *,
    n_online_steps: int = 30,
    n0_init: int = 8,
    batch_reps: int = 2,
    seed: int = 1001,
    pool_size: int = 160,
    gate_mode: str = "auto",
) -> SequentialRunResult:
    """Same-representation SCBO control (`SCBO-on-Psi`): SCBO operating on zero-leakage `psi(x) in R^12`.

    Isolates whether our debiased spectral meta-prior + shift-invariant HVD + Gate-Power KG
    outperform a SOTA trust-region constrained BO method even when SCBO is granted the exact
    same dimension-invariant operational exposure representation `psi(x)` and smooth DCT proposals.
    """
    t0 = time.perf_counter()
    rng_init = np.random.default_rng(seed)
    rng_search = np.random.default_rng(seed + 100_003)
    verif_seed = seed + 200_007
    calls_start = problem.search_calls
    z_a = float(norm.ppf(1.0 - problem.alpha))
    n_ch = int(problem.num_control_channels)

    init_recs = _shared_cold_init(problem, n0_init, batch_reps, rng_init)
    all_recs = list(init_recs)
    sq_est_obj = float(init_recs[0].mean_obj)

    tr_len = 0.35
    succ_cnt, fail_cnt = 0, 0

    def _rec_score(r: SufficientStatsRecord) -> tuple[int, float, float]:
        sd = float(np.sqrt(max(r.ssq_con / max(r.n_reps - 1, 1), 1e-6)))
        m = r.mean_con + z_a * sd
        return (0, r.mean_obj, m) if (m <= 0.0 and r.violation_count == 0) else (1, m, r.mean_obj)

    best_center = min(all_recs, key=_rec_score)
    best_score = _rec_score(best_center)

    gp_f = RawInputExactGP(map_saas=True, saas_alpha=0.15)
    gp_g = RawInputExactGP(map_saas=True, saas_alpha=0.15)
    gp_logv = RawInputExactGP(map_saas=False)

    def _std_psi(psi_obs: np.ndarray, psi_c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu = np.mean(psi_obs, axis=0)
        sd = np.maximum(np.std(psi_obs, axis=0, ddof=1), 1e-3)
        return (
            np.clip((psi_obs - mu[None, :]) / sd[None, :], -4.0, 4.0),
            np.clip((psi_c - mu[None, :]) / sd[None, :], -4.0, 4.0),
        )

    for step_i in range(int(n_online_steps)):
        _, yf, yg, vf, vg = _records_arrays(all_recs)
        psi_obs_raw = np.array([r.exposure.psi for r in all_recs], dtype=float)
        s2_raw_g = np.array([max(r.ssq_con / max(r.n_reps - 1, 1), 1e-5) for r in all_recs], dtype=float)

        # Build SCBO trust-region Sobol candidates + smooth DCT local perturbations
        n_sob = pool_size // 2
        lo_tr = np.maximum(best_center.x_norm - 0.5 * tr_len, 0.0)
        hi_tr = np.minimum(best_center.x_norm + 0.5 * tr_len, 1.0)
        sobol = qmc.Sobol(d=problem.d, scramble=True, seed=int(seed + 37 * step_i))
        m_pow2 = int(np.ceil(np.log2(max(int(n_sob), 2))))
        u_sob = sobol.random_base2(m_pow2)[: int(n_sob)]
        X_tr = lo_tr[None, :] + u_sob * (hi_tr - lo_tr)[None, :]
        prob_pert = min(20.0 / problem.d, 1.0)
        mask = rng_search.uniform(size=(n_sob, problem.d)) < prob_pert
        no_pert = ~np.any(mask, axis=1)
        if np.any(no_pert):
            rand_cols = rng_search.integers(0, problem.d, size=int(np.sum(no_pert)))
            mask[no_pert, rand_cols] = True
        X_sob = np.where(mask, X_tr, best_center.x_norm[None, :])

        dct_cands: list[np.ndarray] = []
        anchor = problem.domain_heuristic_anchor()
        while len(dct_cands) < (pool_size - n_sob):
            idx_d = len(dct_cands)
            rad = (0.03, 0.07, 0.15)[idx_d % 3] * (tr_len / 0.35)
            if idx_d % 3 != 2:
                delta = sample_smooth_dct_delta(rng_search, problem.d, radius=rad, k_modes=12, num_channels=n_ch)
                dct_cands.append(np.clip(best_center.x_norm + delta, 0.0, 1.0))
            else:
                delta = sample_smooth_dct_delta(rng_search, problem.d, radius=0.20, k_modes=12, num_channels=n_ch)
                dct_cands.append(np.clip(anchor + delta, 0.0, 1.0))

        X_cand = np.vstack([X_sob, np.asarray(dct_cands, dtype=float)])[:pool_size]
        exps_cand = [problem.nominal_exposure(xc) for xc in X_cand]
        psi_cand_raw = np.array([e.psi for e in exps_cand], dtype=float)
        Psi_obs, Psi_cand = _std_psi(psi_obs_raw, psi_cand_raw)

        gp_f.fit(Psi_obs, yf, vf)
        gp_g.fit(Psi_obs, yg, vg)
        gp_logv.fit(Psi_obs, np.log(s2_raw_g), np.full(len(all_recs), 0.5))

        f_ts = gp_f.sample_posterior(Psi_cand, rng_search)
        g_lat_ts = gp_g.sample_posterior(Psi_cand, rng_search)
        mlogv, _ = gp_logv.predict(Psi_cand)
        v_pred = np.exp(np.clip(mlogv, -12.0, 4.0))
        g_ts = g_lat_ts + z_a * np.sqrt(v_pred)

        feas = np.where(g_ts <= 0.0)[0]
        idx = int(feas[np.argmin(f_ts[feas])]) if feas.size > 0 else int(np.argmin(g_ts))

        x_next = X_cand[idx]
        samp = problem.simulate_replications(x_next, batch_reps, rng_search, split="search")
        rec_new = SufficientStatsRecord.from_samples(x_next, samp, exps_cand[idx])
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
    psi_obs_raw = np.array([r.exposure.psi for r in all_recs], dtype=float)
    Psi_obs, _ = _std_psi(psi_obs_raw, psi_obs_raw)
    s2_raw_g = np.array([max(r.ssq_con / max(r.n_reps - 1, 1), 1e-5) for r in all_recs], dtype=float)
    gp_f.fit(Psi_obs, yf, vf)
    gp_g.fit(Psi_obs, yg, vg)
    gp_logv.fit(Psi_obs, np.log(s2_raw_g), np.full(len(all_recs), 0.5))

    mf, _ = gp_f.predict(Psi_obs)
    mg, sg = gp_g.predict(Psi_obs)
    mlogv, slogv = gp_logv.predict(Psi_obs)
    ucb_vg = np.exp(np.clip(mlogv + 0.5 * slogv, -12.0, 4.0))

    shortlist = build_feasible_aware_shortlist(
        candidate_pool=list(X),
        mean_obj=mf,
        mean_con=mg,
        std_con=sg,
        ucb_aleat_var_con=ucb_vg,
        r_perp_con=np.full(len(X), 1e-4),
        initial_records=all_recs,
        alpha=problem.alpha,
        reference_obj=sq_est_obj,
    )
    receipt = verify_and_deploy_shortlist(problem, shortlist, seed=verif_seed, gate_mode=gate_mode)
    search_used = int(problem.search_calls - calls_start)
    return SequentialRunResult(
        method_name="SCBO-on-Psi",
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


def pretrain_rgpe_psi_source_ensemble(
    source_archive: Sequence[LegacyTrajectoryRecord],
) -> tuple[list[RawInputExactGP], list[RawInputExactGP], np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Offline source GP ensemble fitting directly on standardized `psi in R^12` for `RGPE-on-Psi`."""
    t_pre = time.perf_counter()
    psi_all = np.array([r.psi for r in source_archive], dtype=float)
    psi_mean = np.mean(psi_all, axis=0)
    psi_std = np.maximum(np.std(psi_all, axis=0, ddof=1), 1e-4)

    src_tasks = sorted({r.task_id for r in source_archive})
    src_models_f: list[RawInputExactGP] = []
    src_models_g: list[RawInputExactGP] = []
    elite_dct_2ch: list[np.ndarray] = []
    elite_dct_1ch: list[np.ndarray] = []

    for tid in src_tasks:
        rows = [r for r in source_archive if r.task_id == tid]
        if len(rows) < 5:
            continue
        Psi_s = np.clip((np.array([r.psi for r in rows], dtype=float) - psi_mean[None, :]) / psi_std[None, :], -4.0, 4.0)
        yfs = np.array([r.mean_obj for r in rows], dtype=float)
        ygs = np.array([r.mean_con for r in rows], dtype=float)
        vfs, vgs = _archive_row_noise_vars(rows)
        src_models_f.append(RawInputExactGP().fit(Psi_s, yfs, vfs))
        src_models_g.append(RawInputExactGP().fit(Psi_s, ygs, vgs))

        best_r = min(
            rows,
            key=lambda r: (
                0 if (r.mean_con <= 0.0 and r.violation_count == 0) else 1,
                r.mean_obj if (r.mean_con <= 0.0 and r.violation_count == 0) else r.mean_con,
            ),
        )
        elite_dct_2ch.append(policy_to_dct_profile(np.asarray(best_r.x_norm), 12, num_channels=2))
        elite_dct_1ch.append(policy_to_dct_profile(np.asarray(best_r.x_norm), 12, num_channels=1))

    return (
        src_models_f,
        src_models_g,
        psi_mean,
        psi_std,
        np.asarray(elite_dct_2ch, dtype=float),
        np.asarray(elite_dct_1ch, dtype=float),
        float(time.perf_counter() - t_pre),
    )


def run_rgpe_on_psi_baseline(
    problem: BaseDynamicORProblem,
    source_archive: Sequence[LegacyTrajectoryRecord],
    *,
    n_online_steps: int = 30,
    n0_init: int = 8,
    batch_reps: int = 2,
    seed: int = 1001,
    pool_size: int = 160,
    gate_mode: str = "auto",
    pretrained_bundle: tuple | None = None,
) -> SequentialRunResult:
    """Same-representation RGPE control (`RGPE-on-Psi`): RGPE ensemble directly on `psi(x) in R^12`.

    Grants Feurer et al. (2018) / BoTorch RGPE the exact same zero-leakage `psi(x) in R^12`
    representation, DCT cross-horizon elite warm-starts, and Gate-Power TS acquisition, cleanly
    isolating our debiased spectral subspace + shift-invariant HVD + empirical-Bayes tempering.
    """
    if pretrained_bundle is None:
        src_models_f, src_models_g, psi_mean, psi_std, dct_2ch, dct_1ch, _ = pretrain_rgpe_psi_source_ensemble(source_archive)
    else:
        src_models_f, src_models_g, psi_mean, psi_std, dct_2ch, dct_1ch, _ = pretrained_bundle

    t0 = time.perf_counter()
    rng_init = np.random.default_rng(seed)
    rng_search = np.random.default_rng(seed + 100_003)
    verif_seed = seed + 200_007
    calls_start = problem.search_calls
    n_ch = int(problem.num_control_channels)

    dct_bank = dct_1ch if (n_ch == 1 and dct_1ch.size > 0) else dct_2ch
    bank_ch = 1 if (n_ch == 1 and dct_1ch.size > 0) else 2
    src_elites = [dct_profile_to_policy(prof, problem.d, num_channels=bank_ch) for prof in dct_bank]

    init_recs = _shared_cold_init(problem, n0_init, batch_reps, rng_init)
    all_recs = list(init_recs)
    sq_est_obj = float(init_recs[0].mean_obj)
    tgt_f = RawInputExactGP()
    tgt_g = RawInputExactGP()

    def _norm_psi(exps: Sequence) -> np.ndarray:
        raw = np.array([e.psi for e in exps], dtype=float)
        return np.clip((raw - psi_mean[None, :]) / psi_std[None, :], -4.0, 4.0)

    def _ranking_weights(
        models: list[RawInputExactGP],
        Psi_t: np.ndarray,
        y_t: np.ndarray,
        tgt_gp: RawInputExactGP,
    ) -> np.ndarray:
        if not models or len(y_t) < 3:
            return np.r_[np.zeros(len(models)), 1.0]
        src_means_sds = [m.predict(Psi_t) for m in models]
        tgt_loo_mean, tgt_loo_sd = tgt_gp.loocv_predict()
        all_means = np.asarray([ms[0] for ms in src_means_sds] + [tgt_loo_mean], dtype=float)
        all_sds = np.asarray([ms[1] for ms in src_means_sds] + [tgt_loo_sd], dtype=float)

        n_mc = 64
        wins = np.zeros(len(models) + 1, dtype=float)
        n_obs = len(y_t)
        for _ in range(n_mc):
            b_idx = rng_search.integers(0, n_obs, size=n_obs)
            yb = y_t[b_idx]
            diff_y = yb[:, None] < yb[None, :]
            draws = all_means[:, b_idx] + all_sds[:, b_idx] * rng_search.normal(size=(len(models) + 1, n_obs))
            losses = np.empty(len(models) + 1, dtype=float)
            for m_i in range(len(models) + 1):
                diff_p = draws[m_i, :, None] < draws[m_i, None, :]
                losses[m_i] = float(np.mean(diff_y ^ diff_p))
            best_loss = float(np.min(losses))
            tied = np.where(np.abs(losses - best_loss) <= 1e-12)[0]
            wins[tied] += 1.0 / float(tied.size)
        w = wins / float(n_mc)
        tgt_w = w[-1]
        keep = w >= 0.5 * max(tgt_w, 0.05)
        keep[-1] = True
        w = np.where(keep, w, 0.0)
        return w / max(float(np.sum(w)), 1e-8)

    k_crit = max(exact_binomial_critical_k(112, problem.alpha, 0.025), 0)
    anchor = problem.domain_heuristic_anchor()

    for _ in range(int(n_online_steps)):
        _, yf, yg, vf, vg = _records_arrays(all_recs)
        Psi_obs = _norm_psi([r.exposure for r in all_recs])
        tgt_f.fit(Psi_obs, yf, vf)
        tgt_g.fit(Psi_obs, yg, vg)
        wf = _ranking_weights(src_models_f, Psi_obs, yf, tgt_f)
        wg = _ranking_weights(src_models_g, Psi_obs, yg, tgt_g)

        cands_list: list[np.ndarray] = []
        for e in src_elites[:16]:
            delta = sample_smooth_dct_delta(rng_search, problem.d, radius=0.06, k_modes=12, num_channels=n_ch)
            cands_list.append(np.clip(e + delta, 0.0, 1.0))
        while len(cands_list) < pool_size:
            idx_c = len(cands_list)
            if idx_c % 2 == 0:
                p = all_recs[idx_c % len(all_recs)].x_norm
                rad = (0.04, 0.09, 0.16)[idx_c % 3]
                delta = sample_smooth_dct_delta(rng_search, problem.d, radius=rad, k_modes=12, num_channels=n_ch)
                cands_list.append(np.clip(p + delta, 0.0, 1.0))
            else:
                delta = sample_smooth_dct_delta(rng_search, problem.d, radius=0.22, k_modes=12, num_channels=n_ch)
                cands_list.append(np.clip(anchor + delta, 0.0, 1.0))
        X_cand = np.asarray(cands_list[:pool_size], dtype=float)
        exps_cand = [problem.nominal_exposure(xc) for xc in X_cand]
        Psi_cand = _norm_psi(exps_cand)

        f_ens = np.zeros(len(X_cand), dtype=float)
        for w_i, mod_f in zip(wf, src_models_f + [tgt_f]):
            if w_i > 1e-6:
                f_ens += float(w_i) * mod_f.sample_posterior(Psi_cand, rng_search)
        g_ens = np.zeros(len(X_cand), dtype=float)
        for w_i, mod_g in zip(wg, src_models_g + [tgt_g]):
            if w_i > 1e-6:
                g_ens += float(w_i) * mod_g.sample_posterior(Psi_cand, rng_search)

        avg_vg = float(np.median([max(r.ssq_con / max(r.n_reps - 1, 1), 1e-4) for r in all_recs]))
        p_draw = ndtr(g_ens / np.sqrt(avg_vg))
        c_pow = exact_binomial_pass_power(p_draw, 112, k_crit)
        spread_f = max(float(np.std(f_ens)), 1e-8)
        j_ref = max(sq_est_obj, float(np.min(f_ens)) + 0.20 * spread_f)
        cert_idx = np.where(c_pow >= 0.50)[0]
        if cert_idx.size > 0:
            util = np.maximum(j_ref - f_ens[cert_idx], 1e-4 * spread_f) * c_pow[cert_idx]
            chosen = int(cert_idx[int(np.argmax(util))])
        else:
            chosen = int(np.argmax(c_pow - 0.02 * (f_ens - float(np.min(f_ens))) / spread_f))

        x_next = X_cand[chosen]
        samp = problem.simulate_replications(x_next, batch_reps, rng_search, split="search")
        all_recs.append(SufficientStatsRecord.from_samples(x_next, samp, exps_cand[chosen]))

    X, yf, yg, vf, vg = _records_arrays(all_recs)
    Psi_obs = _norm_psi([r.exposure for r in all_recs])
    tgt_f.fit(Psi_obs, yf, vf)
    tgt_g.fit(Psi_obs, yg, vg)
    wf = _ranking_weights(src_models_f, Psi_obs, yf, tgt_f)
    wg = _ranking_weights(src_models_g, Psi_obs, yg, tgt_g)
    mf_all = [m.predict(Psi_obs) for m in src_models_f] + [tgt_f.predict(Psi_obs)]
    mg_all = [m.predict(Psi_obs) for m in src_models_g] + [tgt_g.predict(Psi_obs)]
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
        r_perp_con=np.full(len(X), 1e-4),
        initial_records=all_recs,
        alpha=problem.alpha,
        reference_obj=sq_est_obj,
    )
    receipt = verify_and_deploy_shortlist(problem, shortlist, seed=verif_seed, gate_mode=gate_mode)
    search_used = int(problem.search_calls - calls_start)
    return SequentialRunResult(
        method_name="RGPE-on-Psi",
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
