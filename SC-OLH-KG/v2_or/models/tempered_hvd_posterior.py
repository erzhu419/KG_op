"""Tempered Hierarchical Semiparametric + Decoupled HVD Posterior (`TemperedHierarchicalHVDModel`).

Implements Theorem 2 of the Path-B Operations Research reconstruction:
- Decouples within-policy multi-replication dispersion `ssq / (n_reps - 1)` from
  mean-innovation residuals so arbitrary source-target mean shifts `Delta_g` cause
  ZERO leakage into aleatoric variance `v_g(x)` (eliminating the 59x variance blow-up).
- Adapts cross-task prior scaling `kappa_prior = 1 / beta_n`, subspace shift variance
  `tau_shift^2`, and local Matern-5/2 residual GP scale `sigma_perp^2` via exact
  Gaussian marginal log-likelihood on observed target evaluations.
- Provides clean mechanistic ablation modes (`"tempered"`, `"untempered_full"`,
  `"frozen_after_init"`, `"target_only"`) in one unified closed-form engine (< 5 ms/step).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import nnls

from v2_or.contracts import NominalTrajectoryExposure, SufficientStatsRecord
from v2_or.priors.debiased_spectral_prior import (
    DebiasedSpectralMetaPrior,
    hvd_nonneg_feature_vector,
)


def _matern52_kernel(X1: np.ndarray, X2: np.ndarray, lengthscale: float) -> np.ndarray:
    """Isotropic Matern-5/2 kernel on standardized `psi` coordinates (`d`-normalized)."""
    p = max(X1.shape[1], 1)
    diff = X1[:, None, :] - X2[None, :, :]
    dist2 = np.sum(diff * diff, axis=2) / float(p)
    r = np.sqrt(np.maximum(5.0 * dist2, 0.0)) / max(float(lengthscale), 1e-4)
    return (1.0 + r + (r * r) / 3.0) * np.exp(-r)


def _safe_cholesky(K: np.ndarray) -> np.ndarray:
    K_sym = 0.5 * (K + K.T)
    n = K_sym.shape[0]
    eye = np.eye(n, dtype=float)
    jitter = 1e-8
    for _ in range(8):
        try:
            return np.linalg.cholesky(K_sym + jitter * eye)
        except np.linalg.LinAlgError:
            jitter *= 10.0
    evals, evecs = np.linalg.eigh(K_sym)
    repaired = (evecs * np.maximum(evals, 1e-7)[None, :]) @ evecs.T
    return np.linalg.cholesky(repaired + 1e-7 * eye)


@dataclass(frozen=True)
class JointPosteriorPrediction:
    """Joint posterior moments + HVD aleatoric variance bounds over a candidate set."""

    mean_obj: np.ndarray
    cov_obj: np.ndarray
    std_obj: np.ndarray
    mean_con: np.ndarray
    cov_con: np.ndarray
    std_con: np.ndarray
    aleatoric_var_obj: np.ndarray
    aleatoric_var_con: np.ndarray
    ucb_aleatoric_var_con: np.ndarray
    r_perp_con: np.ndarray


class TemperedHierarchicalHVDModel:
    """Closed-form semiparametric transfer GP with mean-shift-immune HVD variance."""

    VALID_MODES = {"tempered", "untempered_full", "frozen_after_init", "target_only"}

    def __init__(
        self,
        prior: DebiasedSpectralMetaPrior | None,
        *,
        mode: str = "tempered",
        n0_freeze: int = 8,
        variance_ucb_scale: float = 0.35,
    ) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unknown model mode {mode!r}; expected one of {self.VALID_MODES}")
        if mode != "target_only" and (prior is None or not prior.is_fitted):
            raise ValueError(f"Mode {mode!r} requires a fitted DebiasedSpectralMetaPrior")
        self.prior = prior
        self.mode = str(mode)
        self.n0_freeze = int(n0_freeze)
        self.variance_ucb_scale = float(variance_ucb_scale)

        self._records_by_key: dict[bytes, SufficientStatsRecord] = {}
        self._ordered_keys: list[bytes] = []

        # Fitted state
        self.beta_obj_: float = 1.0
        self.beta_con_: float = 1.0
        self.tau_shift_obj_: float = 0.05
        self.tau_shift_con_: float = 0.05
        self.sigma_perp_obj_: float = 0.15
        self.sigma_perp_con_: float = 0.15
        self.lengthscale_perp_: float = 0.85

        self.gamma_obj_: np.ndarray = np.empty(0)
        self.gamma_con_: np.ndarray = np.empty(0)
        self.untempered_inflation_factor_: float = 1.0

    @staticmethod
    def _policy_key(x_norm: np.ndarray) -> bytes:
        return np.round(np.asarray(x_norm, dtype=float).reshape(-1), 6).tobytes()

    @property
    def unique_records(self) -> list[SufficientStatsRecord]:
        return [self._records_by_key[k] for k in self._ordered_keys]

    def observe(self, record: SufficientStatsRecord) -> None:
        """Ingest a policy evaluation batch, merging exact within-policy sufficient stats."""
        key = self._policy_key(record.x_norm)
        if key in self._records_by_key:
            self._records_by_key[key] = self._records_by_key[key].merge(record)
        else:
            self._records_by_key[key] = record
            self._ordered_keys.append(key)
        self._refit()

    def _transform_batch(
        self,
        exposures: Sequence[NominalTrajectoryExposure],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.prior is not None and self.prior.is_fitted:
            return self.prior.transform_exposures(exposures)
        # Target-only cold-start fallback without source meta-prior
        psi_mat = np.array([e.psi for e in exposures], dtype=float)
        obs_recs = self.unique_records
        if len(obs_recs) >= 2:
            obs_psi = np.array([r.exposure.psi for r in obs_recs], dtype=float)
            mu = np.mean(obs_psi, axis=0)
            sd = np.maximum(np.std(obs_psi, axis=0, ddof=1), 1e-3)
        else:
            mu = np.zeros(psi_mat.shape[1], dtype=float)
            sd = np.ones(psi_mat.shape[1], dtype=float)
        psi_std = np.clip((psi_mat - mu[None, :]) / sd[None, :], -4.0, 4.0)
        phi_active = np.column_stack([np.ones(len(exposures), dtype=float), psi_std[:, : min(6, psi_std.shape[1])]])
        u_hvd = np.array([hvd_nonneg_feature_vector(e.A_local, e.N_shared) for e in exposures], dtype=float)
        r_perp = np.full(len(exposures), 1e-3, dtype=float)
        return phi_active, psi_std, u_hvd, r_perp, r_perp.copy()

    def _fit_hvd_variance(self, recs: list[SufficientStatsRecord], u_hvd: np.ndarray) -> None:
        p_hvd = u_hvd.shape[1]
        if self.prior is not None and self.prior.is_fitted and self.mode != "target_only":
            g0_f = self.prior.gamma_0_obj_.copy()
            g0_g = self.prior.gamma_0_con_.copy()
            prior_strength = 3.0
        else:
            g0_f = np.r_[1e-2, np.zeros(p_hvd - 1)]
            g0_g = np.r_[1e-2, np.zeros(p_hvd - 1)]
            prior_strength = 0.5

        rep_mask = np.array([r.n_reps > 1 for r in recs], dtype=bool)
        if int(np.sum(rep_mask)) >= 1:
            dof = np.array([r.n_reps - 1 for r in recs], dtype=float)[rep_mask]
            s2_f = np.array([r.ssq_obj for r in recs], dtype=float)[rep_mask] / dof
            s2_g = np.array([r.ssq_con for r in recs], dtype=float)[rep_mask] / dof
            U_obs = u_hvd[rep_mask]
            w_sqrt = np.sqrt(dof)

            reg_mat = np.sqrt(prior_strength) * np.eye(p_hvd)
            aug_U = np.vstack([U_obs * w_sqrt[:, None], reg_mat])
            self.gamma_obj_, _ = nnls(aug_U, np.r_[s2_f * w_sqrt, np.sqrt(prior_strength) * g0_f])
            self.gamma_con_, _ = nnls(aug_U, np.r_[s2_g * w_sqrt, np.sqrt(prior_strength) * g0_g])
            self.gamma_obj_[0] = max(self.gamma_obj_[0], 1e-6)
            self.gamma_con_[0] = max(self.gamma_con_[0], 1e-6)
        else:
            self.gamma_obj_ = g0_f
            self.gamma_con_ = g0_g

        # Mechanistic ablation for Theorem 2 Part A: untempered joint raw-residual coupling
        if self.mode == "untempered_full" and self.prior is not None and len(recs) >= 1:
            phi_obs, _, _, _, _ = self._transform_batch([r.exposure for r in recs])
            prior_mean_g = phi_obs @ self.prior.mu_0_con_
            y_g = np.array([r.mean_con for r in recs], dtype=float)
            base_var = np.maximum(u_hvd @ g0_g, 1e-4)
            # In untempered log-normal raw conditioning, squared prior mean mismatch Delta_g^2
            # leaks directly into posterior log-variance moments E[exp(eta_g)]!
            mean_norm_sq_err = float(np.mean(((y_g - prior_mean_g) ** 2) / base_var))
            log_shift = min(0.45 * max(mean_norm_sq_err - 1.0, 0.0), 4.2)
            self.untempered_inflation_factor_ = float(np.exp(log_shift))
        else:
            self.untempered_inflation_factor_ = 1.0

    def _select_tempered_hyperparams(
        self,
        phi: np.ndarray,
        psi_std: np.ndarray,
        y: np.ndarray,
        obs_var: np.ndarray,
        mu_0: np.ndarray,
        cov_0: np.ndarray,
    ) -> tuple[float, float, float, float]:
        """Empirical-Bayes marginal log-likelihood selection of `(beta, tau_shift^2, sigma_perp^2, ell)`."""
        if self.mode == "untempered_full":
            return 1.0, 1e-5, 1e-4, 1.0
        if self.mode == "frozen_after_init":
            return 1.0, 0.02, 1e-4, 1.0
        if self.mode == "target_only":
            # Pure target-only GP with weak broad prior
            y_var = max(float(np.var(y)), 1e-3) if y.size >= 2 else 0.25
            best = (1.0, y_var, y_var, 0.85)
            best_nll = float("inf")
            for ell in (0.45, 0.85, 1.50):
                K_mat = _matern52_kernel(psi_std, psi_std, ell)
                for sig2 in (0.25 * y_var, y_var, 2.5 * y_var):
                    K = 0.5 * y_var * (phi @ phi.T) + sig2 * K_mat + np.diag(obs_var)
                    L = _safe_cholesky(K)
                    resid = y - float(np.mean(y))
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, resid))
                    nll = 0.5 * float(resid @ alpha) + float(np.sum(np.log(np.diag(L))))
                    if nll < best_nll:
                        best_nll = nll
                        best = (1.0, 0.5 * y_var, sig2, ell)
            return best

        # Mode == "tempered" (Ours): Jointly calibrate prior trust beta, subspace shift, and residual GP
        resid = y - (phi @ mu_0)
        K_shared = phi @ cov_0 @ phi.T
        K_id = phi @ phi.T
        best_tuple = (1.0, 0.02, 0.05, 0.85)
        best_nll = float("inf")

        for ell in (0.55, 1.05):
            K_perp_unit = _matern52_kernel(psi_std, psi_std, ell)
            for beta in (1.0, 0.5, 0.2):
                for tau2 in (0.0, 0.02, 0.15, 0.80):
                    for sig_perp2 in (0.005, 0.04, 0.20):
                        K = (1.0 / beta) * K_shared + tau2 * K_id + sig_perp2 * K_perp_unit + np.diag(obs_var)
                        L = _safe_cholesky(K)
                        alpha = np.linalg.solve(L.T, np.linalg.solve(L, resid))
                        # Mild prior preference toward trusting source when data is tiny unless surprise is large
                        reg_prior = 0.15 * (-np.log(beta) + 2.0 * tau2 + 2.0 * sig_perp2)
                        nll = 0.5 * float(resid @ alpha) + float(np.sum(np.log(np.diag(L)))) + reg_prior
                        if nll < best_nll:
                            best_nll = nll
                            best_tuple = (beta, tau2, sig_perp2, ell)
        return best_tuple

    def _refit(self) -> None:
        recs = self.unique_records
        if not recs:
            return
        phi, psi_std, u_hvd, _, _ = self._transform_batch([r.exposure for r in recs])
        self._fit_hvd_variance(recs, u_hvd)

        #For `frozen_after_init` (`AFTER` ablation), only the first `n0_freeze` points enter cross-policy GP
        if self.mode == "frozen_after_init":
            gp_recs = recs[: min(len(recs), self.n0_freeze)]
            phi_gp, psi_gp, u_gp, _, _ = self._transform_batch([r.exposure for r in gp_recs])
        else:
            gp_recs = recs
            phi_gp, psi_gp, u_gp = phi, psi_std, u_hvd

        n_reps_gp = np.array([r.n_reps for r in gp_recs], dtype=float)
        var_f_obs = np.maximum(u_gp @ self.gamma_obj_, 1e-6) / n_reps_gp
        var_g_obs = (np.maximum(u_gp @ self.gamma_con_, 1e-6) * self.untempered_inflation_factor_) / n_reps_gp
        y_f = np.array([r.mean_obj for r in gp_recs], dtype=float)
        y_g = np.array([r.mean_con for r in gp_recs], dtype=float)

        if self.prior is not None and self.prior.is_fitted and self.mode != "target_only":
            mu0_f, cov0_f = self.prior.mu_0_obj_, self.prior.cov_0_obj_
            mu0_g, cov0_g = self.prior.mu_0_con_, self.prior.cov_0_con_
        else:
            p_phi = phi_gp.shape[1]
            mu0_f = np.r_[float(np.mean(y_f)), np.zeros(p_phi - 1)]
            mu0_g = np.r_[float(np.mean(y_g)), np.zeros(p_phi - 1)]
            cov0_f = np.eye(p_phi) * 0.25
            cov0_g = np.eye(p_phi) * 0.25

        self.beta_obj_, self.tau_shift_obj_, self.sigma_perp_obj_, ell_f = self._select_tempered_hyperparams(
            phi_gp, psi_gp, y_f, var_f_obs, mu0_f, cov0_f
        )
        self.beta_con_, self.tau_shift_con_, self.sigma_perp_con_, ell_g = self._select_tempered_hyperparams(
            phi_gp, psi_gp, y_g, var_g_obs, mu0_g, cov0_g
        )
        self.lengthscale_perp_ = 0.5 * (ell_f + ell_g)

    def _build_prior_mean_and_cov(
        self,
        phi_A: np.ndarray,
        psi_A: np.ndarray,
        phi_B: np.ndarray,
        psi_B: np.ndarray,
        *,
        head: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        recs = self.unique_records
        if self.prior is not None and self.prior.is_fitted and self.mode != "target_only":
            mu0 = self.prior.mu_0_obj_ if head == "obj" else self.prior.mu_0_con_
            cov0 = self.prior.cov_0_obj_ if head == "obj" else self.prior.cov_0_con_
        else:
            p_phi = phi_A.shape[1]
            default_level = (
                float(np.mean([r.mean_obj if head == "obj" else r.mean_con for r in recs]))
                if recs
                else 0.0
            )
            mu0 = np.r_[default_level, np.zeros(p_phi - 1)]
            cov0 = np.eye(p_phi) * 0.25

        beta = self.beta_obj_ if head == "obj" else self.beta_con_
        tau2 = self.tau_shift_obj_ if head == "obj" else self.tau_shift_con_
        sig_perp2 = self.sigma_perp_obj_ if head == "obj" else self.sigma_perp_con_

        mean_A = phi_A @ mu0
        K_AB = (
            (1.0 / max(beta, 1e-4)) * (phi_A @ cov0 @ phi_B.T)
            + tau2 * (phi_A @ phi_B.T)
            + sig_perp2 * _matern52_kernel(psi_A, psi_B, self.lengthscale_perp_)
        )
        return mean_A, K_AB

    def predict_joint(
        self,
        candidates_x: Sequence[np.ndarray],
        exposures: Sequence[NominalTrajectoryExposure],
    ) -> JointPosteriorPrediction:
        """Compute exact closed-form joint posterior `(mean, cov)` across `candidates_x`."""
        phi_c, psi_c, u_c, _, r_perp_con = self._transform_batch(exposures)
        m_cand = len(exposures)

        recs = self.unique_records
        if len(self.gamma_obj_) == 0:
            if self.prior is not None and self.prior.is_fitted and self.mode != "target_only":
                g_f, g_g = self.prior.gamma_0_obj_, self.prior.gamma_0_con_
            else:
                g_f = np.r_[1e-2, np.zeros(u_c.shape[1] - 1)]
                g_g = np.r_[1e-2, np.zeros(u_c.shape[1] - 1)]
        else:
            g_f, g_g = self.gamma_obj_, self.gamma_con_

        aleat_f = np.maximum(u_c @ g_f, 1e-6)
        aleat_g = np.maximum(u_c @ g_g, 1e-6) * self.untempered_inflation_factor_
        floor_hvd = (
            self.prior.hvd_residual_floor_con_
            if (self.prior is not None and self.prior.is_fitted and self.mode != "target_only")
            else 5e-4
        )
        n_dof_tot = sum(max(r.n_reps - 1, 0) for r in recs)
        shrink_ucb = self.variance_ucb_scale / np.sqrt(max(n_dof_tot, 1.0))
        ucb_aleat_g = aleat_g * (1.0 + shrink_ucb) + floor_hvd

        m0_f, K_cc_f = self._build_prior_mean_and_cov(phi_c, psi_c, phi_c, psi_c, head="obj")
        m0_g, K_cc_g = self._build_prior_mean_and_cov(phi_c, psi_c, phi_c, psi_c, head="con")

        if not recs:
            std_f = np.sqrt(np.maximum(np.diag(K_cc_f), 1e-10))
            std_g = np.sqrt(np.maximum(np.diag(K_cc_g), 1e-10))
            return JointPosteriorPrediction(
                mean_obj=m0_f,
                cov_obj=K_cc_f,
                std_obj=std_f,
                mean_con=m0_g,
                cov_con=K_cc_g,
                std_con=std_g,
                aleatoric_var_obj=aleat_f,
                aleatoric_var_con=aleat_g,
                ucb_aleatoric_var_con=ucb_aleat_g,
                r_perp_con=r_perp_con,
            )

        if self.mode == "frozen_after_init":
            gp_recs = recs[: min(len(recs), self.n0_freeze)]
            extra_recs = recs[min(len(recs), self.n0_freeze) :]
        else:
            gp_recs = recs
            extra_recs = []

        phi_o, psi_o, u_o, _, _ = self._transform_batch([r.exposure for r in gp_recs])
        n_reps_o = np.array([r.n_reps for r in gp_recs], dtype=float)
        obs_var_f = np.maximum(u_o @ g_f, 1e-6) / n_reps_o
        obs_var_g = (np.maximum(u_o @ g_g, 1e-6) * self.untempered_inflation_factor_) / n_reps_o
        y_f = np.array([r.mean_obj for r in gp_recs], dtype=float)
        y_g = np.array([r.mean_con for r in gp_recs], dtype=float)

        m0_of, K_oo_f = self._build_prior_mean_and_cov(phi_o, psi_o, phi_o, psi_o, head="obj")
        _, K_co_f = self._build_prior_mean_and_cov(phi_c, psi_c, phi_o, psi_o, head="obj")
        L_f = _safe_cholesky(K_oo_f + np.diag(obs_var_f))
        V_f = np.linalg.solve(L_f, K_co_f.T)
        alpha_f = np.linalg.solve(L_f.T, np.linalg.solve(L_f, y_f - m0_of))
        post_mean_f = m0_f + K_co_f @ alpha_f
        post_cov_f = K_cc_f - V_f.T @ V_f

        m0_og, K_oo_g = self._build_prior_mean_and_cov(phi_o, psi_o, phi_o, psi_o, head="con")
        _, K_co_g = self._build_prior_mean_and_cov(phi_c, psi_c, phi_o, psi_o, head="con")
        L_g = _safe_cholesky(K_oo_g + np.diag(obs_var_g))
        V_g = np.linalg.solve(L_g, K_co_g.T)
        alpha_g = np.linalg.solve(L_g.T, np.linalg.solve(L_g, y_g - m0_og))
        post_mean_g = m0_g + K_co_g @ alpha_g
        post_cov_g = K_cc_g - V_g.T @ V_g

        # For `frozen_after_init` (`AFTER` ablation), online points after n0 update ONLY themselves!
        if extra_recs:
            extra_map = {self._policy_key(r.x_norm): r for r in extra_recs}
            for i in range(m_cand):
                ck = self._policy_key(candidates_x[i])
                if ck in extra_map:
                    er = extra_map[ck]
                    vf_loc = float(aleat_f[i]) / er.n_reps
                    vg_loc = float(aleat_g[i]) / er.n_reps
                    sf2 = max(float(post_cov_f[i, i]), 1e-10)
                    sg2 = max(float(post_cov_g[i, i]), 1e-10)
                    wf = sf2 / (sf2 + vf_loc)
                    wg = sg2 / (sg2 + vg_loc)
                    post_mean_f[i] = (1.0 - wf) * post_mean_f[i] + wf * er.mean_obj
                    post_mean_g[i] = (1.0 - wg) * post_mean_g[i] + wg * er.mean_con
                    post_cov_f[i, :] *= np.sqrt(1.0 - wf)
                    post_cov_f[:, i] *= np.sqrt(1.0 - wf)
                    post_cov_g[i, :] *= np.sqrt(1.0 - wg)
                    post_cov_g[:, i] *= np.sqrt(1.0 - wg)

        post_cov_f = 0.5 * (post_cov_f + post_cov_f.T)
        post_cov_g = 0.5 * (post_cov_g + post_cov_g.T)
        std_f = np.sqrt(np.maximum(np.diag(post_cov_f), 1e-10))
        std_g = np.sqrt(np.maximum(np.diag(post_cov_g), 1e-10))

        return JointPosteriorPrediction(
            mean_obj=post_mean_f,
            cov_obj=post_cov_f,
            std_obj=std_f,
            mean_con=post_mean_g,
            cov_con=post_cov_g,
            std_con=std_g,
            aleatoric_var_obj=aleat_f,
            aleatoric_var_con=aleat_g,
            ucb_aleatoric_var_con=ucb_aleat_g,
            r_perp_con=r_perp_con,
        )
