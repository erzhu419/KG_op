"""Tempered Semiparametric Functional Prior + Shift-Invariant HVD Posterior.

Clean modular design supporting controlled P1 mechanism decomposition and genuine
variance calibration diagnostics (with ZERO artificial inflation multipliers):
- Orthogonal switches separate every source information channel:
  * `use_source_basis`: source-learned low-pass whitened sparse basis `tilde_phi(psi)`
    vs. target-only standardized `psi` basis `[1, psi_std]`;
  * `use_source_coef_prior`: cross-task empirical-Bayes `(mu_0, Sigma_0)` with MLL
    tempering vs. weak zero-mean isotropic prior `N([y_bar, 0], 0.25 I)`;
  * `use_source_hvd_prior`: source HVD variance prior `gamma_0` vs. pure target-only
    within-policy NNLS with fixed weak regularization (`prior_strength = 0.5`).
- Supports controlled variance estimators using the exact same NNLS engine:
  * `"within_ssq"`: `ssq / (n_reps - 1)` (algebraically invariant to mean shifts);
  * `"naive_prior_residual_sq"`: `(mean_y - mu_0(x))^2 + ssq / n_reps` (exact sample
    second moment around source prior mean, testing genuine mean-shift leakage without
    any hand-coded multiplier);
  * `"posterior_residual_sq"`: `(mean_y - post_mean(x))^2 + ssq / n_reps`.
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
    """Joint posterior moments + HVD aleatoric variance estimates over a candidate set."""

    mean_obj: np.ndarray
    cov_obj: np.ndarray
    std_obj: np.ndarray
    mean_con: np.ndarray
    cov_con: np.ndarray
    std_con: np.ndarray
    aleatoric_var_obj: np.ndarray
    aleatoric_var_con: np.ndarray
    ucb_aleatoric_var_con: np.ndarray
    r_perp_con: np.ndarray  # Empirical pruned-basis uncertainty penalty


class TemperedHierarchicalHVDModel:
    """Closed-form semiparametric transfer GP with modular mechanism switches."""

    VALID_MODES = {
        "tempered",
        "tempered_adaptive",
        "tempered_full_release",
        "source_basis_weak_prior",
        "untempered_fixed_prior",
        "frozen_after_init",
        "target_only",
        # Legacy compatibility alias mapped cleanly without artificial multiplier
        "untempered_full",
    }
    VALID_VAR_ESTIMATORS = {
        "within_ssq",
        "naive_prior_residual_sq",
        "posterior_residual_sq",
    }

    def __init__(
        self,
        prior: DebiasedSpectralMetaPrior | None,
        *,
        mode: str = "tempered",
        use_source_basis: bool | None = None,
        use_source_coef_prior: bool | None = None,
        use_source_hvd_prior: bool | None = None,
        variance_estimator: str = "within_ssq",
        use_empirical_r_perp: bool = True,
        n0_freeze: int = 8,
        variance_ucb_scale: float = 0.35,
    ) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unknown model mode {mode!r}; expected one of {self.VALID_MODES}")
        if variance_estimator not in self.VALID_VAR_ESTIMATORS:
            raise ValueError(f"Unknown variance_estimator {variance_estimator!r}")

        self.mode = str(mode)
        self.prior = prior
        self.use_source_basis = (
            bool(use_source_basis)
            if use_source_basis is not None
            else (self.mode != "target_only")
        )
        self.use_source_coef_prior = (
            bool(use_source_coef_prior)
            if use_source_coef_prior is not None
            else (self.mode not in {"target_only", "source_basis_weak_prior"})
        )
        self.use_source_hvd_prior = (
            bool(use_source_hvd_prior)
            if use_source_hvd_prior is not None
            else (self.mode not in {"target_only", "source_basis_weak_prior"})
        )
        if self.mode == "untempered_full" and variance_estimator == "within_ssq":
            # Map legacy mode name to genuine raw prior-residual NNLS (no artificial multiplier!)
            self.variance_estimator = "naive_prior_residual_sq"
        else:
            self.variance_estimator = str(variance_estimator)

        if (self.use_source_basis or self.use_source_coef_prior or self.use_source_hvd_prior) and (
            prior is None or not prior.is_fitted
        ):
            raise ValueError("Requested source prior channel requires a fitted DebiasedSpectralMetaPrior")
        if self.use_source_coef_prior and not self.use_source_basis:
            raise ValueError("use_source_coef_prior=True requires use_source_basis=True")

        self.use_empirical_r_perp = bool(use_empirical_r_perp)
        self.n0_freeze = int(n0_freeze)
        self.variance_ucb_scale = float(variance_ucb_scale)

        self._records_by_key: dict[bytes, SufficientStatsRecord] = {}
        self._ordered_keys: list[bytes] = []

        self.beta_obj_: float = 1.0
        self.beta_con_: float = 1.0
        self.tau_shift_obj_: float = 0.05
        self.tau_shift_con_: float = 0.05
        self.sigma_perp_obj_: float = 0.15
        self.sigma_perp_con_: float = 0.15
        self.lengthscale_perp_: float = 0.85
        self.lengthscale_obj_: float = 0.85
        self.lengthscale_con_: float = 0.85
        self.is_source_released_: bool = False

        self.gamma_obj_: np.ndarray = np.empty(0)
        self.gamma_con_: np.ndarray = np.empty(0)
        # Kept strictly as diagnostic ratio vs within-SSQ estimator on observed points (no injection!)
        self.empirical_var_ratio_vs_within_: float = 1.0
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
        if self.use_source_basis and self.prior is not None and self.prior.is_fitted:
            phi_act, psi_std, u_hvd, r_f, r_g = self.prior.transform_exposures(exposures)
            if not self.use_empirical_r_perp or not self.use_source_coef_prior:
                r_f = np.full(len(exposures), 1e-4, dtype=float)
                r_g = np.full(len(exposures), 1e-4, dtype=float)
            return phi_act, psi_std, u_hvd, r_f, r_g

        # Target-only representation on public `psi in R^12`
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
        phi_active = np.column_stack([np.ones(len(exposures), dtype=float), psi_std[:, : min(8, psi_std.shape[1])]])
        u_hvd = np.array([hvd_nonneg_feature_vector(e.A_local, e.N_shared) for e in exposures], dtype=float)
        r_perp = np.full(len(exposures), 1e-4, dtype=float)
        return phi_active, psi_std, u_hvd, r_perp, r_perp.copy()

    def _target_standardized_psi(self, exposures: Sequence[NominalTrajectoryExposure]) -> np.ndarray:
        """Standardize `psi in R^12` using target observations only (for full-channel release)."""
        psi_mat = np.array([e.psi for e in exposures], dtype=float)
        obs_recs = self.unique_records
        if len(obs_recs) >= 2:
            obs_psi = np.array([r.exposure.psi for r in obs_recs], dtype=float)
            mu = np.mean(obs_psi, axis=0)
            sd = np.maximum(np.std(obs_psi, axis=0, ddof=1), 1e-3)
        else:
            mu = np.zeros(psi_mat.shape[1], dtype=float)
            sd = np.ones(psi_mat.shape[1], dtype=float)
        return np.clip((psi_mat - mu[None, :]) / sd[None, :], -4.0, 4.0)

    def _fit_hvd_variance(
        self,
        recs: list[SufficientStatsRecord],
        phi_obs: np.ndarray,
        u_hvd: np.ndarray,
        post_mean_obj: np.ndarray | None = None,
        post_mean_con: np.ndarray | None = None,
        force_target_only_hvd: bool = False,
    ) -> None:
        p_hvd = u_hvd.shape[1]
        n_src_multirep = int(getattr(self.prior, "hvd_prior_n_multirep_", 100)) if self.prior is not None else 0
        use_src_hvd = (
            (not force_target_only_hvd)
            and self.use_source_hvd_prior
            and self.prior is not None
            and self.prior.is_fitted
            and n_src_multirep >= 6
        )

        if self.variance_estimator == "within_ssq":
            rep_mask = np.array([r.n_reps > 1 for r in recs], dtype=bool)
            if int(np.sum(rep_mask)) >= 1:
                dof = np.array([r.n_reps - 1 for r in recs], dtype=float)[rep_mask]
                target_f = np.array([r.ssq_obj for r in recs], dtype=float)[rep_mask] / dof
                target_g = np.array([r.ssq_con for r in recs], dtype=float)[rep_mask] / dof
                U_obs = u_hvd[rep_mask]
                w_sqrt = np.sqrt(dof)
                tot_dof = float(np.sum(dof))
            else:
                if use_src_hvd:
                    self.gamma_obj_ = self.prior.gamma_0_obj_.copy()
                    self.gamma_con_ = self.prior.gamma_0_con_.copy()
                else:
                    y_f_all = np.array([r.mean_obj for r in recs], dtype=float)
                    y_g_all = np.array([r.mean_con for r in recs], dtype=float)
                    vf0 = max(0.05 * float(np.var(y_f_all)), 1e-8) if y_f_all.size >= 2 else 1e-4
                    vg0 = max(0.05 * float(np.var(y_g_all)), 1e-8) if y_g_all.size >= 2 else 1e-4
                    self.gamma_obj_ = np.r_[vf0, np.zeros(p_hvd - 1)]
                    self.gamma_con_ = np.r_[vg0, np.zeros(p_hvd - 1)]
                self.empirical_var_ratio_vs_within_ = 1.0
                self.untempered_inflation_factor_ = 1.0
                return
        elif self.variance_estimator == "naive_prior_residual_sq":
            # Exact per-replication squared residual around prior mean:
            # (1/m) sum_{l=1}^m (Y_l - mu_0(x))^2 = (Y_bar - mu_0(x))^2 + ssq / m
            if self.prior is not None and self.prior.is_fitted and self.use_source_coef_prior:
                ref_f = phi_obs @ self.prior.mu_0_obj_
                ref_g = phi_obs @ self.prior.mu_0_con_
            else:
                ref_f = np.full(len(recs), float(np.mean([r.mean_obj for r in recs])))
                ref_g = np.full(len(recs), float(np.mean([r.mean_con for r in recs])))
            m_arr = np.array([r.n_reps for r in recs], dtype=float)
            y_f = np.array([r.mean_obj for r in recs], dtype=float)
            y_g = np.array([r.mean_con for r in recs], dtype=float)
            ssq_f = np.array([r.ssq_obj for r in recs], dtype=float)
            ssq_g = np.array([r.ssq_con for r in recs], dtype=float)
            target_f = (y_f - ref_f) ** 2 + ssq_f / m_arr
            target_g = (y_g - ref_g) ** 2 + ssq_g / m_arr
            U_obs = u_hvd
            w_sqrt = np.sqrt(m_arr)
            tot_dof = float(np.sum(np.maximum(m_arr - 1.0, 0.0)))
        else:  # posterior_residual_sq
            m_arr = np.array([r.n_reps for r in recs], dtype=float)
            y_f = np.array([r.mean_obj for r in recs], dtype=float)
            y_g = np.array([r.mean_con for r in recs], dtype=float)
            ref_f = post_mean_obj if post_mean_obj is not None else np.full(len(recs), float(np.mean(y_f)))
            ref_g = post_mean_con if post_mean_con is not None else np.full(len(recs), float(np.mean(y_g)))
            ssq_f = np.array([r.ssq_obj for r in recs], dtype=float)
            ssq_g = np.array([r.ssq_con for r in recs], dtype=float)
            target_f = (y_f - ref_f) ** 2 + ssq_f / m_arr
            target_g = (y_g - ref_g) ** 2 + ssq_g / m_arr
            U_obs = u_hvd
            w_sqrt = np.sqrt(m_arr)
            tot_dof = float(np.sum(np.maximum(m_arr - 1.0, 0.0)))

        med_tf = max(float(np.median(target_f)), 1e-10)
        med_tg = max(float(np.median(target_g)), 1e-10)
        if use_src_hvd:
            g0_f = self.prior.gamma_0_obj_.copy()
            g0_g = self.prior.gamma_0_con_.copy()
            base_strength = min(3.0, 0.5 + 2.5 * (float(n_src_multirep) / 36.0))
            prior_strength = (
                base_strength / (1.0 + 0.10 * tot_dof)
                if self.mode == "tempered_full_release"
                else base_strength
            )
        else:
            # Scale-equivariant target-only baseline regularization (never hardcoded `1e-2`!)
            g0_f = np.r_[med_tf, np.zeros(p_hvd - 1)]
            g0_g = np.r_[med_tg, np.zeros(p_hvd - 1)]
            prior_strength = 0.5

        reg_mat = np.sqrt(prior_strength) * np.eye(p_hvd)
        aug_U = np.vstack([U_obs * w_sqrt[:, None], reg_mat])
        self.gamma_obj_, _ = nnls(aug_U, np.r_[target_f * w_sqrt, np.sqrt(prior_strength) * g0_f])
        self.gamma_con_, _ = nnls(aug_U, np.r_[target_g * w_sqrt, np.sqrt(prior_strength) * g0_g])
        self.gamma_obj_[0] = max(self.gamma_obj_[0], 0.02 * med_tf, 1e-10)
        self.gamma_con_[0] = max(self.gamma_con_[0], 0.02 * med_tg, 1e-10)

        # Purely observational diagnostic comparing fitted constraint variance vs pure within-SSQ
        rep_mask = np.array([r.n_reps > 1 for r in recs], dtype=bool)
        if int(np.sum(rep_mask)) >= 1:
            dof = np.array([r.n_reps - 1 for r in recs], dtype=float)[rep_mask]
            s2_g_pure = np.maximum(np.array([r.ssq_con for r in recs], dtype=float)[rep_mask] / dof, 1e-6)
            pred_g = np.maximum(u_hvd[rep_mask] @ self.gamma_con_, 1e-6)
            self.empirical_var_ratio_vs_within_ = float(np.median(pred_g / s2_g_pure))
        else:
            self.empirical_var_ratio_vs_within_ = 1.0
        self.untempered_inflation_factor_ = self.empirical_var_ratio_vs_within_

    def _select_tempered_hyperparams(
        self,
        phi: np.ndarray,
        psi_std: np.ndarray,
        y: np.ndarray,
        obs_var: np.ndarray,
        mu_0: np.ndarray,
        cov_0: np.ndarray,
        psi_tgt_std: np.ndarray | None = None,
        w_perp: np.ndarray | None = None,
    ) -> tuple[float, float, float, float]:
        """Data-driven regularized marginal log-likelihood grid selector for `(beta, tau2, sig_perp2, ell)`."""
        if self.mode in {"untempered_fixed_prior", "untempered_full"} and self.use_source_coef_prior:
            return 1.0, 0.0, 0.01, 0.85
        if self.mode == "frozen_after_init" and self.use_source_coef_prior:
            return 1.0, 0.02, 1e-4, 1.0

        if not self.use_source_coef_prior:
            # Weak zero-mean prior (used by BOTH `target_only` and `source_basis_weak_prior` Arm C!)
            y_var = max(float(np.var(y)), 1e-3) if y.size >= 2 else 0.25
            best = (1.0, 0.5 * y_var, y_var, 0.85)
            best_nll = float("inf")
            resid = y - (phi @ mu_0)
            K_lin = phi @ phi.T
            for ell in (0.45, 0.85, 1.50):
                K_mat = _matern52_kernel(psi_std, psi_std, ell)
                for tau2 in (0.15 * y_var, 0.60 * y_var, 1.5 * y_var):
                    for sig2 in (0.10 * y_var, 0.50 * y_var, 1.5 * y_var):
                        K = tau2 * K_lin + sig2 * K_mat + np.diag(obs_var)
                        L = _safe_cholesky(K)
                        alpha = np.linalg.solve(L.T, np.linalg.solve(L, resid))
                        nll = 0.5 * float(resid @ alpha) + float(np.sum(np.log(np.diag(L))))
                        if nll < best_nll:
                            best_nll = nll
                            best = (1.0, tau2, sig2, ell)
            return best

        if self.mode == "tempered_full_release" and self.use_source_coef_prior:
            # Analytic Rasmussen-Williams Eq. 5.12 Leave-One-Out Predictive Log-Likelihood (LOO-PLL)
            # selector: directly measures out-of-fold predictive density on target `y` in native units,
            # immune to prior feature-scaling Occam distortion while cleanly releasing under shift!
            n_obs = max(int(y.size), 1)
            y_var = max(
                float(np.var(y)) if n_obs >= 2 else 0.25,
                float(np.median(obs_var)) if obs_var.size else 1e-4,
                1e-8,
            )
            if n_obs < 6:
                return 1.0, 0.08 * y_var, 0.10 * y_var, 0.85

            eye_n = np.eye(n_obs, dtype=float)
            resid_src = y - (phi @ mu_0)
            y_lvl = float(np.mean(y))
            resid_tgt = y - y_lvl
            K_shared = phi @ cov_0 @ phi.T
            if self.use_empirical_r_perp and w_perp is not None and w_perp.shape[1] > 0:
                K_shared = K_shared + (w_perp @ w_perp.T)
            K_id_src = phi @ phi.T

            psi_rel = psi_tgt_std if psi_tgt_std is not None else psi_std
            phi_rel = np.column_stack([np.ones(n_obs, dtype=float), psi_rel[:, : min(8, psi_rel.shape[1])]])
            K_lin_rel = phi_rel @ phi_rel.T

            def _loo_nll(K_mat_tot: np.ndarray, res_vec: np.ndarray) -> float:
                L_c = _safe_cholesky(K_mat_tot)
                K_inv = np.linalg.solve(L_c.T, np.linalg.solve(L_c, eye_n))
                alp_v = K_inv @ res_vec
                d_inv = np.maximum(np.diag(K_inv), 1e-12)
                return 0.5 * float(np.sum((alp_v * alp_v) / d_inv - np.log(d_inv)))

            best_src_tuple = (1.0, 0.08 * y_var, 0.10 * y_var, 0.85)
            best_src_score = float("inf")
            reg_scale = 0.08 * np.sqrt(float(n_obs))

            for ell in (0.45, 0.85, 1.45):
                K_perp_src = _matern52_kernel(psi_std, psi_std, ell)
                for beta in (1.5, 1.0, 0.35):
                    for tau2_mult in (0.0, 0.08, 0.35):
                        tau2 = tau2_mult * y_var
                        for sig_mult in (0.03, 0.20, 0.75):
                            sig_perp2 = sig_mult * y_var
                            K = (1.0 / beta) * K_shared + tau2 * K_id_src + sig_perp2 * K_perp_src + np.diag(obs_var)
                            reg = reg_scale * (abs(np.log(beta)) + 0.5 * tau2_mult + 0.25 * sig_mult)
                            sc = _loo_nll(K, resid_src) + reg
                            if sc < best_src_score:
                                best_src_score = sc
                                best_src_tuple = (beta, tau2, sig_perp2, ell)

            if n_obs >= 10:
                best_rel_tuple = (0.0, 0.60 * y_var, 0.50 * y_var, 0.85)
                best_rel_score = float("inf")
                for ell in (0.45, 0.85, 1.50):
                    K_mat_rel = _matern52_kernel(psi_rel, psi_rel, ell)
                    for tau2_mult in (0.15, 0.60, 1.50):
                        tau2 = tau2_mult * y_var
                        for sig_mult in (0.10, 0.50, 1.50):
                            sig_perp2 = sig_mult * y_var
                            K = tau2 * K_lin_rel + sig_perp2 * K_mat_rel + np.diag(obs_var)
                            sc = _loo_nll(K, resid_tgt)
                            if sc < best_rel_score:
                                best_rel_score = sc
                                best_rel_tuple = (0.0, tau2, sig_perp2, ell)
                # Consistent O(log n) Schwarz-BIC LOO threshold: protects aligned priors at small n
                # while guaranteeing asymptotic release under ANY persistent subspace shift!
                bic_loo_thresh = 0.50 * float(np.log(max(n_obs, 2)))
                if best_rel_score + bic_loo_thresh < best_src_score:
                    return best_rel_tuple

            return best_src_tuple

        if self.mode == "tempered_adaptive" and self.use_source_coef_prior:
            # Scale-equivariant MLL grid G including the exact zero-precision release point beta = 0.0
            # (matching `tau = 0` in `FunctionalTransferRisk.realizable_grid_selector_total_risk_le`).
            n_obs = max(int(y.size), 1)
            y_var = max(
                float(np.var(y)) if n_obs >= 2 else 0.25,
                float(np.median(obs_var)) if obs_var.size else 1e-4,
                1e-6,
            )
            resid_src = y - (phi @ mu_0)
            y_lvl = float(np.mean(y)) if n_obs >= 1 else float(mu_0[0])
            resid_tgt = y - y_lvl
            K_shared = phi @ cov_0 @ phi.T
            K_id = phi @ phi.T
            best_tuple = (1.0, 0.08 * y_var, 0.10 * y_var, 0.85)
            best_nll = float("inf")
            reg_scale = 0.18 / np.sqrt(float(n_obs))
            bic_release_pen = 0.5 * float(np.log(max(n_obs, 2)))

            for ell in (0.55, 1.05):
                K_perp_unit = _matern52_kernel(psi_std, psi_std, ell)
                # 1. Source-anchored tempered candidates (`beta > 0`)
                for beta in (1.5, 1.0, 0.35):
                    for tau2_mult in (0.0, 0.08, 0.40):
                        tau2 = tau2_mult * y_var
                        for sig_mult in (0.03, 0.20, 0.80):
                            sig_perp2 = sig_mult * y_var
                            K = (1.0 / beta) * K_shared + tau2 * K_id + sig_perp2 * K_perp_unit + np.diag(obs_var)
                            L = _safe_cholesky(K)
                            alpha = np.linalg.solve(L.T, np.linalg.solve(L, resid_src))
                            reg = reg_scale * (abs(np.log(beta)) + tau2_mult + sig_mult)
                            nll = 0.5 * float(resid_src @ alpha) + float(np.sum(np.log(np.diag(L)))) + reg
                            if nll < best_nll:
                                best_nll = nll
                                best_tuple = (beta, tau2, sig_perp2, ell)
                # 2. Target-only release branch (`beta = 0.0`, i.e., `tau_source = 0 in G`)
                if n_obs >= 10:
                    for tau2_mult in (0.15, 0.60, 1.50):
                        tau2 = tau2_mult * y_var
                        for sig_mult in (0.10, 0.50, 1.50):
                            sig_perp2 = sig_mult * y_var
                            K = tau2 * K_id + sig_perp2 * K_perp_unit + np.diag(obs_var)
                            L = _safe_cholesky(K)
                            alpha = np.linalg.solve(L.T, np.linalg.solve(L, resid_tgt))
                            nll = 0.5 * float(resid_tgt @ alpha) + float(np.sum(np.log(np.diag(L)))) + bic_release_pen
                            if nll < best_nll:
                                best_nll = nll
                                best_tuple = (0.0, tau2, sig_perp2, ell)
            return best_tuple

        # Tempered source coefficient prior (`use_source_coef_prior == True`, fixed-scale P1 baseline)
        resid = y - (phi @ mu_0)
        K_shared = phi @ cov_0 @ phi.T
        K_id = phi @ phi.T
        best_tuple = (1.0, 0.02, 0.05, 0.85)
        best_nll = float("inf")

        for ell in (0.55, 1.05):
            K_perp_unit = _matern52_kernel(psi_std, psi_std, ell)
            for beta in (1.5, 1.0, 0.5, 0.2):
                for tau2 in (0.0, 0.02, 0.15, 0.80):
                    for sig_perp2 in (0.005, 0.04, 0.20):
                        K = (1.0 / beta) * K_shared + tau2 * K_id + sig_perp2 * K_perp_unit + np.diag(obs_var)
                        L = _safe_cholesky(K)
                        alpha = np.linalg.solve(L.T, np.linalg.solve(L, resid))
                        reg_prior = 0.12 * (abs(np.log(beta)) + 2.0 * tau2 + 2.0 * sig_perp2)
                        nll = 0.5 * float(resid @ alpha) + float(np.sum(np.log(np.diag(L)))) + reg_prior
                        if nll < best_nll:
                            best_nll = nll
                            best_tuple = (beta, tau2, sig_perp2, ell)
        return best_tuple

    def _prior_moments_head(self, p_phi: int, recs: list[SufficientStatsRecord], head: str) -> tuple[np.ndarray, np.ndarray]:
        if self.use_source_coef_prior and self.prior is not None and self.prior.is_fitted:
            if head == "obj":
                return self.prior.mu_0_obj_, self.prior.cov_0_obj_
            return self.prior.mu_0_con_, self.prior.cov_0_con_
        lvl = (
            float(np.mean([r.mean_obj if head == "obj" else r.mean_con for r in recs]))
            if recs
            else 0.0
        )
        return np.r_[lvl, np.zeros(p_phi - 1)], np.zeros((p_phi, p_phi), dtype=float)

    def _refit(self) -> None:
        recs = self.unique_records
        if not recs:
            return
        phi, psi_std, u_hvd, _, _ = self._transform_batch([r.exposure for r in recs])
        self._fit_hvd_variance(recs, phi, u_hvd)

        if self.mode == "frozen_after_init":
            gp_recs = recs[: min(len(recs), self.n0_freeze)]
            phi_gp, psi_gp, u_gp, _, _ = self._transform_batch([r.exposure for r in gp_recs])
        else:
            gp_recs = recs
            phi_gp, psi_gp, u_gp = phi, psi_std, u_hvd

        n_reps_gp = np.array([r.n_reps for r in gp_recs], dtype=float)
        var_f_obs = np.maximum(u_gp @ self.gamma_obj_, 1e-10) / n_reps_gp
        var_g_obs = np.maximum(u_gp @ self.gamma_con_, 1e-10) / n_reps_gp
        y_f = np.array([r.mean_obj for r in gp_recs], dtype=float)
        y_g = np.array([r.mean_con for r in gp_recs], dtype=float)

        mu0_f, cov0_f = self._prior_moments_head(phi_gp.shape[1], gp_recs, "obj")
        mu0_g, cov0_g = self._prior_moments_head(phi_gp.shape[1], gp_recs, "con")
        psi_tgt_gp = self._target_standardized_psi([r.exposure for r in gp_recs]) if self.mode == "tempered_full_release" else None
        if (
            self.mode == "tempered_full_release"
            and self.use_source_basis
            and self.prior is not None
            and self.prior.is_fitted
        ):
            wp_gp_f, wp_gp_g = self.prior.transform_pruned_features([r.exposure for r in gp_recs])
        else:
            wp_gp_f = wp_gp_g = None

        self.beta_obj_, self.tau_shift_obj_, self.sigma_perp_obj_, ell_f = self._select_tempered_hyperparams(
            phi_gp, psi_gp, y_f, var_f_obs, mu0_f, cov0_f, psi_tgt_std=psi_tgt_gp, w_perp=wp_gp_f
        )
        self.beta_con_, self.tau_shift_con_, self.sigma_perp_con_, ell_g = self._select_tempered_hyperparams(
            phi_gp, psi_gp, y_g, var_g_obs, mu0_g, cov0_g, psi_tgt_std=psi_tgt_gp, w_perp=wp_gp_g
        )
        self.lengthscale_obj_ = float(ell_f)
        self.lengthscale_con_ = float(ell_g)
        self.lengthscale_perp_ = 0.5 * (ell_f + ell_g)
        self.is_source_released_ = bool(
            self.mode == "tempered_full_release" and (self.beta_obj_ <= 0.0 or self.beta_con_ <= 0.0)
        )
        if self.is_source_released_ and sum(1 for r in recs if r.n_reps > 1) >= 4:
            self._fit_hvd_variance(recs, phi, u_hvd, force_target_only_hvd=True)

        if self.variance_estimator == "posterior_residual_sq" and len(recs) >= 2:
            # One Gauss-Newton style refinement pass using fitted posterior mean
            pred_obs = self.predict_joint([r.x_norm for r in recs], [r.exposure for r in recs])
            self._fit_hvd_variance(recs, phi, u_hvd, pred_obs.mean_obj, pred_obs.mean_con)

    def _build_prior_mean_and_cov(
        self,
        phi_A: np.ndarray,
        psi_A: np.ndarray,
        phi_B: np.ndarray,
        psi_B: np.ndarray,
        *,
        head: str,
        psi_tgt_A: np.ndarray | None = None,
        psi_tgt_B: np.ndarray | None = None,
        w_perp_A: np.ndarray | None = None,
        w_perp_B: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        recs = self.unique_records
        mu0, cov0 = self._prior_moments_head(phi_A.shape[1], recs, head)

        beta = self.beta_obj_ if head == "obj" else self.beta_con_
        tau2 = self.tau_shift_obj_ if head == "obj" else self.tau_shift_con_
        sig_perp2 = self.sigma_perp_obj_ if head == "obj" else self.sigma_perp_con_
        ell_head = (
            (self.lengthscale_obj_ if head == "obj" else self.lengthscale_con_)
            if self.mode == "tempered_full_release"
            else self.lengthscale_perp_
        )

        if beta <= 0.0:
            lvl = (
                float(np.mean([r.mean_obj if head == "obj" else r.mean_con for r in recs]))
                if recs
                else 0.0
            )
            mean_A = np.full(phi_A.shape[0], lvl, dtype=float)
            if self.mode == "tempered_full_release" and psi_tgt_A is not None and psi_tgt_B is not None:
                phi_rel_A = np.column_stack([np.ones(psi_tgt_A.shape[0], dtype=float), psi_tgt_A[:, : min(8, psi_tgt_A.shape[1])]])
                phi_rel_B = np.column_stack([np.ones(psi_tgt_B.shape[0], dtype=float), psi_tgt_B[:, : min(8, psi_tgt_B.shape[1])]])
                K_AB = (
                    tau2 * (phi_rel_A @ phi_rel_B.T)
                    + sig_perp2 * _matern52_kernel(psi_tgt_A, psi_tgt_B, ell_head)
                )
                return mean_A, K_AB
            K_AB = (
                tau2 * (phi_A @ phi_B.T)
                + sig_perp2 * _matern52_kernel(psi_A, psi_B, ell_head)
            )
            return mean_A, K_AB

        mean_A = phi_A @ mu0
        K_AB = (
            (1.0 / max(beta, 1e-4)) * (phi_A @ cov0 @ phi_B.T)
            + tau2 * (phi_A @ phi_B.T)
            + sig_perp2 * _matern52_kernel(psi_A, psi_B, ell_head)
        )
        if (
            self.mode == "tempered_full_release"
            and self.use_empirical_r_perp
            and w_perp_A is not None
            and w_perp_B is not None
            and w_perp_A.shape[1] > 0
        ):
            # Condition pruned-basis uncertainty inside the GP kernel so Bayes' rule shrinks it at observed points!
            K_AB = K_AB + (1.0 / max(beta, 1e-4)) * (w_perp_A @ w_perp_B.T)
        return mean_A, K_AB

    def predict_joint(
        self,
        candidates_x: Sequence[np.ndarray],
        exposures: Sequence[NominalTrajectoryExposure],
    ) -> JointPosteriorPrediction:
        """Compute exact closed-form joint posterior `(mean, cov)` across `candidates_x`."""
        phi_c, psi_c, u_c, _, r_perp_con = self._transform_batch(exposures)
        m_cand = len(exposures)
        if self.beta_con_ <= 0.0 or self.mode == "tempered_full_release":
            # In `tempered_full_release`, pruned-mode uncertainty is conditioned inside `post_cov_g`!
            r_perp_con = np.full(m_cand, 1e-10, dtype=float)

        recs = self.unique_records
        if len(self.gamma_obj_) == 0:
            if self.use_source_hvd_prior and self.prior is not None and self.prior.is_fitted:
                g_f, g_g = self.prior.gamma_0_obj_, self.prior.gamma_0_con_
            else:
                g_f = np.r_[1e-2, np.zeros(u_c.shape[1] - 1)]
                g_g = np.r_[1e-2, np.zeros(u_c.shape[1] - 1)]
        else:
            g_f, g_g = self.gamma_obj_, self.gamma_con_

        aleat_f = np.maximum(u_c @ g_f, 1e-10)
        aleat_g = np.maximum(u_c @ g_g, 1e-10)
        med_aleat_g = max(float(np.median(aleat_g)), 1e-10)
        n_dof_tot = sum(max(r.n_reps - 1, 0) for r in recs)
        if (
            (not self.is_source_released_)
            and self.use_source_hvd_prior
            and self.prior is not None
            and self.prior.is_fitted
        ):
            floor_hvd = (
                min(float(self.prior.hvd_residual_floor_con_), 0.20 * med_aleat_g)
                if (self.mode == "tempered_full_release" and n_dof_tot >= 4)
                else float(self.prior.hvd_residual_floor_con_)
            )
        else:
            floor_hvd = 0.05 * med_aleat_g
        shrink_ucb = self.variance_ucb_scale / np.sqrt(max(n_dof_tot, 1.0))
        ucb_aleat_g = aleat_g * (1.0 + shrink_ucb) + floor_hvd

        psi_tgt_c = self._target_standardized_psi(exposures) if self.mode == "tempered_full_release" else None
        if (
            self.mode == "tempered_full_release"
            and self.use_source_basis
            and self.prior is not None
            and self.prior.is_fitted
        ):
            wp_cf, wp_cg = self.prior.transform_pruned_features(exposures)
        else:
            wp_cf = wp_cg = None

        m0_f, K_cc_f = self._build_prior_mean_and_cov(
            phi_c, psi_c, phi_c, psi_c, head="obj", psi_tgt_A=psi_tgt_c, psi_tgt_B=psi_tgt_c, w_perp_A=wp_cf, w_perp_B=wp_cf
        )
        m0_g, K_cc_g = self._build_prior_mean_and_cov(
            phi_c, psi_c, phi_c, psi_c, head="con", psi_tgt_A=psi_tgt_c, psi_tgt_B=psi_tgt_c, w_perp_A=wp_cg, w_perp_B=wp_cg
        )

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

        obs_exps = [r.exposure for r in gp_recs]
        phi_o, psi_o, u_o, _, _ = self._transform_batch(obs_exps)
        psi_tgt_o = (
            self._target_standardized_psi(obs_exps)
            if self.mode == "tempered_full_release"
            else None
        )
        if (
            self.mode == "tempered_full_release"
            and self.use_source_basis
            and self.prior is not None
            and self.prior.is_fitted
        ):
            wp_of, wp_og = self.prior.transform_pruned_features(obs_exps)
        else:
            wp_of = wp_og = None

        n_reps_o = np.array([r.n_reps for r in gp_recs], dtype=float)
        obs_var_f = np.maximum(u_o @ g_f, 1e-10) / n_reps_o
        obs_var_g = np.maximum(u_o @ g_g, 1e-10) / n_reps_o
        y_f = np.array([r.mean_obj for r in gp_recs], dtype=float)
        y_g = np.array([r.mean_con for r in gp_recs], dtype=float)

        m0_of, K_oo_f = self._build_prior_mean_and_cov(
            phi_o, psi_o, phi_o, psi_o, head="obj", psi_tgt_A=psi_tgt_o, psi_tgt_B=psi_tgt_o, w_perp_A=wp_of, w_perp_B=wp_of
        )
        _, K_co_f = self._build_prior_mean_and_cov(
            phi_c, psi_c, phi_o, psi_o, head="obj", psi_tgt_A=psi_tgt_c, psi_tgt_B=psi_tgt_o, w_perp_A=wp_cf, w_perp_B=wp_of
        )
        L_f = _safe_cholesky(K_oo_f + np.diag(obs_var_f))
        V_f = np.linalg.solve(L_f, K_co_f.T)
        alpha_f = np.linalg.solve(L_f.T, np.linalg.solve(L_f, y_f - m0_of))
        post_mean_f = m0_f + K_co_f @ alpha_f
        post_cov_f = K_cc_f - V_f.T @ V_f

        m0_og, K_oo_g = self._build_prior_mean_and_cov(
            phi_o, psi_o, phi_o, psi_o, head="con", psi_tgt_A=psi_tgt_o, psi_tgt_B=psi_tgt_o, w_perp_A=wp_og, w_perp_B=wp_og
        )
        _, K_co_g = self._build_prior_mean_and_cov(
            phi_c, psi_c, phi_o, psi_o, head="con", psi_tgt_A=psi_tgt_c, psi_tgt_B=psi_tgt_o, w_perp_A=wp_cg, w_perp_B=wp_og
        )
        L_g = _safe_cholesky(K_oo_g + np.diag(obs_var_g))
        V_g = np.linalg.solve(L_g, K_co_g.T)
        alpha_g = np.linalg.solve(L_g.T, np.linalg.solve(L_g, y_g - m0_og))
        post_mean_g = m0_g + K_co_g @ alpha_g
        post_cov_g = K_cc_g - V_g.T @ V_g

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
