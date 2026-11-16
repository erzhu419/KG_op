"""Debiased Low-Frequency Orthogonal Sparse Meta-Prior (`DebiasedSpectralMetaPrior`).

Implements Theorem 1 of the Path-B Operations Research reconstruction:
1. Inverse-occupancy kernel density debiasing on the `psi(x)` trajectory manifold to
   correct adaptive/survivor bias in legacy solver logs (`SPSA`, `CMA-ES`, `NSGA-II`,
   `Heuristic-Powell`).
2. Graph-Laplacian Rayleigh-quotient low-pass filtering to suppress high-frequency
   noise artifacts before sparse selection.
3. Weighted Gram-matrix whitening orthogonalization (`<phi_j, phi_k>_{nu_w} = delta_{jk}`).
4. Cross-task invariance sparse coordinate selection (`r << d`), routing all pruned
   spectral/functional energy into an explicit conservative residual envelope `R_perp(x)`.
5. Mean-shift-invariant Empirical-Bayes HVD variance hyperprior fitted exclusively on
   within-policy multi-replication dispersion `ssq / (n_reps - 1)`.
6. Dimension-invariant Discrete Cosine Transform (DCT) elite policy templates enabling
   cross-horizon candidate warm-starting (`d_s != d_t`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.fft import dct, idct
from scipy.optimize import nnls

from v2_or.contracts import LegacyTrajectoryRecord, NominalTrajectoryExposure


def _build_raw_dictionary(psi_std: np.ndarray) -> np.ndarray:
    """Expand standardized `psi` `(N, p)` into low-order smooth functional candidates."""
    z = np.clip(np.asarray(psi_std, dtype=float), -4.0, 4.0)
    n, p = z.shape
    cols = [z]
    # Smooth bounded low-frequency nonlinearities (Legendre P2 + half-period trig)
    cols.append((z ** 2 - 1.0) / np.sqrt(2.0))
    cols.append(np.sin(0.5 * np.pi * np.clip(z / 2.5, -1.0, 1.0)))
    # Key cross-interactions between policy action exposure (first 4 coords) and
    # risk/regime exposure (last 5 coords)
    left_idx = list(range(min(4, p)))
    right_idx = list(range(max(0, p - 4), p))
    inter_cols = []
    for i in left_idx:
        for j in right_idx:
            if i != j:
                inter_cols.append((z[:, i] * z[:, j])[:, None])
    if inter_cols:
        cols.extend(inter_cols)
    out = np.hstack(cols)
    assert out.shape[0] == n
    return out


def hvd_nonneg_feature_vector(A_local: np.ndarray, N_shared: np.ndarray) -> np.ndarray:
    """Nonnegative HVD variance basis `[1, A^2, N_i^2, (N_i + N_j)^2 / 2, N]`."""
    a = np.asarray(A_local, dtype=float).reshape(-1)
    n = np.maximum(np.asarray(N_shared, dtype=float).reshape(-1), 0.0)
    pair_quads = []
    for i in range(n.size):
        pair_quads.append(n[i] ** 2)
        for j in range(i + 1, n.size):
            pair_quads.append(0.5 * ((n[i] + n[j]) ** 2))
    return np.concatenate([
        np.array([1.0], dtype=float),
        a ** 2,
        np.asarray(pair_quads, dtype=float),
        n,
    ])


def policy_to_dct_profile(x_norm: np.ndarray, k_modes: int = 12) -> np.ndarray:
    """Compress `x in [0, 1]^d` via 2-channel (even/odd interleaved) DCT-II modes.

    Handles both single-channel temporal curves (smooth even/odd hours) and
    two-channel operational policies (e.g., regular vs. expedited order, staffing
    vs. surge threshold, or arterial vs. cross-street signal phases) across arbitrary `d`.
    """
    arr = np.asarray(x_norm, dtype=float).reshape(-1)
    k_half = max(int(k_modes) // 2, 1)
    ch0 = arr[0::2] - 0.5
    ch1 = arr[1::2] - 0.5 if arr.size > 1 else ch0

    c0 = dct(ch0, type=2, norm="ortho") / np.sqrt(max(ch0.size, 1))
    c1 = dct(ch1, type=2, norm="ortho") / np.sqrt(max(ch1.size, 1))

    out0 = np.zeros(k_half, dtype=float)
    out1 = np.zeros(int(k_modes) - k_half, dtype=float)
    out0[: min(k_half, c0.size)] = c0[: min(k_half, c0.size)]
    out1[: min(out1.size, c1.size)] = c1[: min(out1.size, c1.size)]
    return np.concatenate([out0, out1])


def dct_profile_to_policy(profile: np.ndarray, target_d: int) -> np.ndarray:
    """Reconstruct a `[0, 1]^{target_d}` policy from 2-channel DCT modes."""
    d = int(target_d)
    prof = np.asarray(profile, dtype=float).reshape(-1)
    k_half = max(prof.size // 2, 1)
    p0, p1 = prof[:k_half], prof[k_half:]

    n0 = (d + 1) // 2
    n1 = d // 2
    pad0 = np.zeros(max(n0, 1), dtype=float)
    t0 = min(p0.size, pad0.size)
    pad0[:t0] = p0[:t0] * np.sqrt(max(n0, 1))
    rec0 = 0.5 + idct(pad0, type=2, norm="ortho")

    out = np.empty(d, dtype=float)
    out[0::2] = rec0[:n0]
    if n1 > 0:
        pad1 = np.zeros(n1, dtype=float)
        t1 = min(p1.size, pad1.size)
        pad1[:t1] = p1[:t1] * np.sqrt(max(n1, 1))
        rec1 = 0.5 + idct(pad1, type=2, norm="ortho")
        out[1::2] = rec1[:n1]
    return np.clip(out, 0.0, 1.0)


def sample_smooth_dct_delta(
    rng: np.random.Generator,
    target_d: int,
    radius: float,
    k_modes: int = 12,
) -> np.ndarray:
    """Sample a zero-centered 2-channel smooth DCT perturbation preserving parent profiles."""
    d = int(target_d)
    k_half = max(int(k_modes) // 2, 1)
    decay0 = float(radius) / np.sqrt(np.arange(1, k_half + 1, dtype=float))
    decay1 = float(radius) / np.sqrt(np.arange(1, int(k_modes) - k_half + 1, dtype=float))
    p0 = rng.normal(0.0, decay0)
    p1 = rng.normal(0.0, decay1)

    n0 = (d + 1) // 2
    n1 = d // 2
    pad0 = np.zeros(max(n0, 1), dtype=float)
    t0 = min(p0.size, pad0.size)
    pad0[:t0] = p0[:t0] * np.sqrt(max(n0, 1))
    delta = np.empty(d, dtype=float)
    delta[0::2] = idct(pad0, type=2, norm="ortho")[:n0]
    if n1 > 0:
        pad1 = np.zeros(n1, dtype=float)
        t1 = min(p1.size, pad1.size)
        pad1[:t1] = p1[:t1] * np.sqrt(max(n1, 1))
        delta[1::2] = idct(pad1, type=2, norm="ortho")[:n1]
    return delta


@dataclass(frozen=True)
class PriorDiagnostics:
    """Quantitative diagnostics verifying Theorem 1 mechanisms on the fitted archive."""

    n_source_tasks: int
    n_archive_points: int
    raw_gram_cond: float
    debiased_orth_gram_cond: float
    dictionary_size: int
    lowpass_kept_count: int
    active_rank: int
    mean_lowfreq_energy_ratio: float
    effective_sample_size_ratio: float


class DebiasedSpectralMetaPrior:
    """Learned low-frequency orthogonal sparse functional & HVD meta-prior."""

    def __init__(
        self,
        *,
        active_rank: int = 8,
        max_weight_cap: float = 6.0,
        lowpass_roughness_quantile: float = 0.75,
        enable_debiasing: bool = True,
        enable_lowpass: bool = True,
        enable_orthogonalization: bool = True,
        enable_sparse_pruning: bool = True,
        dct_modes: int = 12,
    ) -> None:
        self.active_rank = int(active_rank)
        self.max_weight_cap = float(max_weight_cap)
        self.lowpass_roughness_quantile = float(lowpass_roughness_quantile)
        self.enable_debiasing = bool(enable_debiasing)
        self.enable_lowpass = bool(enable_lowpass)
        self.enable_orthogonalization = bool(enable_orthogonalization)
        self.enable_sparse_pruning = bool(enable_sparse_pruning)
        self.dct_modes = int(dct_modes)

        self.is_fitted: bool = False
        self.psi_mean_: np.ndarray = np.empty(0)
        self.psi_std_: np.ndarray = np.empty(0)
        self.lowpass_mask_: np.ndarray = np.empty(0, dtype=bool)
        self.dict_center_: np.ndarray = np.empty(0)
        self.whitening_mat_: np.ndarray = np.empty((0, 0))
        self.active_indices_: np.ndarray = np.empty(0, dtype=int)
        self.pruned_indices_: np.ndarray = np.empty(0, dtype=int)

        # Functional Empirical-Bayes hyperpriors over [intercept, active_phi_1..r]
        self.mu_0_obj_: np.ndarray = np.empty(0)
        self.cov_0_obj_: np.ndarray = np.empty((0, 0))
        self.mu_0_con_: np.ndarray = np.empty(0)
        self.cov_0_con_: np.ndarray = np.empty((0, 0))

        # Pruned residual energy envelopes
        self.pruned_energy_obj_: np.ndarray = np.empty(0)
        self.pruned_energy_con_: np.ndarray = np.empty(0)
        self.highfreq_floor_obj_: float = 1e-3
        self.highfreq_floor_con_: float = 1e-3

        # Nonnegative HVD variance hyperpriors
        self.gamma_0_obj_: np.ndarray = np.empty(0)
        self.gamma_0_con_: np.ndarray = np.empty(0)
        self.hvd_residual_floor_con_: float = 1e-4

        # DCT elite profile bank for cross-horizon candidate warm-starting
        self.elite_dct_profiles_: np.ndarray = np.empty((0, 0))
        self.diagnostics_: PriorDiagnostics | None = None

    def fit(self, records: Sequence[LegacyTrajectoryRecord]) -> "DebiasedSpectralMetaPrior":
        if len(records) < 12:
            raise ValueError("Need at least 12 legacy trajectory records to fit meta-prior")

        psi_mat = np.array([r.psi for r in records], dtype=float)
        n_pts, _ = psi_mat.shape

        # Standardize psi robustly
        self.psi_mean_ = np.mean(psi_mat, axis=0)
        self.psi_std_ = np.maximum(np.std(psi_mat, axis=0, ddof=1), 1e-4)
        psi_std = (psi_mat - self.psi_mean_) / self.psi_std_

        # 1. Inverse-Occupancy Kernel Density Debiasing on psi-manifold
        pairwise_sq = np.sum((psi_std[:, None, :] - psi_std[None, :, :]) ** 2, axis=2)
        positive_dists = pairwise_sq[pairwise_sq > 1e-10]
        med_sq = float(np.median(positive_dists)) if positive_dists.size > 0 else float(psi_std.shape[1])
        bandwidth_sq = max(med_sq, 0.25)
        affinity = np.exp(-0.5 * pairwise_sq / bandwidth_sq)

        if self.enable_debiasing:
            occ_density = np.mean(affinity, axis=1)
            inv_occ = 1.0 / np.maximum(occ_density, 1e-6)
            inv_occ = inv_occ / float(np.median(inv_occ))
            weights = np.minimum(inv_occ, self.max_weight_cap)
            weights = weights / float(np.sum(weights))
        else:
            weights = np.full(n_pts, 1.0 / n_pts, dtype=float)

        ess_ratio = float((1.0 / np.sum(weights ** 2)) / n_pts)

        # 2. Candidate Dictionary & Graph-Laplacian Spectral Low-Pass Filtering
        raw_dict = _build_raw_dictionary(psi_std)
        m_dict = raw_dict.shape[1]

        raw_centered = raw_dict - np.mean(raw_dict, axis=0, keepdims=True)
        raw_gram = (raw_centered.T @ raw_centered) / float(n_pts)
        eig_raw = np.linalg.eigvalsh(raw_gram)
        raw_cond = float(np.max(eig_raw) / max(float(np.min(np.abs(eig_raw))), 1e-12))

        deg = np.sum(affinity, axis=1)
        inv_sqrt_deg = 1.0 / np.sqrt(np.maximum(deg, 1e-8))
        norm_aff = (affinity * inv_sqrt_deg[:, None]) * inv_sqrt_deg[None, :]
        laplacian = np.eye(n_pts, dtype=float) - norm_aff

        roughness = np.empty(m_dict, dtype=float)
        for j in range(m_dict):
            col = raw_centered[:, j]
            denom = float(col @ col) + 1e-10
            roughness[j] = float(col @ (laplacian @ col)) / denom

        if self.enable_lowpass and m_dict > self.active_rank + 2:
            cutoff = float(np.quantile(roughness, self.lowpass_roughness_quantile))
            keep_mask = roughness <= cutoff
            if int(np.sum(keep_mask)) < max(self.active_rank, 4):
                order = np.argsort(roughness)
                keep_mask = np.zeros(m_dict, dtype=bool)
                keep_mask[order[: max(self.active_rank + 2, 4)]] = True
        else:
            keep_mask = np.ones(m_dict, dtype=bool)

        self.lowpass_mask_ = keep_mask
        mean_lowfreq_ratio = float(np.mean(1.0 - np.clip(roughness[keep_mask] / 2.0, 0.0, 1.0)))

        dict_lp = raw_dict[:, keep_mask]
        self.dict_center_ = np.sum(dict_lp * weights[:, None], axis=0)
        dict_lp_c = dict_lp - self.dict_center_[None, :]

        # 3. Weighted Gram Truncated-Spectral Whitening Orthogonalization
        gram_w = (dict_lp_c.T * weights[None, :]) @ dict_lp_c
        gram_w = 0.5 * (gram_w + gram_w.T)
        if self.enable_orthogonalization:
            evals, evecs = np.linalg.eigh(gram_w)
            pos_tol = max(float(np.max(evals)) * 1e-5, 1e-8)
            pos_mask = evals > pos_tol
            if int(np.sum(pos_mask)) < 2:
                pos_mask[-min(2, evals.size) :] = True
            evals_pos = np.maximum(evals[pos_mask], pos_tol)
            evecs_pos = evecs[:, pos_mask]
            # Shape (m_pos, m_lp): projects onto strictly positive-variance orthonormal subspace
            self.whitening_mat_ = (evecs_pos * (1.0 / np.sqrt(evals_pos))[None, :]).T
        else:
            diag_std = np.sqrt(np.maximum(np.diag(gram_w), 1e-6))
            self.whitening_mat_ = np.diag(1.0 / diag_std)

        orth_all = dict_lp_c @ self.whitening_mat_.T  # (N, m_pos)
        orth_gram = (orth_all.T * weights[None, :]) @ orth_all
        eig_orth = np.linalg.eigvalsh(0.5 * (orth_gram + orth_gram.T))
        orth_cond = float(np.max(eig_orth) / max(float(np.min(np.abs(eig_orth))), 1e-12))

        # 4. Per-Task GLS Fitting & Cross-Task Invariance Sparse Selection
        m_lp = orth_all.shape[1]
        task_ids = sorted({r.task_id for r in records})
        task_arr = np.array([r.task_id for r in records])
        y_obj = np.array([r.mean_obj for r in records], dtype=float)
        y_con = np.array([r.mean_con for r in records], dtype=float)
        n_reps = np.array([r.n_reps for r in records], dtype=float)

        theta_obj_tasks = []
        theta_con_tasks = []
        res_var_obj = []
        res_var_con = []

        full_design = np.column_stack([np.ones(n_pts, dtype=float), orth_all])
        reg_eye = np.diag(np.r_[1e-3, np.full(m_lp, 0.08)])

        for tid in task_ids:
            mask = task_arr == tid
            if int(np.sum(mask)) < 4:
                continue
            X_t = full_design[mask]
            w_t = weights[mask] * n_reps[mask]
            w_t = w_t / max(float(np.mean(w_t)), 1e-8)
            lhs = (X_t.T * w_t[None, :]) @ X_t + reg_eye
            th_f = np.linalg.solve(lhs, (X_t.T * w_t[None, :]) @ y_obj[mask])
            th_g = np.linalg.solve(lhs, (X_t.T * w_t[None, :]) @ y_con[mask])
            theta_obj_tasks.append(th_f)
            theta_con_tasks.append(th_g)
            res_var_obj.append(float(np.mean((y_obj[mask] - X_t @ th_f) ** 2)))
            res_var_con.append(float(np.mean((y_con[mask] - X_t @ th_g) ** 2)))

        theta_obj_mat = np.asarray(theta_obj_tasks, dtype=float)  # (M, 1 + m_lp)
        theta_con_mat = np.asarray(theta_con_tasks, dtype=float)

        if self.enable_sparse_pruning and m_lp > self.active_rank:
            coef_f = theta_obj_mat[:, 1:]
            coef_g = theta_con_mat[:, 1:]
            mean_f, var_f = np.mean(coef_f, axis=0), np.var(coef_f, axis=0) + 1e-4
            mean_g, var_g = np.mean(coef_g, axis=0), np.var(coef_g, axis=0) + 1e-4
            # Invariance stability ratio + squared effect energy (prioritizing constraint safety boundary)
            invariance_score = (
                (mean_f ** 2 / var_f) + 0.5 * np.mean(coef_f ** 2, axis=0)
                + 1.5 * ((mean_g ** 2 / var_g) + 0.5 * np.mean(coef_g ** 2, axis=0))
            )
            order = np.argsort(-invariance_score)
            r_keep = min(self.active_rank, m_lp)
            self.active_indices_ = np.sort(order[:r_keep])
            self.pruned_indices_ = np.sort(order[r_keep:])
        else:
            self.active_indices_ = np.arange(m_lp, dtype=int)
            self.pruned_indices_ = np.empty(0, dtype=int)

        keep_cols = np.r_[0, 1 + self.active_indices_]
        th_f_keep = theta_obj_mat[:, keep_cols]
        th_g_keep = theta_con_mat[:, keep_cols]

        self.mu_0_obj_ = np.mean(th_f_keep, axis=0)
        self.mu_0_con_ = np.mean(th_g_keep, axis=0)

        if len(theta_obj_mat) >= 2:
            cov_f = np.cov(th_f_keep, rowvar=False, ddof=1)
            cov_g = np.cov(th_g_keep, rowvar=False, ddof=1)
        else:
            cov_f = np.eye(keep_cols.size) * 0.1
            cov_g = np.eye(keep_cols.size) * 0.1

        cov_f = np.atleast_2d(cov_f)
        cov_g = np.atleast_2d(cov_g)
        # Shrinkage stabilization so Empirical-Bayes covariance is strictly PD
        shrink = 0.15
        diag_f = np.diag(np.maximum(np.diag(cov_f), 1e-3))
        diag_g = np.diag(np.maximum(np.diag(cov_g), 1e-3))
        self.cov_0_obj_ = (1.0 - shrink) * cov_f + shrink * diag_f + 1e-4 * np.eye(keep_cols.size)
        self.cov_0_con_ = (1.0 - shrink) * cov_g + shrink * diag_g + 1e-4 * np.eye(keep_cols.size)

        if self.pruned_indices_.size > 0:
            pruned_cols = 1 + self.pruned_indices_
            self.pruned_energy_obj_ = np.quantile(theta_obj_mat[:, pruned_cols] ** 2, 0.80, axis=0)
            self.pruned_energy_con_ = np.quantile(theta_con_mat[:, pruned_cols] ** 2, 0.85, axis=0)
        else:
            self.pruned_energy_obj_ = np.empty(0, dtype=float)
            self.pruned_energy_con_ = np.empty(0, dtype=float)

        self.highfreq_floor_obj_ = max(float(np.median(res_var_obj)) * 0.25, 1e-5)
        self.highfreq_floor_con_ = max(float(np.median(res_var_con)) * 0.25, 1e-5)

        # 5. Mean-Shift-Invariant Nonnegative HVD Variance Hyperprior
        rep_mask = np.array([r.n_reps > 1 for r in records], dtype=bool)
        hvd_feats = np.array(
            [hvd_nonneg_feature_vector(np.asarray(r.A_local), np.asarray(r.N_shared)) for r in records],
            dtype=float,
        )
        if int(np.sum(rep_mask)) >= 6:
            U_rep = hvd_feats[rep_mask]
            dof = np.array([r.n_reps - 1 for r in records], dtype=float)[rep_mask]
            s2_obj = np.array([r.ssq_obj for r in records], dtype=float)[rep_mask] / dof
            s2_con = np.array([r.ssq_con for r in records], dtype=float)[rep_mask] / dof
            w_sqrt = np.sqrt(weights[rep_mask] * dof)
            w_sqrt = w_sqrt / max(float(np.mean(w_sqrt)), 1e-8)

            reg_block = 0.05 * np.eye(U_rep.shape[1])
            aug_U = np.vstack([U_rep * w_sqrt[:, None], reg_block])
            gamma_f, _ = nnls(aug_U, np.r_[s2_obj * w_sqrt, np.zeros(U_rep.shape[1])])
            gamma_g, _ = nnls(aug_U, np.r_[s2_con * w_sqrt, np.zeros(U_rep.shape[1])])
            gamma_f[0] = max(gamma_f[0], 1e-5)
            gamma_g[0] = max(gamma_g[0], 1e-5)
            pred_con = U_rep @ gamma_g
            pos_resid = np.maximum(s2_con - pred_con, 0.0)
            self.hvd_residual_floor_con_ = max(float(np.quantile(pos_resid, 0.75)) * 0.25, 1e-5)
        else:
            p_hvd = hvd_feats.shape[1]
            gamma_f = np.r_[1e-3, np.zeros(p_hvd - 1)]
            gamma_g = np.r_[1e-3, np.zeros(p_hvd - 1)]
            self.hvd_residual_floor_con_ = 1e-4

        self.gamma_0_obj_ = gamma_f
        self.gamma_0_con_ = gamma_g

        # 6. Extract DCT-Compressed Elite Source Templates per Source Task
        elite_profiles = []
        for tid in task_ids:
            t_recs = [r for r in records if r.task_id == tid]
            # Rank by empirical margin feasibility then objective
            t_recs.sort(
                key=lambda r: (
                    0 if (r.mean_con + 1.645 * np.sqrt(max(r.ssq_con / max(r.n_reps - 1, 1), 1e-8))) <= 0.0 else 1,
                    r.mean_con if (r.mean_con + 1.645 * np.sqrt(max(r.ssq_con / max(r.n_reps - 1, 1), 1e-8))) > 0.0 else r.mean_obj,
                )
            )
            for top_r in t_recs[:2]:
                elite_profiles.append(policy_to_dct_profile(np.asarray(top_r.x_norm), self.dct_modes))
        self.elite_dct_profiles_ = np.asarray(elite_profiles, dtype=float)

        self.diagnostics_ = PriorDiagnostics(
            n_source_tasks=len(task_ids),
            n_archive_points=n_pts,
            raw_gram_cond=raw_cond,
            debiased_orth_gram_cond=orth_cond,
            dictionary_size=m_dict,
            lowpass_kept_count=int(np.sum(keep_mask)),
            active_rank=int(self.active_indices_.size),
            mean_lowfreq_energy_ratio=mean_lowfreq_ratio,
            effective_sample_size_ratio=ess_ratio,
        )
        self.is_fitted = True
        return self

    def transform_exposures(
        self,
        exposures: Sequence[NominalTrajectoryExposure],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Map exposures into `(Phi_active, psi_std, U_hvd, R_perp_obj, R_perp_con)`."""
        if not self.is_fitted:
            raise RuntimeError("DebiasedSpectralMetaPrior must be fitted before transform")
        psi_mat = np.array([e.psi for e in exposures], dtype=float)
        psi_std = (psi_mat - self.psi_mean_[None, :]) / self.psi_std_[None, :]
        raw_dict = _build_raw_dictionary(psi_std)
        dict_lp_c = raw_dict[:, self.lowpass_mask_] - self.dict_center_[None, :]
        orth_all = np.clip(dict_lp_c @ self.whitening_mat_.T, -5.0, 5.0)

        phi_active = np.column_stack([
            np.ones(len(exposures), dtype=float),
            orth_all[:, self.active_indices_],
        ])
        if self.pruned_indices_.size > 0:
            orth_pruned_sq = orth_all[:, self.pruned_indices_] ** 2
            r_perp_obj = orth_pruned_sq @ self.pruned_energy_obj_ + self.highfreq_floor_obj_
            r_perp_con = orth_pruned_sq @ self.pruned_energy_con_ + self.highfreq_floor_con_
        else:
            r_perp_obj = np.full(len(exposures), self.highfreq_floor_obj_, dtype=float)
            r_perp_con = np.full(len(exposures), self.highfreq_floor_con_, dtype=float)

        u_hvd = np.array(
            [hvd_nonneg_feature_vector(e.A_local, e.N_shared) for e in exposures],
            dtype=float,
        )
        return phi_active, psi_std, u_hvd, r_perp_obj, r_perp_con

    def warmstart_candidates(
        self,
        target_d: int,
        n_candidates: int,
        rng: np.random.Generator,
        anchor: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        """Generate dimension-adapted initial candidates via DCT source templates + anchor modulations."""
        cands: list[np.ndarray] = []
        anc = np.clip(np.asarray(anchor, dtype=float).reshape(-1), 0.0, 1.0) if anchor is not None else None
        if anc is not None:
            cands.append(anc)
        if self.elite_dct_profiles_.size > 0:
            n_from_src = min(max((n_candidates * 2) // 3, 2), len(self.elite_dct_profiles_))
            chosen = rng.choice(len(self.elite_dct_profiles_), size=n_from_src, replace=False)
            for k_i, idx in enumerate(chosen):
                base = dct_profile_to_policy(self.elite_dct_profiles_[idx], target_d)
                if anc is not None and k_i % 2 == 0:
                    # Preserve target-specific public forecast shape while shifting channel levels/tilts to source elite
                    shift = base - 0.50
                    cands.append(np.clip(0.55 * anc + 0.45 * base + 0.25 * shift, 0.0, 1.0))
                else:
                    cands.append(base)
        while len(cands) < n_candidates:
            if anc is not None and len(cands) % 2 == 0:
                delta = sample_smooth_dct_delta(rng, target_d, radius=0.14, k_modes=self.dct_modes)
                cands.append(np.clip(anc + delta, 0.0, 1.0))
            else:
                prof = rng.normal(0.0, 0.22 / np.arange(1, self.dct_modes + 1), size=self.dct_modes)
                cands.append(dct_profile_to_policy(prof, target_d))
        return cands[:n_candidates]
