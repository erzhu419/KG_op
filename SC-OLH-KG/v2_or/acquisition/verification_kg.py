"""Acquisition Strategies, Controlled Proposal Pools & Modular Sequential Search Engine.

Supports both one-step lookahead and Thompson/cEI acquisitions under a unified,
strictly modular interface so every source-information channel (`init_mode`,
`use_source_basis`, `use_source_coef_prior`, `use_source_hvd_prior`, `proposal_mode`,
`acquisition_type`, `nomination_mode`) can be toggled independently for clean causal
attribution:
- `"gate_power_ts"`: Joint-posterior Cholesky Thompson Sampling maximizing drawn
  expected certified cost saving `(J_sq - f_draw)_+ * C_{K_0}(Phi(g_draw / sigma_g))`
  using exact terminal binomial gate power `C_{K_0}(p)`.
- `"gate_power_vkg"`: Rank-one fantasy lookahead for a binomial-gate surrogate.
  The finite fantasy rule conserves each current gate power before computing
  the gain in the maximum across evaluation targets.
- `"constrained_ts"`: Joint-posterior Cholesky Constrained Thompson Sampling.
- `"v_kg_legacy"` / `"v_kg"`: Preserved Gaussian-margin approximate lookahead for
  controlled ablation against `"gate_power_vkg"` and `"gate_power_ts"`.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Sequence

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import ndtr, ndtri
from scipy.stats import norm, qmc

from v2_or.contracts import (
    BaseDynamicORProblem,
    NominalTrajectoryExposure,
    SufficientStatsRecord,
)
from v2_or.models.tempered_hvd_posterior import (
    JointPosteriorPrediction,
    TemperedHierarchicalHVDModel,
    _safe_cholesky,
)
from v2_or.priors.debiased_spectral_prior import (
    DebiasedSpectralMetaPrior,
    dct_profile_to_policy,
    sample_smooth_dct_delta,
)
from v2_or.verification.gated_verifier_v2 import (
    TerminalDeploymentReceipt,
    build_feasible_aware_shortlist,
    exact_binomial_critical_k,
    exact_binomial_pass_power,
    posterior_binomial_gate_power,
    verify_and_deploy_shortlist,
)


def _gauss_hermite_2d(order: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return standard normal 2D quadrature `(zf, zg, weights)` of size `order^2`."""
    pts, wts = hermgauss(int(order))
    z_1d = pts * np.sqrt(2.0)
    w_1d = wts / np.sqrt(np.pi)
    zf, zg = np.meshgrid(z_1d, z_1d, indexing="ij")
    w2 = np.outer(w_1d, w_1d)
    return zf.reshape(-1), zg.reshape(-1), w2.reshape(-1)


def _gauss_hermite_1d(order: int = 7) -> tuple[np.ndarray, np.ndarray]:
    pts, wts = hermgauss(int(order))
    return pts * np.sqrt(2.0), wts / wts.sum()


_GH5_ZF, _GH5_ZG, _GH5_W = _gauss_hermite_2d(5)
_GH5_Z, _GH5_P = _gauss_hermite_1d(5)
_GH7_Z, _GH7_W = _gauss_hermite_1d(7)
_SLOT_SIZES = np.array([112, 128, 140], dtype=int)


def _build_antithetic_sobol_normals(
    dim: int,
    half_pow2: int,
    seed: int,
    *,
    unit_var: bool = True,
) -> np.ndarray:
    """Deterministic zero-mean antithetic scrambled Sobol normal table."""
    u = qmc.Sobol(d=int(dim), scramble=True, seed=int(seed)).random_base2(int(half_pow2))
    z = ndtri(np.clip(u, 1e-6, 1.0 - 1e-6))
    out = np.vstack([z, -z])
    if unit_var:
        sd = np.maximum(np.std(out, axis=0, ddof=0), 1e-12)
        out = out / sd[None, :]
    return out


_ECKG_OUTER_32 = _build_antithetic_sobol_normals(4, 3, 20261120, unit_var=True)
_ECKG_INNER_128 = _build_antithetic_sobol_normals(3, 6, 20261121, unit_var=False)


def score_verification_kg(
    pred: JointPosteriorPrediction,
    *,
    alpha: float = 0.05,
    batch_reps: int = 2,
    verif_budget_slot1: int = 112,
    max_eval_targets: int = 80,
    residual_ucb_weight: float = 0.04,
) -> np.ndarray:
    """Legacy Gaussian-margin approximate lookahead (`v_kg_legacy`, kept for P1 ablation)."""
    m_cand = pred.mean_obj.size
    z_a = float(norm.ppf(1.0 - float(alpha)))
    q_margin = pred.mean_con + z_a * np.sqrt(np.maximum(pred.ucb_aleatoric_var_con, 1e-8))
    verif_var = pred.aleatoric_var_con / float(max(verif_budget_slot1, 1))
    s_tot_g = np.sqrt(pred.std_con ** 2 + 0.25 * pred.r_perp_con + verif_var)
    pass_prob = ndtr(-q_margin / np.maximum(s_tot_g, 1e-6))

    safe_like = pass_prob >= 0.25
    if np.any(safe_like):
        j_ref = float(np.quantile(pred.mean_obj[safe_like], 0.80) + 0.25 * np.std(pred.mean_obj[safe_like]) + 0.05)
    else:
        j_ref = float(np.quantile(pred.mean_obj, 0.85) + 0.15)

    curr_utility = np.maximum(j_ref - pred.mean_obj, 0.0) * pass_prob
    boundary_score = pass_prob * (1.0 - pass_prob)
    rank_composite = -curr_utility - 0.15 * boundary_score
    eval_idx = np.argsort(rank_composite)[: min(int(max_eval_targets), m_cand)]
    u_curr_max = float(np.max(curr_utility[eval_idx]))

    denom_f = np.sqrt(np.maximum(np.diag(pred.cov_obj) + pred.aleatoric_var_obj / float(batch_reps), 1e-10))
    denom_g = np.sqrt(np.maximum(np.diag(pred.cov_con) + pred.aleatoric_var_con / float(batch_reps), 1e-10))

    tilde_f = pred.cov_obj[:, eval_idx] / denom_f[:, None]
    tilde_g = pred.cov_con[:, eval_idx] / denom_g[:, None]

    mu_f_eval = pred.mean_obj[eval_idx]
    q_g_eval = q_margin[eval_idx]
    s2_g_eval = pred.std_con[eval_idx] ** 2
    rem_var_eval = 0.25 * pred.r_perp_con[eval_idx] + verif_var[eval_idx]

    s_next_g = np.sqrt(np.maximum(s2_g_eval[None, :] - tilde_g ** 2, 1e-10) + rem_var_eval[None, :])
    mu_f_next = mu_f_eval[None, None, :] + tilde_f[:, None, :] * _GH5_ZF[None, :, None]
    q_g_next = q_g_eval[None, None, :] + tilde_g[:, None, :] * _GH5_ZG[None, :, None]
    p_pass_next = ndtr(-q_g_next / np.maximum(s_next_g[:, None, :], 1e-6))

    u_next = np.maximum(j_ref - mu_f_next, 0.0) * p_pass_next
    best_u_per_fantasy = np.max(u_next, axis=2)
    expected_best_u = best_u_per_fantasy @ _GH5_W
    v_kg = np.maximum(expected_best_u - u_curr_max, 0.0)

    boundary_weight = 0.25 + 4.0 * pass_prob * (1.0 - pass_prob)
    ucb_perp = boundary_weight * (pred.std_obj + pred.std_con + np.sqrt(pred.r_perp_con))
    return v_kg + float(residual_ucb_weight) * ucb_perp


def _align_fantasy_gate_power(raw_power: np.ndarray, current_power: np.ndarray) -> np.ndarray:
    """Align power at each query/target pair to its current discrete-rule mean.

    The input has shape (queries, constraint fantasy nodes, targets). When the
    fantasy mean is too high, scale the powers toward zero. When it is too low,
    scale the complements toward one. The denominators on the selected branch
    are positive, including when either mean is at an endpoint. The correction
    preserves the weak ordering of the fantasy powers and keeps them in [0, 1].
    """
    mean = np.einsum("qge,g->qe", raw_power, _GH5_P)
    target = current_power[None, :]
    lower_power = mean > target
    raise_power = mean < target
    keep_power = np.divide(target, mean, out=np.ones_like(mean), where=lower_power)
    keep_failure = np.divide(1.0 - target, 1.0 - mean, out=np.ones_like(mean), where=raise_power)
    return np.where(
        lower_power[:, None, :],
        keep_power[:, None, :] * raw_power,
        np.where(raise_power[:, None, :], 1.0 - keep_failure[:, None, :] * (1.0 - raw_power), raw_power),
    )


def score_gate_power_vkg(
    pred: JointPosteriorPrediction,
    *,
    alpha: float = 0.05,
    batch_reps: int = 2,
    verif_budget_slot0: int = 112,
    delta_slot0: float = 0.025,
    status_quo_obj: float | None = None,
    max_eval_targets: int = 64,
    common_target_screen: bool = False,
) -> np.ndarray:
    """Rank-one lookahead using a mean-aligned finite Gaussian fantasy rule.

    The binomial CDF is evaluated at a Gaussian response proxy. The discrete
    alignment removes tower drift between the current and fantasy quadrature;
    it is not an exact integration of the posterior or the terminal cascade.
    """
    k_crit = exact_binomial_critical_k(int(verif_budget_slot0), float(alpha), float(delta_slot0))
    if k_crit < 0:
        return np.zeros(pred.mean_obj.size, dtype=float)
    return _score_rank_one_kg(
        pred, alpha, batch_reps, status_quo_obj, max_eval_targets, common_target_screen,
        (int(verif_budget_slot0), float(delta_slot0), k_crit),
    )


def score_chance_feasibility_kg(
    pred: JointPosteriorPrediction,
    *,
    alpha: float = 0.05,
    batch_reps: int = 2,
    status_quo_obj: float | None = None,
    max_eval_targets: int = 64,
    common_target_screen: bool = True,
) -> np.ndarray:
    """Same KG lookahead with posterior business chance-feasibility as the weight."""
    return _score_rank_one_kg(pred, alpha, batch_reps, status_quo_obj, max_eval_targets, common_target_screen, None)


def _batch_eval_cascade_rosters(
    mu_f_sq: np.ndarray,
    r_mu_f: np.ndarray,
    r_cov_f: np.ndarray,
    r_av_f: np.ndarray,
    r_mu_g: np.ndarray,
    r_cov_g: np.ndarray,
    r_sig_g: np.ndarray,
    n_slots: np.ndarray,
    has_guard: np.ndarray,
    *,
    alpha: float = 0.05,
    familywise_delta: float = 0.05,
    fast_k1: bool = True,
) -> np.ndarray:
    """Vectorized correlated 3-slot guarded cascade net saving `E[J(x_sq) - l(S, w)]`."""
    b_cnt = int(mu_f_sq.size)
    if b_cnt == 0:
        return np.zeros(0, dtype=float)

    c0 = mu_f_sq - r_mu_f[:, 0]
    c2 = np.where(n_slots >= 2, mu_f_sq - r_mu_f[:, 1], 0.0)
    c3 = np.where(n_slots >= 3, mu_f_sq - r_mu_f[:, 2], 0.0)

    diff = r_mu_f[:, 0] - r_mu_f[:, 1]
    v_diff = np.maximum(r_cov_f[:, 0, 0] + r_cov_f[:, 1, 1] - 2.0 * r_cov_f[:, 0, 1], 0.0)
    s_obs = np.sqrt(np.maximum(v_diff + r_av_f[:, 0] / float(_SLOT_SIZES[0]) + r_av_f[:, 1] / float(_SLOT_SIZES[1]), 1e-16))
    z_d = diff / s_obs
    guard_corr = diff * ndtr(z_d) + (v_diff / s_obs) * (np.exp(-0.5 * z_d * z_d) / np.sqrt(2.0 * np.pi))
    c1 = np.where(has_guard & (n_slots >= 2), guard_corr, 0.0)

    cov_sym = 0.5 * (r_cov_g + np.swapaxes(r_cov_g, 1, 2)) + 1e-10 * np.eye(3, dtype=float)[None, :, :]
    try:
        L_g = np.linalg.cholesky(cov_sym)
    except np.linalg.LinAlgError:
        evals, evecs = np.linalg.eigh(cov_sym)
        cov_rep = np.einsum("bij,bj,bkj->bik", evecs, np.maximum(evals, 1e-9), evecs) + 1e-9 * np.eye(3)[None, :, :]
        L_g = np.linalg.cholesky(cov_rep)

    g_draws = r_mu_g[:, None, :] + np.einsum("mk,bjk->bmj", _ECKG_INNER_128, L_g)
    p_viol = ndtr(g_draws / np.maximum(r_sig_g[:, None, :], 1e-8))

    if fast_k1:
        nm1 = (_SLOT_SIZES - 1)[None, None, :]
        a = ((1.0 - p_viol) ** nm1) * (1.0 + nm1 * p_viol)
    else:
        a = np.zeros_like(p_viol)
        for s_len in (1, 2, 3):
            sel = np.where(n_slots == s_len)[0]
            if sel.size == 0:
                continue
            fracs = (1.0,) if s_len == 1 else ((0.625, 0.375) if s_len == 2 else (0.50, 0.30, 0.20))
            for j, frac in enumerate(fracs):
                kc = exact_binomial_critical_k(int(_SLOT_SIZES[j]), float(alpha), float(familywise_delta) * frac)
                a[sel, :, j] = exact_binomial_pass_power(p_viol[sel, :, j], int(_SLOT_SIZES[j]), kc)

    a[:, :, 1] = np.where((n_slots >= 2)[:, None], a[:, :, 1], 0.0)
    a[:, :, 2] = np.where((n_slots >= 3)[:, None], a[:, :, 2], 0.0)

    a0 = a[:, :, 0]
    a1 = a[:, :, 1]
    a2 = a[:, :, 2]
    m0 = np.mean(a0, axis=1)
    m1 = np.mean(a0 * a1, axis=1)
    m2 = np.mean((1.0 - a0) * a1, axis=1)
    m3 = np.mean((1.0 - a0) * (1.0 - a1) * a2, axis=1)
    return c0 * m0 + c1 * m1 + c2 * m2 + c3 * m3


def score_eligible_cascade_kg(
    pred: JointPosteriorPrediction,
    model: TemperedHierarchicalHVDModel,
    cands: Sequence[np.ndarray],
    exps: Sequence[NominalTrajectoryExposure],
    *,
    alpha: float = 0.05,
    batch_reps: int = 2,
    familywise_delta: float = 0.05,
    status_quo_obj: float | None = None,
    nomination_mode: str = "cascade_bellman_3slot",
    top_k_contenders: int = 16,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Two-stage correlation-preserving Eligible-Cascade KG (`EC-KG`) on `D_n U {q}`.

    Implements the exact dual VOI decomposition (`SequentialTheory.lean`):
        VOI_EC(q) = VOI_refine(q) + VOI_admit(q) >= 0,
    where `VOI_refine(q) >= 0` values epistemic refinement of already-evaluated
    rosters in `D_n` via the switch-gain tower identity on shared QMC draws, and
    `VOI_admit(q) >= 0` values admitting an unevaluated query `q` into terminal
    verification eligibility `D_n U {q}` (`VOI_admit(q) == 0` for revisits `q in D_n`).
    """
    t_start = time.perf_counter()
    m_cand = int(pred.mean_obj.size)
    if m_cand == 0:
        empty = np.zeros(0, dtype=float)
        return (empty, {}) if return_diagnostics else empty

    obs_recs = model.unique_records
    key_to_cand = {model._policy_key(x): idx for idx, x in enumerate(cands)}
    elig_base_idx: list[int] = []
    elig_base_recs: list[SufficientStatsRecord] = []
    for rec in obs_recs:
        k = model._policy_key(rec.x_norm)
        if k in key_to_cand:
            elig_base_idx.append(key_to_cand[k])
            elig_base_recs.append(rec)

    if not elig_base_idx:
        scores_fallback = score_gate_power_vkg(
            pred, alpha=alpha, batch_reps=batch_reps, status_quo_obj=status_quo_obj
        )
        return (scores_fallback, {}) if return_diagnostics else scores_fallback

    base_arr = np.asarray(elig_base_idx, dtype=int)
    m_obs = int(base_arr.size)
    base_pos_by_cand = {int(c_i): pos for pos, c_i in enumerate(elig_base_idx)}
    sq_cand_idx = int(base_arr[0])

    fast_k1 = all(
        exact_binomial_critical_k(int(n), float(alpha), float(familywise_delta) * frac) == 1
        for frac_tuple in ((1.0,), (0.625, 0.375), (0.50, 0.30, 0.20))
        for n, frac in zip(_SLOT_SIZES, frac_tuple)
    )

    av_f = np.maximum(pred.aleatoric_var_obj, 1e-10)
    av_g = np.maximum(pred.aleatoric_var_con, 1e-10)
    av_ucb_g = np.maximum(pred.ucb_aleatoric_var_con, 1e-8)
    sig_ucb_g = np.sqrt(av_ucb_g)
    rp_g = np.maximum(pred.r_perp_con, 0.0)
    cov_g_eff = pred.cov_con + np.diag(0.25 * rp_g)
    std_g_base = np.sqrt(np.maximum(np.diag(pred.cov_con), 0.0))

    obs_anchor_val = float(elig_base_recs[0].mean_obj)
    sq_ref = (
        float(status_quo_obj)
        if (status_quo_obj is not None and np.isfinite(status_quo_obj))
        else 0.5 * (obs_anchor_val + float(pred.mean_obj[sq_cand_idx]))
    )

    # Current eligible shortlist S_curr on D_n
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
        nomination_mode=nomination_mode,
        reference_obj=sq_ref,
        familywise_delta=familywise_delta,
    )
    base_round_map: dict[bytes, int] = {}
    for pos, x_b in enumerate(base_cands):
        rk = np.round(x_b, 5).tobytes()
        if rk not in base_round_map:
            base_round_map[rk] = int(elig_base_idx[pos])
    curr_roster_cands = tuple(base_round_map[np.round(x, 5).tobytes()] for _, x in curr_sl)
    curr_guard = len(curr_sl) >= 2 and curr_sl[1][0] in {"empirical_feasible_incumbent", "safe_initial_incumbent"}

    def _pack_single_roster(r_tup: tuple[int, ...], guard_flag: bool) -> tuple[np.ndarray, int, bool]:
        padded = np.array([r_tup[0], r_tup[1] if len(r_tup) >= 2 else r_tup[0], r_tup[2] if len(r_tup) >= 3 else r_tup[0]], dtype=int)
        return padded, len(r_tup), bool(guard_flag)

    curr_pad, curr_len, curr_g_flag = _pack_single_roster(curr_roster_cands, curr_guard)
    u_curr_raw = float(
        _batch_eval_cascade_rosters(
            np.array([pred.mean_obj[sq_cand_idx]]),
            pred.mean_obj[curr_pad][None, :],
            pred.cov_obj[np.ix_(curr_pad, curr_pad)][None, :, :],
            av_f[curr_pad][None, :],
            pred.mean_con[curr_pad][None, :],
            cov_g_eff[np.ix_(curr_pad, curr_pad)][None, :, :],
            sig_ucb_g[curr_pad][None, :],
            np.array([curr_len], dtype=int),
            np.array([curr_g_flag], dtype=bool),
            alpha=alpha,
            familywise_delta=familywise_delta,
            fast_k1=fast_k1,
        )[0]
    )
    curr_active = bool(u_curr_raw > 0.0)

    # Stage A: Fast vectorized eligible-continuation proxy across all M queries
    s_single = score_gate_power_vkg(
        pred, alpha=alpha, batch_reps=batch_reps, status_quo_obj=sq_ref
    )
    s_feas = score_chance_feasibility_kg(
        pred, alpha=alpha, batch_reps=batch_reps, status_quo_obj=sq_ref
    )

    p0_all = posterior_binomial_gate_power(
        pred.mean_con, pred.std_con, av_ucb_g,
        n_samples=int(_SLOT_SIZES[0]), alpha=float(alpha), delta_slot=0.50 * float(familywise_delta), r_perp_con=rp_g,
    )
    p2_all = posterior_binomial_gate_power(
        pred.mean_con, pred.std_con, av_ucb_g,
        n_samples=int(_SLOT_SIZES[2]), alpha=float(alpha), delta_slot=0.20 * float(familywise_delta), r_perp_con=rp_g,
    )
    j_rem_curr = float(pred.mean_obj[sq_cand_idx] - max(u_curr_raw, 0.0))

    denom_f = np.sqrt(np.maximum(np.diag(pred.cov_obj) + av_f / float(batch_reps), 1e-10))
    denom_g = np.sqrt(np.maximum(np.diag(pred.cov_con) + av_g / float(batch_reps), 1e-10))
    tilde_f_all = pred.cov_obj / denom_f[:, None]
    tilde_g_all = pred.cov_con / denom_g[:, None]

    # Evaluate Stage-A eligible gain: refinement on D_n + admission of q itself
    tf_base = tilde_f_all[:, base_arr]
    tg_base = tilde_g_all[:, base_arr]
    s2_g_all = np.maximum(np.diag(cov_g_eff), 1e-12)
    curr_elig_u = np.maximum(j_rem_curr - pred.mean_obj[base_arr], 0.0) * p0_all[base_arr]
    u_base_max = float(np.max(curr_elig_u))

    s_next_base = np.sqrt(np.maximum(s2_g_all[None, base_arr] - tg_base ** 2, 0.0))
    mu_g_base_next = pred.mean_con[None, None, base_arr] + tg_base[:, None, :] * _GH5_Z[None, :, None]
    k0_crit = max(exact_binomial_critical_k(int(_SLOT_SIZES[0]), float(alpha), 0.50 * float(familywise_delta)), 0)
    g_lat_b = mu_g_base_next[:, :, :, None] + s_next_base[:, None, :, None] * _GH7_Z[None, None, None, :]
    p_lat_b = ndtr(g_lat_b / sig_ucb_g[None, None, base_arr, None])
    raw_gate_b = np.clip(exact_binomial_pass_power(p_lat_b, int(_SLOT_SIZES[0]), k0_crit) @ _GH7_W, 0.0, 1.0)
    p_gate_b = _align_fantasy_gate_power(raw_gate_b, p0_all[base_arr])
    p_gate_b = np.where(tg_base[:, None, :] == 0.0, p0_all[None, None, base_arr], p_gate_b)

    imp_b = np.maximum((j_rem_curr - pred.mean_obj[base_arr])[None, None, :] - tf_base[:, None, :] * _GH5_Z[None, :, None], 0.0)
    u_b_next = imp_b[:, :, None, :] * p_gate_b[:, None, :, :]
    max_b_next = np.max(u_b_next, axis=-1)

    # Self-admission utility when q is queried (for new candidates q not in D_n)
    tf_self = np.diag(tilde_f_all)
    tg_self = np.diag(tilde_g_all)
    s_next_self = np.sqrt(np.maximum(s2_g_all - tg_self ** 2, 0.0))
    mu_g_self_next = pred.mean_con[:, None] + tg_self[:, None] * _GH5_Z[None, :]
    g_lat_s = mu_g_self_next[:, :, None] + s_next_self[:, None, None] * _GH7_Z[None, None, :]
    p_lat_s = ndtr(g_lat_s / sig_ucb_g[:, None, None])
    raw_gate_s = np.clip(exact_binomial_pass_power(p_lat_s, int(_SLOT_SIZES[0]), k0_crit) @ _GH7_W, 0.0, 1.0)
    p_gate_s = _align_fantasy_gate_power(np.swapaxes(raw_gate_s[:, :, None], 0, 2), p0_all)[0].T
    p_gate_s = np.where((tg_self == 0.0)[:, None], p0_all[:, None], p_gate_s)
    imp_s = np.maximum((j_rem_curr - pred.mean_obj)[:, None] - tf_self[:, None] * _GH5_Z[None, :], 0.0)
    u_s_next = imp_s[:, :, None] * p_gate_s[:, None, :]
    # Also value backup safety-net admission via p2_all when primary power is low
    backup_bonus = 0.08 * np.maximum(pred.mean_obj[sq_cand_idx] - pred.mean_obj, 0.0) * p2_all
    max_elig_next = np.maximum(max_b_next, u_s_next)
    stage_a_proxy = np.maximum(np.einsum("qfg,f,g->q", max_elig_next, _GH5_P, _GH5_P) - u_base_max, 0.0) + 0.02 * backup_bonus

    # Construct contender set K_n: current shortlist + single-slot/feas winners + top revisits + top new queries
    contender_set: list[int] = [int(c) for c in curr_roster_cands]
    contender_set.append(int(np.argmax(s_single)))
    contender_set.append(int(np.argmax(s_feas)))
    revisit_cands = [i for i in elig_base_idx]
    new_cands = [i for i in range(m_cand) if i not in base_pos_by_cand]
    if revisit_cands:
        rev_order = sorted(revisit_cands, key=lambda idx: -float(stage_a_proxy[idx]))
        contender_set.extend(rev_order[:6])
    if new_cands:
        new_order = sorted(new_cands, key=lambda idx: -float(stage_a_proxy[idx]))
        contender_set.extend(new_order[: max(int(top_k_contenders) - 8, 8)])
    # Deduplicate preserving priority up to top_k_contenders
    seen_q: set[int] = set()
    contenders: list[int] = []
    for q_idx in contender_set:
        if q_idx not in seen_q:
            seen_q.add(q_idx)
            contenders.append(q_idx)
        if len(contenders) >= max(int(top_k_contenders), 6):
            break

    # Stage B: Exact correlated 3-slot nomination & switch-gain integration over K_n x 32 Sobol fantasies
    n_outer = int(_ECKG_OUTER_32.shape[0])
    eval_sq_list: list[float] = []
    eval_mu_f: list[np.ndarray] = []
    eval_cov_f: list[np.ndarray] = []
    eval_av_f: list[np.ndarray] = []
    eval_mu_g: list[np.ndarray] = []
    eval_cov_g: list[np.ndarray] = []
    eval_sig_g: list[np.ndarray] = []
    eval_nslots: list[int] = []
    eval_guard: list[bool] = []
    # Map (q_pos_in_contenders, outer_s, role) -> index in batch (-1 means empty/abstention = 0.0)
    slot_ptr = np.full((len(contenders), n_outer, 3), -1, dtype=int)

    for q_pos, q in enumerate(contenders):
        is_revisit = q in base_pos_by_cand
        elig_idx_q = base_arr if is_revisit else np.append(base_arr, q)
        elig_cands_q = base_cands if is_revisit else [*base_cands, cands[q]]

        tf_q = tilde_f_all[q, elig_idx_q]
        tg_q = tilde_g_all[q, elig_idx_q]
        zero_innov = bool(np.max(np.abs(tf_q)) < 1e-14 and np.max(np.abs(tg_q)) < 1e-14 and is_revisit)
        if zero_innov:
            continue

        cov_f_post = pred.cov_obj[np.ix_(elig_idx_q, elig_idx_q)] - np.outer(tf_q, tf_q)
        cov_g_post_raw = pred.cov_con[np.ix_(elig_idx_q, elig_idx_q)] - np.outer(tg_q, tg_q)
        std_g_post = np.sqrt(np.maximum(np.diag(cov_g_post_raw), 0.0))
        cov_g_post_eff = cov_g_post_raw + np.diag(0.25 * rp_g[elig_idx_q])

        lat_v = np.array([max(float(pred.cov_obj[q, q]), 0.0), max(float(pred.cov_con[q, q]), 0.0)])
        obs_v = np.array([float(av_f[q]), float(av_g[q])])
        means_q = np.array([float(pred.mean_obj[q]), float(pred.mean_con[q])])
        centers = means_q[None, :] + _ECKG_OUTER_32[:, :2] * np.sqrt(lat_v + obs_v / float(batch_reps))[None, :]
        contrasts = _ECKG_OUTER_32[:, 2:] * np.sqrt(obs_v / 2.0)[None, :]

        # Map local index inside elig_idx_q back to global candidate index
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

        curr_local_pad = np.array([base_pos_by_cand[int(c)] for c in curr_pad], dtype=int)

        for s in range(n_outer):
            zf_s = float(_ECKG_OUTER_32[s, 0])
            zg_s = float(_ECKG_OUTER_32[s, 1])
            mf_s = pred.mean_obj[elig_idx_q] + zf_s * tf_q
            mg_s = pred.mean_con[elig_idx_q] + zg_s * tg_q
            mf_sq_s = float(mf_s[0])
            ref_s = 0.5 * (obs_anchor_val + mf_sq_s)

            if batch_reps == 2:
                samps_s = np.stack([centers[s] + contrasts[s], centers[s] - contrasts[s]], axis=0)
            else:
                samps_s = np.repeat(centers[s : s + 1], int(batch_reps), axis=0)
                samps_s[0] += contrasts[s]
                if batch_reps > 1:
                    samps_s[1] -= contrasts[s]
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
                nomination_mode=nomination_mode,
                reference_obj=ref_s,
                familywise_delta=familywise_delta,
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
                    nomination_mode=nomination_mode,
                    reference_obj=ref_s,
                    familywise_delta=familywise_delta,
                )
                loc_full = tuple(round_to_local_full[np.round(x, 5).tobytes()] for _, x in sl_full)
                grd_full = len(sl_full) >= 2 and sl_full[1][0] in {"empirical_feasible_incumbent", "safe_initial_incumbent"}

            local_cache: dict[tuple[tuple[int, ...], bool], int] = {}

            def _register_local(r_loc: tuple[int, ...], g_fl: bool) -> int:
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

            # Role 0: S_curr* (if curr_active else abstention -> -1)
            if curr_active:
                curr_loc_tup = tuple(int(v) for v in curr_local_pad[:curr_len])
                slot_ptr[q_pos, s, 0] = _register_local(curr_loc_tup, curr_g_flag)
            # Role 1: post-query D_n nomination
            slot_ptr[q_pos, s, 1] = _register_local(loc_d, grd_d)
            # Role 2: post-query D_n U {q} nomination
            slot_ptr[q_pos, s, 2] = slot_ptr[q_pos, s, 1] if is_revisit else _register_local(loc_full, grd_full)

    if eval_sq_list:
        u_batch = _batch_eval_cascade_rosters(
            np.asarray(eval_sq_list, dtype=float),
            np.asarray(eval_mu_f, dtype=float),
            np.asarray(eval_cov_f, dtype=float),
            np.asarray(eval_av_f, dtype=float),
            np.asarray(eval_mu_g, dtype=float),
            np.asarray(eval_cov_g, dtype=float),
            np.asarray(eval_sig_g, dtype=float),
            np.asarray(eval_nslots, dtype=int),
            np.asarray(eval_guard, dtype=bool),
            alpha=alpha,
            familywise_delta=familywise_delta,
            fast_k1=fast_k1,
        )
    else:
        u_batch = np.zeros(0, dtype=float)

    voi_refine = np.zeros(m_cand, dtype=float)
    voi_admit = np.zeros(m_cand, dtype=float)
    voi_total = np.zeros(m_cand, dtype=float)

    # Give non-contenders a tiny tie-breaking fraction of stage_a_proxy strictly below contenders if all are 0
    for q_pos, q in enumerate(contenders):
        ptrs = slot_ptr[q_pos]
        if np.all(ptrs == -1):
            continue
        u_ref_star = np.where(ptrs[:, 0] >= 0, u_batch[ptrs[:, 0]], 0.0)
        u_d = np.where(ptrs[:, 1] >= 0, u_batch[ptrs[:, 1]], 0.0)
        u_full = np.where(ptrs[:, 2] >= 0, u_batch[ptrs[:, 2]], 0.0)

        m0_max = np.maximum(0.0, np.maximum(u_ref_star, u_d))
        m1_max = np.maximum(m0_max, u_full)
        ref_gain = float(np.mean(np.maximum(m0_max - u_ref_star, 0.0)))
        adm_gain = float(np.mean(np.maximum(m1_max - m0_max, 0.0)))
        voi_refine[q] = ref_gain
        voi_admit[q] = adm_gain
        voi_total[q] = ref_gain + adm_gain

    is_rev_mask = np.zeros(m_cand, dtype=bool)
    is_rev_mask[base_arr] = True
    zero_rev_mask = is_rev_mask & (np.max(np.abs(tf_base), axis=1) < 1e-14) & (np.max(np.abs(tg_base), axis=1) < 1e-14)
    tie_break = np.where(zero_rev_mask, 0.0, 1e-9 * stage_a_proxy + 1e-11 * s_single)
    select_scores = voi_total if float(np.max(voi_total)) > 1e-14 else tie_break

    chosen_q = int(np.argmax(select_scores))
    single_q = int(np.argmax(s_single))
    feas_q = int(np.argmax(s_feas))
    step_ms = 1000.0 * (time.perf_counter() - t_start)

    if not return_diagnostics:
        return select_scores

    diag = {
        "chosen_query": chosen_q,
        "single_slot_query": single_q,
        "feasible_kg_query": feas_q,
        "disagreed_with_single_slot": bool(chosen_q != single_q),
        "disagreed_with_feasible_kg": bool(chosen_q != feas_q),
        "chosen_is_revisit": bool(chosen_q in base_pos_by_cand),
        "chosen_voi_total": float(voi_total[chosen_q]),
        "chosen_voi_refine": float(voi_refine[chosen_q]),
        "chosen_voi_admit": float(voi_admit[chosen_q]),
        "contenders_evaluated": len(contenders),
        "rosters_integrated": len(eval_sq_list),
        "step_wall_ms": float(step_ms),
    }
    return voi_total, diag


def _chance_feasibility_probability(margin: np.ndarray, sd: np.ndarray) -> np.ndarray:
    z = np.divide(margin, sd, out=np.zeros_like(margin), where=sd > 0.0)
    return np.where(sd > 0.0, ndtr(z), margin >= 0.0)


def _score_rank_one_kg(
    pred: JointPosteriorPrediction,
    alpha: float,
    batch_reps: int,
    status_quo_obj: float | None,
    max_eval_targets: int,
    common_target_screen: bool,
    gate: tuple[int, float, int] | None,
) -> np.ndarray:
    m_cand = pred.mean_obj.size
    if not m_cand:
        return np.zeros(m_cand, dtype=float)

    av_ucb = np.maximum(pred.ucb_aleatoric_var_con, 1e-8)
    s2_constraint = pred.std_con ** 2 + 0.25 * np.maximum(pred.r_perp_con, 0.0)
    if gate is None:
        feasibility_limit = norm.ppf(alpha) * np.sqrt(av_ucb)
        pass_power_curr = _chance_feasibility_probability(feasibility_limit - pred.mean_con, np.sqrt(s2_constraint))
    else:
        n_verif, delta_slot, k_crit = gate
        pass_power_curr = posterior_binomial_gate_power(
            pred.mean_con, pred.std_con, av_ucb,
            n_samples=n_verif, alpha=float(alpha), delta_slot=delta_slot, r_perp_con=pred.r_perp_con,
        )
    spread_f = max(float(np.std(pred.mean_obj)), 1e-8)
    if status_quo_obj is not None and np.isfinite(status_quo_obj):
        j_ref = max(float(status_quo_obj), float(np.min(pred.mean_obj)) + 0.20 * spread_f)
    else:
        j_ref = float(np.quantile(pred.mean_obj, 0.85)) + 0.25 * spread_f

    curr_utility = np.maximum(j_ref - pred.mean_obj, 0.0) * pass_power_curr
    boundary_info = pass_power_curr * (1.0 - pass_power_curr) * spread_f
    eval_count = min(int(max_eval_targets), m_cand)
    if common_target_screen:
        eval_idx = np.argsort(pred.mean_obj, kind="stable")[:eval_count]
    else:
        eval_idx = np.argsort(-curr_utility - 0.12 * boundary_info)[:eval_count]
    incumbent = int(np.argmax(curr_utility[eval_idx]))

    denom_f = np.sqrt(np.maximum(np.diag(pred.cov_obj) + pred.aleatoric_var_obj / float(batch_reps), 1e-10))
    denom_g = np.sqrt(np.maximum(np.diag(pred.cov_con) + pred.aleatoric_var_con / float(batch_reps), 1e-10))
    tilde_f = pred.cov_obj[:, eval_idx] / denom_f[:, None]
    tilde_g = pred.cov_con[:, eval_idx] / denom_g[:, None]

    mu_f_eval = pred.mean_obj[eval_idx]
    mu_g_eval = pred.mean_con[eval_idx]
    current_power = pass_power_curr[eval_idx]
    s2_g_eval = s2_constraint[eval_idx]
    sig_aleat_eval = np.sqrt(av_ucb[eval_idx])

    s_next_g = np.sqrt(np.maximum(s2_g_eval[None, :] - tilde_g ** 2, 0.0))
    # The gate depends only on the constraint fantasy node, not the objective
    # fantasy node. Integrate it once per pair with the 7-node conditional rule.
    mu_g_next = mu_g_eval[None, None, :] + tilde_g[:, None, :] * _GH5_Z[None, :, None]
    if gate is None:
        raw_gate = _chance_feasibility_probability(feasibility_limit[None, None, eval_idx] - mu_g_next, s_next_g[:, None, :])
    else:
        g_lat = mu_g_next[:, :, :, None] + s_next_g[:, None, :, None] * _GH7_Z[None, None, None, :]
        p_lat = ndtr(g_lat / sig_aleat_eval[None, None, :, None])
        c_lat = exact_binomial_pass_power(p_lat, n_verif, k_crit)
        raw_gate = np.clip(c_lat @ _GH7_W, 0.0, 1.0)
    p_gate_next = _align_fantasy_gate_power(raw_gate, current_power)
    p_gate_next = np.where(tilde_g[:, None, :] == 0.0, current_power[None, None, :], p_gate_next)

    # Sum the two nonnegative gains rather than subtract nearly equal
    # expectations. The current best target is always available as a reference
    # in the fantasy. Its own expected gain is just the hinge's Jensen gap,
    # times its current gate power, because the fantasy heads are factored.
    margin = j_ref - mu_f_eval
    f_innovation = tilde_f[:, None, :] * _GH5_Z[None, :, None]
    improvement = np.maximum(margin[None, None, :] - f_innovation, 0.0)
    u_next = improvement[:, :, None, :] * p_gate_next[:, None, :, :]
    switch_gain = np.max(u_next, axis=-1) - u_next[..., incumbent]
    hinge_gap = np.maximum(
        f_innovation[..., incumbent] - abs(margin[incumbent]), 0.0
    )
    return (
        np.einsum("qfg,f,g->q", switch_gain, _GH5_P, _GH5_P)
        + current_power[incumbent] * (hinge_gap @ _GH5_P)
    )


def score_gate_power_ts(
    pred: JointPosteriorPrediction,
    rng: np.random.Generator,
    *,
    alpha: float = 0.05,
    verif_budget_slot0: int = 112,
    delta_slot0: float = 0.025,
    status_quo_obj: float | None = None,
) -> int:
    """Joint-posterior Cholesky Thompson Sampling maximizing terminal gate-power expected net saving."""
    Lf = _safe_cholesky(pred.cov_obj)
    Lg = _safe_cholesky(pred.cov_con)
    f_draw = pred.mean_obj + Lf @ rng.normal(size=pred.mean_obj.size)
    g_draw = pred.mean_con + Lg @ rng.normal(size=pred.mean_con.size)

    sig_aleat = np.sqrt(np.maximum(pred.ucb_aleatoric_var_con + 0.25 * pred.r_perp_con, 1e-8))
    p_draw = ndtr(g_draw / sig_aleat)
    k_crit = max(exact_binomial_critical_k(int(verif_budget_slot0), float(alpha), float(delta_slot0)), 0)
    c_power = exact_binomial_pass_power(p_draw, int(verif_budget_slot0), k_crit)

    spread_f = max(float(np.std(f_draw)), 1e-8)
    eps_f = 1e-4 * spread_f
    if status_quo_obj is not None and np.isfinite(status_quo_obj):
        j_ref = max(float(status_quo_obj), float(np.min(f_draw)) + 0.20 * spread_f)
    else:
        j_ref = float(np.quantile(f_draw, 0.85)) + 0.25 * spread_f

    certifiable = np.where(c_power >= 0.50)[0]
    if certifiable.size > 0:
        pos_c = np.where(f_draw[certifiable] < j_ref - eps_f)[0]
        if pos_c.size > 0:
            sub_c = certifiable[pos_c]
            util = (j_ref - f_draw[sub_c]) * c_power[sub_c]
            return int(sub_c[int(np.argmax(util))])
        return int(certifiable[int(np.argmin(f_draw[certifiable]))])
    if float(np.max(c_power)) >= 0.05:
        j_hi = max(j_ref, float(np.quantile(f_draw, 0.85)) + 0.25 * spread_f)
        util = np.maximum(j_hi - f_draw, 0.0) * c_power
        if float(np.max(util)) > 0.0:
            return int(np.argmax(util))
        return int(np.argmax(c_power))
    return int(np.argmin(g_draw / sig_aleat))


def score_constrained_ts(
    pred: JointPosteriorPrediction,
    rng: np.random.Generator,
    *,
    alpha: float = 0.05,
) -> int:
    """Joint-posterior Cholesky Constrained Thompson Sampling."""
    z_a = float(norm.ppf(1.0 - float(alpha)))
    Lf = _safe_cholesky(pred.cov_obj)
    Lg = _safe_cholesky(pred.cov_con)
    f_draw = pred.mean_obj + Lf @ rng.normal(size=pred.mean_obj.size)
    g_draw = pred.mean_con + Lg @ rng.normal(size=pred.mean_con.size) + z_a * np.sqrt(
        np.maximum(pred.ucb_aleatoric_var_con, 1e-8)
    )
    feas = np.where(g_draw <= 0.0)[0]
    if feas.size > 0:
        return int(feas[np.argmin(f_draw[feas])])
    return int(np.argmin(g_draw))


def score_constrained_ei(
    pred: JointPosteriorPrediction,
    *,
    alpha: float = 0.05,
    verif_budget_slot0: int = 112,
    delta_slot0: float = 0.025,
) -> np.ndarray:
    """Gate-Power Constrained Expected Improvement (`EI_f * P_gate`)."""
    p_gate = posterior_binomial_gate_power(
        pred.mean_con,
        pred.std_con,
        pred.ucb_aleatoric_var_con,
        n_samples=int(verif_budget_slot0),
        alpha=float(alpha),
        delta_slot=float(delta_slot0),
        r_perp_con=pred.r_perp_con,
    )
    feas_like = p_gate >= 0.50
    f_best = float(np.min(pred.mean_obj[feas_like])) if np.any(feas_like) else float(np.median(pred.mean_obj))
    s_f = np.maximum(pred.std_obj, 1e-6)
    u = (f_best - pred.mean_obj) / s_f
    ei = s_f * (u * ndtr(u) + norm.pdf(u))
    return ei * p_gate


def generate_dynamic_candidate_pool(
    problem: BaseDynamicORProblem,
    model: TemperedHierarchicalHVDModel,
    prior: DebiasedSpectralMetaPrior | None,
    rng: np.random.Generator,
    *,
    pool_size: int = 160,
    enable_source_templates: bool = True,
) -> tuple[list[np.ndarray], list[NominalTrajectoryExposure]]:
    """Build a multi-scale candidate pool with explicit control over source-template proposals."""
    d = problem.d
    n_ch = int(problem.num_control_channels)
    obs_recs = model.unique_records
    cands: list[np.ndarray] = [r.x_norm.copy() for r in obs_recs]

    if obs_recs:
        scored_obs = sorted(
            obs_recs,
            key=lambda r: (
                0 if (r.mean_con + 2.05 * np.sqrt(max(r.ssq_con / max(r.n_reps - 1, 1), 1e-8))) <= 0.01 else 1,
                r.mean_obj
                if (r.mean_con + 2.05 * np.sqrt(max(r.ssq_con / max(r.n_reps - 1, 1), 1e-8))) <= 0.01
                else r.mean_con,
            ),
        )
        elites = [r.x_norm for r in scored_obs[: min(4, len(scored_obs))]]
    else:
        elites = [problem.domain_heuristic_anchor()]

    anchor = problem.domain_heuristic_anchor()
    n_local = max((pool_size - len(cands)) * 2 // 3, 40)
    for idx in range(n_local):
        parent = elites[0] if (idx % 2 == 0) else elites[(idx // 2) % len(elites)]
        radius = (0.025, 0.06, 0.13)[idx % 3]
        mode_sel = idx % 3
        if mode_sel == 0:
            delta = sample_smooth_dct_delta(rng, d, radius=radius, k_modes=12, num_channels=n_ch)
            cands.append(np.clip(parent + delta, 0.0, 1.0))
        elif mode_sel == 1:
            child = parent.copy()
            if n_ch == 1:
                scale = float(rng.uniform(0.80, 1.15))
                child *= scale
            else:
                scale = rng.uniform(0.75, 1.15, size=2)
                child[0::2] *= scale[0]
                if d > 1:
                    child[1::2] *= scale[1]
            delta = sample_smooth_dct_delta(rng, d, radius=0.5 * radius, k_modes=8, num_channels=n_ch)
            cands.append(np.clip(child + delta, 0.0, 1.0))
        else:
            step = rng.normal(0.0, radius, size=d)
            mask = rng.uniform(size=d) < min(20.0 / d, 0.55)
            child = parent.copy()
            if np.any(mask):
                child[mask] = np.clip(child[mask] + step[mask], 0.0, 1.0)
            else:
                child = np.clip(child + 0.5 * step, 0.0, 1.0)
            cands.append(child)

    if enable_source_templates and prior is not None and prior.is_fitted and prior.elite_dct_profiles_.size > 0:
        src_cands = prior.warmstart_candidates(d, 20, rng, anchor=anchor, num_channels=n_ch)
        best_e = elites[0]
        for sc in src_cands:
            w_mix = float(rng.uniform(0.35, 1.0))
            cands.append(np.clip(w_mix * sc + (1.0 - w_mix) * best_e, 0.0, 1.0))

    while len(cands) < pool_size:
        if len(cands) % 2 == 0:
            delta = sample_smooth_dct_delta(rng, d, radius=0.24, k_modes=12, num_channels=n_ch)
            cands.append(np.clip(anchor + delta, 0.0, 1.0))
        else:
            prof = rng.normal(0.0, 0.25 / np.arange(1, 13), size=12)
            cands.append(dct_profile_to_policy(prof, d, num_channels=n_ch))

    cands = cands[:pool_size]
    obs_exp_map = {model._policy_key(r.x_norm): r.exposure for r in obs_recs}
    exposures = []
    for x in cands:
        key = model._policy_key(x)
        if key in obs_exp_map:
            exposures.append(obs_exp_map[key])
        else:
            exposures.append(problem.nominal_exposure(x))
    return cands, exposures


@dataclass(frozen=True)
class SequentialRunResult:
    """Complete metrics and trace of a sequential search + gated terminal verification run."""

    method_name: str
    task_id: str
    seed: int
    n0_init_points: int
    n_online_steps: int
    search_calls_used: int
    verification_calls_used: int
    total_target_calls: int
    wall_time_sec: float
    deployed_policy: np.ndarray | None
    certified: bool
    selected_role: str
    posterior_beta_con: float
    posterior_tau_shift_con: float
    untempered_inflation_factor: float
    n_unique_policies_evaluated: int
    n_replication_queries: int
    receipt: TerminalDeploymentReceipt
    mechanism_receipt: dict | None = None


def build_cold_dct_initial_points(
    problem: BaseDynamicORProblem,
    n0_init: int,
    rng_init: np.random.Generator,
) -> list[np.ndarray]:
    """Shared source-free initial design: operational anchor + smooth DCT + box draws."""
    n_ch = int(problem.num_control_channels)
    anchor = problem.domain_heuristic_anchor()
    pts: list[np.ndarray] = [anchor.copy()]
    while len(pts) < int(n0_init):
        if len(pts) % 3 == 1:
            delta = sample_smooth_dct_delta(rng_init, problem.d, radius=0.15, k_modes=12, num_channels=n_ch)
            pts.append(np.clip(anchor + delta, 0.0, 1.0))
        elif len(pts) % 3 == 2:
            prof = rng_init.normal(0.0, 0.22 / np.arange(1, 13), size=12)
            pts.append(dct_profile_to_policy(prof, problem.d, num_channels=n_ch))
        else:
            pts.append(rng_init.uniform(0.05, 0.95, size=problem.d))
    return pts[: int(n0_init)]


def build_hybrid_warmstart_points(
    problem: BaseDynamicORProblem,
    prior: DebiasedSpectralMetaPrior,
    n0_init: int,
    rng_init: np.random.Generator,
) -> list[np.ndarray]:
    """Balanced initial design: anchor + top source DCT templates + target exploratory DCT/box points."""
    n_tot = int(n0_init)
    if n_tot <= 2:
        return build_cold_dct_initial_points(problem, n_tot, rng_init)
    n_warm = max((n_tot * 3) // 4, 2)
    warm_pts = prior.warmstart_candidates(
        problem.d,
        n_warm,
        rng_init,
        anchor=problem.domain_heuristic_anchor(),
        num_channels=int(problem.num_control_channels),
    )
    cold_pts = build_cold_dct_initial_points(problem, n_tot, rng_init)
    out = list(warm_pts)
    for cp in cold_pts[1:]:
        if len(out) >= n_tot:
            break
        out.append(cp)
    return out[:n_tot]


def build_balanced_warmstart_points(
    problem: BaseDynamicORProblem,
    prior: DebiasedSpectralMetaPrior,
    n0_init: int,
    rng_init: np.random.Generator,
) -> list[np.ndarray]:
    """50/50 design: operational anchor + diverse source templates + exploratory target DCT/box points."""
    n_tot = int(n0_init)
    if n_tot <= 2:
        return build_cold_dct_initial_points(problem, n_tot, rng_init)
    n_warm = max(n_tot // 2, 2)
    warm_pts = prior.warmstart_candidates(
        problem.d,
        n_warm,
        rng_init,
        anchor=problem.domain_heuristic_anchor(),
        num_channels=int(problem.num_control_channels),
    )
    cold_pts = build_cold_dct_initial_points(problem, n_tot, rng_init)
    out = list(warm_pts)
    for cp in cold_pts[1:]:
        if len(out) >= n_tot:
            break
        out.append(cp)
    return out[:n_tot]


def _initial_release_evidence(
    prior: DebiasedSpectralMetaPrior,
    records: Sequence[SufficientStatsRecord],
) -> dict:
    """Score the later initialization batches from the earlier ones, without new calls."""
    cut = len(records) // 2
    train, test = records[:cut], records[cut:]
    held_means = np.array([r.mean_con for r in test])
    held_n = np.array([r.n_reps for r in test])
    scores = {}
    for kind in ("weak_target", "historical_full"):
        model = TemperedHierarchicalHVDModel(
            prior, mode="matched_coefficients", coefficient_prior=kind,
            use_source_basis=True, use_source_hvd_prior=False, use_empirical_r_perp=False,
        )
        for record in train:
            model.observe(record)
        model.freeze_matched_scales(train)
        prediction = model.predict_joint([r.x_norm for r in test], [r.exposure for r in test])
        observed_cov = prediction.cov_con + np.diag(prediction.aleatoric_var_con / held_n)
        factor = _safe_cholesky(observed_cov)
        standardized = np.linalg.solve(factor, held_means - prediction.mean_con)
        log_normalizer = len(test) * np.log(2.0 * np.pi) + 2.0 * np.log(np.diag(factor)).sum()
        scores[kind] = -0.5 * float(standardized @ standardized + log_normalizer)
    ratio = scores["weak_target"] - scores["historical_full"]
    threshold = float(np.log(10.0))
    return {
        "n_train_batches": len(train), "n_held_batches": len(test),
        "log_predictive_weak": scores["weak_target"], "log_predictive_full": scores["historical_full"],
        "log_ratio_weak_over_full": ratio, "threshold": threshold, "recommend_weak": ratio > threshold,
    }


def _variance_summary(records: Sequence[SufficientStatsRecord]) -> dict:
    dof = sum(r.n_reps - 1 for r in records)
    return {"dof": dof, **{f"{head}_var": sum(getattr(r, f"ssq_{head}") for r in records) / dof if dof else None
                          for head in ("obj", "con")}}


def _observed_variances(
    batches: Sequence[SufficientStatsRecord], model: TemperedHierarchicalHVDModel, receipt: TerminalDeploymentReceipt,
) -> dict:
    policies: dict[bytes, list[SufficientStatsRecord]] = {}
    for batch in batches:
        policies.setdefault(model._policy_key(batch.x_norm), []).append(batch)
    merged = {model._policy_key(record.x_norm): record for record in model.unique_records}

    def describe(keys: Sequence[bytes]) -> dict:
        return {"n_policies": len(keys), "n_batches": sum(len(policies[key]) for key in keys),
                "fixed_batch": _variance_summary([r for key in keys for r in policies[key]]),
                "pooled": _variance_summary([merged[key] for key in keys])}

    result = {"all": describe(list(policies)), "revisited": describe([k for k, v in policies.items() if len(v) > 1])}
    for name, policy in (("primary", receipt.primary_policy), ("deployed", receipt.deployed_policy)):
        if policy is not None:
            result[name] = describe([model._policy_key(policy)])
    return result


def run_tls_olh_kg(
    problem: BaseDynamicORProblem,
    prior: DebiasedSpectralMetaPrior | None,
    *,
    n_online_steps: int = 30,
    n0_init: int = 8,
    batch_reps: int = 2,
    seed: int = 1001,
    model_mode: str = "tempered",
    acquisition_type: str = "gate_power_ts",
    init_mode: str = "auto",
    proposal_mode: str = "auto",
    use_source_basis: bool | None = None,
    use_source_coef_prior: bool | None = None,
    use_source_hvd_prior: bool | None = None,
    variance_estimator: str = "within_ssq",
    use_empirical_r_perp: bool = True,
    nomination_mode: str = "binomial_gate_power",
    pool_size: int = 160,
    method_label: str = "TLS-OLH-KG",
    gate_mode: str = "auto",
    coefficient_prior: str | None = None,
) -> SequentialRunResult:
    """Execute initial design + `n_online_steps` sequential queries + gated verification."""
    t0 = time.perf_counter()
    rng_init = np.random.default_rng(seed)
    rng_search = np.random.default_rng(seed + 100_003)
    verif_seed = seed + 200_007
    matched = model_mode == "matched_coefficients"
    if matched and (init_mode != "cold_dct" or proposal_mode != "local_dct_only" or n0_init < 4):
        raise ValueError("Matched runs require common cold initialization and source-free candidate proposals")

    calls_start = problem.search_calls
    model = TemperedHierarchicalHVDModel(
        prior,
        mode=model_mode,
        use_source_basis=use_source_basis,
        use_source_coef_prior=use_source_coef_prior,
        use_source_hvd_prior=use_source_hvd_prior,
        variance_estimator=variance_estimator,
        use_empirical_r_perp=use_empirical_r_perp,
        n0_freeze=n0_init,
        coefficient_prior=coefficient_prior,
    )

    use_balanced_init = False
    if init_mode == "auto":
        use_warm_init = bool(prior is not None and prior.is_fitted and model.use_source_coef_prior)
        use_hybrid_init = bool(use_warm_init and model_mode == "tempered_full_release")
        use_balanced_init = bool(use_warm_init and model_mode == "tempered_sure_release")
    elif init_mode == "source_warmstart":
        use_warm_init = True
        use_hybrid_init = False
    elif init_mode == "hybrid_warmstart":
        use_warm_init = True
        use_hybrid_init = True
    elif init_mode == "balanced_warmstart":
        use_warm_init = True
        use_hybrid_init = False
        use_balanced_init = True
    elif init_mode == "cold_dct":
        use_warm_init = False
        use_hybrid_init = False
    else:
        raise ValueError(f"Unknown init_mode {init_mode!r}")

    if proposal_mode == "auto":
        use_src_templates = bool(prior is not None and prior.is_fitted and model.use_source_coef_prior)
        adaptive_template_release = bool(model_mode in {"tempered_full_release", "tempered_sure_release"})
    elif proposal_mode == "source_templates":
        use_src_templates = True
        adaptive_template_release = False
    elif proposal_mode == "adaptive_release":
        use_src_templates = bool(prior is not None and prior.is_fitted)
        adaptive_template_release = True
    elif proposal_mode == "local_dct_only":
        use_src_templates = False
        adaptive_template_release = False
    else:
        raise ValueError(f"Unknown proposal_mode {proposal_mode!r}")

    if use_warm_init:
        if prior is None or not prior.is_fitted:
            raise ValueError(f"init_mode={init_mode!r} requires a fitted prior")
        if use_balanced_init:
            init_points = build_balanced_warmstart_points(problem, prior, n0_init, rng_init)
        elif use_hybrid_init:
            init_points = build_hybrid_warmstart_points(problem, prior, n0_init, rng_init)
        else:
            init_points = prior.warmstart_candidates(
                problem.d,
                n0_init,
                rng_init,
                anchor=problem.domain_heuristic_anchor(),
                num_channels=int(problem.num_control_channels),
            )
    else:
        init_points = build_cold_dct_initial_points(problem, n0_init, rng_init)

    initial_records: list[SufficientStatsRecord] = []
    for x0 in init_points:
        exp0 = problem.nominal_exposure(x0)
        samp0 = problem.simulate_replications(x0, batch_reps, rng_init, split="search")
        rec0 = SufficientStatsRecord.from_samples(x0, samp0, exp0)
        initial_records.append(rec0)
        model.observe(rec0)

    mechanism = None
    track_prospective = bool(
        model_mode in {"tempered_sure_release", "tempered_full_release", "target_only"}
        or acquisition_type == "eligible_cascade_kg"
    )
    if matched:
        evidence = _initial_release_evidence(prior, initial_records)
        if coefficient_prior == "initial_release" and evidence["recommend_weak"]:
            model.coefficient_prior = "weak_target"
        model.freeze_matched_scales(initial_records)
        evidence["applied_weak"] = model.coefficient_prior == "weak_target"
        mechanism = {"configured_coefficient_prior": coefficient_prior, "active_coefficient_prior": model.coefficient_prior,
                     "early_release": evidence, "first_query_index": None}
        batch_history = initial_records.copy()
    elif track_prospective:
        mechanism = {
            "model_mode": model_mode,
            "acquisition_type": acquisition_type,
            "init_sure_lambda_obj": float(model.sure_lambda_obj_),
            "init_sure_lambda_con": float(model.sure_lambda_con_),
            "init_beta_obj": float(model.beta_obj_),
            "init_beta_con": float(model.beta_con_),
            "init_source_released": bool(model.is_source_released_),
            "first_query_index": None,
        }
        batch_history = initial_records.copy()

    # Initial point 0 is always `domain_heuristic_anchor()`, giving an online sample estimate of status-quo cost
    sq_est_obj = float(initial_records[0].mean_obj) if initial_records else None

    n_rep_queries = 0
    eckg_step_logs: list[dict] = []
    if acquisition_type != "stop" and n_online_steps > 0:
        for _step in range(int(n_online_steps)):
            enable_tpl_step = bool(
                use_src_templates and not (adaptive_template_release and getattr(model, "is_source_released_", False))
            )
            cands, exps = generate_dynamic_candidate_pool(
                problem,
                model,
                prior,
                rng_search,
                pool_size=pool_size,
                enable_source_templates=enable_tpl_step,
            )
            pred = model.predict_joint(cands, exps)
            sq_ref_step = (
                0.5 * float(sq_est_obj) + 0.5 * float(pred.mean_obj[0])
                if (sq_est_obj is not None and len(pred.mean_obj) > 0)
                else sq_est_obj
            )

            if acquisition_type in {"v_kg", "v_kg_legacy"}:
                scores = score_verification_kg(
                    pred,
                    alpha=problem.alpha,
                    batch_reps=batch_reps,
                    residual_ucb_weight=0.04 / np.sqrt(float(_step) + 1.0),
                )
                chosen_idx = int(np.argmax(scores))
            elif acquisition_type == "gate_power_vkg":
                scores = score_gate_power_vkg(
                    pred,
                    alpha=problem.alpha,
                    batch_reps=batch_reps,
                    status_quo_obj=sq_ref_step,
                    common_target_screen=matched,
                )
                chosen_idx = int(np.argmax(scores))
            elif acquisition_type == "chance_feasibility_kg":
                scores = score_chance_feasibility_kg(
                    pred,
                    alpha=problem.alpha,
                    batch_reps=batch_reps,
                    status_quo_obj=sq_ref_step,
                    common_target_screen=matched or True,
                )
                chosen_idx = int(np.argmax(scores))
            elif acquisition_type == "eligible_cascade_kg":
                scores, step_diag = score_eligible_cascade_kg(
                    pred,
                    model,
                    cands,
                    exps,
                    alpha=problem.alpha,
                    batch_reps=batch_reps,
                    status_quo_obj=sq_ref_step,
                    nomination_mode=nomination_mode,
                    return_diagnostics=True,
                )
                chosen_idx = int(step_diag["chosen_query"])
                eckg_step_logs.append(step_diag)
            elif acquisition_type == "gate_power_ts":
                chosen_idx = score_gate_power_ts(
                    pred,
                    rng_search,
                    alpha=problem.alpha,
                    status_quo_obj=sq_ref_step,
                )
            elif acquisition_type == "constrained_ts":
                chosen_idx = score_constrained_ts(pred, rng_search, alpha=problem.alpha)
            elif acquisition_type == "cei":
                scores = score_constrained_ei(pred, alpha=problem.alpha)
                chosen_idx = int(np.argmax(scores))
            else:
                raise ValueError(f"Unknown acquisition_type {acquisition_type!r}")

            x_next = cands[chosen_idx]
            if model._policy_key(x_next) in model._records_by_key:
                n_rep_queries += 1
            samp = problem.simulate_replications(x_next, batch_reps, rng_search, split="search")
            rec = SufficientStatsRecord.from_samples(x_next, samp, exps[chosen_idx])
            model.observe(rec)
            if mechanism is not None:
                if _step == 0:
                    mechanism["first_query_index"] = chosen_idx
                batch_history.append(rec)

    eval_recs = model.unique_records
    term_cands = [r.x_norm.copy() for r in eval_recs]
    term_exps = [r.exposure for r in eval_recs]
    term_pred = model.predict_joint(term_cands, term_exps)
    sq_ref_term = (
        0.5 * float(sq_est_obj) + 0.5 * float(term_pred.mean_obj[0])
        if (sq_est_obj is not None and len(term_pred.mean_obj) > 0)
        else sq_est_obj
    )
    shortlist = build_feasible_aware_shortlist(
        candidate_pool=term_cands,
        mean_obj=term_pred.mean_obj,
        mean_con=term_pred.mean_con,
        std_con=term_pred.std_con,
        ucb_aleat_var_con=term_pred.ucb_aleatoric_var_con,
        r_perp_con=term_pred.r_perp_con,
        initial_records=eval_recs,
        alpha=problem.alpha,
        nomination_mode=nomination_mode,
        reference_obj=sq_ref_term,
    )

    receipt = verify_and_deploy_shortlist(
        problem,
        shortlist,
        seed=verif_seed,
        gate_mode=gate_mode,
    )
    if mechanism is not None:
        mechanism["variance"] = _observed_variances(batch_history, model, receipt)
        if not matched:
            mechanism.update({
                "final_sure_lambda_obj": float(model.sure_lambda_obj_),
                "final_sure_lambda_con": float(model.sure_lambda_con_),
                "final_beta_obj": float(model.beta_obj_),
                "final_beta_con": float(model.beta_con_),
                "final_source_released": bool(model.is_source_released_),
                "source_cov_shrink": float(model.source_cov_shrink_),
            })
            if eckg_step_logs:
                mechanism["eckg_summary"] = {
                    "n_steps": len(eckg_step_logs),
                    "disagreed_with_single_slot_steps": sum(int(d["disagreed_with_single_slot"]) for d in eckg_step_logs),
                    "disagreed_with_feasible_kg_steps": sum(int(d["disagreed_with_feasible_kg"]) for d in eckg_step_logs),
                    "chosen_revisit_steps": sum(int(d["chosen_is_revisit"]) for d in eckg_step_logs),
                    "mean_voi_total": float(np.mean([d["chosen_voi_total"] for d in eckg_step_logs])),
                    "mean_voi_refine": float(np.mean([d["chosen_voi_refine"] for d in eckg_step_logs])),
                    "mean_voi_admit": float(np.mean([d["chosen_voi_admit"] for d in eckg_step_logs])),
                    "mean_step_wall_ms": float(np.mean([d["step_wall_ms"] for d in eckg_step_logs])),
                }
    wall_sec = time.perf_counter() - t0
    search_used = int(problem.search_calls - calls_start)

    return SequentialRunResult(
        method_name=method_label,
        task_id=problem.task_id,
        seed=int(seed),
        n0_init_points=int(n0_init),
        n_online_steps=0 if acquisition_type == "stop" else int(n_online_steps),
        search_calls_used=search_used,
        verification_calls_used=receipt.verification_calls_used,
        total_target_calls=search_used + receipt.verification_calls_used,
        wall_time_sec=float(wall_sec),
        deployed_policy=receipt.deployed_policy,
        certified=receipt.certified,
        selected_role=receipt.selected_role,
        posterior_beta_con=float(model.beta_con_),
        posterior_tau_shift_con=float(model.tau_shift_con_),
        untempered_inflation_factor=float(model.empirical_var_ratio_vs_within_),
        n_unique_policies_evaluated=len(model.unique_records),
        n_replication_queries=n_rep_queries,
        receipt=receipt,
        mechanism_receipt=mechanism,
    )
