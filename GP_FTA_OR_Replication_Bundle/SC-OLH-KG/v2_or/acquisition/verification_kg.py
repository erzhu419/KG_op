"""Acquisition Strategies, Controlled Proposal Pools & Modular Sequential Search Engine.

Supports both one-step lookahead and Thompson/cEI acquisitions under a unified,
strictly modular interface so every source-information channel (`init_mode`,
`use_source_basis`, `use_source_coef_prior`, `use_source_hvd_prior`, `proposal_mode`,
`acquisition_type`, `nomination_mode`) can be toggled independently for clean causal
attribution:
- `"gate_power_ts"`: Joint-posterior Cholesky Thompson Sampling maximizing drawn
  expected certified cost saving `(J_sq - f_draw)_+ * C_{K_0}(Phi(g_draw / sigma_g))`
  using exact terminal binomial gate power `C_{K_0}(p)`.
- `"gate_power_vkg"`: Rank-one fantasy lookahead maximizing one-step expected gain in
  `max_{x'} (J_sq - mu_f(x'))_+ * E[C_{K_0}(p(x'))]` (correcting the 23x boundary
  overestimation of naive Gaussian `z_{1-alpha}` margins).
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
from scipy.special import ndtr
from scipy.stats import norm

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
    return pts * np.sqrt(2.0), wts / np.sqrt(np.pi)


_GH5_ZF, _GH5_ZG, _GH5_W = _gauss_hermite_2d(5)
_GH7_Z, _GH7_W = _gauss_hermite_1d(7)


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


def score_gate_power_vkg(
    pred: JointPosteriorPrediction,
    *,
    alpha: float = 0.05,
    batch_reps: int = 2,
    verif_budget_slot0: int = 112,
    delta_slot0: float = 0.025,
    status_quo_obj: float | None = None,
    max_eval_targets: int = 64,
) -> np.ndarray:
    """One-step lookahead KG using exact terminal binomial gate power `C_{K_0}(p)`."""
    m_cand = pred.mean_obj.size
    av_ucb = np.maximum(pred.ucb_aleatoric_var_con, 1e-8)
    pass_power_curr = posterior_binomial_gate_power(
        pred.mean_con,
        pred.std_con,
        av_ucb,
        n_samples=int(verif_budget_slot0),
        alpha=float(alpha),
        delta_slot=float(delta_slot0),
        r_perp_con=pred.r_perp_con,
    )
    spread_f = max(float(np.std(pred.mean_obj)), 1e-8)
    if status_quo_obj is not None and np.isfinite(status_quo_obj):
        j_ref = max(float(status_quo_obj), float(np.min(pred.mean_obj)) + 0.20 * spread_f)
    else:
        j_ref = float(np.quantile(pred.mean_obj, 0.85)) + 0.25 * spread_f

    curr_utility = np.maximum(j_ref - pred.mean_obj, 0.0) * pass_power_curr
    boundary_info = pass_power_curr * (1.0 - pass_power_curr) * spread_f
    eval_idx = np.argsort(-curr_utility - 0.12 * boundary_info)[: min(int(max_eval_targets), m_cand)]
    u_curr_max = float(np.max(curr_utility[eval_idx]))

    denom_f = np.sqrt(np.maximum(np.diag(pred.cov_obj) + pred.aleatoric_var_obj / float(batch_reps), 1e-10))
    denom_g = np.sqrt(np.maximum(np.diag(pred.cov_con) + pred.aleatoric_var_con / float(batch_reps), 1e-10))
    tilde_f = pred.cov_obj[:, eval_idx] / denom_f[:, None]
    tilde_g = pred.cov_con[:, eval_idx] / denom_g[:, None]

    mu_f_eval = pred.mean_obj[eval_idx]
    mu_g_eval = pred.mean_con[eval_idx]
    s2_g_eval = pred.std_con[eval_idx] ** 2 + 0.25 * pred.r_perp_con[eval_idx]
    sig_aleat_eval = np.sqrt(av_ucb[eval_idx])

    s_next_g = np.sqrt(np.maximum(s2_g_eval[None, :] - tilde_g ** 2, 1e-10))
    k_crit = max(exact_binomial_critical_k(int(verif_budget_slot0), float(alpha), float(delta_slot0)), 0)

    # (m_cand, 25, m_eval) fantasy posterior means, integrated over 7 GH nodes for exact C_{K_0}(p)
    mu_f_next = mu_f_eval[None, None, :] + tilde_f[:, None, :] * _GH5_ZF[None, :, None]
    mu_g_next = mu_g_eval[None, None, :] + tilde_g[:, None, :] * _GH5_ZG[None, :, None]
    g_lat = mu_g_next[:, :, :, None] + s_next_g[:, None, :, None] * _GH7_Z[None, None, None, :]
    p_lat = ndtr(g_lat / sig_aleat_eval[None, None, :, None])
    c_lat = exact_binomial_pass_power(p_lat, int(verif_budget_slot0), k_crit)
    p_gate_next = np.tensordot(c_lat, _GH7_W, axes=([3], [0]))  # (m_cand, 25, m_eval)

    u_next = np.maximum(j_ref - mu_f_next, 0.0) * p_gate_next
    best_u_per_fantasy = np.max(u_next, axis=2)
    expected_best_u = best_u_per_fantasy @ _GH5_W
    return np.maximum(expected_best_u - u_curr_max, 0.0)


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
) -> SequentialRunResult:
    """Execute initial design + `n_online_steps` sequential queries + gated verification."""
    t0 = time.perf_counter()
    rng_init = np.random.default_rng(seed)
    rng_search = np.random.default_rng(seed + 100_003)
    verif_seed = seed + 200_007

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
    )

    if init_mode == "auto":
        use_warm_init = bool(prior is not None and prior.is_fitted and model.use_source_coef_prior)
        use_hybrid_init = bool(use_warm_init and model_mode == "tempered_full_release")
    elif init_mode == "source_warmstart":
        use_warm_init = True
        use_hybrid_init = False
    elif init_mode == "hybrid_warmstart":
        use_warm_init = True
        use_hybrid_init = True
    elif init_mode == "cold_dct":
        use_warm_init = False
        use_hybrid_init = False
    else:
        raise ValueError(f"Unknown init_mode {init_mode!r}")

    if proposal_mode == "auto":
        use_src_templates = bool(prior is not None and prior.is_fitted and model.use_source_coef_prior)
        adaptive_template_release = bool(model_mode == "tempered_full_release")
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
        if use_hybrid_init:
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

    # Initial point 0 is always `domain_heuristic_anchor()`, giving an online sample estimate of status-quo cost
    sq_est_obj = float(initial_records[0].mean_obj) if initial_records else None

    n_rep_queries = 0
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
                )
                chosen_idx = int(np.argmax(scores))
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
    )
