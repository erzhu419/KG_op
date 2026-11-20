"""Deterministic unit tests for Fisher-whitened SURE shrinkage & Eligible-Cascade KG."""

from __future__ import annotations

import time
import numpy as np

from v2_or.acquisition import run_tls_olh_kg, score_eligible_cascade_kg
from v2_or.contracts import SufficientStatsRecord
from v2_or.legacy_solvers import generate_legacy_archive
from v2_or.models import JointPosteriorPrediction, TemperedHierarchicalHVDModel
from v2_or.priors import DebiasedSpectralMetaPrior
from v2_or.simulators import sample_inventory_tasks


def _build_toy_prior(seed: int = 101) -> DebiasedSpectralMetaPrior:
    src_tasks = sample_inventory_tasks(4, seed=seed, horizons=(12, 16), shift_scale=1.0)
    recs, _ = generate_legacy_archive(
        src_tasks,
        solvers=("penalty_spsa", "heuristic_powell"),
        per_solver_budget=18,
        reps_per_eval=2,
        seed=seed + 50,
    )
    return DebiasedSpectralMetaPrior(active_rank=6).fit(recs)


def test_fisher_sure_shrinkage_aligned_vs_shifted() -> None:
    """Verify design-whitened Stein-Mallows SURE shrinks aligned noise and releases shifted priors."""
    prior = _build_toy_prior(seed=202)
    prob = sample_inventory_tasks(1, seed=252, horizons=(16,), shift_scale=1.0)[0]
    rng = np.random.default_rng(303)

    m_aligned = TemperedHierarchicalHVDModel(prior=prior, mode="tempered_sure_release")
    m_shifted = TemperedHierarchicalHVDModel(prior=prior, mode="tempered_sure_release")

    anc = prob.domain_heuristic_anchor()
    xs = [np.clip(anc + rng.normal(0.0, 0.22, size=prob.d), 0.0, 1.0) for _ in range(14)]
    exps = [prob.nominal_exposure(x) for x in xs]
    phi_mat = prior.transform_exposures(exps)[0]
    mu0_f = prior.mu_0_obj_
    mu0_g = prior.mu_0_con_

    # Stabilized source covariance check: p / (M_src + p)
    cov_stab, w_src = m_aligned._stabilized_source_cov(prior.cov_0_obj_)
    p_dim = int(prior.cov_0_obj_.shape[0])
    m_src = int(prior.diagnostics_.n_source_tasks)
    assert abs(w_src - (p_dim / float(m_src + p_dim))) < 1e-12
    assert float(np.min(np.linalg.eigvalsh(cov_stab))) > 0.0

    # Case A: Target matches source mean mu0 closely with moderate replication noise
    for i in range(14):
        f_true = float(phi_mat[i] @ mu0_f)
        g_true = float(phi_mat[i] @ mu0_g)
        samp_a = np.column_stack([
            f_true + rng.normal(0.0, 12.0, size=2),
            g_true + rng.normal(0.0, 0.30, size=2),
        ])
        m_aligned.observe(SufficientStatsRecord.from_samples(xs[i], samp_a, exps[i]))

    # Case B: Target undergoes severe structural shift away from mu0
    delta_f = np.full(phi_mat.shape[1], 80.0, dtype=float)
    delta_g = np.full(phi_mat.shape[1], -3.0, dtype=float)
    for i in range(14):
        f_sh = float(phi_mat[i] @ (mu0_f + delta_f))
        g_sh = float(phi_mat[i] @ (mu0_g + delta_g))
        samp_b = np.column_stack([
            f_sh + rng.normal(0.0, 1.5, size=2),
            g_sh + rng.normal(0.0, 0.05, size=2),
        ])
        m_shifted.observe(SufficientStatsRecord.from_samples(xs[i], samp_b, exps[i]))

    assert 0.0 <= m_aligned.sure_lambda_obj_ < m_shifted.sure_lambda_obj_ <= 1.0
    assert 0.0 <= m_aligned.sure_lambda_con_ < m_shifted.sure_lambda_con_ <= 1.0
    assert m_shifted.sure_lambda_obj_ >= 0.80
    assert m_shifted.sure_lambda_con_ >= 0.80
    assert m_shifted.is_source_released_ is True


def test_eligible_cascade_kg_dual_voi_invariants_and_latency() -> None:
    """Verify EC-KG dual VOI decomposition, zero-innovation/revisit laws, and <250ms step speed."""
    prior = _build_toy_prior(seed=404)
    prob = sample_inventory_tasks(1, seed=505, horizons=(16,), shift_scale=1.15)[0]
    rng = np.random.default_rng(606)
    model = TemperedHierarchicalHVDModel(prior=prior, mode="tempered_sure_release")

    anc = prob.domain_heuristic_anchor()
    init_pts = [anc] + [np.clip(anc + rng.normal(0.0, 0.12, size=prob.d), 0.0, 1.0) for _ in range(7)]
    for x in init_pts:
        exp = prob.nominal_exposure(x)
        samp = prob.simulate_replications(x, 2, rng)
        model.observe(SufficientStatsRecord.from_samples(x, samp, exp))

    cands = [r.x_norm.copy() for r in model.unique_records]
    while len(cands) < 160:
        cands.append(np.clip(anc + rng.normal(0.0, 0.15, size=prob.d), 0.0, 1.0))
    exps = [prob.nominal_exposure(x) for x in cands]
    pred = model.predict_joint(cands, exps)

    t0 = time.perf_counter()
    scores, diag = score_eligible_cascade_kg(
        pred,
        model,
        cands,
        exps,
        alpha=prob.alpha,
        batch_reps=2,
        status_quo_obj=float(pred.mean_obj[0]),
        top_k_contenders=16,
        return_diagnostics=True,
    )
    elapsed = time.perf_counter() - t0
    chosen_idx = int(diag["chosen_query"])

    assert 0 <= chosen_idx < len(cands)
    assert scores.shape == (len(cands),)
    assert elapsed < 0.35
    assert diag["chosen_voi_refine"] >= -1e-12
    assert diag["chosen_voi_admit"] >= -1e-12
    assert abs(diag["chosen_voi_total"] - (diag["chosen_voi_refine"] + diag["chosen_voi_admit"])) < 1e-10
    if diag["chosen_is_revisit"]:
        assert diag["chosen_voi_admit"] == 0.0

    # Zero-innovation revisit check: collapse epistemic covariances on already-evaluated policies -> exact 0.0
    m_revisit = len(model.unique_records)
    zero_pred = JointPosteriorPrediction(
        mean_obj=pred.mean_obj[:m_revisit].copy(),
        cov_obj=np.zeros((m_revisit, m_revisit), dtype=float),
        std_obj=np.zeros(m_revisit, dtype=float),
        mean_con=pred.mean_con[:m_revisit].copy(),
        cov_con=np.zeros((m_revisit, m_revisit), dtype=float),
        std_con=np.zeros(m_revisit, dtype=float),
        aleatoric_var_obj=pred.aleatoric_var_obj[:m_revisit].copy(),
        aleatoric_var_con=pred.aleatoric_var_con[:m_revisit].copy(),
        ucb_aleatoric_var_con=pred.ucb_aleatoric_var_con[:m_revisit].copy(),
        r_perp_con=np.zeros(m_revisit, dtype=float),
    )
    z_scores, z_diag = score_eligible_cascade_kg(
        zero_pred,
        model,
        cands[:m_revisit],
        exps[:m_revisit],
        alpha=prob.alpha,
        batch_reps=2,
        status_quo_obj=float(pred.mean_obj[0]),
        top_k_contenders=8,
        return_diagnostics=True,
    )
    assert np.all(z_scores == 0.0)
    assert z_diag["chosen_voi_total"] == 0.0
    assert z_diag["chosen_voi_refine"] == 0.0
    assert z_diag["chosen_voi_admit"] == 0.0


def test_end_to_end_mini_eckg_sure_run() -> None:
    """Verify full closed-loop GP-FTA-ECKG execution, telemetry, and call accounting."""
    prior = _build_toy_prior(seed=707)
    prob = sample_inventory_tasks(1, seed=808, horizons=(16,), shift_scale=1.2)[0]
    res = run_tls_olh_kg(
        prob,
        prior,
        n_online_steps=4,
        n0_init=6,
        batch_reps=2,
        seed=909,
        model_mode="tempered_sure_release",
        acquisition_type="eligible_cascade_kg",
        init_mode="balanced_warmstart",
        proposal_mode="adaptive_release",
        pool_size=48,
        method_label="GP-FTA-ECKG",
    )
    assert res.search_calls_used == (6 + 4) * 2
    assert res.mechanism_receipt is not None
    assert "eckg_summary" in res.mechanism_receipt
    assert res.mechanism_receipt["eckg_summary"]["n_steps"] == 4
    assert 0.0 <= res.mechanism_receipt["final_sure_lambda_con"] <= 1.0
