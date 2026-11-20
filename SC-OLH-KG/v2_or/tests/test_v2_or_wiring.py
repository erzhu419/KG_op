"""Deterministic local wiring, zero-leakage, LOOCV, adapter & contract tests for `v2_or`.

Strictly lightweight (< 5s single-threaded) per `AGENTS.md`: tests mathematical
identities, private-array perturbation invariance across all 4 families, multi-channel
policy warping, analytic GP LOOCV exactness, MAP-SAAS gradient accuracy, exact binomial
gate power, strict abstention (`deployed_policy is None`), and modular P1 switches.
"""

from __future__ import annotations

from dataclasses import replace
import time

import numpy as np
from scipy.stats import binom, norm

from v2_or.acquisition import run_tls_olh_kg, score_gate_power_vkg
from v2_or.baselines import (
    RawInputExactGP,
    run_direct_legacy_baseline,
    run_hetgp_kg_baseline,
    run_hyperbo_transfer_baseline,
    run_rgpe_on_psi_baseline,
    run_rgpe_transfer_baseline,
    run_saasbo_baseline,
    run_scbo_baseline,
    run_scbo_on_psi_baseline,
    run_warmstart_pattern_search_baseline,
)
from v2_or.contracts import SufficientStatsRecord, warp_multichannel_policy
from v2_or.legacy_solvers import generate_legacy_archive
from v2_or.baselines.sota_baselines import _ard_matern52
from v2_or.models import TemperedHierarchicalHVDModel
from v2_or.priors import DebiasedSpectralMetaPrior
from v2_or.simulators import (
    OPSDGridStorageBenchmark,
    discover_valid_opsd_markets,
    sample_inventory_tasks,
    sample_queueing_tasks,
    sample_traffic_tasks,
)
from v2_or.verification import (
    exact_binomial_critical_k,
    exact_binomial_gate,
    gaussian_noncentral_t_gate,
    posterior_binomial_gate_power,
    select_empirical_feasible_incumbent,
    verify_and_deploy_shortlist,
)


def test_zero_oracle_leakage_invariance_all_four_families() -> None:
    """Mutating private task laws / private arrays MUST leave `nominal_exposure(x)` bit-identical."""
    rng = np.random.default_rng(42)

    # 1. Inventory DES
    inv_a = sample_inventory_tasks(1, seed=10, horizons=(24,))[0]
    mut_inv_law = replace(
        inv_a._law,
        regime_drift=inv_a._law.regime_drift + 0.55,
        shortage_penalty=99.0,
        sla_max_shortage_rate=0.01,
    )
    inv_b = type(inv_a)(
        task_id="inv_mut",
        demand_forecast=inv_a.demand_forecast.copy(),
        capacity=inv_a.capacity,
        initial_stock=inv_a.initial_stock,
        law=mut_inv_law,
    )
    x_inv = rng.uniform(0.0, 1.0, size=inv_a.d)
    assert inv_a.nominal_exposure(x_inv).psi.shape == (12,)
    assert np.max(np.abs(inv_a.nominal_exposure(x_inv).psi - inv_b.nominal_exposure(x_inv).psi)) == 0.0
    assert inv_a.search_calls == 0

    # 2. Queueing DES
    q_a = sample_queueing_tasks(1, seed=20, horizons=(24,))[0]
    mut_q_law = replace(
        q_a._law,
        regime_burst_mean=q_a._law.regime_burst_mean + 0.40,
        waiting_cost=50.0,
        sla_max_breach_fraction=0.02,
    )
    q_b = type(q_a)(
        task_id="q_mut",
        arrival_forecast=q_a.arrival_forecast.copy(),
        c_min=q_a.c_min,
        c_max=q_a.c_max,
        initial_queue=q_a.initial_queue,
        law=mut_q_law,
    )
    x_q = rng.uniform(0.0, 1.0, size=q_a.d)
    assert q_a.nominal_exposure(x_q).psi.shape == (12,)
    assert np.max(np.abs(q_a.nominal_exposure(x_q).psi - q_b.nominal_exposure(x_q).psi)) == 0.0

    # 3. Traffic Signal Network DES
    tr_a = sample_traffic_tasks(1, seed=30, tls_counts=(11,))[0]
    mut_tr_law = replace(
        tr_a._law,
        wave_drift=tr_a._law.wave_drift + 0.35,
        weather_sat_mean=0.65,
        tau_co2_ratio=0.90,
    )
    tr_b = type(tr_a)(
        task_id="tr_mut",
        lo=tr_a.lo,
        hi=tr_a.hi,
        defaults=tr_a.defaults,
        var_tls=tr_a.var_tls,
        fixed_overhead=tr_a.fixed_overhead,
        public_arrivals=tr_a.public_arrivals,
        law=mut_tr_law,
    )
    x_tr = rng.uniform(0.0, 1.0, size=tr_a.d)
    assert tr_a.nominal_exposure(x_tr).psi.shape == (12,)
    assert np.max(np.abs(tr_a.nominal_exposure(x_tr).psi - tr_b.nominal_exposure(x_tr).psi)) == 0.0

    # 4. Real-data OPSD Storage: directly overwrite private `_net_error` array and verify bit-identity!
    opsd = OPSDGridStorageBenchmark(market="DK_1", d=96, buffer_hours=24)
    x_op = rng.uniform(0.0, 1.0, size=opsd.d)
    psi_before = opsd.nominal_exposure(x_op).psi.copy()
    opsd._net_error = np.full_like(opsd._net_error, 0.99)
    psi_after = opsd.nominal_exposure(x_op).psi
    assert psi_before.shape == (12,)
    assert np.max(np.abs(psi_before - psi_after)) == 0.0
    assert opsd.search_calls == 0


def test_multichannel_policy_warping_and_loocv_exactness() -> None:
    """Verify GPT-6's `[0.2, 0.8]*3 -> [0.2, 0.8]*4` adapter test & exact GP LOOCV."""
    src_const = np.array([0.2, 0.8, 0.2, 0.8, 0.2, 0.8], dtype=float)
    warped = warp_multichannel_policy(src_const, target_d=8, num_channels=2)
    expected = np.array([0.2, 0.8, 0.2, 0.8, 0.2, 0.8, 0.2, 0.8], dtype=float)
    assert np.allclose(warped, expected, atol=1e-12)

    inv_prob = sample_inventory_tasks(1, seed=12, horizons=(16,))[0]
    assert np.allclose(inv_prob.warp_source_policy(src_const)[0::2], 0.2, atol=1e-12)
    assert np.allclose(inv_prob.warp_source_policy(src_const)[1::2], 0.8, atol=1e-12)

    # Verify analytic GP LOOCV against brute-force leave-one-out refits
    rng = np.random.default_rng(123)
    X = rng.uniform(0.0, 1.0, size=(10, 6))
    y = np.sin(3.0 * X[:, 0]) + 0.5 * X[:, 1]
    nv = np.full(10, 0.02)
    gp = RawInputExactGP(map_saas=True).fit(X, y, nv)
    loo_m, loo_s = gp.loocv_predict()

    # Brute-force leave-one-out using the exact same fitted kernel parameters `(inv_ls, outscale)`
    for i in range(10):
        mask = np.arange(10) != i
        X_tr, y_tr, nv_tr = X[mask], (y[mask] - gp.y_mean) / gp.y_std, gp.nv_norm[mask]
        K_tr = _ard_matern52(X_tr, X_tr, gp.inv_ls, gp.outscale) + np.diag(nv_tr)
        K_te = _ard_matern52(X[i : i + 1], X_tr, gp.inv_ls, gp.outscale)
        alp = np.linalg.solve(K_tr, y_tr)
        bf_mean = gp.y_mean + gp.y_std * float((K_te @ alp)[0])
        bf_var = (gp.y_std ** 2) * float(gp.outscale + gp.nv_norm[i] - (K_te @ np.linalg.solve(K_tr, K_te.T))[0, 0])
        # Note: Rasmussen-Williams Eq. 5.12 gives LOO latent + noise variance `1 / [K^{-1}]_{ii}`
        assert abs(loo_m[i] - bf_mean) < 1e-6
        assert abs(loo_s[i] - np.sqrt(bf_var)) < 1e-6


def test_exact_gate_power_and_strict_abstention_contract() -> None:
    """Verify `C_{112}(0.05) = 0.0220578` at `s_g -> 0` and strict `deployed_policy is None` on abstention."""
    assert exact_binomial_critical_k(112, 0.05, 0.025) == 1
    assert exact_binomial_critical_k(128, 0.05, 0.015) == 1
    assert exact_binomial_critical_k(140, 0.05, 0.010) == 1

    # GPT-6 Section 8.4 boundary check: p = 0.05 -> mu_g = -z_{0.95} * sigma_g, s_g = 0
    z_95 = float(norm.ppf(0.95))
    exact_pass = float(binom.cdf(1, 112, 0.05))
    pred_pass = float(
        posterior_binomial_gate_power(
            mean_con=np.array([-z_95 * 0.2]),
            std_con=np.array([0.0]),
            aleat_var_con=np.array([0.04]),
            n_samples=112,
            alpha=0.05,
            delta_slot=0.025,
        )[0]
    )
    assert abs(pred_pass - exact_pass) < 1e-10
    assert abs(pred_pass - 0.0220578) < 1e-6

    # Strict abstention contract check on an infeasible candidate shortlist
    prob = sample_traffic_tasks(1, seed=707, tls_counts=(11,))[0]
    bad_pol = np.zeros(prob.d)
    rcpt = verify_and_deploy_shortlist(prob, [("posterior_primary", bad_pol)], seed=99)
    if not rcpt.certified:
        assert rcpt.decision_state == "abstention"
        assert rcpt.deployed_policy is None
        assert rcpt.selected_uncertified_candidate is not None
        assert rcpt.primary_policy.shape == (prob.d,)


def test_genuine_variance_shift_immunity_without_artificial_multiplier() -> None:
    """Verify `within_ssq` vs `naive_prior_residual_sq` using the exact same NNLS (no multiplier!)."""
    src_tasks = sample_queueing_tasks(2, seed=303, horizons=(16,))
    records, _ = generate_legacy_archive(
        src_tasks,
        solvers=("penalty_spsa", "constrained_cmaes"),
        per_solver_budget=20,
        reps_per_eval=2,
        seed=404,
    )
    prior = DebiasedSpectralMetaPrior(active_rank=6).fit(records)
    tgt = sample_queueing_tasks(1, seed=505, horizons=(24,))[0]

    m_within = TemperedHierarchicalHVDModel(prior, mode="tempered", variance_estimator="within_ssq")
    m_naive = TemperedHierarchicalHVDModel(prior, mode="tempered", variance_estimator="naive_prior_residual_sq")

    rng = np.random.default_rng(606)
    for _ in range(5):
        x = rng.uniform(0.1, 0.9, size=tgt.d)
        exp = tgt.nominal_exposure(x)
        fake_samples = np.column_stack([
            1.5 + rng.normal(0.0, 0.02, size=3),
            2.0 + rng.normal(0.0, 0.01, size=3),
        ])
        rec = SufficientStatsRecord.from_samples(x, fake_samples, exp)
        m_within.observe(rec)
        m_naive.observe(rec)

    cands = [rng.uniform(0.1, 0.9, size=tgt.d) for _ in range(48)]
    exps = [tgt.nominal_exposure(c) for c in cands]
    t0 = time.perf_counter()
    p_within = m_within.predict_joint(cands, exps)
    scores = score_gate_power_vkg(p_within, alpha=0.05, status_quo_obj=2.5)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    p_naive = m_naive.predict_joint(cands, exps)
    assert elapsed_ms < 400.0
    assert scores.shape == (48,) and np.all(np.isfinite(scores))
    assert float(np.mean(p_naive.aleatoric_var_con)) > 8.0 * float(np.mean(p_within.aleatoric_var_con))
    assert float(np.mean(p_within.aleatoric_var_con)) < 0.02


def test_all_methods_and_p1_ladder_micro_wiring() -> None:
    """Verify micro-wiring of P1 mechanism switches and faithful external baselines."""
    prob = sample_traffic_tasks(1, seed=707, tls_counts=(11,))[0]
    x_infeas = np.zeros(prob.d)
    x_safe = prob.domain_heuristic_anchor()
    r_infeas = SufficientStatsRecord.from_samples(
        x_infeas, np.array([[0.50, 0.45], [0.52, 0.40]]), prob.nominal_exposure(x_infeas)
    )
    r_safe = SufficientStatsRecord.from_samples(
        x_safe, np.array([[0.95, -0.25], [0.97, -0.22]]), prob.nominal_exposure(x_safe)
    )
    chosen = select_empirical_feasible_incumbent([r_infeas, r_safe], alpha=0.05)
    assert np.allclose(chosen, x_safe)

    ok_safe, p_safe, _, _, k_c = exact_binomial_gate(np.full(112, -0.1), alpha=0.05, delta_slot=0.025)
    assert ok_safe and p_safe < 0.005 and k_c == 1
    ok_g, _, _, _ = gaussian_noncentral_t_gate(np.full(80, -0.30) + np.linspace(-0.02, 0.02, 80), alpha=0.05, delta_slot=0.025)
    assert ok_g

    src_tr = sample_traffic_tasks(2, seed=808, tls_counts=(11,))
    arc, _ = generate_legacy_archive(
        src_tr, solvers=("penalty_spsa", "heuristic_powell"), per_solver_budget=12, reps_per_eval=2, seed=809
    )
    prior = DebiasedSpectralMetaPrior(active_rank=4).fit(arc)

    # P1 Arm C (Source basis + weak prior) & Arm D (Source basis + source coef prior, target HVD only)
    res_c = run_tls_olh_kg(
        prob,
        prior,
        n_online_steps=2,
        n0_init=4,
        batch_reps=2,
        pool_size=24,
        seed=11,
        model_mode="source_basis_weak_prior",
        init_mode="source_warmstart",
        proposal_mode="local_dct_only",
        use_source_basis=True,
        use_source_coef_prior=False,
        use_source_hvd_prior=False,
        acquisition_type="gate_power_ts",
    )
    res_d = run_tls_olh_kg(
        prob,
        prior,
        n_online_steps=2,
        n0_init=4,
        batch_reps=2,
        pool_size=24,
        seed=11,
        model_mode="tempered",
        init_mode="source_warmstart",
        proposal_mode="local_dct_only",
        use_source_basis=True,
        use_source_coef_prior=True,
        use_source_hvd_prior=False,
        acquisition_type="gate_power_ts",
    )
    res_g = run_warmstart_pattern_search_baseline(prob, prior, n_online_steps=2, n0_init=4, batch_reps=2, seed=11)
    res_scbo = run_scbo_baseline(prob, n_online_steps=2, n0_init=4, batch_reps=2, pool_size=24, seed=11)
    res_saas = run_saasbo_baseline(prob, n_online_steps=2, n0_init=4, batch_reps=2, pool_size=24, seed=11)
    res_het = run_hetgp_kg_baseline(prob, n_online_steps=2, n0_init=4, batch_reps=2, pool_size=24, seed=11)
    res_rgpe = run_rgpe_transfer_baseline(prob, arc, n_online_steps=2, n0_init=4, batch_reps=2, pool_size=24, seed=11)
    res_hyp = run_hyperbo_transfer_baseline(prob, arc, n_online_steps=2, n0_init=4, batch_reps=2, pool_size=24, seed=11)
    res_spsa = run_direct_legacy_baseline(prob, solver_name="penalty_spsa", n_online_steps=2, n0_init=4, batch_reps=2, seed=11)

    for res in (res_c, res_d, res_g, res_scbo, res_saas, res_het, res_rgpe, res_hyp, res_spsa):
        assert res.search_calls_used == 12
        if res.certified:
            assert res.deployed_policy is not None and res.deployed_policy.shape == (prob.d,)
            assert res.receipt.decision_state == "certified_deployment"
        else:
            assert res.deployed_policy is None
            assert res.receipt.decision_state == "abstention"

    markets = discover_valid_opsd_markets(d=168)
    assert len(markets) >= 10

    # Verify `tempered_adaptive` releases source prior (`beta == 0.0`) under severe target shift
    m_adapt = TemperedHierarchicalHVDModel(prior, mode="tempered_adaptive")
    rng = np.random.default_rng(919)
    for k in range(12):
        xk = rng.uniform(0.1, 0.9, size=prob.d)
        exp_k = prob.nominal_exposure(xk)
        # Inject large linear + offset shift conflicting with source prior slope
        y_shift = 25.0 - 8.0 * float(exp_k.psi[0])
        samps = np.column_stack([
            y_shift + rng.normal(0.0, 0.01, size=3),
            -0.5 + rng.normal(0.0, 0.01, size=3),
        ])
        m_adapt.observe(SufficientStatsRecord.from_samples(xk, samps, exp_k))
    assert m_adapt.beta_obj_ == 0.0
    pred_ad = m_adapt.predict_joint([x_safe], [prob.nominal_exposure(x_safe)])
    assert np.all(np.isfinite(pred_ad.mean_obj))


def test_u1_scale_equivariance_full_release_on_psi_and_messy_archive() -> None:
    """Verify objective scale-equivariance, `tempered_full_release`, `*-on-Psi`, and `n_reps=1` messy archives."""
    from dataclasses import replace as dc_replace
    from v2_or.legacy_solvers import run_constrained_cmaes, run_penalty_spsa
    from v2_or.verification import build_feasible_aware_shortlist

    prob = sample_inventory_tasks(1, seed=1201, horizons=(16,))[0]
    rng = np.random.default_rng(1202)
    cands = [prob.domain_heuristic_anchor()] + [rng.uniform(0.1, 0.9, size=prob.d) for _ in range(15)]
    exps = [prob.nominal_exposure(c) for c in cands]

    # 1. Exact affine objective scale equivariance (OPSD 0.05x scale vs Inventory 50x scale)
    m_tgt = TemperedHierarchicalHVDModel(None, mode="target_only")
    recs = []
    for c in cands[:6]:
        s = prob.simulate_replications(c, 2, rng, split="search")
        rec = SufficientStatsRecord.from_samples(c, s, prob.nominal_exposure(c))
        recs.append(rec)
        m_tgt.observe(rec)

    pred_base = m_tgt.predict_joint(cands[:6], exps[:6])
    scale = 1e-3
    pred_tiny = dc_replace(
        pred_base,
        mean_obj=pred_base.mean_obj * scale,
        cov_obj=pred_base.cov_obj * (scale ** 2),
        std_obj=pred_base.std_obj * scale,
        aleatoric_var_obj=pred_base.aleatoric_var_obj * (scale ** 2),
    )
    sq_base = float(recs[0].mean_obj)
    sc_base = score_gate_power_vkg(pred_base, alpha=0.05, status_quo_obj=sq_base)
    sc_tiny = score_gate_power_vkg(pred_tiny, alpha=0.05, status_quo_obj=sq_base * scale)
    assert np.allclose(sc_tiny, sc_base * scale, rtol=1e-6, atol=1e-10)
    assert int(np.argmax(sc_base)) == int(np.argmax(sc_tiny))

    recs_tiny = [dc_replace(r, mean_obj=r.mean_obj * scale, ssq_obj=r.ssq_obj * (scale ** 2)) for r in recs]
    sl_base = build_feasible_aware_shortlist(
        candidate_pool=cands[:6],
        mean_obj=pred_base.mean_obj,
        mean_con=pred_base.mean_con,
        std_con=pred_base.std_con,
        ucb_aleat_var_con=pred_base.ucb_aleatoric_var_con,
        r_perp_con=pred_base.r_perp_con,
        initial_records=recs,
        alpha=0.05,
        reference_obj=sq_base,
    )
    sl_tiny = build_feasible_aware_shortlist(
        candidate_pool=cands[:6],
        mean_obj=pred_tiny.mean_obj,
        mean_con=pred_tiny.mean_con,
        std_con=pred_tiny.std_con,
        ucb_aleat_var_con=pred_tiny.ucb_aleatoric_var_con,
        r_perp_con=pred_tiny.r_perp_con,
        initial_records=recs_tiny,
        alpha=0.05,
        reference_obj=sq_base * scale,
    )
    for (_, xb), (_, xt) in zip(sl_base, sl_tiny):
        assert np.allclose(xb, xt)

    # 2. Messy archive with 50%+ single-rep (`n_reps=1, ssq=0`) & mismatched `solver_alpha`
    src_tasks = sample_inventory_tasks(2, seed=1301, horizons=(16,))
    messy_recs = []
    for idx_t, st in enumerate(src_tasks):
        r1 = run_penalty_spsa(st, budget_calls=14, seed=1400 + idx_t, reps_schedule=[1, 1, 2], solver_alpha=0.10)
        r2 = run_constrained_cmaes(st, budget_calls=14, seed=1500 + idx_t, reps_schedule=[1, 1, 3], solver_alpha=0.02)
        messy_recs.extend(r1.trajectory_log)
        messy_recs.extend(r2.trajectory_log)
    assert any(r.n_reps == 1 and r.ssq_con == 0.0 for r in messy_recs)
    prior_messy = DebiasedSpectralMetaPrior(active_rank=4).fit(messy_recs)

    # 3. `tempered_full_release` + `hybrid_warmstart` + `adaptive_release` + `SCBO-on-Psi` + `RGPE-on-Psi`
    prob_eval = sample_inventory_tasks(1, seed=1601, horizons=(20,))[0]
    res_full = run_tls_olh_kg(
        prob_eval,
        prior_messy,
        n_online_steps=2,
        n0_init=4,
        batch_reps=2,
        pool_size=24,
        seed=21,
        model_mode="tempered_full_release",
        init_mode="hybrid_warmstart",
        proposal_mode="adaptive_release",
        acquisition_type="gate_power_vkg",
        nomination_mode="cascade_bellman_3slot",
    )
    res_scbo_psi = run_scbo_on_psi_baseline(prob_eval, n_online_steps=2, n0_init=4, batch_reps=2, pool_size=24, seed=21)
    res_rgpe_psi = run_rgpe_on_psi_baseline(
        prob_eval, messy_recs, n_online_steps=2, n0_init=4, batch_reps=2, pool_size=24, seed=21
    )
    for r_item in (res_full, res_scbo_psi, res_rgpe_psi):
        assert r_item.search_calls_used == 12

    # 4. Verify `tempered_full_release` triggers `is_source_released_ == True` under severe shift
    m_rel = TemperedHierarchicalHVDModel(prior_messy, mode="tempered_full_release")
    for k in range(10):
        xk = rng.uniform(0.1, 0.9, size=prob_eval.d)
        exp_k = prob_eval.nominal_exposure(xk)
        samps = np.column_stack([
            40.0 - 12.0 * float(exp_k.psi[0]) + rng.normal(0.0, 0.01, size=2),
            -0.6 + rng.normal(0.0, 0.01, size=2),
        ])
        m_rel.observe(SufficientStatsRecord.from_samples(xk, samps, exp_k))
    assert m_rel.is_source_released_ is True
