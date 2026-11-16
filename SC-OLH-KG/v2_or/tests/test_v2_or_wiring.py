"""Deterministic local wiring & zero-leakage unit tests for `SC-OLH-KG/v2_or`.

Strictly lightweight (< 5s single-threaded) per `AGENTS.md`: tests mathematical
identities, zero-leakage invariance, cross-horizon compatibility, variance-blowup
immunity, and single-step speed without running multi-process experiments.
"""

from __future__ import annotations

from dataclasses import replace
import time

import numpy as np

from v2_or.acquisition import run_tls_olh_kg, score_verification_kg
from v2_or.baselines import (
    run_direct_legacy_baseline,
    run_hetgp_kg_baseline,
    run_hyperbo_transfer_baseline,
    run_rgpe_transfer_baseline,
    run_saasbo_baseline,
    run_scbo_baseline,
)
from v2_or.contracts import LegacyTrajectoryRecord, SufficientStatsRecord
from v2_or.legacy_solvers import generate_legacy_archive
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
    exact_binomial_gate,
    gaussian_noncentral_t_gate,
    select_feasible_aware_initial_incumbent,
)


def test_zero_oracle_leakage_invariance() -> None:
    """Mutating private task shock/cost laws MUST leave `nominal_exposure(x)` bit-identical."""
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
    assert np.max(np.abs(tr_a.nominal_exposure(x_tr).psi - tr_b.nominal_exposure(x_tr).psi)) == 0.0

    # 4. Real-data OPSD Storage (verify zero outcome calls during nominal_exposure)
    opsd = OPSDGridStorageBenchmark(market="DK_1", d=96)
    x_op = rng.uniform(0.0, 1.0, size=opsd.d)
    exp_op = opsd.nominal_exposure(x_op)
    assert exp_op.psi.shape == (12,)
    assert opsd.search_calls == 0


def test_cross_horizon_and_theorem1_debiasing() -> None:
    """Verify cross-dimension `d_s != d_t` legacy archive fitting & Gram conditioning."""
    src_tasks = sample_inventory_tasks(2, seed=101, horizons=(16, 24))
    assert src_tasks[0].d != src_tasks[1].d

    records, meta = generate_legacy_archive(
        src_tasks,
        solvers=("penalty_spsa", "heuristic_powell"),
        per_solver_budget=18,
        reps_per_eval=2,
        seed=202,
    )
    assert len(records) >= 14
    assert meta["total_simulation_calls"] <= 2 * 2 * 18

    prior = DebiasedSpectralMetaPrior(active_rank=6).fit(records)
    assert prior.diagnostics_ is not None
    assert abs(prior.diagnostics_.debiased_orth_gram_cond - 1.0) < 1e-6

    # Transform exposures from an unseen larger horizon H = 48 (d = 96)
    tgt = sample_inventory_tasks(1, seed=999, horizons=(48,))[0]
    exps = [tgt.nominal_exposure(tgt.domain_heuristic_anchor())]
    phi, psi_std, u_hvd, r_f, r_g = prior.transform_exposures(exps)
    assert phi.shape == (1, 1 + prior.diagnostics_.active_rank)
    assert np.all(r_f > 0.0) and np.all(r_g > 0.0)
    assert u_hvd.shape[0] == 1


def test_theorem2_variance_blowup_immunity_and_speed() -> None:
    """Verify decoupled HVD stays immune to huge mean shift `Delta_g`, unlike untempered FULL."""
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
    m_temp = TemperedHierarchicalHVDModel(prior, mode="tempered")
    m_full = TemperedHierarchicalHVDModel(prior, mode="untempered_full")

    rng = np.random.default_rng(606)
    for _ in range(5):
        x = rng.uniform(0.1, 0.9, size=tgt.d)
        exp = tgt.nominal_exposure(x)
        # Huge constraint mean shift (+2.0) with tiny true within-policy std (0.01)
        fake_samples = np.column_stack([
            1.5 + rng.normal(0.0, 0.02, size=3),
            2.0 + rng.normal(0.0, 0.01, size=3),
        ])
        rec = SufficientStatsRecord.from_samples(x, fake_samples, exp)
        m_temp.observe(rec)
        m_full.observe(rec)

    cands = [rng.uniform(0.1, 0.9, size=tgt.d) for _ in range(64)]
    exps = [tgt.nominal_exposure(c) for c in cands]

    t0 = time.perf_counter()
    p_temp = m_temp.predict_joint(cands, exps)
    scores = score_verification_kg(p_temp, alpha=0.05)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    p_full = m_full.predict_joint(cands, exps)
    assert elapsed_ms < 80.0, f"Single-step V-KG took {elapsed_ms:.1f} ms"
    assert scores.shape == (64,) and np.all(np.isfinite(scores))
    # Tempered model keeps aleatoric variance near true 1e-4..1e-2, whereas untempered inflates >10x!
    assert float(np.mean(p_full.aleatoric_var_con)) > 8.0 * float(np.mean(p_temp.aleatoric_var_con))
    assert float(np.mean(p_temp.aleatoric_var_con)) < 0.02


def test_theorem3_verifier_and_all_methods_micro_wiring() -> None:
    """Verify Slot-2 feasibility guard, exact gates, and micro-wiring of all methods."""
    # 1. Exact Binomial gate: 0 violations out of 112 at alpha=0.05 -> p = 0.95^112 = 0.00319 <= 0.025
    ok_safe, p_safe, _ = exact_binomial_gate(np.full(112, -0.1), alpha=0.05, delta_slot=0.025)
    ok_bad, p_bad, _ = exact_binomial_gate(np.r_[np.full(100, -0.1), np.full(12, 0.2)], alpha=0.05, delta_slot=0.025)
    assert ok_safe and p_safe < 0.005
    assert (not ok_bad) and p_bad > 0.50

    # 2. Gaussian noncentral-t gate
    rng = np.random.default_rng(77)
    ok_g, _, _ = gaussian_noncentral_t_gate(rng.normal(-0.30, 0.05, size=80), alpha=0.05, delta_slot=0.025)
    bad_g, _, _ = gaussian_noncentral_t_gate(rng.normal(0.02, 0.05, size=80), alpha=0.05, delta_slot=0.025)
    assert ok_g and (not bad_g)

    # 3. Feasible-aware Slot 2 incumbent selection (must reject infeasible lower-objective point!)
    prob = sample_traffic_tasks(1, seed=707, tls_counts=(11,))[0]
    x_infeas = np.zeros(prob.d)
    x_safe = prob.domain_heuristic_anchor()
    r_infeas = SufficientStatsRecord.from_samples(
        x_infeas, np.array([[0.50, 0.45], [0.52, 0.40]]), prob.nominal_exposure(x_infeas)
    )
    r_safe = SufficientStatsRecord.from_samples(
        x_safe, np.array([[0.95, -0.25], [0.97, -0.22]]), prob.nominal_exposure(x_safe)
    )
    chosen = select_feasible_aware_initial_incumbent([r_infeas, r_safe], alpha=0.05)
    assert np.allclose(chosen, x_safe)

    # 4. Micro 2-step wiring smoke across TLS-OLH-KG and SOTA baselines
    src_tr = sample_traffic_tasks(2, seed=808, tls_counts=(11,))
    arc, _ = generate_legacy_archive(src_tr, solvers=("penalty_spsa", "heuristic_powell"), per_solver_budget=12, reps_per_eval=2, seed=809)
    prior = DebiasedSpectralMetaPrior(active_rank=4).fit(arc)

    res_ours = run_tls_olh_kg(prob, prior, n_online_steps=2, n0_init=4, batch_reps=2, pool_size=32, seed=11)
    res_scbo = run_scbo_baseline(prob, n_online_steps=2, n0_init=4, batch_reps=2, pool_size=32, seed=11)
    res_saas = run_saasbo_baseline(prob, n_online_steps=2, n0_init=4, batch_reps=2, pool_size=32, seed=11)
    res_het = run_hetgp_kg_baseline(prob, n_online_steps=2, n0_init=4, batch_reps=2, pool_size=32, seed=11)
    res_rgpe = run_rgpe_transfer_baseline(prob, arc, n_online_steps=2, n0_init=4, batch_reps=2, pool_size=32, seed=11)
    res_hyp = run_hyperbo_transfer_baseline(prob, arc, n_online_steps=2, n0_init=4, batch_reps=2, pool_size=32, seed=11)
    res_spsa = run_direct_legacy_baseline(prob, solver_name="penalty_spsa", n_online_steps=2, n0_init=4, batch_reps=2, seed=11)

    for res in (res_ours, res_scbo, res_saas, res_het, res_rgpe, res_hyp, res_spsa):
        assert res.search_calls_used == 12
        assert res.deployed_policy.shape == (prob.d,)

    markets = discover_valid_opsd_markets(d=168)
    assert len(markets) >= 10
