"""Wiring tests for search updates, eligibility, and the joint value brackets."""
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose
from scipy.special import ndtr, roots_hermitenorm
from scipy.stats import binom

from score_eligible_prefix import (
    first_and_envelope, integrate_full, native_draw, paired_fantasies, prove_gate_counts, RosterDraw,
)
from score_cascade_prefix import SIZES
from v2_or.contracts import NominalTrajectoryExposure, SufficientStatsRecord
from v2_or.models.tempered_hvd_posterior import TemperedHierarchicalHVDModel


def test_batch_fantasy_preserves_sample_dispersion_and_violation_count():
    prediction = SimpleNamespace(
        mean_obj=np.array([2.0]), mean_con=np.array([0.0]),
        cov_obj=np.array([[0.25]]), cov_con=np.array([[0.25]]),
        aleatoric_var_obj=np.array([2.0]), aleatoric_var_con=np.array([2.0]),
    )
    samples = paired_fantasies(prediction, 0, np.array([[0.0, 0.0, 1.0, 3.0]]))[0]
    exposure = NominalTrajectoryExposure([0.0], [1.0], [1.0])
    record = SufficientStatsRecord.from_samples(np.array([0.25]), samples, exposure)
    assert (record.n_reps, record.violation_count) == (2, 1)
    assert_allclose([record.mean_obj, record.mean_con, record.ssq_obj, record.ssq_con], [2.0, 0.0, 2.0, 18.0])


def test_native_nomination_admits_one_new_policy_and_does_not_mutate_history():
    def basis(exposures):
        positions = np.array([e.psi[0] for e in exposures])
        features = np.column_stack([np.ones(len(positions)), positions])
        return features, features, features, np.zeros(len(positions)), np.zeros(len(positions))

    prior = SimpleNamespace(
        is_fitted=True, mu_0_obj_=np.array([2.0, -0.25]), mu_0_con_=np.array([-0.5, -0.1]),
        cov_0_obj_=np.eye(2) * 0.02, cov_0_con_=np.eye(2) * 0.02, transform_exposures=basis,
    )
    model = TemperedHierarchicalHVDModel(
        prior, mode="matched_coefficients", coefficient_prior="historical_full",
        use_source_hvd_prior=False, use_empirical_r_perp=False,
    )

    def record(x, obj):
        exposure = NominalTrajectoryExposure([x], [1.0], [1.0])
        samples = np.array([[obj - 0.05, -1.1], [obj + 0.05, -0.9]])
        return SufficientStatsRecord.from_samples(np.array([x]), samples, exposure)

    initial = [record(0.0, 2.5), record(0.25, 2.4)]
    for item in initial:
        model.observe(item)
    model.freeze_matched_scales(initial)
    passed = native_draw(model, record(0.5, 2.0), 0.05, 2.5, 2.5, "cascade_bellman_3slot")
    revisited = native_draw(model, record(0.25, 2.0), 0.05, 2.5, 2.5, "cascade_bellman_3slot")
    assert sorted(passed.policy_indexes) == [0, 1, 2]
    assert sorted(revisited.policy_indexes) == [0, 1]
    assert model.unique_records == initial


def test_shortlist_lengths_and_signed_gain_envelopes():
    prove_gate_counts(0.05)
    mean = np.array([-1.8, -2.1, -1.5])
    cov = np.diag([0.04, 0.055, 0.03]) ** 2
    gains = np.array([-0.9, 0.2, -0.2, 0.35])
    draws = [
        RosterDraw(tuple(range(n)), n >= 2, np.where([1, n >= 2, n >= 2, n == 3], gains, 0),
                   mean[:n], cov[:n, :n], np.ones(n))
        for n in (1, 2, 3)
    ]
    evaluated = integrate_full(draws)
    first24, first48, lower, upper = first_and_envelope(draws)
    z, weights = roots_hermitenorm(64)
    p0, p1, p2 = weights @ binom.cdf(1, SIZES, ndtr(mean + z[:, None] * np.sqrt(np.diag(cov)))) / np.sqrt(2 * np.pi)
    exact_terms = np.array([p0, p0 * p1, (1 - p0) * p1, (1 - p0) * (1 - p1) * p2])
    exact = np.array([d.gains @ exact_terms for d in draws])
    assert np.all(first24 < 0) and np.all(first48 < 0)
    assert_allclose(first48[0], evaluated[:, 0])
    assert np.all(lower <= exact) and np.all(exact <= upper)
    assert_allclose(evaluated, np.broadcast_to(exact, evaluated.shape), atol=1e-3)
