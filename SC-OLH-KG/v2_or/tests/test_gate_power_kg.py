"""Small deterministic cases for the gate-power KG quadrature."""

from dataclasses import replace

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from v2_or.acquisition.verification_kg import (
    _GH5_P,
    _align_fantasy_gate_power,
    score_gate_power_vkg,
)
from v2_or.models.tempered_hvd_posterior import JointPosteriorPrediction


def prediction(mean_obj, mean_con, cov_obj, cov_con, noise):
    mean_obj = np.asarray(mean_obj, dtype=float)
    cov_obj = np.asarray(cov_obj, dtype=float)
    cov_con = np.asarray(cov_con, dtype=float)
    variance = np.full(mean_obj.size, noise, dtype=float)
    return JointPosteriorPrediction(
        mean_obj=mean_obj,
        cov_obj=cov_obj,
        std_obj=np.sqrt(np.diag(cov_obj)),
        mean_con=np.asarray(mean_con, dtype=float),
        cov_con=cov_con,
        std_con=np.sqrt(np.diag(cov_con)),
        aleatoric_var_obj=variance,
        aleatoric_var_con=variance,
        ucb_aleatoric_var_con=variance,
        r_perp_con=np.zeros(mean_obj.size),
    )


def test_fantasy_gate_mean_alignment_is_bounded_and_order_preserving():
    # Two queries, five quadrature nodes, four candidate targets. Include
    # both mean correction directions and raw distributions at either endpoint.
    raw = np.broadcast_to(np.linspace(0.1, 0.9, 5)[None, :, None], (2, 5, 4)).copy()
    raw[0, :, 2] = 0.0
    raw[1, :, 3] = 1.0
    current = np.array([0.1, 0.9, 0.0, 1.0])
    aligned = _align_fantasy_gate_power(raw, current)
    assert np.all((aligned >= 0.0) & (aligned <= 1.0))
    assert np.all(np.diff(aligned, axis=1) >= 0.0)
    assert_allclose(
        np.einsum("qge,g->qe", aligned, _GH5_P), np.tile(current, (2, 1)), atol=1e-15, rtol=0.0
    )


def test_kg_does_not_reward_the_unrelated_query_in_audit_counterexample():
    # Only the first evaluation target is screened in. The second query has
    # zero covariance with it. The old score ranked the second query first.
    pred = prediction([0.0, 2.0], [-1.0, -1.0], np.eye(2), np.eye(2), 1.0)
    scores = score_gate_power_vkg(pred, status_quo_obj=1.0, max_eval_targets=1)
    assert scores[1] == 0.0
    assert scores[0] > 0.0


def test_one_target_with_constant_objective_has_zero_gate_only_kg():
    pred = prediction([1.0], [-0.10], [[0.0]], [[0.06**2]], 0.003)
    assert_array_equal(score_gate_power_vkg(pred, status_quo_obj=2.0), [0.0])
    assert_array_equal(
        score_gate_power_vkg(replace(pred, mean_con=np.array([-0.06])), status_quo_obj=2.0),
        [0.0],
    )


def test_gate_information_can_change_the_preferred_target():
    # The current targets are tied. Observing either independent constraint
    # reveals whether to prefer that target over the unchanged alternative.
    pred = prediction([1.0, 1.0], [-0.10, -0.10], np.zeros((2, 2)), np.eye(2) * 0.06**2, 0.003)
    scores = score_gate_power_vkg(pred, status_quo_obj=2.0)
    assert np.all(scores > 0.0)
    assert_allclose(scores[0], scores[1], atol=1e-15, rtol=0.0)
    assert_array_equal(
        score_gate_power_vkg(pred, status_quo_obj=2.0, verif_budget_slot0=1),
        [0.0, 0.0],
    )
