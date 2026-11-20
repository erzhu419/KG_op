"""Deterministic wiring checks for the prospectively matched comparisons."""

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_allclose

from v2_or.acquisition.verification_kg import (
    _observed_variances,
    run_tls_olh_kg,
    score_chance_feasibility_kg,
    score_gate_power_vkg,
)
from v2_or.contracts import BaseDynamicORProblem, NominalTrajectoryExposure, SufficientStatsRecord
from v2_or.models.tempered_hvd_posterior import JointPosteriorPrediction, TemperedHierarchicalHVDModel


class ProgrammedProblem(BaseDynamicORProblem):
    d = 6

    def __init__(self, offset):
        super().__init__()
        self.offset = offset
        self.query_path = []

    def nominal_exposure(self, x_norm):
        return NominalTrajectoryExposure(np.r_[x_norm[:2], np.zeros(10)], np.ones(1), np.ones(1))

    def _simulate_raw(self, x_norm, reps, rng, split):
        if split == "search":
            self.query_path.append(x_norm.copy())
        noise = rng.standard_normal((reps, 2))
        return np.column_stack((1.0 + x_norm[0] + 0.05 * noise[:, 0],
                                self.offset + 0.01 * x_norm[1] + 0.015 * noise[:, 1]))


class FrozenSubstitute:
    is_fitted = True
    mu_0_obj_ = np.array([1.0, 1.0, 0.0])
    mu_0_con_ = np.array([-0.10, 0.0, 0.01])
    cov_0_obj_ = np.array([[0.01, 0.008, 0.0], [0.008, 0.02, 0.0], [0.0, 0.0, 0.005]])
    cov_0_con_ = cov_0_obj_ / 100.0

    @staticmethod
    def transform_exposures(exposures):
        points = np.array([e.psi[:2] for e in exposures])
        levels = np.ones(len(exposures))
        return np.column_stack([levels, points]), points, levels[:, None], levels, levels


def _model(kind):
    return TemperedHierarchicalHVDModel(
        FrozenSubstitute(), mode="matched_coefficients", coefficient_prior=kind,
        use_source_basis=True, use_source_hvd_prior=False, use_empirical_r_perp=False,
    )


def _initial_record(x, responses):
    xp = np.full(6, x)
    return SufficientStatsRecord.from_samples(xp, np.array(responses), ProgrammedProblem(0.0).nominal_exposure(xp))


def test_full_and_diagonal_share_coefficient_marginals_and_frozen_scales():
    records = [_initial_record(0.2, [[1, -1], [2, 0]]), _initial_record(0.8, [[1, 1], [2, 2]])]
    models = [_model(kind) for kind in ("weak_target", "historical_diagonal", "historical_full")]
    for model in models:
        for batch in records:
            model.observe(batch)
        model.freeze_matched_scales(records)
    for head in ("obj", "con"):
        prior_diagonal, prior_full = [m._prior_moments_head(3, records, head) for m in models[1:]]
        assert_array_equal(prior_diagonal[0], prior_full[0])
        assert_array_equal(np.diag(prior_diagonal[1]), np.diag(prior_full[1]))
        assert np.count_nonzero(prior_diagonal[1]) == 3
        assert np.count_nonzero(prior_full[1]) > 3
        frozen = getattr(models[0], f"tau_shift_{head}_")
        assert all(getattr(m, f"tau_shift_{head}_") == frozen for m in models)
        for m in models:
            m.observe(_initial_record(0.3, [[1, 100], [2, 110]]))
            assert getattr(m, f"tau_shift_{head}_") == frozen
    candidate = np.full(6, 0.5)
    exposures = [ProgrammedProblem(0.0).nominal_exposure(candidate)]
    predictions = [m.predict_joint([candidate], exposures) for m in models]
    assert_allclose([p.aleatoric_var_con for p in predictions], [predictions[0].aleatoric_var_con] * 3)
    assert not np.allclose(predictions[1].cov_obj, predictions[2].cov_obj)


@pytest.mark.parametrize("offset, expect_weak", [(-0.10, False), (1.0, True)])
def test_prospective_release_reproduces_exact_selected_paid_path(offset, expect_weak):
    common = dict(
        n_online_steps=3, n0_init=8, batch_reps=2, seed=991, model_mode="matched_coefficients",
        acquisition_type="gate_power_vkg", init_mode="cold_dct", proposal_mode="local_dct_only",
        use_source_basis=True, use_source_hvd_prior=False, use_empirical_r_perp=False, pool_size=50,
    )
    runs = {}
    for kind in ("weak_target", "historical_full", "initial_release"):
        prob = ProgrammedProblem(offset)
        run = run_tls_olh_kg(prob, FrozenSubstitute(), coefficient_prior=kind, **common)
        runs[kind] = run, prob.query_path
    evidence = [r.mechanism_receipt["early_release"] for r, _ in runs.values()]
    assert all(e["recommend_weak"] == expect_weak for e in evidence)
    assert len({e["log_ratio_weak_over_full"] for e in evidence}) == 1
    adaptive, adaptive_path = runs["initial_release"]
    chosen, chosen_path = runs["weak_target" if expect_weak else "historical_full"]
    assert_array_equal(adaptive_path, chosen_path)
    assert adaptive.receipt.certificates == chosen.receipt.certificates
    assert adaptive.total_target_calls == chosen.total_target_calls
    assert adaptive.mechanism_receipt["variance"] == chosen.mechanism_receipt["variance"]


def test_pooled_variance_keeps_between_batch_term_separate():
    first, further = [_initial_record(0.2, [[1, a], [1, a + 2]]) for a in (-1, 4)]
    model = _model("weak_target")
    for record in (first, further):
        model.observe(record)
    rcpt = SimpleNamespace(primary_policy=first.x_norm, deployed_policy=None)
    result = _observed_variances([first, further], model, rcpt)
    assert result["primary"]["fixed_batch"] == {"dof": 2, "obj_var": 0.0, "con_var": 2.0}
    assert result["primary"]["pooled"]["con_var"] == pytest.approx(29 / 3)
    assert result["revisited"]["n_batches"] == 2


def test_matched_chance_feasibility_uses_common_deterministic_screen():
    identity = np.eye(2)
    pred = JointPosteriorPrediction(
        mean_obj=np.array([0.0, 2.0]), cov_obj=identity, std_obj=np.ones(2),
        mean_con=np.array([-1.0, -1.0]), cov_con=identity, std_con=np.ones(2),
        aleatoric_var_obj=np.ones(2), aleatoric_var_con=np.ones(2),
        ucb_aleatoric_var_con=np.ones(2), r_perp_con=np.zeros(2),
    )
    gate = score_gate_power_vkg(pred, status_quo_obj=1.0, max_eval_targets=1, common_target_screen=True)
    chance = score_chance_feasibility_kg(pred, status_quo_obj=1.0, max_eval_targets=1)
    assert gate[1] == chance[1] == 0.0
    assert gate[0] > 0.0 and chance[0] > 0.0 and gate[0] != chance[0]
