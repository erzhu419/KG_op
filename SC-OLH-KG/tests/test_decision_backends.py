from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from acquisition.decision_backends import (  # noqa: E402
    minimization_expected_improvement,
    normal_positive_part,
    score_decision_backend,
)
from algorithms.single_olhkg import (  # noqa: E402
    SingleOLHKGAlgorithm,
    SingleOLHKGConfig,
)
from problems.rzdt import RZDT1  # noqa: E402
from problems.single_objective import ScalarizedProblem  # noqa: E402


class DummyGPR:
    def __init__(self, sign=1.0):
        self.sign = float(sign)
        self.a = np.array([0.1, 0.2 * self.sign, -0.1 * self.sign])
        self.C = np.diag([0.02, 0.05, 0.03])
        self.lambda_i = 0.01
        self.sol_to_idx = {}

    @staticmethod
    def augmented_feature_matrix(candidates):
        x = np.asarray(candidates, dtype=float) / 10.0
        return np.column_stack([np.ones(len(x)), x[:, :2]])

    def posterior_mean_many(self, candidates):
        return self.augmented_feature_matrix(candidates) @ self.a

    def posterior_var_many(self, candidates):
        phi = self.augmented_feature_matrix(candidates)
        return np.einsum("ij,jk,ik->i", phi, self.C, phi) + self.lambda_i


class DummyVariance:
    @staticmethod
    def predict_certification_variance_many(output_index, candidates, problem):
        del output_index, problem
        x = np.asarray(candidates, dtype=float)
        return 0.01 + 0.001 * x[:, 0]


class DummyProblem:
    d = 2
    alpha = 0.05
    tau = 0.5

    @staticmethod
    def int_bounds():
        return np.zeros(2, dtype=int), np.full(2, 10, dtype=int)

    @staticmethod
    def source_mean_prior_predict_many(candidates, output_index=1):
        x = np.asarray(candidates, dtype=float) / 10.0
        if int(output_index) == 0:
            return 0.1 + 0.3 * x[:, 0]
        return 0.2 + 0.2 * x[:, 1]


def _score(name, seed=17):
    candidates = [(0, 0), (2, 8), (6, 3), (10, 10)]
    return score_decision_backend(
        name,
        candidates,
        DummyGPR(sign=1.0),
        DummyGPR(sign=-1.0),
        DummyVariance(),
        DummyProblem(),
        observed=[((0, 0), np.array([0.0, 0.0]))],
        rng=np.random.default_rng(seed),
        iteration=3,
        seed=seed,
        risk_penalty=5.0,
    )


def test_gaussian_loss_primitives_are_finite_and_nonnegative():
    positive = normal_positive_part(
        np.array([-1.0, 0.0, 1.0]), np.array([0.5, 0.5, 0.5]))
    improvement = minimization_expected_improvement(
        0.0, np.array([-1.0, 0.0, 1.0]), np.array([0.5, 0.5, 0.5]))
    assert np.all(np.isfinite(positive))
    assert np.all(positive >= 0.0)
    assert np.all(np.isfinite(improvement))
    assert np.all(improvement >= 0.0)


def test_all_nonlegacy_backends_return_finite_shared_posterior_scores():
    for name in (
        "n0_best",
        "random",
        "sobol",
        "risk_ts",
        "bayes_risk_ei",
        "constrained_ei",
        "transfer_utility",
    ):
        result = _score(name)
        assert result["backend"] == name
        assert result["total"].shape == (4,)
        assert np.all(np.isfinite(result["total"]))
        assert np.all(result["constraint_aleatoric"] > 0.0)
        assert result["posterior_source"] == "single_cumulative_hvd"


def test_risk_aware_thompson_sampling_is_reproducible_for_fixed_seed():
    first = _score("risk_ts", seed=81)
    second = _score("risk_ts", seed=81)
    different = _score("risk_ts", seed=82)
    np.testing.assert_allclose(first["total"], second["total"])
    assert not np.allclose(first["total"], different["total"])


def test_transfer_utility_declares_frozen_source_prior_use():
    result = _score("transfer_utility")
    assert result["transfer_utility_status"] == "source_mean_prior_active"
    assert np.all(np.isfinite(result["transfer_utility"]))


def test_risk_ts_replaces_exact_kg_without_changing_target_budget():
    problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
    algorithm = SingleOLHKGAlgorithm(
        problem,
        SingleOLHKGConfig(
            N=3,
            n0=2,
            K1=2,
            K2=0,
            decision_backend="risk_ts",
            exact_kg_mc_samples=2,
            use_state_coupling=False,
            use_state_basis=False,
            use_problem_initial_samples=False,
            use_boundary_initial_samples=False,
            use_recommendation_refinement=False,
            recommendation_axis_oracle=False,
            recommendation_calibration=False,
            finalist_replication_budget=0,
            eval_pool_size=8,
            evaluate_interval=0,
            seed=91,
        ),
    )
    result = algorithm.run(verbose=False)
    assert result["n_simulations"] == 3
    assert len(algorithm.iteration_log) == 1
    row = algorithm.iteration_log[0]
    assert row["decision_backend"] == "risk_ts"
    assert "exact_kg_mc_samples" not in row
    assert row["selection_policy"] == "acquisition"
    assert result["decision_backend_terminal_used"] is True
    assert result["decision_backend_contract"]["coherent"] is True
    assert result["adaptive_outcome_audit"][
        "target_oracle_used_for_decision"] is False
    assert result["certificate_outcome_audit"]["status"] == "audited"
    assert result["certificate_outcome_audit"][
        "target_oracle_used_for_decision"] is False
