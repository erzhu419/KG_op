from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import acquisition.decision_backends as decision_backends  # noqa: E402
from acquisition.decision_backends import (  # noqa: E402
    _constraint_epistemic_reduction,
    feasible_first_terminal_order,
    minimization_expected_improvement,
    normal_positive_part,
    score_decision_backend,
)
from algorithms.single_olhkg import (  # noqa: E402
    SingleOLHKGAlgorithm,
    SingleOLHKGConfig,
)
from core.designs import next_sobol_integer_candidate  # noqa: E402
from problems.rzdt import RZDT1  # noqa: E402
from problems.single_objective import ScalarizedProblem  # noqa: E402
from performance.benchmark_lodo_meta_prior import (  # noqa: E402
    post_run_variance_calibration_audit,
)


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

    @staticmethod
    def cumulative_risk_features(point, output_index=1):
        del output_index
        x = np.asarray(point, dtype=float) / 10.0
        return np.asarray([1.0, x[0] ** 2, x[1] ** 2], dtype=float)


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


def test_feasible_first_terminal_order_is_lexicographic():
    order, mode = feasible_first_terminal_order(
        [0.0, 10.0, 1.0],
        [0.90, 0.01, 0.02],
        maximum_violation_probability=0.05,
    )
    assert order.tolist() == [2, 1, 0]
    assert mode == "posterior_feasible_objective"

    order, mode = feasible_first_terminal_order(
        [0.0, 10.0, 1.0],
        [0.90, 0.50, 0.60],
        maximum_violation_probability=0.05,
    )
    assert order.tolist() == [1, 2, 0]
    assert mode == "minimum_posterior_violation"


def test_decision_central_hvd_does_not_relax_theory_margin():
    class SplitVariance(DummyVariance):
        @staticmethod
        def predict_variance_many(output_index, candidates, problem):
            del output_index, problem
            return np.full(len(candidates), 0.01, dtype=float)

        @staticmethod
        def predict_certification_variance_many(
            output_index, candidates, problem,
        ):
            del output_index, problem
            return np.full(len(candidates), 0.25, dtype=float)

    candidates = [(0, 0), (2, 8), (6, 3), (10, 10)]
    kwargs = dict(
        backend="n0_best",
        candidates=candidates,
        obj_gpr=DummyGPR(sign=1.0),
        con_gpr=DummyGPR(sign=-1.0),
        variance_model=SplitVariance(),
        problem=DummyProblem(),
        observed=[],
        seed=23,
    )
    upper = score_decision_backend(
        **kwargs, decision_aleatoric_mode="certification_upper")
    central = score_decision_backend(
        **kwargs, decision_aleatoric_mode="posterior_central")
    np.testing.assert_allclose(upper["constraint_aleatoric"], 0.25)
    np.testing.assert_allclose(central["constraint_aleatoric"], 0.01)
    np.testing.assert_allclose(central["theory_margin"], upper["theory_margin"])
    assert np.any(np.abs(
        central["bayes_risk"] - upper["bayes_risk"]) > 1e-8)


def test_failure_probability_is_the_posterior_binary_violation_loss():
    class CentralVariance(DummyVariance):
        @staticmethod
        def predict_variance_many(output_index, candidates, problem):
            return DummyVariance.predict_certification_variance_many(
                output_index, candidates, problem)

    candidates = [(0, 0), (2, 8), (6, 3), (10, 10)]
    result = score_decision_backend(
        "n0_best",
        candidates,
        DummyGPR(sign=1.0),
        DummyGPR(sign=-1.0),
        CentralVariance(),
        DummyProblem(),
        observed=[],
        seed=29,
        risk_penalty=5.0,
        decision_aleatoric_mode="posterior_central",
        violation_loss_mode="failure_probability",
    )
    np.testing.assert_allclose(
        result["violation_loss"], result["probability_violation"])
    np.testing.assert_allclose(
        result["bayes_risk"],
        result["objective_mean"] + 5.0 * result["probability_violation"],
    )
    assert np.all(result["probability_violation"] >= 0.0)
    assert np.all(result["probability_violation"] <= 1.0)
    assert result["violation_loss_mode"] == "failure_probability"


def test_nominal_decision_mixture_does_not_relax_robust_certificate():
    class DummyTaskEnsemble:
        @staticmethod
        def mixture_moments_many(output_index, candidates, certification=False):
            del certification
            n = len(candidates)
            if int(output_index) == 0:
                return SimpleNamespace(
                    mean=np.linspace(0.1, 0.2, n),
                    epistemic=np.full(n, 0.02),
                )
            return SimpleNamespace(
                mean=np.linspace(0.05, 0.15, n),
                epistemic=np.full(n, 0.01),
                aleatoric=np.full(n, 0.01),
                between_mean=np.full(n, 0.005),
            )

        @staticmethod
        def robust_moments_many(output_index, candidates, certification=True):
            del output_index, certification
            n = len(candidates)
            return SimpleNamespace(
                mean_upper=np.linspace(0.35, 0.45, n),
                epistemic_upper=np.full(n, 0.09),
                aleatoric_upper=np.full(n, 0.16),
                nominal=SimpleNamespace(between_mean=np.full(n, 0.005)),
            )

        @staticmethod
        def robust_chance_margin_many(
            candidates, *, beta_g, z_alpha, tau, certification,
        ):
            del beta_g, z_alpha, tau
            assert certification
            return SimpleNamespace(
                upper=np.linspace(0.7, 1.0, len(candidates)))

    candidates = [(0, 0), (2, 8), (6, 3), (10, 10)]
    kwargs = dict(
        backend="n0_best",
        candidates=candidates,
        obj_gpr=DummyGPR(),
        con_gpr=DummyGPR(),
        variance_model=DummyVariance(),
        problem=DummyProblem(),
        observed=[],
        task_ensemble=DummyTaskEnsemble(),
        decision_aleatoric_mode="posterior_central",
        robust_certificate_mode="joint_tangent",
    )
    robust = score_decision_backend(
        **kwargs, decision_ambiguity_mode="kl_robust")
    nominal = score_decision_backend(
        **kwargs, decision_ambiguity_mode="posterior_nominal")
    np.testing.assert_allclose(robust["theory_margin"], nominal["theory_margin"])
    np.testing.assert_allclose(nominal["constraint_mean"], [0.05, 0.0833333333,
                                                             0.1166666667, 0.15])
    assert nominal["posterior_source"] == "task_posterior_nominal_cumulative"
    assert robust["posterior_source"] == "task_posterior_robust_cumulative"
    assert np.all(nominal["bayes_risk"] < robust["bayes_risk"])
    assert nominal["decision_ambiguity_mode"] == "posterior_nominal"


def test_all_nonlegacy_backends_return_finite_shared_posterior_scores():
    for name in (
        "n0_best",
        "random",
        "sobol",
        "sobol_new",
        "sobol_hvd_voi",
        "sobol_joint_voi",
        "sobol_exact_joint_voi",
        "certificate_depth_new",
        "risk_ts",
        "constrained_ts",
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


def test_sobol_new_excludes_already_observed_candidates():
    result = _score("sobol_new")
    assert result["total"][0] < -1e250
    assert int(np.argmax(result["total"])) != 0


def test_sobol_new_uniquely_selects_injected_canonical_continuation():
    problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
    observed = [(0, 0, 0), (20, 20, 20)]
    canonical = next_sobol_integer_candidate(
        problem, 217, observed=observed)
    candidates = [canonical, (1, 1, 1), (19, 19, 19), observed[0]]
    result = score_decision_backend(
        "sobol_new",
        candidates,
        DummyGPR(sign=1.0),
        DummyGPR(sign=-1.0),
        DummyVariance(),
        problem,
        observed=[(point, np.array([0.0, 0.0])) for point in observed],
        rng=np.random.default_rng(217),
        iteration=7,
        seed=217,
        risk_penalty=5.0,
    )
    assert result["canonical_sobol_injected"] is True
    assert result["canonical_sobol_index"] == 0
    assert int(np.argmax(result["total"])) == 0


def test_certificate_depth_new_uses_theory_margin_and_excludes_observed():
    result = _score("certificate_depth_new")
    assert result["total"][0] < -1e250
    new_indices = np.arange(1, len(result["total"]))
    selected = int(np.argmax(result["total"]))
    expected = int(new_indices[np.argmin(result["theory_margin"][new_indices])])
    assert selected == expected
    assert result["robust_certificate_mode"] == "separable"


def test_certificate_depth_uses_joint_task_margin_when_configured():
    class DummyTaskEnsemble:
        @staticmethod
        def mixture_moments_many(output_index, candidates, certification=False):
            del output_index, certification
            n = len(candidates)
            return SimpleNamespace(
                mean=np.linspace(0.1, 0.4, n),
                epistemic=np.full(n, 0.02),
            )

        @staticmethod
        def robust_moments_many(output_index, candidates, certification=True):
            del output_index, certification
            n = len(candidates)
            return SimpleNamespace(
                mean_upper=np.linspace(0.2, 0.5, n),
                epistemic_upper=np.full(n, 0.03),
                aleatoric_upper=np.full(n, 0.01),
                nominal=SimpleNamespace(between_mean=np.zeros(n)),
            )

        @staticmethod
        def robust_chance_margin_many(
            candidates, *, beta_g, z_alpha, tau, certification,
        ):
            assert beta_g == 2.0
            assert z_alpha > 0.0
            assert tau == DummyProblem.tau
            assert certification
            return SimpleNamespace(
                upper=np.asarray([0.4, 0.2, -0.3, 0.1], dtype=float))

    candidates = [(0, 0), (2, 8), (6, 3), (10, 10)]
    result = score_decision_backend(
        "certificate_depth_new",
        candidates,
        DummyGPR(),
        DummyGPR(),
        DummyVariance(),
        DummyProblem(),
        observed=[((0, 0), np.array([0.0, 0.0]))],
        task_ensemble=DummyTaskEnsemble(),
        certification_beta_g=2.0,
        robust_certificate_mode="joint_tangent",
    )
    np.testing.assert_allclose(result["theory_margin"], [0.4, 0.2, -0.3, 0.1])
    assert int(np.argmax(result["total"])) == 2


def test_sobol_hvd_voi_compares_one_sobol_new_point_with_replication():
    result = _score("sobol_hvd_voi")
    assert result["hvd_sobol_new_index"] is not None
    assert result["hvd_action_is_replicate"][0] == 1.0
    assert result["hvd_action_reliability"][0] == 1.0
    assert np.all(result["hvd_information_reduction"] >= 0.0)
    assert int(np.argmax(result["total"])) == 0


def test_joint_voi_uses_injected_canonical_point_as_its_only_new_action():
    problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
    observed = [(0, 0, 0), (20, 20, 20)]
    canonical = next_sobol_integer_candidate(
        problem, 219, observed=observed)
    candidates = [observed[0], canonical, (1, 1, 1), (19, 19, 19)]
    result = score_decision_backend(
        "sobol_joint_voi",
        candidates,
        DummyGPR(sign=1.0),
        DummyGPR(sign=-1.0),
        DummyVariance(),
        problem,
        observed=[(point, np.array([0.0, 0.0])) for point in observed],
        rng=np.random.default_rng(219),
        iteration=4,
        seed=219,
        risk_penalty=5.0,
    )
    assert result["canonical_sobol_injected"] is True
    assert result["canonical_sobol_index"] == 1
    assert result["hvd_sobol_new_index"] == 1


def test_joint_voi_selects_new_point_when_constraint_epistemic_gain_dominates():
    class NoHVDInformation(DummyVariance):
        @staticmethod
        def information_reduction_many(
            output_index,
            action_points,
            reference_points,
            problem,
            **kwargs,
        ):
            del output_index, reference_points, problem, kwargs
            return np.zeros(len(action_points), dtype=float)

    class DirectionalGPR(DummyGPR):
        def __init__(self):
            super().__init__(sign=1.0)
            self.C = np.diag([1e-4, 1e-4, 8.0])
            self.lambda_i = 1e-4

    candidates = [(0, 0), (10, 10)]
    result = score_decision_backend(
        "sobol_joint_voi",
        candidates,
        DummyGPR(),
        DirectionalGPR(),
        NoHVDInformation(),
        DummyProblem(),
        observed=[((0, 0), np.array([0.0, 0.0]))],
        iteration=1,
        seed=23,
    )
    assert int(np.argmax(result["total"])) == 1
    assert (
        result["constraint_epistemic_information_reduction"][1]
        > result["constraint_epistemic_information_reduction"][0]
    )
    np.testing.assert_allclose(
        result["joint_information_reduction"],
        result["constraint_epistemic_information_reduction"]
        + result["hvd_margin_information_reduction"],
    )
    assert result["joint_information_unit"] == "chance_margin_response"
    assert result["joint_information_contract"] == "sqrt_radius_reduction_v2"


def test_joint_voi_gpr_reduction_matches_nominal_posterior_update_noise():
    class SplitVariance(DummyVariance):
        @staticmethod
        def predict_variance_many(output_index, candidates, problem):
            del output_index, problem
            return np.full(len(candidates), 0.1, dtype=float)

        @staticmethod
        def predict_certification_variance_many(
            output_index, candidates, problem,
        ):
            del output_index, problem
            return np.full(len(candidates), 100.0, dtype=float)

    point = (10, 10)
    model = DummyGPR()
    total_variance = float(model.posterior_var_many([point])[0])
    gain = _constraint_epistemic_reduction(
        [point],
        [point],
        model,
        SplitVariance(),
        DummyProblem(),
        reference_weights=[1.0],
    )
    expected = total_variance ** 2 / (total_variance + 0.1)
    np.testing.assert_allclose(gain, [expected], rtol=1e-12, atol=1e-12)


def test_joint_voi_can_select_replication_when_hvd_gain_dominates():
    class ReplicationHVD(DummyVariance):
        @staticmethod
        def information_reduction_many(
            output_index,
            action_points,
            reference_points,
            problem,
            **kwargs,
        ):
            del output_index, reference_points, problem, kwargs
            return np.asarray([
                100.0 if tuple(point) == (0, 0) else 0.0
                for point in action_points
            ])

    candidates = [(0, 0), (10, 10)]
    result = score_decision_backend(
        "sobol_joint_voi",
        candidates,
        DummyGPR(),
        DummyGPR(),
        ReplicationHVD(),
        DummyProblem(),
        observed=[((0, 0), np.array([0.0, 0.0]))],
        iteration=1,
        seed=29,
    )
    assert int(np.argmax(result["total"])) == 0
    assert result["hvd_action_is_replicate"][0] == 1.0


def test_joint_voi_new_only_excludes_observed_points_from_action_space():
    class ReplicationHVD(DummyVariance):
        @staticmethod
        def information_reduction_many(
            output_index,
            action_points,
            reference_points,
            problem,
            **kwargs,
        ):
            del output_index, reference_points, problem, kwargs
            return np.asarray([
                100.0 if tuple(point) == (0, 0) else 0.0
                for point in action_points
            ])

    candidates = [(0, 0), (10, 10)]
    result = score_decision_backend(
        "sobol_joint_voi",
        candidates,
        DummyGPR(),
        DummyGPR(),
        ReplicationHVD(),
        DummyProblem(),
        observed=[((0, 0), np.array([0.0, 0.0]))],
        iteration=1,
        seed=29,
        canonical_sobol_candidate=(10, 10),
        allow_replication_actions=False,
    )
    assert int(np.argmax(result["total"])) == 1
    assert result["replication_actions_enabled"] is False
    assert result["hvd_action_is_replicate"][0] == 0.0


def test_exact_joint_voi_declares_only_canonical_new_and_replicate_actions():
    problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
    observed = [(0, 0, 0), (20, 20, 20)]
    canonical = next_sobol_integer_candidate(
        problem, 231, observed=observed)
    candidates = [observed[0], canonical, (1, 1, 1), observed[1]]
    result = score_decision_backend(
        "sobol_exact_joint_voi",
        candidates,
        DummyGPR(),
        DummyGPR(),
        DummyVariance(),
        problem,
        observed=[(point, np.array([0.0, 0.0])) for point in observed],
        iteration=5,
        seed=231,
        canonical_sobol_candidate=canonical,
        allow_replication_actions=True,
    )
    active = result["evaluate_or_replicate_active_indices"].tolist()
    assert active == [1, 0, 3]
    assert result["evaluate_or_replicate_exact_refit_required"] is True
    assert result["evaluate_or_replicate_new_action_count"] == 1
    assert result["evaluate_or_replicate_replication_action_count"] == 2
    assert result["total"][2] < -1e250
    assert np.all(result["total"][active] == 0.0)


def test_exact_joint_voi_expands_new_actions_by_posterior_risk():
    problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
    observed = [(0, 0, 0), (20, 20, 20)]
    canonical = next_sobol_integer_candidate(
        problem, 232, observed=observed)
    candidates = [
        observed[0], canonical, (1, 1, 1), (6, 7, 8),
        (19, 19, 19), observed[1],
    ]
    result = score_decision_backend(
        "sobol_exact_joint_voi",
        candidates,
        DummyGPR(),
        DummyGPR(),
        DummyVariance(),
        problem,
        observed=[(point, np.array([0.0, 0.0])) for point in observed],
        iteration=5,
        seed=232,
        canonical_sobol_candidate=canonical,
        allow_replication_actions=True,
        evaluate_or_replicate_new_action_count=3,
        evaluate_or_replicate_new_action_policy=(
            "canonical_plus_posterior_risk"),
    )
    new_indices = [1, 2, 3, 4]
    posterior_order = sorted(
        new_indices,
        key=lambda index: (float(result["bayes_risk"][index]), index),
    )
    expected_new = [1] + [
        index for index in posterior_order if index != 1
    ][:2]
    assert result[
        "evaluate_or_replicate_new_action_indices"].tolist() == expected_new
    active = result["evaluate_or_replicate_active_indices"].tolist()
    assert active == expected_new + [0, 5]
    assert result["evaluate_or_replicate_new_action_count"] == 3
    assert result["evaluate_or_replicate_replication_action_count"] == 2
    assert result["evaluate_or_replicate_new_action_policy"] == (
        "canonical_plus_posterior_risk")


def test_v52_action_set_is_literal_superset_of_v51_subset():
    candidates = [
        (0, 0), (1, 1), (2, 8), (3, 7), (4, 6),
        (5, 5), (6, 4), (7, 3), (8, 2), (10, 10),
    ]
    observed = [candidates[0], candidates[-1]]
    result = score_decision_backend(
        "sobol_exact_joint_voi",
        candidates,
        DummyGPR(),
        DummyGPR(),
        DummyVariance(),
        DummyProblem(),
        observed=[(point, np.array([0.0, 0.0])) for point in observed],
        iteration=5,
        seed=232,
        canonical_sobol_candidate=candidates[1],
        allow_replication_actions=True,
        evaluate_or_replicate_new_action_count=6,
        evaluate_or_replicate_new_action_policy=(
            "canonical_plus_posterior_risk_certificate_coverage"),
        evaluate_or_replicate_baseline_new_action_count=4,
    )
    active = set(result["evaluate_or_replicate_active_indices"].tolist())
    baseline = set(result[
        "evaluate_or_replicate_baseline_indices"].tolist())
    supplemental = result[
        "evaluate_or_replicate_supplemental_indices"].tolist()
    assert baseline < active
    assert len(baseline) == 6
    assert len(supplemental) == 2
    assert set(supplemental).issubset(active - baseline)
    assert result[
        "evaluate_or_replicate_supplemental_labels"] == [
            "certificate_depth", "psi_coverage",
        ]
    assert result["risk_coordinate_coverage_source"] == (
        "observable_cumulative_risk")


def test_v54_pareto_support_is_literal_superset_with_auditable_labels():
    candidates = [
        (0, 0), (1, 9), (2, 8), (3, 7), (4, 6), (5, 5),
        (6, 4), (7, 3), (8, 2), (9, 1), (10, 10), (10, 0),
    ]
    observed = [candidates[0], candidates[10]]
    result = score_decision_backend(
        "sobol_exact_joint_voi",
        candidates,
        DummyGPR(),
        DummyGPR(),
        DummyVariance(),
        DummyProblem(),
        observed=[(point, np.array([0.0, 0.0])) for point in observed],
        iteration=5,
        seed=237,
        canonical_sobol_candidate=candidates[1],
        allow_replication_actions=True,
        evaluate_or_replicate_new_action_count=10,
        evaluate_or_replicate_new_action_policy=(
            "canonical_plus_posterior_pareto_support"),
        evaluate_or_replicate_baseline_new_action_count=4,
    )
    active = result["evaluate_or_replicate_active_indices"].tolist()
    baseline = set(result[
        "evaluate_or_replicate_baseline_indices"].tolist())
    labels = result["evaluate_or_replicate_active_labels"]
    assert baseline.issubset(set(active))
    assert len(labels) == len(active)
    assert labels.count("v51_baseline_new") == 4
    assert labels.count("replicate") == 2
    assert result["evaluate_or_replicate_supplemental_labels"]
    assert set(result["evaluate_or_replicate_supplemental_labels"]).issubset({
        "bayes_risk_ei",
        "constrained_ei",
        "chance_boundary",
        "chance_boundary_information",
        "certificate_depth",
        "constraint_margin_information",
        "hvd_margin_information",
        "joint_margin_information",
        "psi_coverage",
    })
    assert result["evaluate_or_replicate_new_action_policy"] == (
        "canonical_plus_posterior_pareto_support")


@pytest.mark.parametrize(
    ("mode", "expected_labels"),
    [
        (
            "epistemic",
            {
                "guard_epistemic_neighbor",
                "guard_epistemic_local_information",
            },
        ),
        (
            "aleatoric",
            {
                "guard_aleatoric_neighbor",
                "guard_aleatoric_boundary",
            },
        ),
        (
            "interior",
            {
                "guard_safe_interior_depth",
                "guard_safe_interior_risk",
            },
        ),
    ],
)
def test_v58_guard_support_is_a_literal_v51_action_superset(
    mode, expected_labels,
):
    candidates = [
        (0, 0), (1, 9), (2, 8), (3, 7), (4, 6), (5, 5),
        (6, 4), (7, 3), (8, 2), (9, 1), (10, 10), (10, 0),
    ]
    observed = [candidates[0], candidates[10]]
    result = score_decision_backend(
        "sobol_exact_joint_voi",
        candidates,
        DummyGPR(),
        DummyGPR(),
        DummyVariance(),
        DummyProblem(),
        observed=[(point, np.array([0.0, 0.0])) for point in observed],
        iteration=5,
        seed=239,
        canonical_sobol_candidate=candidates[1],
        allow_replication_actions=True,
        evaluate_or_replicate_new_action_count=6,
        evaluate_or_replicate_new_action_policy=(
            "canonical_plus_posterior_guard_decomposition"),
        evaluate_or_replicate_baseline_new_action_count=4,
        guard_decomposition={
            "status": "ok",
            "dominant_mode": mode,
            "anchor": list(candidates[0]),
            "target_oracle_used": False,
        },
    )
    active = set(result["evaluate_or_replicate_active_indices"].tolist())
    baseline = set(result[
        "evaluate_or_replicate_baseline_indices"].tolist())
    assert baseline < active
    assert len(baseline) == 6
    assert set(result[
        "evaluate_or_replicate_supplemental_labels"]) == expected_labels
    support = result["guard_decomposition_support"]
    assert support["dominant_mode"] == mode
    assert support["target_oracle_used"] is False
    if mode == "aleatoric":
        labels = result["evaluate_or_replicate_active_labels"]
        assert "guard_aleatoric_anchor_replicate" in labels


def test_v52_one_step_guard_falls_back_until_advantage_exceeds_two_eta():
    problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
    algorithm = SingleOLHKGAlgorithm(
        problem,
        SingleOLHKGConfig(
            policy_improvement_mode="action_superset",
            policy_improvement_mc_error_bound=0.1,
            use_state_coupling=False,
            use_state_basis=False,
        ),
    )
    candidates = [(0, 0, 0), (1, 1, 1), (2, 2, 2)]
    backend = {
        "evaluate_or_replicate_active_indices": np.array([0, 1, 2]),
        "evaluate_or_replicate_baseline_indices": np.array([0, 1]),
    }
    algorithm._last_exact_kg_raw_scores = np.array([1.0, 0.8, 1.19])
    selected, info = algorithm._guarded_one_step_policy_improvement(
        candidates,
        algorithm._last_exact_kg_raw_scores,
        backend,
    )
    assert selected == 0
    assert info["status"] == "baseline_mc_guard"
    assert info["switched"] is False

    algorithm._last_exact_kg_raw_scores = np.array([1.0, 0.8, 1.21])
    selected, info = algorithm._guarded_one_step_policy_improvement(
        candidates,
        algorithm._last_exact_kg_raw_scores,
        backend,
    )
    assert selected == 2
    assert info["status"] == "superset_switched"
    assert info["switched"] is True


def test_v53_certificate_deficit_terminal_uses_minimum_theory_margin():
    problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
    algorithm = SingleOLHKGAlgorithm(
        problem,
        SingleOLHKGConfig(
            exact_kg_terminal_mode="certificate_deficit",
            use_state_coupling=False,
            use_state_basis=False,
        ),
    )
    margins = np.array([0.3, 0.7])
    algorithm._terminal_certificate_components = (
        lambda *args, **kwargs: {"margin": margins})
    value = algorithm._terminal_value_from_models(
        [], object(), [(0, 0, 0), (1, 1, 1)])
    assert value == pytest.approx(0.3)

    margins[:] = [-0.2, 0.7]
    value = algorithm._terminal_value_from_models(
        [], object(), [(0, 0, 0), (1, 1, 1)])
    assert value == 0.0


def test_v53_guard_requires_both_risk_and_certificate_two_eta_gains():
    problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
    algorithm = SingleOLHKGAlgorithm(
        problem,
        SingleOLHKGConfig(
            policy_improvement_mode="certificate_constrained",
            policy_improvement_mc_error_bound=0.1,
            policy_improvement_certificate_mc_error_bound=0.05,
            use_state_coupling=False,
            use_state_basis=False,
        ),
    )
    candidates = [(0, 0, 0), (1, 1, 1), (2, 2, 2)]
    backend = {
        "evaluate_or_replicate_active_indices": np.array([0, 1, 2]),
        "evaluate_or_replicate_baseline_indices": np.array([0, 1]),
    }

    algorithm._last_exact_kg_raw_scores = np.array([1.0, 0.8, 1.19])
    selected, info = algorithm._guarded_certificate_deficit_policy_improvement(
        candidates,
        algorithm._last_exact_kg_raw_scores,
        np.array([0.0, 0.2, 1.0]),
        backend,
    )
    assert selected == 0
    assert info["status"] == "no_risk_admissible_challenger"

    algorithm._last_exact_kg_raw_scores = np.array([1.0, 0.8, 1.21])
    selected, info = algorithm._guarded_certificate_deficit_policy_improvement(
        candidates,
        algorithm._last_exact_kg_raw_scores,
        np.array([0.0, 0.2, 0.09]),
        backend,
    )
    assert selected == 0
    assert info["status"] == "certificate_mc_guard"

    selected, info = algorithm._guarded_certificate_deficit_policy_improvement(
        candidates,
        algorithm._last_exact_kg_raw_scores,
        np.array([0.0, 0.2, 0.11]),
        backend,
    )
    assert selected == 2
    assert info["status"] == "certificate_constrained_switched"
    assert info["switched"] is True
    assert info["conditional_noninferiority_contract"] == (
        "uniform_risk_and_certificate_mc_errors_imply_joint_"
        "posterior_improvement")


def test_v53_current_terminal_normalization_preserves_ranking_and_scales_eta():
    problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
    algorithm = SingleOLHKGAlgorithm(
        problem,
        SingleOLHKGConfig(
            policy_improvement_mode="certificate_constrained",
            policy_improvement_score_normalization="current_terminal",
            policy_improvement_mc_error_bound=0.1,
            policy_improvement_certificate_mc_error_bound=0.05,
            use_state_coupling=False,
            use_state_basis=False,
        ),
    )
    candidates = [(0, 0, 0), (1, 1, 1), (2, 2, 2)]
    backend = {
        "evaluate_or_replicate_active_indices": np.array([0, 1, 2]),
        "evaluate_or_replicate_baseline_indices": np.array([0, 1]),
    }
    algorithm._last_exact_kg_current_value = 100.0
    algorithm._last_certificate_deficit_current_value = 2.0
    algorithm._last_exact_kg_raw_scores = np.array([100.0, 80.0, 121.0])
    certificate_scores = np.array([0.0, 0.4, 0.22])

    selected, info = algorithm._guarded_certificate_deficit_policy_improvement(
        candidates,
        algorithm._last_exact_kg_raw_scores,
        certificate_scores,
        backend,
    )

    assert selected == 2
    assert info["status"] == "certificate_constrained_switched"
    assert info["score_normalization"] == "current_terminal"
    assert info["risk_score_scale"] == pytest.approx(100.0)
    assert info["certificate_score_scale"] == pytest.approx(2.0)
    assert info["estimated_risk_advantage"] == pytest.approx(0.21)
    assert info["estimated_certificate_advantage"] == pytest.approx(0.11)
    assert info["raw_estimated_risk_advantage"] == pytest.approx(21.0)
    assert info["raw_estimated_certificate_advantage"] == pytest.approx(0.22)
    assert info[
        "risk_mc_uniform_error_bound_raw_equivalent"] == pytest.approx(10.0)
    assert info[
        "certificate_mc_uniform_error_bound_raw_equivalent"
    ] == pytest.approx(0.1)
    assert algorithm._policy_improvement_contract_id() == (
        "v53_constrained_certificate_deficit_v2")


def test_v53_certificate_pass_requires_nested_common_random_numbers():
    problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
    algorithm = SingleOLHKGAlgorithm(
        problem,
        SingleOLHKGConfig(
            exact_kg_sampling_mode="iid",
            use_state_coupling=False,
            use_state_basis=False,
        ),
    )
    with pytest.raises(ValueError, match="common random numbers"):
        algorithm._exact_certificate_deficit_scores_for_actions(
            [(0, 0, 0)], [(0, 0, 0)], [0])


def test_v52_rollout_guard_uses_one_step_action_as_fallback():
    problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
    algorithm = SingleOLHKGAlgorithm(
        problem,
        SingleOLHKGConfig(
            policy_improvement_mode="guarded_rollout",
            policy_improvement_rollout_depth=2,
            policy_improvement_rollout_max_arms=2,
            policy_improvement_rollout_mc_error_bound=0.1,
            use_state_coupling=False,
            use_state_basis=False,
        ),
    )
    candidates = [(0, 0, 0), (1, 1, 1)]
    algorithm._last_exact_kg_raw_scores = np.array([1.0, 1.1])

    def rollout(_arms, _terminal_pool, *, depth, stage):
        del depth, stage
        return _arms[1], {
            "terminal_kg_expected_values": [1.0, 0.7],
            "terminal_kg_selected_index": 1,
            "terminal_kg_raw_gains": [0.0, 0.3],
            "terminal_kg_clipped_gains": [0.0, 0.3],
        }

    algorithm._terminal_replication_kg_candidate = rollout
    selected, info = algorithm._guarded_rollout_policy_improvement(
        candidates,
        candidates,
        algorithm._last_exact_kg_raw_scores,
        np.array([0, 1]),
        0,
        stage=3,
    )
    assert selected == 1
    assert info["status"] == "rollout_switched"
    assert info["switched"] is True


def test_risk_aware_thompson_sampling_is_reproducible_for_fixed_seed():
    first = _score("risk_ts", seed=81)
    second = _score("risk_ts", seed=81)
    different = _score("risk_ts", seed=82)
    np.testing.assert_allclose(first["total"], second["total"])
    assert not np.allclose(first["total"], different["total"])


def test_constrained_ts_is_lexicographically_feasible_first(monkeypatch):
    draws = iter([
        np.asarray([0.0, 3.0, 1.0, 2.0]),
        np.asarray([1.0, -1.0, 1.0, -1.0]),
    ])
    monkeypatch.setattr(
        decision_backends,
        "_joint_gpr_draw",
        lambda *args, **kwargs: next(draws),
    )
    result = _score("constrained_ts", seed=81)
    assert result["sampled_feasible_count"] == 2
    assert result["sampled_feasibility_mode"] == "feasible_objective"
    assert int(np.argmax(result["total"])) == 3
    assert result["total"][0] < -1e250
    assert result["total"][2] < -1e250


def test_constrained_ts_minimizes_violation_when_draw_has_no_feasible_policy(
    monkeypatch,
):
    draws = iter([
        np.asarray([0.0, 3.0, 1.0, 2.0]),
        np.asarray([1.0, 0.9, 0.8, 0.7]),
    ])
    monkeypatch.setattr(
        decision_backends,
        "_joint_gpr_draw",
        lambda *args, **kwargs: next(draws),
    )
    result = _score("constrained_ts", seed=82)
    assert result["sampled_feasible_count"] == 0
    assert result["sampled_feasibility_mode"] == "minimum_violation"
    assert int(np.argmax(result["total"])) == 3


def test_transfer_utility_declares_frozen_source_prior_use():
    result = _score("transfer_utility")
    assert result["transfer_utility_status"] == "source_mean_prior_active"
    assert np.all(np.isfinite(result["transfer_utility"]))


def test_exact_joint_backend_runs_refit_voi_on_declared_action_set():
    problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
    algorithm = SingleOLHKGAlgorithm(
        problem,
        SingleOLHKGConfig(
            N=3,
            n0=2,
            K1=2,
            K2=0,
            decision_backend="sobol_exact_joint_voi",
            adaptive_replication_voi=True,
            replication_candidate_count=2,
            exact_kg_mc_samples=2,
            exact_kg_jobs=1,
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
            seed=93,
        ),
    )
    result = algorithm.run(verbose=False)
    row = algorithm.iteration_log[0]
    assert result["n_simulations"] == 3
    assert row["decision_backend"] == "sobol_exact_joint_voi"
    assert row["exact_kg_full_posterior_refit"] is True
    assert row["exact_kg_refits_gpr_hc3_hvd"] is True
    assert row["exact_kg_active_action_count"] == 3
    assert len(row["exact_kg_active_indices"]) == 3
    assert row["decision_evaluate_or_replicate_new_action_count"] == 1
    assert row[
        "decision_evaluate_or_replicate_replication_action_count"] == 2
    assert result["adaptive_replication_voi"][
        "exact_refit_action_value"] is True
    assert result["adaptive_replication_voi"]["unified_exact_voi"] is True
    assert result["adaptive_replication_voi"]["target_oracle_used"] is False


def test_v52_superset_runs_end_to_end_without_changing_v51_terminal_rule():
    problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
    algorithm = SingleOLHKGAlgorithm(
        problem,
        SingleOLHKGConfig(
            N=3,
            n0=2,
            K1=5,
            K2=0,
            decision_backend="sobol_exact_joint_voi",
            decision_recommend_observed_only=True,
            exact_kg_terminal_mode="bayes_risk",
            adaptive_replication_voi=True,
            replication_candidate_count=2,
            exact_kg_mc_samples=2,
            exact_kg_jobs=1,
            evaluate_or_replicate_new_action_count=3,
            evaluate_or_replicate_baseline_new_action_count=1,
            evaluate_or_replicate_new_action_policy=(
                "canonical_plus_posterior_risk_certificate_coverage"),
            policy_improvement_mode="action_superset",
            policy_improvement_mc_error_bound=1e6,
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
            seed=194,
        ),
    )
    result = algorithm.run(verbose=False)
    row = algorithm.iteration_log[0]
    assert result["n_simulations"] == 3
    assert row["selection_policy"] == "safeguarded_policy_improvement"
    assert row["policy_improvement_one_step"]["switched"] is False
    assert row["policy_improvement_rollout"]["status"] == "disabled"
    labels = row["decision_evaluate_or_replicate_supplemental_labels"]
    assert labels
    assert labels[0] == "certificate_depth"
    assert set(labels).issubset({"certificate_depth", "psi_coverage"})
    assert result["decision_backend_contract"][
        "policy_improvement_contract"] == (
            "v52_safeguarded_policy_improvement_v1")
    assert result["decision_backend_terminal_rule"] == "posterior_bayes_risk"


def test_v53_runs_two_terminal_passes_and_disables_rollout():
    problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
    algorithm = SingleOLHKGAlgorithm(
        problem,
        SingleOLHKGConfig(
            N=3,
            n0=2,
            K1=5,
            K2=0,
            decision_backend="sobol_exact_joint_voi",
            decision_recommend_observed_only=True,
            exact_kg_terminal_mode="bayes_risk",
            exact_kg_sampling_mode="antithetic_nested",
            adaptive_replication_voi=True,
            replication_candidate_count=2,
            exact_kg_mc_samples=2,
            exact_kg_jobs=1,
            evaluate_or_replicate_new_action_count=3,
            evaluate_or_replicate_baseline_new_action_count=1,
            evaluate_or_replicate_new_action_policy=(
                "canonical_plus_posterior_risk_certificate_coverage"),
            policy_improvement_mode="certificate_constrained",
            policy_improvement_mc_error_bound=1e6,
            policy_improvement_certificate_mc_error_bound=1e6,
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
            seed=195,
        ),
    )
    result = algorithm.run(verbose=False)
    row = algorithm.iteration_log[0]
    assert result["n_simulations"] == 3
    assert row["selection_policy"] == "safeguarded_policy_improvement"
    assert row["policy_improvement_rollout"]["status"] == (
        "disabled_by_v53_contract")
    assert row["policy_improvement_contract_id"] == (
        "v53_constrained_certificate_deficit_v1")
    assert row["exact_kg_terminal_value_contract"].startswith("bayes_risk:")
    assert row["certificate_deficit_contract"].startswith(
        "certificate_deficit:")
    assert len(row["certificate_deficit_raw_scores_active"]) == (
        row["exact_kg_active_action_count"])
    assert result["decision_backend_contract"][
        "policy_improvement_contract"] == (
            "v53_constrained_certificate_deficit_v1")
    assert result["decision_backend_contract"][
        "policy_improvement_certificate_mc_error_bound"] == 1e6
    assert algorithm.config.exact_kg_terminal_mode == "bayes_risk"



def test_joint_terminal_head_reuse_matches_legacy_two_pass_scores():
    def run(reuse):
        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        algorithm = SingleOLHKGAlgorithm(
            problem,
            SingleOLHKGConfig(
                N=3,
                n0=2,
                K1=5,
                K2=0,
                decision_backend="sobol_exact_joint_voi",
                decision_recommend_observed_only=True,
                exact_kg_terminal_mode="bayes_risk",
                exact_kg_sampling_mode="antithetic_nested",
                exact_kg_mc_samples=4,
                exact_kg_jobs=1,
                exact_kg_joint_terminal_reuse=reuse,
                adaptive_replication_voi=True,
                replication_candidate_count=2,
                evaluate_or_replicate_new_action_count=3,
                evaluate_or_replicate_baseline_new_action_count=1,
                evaluate_or_replicate_new_action_policy=(
                    "canonical_plus_posterior_risk_certificate_coverage"),
                policy_improvement_mode="certificate_constrained",
                policy_improvement_mc_error_bound=1e6,
                policy_improvement_certificate_mc_error_bound=1e6,
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
                seed=397,
            ),
        )
        algorithm.run(verbose=False)
        return algorithm.iteration_log[0]

    legacy = run(False)
    reused = run(True)
    for field in (
        "exact_kg_raw_scores_active",
        "exact_kg_policy_scores_active",
        "certificate_deficit_raw_scores_active",
        "certificate_deficit_policy_scores_active",
        "certificate_deficit_expected_values_active",
    ):
        np.testing.assert_allclose(reused[field], legacy[field], atol=1e-12)
    assert reused["policy_improvement_selected_index"] == (
        legacy["policy_improvement_selected_index"])
    assert legacy["exact_kg_joint_terminal_head_reuse"] is False
    assert reused["exact_kg_joint_terminal_head_reuse"] is True

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
            truth_pool_diagnostics=True,
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
    assert result["online_action_trace_target_oracle_used"] is False
    assert result["online_action_sequence_fingerprint"] is not None
    assert result["target_design_fingerprint"] is not None
    assert len(result["online_action_trace"]) == 1
    trace = result["online_action_trace"][0]
    assert trace["candidate_source"] != "missing"
    assert trace["n_candidates"] > 0
    assert np.isfinite(trace["selected_score"])
    assert np.isfinite(trace["decision_bayes_risk"])
    assert isinstance(trace["task_expert_allocation"], dict)
    assert isinstance(trace["task_expert_proposal_weights"], dict)
    assert len(trace["observed_response"]) == 2
    assert trace["target_call"] == 3
    assert np.isfinite(trace["true_objective_post_run"])
    assert np.isfinite(trace["true_chance_margin_post_run"])
    assert isinstance(trace["true_feasible_post_run"], bool)
    assert trace["truth_join_timing"] == (
        "post_run_after_all_decisions_frozen")
    assert trace["target_oracle_used_for_decision"] is False
    assert result["online_action_trace_truth_available"] is True
    randomness = result["simulation_randomness_contract"]
    assert randomness["mode"] == "evaluation_indexed_seed_sequence"
    assert randomness["proposal_rng_independent"] is True
    assert randomness["target_oracle_used"] is False
    proposal_randomness = result["proposal_randomness_contract"]
    assert proposal_randomness[
        "mode"] == "iteration_and_namespace_seed_sequence"
    assert proposal_randomness["component_streams_independent"] is True
    assert proposal_randomness["target_oracle_used"] is False

    audit = post_run_variance_calibration_audit(
        algorithm, problem, seed=91, audit_size=8)
    assert audit["status"] == "audited"
    assert audit["audit_size"] == 8
    assert np.isfinite(audit["log_variance_rmse"])
    assert 0.0 <= audit["variance_upper_coverage"] <= 1.0
    assert audit["target_oracle_used_for_decision"] is False


def test_sobol_new_algorithm_uses_model_independent_canonical_action():
    problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
    algorithm = SingleOLHKGAlgorithm(
        problem,
        SingleOLHKGConfig(
            N=3,
            n0=2,
            K1=2,
            K2=0,
            decision_backend="sobol_new",
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
            seed=218,
        ),
    )
    result = algorithm.run(verbose=False)
    row = algorithm.iteration_log[0]
    assert row["decision_canonical_sobol_injected"] is True
    assert row["candidate_source_selected"] == "sobol_continuation"
    assert row["online_terminal_solve_skipped"] is True
    assert row["online_terminal_pool_deferred"] is False
    assert row["t_posterior_solve"] < 0.05
    assert algorithm._canonical_sobol_sequence is not None
    assert len(algorithm._canonical_sobol_sequence) >= 64
    assert result["online_action_trace"][0][
        "candidate_source"] == "sobol_continuation"


def test_simulation_noise_is_independent_of_proposal_rng_consumption():
    def make_algorithm():
        return SingleOLHKGAlgorithm(
            ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03)),
            SingleOLHKGConfig(
                N=2,
                n0=2,
                seed=117,
                use_state_coupling=False,
                use_state_basis=False,
            ),
        )

    first = make_algorithm()
    second = make_algorithm()
    first.rng.random(1000)
    x = (5, 10, 15)
    np.testing.assert_allclose(
        first._simulate_and_store(x),
        second._simulate_and_store(x),
    )
    second.rng.random(37)
    np.testing.assert_allclose(
        first._simulate_and_store(x),
        second._simulate_and_store(x),
    )


def test_proposal_component_streams_are_order_independent():
    algorithm = SingleOLHKGAlgorithm(
        ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03)),
        SingleOLHKGConfig(
            N=2,
            n0=2,
            seed=118,
            use_state_coupling=False,
            use_state_basis=False,
        ),
    )
    first = algorithm._proposal_rng(3, "task_expert:ordered")
    first.random(500)
    reference = algorithm._proposal_rng(3, "constraint_uncertain").random(8)
    second = algorithm._proposal_rng(3, "constraint_uncertain").random(8)
    np.testing.assert_allclose(reference, second)
    assert not np.allclose(
        reference,
        algorithm._proposal_rng(3, "task_expert:ordered").random(8),
    )


def test_v54_paired_difference_guard_uses_action_specific_crn_radius():
    problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
    algorithm = SingleOLHKGAlgorithm(
        problem,
        SingleOLHKGConfig(
            policy_improvement_mode="certificate_constrained",
            policy_improvement_score_transform="bounded_current_gain",
            policy_improvement_guard_mode="paired_nested_difference",
            policy_improvement_pairwise_prefix_samples=32,
            policy_improvement_pairwise_error_multiplier=1.25,
            exact_kg_mc_samples=128,
            use_state_coupling=False,
            use_state_basis=False,
        ),
    )
    candidates = [(0, 0, 0), (1, 1, 1), (2, 2, 2)]
    backend = {
        "evaluate_or_replicate_active_indices": np.array([0, 1, 2]),
        "evaluate_or_replicate_baseline_indices": np.array([0, 1]),
    }
    algorithm._last_exact_kg_raw_scores = np.array([2.0, 1.0, 100.0])
    algorithm._last_exact_kg_policy_scores = np.array([0.3, 0.3, 0.7])
    algorithm._last_certificate_deficit_raw_scores = np.array([0.0, 0.0, 1.0])
    algorithm._last_certificate_deficit_policy_scores = np.array([0.0, 0.0, 0.4])
    algorithm._last_pairwise_prefix_risk_policy_scores = np.array(
        [0.3, 0.3, 0.68])
    algorithm._last_pairwise_prefix_certificate_policy_scores = np.array(
        [0.0, 0.0, 0.38])
    algorithm._last_pairwise_prefix_sample_count = 32
    algorithm._last_pairwise_high_sample_count = 128

    selected, info = algorithm._guarded_certificate_deficit_policy_improvement(
        candidates,
        algorithm._last_exact_kg_raw_scores,
        algorithm._last_certificate_deficit_raw_scores,
        backend,
    )
    assert selected == 2
    assert info["switched"] is True
    assert info["guard_mode"] == "paired_nested_difference"
    assert info["risk_switch_threshold"] == pytest.approx(0.025)
    assert info["certificate_switch_threshold"] == pytest.approx(0.025)
    assert info["conditional_noninferiority_contract"] == (
        "paired_difference_error_bounds_imply_joint_posterior_improvement")
    assert algorithm._policy_improvement_contract_id() == (
        "v54_paired_difference_guard_v1")

    algorithm._last_pairwise_prefix_risk_policy_scores[2] = 0.2
    selected, info = algorithm._guarded_certificate_deficit_policy_improvement(
        candidates,
        algorithm._last_exact_kg_raw_scores,
        algorithm._last_certificate_deficit_raw_scores,
        backend,
    )
    assert selected == 0
    assert info["status"] == "no_risk_admissible_challenger"


def test_v55_current_relative_guard_maximizes_positive_joint_lcb():
    problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
    algorithm = SingleOLHKGAlgorithm(
        problem,
        SingleOLHKGConfig(
            policy_improvement_mode="certificate_constrained",
            policy_improvement_score_transform="bounded_current_gain",
            policy_improvement_guard_mode="paired_nested_absolute",
            policy_improvement_pairwise_error_multiplier=1.0,
            use_state_coupling=False,
            use_state_basis=False,
        ),
    )
    candidates = [(0, 0, 0), (1, 1, 1), (2, 2, 2)]
    backend_score = {
        "evaluate_or_replicate_active_indices": np.array([0, 1, 2]),
        "evaluate_or_replicate_baseline_indices": np.array([0]),
    }
    risk = np.array([0.8, 0.3, 0.4])
    certificate = np.array([-0.2, 0.4, 0.35])
    algorithm._last_exact_kg_raw_scores = risk.copy()
    algorithm._last_exact_kg_policy_scores = risk.copy()
    algorithm._last_certificate_deficit_raw_scores = certificate.copy()
    algorithm._last_certificate_deficit_policy_scores = certificate.copy()
    algorithm._last_pairwise_prefix_risk_policy_scores = np.array([
        0.8, 0.28, 0.39])
    algorithm._last_pairwise_prefix_certificate_policy_scores = np.array([
        -0.2, 0.38, 0.34])
    algorithm._last_pairwise_prefix_sample_count = 32
    algorithm._last_pairwise_high_sample_count = 128

    selected, info = algorithm._guarded_certificate_deficit_policy_improvement(
        candidates, risk, certificate, backend_score)
    assert selected == 2
    assert info["status"] == "current_relative_joint_switched"
    assert info["joint_lcb_by_index"]["2"] == pytest.approx(0.34)
    assert info["current_relative_admissible_indices"] == [1, 2]
    assert info["conditional_noninferiority_contract"] == (
        "nested_absolute_error_bounds_imply_current_relative_joint_"
        "improvement")
    assert algorithm._policy_improvement_contract_id() == (
        "v55_current_relative_joint_improvement_v1")

    algorithm._last_pairwise_prefix_risk_policy_scores = np.array([
        0.8, -0.2, -0.2])
    algorithm._last_pairwise_prefix_certificate_policy_scores = np.array([
        -0.2, -0.2, -0.2])
    selected, info = algorithm._guarded_certificate_deficit_policy_improvement(
        candidates, risk, certificate, backend_score)
    assert selected == 0
    assert info["status"] == "no_current_relative_joint_admissible_action"
    assert info["current_relative_admissible_indices"] == []


def test_v56_betting_eprocess_detects_small_stable_positive_gain():
    lambdas = np.geomspace(0.001, 1.0, 24)
    positive = SingleOLHKGAlgorithm._betting_mixture_log_evalue_path(
        np.full(4096, 0.004, dtype=float), lambdas)
    null = SingleOLHKGAlgorithm._betting_mixture_log_evalue_path(
        np.zeros(4096, dtype=float), lambdas)
    assert positive[-1] > np.log(400.0)
    assert null[-1] == pytest.approx(0.0)


def test_v56_confirmation_stream_is_reproducible_and_not_main_rng_driven():
    problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
    algorithm = SingleOLHKGAlgorithm(
        problem,
        SingleOLHKGConfig(
            seed=991,
            use_state_coupling=False,
            use_state_basis=False,
        ),
    )
    z_first, u_first, seed_first = (
        algorithm._independent_confirmation_sample_plan(8))
    algorithm.rng.random(1000)
    z_second, u_second, seed_second = (
        algorithm._independent_confirmation_sample_plan(8))
    np.testing.assert_array_equal(z_first, z_second)
    np.testing.assert_array_equal(u_first, u_second)
    assert seed_first == seed_second


def test_v56_guard_uses_independent_confirmation_or_literal_v51_fallback(
    monkeypatch,
):
    problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
    algorithm = SingleOLHKGAlgorithm(
        problem,
        SingleOLHKGConfig(
            policy_improvement_mode="certificate_constrained",
            policy_improvement_score_transform="bounded_current_gain",
            policy_improvement_guard_mode="independent_confirmation",
            use_state_coupling=False,
            use_state_basis=False,
        ),
    )
    candidates = [(0, 0, 0), (1, 1, 1), (2, 2, 2)]
    backend = {
        "evaluate_or_replicate_active_indices": np.array([0, 1, 2]),
        "evaluate_or_replicate_baseline_indices": np.array([0]),
    }
    risk = np.array([0.8, 0.3, 0.4])
    certificate = np.array([-0.2, 0.4, 0.35])
    algorithm._last_exact_kg_raw_scores = risk.copy()
    algorithm._last_exact_kg_policy_scores = risk.copy()
    algorithm._last_certificate_deficit_raw_scores = certificate.copy()
    algorithm._last_certificate_deficit_policy_scores = certificate.copy()
    monkeypatch.setattr(
        algorithm,
        "_independent_policy_improvement_confirmation",
        lambda candidate, terminal_pool: {
            "status": "joint_confirmation_passed",
            "passed": True,
            "sample_count": 32,
            "target_oracle_used": False,
        },
    )
    selected, info = algorithm._guarded_certificate_deficit_policy_improvement(
        candidates, risk, certificate, backend, terminal_pool=candidates)
    assert selected == 2
    assert info["pilot_index"] == 2
    assert info["switched"] is True
    assert algorithm._policy_improvement_contract_id() == (
        "v56_independent_confirmation_finite_look_v1")

    monkeypatch.setattr(
        algorithm,
        "_independent_policy_improvement_confirmation",
        lambda candidate, terminal_pool: {
            "status": "joint_confirmation_failed",
            "passed": False,
            "sample_count": 32,
            "target_oracle_used": False,
        },
    )
    selected, info = algorithm._guarded_certificate_deficit_policy_improvement(
        candidates, risk, certificate, backend, terminal_pool=candidates)
    assert selected == 0
    assert info["status"] == "independent_confirmation_fallback"
    assert info["selected_index"] == 0


def test_v56_independent_confirmation_runs_end_to_end():
    problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
    algorithm = SingleOLHKGAlgorithm(
        problem,
        SingleOLHKGConfig(
            N=3,
            n0=2,
            K1=5,
            K2=0,
            decision_backend="sobol_exact_joint_voi",
            decision_recommend_observed_only=True,
            exact_kg_terminal_mode="bayes_risk",
            exact_kg_sampling_mode="antithetic_nested",
            exact_kg_mc_samples=2,
            exact_kg_jobs=2,
            exact_kg_parallel_backend="process_fork",
            adaptive_replication_voi=True,
            replication_candidate_count=2,
            evaluate_or_replicate_new_action_count=3,
            evaluate_or_replicate_baseline_new_action_count=1,
            evaluate_or_replicate_new_action_policy=(
                "canonical_plus_posterior_pareto_support"),
            policy_improvement_mode="certificate_constrained",
            policy_improvement_score_transform="bounded_current_gain",
            policy_improvement_guard_mode="independent_confirmation",
            policy_improvement_confirmation_samples=4,
            policy_improvement_confirmation_batch_samples=2,
            policy_improvement_confirmation_delta=0.5,
            policy_improvement_confirmation_jobs=2,
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
            seed=359,
        ),
    )
    result = algorithm.run(verbose=False)
    info = algorithm.iteration_log[0]["policy_improvement_one_step"]
    confirmation = info["confirmation"]
    assert result["n_simulations"] == 3
    assert info["guard_mode"] == "independent_confirmation"
    assert confirmation["sample_count"] == 4
    assert confirmation["stream_mode"] == "stage_keyed_independent_iid"
    assert confirmation["pilot_stream_independent"] is True
    assert confirmation["maximum_look_count"] == 2
    assert confirmation["head_stage_look_alpha"] == pytest.approx(0.125)
    assert confirmation["log_evalue_threshold"] == pytest.approx(np.log(8.0))
    assert confirmation["error_spending"] == (
        "bonferroni_two_heads_fixed_horizon_finite_looks")
    assert np.isfinite(confirmation["risk_sample_mean"])
    assert np.isfinite(confirmation["certificate_sample_mean"])
    assert result["decision_backend_contract"][
        "policy_improvement_guard_mode"] == "independent_confirmation"


def test_v55_current_relative_prefix_runs_end_to_end():
    problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
    algorithm = SingleOLHKGAlgorithm(
        problem,
        SingleOLHKGConfig(
            N=3,
            n0=2,
            K1=5,
            K2=0,
            decision_backend="sobol_exact_joint_voi",
            decision_recommend_observed_only=True,
            exact_kg_terminal_mode="bayes_risk",
            exact_kg_sampling_mode="antithetic_nested",
            exact_kg_mc_samples=4,
            exact_kg_jobs=1,
            adaptive_replication_voi=True,
            replication_candidate_count=2,
            evaluate_or_replicate_new_action_count=3,
            evaluate_or_replicate_baseline_new_action_count=1,
            evaluate_or_replicate_new_action_policy=(
                "canonical_plus_posterior_pareto_support"),
            policy_improvement_mode="certificate_constrained",
            policy_improvement_score_transform="bounded_current_gain",
            policy_improvement_guard_mode="paired_nested_absolute",
            policy_improvement_pairwise_prefix_samples=2,
            policy_improvement_pairwise_error_multiplier=1.25,
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
            seed=297,
        ),
    )
    result = algorithm.run(verbose=False)
    row = algorithm.iteration_log[0]
    info = row["policy_improvement_one_step"]
    assert result["n_simulations"] == 3
    assert result["initialization_time_sec"] >= 0.0
    assert result["finalization_time_sec"] >= 0.0
    assert info["guard_mode"] == "paired_nested_absolute"
    assert info["pairwise_prefix_sample_count"] == 2
    assert info["pairwise_high_sample_count"] == 4
    assert info["conditional_noninferiority_contract"] == (
        "nested_absolute_error_bounds_imply_current_relative_joint_"
        "improvement")
    assert row["policy_improvement_contract_id"] == (
        "v55_current_relative_joint_improvement_v1")
    assert len(row["pairwise_prefix_risk_policy_scores_active"]) == (
        row["exact_kg_active_action_count"])
    for name in (
        "clone", "predictive_sample", "joint_update", "robust_terminal",
    ):
        assert np.isfinite(row[f"exact_kg_time_{name}_mean"])
    assert result["decision_backend_contract"][
        "policy_improvement_guard_mode"] == "paired_nested_absolute"


def test_v55_nested_prefix_reuse_matches_legacy_second_pass():
    def run(reuse, jobs=1, backend="thread"):
        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        algorithm = SingleOLHKGAlgorithm(
            problem,
            SingleOLHKGConfig(
                N=3,
                n0=2,
                K1=5,
                K2=0,
                decision_backend="sobol_exact_joint_voi",
                decision_recommend_observed_only=True,
                exact_kg_terminal_mode="bayes_risk",
                exact_kg_sampling_mode="antithetic_nested",
                exact_kg_mc_samples=4,
                exact_kg_jobs=jobs,
                exact_kg_parallel_backend=backend,
                exact_kg_joint_terminal_reuse=True,
                exact_kg_reuse_nested_prefix=reuse,
                adaptive_replication_voi=True,
                replication_candidate_count=2,
                evaluate_or_replicate_new_action_count=3,
                evaluate_or_replicate_baseline_new_action_count=1,
                evaluate_or_replicate_new_action_policy=(
                    "canonical_plus_posterior_pareto_support"),
                policy_improvement_mode="certificate_constrained",
                policy_improvement_score_transform="bounded_current_gain",
                policy_improvement_guard_mode="paired_nested_absolute",
                policy_improvement_pairwise_prefix_samples=2,
                policy_improvement_pairwise_error_multiplier=1.25,
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
                seed=1297,
            ),
        )
        algorithm.run(verbose=False)
        return algorithm.iteration_log[0]

    legacy = run(False)
    reused = run(True)
    chunked = run(True, jobs=8, backend="process_fork")
    fields = (
        "exact_kg_raw_scores_active",
        "exact_kg_policy_scores_active",
        "certificate_deficit_raw_scores_active",
        "certificate_deficit_policy_scores_active",
        "pairwise_prefix_risk_policy_scores_active",
        "pairwise_prefix_certificate_policy_scores_active",
    )
    for field in fields:
        np.testing.assert_allclose(reused[field], legacy[field], atol=1e-12)
        np.testing.assert_allclose(chunked[field], reused[field], atol=1e-12)
    assert reused["policy_improvement_selected_index"] == (
        legacy["policy_improvement_selected_index"])
    assert chunked["policy_improvement_selected_index"] == (
        reused["policy_improvement_selected_index"])
    assert legacy["pairwise_prefix_reused_from_high_pass"] is False
    assert reused["pairwise_prefix_reused_from_high_pass"] is True
    assert chunked["pairwise_prefix_reused_from_high_pass"] is True
    assert chunked["exact_kg_chunks_per_candidate"] > 1


def test_v54_paired_prefix_runs_end_to_end():
    problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
    algorithm = SingleOLHKGAlgorithm(
        problem,
        SingleOLHKGConfig(
            N=3,
            n0=2,
            K1=5,
            K2=0,
            decision_backend="sobol_exact_joint_voi",
            decision_recommend_observed_only=True,
            exact_kg_terminal_mode="bayes_risk",
            exact_kg_sampling_mode="antithetic_nested",
            exact_kg_mc_samples=4,
            exact_kg_jobs=1,
            adaptive_replication_voi=True,
            replication_candidate_count=2,
            evaluate_or_replicate_new_action_count=3,
            evaluate_or_replicate_baseline_new_action_count=1,
            evaluate_or_replicate_new_action_policy=(
                "canonical_plus_posterior_risk_certificate_coverage"),
            policy_improvement_mode="certificate_constrained",
            policy_improvement_score_transform="bounded_current_gain",
            policy_improvement_guard_mode="paired_nested_difference",
            policy_improvement_pairwise_prefix_samples=2,
            policy_improvement_pairwise_error_multiplier=1.25,
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
            seed=296,
        ),
    )
    result = algorithm.run(verbose=False)
    row = algorithm.iteration_log[0]
    info = row["policy_improvement_one_step"]
    assert result["n_simulations"] == 3
    assert info["guard_mode"] == "paired_nested_difference"
    assert info["pairwise_prefix_sample_count"] == 2
    assert info["pairwise_high_sample_count"] == 4
    assert info["conditional_noninferiority_contract"] == (
        "paired_difference_error_bounds_imply_joint_posterior_improvement")
    assert row["policy_improvement_contract_id"] == (
        "v54_paired_difference_guard_v1")
    assert len(row["pairwise_prefix_risk_policy_scores_active"]) == (
        row["exact_kg_active_action_count"])
    assert result["decision_backend_contract"][
        "policy_improvement_guard_mode"] == "paired_nested_difference"


def test_v53_bounded_gain_clips_fantasies_and_keeps_literal_v51_fallback():
    problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
    algorithm = SingleOLHKGAlgorithm(
        problem,
        SingleOLHKGConfig(
            policy_improvement_mode="certificate_constrained",
            policy_improvement_score_transform="bounded_current_gain",
            policy_improvement_mc_error_bound=0.1,
            policy_improvement_certificate_mc_error_bound=0.05,
            use_state_coupling=False,
            use_state_basis=False,
        ),
    )
    assert algorithm._policy_improvement_sample_gain(0.4, -1000.0) == 1.0
    assert algorithm._policy_improvement_sample_gain(0.4, 1000.0) == -1.0
    assert algorithm._policy_improvement_sample_gain(0.4, 0.15) == pytest.approx(0.25)

    candidates = [(0, 0, 0), (1, 1, 1), (2, 2, 2)]
    backend = {
        "evaluate_or_replicate_active_indices": np.array([0, 1, 2]),
        "evaluate_or_replicate_baseline_indices": np.array([0, 1]),
    }
    algorithm._last_exact_kg_raw_scores = np.array([2.0, 1.0, 100.0])
    algorithm._last_exact_kg_policy_scores = np.array([0.3, 0.9, 0.7])
    algorithm._last_certificate_deficit_raw_scores = np.array([0.0, 0.0, 100.0])
    algorithm._last_certificate_deficit_policy_scores = np.array([0.0, 0.0, 0.4])
    selected, info = algorithm._guarded_certificate_deficit_policy_improvement(
        candidates,
        algorithm._last_exact_kg_raw_scores,
        algorithm._last_certificate_deficit_raw_scores,
        backend,
    )
    assert info["baseline_index"] == 0
    assert selected == 2
    assert info["score_transform"] == "bounded_current_gain"
    assert algorithm._policy_improvement_contract_id() == (
        "v53_constrained_certificate_deficit_v3")

    algorithm.config.policy_improvement_score_normalization = "current_terminal"
    with pytest.raises(ValueError, match="cannot be normalized twice"):
        algorithm._policy_improvement_score_scales()
