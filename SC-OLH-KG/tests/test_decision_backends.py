from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from acquisition.decision_backends import (  # noqa: E402
    _constraint_epistemic_reduction,
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
