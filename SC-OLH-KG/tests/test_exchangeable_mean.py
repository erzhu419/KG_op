import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.gpr import ParametricGPR  # noqa: E402
from algorithms.single_olhkg import (  # noqa: E402
    SingleOLHKGAlgorithm,
    SingleOLHKGConfig,
)
from problems.rzdt import make_problem  # noqa: E402
from problems.single_objective import ScalarizedProblem  # noqa: E402
from representation.exchangeable_mean import (  # noqa: E402
    ExchangeableBoundaryMeanCoordinate,
)
from representation.meta_prior import (  # noqa: E402
    LearnedMetaPrior,
    MetaPriorProblemAdapter,
)
from representation.observable_exposure import (  # noqa: E402
    ObservableStateExposure,
)


def _exposure(means, scales=None):
    means = np.asarray(means, dtype=float)
    scales = (
        0.05 + 0.1 * means
        if scales is None
        else np.asarray(scales, dtype=float)
    )
    return ObservableStateExposure(
        means,
        scales,
        occupancy=np.asarray([np.mean(means), np.std(means)]),
        dynamics=np.asarray([np.mean(means), np.std(means)]),
    )


def _source_rows(seed=801, n=40):
    rng = np.random.default_rng(seed)
    exposures = []
    targets = []
    domains = []
    for domain, coefficient in (
        ("source_a", np.asarray([1.8, -0.9, 0.4])),
        ("source_b", np.asarray([-1.2, 0.7, 1.5])),
    ):
        for _ in range(n):
            means = rng.uniform(0.05, 0.95, size=3)
            exposures.append(_exposure(means))
            targets.append(float(coefficient @ means - 0.15))
            domains.append(domain)
    return exposures, np.asarray(targets), domains


class _ExposureProblem:
    def __init__(self, exposure, tau=1.0, sigma_level=0.04):
        self.exposure = exposure
        self.tau = float(tau)
        self.sigma_level = float(sigma_level)

    def int_bounds(self):
        return np.asarray([0]), np.asarray([0])

    def observable_state_exposure(self, _x):
        return self.exposure


class _IndexedBasis:
    def __init__(self, coordinate, exposures):
        self.coordinate = coordinate
        self.exposures = list(exposures)
        self.feature_dim = int(coordinate.feature_dim)

    def features(self, x):
        return self.coordinate.features_profile(
            self.exposures[int(np.asarray(x).reshape(-1)[0])])

    def features_many(self, points):
        return np.vstack([self.features(point) for point in points])


def test_exchangeable_features_are_channel_permutation_equivariant():
    exposures, targets, domains = _source_rows()
    coordinate = ExchangeableBoundaryMeanCoordinate().fit(
        exposures, targets, domains)
    original = _exposure([0.2, 0.7, 0.4], [0.1, 0.3, 0.2])
    permutation = np.asarray([2, 0, 1])
    permuted = _exposure(
        original.channel_means[permutation],
        original.channel_scales[permutation],
    )
    first = coordinate.features_profile(original)
    second = coordinate.features_profile(permuted)
    block = coordinate.channel_block_dim
    for new_channel, old_channel in enumerate(permutation):
        np.testing.assert_allclose(
            second[new_channel * block:(new_channel + 1) * block],
            first[old_channel * block:(old_channel + 1) * block],
            atol=1e-12,
            rtol=0.0,
        )
    np.testing.assert_allclose(
        second[coordinate._global_index:],
        first[coordinate._global_index:],
        atol=1e-12,
        rtol=0.0,
    )


def test_exchangeable_source_prior_discards_source_role_identity():
    exposures, targets, domains = _source_rows(seed=802)
    permutation = np.asarray([2, 0, 1])
    permuted = [
        _exposure(
            exposure.channel_means[permutation],
            exposure.channel_scales[permutation],
        )
        for exposure in exposures
    ]
    first = ExchangeableBoundaryMeanCoordinate().fit(
        exposures, targets, domains)
    second = ExchangeableBoundaryMeanCoordinate().fit(
        permuted, targets, domains)
    problem = _ExposureProblem(_exposure([0.3, 0.4, 0.8]))
    first_prior = first.source_parametric_prior(problem)
    second_prior = second.source_parametric_prior(problem)
    np.testing.assert_allclose(
        first_prior["mean"], second_prior["mean"], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(
        first_prior["covariance"],
        second_prior["covariance"],
        atol=1e-8,
        rtol=1e-8,
    )
    assert not first_prior["diagnostics"][
        "source_role_identity_transferred"]
    assert first_prior["diagnostics"]["permutation_equivariant"]
    assert first_prior["diagnostics"]["prior_kind"] == (
        "exchangeable_empirical_bayes_gaussian_hyperlaw")
    assert first_prior["diagnostics"]["target_task_law"] == (
        "single_gaussian_draw")
    assert first_prior["diagnostics"][
        "source_domain_identity_marginalized"]
    assert not first_prior["diagnostics"][
        "source_components_retained_in_target_posterior"]
    assert first_prior["diagnostics"][
        "within_source_covariance_included"]
    assert first_prior["diagnostics"][
        "between_source_covariance_included"]


def test_charged_target_data_learn_channel_specific_opposite_signs():
    exposures, targets, domains = _source_rows(seed=803)
    coordinate = ExchangeableBoundaryMeanCoordinate().fit(
        exposures, targets, domains)
    rng = np.random.default_rng(804)
    target_exposures = [
        _exposure(rng.uniform(0.02, 0.98, size=2)) for _ in range(80)
    ]
    basis = _IndexedBasis(coordinate, target_exposures)
    model = ParametricGPR(d=1, basis_map=basis, lambda_i=1e-8)
    prior = coordinate.source_parametric_prior(
        _ExposureProblem(target_exposures[0]))
    model.set_parametric_prior(
        prior["mean"], lambda_i=1e-8, prior_var=prior["covariance"])
    true_beta = np.zeros(1 + coordinate.feature_dim, dtype=float)
    true_beta[1] = 2.0
    true_beta[1 + coordinate.channel_block_dim] = -2.0
    for index in range(len(target_exposures)):
        x = (index,)
        value = float(model.basis(x) @ true_beta)
        model.update(x, value, 1e-6)
    assert model.a[1] > 0.5
    assert model.a[1 + coordinate.channel_block_dim] < -0.5


def test_exchangeable_meta_prior_is_oracle_free_and_keeps_hvd_head_separate():
    def problem(name, d=12):
        return ScalarizedProblem(
            make_problem(name, d=d, L=100, sigma=0.04))

    sources = [
        ("InventorySupplyChain", problem("InventorySupplyChain")),
        ("QueueResourceControl", problem("QueueResourceControl")),
    ]
    prior = LearnedMetaPrior(
        local_dim=3,
        shared_dim=2,
        component_stage="coordinate",
        observable_mean_coordinate=True,
        observable_mean_mode="boundary_aligned",
        observable_mean_training_target="chance_margin",
        observable_mean_input_mode="observable_state_exposure",
        observable_mean_descriptor_mode="exchangeable_equivariant",
        observable_mean_feature_mode="linear",
        observable_variance_input_mode="observable_state_exposure",
        source_observation_mode="replicated",
        source_observation_replicates=2,
        source_design_mode="universal_mixture",
        source_universal_fraction=1.0,
        teacher_records_per_domain=0,
        seed=805,
    ).fit_from_source_problems(
        sources,
        n_records_per_domain=12,
        rng=np.random.default_rng(805),
    )
    target = MetaPriorProblemAdapter(
        problem("FactorShockStatePolicyRZDT1", d=100), prior)
    x = target.sample_random(np.random.default_rng(806))
    variance_features_before = target.cumulative_risk_features(x).copy()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("target outcome or oracle was called")

    target.base.true_objective = forbidden
    target.base.true_constraint_mean = forbidden
    target.base.true_sigma = forbidden
    target.base.risk_exposures = forbidden
    basis = target.gpr_basis_map(output_index=1)
    assert basis.features(x).shape == (13,)
    assert len(basis.source_parametric_prior_components()) == 2
    source_prior = basis.source_parametric_prior()
    posterior_mean = source_prior["mean"].copy()
    posterior_mean[1 + prior.observable_mean_model.channel_block_dim] += 0.5
    posterior_diagnostics = basis.posterior_coefficient_diagnostics(
        posterior_mean, source_prior["covariance"])
    assert posterior_diagnostics["source_prior_exchangeable"]
    assert posterior_diagnostics["target_channel_roles_differentiated"]
    np.testing.assert_allclose(
        target.cumulative_risk_features(x), variance_features_before)
    diagnostics = basis.diagnostics()
    assert diagnostics["source_role_identity_transferred"] is False
    assert diagnostics["target_oracle_used"] is False
    contract = target.mean_risk_coordinate_contract()
    assert contract["exchangeable_channel_role_posterior"]
    assert contract["target_channel_roles_learned_from_charged_data"]
    assert contract["separate_mean_variance_heads"]
    assert contract["shared_observable_exposure_input"]
    assert contract["source_role_identity_transferred"] is False


def test_frozen_exchangeable_hyperlaw_conditions_without_source_mixture():
    def problem(name, d=12):
        return ScalarizedProblem(
            make_problem(name, d=d, L=100, sigma=0.04))

    prior = LearnedMetaPrior(
        local_dim=3,
        shared_dim=2,
        component_stage="coordinate",
        observable_mean_coordinate=True,
        observable_mean_mode="boundary_aligned",
        observable_mean_training_target="chance_margin",
        observable_mean_input_mode="observable_state_exposure",
        observable_mean_descriptor_mode="exchangeable_equivariant",
        observable_mean_feature_mode="linear",
        observable_variance_input_mode="observable_state_exposure",
        source_observation_mode="replicated",
        source_observation_replicates=2,
        source_design_mode="universal_mixture",
        source_universal_fraction=1.0,
        teacher_records_per_domain=0,
        seed=807,
    ).fit_from_source_problems(
        [
            ("InventorySupplyChain", problem("InventorySupplyChain")),
            ("QueueResourceControl", problem("QueueResourceControl")),
        ],
        n_records_per_domain=12,
        rng=np.random.default_rng(807),
    )

    for index, mode in enumerate((
        "none",
        "predictive_scale",
        "predictive_scale_directional",
        "predictive_scale_upper_target",
        "predictive_scale_upper",
        "predictive_sandwich_hc3",
        "predictive_sandwich_hc3_task",
        "predictive_scale_sandwich_hc3",
        "predictive_scale_sandwich_hc3_task",
        "predictive_scale_sandwich_hc3_confidence",
        "predictive_scale_sandwich_hc3_task_confidence",
    )):
        target = MetaPriorProblemAdapter(
            problem("FactorShockStatePolicyRZDT1"), prior)
        algorithm = SingleOLHKGAlgorithm(
            target,
            SingleOLHKGConfig(
                N=4,
                n0=4,
                K1=1,
                K2=0,
                posterior_pool_size=6,
                posterior_keep=2,
                axis_candidate_count=0,
                state_candidate_count=0,
                eval_pool_size=6,
                evaluate_interval=0,
                use_problem_initial_samples=True,
                use_boundary_initial_samples=False,
                use_recommendation_refinement=False,
                recommendation_axis_oracle=False,
                recommendation_axis_candidate_count=0,
                task_posterior_mode="off",
                source_constraint_mean_coefficient_prior=True,
                source_constraint_mean_adaptation_mode=(
                    "sequential_aggregate_hyperlaw"),
                source_constraint_mean_deviation_mode="latent_shared",
                source_constraint_mean_misspecification_mode=mode,
                source_constraint_mean_null_weight=0.0,
                hvd_source_task_weight_mode="independent",
                hvd_cumulative_target_evidence_mode="replication_only",
                seed=808 + index,
            ),
        )
        samples = algorithm._initial_samples()
        algorithm._fit_initial_belief(samples)
        diagnostics = algorithm.gpr[1].source_parametric_prior_diagnostics
        assert diagnostics["adaptation_mode"] == (
            "sequential_single_aggregate_hyperlaw")
        assert diagnostics["prior_kind"] == (
            "exchangeable_empirical_bayes_gaussian_hyperlaw")
        assert diagnostics["source_domain_identity_marginalized"]
        assert not diagnostics[
            "source_components_retained_in_target_posterior"]
        assert diagnostics["component_names"] == ["source:aggregate"]
        assert diagnostics["single_aggregate_component_count"] == 1
        assert not diagnostics["target_null_component_retained"]
        assert diagnostics["posterior_component_count"] == 1
        assert diagnostics["posterior_projection"] == (
            "single_gaussian_identity_projection")
        assert abs(diagnostics[
            "between_component_covariance_trace"]) <= 1e-8
        assert diagnostics["posterior_target_data_used"]
        assert diagnostics["target_observation_count"] == len(samples)
        assert not diagnostics["target_oracle_used"]
        assert diagnostics["source_mean_misspecification_mode"] == mode
        assert algorithm.variance_model.cumulative_source_task_posterior[1] is None
        if mode == "none":
            assert not diagnostics["source_mean_misspecification_applied"]
            assert diagnostics["source_mean_misspecification_scale"] == 1.0
        else:
            assert diagnostics["source_mean_misspecification_applied"]
            assert diagnostics["source_mean_misspecification_scale"] >= 1.0
            assert diagnostics[
                "misspecification_uncertainty_can_only_increase"]
            covariance_trace_before = diagnostics[
                "source_mean_prior_covariance_trace_before"]
            covariance_trace_after = diagnostics[
                "source_mean_prior_covariance_trace_after"]
            assert covariance_trace_after >= (
                covariance_trace_before
                - 1e-12 * max(1.0, abs(covariance_trace_before))
            )
            if mode in {
                "predictive_scale_upper_target",
                "predictive_scale_upper",
            }:
                assert diagnostics[
                    "source_mean_misspecification_application"] == (
                        "posterior_covariance_only")
                assert diagnostics["source_mean_posterior_mean_preserved"]
            if mode in {
                "predictive_sandwich_hc3",
                "predictive_sandwich_hc3_task",
                "predictive_scale_sandwich_hc3",
                "predictive_scale_sandwich_hc3_task",
                "predictive_scale_sandwich_hc3_confidence",
                "predictive_scale_sandwich_hc3_task_confidence",
            }:
                expected_application = (
                    "source_prior_scale_then_posterior_sandwich"
                    if mode.startswith("predictive_scale_sandwich")
                    else "posterior_sandwich_covariance_only"
                )
                assert diagnostics[
                    "source_mean_misspecification_application"
                ] == expected_application
                assert diagnostics["source_mean_posterior_mean_preserved"]
                assert diagnostics["source_mean_sandwich_applied"]
                assert diagnostics[
                    "source_mean_sandwich_covariance_trace"] >= 0.0
                assert diagnostics[
                    "source_mean_prior_scaled_before_conditioning"
                ] == mode.startswith("predictive_scale_sandwich")
                assert diagnostics[
                    "source_mean_sandwich_decision_authority"
                ] == (
                    "confidence_only"
                    if mode.endswith("_confidence")
                    else "joint_predictive"
                )

        x = tuple(int(value) for value in target.sample_random(
            np.random.default_rng(820 + index)))
        y = target.simulate(x, np.random.default_rng(830 + index))
        sigma2 = algorithm.variance_model.predict_variance(1, x, target)
        exact_kg_clone = algorithm._clone_gpr_for_exact_kg(algorithm.gpr[1])
        assert exact_kg_clone._finite_mixture_misspecification_mode == mode
        assert exact_kg_clone._finite_mixture_misspecification_ridge == 1.0
        original_decision_covariance = (
            None
            if algorithm.gpr[1]._decision_covariance is None
            else algorithm.gpr[1]._decision_covariance.copy()
        )
        if mode.endswith("_confidence"):
            assert exact_kg_clone._decision_covariance is not (
                algorithm.gpr[1]._decision_covariance)
            np.testing.assert_allclose(
                exact_kg_clone._decision_covariance,
                original_decision_covariance,
            )
        exact_kg_clone.update(x, float(y[1]), float(sigma2))
        assert exact_kg_clone._finite_mixture_update_count == 1
        assert algorithm.gpr[1]._finite_mixture_update_count == 0
        assert len(exact_kg_clone._finite_mixture_target_history) == (
            len(samples) + 1)
        assert len(algorithm.gpr[1]._finite_mixture_target_history) == len(samples)
        if mode.endswith("_confidence"):
            np.testing.assert_allclose(
                algorithm.gpr[1]._decision_covariance,
                original_decision_covariance,
            )

        algorithm.gpr[1].update(x, float(y[1]), float(sigma2))
        updated = algorithm.gpr[1].source_parametric_prior_diagnostics
        assert updated["target_observation_count"] == len(samples) + 1
        assert updated["online_mixture_update_count"] == 1
        assert updated["component_names"] == ["source:aggregate"]
        assert updated["posterior_component_count"] == 1
        assert updated["source_mean_misspecification_scale_trajectory"][-1][
            "target_observation_count"] == len(samples) + 1
