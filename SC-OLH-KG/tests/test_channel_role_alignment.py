import numpy as np
from itertools import permutations

from problems.rzdt import (
    FactorShockStatePolicyRZDT1,
    InventorySupplyChainProblem,
    QueueResourceControlProblem,
)
from representation.channel_role_alignment import (
    EquivariantChannelRoleAligner,
)
from representation.observable_exposure import (
    ObservableStateExposure,
    get_observable_state_exposure,
    observable_state_descriptor_names,
    partially_aligned_observable_state_descriptor,
    role_aligned_observable_state_descriptor,
)


def _exposure(means, scales):
    return ObservableStateExposure(
        channel_means=np.asarray(means, dtype=float),
        channel_scales=np.asarray(scales, dtype=float),
        occupancy=np.linspace(0.0, 1.0, 13),
        dynamics=np.linspace(-0.5, 0.5, 12),
    )


def test_role_aligned_descriptor_is_equivariant_to_channel_permutation():
    exposure = _exposure([0.2, 0.8, 0.4], [0.1, 0.3, 0.2])
    assignment = np.asarray([2, 0, 1], dtype=int)
    permutation = np.asarray([2, 0, 1], dtype=int)
    permuted = _exposure(
        exposure.channel_means[permutation],
        exposure.channel_scales[permutation],
    )
    first = role_aligned_observable_state_descriptor(exposure, assignment)
    second = role_aligned_observable_state_descriptor(
        permuted, assignment[permutation])
    np.testing.assert_allclose(first, second, atol=0.0, rtol=0.0)
    assert len(first) == len(observable_state_descriptor_names("role_aligned"))


def test_source_role_alignment_is_deterministic_and_fixed_dimension():
    rng = np.random.default_rng(730)
    exposures = []
    domains = []
    margins = []
    for domain, permutation in (
        ("left", np.asarray([0, 1, 2])),
        ("rotated", np.asarray([2, 0, 1])),
    ):
        for _ in range(32):
            base = rng.uniform(0.1, 0.9, size=3)
            scale = 0.05 + 0.2 * base
            exposures.append(_exposure(base[permutation], scale[permutation]))
            domains.append(domain)
            margins.append(float(base[0] - 0.5 * base[1]))
    first = EquivariantChannelRoleAligner(seed=731).fit(
        exposures, domains, margins)
    second = EquivariantChannelRoleAligner(seed=731).fit(
        exposures, domains, margins)
    assert first.diagnostics()["source_assignments"] == (
        second.diagnostics()["source_assignments"])
    probe_left = first.source_descriptor("left", exposures[0])
    probe_rotated = first.source_descriptor("rotated", exposures[32])
    assert probe_left.shape == probe_rotated.shape
    assert np.all(np.isfinite(probe_left))
    diagnostics = first.diagnostics()
    assert diagnostics["target_labels_used"] is False
    assert diagnostics["target_oracle_used"] is False


def test_target_role_matching_uses_only_unlabelled_observable_exposure():
    sources = (
        FactorShockStatePolicyRZDT1(d=30),
        InventorySupplyChainProblem(d=30),
    )
    exposures = []
    domains = []
    margins = []
    for domain_index, problem in enumerate(sources):
        rng = np.random.default_rng(740 + domain_index)
        for _ in range(24):
            x = problem.sample_random(rng)
            exposures.append(get_observable_state_exposure(problem, x))
            domains.append(type(problem).__name__)
            margins.append(float(rng.normal()))
    aligner = EquivariantChannelRoleAligner(
        target_pool_size=24, seed=742).fit(exposures, domains, margins)
    target = QueueResourceControlProblem(d=30)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("target outcome/oracle was called")

    target.true_objectives = forbidden
    target.true_objective = forbidden
    target.true_constraint_mean = forbidden
    target.true_sigma = forbidden
    target.risk_exposures = forbidden
    assignment = aligner.target_assignment(target)
    calibration = aligner.target_epistemic_calibration(target)
    x = target.sample_random(np.random.default_rng(743))
    descriptor = aligner.descriptor(
        target, get_observable_state_exposure(target, x))
    assert len(assignment) == 3
    assert np.all(np.isfinite(descriptor))
    target_diagnostics = next(iter(
        aligner.diagnostics()["target_matches"].values()))
    assert target_diagnostics["target_labels_used"] is False
    assert target_diagnostics["target_oracle_used"] is False
    assert 0.0 <= calibration["source_role_trust"] <= 1.0
    assert calibration["target_labels_used"] is False
    assert calibration["target_oracle_used"] is False


def test_source_geometry_assignment_prior_preserves_unlabelled_hard_match():
    sources = (
        FactorShockStatePolicyRZDT1(d=30),
        InventorySupplyChainProblem(d=30),
    )
    exposures = []
    domains = []
    margins = []
    for domain_index, problem in enumerate(sources):
        rng = np.random.default_rng(744 + domain_index)
        for _ in range(24):
            x = problem.sample_random(rng)
            exposures.append(get_observable_state_exposure(problem, x))
            domains.append(type(problem).__name__)
            margins.append(float(rng.normal()))
    aligner = EquivariantChannelRoleAligner(
        target_pool_size=24, seed=746).fit(exposures, domains, margins)
    target = QueueResourceControlProblem(d=30)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("target outcome/oracle was called")

    target.true_objectives = forbidden
    target.true_objective = forbidden
    target.true_constraint_mean = forbidden
    target.true_sigma = forbidden
    target.risk_exposures = forbidden
    assignments = tuple(permutations(range(aligner.n_roles), 3))
    weight, diagnostics = aligner.target_assignment_prior(
        target, assignments, temperature_scale=0.5)
    hard = tuple(int(value) for value in aligner.target_assignment(target))
    assert len(weight) == len(assignments)
    assert np.all(weight > 0.0)
    assert np.isclose(float(np.sum(weight)), 1.0)
    assert assignments[int(np.argmax(weight))] == hard
    assert diagnostics["maximum_prior_matches_hard_assignment"] is True
    assert diagnostics["target_labels_used"] is False
    assert diagnostics["target_oracle_used"] is False
    assert diagnostics["permutation_equivariant"] is True


def test_boundary_assignment_update_is_channel_permutation_equivariant():
    assignments = tuple(permutations(range(3), 2))
    prior = np.full(len(assignments), 1.0 / len(assignments))
    target_mean = np.asarray([0.85, -0.70])
    target_variance = np.asarray([0.08, 0.08])
    role_mean = np.asarray([-0.75, 0.10, 0.90])
    role_variance = np.asarray([0.04, 0.04, 0.04])
    posterior, _ = EquivariantChannelRoleAligner.boundary_assignment_update(
        assignments,
        prior,
        target_mean,
        target_variance,
        role_mean,
        role_variance,
    )
    selected = assignments[int(np.argmax(posterior))]
    assert selected == (2, 0)

    permutation = np.asarray([1, 0], dtype=int)
    permuted_assignments = tuple(
        tuple(np.asarray(assignment, dtype=int)[permutation])
        for assignment in assignments
    )
    permuted_prior = prior.copy()
    permuted, _ = EquivariantChannelRoleAligner.boundary_assignment_update(
        permuted_assignments,
        permuted_prior,
        target_mean[permutation],
        target_variance[permutation],
        role_mean,
        role_variance,
    )
    np.testing.assert_allclose(permuted, posterior, atol=1e-15, rtol=0.0)


def test_target_boundary_assignment_uses_charged_labels_without_oracle():
    sources = (
        InventorySupplyChainProblem(d=18),
        QueueResourceControlProblem(d=18),
    )
    exposures = []
    domains = []
    margins = []
    for index, problem in enumerate(sources):
        rng = np.random.default_rng(748 + index)
        for _ in range(32):
            x = problem.sample_random(rng)
            exposure = get_observable_state_exposure(problem, x)
            exposures.append(exposure)
            domains.append(problem.problem_name)
            margins.append(float(
                exposure.channel_means[0]
                - exposure.channel_means[-1]))
    aligner = EquivariantChannelRoleAligner(
        target_pool_size=24, seed=750).fit(exposures, domains, margins)
    target = QueueResourceControlProblem(d=18)
    rng = np.random.default_rng(751)
    samples = [target.sample_random(rng) for _ in range(10)]
    target_exposures = [
        get_observable_state_exposure(target, x) for x in samples
    ]
    observations = np.asarray([
        target.tau - (
            exposure.channel_means[0] - exposure.channel_means[-1]
        )
        for exposure in target_exposures
    ])
    assignments = tuple(permutations(range(aligner.n_roles), 3))
    geometry, _ = aligner.target_assignment_prior(target, assignments)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("target oracle was called")

    target.true_objective = forbidden
    target.true_constraint_mean = forbidden
    target.true_sigma = forbidden
    posterior, diagnostics = aligner.target_boundary_assignment_posterior(
        target,
        assignments,
        samples,
        observations,
        np.full(len(samples), 0.01),
        geometry_prior_weights=geometry,
    )
    assert np.isclose(float(np.sum(posterior)), 1.0)
    assert np.all(np.isfinite(posterior))
    assert diagnostics["target_observation_count"] == 10
    assert diagnostics["target_labels_used"] is True
    assert diagnostics["target_oracle_used"] is False
    assert diagnostics["permutation_equivariant"] is True
    assert diagnostics["effective_assignment_count_after"] > 0.0


def test_role_match_epistemic_trust_decreases_for_unsupported_signature():
    rng = np.random.default_rng(750)
    exposures = []
    domains = []
    margins = []
    for domain in ("left", "right"):
        for _ in range(24):
            means = rng.normal([0.2, 0.5, 0.8], 0.02)
            exposures.append(_exposure(means, 0.1 + 0.05 * means))
            domains.append(domain)
            margins.append(float(means[0] - means[2]))
    aligner = EquivariantChannelRoleAligner(
        target_pool_size=16, seed=751).fit(exposures, domains, margins)
    target = QueueResourceControlProblem(d=12)
    aligner.target_assignment(target)
    key = next(iter(aligner._target_cache))
    aligner._target_cache[key]["matching_loss"] = 0.0
    supported = aligner.target_epistemic_calibration(target)
    aligner._target_cache[key]["matching_loss"] = 10_000.0
    unsupported = aligner.target_epistemic_calibration(target)
    assert supported["source_role_trust"] == 1.0
    assert unsupported["source_role_trust"] < supported["source_role_trust"]
    assert unsupported["normalized_matching_loss"] > 1.0


def test_role_cardinality_support_is_outcome_free_and_source_defined():
    sources = (
        InventorySupplyChainProblem(d=18),
        QueueResourceControlProblem(d=18),
    )
    exposures = []
    domains = []
    margins = []
    for index, problem in enumerate(sources):
        rng = np.random.default_rng(760 + index)
        for _ in range(20):
            x = problem.sample_random(rng)
            exposures.append(get_observable_state_exposure(problem, x))
            domains.append(type(problem).__name__)
            margins.append(float(rng.normal()))
    aligner = EquivariantChannelRoleAligner(
        target_pool_size=16, seed=762).fit(exposures, domains, margins)
    unsupported = aligner.target_support_diagnostics(
        FactorShockStatePolicyRZDT1(d=18))
    supported = aligner.target_support_diagnostics(
        InventorySupplyChainProblem(d=18))
    assert unsupported["source_channel_count_support"] == [3]
    assert unsupported["target_channel_count"] == 2
    assert unsupported["channel_cardinality_supported"] is False
    assert supported["channel_cardinality_supported"] is True
    assert unsupported["selection_uses_target_labels"] is False
    assert unsupported["selection_uses_target_oracle"] is False


def test_partial_role_descriptor_is_equivariant_and_retains_missing_mass():
    exposure = _exposure([0.2, 0.8], [0.1, 0.3])
    weights = np.asarray([
        [0.8, 0.2, 0.0],
        [0.1, 0.3, 0.6],
    ])
    permutation = np.asarray([1, 0])
    permuted = _exposure(
        exposure.channel_means[permutation],
        exposure.channel_scales[permutation],
    )
    first = partially_aligned_observable_state_descriptor(
        exposure, weights, n_roles=3)
    second = partially_aligned_observable_state_descriptor(
        permuted, weights[permutation], n_roles=3)
    np.testing.assert_allclose(first, second, atol=1e-15, rtol=0.0)
    role_mask = first[8:12]
    assert np.isclose(np.sum(role_mask), 2.0)
    assert np.all(role_mask[:3] <= 1.0)


def test_partial_transport_learns_source_only_temperature_and_matches_cardinality():
    sources = (
        InventorySupplyChainProblem(d=18),
        QueueResourceControlProblem(d=18),
    )
    exposures = []
    domains = []
    margins = []
    for index, problem in enumerate(sources):
        rng = np.random.default_rng(780 + index)
        for _ in range(20):
            x = problem.sample_random(rng)
            exposure = get_observable_state_exposure(problem, x)
            exposures.append(exposure)
            domains.append(type(problem).__name__)
            margins.append(float(
                exposure.channel_means[0]
                - 0.5 * exposure.channel_means[-1]))
    aligner = EquivariantChannelRoleAligner(
        target_pool_size=16,
        seed=782,
        partial_transport=True,
        transport_temperature_grid=(0.1, 0.5, 1.0),
    ).fit(exposures, domains, margins)
    target = FactorShockStatePolicyRZDT1(d=18)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("target outcome/oracle was called")

    target.true_objective = forbidden
    target.true_constraint_mean = forbidden
    target.true_sigma = forbidden
    target.risk_exposures = forbidden
    weights = aligner.target_transport_weights(target)
    x = target.sample_random(np.random.default_rng(783))
    descriptor = aligner.transport_descriptor(
        target, get_observable_state_exposure(target, x))
    calibration = aligner.target_epistemic_calibration(target)
    diagnostics = aligner.diagnostics()
    assert weights.shape == (2, 3)
    np.testing.assert_allclose(np.sum(weights, axis=1), 1.0)
    assert np.max(np.sum(weights, axis=0)) <= 1.0 + 1e-7
    assert np.all(np.isfinite(descriptor))
    assert diagnostics["transport_selection"]["target_data_used"] is False
    assert diagnostics["transport_selection"]["target_oracle_used"] is False
    assert diagnostics["transport_selection"]["selected_temperature"] in {
        0.1, 0.5, 1.0}
    assert calibration["partial_transport"] is True
    assert calibration["epistemic_covariance_scale"] >= 1.0
    assert calibration["target_labels_used"] is False
    assert calibration["target_oracle_used"] is False


def test_intervention_response_transport_aligns_source_roles_without_outcomes():
    sources = (
        InventorySupplyChainProblem(d=60),
        QueueResourceControlProblem(d=60),
    )
    exposures = []
    profiles = []
    domains = []
    margins = []
    for index, problem in enumerate(sources):
        rng = np.random.default_rng(800 + index)
        for _ in range(48):
            x = problem.sample_random(rng)
            exposures.append(get_observable_state_exposure(problem, x))
            profiles.append(problem.normalize(x))
            domains.append(problem.problem_name)
            margins.append(float(rng.normal()))
    aligner = EquivariantChannelRoleAligner(
        target_pool_size=32,
        seed=803,
        partial_transport=True,
        signature_mode="intervention_response",
        barycentric_transport=True,
        transport_temperature_grid=(0.1, 0.5),
    ).fit(
        exposures,
        domains,
        margins,
        profiles=profiles,
        source_problems=[
            (problem.problem_name, problem) for problem in sources
        ],
    )
    diagnostics = aligner.diagnostics()
    assert diagnostics["source_assignments"][sources[0].problem_name] == (
        diagnostics["source_assignments"][sources[1].problem_name])
    assert diagnostics["signature_mode"] == "intervention_response"
    assert diagnostics["barycentric_transport"] is True
    assert diagnostics["source_signature_pool"] == (
        "deterministic_unlabeled_intervention_pool")

    target = FactorShockStatePolicyRZDT1(d=60)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("target outcome/oracle was called")

    target.true_objective = forbidden
    target.true_constraint_mean = forbidden
    target.true_sigma = forbidden
    target.risk_exposures = forbidden
    weights = aligner.target_transport_weights(target)
    np.testing.assert_allclose(np.sum(weights, axis=1), 1.0, atol=1e-7)
    assert weights.shape == (2, 3)
    target_match = next(iter(
        aligner.diagnostics()["target_matches"].values()))
    assert target_match["transport_geometry"] == "barycentric_response"
    assert target_match["solver_status"] == "entropic_partial_transport"
    assert target_match["target_labels_used"] is False
    assert target_match["target_oracle_used"] is False


def test_square_barycentric_transport_reports_complete_geometry_contract():
    signatures = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    weights, diagnostics = EquivariantChannelRoleAligner._partial_assignment(
        signatures,
        signatures,
        0.5,
        barycentric=True,
    )
    np.testing.assert_allclose(weights, np.eye(2))
    assert diagnostics["solver_status"] == "square_hard_assignment"
    assert diagnostics["optimizer_success"] is True
    assert diagnostics["optimizer_candidate_feasible"] is True
    assert diagnostics["transport_geometry"] == "barycentric_response"
