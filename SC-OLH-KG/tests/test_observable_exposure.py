import numpy as np

from problems.rzdt import (
    FactorShockStatePolicyRZDT1,
    InventorySupplyChainProblem,
    QueueResourceControlProblem,
)
from representation.observable_exposure import (
    ObservableStateExposure,
    canonical_observable_state_descriptor,
    get_observable_state_exposure,
    observable_state_descriptor_names,
)


def test_observable_state_descriptor_is_fixed_dimension_and_finite():
    problems = (
        FactorShockStatePolicyRZDT1(d=1000),
        InventorySupplyChainProblem(d=1000),
        QueueResourceControlProblem(d=1000),
    )
    descriptors = []
    for index, problem in enumerate(problems):
        x = problem.sample_random(np.random.default_rng(610 + index))
        exposure = get_observable_state_exposure(problem, x)
        assert exposure is not None
        assert exposure.meta["target_outcomes_used"] is False
        assert exposure.meta["target_risk_provider_used"] is False
        descriptor = canonical_observable_state_descriptor(exposure)
        assert np.all(np.isfinite(descriptor))
        descriptors.append(descriptor)
    assert len({len(value) for value in descriptors}) == 1
    assert len(descriptors[0]) == len(observable_state_descriptor_names())


def test_factor_observable_exposure_does_not_use_hidden_targets():
    problem = FactorShockStatePolicyRZDT1(d=100)
    x = problem.sample_random(np.random.default_rng(613))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("hidden target formula was called")

    problem.true_objectives = forbidden
    problem.true_sigma = forbidden
    problem.risk_exposures = forbidden
    descriptor = canonical_observable_state_descriptor(
        problem.observable_state_exposure(x))
    assert np.all(np.isfinite(descriptor))


def test_set_invariant_descriptor_ignores_channel_permutation():
    exposure = ObservableStateExposure(
        channel_means=np.asarray([0.2, 0.8, 0.4]),
        channel_scales=np.asarray([0.1, 0.3, 0.2]),
        occupancy=np.linspace(0.0, 1.0, 13),
        dynamics=np.linspace(-0.5, 0.5, 12),
        channel_names=("left", "right", "middle"),
    )
    permutation = np.asarray([2, 0, 1], dtype=int)
    permuted = ObservableStateExposure(
        channel_means=exposure.channel_means[permutation],
        channel_scales=exposure.channel_scales[permutation],
        occupancy=exposure.occupancy.copy(),
        dynamics=exposure.dynamics.copy(),
        channel_names=tuple(exposure.channel_names[index]
                            for index in permutation),
    )
    first = canonical_observable_state_descriptor(
        exposure, mode="set_invariant")
    second = canonical_observable_state_descriptor(
        permuted, mode="set_invariant")
    np.testing.assert_allclose(first, second, rtol=0.0, atol=1e-12)
    assert len(first) == len(observable_state_descriptor_names(
        "set_invariant"))
