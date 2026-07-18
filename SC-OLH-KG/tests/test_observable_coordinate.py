import numpy as np
import pytest

from problems.rzdt import InventorySupplyChainProblem
from representation.observable_coordinate import (
    SourceLearnedObservableCoordinate,
    observable_profile_library,
)


def test_observable_library_is_fixed_finite_and_triadic():
    short = observable_profile_library(np.linspace(0.0, 1.0, 5))
    long = observable_profile_library(np.linspace(0.0, 1.0, 50))
    assert short.shape == long.shape
    assert len(short) > 50
    assert np.all(np.isfinite(short))
    assert np.all(np.isfinite(long))


def test_observable_library_represents_generic_quadratic_exposure_pocket():
    rng = np.random.default_rng(91)
    profiles = rng.uniform(size=(300, 50))

    def pocket(profile):
        means = np.asarray([
            np.mean(segment) for segment in np.array_split(profile, 3)
        ])
        return float(
            (means[0] - 0.56) ** 2
            + (means[1] - 0.34) ** 2
            + (means[2] - 0.44) ** 2
            + 0.4 * np.std(profile) ** 2
        )

    matrix = np.vstack([
        observable_profile_library(profile) for profile in profiles
    ])
    design = np.column_stack([np.ones(len(matrix)), matrix])
    target = np.asarray([pocket(profile) for profile in profiles])
    beta = np.linalg.lstsq(design[:200], target[:200], rcond=1e-10)[0]
    error = design[200:] @ beta - target[200:]
    assert float(np.max(np.abs(error))) < 1e-8


def test_source_coordinate_uses_source_rows_only_and_is_reproducible():
    rng = np.random.default_rng(12)
    profiles = [rng.uniform(size=50) for _ in range(24)]
    domains = np.asarray(["a"] * 12 + ["b"] * 12, dtype=object)
    margins = np.asarray([
        (profile[:17].mean() - 0.5) ** 2
        + profile[17:34].mean() - 0.25
        for profile in profiles
    ])
    first = SourceLearnedObservableCoordinate().fit(
        profiles, margins, domains)
    second = SourceLearnedObservableCoordinate().fit(
        profiles, margins, domains)
    probe = rng.uniform(size=50)
    assert first.feature_dim == 6
    assert np.allclose(
        first.features_profile(probe), second.features_profile(probe))
    assert first.diagnostics()["target_oracle_used"] is False
    prior = first.source_parametric_prior(
        InventorySupplyChainProblem(d=50))
    assert prior["mean"].shape == (first.feature_dim + 1,)
    assert prior["covariance"].shape == (
        first.feature_dim + 1,
        first.feature_dim + 1,
    )
    assert float(np.min(np.linalg.eigvalsh(prior["covariance"]))) > 0.0
    assert prior["deviation_variance"] > 0.0
    assert prior["diagnostics"]["target_data_used"] is False
    assert prior["diagnostics"]["target_oracle_used"] is False
    components = first.source_parametric_prior_components(
        InventorySupplyChainProblem(d=50))
    assert {component["domain"] for component in components} == {"a", "b"}
    assert sum(component["prior_weight"] for component in components) == pytest.approx(1.0)
    for component in components:
        assert component["mean"].shape == (first.feature_dim + 1,)
        assert component["covariance"].shape == (
            first.feature_dim + 1,
            first.feature_dim + 1,
        )
        assert float(np.min(np.linalg.eigvalsh(
            component["covariance"]))) > 0.0
        assert component["diagnostics"]["target_data_used"] is False
        assert component["diagnostics"]["target_oracle_used"] is False


def test_consensus_coordinate_preserves_boundary_under_domain_rescaling():
    rng = np.random.default_rng(120)
    base_profiles = [rng.uniform(size=50) for _ in range(32)]
    profiles = base_profiles + [profile.copy() for profile in base_profiles]
    signed = np.asarray([
        profile[:17].mean() - 0.45
        for profile in base_profiles
    ])
    model = SourceLearnedObservableCoordinate(
        output_mode="consensus",
    ).fit(
        profiles,
        np.concatenate([signed, 100.0 * signed]),
        np.asarray(["base"] * 32 + ["scaled"] * 32, dtype=object),
    )
    probe = rng.uniform(size=50)
    features = model.features_profile(probe)
    diagnostics = model.diagnostics()
    scales = {
        row["domain"]: row["target_scale"]
        for row in diagnostics["models"]
    }
    assert model.feature_dim == 2
    assert np.all(np.isfinite(features))
    assert features[1] < 1e-8
    assert scales["scaled"] / scales["base"] == pytest.approx(100.0)
    assert diagnostics["boundary_zero_preserved"] is True


def test_consensus_coordinate_downweights_unreliable_source_atom():
    rng = np.random.default_rng(121)
    base_profiles = [rng.uniform(size=50) for _ in range(40)]
    profiles = base_profiles + [profile.copy() for profile in base_profiles]
    learnable = np.asarray([
        profile[:17].mean() - 0.45
        for profile in base_profiles
    ])
    unlearnable = rng.normal(size=len(base_profiles))
    model = SourceLearnedObservableCoordinate(
        output_mode="consensus",
    ).fit(
        profiles,
        np.concatenate([learnable, unlearnable]),
        np.asarray(["learnable"] * 40 + ["noise"] * 40, dtype=object),
    )
    reliability = {
        row["domain"]: row["reliability"]
        for row in model.diagnostics()["models"]
    }
    assert reliability["learnable"] > reliability["noise"]


def test_source_affine_coordinate_preserves_one_full_shape_per_component():
    rng = np.random.default_rng(122)
    base_profiles = [rng.uniform(size=50) for _ in range(40)]
    profiles = base_profiles + [profile.copy() for profile in base_profiles]
    signed = np.asarray([
        profile[:17].mean() - 0.45
        for profile in base_profiles
    ])
    model = SourceLearnedObservableCoordinate(
        output_mode="source_affine",
    ).fit(
        profiles,
        np.concatenate([signed, 0.2 + 2.0 * signed]),
        np.asarray(["a"] * 40 + ["b"] * 40, dtype=object),
    )
    problem = InventorySupplyChainProblem(d=50)
    probe = model.features_profile(rng.uniform(size=50))
    components = model.source_parametric_prior_components(problem)
    assert model.feature_dim == 2
    assert probe.shape == (2,)
    assert np.all(np.isfinite(probe))
    assert {component["domain"] for component in components} == {"a", "b"}
    for component in components:
        active_atom = component["diagnostics"]["active_atom_index"]
        active_coefficient = 1 + active_atom
        inactive_coefficient = 1 + (1 - active_atom)
        assert component["mean"].shape == (3,)
        assert component["mean"][inactive_coefficient] == pytest.approx(0.0)
        assert (
            component["covariance"][active_coefficient, active_coefficient]
            > 100.0
            * component["covariance"][
                inactive_coefficient, inactive_coefficient]
        )
        assert component["diagnostics"]["component_kind"] == (
            "source_boundary_affine")
        assert len(component["diagnostics"]["affine_calibrations"]) == 2
        assert component["diagnostics"]["target_data_used"] is False
        assert component["diagnostics"]["target_oracle_used"] is False
    diagnostics = model.diagnostics()
    assert diagnostics["boundary_zero_preserved"] is True
    assert diagnostics["source_parametric_prior"]["coordinate"] == (
        "eta_source_affine")


def test_source_rank_coordinate_is_invariant_to_strict_margin_rescaling():
    rng = np.random.default_rng(123)
    base_profiles = [rng.uniform(size=50) for _ in range(48)]
    profiles = base_profiles + [profile.copy() for profile in base_profiles]
    margin = np.asarray([
        profile[:17].mean() + 0.4 * profile[17:34].mean()
        for profile in base_profiles
    ])
    domains = np.asarray(["a"] * 48 + ["b"] * 48, dtype=object)
    first = SourceLearnedObservableCoordinate(
        output_mode="source_rank",
    ).fit(
        profiles,
        np.concatenate([margin, margin]),
        domains,
    )
    transformed = SourceLearnedObservableCoordinate(
        output_mode="source_rank",
    ).fit(
        profiles,
        np.concatenate([margin, np.exp(margin)]),
        domains,
    )
    probes = [rng.uniform(size=50) for _ in range(8)]
    first_features = np.vstack([
        first.features_profile(profile) for profile in probes
    ])
    transformed_features = np.vstack([
        transformed.features_profile(profile) for profile in probes
    ])
    np.testing.assert_allclose(
        first_features, transformed_features, atol=0.0, rtol=0.0)
    assert first.feature_dim == 2
    assert np.all(first_features[:, 0] >= 0.0)
    assert np.all(first_features[:, 0] <= 1.5)
    assert np.all(first_features[:, 1] >= 0.0)
    diagnostics = transformed.diagnostics()
    assert diagnostics["source_rank"][
        "strict_monotone_scale_invariant"] is True
    assert diagnostics["source_rank"]["target_data_used"] is False
    assert diagnostics["source_rank"]["target_oracle_used"] is False
    assert diagnostics["source_parametric_prior"]["coordinate"] == (
        "eta_source_rank")


def test_source_rank_features_many_matches_scalar_path():
    rng = np.random.default_rng(124)
    profiles = [rng.uniform(size=50) for _ in range(64)]
    domains = np.asarray(["a"] * 32 + ["b"] * 32, dtype=object)
    margins = np.asarray([
        profile[:17].mean() - profile[17:34].mean()
        for profile in profiles
    ])
    model = SourceLearnedObservableCoordinate(
        output_mode="source_rank",
    ).fit(profiles, margins, domains)
    problem = InventorySupplyChainProblem(d=50)
    points = [tuple(rng.integers(0, 101, size=50)) for _ in range(7)]
    scalar = np.vstack([model.features(problem, point) for point in points])
    np.testing.assert_allclose(
        model.features_many(problem, points), scalar, atol=1e-12, rtol=1e-12)


def test_observable_library_accepts_problem_normalization():
    problem = InventorySupplyChainProblem(d=50)
    x = tuple([50] * problem.d)
    profile = problem.normalize(x)
    features = observable_profile_library(profile)
    assert features.ndim == 1
    assert np.all(np.isfinite(features))
