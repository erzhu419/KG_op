import numpy as np
import pytest

from problems.rzdt import InventorySupplyChainProblem
from representation.observable_coordinate import (
    SourceLearnedObservableCoordinate,
    observable_profile_library,
)
from representation.boundary_coordinate import (
    SourceAlignedBoundaryCoordinate,
    select_boundary_coordinate_candidates,
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


def test_boundary_aligned_coordinate_separates_representation_and_mean_targets():
    rng = np.random.default_rng(125)
    profiles = []
    boundary = []
    means = []
    domains = []
    for domain, nuisance_sign in (("a", -1.0), ("b", 1.0)):
        for value in np.linspace(0.12, 0.88, 48):
            positions = (np.arange(50, dtype=float) + 0.5) / 50.0
            profile = (
                value
                + 0.10 * np.cos(np.pi * positions)
                + nuisance_sign * 0.15 * np.cos(8.0 * np.pi * positions)
                + rng.normal(0.0, 0.005, size=50)
            )
            profiles.append(np.clip(profile, 0.0, 1.0))
            means.append(0.5 * (value - 0.50))
            boundary.append(value - 0.50 + 0.08 * nuisance_sign)
            domains.append(domain)
    model = SourceAlignedBoundaryCoordinate(latent_dim=2).fit(
        profiles,
        boundary,
        domains,
        coefficient_targets=means,
    )
    features = np.vstack([
        model.features_profile(profile) for profile in profiles
    ])
    correlation = np.corrcoef(features[:, 0], boundary)[0, 1]
    assert abs(float(correlation)) > 0.80
    assert model.feature_dim == 2
    assert np.all(np.isfinite(features))
    diagnostics = model.diagnostics()
    assert diagnostics["alignment"]["representation_training_target"] == (
        "chance_margin_bin")
    assert diagnostics["alignment"]["coefficient_prior_training_target"] == (
        "constraint_mean")
    assert diagnostics["target_oracle_used"] is False
    prior = model.source_parametric_prior(
        InventorySupplyChainProblem(d=50))
    assert prior["mean"].shape == (3,)
    assert float(np.min(np.linalg.eigvalsh(prior["covariance"]))) > 0.0


def test_source_tanh_boundary_coordinate_is_bounded_and_source_selected():
    rng = np.random.default_rng(127)
    profiles = []
    boundary = []
    means = []
    domains = []
    for domain, shift in (("left", -0.08), ("right", 0.08)):
        for value in np.linspace(0.08, 0.92, 32):
            profile = np.clip(
                value + shift + rng.normal(0.0, 0.01, size=40),
                0.0,
                1.0,
            )
            profiles.append(profile)
            boundary.append(value - 0.5 + shift)
            means.append(0.4 * (value - 0.5) + 0.2 * shift)
            domains.append(domain)
    model = SourceAlignedBoundaryCoordinate(
        latent_dim=3,
        feature_mode="linear",
        latent_transform="source_tanh",
    ).fit(
        profiles,
        boundary,
        domains,
        coefficient_targets=means,
    )
    source_features = np.vstack([
        model.features_profile(profile) for profile in profiles
    ])
    extreme = model.features_profile(np.full(40, 10.0))
    assert np.max(np.abs(source_features)) <= 1.0 + 1e-12
    assert np.max(np.abs(extreme)) <= 1.0 + 1e-12
    diagnostics = model.diagnostics()["latent_transform_diagnostics"]
    assert diagnostics["status"] == "source_lodo_selected"
    assert diagnostics["selected_temperature"] in {0.5, 1.0, 2.0, 4.0}
    assert diagnostics["selection_uses_target_data"] is False
    assert diagnostics["selection_uses_target_oracle"] is False
    assert len(diagnostics["candidate_scores"]) == 4


def test_source_support_clip_preserves_source_support_and_bounds_target_shift():
    rng = np.random.default_rng(128)
    profiles = []
    boundary = []
    means = []
    domains = []
    for domain, shift in (("left", -0.06), ("right", 0.06)):
        for value in np.linspace(0.10, 0.90, 36):
            profile = np.clip(
                value + shift + rng.normal(0.0, 0.008, size=40),
                0.0,
                1.0,
            )
            profiles.append(profile)
            boundary.append(value - 0.5 + shift)
            means.append(0.5 * (value - 0.5) + 0.1 * shift)
            domains.append(domain)
    identity = SourceAlignedBoundaryCoordinate(
        latent_dim=3,
        feature_mode="linear",
        latent_transform="identity",
    ).fit(profiles, boundary, domains, coefficient_targets=means)
    clipped = SourceAlignedBoundaryCoordinate(
        latent_dim=3,
        feature_mode="linear",
        latent_transform="source_support_clip",
    ).fit(profiles, boundary, domains, coefficient_targets=means)

    diagnostics = clipped.diagnostics()["latent_transform_diagnostics"]
    bounds = np.asarray(diagnostics["support_bounds"], dtype=float)
    source_features = clipped.features_many(
        InventorySupplyChainProblem(d=40),
        [tuple(np.rint(100.0 * row).astype(int)) for row in profiles],
    )
    extreme = clipped.features_profile(np.full(40, 100.0))
    assert diagnostics["status"] == "source_lodo_selected"
    assert diagnostics["selected_quantile"] in {0.8, 0.9, 0.95, 1.0}
    assert diagnostics["selection_uses_target_data"] is False
    assert diagnostics["selection_uses_target_oracle"] is False
    assert clipped.feature_dim == identity.feature_dim == 3
    assert np.all(np.max(np.abs(source_features), axis=0) <= bounds + 1e-12)
    assert np.all(np.abs(extreme) <= bounds + 1e-12)


def test_source_support_residual_adds_bounded_target_discrepancy_channel():
    profiles = []
    boundary = []
    means = []
    domains = []
    for domain, shift in (("a", -0.05), ("b", 0.05)):
        for value in np.linspace(0.15, 0.85, 32):
            profile = np.clip(value + shift, 0.0, 1.0) * np.ones(40)
            profiles.append(profile)
            boundary.append(value - 0.5 + shift)
            means.append(0.4 * (value - 0.5))
            domains.append(domain)
    model = SourceAlignedBoundaryCoordinate(
        latent_dim=3,
        feature_mode="linear",
        latent_transform="source_support_residual",
    ).fit(profiles, boundary, domains, coefficient_targets=means)
    in_support = model.features_profile(profiles[len(profiles) // 2])
    extreme = model.features_profile(np.full(40, 100.0))
    diagnostics = model.diagnostics()["latent_transform_diagnostics"]

    assert model.feature_dim == 4
    assert diagnostics["residual_channel"] is True
    assert diagnostics["residual_channel_index"] == 3
    assert 0.0 <= in_support[-1] <= 1.0
    assert 0.0 < extreme[-1] <= 1.0
    assert np.all(np.isfinite(extreme))


def test_boundary_feature_library_supports_nonlinear_chance_geometry():
    rng = np.random.default_rng(126)
    profiles = rng.uniform(0.0, 1.0, size=(96, 50))
    domains = np.asarray(["a"] * 48 + ["b"] * 48, dtype=object)
    local = np.column_stack([
        np.mean(profiles[:, :17], axis=1),
        np.mean(profiles[:, 17:34], axis=1),
        np.mean(profiles[:, 34:], axis=1),
    ])
    boundary = (
        (local[:, 0] - 0.55) ** 2
        + (local[:, 1] - 0.35) ** 2
        + (local[:, 2] - 0.45) ** 2
        - 0.025
    )
    means = local[:, 0] - local[:, 1]
    expected = {
        "linear": 3,
        "diagonal_quadratic": 6,
        "full_quadratic": 9,
    }
    for mode, dimension in expected.items():
        model = SourceAlignedBoundaryCoordinate(
            latent_dim=3, feature_mode=mode).fit(
                profiles,
                boundary,
                domains,
                coefficient_targets=means,
            )
        features = model.features_many(
            InventorySupplyChainProblem(d=50),
            [tuple(np.rint(100.0 * row).astype(int)) for row in profiles[:5]],
        )
        assert model.feature_dim == dimension
        assert features.shape == (5, dimension)
        assert np.all(np.isfinite(features))
        assert model.diagnostics()["boundary_feature_mode"] == mode


def test_boundary_coordinate_selector_allocates_safe_boundary_and_coverage():
    features = np.column_stack([
        np.linspace(-2.0, 2.0, 21),
        np.cos(np.linspace(0.0, 2.0 * np.pi, 21)),
    ])
    posterior_mean = np.linspace(-1.0, 1.0, 21)
    posterior_variance = np.linspace(0.05, 0.20, 21)
    chance_margin = posterior_mean + 0.10
    selection = select_boundary_coordinate_candidates(
        features,
        features[[0, 10]],
        posterior_mean,
        posterior_variance,
        chance_margin,
        count=10,
        safe_fraction=0.30,
        boundary_fraction=0.40,
        coverage_fraction=0.30,
    )
    assert len(selection.indices) == 10
    assert len(set(selection.indices)) == 10
    assert set(selection.roles) == {"safe", "boundary", "coverage"}
    assert selection.diagnostics["role_counts"] == {
        "boundary": 4,
        "coverage": 3,
        "safe": 3,
    }
    assert selection.diagnostics["target_oracle_used"] is False
