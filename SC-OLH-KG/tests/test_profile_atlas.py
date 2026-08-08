import itertools
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.profile_atlas import (  # noqa: E402
    ProfileAtlasConfig,
    SourceProfileRecord,
    SourceScoredProfileAtlas,
    covering_radius,
    farthest_first_indices,
    generic_dct_maximin,
    percentile_ranks,
    profile_cosine_coordinate,
    profile_quadrature_weights,
    regular_profile_nodes,
)


def _records():
    nodes = regular_profile_nodes(32)
    profiles = {
        "low": np.full(32, 0.2),
        "ramp": np.linspace(0.1, 0.9, 32),
        "high": np.full(32, 0.8),
        "wave": 0.5 + 0.25 * np.cos(np.pi * nodes),
    }
    rows = []
    for task_index, descriptor in enumerate(((0.0,), (1.0,))):
        for profile_index, (profile_id, profile) in enumerate(profiles.items()):
            objective = 0.1 * profile_index + 0.02 * task_index
            constraint = -0.4 + 0.25 * profile_index + 0.01 * task_index
            rows.append(SourceProfileRecord(
                task_id=f"source_{task_index}",
                profile_id=profile_id,
                profile=tuple(profile),
                objective_samples=(objective - 0.01, objective, objective + 0.01),
                constraint_samples=(
                    constraint - 0.01, constraint, constraint + 0.01),
                descriptor=descriptor,
                nodes=tuple(nodes),
            ))
    return rows


def test_irregular_quadrature_and_cosine_coordinate_are_finite_and_stable():
    nodes = np.asarray([0.03, 0.11, 0.31, 0.72, 0.96])
    weights = profile_quadrature_weights(nodes)
    assert np.all(weights > 0.0)
    assert np.isclose(np.sum(weights), 1.0)
    profile = 0.4 + 0.2 * np.cos(np.pi * nodes)
    coordinate = profile_cosine_coordinate(
        profile, nodes=nodes, max_frequency=3)
    assert coordinate.shape == (8,)
    assert np.all(np.isfinite(coordinate))


def test_cosine_coordinate_is_consistent_under_grid_refinement():
    coarse = regular_profile_nodes(64)
    fine = regular_profile_nodes(4096)
    function = lambda nodes: 0.45 + 0.18 * np.cos(np.pi * nodes) - 0.07 * np.cos(
        3.0 * np.pi * nodes)
    coarse_coordinate = profile_cosine_coordinate(
        function(coarse), nodes=coarse, max_frequency=5,
        include_diagonal_quadratic=False)
    fine_coordinate = profile_cosine_coordinate(
        function(fine), nodes=fine, max_frequency=5,
        include_diagonal_quadratic=False)
    assert np.max(np.abs(coarse_coordinate - fine_coordinate)) < 2e-4


def test_percentile_ranks_use_average_ties():
    ranks = percentile_ranks([3.0, 1.0, 1.0, 4.0])
    assert np.allclose(ranks, [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0, 1.0])


def test_farthest_first_has_two_approx_radius_on_small_finite_metric():
    coordinates = np.asarray([[0.0], [0.2], [0.4], [1.0], [1.1]])
    selected = farthest_first_indices(coordinates, 2, initial_index=1)
    greedy_radius = covering_radius(coordinates, selected)
    optimum = min(
        covering_radius(coordinates, centers)
        for centers in itertools.combinations(range(len(coordinates)), 2)
    )
    assert greedy_radius <= 2.0 * optimum + 1e-12


def test_profile_atlas_is_source_only_deterministic_and_dimension_equivariant():
    config = ProfileAtlasConfig(n0=3, max_frequency=4)
    first = SourceScoredProfileAtlas(config).fit(
        _records(), target_descriptor=(0.25,)).selected()
    second = SourceScoredProfileAtlas(config).fit(
        _records(), target_descriptor=(0.25,)).selected()
    assert first == second
    assert first.diagnostics["target_outcomes_used"] is False
    assert first.diagnostics["target_oracle_used"] is False
    assert first.diagnostics["descriptor_conditioned"] is True
    assert len(first.members) == 3
    low_dimension = first.target_profiles(17)
    high_dimension = first.target_profiles(1000)
    assert len(low_dimension[0]) == 17
    assert len(high_dimension[0]) == 1000
    assert [member.profile_id for member in first.members] == [
        member.profile_id for member in second.members]


def test_generic_dct_control_never_receives_source_outcomes():
    profiles = [record.profile for record in _records()[:4]]
    selected, audit = generic_dct_maximin(profiles, 3, max_frequency=4)
    assert len(selected) == 3
    assert len(set(selected)) == 3
    assert audit["source_outcomes_used"] is False
    assert audit["target_outcomes_used"] is False
    assert audit["target_oracle_used"] is False
