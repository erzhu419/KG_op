import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from problems.randomized_profiles import (  # noqa: E402
    PROFILE_STRESS_REGIMES,
    RandomizedOrderedProfileProblem,
    generate_structural_profile_library,
    source_profile_records,
)


def test_every_registered_profile_regime_is_finite_and_nontrivial():
    library = generate_structural_profile_library(64, dimension=96, seed=7)
    for regime in PROFILE_STRESS_REGIMES:
        task = RandomizedOrderedProfileProblem(
            regime=regime,
            role="target",
            task_seed=3,
            family_seed=11,
            d=128,
            calibration_library=library,
        )
        points = [task.point_from_structural_profile(item) for item in library]
        objectives = np.asarray([task.true_objective(point) for point in points])
        probabilities = np.asarray([
            task.true_feasibility_probability(point) for point in points])
        assert np.all(np.isfinite(objectives))
        assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
        assert 1 <= task.effective_rank <= task.d
        assert np.any([task.is_truly_feasible(point) for point in points])


def test_declared_permutation_schema_round_trips_semantic_profile():
    task = RandomizedOrderedProfileProblem(
        regime="coordinate_permutation",
        role="target",
        task_seed=5,
        d=64,
    )
    semantic = 0.4 + 0.2 * np.cos(np.pi * task.nodes)
    point = task.encode_semantic_profile(semantic)
    recovered = task.semantic_profile(point)
    assert np.max(np.abs(recovered - semantic)) <= 0.0051


def test_chance_margin_matches_declared_probability_boundary():
    task = RandomizedOrderedProfileProblem(
        regime="aligned_low_frequency",
        role="target",
        task_seed=2,
        d=64,
    )
    point = task.encode_semantic_profile(task._safe_profile)
    probability = task.true_feasibility_probability(point)
    assert task.true_chance_margin(point) < 0.0
    assert probability > 1.0 - task.alpha


def test_source_archive_contains_only_replicated_source_observations():
    library = generate_structural_profile_library(16, dimension=32, seed=9)
    tasks = [
        RandomizedOrderedProfileProblem(
            regime="aligned_low_frequency",
            role="source",
            task_seed=seed,
            d=48,
            calibration_library=library,
        )
        for seed in (1, 2)
    ]
    records = source_profile_records(tasks, library, replications=3, seed=13)
    assert len(records) == 2 * len(library)
    assert all(len(record.objective_samples) == 3 for record in records)
    assert all(record.task_id.startswith("RandomizedProfile") for record in records)
