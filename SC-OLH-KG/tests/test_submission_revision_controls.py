"""Checks detect selection-rule mistakes before expensive supplemental cells."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.profile_atlas import ProfileAtlasConfig, SourceProfileRecord, SourceScoredProfileAtlas
from performance.submission_revision_controls import (
    greedy_portfolio_indices, source_greedy_portfolio,
    standardized_generic_dct_maximin,
)
from problems.randomized_profiles import generate_structural_profile_library


def test_portfolio_selects_complementary_task_winners_before_better_average():
    # Mean-only selection would choose 0,3; the specified portfolio needs 0,1.
    scores = [[0, 1, 0.8, 0.4], [0.8, 0, 0.8, 0.5]]
    selected, path = greedy_portfolio_indices(scores, 4)
    assert selected == (0, 1, 3, 2)
    assert path == [0.4, 0.0, 0.0, 0.0]


def test_portfolio_stable_ties_and_source_statistics_match_atlas():
    records = tuple(SourceProfileRecord(
        task_id=f"task{task}", profile_id=f"profile{profile}",
        profile=(0.1 + profile * 0.2,) * 8,
        objective_samples=(float(profile + task), float(profile + task + 1)),
        constraint_samples=(float(3-profile), float(3-profile) + 0.1),
    ) for task in range(2) for profile in range(4))
    selected = source_greedy_portfolio(records, 4)
    original = SourceScoredProfileAtlas(ProfileAtlasConfig(n0=4, max_frequency=2)).fit(records).selected()
    lookup = {m.profile_id: m for m in original.members}
    for member in selected.members:
        assert member.safety_rank == lookup[member.profile_id].safety_rank
        assert member.objective_rank == lookup[member.profile_id].objective_rank
    assert [m.profile_id for m in selected.members] == [f"profile{i}" for i in range(4)]
    assert selected.diagnostics["target_outcomes_used"] is False


def test_standardized_control_matches_existing_real_library_diagnostic():
    # Detect a different node convention, column scaling or first-center rule.
    import json
    library = generate_structural_profile_library(64, dimension=128, seed=20261799, maximum_frequency=40)
    indices, diagnostics = standardized_generic_dct_maximin(
        [p.values for p in library], 10, nodes=library[0].nodes,
        max_frequency=8, frequency_penalty=0.25, include_diagonal_quadratic=True)
    previous = json.loads((ROOT / "docs/submission_review_20260907/kg_standardized_generic_diagnostic.json").read_text())["rows"]
    ids = [library[index].profile_id for index in indices]
    assert all(ids == row["selected_profile_ids"] for row in previous)
    assert diagnostics["source_outcomes_used"] is False
