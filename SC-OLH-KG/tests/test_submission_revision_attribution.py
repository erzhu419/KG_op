from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pytest
from core.profile_atlas import ProfileAtlasConfig, SourceScoredProfileAtlas
from problems.randomized_profiles import PROFILE_STRESS_REGIMES, RandomizedOrderedProfileProblem, generate_structural_profile_library, source_profile_records
from performance.benchmark_profile_stress_suite import _stable_seed
from performance.submission_revision_attribution import selections
from performance.submission_revision_controls import standardized_generic_dct_maximin


@pytest.mark.parametrize("regime", PROFILE_STRESS_REGIMES)
def test_factorial_preserves_both_existing_methods_and_fixes_first_centers(regime):
    family = 20260808
    library = generate_structural_profile_library(64, dimension=128, seed=family+991, maximum_frequency=40)
    sources = [RandomizedOrderedProfileProblem(regime=regime, role="source", task_seed=10000+i, family_seed=family+_stable_seed(regime), d=256) for i in range(2)]
    records = source_profile_records(sources, library, replications=3, seed=family+1237)
    designs = selections(records)
    ids = lambda selection: [m.profile_id for m in selection.members]
    assert ids(designs["source_atlas"]) == ids(SourceScoredProfileAtlas(ProfileAtlasConfig()).fit(records).selected())
    generic, _ = standardized_generic_dct_maximin([p.values for p in library], 10, nodes=library[0].nodes, max_frequency=8, frequency_penalty=.25, include_diagonal_quadratic=True)
    assert ids(designs["standardized_generic_dct_maximin"]) == [library[i].profile_id for i in generic]
    assert designs["source_best_z"].members[0].profile_id == designs["source_atlas"].members[0].profile_id == designs["source_best_raw_l2"].members[0].profile_id
    assert designs["medoid_augmented"].members[0].profile_id == designs["standardized_generic_dct_maximin"].members[0].profile_id
    assert all(len(ids(d)) == len(set(ids(d))) == 10 for d in designs.values())
