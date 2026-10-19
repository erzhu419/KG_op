"""Catch inherited outcomes, stream collisions and four-cell budget/source drift."""
from copy import deepcopy

import pytest

from performance import register_mainline_reference_coverage_confirmation as reg
from performance import run_mainline_observation_seed_replication as replication


def fixture_plan():
    reference = reg.read(reg.ROOT / reg.REFERENCE)
    vectors = [dict(task_geometry_shift=[i / 10000, 0., 0.], task_geometry_radius_scale=1.) for i in range(24)]
    return reference, reg.make_plan(reference, vectors, 'substitute-time')


def test_four_cell_registration_preserves_sources_search_and_all_controls(tmp_path, monkeypatch):
    reference, plan = fixture_plan()
    before = deepcopy(reference)
    for key in ('search_arms', 'terminal_arms', 'randomness', 'old_protocol', 'candidate_protocol'):
        assert plan[key] == reference[key]
    assert {k: v for k, v in plan['frozen'].items() if k != 'terminal'} == {
        k: v for k, v in reference['frozen'].items() if k != 'terminal'}
    assert not {'completed_utc', 'summary_file', 'execution_failures', 'execution_started_utc'} & plan.keys()
    assert plan['primary_contrast'] == ['reference_two_comparison', 'original_old']
    assert plan['budget']['registered_terminal_cells'] == 24 * 9 * 4
    assert plan['budget']['maximum_target_physical_calls'] == 111264
    assert plan['budget']['maximum_logical_target_calls_four_variants'] == 315072
    assert len({t['geometry_task_seed'] for t in plan['tasks']}) == 24
    parent = reg.read(reg.ROOT / plan['source_manifest'])
    for family in parent['families']:
        for key in ('search_manifest', 'terminal_manifest'):
            family[key] = str(reg.ROOT / family[key])
    reg.save(tmp_path / 'parent.json', parent)
    plan['source_manifest'] = str(tmp_path / 'parent.json')
    monkeypatch.setattr(replication, 'ROOT', tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError('Metadata preparation must not launch workers or draw target observations')
    monkeypatch.setattr(replication.subprocess, 'run', forbidden)
    monkeypatch.setattr(reg.np.random, 'default_rng', forbidden)
    prepared = replication.materialize(plan)
    for family in prepared['replicates'][0]['families']:
        source = reg.read(tmp_path / family['search_manifest'])
        terminal = reg.read(tmp_path / family['terminal_manifest'])
        original = reg.read(next(f['search_manifest'] for f in parent['families'] if f['domain'] == family['domain']))
        for key in ('task', 'anchor', 'frozen_assets', 'frontends', 'role_prior', 'point_only', 'after_init_point_only'):
            assert source[key] == original[key]
        assert len(source['target_seeds']) == 8 and terminal['arms'] == reference['terminal_arms']
        assert 'fixed_initialization_file_by_task' not in source and 'fixed_neighbors_file_by_task' not in source
        for seed in source['target_seeds']:
            assert source['observation_seed_by_task'][str(seed)] == seed + 100
            assert terminal['verification_seed_by_task'][str(seed)] == seed + 200
    assert reference == before
    assert not list(tmp_path.rglob('search.json')) and not list(tmp_path.rglob('truth.json'))


def test_registration_rejects_previous_seed_or_geometry_without_replacement():
    reference, plan = fixture_plan()
    reg.check_freshness(plan, [reference])
    with pytest.raises(AssertionError, match='seed collision'):
        reg.check_freshness(plan, [dict(verification_seed=2026110201)])
    with pytest.raises(AssertionError):
        reg.check_freshness(plan, [plan['tasks'][0]['geometry']])


def test_registration_requires_cluster_and_passed_checks_before_metadata_draw(tmp_path, monkeypatch):
    monkeypatch.setattr(reg, 'ROOT', tmp_path)
    monkeypatch.setattr(reg.socket, 'gethostname', lambda: 'node001')
    def forbidden(*args, **kwargs):
        raise AssertionError('Should not draw metadata before checks')
    monkeypatch.setattr(reg.np.random, 'default_rng', forbidden)
    reg.save(tmp_path / 'failed.json', dict(status='failed'))
    with pytest.raises(AssertionError, match='Substitute checks'):
        reg.register(checks='failed.json')
    monkeypatch.setattr(reg.socket, 'gethostname', lambda: 'local')
    with pytest.raises(AssertionError, match='local'):
        reg.register(checks='failed.json')
