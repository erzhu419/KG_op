"""Metadata-only checks for fresh two-stream, old-only cohort registration."""
from copy import deepcopy

import numpy as np
import pytest

from performance import prepare_mainline_precision_confirmation as reg


def inputs(n=24):
    reference = reg.read(reg.ROOT / reg.REFERENCE)
    geometries = [dict(task_geometry_shift=[i / 20000, 0., 0.],
        task_geometry_radius_scale=1.) for i in range(n)]
    preparation = dict(name='mainline_precision_confirmation_preparation_v1',
        output_dir='paper_artifacts/mainline_precision_confirmation_preparation_v1')
    precision = dict(status='selected', selection_pass=True, selected_N=n)
    return reference, geometries, preparation, precision


def stage_sources(tmp_path, reference):
    """Copy only small source recipes; model/replay files are unopened fixtures."""
    parent = reg.read(reg.ROOT / reference['source_manifest'])
    for family in parent['families']:
        source = reg.read(reg.ROOT / family['search_manifest'])
        for key, value in source['frozen_assets'].items():
            if key.endswith('_file'):
                target = tmp_path / value
                target.parent.mkdir(parents=True, exist_ok=True)
                target.touch()
        reg.save(tmp_path / family['search_manifest'], source)
        terminal = reg.read(reg.ROOT / family['terminal_manifest'])
        reg.save(tmp_path / family['terminal_manifest'], terminal)
        reg.save(tmp_path / terminal['output_dir'] / 'replay_check.json', dict(status='passed'))
    reg.save(tmp_path / reference['source_manifest'], parent)
    reg.save(tmp_path / reg.REFERENCE, reference)


def test_paired_streams_preserve_frozen_old_rule_and_separate_realized_banks(tmp_path, monkeypatch):
    reference, geometries, preparation, precision = inputs()
    before = deepcopy(reference)
    plan, children = reg.make_plans(reference, geometries, preparation, precision, 'fixture-time')
    assert reference == before and plan['selected_N'] == 24
    assert plan['terminal_protocols'] == ['old'] and 'candidate_protocol' not in plan
    assert 'new_terminal' not in plan['frozen']
    assert plan['search_arms'] == reference['search_arms']
    assert plan['terminal_arms'] == reference['terminal_arms']
    for key in ('d', 'search_calls', 'n0', 'adaptive_calls', 'candidate_budgets',
                'objective_comparison_budget_per_policy', 'familywise_delta',
                'objective_comparison_delta', 'terminal_maximum_violation_probability', 'verification_method'):
        assert plan['frozen'][key] == reference['frozen'][key]
    assert plan['execution_preparation']['entrypoint'] is None
    assert not plan['execution_started'] and not plan['execution_preparation']['independent_scientific_execution']
    assert children[0][1]['tasks'] == children[1][1]['tasks'] == plan['tasks']
    assert all(t['optimizer_seed'] == t['geometry_task_seed'] for t in plan['tasks'])
    assert all('initial_policy_file' not in t and 'shared_neighbors_file' not in t for t in plan['tasks'])
    assert {d: sum(t['domain'] == d for t in plan['tasks']) for d in {t['domain'] for t in plan['tasks']}} == {
        d: 8 for d in {t['domain'] for t in plan['tasks']}}
    stage_sources(tmp_path, reference)
    monkeypatch.setattr(reg.replication, 'ROOT', tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError('Metadata materialization cannot run science or draw geometries')
    monkeypatch.setattr(reg.replication.subprocess, 'run', forbidden)
    monkeypatch.setattr(reg.np.random, 'default_rng', forbidden)
    prepared = [reg.replication.materialize(child) for _, child in children]
    result = reg.validate_metadata(plan, children, prepared, root=tmp_path)
    assert result['family_plans'] == 6 and result['search_histories'] == 336
    assert result['terminal_cells'] == 432
    seed_maps = [child['replicates'][0]['seed_by_task'] for _, child in children]
    seeds = [value for mapping in seed_maps for row in mapping.values() for value in row.values()]
    assert len(set(seeds)) == 96
    assert not list(tmp_path.rglob('search.json')) and not list(tmp_path.rglob('truth.json'))


@pytest.mark.parametrize('n', [24, 384])
def test_cost_bound_uses_shared_initialization_and_old_only_mutually_exclusive_branches(n):
    budget = reg.budget_for(n)
    assert budget['unique_search_histories'] == 14 * n
    assert budget['registered_terminal_cells'] == 18 * n
    assert budget['initial_physical_calls'] == 20 * n
    assert budget['online_search_physical_calls'] == 36 * n
    assert budget['reachable_successful_target_physical_calls'] == 6104 * n
    assert budget['maximum_target_physical_calls'] == 6392 * n
    assert budget['maximum_expanded_logical_target_calls'] == 6564 * n
    assert budget['maximum_verification_calls_per_cell'] == 352


def test_collision_scan_covers_all_streams_map_keys_lists_and_geometry():
    reference, geometries, preparation, precision = inputs(384)
    plan, children = reg.make_plans(reference, geometries, preparation, precision, 'fixture-time')
    reg.check_freshness(plan, children, [reference])
    collisions = [dict(target_seeds=[2026160384]),
        dict(nested=dict(seed_by_task={'2026120001': {'other': 0}})),
        dict(replicates=[dict(seed_by_task={'1': dict(observation_seed=2026150001)})])]
    for document in collisions:
        with pytest.raises(AssertionError, match='seed collision'):
            reg.check_freshness(plan, children, [document])
    with pytest.raises(AssertionError, match='geometry collision'):
        reg.check_freshness(plan, children, [dict(geometry=plan['tasks'][-1]['geometry'])])


def test_registration_gates_and_unmet_precision_do_not_draw_metadata(tmp_path, monkeypatch):
    _, _, preparation, precision = inputs()
    monkeypatch.setattr(reg.socket, 'gethostname', lambda: 'local')
    def forbidden(*args, **kwargs):
        raise AssertionError('No metadata RNG before successful admission and selected N')
    monkeypatch.setattr(reg.np.random, 'default_rng', forbidden)
    with pytest.raises(AssertionError, match='local'):
        reg.register(preparation, precision, 'checks.json', root=tmp_path)
    monkeypatch.setattr(reg.socket, 'gethostname', lambda: 'node006')
    reg.save(tmp_path / 'checks.json', dict(status='failed'))
    with pytest.raises(AssertionError, match='Substitute checks'):
        reg.register(preparation, precision, 'checks.json', root=tmp_path)
    reg.save(tmp_path / 'checks.json', dict(status='passed'))
    result = reg.register(preparation, dict(status='hold', selected_N=None), 'checks.json', root=tmp_path)
    assert result['status'] == 'not_registered_precision_target_unmet'
    assert result['geometry_uniform_draws'] == result['new_target_initial_calls'] == 0
    assert not (tmp_path / reg.MANIFEST).exists()


def test_registration_only_draws_metadata_once_and_retains_selected_plan(tmp_path, monkeypatch):
    reference, geometries, preparation, precision = inputs()
    stage_sources(tmp_path, reference)
    monkeypatch.setattr(reg.replication, 'ROOT', tmp_path)
    monkeypatch.setattr(reg.socket, 'gethostname', lambda: 'node003')
    reg.save(tmp_path / 'checks.json', dict(status='passed'))
    class MetadataRng:
        def __init__(self):
            self.index = 0
        def uniform(self, low, high, size=None):
            geometry = geometries[self.index]
            if size == 3:
                assert (low, high) == (-.035, .035)
                return np.asarray(geometry['task_geometry_shift'])
            assert (low, high, size) == (.95, 1.05, None)
            self.index += 1
            return geometry['task_geometry_radius_scale']
    draws = []
    def fixture_rng(seed):
        draws.append(seed)
        return MetadataRng()
    monkeypatch.setattr(reg.np.random, 'default_rng', fixture_rng)
    def forbidden(*args, **kwargs):
        raise AssertionError('Registration cannot launch science')
    monkeypatch.setattr(reg.replication.subprocess, 'run', forbidden)
    result = reg.register(preparation, precision, 'checks.json', root=tmp_path)
    assert result['status'] == 'registered_preparation_complete'
    assert result['geometry_uniform_draws'] == 96 and result['models_loaded'] == 0
    assert not result['truth_read'] and not result['execution_started']
    assert all(result[key] == 0 for key in reg.ZERO)
    assert reg.register(preparation, precision, 'checks.json', root=tmp_path) == result
    assert draws == [reg.GEOMETRY_SEED]
    assert (tmp_path / preparation['output_dir'] / 'registration/attempts.jsonl').is_file()
    assert not (tmp_path / reg.OUTPUT / 'registration/attempts.jsonl').exists()
    plan = reg.read(tmp_path / reg.MANIFEST)
    assert plan['status'] == 'registered_not_started' and plan['selected_N'] == 24
    assert not list(tmp_path.rglob('search.json')) and not list(tmp_path.rglob('truth.json'))
