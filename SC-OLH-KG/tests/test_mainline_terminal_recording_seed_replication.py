"""Metadata fixtures for stream routing, fixed identities and collision detection."""
from copy import deepcopy

import pytest

from performance import register_mainline_terminal_recording_seed_replication as reg
from performance import run_mainline_observation_seed_replication as replication


def test_fixed_geometry_streams_materialize_without_sampling_or_stale_primary(tmp_path, monkeypatch):
    reference = reg.read(reg.ROOT/reg.REFERENCE)
    before = deepcopy(reference)
    outer, children = reg.plans(reference, 'fixture-time')
    assert reference == before
    assert outer['status'] == 'registered_not_started' and not outer['execution_started']
    assert outer['primary_terminal'] == 'old' and outer['primary_candidate'] == reg.AFTER
    assert outer['primary_reference'] == reg.POINT
    assert outer['budget']['maximum_target_physical_calls'] == 160320
    assert outer['budget']['maximum_logical_target_calls_both_protocols'] == 315072
    assert outer['budget']['registered_terminal_cells'] == 864
    assert outer['historical_accounts'][:len(reference['historical_accounts'])] == reference['historical_accounts']
    assert 'Stage65' not in str(outer) and 'every frozen action_ids' in outer['analysis']['truth']
    assert all(child['analysis']['primary'] == outer['analysis']['primary'] for _,child in children)
    parent = reg.read(reg.ROOT/reference['source_manifest'])
    for family in parent['families']:
        for field in ('search_manifest','terminal_manifest'):
            family[field] = str(reg.ROOT/family[field])
    reg.save(tmp_path/'parent.json',parent)
    monkeypatch.setattr(replication, 'ROOT', tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError('Metadata materialization must not start science')
    monkeypatch.setattr(replication.subprocess, 'run', forbidden)
    families = []
    for number, (_,child) in enumerate(children,1):
        child['source_manifest'] = str(tmp_path/'parent.json')
        assert {k:v for k,v in child['frozen']['new_terminal'].items() if k not in ('independence','reuse')} == {
            k:v for k,v in reference['frozen']['new_terminal'].items() if k not in ('independence','reuse')}
        assert child['terminal_arms'] == reference['terminal_arms']
        for old,task in zip(reference['tasks'], child['tasks'], strict=True):
            assert {k:task[k] for k in old} == old
            assert reference['output_dir'] in task['initial_policy_file']
            assert reference['output_dir'] in task['shared_neighbors_file']
        prepared = replication.materialize(child)
        families.extend(prepared['replicates'][0]['families'])
        for family in prepared['replicates'][0]['families']:
            source = reg.read(tmp_path/family['search_manifest'])
            terminal = reg.read(tmp_path/family['terminal_manifest'])
            assert len(source['target_seeds']) == 8
            for seed in source['target_seeds']:
                assert source['observation_seed_by_task'][str(seed)] == seed + (1100 if number == 1 else 2100)
                assert terminal['verification_seed_by_task'][str(seed)] == seed + (1200 if number == 1 else 2200)
                assert source['fixed_initialization_file_by_task'][str(seed)].endswith(f'seed_{seed}/initialization.json')
                assert source['fixed_neighbors_file_by_task'][str(seed)].endswith(f'seed_{seed}/neighbors.npz')
            assert '864' in terminal['truth_barrier']
    assert len(families) == 6
    assert not list(tmp_path.rglob('search.json')) and not list(tmp_path.rglob('truth.json'))


def test_new_stream_collisions_include_nested_seed_maps_and_lists():
    reference = reg.read(reg.ROOT/reg.REFERENCE)
    _,children = reg.plans(reference, 'fixture-time')
    reg.check_seed_collisions(children, [reference])
    with pytest.raises(AssertionError, match='seed collision'):
        reg.check_seed_collisions(children, [dict(replicates=[dict(seed_by_task={'1':dict(observation_seed=2026108101)})])])
    with pytest.raises(AssertionError, match='seed collision'):
        reg.check_seed_collisions(children, [dict(target_seeds=[2026109224])])
