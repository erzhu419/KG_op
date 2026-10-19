"""Target metadata substitutes: no registered RNG, simulation or model loads."""
from copy import deepcopy

import pytest

from performance import prepare_mainline_source_alignment_targets as reg


def inputs():
    control = reg.read(reg.ROOT / reg.CONTROL)
    reference = reg.read(reg.ROOT / control['reference_manifest'])
    geometries = [dict(task_geometry_shift=[i / 20000, 0., 0.],
        task_geometry_radius_scale=1.) for i in range(144)]
    return control, reference, geometries


def test_full_metadata_preserves_sources_old_arms_randomness_and_budgets():
    control, reference, geometries = inputs()
    before = deepcopy(control), deepcopy(reference)
    plan, children = reg.make_plans(control, reference, geometries, 'fixture')
    assert (control, reference) == before
    assert len(plan['tasks']) == 144 and len(plan['search_arms']) == 9 and len(plan['terminal_arms']) == 11
    assert plan['terminal_arms'][:9] == reference['terminal_arms']
    assert plan['randomness'] == reference['randomness']
    for key in ('d', 'search_calls', 'n0', 'adaptive_calls', 'candidate_budgets',
                'objective_comparison_budget_per_policy', 'familywise_delta',
                'objective_comparison_delta', 'terminal_maximum_violation_probability', 'verification_method'):
        assert plan['frozen'][key] == reference['frozen'][key]
    assert children[0][1]['tasks'] == children[1][1]['tasks'] == plan['tasks']
    assert [s['name'] for s in plan['streams']] == ['stream1', 'stream2']
    assert all(sum(t['domain'] == f['domain'] for t in plan['tasks']) == 48 for f in control['families'])
    mapping = [v for _, c in children for row in c['replicates'][0]['seed_by_task'].values() for v in row.values()]
    assert len(set(mapping)) == 576
    assert children[0][1]['replicates'][0]['seed_by_task']['2026200001'] == {
        'observation_seed': 2026210001, 'verification_seed': 2026220001}
    assert children[1][1]['replicates'][0]['seed_by_task']['2026200144'] == {
        'observation_seed': 2026230144, 'verification_seed': 2026240144}
    assert plan['saved_score_contract']['planned_saved_query_states'] == 4320
    assert not plan['target_execution_authorized'] and not plan['target_execution_started']
    assert plan['target_execution_preparation']['status'] == 'pending'
    budget = plan['budget']
    for key, expected in dict(unique_search_histories=2592, registered_terminal_cells=3168,
        terminal_cells=3168, initial_physical_calls=2880, online_search_physical_calls=6912,
        expanded_logical_search_calls=40320, maximum_verification_calls_per_cell=352,
        maximum_target_physical_calls=1124928, reachable_successful_target_physical_calls=1074240,
        maximum_expanded_logical_target_calls=1155456).items():
        assert budget[key] == expected


def test_family_materialization_keeps_frontend_and_changes_only_added_backend_paths(tmp_path, monkeypatch):
    control, reference, geometries = inputs()
    _, children = reg.make_plans(control, reference, geometries, 'fixture')
    parent = reg.read(reg.ROOT / reference['source_manifest'])
    reg.save(tmp_path / reference['source_manifest'], parent)
    originals = {}
    for family in parent['families']:
        for key in ('search_manifest', 'terminal_manifest'):
            data = reg.read(reg.ROOT / family[key])
            reg.save(tmp_path / family[key], data)
            if key == 'search_manifest':
                originals[family['domain']] = data
    monkeypatch.setattr(reg.replication, 'ROOT', tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError('Metadata cannot draw RNG or execute a process')
    monkeypatch.setattr(reg.np.random, 'PCG64', forbidden)
    monkeypatch.setattr(reg.replication.subprocess, 'run', forbidden)
    for _, child in children:
        prepared = reg.materialize(child, tmp_path)
        for family in prepared['replicates'][0]['families']:
            source = reg.read(tmp_path / family['search_manifest'])
            original = originals[family['domain']]
            for key in ('task', 'anchor', 'frozen_assets', 'frontends', 'neighbor_count',
                        'neighbor_radius', 'neighbor_safe_margin_scale', 'role_prior',
                        'point_only', 'after_init_point_only'):
                assert source[key] == original[key], key
            assert source['source_permuted_assets'] == child['source_permuted_assets_by_family'][family['domain']]
            assert source['arms'] == [dict(name=a, online=a != 'initialization_only') for a in child['search_arms']]
            assert 'fixed_initialization_file_by_task' not in source and 'fixed_neighbors_file_by_task' not in source
            terminal = reg.read(tmp_path / family['terminal_manifest'])
            assert terminal['arms'] == control['terminal_arms']
            assert terminal['protocols'] == {control['old_protocol']: [80, 128, 128]}
    assert not list(tmp_path.rglob('search.json')) and not list(tmp_path.rglob('truth.json'))


def test_seal_requires_all_six_assets_without_loading_them(tmp_path):
    control, _, _ = inputs()
    seal = reg.read(reg.ROOT / reg.SEAL)
    fixture = deepcopy(seal)
    for family in fixture['families']:
        for model in family['models']:
            path = tmp_path / model['model_file']
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'not a pickle; metadata only')
            stat = path.stat()
            model.update(size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    retained = reg.require_sources(control, fixture, tmp_path)
    assert retained['families'][1]['low_ess_head_diagnostics'] == 6
    assert '99.84' in retained['numerical_limits']
    missing = tmp_path / fixture['families'][1]['models'][1]['model_file']
    missing.unlink()
    with pytest.raises(FileNotFoundError):
        reg.require_sources(control, fixture, tmp_path)


def test_prechecks_and_interrupted_registration_never_redraw(tmp_path, monkeypatch):
    monkeypatch.setattr(reg.socket, 'gethostname', lambda: 'node006')
    def forbidden(*args, **kwargs):
        raise AssertionError('No RNG before checks or after an interrupted registration')
    monkeypatch.setattr(reg.np.random, 'PCG64', forbidden)
    reg.save(tmp_path / 'checks.json', dict(status='failed'))
    with pytest.raises(AssertionError, match='checks must pass'):
        reg.register('checks.json', tmp_path)
    reg.save(tmp_path / 'checks.json', dict(status='passed'))
    reg.save(tmp_path / reg.OUTPUT / 'target_registration/started.json', dict(status='started'))
    with pytest.raises(AssertionError, match='do not redraw'):
        reg.register('checks.json', tmp_path)


def test_registered_seed_collisions_stop_without_selecting_replacement():
    control, reference, geometries = inputs()
    plan, children = reg.make_plans(control, reference, geometries, 'fixture')
    reg.check_freshness(plan, children, [reference])
    for prior in (dict(target_seeds=[2026200001]),
                  dict(seed_by_task={'2026200144': {}}),
                  dict(verification_seed=2026240144)):
        with pytest.raises(AssertionError, match='seed collision'):
            reg.check_freshness(plan, children, [prior])
    with pytest.raises(AssertionError, match='geometry collision'):
        reg.check_freshness(plan, children, [plan['tasks'][0]['geometry']])
