"""Exercise registration and global stage ordering without executing any worker."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from performance import run_mainline_observation_seed_replication as runner

PROJECT = runner.ROOT


@pytest.fixture
def registration(tmp_path, monkeypatch):
    plan = runner.read(runner.MANIFEST)
    parent = runner.read(PROJECT / plan['source_manifest'])
    for family in parent['families']:
        for key in ('search_manifest', 'terminal_manifest'):
            family[key] = str(PROJECT / family[key])
    parent_file = tmp_path / 'parent.json'
    runner.save(parent_file, parent)
    plan['source_manifest'] = str(parent_file)
    plan['execution_preparation']['status'] = 'validated'
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    return plan, tmp_path


@pytest.fixture
def fresh_registration(registration):
    plan, root = registration
    parent = runner.read(plan['source_manifest'])
    old_tasks = plan['tasks']
    tasks = []
    for family in parent['families']:
        domain = family['domain']
        source = runner.read(family['search_manifest'])
        # A fresh cohort must not inherit physical banks from any copied template.
        source['fixed_initialization_file_by_task'] = {'old': 'old/initialization.json'}
        source['fixed_neighbors_file_by_task'] = {'old': 'old/neighbors.npz'}
        family['search_manifest'] = str(root / f'{domain}_template.json')
        runner.save(family['search_manifest'], source)
        template = next(t for t in old_tasks if t['domain'] == domain)
        for index in range(8):
            task = deepcopy(template)
            seed = 2026103001 + len(tasks)
            task.update(geometry_task_seed=seed, optimizer_seed=seed,
                        source_search_manifest=family['search_manifest'])
            task.pop('initial_policy_file'); task.pop('shared_neighbors_file')
            task['geometry'] = dict(task_id=f'fresh_{domain}_{index}', problem_kwargs=dict(
                task_geometry_shift=[.001*index, -.001*index, 0.], task_geometry_radius_scale=1.))
            tasks.append(task)
    runner.save(plan['source_manifest'], parent)
    plan.update(tasks=tasks, geometry_seed=2026103000,
        geometry_description='24 registered new geometries, eight per family; no outcome selection.',
        statistical_unit='24 new geometries with one paired observation and verification stream each.',
        replicates=[dict(replicate=1, output_dir='fresh/replicate_1', seed_by_task={
            str(t['geometry_task_seed']): dict(observation_seed=t['geometry_task_seed']+100,
                                              verification_seed=t['geometry_task_seed']+200)
            for t in tasks})])
    plan['frozen']['initial_policies'] = 'Unchanged source-only design80 with ten fresh shared raws per new geometry.'
    plan['budget'].update(geometry_count=24, observation_replicates=1)
    return plan, root


def test_materialize_preserves_all_tasks_and_separates_streams(registration):
    plan, root = registration
    frozen = deepcopy(plan)
    prepared = runner.materialize(plan)
    assert plan == frozen
    assert len(prepared['replicates']) == 2
    seeds = []
    for replica in prepared['replicates']:
        mapping = next(r['seed_by_task'] for r in plan['replicates'] if r['replicate'] == replica['replicate'])
        for family in replica['families']:
            search = runner.read(root / family['search_manifest'])
            terminal = runner.read(root / family['terminal_manifest'])
            assert len(search['arms']) == 7 and len(terminal['arms']) == 9
            assert 'execution_failures' not in search and 'execution_failures' not in terminal
            assert terminal['protocols'] == {runner.PROTOCOL: [80, 128, 128]}
            for seed in search['target_seeds']:
                task = next(t for t in plan['tasks'] if t['geometry_task_seed'] == seed)
                assert search['observation_seed_by_task'][str(seed)] == mapping[str(seed)]['observation_seed']
                assert terminal['verification_seed_by_task'][str(seed)] == mapping[str(seed)]['verification_seed']
                assert search['geometry_by_seed'][str(seed)] == task['geometry']
                assert search['fixed_initialization_file_by_task'][str(seed)] == task['initial_policy_file']
                assert search['fixed_neighbors_file_by_task'][str(seed)] == task['shared_neighbors_file']
                seeds.append(seed)
    assert len(seeds) == 24 and seeds.count(2026102120) == 2
    assert len(runner.search_jobs(prepared, True)) == 24
    assert len(runner.search_jobs(prepared, False)) == 144
    assert not list(root.rglob('search.json')) and not list(root.rglob('verification.json'))
    assert runner.materialize(plan) == prepared


def test_fresh24_materialization_removes_old_banks_and_preserves_frozen_contract(fresh_registration):
    plan, root = fresh_registration
    frozen = deepcopy(plan)
    parent = runner.read(plan['source_manifest'])
    prepared = runner.materialize(plan)
    assert plan == frozen and len(prepared['replicates']) == 1
    replica = prepared['replicates'][0]
    retained = ('task', 'anchor', 'frozen_assets', 'neighbor_count', 'neighbor_radius',
                'neighbor_safe_margin_scale', 'neighbor_rng', 'decision_protocol',
                'frontends', 'role_prior', 'point_only', 'after_init_point_only')
    seeds = []
    for family in replica['families']:
        template = next(f for f in parent['families'] if f['domain'] == family['domain'])
        original = runner.read(template['search_manifest'])
        search = runner.read(root / family['search_manifest'])
        terminal = runner.read(root / family['terminal_manifest'])
        assert {k:search[k] for k in retained} == {k:original[k] for k in retained}
        assert search['task']['d'] == 1000 and search['anchor']['search_calls'] == 13
        assert search['decision_protocol']['maximum_verification_calls_per_recommendation'] == 352
        assert 'fixed_initialization_file_by_task' not in search
        assert 'fixed_neighbors_file_by_task' not in search
        assert search['geometry_seed'] == plan['geometry_seed']
        assert search['geometry'] == plan['geometry_description']
        assert search['initialization'] == plan['frozen']['initial_policies']
        assert terminal['statistical_unit'] == plan['statistical_unit']
        assert len(search['target_seeds']) == 8
        assert terminal['protocols'] == {runner.PROTOCOL: [80, 128, 128]}
        assert terminal['arms'] == plan['terminal_arms']
        for seed in search['target_seeds']:
            task = next(t for t in plan['tasks'] if t['geometry_task_seed'] == seed)
            assert search['geometry_by_seed'][str(seed)] == task['geometry']
            assert search['observation_seed_by_task'][str(seed)] == seed+100
            assert terminal['verification_seed_by_task'][str(seed)] == seed+200
            seeds.append(seed)
    assert sorted(seeds) == list(range(2026103001, 2026103025))
    assert len(runner.search_jobs(prepared, True)) == 24
    assert len(runner.search_jobs(prepared, False)) == 144
    assert not list(root.rglob('search.json')) and not list(root.rglob('truth.json'))


@pytest.mark.parametrize('structural_failure', [False, True])
@pytest.mark.parametrize('registration_fixture', ['registration', 'fresh_registration'])
def test_global_barriers_and_modified_only_workers(request, monkeypatch, structural_failure, registration_fixture):
    plan, root = request.getfixturevalue(registration_fixture)
    prepared = runner.materialize(plan)
    events = []
    expected_cells = 215 if structural_failure else 216
    target = (1, plan['tasks'][9]['geometry_task_seed'], runner.STRUCTURAL)

    def children():
        return [(r, f) for r in prepared['replicates'] for f in r['families']]

    def stage(output, name, jobs, workers):
        events.append((name, len(jobs)))
        if name == 'fixed-neighbors':
            assert len(list(root.rglob('search.json'))) == 24
        if name == 'online-search':
            assert len(list(root.rglob('neighbors.json'))) == 24
        if name == 'native-terminals':
            assert len(list(root.rglob('search.json'))) == (167 if structural_failure else 168)
        if name == 'verification':
            assert len(list(root.rglob('input.json'))) == expected_cells
            assert len(list(root.rglob('native_terminal.json'))) == 48
        failed = []
        for job in jobs:
            command = job['command']
            if name in ('initialization', 'online-search'):
                source = runner.read(root / job['search_manifest'])
                directory = root / source['output_dir'] / f"seed_{job['seed']}" / job['arm']
                if structural_failure and (job['replicate'], job['seed'], job['arm']) == target:
                    runner.save(directory / 'failed_search_raw_1.json', [dict(raw=[1, 2]), dict(raw=[3, 4])])
                    (directory / 'failed_search_engine.pkl').write_bytes(b'synthetic-preserved-state')
                    (directory / 'attempts.jsonl').write_text(json.dumps(dict(error='synthetic structural overflow'))+'\n')
                    failed.append(dict(job, returncode=1))
                else:
                    runner.save(directory / 'search.json', {})
            elif name == 'fixed-neighbors':
                source = runner.read(root / command[-1])
                for seed in source['target_seeds']:
                    runner.save(root / source['output_dir'] / f'seed_{seed}/neighbors.json', {})
            else:
                child = runner.read(root / command[command.index('--child-manifest')+1])
                seed = int(command[command.index('--seed')+1]); arm = command[command.index('--arm')+1]
                directory = root / child['output_dir'] / f'seed_{seed}' / arm
                if name == 'native-terminals':
                    runner.save(directory / 'native_terminal.json', {})
                else:
                    assert list(child['protocols']) == [runner.PROTOCOL]
                    runner.save(directory / runner.PROTOCOL / 'verification.json', {})
        return failed

    def freeze(source, terminal):
        assert len(list(root.rglob('native_terminal.json'))) == 48
        for seed, arm, protocol in runner.verifier.cases(terminal):
            runner.save(root / terminal['output_dir'] / f'seed_{seed}' / arm / protocol / 'input.json', {})

    def score(actual_plan, actual_prepared):
        assert actual_prepared == prepared
        assert len(list(root.rglob('verification.json'))) == expected_cells
        events.append(('score', expected_cells))
        return dict(rows=[dict(logical_total_calls=13)] * expected_cells)

    monkeypatch.setattr(runner, 'run_stage', stage)
    monkeypatch.setattr(runner, 'prepare_terminal', freeze)
    monkeypatch.setattr(runner, 'costs', lambda _: dict(cost=dict(new_target_initial_calls=0,
        new_target_search_calls=0, new_verification_calls=0)))
    monkeypatch.setitem(sys.modules, 'performance.summarize_mainline_observation_seed_replication', SimpleNamespace(score=score))
    result = runner.run(plan, 3, root / 'registration.json')
    assert events == [('initialization', 24), ('fixed-neighbors', len(children())), ('online-search', 144),
                      ('native-terminals', 48), ('verification', expected_cells), ('score', expected_cells)]
    assert result['logical_target_calls'] == 13*expected_cells+(12 if structural_failure else 0)
    assert result['status'] == ('complete_with_search_failure' if structural_failure else 'complete')
    if structural_failure:
        assert plan['execution_failures'][0]['logical_search_calls'] == 12
        assert plan['execution_failures'][0]['new_search_calls'] == 2
        assert runner.search_jobs(prepared, False) == []


def test_initial_failure_prevents_search_and_verification(registration, monkeypatch):
    plan, root = registration
    calls = []
    def fail(output, name, jobs, workers):
        calls.append(name)
        return [jobs[0]]
    monkeypatch.setattr(runner, 'run_stage', fail)
    with pytest.raises(RuntimeError, match='Initialization failed'):
        runner.run(plan, 1, root / 'registration.json')
    assert calls == ['initialization']
