"""Small two-stream substitutes for old-only execution and global truth barriers."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from performance import run_mainline_precision_confirmation_science as science
from test_mainline_reference_coverage_cohort import write_search
from test_mainline_reference_coverage_execution import Substitute, substitute_verifiers


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    for module in (science, science.run, science.run.replication, science.run.verifier,
                   science.previous_dispatch):
        monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(science.socket, 'gethostname', lambda: 'node005')
    arms = ['legacy_native_neighbors', 'same_source_gp', 'current_structural_mixture',
        'legacy_role_full', 'legacy_role_after_init_point_only', 'legacy_role_point_only', science.run.STOP]
    cells = [dict(name=a, history_arm=a, terminal_backend='original') for a in arms]
    cells += [dict(name=a + '_native_terminal', history_arm=a, terminal_backend='native') for a in science.run.ROLES]
    task = dict(domain='fixture', geometry_task_seed=17, optimizer_seed=17,
        geometry=dict(problem_kwargs=dict(task_geometry_shift=[0., .01, -.01], task_geometry_radius_scale=1.)))
    frozen = dict(d=1000, search_calls=13, candidate_budgets=[80, 128, 128])
    plan = dict(name='fixture', status='registered_not_started', execution_started=False,
        output_dir='cohort', selected_N=1, tasks=[task], search_arms=arms, terminal_arms=cells,
        terminal_protocols=['old'], old_protocol=science.run.PROTOCOL, frozen=frozen,
        budget=dict(maximum_verification_calls_per_cell=352, registered_terminal_cells=18),
        execution_preparation=dict(status='pending_old_only_two_stream_adapter'),
        streams=[dict(name=f'observation_{i}', manifest=f'child_{i}.json') for i in (1, 2)])
    sources = []
    for i, descriptor in enumerate(plan['streams'], 1):
        output = f'cohort/stream_{i}'
        source = dict(domain='fixture', output_dir=f'{output}/search', target_seeds=[17],
            task=dict(d=1000, L=1, sigma=.1, alpha=.05, weights=[1.]),
            arms=[dict(name=a, online=a != science.run.STOP) for a in arms],
            geometry_by_seed={'17': task['geometry']}, observation_seed_by_task={'17': 100 + i},
            decision_protocol=dict(verification_method='normal_quantile_tolerance'))
        terminal = dict(output_dir=f'{output}/terminal', target_seeds=[17], arms=cells,
            protocols={science.run.PROTOCOL: [80, 128, 128]},
            verification_seed_by_task={'17': 200 + i}, inherited_replay_check_file='old_replay.json')
        child = dict(output_dir=output, tasks=deepcopy(plan['tasks']), search_arms=arms,
            terminal_arms=cells, terminal_protocols=['old'], frozen=frozen,
            status='registered_not_started', execution_started=False,
            execution_preparation=deepcopy(plan['execution_preparation']),
            replicates=[dict(replicate=i, seed_by_task={'17': dict(observation_seed=100 + i, verification_seed=200 + i)})])
        family = dict(domain='fixture', search_manifest=f'source_{i}.json', terminal_manifest=f'terminal_{i}.json')
        science.save(tmp_path / family['search_manifest'], source)
        science.save(tmp_path / family['terminal_manifest'], terminal)
        science.save(tmp_path / descriptor['manifest'], child)
        science.save(tmp_path / output / 'prepared.json', dict(replicates=[dict(replicate=i, families=[family])]))
        sources.append(source)
    science.save(tmp_path / 'old_replay.json', dict(status='passed'))
    science.save(tmp_path / 'checks.json', dict(status='passed'))
    science.save(tmp_path / 'plan.json', plan)
    science.admit('plan.json', 'checks.json')
    return science.read(tmp_path / 'plan.json'), sources, tmp_path


def record_search(root, source, arm):
    write_search(root, source, 17, arm)
    path = root / source['output_dir'] / 'seed_17' / arm / 'search.json'
    saved = science.read(path)
    saved['terminal']['objective_incumbent_position'] = 1  # Exercise the real distinct-challenger transform.
    for row in saved['history']:
        row['raw'] = [row['raw'][0] + source['observation_seed_by_task']['17']]
    science.save(path, saved)
    if arm in science.run.ROLES:
        science.save(path.parent / 'native_terminal.json', dict(terminal=deepcopy(saved['terminal'])))


def finish_stage(plan, stage, descriptor, failures=()):
    if descriptor['jobs']:
        science.save(science.stage_path(plan, stage) / f"shard_0_{len(descriptor['jobs'])}" / 'result.json',
            dict(labels=[job['label'] for job in descriptor['jobs']], failed=list(failures)))
    return science.finish(science.stage_path(plan, stage) / 'jobs.json')


def through_inputs(cohort):
    plan, sources, root = cohort
    initial = science.describe('plan.json', 'initial')
    assert len(initial['jobs']) == 2
    for source in sources:
        record_search(root, source, science.run.STOP)
    finish_stage(plan, 'initial', initial)
    neighbors = science.describe('plan.json', 'neighbors')
    assert {job['search_manifest'] for job in neighbors['jobs']} == {'source_1.json', 'source_2.json'}
    for job in neighbors['jobs']:
        source = science.read(root / job['search_manifest'])
        case = root / source['output_dir'] / 'seed_17'
        science.save(case / 'neighbors.json', dict(source=job['search_manifest'],
            own_stop_raw=science.read(case / science.run.STOP / 'search.json')['history'][0]['raw']))
        (case / 'neighbors.npz').touch()
    finish_stage(plan, 'neighbors', neighbors)
    online = science.describe('plan.json', 'online')
    assert len(online['jobs']) == 12 and len({j['label'] for j in online['jobs']}) == 12
    for source in sources:
        for arm in plan['search_arms']:
            if arm != science.run.STOP:
                record_search(root, source, arm)
    finish_stage(plan, 'online', online)
    native = science.describe('plan.json', 'native')
    assert native['jobs'] == []  # Reuse the fixture's already frozen native outputs.
    finish_stage(plan, 'native', native)
    return science.require_frozen_inputs(plan)


def test_two_stream_old_only_e2e_uses_real_input_transform_verifier_and_global_last_cell_gate(cohort, monkeypatch):
    plan, sources, root = cohort
    frozen = through_inputs(cohort)
    assert frozen['registered_terminal_cells'] == frozen['frozen_cells'] == 18
    assert all('new_input' not in row for row in frozen['rows'])
    initial_rows = [science.read(root / source['output_dir'] / 'seed_17' / science.run.STOP / 'search.json')['history'] for source in sources]
    assert [r['point'] for r in initial_rows[0]] == [r['point'] for r in initial_rows[1]]
    assert initial_rows[0][0]['raw'] != initial_rows[1][0]['raw']
    banks = [science.read(root / source['output_dir'] / 'seed_17/neighbors.json') for source in sources]
    assert banks[0]['source'] != banks[1]['source'] and banks[0]['own_stop_raw'] != banks[1]['own_stop_raw']
    assert science.read(root / 'plan.json')['execution_started']
    substitute_verifiers(monkeypatch, (True, True, True))
    built = []
    def factory(domain, d, L, sigma, alpha, weights, *, problem_kwargs):
        assert problem_kwargs == plan['tasks'][0]['geometry']['problem_kwargs']
        problem = Substitute()
        problem.d, problem.tau, problem.alpha = 1000, 0., alpha
        problem.base = SimpleNamespace(**problem_kwargs)
        built.append(problem)
        return problem
    monkeypatch.setattr(science.run.verifier.benchmark, 'build_scalarized_problem', factory)
    descriptor = science.describe('plan.json', 'terminal')
    assert len(descriptor['jobs']) == 18
    for row in frozen['rows']:
        item = science.read(root / row['old_input'])
        assert item['protocol'] == science.run.PROTOCOL and item['candidate_budgets'] == [80, 128, 128]
        assert [r['point'] for r in item['frozen_shortlist']] == [[1], [0], [2]]
        assert item['objective_incumbent_position'] == 2 and item['objective_comparison_budget'] == 8
        assert item['verification_seed'] == (201 if row['stream'] == 'observation_1' else 202)
    for row in frozen['rows'][:-1]:
        result = science.terminal(plan, row['stream'], row['seed'], row['arm'])
        assert result['new_verification_calls'] == 224 and result['deployed'] == [1]
    with pytest.raises(AssertionError, match='Unfinished old terminal'):
        science.seal_results(plan)
    with pytest.raises(RuntimeError, match='before truth'):
        science.require_seal(plan)
    last = frozen['rows'][-1]
    science.terminal(plan, last['stream'], last['seed'], last['arm'])
    finish_stage(plan, 'terminal', descriptor)
    seal = science.require_seal(plan)
    assert seal['completed_old'] == seal['registered_terminal_cells'] == 18
    assert seal['costs']['cost']['new_verification_calls'] == 18 * 224
    assert len(built) == 18 and all(len(problem.calls) == 224 for problem in built)
    assert all('new' not in row for row in seal['cases'])
    for _, child in science.children(plan):
        assert not science.read(root / child['output_dir'] / 'execution_status.json')['truth_access_allowed']
    science.terminal(plan, last['stream'], last['seed'], last['arm'])
    assert len(built) == 18  # Completed science is reused without a new problem or fee.


def test_one_bank_failure_keeps_stop_and_other_stream_but_retains_all_eighteen_cells(cohort):
    plan, sources, root = cohort
    descriptor = science.describe('plan.json', 'initial')
    for source in sources:
        record_search(root, source, science.run.STOP)
    finish_stage(plan, 'initial', descriptor)
    descriptor = science.describe('plan.json', 'neighbors')
    failed = dict(descriptor['jobs'][0], returncode=7)
    second = root / sources[1]['output_dir'] / 'seed_17'
    science.save(second / 'neighbors.json', dict(own_stream=2))
    (second / 'neighbors.npz').touch()
    finish_stage(plan, 'neighbors', descriptor, [failed])
    online = science.describe('plan.json', 'online')
    assert len(online['jobs']) == 6 and {job['stream'] for job in online['jobs']} == {'observation_2'}
    for arm in plan['search_arms']:
        if arm != science.run.STOP:
            record_search(root, sources[1], arm)
    finish_stage(plan, 'online', online)
    native = science.describe('plan.json', 'native')
    finish_stage(plan, 'native', native)
    frozen = science.require_frozen_inputs(plan)
    assert frozen['missing_cells'] == 8 and frozen['frozen_cells'] == 10
    first = [r for r in frozen['rows'] if r['stream'] == 'observation_1']
    assert next(r for r in first if r['arm'] == science.run.STOP)['status'] == 'frozen'
    assert all(r['failure']['status'] == 'neighbor_dependency_failed' for r in first if r['status'] == 'missing')
    assert len(science.describe('plan.json', 'terminal')['jobs']) == 10


def test_worker_rechecks_cross_stream_barrier_and_missing_shard_is_not_scientific_failure(cohort, monkeypatch):
    plan, _, root = cohort
    initial = science.describe('plan.json', 'initial')
    with pytest.raises(AssertionError, match='Unfinished or overlapping'):
        science.finish(science.stage_path(plan, 'initial') / 'jobs.json')
    path = science.stage_path(plan, 'online') / 'jobs.json'
    science.save(path, dict(manifest='plan.json', stage='online', jobs=initial['jobs']))
    runner = Mock(side_effect=AssertionError('premature science'))
    monkeypatch.setattr(science.run.replication, 'run_stage', runner)
    with pytest.raises(ValueError, match='neighbors'):
        science.worker(path, 0, 1, 1)
    assert not runner.called
    monkeypatch.setattr(science.socket, 'gethostname', lambda: 'local')
    with pytest.raises(RuntimeError, match='node001-node006'):
        science.worker(path, 0, 1, 1)


def test_initial_policy_identity_compared_without_requiring_same_observations(cohort):
    plan, sources, root = cohort
    initial = science.describe('plan.json', 'initial')
    for source in sources:
        record_search(root, source, science.run.STOP)
    path = root / sources[1]['output_dir'] / 'seed_17' / science.run.STOP / 'search.json'
    saved = science.read(path)
    saved['history'][0]['point'] = [999]
    science.save(path, saved)
    with pytest.raises(AssertionError, match='physical policies differ'):
        finish_stage(plan, 'initial', initial)
    assert not (science.stage_path(plan, 'initial') / 'completion.json').exists()


def test_failed_terminal_without_ledger_stays_unknown_and_completed_cells_are_not_reexecuted(cohort, monkeypatch):
    plan, _, root = cohort
    frozen = through_inputs(cohort)
    descriptor = science.describe('plan.json', 'terminal')
    failed_job = dict(descriptor['jobs'][0], returncode=9)
    for row in frozen['rows'][1:]:
        science.save((root / row['old_input']).parent / 'verification.json',
            dict(new_verification_calls=224, verification=dict(max_verification_budget=352, verification_budget=224)))
    finish_stage(plan, 'terminal', descriptor, [failed_job])
    seal = science.require_seal(plan)
    assert seal['completed_old'] == 17 and seal['failed_old'] == 1
    failure = science.read(root / seal['cases'][0]['old']['failure_file'])
    assert failure['new_verification_calls'] is None and not failure['call_accounting_complete']
    runner = Mock(side_effect=AssertionError('resolved cells must not execute'))
    monkeypatch.setattr(science.run.replication, 'run_stage', runner)
    result = science.worker(science.stage_path(plan, 'terminal') / 'jobs.json', 0, 1, 1)
    assert result['executed_jobs'] == 0 and result['reused_jobs'] == 1 and not runner.called
    assert science.terminal(plan, failed_job['stream'], failed_job['seed'], failed_job['arm']) == failure
