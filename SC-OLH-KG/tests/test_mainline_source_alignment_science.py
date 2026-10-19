"""Nine-history / eleven-terminal substitutes, including the full 3168-cell gate."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from performance import run_mainline_source_alignment_science as science
from test_mainline_reference_coverage_cohort import write_search
from test_mainline_reference_coverage_execution import Substitute, substitute_verifiers


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    for module in (science, science.run, science.run.replication, science.run.verifier,
                   science.previous_dispatch):
        monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(science.socket, 'gethostname', lambda: 'node005')
    arms = ['legacy_native_neighbors', 'same_source_gp', 'current_structural_mixture',
        'legacy_role_full', 'legacy_role_after_init_point_only', 'legacy_role_point_only', science.run.STOP,
        'source_permuted_role_after_init_point_only', 'source_permuted_role_point_only']
    cells = [dict(name=a, history_arm=a, terminal_backend='original') for a in arms]
    cells += [dict(name=a + '_native_terminal', history_arm=a, terminal_backend='native') for a in science.run.ROLES]
    task = dict(domain='fixture', geometry_task_seed=17, optimizer_seed=17,
        geometry=dict(problem_kwargs=dict(task_geometry_shift=[0., .01, -.01], task_geometry_radius_scale=1.)))
    frozen = dict(d=1000, search_calls=13, candidate_budgets=[80, 128, 128])
    plan = dict(name='fixture', status='registered_not_started', execution_started=False,
        target_execution_started=False, target_execution_authorized=False,
        source_control_manifest='source_control.json', source_asset_seal_file='source_seal.json',
        output_dir='cohort', selected_N=1, tasks=[task], search_arms=arms, terminal_arms=cells,
        terminal_protocols=['old'], old_protocol=science.run.PROTOCOL, frozen=frozen,
        budget=dict(maximum_verification_calls_per_cell=352, registered_terminal_cells=22),
        execution_preparation=dict(status='pending_old_only_two_stream_adapter'),
        target_execution_preparation=dict(status='pending'),
        streams=[dict(name=f'observation_{i}', manifest=f'child_{i}.json') for i in (1, 2)])
    originals = dict(source_models_file='original/bundle.pkl', native_source_prior_file='original/native.pkl')
    control_families, sealed_families = [], []
    for index, domain in enumerate(('fixture', 'other1', 'other2')):
        assets = dict(source_models_file=f'permuted/{domain}/bundle.pkl',
                      native_source_prior_file=f'permuted/{domain}/native.pkl')
        row = dict(domain=domain, frozen_assets=deepcopy(originals),
            source_permuted_assets=assets, source_archive_file=f'original/{domain}/archive.json',
            source_permutation_seed=71 + index, source_search_manifest=f'original/{domain}/search.json')
        models = []
        for kind, key in [('structural', 'source_models_file'), ('native', 'native_source_prior_file')]:
            artifact = tmp_path / assets[key]
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text('synthetic sealed source; never load')
            stat = artifact.stat()
            models.append(dict(kind=kind, model_file=assets[key], receipt_status='complete',
                               size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns))
        control_families.append(row)
        sealed_families.append(dict(family=domain, source_archive_file=row['source_archive_file'],
                                   permutation_seed=row['source_permutation_seed'], models=models))
    science.save(tmp_path / 'source_control.json', dict(search_arms=arms, terminal_arms=cells,
        families=control_families, source_fitting_authorized=True, target_execution_authorized=False))
    plan['source_permuted_assets_by_family'] = {
        row['domain']: deepcopy(row['source_permuted_assets']) for row in control_families}
    science.save(tmp_path / 'source_seal.json', dict(status='sealed_sources_only',
        source_models_ready=True, completed_fit_pipelines=6, source_permutations_generated=3,
        source_fitting_recipes_unchanged=True, failures=[], unknown_attempt_costs=[], families=sealed_families,
        target_execution_allowed=False))
    sources = []
    for i, descriptor in enumerate(plan['streams'], 1):
        output = f'cohort/stream_{i}'
        source = dict(domain='fixture', output_dir=f'{output}/search', target_seeds=[17],
            frozen_assets=originals, source_permuted_assets=control_families[0]['source_permuted_assets'],
            task=dict(d=1000, L=1, sigma=.1, alpha=.05, weights=[1.]),
            anchor=dict(n0=10, search_calls=13, initial_design_seed=80),
            neighbor_count=64, neighbor_radius=.08, neighbor_safe_margin_scale=1., neighbor_rng='frozen_fixture',
            arms=[dict(name=a, online=a != science.run.STOP) for a in arms],
            geometry_by_seed={'17': task['geometry']}, observation_seed_by_task={'17': 100 + i},
            decision_protocol=dict(verification_method='normal_quantile_tolerance'))
        terminal = dict(output_dir=f'{output}/terminal', target_seeds=[17], arms=cells,
            protocols={science.run.PROTOCOL: [80, 128, 128]},
            verification_seed_by_task={'17': 200 + i}, inherited_replay_check_file='old_replay.json')
        child = dict(output_dir=output, tasks=deepcopy(plan['tasks']), search_arms=arms,
            terminal_arms=cells, terminal_protocols=['old'], frozen=frozen,
            status='registered_not_started', execution_started=False,
            target_execution_started=False, target_execution_authorized=False,
            target_execution_preparation=deepcopy(plan['target_execution_preparation']),
            execution_preparation=deepcopy(plan['execution_preparation']),
            replicates=[dict(replicate=i, seed_by_task={'17': dict(observation_seed=100 + i, verification_seed=200 + i)})])
        family = dict(domain='fixture', search_manifest=f'source_{i}.json', terminal_manifest=f'terminal_{i}.json')
        science.save(tmp_path / family['search_manifest'], source)
        science.save(tmp_path / 'original/fixture/search.json', source)
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
    assert len(online['jobs']) == 16 and len({j['label'] for j in online['jobs']}) == 16
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
    assert frozen['registered_terminal_cells'] == frozen['frozen_cells'] == 22
    assert all('new_input' not in row for row in frozen['rows'])
    initial_rows = [science.read(root / source['output_dir'] / 'seed_17' / science.run.STOP / 'search.json')['history'] for source in sources]
    assert [r['point'] for r in initial_rows[0]] == [r['point'] for r in initial_rows[1]]
    assert initial_rows[0][0]['raw'] != initial_rows[1][0]['raw']
    banks = [science.read(root / source['output_dir'] / 'seed_17/neighbors.json') for source in sources]
    assert banks[0]['source'] != banks[1]['source'] and banks[0]['own_stop_raw'] != banks[1]['own_stop_raw']
    assert science.read(root / 'plan.json')['execution_started']
    assert science.read(root / 'plan.json')['target_execution_started']
    assert science.read(root / 'source_control.json')['target_execution_authorized'] is False
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
    assert len(descriptor['jobs']) == 22
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
    assert seal['completed_old'] == seal['registered_terminal_cells'] == 22
    assert seal['costs']['cost']['new_verification_calls'] == 22 * 224
    assert len(built) == 22 and all(len(problem.calls) == 224 for problem in built)
    assert all('new' not in row for row in seal['cases'])
    for _, child in science.children(plan):
        assert not science.read(root / child['output_dir'] / 'execution_status.json')['truth_access_allowed']
    science.terminal(plan, last['stream'], last['seed'], last['arm'])
    assert len(built) == 22  # Completed science is reused without a new problem or fee.


def test_one_bank_failure_keeps_stop_and_other_stream_but_retains_all_twenty_two_cells(cohort):
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
    assert len(online['jobs']) == 8 and {job['stream'] for job in online['jobs']} == {'observation_2'}
    for arm in plan['search_arms']:
        if arm != science.run.STOP:
            record_search(root, sources[1], arm)
    finish_stage(plan, 'online', online)
    native = science.describe('plan.json', 'native')
    finish_stage(plan, 'native', native)
    frozen = science.require_frozen_inputs(plan)
    assert frozen['missing_cells'] == 10 and frozen['frozen_cells'] == 12
    first = [r for r in frozen['rows'] if r['stream'] == 'observation_1']
    assert next(r for r in first if r['arm'] == science.run.STOP)['status'] == 'frozen'
    assert all(r['failure']['status'] == 'neighbor_dependency_failed' for r in first if r['status'] == 'missing')
    assert len(science.describe('plan.json', 'terminal')['jobs']) == 12


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
    assert seal['completed_old'] == 21 and seal['failed_old'] == 1
    failure = science.read(root / seal['cases'][0]['old']['failure_file'])
    assert failure['new_verification_calls'] is None and not failure['call_accounting_complete']
    runner = Mock(side_effect=AssertionError('resolved cells must not execute'))
    monkeypatch.setattr(science.run.replication, 'run_stage', runner)
    result = science.worker(science.stage_path(plan, 'terminal') / 'jobs.json', 0, 1, 1)
    assert result['executed_jobs'] == 0 and result['reused_jobs'] == 1 and not runner.called
    assert science.terminal(plan, failed_job['stream'], failed_job['seed'], failed_job['arm']) == failure


def test_source_only_admission_and_generic_validated_flag_cannot_start_target_work(cohort):
    plan, _, root = cohort
    plan['target_execution_authorized'] = False
    plan['target_execution_preparation']['status'] = 'pending'
    plan['execution_preparation']['status'] = 'validated'
    plan['source_fitting_authorized'] = True
    science.save(root / 'plan.json', plan)
    with pytest.raises(ValueError, match='target execution has not been explicitly admitted'):
        science.describe('plan.json', 'initial')
    assert not (root / plan['output_dir'] / 'cluster/initial/jobs.json').exists()


def test_missing_or_changed_sealed_source_asset_blocks_target_admission(cohort):
    plan, _, root = cohort
    seal = science.read(root / plan['source_asset_seal_file'])
    asset = root / seal['families'][0]['models'][0]['model_file']
    asset.write_text('changed synthetic source')
    with pytest.raises(AssertionError):
        science.admit('plan.json', 'checks.json')
    asset.unlink()
    with pytest.raises(FileNotFoundError):
        science.admit('plan.json', 'checks.json')
    assert not (root / plan['output_dir'] / 'cluster/initial/jobs.json').exists()


def test_native_cross_jobs_are_only_original_full_and_after(cohort):
    plan, sources, root = cohort
    through_inputs(cohort)
    path = science.stage_path(plan, 'native') / 'jobs.json'
    path.unlink()
    for source in sources:
        for arm in science.run.ROLES:
            (root / source['output_dir'] / 'seed_17' / arm / 'native_terminal.json').unlink()
    descriptor = science.describe('plan.json', 'native')
    assert len(descriptor['jobs']) == 4
    assert {job['arm'] for job in descriptor['jobs']} == {
        'legacy_role_full', 'legacy_role_after_init_point_only'}


def test_interrupted_worker_is_retained_without_automatic_retry(cohort, monkeypatch):
    plan, _, root = cohort
    descriptor = science.describe('plan.json', 'initial')
    output = science.stage_path(plan, 'initial') / 'shard_0_1'
    science.save(output / 'started.json', dict(assigned_labels=[descriptor['jobs'][0]['label']]))
    runner = Mock(side_effect=AssertionError('must retain interrupted work'))
    monkeypatch.setattr(science.run.replication, 'run_stage', runner)
    with pytest.raises(RuntimeError, match='do not retry it automatically'):
        science.worker(science.stage_path(plan, 'initial') / 'jobs.json', 0, 1, 1)
    assert not runner.called and not (output / 'result.json').exists()


def test_every_one_of_3168_distinct_registered_cells_is_required_before_truth(tmp_path, monkeypatch):
    monkeypatch.setattr(science, 'ROOT', tmp_path)
    names = [f'control_{index}' for index in range(11)]
    plan = dict(output_dir='full_fixture', streams=[dict(name='stream1'), dict(name='stream2')],
        tasks=[dict(geometry_task_seed=seed) for seed in range(144)],
        terminal_arms=[dict(name=name) for name in names])
    cases = [dict(stream=stream['name'], seed=task['geometry_task_seed'], arm=arm,
                  old=dict(status='complete'))
        for stream in plan['streams'] for task in plan['tasks'] for arm in names]
    path = tmp_path / plan['output_dir'] / 'execution_status.json'
    seal = dict(status='sealed', truth_access_allowed=True, registered_streams=2,
                registered_terminal_cells=3168, cases=cases)
    science.save(path, dict(seal, cases=cases[:-1]))
    with pytest.raises(AssertionError):
        science.require_seal(plan)
    science.save(path, dict(seal, cases=cases[:-1] + [deepcopy(cases[0])]))
    with pytest.raises(AssertionError):
        science.require_seal(plan)
    cases[-1]['old']['status'] = 'pending'
    science.save(path, seal)
    with pytest.raises(AssertionError):
        science.require_seal(plan)
    cases[-1]['old']['status'] = 'failed'
    science.save(path, seal)
    assert len(science.require_seal(plan)['cases']) == 3168
