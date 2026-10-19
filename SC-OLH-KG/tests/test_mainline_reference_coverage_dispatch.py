"""Changed barriers, subprocess failure retention, and four-cell dispatch only."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import pytest

from performance import mainline_reference_coverage_dispatch as dispatch
from test_mainline_reference_coverage_cohort import (
    setup_cohort, write_online, record_outcomes,
)
from test_mainline_reference_coverage_execution import (
    inputs, substitute_verifiers, Substitute,
)


def fixture(tmp_path, monkeypatch, seeds=range(24)):
    plan, source = setup_cohort(tmp_path, seeds=seeds)
    plan.update(status='registered_not_started', name='fixture')
    manifest = tmp_path / 'plan.json'
    dispatch.save(manifest, plan)
    for module in (dispatch, dispatch.run, dispatch.run.replication):
        monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(dispatch.socket, 'gethostname', lambda: 'node004')
    return plan, source, manifest


def complete_stage(plan, stage):
    dispatch.save(dispatch.stage_path(plan, stage) / 'completion.json', dict(status='complete'))


def descriptor(plan, stage, jobs):
    path = dispatch.stage_path(plan, stage) / 'jobs.json'
    dispatch.save(path, dict(manifest='plan.json', stage=stage, jobs=jobs))
    return path


def record_shard(plan, stage, jobs, failures=()):
    dispatch.save(dispatch.stage_path(plan, stage) / f'shard_0_{len(jobs)}/result.json',
                  dict(labels=[job['label'] for job in jobs], failed=list(failures)))


def freeze(plan, source, tmp_path):
    dispatch.cohort.freeze_initial_references(plan, root=tmp_path)
    write_online(tmp_path, source)
    return dispatch.cohort.freeze_cohort_inputs(plan, root=tmp_path)


def test_completed_manifest_and_local_scientific_worker_rejected_before_work(tmp_path, monkeypatch):
    manifest = tmp_path / 'plan.json'
    dispatch.save(manifest, dict(status='complete'))
    with pytest.raises(ValueError, match='cannot be restarted'):
        dispatch.describe(manifest, 'initial')
    monkeypatch.setattr(dispatch.socket, 'gethostname', lambda: 'local-machine')
    with pytest.raises(RuntimeError, match='node001-node006'):
        dispatch.worker('absent.json', 0, 1, 1)
    with pytest.raises(RuntimeError, match='node001-node006'):
        dispatch.terminal({}, 1, 'arm')


def test_initial_completion_freezes_all_references_before_online_descriptor(tmp_path, monkeypatch):
    plan, source, manifest = fixture(tmp_path, monkeypatch)
    first = dispatch.describe(manifest, 'initial')
    assert first['jobs'] == []  # All 24 saved STOP histories are reused.
    assert dispatch.finish(dispatch.stage_path(plan, 'initial') / 'jobs.json')['status'] == 'complete'
    references = dispatch.cohort.require_initial_references(plan, root=tmp_path)
    assert references['frozen_references'] == 24
    with pytest.raises(ValueError, match='neighbors'):
        dispatch.describe(manifest, 'online')
    complete_stage(plan, 'neighbors')
    online = dispatch.describe(manifest, 'online')
    assert len(online['jobs']) == 144
    assert {job['seed'] for job in online['jobs']} == set(range(24))
    assert all(job['command'][3] == dispatch.run.MODULE for job in online['jobs'])
    assert all(job['command'][4] == 'search' for job in online['jobs'])
    assert not (tmp_path / 'cohort/search_status.json').exists()


def test_saved_descriptor_worker_still_requires_global_reference_gate(tmp_path, monkeypatch):
    plan, source, _ = fixture(tmp_path, monkeypatch, seeds=[0])
    complete_stage(plan, 'neighbors')
    job = dict(label='online', seed=0, command=['python', 'unused'])
    path = descriptor(plan, 'online', [job])
    runner = Mock(side_effect=AssertionError('must not execute'))
    monkeypatch.setattr(dispatch.run.replication, 'run_stage', runner)
    with pytest.raises(ValueError, match='initial_references'):
        dispatch.worker(path, 0, 1, 1)
    assert not runner.called


def test_initial_reference_admission_failure_retains_six_searches_and_36_cells(tmp_path, monkeypatch):
    plan, source, manifest = fixture(tmp_path, monkeypatch, seeds=[0])
    path = tmp_path / 'search/seed_0/initialization_only/search.json'
    search = dispatch.read(path)
    search['terminal'].pop('terminal_recording')
    dispatch.save(path, search)
    dispatch.describe(manifest, 'initial')
    dispatch.finish(dispatch.stage_path(plan, 'initial') / 'jobs.json')
    complete_stage(plan, 'neighbors')
    assert dispatch.describe(manifest, 'online')['jobs'] == []
    retained = dispatch.read(tmp_path / 'source.json')['execution_failures']
    assert len(retained) == 6 and all(row['new_search_calls'] == 0 for row in retained)
    frozen = dispatch.cohort.freeze_cohort_inputs(plan, root=tmp_path)
    assert frozen['registered_terminal_cells'] == 36
    assert all(row['status'] == 'missing' for row in frozen['rows'])
    seal = dispatch.cohort.seal_cohort_results(plan, root=tmp_path)
    assert seal['missing_cells'] == 36 and seal['physical_verification_calls'] == 0


def test_startup_search_failure_has_no_retry_and_retains_all_dependencies(tmp_path, monkeypatch):
    plan, source, _ = fixture(tmp_path, monkeypatch, seeds=[0])
    (tmp_path / 'search/seed_0/initialization_only/search.json').unlink()
    job = dict(label='failed_stop', seed=0, arm=dispatch.run.STOP, replicate=1,
        search_manifest='source.json', terminal_manifest='terminal.json', returncode=1)
    path = descriptor(plan, 'initial', [job])
    record_shard(plan, 'initial', [job], [job])
    result = dispatch.finish(path)
    assert result['failed_jobs'] == [job]
    source = dispatch.read(tmp_path / 'source.json')
    terminal = dispatch.read(tmp_path / 'terminal.json')
    assert len(source['execution_failures']) == 7
    assert len(terminal['execution_failures']) == 9
    assert all(row['new_initial_calls'] is None and row['new_search_calls'] is None
               for row in source['execution_failures'])
    assert dispatch.finish(path) == result


def test_finish_rejects_missing_or_overlapping_shards(tmp_path, monkeypatch):
    plan, _, _ = fixture(tmp_path, monkeypatch, seeds=[0])
    jobs = [dict(label='a'), dict(label='b')]
    path = descriptor(plan, 'neighbors', jobs)
    record_shard(plan, 'neighbors', jobs[:1])
    with pytest.raises(ValueError, match='Unfinished or overlapping'):
        dispatch.finish(path)
    dispatch.save(dispatch.stage_path(plan, 'neighbors') / 'shard_1_2/result.json',
                  dict(labels=['a'], failed=[]))
    with pytest.raises(ValueError, match='Unfinished or overlapping'):
        dispatch.finish(path)


def test_neighbor_failure_blocks_online_without_auto_retry(tmp_path, monkeypatch):
    plan, _, manifest = fixture(tmp_path, monkeypatch, seeds=[0])
    job = dict(label='neighbor_0', returncode=1)
    path = descriptor(plan, 'neighbors', [job])
    record_shard(plan, 'neighbors', [job], [job])
    with pytest.raises(RuntimeError, match='online search remains blocked'):
        dispatch.finish(path)
    with pytest.raises(ValueError, match='neighbors'):
        dispatch.describe(manifest, 'online')


def test_all216_terminal_jobs_have_isolated_subprocess_and864_frozen_cells(tmp_path, monkeypatch):
    plan, source, manifest = fixture(tmp_path, monkeypatch)
    frozen = freeze(plan, source, tmp_path)
    complete_stage(plan, 'native')
    jobs = dispatch.describe(manifest, 'terminal')['jobs']
    assert len(jobs) == 216 and frozen['registered_terminal_cells'] == 864
    assert len({(job['seed'], job['arm']) for job in jobs}) == 216
    assert all(job['command'][3:5] == [dispatch.MODULE, 'terminal'] for job in jobs)
    assert all('--seed' in job['command'] and '--arm' in job['command'] for job in jobs)
    assert not (tmp_path / 'cohort/execution_status.json').exists()


def test_terminal_worker_and_direct_execution_require_global_frozen_inputs(tmp_path, monkeypatch):
    plan, _, _ = fixture(tmp_path, monkeypatch, seeds=[0])
    job = dict(label='terminal', seed=0, arm='initialization_only', command=['python', 'unused'])
    path = descriptor(plan, 'terminal', [job])
    with pytest.raises(ValueError, match='frozen_inputs'):
        dispatch.worker(path, 0, 1, 1)
    with pytest.raises(ValueError, match='frozen_inputs'):
        dispatch.terminal(plan, 0, 'initialization_only')


def test_terminal_calls_registered_factory_and_reuses_completed_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatch, 'ROOT', tmp_path)
    monkeypatch.setattr(dispatch.socket, 'gethostname', lambda: 'node006')
    variants = inputs()
    descriptors = {}
    for name, value in variants.items():
        path = Path('inputs') / name / 'input.json'
        dispatch.save(tmp_path / path, value)
        descriptors[name] = dict(status='frozen', input_file=str(path))
    row = dict(seed=8001, arm='legacy_role_after_init_point_only', status='frozen',
               variants=descriptors, stream='replicate_1', execution_dir='execution')
    monkeypatch.setattr(dispatch.cohort, 'require_frozen_inputs', lambda *a, **k: dict(rows=[row]))
    substitute_verifiers(monkeypatch, (True, True, True))
    problem = Substitute()
    factory = Mock(return_value=problem)
    monkeypatch.setattr(dispatch, 'build_problem', factory)
    result = dispatch.terminal({}, 8001, row['arm'])
    assert result['status'] == 'complete' and factory.called
    assert all(item.args[0] in variants.values() for item in factory.call_args_list)
    assert all(cell['logical_verification_calls'] <= 352 for cell in result['variants'].values())
    paid = len(problem.calls)
    factory.reset_mock()
    assert dispatch.terminal({}, 8001, row['arm']) == result
    assert len(problem.calls) == paid and not factory.called


def test_failed_terminal_subprocess_preserves_paid_prefix_and_seals_all864(tmp_path, monkeypatch):
    plan, source, _ = fixture(tmp_path, monkeypatch)
    frozen = freeze(plan, source, tmp_path)
    record_outcomes(tmp_path, frozen)
    row = frozen['rows'][0]
    directory = tmp_path / row['execution_dir']
    (directory / 'execution.json').unlink()
    lost = 'reference_two_comparison'
    (directory / lost / 'verification.json').unlink()
    dispatch.save(directory / 'raw_blocks/block_0000.json', dict(raw=[dict(paid_by_variant=lost)],
        failed_calls=[dict(variant=lost)]))
    job = dict(label='interrupted', seed=row['seed'], arm=row['arm'], returncode=-15)
    path = descriptor(plan, 'terminal', [job])
    record_shard(plan, 'terminal', [job], [job])
    before = dispatch.cohort.require_frozen_inputs(plan, root=tmp_path)
    with pytest.raises(ValueError, match='execution_status'):
        dispatch.cohort.require_truth_seal(plan, root=tmp_path)
    dispatch.finish(path)
    seal = dispatch.cohort.require_truth_seal(plan, root=tmp_path)
    assert seal['registered_terminal_cells'] == 864 and seal['failed_cells'] == 1
    assert seal['completed_cells'] == 863 and seal['missing_cells'] == 0
    assert seal['physical_verification_calls'] == seal['failed_simulation_calls'] == 1
    failure = dispatch.read(directory / lost / 'failure.json')
    assert failure['logical_verification_calls'] is None
    assert failure['cumulative_new_verification_calls'] == 1
    assert failure['cumulative_failed_simulation_calls'] == 1
    assert dispatch.cohort.require_frozen_inputs(plan, root=tmp_path) == before


def test_native_startup_failure_keeps_missing_cells_without_loading_model(tmp_path, monkeypatch):
    plan, source, _ = fixture(tmp_path, monkeypatch, seeds=[0])
    dispatch.cohort.freeze_initial_references(plan, root=tmp_path)
    write_online(tmp_path, source)
    (tmp_path / 'search/seed_0/online_0/native_terminal.json').unlink()
    job = dict(label='native_0', seed=0, arm='online_0', search_manifest='source.json', returncode=1)
    path = descriptor(plan, 'native', [job])
    record_shard(plan, 'native', [job], [job])
    dispatch.finish(path)
    frozen = dispatch.cohort.require_frozen_inputs(plan, root=tmp_path)
    missing = [row for row in frozen['rows'] if row['status'] == 'missing']
    assert len(missing) == 1 and missing[0]['arm'] == 'online_0_native'
    assert all(value['status'] == 'missing' for value in missing[0]['variants'].values())


def test_completed_worker_shard_does_not_reinvoke_subprocesses(tmp_path, monkeypatch):
    plan, _, _ = fixture(tmp_path, monkeypatch, seeds=[0])
    job = dict(label='neighbor', command=['python', 'unused'])
    path = descriptor(plan, 'neighbors', [job])
    runner = Mock(return_value=[])
    monkeypatch.setattr(dispatch.run.replication, 'run_stage', runner)
    assert dispatch.worker(path, 0, 1, 12) == []
    assert runner.call_args.args[-1] == 1
    assert dispatch.worker(path, 0, 1, 12) == []
    assert runner.call_count == 1
