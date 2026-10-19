"""Driver/collector substitutes: saved JSON and fake workers, no source probes."""
import json
from types import SimpleNamespace

import pytest

from performance import run_mainline_source_finite_probes as run


def plan():
    value = json.loads((run.ROOT / run.MANIFEST).read_text())
    value['execution_authorized'] = True
    value['output_dir'] = 'synthetic_finite_batch'
    return value


def allowed(monkeypatch):
    monkeypatch.setattr(run.core, 'require_execution', lambda p: None)
    monkeypatch.setattr(run.core, '_dependencies', lambda: pytest.fail('Scientific dependencies must not load'))


def put(root, p, case, filename, value):
    path = root / p['output_dir'] / 'cases' / case['case_id'] / filename
    if filename.endswith('.jsonl'):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(row)+'\n' for row in value))
    else:
        run.core.save(path, value)
    return path


def test_fresh_process_pool_contract_failure_seals_and_shard_reuses(tmp_path, monkeypatch):
    allowed(monkeypatch)
    p = plan(); calls, pool_options, contexts = [], [], []

    def fake_case(plan, case_id, root):
        calls.append(case_id)
        return dict(status='resource_stop' if len(calls) == 1 else 'complete')

    class Pool:
        def __init__(self, **kwargs):
            pool_options.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def imap_unordered(self, function, arguments, chunksize):
            assert chunksize == 1
            return iter([function(item) for item in reversed(arguments)])

    def context(name):
        contexts.append(name)
        return SimpleNamespace(Pool=Pool)

    monkeypatch.setattr(run.core, 'run_case', fake_case)
    monkeypatch.setattr(run.multiprocessing, 'get_context', context)
    result = run.batch(p, 'manifest.json', 0, 3, 2, root=tmp_path)
    assert result['status'] == 'finished' and len(calls) == 3
    assert len(result['noncomplete_cases']) == 1
    assert contexts == ['spawn'] and pool_options == [dict(processes=2, maxtasksperchild=1)]
    assert [row['case_id'] for row in result['outcomes']] == [row['case_id'] for row in p['cases'][:3]]
    assert result['fresh_process_per_case'] and result['automatic_retries'] == 0
    assert run.batch(p, 'manifest.json', 0, 3, 2, root=tmp_path) == result
    assert len(calls) == 3


def test_started_only_worker_never_calls_core_and_receipt_reuses(tmp_path, monkeypatch):
    allowed(monkeypatch)
    p = plan(); case = p['cases'][0]
    put(tmp_path, p, case, 'started.json', {'started_utc': 'saved'})
    monkeypatch.setattr(run.core, 'run_case', lambda *_args, **_kwargs: pytest.fail('No retry'))
    result = run.case_worker((p, case['case_id'], str(tmp_path)))
    assert result['status'] == 'retained_missing' and result['core_invoked'] is False
    assert run.case_worker((p, case['case_id'], str(tmp_path))) == result
    assert not (tmp_path / p['output_dir'] / 'cases' / case['case_id'] / 'result.json').exists()


def test_worker_exception_stays_outside_scientific_result(tmp_path, monkeypatch):
    allowed(monkeypatch)
    p = plan(); case = p['cases'][0]

    def error(*args, **kwargs):
        raise OSError('synthetic worker exception')

    monkeypatch.setattr(run.core, 'run_case', error)
    row = run.case_worker((p, case['case_id'], str(tmp_path)))
    assert row['status'] == 'worker_error'
    assert row['process_cpu_seconds'] >= 0
    assert not (tmp_path / p['output_dir'] / 'cases' / case['case_id'] / 'result.json').exists()


def fixture(root):
    p = plan(); dispatch = []
    statuses = ['kernel_exception', 'resource_stop', 'input_or_projection_error', 'unknown'] + ['complete']*8
    for index, (case, status) in enumerate(zip(p['cases'], statuses)):
        dispatch.append(dict(case_id=case['case_id'], status='retained_missing' if status == 'unknown' else status))
        worker = dict(status=dispatch[-1]['status'], process_cpu_seconds=.5 if status == 'unknown' else 12.,
            core_result_reused=False, core_invoked=status != 'unknown')
        put(root, p, case, 'worker.json', worker)
        if status == 'unknown':
            put(root, p, case, 'started.json', {'status': 'started'})
            continue
        entered = status != 'input_or_projection_error'
        returned = status == 'complete'
        cost = {key: 0 for key in run.COUNTERS}
        cost.update(local_raw_moment_attempts=int(entered), completed_calls=int(returned),
            failed_calls=int(status == 'kernel_exception'), resource_stopped_calls=int(status == 'resource_stop'),
            unreturned_calls=int(entered and not returned), quad_vec_attempts=int(status != 'resource_stop' and entered),
            process_cpu_seconds=9., wall_seconds=15., recording_errors=[])
        put(root, p, case, 'result.json', dict(case=case, status=status, cost=cost, attempt_count=1,
            automatic_retries=0, not_reached_kernel_calls=512-int(entered),
            zero_work={key: 0 for key in p['future_zero_work']}))
        put(root, p, case, 'attempts.jsonl', [dict(process_cpu_seconds=10., wall_seconds=16., status=status)])
        context = dict(asset_group=case['asset_group'], association=case['association'],
            source_family=case['source_family'], identity='spectral/head_specific/observed_task',
            head='g', component_start=64, component_stop=128, prefix=1, policy_index=1, repeat_index=0)
        if status == 'kernel_exception':
            diagnostic = dict(values=dict(error=2e-9, relative_error=2e-8, minimum_normalizer=.1,
                integral=[[12345]*6], per_node_Z=[12345], shifts=[99999], argmin_Z_node=65,
                **{'info.status': 0, 'info.success': True, 'info.message': 'saved', 'info.neval': 60}),
                missing_fields={})
        elif status == 'resource_stop':
            diagnostic = dict(values=dict(error=None, relative_error=None, minimum_normalizer=None),
                              missing_fields=dict(error='not_reached_in_kernel_frame'))
        else:
            diagnostic = dict(values=dict(integration_absolute_error=1e-10, integration_relative_error=1e-9,
                minimum_normalizer=.1, integration_status=2, bounded_error_accepted=True,
                roundoff_accepted=True, integrand_evaluations=40),
                missing_fields=dict(shifts='not_exposed_by_return'))
        call = dict(call_id=1, status='returned' if returned else status, entered_kernel=entered,
            returned=returned, context=context, diagnostics=diagnostic,
            error='Batched raw moment error exceeds frozen tolerance: 2e-8' if status == 'kernel_exception' else None)
        put(root, p, case, 'call_diagnostics.jsonl', [call] if entered else [])
    run.core.save(root / p['output_dir'] / 'dispatch/shard_0_12/result.json',
                  dict(status='finished', outcomes=dispatch, coordinator_cpu_seconds=3.))
    return p


def test_collect_preserves_missing_scalars_cost_layers_and_accepted_status2(tmp_path, monkeypatch):
    allowed(monkeypatch)
    monkeypatch.setattr(run.core, 'run_case', lambda *_a, **_k: pytest.fail('Collect is read-only'))
    p = fixture(tmp_path)
    completion = run.collect(p, 'manifest.json', root=tmp_path)
    output = tmp_path / p['output_dir']
    summary = run.read(output / 'case_status_and_costs.json')
    conclusion = run.read(output / 'conclusion.json')
    assert completion['status'] == 'complete' and conclusion['registered_cases'] == 12
    assert conclusion['status_counts'] == dict(numerical_rejection=1, resource_stop=1,
        input_or_projection_error=1, unknown=1, complete=8)
    assert conclusion['record_completeness'] == 'partial'
    assert conclusion['bounded_accepted_returned_calls']['known_subtotal'] == 8
    assert conclusion['bounded_accepted_returned_calls']['total'] is None
    worst = conclusion['maximum_recorded_relative_error']
    assert worst['recorded']['integration_relative_error'] == 2e-8
    assert worst['recorded']['integration_absolute_error'] == 2e-9
    assert worst['recorded']['argmin_Z_node'] == 65 and worst['context']['head'] == 'g'
    assert '12345' not in json.dumps(conclusion) and '99999' not in json.dumps(conclusion)
    assert conclusion['stop_calls'][1]['recorded']['minimum_normalizer'] is None
    assert summary['cases'][3]['original_cost'] is None
    assert summary['cases'][3]['not_reached_kernel_calls'] is None
    assert summary['accounting']['case_attempt_cpu_seconds']['total'] is None
    assert summary['accounting']['case_attempt_cpu_seconds']['known_subtotal'] == 110.
    assert summary['accounting']['process_cpu_seconds']['known_subtotal'] == 99.
    assert summary['accounting']['worker_process_cpu_seconds_inclusive']['total'] == 132.5
    assert summary['accounting']['worker_driver_cpu_seconds_excluding_case_attempt']['total'] == 22.5
    assert summary['accounting']['batch_driver_cpu_seconds'] == 3.
    assert conclusion['zero_work']['target_simulation']['total'] is None
    assert conclusion['historical']['unchanged_historical_coverage']['retained_failures'] == 67
    assert len(conclusion['historical']['retained_terminal_arms']) == 11
    assert conclusion['numerical_accuracy'] == 'not_assessed' and conclusion['target_d1000_decision_gain'] == 'not_tested'
    assert run.collect(p, 'manifest.json', root=tmp_path) == completion


def test_collect_barrier_requires_all_dispatch_and_never_fills_missing_result(tmp_path, monkeypatch):
    allowed(monkeypatch)
    p = fixture(tmp_path)
    path = tmp_path / p['output_dir'] / 'dispatch/shard_0_12/result.json'
    partial = run.read(path); partial['outcomes'].pop()
    run.core.save(path, partial)
    with pytest.raises(ValueError, match='twelve dispatch'):
        run.collect(p, 'manifest.json', root=tmp_path)
    assert not (tmp_path / p['output_dir'] / 'conclusion.json').exists()


def test_stage100_gate_does_not_import_science_or_write_before_authorization(tmp_path, monkeypatch):
    p = plan(); p['execution_authorized'] = False
    monkeypatch.setattr(run.core, '_dependencies', lambda: pytest.fail('No dependency imports'))
    with pytest.raises(PermissionError):
        run.batch(p, 'manifest.json', 0, 12, 2, root=tmp_path)
    with pytest.raises(PermissionError):
        run.collect(p, 'manifest.json', root=tmp_path)
    assert not (tmp_path / p['output_dir']).exists()
