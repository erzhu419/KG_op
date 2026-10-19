"""Synthetic sources and quadrature returns only; never run a real integral."""
from contextlib import contextmanager
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from performance import mainline_source_finite_numerical_probe as probe


def plan():
    result = json.loads((probe.ROOT / probe.MANIFEST).read_text())
    result['execution_authorized'] = True
    result['output_dir'] = 'synthetic_probe'
    return result


def costs():
    return dict(local_raw_moment_attempts=0, completed_calls=0, unreturned_calls=0,
                failed_calls=0, resource_stopped_calls=0)


class NoTimers:
    def __init__(self, *_):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def check(self):
        pass

    @contextmanager
    def single_call(self):
        yield


def recorder(tmp_path, kernel, *, budget=None):
    context = dict(identity='identity0', head='g', component_start=64,
                   component_stop=128, prefix=1, policy_index=1, repeat_index=0)
    return probe.MomentRecorder(kernel, context, tmp_path, costs(), budget or NoTimers(), plan()['limits'])


def read_calls(path):
    return [json.loads(line) for line in (path / 'call_diagnostics.jsonl').read_text().splitlines()]


def test_original_arguments_return_object_and_input_before_mutation(tmp_path):
    result = {'integration_status': 0, 'minimum_normalizer': .1}
    received = []
    means, covariance = [[1., 2.]], [[[3., 0.], [0., 4.]]]

    def kernel(*args, **kwargs):
        received.append((args, kwargs))
        assert json.loads((tmp_path / 'inputs/call_0001.json').read_text())['arguments']['means'] == [[1., 2.]]
        means[0][0] = 10.
        return result

    wrapped = recorder(tmp_path, kernel)
    returned = wrapped(means, covariance, label=5., replicates=1)
    assert returned is result
    assert len(received) == 1 and received[0][0][0] is means and received[0][0][1] is covariance
    assert received[0][1] == dict(label=5., replicates=1)
    row = read_calls(tmp_path)[0]
    assert row['diagnostics']['values']['per_node_Z'] is None
    assert row['diagnostics']['missing_fields']['shifts'] == 'not_exposed_by_return'
    assert wrapped.cost['completed_calls'] == wrapped.cost['local_raw_moment_attempts'] == 1


def test_same_exception_frame_and_missing_fields_are_preserved(tmp_path):
    original = RuntimeError('Batched raw moment error exceeds frozen tolerance: 2e-8')

    def kernel(means, covariances, label):
        integral = [[.2, 0, 0, 0, 0, 0], [.1, 0, 0, 0, 0, 0]]
        error, relative_error, shifts = 2e-9, 2e-8, [1., 2.]
        info = SimpleNamespace(status=0, success=True, message='ok', neval=9)
        raise original

    wrapped = recorder(tmp_path, kernel)
    with pytest.raises(RuntimeError) as caught:
        wrapped([[0., 0.]], [[[1., 0.], [0., 1.]]], 0.)
    assert caught.value is original
    row = read_calls(tmp_path)[0]
    assert row['diagnostics']['kernel_frame_found']
    assert row['diagnostics']['values']['argmin_Z_node'] == 65
    assert row['diagnostics']['values']['info.success'] is True
    assert row['diagnostics']['values']['error'] == 2e-9
    assert row['diagnostics']['missing_fields']['covariance'] == 'not_reached_in_kernel_frame'
    assert wrapped.cost['failed_calls'] == wrapped.cost['unreturned_calls'] == 1


def test_early_exception_nonfinite_evidence_and_logger_failure_do_not_replay(tmp_path, monkeypatch):
    original = ValueError('pre-integral error')
    count = []

    def kernel(*_):
        count.append(1)
        relative_error = float('inf')
        raise original

    wrapped = recorder(tmp_path, kernel)
    with pytest.raises(ValueError) as caught:
        wrapped([[float('nan'), 0.]], [], 0.)
    assert caught.value is original and count == [1]
    row = read_calls(tmp_path)[0]
    assert row['diagnostics']['values']['relative_error'] == {'nonfinite': 'inf'}
    assert row['diagnostics']['values']['minimum_normalizer'] is None
    assert row['diagnostics']['missing_fields']['integral'] == 'not_reached_in_kernel_frame'
    write = wrapped._write

    def failing_logger(kind, value, call_id):
        if kind == 'diagnostic':
            raise OSError('disk error')
        return write(kind, value, call_id)

    monkeypatch.setattr(wrapped, '_write', failing_logger)
    with pytest.raises(ValueError) as caught:
        wrapped([], [], 0.)
    assert caught.value is original and count == [1, 1]
    assert wrapped.recording_errors[0]['error'] == "OSError('disk error')"


def test_input_write_failure_and_call_cap_never_enter_original(tmp_path, monkeypatch):
    called = []
    wrapped = recorder(tmp_path, lambda *_: called.append(True))
    monkeypatch.setattr(wrapped, '_write', lambda *_: (_ for _ in ()).throw(OSError('input disk error')))
    with pytest.raises(probe.RecordingError):
        wrapped([], [], 0.)
    assert called == [] and wrapped.cost['local_raw_moment_attempts'] == 0
    wrapped.cost['local_raw_moment_attempts'] = 512
    with pytest.raises(probe.ResourceStop, match='maximum_raw'):
        wrapped([], [], 0.)
    assert called == []


@pytest.mark.parametrize('ratio', [1e-10, 2e-8])
def test_frozen_kernel_with_synthetic_quad_result_keeps_arguments_and_guard(tmp_path, monkeypatch, ratio):
    import numpy as np
    from representation import vectorized_joint_function_risk as module
    calls = []
    integral = np.array([[1., 0., 1., 0., 1., 0.], [.5, 0., .5, 0., .5, 0.]])

    def fake_quad(integrand, low, high, **kwargs):
        # Never call integrand or the real numerical integrator.
        calls.append((low, high, kwargs))
        return integral, ratio * .5, SimpleNamespace(status=0, success=True, message='synthetic', neval=0)

    monkeypatch.setattr(module, 'quad_vec', fake_quad)
    wrapped = recorder(tmp_path, module.local_raw_moments)
    means = np.zeros((2, 2)); covariance = np.repeat(np.eye(2)[None], 2, axis=0)
    if ratio > 1e-8:
        with pytest.raises(RuntimeError, match='exceeds frozen tolerance'):
            wrapped(means, covariance, 0.)
        row = read_calls(tmp_path)[0]
        assert row['diagnostics']['values']['relative_error'] == ratio
        assert row['diagnostics']['values']['covariance'] == np.repeat(np.eye(2)[None], 2, axis=0).tolist()
    else:
        result = wrapped(means, covariance, 0.)
        assert result['integration_relative_error'] == ratio
    assert len(calls) == 1 and calls[0][:2] == (-12., 12.)
    options = calls[0][2]
    assert {key: options[key] for key in ('norm', 'full_output', 'epsabs', 'epsrel', 'limit', 'cache_size', 'workers')} == dict(
        norm='max', full_output=True, epsabs=1e-12, epsrel=1e-12, limit=128, cache_size=16*1024**2, workers=1)
    assert 'points' in options


class FakeTimers:
    def __init__(self):
        self.handlers, self.armed = {}, {}

    def getsignal(self, sig):
        return self.handlers.get(sig, 0)

    def getitimer(self, kind):
        return self.armed.get(kind, (0., 0.))

    def signal(self, sig, handler):
        self.handlers[sig] = handler

    def setitimer(self, kind, seconds, repeat=0):
        self.armed[kind] = (seconds, repeat)


def test_deadlines_do_not_restart_and_resource_stop_is_distinct(tmp_path):
    elapsed = dict(cpu=0., wall=0.)
    timers = FakeTimers()
    with probe.BudgetScope(plan()['limits'], cpu=lambda: elapsed['cpu'], wall=lambda: elapsed['wall'], timers=timers) as budget:
        elapsed['wall'] = 100.
        with budget.single_call():
            assert timers.armed[probe.signal.ITIMER_REAL][0] == 30
            elapsed['wall'] = 110.
        assert timers.armed[probe.signal.ITIMER_REAL][0] == 490
        elapsed['wall'] = 590.
        with pytest.raises(probe.ResourceStop, match='case_wall_seconds'):
            with budget.single_call():
                assert timers.armed[probe.signal.ITIMER_REAL][0] == 10
                timers.handlers[probe.signal.SIGALRM]()
        elapsed['wall'] = 110.

        def kernel(*_):
            timers.handlers[probe.signal.SIGALRM]()

        wrapped = recorder(tmp_path, kernel, budget=budget)
        with pytest.raises(probe.ResourceStop, match='single_raw_moment_call_wall_seconds'):
            wrapped([], [], 0.)
        assert wrapped.cost['resource_stopped_calls'] == 1 and wrapped.cost['failed_calls'] == 0
        assert read_calls(tmp_path)[0]['diagnostics']['values']['integral'] is None
        elapsed['cpu'] = 300.
        with pytest.raises(probe.ResourceStop, match='case_process_cpu_seconds'):
            budget.check()
    assert timers.armed[probe.signal.ITIMER_REAL][0] == 0
    assert timers.armed[probe.signal.ITIMER_PROF][0] == 0


def worker(monkeypatch):
    monkeypatch.setattr(probe.socket, 'gethostname', lambda: 'node003')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        monkeypatch.setenv(key, '1')
    monkeypatch.setattr(probe, 'BudgetScope', NoTimers)


def synthetic_dependencies(monkeypatch, *, fail_at=None, projection_failure=False):
    calls, loaded, builds, observed = [], [], [], []
    archive_tasks = [SimpleNamespace(name=name, X=SimpleNamespace(shape=(64, 50)),
        X_integer=[[index+1]*50 for index in range(64)],
        Y_replicates=[[[10.+index, 20.+index]]*3 for index in range(64)], tau=3., alpha=.05)
        for name in reversed(probe.FAMILIES)]
    bundle = dict(identities=[f'id{i}' for i in range(8)],
                  components=[({'mean': [None]*1024}, {'mean': [None]*1024}) for _ in range(8)])
    prior = object()
    module = SimpleNamespace()
    module.quad_vec = lambda: None

    def kernel(means, covariance, label, eta_label=None, replicates=1):
        calls.append((label, eta_label, replicates))
        if fail_at == len(calls):
            raise RuntimeError('synthetic guard')
        module.quad_vec()
        return dict(integration_status=0, minimum_normalizer=1.)

    module.local_raw_moments = kernel

    def load_model(path):
        loaded.append(str(path))
        return bundle if path.name == 'source_models.pkl' else prior

    def engine(source, family, d, heldout_family):
        assert source is prior
        builds.append((family, d, heldout_family))
        return SimpleNamespace(problem=SimpleNamespace(tau=3., alpha=.05),
            config=SimpleNamespace(L=100, sigma=.04, alpha=.05, weights='0.5,0.5'))

    def inputs(original, engine, bank):
        assert original is bundle
        observed.append(copy.deepcopy(bank))
        return bundle, [[('Df', 'Ef'), ('Dg', 'Eg')]]*8

    def history(engine, bank, rows):
        assert rows[0]['raw'] == [10., 20.] and rows[1]['raw'] == [11., 21.]
        return [(index, [row['raw'][0], row['raw'][1]-engine.problem.tau]) for index, row in enumerate(rows)]

    def stream(record, D, E, history, head, batch, observer):
        assert (D, E) == [('Df', 'Ef'), ('Dg', 'Eg')][head] and batch == 64
        for start in range(0, 1024, batch):
            for prefix in range(3):
                observer(prefix, start, 0., SimpleNamespace(means=[None]*64), None, None, {0: 0, 1: 1})
                if projection_failure:
                    raise ValueError('projection failed after observer but before original kernel')
                if prefix < 2:
                    module.local_raw_moments([[0., 0.]], [[[1., 0.], [0., 1.]]], history[prefix][1][head], None, 1)
        return None

    deps = SimpleNamespace(load_archive=lambda path: SimpleNamespace(tasks=archive_tasks),
        load_model=load_model, build_source_engine=engine, role_inputs=inputs, history=history,
        stream_head=stream, kernel_module=module)
    monkeypatch.setattr(probe, '_dependencies', lambda: deps)
    return SimpleNamespace(calls=calls, loaded=loaded, builds=builds, observed=observed, module=module, kernel=kernel)


def test_authorization_node_and_threads_gate_precedes_any_dependency(tmp_path, monkeypatch):
    p = plan(); case = p['cases'][0]['case_id']
    called = []
    monkeypatch.setattr(probe, '_dependencies', lambda: called.append(True))
    p['execution_authorized'] = False
    with pytest.raises(PermissionError):
        probe.run_case(p, case, root=tmp_path)
    p['execution_authorized'] = True
    monkeypatch.setattr(probe.socket, 'gethostname', lambda: 'local')
    with pytest.raises(RuntimeError, match='scheduler'):
        probe.run_case(p, case, root=tmp_path)
    monkeypatch.setattr(probe.socket, 'gethostname', lambda: 'node001')
    monkeypatch.setenv('OPENBLAS_NUM_THREADS', '24')
    with pytest.raises(RuntimeError, match='threads=1'):
        probe.run_case(p, case, root=tmp_path)
    assert called == [] and not (tmp_path / p['output_dir']).exists()


def test_fixed_twelve_cases_same_original_rows_and_full_order(tmp_path, monkeypatch):
    worker(monkeypatch)
    p = plan()
    assert len({probe.registered_case(p, row['case_id'])[0]['case_id'] for row in p['cases']}) == 12
    deps = synthetic_dependencies(monkeypatch)
    for association in ('real', 'permuted'):
        case_id = f'InventorySupplyChain__{association}__QueueResourceControl'
        result = probe.run_case(p, case_id, root=tmp_path)
        assert result['status'] == 'complete'
        assert result['cost']['local_raw_moment_attempts'] == result['cost']['quad_vec_attempts'] == 512
        assert result['cost']['source_model_file_load_attempts'] == result['cost']['source_model_file_loads'] == 2
        assert result['cost']['source_conditioning_calls'] == 16
        assert result['not_reached_kernel_calls'] == 0
        assert result['unchanged_historical_coverage']['retained_failures'] == 67
        assert len(result['retained_terminal_arms']) == 11 and result['zero_work']['old_failed_query_replay'] == 0
        directory = tmp_path / p['output_dir'] / 'cases' / case_id
        contexts = [row['context'] for row in read_calls(directory)]
        expected = [(f'id{i}', head, start, prefix) for i in range(8) for head in ('f', 'g')
                    for start in range(0, 1024, 64) for prefix in (0, 1)]
        assert [(c['identity'], c['head'], c['component_start'], c['prefix']) for c in contexts] == expected
        assert all(c['component_stop'] == c['component_start']+64 for c in contexts)
    assert deps.calls[:512] == deps.calls[512:]
    assert deps.observed == [[(1,)*50, (2,)*50]]*2
    assert deps.builds == [('QueueResourceControl', 50, 'InventorySupplyChain')]*2
    assert deps.module.local_raw_moments is deps.kernel
    # Completed results are reused without any second model load or kernel call.
    probe.run_case(p, case_id, root=tmp_path)
    assert len(deps.calls) == 1024 and len(deps.loaded) == 4


def test_first_failure_retained_no_retry_and_unreached_not_zero(tmp_path, monkeypatch):
    worker(monkeypatch)
    p = plan(); case = p['cases'][0]['case_id']
    deps = synthetic_dependencies(monkeypatch, fail_at=2)
    result = probe.run_case(p, case, root=tmp_path)
    assert result['status'] == 'kernel_exception'
    assert result['cost']['local_raw_moment_attempts'] == 2
    assert result['cost']['quad_vec_attempts'] == 1
    assert result['cost']['completed_calls'] == result['cost']['failed_calls'] == result['cost']['unreturned_calls'] == 1
    assert result['not_reached_kernel_calls'] == 510
    assert result['cost']['process_cpu_seconds'] >= 0 and result['cost']['wall_seconds'] >= 0
    probe.run_case(p, case, root=tmp_path)
    assert len(deps.calls) == 2
    other = p['cases'][1]['case_id']
    probe.save(tmp_path / p['output_dir'] / 'cases' / other / 'started.json', {'status': 'started'})
    with pytest.raises(RuntimeError, match='retry'):
        probe.run_case(p, other, root=tmp_path)
    assert len(deps.loaded) == 2


def test_observer_is_not_a_kernel_attempt(tmp_path, monkeypatch):
    worker(monkeypatch)
    p = plan()
    deps = synthetic_dependencies(monkeypatch, projection_failure=True)
    result = probe.run_case(p, p['cases'][0]['case_id'], root=tmp_path)
    assert result['status'] == 'input_or_projection_error'
    assert result['cost']['local_raw_moment_attempts'] == result['cost']['quad_vec_attempts'] == 0
    assert result['not_reached_kernel_calls'] == 512 and deps.calls == []


def test_expired_after_input_never_counts_unentered_kernel(tmp_path, monkeypatch):
    elapsed = dict(wall=0.)
    timers = FakeTimers()
    invoked = []
    with probe.BudgetScope(plan()['limits'], cpu=lambda: 0., wall=lambda: elapsed['wall'], timers=timers) as budget:
        wrapped = recorder(tmp_path, lambda *_: invoked.append(True), budget=budget)
        write = wrapped._write

        def input_uses_remaining_budget(kind, value, call_id):
            write(kind, value, call_id)
            if kind == 'input':
                elapsed['wall'] = 600.

        monkeypatch.setattr(wrapped, '_write', input_uses_remaining_budget)
        with pytest.raises(probe.ResourceStop, match='case_wall_seconds'):
            wrapped([], [], 0.)
        assert invoked == [] and wrapped.cost['local_raw_moment_attempts'] == 0
        assert wrapped.cost['unreturned_calls'] == wrapped.cost['resource_stopped_calls'] == 0
        row = read_calls(tmp_path)[0]
        assert not row['entered_kernel'] and not row['returned']
        assert timers.armed[probe.signal.ITIMER_REAL][0] == 0
        assert timers.armed[probe.signal.ITIMER_PROF][0] == 0


def test_post_return_deadline_retains_successful_evidence(tmp_path):
    elapsed = dict(wall=0.)
    timers = FakeTimers()
    returned = dict(integration_relative_error=1e-10, minimum_normalizer=.2,
                    integration_absolute_error=2e-11, integration_status=0)

    def kernel(*_):
        elapsed['wall'] = 30.
        return returned

    with probe.BudgetScope(plan()['limits'], cpu=lambda: 0., wall=lambda: elapsed['wall'], timers=timers) as budget:
        wrapped = recorder(tmp_path, kernel, budget=budget)
        with pytest.raises(probe.ResourceStop, match='single_raw_moment_call_wall_seconds'):
            wrapped([], [], 0.)
        assert wrapped.cost['completed_calls'] == 1 and wrapped.cost['unreturned_calls'] == 0
        row = read_calls(tmp_path)[0]
        assert row['returned'] and row['status'] == 'resource_stop'
        assert row['diagnostics']['values']['minimum_normalizer'] == .2
        assert row['diagnostics']['values']['integration_absolute_error'] == 2e-11
        assert timers.armed[probe.signal.ITIMER_REAL][0] == 0


@pytest.mark.parametrize('write_kind', ['input', 'diagnostic'])
def test_write_deadline_keeps_resource_category(tmp_path, monkeypatch, write_kind):
    count = []
    wrapped = recorder(tmp_path, lambda *_: count.append(1) or {})
    write = wrapped._write

    def interrupted(kind, value, call_id):
        if kind == write_kind:
            raise probe.ResourceStop('case_process_cpu_seconds')
        return write(kind, value, call_id)

    monkeypatch.setattr(wrapped, '_write', interrupted)
    with pytest.raises(probe.ResourceStop, match='case_process_cpu_seconds'):
        wrapped([], [], 0.)
    assert len(count) == (write_kind == 'diagnostic')
