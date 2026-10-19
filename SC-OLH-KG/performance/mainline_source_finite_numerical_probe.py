"""Finite source-only probes; real execution requires separate authorization.

The production kernel is never edited. Inputs are written before each call;
exception diagnostics come from that same call's traceback, never a replay.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import socket
import time
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = 'performance/manifests/mainline_source_finite_numerical_validation_v1.json'
FAMILIES = ('InventorySupplyChain', 'QueueResourceControl', 'FactorShockStatePolicyRZDT1')
SUCCESS_FIELDS = ('integration_absolute_error', 'minimum_normalizer',
    'integration_relative_error', 'integration_status', 'integrand_evaluations',
    'bounded_error_accepted', 'roundoff_accepted')
HIDDEN_FIELDS = ('per_node_Z', 'argmin_Z_node', 'shifts', 'info.success', 'info.message')


def now():
    return datetime.now(timezone.utc).isoformat()


def json_value(value):
    """Retain nonfinite evidence explicitly instead of losing the exception."""
    if hasattr(value, 'tolist'):
        return json_value(value.tolist())
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite': repr(value)}
    return value


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(json_value(value), indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def append(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(json_value(value), allow_nan=False) + '\n')


class ResourceStop(RuntimeError):
    pass


class RecordingError(RuntimeError):
    pass


class BudgetScope:
    """One worker's absolute case deadlines, shortened only during a call."""
    def __init__(self, limits, *, cpu=time.process_time, wall=time.monotonic, timers=signal):
        self.limits, self.cpu, self.wall, self.timers = limits, cpu, wall, timers

    def __enter__(self):
        self.cpu_start, self.wall_start = self.cpu(), self.wall()
        self.stopped = False
        self.deadline = self.wall_start + self.limits['case_wall_seconds']
        self.wall_reason = 'case_wall_seconds'
        self.previous = [(kind, self.timers.getsignal(sig), self.timers.getitimer(kind))
                         for kind, sig in ((signal.ITIMER_REAL, signal.SIGALRM),
                                           (signal.ITIMER_PROF, signal.SIGPROF))]
        if any(timer[0] for _, _, timer in self.previous):
            raise RuntimeError('Probe worker already has an active interval timer')
        self.timers.signal(signal.SIGALRM, self._wall_expired)
        self.timers.signal(signal.SIGPROF, self._cpu_expired)
        self.timers.setitimer(signal.ITIMER_REAL, self.limits['case_wall_seconds'])
        self.timers.setitimer(signal.ITIMER_PROF, self.limits['case_process_cpu_seconds'])
        return self

    def _wall_expired(self, *_):
        raise ResourceStop(self.wall_reason)

    def _cpu_expired(self, *_):
        raise ResourceStop('case_process_cpu_seconds')

    def check(self):
        if self.cpu() - self.cpu_start >= self.limits['case_process_cpu_seconds']:
            raise ResourceStop('case_process_cpu_seconds')
        if self.wall() >= self.deadline:
            raise ResourceStop('case_wall_seconds')

    def disarm(self):
        self.stopped = True
        self.timers.setitimer(signal.ITIMER_REAL, 0)
        self.timers.setitimer(signal.ITIMER_PROF, 0)

    @contextmanager
    def single_call(self):
        try:
            self.check()
            started = self.wall()
            remaining = self.deadline - started
            cap = self.limits['single_raw_moment_call_wall_seconds']
            self.wall_reason = 'case_wall_seconds' if remaining <= cap else 'single_raw_moment_call_wall_seconds'
            self.timers.setitimer(signal.ITIMER_REAL, min(remaining, cap))
            yield
            self.check()
            if self.wall() - started >= cap:
                raise ResourceStop('single_raw_moment_call_wall_seconds')
        except BaseException:
            # This case stops on its first error. Do not interrupt preservation
            # of that original error with a second timer signal.
            self.disarm()
            raise
        finally:
            self.wall_reason = 'case_wall_seconds'
            if not self.stopped:
                remaining = self.deadline - self.wall()
                if remaining > 0:
                    self.timers.setitimer(signal.ITIMER_REAL, remaining)
                else:
                    self.disarm()

    def __exit__(self, *_):
        for (kind, handler, timer), sig in zip(self.previous, (signal.SIGALRM, signal.SIGPROF)):
            self.timers.setitimer(kind, 0)
            self.timers.signal(sig, handler)
            self.timers.setitimer(kind, *timer)


def exception_diagnostics(error, kernel, context):
    frame = None
    trace = error.__traceback__
    while trace is not None:
        if trace.tb_frame.f_code is kernel.__code__:
            frame = trace.tb_frame
            break
        trace = trace.tb_next
    values = frame.f_locals if frame is not None else {}
    missing = {}

    def recorded(name):
        if name not in values:
            missing[name] = 'not_reached_in_kernel_frame' if frame is not None else 'kernel_frame_absent'
        return values.get(name)

    result = {key: recorded(key) for key in ('integral', 'error', 'relative_error',
        'shifts', 'statistics', 'displacement', 'covariance', 'roundoff_accepted', 'bounded_error_accepted')}
    info = values.get('info')
    for name in ('success', 'status', 'message', 'neval'):
        key = 'info.' + name
        result[key] = getattr(info, name, None)
        if info is None or not hasattr(info, name):
            missing[key] = 'not_reached_in_kernel_frame' if frame is not None else 'kernel_frame_absent'
    result.update(per_node_Z=None, minimum_normalizer=None, argmin_Z_node=None)
    if result['integral'] is not None:
        # This reads the already returned first column, without another integral.
        normalizers = [row[0] for row in result['integral']]
        result['per_node_Z'] = normalizers
        if normalizers and all(math.isfinite(float(value)) for value in normalizers):
            index = min(range(len(normalizers)), key=normalizers.__getitem__)
            result['minimum_normalizer'] = normalizers[index]
            result['argmin_Z_node'] = context['component_start'] + index
        else:
            missing.update(minimum_normalizer='nonfinite_or_empty_normalizers',
                           argmin_Z_node='nonfinite_or_empty_normalizers')
    else:
        missing.update({key: 'integral_not_available' for key in
                        ('per_node_Z', 'minimum_normalizer', 'argmin_Z_node')})
    return dict(values=result, missing_fields=missing, kernel_frame_found=frame is not None)


def success_diagnostics(result):
    diagnostics = dict(values={key: result.get(key) for key in SUCCESS_FIELDS},
        missing_fields={key: 'not_exposed_by_return' for key in HIDDEN_FIELDS})
    diagnostics['values'].update({key: None for key in HIDDEN_FIELDS})
    for key in SUCCESS_FIELDS:
        if key not in result:
            diagnostics['missing_fields'][key] = 'not_exposed_by_return'
    return diagnostics


class MomentRecorder:
    def __init__(self, kernel, context, directory, cost, budget, limits):
        self.kernel, self.context, self.directory = kernel, context, Path(directory)
        self.cost, self.budget, self.limits = cost, budget, limits
        self.last_context = None
        self.recording_errors = []

    def _write(self, kind, value, call_id):
        if kind == 'input':
            save(self.directory / 'inputs' / f'call_{call_id:04d}.json', value)
        else:
            append(self.directory / 'call_diagnostics.jsonl', value)

    def __call__(self, *args, **kwargs):
        self.budget.check()
        if self.cost['local_raw_moment_attempts'] >= self.limits['maximum_raw_moment_calls_per_case']:
            raise ResourceStop('maximum_raw_moment_calls_per_case')
        names = ('means', 'covariances', 'label', 'eta_label', 'replicates')
        supplied = dict(eta_label=None, replicates=1)
        supplied.update(zip(names, args)); supplied.update(kwargs)
        call_id = self.cost['local_raw_moment_attempts'] + 1
        context = dict(self.context)
        self.last_context = context
        try:
            self._write('input', dict(call_id=call_id, context=context,
                arguments={name: supplied.get(name) for name in names}), call_id)
        except ResourceStop:
            raise
        except Exception as error:
            raise RecordingError(f'Input packet was not saved: {error!r}') from error
        cpu, wall = time.process_time(), time.monotonic()
        entered = returned = False
        try:
            with self.budget.single_call():
                entered = True
                self.cost['local_raw_moment_attempts'] += 1
                result = self.kernel(*args, **kwargs)
                returned = True
                self.cost['completed_calls'] += 1
        except BaseException as error:
            if entered and not returned:
                self.cost['unreturned_calls'] += 1
            if entered:
                self.cost['resource_stopped_calls' if isinstance(error, ResourceStop) else 'failed_calls'] += 1
            row = dict(call_id=call_id, context=context, entered_kernel=entered, returned=returned,
                status='resource_stop' if isinstance(error, ResourceStop) else 'kernel_exception',
                error=repr(error), diagnostics=(success_diagnostics(result) if returned else
                    exception_diagnostics(error, self.kernel, context)),
                process_cpu_seconds=time.process_time()-cpu, wall_seconds=time.monotonic()-wall)
            try:
                self._write('diagnostic', row, call_id)
            except Exception as recording_error:
                # Never replace the original numeric exception with a logger error.
                self.recording_errors.append(dict(call_id=call_id, error=repr(recording_error)))
            raise
        diagnostics = success_diagnostics(result)
        try:
            self._write('diagnostic', dict(call_id=call_id, context=context, status='returned',
                returned=True, diagnostics=diagnostics, process_cpu_seconds=time.process_time()-cpu,
                wall_seconds=time.monotonic()-wall), call_id)
        except ResourceStop:
            raise
        except Exception as error:
            raise RecordingError(f'Returned call diagnostic was not saved: {error!r}') from error
        return result


def registered_case(plan, case_id):
    expected = [(group, association, source) for group in FAMILIES
                for association in ('real', 'permuted') for source in FAMILIES if source != group]
    cases = plan['cases']
    if len(cases) != 12 or {(row['asset_group'], row['association'], row['source_family'])
                          for row in cases} != set(expected):
        raise ValueError('Retain all twelve registered source cases')
    if any(row['policy_indices'] != [0, 1] or row['repeat_indices'] != [0, 0] for row in cases):
        raise ValueError('Retain original policies 0,1 and repeat 0')
    case = next(row for row in cases if row['case_id'] == case_id)
    group = next(row for row in plan['asset_groups'] if row['asset_group'] == case['asset_group'])
    if set(group['actual_source_families']) != set(FAMILIES)-{case['asset_group']}:
        raise ValueError('Source group must exclude its held-out family')
    contract = plan['fixed_probe_contract']
    for key, value in dict(identities=8, heads=['f', 'g'], nodes_per_identity_head=1024,
                          batch_size=64, raw_rows_per_case=2, case_count=12).items():
        if contract[key] != value:
            raise ValueError(f'Frozen probe contract changed: {key}')
    for key, value in dict(maximum_attempts_per_case=1, maximum_raw_moment_calls_per_case=512,
        case_process_cpu_seconds=300, case_wall_seconds=600, single_raw_moment_call_wall_seconds=30).items():
        if plan['limits'][key] != value:
            raise ValueError(f'Frozen resource bound changed: {key}')
    return case, group


def require_execution(plan):
    if plan.get('execution_authorized') is not True:
        raise PermissionError('Real finite source probes are not authorized')
    host = socket.gethostname().split('.')[0]
    if host not in {f'node{index:03d}' for index in range(1, 7)}:
        raise RuntimeError('Finite source probes must run through scheduler on node001-node006')
    if any(os.environ.get(key) != '1' for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')):
        raise RuntimeError('Each probe worker requires OMP/MKL/OpenBLAS threads=1')


def _dependencies():
    # Only called inside the authorized worker's case budget.
    from baselines.transfer_archive import FrozenTransferArchive
    from performance.mainline_role_source_holdout import build_source_engine
    from performance.mainline_role_prior import role_inputs, _history
    from performance.diagnose_mainline_initialization import stream_head
    from performance.run_mainline_legacy_backend_bridge import load_pickle
    from representation import vectorized_joint_function_risk
    return SimpleNamespace(load_archive=FrozenTransferArchive.load, load_model=load_pickle,
        build_source_engine=build_source_engine, role_inputs=role_inputs, history=_history,
        stream_head=stream_head, kernel_module=vectorized_joint_function_risk)


def _probe(plan, case, group, directory, cost, budget, root):
    deps = _dependencies()
    archive = deps.load_archive(root / group['original_source_archive_file'])
    task = next(task for task in archive.tasks if task.name == case['source_family'])
    if task.X.shape[1] != 50:
        raise ValueError('Use the actual registered d=50 source task')
    bank = [tuple(int(value) for value in task.X_integer[index]) for index in (0, 1)]
    rows = [dict(point=list(bank[index]), raw=list(task.Y_replicates[index][0])) for index in (0, 1)]
    if any(len(row['raw']) != 2 for row in rows):
        raise ValueError('The fixed source record must retain its complete f/g pair')
    assets = group['assets'][case['association']]
    models = []
    for key in ('source_models_file', 'native_source_prior_file'):
        cost['source_model_file_load_attempts'] += 1
        model = deps.load_model(root / assets[key])
        cost['source_model_file_loads'] += 1
        models.append(model)
    bundle, prior = models
    engine = deps.build_source_engine(prior, task.name, d=50, heldout_family=case['asset_group'])
    if engine.problem.tau != task.tau or engine.problem.alpha != task.alpha:
        raise ValueError('Source projection and archived response scales differ')
    cost['source_projection_attempts'] += 1
    source, designs = deps.role_inputs(bundle, engine, bank)
    history = deps.history(engine, bank, rows)
    cost['source_projection_calls'] += 1
    if len(source['identities']) != 8 or len(source['components']) != 8 or len(designs) != 8:
        raise ValueError('Preserve all eight source identities')
    save(directory / 'fixed_inputs.json', dict(case=case, d=50, tau=task.tau,
        alpha=task.alpha, bank=bank, original_rows=rows, archive_file=group['original_source_archive_file'],
        source_assets=assets, source_config={key: getattr(engine.config, key) for key in
                                            ('L', 'sigma', 'alpha', 'weights')}))
    context = dict(case, tau=task.tau)
    recorder = MomentRecorder(deps.kernel_module.local_raw_moments, context, directory,
                              cost, budget, plan['limits'])
    original_quad = deps.kernel_module.quad_vec

    def counted_quad(*args, **kwargs):
        cost['quad_vec_attempts'] += 1
        return original_quad(*args, **kwargs)

    try:
        with patch.object(deps.kernel_module, 'local_raw_moments', recorder), \
                patch.object(deps.kernel_module, 'quad_vec', counted_quad):
            for identity, pair, design_pair in zip(source['identities'], source['components'], designs):
                if len(pair) != 2 or len(design_pair) != 2:
                    raise ValueError('Preserve f/g heads')
                for head, (record, (D, E)) in enumerate(zip(pair, design_pair)):
                    if len(record['mean']) != 1024:
                        raise ValueError('Preserve all 1024 nodes')

                    def observe(prefix, start, mass, field, LF, unobserved, local):
                        budget.check()
                        if prefix < len(history):
                            context.update(identity=identity, head=('f', 'g')[head],
                                component_start=start, component_stop=start+len(field.means),
                                prefix=prefix, policy_index=case['policy_indices'][prefix],
                                repeat_index=case['repeat_indices'][prefix],
                                original_raw_pair=rows[prefix]['raw'])

                    cost['source_conditioning_attempts'] += 1
                    deps.stream_head(record, D, E, history, head, 64, observer=observe)
                    cost['source_conditioning_calls'] += 1
    finally:
        cost['last_kernel_context'] = recorder.last_context
        cost['recording_errors'] = recorder.recording_errors


def run_case(plan, case_id, *, root=ROOT):
    require_execution(plan)
    case, group = registered_case(plan, case_id)
    directory = Path(root) / plan['output_dir'] / 'cases' / case_id
    result_path = directory / 'result.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    if (directory / 'started.json').exists():
        raise RuntimeError('Retain the started case; automatic retry is forbidden')
    cpu, wall = time.process_time(), time.monotonic()
    cost = {key: 0 for key in ('source_model_file_load_attempts', 'source_model_file_loads',
        'source_projection_attempts', 'source_projection_calls', 'source_conditioning_attempts',
        'source_conditioning_calls', 'local_raw_moment_attempts', 'quad_vec_attempts', 'completed_calls',
        'failed_calls', 'resource_stopped_calls', 'unreturned_calls')}
    result = dict(case=case, status='input_or_projection_error', started_utc=now(),
        cost=cost, attempt_count=1, automatic_retries=0,
        source_asset_seal_file=plan['source_asset_seal_file'],
        retained_terminal_arms=plan['retained_terminal_arms'],
        unchanged_historical_coverage=plan['unchanged_historical_coverage'],
        retained_source_warnings=plan['retained_source_warnings'],
        historical_accounts=plan['historical_accounts'],
        numerical_accuracy='not_assessed', target_d1000_decision_gain='not_tested',
        zero_work={key: 0 for key in plan['future_zero_work']})
    try:
        with BudgetScope(plan['limits']) as budget:
            save(directory / 'started.json', dict(case=case, started_utc=result['started_utc'], attempt=1))
            _probe(plan, case, group, directory, cost, budget, Path(root))
            if cost['local_raw_moment_attempts'] != 512 or cost['completed_calls'] != 512:
                raise ValueError('The full fixed probe did not reach all 512 original calls')
            budget.check()
            result['status'] = 'complete'
    except ResourceStop as error:
        result.update(status='resource_stop', error=repr(error))
    except RecordingError as error:
        result.update(status='recording_error', error=repr(error))
    except Exception as error:
        result.update(status='kernel_exception' if cost['failed_calls'] else 'input_or_projection_error',
                      error=repr(error))
    result.update(completed_utc=now(), not_reached_kernel_calls=512-cost['local_raw_moment_attempts'])
    cost.update(process_cpu_seconds=time.process_time()-cpu, wall_seconds=time.monotonic()-wall)
    save(result_path, result)
    # Final accounting includes the result write; this small append is coordination overhead.
    append(directory / 'attempts.jsonl', dict(case_id=case_id, status=result['status'],
        process_cpu_seconds=time.process_time()-cpu, wall_seconds=time.monotonic()-wall,
        result_costs_are_nested=True, automatic_retries=0))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=MANIFEST)
    parser.add_argument('--case', required=True)
    args = parser.parse_args()
    plan = json.loads((ROOT / args.manifest).read_text())
    result = run_case(plan, args.case)
    print(json.dumps(dict(case_id=args.case, status=result['status'], cost=result['cost'])))
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
