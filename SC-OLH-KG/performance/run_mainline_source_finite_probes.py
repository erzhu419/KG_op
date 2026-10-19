"""Dispatch the twelve frozen source probes and summarize saved records only."""
import argparse
from collections import Counter
import json
import math
import multiprocessing
import os
from pathlib import Path
import socket
import time

from performance import mainline_source_finite_numerical_probe as core


ROOT, MANIFEST = core.ROOT, core.MANIFEST
COUNTERS = ('source_model_file_load_attempts', 'source_model_file_loads',
    'source_projection_attempts', 'source_projection_calls', 'source_conditioning_attempts',
    'source_conditioning_calls', 'local_raw_moment_attempts', 'quad_vec_attempts',
    'completed_calls', 'failed_calls', 'resource_stopped_calls', 'unreturned_calls')


def read(path):
    return json.loads(Path(path).read_text())


def _case_directory(root, plan, case_id):
    return Path(root) / plan['output_dir'] / 'cases' / case_id


def case_worker(arguments):
    """Runs once in a fresh process; expected scientific failures are results."""
    plan, case_id, root = arguments
    core.require_execution(plan)
    directory = _case_directory(root, plan, case_id)
    receipt = directory / 'worker.json'
    if receipt.exists():
        return read(receipt)
    cpu, wall = time.process_time(), time.monotonic()
    reused = (directory / 'result.json').exists()
    row = dict(case_id=case_id, process_id=os.getpid(), node=socket.gethostname(),
               core_result_reused=reused, automatic_retries=0, started_utc=core.now())
    if not reused and (directory / 'started.json').exists():
        row.update(status='retained_missing', error='Started case has no result; it was not rerun',
                   core_invoked=False)
    else:
        try:
            result = core.run_case(plan, case_id, root=Path(root))
            row.update(status=result['status'], core_invoked=True)
        except BaseException as error:
            # Do not replace the core's result, input packets, or started marker.
            row.update(status='worker_error', error=repr(error), core_invoked=True)
    row.update(process_cpu_seconds=time.process_time()-cpu, wall_seconds=time.monotonic()-wall,
               completed_utc=core.now())
    core.save(receipt, row)
    return row


def batch(plan, manifest, start, end, workers, *, root=ROOT):
    core.require_execution(plan)
    if not 0 <= start < end <= len(plan['cases']) == 12 or workers < 1:
        raise ValueError('Use a registered case slice and positive allocated worker count')
    for case in plan['cases']:
        core.registered_case(plan, case['case_id'])
    directory = Path(root) / plan['output_dir'] / 'dispatch' / f'shard_{start}_{end}'
    result_file = directory / 'result.json'
    if result_file.exists():
        return read(result_file)
    if (directory / 'started.json').exists():
        raise RuntimeError('Retain this unfinished shard; no automatic resubmission')
    cpu, wall = time.process_time(), time.monotonic()
    arguments = [(plan, case['case_id'], str(root)) for case in plan['cases'][start:end]]
    core.save(directory / 'started.json', dict(manifest=str(manifest), start=start, end=end,
        case_ids=[item[1] for item in arguments], workers=min(workers, end-start),
        started_utc=core.now(), automatic_retries=0))
    rows = []
    # Python 3.10 supports maxtasksperchild here. Every probe has a fresh main
    # thread and process for the frozen core's signal timers and CPU accounting.
    with multiprocessing.get_context('spawn').Pool(
            processes=min(workers, end-start), maxtasksperchild=1) as pool:
        for row in pool.imap_unordered(case_worker, arguments, chunksize=1):
            rows.append(row)
            print('SCHEDULEURM_PROGRESS ' + json.dumps(dict(current=len(rows), total=end-start,
                unit='source_case', elapsed_seconds=time.monotonic()-wall)), flush=True)
    order = {case['case_id']: index for index, case in enumerate(plan['cases'])}
    rows.sort(key=lambda row: order[row['case_id']])
    result = dict(status='finished', start=start, end=end, outcomes=rows,
        workers=min(workers, end-start), fresh_process_per_case=True, node=socket.gethostname(),
        coordinator_cpu_seconds=time.process_time()-cpu, wall_seconds=time.monotonic()-wall,
        noncomplete_cases=[row['case_id'] for row in rows if row['status'] != 'complete'],
        automatic_retries=0, completed_utc=core.now())
    core.save(result_file, result)
    return result


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def totals(rows, key):
    known = [row[key] for row in rows if finite(row.get(key))]
    return dict(total=sum(known) if len(known) == len(rows) else None,
        known_subtotal=sum(known), known_count=len(known), registered_count=len(rows),
        unknown_case_ids=[row['case_id'] for row in rows if not finite(row.get(key))])


def saved(path, issues, *, lines=False):
    path = Path(path)
    if not path.exists():
        issues.append(dict(file=str(path), reason='missing_saved_record'))
        return None
    try:
        if lines:
            return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return read(path)
    except (OSError, ValueError) as error:
        issues.append(dict(file=str(path), reason='unreadable_saved_record', error=repr(error)))
        return None


def compact_call(row):
    values = row.get('diagnostics', {}).get('values', {})
    aliases = dict(integration_absolute_error='error', integration_relative_error='relative_error',
        integration_status='info.status', integrand_evaluations='info.neval')
    scalar = {key: values.get(key, values.get(aliases.get(key))) for key in core.SUCCESS_FIELDS}
    scalar.update({key: values.get(key) for key in ('argmin_Z_node', 'info.success', 'info.message')})
    return dict(call_id=row.get('call_id'), status=row.get('status'), returned=row.get('returned'),
        entered_kernel=row.get('entered_kernel'), context=row.get('context'), error=row.get('error'),
        recorded=scalar, missing_fields=row.get('diagnostics', {}).get('missing_fields', {}))


def extreme(calls, key, *, smallest=False):
    known = [row for row in calls if finite(row['recorded'].get(key))]
    return (min if smallest else max)(known, key=lambda row: row['recorded'][key]) if known else None


def case_summary(plan, case, root, dispatch):
    directory = _case_directory(root, plan, case['case_id'])
    issues = []
    result = saved(directory / 'result.json', issues)
    calls = saved(directory / 'call_diagnostics.jsonl', issues, lines=True)
    attempts = saved(directory / 'attempts.jsonl', issues, lines=True)
    worker = saved(directory / 'worker.json', issues)
    compact = [compact_call(row) for row in calls or []]
    cost = (result or {}).get('cost', {})
    stop_calls = [row for row in compact if row['status'] != 'returned']
    status = result.get('status') if result else 'unknown'
    numeric_rejections = [row for row in stop_calls if row['status'] == 'kernel_exception' and
        any(text in (row['error'] or '') for text in
            ('Batched raw moment error exceeds frozen tolerance', 'Batched raw moment integration failed'))]
    category = 'numerical_rejection' if status == 'kernel_exception' and numeric_rejections else status
    attempt_cpu = (sum(row['process_cpu_seconds'] for row in attempts)
        if attempts and all(finite(row.get('process_cpu_seconds')) for row in attempts) else None)
    worker_cpu = (worker or {}).get('process_cpu_seconds')
    # A reused historical result was not paid again by the current worker.
    worker_overhead = (worker_cpu if worker and (worker['core_result_reused'] or not worker['core_invoked'])
        else worker_cpu-attempt_cpu if finite(worker_cpu) and finite(attempt_cpu) else None)
    return dict(case=case, status=status, category=category, error=(result or {}).get('error'),
        worker_status=(worker or {}).get('status'), dispatch_status=dispatch.get(case['case_id'], {}).get('status'),
        result_file=str(directory / 'result.json'), attempt_count=(result or {}).get('attempt_count'),
        saved_attempt_rows=len(attempts) if attempts is not None else None,
        automatic_retries=(result or {}).get('automatic_retries'),
        not_reached_kernel_calls=(result or {}).get('not_reached_kernel_calls'),
        original_cost=cost if result else None, case_attempt_cpu_seconds=attempt_cpu,
        worker_process_cpu_seconds_inclusive=worker_cpu,
        worker_driver_cpu_seconds_excluding_case_attempt=worker_overhead,
        saved_call_rows=len(calls) if calls is not None else None,
        saved_entered_kernel_rows=(sum(row.get('entered_kernel', row.get('status') == 'returned') is True
            for row in calls) if calls is not None else None),
        saved_returned_rows=(sum(row.get('returned') is True for row in calls) if calls is not None else None),
        bounded_accepted_returned_calls=sum(row['returned'] is True and
            row['recorded'].get('bounded_error_accepted') is True for row in compact) if calls is not None else None,
        maximum_recorded_relative_error=extreme(compact, 'integration_relative_error'),
        minimum_recorded_normalizer=extreme(compact, 'minimum_normalizer', smallest=True),
        nonfinite_error_records=[row for row in compact if isinstance(
            row['recorded'].get('integration_relative_error'), dict)],
        stop_calls=stop_calls, numeric_rejection_call_count=len(numeric_rejections) if calls is not None else None,
        zero_work=(result or {}).get('zero_work'), record_issues=issues,
        record_counts_match=(calls is not None and result is not None and
            sum(row.get('entered_kernel', row.get('status') == 'returned') is True for row in calls)
            == cost.get('local_raw_moment_attempts')))


def collect(plan, manifest, *, root=ROOT):
    core.require_execution(plan)
    root, output = Path(root), Path(root) / plan['output_dir']
    if (output / 'completion.json').exists():
        return read(output / 'completion.json')
    cpu, wall = time.process_time(), time.monotonic()
    shards = [read(path) for path in sorted((output / 'dispatch').glob('shard_*/result.json'))]
    dispatch = {row['case_id']: row for shard in shards for row in shard['outcomes']}
    registered = {case['case_id'] for case in plan['cases']}
    if len(registered) != 12 or set(dispatch) != registered or any(shard['status'] != 'finished' for shard in shards):
        raise ValueError('Seal all twelve dispatch outcomes before collecting; missing science remains unknown')
    for case in plan['cases']:
        core.registered_case(plan, case['case_id'])
    rows = [case_summary(plan, case, root, dispatch) for case in plan['cases']]
    records = [dict(case_id=row['case']['case_id'], **(row['original_cost'] or {}),
        case_attempt_cpu_seconds=row['case_attempt_cpu_seconds'],
        worker_process_cpu_seconds_inclusive=row['worker_process_cpu_seconds_inclusive'],
        worker_driver_cpu_seconds_excluding_case_attempt=row['worker_driver_cpu_seconds_excluding_case_attempt'])
        for row in rows]
    accounting = {key: totals(records, key) for key in (*COUNTERS, 'process_cpu_seconds',
        'case_attempt_cpu_seconds', 'worker_process_cpu_seconds_inclusive',
        'worker_driver_cpu_seconds_excluding_case_attempt')}
    accounting.update(batch_driver_cpu_seconds=sum(shard['coordinator_cpu_seconds'] for shard in shards),
        scope='Case result CPU is nested in case attempt CPU, which is nested in worker/wrapper CPU. '
              'Use leaf case attempts once; worker driver and batch/collection CPU are separate. '
              'Missing costs are null with known subtotals; retained historical fees are not charged again.')
    status_counts = dict(Counter(row['category'] for row in rows))
    all_complete = all(row['status'] == 'complete' for row in rows)
    executable = ('no_failure_on_fixed_inputs' if all_complete else
        'numerical_rejection_on_fixed_inputs' if status_counts.get('numerical_rejection') else
        'resource_limited' if status_counts.get('resource_stop') else 'incomplete_or_input_failure')
    historical = {key: plan[key] for key in ('retained_terminal_arms', 'unchanged_historical_coverage',
        'retained_source_warnings', 'historical_accounts', 'historical_failure_fees')}
    zero_work = {key: totals([dict(case_id=row['case']['case_id'], value=(row['zero_work'] or {}).get(key))
        for row in rows], 'value') for key in plan['future_zero_work']}
    summaries = dict(status='collected', manifest=str(manifest), registered_cases=12, cases=rows,
        accounting=accounting, zero_work=zero_work, historical=historical,
        collection_cpu_seconds=time.process_time()-cpu, collection_wall_seconds=time.monotonic()-wall)
    core.save(output / 'case_status_and_costs.json', summaries)
    extrema = [row['maximum_recorded_relative_error'] for row in rows if row['maximum_recorded_relative_error']]
    minima = [row['minimum_recorded_normalizer'] for row in rows if row['minimum_recorded_normalizer']]
    conclusion = dict(status='complete', registered_cases=12, status_counts=status_counts,
        scientific_result_cases=sum(row['status'] != 'unknown' for row in rows),
        record_completeness='complete' if all(not row['record_issues'] and row['record_counts_match'] and
            not (row['original_cost'] or {}).get('recording_errors') for row in rows) else 'partial',
        fixed_input_executability=executable, numerical_accuracy='not_assessed',
        target_d1000_decision_gain='not_tested', diagnosis='cause_unidentified',
        maximum_recorded_relative_error=extreme(extrema, 'integration_relative_error'),
        minimum_recorded_normalizer=extreme(minima, 'minimum_normalizer', smallest=True),
        stop_calls=[call for row in rows for call in row['stop_calls']],
        input_or_missing_cases=[dict(case=row['case'], status=row['status'], error=row['error'],
            worker_status=row['worker_status'], record_issues=row['record_issues']) for row in rows
            if row['status'] in ('unknown', 'input_or_projection_error', 'recording_error')],
        bounded_accepted_returned_calls=totals([dict(case_id=row['case']['case_id'],
            value=row['bounded_accepted_returned_calls']) for row in rows], 'value'),
        accounting=accounting, zero_work=zero_work, historical=historical,
        interpretation='A shared error estimate and argmin of shifted Z locate a recorded block; '
            'they do not identify a unique bad node, certify accuracy, or explain the old target failures.',
        completed_utc=core.now())
    core.save(output / 'conclusion.json', conclusion)
    completion = dict(status='complete', registered_cases=12, status_counts=status_counts,
        fixed_input_executability=executable, record_completeness=conclusion['record_completeness'],
        collection_cpu_seconds=time.process_time()-cpu, collection_wall_seconds=time.monotonic()-wall,
        collection_summary_cpu_is_nested=True, automatic_retries=0,
        conclusion_file=str(output / 'conclusion.json'), cases_file=str(output / 'case_status_and_costs.json'),
        completed_utc=core.now())
    core.save(output / 'completion.json', completion)
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=MANIFEST)
    parser.add_argument('--action', choices=('batch', 'collect'), required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=12)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    plan = read(ROOT / args.manifest)
    result = (batch(plan, args.manifest, args.start, args.end, args.workers) if args.action == 'batch'
              else collect(plan, args.manifest))
    print(json.dumps({key: value for key, value in result.items() if key not in ('outcomes',)}), flush=True)
    # Retained scientific failures are sealed outcomes, not scheduler retries.
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
