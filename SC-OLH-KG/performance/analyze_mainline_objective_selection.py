"""Scheduler-only source/target objective-selection attribution from saved arrays."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import time
import traceback

from performance.analyze_mainline_three_stream_attribution import ZERO, read, save

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = 'performance/manifests/mainline_objective_selection_attribution_v1.json'


def item_directory(plan, index):
    output = ROOT / plan['output_dir']
    if index == len(plan['target_seeds']):
        return output / 'source'
    return output / 'cases' / f"seed_{plan['target_seeds'][index]}"


def worker(arguments):
    manifest, index = arguments
    start, wall = time.process_time(), time.monotonic()
    plan = read(ROOT / manifest)
    directory = item_directory(plan, index)
    if (directory / 'compact.json').is_file():
        return dict(item=index, status='reused')
    directory.mkdir(parents=True, exist_ok=True)
    status, error = 'complete', None
    try:
        if index == len(plan['target_seeds']):
            from performance.mainline_objective_selection_source import analyze_source, compact_source
            raw = analyze_source(ROOT, plan)
            compact = compact_source(raw)
        else:
            from performance.mainline_objective_selection_target import analyze_task, compact_task
            raw = analyze_task(ROOT, plan, plan['target_seeds'][index])
            compact = compact_task(raw)
            if [s['name'] for s in raw['streams']] != [s['name'] for s in plan['streams']]:
                raise ValueError('Target output omitted or reordered a registered stream')
            if any(len(s['states']) != 9 for s in raw['streams']):
                raise ValueError('A target stream omitted one of its nine saved role choices')
        save(directory / 'analysis.json', raw)
        save(directory / 'compact.json', compact)
    except Exception:
        status, error = 'failed', traceback.format_exc()
    account = dict(item=index, status=status, error=error, node=socket.gethostname(),
        utc=datetime.now(timezone.utc).isoformat(), cpu_seconds=time.process_time()-start,
        wall_seconds=time.monotonic()-wall, **ZERO)
    with (directory / 'attempts.jsonl').open('a') as handle:
        handle.write(json.dumps(account) + '\n')
    return account


def collect(plan):
    from performance.mainline_objective_selection_target import aggregate_tasks
    output = ROOT / plan['output_dir']
    targets, attempts, missing, source = [], [], [], None
    for index in range(plan['expected']['items']):
        directory = item_directory(plan, index)
        ledger = directory / 'attempts.jsonl'
        if ledger.is_file():
            attempts.extend(json.loads(line) for line in ledger.read_text().splitlines())
        if not (directory / 'compact.json').is_file():
            missing.append(index)
            continue
        result = read(directory / 'compact.json')
        if index == len(plan['target_seeds']):
            source = result
        else:
            targets.append(result)
    target_summary = aggregate_tasks(targets)
    if not missing:
        if target_summary['coverage']['states'] != plan['expected']['states']:
            raise ValueError('Target summary dropped a registered saved choice')
        for key in ('source_groups', 'arm_cases', 'snapshots'):
            expected_key = {'arm_cases': 'source_arm_cases', 'snapshots': 'source_snapshots'}.get(key, key)
            if source['coverage'][key] != plan['expected'][expected_key]:
                raise ValueError(f'Source summary dropped registered {key}')
    save(output / 'all_target_details.json', targets)
    save(output / 'focus_details.json', [r for r in targets if r['seed'] in plan['focus_seeds']])
    summary = dict(status='complete' if not missing else 'incomplete', missing_items=missing,
        targets=target_summary, source=source, statistical_unit=plan['statistical_unit'],
        limits=plan['limits'], all_control_evidence=plan['all_control_evidence'],
        recorded_inference_files=plan['recorded_inference_files'],
        historical_accounts=plan['historical_accounts'], new_work=dict(ZERO))
    save(output / 'summary.json', summary)
    dispatches = [read(p) for p in (output / 'dispatch').glob('*.json')]
    validation = [dict(file=str(p.relative_to(ROOT)), **read(p))
                  for p in (output / 'validation').glob('*.json')]
    return dict(status=summary['status'], completed_target_geometries=len(targets),
        source_complete=source is not None, missing_items=missing, worker_attempts=attempts,
        failed_worker_attempts=[r for r in attempts if r['status'] == 'failed'],
        worker_cpu_seconds=sum(r['cpu_seconds'] for r in attempts),
        coordinator_cpu_seconds=sum(r['coordinator_cpu_seconds'] for r in dispatches),
        registration_cpu_seconds=read(output / 'registration.json')['cpu_seconds'],
        validation=validation, measured_validation_cpu_seconds=sum(r.get('cpu_seconds') or 0. for r in validation),
        unmetered_validation=[r['file'] for r in validation if r.get('cpu_seconds') is None],
        historical_accounts=plan['historical_accounts'], new_work=dict(ZERO),
        accounting_scope='New recorded analysis and fixture CPU only; historical accounts remain linked, not charged again.')


def main():
    start, wall = time.process_time(), time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=DEFAULT)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--collect', action='store_true')
    args = parser.parse_args()
    node = socket.gethostname().split('.')[0]
    if node not in {f'node{i:03d}' for i in range(1, 7)}:
        raise RuntimeError('Use scheduler on node001-node006 for saved scientific inputs')
    plan = read(ROOT / args.manifest)
    output = ROOT / plan['output_dir']
    if args.collect:
        result = collect(plan)
        result.update(collector_cpu_seconds=time.process_time()-start,
                      collector_wall_seconds=time.monotonic()-wall, collector_node=node)
        result['total_new_recorded_cpu_seconds'] = sum(result[key] for key in (
            'worker_cpu_seconds', 'coordinator_cpu_seconds', 'collector_cpu_seconds',
            'registration_cpu_seconds', 'measured_validation_cpu_seconds'))
        save(output / 'completion.json', result)
        if result['status'] == 'complete':
            plan.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
                        summary_file=str(Path(plan['output_dir']) / 'summary.json'))
            save(ROOT / args.manifest, plan)
        print(json.dumps({k:result[k] for k in ('status', 'completed_target_geometries',
            'source_complete', 'missing_items', 'failed_worker_attempts',
            'total_new_recorded_cpu_seconds')}), flush=True)
        if result['status'] != 'complete':
            raise SystemExit(1)
    else:
        indices = range(plan['expected']['items'])[args.start:args.end]
        results = []
        def progress():
            print('SCHEDULEURM_PROGRESS ' + json.dumps(dict(current=len(results), total=len(indices),
                unit='saved_input_item', elapsed_seconds=time.monotonic()-wall)), flush=True)
        progress()
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            pending = [executor.submit(worker, (args.manifest, index)) for index in indices]
            for future in as_completed(pending):
                results.append(future.result())
                progress()
        save(output / 'dispatch' / f'{args.start}_{args.end}_{time.time_ns()}.json',
             dict(node=node, start=args.start, end=args.end, workers=args.workers, results=results,
                  coordinator_cpu_seconds=time.process_time()-start, wall_seconds=time.monotonic()-wall))
        if any(r['status'] == 'failed' for r in results):
            raise SystemExit(1)


if __name__ == '__main__':
    main()
