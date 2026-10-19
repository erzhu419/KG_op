"""Scheduler-only, zero-call attribution of three already sealed observation streams."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import time
import traceback

from performance.analyze_mainline_confirmation_attribution import (
    AFTER, FULL, POINT, ZERO, contrast, deployment_contributions, read,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = 'performance/manifests/mainline_three_stream_attribution_v1.json'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n')


def worker(args):
    manifest, seed = args
    start, wall = time.process_time(), time.monotonic()
    plan = read(ROOT / manifest)
    directory = ROOT / plan['output_dir'] / 'cases' / f'seed_{seed}'
    if (directory / 'compact.json').is_file():
        return dict(seed=seed, status='reused')
    directory.mkdir(parents=True, exist_ok=True)
    status, error = 'complete', None
    try:
        from performance import mainline_three_stream_query_attribution as query
        from performance import mainline_three_stream_terminal_attribution as terminal
        registry = {}
        q = query.analyze_task(ROOT, plan, seed, policy_registry=registry)
        t = terminal.analyze_task(ROOT, plan, seed, policy_registry=registry)
        if [s['name'] for s in q['streams']] != [s['name'] for s in plan['streams']]:
            raise ValueError('Query attribution omitted or reordered a registered stream')
        if [s['name'] for s in t['streams']] != [s['name'] for s in plan['streams']]:
            raise ValueError('Terminal attribution omitted or reordered a registered stream')
        if any(s['query']['reconstruction']['recorded_choices'] != 9 for s in q['streams']):
            raise ValueError('Each stream must retain all nine role queries')
        if any(len(s['terminal']['arms']) != 9 for s in t['streams']):
            raise ValueError('Each stream must retain all nine terminal controls')
        save(directory / 'analysis.json', dict(seed=seed, query=q, terminal=t))
        save(directory / 'policy_vectors.json', [dict(policy_id=value, point=key)
                                               for key, value in registry.items()])
        save(directory / 'compact.json', dict(seed=seed, query=query.compact_task(q),
                                              terminal=terminal.compact_task(t)))
    except Exception:
        status, error = 'failed', traceback.format_exc()
    account = dict(seed=seed, status=status, error=error, node=socket.gethostname(),
        utc=datetime.now(timezone.utc).isoformat(), cpu_seconds=time.process_time()-start,
        wall_seconds=time.monotonic()-wall, **ZERO)
    with (directory / 'attempts.jsonl').open('a') as handle:
        handle.write(json.dumps(account) + '\n')
    return account


def collect(plan):
    from performance import mainline_three_stream_query_attribution as query
    from performance import mainline_three_stream_terminal_attribution as terminal
    output = ROOT / plan['output_dir']
    results, attempts, missing = [], [], []
    for seed in plan['target_seeds']:
        directory = output / 'cases' / f'seed_{seed}'
        ledger = directory / 'attempts.jsonl'
        if ledger.is_file():
            attempts.extend(json.loads(line) for line in ledger.read_text().splitlines())
        if (directory / 'compact.json').is_file():
            results.append(read(directory / 'compact.json'))
        else:
            missing.append(seed)
    queries = query.aggregate_queries([r['query'] for r in results])
    terminals = terminal.aggregate_terminals([r['terminal'] for r in results])
    expected = plan['expected']
    if not missing and (queries['reused_choices'] != expected['reused_original_choices'] or
            queries['newly_reconstructed_choices'] != expected['newly_reconstructed_saved_choices'] or
            terminals['registered_pairs'] != expected['terminal_pairs']):
        raise ValueError('Three-stream aggregation dropped registered choices or controls')
    paths, effects = [], []
    for descriptor in plan['streams']:
        name = descriptor['name']
        tasks, compact_queries = [], []
        for result in results:
            q = next(s['query'] for s in result['query']['streams'] if s['name'] == name)
            t = next(s['terminal'] for s in result['terminal']['streams'] if s['name'] == name)
            arms = {r['arm']: r for r in t['arms']}
            row = dict(seed=result['seed'], domain=t['domain'],
                AFTER_minus_POINT=contrast(arms[AFTER]['new'], arms[POINT]['new']),
                FULL_minus_AFTER=contrast(arms[FULL]['new'], arms[AFTER]['new']),
                AFTER_terminal_delta=arms[AFTER]['new_minus_old'])
            tasks.append(row)
            compact_queries.append(q)
        paths.append(dict(name=name, **deployment_contributions(tasks, compact_queries)))
        effects.append(dict(name=name, tasks=tasks))
    focal = [r for r in results if r['seed'] in plan['focus_seeds']]
    save(output / 'focus_details.json', focal)
    save(output / 'all_task_details.json', results)
    # Large focal exposure trees live once in focus_details; the summary keeps small counts.
    terminal_summary = {k: v for k, v in terminals.items() if k != 'focal_exposure'}
    terminal_summary['focal_exposure_counts'] = [dict(seed=r['seed'], policies=len(r['policies']))
                                               for r in terminals['focal_exposure']]
    summary = dict(status='complete' if not missing else 'incomplete',
        registered_geometries=24, complete_geometries=len(results), missing=missing,
        queries=queries, terminals=terminal_summary, deployment_contributions=paths,
        effects=effects, statistical_unit=plan['statistical_unit'], limits=plan['limits'],
        recorded_inference_files=plan['recorded_inference_files'],
        historical_accounts=plan['historical_accounts'], new_work=dict(ZERO),
        files=dict(focus='focus_details.json', complete_details='all_task_details.json'))
    save(output / 'summary.json', summary)
    dispatches = [read(p) for p in (output / 'dispatch').glob('*.json')]
    validation = [dict(file=str(p.relative_to(ROOT)), **read(p))
                  for p in (output / 'validation').glob('*.json')]
    return dict(status=summary['status'], completed_geometries=len(results), missing=missing,
        reused_original_choices=queries['reused_choices'],
        newly_reconstructed_choices=queries['newly_reconstructed_choices'],
        terminal_pairs=terminals['registered_pairs'], worker_attempts=attempts,
        failed_worker_attempts=[r for r in attempts if r['status'] == 'failed'],
        worker_cpu_seconds=sum(r['cpu_seconds'] for r in attempts),
        coordinator_cpu_seconds=sum(r['coordinator_cpu_seconds'] for r in dispatches),
        registration_cpu_seconds=read(output / 'registration.json')['cpu_seconds'],
        validation=validation, measured_validation_cpu_seconds=sum(
            r.get('cpu_seconds') or 0. for r in validation),
        unmetered_validation=[r['file'] for r in validation if r.get('cpu_seconds') is None],
        historical_accounts=plan['historical_accounts'], new_work=dict(ZERO),
        accounting_scope='New recorded analysis CPU only. Linked historical accounts are not added again.')


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
        raise RuntimeError('Run saved-trajectory analysis through scheduler on node001-node006')
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
        print(json.dumps({k: result[k] for k in ('status', 'completed_geometries',
            'reused_original_choices', 'newly_reconstructed_choices', 'terminal_pairs',
            'failed_worker_attempts', 'total_new_recorded_cpu_seconds')}), flush=True)
        if result['status'] != 'complete':
            raise SystemExit(1)
    else:
        seeds = plan['target_seeds'][args.start:args.end]
        results = []
        def progress():
            print('SCHEDULEURM_PROGRESS ' + json.dumps(dict(current=len(results), total=len(seeds),
                unit='geometry', elapsed_seconds=time.monotonic()-wall)), flush=True)
        progress()
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            pending = [executor.submit(worker, (args.manifest, seed)) for seed in seeds]
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
