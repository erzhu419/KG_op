"""Scheduler-only attribution from saved observations, scores and terminal records."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
DEFAULT = 'performance/manifests/mainline_observed_terminal_attribution_v1.json'
ZERO = {key: 0 for key in (
    'new_source_calls', 'new_target_initial_calls', 'new_target_search_calls',
    'new_verification_calls', 'new_truth_function_calls', 'new_source_fits',
    'source_model_restorations', 'engine_restorations', 'posterior_restorations',
    'new_raw_replays', 'new_rng_draws', 'download_bytes')}


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, separators=(',', ':'), allow_nan=False)+'\n')


def worker(args):
    manifest, seed = args
    start, wall = time.process_time(), time.monotonic()
    plan = read(ROOT/manifest)
    output = ROOT/plan['output_dir']/'cases'/f'seed_{seed}'
    account_path = output/'attempt.json'
    if account_path.exists():
        return dict(read(account_path), reused=True)
    output.mkdir(parents=True, exist_ok=True)
    status, error = 'complete', None
    try:
        from performance import mainline_observed_shortlist_attribution as shortlist
        from performance import mainline_saved_terminal_comparison as comparison
        registry = {}
        s = shortlist.analyze_task(ROOT, plan, seed, policy_registry=registry)
        t = comparison.analyze_task(ROOT, plan, seed, policy_registry=registry)
        save(output/'analysis.json', dict(seed=seed, shortlists=s, comparisons=t))
        save(output/'policy_vectors.json', [dict(policy_id=value, point=point)
                                           for point, value in registry.items()])
        save(output/'compact.json', dict(seed=seed, shortlists=shortlist.compact_task(s),
                                        comparisons=comparison.compact_task(t)))
    except Exception:
        status, error = 'failed', traceback.format_exc()
    account = dict(seed=seed, status=status, error=error, node=socket.gethostname(),
        utc=datetime.now(timezone.utc).isoformat(), cpu_seconds=time.process_time()-start,
        wall_seconds=time.monotonic()-wall, **ZERO)
    save(account_path, account)
    return account


def collect(plan):
    from performance import mainline_observed_shortlist_attribution as shortlist
    from performance import mainline_saved_terminal_comparison as comparison
    output = ROOT/plan['output_dir']
    results, attempts, missing = [], [], []
    for seed in plan['target_seeds']:
        directory = output/'cases'/f'seed_{seed}'
        if (directory/'attempt.json').exists():
            attempts.append(read(directory/'attempt.json'))
        if (directory/'compact.json').exists():
            results.append(read(directory/'compact.json'))
        else:
            missing.append(seed)
    summary = dict(status='complete' if not missing else 'incomplete',
        registered_geometries=len(plan['target_seeds']), completed_geometries=len(results),
        missing=missing, shortlists=shortlist.aggregate_tasks([r['shortlists'] for r in results]),
        comparisons=comparison.aggregate_tasks([r['comparisons'] for r in results]),
        statistical_unit=plan['statistical_unit'], limits=plan['limits'],
        all_control_evidence=plan['all_control_evidence'],
        historical_accounts=plan['historical_accounts'], new_work=dict(ZERO))
    save(output/'summary.json', summary)
    save(output/'all_task_details.json', results)
    save(output/'focus_details.json', [r for r in results if r['seed'] in plan['focus_seeds']])
    dispatches = [read(p) for p in (output/'dispatch').glob('*.json')]
    validation = [dict(file=str(p.relative_to(ROOT)), **read(p))
                  for p in (output/'validation').glob('*.json')]
    return dict(status=summary['status'], completed_geometries=len(results), missing=missing,
        attempts=attempts, failed_attempts=[a for a in attempts if a['status'] != 'complete'],
        analysis_cpu_seconds=sum(a['cpu_seconds'] for a in attempts),
        coordinator_cpu_seconds=sum(d['cpu_seconds'] for d in dispatches),
        validation=validation, validation_cpu_seconds=sum(v.get('cpu_seconds') or 0 for v in validation),
        historical_accounts=plan['historical_accounts'], new_work=dict(ZERO),
        accounting_scope='New saved-record analysis and targeted tests; earlier paid calls remain linked and are not added again.')


def main():
    start, wall = time.process_time(), time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=DEFAULT)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=24)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--collect', action='store_true')
    args = parser.parse_args()
    node = socket.gethostname().split('.')[0]
    if node not in {f'node{i:03d}' for i in range(1, 7)}:
        raise RuntimeError('Run saved-record attribution through scheduler on node001-node006')
    plan = read(ROOT/args.manifest)
    output = ROOT/plan['output_dir']
    seal = read(ROOT/plan['cohort_dir']/'execution_status.json')
    expected = {(seed, arm['name']) for seed in plan['target_seeds'] for arm in plan['terminal_arms']}
    actual = {(case['seed'], case['arm']) for case in seal['cases']}
    if seal['status'] != 'sealed' or actual != expected or len(seal['cases']) != len(expected):
        raise ValueError('Retain the sealed full24 by9 terminal cohort')
    if args.collect:
        result = collect(plan)
        result.update(collector_cpu_seconds=time.process_time()-start,
                      collector_wall_seconds=time.monotonic()-wall, node=node)
        result['total_new_recorded_cpu_seconds'] = sum(result[key] for key in (
            'analysis_cpu_seconds', 'coordinator_cpu_seconds', 'validation_cpu_seconds',
            'collector_cpu_seconds'))
        save(output/'completion.json', result)
        if result['status'] == 'complete':
            plan.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat())
            save(ROOT/args.manifest, plan)
        print(json.dumps({key: result[key] for key in ('status', 'completed_geometries',
            'missing', 'total_new_recorded_cpu_seconds')}), flush=True)
    else:
        seeds = plan['target_seeds'][args.start:args.end]
        if not seeds or args.workers < 1:
            raise ValueError('Registered slice and positive worker count required')
        outcomes = []
        with ProcessPoolExecutor(max_workers=min(args.workers, len(seeds))) as executor:
            pending = [executor.submit(worker, (args.manifest, seed)) for seed in seeds]
            for future in as_completed(pending):
                outcomes.append(future.result())
                print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=len(outcomes),
                    total=len(seeds), unit='geometry')), flush=True)
        save(output/'dispatch'/f'{args.start}_{args.end}_{time.time_ns()}.json',
            dict(node=node, outcomes=outcomes, cpu_seconds=time.process_time()-start,
                 wall_seconds=time.monotonic()-wall, **ZERO))
        print(json.dumps(dict(completed=len(outcomes), failed=sum(
            row['status'] != 'complete' for row in outcomes))), flush=True)
        # A recorded analysis failure stays visible; scheduler retries must not replace it.


if __name__ == '__main__':
    main()
