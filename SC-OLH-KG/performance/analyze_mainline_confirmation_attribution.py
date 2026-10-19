"""Scheduler-only attribution of the sealed Stage65 cohort; saved evidence only."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
DEFAULT = 'performance/manifests/mainline_confirmation_attribution_v1.json'
ZERO = dict(new_source_calls=0, new_target_initial_calls=0, new_target_search_calls=0,
    new_verification_calls=0, new_truth_function_calls=0, new_source_fits=0,
    source_model_restorations=0, posterior_restorations=0, raw_conditioning=0,
    raw_replays=0, new_rng_draws=0)
FULL, AFTER, POINT = 'legacy_role_full', 'legacy_role_after_init_point_only', 'legacy_role_point_only'


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def worker(args):
    manifest, seed = args
    start, wall = time.process_time(), time.monotonic()
    plan = read(ROOT / manifest)
    directory = ROOT / plan['output_dir'] / 'cases' / f'seed_{seed}'
    if (directory / 'analysis.json').is_file():
        return dict(seed=seed, status='reused')
    directory.mkdir(parents=True, exist_ok=True)
    status, error = 'complete', None
    try:
        from performance.mainline_confirmation_query_attribution import analyze_task as query
        from performance.mainline_confirmation_terminal_attribution import analyze_task as terminal
        result = dict(seed=seed, status='complete', query=query(ROOT, plan, seed),
                      terminal=terminal(ROOT, plan, seed))
        if result['query']['reconstruction']['recorded_choices'] != 9:
            raise ValueError('A task must retain all three saved choices in each of three role arms')
        if len(result['terminal']['arms']) != 9:
            raise ValueError('A task must retain all nine terminal controls')
        save(directory / 'analysis.json', result)
    except Exception:
        status, error = 'failed', traceback.format_exc()
    account = dict(seed=seed, status=status, error=error, node=socket.gethostname(),
        utc=datetime.now(timezone.utc).isoformat(), cpu_seconds=time.process_time()-start,
        wall_seconds=time.monotonic()-wall, **ZERO)
    with (directory / 'attempts.jsonl').open('a') as handle:
        handle.write(json.dumps(account) + '\n')
    return account


def contrast(left, right):
    return {k: None if left[k] is None or right[k] is None else left[k]-right[k]
            for k in ('A', 'C', 'false_certificate', 'R', 'G', 'P', 'L')}


def metric_decomposition(row, protocol):
    metric = row[protocol]
    baseline = row['best_initial_quality'] if row['quality_baseline_defined'] else None
    reference = None if baseline is None else metric['C'] * baseline
    if reference is not None and metric['R'] is not None:
        if abs(metric['R'] - (reference + metric['P'] - metric['L'])) > 1e-12:
            raise ValueError('Saved signed quality and certificate/reference contributions disagree')
    return dict(**metric, certificate_reference=reference,
                baseline_defined=baseline is not None)


def deployment_contributions(tasks, queries):
    groups, rows = {}, []
    for task, query in zip(tasks, queries):
        deployed = query['after_new']
        other = deployed['paths']['POINT']
        search, terminal = other['search'], other['terminals']['new']
        gap = ('not_in_saved_candidate_pools' if search['first_candidate_prefix'] is None else
               'candidate_not_observed' if not search['observed_indices'] else
               'observed_not_shortlisted' if terminal['shortlist_position'] is None else
               terminal['disposition'])
        delta = task['AFTER_minus_POINT']
        row = dict(seed=task['seed'], origin=deployed['origin'], POINT_path=gap,
            AFTER_actual_certificate=deployed['certified'], delta=delta)
        rows.append(row)
        for field, value in (('origin', row['origin']), ('POINT_path', gap)):
            key = field + ':' + value
            group = groups.setdefault(key, dict(tasks=0, seeds=[], positive_G_seeds=[],
                negative_G_seeds=[], undefined_G_seeds=[], sum_delta_G=0.,
                sum_positive_delta_G=0., sum_delta_C=0, sum_delta_R=0.))
            group['tasks'] += 1; group['seeds'].append(task['seed'])
            group['sum_delta_C'] += delta['C']; group['sum_delta_R'] += delta['R']
            g = delta['G']
            if g is None:
                group['undefined_G_seeds'].append(task['seed'])
            else:
                group['sum_delta_G'] += g
                group['sum_positive_delta_G'] += max(g,0.)
                if abs(g)>1e-12:
                    group['positive_G_seeds' if g>0 else 'negative_G_seeds'].append(task['seed'])
    positive = sum(max(row['delta']['G'] or 0.,0.) for row in rows)
    for group in groups.values():
        group['share_of_positive_G'] = group['sum_positive_delta_G']/positive if positive else None
    return dict(groups=groups,rows=rows,
        interpretation='Observed paths grouped descriptively; not an exclusive causal mediation estimate')


def collect(plan):
    from performance.mainline_confirmation_query_attribution import compact_task as query_compact
    from performance.mainline_confirmation_query_attribution import aggregate_queries
    from performance.mainline_confirmation_terminal_attribution import compact_task as terminal_compact
    from performance.mainline_confirmation_terminal_attribution import aggregate_terminals
    output = ROOT / plan['output_dir']
    tasks, queries, terminals, attempts, missing = [], [], [], [], []
    reconstruction = dict(recorded_choices=0, max_score_error=0.)
    for seed in plan['target_seeds']:
        directory = output / 'cases' / f'seed_{seed}'
        if (directory / 'attempts.jsonl').is_file():
            attempts += [json.loads(line) for line in (directory/'attempts.jsonl').read_text().splitlines()]
        if not (directory / 'analysis.json').is_file():
            missing.append(seed)
            continue
        result = read(directory / 'analysis.json')
        q, t = result['query'], result['terminal']
        queries.append(query_compact(q)); terminals.append(terminal_compact(t))
        rec = q['reconstruction']
        reconstruction['recorded_choices'] += rec['recorded_choices']
        reconstruction['max_score_error'] = max(reconstruction['max_score_error'],rec['max_score_error'])
        arms = {r['arm']:r for r in t['arms']}
        roles = {}
        for arm in (FULL, AFTER, POINT):
            row = arms[arm]
            roles[arm] = dict(old=metric_decomposition(row,'old'),
                new=metric_decomposition(row,'new'),
                old_deployed=row['protocols']['old']['deployed'],
                new_deployed=row['protocols']['new']['deployed'])
        ap = contrast(arms[AFTER]['new'], arms[POINT]['new'])
        fa = contrast(arms[FULL]['new'], arms[AFTER]['new'])
        tasks.append(dict(seed=seed, domain=t['domain'], roles=roles,
            AFTER_minus_POINT=ap, FULL_minus_AFTER=fa,
            AFTER_terminal_delta=arms[AFTER]['new_minus_old'],
            AFTER_third_contribution=arms[AFTER]['third_challenge'],
            AFTER_prechange_contribution=arms[AFTER]['prechange']))
    terminal_count = sum(len(t['arms']) for t in terminals)
    if not missing and (reconstruction['recorded_choices'] != plan['expected_recorded_choices']
                        or terminal_count != plan['expected_paired_cells']):
        raise ValueError('Attribution dropped registered query choices or terminal pairs')
    source = read(ROOT / plan['source_output_dir'] / 'summary.json')
    # Reuse Stage65 inference and costs; this audit cannot create a new confirmation sample.
    inference = dict(primary=source['primary'], secondary=[{k:v for k,v in r.items() if k!='rows'}
        for r in source['secondary']], controls=source['arms'])
    save(output/'recorded_inference.json', inference)
    save(output/'query_details.json', queries)
    save(output/'terminal_details.json', terminals)
    summary = dict(status='complete' if not missing else 'incomplete',
        source_summary=str(Path(plan['source_output_dir'])/'summary.json'),
        registered_tasks=len(plan['target_seeds']), complete_tasks=len(tasks), missing=missing,
        terminal_pairs=terminal_count, reconstruction=reconstruction,
        tasks=tasks, queries=aggregate_queries(queries), terminals=aggregate_terminals(terminals),
        deployment_contributions=deployment_contributions(tasks,queries),
        scope=plan['scope'], limits=plan['limits'], historical_cost_snapshot=source['costs'],
        final_historical_accounts=plan['historical_accounts'],
        accounting_note='Historical summary predates its final audit; final historical totals use linked completion.json',
        new_work=dict(ZERO), statistical_unit=plan['statistical_unit'])
    save(output/'summary.json', summary)
    dispatch = [read(path) for path in (output/'dispatch').glob('*.json')]
    return dict(status=summary['status'], completed_tasks=len(tasks), missing=missing,
        recorded_choices=reconstruction['recorded_choices'], terminal_pairs=terminal_count,
        worker_cpu_seconds=sum(r['cpu_seconds'] for r in attempts),
        worker_attempt_count=len(attempts), failed_worker_attempts=[r for r in attempts if r['status']=='failed'],
        worker_nodes=sorted({r['node'] for r in attempts}), **ZERO,
        coordinator_cpu_seconds=sum(r['coordinator_cpu_seconds'] for r in dispatch),
        dispatch_count=len(dispatch),
        historical_accounts=plan['historical_accounts'],
        files=['summary.json','completion.json','query_details.json','terminal_details.json',
               'recorded_inference.json'],
        accounting_scope='New audit CPU only; linked prior accounts are not summed again')


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
    if node not in {f'node{i:03d}' for i in range(1,7)}:
        raise RuntimeError('Run saved-cohort analysis through scheduler on node001-node006')
    plan = read(ROOT / args.manifest)
    output = ROOT / plan['output_dir']
    if args.collect:
        result = collect(plan)
        result.update(collector_cpu_seconds=time.process_time()-start,
                      collector_wall_seconds=time.monotonic()-wall, collector_node=node)
        registration = read(output/'registration.json')
        result['registration_cpu_seconds'] = registration['cpu_seconds']
        result['total_new_recorded_cpu_seconds'] = (result['worker_cpu_seconds'] +
            result['collector_cpu_seconds'] + result['registration_cpu_seconds'] +
            result['coordinator_cpu_seconds'])
        save(output/'completion.json',result)
        print(json.dumps({k:result[k] for k in ('status','completed_tasks','recorded_choices',
              'terminal_pairs','failed_worker_attempts','total_new_recorded_cpu_seconds')}),flush=True)
        if result['status'] != 'complete':
            raise SystemExit(1)
    else:
        seeds = plan['target_seeds'][args.start:args.end]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(executor.map(worker,[(args.manifest,s) for s in seeds]))
        batch = dict(start=args.start,end=args.end,workers=args.workers,node=node,results=results,
            coordinator_cpu_seconds=time.process_time()-start,
            wall_seconds=time.monotonic()-wall,**ZERO)
        save(output/'dispatch'/f'{args.start}_{args.end}_{time.time_ns()}.json',batch)
        print(json.dumps(batch),flush=True)
        if any(r['status']=='failed' for r in results):
            raise SystemExit(1)


if __name__ == '__main__':
    main()
