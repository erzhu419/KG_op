"""Two new observation streams: score after a joint seal, then average by geometry."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import socket

from performance import summarize_mainline_two_comparison_confirmation as single
from performance.prepare_mainline_two_comparison_terminal import ZERO
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save
from performance.run_mainline_two_comparison_confirmation import finalize_accounting
from performance.summarize_mainline_two_comparison_development import (
    AFTER, FULL, POINT, FIELDS, METRICS, difference, summarize_rows,
)


MANIFEST = ROOT / 'performance/manifests/mainline_terminal_two_comparison_seed_replication_v1.json'
ORIGINS = ('initial', 'shared', 'dynamic')


def children(plan):
    return [(descriptor, read(ROOT / descriptor['manifest'])) for descriptor in plan['streams']]


def preload_truth(child, previous_plans):
    """Reuse exact-policy truth for identical frozen geometry, only after joint sealing."""
    directory = ROOT / child['output_dir']
    with attempt(directory / 'truth_cache_reuse', 'reuse-sealed-same-geometry-truth') as cost:
        cost.update(ZERO)
        current = read(directory / 'execution_status.json')
        current_directories = {row['seed']: Path(row['source_case']).parent for row in current['cases']
                               if row.get('source_case')}
        prior = []
        for plan in previous_plans:
            seal = read(ROOT / plan['output_dir'] / 'execution_status.json')
            paths = {row['seed']: Path(row['source_case']).parent for row in seal['cases']
                     if row.get('source_case')}
            prior.append((plan, {row['geometry_task_seed']: row for row in plan['tasks']}, paths))
        rows = []
        for task in child['tasks']:
            seed = task['geometry_task_seed']
            if seed not in current_directories:
                rows.append(dict(seed=seed, status='no_available_search_directory'))
                continue
            target = ROOT / current_directories[seed]
            if not (target / 'initialization.json').is_file():
                rows.append(dict(seed=seed, status='initialization_unavailable'))
                continue
            initial = [row['point'] for row in read(target / 'initialization.json')['history']]
            cache_path = target / 'truth.json'
            cache = {tuple(row['point']): row['truth'] for row in read(cache_path)} if cache_path.is_file() else {}
            before, provenance = len(cache), []
            for previous, tasks, paths in prior:
                old_task = tasks[seed]
                if any(task[key] != old_task[key] for key in ('domain', 'geometry', 'optimizer_seed')):
                    raise ValueError('Truth reuse requires identical geometry and optimizer seed')
                if seed not in paths:
                    continue
                old_directory = ROOT / paths[seed]
                old_cache = old_directory / 'truth.json'
                if not old_cache.is_file():
                    continue
                if initial != [row['point'] for row in read(old_directory / 'initialization.json')['history']]:
                    raise ValueError('Truth reuse requires identical initial physical policies')
                added = 0
                for row in read(old_cache):
                    point = tuple(row['point'])
                    if point not in cache:
                        cache[point] = row['truth']
                        added += 1
                provenance.append(dict(source_file=str(old_cache.relative_to(ROOT)), added_policies=added))
            save(cache_path, [dict(point=list(point), truth=value) for point, value in cache.items()])
            rows.append(dict(seed=seed, status='reused', before_policies=before, after_policies=len(cache),
                added_policies=len(cache) - before, target_file=str(cache_path.relative_to(ROOT)), sources=provenance))
        result = dict(tasks=rows, added_policies=sum(row.get('added_policies', 0) for row in rows),
                      new_truth_function_calls=0)
        save(directory / 'truth_cache_reuse.json', result)
    return result


def score_streams(plan):
    """No child may open truth until every new old/new terminal is sealed."""
    streams = children(plan)
    # Validate both complete file sets first; a missing second stream blocks first-stream truth.
    with attempt(ROOT / plan['output_dir'] / 'score_barrier', 'validate-both-new-terminal-seals') as cost:
        cost.update(ZERO)
        for _, child in streams:
            single.load_sealed(child)
    previous = [read(ROOT / plan['reference_manifest'])]
    results = []
    for descriptor, child in streams:
        output = ROOT / child['output_dir']
        reused = (output / 'summary.json').is_file()
        if not reused:
            preload_truth(child, previous)
        scored = read(output / 'summary.json') if reused else single.score(child)
        scored.update(scope=child['scope'], statistical_unit=child['statistical_unit'], limits=[
            'This child is one new observation/verification stream on the 24 fixed Stage65 geometries, not a fresh-geometry cohort.',
            'The parent analysis averages both required new streams within each geometry; stream and arm cells are not independent geometry samples.',
            'The opened Stage65 stream is a separate descriptive reference and is excluded from new-stream estimates.',
            'Missing executions and undefined initial-policy baselines remain null; available-subset estimates retain their explicit sample counts.',
            'Largest-positive-geometry removal is descriptive sensitivity; all registered geometries remain in the primary estimate.',
            'Paired old/new terminal cells share a search history; terminal conversion is not new search learning.'])
        save(output / 'summary.json', scored)
        if not (output / 'completion.json').is_file():
            finalize_accounting(child, ROOT / descriptor['manifest'])
        results.append(dict(name=descriptor['name'], replicate=descriptor['replicate'],
            status=scored['status'], reused_summary=reused,
            summary_file=str(output.relative_to(ROOT) / 'summary.json'),
            completion_file=str(output.relative_to(ROOT) / 'completion.json')))
        previous.append(child)
    return results


def origin(label):
    if label is None:
        return None
    if label.startswith('initial:'):
        return 'initial'
    if label.startswith('neighbor:'):
        return 'shared'
    if ':history:' in label:
        return 'dynamic'
    raise ValueError(f'Unrecognized saved deployment label: {label}')


def origin_rows(rows):
    """Allocate saved outcomes by actual deployment origin; do not invent certificates."""
    result = []
    for source in rows:
        row = dict(source)
        values = {name: {} for name in ORIGINS}
        for field in ('old', 'new', 'B'):
            label = source.get(field + '_reported_deployed', source.get(field + '_deployed'))
            selected = origin(label)
            for name in ORIGINS:
                values[name][field] = {metric: None if value is None else value if selected == name else 0.
                                      for metric, value in source[field].items()}
        for name in ORIGINS:
            fields = values[name]
            fields['new_minus_old'] = difference(fields['new'], fields['old'])
            fields['third_challenge'] = difference(fields['new'], fields['B'])
            fields['prechange'] = difference(fields['B'], fields['old'])
        row['origin_contributions'] = values
        result.append(row)
    return result


def mean_known(values):
    """A geometry's two-stream mean is unavailable if either required stream is missing."""
    return None if any(value is None for value in values) else sum(values) / len(values)


def geometry_average(stream_rows):
    """Exactly one row per geometry and arm, with paired streams averaged first."""
    indexed = [{(row['seed'], row['arm']): row for row in rows} for rows in stream_rows]
    if len(indexed) != 2 or set(indexed[0]) != set(indexed[1]):
        raise ValueError('The two registered streams must retain the same geometry-arm grid')
    result = []
    for key in indexed[0]:
        source = [index[key] for index in indexed]
        anchor = source[0]
        row = {name: anchor[name] for name in ('seed', 'domain', 'arm', 'history_arm')}
        row.update(status='complete' if all(s['status'] == 'complete' for s in source) else 'incomplete_pair',
            stream_count=2, complete_stream_count=sum(s['status'] == 'complete' for s in source),
            quality_baseline_defined=all(s['quality_baseline_defined'] for s in source),
            quality_reference_status=[s['quality_reference_status'] for s in source],
            best_initial_quality=mean_known([s['best_initial_quality'] for s in source]))
        for field in FIELDS:
            row[field] = {metric: mean_known([s[field][metric] for s in source]) for metric in METRICS}
        row['origin_contributions'] = {name: {field: {
            metric: mean_known([s['origin_contributions'][name][field][metric] for s in source])
            for metric in METRICS} for field in FIELDS} for name in ORIGINS}
        result.append(row)
    return result


def origin_summary(rows, arms):
    output = []
    for name in ORIGINS:
        selected = [dict(seed=row['seed'], domain=row['domain'], arm=row['arm'], status=row['status'],
                         **row['origin_contributions'][name]) for row in rows]
        groups, secondary = summarize_rows(selected, arms)
        single.add_sensitivity(groups, secondary, selected)
        output.append(dict(origin=name, arms=groups, secondary=secondary))
    return output


def summarize_unit(rows, arms):
    groups, secondary = summarize_rows(rows, arms)
    single.add_sensitivity(groups, secondary, rows)
    return dict(registered_geometries=len({row['seed'] for row in rows}),
        rows=rows, arms=groups, primary=next(row for row in groups if row['arm'] == AFTER),
        secondary=secondary, origins=origin_summary(rows, arms))


def sign(value):
    return 'unknown' if value is None else 'positive' if value > 1e-12 else 'negative' if value < -1e-12 else 'zero'


def replication_directions(stream_rows):
    indexed = [{(row['seed'], row['arm']): row for row in rows} for rows in stream_rows]
    contrasts = []
    for name, left, right in (('AFTER_new_minus_old', AFTER, None),
                              ('AFTER_minus_POINT', AFTER, POINT), ('FULL_minus_AFTER', FULL, AFTER)):
        rows = []
        for seed in dict.fromkeys(row['seed'] for row in stream_rows[0]):
            effects = [index[seed, left]['new_minus_old'] if right is None else
                difference(index[seed, left]['new'], index[seed, right]['new']) for index in indexed]
            rows.append(dict(seed=seed, domain=indexed[0][seed, left]['domain'],
                stream_effects=effects, directions={metric: [sign(effect[metric]) for effect in effects]
                                                   for metric in ('C', 'R', 'G')}))
        contrasts.append(dict(contrast=name, rows=rows,
            direction_counts={metric: dict(Counter('/'.join(row['directions'][metric]) for row in rows))
                              for metric in ('C', 'R', 'G')}))
    return contrasts


def _validate_reference(rows, reference):
    old = {(row['seed'], row['arm']): row for row in reference['rows']}
    if {(row['seed'], row['arm']) for row in rows} != set(old):
        raise ValueError('New streams and the frozen reference must keep the same geometry-arm grid')
    for row in rows:
        before = old[row['seed'], row['arm']]
        if row['quality_reference_status'] == 'initial_reference_unavailable':
            continue
        if row['quality_reference_status'] != before['quality_reference_status'] or (
            row['best_initial_quality'] is not None and
            abs(row['best_initial_quality'] - before['best_initial_quality']) > 1e-12):
            raise ValueError('A new stream changed the frozen true-feasible initial-policy baseline')


def compact_estimate(estimate):
    known = estimate['conditional_available']
    result = dict(n=known['n'], mean=known['mean'], ci95=known['descriptive_95pct_t_interval'],
        positive=known['positive'], zero=known['zero'], negative=known['negative'],
        largest_positive_share=known['largest_positive_share'])
    sensitivity = estimate.get('leave_largest_positive_geometry_out', {})
    if sensitivity.get('excluded_seed') is not None:
        result['drop_largest_positive'] = dict(seed=sensitivity['excluded_seed'],
            n=sensitivity['remaining_available'], mean=sensitivity['conditional_available_leave_one_out_mean'])
    return result


def compact_unit(unit):
    fields, metrics = ('old', 'new', 'new_minus_old'), ('C', 'R', 'G')
    arms = []
    for group in unit['arms']:
        arm = dict(arm=group['arm'], **{field: {
            metric: compact_estimate(group[field][metric]) for metric in metrics} for field in fields})
        arm['false_certificate_totals'] = {field: group[field]['false_certificate'][
            'conditional_available']['positive_sum'] for field in ('old', 'new')}
        if group['arm'] == AFTER:
            arm.update({field: {metric: compact_estimate(group[field][metric]) for metric in metrics}
                        for field in ('third_challenge', 'prechange')})
        arms.append(arm)
    secondary = [dict(left=group['left'], right=group['right'], **{field: {
        metric: compact_estimate(group[field][metric]) for metric in metrics}
        for field in ('old_contrast', 'new_contrast', 'change')}) for group in unit['secondary']]
    indexed = {(row['seed'], row['arm']): row for row in unit['rows']}
    tasks = []
    for seed in dict.fromkeys(row['seed'] for row in unit['rows']):
        after, point, full = (indexed[seed, arm] for arm in (AFTER, POINT, FULL))
        task = dict(seed=seed, domain=after['domain'])
        for name, left, right in (('primary', after, None), ('AFTER_minus_POINT', after, point),
                                  ('FULL_minus_AFTER', full, after)):
            values = left['new_minus_old'] if right is None else difference(left['new'], right['new'])
            by_origin = {origin: (left['origin_contributions'][origin]['new_minus_old']['G']
                if right is None else difference(left['origin_contributions'][origin]['new'],
                    right['origin_contributions'][origin]['new'])['G']) for origin in ORIGINS}
            task[name] = dict(**{metric: values[metric] for metric in metrics}, origin_G=by_origin)
        tasks.append(task)
    origins = []
    for group in unit['arms']:
        rows = [row for row in unit['rows'] if row['arm'] == group['arm']]
        origin_record = dict(arm=group['arm'])
        for name in ORIGINS:
            fields_by_origin = {}
            for field in fields:
                values = [row['origin_contributions'][name][field]['G'] for row in rows]
                known = [value for value in values if value is not None]
                fields_by_origin[field + '_G'] = dict(n=len(known), sum=sum(known),
                    mean=sum(known) / len(known) if known else None)
            for protocol in ('old', 'new'):
                certificates = [row['origin_contributions'][name][protocol]['C'] for row in rows]
                fields_by_origin[protocol + '_true_feasible_deployments'] = dict(
                    n=sum(value is not None for value in certificates),
                    total=sum(value for value in certificates if value is not None))
            origin_record[name] = fields_by_origin
        origins.append(origin_record)
    return dict(registered_geometries=unit['registered_geometries'], arms=arms,
                secondary=secondary, origins=origins, tasks=tasks)


def compact_summary(result):
    streams = [dict(name=stream['name'], replicate=stream['replicate'], status=stream['status'],
                    **compact_unit(stream)) for stream in result['streams']]
    false_certificates = [dict(stream=stream['name'], seed=row['seed'], domain=row['domain'],
        arm=row['arm'], protocol=protocol, policy=row.get(protocol + '_reported_deployed'))
        for stream in result['streams'] for row in stream['rows'] for protocol in ('old', 'new')
        if row[protocol]['false_certificate'] is not None and row[protocol]['false_certificate'] > 0]
    failures = [{key: row.get(key) for key in ('stream', 'seed', 'domain', 'arm', 'protocol',
                'status', 'reason', 'files')} for row in result['failures']]
    accounts = [{key: account[key] for key in ('name', 'replicate', 'completion_file',
        'truth_cache_reuse_file', 'recorded_cpu_seconds', 'new_work', 'new_physical_target_calls', 'terminal_costs')}
        | dict(failed_attempt_count=len(account['failed_attempts']),
               execution_failures=account['execution_failures']) for account in result['costs']['streams']]
    return dict(status=result['status'], statistical_unit=result['statistical_unit'],
        estimates='mean and ci95 use explicitly counted available geometries; positive/zero/negative are signed-value counts. Geometry-averaged deployment totals can be fractional.',
        primary_arm=AFTER, streams=streams, geometry_averaged=compact_unit(result['geometry_averaged']),
        replication_directions=[dict(contrast=row['contrast'], direction_counts=row['direction_counts'])
                                for row in result['replication_directions']],
        failures=failures, false_certificates=false_certificates, costs=dict(streams=accounts),
        reference={key: result['reference'][key] for key in ('role', 'summary_file', 'completion_file')},
        detailed_summary_file='summary.json', completion_file='completion.json', limits=result['limits'])


def summarize(plan):
    output = ROOT / plan['output_dir']
    streams = children(plan)
    with attempt(output / 'aggregation', 'aggregate-two-new-streams-by-geometry') as cost:
        cost.update(ZERO)
        reference_plan = read(ROOT / plan['reference_manifest'])
        reference_output = ROOT / reference_plan['output_dir']
        reference = read(reference_output / 'summary.json')
        arms = single.arm_names(streams[0][1])
        per_stream, stream_rows, accounts, failures = [], [], [], []
        for descriptor, child in streams:
            directory = ROOT / child['output_dir']
            summary, account = read(directory / 'summary.json'), read(directory / 'completion.json')
            rows = origin_rows(summary['rows'])
            _validate_reference(rows, reference)
            stream_rows.append(rows)
            per_stream.append(dict(name=descriptor['name'], replicate=descriptor['replicate'],
                status=summary['status'], completed_outcome_cells=summary['completed_outcome_cells'],
                **summarize_unit(rows, arms)))
            accounts.append(dict(name=descriptor['name'], replicate=descriptor['replicate'],
                completion_file=str(directory.relative_to(ROOT) / 'completion.json'),
                truth_cache_reuse_file=str(directory.relative_to(ROOT) / 'truth_cache_reuse.json'),
                **{key: account[key] for key in ('recorded_cpu_seconds', 'new_work',
                    'new_physical_target_calls', 'terminal_costs', 'failed_attempts', 'execution_failures')}))
            failures.extend(dict(stream=descriptor['name'], **row) for row in summary['outcomes']
                            if row['status'] != 'complete')
        averaged = geometry_average(stream_rows)
        result = dict(status='complete' if not failures else 'complete_with_missing_cells',
            statistical_unit='24 fixed geometries; average the two new streams within each geometry before inference.',
            geometry_averaged=summarize_unit(averaged, arms), streams=per_stream,
            replication_directions=replication_directions(stream_rows), failures=failures,
            reference=dict(role='Opened Stage65 stream, descriptive reference only; excluded from all new means.',
                summary_file=str(reference_output.relative_to(ROOT) / 'summary.json'),
                completion_file=str(reference_output.relative_to(ROOT) / 'completion.json'),
                primary=reference['primary'], secondary=reference['secondary'], arms=reference['arms']),
            costs=dict(streams=accounts), limits=[
                'The repeated streams test observation and verification noise on fixed geometries, not new task geometry generalization.',
                'The 48 stream-geometry observations are not 48 independent geometries; the old stream is not pooled.',
                'Unavailable outcomes and undefined initial baselines stay null. Missing one stream makes its geometry-mean metric unknown.',
                'Origin contributions are deployment accounting, not interventions or causal effects of candidate origin.',
                'Initial/neighbor/history labels mean fixed initial/shared candidate/dynamically generated policies within each task.'])
        save(output / 'summary.json', result)
        compact = compact_summary(result)
    # Parent attempt ledgers exclude child trees: their final accounts already charge them once.
    child_outputs = [ROOT / child['output_dir'] for _, child in streams]
    parent_attempts = []
    ledger_paths = list(output.rglob('attempts.jsonl'))
    amendment_file = output / 'registration' / 'metadata_amendments.jsonl'
    if amendment_file.is_file():
        ledger_paths.append(amendment_file)
    for path in sorted(ledger_paths):
        if any(path.is_relative_to(directory) for directory in child_outputs):
            continue
        parent_attempts.extend(dict(json.loads(line), ledger=str(path.relative_to(ROOT)))
                               for line in path.read_text().splitlines() if line.strip())
    keys = set().union(*(account['new_work'] for account in accounts))
    new_work = {key: None if any(account['new_work'].get(key) is None for account in accounts) else
                sum(account['new_work'][key] for account in accounts) for key in keys
                if key != 'randomized_new_observations'}
    new_work['randomized_new_observations'] = any(a['new_work']['randomized_new_observations'] for a in accounts)
    parent_cpu = sum(row.get('cpu_seconds', 0.) for row in parent_attempts)
    child_cpu = sum(account['recorded_cpu_seconds'] for account in accounts)
    completion = dict(status=result['status'], completed_utc=datetime.now(timezone.utc).isoformat(),
        registered_geometries=len(averaged) // len(arms), registered_new_streams=len(streams),
        registered_paired_cells=sum(len(rows) for rows in stream_rows),
        recorded_cpu_seconds=parent_cpu + child_cpu, recorded_child_cpu_seconds=child_cpu,
        recorded_parent_cpu_seconds=parent_cpu, parent_attempts=parent_attempts,
        failed_parent_attempts=[row for row in parent_attempts if row['status'] == 'failed'],
        new_work=new_work, new_physical_target_calls=sum(a['new_physical_target_calls'] for a in accounts),
        streams=accounts, execution_failure_cells=failures,
        historical_accounts=plan.get('historical_accounts', []),
        accounting_note='Each new child final account plus non-child parent attempt ledgers is charged once. '
            'Historical Stage65 costs are linked, never added. Parent aggregation makes zero scientific calls.')
    save(output / 'completion.json', completion)
    compact['costs'].update(recorded_cpu_seconds=completion['recorded_cpu_seconds'],
        recorded_parent_cpu_seconds=completion['recorded_parent_cpu_seconds'],
        new_physical_target_calls=completion['new_physical_target_calls'],
        new_work=completion['new_work'], failed_parent_attempt_count=len(completion['failed_parent_attempts']))
    (output / 'compact_summary.json').write_text(json.dumps(compact, separators=(',', ':')) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('score', 'summarize'))
    parser.add_argument('--manifest', default=str(MANIFEST))
    args = parser.parse_args()
    if socket.gethostname().split('.')[0] not in {f'node{i:03d}' for i in range(1, 7)}:
        raise RuntimeError('Run replication scoring through scheduler on node001-node006')
    plan = read(args.manifest)
    if args.stage == 'score':
        score_streams(plan)
    result = summarize(plan)
    completion = read(ROOT / plan['output_dir'] / 'completion.json')
    plan.update(status=result['status'], completed_utc=completion['completed_utc'],
        summary_file=str(Path(plan['output_dir']) / 'summary.json'),
        compact_summary_file=str(Path(plan['output_dir']) / 'compact_summary.json'),
        completion_file=str(Path(plan['output_dir']) / 'completion.json'))
    save(args.manifest, plan)
    print(result['status'], flush=True)


if __name__ == '__main__':
    main()
