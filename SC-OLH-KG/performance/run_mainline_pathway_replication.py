"""Scheduler-only replication of saved discovery paths across three completed streams."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import time
import traceback

from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save
from performance.prepare_mainline_two_comparison_terminal import ZERO as EXISTING_ZERO
from performance.analyze_mainline_frozen_propagation_paths import (
    metric_stats, compact_estimate, decomposition,
)
from performance.summarize_mainline_two_comparison_development import (
    AFTER, POINT, FULL, METRICS, difference, estimate,
)
from performance.summarize_mainline_confirmation_seed_replication import mean_known, sign
from performance.mainline_pathway_replication import CATEGORIES


MANIFEST = 'performance/manifests/mainline_pathway_replication_v1.json'
ZERO = dict(EXISTING_ZERO, new_candidate_scores=0, truth_problem_constructions=0)
PROTOCOLS = ('old', 'new')


def arm_names(plan):
    return [row['name'] for row in plan['terminal_arms']]


def require_source(plan, *, root=ROOT):
    """Admit only already sealed/scored pools; manifest snapshots are not runtime state."""
    root = Path(root)
    seeds = {task['geometry_task_seed'] for task in plan['tasks']}
    expected = {(seed, arm) for seed in seeds for arm in arm_names(plan)}
    anchor = next(row for row in plan['streams'] if row['name'] == 'reference')
    geometry = {task['geometry_task_seed']: (task['domain'], task['geometry'])
                for task in read(root / anchor['source_manifest'])['tasks']}
    completed = []
    for directory in plan['completed_source_batches']:
        completion = read(root / directory / 'completion.json')
        if completion['status'] != 'complete':
            raise ValueError('Every source batch must already be complete')
        completed.append(directory + '/completion.json')
    for descriptor in plan['streams']:
        source = read(root / descriptor['source_manifest'])
        if ({task['geometry_task_seed'] for task in source['tasks']} != seeds
                or source['terminal_arms'] != plan['terminal_arms']):
            raise ValueError('Registered source geometry or controls changed')
        if ({task['geometry_task_seed']: (task['domain'], task['geometry']) for task in source['tasks']} != geometry
                or any(geometry[task['geometry_task_seed']][0] != task['domain'] for task in plan['tasks'])):
            raise ValueError('The three paired streams must share identical geometry and family metadata')
        directory = root / descriptor['source_output_dir']
        seal = read(directory / 'execution_status.json')
        actual = [(row['seed'], row['arm']) for row in seal['cases']]
        if (seal['status'] != 'sealed' or len(actual) != len(expected) or set(actual) != expected
                or any(row[p]['status'] not in ('complete', 'failed', 'missing')
                       for row in seal['cases'] for p in PROTOCOLS)):
            raise ValueError('Every stream needs the complete old/new terminal seal')
        inventory = read(directory / 'mechanism/inventory.json')
        if (inventory['status'] != 'sealed' or len(inventory['tasks']) != len(seeds)
                or {row['seed'] for row in inventory['tasks']} != seeds):
            raise ValueError('Every stream needs the complete frozen candidate inventory')
        for seed in seeds:
            folder = directory / 'mechanism' / f'seed_{seed}'
            result = read(folder / 'result.json')
            if (result['status'] != 'complete'
                    or result['cached_inventory_policies'] != result['distinct_inventory_policies']
                    or not (folder / 'analysis.json').is_file()
                    or not (root / descriptor['attribution_file_pattern'].format(seed=seed)).is_file()):
                raise ValueError('Already complete full-pool scoring and attribution are required')
    independent = [row for row in plan['streams'] if row['batch'] == 'independent']
    if len(independent) != 2 or len(plan['streams']) != 3:
        raise ValueError('Keep one reference and exactly two independent observation streams')
    parent = root / independent[0]['source_output_dir']
    global_inventory = read(parent.parent / 'mechanism/inventory.json')
    if (global_inventory['status'] != 'sealed' or global_inventory['registered_streams'] != 2
            or global_inventory['registered_task_streams'] != 2 * len(seeds)
            or global_inventory['registered_terminal_cells'] != 4 * len(expected)):
        raise ValueError('Both independent streams must have their original global seal')
    return dict(status='admitted', registered_geometries=len(seeds),
        registered_task_streams=3 * len(seeds), registered_terminal_outcomes=6 * len(expected),
        completed_source_batches=completed, **ZERO)

def register(plan, manifest, checks, *, root=ROOT):
    root, output = Path(root), Path(root) / plan['output_dir']
    target = output / 'registration' / 'result.json'
    if target.exists():
        return read(target)
    if not checks or read(root / checks)['status'] != 'passed':
        raise ValueError('Targeted substitute checks must pass before saved-record analysis')
    with attempt(target.parent, 'register-pathway-replication') as cost:
        cost.update(ZERO)
        result = require_source(plan, root=root)
        result.update(checks_file=checks, status='registered', registered_utc=datetime.now(timezone.utc).isoformat())
        save(target, result)
        plan.update(status='registered', registered_utc=result['registered_utc'])
        save(root / manifest, plan)
    return result


def failed_task(plan, task, error):
    seed = task['geometry_task_seed']
    return dict(status='failed', seed=seed, domain=task['domain'], error=error,
        streams=[dict(name=descriptor['name'], batch=descriptor['batch'], status='failed',
            outcomes=[dict(arm=arm, protocol=protocol, status='analysis_failed',
                metrics=dict.fromkeys(METRICS), discovery_category='unavailable_saved_path',
                deployed_policy_id=None, reported_policy_id=None, actual_deployed_quality=None,
                first_query_prefix=None, returned_repeat_count=None)
                for arm in arm_names(plan) for protocol in PROTOCOLS],
            queries=[], policy_paths=[], comparisons=[])
            for descriptor in plan['streams']],
        missingness='Saved science remains unchanged; failed analysis does not imply zero gain.')


def validate_task(plan, task, result):
    expected = {(arm, protocol) for arm in arm_names(plan) for protocol in PROTOCOLS}
    streams = {row['name']: row for row in result['streams']}
    if (result['seed'] != task['geometry_task_seed'] or len(streams) != len(result['streams'])
            or set(streams) != {row['name'] for row in plan['streams']}):
        raise ValueError('Every registered geometry must retain all three streams')
    for descriptor in plan['streams']:
        stream = streams[descriptor['name']]
        actual = [(row['arm'], row['protocol']) for row in stream['outcomes']]
        if stream['batch'] != descriptor['batch'] or len(actual) != len(expected) or set(actual) != expected:
            raise ValueError('Every stream must retain all controls and both terminals')
        for row in stream['outcomes']:
            if set(METRICS) - row['metrics'].keys() or row['discovery_category'] not in CATEGORIES:
                raise ValueError('Saved metrics and frozen discovery categories are required')

def geometry_worker(args):
    manifest, seed, root = args
    root = Path(root)
    plan = read(root / manifest)
    task = next(row for row in plan['tasks'] if row['geometry_task_seed'] == seed)
    directory = root / plan['output_dir'] / 'cases' / f'seed_{seed}'
    target = directory / 'result.json'
    if target.exists():
        return dict(seed=seed, status=read(target)['status'], reused=True)
    try:
        with attempt(directory, 'analyze-pathway-replication', task['domain']) as cost:
            cost.update(ZERO, seed=seed)
            from performance.mainline_pathway_replication import analyze_task
            result = analyze_task(root, plan, seed)
            validate_task(plan, task, result)
            save(target, result)
    except Exception:
        result = failed_task(plan, task, traceback.format_exc())
        save(target, result)
    return dict(seed=seed, status=result['status'])


def batch(plan, manifest, start, end, workers, *, root=ROOT):
    root, output = Path(root), Path(root) / plan['output_dir']
    if not (output / 'registration' / 'result.json').exists():
        raise ValueError('Register the completed source cohort before analysis')
    if not 0 <= start < end <= len(plan['tasks']) or workers < 1:
        raise ValueError('A registered geometry slice and positive worker count are required')
    shard = output / 'dispatch' / f'shard_{start}_{end}'
    if (shard / 'result.json').exists():
        return read(shard / 'result.json')
    cpu, wall = time.process_time(), time.monotonic()
    rows = []
    with ProcessPoolExecutor(max_workers=min(workers, end - start)) as executor:
        pending = {executor.submit(geometry_worker, (manifest, task['geometry_task_seed'], str(root))): task
                   for task in plan['tasks'][start:end]}
        for future in as_completed(pending):
            task = pending[future]
            try:
                rows.append(future.result())
            except Exception:
                target = output / 'cases' / f"seed_{task['geometry_task_seed']}" / 'result.json'
                if not target.exists():
                    result = failed_task(plan, task, traceback.format_exc())
                    result.update(unclosed_attempt_cpu_seconds=None)
                    save(target, result)
                rows.append(dict(seed=task['geometry_task_seed'], status=read(target)['status']))
            print('SCHEDULEURM_PROGRESS ' + json.dumps(dict(current=len(rows), total=end - start,
                unit='geometry', elapsed_seconds=time.monotonic() - wall)), flush=True)
    result = dict(status='finished', outcomes=rows, node=socket.gethostname(),
        coordinator_cpu_seconds=time.process_time() - cpu, wall_seconds=time.monotonic() - wall,
        failures=[row for row in rows if row['status'] != 'complete'], **ZERO)
    save(shard / 'result.json', result)
    return result


COMPONENTS = ('positive_gain_component', 'avoided_loss_component', 'net_gain')
REQUERY = ('requery', 'no_requery', 'unknown')


def requery_category(row):
    count = row.get('returned_repeat_count')
    return 'unknown' if count is None else 'requery' if count > 0 else 'no_requery'


def partition_components(left, right, labels, left_label, right_label):
    """Partition both actual deployments, including POINT gains and AFTER losses."""
    delta = decomposition(left['metrics'], right['metrics'])
    known = all(row['metrics'][key] is not None for row in (left, right) for key in ('G', 'P', 'L'))
    categories = {}
    for category in labels:
        a, b = int(left_label == category), int(right_label == category)
        positive = left['metrics']['P'] * a - right['metrics']['P'] * b if known else None
        avoided = right['metrics']['L'] * b - left['metrics']['L'] * a if known else None
        categories[category] = dict(positive_gain_component=positive, avoided_loss_component=avoided,
            net_gain=None if not known else positive + avoided)
    if known and any(abs(sum(row[key] for row in categories.values()) - expected) > 1e-12
            for key, expected in (('positive_gain_component', delta['positive_gain_component']),
                ('avoided_loss_component', delta['avoided_loss_component']), ('net_gain', delta['delta']['G']))):
        raise ValueError('Signed category components must sum to the original contrast')
    return categories


def category_decomposition(left, right):
    return dict(decomposition(left['metrics'], right['metrics']),
        categories=partition_components(left, right, CATEGORIES, left['discovery_category'], right['discovery_category']),
        requery=partition_components(left, right, REQUERY, requery_category(left), requery_category(right)))


def mean_metrics(rows):
    return {key: mean_known([row[key] for row in rows]) for key in METRICS}


def average_effect(rows):
    result = {key: rows[0][key] for key in ('seed', 'domain')}
    for field in ('left', 'right', 'delta'):
        result[field] = mean_metrics([row[field] for row in rows])
    for field in ('positive_gain_component', 'avoided_loss_component'):
        result[field] = mean_known([row[field] for row in rows])
    for field, labels in (('categories', CATEGORIES), ('requery', REQUERY)):
        result[field] = {category: {key: mean_known([row[field][category][key] for row in rows])
            for key in COMPONENTS} for category in labels}
    result['stream_details'] = [{key: row.get(key) for key in ('stream', 'left_status', 'right_status',
        'left_deployment', 'right_deployment')} for row in rows]
    return result


def deployment_detail(row):
    return {key: row.get(key) for key in ('deployed_policy_id', 'reported_policy_id',
        'actual_deployed_quality', 'discovery_category', 'first_query_prefix', 'returned_repeat_count',
        'initially_observed', 'returned_adaptive_count', 'returned_query_prefixes')}


def summarize_effect(rows):
    return dict(rows=rows, delta=metric_stats(rows, 'delta'),
        positive_gain_component=estimate([row['positive_gain_component'] for row in rows]),
        avoided_loss_component=estimate([row['avoided_loss_component'] for row in rows]),
        categories={category: {key: estimate([row['categories'][category][key] for row in rows])
            for key in COMPONENTS} for category in CATEGORIES},
        requery={category: {key: estimate([row['requery'][category][key] for row in rows])
            for key in COMPONENTS} for category in REQUERY},
        by_family=[dict(domain=domain, delta=metric_stats([row for row in rows if row['domain'] == domain], 'delta'),
            positive_gain_component=estimate([row['positive_gain_component'] for row in rows if row['domain'] == domain]),
            avoided_loss_component=estimate([row['avoided_loss_component'] for row in rows if row['domain'] == domain]))
            for domain in dict.fromkeys(row['domain'] for row in rows)])


def stream_unit(tasks, plan, name):
    streams = {task['seed']: next(row for row in task['streams'] if row['name'] == name) for task in tasks}
    indexed = {(seed, row['arm'], row['protocol']): row for seed, stream in streams.items() for row in stream['outcomes']}
    controls = []
    for arm in arm_names(plan):
        for protocol in PROTOCOLS:
            rows = [dict(seed=task['seed'], domain=task['domain'], values=indexed[task['seed'], arm, protocol]['metrics'])
                    for task in tasks]
            controls.append(dict(arm=arm, protocol=protocol, rows=rows, metrics=metric_stats(rows, 'values'),
                outcome_counts=dict(Counter(indexed[task['seed'], arm, protocol]['status'] for task in tasks)),
                category_counts=dict(Counter(indexed[task['seed'], arm, protocol]['discovery_category'] for task in tasks))))
    comparisons = []
    for left, right in ((AFTER, POINT), (FULL, AFTER)):
        for protocol in PROTOCOLS:
            rows = []
            for task in tasks:
                a, b = (indexed[task['seed'], arm, protocol] for arm in (left, right))
                rows.append(dict(seed=task['seed'], domain=task['domain'], stream=name,
                    left_status=a['status'], right_status=b['status'], left=a['metrics'], right=b['metrics'],
                    left_deployment=deployment_detail(a), right_deployment=deployment_detail(b),
                    **category_decomposition(a, b)))
            comparisons.append(dict(left=left, right=right, protocol=protocol, **summarize_effect(rows)))
    return dict(name=name, registered_geometries=len(tasks), controls=controls, comparisons=comparisons)


def independent_unit(units):
    """Average two existing signed effects per geometry, never recompute P from mean G."""
    if len(units) != 2:
        raise ValueError('Two independent streams are required for the geometry mean')
    controls = []
    for first, second in zip(*(unit['controls'] for unit in units)):
        assert (first['arm'], first['protocol']) == (second['arm'], second['protocol'])
        rows = []
        for a, b in zip(first['rows'], second['rows']):
            assert a['seed'] == b['seed']
            rows.append(dict(seed=a['seed'], domain=a['domain'], values=mean_metrics([a['values'], b['values']])))
        controls.append(dict(arm=first['arm'], protocol=first['protocol'], rows=rows,
            metrics=metric_stats(rows, 'values'), per_stream_outcome_counts=[unit['outcome_counts'] for unit in (first, second)],
            per_stream_category_counts=[unit['category_counts'] for unit in (first, second)]))
    comparisons = []
    for first, second in zip(*(unit['comparisons'] for unit in units)):
        assert (first['left'], first['right'], first['protocol']) == (second['left'], second['right'], second['protocol'])
        rows = []
        for a, b in zip(first['rows'], second['rows']):
            assert a['seed'] == b['seed']
            rows.append(average_effect([a, b]))
        comparisons.append(dict(left=first['left'], right=first['right'], protocol=first['protocol'], **summarize_effect(rows)))
    return dict(name='independent', registered_geometries=units[0]['registered_geometries'],
        streams=[unit['name'] for unit in units], controls=controls, comparisons=comparisons)


def replication_directions(reference, independent, fresh):
    result = []
    for original, averaged, first, second in zip(reference['comparisons'], independent['comparisons'],
                                               fresh[0]['comparisons'], fresh[1]['comparisons']):
        rows = []
        for old, mean, a, b in zip(original['rows'], averaged['rows'], first['rows'], second['rows']):
            by_category = {}
            for category in CATEGORIES:
                gains = [row['left']['P'] for row in (a, b)]
                categories = [row['left_deployment']['discovery_category'] for row in (a, b)]
                known_positive = [None if gain is None else gain > 1e-12 and cat == category
                                  for gain, cat in zip(gains, categories)]
                net = [row['categories'][category]['net_gain'] for row in (a, b)]
                by_category[category] = dict(stream_components=[row['categories'][category] for row in (a, b)],
                    left_own_positive=[value for value in known_positive],
                    left_own_positive_both=None if None in known_positive else all(known_positive),
                    signed_contribution_positive_both=None if None in net else all(value > 1e-12 for value in net))
            gains = [row['delta']['G'] for row in (a, b)]
            repeat = []
            for row in (a, b):
                gain, count = row['left']['P'], row['left_deployment']['returned_repeat_count']
                repeat.append(None if gain is None else False if gain <= 1e-12
                              else None if count is None else count > 0)
            repeat_net = [row['requery']['requery']['net_gain'] for row in (a, b)]
            rows.append(dict(seed=old['seed'], domain=old['domain'], reference_delta=old['delta'],
                independent_stream_deltas=[a['delta'], b['delta']], independent_mean_delta=mean['delta'],
                independent_minus_reference=difference(mean['delta'], old['delta']),
                directions={metric: [sign(row['delta'][metric]) for row in (old, a, b)] for metric in ('C', 'G')},
                total_signed_gain_positive_both=None if None in gains else all(value > 1e-12 for value in gains),
                left_stream_deployments=[a['left_deployment'], b['left_deployment']], categories=by_category,
                requery=dict(stream_components=[row['requery'] for row in (a, b)],
                    left_own_positive_both=None if None in repeat else all(repeat),
                    signed_contribution_positive_both=None if None in repeat_net else all(value > 1e-12 for value in repeat_net))))
        result.append(dict(left=original['left'], right=original['right'], protocol=original['protocol'], rows=rows,
            direction_counts={metric: dict(Counter('/'.join(row['directions'][metric]) for row in rows)) for metric in ('C', 'G')},
            independent_minus_reference=metric_stats(rows, 'independent_minus_reference'),
            total_signed_gain_positive_both=event_count([row['total_signed_gain_positive_both'] for row in rows]),
            category_agreement={category: {key: event_count([row['categories'][category][key] for row in rows])
                for key in ('left_own_positive_both', 'signed_contribution_positive_both')} for category in CATEGORIES},
            requery_agreement={key: event_count([row['requery'][key] for row in rows])
                for key in ('left_own_positive_both', 'signed_contribution_positive_both')},
            requery_scope='Orthogonal signed partition of the same gain; never add it to discovery-category components.'))
    return result


def event_count(values):
    return dict(registered_geometries=len(values), defined_geometries=sum(value is not None for value in values),
        true=sum(value is True for value in values), false=sum(value is False for value in values),
        unknown=sum(value is None for value in values))


def summarize(tasks, plan):
    indexed = {row['seed']: row for row in tasks}
    if len(tasks) != len(plan['tasks']) or set(indexed) != {row['geometry_task_seed'] for row in plan['tasks']}:
        raise ValueError('Every registered geometry needs an analysis or a retained failure')
    tasks = [indexed[task['geometry_task_seed']] for task in plan['tasks']]
    for task, result in zip(plan['tasks'], tasks):
        validate_task(plan, task, result)
    units = [stream_unit(tasks, plan, descriptor['name']) for descriptor in plan['streams']]
    reference = next(unit for unit in units if unit['name'] == 'reference')
    fresh = [unit for unit in units if unit['name'] != 'reference']
    independent = independent_unit(fresh)
    all_streams = [stream for task in tasks for stream in task['streams']]
    primary = next(row for row in independent['comparisons'] if (row['left'], row['right'], row['protocol']) == (AFTER, POINT, 'old'))
    return dict(status='complete' if all(stream['status'] == 'complete' for stream in all_streams) else 'complete_with_analysis_failures',
        registered_geometries=len(tasks), completed_geometries=sum(task['status'] == 'complete' for task in tasks),
        registered_task_streams=len(all_streams), completed_task_streams=sum(stream['status'] == 'complete' for stream in all_streams),
        registered_terminal_outcomes=sum(len(stream['outcomes']) for stream in all_streams),
        streams=units, independent=independent, primary=primary,
        replication=replication_directions(reference, independent, fresh),
        analysis_failures=[dict(seed=task['seed'], domain=task['domain'], stream=stream['name'], error=stream.get('error', task.get('error')))
            for task in tasks for stream in task['streams'] if stream['status'] != 'complete'],
        context=plan.get('context'), historical_accounts=plan['historical_accounts'],
        statistical_unit=plan['statistical_unit'], limits=plan['limits'], new_work=dict(ZERO))


def compact_unit(unit):
    controls = []
    for row in unit['controls']:
        item = {key: row[key] for key in ('arm', 'protocol', 'outcome_counts', 'category_counts',
                'per_stream_outcome_counts', 'per_stream_category_counts') if key in row}
        item['metrics'] = {metric: dict(mean=row['metrics'][metric]['all_registered_mean'],
            defined=row['metrics'][metric]['available_geometries'],
            available_mean=row['metrics'][metric]['conditional_available']['mean']) for metric in METRICS}
        controls.append(item)
    comparisons = []
    for row in unit['comparisons']:
        item = {key: row[key] for key in ('left', 'right', 'protocol', 'rows', 'positive_gain_component',
            'avoided_loss_component', 'categories', 'requery')}
        item['delta'] = {key: compact_estimate(row['delta'][key]) for key in ('C', 'G', 'P', 'L')}
        item['by_family'] = [dict(domain=family['domain'],
            delta={key: compact_estimate(family['delta'][key]) for key in ('C', 'G', 'P', 'L')}) for family in row['by_family']]
        comparisons.append(item)
    return dict(name=unit['name'], registered_geometries=unit['registered_geometries'],
        controls=controls, comparisons=comparisons)


def compact_task(task):
    streams = []
    for stream in task['streams']:
        queries = [{key: query.get(key) for key in ('arm', 'paid_prefix', 'status', 'selected_policy_id',
            'previously_observed', 'observation_returned', 'returned_policy_matches', 'replay_valid',
            'mode', 'query_action_kind', 'discovery_category', 'pool_truth_complete', 'counts',
            'safe_competition', 'selected_truth', 'selected_saved_scores')} for query in stream['queries']]
        paths = []
        for path in stream['policy_paths']:
            if path['history_arm'] not in (AFTER, POINT):
                continue
            item = {key: path.get(key) for key in ('history_arm', 'policy_id', 'origin', 'observation_indices',
                'truth', 'quality_gain_to_initial', 'improving_safe', 'discovery_category', 'first_query_prefix',
                'returned_repeat_count', 'returned_adaptive_count', 'initially_observed', 'returned_query_prefixes')}
            item['terminals'] = [dict(arm=terminal['arm'], protocol=terminal['protocol'], status=terminal['status'],
                path={key: terminal['path'].get(key) for key in ('disposition', 'shortlist_position',
                    'safety_attempted', 'safety_passed', 'safety_upper_margin', 'actual_objective_comparisons',
                    'deployed', 'counterfactual_certificate', 'counterfactual_objective_comparison')})
                for terminal in path.get('terminals', []) if terminal['arm'] in (AFTER, POINT)]
            item['observed_in_histories'] = [{key: exposure.get(key) for key in
                ('history_arm', 'returned_observation_indices')} for exposure in path.get('exposures', [])
                if exposure['history_arm'] in (AFTER, POINT)]
            paths.append(item)
        streams.append(dict(name=stream['name'], batch=stream['batch'], status=stream['status'],
            baseline=stream.get('baseline'), coverage=stream.get('coverage'), queries=queries,
            policy_paths=paths, query_comparisons=[row for row in stream.get('query_comparisons', [])
                if row.get('contrast') == 'AP']))
    return dict(seed=task['seed'], domain=task['domain'], status=task['status'], streams=streams)


def compact_summary(result, tasks):
    independent = compact_unit(result['independent'])
    return dict(status=result['status'], registered_geometries=result['registered_geometries'],
        completed_geometries=result['completed_geometries'], registered_task_streams=result['registered_task_streams'],
        completed_task_streams=result['completed_task_streams'], registered_terminal_outcomes=result['registered_terminal_outcomes'],
        streams=[compact_unit(unit) for unit in result['streams']], independent=independent,
        primary=next(row for row in independent['comparisons'] if row['left'] == AFTER and row['protocol'] == 'old'),
        replication=result['replication'], paths=[compact_task(task) for task in tasks],
        analysis_failures=result['analysis_failures'], context=result['context'],
        new_work=result['new_work'], limits=result['limits'])

def collect(plan, manifest, *, root=ROOT):
    root, output = Path(root), Path(root) / plan['output_dir']
    if (output / 'completion.json').exists():
        return read(output / 'completion.json')
    tasks = []
    for task in plan['tasks']:
        path = output / 'cases' / f"seed_{task['geometry_task_seed']}" / 'result.json'
        if not path.exists():
            raise ValueError('Every registered geometry needs a result or retained analysis failure')
        tasks.append(read(path))
    with attempt(output / 'aggregation', 'aggregate-pathway-replication') as cost:
        cost.update(ZERO)
        result = summarize(tasks, plan)
        save(output / 'summary.json', result)
        (output / 'readout.json').write_text(json.dumps(compact_summary(result, tasks), separators=(',', ':')) + '\n')
    attempts = [dict(json.loads(line), ledger=str(path.relative_to(root)))
        for scope in ('registration', 'cases', 'aggregation')
        for path in (output / scope).rglob('attempts.jsonl')
        for line in path.read_text().splitlines() if line.strip()]
    shards = [read(path) for path in (output / 'dispatch').glob('shard_*/result.json')]
    checks = [dict(file=str(path.relative_to(root)), **read(path))
              for path in sorted((output / 'cluster_control' / 'validation').glob('*/checks.json'))]
    preparation = [dict(file=str(path.relative_to(root)), **read(path))
                   for path in sorted((output / 'cluster_control').glob('*preflight*.json'))]
    if any(row.get(key, 0) for row in attempts for key in ZERO):
        raise ValueError('Saved-record analysis ledger contains newly charged scientific work')
    completion = dict(status=result['status'], completed_utc=datetime.now(timezone.utc).isoformat(),
        registered_geometries=len(plan['tasks']), completed_geometries=result['completed_geometries'],
        registered_task_streams=result['registered_task_streams'], completed_task_streams=result['completed_task_streams'],
        registered_terminal_outcomes=result['registered_terminal_outcomes'],
        analysis_failures=result['analysis_failures'], attempts=attempts,
        failed_attempts=[row for row in attempts if row['status'] == 'failed'],
        recorded_analysis_cpu_seconds=sum(row['cpu_seconds'] for row in attempts),
        recorded_coordinator_cpu_seconds=sum(row['coordinator_cpu_seconds'] for row in shards),
        unclosed_attempt_cpu_seconds=None if any('unclosed_attempt_cpu_seconds' in row for row in tasks) else 0.,
        validations=checks, preparation=preparation, new_work=dict(ZERO), historical_accounts=plan['historical_accounts'],
        accounting='Attempt CPU and coordinator CPU are separate and counted once; validation stays separate. '
            'Wrapper CPU overlaps children. Historical science, failed attempts and fees remain referenced without recharging.')
    save(output / 'completion.json', completion)
    plan.update(status=result['status'], completed_utc=completion['completed_utc'],
        summary_file=plan['output_dir'] + '/summary.json', completion_file=plan['output_dir'] + '/completion.json')
    save(root / manifest, plan)
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('register', 'batch', 'collect'))
    parser.add_argument('--manifest', default=MANIFEST)
    parser.add_argument('--checks')
    for key in ('start', 'end', 'workers'):
        parser.add_argument('--' + key, type=int)
    args = parser.parse_args()
    if socket.gethostname().split('.')[0] not in {f'node{i:03d}' for i in range(1, 7)}:
        raise RuntimeError('Run saved-record analysis through scheduler on node001-node006')
    plan = read(ROOT / args.manifest)
    if args.stage == 'register':
        result = register(plan, args.manifest, args.checks)
    elif args.stage == 'batch':
        result = batch(plan, args.manifest, args.start, args.end, args.workers)
    else:
        result = collect(plan, args.manifest)
    print(json.dumps({key: result[key] for key in ('status', 'registered_geometries', 'completed_geometries',
        'registered_terminal_outcomes') if key in result}), flush=True)


if __name__ == '__main__':
    main()
