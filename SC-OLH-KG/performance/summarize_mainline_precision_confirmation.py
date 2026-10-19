"""Score the globally sealed old-terminal cohort and summarize paired geometries."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import time
import traceback
from unittest.mock import patch

import numpy as np

from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save
from performance import run_mainline_safe_unsampled_audit as audit
from performance import mainline_safe_unsampled_competition as competition
from performance.mainline_pathway_replication import _discovery, _query, CATEGORIES
from performance.mainline_confirmation_query_attribution import terminal_path
from performance.summarize_mainline_two_comparison_confirmation import initial_reference
from performance.summarize_mainline_two_comparison_development import AFTER, POINT, FULL, METRICS, metrics
from performance.analyze_mainline_frozen_propagation_paths import metric_stats, compact_estimate
from performance.run_mainline_pathway_replication import (
    category_decomposition, deployment_detail, summarize_effect, independent_unit, compact_unit,
)
from performance.summarize_mainline_confirmation_seed_replication import sign


MANIFEST = 'performance/manifests/mainline_precision_confirmation_v1.json'
ZERO = dict(audit.ZERO, new_candidate_scores=0, raw_replays=0)


def arm_names(plan):
    return [row['name'] for row in plan['terminal_arms']]


def child_plans(plan, root):
    return [(descriptor, read(Path(root) / descriptor['manifest'])) for descriptor in plan['streams']]


def global_seal(plan, root):
    value = read(Path(root) / plan['output_dir'] / 'execution_status.json')
    expected = {(stream['name'], task['geometry_task_seed'], arm)
                for stream in plan['streams'] for task in plan['tasks'] for arm in arm_names(plan)}
    actual = [(row['stream'], row['seed'], row['arm']) for row in value['cases']]
    if (value['status'] != 'sealed' or not value.get('truth_access_allowed')
            or len(actual) != len(expected) or set(actual) != expected):
        raise ValueError('All registered old-terminal outcomes or failures must be globally sealed before truth')
    for row in value['cases']:
        descriptor = row['old']
        if descriptor['status'] not in ('complete', 'failed', 'missing'):
            raise ValueError('Retain every old-terminal outcome or failure')
        if descriptor['status'] == 'complete':
            for field in ('input_file', 'verification_file'):
                if not (Path(root) / descriptor[field]).is_file():
                    raise ValueError('A completed sealed old-terminal file is absent')
        elif descriptor.get('failure_file') and not (Path(root) / descriptor['failure_file']).is_file():
            raise ValueError('A retained terminal failure file is absent')
    return value


def load_cases(descriptors, root):
    """Only the current geometry's nine descriptors are loaded for one stream."""
    result = []
    for original in descriptors:
        row, old = dict(original), dict(original['old'])
        path = old.get('input_file')
        old['item'] = read(Path(root) / path) if path and (Path(root) / path).is_file() else None
        old['record'] = read(Path(root) / old['verification_file']) if old['status'] == 'complete' else None
        row['old'] = old
        result.append(row)
    return result


def inventory_points(initial, bundles, cases):
    points = dict.fromkeys(tuple(row['point']) for row in initial)
    for bundle in bundles.values():
        points.update(dict.fromkeys(tuple(row['point']) for row in bundle['history']))
        for stage in bundle['audit']['stages']:
            points.update(dict.fromkeys(bundle['points'][i] for i in stage['action_ids']))
    for case in cases:
        old = case['old']
        if old['item'] is not None:
            points.update(dict.fromkeys(tuple(row['point']) for row in old['item']['frozen_shortlist']))
        if old['status'] == 'complete':
            points[tuple(old['record']['deployed'])] = None
    return list(points)


def freeze(plan, *, root=ROOT):
    root, output = Path(root), Path(root) / plan['output_dir']
    target = output / 'mechanism/inventory.json'
    if target.exists():
        return read(target)
    seal = global_seal(plan, root)
    children = child_plans(plan, root)
    indices = []
    with attempt(output / 'mechanism/inventory', 'freeze-complete-old-terminal-pools') as cost:
        cost.update(ZERO)
        for task in plan['tasks']:
            seed = task['geometry_task_seed']
            folder = output / 'mechanism' / f'seed_{seed}'
            streams, union = [], {}
            for descriptor, child in children:
                descriptors = [row for row in seal['cases'] if row['seed'] == seed and row['stream'] == descriptor['name']]
                item = dict(name=descriptor['name'], manifest=descriptor['manifest'], cases=descriptors)
                try:
                    cases = load_cases(descriptors, root)
                    with patch.object(audit, 'ROOT', root):
                        directory, source, initial, shared, bundles, _ = audit.task_material(child, task, cases)
                    points = inventory_points(initial, bundles, cases)
                    union.update(dict.fromkeys(points))
                    path = folder / descriptor['name'] / 'policies.npz'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path, points=np.asarray(points, dtype=np.int16).reshape((-1, child['frozen']['d'])))
                    item.update(status='frozen', policies_file=str(path.relative_to(root)), distinct_policies=len(points),
                        source_directory=str(directory.relative_to(root)), initial_rows=len(initial))
                except Exception:
                    item.update(status='failed', error=traceback.format_exc(), distinct_policies=None)
                streams.append(item)
            union_path = folder / 'policies.npz'
            union_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(union_path, points=np.asarray(list(union), dtype=np.int16).reshape((-1, plan['frozen']['d'])))
            index = dict(seed=seed, domain=task['domain'], streams=streams, union_policies_file=str(union_path.relative_to(root)),
                distinct_policy_task_units=len(union), maximum_analytic_function_calls=3 * len(union),
                inventory_complete=all(row['status'] == 'frozen' for row in streams))
            index_path = folder / 'index.json'
            save(index_path, index)
            indices.append(dict(seed=seed, index_file=str(index_path.relative_to(root)),
                distinct_policy_task_units=len(union), inventory_complete=index['inventory_complete']))
        complete = all(row['inventory_complete'] for row in indices)
        total = sum(row['distinct_policy_task_units'] for row in indices)
        result = dict(status='sealed', registered_geometries=len(plan['tasks']), registered_task_streams=2 * len(plan['tasks']),
            registered_terminal_cells=len(seal['cases']), truth_access_allowed=True, truth_used=False,
            tasks=indices, inventory_complete=complete, known_distinct_policy_task_units=total,
            maximum_analytic_function_calls=3 * total if complete else None,
            inventory_failures=[row for row in indices if not row['inventory_complete']], **ZERO)
        save(target, result)
    return result


def require_inventory(plan, root):
    value = read(Path(root) / plan['output_dir'] / 'mechanism/inventory.json')
    expected = 2 * len(plan['tasks']) * len(arm_names(plan))
    if value['status'] != 'sealed' or not value['truth_access_allowed'] or value['registered_terminal_cells'] != expected:
        raise ValueError('The complete global terminal seal and frozen pool inventory must precede scoring')
    return value


def failed_stream(plan, task, descriptor, error):
    return dict(name=descriptor['name'], seed=task['geometry_task_seed'], domain=task['domain'], status='failed', error=error,
        outcomes=[dict(seed=task['geometry_task_seed'], domain=task['domain'], arm=arm['name'], history_arm=arm['history_arm'],
            protocol='old', status='scoring_failed', metrics=dict.fromkeys(METRICS), discovery_category='unknown',
            reported_policy_id=None, deployed_policy_id=None, actual_deployed_quality=None,
            returned_repeat_count=None, first_query_prefix=None) for arm in plan['terminal_arms']],
        queries=[], policy_paths=[], mechanism=None, coverage=None)


def attach_paths(raw, cases, bundles, initial, cache, physical_ids, best, reference_status, histories=None):
    """Reuse frozen first-query classification and actual old-terminal paths."""
    by_point = {tuple(row['point']): row['policy_id'] for row in raw['policies']}
    state_index = {(row['arm'], row['paid_prefix']): row for row in raw['states']}
    histories = histories or {}
    outcomes = []
    for case in cases:
        old, arm = case['old'], case['history_arm']
        saved = old['record']
        point = tuple(saved['deployed']) if saved is not None else None
        value = cache.get(point)
        metric = metrics(saved['verification']['certified'], value, best) if saved is not None else metrics(failed=True)
        history = bundles.get(arm, {}).get('history', histories.get(arm, initial))
        indices = [i for i, row in enumerate(history) if tuple(row['point']) == point]
        observed = dict(policy_id=by_point.get(point), observation_indices=indices) if indices else None
        discovery = _discovery(arm, observed, state_index)
        if arm not in bundles and arm not in histories and arm != 'initialization_only':
            discovery.update(returned_repeat_count=None, returned_adaptive_count=None, returned_query_prefixes=None)
        if metric['C'] == 0:
            discovery.update(discovery_category='no_true_certified_deployment', first_query_prefix=None,
                returned_repeat_count=None, returned_adaptive_count=None, initially_observed=None, returned_query_prefixes=None)
        outcomes.append(dict(seed=case['seed'], domain=case['domain'], arm=case['arm'], history_arm=arm, protocol='old',
            status=old['status'], metrics=metric, quality_reference_status=reference_status,
            best_initial_quality=None if best is None else best['quality'],
            reported_policy_id=physical_ids.get(point), deployed_policy_id=physical_ids.get(point) if metric['C'] == 1 else None,
            actual_deployed_quality=value['quality'] if metric['C'] == 1 else None,
            selected_truth=value, **discovery,
            verification_budget=None if saved is None else saved['verification'].get('verification_budget')))
    mapping = {row['policy_id']: physical_ids[tuple(row['point'])] for row in raw['policies'] if tuple(row['point']) in physical_ids}
    queries = [_query(state, mapping) for state in raw['states']]
    for query in queries:
        if query['observation_returned'] is True and query['returned_policy_matches'] is True:
            state = state_index[query['arm'], query['paid_prefix']]
            point = next(point for point, pid in by_point.items() if pid == state['selected']['policy_id'])
            indices = [i for i, row in enumerate(bundles[query['arm']]['history']) if tuple(row['point']) == point]
            query['discovery_category'] = _discovery(query['arm'], dict(
                policy_id=state['selected']['policy_id'], observation_indices=indices), state_index)['discovery_category']
    paths = []
    for arm, bundle in bundles.items():
        history = bundle['history']
        for point in dict.fromkeys(tuple(row['point']) for row in history[10:]):
            indices = [i for i, row in enumerate(history) if tuple(row['point']) == point]
            observed = dict(policy_id=by_point.get(point), observation_indices=indices)
            value = cache.get(point)
            paths.append(dict(history_arm=arm, policy_id=physical_ids.get(point), observation_indices=indices,
                truth=value, quality_gain_to_initial=None if best is None or value is None else value['quality'] - best['quality'],
                **_discovery(arm, observed, state_index),
                terminals=[dict(arm=case['arm'], path=terminal_path(point, case['old']['item'], case['old']['record'], case['old']['status']))
                           for case in cases]))
    return outcomes, queries, paths


def score_stream(plan, task, descriptor, child, index, cache, physical_ids, root):
    seed = task['geometry_task_seed']
    folder = Path(root) / plan['output_dir'] / 'cases' / f'seed_{seed}' / descriptor['name']
    target = folder / 'result.json'
    if target.exists():
        return read(target)
    try:
        with attempt(folder, 'score-sealed-complete-old-pool', task['domain']) as cost:
            cost.update(ZERO, seed=seed, stream=descriptor['name'])
            if index['status'] != 'frozen':
                raise ValueError('Retain failed inventory: ' + index.get('error', 'unavailable'))
            cases = load_cases(index['cases'], root)
            with patch.object(audit, 'ROOT', Path(root)):
                _, source, initial, shared, bundles, _ = audit.task_material(child, task, cases)
            points = audit.load_bank(Path(root) / index['policies_file'])
            anchor = next((row['old']['item'] for row in cases if row['old']['item'] is not None), None)
            from performance.run_mainline_two_comparison_development import build_problem
            factory = lambda: build_problem(anchor) if anchor is not None else audit.truth_problem(task, source, [])
            try:
                audit.score_points(points, cache, factory, cost)
            finally:
                audit.cache_save(folder.parent / 'truth.json', cache)
            best, status = initial_reference(initial, cache)
            old_metrics = [dict(arm=row['arm'], old=(metrics(row['old']['record']['verification']['certified'],
                cache.get(tuple(row['old']['record']['deployed'])), best) if row['old']['record'] is not None else metrics(failed=True)))
                for row in cases]
            raw = competition.analyze_task(seed, task['domain'], bundles, cache, initial, shared, cases, old_metrics,
                protocols=('old',), include_substitutions=False)
            save(folder / 'analysis.json', raw)
            if raw['coverage']['invalid_replays']:
                raise ValueError('Saved scores do not reproduce captured choices')
            histories = {}
            for row in cases:
                arm = row['history_arm']
                path = Path(root) / row['source_case'] / 'search.json'
                if arm not in bundles and arm not in histories and path.exists():
                    histories[arm] = read(path)['history']
            outcomes, queries, paths = attach_paths(raw, cases, bundles, initial, cache, physical_ids, best, status, histories)
            result = dict(name=descriptor['name'], seed=seed, domain=task['domain'], status='complete',
                search_coverage_status=raw['status'], outcomes=outcomes, queries=queries, policy_paths=paths,
                mechanism=competition.compact_task(raw), coverage=raw['coverage'], quality_reference_status=status,
                distinct_inventory_policies=len(points), cached_inventory_policies=sum(point in cache for point in points),
                new_truth_function_calls=cost['new_truth_function_calls'], truth_problem_constructions=cost['truth_problem_constructions'])
            save(target, result)
    except Exception:
        result = failed_stream(plan, task, descriptor, traceback.format_exc())
        save(target, result)
    return result


def score_task(plan, seed, *, root=ROOT):
    root, output = Path(root), Path(root) / plan['output_dir']
    gate = require_inventory(plan, root)
    task = next(row for row in plan['tasks'] if row['geometry_task_seed'] == seed)
    target = output / 'cases' / f'seed_{seed}' / 'result.json'
    if target.exists():
        return read(target)
    index_path = next(row['index_file'] for row in gate['tasks'] if row['seed'] == seed)
    index = read(root / index_path)
    points = audit.load_bank(root / index['union_policies_file'])
    physical_ids = {point: f'P{i:04d}' for i, point in enumerate(points)}
    cache = audit.cache_read(target.parent / 'truth.json')
    streams = []
    for (descriptor, child), entry in zip(child_plans(plan, root), index['streams']):
        if descriptor['name'] != entry['name']:
            raise ValueError('Frozen stream order changed')
        streams.append(score_stream(plan, task, descriptor, child, entry, cache, physical_ids, root))
    result = dict(seed=seed, domain=task['domain'], status='complete' if all(row['status'] == 'complete' for row in streams) else 'scoring_failed',
        streams=streams, distinct_policy_task_units=len(points), cached_policies=len(cache),
        maximum_analytic_function_calls=index['maximum_analytic_function_calls'],
        truth_cache_scope='Exact physical-policy fields shared only within this geometry; simulation observations never shared across streams.')
    save(target, result)
    return result


def score_worker(args):
    manifest, seed, root = args
    root = Path(root)
    plan = read(root / manifest)
    try:
        result = score_task(plan, seed, root=root)
    except Exception:
        task = next(row for row in plan['tasks'] if row['geometry_task_seed'] == seed)
        result = dict(seed=seed, domain=task['domain'], status='scoring_failed', error=traceback.format_exc(),
            streams=[failed_stream(plan, task, descriptor, traceback.format_exc()) for descriptor in plan['streams']],
            unclosed_scoring_cpu_seconds=None)
        save(root / plan['output_dir'] / 'cases' / f'seed_{seed}' / 'result.json', result)
    return dict(seed=seed, status=result['status'])


def score_batch(plan, manifest, start, end, workers, *, root=ROOT):
    root, output = Path(root), Path(root) / plan['output_dir']
    require_inventory(plan, root)
    if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(plan['tasks'])
            and isinstance(workers, int) and workers > 0):
        raise ValueError('Select a nonempty registered geometry slice and positive workers')
    target = output / 'scoring_dispatch' / f'shard_{start}_{end}' / 'result.json'
    if target.exists():
        return read(target)
    cpu, wall = time.process_time(), time.monotonic()
    results = []
    with ProcessPoolExecutor(max_workers=min(workers, end - start)) as executor:
        futures = {executor.submit(score_worker, (manifest, task['geometry_task_seed'], str(root))): task
                   for task in plan['tasks'][start:end]}
        for future in as_completed(futures):
            task = futures[future]
            try:
                results.append(future.result())
            except Exception:
                error = traceback.format_exc()
                case_path = output / 'cases' / f"seed_{task['geometry_task_seed']}" / 'result.json'
                if case_path.exists():
                    result = read(case_path)
                else:
                    result = dict(seed=task['geometry_task_seed'], domain=task['domain'], status='scoring_failed',
                        streams=[failed_stream(plan, task, descriptor, error) for descriptor in plan['streams']],
                        unclosed_scoring_cpu_seconds=None)
                    save(case_path, result)
                results.append(dict(seed=task['geometry_task_seed'], status=result['status']))
            print('SCHEDULEURM_PROGRESS ' + json.dumps(dict(current=len(results), total=end-start, unit='geometry',
                elapsed_seconds=time.monotonic()-wall)), flush=True)
    result = dict(status='finished', results=results, coordinator_cpu_seconds=time.process_time()-cpu,
        wall_seconds=time.monotonic()-wall, node=socket.gethostname(), **ZERO)
    save(target, result)
    return result


def stream_summary(plan, name, tasks):
    streams = [next(row for row in task['streams'] if row['name'] == name) for task in tasks]
    outcomes = [row for stream in streams for row in stream['outcomes']]
    expected = {(task['geometry_task_seed'], arm) for task in plan['tasks'] for arm in arm_names(plan)}
    actual = [(row['seed'], row['arm']) for row in outcomes]
    if len(actual) != len(expected) or set(actual) != expected or any(row['protocol'] != 'old' for row in outcomes):
        raise ValueError('Every stream must preserve every registered geometry and all nine old-terminal controls')
    indexed = {(row['seed'], row['arm']): row for row in outcomes}
    controls = []
    for arm in arm_names(plan):
        rows = [dict(seed=task['geometry_task_seed'], domain=task['domain'],
            values=indexed[task['geometry_task_seed'], arm]['metrics']) for task in plan['tasks']]
        controls.append(dict(arm=arm, protocol='old', rows=rows, metrics=metric_stats(rows, 'values'),
            outcome_counts=dict(Counter(indexed[task['geometry_task_seed'], arm]['status'] for task in plan['tasks'])),
            category_counts=dict(Counter(indexed[task['geometry_task_seed'], arm]['discovery_category'] for task in plan['tasks'])),
            by_family=[dict(domain=domain, metrics=metric_stats([row for row in rows if row['domain'] == domain], 'values'))
                for domain in dict.fromkeys(task['domain'] for task in plan['tasks'])]))
    contrasts = []
    for left, right in ((AFTER, POINT), (FULL, AFTER)):
        rows = []
        for task in plan['tasks']:
            a, b = (indexed[task['geometry_task_seed'], arm] for arm in (left, right))
            rows.append(dict(seed=task['geometry_task_seed'], domain=task['domain'], stream=name,
                left_status=a['status'], right_status=b['status'], left=a['metrics'], right=b['metrics'],
                left_deployment=deployment_detail(a), right_deployment=deployment_detail(b), **category_decomposition(a, b)))
        contrasts.append(dict(left=left, right=right, protocol='old', **summarize_effect(rows)))
    return dict(name=name, registered_geometries=len(plan['tasks']), controls=controls, comparisons=contrasts)


def summarize_loaded(plan, tasks):
    indexed = {task['seed']: task for task in tasks}
    expected = {task['geometry_task_seed'] for task in plan['tasks']}
    if len(tasks) != len(expected) or set(indexed) != expected:
        raise ValueError('All registered geometries require completed scoring or a retained failure')
    for task in tasks:
        if len(task['streams']) != 2 or {row['name'] for row in task['streams']} != {row['name'] for row in plan['streams']}:
            raise ValueError('Every geometry must retain both independent observation streams')
    tasks = [indexed[task['geometry_task_seed']] for task in plan['tasks']]
    units = [stream_summary(plan, descriptor['name'], tasks) for descriptor in plan['streams']]
    averaged = independent_unit(units)
    for control in averaged['controls']:
        control['by_family'] = [dict(domain=domain, metrics=metric_stats(
            [row for row in control['rows'] if row['domain'] == domain], 'values'))
            for domain in dict.fromkeys(task['domain'] for task in plan['tasks'])]
    directions = []
    for a, b, average in zip(units[0]['comparisons'], units[1]['comparisons'], averaged['comparisons']):
        paired = []
        for first, second, center in zip(a['rows'], b['rows'], average['rows']):
            paired.append(dict(seed=first['seed'], domain=first['domain'], stream_deltas=[first['delta'], second['delta']],
                geometry_mean_delta=center['delta'], directions={metric: [sign(first['delta'][metric]), sign(second['delta'][metric])]
                    for metric in ('C', 'G')}))
        directions.append(dict(left=a['left'], right=a['right'], protocol='old', rows=paired,
            counts={metric: dict(Counter('/'.join(row['directions'][metric]) for row in paired)) for metric in ('C', 'G')}))
    primary = next(row for row in averaged['comparisons'] if row['left'] == AFTER)
    failures = [dict(seed=task['seed'], domain=task['domain'], stream=stream['name'], error=stream.get('error'))
                for task in tasks for stream in task['streams'] if stream['status'] != 'complete']
    failed_outcomes = [dict(stream=stream['name'], **{key: row.get(key) for key in ('seed', 'domain', 'arm', 'status')})
        for task in tasks for stream in task['streams'] for row in stream['outcomes']
        if row['status'] not in ('complete', 'scoring_failed')]
    historical = plan.get('precision_selection', {})
    return dict(status='complete' if not failures and not failed_outcomes else 'complete_with_retained_failures',
        registered_geometries=len(tasks), registered_task_streams=2 * len(tasks),
        registered_terminal_cells=2 * len(tasks) * len(arm_names(plan)),
        completed_scoring_geometries=sum(task['status'] == 'complete' for task in tasks),
        completed_scoring_streams=sum(stream['status'] == 'complete' for task in tasks for stream in task['streams']),
        streams=units, independent=averaged, primary=primary, stream_directions=directions,
        scoring_failures=failures, terminal_failures=failed_outcomes,
        historical_context=[dict(name=row['name'], registered_geometries=row['registered_geometries'], metrics=row['metrics'])
                            for row in historical.get('cohorts', [])],
        historical_reference=None if not historical.get('reference_context') else {
            key: historical['reference_context'][key] for key in ('name', 'registered_geometries', 'metrics')},
        statistical_unit=plan['statistical_unit'],
        limits=['Both independent streams are averaged within each registered geometry before paired 95% t intervals.',
            'Historical cohorts are shown separately and never added to the new confirmation denominator.',
            'Missing baselines and failed effects remain null; no certificate is zero G only when the baseline is defined.',
            'Family, mechanism and repeated-query slices are descriptive; no separate multiple-testing superiority claims.',
            'Discovery-category and requery components are orthogonal partitions of the same gain, not additive mechanisms.'])


def compact_paths(task):
    streams = []
    for stream in task['streams']:
        paths = []
        for path in stream['policy_paths']:
            if path['history_arm'] not in (AFTER, POINT):
                continue
            paths.append(dict(history_arm=path['history_arm'], policy_id=path['policy_id'], truth=path['truth'],
                quality_gain_to_initial=path['quality_gain_to_initial'], discovery_category=path['discovery_category'],
                first_query_prefix=path['first_query_prefix'], returned_repeat_count=path['returned_repeat_count'],
                returned_query_prefixes=path.get('returned_query_prefixes'),
                terminals=[dict(arm=row['arm'], disposition=row['path'].get('disposition'),
                    shortlist_position=row['path'].get('shortlist_position'), safety_passed=row['path'].get('safety_passed'),
                    safety_attempted=row['path'].get('safety_attempted'), deployed=row['path'].get('deployed'))
                    for row in path['terminals'] if row['arm'] in (AFTER, POINT)]))
        streams.append(dict(name=stream['name'], status=stream['status'], coverage=stream['coverage'],
            quality_reference_status=stream.get('quality_reference_status'),
            queries=[{key: row.get(key) for key in ('arm', 'paid_prefix', 'status', 'selected_policy_id',
                'query_action_kind', 'discovery_category', 'pool_truth_complete', 'counts', 'safe_competition', 'selected_truth')}
                for row in stream['queries']], policy_paths=paths))
    return dict(seed=task['seed'], domain=task['domain'], streams=streams)


def compact_readout(result, tasks, accounts):
    units = [compact_unit(unit) for unit in result['streams']]
    independent = compact_unit(result['independent'])
    # Keep rich signed rows once, for the registered geometry-mean AP contrast.
    for unit in [*units, independent]:
        for contrast in unit['comparisons']:
            if unit is independent and contrast['left'] == AFTER:
                continue
            contrast['rows'] = [{key: row[key] for key in ('seed', 'domain', 'delta', 'positive_gain_component',
                'avoided_loss_component', 'left_deployment', 'right_deployment', 'stream_details') if key in row}
                for row in contrast['rows']]
    primary = next(row for row in independent['comparisons'] if row['left'] == AFTER)
    return dict(status=result['status'], registered_geometries=result['registered_geometries'],
        registered_task_streams=result['registered_task_streams'], registered_terminal_cells=result['registered_terminal_cells'],
        completed_scoring_geometries=result['completed_scoring_geometries'], completed_scoring_streams=result['completed_scoring_streams'],
        streams=units, independent=independent, primary=primary, stream_directions=result['stream_directions'],
        paths=[compact_paths(task) for task in tasks], historical_context=result['historical_context'],
        historical_reference=result['historical_reference'], scoring_failures=result['scoring_failures'],
        terminal_failures=result['terminal_failures'], costs=accounts, limits=result['limits'])


def target_fee_unknowns(seal, root):
    """Scoring failures do not erase already recorded simulation charges."""
    unknown = []
    for manifest in dict.fromkeys(row['search_manifest'] for row in seal['cases']):
        for failure in read(Path(root) / manifest).get('execution_failures', []):
            if failure.get('call_accounting_complete') is False or failure.get('new_search_calls') is None:
                unknown.append(dict(stage='search', manifest=manifest, failure=failure))
    for row in seal['cases']:
        old = row['old']
        if old['status'] != 'failed':
            continue  # Missing input means verification was never invoked.
        failure = read(Path(root) / old['failure_file']) if old.get('failure_file') else old.get('failure', {})
        if failure.get('call_accounting_complete') is False or failure.get('new_verification_calls') is None:
            unknown.append(dict(stage='verify', stream=row['stream'], seed=row['seed'], arm=row['arm'], failure=failure))
    return unknown


def accounting(plan, tasks, *, root=ROOT, seal=None):
    root, output = Path(root), Path(root) / plan['output_dir']
    attempts = [dict(json.loads(line), ledger=str(path.relative_to(root))) for path in output.rglob('attempts.jsonl')
                for line in path.read_text().splitlines() if line.strip()]
    fields = ('new_target_initial_calls', 'new_target_search_calls', 'new_verification_calls',
              'new_truth_function_calls', 'new_source_calls', 'new_source_fits', 'cpu_seconds')
    known = {key: sum(row.get(key) or 0 for row in attempts) for key in fields}
    failed = [row for row in attempts if row['status'] == 'failed']
    seal = read(output / 'execution_status.json') if seal is None else seal
    unknown_target = target_fee_unknowns(seal, root)
    incomplete = bool(unknown_target or any(row.get(key) is None for row in attempts
        for key in ('new_target_initial_calls', 'new_target_search_calls', 'new_verification_calls') if key in row))
    physical = sum(known[key] for key in ('new_target_initial_calls', 'new_target_search_calls', 'new_verification_calls'))
    shards = [read(path) for path in (output / 'scoring_dispatch').glob('shard_*/result.json')]
    verification = [row.get('verification_budget') for task in tasks for stream in task['streams'] for row in stream['outcomes']]
    known_verify = [value for value in verification if value is not None]
    logical_search = 2 * len(plan['tasks']) * sum(10 if arm['history_arm'] == 'initialization_only' else 13 for arm in plan['terminal_arms'])
    return dict(known_recorded=known, physical_target_calls=physical if not incomplete else None,
        known_recorded_physical_target_calls=physical, physical_accounting_scope='known_recorded_lower_bound' if incomplete else 'recorded_complete_execution',
        unknown_target_fee_records=unknown_target,
        stage_attempts=dict(Counter(row['stage'] for row in attempts)), attempts=attempts, failed_attempts=failed,
        recorded_scoring_coordinator_cpu_seconds=sum(row['coordinator_cpu_seconds'] for row in shards),
        unclosed_scoring_cpu_seconds=None if any('unclosed_scoring_cpu_seconds' in task for task in tasks) else 0.,
        completed_cell_verification_budget_sum=sum(known_verify), cells_with_recorded_verification_budget=len(known_verify),
        successful_design_logical_search_calls=logical_search,
        all_registered_successful_logical_target_calls=logical_search+sum(known_verify) if len(known_verify)==len(verification) else None,
        source_model_restorations=None, engine_restorations=None, posterior_restorations=None,
        failed_simulator_invocations=None, simulation_rng_draws=None,
        instrumentation='Historical execution paths do not fully instrument restorations, failed simulator invocations or internal RNG draws; these totals are unknown.',
        historical_accounts=plan['historical_accounts'],
        accounting='Read each physical attempt ledger once. Completed-cell logical budgets and shared-search design counts are separate, '
            'not added to physical calls. Wrapper CPU overlaps children; validation and historical fees are separate.')


def headline(result, accounts):
    """Small default transfer: complete control statistics and 144 paired effects."""
    independent = compact_unit(result['independent'])
    for contrast in independent['comparisons']:
        contrast['rows'] = [{key: row[key] for key in ('seed', 'domain', 'delta',
            'positive_gain_component', 'avoided_loss_component')} for row in contrast['rows']]
    return dict(status=result['status'], registered_geometries=result['registered_geometries'],
        registered_task_streams=result['registered_task_streams'], registered_terminal_cells=result['registered_terminal_cells'],
        independent=independent, stream_directions=result['stream_directions'],
        per_stream=[dict(name=unit['name'], controls=compact_unit(unit)['controls'],
            comparisons=[dict(left=row['left'], right=row['right'],
                delta={key: compact_estimate(row['delta'][key]) for key in ('C', 'G', 'P', 'L')})
                for row in unit['comparisons']]) for unit in result['streams']],
        historical_context=result['historical_context'], historical_reference=result['historical_reference'],
        scoring_failures=result['scoring_failures'], terminal_failures=result['terminal_failures'],
        costs=accounts, limits=result['limits'])


def collect(plan, manifest, *, root=ROOT):
    root, output = Path(root), Path(root) / plan['output_dir']
    target = output / 'completion.json'
    if target.exists():
        return read(target)
    inventory = require_inventory(plan, root)
    seal = read(output / 'execution_status.json')
    tasks = []
    for task in plan['tasks']:
        path = output / 'cases' / f"seed_{task['geometry_task_seed']}" / 'result.json'
        if not path.exists():
            raise ValueError('Every registered geometry needs scoring or a retained failure before collect')
        tasks.append(read(path))
    with attempt(output / 'aggregation', 'aggregate-old-terminal-precision-confirmation') as cost:
        cost.update(ZERO)
        result = summarize_loaded(plan, tasks)
        result['terminal_failures'] = [dict(stream=row['stream'], seed=row['seed'], domain=row['domain'],
            arm=row['arm'], status=row['old']['status'], failure_file=row['old'].get('failure_file'),
            failure=row['old'].get('failure')) for row in seal['cases'] if row['old']['status'] != 'complete']
        result['status'] = 'complete_with_retained_failures' if result['scoring_failures'] or result['terminal_failures'] else 'complete'
        save(output / 'summary.json', result)
    accounts = accounting(plan, tasks, root=root, seal=seal)
    accounts.update(registered_pool_inventory_complete=inventory['inventory_complete'],
        known_distinct_policy_task_units=inventory['known_distinct_policy_task_units'],
        maximum_analytic_function_calls=inventory['maximum_analytic_function_calls'])
    compact_accounts = {key: value for key, value in accounts.items() if key not in ('attempts', 'failed_attempts', 'historical_accounts')}
    compact_accounts['failed_attempt_count'] = len(accounts['failed_attempts'])
    (output / 'readout.json').write_text(json.dumps(compact_readout(result, tasks, compact_accounts), separators=(',', ':')) + '\n')
    (output / 'headline.json').write_text(json.dumps(headline(result, compact_accounts), separators=(',', ':')) + '\n')
    checks = [dict(file=str(path.relative_to(root)), **read(path))
              for path in sorted((output / 'cluster_control/validation').glob('*/checks.json'))]
    completion = dict(status=result['status'], completed_utc=datetime.now(timezone.utc).isoformat(),
        registered_geometries=len(plan['tasks']), registered_task_streams=2 * len(plan['tasks']),
        registered_terminal_cells=result['registered_terminal_cells'], completed_scoring_geometries=result['completed_scoring_geometries'],
        completed_scoring_streams=result['completed_scoring_streams'], scoring_failures=result['scoring_failures'],
        terminal_failures=result['terminal_failures'], costs=accounts, validations=checks,
        historical_accounts=plan['historical_accounts'], summary_file=plan['output_dir']+'/summary.json',
        readout_file=plan['output_dir']+'/readout.json', headline_file=plan['output_dir']+'/headline.json')
    save(target, completion)
    plan.update(status=result['status'], execution_started=True, completed_utc=completion['completed_utc'],
        summary_file=completion['summary_file'], completion_file=plan['output_dir']+'/completion.json')
    save(root / manifest, plan)
    for descriptor, child in child_plans(plan, root):
        failed = any(row['stream'] == descriptor['name'] for row in result['scoring_failures'] + result['terminal_failures'])
        child.update(status='complete_with_retained_failures' if failed else 'complete', execution_started=True,
            completed_utc=completion['completed_utc'], global_completion_file=plan['output_dir']+'/completion.json')
        save(root / descriptor['manifest'], child)
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'score-batch', 'collect'))
    parser.add_argument('--manifest', default=MANIFEST)
    for key in ('start', 'end', 'workers'):
        parser.add_argument('--'+key, type=int)
    args = parser.parse_args()
    if socket.gethostname().split('.')[0] not in {f'node{i:03d}' for i in range(1, 7)}:
        raise RuntimeError('Execute scientific work through scheduler on node001-node006')
    plan = read(ROOT / args.manifest)
    if args.stage == 'freeze':
        result = freeze(plan)
    elif args.stage == 'score-batch':
        result = score_batch(plan, args.manifest, args.start, args.end, args.workers)
    else:
        result = collect(plan, args.manifest)
    print(json.dumps({key: result[key] for key in ('status', 'registered_geometries', 'registered_task_streams',
        'registered_terminal_cells', 'completed_scoring_geometries') if key in result}), flush=True)


if __name__ == '__main__':
    main()
