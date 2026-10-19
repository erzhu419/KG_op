"""Explain the sealed source-alignment cohort using saved paths and metrics only."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import time
import traceback
from unittest.mock import patch

from performance import mainline_precision_propagation_paths as paths
from performance import mainline_pathway_replication as pathway
from performance import mainline_source_alignment_summary as science
from performance.analyze_mainline_precision_propagation_paths import terminal_counts
from performance.analyze_mainline_frozen_propagation_paths import metric_stats
from performance.run_mainline_pathway_replication import REQUERY, requery_category
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save
from performance.summarize_mainline_confirmation_seed_replication import mean_known, sign
from performance.summarize_mainline_two_comparison_development import METRICS, estimate
from performance.summarize_mainline_source_alignment import ZERO


MANIFEST = 'performance/manifests/mainline_source_alignment_paths_v1.json'
AFTER, POINT, FULL = science.AFTER, science.POINT, science.FULL
PERMUTED_AFTER, PERMUTED_POINT = science.PERMUTED_AFTER, science.PERMUTED_POINT
CORE = (AFTER, POINT, PERMUTED_AFTER, PERMUTED_POINT)
ROLES = (*CORE, FULL)
COMPONENTS = ('C', 'positive_gain', 'avoided_loss', 'G')
COMPONENT_METRIC = dict(C='C', positive_gain='P', avoided_loss='L', G='G')
CERTIFICATION = ('both_true_certified', 'left_only_true_certified',
                 'right_only_true_certified', 'neither_true_certified', 'unknown')
QUALITY = tuple(origin + '_' + quality for origin in ('initial', 'new', 'unknown_path')
                for quality in ('positive', 'zero', 'negative')) + (
                    'no_true_certified_deployment', 'unknown_baseline', 'unavailable')
PARTITIONS = dict(certification=CERTIFICATION, discovery=pathway.CATEGORIES,
                  requery=REQUERY, quality=QUALITY)
QUERY_CONTRASTS = ('AP', 'source_AFTER', 'source_POINT', 'permuted_AP', 'FA')
LIMITS = [
    'Each partition is an alternative signed decomposition of actual deployments; partitions are not added together.',
    'New-policy positive quality is a saved deployment category, not an identified causal mediation effect.',
    'Prefix10 is the initialization comparison; a later first divergence is common-state only with identical paid history, physical pool and returned observations.',
    'Pre-initialization predictions are absent; no unrecorded initialization delta is reconstructed.',
    'The four-arm interaction uses common support separately for each metric and both required streams.',
    'Missing outcomes and missing initial quality baselines remain null; partial saved paths do not become scientific failures.',
    'Failed-prefix policies without a completed search record do not acquire an invented terminal observed-pool list.',
]


def arm_names(plan):
    return [row['name'] for row in plan['terminal_arms']]


def quality_category(row):
    values = row['metrics']
    if values['C'] is None:
        return 'unavailable'
    if not values['C']:
        return 'no_true_certified_deployment'
    if values['G'] is None:
        return 'unknown_baseline'
    origin = ('initial' if row.get('initially_observed') is True or row.get('discovery_category') == 'initial'
              else 'new' if row.get('initially_observed') is False or row.get('first_query_prefix') is not None
              else 'unknown_path')
    return origin + '_' + ('positive' if values['G'] > 1e-12 else 'negative' if values['G'] < -1e-12 else 'zero')


def _certification(left, right):
    a, b = left['metrics']['C'], right['metrics']['C']
    if a is None or b is None:
        return 'unknown'
    return ('both_true_certified' if a and b else 'left_only_true_certified' if a else
            'right_only_true_certified' if b else 'neither_true_certified')


def _linear(rows, weights):
    return {key: None if any(row['metrics'][key] is None for row in rows) else
            sum(weight * row['metrics'][key] for row, weight in zip(rows, weights)) for key in METRICS}


def _signed(value, component):
    return -value if component == 'avoided_loss' else value


def pair_components(left, right):
    """Assign each actual deployment once; a missing metric nulls its whole partition."""
    delta = _linear((left, right), (1, -1))
    labels = dict(certification=(_certification(left, right),) * 2,
        discovery=tuple(row.get('discovery_category', 'unknown') for row in (left, right)),
        requery=tuple(requery_category(row) for row in (left, right)),
        quality=tuple(quality_category(row) for row in (left, right)))
    partitions = {}
    for name, options in PARTITIONS.items():
        a, b = labels[name]
        if a not in options or b not in options:
            raise ValueError('Unknown saved deployment category: ' + name)
        partitions[name] = {label: {component: None if delta[key] is None else _signed(
            left['metrics'][key] * int(a == label) - right['metrics'][key] * int(b == label), component)
            for component, key in COMPONENT_METRIC.items()} for label in options}
    result = dict(delta=delta, partitions=partitions)
    _check_partition_sums(result)
    return result


def interaction_components(real_after, real_point, permuted_after, permuted_point):
    """Real AP minus permuted AP, with the same four-arm support for every component."""
    a, b = pair_components(real_after, real_point), pair_components(permuted_after, permuted_point)
    delta = _linear((real_after, real_point, permuted_after, permuted_point), (1, -1, -1, 1))
    partitions = {name: {label: {component: None if delta[key] is None else
        a['partitions'][name][label][component] - b['partitions'][name][label][component]
        for component, key in COMPONENT_METRIC.items()} for label in options} for name, options in PARTITIONS.items()}
    result = dict(delta=delta, partitions=partitions)
    _check_partition_sums(result)
    return result


def _check_partition_sums(row):
    for partition in row['partitions'].values():
        for component, key in COMPONENT_METRIC.items():
            values, expected = [entry[component] for entry in partition.values()], row['delta'][key]
            if expected is None:
                if any(value is not None for value in values):
                    raise ValueError('Missing scientific endpoints must stay null in every component')
            elif any(value is None for value in values) or abs(sum(values) - _signed(expected, component)) > 1e-12:
                raise ValueError('Saved category components must sum to the scientific contrast')


def mean_components(rows):
    result = dict(delta={key: mean_known([row['delta'][key] for row in rows]) for key in METRICS},
        partitions={name: {label: {component: mean_known([
            row['partitions'][name][label][component] for row in rows]) for component in COMPONENTS}
            for label in options} for name, options in PARTITIONS.items()})
    _check_partition_sums(result)
    return result


def query_scope(row):
    if row.get('paid_prefix') != 10:
        if (row.get('status') == 'available' and row.get('first_divergence') is True
                and row.get('same_paid_history') is True and row.get('same_physical_pool') is True
                and row.get('both_observations_returned') is True):
            return 'later_common_state_first_divergence'
        return 'later_adaptive_path'
    if row.get('status') != 'available':
        return 'initial_unavailable'
    return ('initial_common_state' if row.get('same_paid_history') is True and
            row.get('same_physical_pool') is True else 'initial_not_common')


def _effects(outcomes):
    rows = {row['arm']: row for row in outcomes}
    result = []
    for name, terms, role in science.CONTRASTS:
        component = (interaction_components(*(rows[arm] for arm in CORE)) if len(terms) == 4 else
                     pair_components(*(rows[arm] for arm, _ in terms)))
        result.append(dict(name=name, inference_role=role, terms=[dict(arm=a, weight=w) for a, w in terms],
            **component, deployments={a: deepcopy(rows[a]) for a, _ in terms}))
    return result


def _enrich_stream(stream):
    outcomes = {row['arm']: row for row in stream['outcomes']}
    stream['comparisons'] = _effects(stream['outcomes'])
    for row in stream.get('query_comparisons', []):
        row['attribution_scope'] = query_scope(row)
        row['common_state_first_divergence'] = bool(row.get('query_changed') and
            row.get('both_observations_returned') and (row['attribution_scope'] == 'initial_common_state'
            or row['attribution_scope'] == 'later_common_state_first_divergence'))
        row['focal_deployment_links'] = [{
            'policy_id': policy['policy_id'], 'deployments': [dict(arm=arm, metrics=deepcopy(outcome['metrics']),
                discovery_category=outcome.get('discovery_category'), quality_category=quality_category(outcome))
                for arm, outcome in outcomes.items() if outcome.get('deployed_policy_id') == policy['policy_id']]}
                for policy in row.get('focal_policies', [])]
    initial = {row['contrast']: row for row in stream.get('query_comparisons', []) if row['paid_prefix'] == 10}
    core_pairs = [initial.get(label) for label in QUERY_CONTRASTS[:4]]
    stream['four_arm_initial_common_state'] = (None if any(row is None or row.get('status') != 'available'
        for row in core_pairs) else all(query_scope(row) == 'initial_common_state' for row in core_pairs))
    for cell in stream.get('policy_terminal_paths', []):
        cell['observed_pool_coverage'] = 'completed_history' if cell['status'] == 'complete' else 'saved_prefix_only'
        if cell['status'] != 'complete' and not cell['observed']:
            cell['final_observed_pool_truth'] = None
    cells = {row['arm']: row for row in stream.get('policy_terminal_paths', [])}
    stream['history_coverage'] = []
    for arm, outcome in outcomes.items():
        history_arm = outcome.get('history_arm', arm)
        queries = [row for row in stream.get('queries', []) if row['arm'] == history_arm]
        captured = [row for row in queries if row.get('status') == 'captured']
        cell = cells.get(arm)
        complete = cell is not None and cell['status'] == 'complete'
        stream['history_coverage'].append(dict(arm=arm, history_arm=history_arm,
            scientific_status=outcome['status'], completed_observed_pool_available=complete,
            observed_pool_size=len(cell['observed']) if complete else None,
            query_recording_scope='saved_role_states' if history_arm in ROLES else 'not_in_five_role_pool_recording',
            registered_query_rows=len(queries) if history_arm in ROLES else None,
            captured_query_states=len(captured) if history_arm in ROLES else None,
            unavailable_query_states=3-len(captured) if history_arm in ROLES else None,
            known_returned_query_prefix=sum(row.get('observation_returned') is True and
                row.get('returned_policy_matches') is True for row in captured) if history_arm in ROLES else None,
            saved_adaptive_policy_paths=sum(row.get('history_arm') == history_arm for row in stream.get('policy_paths', [])),
            scope='Available saved query events only; missing complete history does not mean zero discoveries or losses.'))
    stream['source_missing_arms'] = [row['arm'] for row in stream['outcomes'] if row['status'] != 'complete']
    return stream


def analyze_task(root, plan, seed):
    """Reuse Stage86's saved-field readers; the five-role extension changes no scorer."""
    with patch.object(paths, 'ROLES', ROLES), patch.object(pathway, 'ROLES', ROLES):
        result = paths.analyze_task(root, plan, seed)
    for stream in result['streams']:
        _enrich_stream(stream)
    result['limits'] = list(LIMITS)
    return result


def require_source(plan, *, root=ROOT):
    root = Path(root)
    source, folder = read(root / plan['source_manifest']), root / plan['source_output_dir']
    completion, seal, inventory = (read(folder / name) for name in (
        'completion.json', 'execution_status.json', 'mechanism/inventory.json'))
    accepted = ('complete', 'complete_with_retained_failures')
    if source['status'] not in accepted or completion['status'] not in accepted or source['output_dir'] != plan['source_output_dir']:
        raise ValueError('The frozen source-alignment confirmation must already be complete')
    for key in ('tasks', 'streams', 'terminal_arms'):
        if source[key] != plan[key]:
            raise ValueError('Frozen cohort metadata changed: ' + key)
    indexed = science.require_seal(source, seal)
    if (inventory['status'] != 'sealed' or not inventory.get('truth_access_allowed')
            or inventory['registered_terminal_cells'] != len(indexed)):
        raise ValueError('The existing saved candidate inventory must already be sealed')
    headline = read(folder / 'headline.json')
    if headline['registered_geometries'] != len(plan['tasks']) or headline['registered_terminal_cells'] != len(indexed):
        raise ValueError('Original summaries must retain every registered unit')
    for task in plan['tasks']:
        if not (folder / 'cases' / f"seed_{task['geometry_task_seed']}" / 'result.json').is_file():
            raise ValueError('Every geometry requires its existing scored record')
    return dict(status='admitted', registered_geometries=len(plan['tasks']),
        registered_task_streams=2 * len(plan['tasks']), registered_terminal_outcomes=len(indexed),
        retained_scientific_failures=sum(row['old']['status'] != 'complete' for row in seal['cases']),
        saved_inventory_complete=inventory['inventory_complete'],
        source_completion=plan['source_output_dir'] + '/completion.json', **ZERO)


def validate_task(plan, task, result):
    streams = {row['name']: row for row in result['streams']}
    if (result['seed'] != task['geometry_task_seed'] or result['domain'] != task['domain']
            or len(streams) != len(result['streams']) or set(streams) != {s['name'] for s in plan['streams']}):
        raise ValueError('Retain both registered streams of each geometry')
    for stream in streams.values():
        rows = stream['outcomes']
        if len(rows) != len(arm_names(plan)) or {(r['arm'], r['protocol']) for r in rows} != {(a, 'old') for a in arm_names(plan)}:
            raise ValueError('Every stream must retain all eleven original scientific outcomes')
        if any(set(row['metrics']) != set(METRICS) for row in rows):
            raise ValueError('Retain all saved scientific metrics')
        if any(row['status'] != 'complete' and any(v is not None for v in row['metrics'].values()) for row in rows):
            raise ValueError('Scientific failures must remain null')


def failed_task(plan, task, error, saved):
    """An explanation failure never overwrites already known scientific outcomes."""
    result = dict(seed=task['geometry_task_seed'], domain=task['domain'], status='analysis_failed', error=error,
        streams=[dict(name=row['name'], status='analysis_failed', error=error, source_status=row['status'],
            outcomes=deepcopy(row['outcomes']), queries=deepcopy(row.get('queries', [])),
            policy_paths=deepcopy(row.get('policy_paths', [])), query_comparisons=[], policy_terminal_paths=[],
            multi_safe_events=[], coverage=deepcopy(row.get('coverage'))) for row in saved['streams']], limits=list(LIMITS))
    for stream in result['streams']:
        _enrich_stream(stream)
    return result


def stream_measures(stream):
    values = dict(four_arm_initial_common_state=stream.get('four_arm_initial_common_state'))
    for label in QUERY_CONTRASTS:
        rows = [r for r in stream.get('query_comparisons', []) if r['contrast'] == label]
        first = next((r for r in rows if r['paid_prefix'] == 10), None)
        available = first is not None and first.get('status') == 'available'
        values.update({label + '.prefix10_available': available,
            label + '.prefix10_common_state': None if not available else query_scope(first) == 'initial_common_state',
            label + '.prefix10_query_changed': first.get('query_changed') if available else None,
            label + '.prefix10_common_returned_change': first.get('common_state_first_divergence') if available else None})
        later = [r for r in rows if r['paid_prefix'] in (11, 12)]
        values[label + '.later_query_changes'] = (None if len(later) != 2 or any(r.get('status') != 'available'
            or r.get('query_changed') is None for r in later) else sum(bool(r['query_changed']) for r in later))
        values[label + '.later_common_state_first_divergence'] = (None if len(later) != 2 or any(
            r.get('status') != 'available' for r in later) else sum(
                query_scope(r) == 'later_common_state_first_divergence' for r in later))
    cells = {row['arm']: row for row in stream.get('policy_terminal_paths', [])}
    outcomes = {row['arm']: row for row in stream['outcomes']}
    for arm in CORE:
        queries = [r for r in stream.get('queries', []) if r['arm'] == arm]
        returned = [r for r in queries if r.get('observation_returned') is True and r.get('returned_policy_matches') is True]
        complete = len(queries) == 3 and all(r.get('status') == 'captured' for r in queries)
        for label, predicate in (
                ('returned_new', lambda q: q.get('query_action_kind') == 'new'),
                ('returned_repeat', lambda q: q.get('query_action_kind') == 'repeat'),
                ('returned_multi_safe', lambda q: q.get('discovery_category') == paths.MULTI),
                ('returned_minimum_violation', lambda q: q.get('mode') == 'minimum_violation')):
            values[arm + '.' + label] = sum(predicate(q) for q in returned) if complete else None
            values[arm + '.known_prefix.' + label] = sum(predicate(q) for q in returned)
        values[arm + '.captured_query_states'] = sum(row.get('status') == 'captured' for row in queries)
        values[arm + '.unavailable_query_states'] = 3 - values[arm + '.captured_query_states']
        values[arm + '.complete_query_coverage'] = complete
        cell = cells.get(arm)
        if cell is not None and cell['status'] == 'complete':
            counts = terminal_counts(cell['observed'], outcomes[arm].get('actual_deployed_quality'))
            values.update({arm + '.terminal.' + key: value for key, value in counts.items()})
    return values


def _statistics(rows):
    return dict(delta=metric_stats(rows, 'delta'), partitions={name: {label: {
        component: estimate([row['partitions'][name][label][component] for row in rows]) for component in COMPONENTS}
        for label in options} for name, options in PARTITIONS.items()})


def _unit(name, task_rows):
    comparisons = []
    for contrast, _, role in science.CONTRASTS:
        rows = [dict(seed=task['seed'], domain=task['domain'], **next(
            row for row in task['comparisons'] if row['name'] == contrast)) for task in task_rows]
        comparisons.append(dict(name=contrast, inference_role=role, rows=rows, **_statistics(rows),
            signed_geometry_seeds={key: {direction: [r['seed'] for r in rows if sign(r['delta'][key]) == direction]
                for direction in ('positive', 'zero', 'negative', 'unknown')} for key in ('C', 'G')},
            by_family=[dict(domain=domain, **_statistics([r for r in rows if r['domain'] == domain]))
                for domain in dict.fromkeys(r['domain'] for r in rows)]))
    return dict(name=name, registered_geometries=len(task_rows), comparisons=comparisons)


def summarize(tasks, plan, saved_science):
    indexed = {row['seed']: row for row in tasks}
    if len(indexed) != len(tasks) or set(indexed) != {row['geometry_task_seed'] for row in plan['tasks']}:
        raise ValueError('Every registered geometry needs a saved analysis or retained analysis failure')
    ordered = [indexed[row['geometry_task_seed']] for row in plan['tasks']]
    for task, result in zip(plan['tasks'], ordered):
        validate_task(plan, task, result)
    units, measure_units, failures = [], [], []
    for descriptor in plan['streams']:
        rows, measures = [], []
        for task in ordered:
            stream = next(row for row in task['streams'] if row['name'] == descriptor['name'])
            rows.append(dict(seed=task['seed'], domain=task['domain'], comparisons=_effects(stream['outcomes'])))
            measures.append(dict(seed=task['seed'], domain=task['domain'], values=stream_measures(stream)))
            if stream['status'] != 'complete':
                failures.append(dict(seed=task['seed'], domain=task['domain'], stream=descriptor['name'], error=stream.get('error')))
        units.append(_unit(descriptor['name'], rows))
        measure_units.append(dict(name=descriptor['name'], rows=measures))
    paired = []
    for index, task in enumerate(ordered):
        comparisons = []
        for number, (name, terms, role) in enumerate(science.CONTRASTS):
            per_stream = [unit['comparisons'][number]['rows'][index] for unit in units]
            comparisons.append(dict(name=name, terms=[dict(arm=a, weight=w) for a, w in terms],
                inference_role=role, **mean_components(per_stream), stream_details=per_stream))
        paired.append(dict(seed=task['seed'], domain=task['domain'], comparisons=comparisons))
    independent = _unit('independent', paired)
    # The scientific summary is an independent saved reference, including every control and fee.
    saved_effects = {row['name']: row for row in saved_science['independent']['comparisons']}
    for effect in independent['comparisons']:
        for key in METRICS:
            actual, expected = effect['delta'][key], saved_effects[effect['name']]['delta'][key]
            if actual['available_geometries'] != expected['available_geometries']:
                raise ValueError('Path decomposition changed the saved scientific denominator')
            a, b = actual['conditional_available']['mean'], expected['conditional_available']['mean']
            if (a is None) != (b is None) or a is not None and abs(a - b) > 1e-12:
                raise ValueError('Path decomposition changed the saved scientific effect')
    keys = sorted({key for unit in measure_units for row in unit['rows'] for key in row['values']})
    averaged = [dict(seed=a['seed'], domain=a['domain'], values={key: mean_known([
        a['values'].get(key), b['values'].get(key)]) for key in keys})
        for a, b in zip(measure_units[0]['rows'], measure_units[1]['rows'])]
    def measures(rows):
        return {key: estimate([row['values'].get(key) for row in rows]) for key in keys}
    for unit in measure_units:
        unit['measures'] = measures(unit['rows'])
    outcomes = [dict(stream=s['name'], **deepcopy(row)) for task in ordered for s in task['streams'] for row in s['outcomes']]
    return dict(status='complete' if not failures else 'complete_with_analysis_failures',
        registered_geometries=len(tasks), completed_geometries=sum(all(s['status'] == 'complete' for s in t['streams']) for t in ordered),
        registered_task_streams=2 * len(tasks), completed_task_streams=2 * len(tasks) - len(failures),
        registered_terminal_outcomes=len(outcomes), retained_scientific_outcomes=outcomes,
        retained_scientific_status_counts=dict(Counter(row['status'] for row in outcomes)),
        streams=units, independent=independent, query_measures=measure_units,
        paired_query_measures=dict(rows=averaged, measures=measures(averaged)),
        saved_science=deepcopy(saved_science), source_headline=plan['source_output_dir'] + '/headline.json',
        analysis_failures=failures, historical_accounts=plan.get('historical_accounts', []),
        statistical_unit='Both required streams averaged within geometry; missing either stays null.',
        limits=[*plan.get('limits', []), *LIMITS], new_work=dict(ZERO))


def compact_summary(result):
    def unit(value):
        return {**{k: v for k, v in value.items() if k != 'comparisons'}, 'comparisons': [
            {k: v for k, v in row.items() if k != 'rows'} for row in value['comparisons']]}
    compact = {k: v for k, v in result.items() if k not in (
        'retained_scientific_outcomes', 'streams', 'independent', 'query_measures', 'paired_query_measures', 'saved_science')}
    compact.update(streams=[unit(row) for row in result['streams']], independent=unit(result['independent']),
        query_measures=[{k: v for k, v in row.items() if k != 'rows'} for row in result['query_measures']],
        paired_query_measures=dict(measures=result['paired_query_measures']['measures']),
        saved_scientific_controls=deepcopy(result['saved_science']['independent']['controls']),
        saved_scientific_costs=deepcopy(result['saved_science']['costs']),
        source_numerical_limits=deepcopy(result['saved_science'].get('source_numerical_limits')),
        source_warning_document=result['saved_science'].get('source_warning_document'))
    return compact


def register(plan, manifest, checks, *, root=ROOT):
    root, output = Path(root), Path(root) / plan['output_dir']
    target = output / 'registration/result.json'
    if target.exists():
        return read(target)
    if not checks or read(root / checks)['status'] != 'passed':
        raise ValueError('Targeted substitutes must pass before saved-record analysis')
    with attempt(target.parent, 'register-source-alignment-paths') as cost:
        cost.update(ZERO)
        result = require_source(plan, root=root)
        result.update(status='registered', checks_file=checks, registered_utc=datetime.now(timezone.utc).isoformat())
        save(target, result)
        plan.update(status='registered', registered_utc=result['registered_utc'])
        save(root / manifest, plan)
    return result


def geometry_worker(args):
    manifest, seed, root = args
    root, plan = Path(root), read(Path(root) / manifest)
    task = next(row for row in plan['tasks'] if row['geometry_task_seed'] == seed)
    folder, target = root / plan['output_dir'] / 'cases' / f'seed_{seed}', None
    target = folder / 'result.json'
    if target.exists():
        return dict(seed=seed, status=read(target)['status'], reused=True)
    saved = read(root / plan['source_output_dir'] / 'cases' / f'seed_{seed}' / 'result.json')
    try:
        with attempt(folder, 'analyze-source-alignment-paths', task['domain']) as cost:
            cost.update(ZERO, seed=seed)
            result = analyze_task(root, plan, seed)
            validate_task(plan, task, result)
            save(target, result)
    except Exception:
        result = failed_task(plan, task, traceback.format_exc(), saved)
        save(target, result)
    return dict(seed=seed, status=result['status'])


def batch(plan, manifest, start, end, workers, *, root=ROOT):
    root, output = Path(root), Path(root) / plan['output_dir']
    if not (output / 'registration/result.json').exists():
        raise ValueError('Register the sealed source cohort before analysis')
    if not (0 <= start < end <= len(plan['tasks']) and workers > 0):
        raise ValueError('A nonempty geometry slice and positive workers are required')
    target = output / 'dispatch' / f'shard_{start}_{end}' / 'result.json'
    if target.exists():
        return read(target)
    cpu, wall, rows = time.process_time(), time.monotonic(), []
    with ProcessPoolExecutor(max_workers=min(workers, end - start)) as executor:
        pending = {executor.submit(geometry_worker, (manifest, task['geometry_task_seed'], str(root))): task
                   for task in plan['tasks'][start:end]}
        for future in as_completed(pending):
            task = pending[future]
            try:
                rows.append(future.result())
            except Exception:
                path = output / 'cases' / f"seed_{task['geometry_task_seed']}" / 'result.json'
                if not path.exists():
                    saved = read(root / plan['source_output_dir'] / 'cases' / f"seed_{task['geometry_task_seed']}" / 'result.json')
                    result = failed_task(plan, task, traceback.format_exc(), saved)
                    result['unclosed_attempt_cpu_seconds'] = None
                    save(path, result)
                rows.append(dict(seed=task['geometry_task_seed'], status=read(path)['status']))
            print('SCHEDULEURM_PROGRESS ' + json.dumps(dict(current=len(rows), total=end-start,
                unit='geometry', elapsed_seconds=time.monotonic()-wall)), flush=True)
    result = dict(status='finished', outcomes=rows, coordinator_cpu_seconds=time.process_time()-cpu,
        wall_seconds=time.monotonic()-wall, node=socket.gethostname(),
        failures=[row for row in rows if row['status'] != 'complete'], **ZERO)
    save(target, result)
    return result


def collect(plan, manifest, *, root=ROOT):
    root, output = Path(root), Path(root) / plan['output_dir']
    target = output / 'completion.json'
    if target.exists():
        return read(target)
    if not (output / 'registration/result.json').exists():
        raise ValueError('Register the sealed source before collection')
    tasks = [read(output / 'cases' / f"seed_{task['geometry_task_seed']}" / 'result.json') for task in plan['tasks']]
    with attempt(output / 'aggregation', 'aggregate-source-alignment-paths') as cost:
        cost.update(ZERO)
        result = summarize(tasks, plan, read(root / plan['source_output_dir'] / 'headline.json'))
        save(output / 'summary.json', result)
        save(output / 'headline.json', compact_summary(result))
    attempts = [dict(json.loads(line), ledger=str(path.relative_to(root))) for scope in ('registration', 'cases', 'aggregation')
        for path in (output / scope).rglob('attempts.jsonl') for line in path.read_text().splitlines() if line.strip()]
    if any(row.get(key, 0) != 0 for row in attempts for key in ZERO):
        raise ValueError('Saved-path analysis must not charge new scientific work or randomness')
    shards = [read(path) for path in (output / 'dispatch').glob('shard_*/result.json')]
    completion = dict(status=result['status'], completed_utc=datetime.now(timezone.utc).isoformat(),
        **{key: result[key] for key in ('registered_geometries', 'completed_geometries', 'registered_task_streams',
            'completed_task_streams', 'registered_terminal_outcomes', 'analysis_failures', 'retained_scientific_status_counts')},
        attempts=attempts, failed_attempts=[row for row in attempts if row['status'] == 'failed'],
        recorded_analysis_cpu_seconds=sum(row['cpu_seconds'] for row in attempts),
        recorded_coordinator_cpu_seconds=sum(row['coordinator_cpu_seconds'] for row in shards),
        unclosed_attempt_cpu_seconds=None if any('unclosed_attempt_cpu_seconds' in row for row in tasks) else 0.,
        new_work=dict(ZERO), historical_accounts=plan.get('historical_accounts', []),
        source_headline=result['source_headline'], headline_file=plan['output_dir'] + '/headline.json',
        accounting='Analysis attempt and coordinator CPU are separate from preserved scientific and check fees. '
            'Wrapper CPU overlaps children; historical unknown counters remain unknown.')
    save(target, completion)
    plan.update(status=result['status'], completed_utc=completion['completed_utc'],
        summary_file=plan['output_dir'] + '/summary.json', completion_file=plan['output_dir'] + '/completion.json')
    save(root / manifest, plan)
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('register', 'batch', 'collect'))
    parser.add_argument('--manifest', default=MANIFEST)
    parser.add_argument('--checks')
    for name in ('start', 'end', 'workers'):
        parser.add_argument('--' + name, type=int)
    args = parser.parse_args()
    if socket.gethostname().split('.')[0] not in {f'node{i:03d}' for i in range(1, 7)}:
        raise RuntimeError('Run saved-path analysis through scheduler on node001-node006')
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
