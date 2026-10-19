"""Scheduler-only explanation of the frozen 144-geometry, two-stream confirmation."""
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
from performance.summarize_mainline_two_comparison_development import AFTER, POINT, FULL, METRICS, estimate
from performance.summarize_mainline_confirmation_seed_replication import mean_known, sign
from performance.run_mainline_pathway_replication import category_decomposition


MANIFEST = 'performance/manifests/mainline_precision_propagation_paths_v1.json'
ZERO = dict(EXISTING_ZERO, new_candidate_scores=0, truth_problem_constructions=0)


def arm_names(plan):
    return [row['name'] for row in plan['terminal_arms']]


def require_source(plan, *, root=ROOT):
    root = Path(root)
    source = read(root / plan['source_manifest'])
    folder = root / plan['source_output_dir']
    completion = read(folder / 'completion.json')
    seal = read(folder / 'execution_status.json')
    inventory = read(folder / 'mechanism/inventory.json')
    accepted = ('complete', 'complete_with_retained_failures')
    if source['status'] not in accepted or completion['status'] not in accepted:
        raise ValueError('The original confirmation must already be complete')
    for key in ('tasks', 'streams', 'terminal_arms'):
        if source[key] != plan[key]:
            raise ValueError('The frozen confirmation metadata changed: ' + key)
    expected = {(stream['name'], task['geometry_task_seed'], arm) for stream in plan['streams']
                for task in plan['tasks'] for arm in arm_names(plan)}
    actual = [(row['stream'], row['seed'], row['arm']) for row in seal['cases']]
    if (seal['status'] != 'sealed' or not seal.get('truth_access_allowed')
            or len(actual) != len(expected) or set(actual) != expected
            or any(row['old']['status'] not in ('complete', 'failed', 'missing') for row in seal['cases'])):
        raise ValueError('Every old-terminal control must be globally sealed')
    if inventory['status'] != 'sealed' or not inventory['inventory_complete']:
        raise ValueError('The original complete candidate inventory must already be frozen')
    headline = read(folder / 'headline.json')
    if (headline['registered_geometries'] != len(plan['tasks'])
            or headline['registered_terminal_cells'] != len(expected)):
        raise ValueError('Saved scientific summaries must retain all registered units')
    for task in plan['tasks']:
        if not (folder / 'cases' / f"seed_{task['geometry_task_seed']}" / 'result.json').is_file():
            raise ValueError('Every geometry requires its original saved score result')
    return dict(status='admitted', registered_geometries=len(plan['tasks']),
        registered_task_streams=2*len(plan['tasks']), registered_terminal_outcomes=len(expected),
        source_completion=plan['source_output_dir']+'/completion.json',
        source_headline=plan['source_output_dir']+'/headline.json', **ZERO)


def register(plan, manifest, checks, *, root=ROOT):
    root, output = Path(root), Path(root)/plan['output_dir']
    target = output/'registration/result.json'
    if target.exists():
        return read(target)
    if not checks or read(root/checks)['status'] != 'passed':
        raise ValueError('Targeted substitutes must pass before saved-path analysis')
    with attempt(target.parent, 'register-precision-propagation-paths') as cost:
        cost.update(ZERO)
        result = require_source(plan, root=root)
        result.update(status='registered', checks_file=checks, registered_utc=datetime.now(timezone.utc).isoformat())
        save(target, result)
        plan.update(status='registered', registered_utc=result['registered_utc'])
        save(root/manifest, plan)
    return result


def failed_task(plan, task, error):
    return dict(seed=task['geometry_task_seed'], domain=task['domain'], status='failed', error=error,
        streams=[dict(name=stream['name'], status='failed', error=error,
            outcomes=[dict(arm=arm, protocol='old', status='analysis_failed', metrics=dict.fromkeys(METRICS),
                discovery_category='unknown', returned_repeat_count=None) for arm in arm_names(plan)],
            queries=[], query_comparisons=[], policy_paths=[], policy_terminal_paths=[], multi_safe_events=[], comparisons=[])
            for stream in plan['streams']],
        missingness='Path analysis failed; the already sealed scientific results remain unchanged in the source headline.')


def validate_task(plan, task, result):
    streams = {row['name']: row for row in result['streams']}
    if (result['seed'] != task['geometry_task_seed'] or result['domain'] != task['domain']
            or len(streams) != len(result['streams']) or set(streams) != {row['name'] for row in plan['streams']}):
        raise ValueError('Each geometry must retain both registered streams')
    for stream in result['streams']:
        actual = [(row['arm'], row['protocol']) for row in stream['outcomes']]
        if len(actual) != len(arm_names(plan)) or set(actual) != {(arm, 'old') for arm in arm_names(plan)}:
            raise ValueError('Every stream must retain all nine old-terminal controls')
        if any(set(METRICS)-row['metrics'].keys() for row in stream['outcomes']):
            raise ValueError('Saved metrics must retain all fields, including null baselines')


def geometry_worker(args):
    manifest, seed, root = args
    root = Path(root)
    plan = read(root/manifest)
    task = next(row for row in plan['tasks'] if row['geometry_task_seed'] == seed)
    folder = root/plan['output_dir']/'cases'/f'seed_{seed}'
    target = folder/'result.json'
    if target.exists():
        return dict(seed=seed, status=read(target)['status'], reused=True)
    try:
        with attempt(folder, 'analyze-precision-propagation-paths', task['domain']) as cost:
            cost.update(ZERO, seed=seed)
            from performance.mainline_precision_propagation_paths import analyze_task
            result = analyze_task(root, plan, seed)
            validate_task(plan, task, result)
            save(target, result)
    except Exception:
        result = failed_task(plan, task, traceback.format_exc())
        save(target, result)
    return dict(seed=seed, status=result['status'])


def batch(plan, manifest, start, end, workers, *, root=ROOT):
    root, output = Path(root), Path(root)/plan['output_dir']
    if not (output/'registration/result.json').exists():
        raise ValueError('Register the sealed source before analysis')
    if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(plan['tasks'])
            and isinstance(workers, int) and workers > 0):
        raise ValueError('A nonempty registered geometry slice and positive workers are required')
    target = output/'dispatch'/f'shard_{start}_{end}'/'result.json'
    if target.exists():
        return read(target)
    cpu, wall, rows = time.process_time(), time.monotonic(), []
    with ProcessPoolExecutor(max_workers=min(workers, end-start)) as executor:
        futures = {executor.submit(geometry_worker, (manifest, task['geometry_task_seed'], str(root))): task
                   for task in plan['tasks'][start:end]}
        for future in as_completed(futures):
            task = futures[future]
            try:
                rows.append(future.result())
            except Exception:
                path = output/'cases'/f"seed_{task['geometry_task_seed']}"/'result.json'
                if not path.exists():
                    result = failed_task(plan, task, traceback.format_exc())
                    result['unclosed_attempt_cpu_seconds'] = None
                    save(path, result)
                rows.append(dict(seed=task['geometry_task_seed'], status=read(path)['status']))
            print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=len(rows), total=end-start,
                unit='geometry', elapsed_seconds=time.monotonic()-wall)), flush=True)
    result = dict(status='finished', outcomes=rows, coordinator_cpu_seconds=time.process_time()-cpu,
        wall_seconds=time.monotonic()-wall, node=socket.gethostname(),
        failures=[row for row in rows if row['status'] != 'complete'], **ZERO)
    save(target, result)
    return result


DISPOSITIONS = ('deployed', 'not_on_frozen_shortlist', 'not_verified_after_earlier_terminal_decision',
    'safety_certificate_failed', 'objective_guard_retained_other_policy', 'verification_unavailable',
    'terminal_input_unavailable')
TERMINAL_REASONS = (*DISPOSITIONS, 'actual_objective_comparison_retained_other_policy',
    'safety_certified_without_actual_objective_comparison', 'verified_but_not_true_certified_deployment')
MULTI = 'multiple_safe_unobserved_competition'


def known_count(values):
    return None if any(value is None for value in values) else sum(bool(value) for value in values)


def terminal_counts(rows, deployed_quality=None):
    paths = [row.get('terminal_path') or {} for row in rows]
    unavailable = ('verification_unavailable', 'terminal_input_unavailable')
    result = dict(observed_policies=len(rows), on_shortlist=known_count([None if not path or
        path.get('disposition') == 'terminal_input_unavailable' else path.get('shortlist_position') is not None for path in paths]),
        safety_attempted=known_count([None if not path or path.get('disposition') in unavailable else
            path.get('safety_attempted') is True for path in paths]),
        safety_passed=known_count([None if not path or path.get('disposition') in unavailable else
            path.get('safety_passed') is True for path in paths]),
        true_certified_deployed=known_count([path.get('deployed') if path else None for path in paths]),
        unknown_terminal_path=sum(path.get('disposition') not in DISPOSITIONS for path in paths))
    result.update({'terminal_reason.'+name: sum(path.get('terminal_reason') == name for path in paths) for name in TERMINAL_REASONS})
    improving, better, unknown_improving, unknown_better = [], [], 0, 0
    for row, path in zip(rows, paths):
        truth, gain = row.get('truth') or {}, row.get('quality_gain_to_initial')
        feasible, quality = truth.get('true_feasible'), truth.get('quality')
        if feasible is False:
            continue
        if feasible is None or gain is None:
            unknown_improving += 1
        elif feasible and gain > 1e-12:
            improving.append(path)
        if feasible is None or quality is None or deployed_quality is None:
            unknown_better += 1
        elif feasible and quality > deployed_quality+1e-12:
            better.append(path)
    result.update(known_improving_safe=len(improving), improving_safe_unknown=unknown_improving,
        known_improving_safe_not_deployed=sum(path.get('deployed') is False for path in improving),
        known_safe_better_than_deployment=len(better), better_than_deployment_unknown=unknown_better)
    for name in TERMINAL_REASONS:
        result['better_than_deployment.'+name] = sum(path.get('terminal_reason') == name for path in better)
    return result


def stream_measures(stream):
    """Describe saved transitions; counts are not independent policy observations."""
    values = {}
    outcomes = {row['arm']: row for row in stream['outcomes']}
    for label, left, right in (('AP', AFTER, POINT), ('FA', FULL, AFTER)):
        parts = category_decomposition(outcomes[left], outcomes[right])['categories'][MULTI]
        values.update({label+'.multi_signed.'+key: value for key, value in parts.items()})
    if stream['status'] != 'complete':
        return values
    for contrast in ('AP', 'FA'):
        rows = [row for row in stream['query_comparisons'] if row['contrast'] == contrast]
        initial = next((row for row in rows if row['paid_prefix'] == 10), {})
        captured = initial.get('status') == 'available'
        common = None if not captured or initial.get('same_paid_history') is None or initial.get('same_physical_pool') is None else bool(
            initial['same_paid_history'] and initial['same_physical_pool'])
        flags = dict(prefix10_same_history_pool=common,
            prefix10_query_changed=initial.get('query_changed') if captured else None,
            changed_query_prefixes=known_count([row.get('query_changed') for row in rows]) if len(rows) == 3 else None,
            common_state_first_divergence=known_count([row.get('common_state_first_divergence')
                if row.get('status') == 'available' else None for row in rows]) if len(rows) == 3 else None,
            later_diverged_history_prefixes=known_count([None if row.get('status') != 'available' or row.get('same_paid_history') is None else
                not row.get('same_paid_history') for row in rows if row['paid_prefix'] > 10]) if len(rows) == 3 else None)
        focal = initial.get('focal_policies', [])
        eligibility = [None if any(row.get(side, {}).get('eligible') is None for side in ('left', 'right'))
            else row['left']['eligible'] != row['right']['eligible'] for row in focal]
        flags['prefix10_selected_policy_eligibility_changes'] = known_count(eligibility) if common and focal else None
        values.update({contrast+'.'+key: None if value is None else float(value) for key, value in flags.items()})
    for arm in (AFTER, POINT, FULL):
        events = [row for row in stream['multi_safe_events'] if row['arm'] == arm]
        distinct = {row['policy_id']: row for row in events}
        prefix = arm+'.multi.'
        values.update({prefix+'returned_query_events': len(events), prefix+'distinct_returned_policies': len(distinct),
            prefix+'new_queries': sum(row['query_action_kind'] == 'new' for row in events),
            prefix+'repeat_queries': sum(row['query_action_kind'] == 'repeat' for row in events)})
        counts = terminal_counts([dict(terminal_path=row['own_terminal_path']) for row in distinct.values()])
        counts = {key: value for key, value in counts.items() if key in ('observed_policies', 'on_shortlist',
            'safety_attempted', 'safety_passed', 'true_certified_deployed', 'unknown_terminal_path') or key.startswith('terminal_reason.')}
        values.update({prefix+key: value for key, value in counts.items()})
    for cell in stream['policy_terminal_paths']:
        values.update({cell['arm']+'.all_observed.'+key: value for key, value in terminal_counts(
            cell['observed'], outcomes[cell['arm']].get('actual_deployed_quality')).items()})
    return values


def summarize_measures(rows):
    keys = sorted({key for row in rows for key in row['values']})
    return dict(registered_geometries=len(rows), rows=rows,
        measures={key: estimate([row['values'].get(key) for row in rows]) for key in keys},
        by_family=[dict(domain=domain, measures={key: estimate([row['values'].get(key) for row in rows
            if row['domain'] == domain]) for key in keys}) for domain in dict.fromkeys(row['domain'] for row in rows)])


def compact_initial(stream):
    rows = []
    for contrast in ('AP', 'FA'):
        row = next((row for row in stream['query_comparisons'] if row['contrast'] == contrast and row['paid_prefix'] == 10), None)
        if row is None:
            rows.append(dict(contrast=contrast, status='unavailable'))
            continue
        focal = []
        for policy in row['focal_policies']:
            delta = policy.get('saved_left_minus_right', {})
            sampled, mean = delta.get('sampled_objective'), delta.get('objective_mean')
            focal.append(dict(policy_id=policy['policy_id'], eligibility_transition=policy.get('eligibility_transition'),
                saved_left_minus_right=delta,
                left={key: policy.get('left', {}).get(key) for key in ('present', 'observed', 'eligible', 'objective_mean',
                    'sampled_objective', 'objective_draw_innovation', 'sampled_margin', 'mean_rank_within_eligible', 'sampled_rank_within_eligible')},
                right={key: policy.get('right', {}).get(key) for key in ('present', 'observed', 'eligible', 'objective_mean',
                    'sampled_objective', 'objective_draw_innovation', 'sampled_margin', 'mean_rank_within_eligible', 'sampled_rank_within_eligible')},
                sampling_innovation_difference=None if sampled is None or mean is None else sampled-mean))
        rows.append(dict(contrast=contrast, status=row['status'], same_paid_history=row.get('same_paid_history'),
            same_physical_pool=row.get('same_physical_pool'), query_changed=row.get('query_changed'),
            selected=row.get('selected'), focal_policies=focal,
            pre_initialization_predictions=None, temporal_initialization_update=None))
    return dict(name=stream['name'], status=stream['status'], comparisons=rows)


def summarize(tasks, plan, saved_science):
    indexed = {row['seed']: row for row in tasks}
    if len(tasks) != len(plan['tasks']) or set(indexed) != {row['geometry_task_seed'] for row in plan['tasks']}:
        raise ValueError('Every registered geometry needs a path result or retained analysis failure')
    ordered = [indexed[task['geometry_task_seed']] for task in plan['tasks']]
    for task, result in zip(plan['tasks'], ordered):
        validate_task(plan, task, result)
    units = []
    for descriptor in plan['streams']:
        rows = [dict(seed=task['seed'], domain=task['domain'], values=stream_measures(next(
            stream for stream in task['streams'] if stream['name'] == descriptor['name']))) for task in ordered]
        units.append(dict(name=descriptor['name'], **summarize_measures(rows)))
    keys = sorted(set(units[0]['measures']) | set(units[1]['measures']))
    averaged = [dict(seed=a['seed'], domain=a['domain'], values={key: mean_known([
        a['values'].get(key), b['values'].get(key)]) for key in keys})
        for a, b in zip(units[0]['rows'], units[1]['rows'])]
    failures = [dict(seed=task['seed'], domain=task['domain'], stream=stream['name'],
        error=stream.get('error', task.get('error'))) for task in ordered for stream in task['streams'] if stream['status'] != 'complete']
    directions = []
    for contrast in ('AP', 'FA'):
        key = contrast+'.multi_signed.net_gain'
        rows = [dict(seed=a['seed'], domain=a['domain'], stream_net_contributions=[a['values'].get(key), b['values'].get(key)],
            directions=[sign(a['values'].get(key)), sign(b['values'].get(key))])
            for a, b in zip(units[0]['rows'], units[1]['rows'])]
        directions.append(dict(contrast=contrast, registered_geometries=len(rows), rows=rows,
            counts=dict(Counter('/'.join(row['directions']) for row in rows))))
    return dict(status='complete' if not failures else 'complete_with_analysis_failures',
        registered_geometries=len(tasks), completed_geometries=sum(row['status'] == 'complete' for row in tasks),
        registered_task_streams=2*len(tasks), completed_task_streams=2*len(tasks)-len(failures),
        registered_terminal_outcomes=2*len(tasks)*len(arm_names(plan)),
        saved_science=saved_science, source_headline=plan['source_output_dir']+'/headline.json',
        streams=units, geometry_mean=summarize_measures(averaged),
        multi_safe_signed_directions=directions,
        initialization=[dict(seed=task['seed'], domain=task['domain'], streams=[compact_initial(stream)
            for stream in task['streams']]) for task in ordered], analysis_failures=failures,
        historical_accounts=plan['historical_accounts'], statistical_unit=plan['statistical_unit'], new_work=dict(ZERO),
        limits=plan['limits']+['Path counts are repeated diagnostics; each geometry contributes its two-stream average.',
            'Signed deployment contributions are assigned once to each arm actual deployment, never once per query.',
            'Source scientific summaries, including all controls and undefined baselines, survive path-analysis failures unchanged.',
            'Saved cross-arm score differences are not unrecorded pre/post initialization updates or counterfactual substitutions.'])


def compact_summary(result):
    def statistics(measures):
        compact = {}
        for key, stat in measures.items():
            if '.multi_signed.' in key:
                compact[key] = stat
                continue
            available = stat['conditional_available']
            compact[key] = dict(registered=stat['registered_geometries'], defined=stat['available_geometries'],
                mean=stat['all_registered_mean'], available_mean=available['mean'],
                available_sum=available['positive_sum']+available['negative_sum'])
        return compact
    def group(row):
        return dict(**{key: value for key, value in row.items() if key not in ('rows', 'measures', 'by_family')},
            measures=statistics(row['measures']), by_family=[dict(domain=family['domain'],
                measures=statistics(family['measures'])) for family in row['by_family']])
    return {**{key: value for key, value in result.items() if key not in ('streams', 'geometry_mean', 'historical_accounts')},
        'streams': [group(row) for row in result['streams']], 'geometry_mean': group(result['geometry_mean'])}


def collect(plan, manifest, *, root=ROOT):
    root, output = Path(root), Path(root)/plan['output_dir']
    target = output/'completion.json'
    if target.exists():
        return read(target)
    if not (output/'registration/result.json').exists():
        raise ValueError('Register the original sealed cohort before collection')
    tasks = []
    for task in plan['tasks']:
        path = output/'cases'/f"seed_{task['geometry_task_seed']}"/'result.json'
        if not path.exists():
            raise ValueError('Every registered geometry needs analysis or retained failure before collect')
        tasks.append(read(path))
    with attempt(output/'aggregation', 'aggregate-precision-propagation-paths') as cost:
        cost.update(ZERO)
        result = summarize(tasks, plan, read(root/plan['source_output_dir']/'headline.json'))
        save(output/'summary.json', result)
        (output/'headline.json').write_text(json.dumps(compact_summary(result), separators=(',', ':'))+'\n')
    attempts = [dict(json.loads(line), ledger=str(path.relative_to(root)))
        for scope in ('registration', 'cases', 'aggregation') for path in (output/scope).rglob('attempts.jsonl')
        for line in path.read_text().splitlines() if line.strip()]
    if any(row.get(key, 0) != 0 for row in attempts for key in ZERO):
        raise ValueError('Saved-path analysis must not charge new scientific work or randomness')
    shards = [read(path) for path in (output/'dispatch').glob('shard_*/result.json')]
    checks = [dict(file=str(path.relative_to(root)), **read(path)) for path in
        sorted((output/'cluster_control/validation').glob('*/checks.json'))]
    completion = dict(status=result['status'], completed_utc=datetime.now(timezone.utc).isoformat(),
        **{key: result[key] for key in ('registered_geometries', 'completed_geometries', 'registered_task_streams',
            'completed_task_streams', 'registered_terminal_outcomes', 'analysis_failures')},
        attempts=attempts, failed_attempts=[row for row in attempts if row['status'] == 'failed'],
        recorded_analysis_cpu_seconds=sum(row['cpu_seconds'] for row in attempts),
        recorded_coordinator_cpu_seconds=sum(row['coordinator_cpu_seconds'] for row in shards),
        unclosed_attempt_cpu_seconds=None if any('unclosed_attempt_cpu_seconds' in row for row in tasks) else 0.,
        validations=checks, new_work=dict(ZERO), historical_accounts=plan['historical_accounts'],
        source_headline=result['source_headline'], headline_file=plan['output_dir']+'/headline.json',
        accounting='Count this analysis attempt and coordinator CPU once; checks and historical science stay separate. '
            'Wrapper CPU overlaps children. Previously recorded restoration unknowns remain historical, not new work.')
    save(target, completion)
    plan.update(status=result['status'], completed_utc=completion['completed_utc'],
        summary_file=plan['output_dir']+'/summary.json', completion_file=plan['output_dir']+'/completion.json')
    save(root/manifest, plan)
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('register', 'batch', 'collect'))
    parser.add_argument('--manifest', default=MANIFEST)
    parser.add_argument('--checks')
    for name in ('start', 'end', 'workers'):
        parser.add_argument('--'+name, type=int)
    args = parser.parse_args()
    if socket.gethostname().split('.')[0] not in {f'node{i:03d}' for i in range(1, 7)}:
        raise RuntimeError('Run saved-path analysis through scheduler on node001-node006')
    plan = read(ROOT/args.manifest)
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
