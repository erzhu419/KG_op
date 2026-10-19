"""Scheduler-only decomposition of the already scored Stage81 decision paths."""
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
from performance.mainline_reference_coverage import VARIANTS
from performance.summarize_mainline_two_comparison_development import (
    AFTER, POINT, FULL, METRICS, difference, estimate,
)
from performance.summarize_mainline_two_comparison_confirmation import leave_largest_positive


MANIFEST = 'performance/manifests/mainline_frozen_propagation_paths_v1.json'
ZERO = dict(EXISTING_ZERO, new_candidate_scores=0, truth_problem_constructions=0)


def arm_names(plan):
    return [row['name'] for row in plan['terminal_arms']]


def require_source(plan, *, root=ROOT):
    """Check completed saved science; never construct a problem or restore a model."""
    root = Path(root)
    source = read(root / plan['reference_manifest'])
    directory = root / plan['reference_output_dir']
    completion = read(directory / 'completion.json')
    seal = read(directory / 'execution_status.json')
    if source['status'] != 'complete' or completion['status'] != 'complete' or seal['status'] != 'sealed':
        raise ValueError('Stage81 completed seal and post-seal scoring are required')
    for key in ('tasks', 'terminal_arms', 'terminal_variants'):
        if source[key] != plan[key]:
            raise ValueError('Frozen Stage81 cohort changed: ' + key)
    expected = {(task['geometry_task_seed'], arm, variant) for task in plan['tasks']
                for arm in arm_names(plan) for variant in VARIANTS}
    cells = [(row['seed'], row['arm'], row['variant']) for row in seal['cells']]
    if len(cells) != len(expected) or set(cells) != expected or not seal.get('truth_access_allowed'):
        raise ValueError('Source seal must retain the full registered grid')
    for task in plan['tasks']:
        seed = task['geometry_task_seed']
        path = directory / 'truth' / f'seed_{seed}' / 'scored.json'
        if not path.exists():
            raise ValueError('Every registered source geometry must already be scored')
        result = read(path)
        actual = [(row['seed'], row['arm'], row['variant']) for row in result['outcomes']]
        task_expected = {key for key in expected if key[0] == seed}
        if (result['status'] != 'complete' or result.get('failure') is not None
                or len(actual) != len(task_expected) or set(actual) != task_expected):
            raise ValueError('Source scoring must retain all four cells and all controls')
    return dict(status='admitted', registered_geometries=len(plan['tasks']),
        registered_terminal_outcomes=len(expected), source_completion=plan['reference_output_dir'] + '/completion.json',
        source_seal=plan['reference_output_dir'] + '/execution_status.json', **ZERO)


def register(plan, manifest, checks, *, root=ROOT):
    root, output = Path(root), Path(root) / plan['output_dir']
    target = output / 'registration' / 'result.json'
    if target.exists():
        return read(target)
    if not checks or read(root / checks)['status'] != 'passed':
        raise ValueError('Targeted substitute checks must pass before saved-record analysis')
    with attempt(target.parent, 'register-frozen-propagation-paths') as cost:
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
        outcomes=[dict(seed=seed, domain=task['domain'], arm=arm, variant=variant,
            status='analysis_failed', metrics=dict.fromkeys(METRICS))
            for arm in arm_names(plan) for variant in VARIANTS],
        policies=[], histories=[], cells=[], comparisons=[], query_comparisons=[],
        missingness='Saved scientific cohort remains unchanged; this failed analysis supplies no inferred values.')


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
        with attempt(directory, 'analyze-frozen-propagation-paths', task['domain']) as cost:
            cost.update(ZERO, seed=seed)
            from performance.mainline_frozen_propagation_paths import analyze_task
            result = analyze_task(root, plan, seed)
            validate_task(plan, task, result)
            save(target, result)
    except Exception:
        result = failed_task(plan, task, traceback.format_exc())
        save(target, result)
    return dict(seed=seed, status=result['status'])


def validate_task(plan, task, result):
    expected = {(arm, variant) for arm in arm_names(plan) for variant in VARIANTS}
    rows = result['outcomes']
    actual = [(row['arm'], row['variant']) for row in rows]
    if (result['seed'] != task['geometry_task_seed'] or len(actual) != len(expected)
            or set(actual) != expected):
        raise ValueError('Analysis must retain every registered arm and terminal cell')
    for row in rows:
        if set(METRICS) - row['metrics'].keys():
            raise ValueError('Saved outcome metrics must preserve undefined fields')


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


def metric_stats(rows, field):
    result = {}
    for metric in METRICS:
        stat = estimate([row[field][metric] for row in rows])
        stat['leave_largest_positive_geometry_out'] = leave_largest_positive(rows, field, metric)
        result[metric] = stat
    return result


def decomposition(left, right):
    delta = difference(left, right)
    positive, loss = delta['P'], None if delta['L'] is None else -delta['L']
    if delta['G'] is not None and positive is not None and loss is not None:
        if abs(delta['G'] - positive - loss) > 1e-12:
            raise ValueError('Saved signed gain does not equal positive gain minus deployment loss')
    return dict(delta=delta, positive_gain_component=positive, avoided_loss_component=loss,
        interpretation='Metric identity, not a causal decomposition of query discovery.')


def summarize(tasks, plan):
    task_index = {row['seed']: row for row in tasks}
    expected = {task['geometry_task_seed'] for task in plan['tasks']}
    if len(tasks) != len(expected) or set(task_index) != expected:
        raise ValueError('Every registered geometry needs an analysis or a retained failure')
    for task in plan['tasks']:
        validate_task(plan, task, task_index[task['geometry_task_seed']])
    outcomes = [row for task in tasks for row in task['outcomes']]
    indexed = {(row['seed'], row['arm'], row['variant']): row for row in outcomes}
    controls = []
    for arm in arm_names(plan):
        for variant in VARIANTS:
            rows = [dict(seed=task['geometry_task_seed'], domain=task['domain'],
                values=indexed[task['geometry_task_seed'], arm, variant]['metrics']) for task in plan['tasks']]
            controls.append(dict(arm=arm, variant=variant, metrics=metric_stats(rows, 'values'),
                outcome_counts=dict(Counter(indexed[task['geometry_task_seed'], arm, variant]['status']
                                            for task in plan['tasks'])),
                by_family=[dict(domain=domain, metrics=metric_stats([row for row in rows if row['domain'] == domain], 'values'))
                           for domain in dict.fromkeys(row['domain'] for row in rows)]))
    comparisons = []
    for left, right in ((AFTER, POINT), (FULL, AFTER)):
        if left not in arm_names(plan) or right not in arm_names(plan):
            continue
        for variant in VARIANTS:
            rows = []
            for task in plan['tasks']:
                seed = task['geometry_task_seed']
                a, b = (indexed[seed, arm, variant] for arm in (left, right))
                row = dict(seed=seed, domain=task['domain'], analysis_status=task_index[seed]['status'],
                    left_status=a['status'], right_status=b['status'], left=a['metrics'], right=b['metrics'],
                    **decomposition(a['metrics'], b['metrics']),
                    details_file=f"{plan['output_dir']}/cases/seed_{seed}/result.json")
                rows.append(row)
            comparisons.append(dict(left=left, right=right, variant=variant, rows=rows,
                delta=metric_stats(rows, 'delta'),
                positive_gain_component=estimate([row['positive_gain_component'] for row in rows]),
                avoided_loss_component=estimate([row['avoided_loss_component'] for row in rows]),
                by_family=[dict(domain=domain, delta=metric_stats([row for row in rows if row['domain'] == domain], 'delta'),
                    positive_gain_component=estimate([row['positive_gain_component'] for row in rows if row['domain'] == domain]),
                    avoided_loss_component=estimate([row['avoided_loss_component'] for row in rows if row['domain'] == domain]))
                    for domain in dict.fromkeys(row['domain'] for row in rows)]))
    primary = next(row for row in comparisons if (row['left'], row['right'], row['variant']) == (AFTER, POINT, 'original_old'))
    no_certificate = [row['seed'] for row in primary['rows'] if row['left']['A'] == 0]
    return dict(status='complete' if all(row['status'] == 'complete' for row in tasks) else 'complete_with_analysis_failures',
        registered_geometries=len(plan['tasks']), completed_geometries=sum(row['status'] == 'complete' for row in tasks),
        registered_terminal_outcomes=len(outcomes), controls=controls, comparisons=comparisons,
        primary=primary, after_no_certificate_seeds=no_certificate,
        analysis_failures=[{key: task.get(key) for key in ('seed', 'domain', 'error')} for task in tasks if task['status'] != 'complete'],
        historical_accounts=plan['historical_accounts'], statistical_unit=plan['statistical_unit'],
        limits=plan['limits'], new_work=dict(ZERO))


def compact_estimate(stat):
    result = {key: stat[key] for key in ('registered_geometries', 'available_geometries', 'all_registered_mean',
        'all_registered_descriptive_95pct_t_interval', 'conditional_available')}
    if 'leave_largest_positive_geometry_out' in stat:
        result['leave_largest_positive_geometry_out'] = stat['leave_largest_positive_geometry_out']
    return result


def compact_task(task):
    """Keep the two deployed/reported policy paths and saved AP query comparisons."""
    if task['status'] != 'complete':
        return {key: task.get(key) for key in ('seed', 'domain', 'status', 'error')}
    outcomes = {(row['arm'], row['variant']): row for row in task['outcomes']}
    focal = {outcomes[arm, 'original_old'].get(key) for arm in (AFTER, POINT)
             for key in ('reported_policy_id', 'deployed_policy_id')} - {None}
    policies = [{key: row.get(key) for key in ('policy_id', 'initial_indices', 'origin',
        'observed_locations', 'truth', 'truth_status')} for row in task['policies'] if row['policy_id'] in focal]
    histories = []
    for history in task['histories']:
        if history['history_arm'] not in (AFTER, POINT):
            continue
        histories.append(dict(history_arm=history['history_arm'], status=history['status'],
            queries=[{key: query.get(key) for key in ('paid_prefix', 'selected_policy_id',
                'actual_query_returned', 'selected_before_observed', 'candidate_count', 'pool_truth', 'selection_replay')}
                for query in history['queries']],
            focal_policy_paths=[row for row in history['policy_paths'] if row['policy_id'] in focal],
            observed_truth=history['observed_truth']))
    terminals = []
    for cell in task['cells']:
        if cell['arm'] not in (AFTER, POINT):
            continue
        outcome = outcomes[cell['arm'], cell['variant']]
        terminals.append(dict(arm=cell['arm'], variant=cell['variant'], status=cell['status'],
            metrics=outcome['metrics'], actual_deployed_quality=outcome['actual_deployed_quality'],
            reported_policy_id=cell['reported_policy_id'], deployed_policy_id=cell['deployed_policy_id'],
            newly_queried_deployment=outcome['newly_queried_deployment'],
            deployment_query_indices=outcome['deployment_query_indices'],
            shortlist_policy_ids=cell['shortlist_policy_ids'], quality_sets=cell['quality_sets'],
            original_selection_record_status=cell['original_selection_record_status'],
            selection_issues=cell['selection_issues'],
            focal_observed=[row for row in cell['observed'] if row['policy_id'] in focal],
            safety_attempts=[{key: row[key] for key in ('candidate_index', 'certified', 'upper_margin',
                'verification_budget') if key in row} for row in cell['safety_attempts']],
            actual_objective_comparisons=[{key: row[key] for key in ('block_index', 'challenger_index',
                'incumbent_index', 'winner_index', 'enabled', 'status', 'sample_mean_difference',
                'one_sided_upper_confidence_bound', 'challenger_dominates', 'simulation_calls') if key in row}
                for row in cell['actual_objective_comparisons']]))
    return dict(seed=task['seed'], domain=task['domain'], status=task['status'],
        policies=policies, histories=histories, terminals=terminals,
        query_comparisons=[row for row in task['query_comparisons'] if (row['left'], row['right']) == (AFTER, POINT)],
        coverage=task.get('coverage'), limits=task.get('limits', []))


def compact_summary(result, tasks):
    controls = [dict(arm=row['arm'], variant=row['variant'], outcome_counts=row['outcome_counts'],
        metrics={metric: dict(mean=row['metrics'][metric]['all_registered_mean'],
            defined=row['metrics'][metric]['available_geometries'],
            available_mean=row['metrics'][metric]['conditional_available']['mean']) for metric in METRICS})
        for row in result['controls']]
    comparisons = []
    for row in result['comparisons']:
        selected = dict(left=row['left'], right=row['right'], variant=row['variant'],
            delta={key: compact_estimate(row['delta'][key]) for key in ('C', 'G', 'P', 'L')},
            positive_gain_component=row['positive_gain_component'], avoided_loss_component=row['avoided_loss_component'])
        if row['left'] == AFTER:
            selected.update(rows=row['rows'], by_family=[dict(domain=family['domain'],
                delta={key: compact_estimate(family['delta'][key]) for key in ('C', 'G', 'P', 'L')},
                positive_gain_component=family['positive_gain_component'], avoided_loss_component=family['avoided_loss_component'])
                for family in row['by_family']])
        comparisons.append(selected)
    return dict(status=result['status'], registered_geometries=result['registered_geometries'],
        completed_geometries=result['completed_geometries'], registered_terminal_outcomes=result['registered_terminal_outcomes'],
        controls=controls, comparisons=comparisons,
        primary=next(row for row in comparisons if row['left'] == AFTER and row['variant'] == 'original_old'),
        after_no_certificate_seeds=result['after_no_certificate_seeds'], analysis_failures=result['analysis_failures'],
        paths=[compact_task(task) for task in tasks],
        task_path_files=[dict(seed=row['seed'], file=f"cases/seed_{row['seed']}/result.json") for row in tasks],
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
    with attempt(output / 'aggregation', 'aggregate-frozen-propagation-paths') as cost:
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
    if any(row.get(key, 0) for row in attempts for key in ZERO):
        raise ValueError('Saved-record analysis ledger contains newly charged scientific work')
    completion = dict(status=result['status'], completed_utc=datetime.now(timezone.utc).isoformat(),
        registered_geometries=len(plan['tasks']), completed_geometries=result['completed_geometries'],
        registered_terminal_outcomes=result['registered_terminal_outcomes'],
        analysis_failures=result['analysis_failures'], attempts=attempts,
        failed_attempts=[row for row in attempts if row['status'] == 'failed'],
        recorded_analysis_cpu_seconds=sum(row['cpu_seconds'] for row in attempts),
        recorded_coordinator_cpu_seconds=sum(row['coordinator_cpu_seconds'] for row in shards),
        unclosed_attempt_cpu_seconds=None if any('unclosed_attempt_cpu_seconds' in row for row in tasks) else 0.,
        validations=checks, new_work=dict(ZERO), historical_accounts=plan['historical_accounts'],
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
