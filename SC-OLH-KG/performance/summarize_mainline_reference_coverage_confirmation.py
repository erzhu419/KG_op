"""Score the registered four-cell cohort after its global terminal seal."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from performance.diagnose_mainline_shared_neighbors import truth_value
from performance.mainline_reference_coverage import VARIANTS
from performance.mainline_reference_coverage_cohort import require_truth_seal
from performance.prepare_mainline_two_comparison_terminal import ZERO
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save
from performance.run_mainline_two_comparison_development import build_problem
from performance.summarize_mainline_two_comparison_confirmation import (
    initial_reference, leave_largest_positive,
)
from performance.summarize_mainline_two_comparison_development import (
    AFTER, POINT, FULL, METRICS, difference, estimate, metrics,
)

MANIFEST = ROOT / 'performance/manifests/mainline_reference_coverage_confirmation_v1.json'
EFFECTS = ('joint', 'reference_only', 'coverage_only', 'interaction')


def arm_names(plan):
    return [row['name'] if isinstance(row, dict) else row for row in plan['terminal_arms']]


def load_sealed(plan, *, root=ROOT):
    """Check the complete registered grid before opening a truth cache."""
    seal = require_truth_seal(plan, root=root)
    expected = {(t['geometry_task_seed'], arm, variant) for t in plan['tasks']
                for arm in arm_names(plan) for variant in VARIANTS}
    actual = [(r['seed'], r['arm'], r['variant']) for r in seal['cells']]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Terminal seal does not contain the complete registered four-cell grid')
    for row in seal['cells']:
        if row['status'] not in ('complete', 'failed', 'missing'):
            raise ValueError('Terminal cell is not complete or a retained failure')
        for field in ('input_file', 'result_file'):
            if row.get(field) and not (Path(root) / row[field]).is_file():
                raise ValueError('Sealed terminal artifact is absent: ' + row[field])
        if row['status'] in ('complete', 'failed') and not all(row.get(key) for key in ('input_file', 'result_file')):
            raise ValueError('Completed or failed cell must retain its input and result')
    return seal


def freeze_truth_inventory(plan, *, root=ROOT):
    """Freeze only initial10 and actual terminal policies, never the unsampled pool."""
    root, output = Path(root), Path(root) / plan['output_dir']
    seal = load_sealed(plan, root=root)
    path = output / 'truth_inventory.json'
    if path.exists():
        return require_inventory(plan, root=root)
    inputs = read(output / 'frozen_inputs.json')['rows']
    rows = []
    for task in plan['tasks']:
        seed = task['geometry_task_seed']
        cases = [row for row in seal['cells'] if row['seed'] == seed]
        sources = [row['source_case'] for row in inputs if row['seed'] == seed]
        initial_path = root / sources[0] / '..' / 'initialization.json'
        initial = read(initial_path)['history'] if initial_path.is_file() else []
        points = dict.fromkeys(tuple(row['point']) for row in initial)
        anchor = None
        for cell in cases:
            if cell.get('input_file'):
                item = read(root / cell['input_file'])
                anchor = anchor or cell['input_file']
                points.update(dict.fromkeys(tuple(row['point']) for row in item['frozen_shortlist']))
            if cell['status'] == 'complete':
                points[tuple(read(root / cell['result_file'])['deployed'])] = None
        if len(points) > 28:
            raise ValueError('Frozen truth union exceeds the registered 28-policy bound')
        rows.append(dict(seed=seed, domain=task['domain'], anchor_input=anchor,
            initial=[dict(point=row['point']) for row in initial],
            points=[list(point) for point in points], required_policies=len(points),
            initial_reference_available=len(initial) == 10))
    result = dict(status='frozen', registered_terminal_cells=len(seal['cells']),
        registered_geometries=len(rows), rows=rows,
        required_policies=sum(row['required_policies'] for row in rows),
        analytic_field_call_upper_bound=3 * sum(row['required_policies'] for row in rows),
        truth_used=False, new_truth_function_calls=0)
    save(path, result)
    return result


def require_inventory(plan, *, root=ROOT):
    # This reads only sealed metadata; truth access is still downstream.
    seal = load_sealed(plan, root=root)
    path = Path(root) / plan['output_dir'] / 'truth_inventory.json'
    if not path.is_file():
        raise ValueError('Truth union must be frozen after the global terminal seal')
    inventory = read(path)
    expected = {task['geometry_task_seed'] for task in plan['tasks']}
    if (inventory['status'] != 'frozen' or inventory['registered_terminal_cells'] != len(seal['cells'])
            or len(inventory['rows']) != len(expected)
            or {row['seed'] for row in inventory['rows']} != expected):
        raise ValueError('Frozen truth inventory lost a registered geometry')
    return inventory


def _labels(item, initial, root):
    labels = {tuple(row['point']): f'initial:{i}' for i, row in reversed(list(enumerate(initial)))}
    path = Path(root) / item['source_case'] / 'search.json'
    if path.is_file():
        for i, row in enumerate(read(path)['history']):
            labels.setdefault(tuple(row['point']), f"{item.get('history_arm', item['arm'])}:history:{i}")
    for i, row in enumerate(item['frozen_shortlist']):
        labels.setdefault(tuple(row['point']), f'terminal:{i}')
    return labels


def score_cell(cell, context, *, root=ROOT):
    record = read(Path(root) / cell['result_file']) if cell.get('result_file') else None
    item = read(Path(root) / cell['input_file']) if cell.get('input_file') else None
    result = {key: cell[key] for key in ('seed', 'domain', 'arm', 'variant', 'status')}
    result.update(history_arm=cell.get('history_arm'), metrics=metrics(failed=True),
        files={key: value for key, value in cell.items() if key.endswith('_file')},
        quality_reference_status=context['reference_status'],
        best_initial_quality=None if context['best'] is None else context['best']['quality'])
    if cell['status'] != 'complete':
        result['failure'] = record or cell.get('failure')
    if record is not None:
        cost_fields = ('logical_verification_calls', 'reused_raw_calls', 'new_verification_calls',
            'cumulative_new_verification_calls', 'failed_simulation_calls',
            'cumulative_failed_simulation_calls', 'cumulative_new_simulation_invocations')
        result['costs'] = {key: record[key] for key in cost_fields if key in record}
    if item is not None:
        labels = _labels(item, context['initial'], root)
        result['shortlist'] = [labels[tuple(row['point'])] for row in item['frozen_shortlist']]
    if cell['status'] == 'complete':
        point, report = tuple(record['deployed']), record['verification']
        truth = context['cache'].get(point)
        result.update(metrics=metrics(report['certified'], truth, context['best']),
            reported_deployed=labels[point], deployed=labels[point] if report['certified'] else None,
            selected_truth=truth, verification={key: value for key, value in report.items()
                if key not in ('attempts',) and 'fingerprint' not in key},
            safety_attempts=[{key: value for key, value in row.items() if 'fingerprint' not in key}
                for row in report.get('attempts', [])])
    return result


def score_task(plan, seed, *, root=ROOT):
    """One terminal-sealed geometry, one attempt; retained failures are not retried."""
    root = Path(root)
    inventory = require_inventory(plan, root=root)
    entry = next(row for row in inventory['rows'] if row['seed'] == seed)
    directory = root / plan['output_dir'] / 'truth' / f'seed_{seed}'
    target = directory / 'scored.json'
    if target.exists():
        return read(target)
    cache_path = directory / 'truth.json'
    cache = {tuple(row['point']): row['truth'] for row in read(cache_path)} if cache_path.exists() else {}
    if cache or (directory / 'attempts.jsonl').exists():
        raise RuntimeError('Interrupted truth attempt requires explicit review; no automatic retry')
    failure, problem, extra = None, None, {}
    with attempt(directory, 'score-reference-coverage-geometry', entry['domain']) as cost:
        cost.update(ZERO, truth_problem_constructions=0)
        try:
            if entry['points'] and entry['anchor_input']:
                problem = build_problem(read(root / entry['anchor_input']))
                cost['truth_problem_constructions'] = 1
                cost['new_rng_draws'] = None  # Factory internals are not instrumented.
                with patch.object(problem, 'simulate', side_effect=RuntimeError('Truth scoring cannot simulate')):
                    for point in entry['points']:
                        truth_value(problem, point, cache, extra, cost)
        except Exception as error:
            failure = dict(error=repr(error), retained_policies=len(cache), automatic_retry=False)
        finally:
            save(cache_path, [dict(point=list(point), truth=value) for point, value in cache.items()])
        cost['truth_failed'] = failure is not None
        best, reference_status = initial_reference(entry['initial'], cache)
        context = dict(cache=cache, initial=entry['initial'], best=best, reference_status=reference_status)
        seal = load_sealed(plan, root=root)
        outcomes = [score_cell(cell, context, root=root) for cell in seal['cells'] if cell['seed'] == seed]
        if cost['new_truth_function_calls'] > 3 * entry['required_policies']:
            raise ValueError('Truth scoring exceeded the frozen analytic-field bound')
        result = dict(status='complete' if failure is None else 'complete_with_truth_failure',
            seed=seed, domain=entry['domain'], outcomes=outcomes, failure=failure,
            quality_reference_status=reference_status, required_policies=entry['required_policies'],
            cached_policies=len(cache), new_truth_function_calls=cost['new_truth_function_calls'],
            truth_problem_constructions=cost['truth_problem_constructions'], new_simulation_calls=0,
            new_source_fits=0, posterior_restorations=0)
        save(target, result)
    return result


def aggregate(rows, fields):
    result = {}
    for field in fields:
        result[field] = {}
        for metric in METRICS:
            stat = estimate([row[field][metric] for row in rows])
            stat['leave_largest_positive_geometry_out'] = leave_largest_positive(rows, field, metric)
            result[field][metric] = stat
    return result


def _families(rows, fields):
    return [dict(domain=domain, **aggregate([r for r in rows if r['domain'] == domain], fields))
            for domain in dict.fromkeys(row['domain'] for row in rows)]


def summarize_outcomes(plan, outcomes):
    indexed = {(r['seed'], r['arm'], r['variant']): r for r in outcomes}
    rows = []
    for task in plan['tasks']:
        for arm in arm_names(plan):
            cells = {variant: indexed[task['geometry_task_seed'], arm, variant] for variant in VARIANTS}
            values = {variant: cells[variant]['metrics'] for variant in VARIANTS}
            joint = difference(values['reference_two_comparison'], values['original_old'])
            reference = difference(values['reference_old'], values['original_old'])
            coverage = difference(values['original_two_comparison'], values['original_old'])
            rows.append(dict(seed=task['geometry_task_seed'], domain=task['domain'], arm=arm,
                **values, joint=joint, reference_only=reference, coverage_only=coverage,
                interaction=difference(difference(joint, reference), coverage),
                cells={name: dict(status=cell['status'], deployed=cell.get('deployed'),
                    reported_deployed=cell.get('reported_deployed'), shortlist=cell.get('shortlist'),
                    quality_reference_status=cell['quality_reference_status']) for name, cell in cells.items()}))
    fields = (*VARIANTS, *EFFECTS)
    groups = []
    for arm in arm_names(plan):
        selected = [row for row in rows if row['arm'] == arm]
        counts = {variant: {status: sum(row['cells'][variant]['status'] == status for row in selected)
            for status in ('complete', 'failed', 'missing')} for variant in VARIANTS}
        groups.append(dict(arm=arm, outcome_counts=counts, **aggregate(selected, fields),
                           by_family=_families(selected, fields)))
    pairs = [(AFTER, POINT), (FULL, AFTER)]
    pairs.extend((arm + '_native_terminal', arm) for arm in (AFTER, FULL)
                 if arm + '_native_terminal' in arm_names(plan))
    secondary = []
    row_map = {(row['seed'], row['arm']): row for row in rows}
    for left, right in pairs:
        if left not in arm_names(plan) or right not in arm_names(plan):
            continue
        contrasts = []
        for task in plan['tasks']:
            a, b = (row_map[task['geometry_task_seed'], arm] for arm in (left, right))
            contrasts.append(dict(seed=a['seed'], domain=a['domain'],
                **{variant: difference(a[variant], b[variant]) for variant in VARIANTS},
                joint_change=difference(a['joint'], b['joint']),
                left_cells=a['cells'], right_cells=b['cells']))
        contrast_fields = (*VARIANTS, 'joint_change')
        secondary.append(dict(left=left, right=right, rows=contrasts,
            **aggregate(contrasts, contrast_fields), by_family=_families(contrasts, contrast_fields)))
    return rows, groups, secondary


def accounting(plan, seal, *, root=ROOT):
    """Count each native ledger and each arm raw union once; wrappers stay separate."""
    root, output = Path(root), Path(root) / plan['output_dir']
    attempts = []
    for path in sorted(output.rglob('attempts.jsonl')):
        attempts.extend(dict(json.loads(line), ledger=str(path.relative_to(root)))
            for line in path.read_text().splitlines() if line.strip())
    initial = sum(row.get('new_target_initial_calls', 0) for row in attempts)
    online = sum(row.get('new_target_search_calls', 0) for row in attempts)
    ledger_terminal = sum(row.get('new_verification_calls', 0) for row in attempts)
    terminal = seal['physical_verification_calls']
    if initial + online + terminal > plan['budget']['maximum_target_physical_calls']:
        raise ValueError('Successful physical spending exceeded the registered bound')
    logical = {name: dict(logical_verification_calls=0, reused_raw_calls=0,
        cumulative_new_verification_calls=0, cumulative_failed_simulation_calls=0,
        paid_rows_in_current_failed_cells=0) for name in VARIANTS}
    interruptions, unclosed_paid = [], 0
    for cell in seal['cells']:
        if not cell.get('result_file'):
            continue
        record = read(root / cell['result_file'])
        if (cell['status'] == 'failed' and record.get('subprocess_job') is not None
                and record.get('reason') == 'Terminal subprocess interrupted; no automatic replay'):
            path = str((Path(cell['result_file']).parent / 'attempts.jsonl'))
            paid_in_ledger = sum(row.get('new_verification_calls', 0)
                                 for row in attempts if row['ledger'] == path)
            missing = record['cumulative_new_verification_calls'] - paid_in_ledger
            if missing < 0:
                raise ValueError('Interrupted cell ledger exceeds its cumulative paid raw')
            unclosed_paid += missing
            interruptions.append(dict(seed=cell['seed'], arm=cell['arm'], variant=cell['variant'],
                result_file=cell['result_file'], unclosed_attempt_physical_calls=missing,
                unclosed_attempt_cpu_seconds=None, unrecorded_inflight_invocations=None))
        for key in ('logical_verification_calls', 'reused_raw_calls',
                    'cumulative_new_verification_calls', 'cumulative_failed_simulation_calls'):
            value = record.get(key, 0)
            if value is None:
                logical[cell['variant']][key] = None
            elif logical[cell['variant']][key] is not None:
                logical[cell['variant']][key] += value
        if cell['status'] == 'failed':
            logical[cell['variant']]['paid_rows_in_current_failed_cells'] += record.get('cumulative_new_verification_calls', 0)
    if ledger_terminal + unclosed_paid != terminal:
        raise ValueError('Attempt terminal spending disagrees with cumulative raw execution spending')
    unknown_search = [row for row in plan.get('execution_failures', [])
                      if row.get('call_accounting_complete') is False]
    accounting_complete = not interruptions and not unknown_search
    known_target = initial + online + terminal
    tests = []
    for path in sorted((output / 'cluster_control').rglob('checks.json')):
        tests.append(dict(file=str(path.relative_to(root)), **read(path)))
    by_stage = {}
    for row in attempts:
        stage = by_stage.setdefault(row['stage'], dict(attempts=0, failed_attempts=0,
            cpu_seconds=0., wall_seconds=0.))
        stage['attempts'] += 1
        stage['failed_attempts'] += row['status'] != 'completed'
        for key in ('cpu_seconds', 'wall_seconds'):
            stage[key] += row.get(key, 0.)
    return dict(shared_initial_physical_calls=initial, shared_online_physical_calls=online,
        unique_terminal_physical_calls=terminal,
        successful_target_physical_calls=known_target if accounting_complete else None,
        known_recorded_successful_target_physical_calls=known_target,
        call_accounting_complete=accounting_complete,
        physical_spending_interpretation='complete_recorded_calls' if accounting_complete else 'known_recorded_lower_bound',
        unclosed_attempt_physical_calls=unclosed_paid, terminal_interruptions=interruptions,
        unclosed_attempt_cpu_seconds=None if interruptions or unknown_search else 0.,
        search_histories_with_unclosed_accounting=unknown_search,
        failed_terminal_simulation_invocations=seal['failed_simulation_calls'],
        target_invocations_including_recorded_terminal_failures=(known_target + seal['failed_simulation_calls']
            if accounting_complete else None),
        known_recorded_target_invocations_including_failures=known_target + seal['failed_simulation_calls'],
        search_failed_invocation_counter=None,
        analytic_truth_field_calls=sum(row.get('new_truth_function_calls', 0) for row in attempts),
        failed_truth_attempts=sum(bool(row.get('truth_failed')) for row in attempts),
        new_source_calls=sum(row.get('new_source_calls', 0) for row in attempts),
        new_source_fits=sum(row.get('new_source_fits', 0) for row in attempts),
        search_model_and_engine_restorations=None, scoring_posterior_restorations=0,
        terminal_logical_and_raw_ownership_by_variant=logical,
        attempt_cpu_seconds=sum(row.get('cpu_seconds', 0.) for row in attempts),
        attempt_wall_seconds_sum=sum(row.get('wall_seconds', 0.) for row in attempts),
        by_stage=by_stage, failed_attempts=[row for row in attempts if row['status'] != 'completed'],
        attempt_ledger_files=sorted({row['ledger'] for row in attempts}), checks=tests,
        historical_accounts=plan.get('historical_accounts', []),
        accounting_note='Each attempt ledger is read once, including registration and failures. '
            'Shared search and same-arm raw blocks are charged once physically; expanded logical cells are separate. '
            'Checks remain separately recorded; wrapper CPU overlaps its children and is not added. '
            'Historical references are retained without summing nested totals. Search loading and failed-invocation '
            'counts are not instrumented. Interrupted raw prefixes remain charged even without a closed ledger; '
            'unrecorded in-flight calls and CPU remain unknown. Summed wall time is not cohort elapsed time.')


def compact_stat(stat):
    available = stat['conditional_available']
    sensitivity = stat['leave_largest_positive_geometry_out']
    return dict(n=stat['registered_geometries'], defined=stat['available_geometries'],
        mean=stat['all_registered_mean'], ci=stat['all_registered_descriptive_95pct_t_interval'],
        available_mean=available['mean'], positive=available['positive'], zero=available['zero'],
        negative=available['negative'], positive_sum=available['positive_sum'],
        negative_sum=available['negative_sum'], largest_positive_share=available['largest_positive_share'],
        remove_largest_positive=dict(status=sensitivity['status'],
            seed=sensitivity.get('excluded_seed'), contribution=sensitivity.get('excluded_contribution'),
            mean=sensitivity.get('all_registered_leave_one_out_mean'),
            available_mean=sensitivity.get('conditional_available_leave_one_out_mean')))


def compact_readout(result):
    """Small transfer artifact; full outcomes and raw policy coordinates stay on server."""
    def effects(group, fields):
        return {field: {key: compact_stat(group[field][key]) for key in ('C', 'G')} for field in fields}
    controls = []
    for arm in result['arms']:
        cells = {}
        for variant in VARIANTS:
            values = arm[variant]
            cells[variant] = dict(counts=arm['outcome_counts'][variant],
                **{key: dict(defined=values[key]['available_geometries'],
                    total=(None if values[key]['conditional_available']['mean'] is None else
                           values[key]['conditional_available']['mean'] * values[key]['available_geometries']))
                   for key in ('A', 'C', 'false_certificate')},
                **{key: dict(mean=values[key]['all_registered_mean'],
                    defined=values[key]['available_geometries'],
                    available_mean=values[key]['conditional_available']['mean']) for key in ('G', 'R')})
        controls.append(dict(arm=arm['arm'], cells=cells, joint=effects(arm, ('joint',))['joint']))
    secondary = []
    for group in result['secondary']:
        fields = (*VARIANTS, 'joint_change')
        secondary.append(dict(left=group['left'], right=group['right'], **effects(group, fields),
            by_family=[dict(domain=row['domain'], **effects(row, fields)) for row in group['by_family']],
            rows=[dict(seed=row['seed'], domain=row['domain'],
                       **{field: {key: row[field][key] for key in ('C', 'G')} for field in fields})
                  for row in group['rows']]))
    primary = result['primary']
    return dict(status=result['status'], registered_geometries=result['registered_geometries'],
        registered_terminal_cells=result['registered_terminal_cells'],
        completed_terminal_cells=result['completed_terminal_cells'],
        failed_terminal_cells=result['failed_terminal_cells'], missing_terminal_cells=result['missing_terminal_cells'],
        primary=dict(arm=primary['arm'], **effects(primary, EFFECTS),
            by_family=[dict(domain=row['domain'], **effects(row, EFFECTS)) for row in primary['by_family']],
            rows=[dict(seed=row['seed'], domain=row['domain'], cells=row['cells'],
                **{field: {key: row[field][key] for key in ('C', 'G')} for field in (*VARIANTS, *EFFECTS)})
                for row in result['primary_rows']]), controls=controls, secondary=secondary,
        truth_failures=[row for row in result['truth_by_task'] if row['failure'] is not None],
        costs={key: value for key, value in result['costs'].items() if key not in
            ('checks', 'attempt_ledger_files', 'historical_accounts', 'failed_attempts', 'accounting_note')},
        failed_attempt_count=len(result['costs']['failed_attempts']),
        check_costs=[{key: row[key] for key in ('file', 'status', 'cpu_seconds', 'wall_seconds') if key in row}
                     for row in result['costs']['checks']])


def collect(plan, *, root=ROOT, manifest_path=None):
    root, output = Path(root), Path(root) / plan['output_dir']
    inventory = require_inventory(plan, root=root)
    seal = load_sealed(plan, root=root)
    scored = []
    for task in plan['tasks']:
        path = output / 'truth' / f"seed_{task['geometry_task_seed']}" / 'scored.json'
        if not path.is_file():
            raise ValueError('Every geometry needs a retained scoring outcome before collection')
        scored.append(read(path))
    with attempt(output / 'summary', 'summarize-reference-coverage-confirmation') as cost:
        cost.update(ZERO)
        outcomes = [cell for row in scored for cell in row['outcomes']]
        if len(outcomes) != len(seal['cells']):
            raise ValueError('Scoring omitted a registered terminal cell')
        rows, arms, secondary = summarize_outcomes(plan, outcomes)
        complete = all(row['status'] == 'complete' for row in outcomes)
        result = dict(status='complete' if complete and all(row['failure'] is None for row in scored)
            else 'complete_with_retained_failures', registered_geometries=len(plan['tasks']),
            registered_terminal_cells=len(outcomes),
            completed_terminal_cells=sum(row['status'] == 'complete' for row in outcomes),
            failed_terminal_cells=sum(row['status'] == 'failed' for row in outcomes),
            missing_terminal_cells=sum(row['status'] == 'missing' for row in outcomes),
            rows=rows, outcomes=outcomes, arms=arms, secondary=secondary,
            primary=next(row for row in arms if row['arm'] == AFTER),
            primary_rows=[row for row in rows if row['arm'] == AFTER],
            truth_by_task=[{key: value for key, value in row.items() if key != 'outcomes'} for row in scored],
            required_truth_policies=inventory['required_policies'],
            statistical_unit=plan['statistical_unit'], primary_definition=plan['analysis']['primary'],
            metric_definitions=dict(A='Actual certificate', C='Certified and truly feasible',
                false_certificate='Certified and truly infeasible', R='C*Qdeployed',
                G='C*(Qdeployed-Qbest_true_feasible_initial10), signed', P='max(G,0)', L='max(-G,0)'),
            limitations=['Previously opened geometries remain historical and are not pooled.',
                'Undefined baselines and missing cells keep full-cohort estimates null; available subsets are labeled.',
                'Largest-positive removal is descriptive sensitivity; it never changes the primary sample.',
                'Known-family new geometries and one stream do not establish noise stability or universal superiority.',
                'Generic terminal gains do not establish propagation gains; retain within-terminal AFTER-minus-POINT.'])
    result['costs'] = accounting(plan, seal, root=root)
    save(output / 'summary.json', result)
    (output / 'readout.json').write_text(json.dumps(compact_readout(result), separators=(',', ':')) + '\n')
    completion = dict(status=result['status'], completed_utc=datetime.now(timezone.utc).isoformat(),
        registered_geometries=len(plan['tasks']), registered_terminal_cells=len(outcomes),
        completed_terminal_cells=result['completed_terminal_cells'],
        failed_terminal_cells=result['failed_terminal_cells'], missing_terminal_cells=result['missing_terminal_cells'],
        costs=result['costs'], summary_file=str((output / 'summary.json').relative_to(root)))
    save(output / 'completion.json', completion)
    if manifest_path is not None:
        plan.update(status=result['status'], completed_utc=completion['completed_utc'],
                    summary_file=completion['summary_file'])
        save(manifest_path, plan)
    return result


def score_batch(plan, manifest_path, start, end, workers, *, root=ROOT):
    root = Path(root)
    require_inventory(plan, root=root)
    tasks = plan['tasks'][start:end]
    logs = root / plan['output_dir'] / 'cluster_control' / 'truth_logs'
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONPATH=str(root))
    def run(task):
        seed = task['geometry_task_seed']
        with (logs / f'seed_{seed}.log').open('a') as stream:
            result = subprocess.run([sys.executable, '-u', '-m',
                'performance.summarize_mainline_reference_coverage_confirmation', 'score-task',
                '--manifest', str(manifest_path), '--seed', str(seed)], cwd=root, env=env,
                stdout=stream, stderr=subprocess.STDOUT)
        return dict(seed=seed, returncode=result.returncode)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run, tasks))
    if any(row['returncode'] for row in results):
        raise RuntimeError('Scoring subprocess failed; retained logs require review, no automatic retry: ' + str(results))
    return dict(status='complete', start=start, end=end, results=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze-truth', 'score-task', 'score-batch', 'collect'))
    parser.add_argument('--manifest', default=str(MANIFEST))
    parser.add_argument('--seed', type=int)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=24)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    plan = read(args.manifest)
    if args.stage == 'freeze-truth':
        result = freeze_truth_inventory(plan)
    elif args.stage == 'score-task':
        result = score_task(plan, args.seed)
    elif args.stage == 'score-batch':
        result = score_batch(plan, args.manifest, args.start, args.end, args.workers)
    else:
        result = collect(plan, manifest_path=args.manifest)
    print(json.dumps({key: value for key, value in result.items()
        if key in ('status', 'registered_terminal_cells', 'completed_terminal_cells', 'seed', 'start', 'end')}), flush=True)
