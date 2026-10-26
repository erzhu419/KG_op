"""Paired backend supplements from two already scored, sealed cohorts only.

No task trajectories, truth caches, models, or posterior objects are opened.
Run the real aggregation through scheduleurm on node001--node006.
"""
import argparse
from collections import Counter
from copy import deepcopy
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import time

from performance.summarize_mainline_two_comparison_development import (
    AFTER, POINT, FULL, METRICS, difference, estimate,
)
from performance.summarize_mainline_confirmation_seed_replication import mean_known, sign

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = 'paper_artifacts/mainline_frozen_backend_comparison_v1'
GP, NATIVE = 'same_source_gp', 'legacy_native_neighbors'
ORIGINAL_ARMS = (FULL, FULL + '_native_terminal', AFTER, AFTER + '_native_terminal',
                 NATIVE, GP, POINT, 'current_structural_mixture', 'initialization_only')
PERMUTED_ARMS = ('source_permuted_role_after_init_point_only', 'source_permuted_role_point_only')
COHORTS = (
    dict(name='stage85', manifest='performance/manifests/mainline_precision_confirmation_v1.json',
         output='paper_artifacts/mainline_precision_confirmation_v1', arms=ORIGINAL_ARMS),
    dict(name='stage93', manifest='performance/manifests/mainline_source_alignment_targets_v1.json',
         output='paper_artifacts/mainline_source_alignment_control_v1', arms=ORIGINAL_ARMS + PERMUTED_ARMS),
)
COMPARISONS = (('AFTER_minus_same_source_GP', GP), ('AFTER_minus_native_neighbors', NATIVE),
               ('AFTER_minus_POINT_reference', POINT))
ZERO = dict(new_simulation_calls=0, new_candidate_scores=0, new_truth_function_calls=0,
            new_source_fits=0, model_or_posterior_restorations=0, rng_draws=0)
DONE = ('complete', 'complete_with_retained_failures')


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n')


def require_sealed(plan, seal, completion, descriptor):
    """Validate the existing whole-cohort seal before opening saved metric rows."""
    tasks = {row['geometry_task_seed']: row['domain'] for row in plan['tasks']}
    streams = [row['name'] for row in plan['streams']]
    arms = [row['name'] for row in plan['terminal_arms']]
    if (len(tasks) != len(plan['tasks']) or not tasks or len(streams) != 2
            or len(set(streams)) != 2 or len(arms) != len(descriptor['arms'])
            or set(arms) != set(descriptor['arms'])):
        raise ValueError('Preserve both registered streams, geometries, and all original control arms')
    expected = {(stream, seed, arm) for stream in streams for seed in tasks for arm in arms}
    actual = [(row['stream'], row['seed'], row['arm']) for row in seal['cases']]
    budget = plan['budget']
    cells = budget.get('registered_terminal_cells', budget.get('terminal_cells'))
    cap = budget.get('maximum_verification_calls_per_cell', budget.get('verification_cap_per_cell'))
    if (seal['status'] != 'sealed' or not seal.get('truth_access_allowed')
            or completion['status'] not in DONE or cells != len(expected) or cap != 352
            or completion['registered_geometries'] != len(tasks)
            or completion['registered_terminal_cells'] != len(expected)
            or len(actual) != len(expected) or set(actual) != expected):
        raise ValueError('The complete registered terminal grid and scientific collection must already be sealed')
    if any(row['domain'] != tasks[row['seed']] or row['old']['status'] not in ('complete', 'failed', 'missing')
           for row in seal['cases']):
        raise ValueError('Retain each sealed terminal failure or missing input explicitly')
    return {(row['stream'], row['seed'], row['arm']): row for row in seal['cases']}


def _average(values):
    return {metric: mean_known([row[metric] for row in values]) for metric in METRICS}


def _same(left, right):
    if left is None or right is None:
        return left is right
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return abs(left - right) <= 1e-12


def _reference(unit):
    found = [row for row in unit['comparisons'] if row.get('protocol') == 'old' and (
        (row.get('left') == AFTER and row.get('right') == POINT)
        or row.get('name') == 'real_AFTER_minus_POINT')]
    if len(found) != 1:
        raise ValueError('Require the original saved AFTER-minus-POINT comparison')
    return found[0]


def _check_reference(rows, saved):
    original = {row['seed']: row for row in saved['rows']}
    if len(original) != len(saved['rows']) or set(original) != {row['seed'] for row in rows}:
        raise ValueError('Saved AFTER-minus-POINT geometry grid changed')
    for row in rows:
        prior = original[row['seed']]
        if prior['domain'] != row['domain'] or any(not _same(row['delta'][metric], prior['delta'][metric]) for metric in METRICS):
            raise ValueError('Saved AFTER-minus-POINT metric or pairing changed')
    for metric in METRICS:
        current, prior = estimate([row['delta'][metric] for row in rows]), saved['delta'][metric]
        if (current['available_geometries'] != prior['available_geometries']
                or current['registered_geometries'] != prior['registered_geometries']
                or not _same(current['all_registered_mean'], prior['all_registered_mean'])
                or not _same(current['conditional_available']['mean'], prior['conditional_available']['mean'])
                or not _same(current['conditional_available']['descriptive_95pct_t_interval'],
                             prior['conditional_available']['descriptive_95pct_t_interval'])):
            raise ValueError('Saved AFTER-minus-POINT estimator changed')


def _stats(rows):
    result = {}
    for metric in METRICS:
        result[metric] = estimate([row['delta'][metric] for row in rows])
        positives = [row for row in rows if sign(row['delta'][metric]) == 'positive']
        largest = max(positives, key=lambda row: (row['delta'][metric], -row['seed'])) if positives else None
        result[metric]['leave_largest_positive_geometry_out'] = None if largest is None else dict(
            excluded_seed=largest['seed'], domain=largest['domain'], excluded_delta=largest['delta'][metric],
            estimate=estimate([row['delta'][metric] for row in rows if row['seed'] != largest['seed']]))
    return result


def _effect(rows, name, right):
    families = list(dict.fromkeys(row['domain'] for row in rows))
    return dict(name=name, left=AFTER, right=right, protocol='old', rows=rows, delta=_stats(rows),
        signed_geometry_seeds={metric: {direction: [row['seed'] for row in rows if sign(row['delta'][metric]) == direction]
            for direction in ('positive', 'zero', 'negative', 'unknown')} for metric in METRICS},
        positive_gain_component=estimate([row['delta']['P'] for row in rows]),
        avoided_loss_component=estimate([None if row['delta']['L'] is None else -row['delta']['L'] for row in rows]),
        by_family=[dict(domain=domain, delta=_stats([row for row in rows if row['domain'] == domain])) for domain in families])


def _index_metrics(plan, science, sealed):
    tasks = {row['geometry_task_seed']: row['domain'] for row in plan['tasks']}
    arms = {row['name'] for row in plan['terminal_arms']}
    stream_names = {row['name'] for row in plan['streams']}
    units = {row['name']: row for row in science['streams']}
    if (science['status'] not in DONE or science['registered_geometries'] != len(tasks)
            or science['registered_terminal_cells'] != len(sealed)
            or len(units) != 2 or set(units) != stream_names):
        raise ValueError('Saved science summary must match the registered sealed cohort')
    scoring_failed = {(row['stream'], row['seed']) for row in science.get('scoring_failures', [])}
    scoring_failed_cells = {(row['stream'], row['seed'], row['arm']) for row in science.get('scoring_failed_outcomes', [])}
    indexed = {}
    for stream, unit in units.items():
        controls = {row['arm']: row for row in unit['controls']}
        if len(controls) != len(unit['controls']) or set(controls) != arms:
            raise ValueError('Keep every saved control, including controls unused by the new comparisons')
        for arm, control in controls.items():
            rows = {row['seed']: row for row in control['rows']}
            if control['protocol'] != 'old' or len(rows) != len(control['rows']) or set(rows) != set(tasks):
                raise ValueError('Saved old-terminal metric rows must retain the registered geometry grid')
            for seed, row in rows.items():
                key = (stream, seed, arm)
                values = row['values']
                terminal_status = sealed[key]['old']['status']
                scoring_failure = (stream, seed) in scoring_failed or key in scoring_failed_cells
                if row['domain'] != tasks[seed] or set(values) != set(METRICS):
                    raise ValueError('Saved family and metric definitions must remain unchanged')
                if (terminal_status != 'complete' or scoring_failure) and any(v is not None for v in values.values()):
                    raise ValueError('Execution and scoring failures must remain all-null outcomes')
                quality = [values[m] for m in ('G', 'P', 'L')]
                if any(v is None for v in quality) != all(v is None for v in quality):
                    raise ValueError('G/P/L must share the saved baseline support')
                if quality[0] is not None and not _same(quality[0], quality[1] - quality[2]):
                    raise ValueError('Saved signed quality must equal P minus L')
                status = 'scoring_failed' if scoring_failure else terminal_status
                quality_status = ('available' if quality[0] is not None else
                    'execution_or_scoring_failure' if status != 'complete' else
                    'undefined_initial_baseline' if values['C'] is not None else 'saved_metric_unavailable')
                indexed[key] = dict(values=deepcopy(values), terminal_status=terminal_status,
                    outcome_status=status, quality_status=quality_status)
    return units, indexed


def summarize_cohort(plan, science, seal, completion, descriptor):
    """Pure saved-metric adapter; fixtures may use a smaller registered grid."""
    sealed = require_sealed(plan, seal, completion, descriptor)
    units, indexed = _index_metrics(plan, science, sealed)
    stream_names = [row['name'] for row in plan['streams']]
    stream_results, independent = [], []
    for stream in stream_names:
        comparisons = []
        for name, right in COMPARISONS:
            rows = []
            for task in plan['tasks']:
                seed = task['geometry_task_seed']
                left_row, right_row = indexed[stream, seed, AFTER], indexed[stream, seed, right]
                rows.append(dict(seed=seed, domain=task['domain'], stream=stream,
                    left=deepcopy(left_row), right=deepcopy(right_row),
                    delta=difference(left_row['values'], right_row['values'])))
            comparisons.append(_effect(rows, name, right))
        _check_reference(comparisons[-1]['rows'], _reference(units[stream]))
        stream_results.append(dict(name=stream, comparisons=comparisons))
    for position, (name, right) in enumerate(COMPARISONS):
        by_stream = [{row['seed']: row for row in unit['comparisons'][position]['rows']} for unit in stream_results]
        rows = []
        for task in plan['tasks']:
            pair = [unit[task['geometry_task_seed']] for unit in by_stream]
            rows.append(dict(seed=task['geometry_task_seed'], domain=task['domain'],
                left=_average([row['left']['values'] for row in pair]),
                right=_average([row['right']['values'] for row in pair]),
                delta=_average([row['delta'] for row in pair]),
                stream_statuses=[dict(stream=row['stream'], left=row['left']['outcome_status'],
                    right=row['right']['outcome_status'], left_quality=row['left']['quality_status'],
                    right_quality=row['right']['quality_status']) for row in pair],
                available_streams={metric: sum(row['delta'][metric] is not None for row in pair) for metric in METRICS}))
        independent.append(_effect(rows, name, right))
    _check_reference(independent[-1]['rows'], _reference(science['independent']))
    original_controls = deepcopy(science['independent']['controls'])
    if {row['arm'] for row in original_controls} != set(descriptor['arms']):
        raise ValueError('Saved independent controls must preserve all original arms')
    retained = {key: deepcopy(science[key]) for key in (
        'terminal_failures', 'scoring_failures', 'scoring_failed_outcomes', 'source_numerical_limits',
        'source_warning_document', 'limits', 'historical_context', 'historical_reference') if key in science}
    if 'source_warning_document' not in retained and plan.get('source_warning_document'):
        retained['source_warning_document'] = plan['source_warning_document']
    return dict(name=descriptor['name'], status=science['status'], registered_geometries=len(plan['tasks']),
        registered_task_streams=2 * len(plan['tasks']), registered_terminal_cells=len(sealed),
        independent=dict(comparisons=independent, controls=original_controls), streams=stream_results,
        original_stream_controls=[dict(name=name, controls=deepcopy(units[name]['controls'])) for name in stream_names],
        saved_AFTER_minus_POINT_consistency='unchanged_all_metrics_rows_and_estimates',
        sealed_terminal_counts=dict(Counter(row['old']['status'] for row in seal['cases'])),
        sealed_terminal_failures=[deepcopy(row) for row in seal['cases'] if row['old']['status'] != 'complete'],
        retained_source_metadata=retained, costs=deepcopy(completion['costs']),
        historical_accounts=deepcopy(completion.get('historical_accounts', plan.get('historical_accounts'))),
        sources=dict(manifest=descriptor['manifest'], summary=descriptor['output'] + '/summary.json',
            seal=descriptor['output'] + '/execution_status.json', completion=descriptor['output'] + '/completion.json',
            costs=descriptor['output'] + '/completion.json#costs'), new_work=dict(ZERO))


def compact_estimate(value, *, concentration=False):
    available = value['conditional_available']
    result = dict(n=available['n'], registered=value['registered_geometries'],
        all_registered_mean=value['all_registered_mean'], conditional_mean=available['mean'],
        conditional_ci95=available['descriptive_95pct_t_interval'])
    if concentration:
        result.update(sign_counts={key: available[key] for key in ('positive', 'zero', 'negative')},
            unknown=value['registered_geometries'] - available['n'], largest_positive_share=available['largest_positive_share'])
        leave = value.get('leave_largest_positive_geometry_out')
        result['leave_largest_positive_out'] = None if leave is None else dict(
            seed=leave['excluded_seed'], domain=leave['domain'], delta=leave['excluded_delta'],
            **compact_estimate(leave['estimate']))
    return result


def compact_headline(result):
    cohorts = []
    for cohort in result['cohorts']:
        comparisons = []
        for index, effect in enumerate(cohort['independent']['comparisons']):
            comparisons.append(dict(name=effect['name'], left=effect['left'], right=effect['right'],
                delta={m: compact_estimate(effect['delta'][m], concentration=True) for m in METRICS},
                positive_gain_component=compact_estimate(effect['positive_gain_component']),
                avoided_loss_component=compact_estimate(effect['avoided_loss_component']),
                by_family=[dict(domain=family['domain'], delta={m: compact_estimate(family['delta'][m]) for m in METRICS})
                    for family in effect['by_family']],
                per_stream=[dict(name=unit['name'], delta={m: compact_estimate(unit['comparisons'][index]['delta'][m]) for m in METRICS})
                    for unit in cohort['streams']],
                signed_lists_and_geometry_rows='summary.json#' + cohort['name'] + '/' + effect['name']))
        accounts = cohort['costs']
        cohorts.append(dict(name=cohort['name'], status=cohort['status'], sources=cohort['sources'],
            registered_geometries=cohort['registered_geometries'], registered_terminal_cells=cohort['registered_terminal_cells'],
            comparisons=comparisons, controls=[dict(arm=row['arm'], protocol=row['protocol'],
                metrics={m: compact_estimate(row['metrics'][m]) for m in METRICS}) for row in cohort['independent']['controls']],
            sealed_terminal_counts=cohort['sealed_terminal_counts'],
            scoring_failure_count=len(cohort['retained_source_metadata'].get('scoring_failures', [])),
            saved_AFTER_minus_POINT_consistency=cohort['saved_AFTER_minus_POINT_consistency'],
            source_numerical_limits=cohort['retained_source_metadata'].get('source_numerical_limits'),
            source_warning_document=cohort['retained_source_metadata'].get('source_warning_document'),
            costs=dict(source=cohort['sources']['costs'], retained_unchanged_in='summary.json#' + cohort['name'] + '/costs',
                **{key: deepcopy(accounts[key]) for key in ('known_recorded', 'physical_target_calls',
                    'known_recorded_physical_target_calls', 'physical_accounting_scope', 'retained_source_known_recorded',
                    'retained_check_known_recorded', 'attempt_scope_counts') if key in accounts}),
            retained_failures_and_historical_accounts='summary.json#' + cohort['name']))
    return dict(status=result['status'], cohorts=cohorts, limits=result['limits'], new_work=dict(ZERO),
        detailed_summary_file='summary.json', geometry_csv_file='geometry_effects.csv',
        analysis_costs='attempts.jsonl (new aggregation CPU only; original costs are not charged again)')


def write_geometry_csv(path, cohorts):
    fields = ['cohort', 'unit', 'comparison', 'seed', 'domain'] + [f'{side}_{m}' for side in ('left', 'right', 'delta') for m in METRICS]
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for cohort in cohorts:
            for unit in [dict(name='independent', **cohort['independent'])] + cohort['streams']:
                for effect in unit['comparisons']:
                    for row in effect['rows']:
                        values = dict(cohort=cohort['name'], unit=unit['name'], comparison=effect['name'], seed=row['seed'], domain=row['domain'])
                        for side in ('left', 'right', 'delta'):
                            saved = row[side].get('values', row[side])
                            values.update({f'{side}_{m}': saved[m] for m in METRICS})
                        writer.writerow(values)


def run(*, root=ROOT, output=OUTPUT, checks, descriptors=COHORTS):
    root, directory = Path(root), Path(root) / output
    validation = read(root / checks)
    if validation.get('status') != 'passed':
        raise ValueError('Stage103 targeted substitute checks must pass before saved-data aggregation')
    if (directory / 'completion.json').exists():
        return read(directory / 'completion.json')
    if any(directory.resolve() == (root / item['output']).resolve() for item in descriptors):
        raise ValueError('New analysis output must not overwrite either frozen source cohort')
    directory.mkdir(parents=True, exist_ok=True)
    cpu, wall = time.process_time(), time.monotonic()
    record = dict(stage='stage103_saved_backend_comparison', status='failed', **ZERO)
    try:
        inputs = []
        for descriptor in descriptors:
            plan = read(root / descriptor['manifest'])
            if plan['output_dir'] != descriptor['output']:
                raise ValueError('Registered source output changed')
            seal = read(root / descriptor['output'] / 'execution_status.json')
            completion = read(root / descriptor['output'] / 'completion.json')
            require_sealed(plan, seal, completion, descriptor)
            inputs.append((descriptor, plan, seal, completion))
        # Both whole-cohort barriers are checked before either metric summary is opened.
        cohorts = [summarize_cohort(plan, read(root / descriptor['output'] / 'summary.json'), seal, completion, descriptor)
                   for descriptor, plan, seal, completion in inputs]
        result = dict(status='complete', cohorts=cohorts, new_work=dict(ZERO), limits=[
            'Post-hoc paired backend supplements; original registered primary comparisons remain unchanged.',
            'Each cohort is separate. Both observation streams must be defined for a geometry metric; missing values are not zero.',
            'Full registered means remain null when any geometry is missing; conditional paired estimates identify their own support.',
            'G = P - L on the same saved support. R and G may have different supports and are not subtracted as a decomposition.',
            '95% t intervals and leave-largest-out summaries are descriptive; no multiplicity-adjusted superiority claim.',
            'Backend contrasts concern complete frozen systems on their existing histories; they do not isolate all prior or frontend mechanisms.',
            'All original controls, failures, source warnings and historical costs remain retained; earlier 24-geometry cohorts are references only.',
            'Original physical, logical, historical and wrapper costs are not added together; this analysis records only its new aggregation CPU.'])
        save(directory / 'summary.json', result)
        save(directory / 'headline.json', compact_headline(result))
        write_geometry_csv(directory / 'geometry_effects.csv', cohorts)
        record['status'] = 'complete'
    except Exception as error:
        record['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        record.update(cpu_seconds=time.process_time() - cpu, wall_seconds=time.monotonic() - wall,
            completed_utc=datetime.now(timezone.utc).isoformat(), node=socket.gethostname())
        with (directory / 'attempts.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
    completion = dict(status='complete', cohorts=[row['name'] for row in cohorts], validation=checks,
        new_work=dict(ZERO), analysis_attempt=record, analysis_ledger=str(Path(output) / 'attempts.jsonl'),
        headline_file=str(Path(output) / 'headline.json'), detailed_summary_file=str(Path(output) / 'summary.json'),
        geometry_csv_file=str(Path(output) / 'geometry_effects.csv'))
    save(directory / 'completion.json', completion)
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', default=OUTPUT)
    parser.add_argument('--checks', required=True)
    args = parser.parse_args()
    if socket.gethostname().split('.')[0] not in {f'node{i:03d}' for i in range(1, 7)}:
        raise RuntimeError('Run actual saved-record aggregation through scheduleurm on node001--node006')
    print(json.dumps(run(root=args.root, output=args.output, checks=args.checks)))


if __name__ == '__main__':
    main()
