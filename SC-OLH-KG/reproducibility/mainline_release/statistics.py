"""Recompute the sealed current-mainline tables from portable cell records only.

Scientific aggregation of the full release runs on the existing CPU scheduler.
The functions are also usable on small deterministic fixtures in any environment.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from math import sqrt
from pathlib import Path
from statistics import fmean, stdev

METRICS = ('A', 'C', 'false_certificate', 'R', 'G', 'P', 'L')
AFTER, POINT, FULL = ('legacy_role_after_init_point_only', 'legacy_role_point_only', 'legacy_role_full')
GP, NATIVE = 'same_source_gp', 'legacy_native_neighbors'
PA, PP = 'source_permuted_role_after_init_point_only', 'source_permuted_role_point_only'
CONTRASTS = (
    ('AFTER_minus_POINT', ((AFTER, 1), (POINT, -1)), 'registered_primary'),
    ('FULL_minus_AFTER', ((FULL, 1), (AFTER, -1)), 'registered_secondary'),
    ('AFTER_minus_same_source_GP', ((AFTER, 1), (GP, -1)), 'stage103_post_hoc'),
    ('AFTER_minus_native_neighbors', ((AFTER, 1), (NATIVE, -1)), 'stage103_post_hoc'),
    ('real_AFTER_minus_permuted_AFTER', ((AFTER, 1), (PA, -1)), 'source_primary'),
    ('source_by_propagation_interaction', ((AFTER, 1), (POINT, -1), (PA, -1), (PP, 1)), 'source_key_secondary'),
    ('permuted_AFTER_minus_POINT', ((PA, 1), (PP, -1)), 'source_secondary'),
    ('real_POINT_minus_permuted_POINT', ((POINT, 1), (PP, -1)), 'source_secondary'),
)


def read_json(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'wt', encoding='utf-8') as stream:
        json.dump(value, stream, separators=(',', ':'), allow_nan=False)
        stream.write('\n')


def complete_mean(values):
    """Both registered observation streams are required; never average one alone."""
    return None if any(value is None for value in values) else fmean(values)


def estimate(values):
    """Individual descriptive t interval, with geometry as the independent unit."""
    known = [value for value in values if value is not None]
    n, registered = len(known), len(values)
    center = fmean(known) if n else None
    interval = None
    if n > 1:
        from scipy.stats import t
        width = float(t.ppf(.975, n - 1)) * stdev(known) / sqrt(n)
        interval = [center - width, center + width]
    positive = [value for value in known if value > 1e-12]
    negative = [value for value in known if value < -1e-12]
    return dict(registered_geometries=registered, available_geometries=n,
        all_registered_mean=center if n == registered else None,
        all_registered_descriptive_95pct_t_interval=interval if n == registered else None,
        conditional_available=dict(n=n, mean=center, descriptive_95pct_t_interval=interval,
            degrees_of_freedom=max(n - 1, 0), positive=len(positive), negative=len(negative),
            zero=n - len(positive) - len(negative), positive_sum=sum(positive), negative_sum=sum(negative),
            largest_positive_share=max(positive) / sum(positive) if positive else None))


def validate(snapshot):
    indexed = {}
    for row in snapshot['cells']:
        key = row['cohort'], row['stream'], row['seed'], row['arm']
        if key in indexed:
            raise ValueError('Duplicate registered cell')
        values = row['metrics']
        if set(values) != set(METRICS):
            raise ValueError('Every cell must carry all original metrics')
        if row['status'] != 'complete' and any(value is not None for value in values.values()):
            raise ValueError('Failure metrics must remain null')
        if row['terminal_status'] != 'complete' and row['status'] == 'complete':
            raise ValueError('A failed terminal cannot become a completed outcome')
        for metric in ('A', 'C', 'false_certificate'):
            if values[metric] is not None and values[metric] not in (0, 1):
                raise ValueError('Cell certification endpoints must be binary')
        if all(values[metric] is not None for metric in ('A', 'C', 'false_certificate')):
            if values['A'] - values['C'] != values['false_certificate']:
                raise ValueError('False certificate must equal A minus C')
        quality = [values[metric] for metric in ('G', 'P', 'L')]
        if any(value is None for value in quality) != all(value is None for value in quality):
            raise ValueError('G/P/L must retain a common baseline support')
        if quality[0] is not None and abs(quality[0] - quality[1] + quality[2]) > 1e-12:
            raise ValueError('G must equal P minus L at the cell level')
        if row.get('baseline_status') == 'undefined_initial_baseline' and quality[0] is not None:
            raise ValueError('An undefined baseline cannot acquire a gain')
        indexed[key] = row
    expected = set()
    for cohort in snapshot['cohorts']:
        if len(cohort['streams']) != 2 or len(set(cohort['streams'])) != 2:
            raise ValueError('Exactly two registered observation streams are required')
        for geometry in cohort['geometries']:
            for stream in cohort['streams']:
                for arm in cohort['arms']:
                    key = cohort['name'], stream, geometry['seed'], arm
                    expected.add(key)
                    if key not in indexed or indexed[key]['domain'] != geometry['domain']:
                        raise ValueError('Registered grid or geometry family changed')
    if set(indexed) != expected:
        raise ValueError('Every registered cell, including failures, must be retained')
    return indexed


def _linear(indexed, cohort, seed, stream, terms, metric):
    values = [(indexed[cohort, stream, seed, arm]['metrics'][metric], weight) for arm, weight in terms]
    return None if any(value is None for value, _ in values) else sum(value * weight for value, weight in values)


def _summary(rows):
    result = {}
    for metric in METRICS:
        values = [row['values'][metric] for row in rows]
        result[metric] = estimate(values)
        positives = [row for row in rows if row['values'][metric] is not None and row['values'][metric] > 1e-12]
        largest = max(positives, key=lambda row: (row['values'][metric], -row['seed'])) if positives else None
        result[metric]['leave_largest_positive_geometry_out'] = None if largest is None else dict(
            excluded_seed=largest['seed'], domain=largest['domain'], excluded_delta=largest['values'][metric],
            estimate=estimate([row['values'][metric] for row in rows if row['seed'] != largest['seed']]))
    return result


def _rows(indexed, cohort, terms, streams):
    return [dict(seed=geometry['seed'], domain=geometry['domain'], values={metric: complete_mean([
        _linear(indexed, cohort['name'], geometry['seed'], stream, terms, metric) for stream in streams])
        for metric in METRICS}) for geometry in cohort['geometries']]


def completion_bounds(indexed, cohort, terms):
    """Sharp finite registered-sample C bounds; these are not confidence intervals.

    Each missing binary cell ranges independently over {0,1}. Averaging the two
    streams inside each geometry preserves the registered equal-weight design.
    No missing cell is modified or used in a point estimate.
    """
    geometry_bounds = []
    for geometry in cohort['geometries']:
        low = high = 0.
        for stream in cohort['streams']:
            for arm, weight in terms:
                value = indexed[cohort['name'], stream, geometry['seed'], arm]['metrics']['C']
                low += weight * value if value is not None else min(weight, 0)
                high += weight * value if value is not None else max(weight, 0)
        geometry_bounds.append(dict(seed=geometry['seed'], low=low / 2, high=high / 2))
    return dict(endpoint='C', registered_geometries=len(geometry_bounds),
        mean_bounds=[fmean(row['low'] for row in geometry_bounds), fmean(row['high'] for row in geometry_bounds)],
        interpretation='Finite registered-sample arbitrary binary completion bounds; not a confidence interval or imputation.',
        point_estimate_unchanged=True, rows=geometry_bounds)


def summarize_cohort(indexed, cohort):
    units = []
    for unit, streams in [('independent', cohort['streams'])] + [(name, [name]) for name in cohort['streams']]:
        controls = []
        for arm in cohort['arms']:
            rows = _rows(indexed, cohort, ((arm, 1),), streams)
            controls.append(dict(arm=arm, rows=rows, metrics=_summary(rows)))
        comparisons = []
        for name, terms, role in CONTRASTS:
            if not all(arm in cohort['arms'] for arm, _ in terms):
                continue
            rows = _rows(indexed, cohort, terms, streams)
            comparisons.append(dict(name=name, inference_role=role, terms=[dict(arm=a, coefficient=w) for a, w in terms],
                rows=rows, delta=_summary(rows),
                positive_gain_component=estimate([row['values']['P'] for row in rows]),
                avoided_loss_component=estimate([None if row['values']['L'] is None else -row['values']['L'] for row in rows]),
                by_family=[dict(domain=domain, delta=_summary([row for row in rows if row['domain'] == domain]))
                    for domain in dict.fromkeys(row['domain'] for row in rows)]))
        units.append(dict(name=unit, controls=controls, comparisons=comparisons))
    sensitivity = [dict(name=name, **completion_bounds(indexed, cohort, terms)) for name, terms, _ in CONTRASTS
        if name in ('real_AFTER_minus_permuted_AFTER', 'source_by_propagation_interaction')
        and all(arm in cohort['arms'] for arm, _ in terms)]
    cells = [row for key, row in indexed.items() if key[0] == cohort['name']]
    return dict(name=cohort['name'], registered_geometries=len(cohort['geometries']),
        registered_cells=len(cells), terminal_counts=dict(Counter(row['terminal_status'] for row in cells)),
        outcome_counts=dict(Counter(row['status'] for row in cells)),
        baseline_counts=dict(Counter(row.get('baseline_status', 'unknown') for row in cells)),
        units=units, missing_sensitivity=sensitivity)


def cost_table(snapshot):
    """Keep physical attempts, successful logical cells and shared costs separate."""
    fields = ('new_source_calls', 'new_source_fits', 'new_target_initial_calls',
              'new_target_search_calls', 'new_verification_calls', 'new_truth_function_calls', 'cpu_seconds')
    attempts = snapshot.get('attempts', [])
    seen = set()
    for row in attempts:
        key = row['cohort'], row['ledger'], row['ledger_row']
        if key in seen:
            raise ValueError('A physical attempt ledger row would be charged twice')
        seen.add(key)
    tables = []
    for cohort in snapshot['cohorts']:
        rows = [row for row in attempts if row['cohort'] == cohort['name']]
        groups = {}
        for row in rows:
            scope = row['accounting_scope']
            # Initialization/neighbor preparation is shared by arms. A CPU row
            # bearing an arm can be attributed, but is never divided into cells.
            shared_initial = row.get('stage') == 'search' and row.get('arm') == 'initialization_only'
            arm = row.get('arm') if scope == 'target' and not shared_initial else None
            groups.setdefault((scope, arm), []).append(row)
        summaries = []
        for (scope, arm), group in groups.items():
            totals = {field: sum(row[field] for row in group if row.get(field) is not None)
                if any(row.get(field) is not None for row in group) else None for field in fields}
            summaries.append(dict(scope=scope, arm=arm, attempt_count=len(group),
                failed_attempts=sum(row.get('status') == 'failed' for row in group),
                failed_cpu_seconds=sum(row.get('cpu_seconds') or 0 for row in group if row.get('status') == 'failed'),
                known_recorded=totals,
                explicit_unknown_fields={field: sum(field in row and row[field] is None for row in group) for field in fields},
                unrecorded_fields={field: sum(field not in row for row in group) for field in fields},
                peak_memory_bytes=None, peak_memory_status='not_recorded_in_historical_attempts'))
        logical = []
        for arm in cohort['arms']:
            cells = [row for row in snapshot['cells'] if row['cohort'] == cohort['name'] and row['arm'] == arm]
            recorded = [row['verification_calls'] for row in cells if row.get('verification_calls') is not None]
            logical.append(dict(arm=arm, registered_cells=len(cells),
                completed_cells=sum(row['status'] == 'complete' for row in cells),
                known_cell_verification_calls=sum(recorded), cells_with_recorded_verification_calls=len(recorded),
                all_cell_verification_calls=sum(recorded) if len(recorded) == len(cells) else None,
                successful_search_calls_per_cell=10 if arm == 'initialization_only' else 13,
                verification_cap=352, logical_total_cap=362 if arm == 'initialization_only' else 365,
                physical_search_cost_owner='history_arm; shared by role/native terminal variants',
                physical_source_cost_owner='shared_source_accounts'))
        tables.append(dict(cohort=cohort['name'], physical_attempt_groups=summaries, logical_cells=logical,
            retained_original_accounts=cohort.get('saved_cost_accounts'),
            note='Physical attempts, cell logical calls, source reuse and wrapper CPU overlap and must not be added. Unknown historical fees remain unknown.'))
    return tables


def recompute(snapshot):
    indexed = validate(snapshot)
    return dict(format='mainline_release_statistics_v1', cohorts=[summarize_cohort(indexed, row) for row in snapshot['cohorts']],
        costs=cost_table(snapshot), interval_policy='Individual descriptive 95% t intervals; no multiplicity-adjusted superiority claims.',
        statistical_unit='Exactly two required observation streams averaged inside each geometry; cohorts remain separate.',
        source_reuse=dict(paired_calls_per_archive=384, scope='One-time shared frozen source archive; not charged per target cell.'),
        false_certificate_policy='Zero observed false certificates does not imply a zero population error probability.',
        new_scientific_work=dict(simulation_calls=0, source_fits=0, posterior_restorations=0, rng_draws=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cells', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = recompute(read_json(args.cells))
    write_json(args.output, result)
    print(json.dumps(dict(status='complete', cohorts=[dict(name=row['name'], cells=row['registered_cells'],
        terminal_counts=row['terminal_counts']) for row in result['cohorts']]), separators=(',', ':')))


if __name__ == '__main__':
    main()
