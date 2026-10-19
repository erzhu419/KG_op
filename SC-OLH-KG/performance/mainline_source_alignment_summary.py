"""Summarize already scored, globally sealed source-alignment cells.

The input is the Stage85 ``score_task`` result schema. This module neither scores
policies nor loads models; the file adapter checks the whole terminal seal before
opening any scored geometry. Run real saved-record aggregation via scheduleurm.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from performance.summarize_mainline_two_comparison_development import (
    AFTER, POINT, FULL, METRICS, estimate,
)
from performance.analyze_mainline_frozen_propagation_paths import metric_stats
from performance.summarize_mainline_confirmation_seed_replication import mean_known, sign


PERMUTED_AFTER = 'source_permuted_role_after_init_point_only'
PERMUTED_POINT = 'source_permuted_role_point_only'
ARMS = (FULL, FULL + '_native_terminal', AFTER, AFTER + '_native_terminal',
        'legacy_native_neighbors', 'same_source_gp', POINT,
        'current_structural_mixture', 'initialization_only', PERMUTED_AFTER, PERMUTED_POINT)
CONTRASTS = (
    ('real_AFTER_minus_permuted_AFTER', ((AFTER, 1), (PERMUTED_AFTER, -1)), 'primary'),
    ('source_by_propagation_interaction',
     ((AFTER, 1), (POINT, -1), (PERMUTED_AFTER, -1), (PERMUTED_POINT, 1)), 'key_secondary'),
    ('real_AFTER_minus_POINT', ((AFTER, 1), (POINT, -1)), 'secondary'),
    ('permuted_AFTER_minus_POINT', ((PERMUTED_AFTER, 1), (PERMUTED_POINT, -1)), 'secondary'),
    ('real_POINT_minus_permuted_POINT', ((POINT, 1), (PERMUTED_POINT, -1)), 'secondary'),
    ('real_FULL_minus_AFTER', ((FULL, 1), (AFTER, -1)), 'secondary'),
)
ZERO = dict(simulation_calls=0, terminal_calls=0, truth_fields=0,
            candidate_score_evaluations=0, source_fit_pipelines=0,
            model_or_posterior_restorations=0, raw_replays=0, rng_draws=0)


def require_seal(plan, seal):
    """Require the materialized registered grid, including every retained failure."""
    tasks, streams = plan['tasks'], plan['streams']
    names = [row['name'] for row in plan['terminal_arms']]
    seeds = [row['geometry_task_seed'] for row in tasks]
    stream_names = [row['name'] for row in streams]
    if len(names) != 11 or set(names) != set(ARMS):
        raise ValueError('Preserve the original nine terminal arms and both permuted arms')
    if len(seeds) != len(set(seeds)) or not seeds or len(set(stream_names)) != 2 or len(stream_names) != 2:
        raise ValueError('Require unique registered geometries and exactly two observation streams')
    expected = {(name, seed, arm) for name in stream_names for seed in seeds for arm in names}
    actual = [(row['stream'], row['seed'], row['arm']) for row in seal['cases']]
    if (plan['budget']['terminal_cells'] != len(expected)
            or plan['budget']['verification_cap_per_cell'] != 352
            or seal['status'] != 'sealed' or not seal.get('truth_access_allowed')
            or len(actual) != len(expected) or set(actual) != expected):
        raise ValueError('All registered 11-arm old-terminal cells must be globally sealed before summary')
    domains = {row['geometry_task_seed']: row['domain'] for row in tasks}
    for row in seal['cases']:
        if row['old']['status'] not in ('complete', 'failed', 'missing') or row['domain'] != domains[row['seed']]:
            raise ValueError('Seal must retain completed old terminals or explicit failures with the registered family')
    return {(row['stream'], row['seed'], row['arm']): row for row in seal['cases']}


def _index_outcomes(plan, tasks, sealed):
    registered = {row['geometry_task_seed']: row['domain'] for row in plan['tasks']}
    indexed_tasks = {row['seed']: row for row in tasks}
    if len(tasks) != len(registered) or set(indexed_tasks) != set(registered):
        raise ValueError('Every registered geometry needs scored records or retained scoring failure')
    stream_names = {row['name'] for row in plan['streams']}
    indexed = {}
    for task in tasks:
        if (task['domain'] != registered[task['seed']] or len(task['streams']) != 2
                or {row['name'] for row in task['streams']} != stream_names):
            raise ValueError('Every geometry must retain its registered family and both streams')
        for stream in task['streams']:
            for row in stream['outcomes']:
                key = (stream['name'], row['seed'], row['arm'])
                if (key in indexed or key not in sealed or row['seed'] != task['seed']
                        or row['domain'] != task['domain'] or row['protocol'] != 'old'):
                    raise ValueError('Scored records must match the complete registered old-terminal grid')
                values = row['metrics']
                if set(values) != set(METRICS):
                    raise ValueError('Retain every original outcome metric, including false certificates')
                old_status = sealed[key]['old']['status']
                if (old_status != 'complete' and row['status'] not in (old_status, 'scoring_failed')
                        or row['status'] != 'complete' and any(value is not None for value in values.values())):
                    raise ValueError('Execution or scoring failures must remain null, never zero outcomes')
                if ('best_initial_quality' in row and row['best_initial_quality'] is None
                        and any(values[metric] is not None for metric in ('G', 'P', 'L'))):
                    raise ValueError('An undefined initial feasible baseline must retain null G/P/L')
                if all(values[metric] is not None for metric in ('G', 'P', 'L')):
                    if abs(values['G'] - values['P'] + values['L']) > 1e-12:
                        raise ValueError('Saved G must equal saved P minus L')
                budget = row.get('verification_budget')
                if budget is not None and budget > 352:
                    raise ValueError('A completed-cell old-terminal budget exceeds the frozen 352 cap')
                indexed[key] = row
    if set(indexed) != set(sealed):
        raise ValueError('Retain all 11 arms even when scoring or terminal execution fails')
    return indexed


def _metric_average(rows):
    return {metric: mean_known([row[metric] for row in rows]) for metric in METRICS}


def _linear_metrics(arms, terms):
    # In particular, the interaction has the same four-arm subset per metric.
    return {metric: None if any(arms[arm][metric] is None for arm, _ in terms)
            else sum(weight * arms[arm][metric] for arm, weight in terms) for metric in METRICS}


def _effect_row(seed, domain, arms, terms, **details):
    delta = _linear_metrics(arms, terms)
    return dict(seed=seed, domain=domain, arm_metrics=arms, delta=delta,
        positive_gain_component=delta['P'],
        avoided_loss_component=None if delta['L'] is None else -delta['L'], **details)


def _signed_members(rows):
    return {metric: {direction: [row['seed'] for row in rows if sign(row['delta'][metric]) == direction]
                    for direction in ('positive', 'zero', 'negative', 'unknown')} for metric in METRICS}


def _effect_summary(name, terms, role, rows):
    families = list(dict.fromkeys(row['domain'] for row in rows))
    return dict(name=name, protocol='old', inference_role=role,
        terms=[dict(arm=arm, coefficient=weight) for arm, weight in terms], rows=rows,
        delta=metric_stats(rows, 'delta'), signed_geometry_seeds=_signed_members(rows),
        positive_gain_component=estimate([row['positive_gain_component'] for row in rows]),
        avoided_loss_component=estimate([row['avoided_loss_component'] for row in rows]),
        by_family=[dict(domain=domain,
            delta=metric_stats([row for row in rows if row['domain'] == domain], 'delta'),
            positive_gain_component=estimate([row['positive_gain_component'] for row in rows if row['domain'] == domain]),
            avoided_loss_component=estimate([row['avoided_loss_component'] for row in rows if row['domain'] == domain]))
            for domain in families])


def _controls(plan, indexed, stream_names):
    controls = []
    for arm in [row['name'] for row in plan['terminal_arms']]:
        rows, retained = [], []
        for task in plan['tasks']:
            seed, domain = task['geometry_task_seed'], task['domain']
            outcomes = [indexed[name, seed, arm] for name in stream_names]
            rows.append(dict(seed=seed, domain=domain, values=_metric_average([row['metrics'] for row in outcomes])))
            retained.extend(outcomes)
        controls.append(dict(arm=arm, protocol='old', rows=rows, metrics=metric_stats(rows, 'values'),
            outcome_counts=dict(Counter(row['status'] for row in retained)),
            category_counts=dict(Counter(row.get('discovery_category', 'unknown') for row in retained)),
            outcome_count_unit='terminal cells, not independent geometries',
            by_family=[dict(domain=domain, metrics=metric_stats(
                [row for row in rows if row['domain'] == domain], 'values'))
                for domain in dict.fromkeys(row['domain'] for row in rows)]))
    return controls


def summarize_loaded(plan, tasks, *, seal, accounts, accounts_source='caller_supplied_saved_accounts'):
    """Use saved Stage85-shaped outcomes; never call a scorer or rebuild a prior."""
    sealed = require_seal(plan, seal)
    indexed = _index_outcomes(plan, tasks, sealed)
    stream_names = [row['name'] for row in plan['streams']]
    streams = []
    for stream in stream_names:
        comparisons = []
        for name, terms, role in CONTRASTS:
            rows = []
            for task in plan['tasks']:
                seed, domain = task['geometry_task_seed'], task['domain']
                outcomes = {arm: indexed[stream, seed, arm] for arm, _ in terms}
                rows.append(_effect_row(seed, domain, {arm: row['metrics'] for arm, row in outcomes.items()}, terms,
                    stream=stream, arm_outcomes=deepcopy(outcomes)))
            comparisons.append(_effect_summary(name, terms, role, rows))
        streams.append(dict(name=stream, registered_geometries=len(plan['tasks']),
            controls=_controls(plan, indexed, [stream]), comparisons=comparisons))
    comparisons, directions = [], []
    for index, (name, terms, role) in enumerate(CONTRASTS):
        rows, paired = [], []
        for first, second in zip(streams[0]['comparisons'][index]['rows'], streams[1]['comparisons'][index]['rows']):
            arms = {arm: _metric_average([first['arm_metrics'][arm], second['arm_metrics'][arm]]) for arm, _ in terms}
            # Average signed stream effects directly. Do not recompute P/L from averaged G.
            row = _effect_row(first['seed'], first['domain'], arms, terms,
                stream_details=[first, second])
            row['delta'] = _metric_average([first['delta'], second['delta']])
            row['positive_gain_component'] = mean_known([first['positive_gain_component'], second['positive_gain_component']])
            row['avoided_loss_component'] = mean_known([first['avoided_loss_component'], second['avoided_loss_component']])
            rows.append(row)
            paired.append(dict(seed=first['seed'], domain=first['domain'],
                stream_deltas=[first['delta'], second['delta']], geometry_mean_delta=row['delta'],
                directions={metric: [sign(first['delta'][metric]), sign(second['delta'][metric])] for metric in ('C', 'G')}))
        comparisons.append(_effect_summary(name, terms, role, rows))
        directions.append(dict(name=name, rows=paired,
            counts={metric: dict(Counter('/'.join(row['directions'][metric]) for row in paired)) for metric in ('C', 'G')}))
    failures = [deepcopy(row) for row in seal['cases'] if row['old']['status'] != 'complete']
    scoring_failures = [dict(seed=task['seed'], domain=task['domain'], stream=stream['name'],
        status=stream['status'], error=stream.get('error')) for task in tasks for stream in task['streams']
        if stream['status'] != 'complete']
    scoring_failed_outcomes = [dict(stream=name, **deepcopy(row)) for (name, _, _), row in indexed.items()
        if row['status'] == 'scoring_failed']
    return dict(status='complete_with_retained_failures' if failures or scoring_failures or scoring_failed_outcomes else 'complete',
        registered_geometries=len(plan['tasks']), registered_task_streams=2 * len(plan['tasks']),
        registered_terminal_cells=len(sealed), streams=streams,
        independent=dict(name='independent', streams=stream_names, registered_geometries=len(plan['tasks']),
            controls=_controls(plan, indexed, stream_names), comparisons=comparisons),
        primary=comparisons[0], interaction=comparisons[1], stream_directions=directions,
        terminal_failures=failures, scoring_failures=scoring_failures, scoring_failed_outcomes=scoring_failed_outcomes,
        retained_scored_outcomes=[dict(stream=name, **deepcopy(row)) for (name, _, _), row in indexed.items()],
        costs=deepcopy(accounts), costs_source=accounts_source,
        accounting='Saved accounts retained unchanged; no physical, logical or historical amounts are added together. '
            'This function does not meter its own CPU; the scheduler caller records aggregation/check CPU separately.',
        historical_accounts=deepcopy(plan.get('historical_accounts')),
        historical_reference_manifest=plan.get('reference_manifest'),
        statistical_unit='Mean of both required observation streams within each registered geometry; original 95% t intervals.',
        limits=['The direct real AFTER minus permuted AFTER contrast and the source-by-propagation interaction are distinct.',
            'Interactions use the common four-arm defined subset for each endpoint, not differences of separate subset means.',
            'Missing stream/cell/baseline effects remain null; a conditional subset mean is not the full registered mean.',
            'Intervals are individual descriptive intervals, not multiplicity-adjusted claims.',
            'All old cohorts remain separate from this fresh cohort; results condition on the source-trained frontend and fixed permutations.',
            'Positive/avoided-loss components are signed metric identities, not identified causal mediation.'], new_work=dict(ZERO))


def summarize_files(plan, *, root, accounts_file):
    """Read only saved seal, scored geometry JSON and a caller-selected saved ledger."""
    root = Path(root)
    output = root / plan['output_dir']
    seal = json.loads((output / 'execution_status.json').read_text())
    require_seal(plan, seal)  # Must precede opening even the first already-scored task.
    tasks = [json.loads((output / 'cases' / f"seed_{task['geometry_task_seed']}" / 'result.json').read_text())
             for task in plan['tasks']]
    accounts = json.loads((root / accounts_file).read_text())
    return summarize_loaded(plan, tasks, seal=seal, accounts=accounts, accounts_source=str(accounts_file))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--accounts-file', required=True)
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    plan = json.loads((root / args.manifest).read_text())
    result = summarize_files(plan, root=root, accounts_file=args.accounts_file)
    target = root / args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
