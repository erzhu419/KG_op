"""Portable saved-cell aggregation checks on small deterministic substitutes."""
from copy import deepcopy
import json

import pytest

from reproducibility.mainline_release import export_cells as exporter
from reproducibility.mainline_release import statistics as stats
from reproducibility.mainline_release import verify_statistics as parity


ARMS = [stats.FULL, stats.AFTER, stats.POINT, stats.GP, stats.NATIVE, stats.PA, stats.PP, 'initialization_only']


def value(gain=0., *, baseline=.5, certified=True):
    c = int(certified)
    reward = .5 + gain if c else 0.
    gain = None if baseline is None else reward - c * baseline
    return dict(A=c, C=c, false_certificate=0, R=reward, G=gain,
        P=None if gain is None else max(gain, 0.), L=None if gain is None else max(-gain, 0.))


def fixture(count=4):
    cohort = dict(name='stage93', streams=['s1', 's2'], arms=ARMS,
        geometries=[dict(seed=seed, domain='Queue' if seed % 2 else 'Inventory') for seed in range(1, count + 1)])
    cells = [dict(cohort=cohort['name'], seed=task['seed'], domain=task['domain'], stream=stream, arm=arm,
        history_arm=arm, protocol='old', status='complete', terminal_status='complete', metrics=value(),
        baseline_status='defined', best_initial_quality=.5, verification_calls=208)
        for task in cohort['geometries'] for stream in cohort['streams'] for arm in ARMS]
    return dict(format='mainline_release_cells_v1', cohorts=[cohort], cells=cells, attempts=[])


def cell(data, seed=1, stream='s1', arm=stats.AFTER):
    return next(row for row in data['cells'] if (row['seed'], row['stream'], row['arm']) == (seed, stream, arm))


def contrast(result, name, unit='independent'):
    return next(row for group in result['cohorts'][0]['units'] if group['name'] == unit
        for row in group['comparisons'] if row['name'] == name)


def fail(row):
    row.update(status='missing', terminal_status='missing', metrics=dict.fromkeys(stats.METRICS), verification_calls=0)


def test_pair_ids_stream_means_and_signed_positive_loss_decomposition():
    data = fixture()
    for stream, gain in [('s1', .3), ('s2', -.1)]:
        cell(data, stream=stream)['metrics'] = value(gain)
        cell(data, stream=stream, arm=stats.GP)['metrics'] = value(.05)
    for stream in ('s1', 's2'):
        cell(data, 2, stream)['metrics'] = value(-.05)
    data['cells'].reverse()
    original = deepcopy(data)
    result = stats.recompute(data)
    effect = contrast(result, 'AFTER_minus_same_source_GP')
    assert effect['rows'][0]['values']['G'] == pytest.approx(.05)
    assert effect['rows'][0]['values']['P'] == pytest.approx(.1)
    assert effect['rows'][0]['values']['L'] == pytest.approx(.05)
    assert effect['positive_gain_component']['all_registered_mean'] == pytest.approx(.025)
    assert effect['avoided_loss_component']['all_registered_mean'] == pytest.approx(-.025)
    assert effect['delta']['G']['all_registered_mean'] == pytest.approx(0.)
    assert effect['delta']['G']['leave_largest_positive_geometry_out']['excluded_seed'] == 1
    assert effect['delta']['G']['conditional_available']['degrees_of_freedom'] == 3
    assert data == original


def test_missing_one_stream_is_null_and_undefined_baseline_is_distinct():
    data = fixture()
    fail(cell(data, stream='s2', arm=stats.PA))
    for stream in ('s1', 's2'):
        for arm in ARMS:
            row = cell(data, 2, stream, arm)
            row.update(metrics=value(baseline=None), baseline_status='undefined_initial_baseline', best_initial_quality=None)
    effect = contrast(stats.recompute(data), 'real_AFTER_minus_permuted_AFTER')
    assert effect['rows'][0]['values']['C'] is None
    assert effect['rows'][1]['values']['C'] == 0
    assert effect['rows'][1]['values']['G'] is None
    assert effect['delta']['C']['available_geometries'] == 3
    assert effect['delta']['G']['available_geometries'] == 2
    assert effect['delta']['G']['all_registered_mean'] is None


def test_interaction_uses_four_arm_common_support_before_geometry_average():
    data = fixture(2)
    for stream in ('s1', 's2'):
        for arm, gain in [(stats.AFTER, .3), (stats.POINT, .1), (stats.PA, .2), (stats.PP, .15)]:
            cell(data, 1, stream, arm)['metrics'] = value(gain)
        cell(data, 2, stream, stats.AFTER)['metrics'] = value(.8)
    fail(cell(data, 2, 's2', stats.PA))
    result = stats.recompute(data)
    interaction = contrast(result, 'source_by_propagation_interaction')
    assert interaction['rows'][0]['values']['G'] == pytest.approx(.15)
    assert interaction['rows'][1]['values']['G'] is None
    assert interaction['delta']['G']['conditional_available']['mean'] == pytest.approx(.15)
    assert interaction['delta']['G']['available_geometries'] == 1
    assert contrast(result, 'AFTER_minus_POINT')['delta']['G']['available_geometries'] == 2


def test_binary_missing_sensitivity_is_completion_bound_and_keeps_point_estimate_null():
    data = fixture(2)
    # Four registered stream cells. Real A = 4, real P = 3,
    # permuted A = 1 observed success + two failures + one missing, P = 4.
    for seed, stream in [(1, 's1'), (1, 's2'), (2, 's1')]:
        cell(data, seed, stream, stats.PA)['metrics'] = value(certified=(seed == 1 and stream == 's1'))
    fail(cell(data, 2, 's2', stats.PA))
    cell(data, 2, 's2', stats.POINT)['metrics'] = value(certified=False)
    result = stats.recompute(data)
    bounds = {row['name']: row for row in result['cohorts'][0]['missing_sensitivity']}
    assert bounds['real_AFTER_minus_permuted_AFTER']['mean_bounds'] == [.5, .75]
    assert bounds['source_by_propagation_interaction']['mean_bounds'] == [.75, 1.]
    assert contrast(result, 'source_by_propagation_interaction')['delta']['C']['all_registered_mean'] is None
    assert cell(data, 2, 's2', stats.PA)['metrics']['C'] is None


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'failure_nonnull', 'gain_identity', 'family'])
def test_reachable_bad_cell_exports_stop_instead_of_repairing(mutation):
    data = fixture(2)
    if mutation == 'missing':
        data['cells'].pop()
    elif mutation == 'duplicate':
        data['cells'].append(deepcopy(data['cells'][0]))
    elif mutation == 'failure_nonnull':
        data['cells'][0]['status'] = 'missing'
    elif mutation == 'gain_identity':
        data['cells'][0]['metrics']['G'] = .1
    else:
        data['cells'][0]['domain'] = 'other'
    with pytest.raises(ValueError):
        stats.recompute(data)


def test_costs_preserve_failed_cpu_shared_initial_and_unknown_fees_without_wrapper_sum():
    data = fixture(2)
    base = dict(cohort='stage93', ledger='target/attempts.jsonl', accounting_scope='target', stage='search', status='completed')
    data['attempts'] = [dict(base, ledger_row=1, arm='initialization_only', cpu_seconds=4., new_target_initial_calls=10),
        dict(base, ledger_row=2, arm=stats.AFTER, cpu_seconds=3., new_target_search_calls=3),
        dict(base, ledger_row=3, arm=stats.PA, status='failed', cpu_seconds=2., new_target_search_calls=None),
        dict(base, ledger='source/attempts.jsonl', ledger_row=1, accounting_scope='retained_source', arm=None,
            cpu_seconds=5., new_source_calls=0, new_source_fits=1)]
    result = stats.cost_table(data)[0]
    groups = {(row['scope'], row['arm']): row for row in result['physical_attempt_groups']}
    assert groups['target', None]['known_recorded']['new_target_initial_calls'] == 10
    assert groups['target', stats.PA]['failed_cpu_seconds'] == 2.
    assert groups['target', stats.PA]['known_recorded']['new_target_search_calls'] is None
    assert groups['target', stats.PA]['explicit_unknown_fields']['new_target_search_calls'] == 1
    assert groups['target', stats.PA]['peak_memory_bytes'] is None
    assert groups['retained_source', None]['known_recorded']['cpu_seconds'] == 5.
    data['attempts'].append(deepcopy(data['attempts'][0]))
    with pytest.raises(ValueError, match='charged twice'):
        stats.cost_table(data)


def export_fixture(tmp_path):
    data = fixture(2)
    plan = dict(output_dir='saved', tasks=[dict(geometry_task_seed=row['seed'], domain=row['domain'])
        for row in data['cohorts'][0]['geometries']], streams=[dict(name=name) for name in ('s1', 's2')],
        terminal_arms=[dict(name=arm, history_arm=arm) for arm in ARMS])
    seal = dict(status='sealed', truth_access_allowed=True, cases=[dict(stream=row['stream'], seed=row['seed'],
        domain=row['domain'], arm=row['arm'], source_case='saved/history', old=dict(status=row['terminal_status'],
            input_file='saved/input.json', verification_file='saved/verification.json')) for row in data['cells']])
    summary = dict(streams=[dict(name=stream, controls=[dict(arm=arm,
        rows=[dict(seed=row['seed'], domain=row['domain'], values=row['metrics']) for row in data['cells']
            if row['stream'] == stream and row['arm'] == arm]) for arm in ARMS]) for stream in ('s1', 's2')])
    for seed in (1, 2):
        result = dict(seed=seed, streams=[dict(name=stream, outcomes=[dict(seed=seed, arm=row['arm'], status=row['status'],
            metrics=row['metrics'], best_initial_quality=.5, verification_budget=208, selected_truth=dict(quality=.5, true_feasible=True))
            for row in data['cells'] if row['seed'] == seed and row['stream'] == stream]) for stream in ('s1', 's2')])
        stats.write_json(tmp_path / 'saved/cases' / f'seed_{seed}/result.json', result)
    stats.write_json(tmp_path / 'plan.json', plan)
    stats.write_json(tmp_path / 'saved/execution_status.json', seal)
    stats.write_json(tmp_path / 'saved/completion.json', dict(status='complete', costs=dict(attempts=[], physical_target_calls=None)))
    stats.write_json(tmp_path / 'saved/summary.json', summary)
    # A headline is deliberately unusable: the exporter must never read it.
    (tmp_path / 'saved/headline.json').write_text('not json')
    return data


def test_export_reads_actual_cell_metrics_and_trace_keys_without_headline_or_models(tmp_path):
    data = export_fixture(tmp_path)
    output = exporter.export(tmp_path, [('stage93', 'plan.json')])
    assert len(output['cells']) == len(data['cells'])
    assert output['cells'][0]['references']['input_file'] == 'saved/input.json'
    assert output['cells'][0]['selected_truth']['quality'] == .5
    stats.write_json(tmp_path / 'cells.json.gz', output)
    assert stats.read_json(tmp_path / 'cells.json.gz') == output


def test_export_whole_grid_seal_precedes_any_saved_scoring_read(tmp_path):
    export_fixture(tmp_path)
    seal = stats.read_json(tmp_path / 'saved/execution_status.json')
    seal['cases'].pop()
    stats.write_json(tmp_path / 'saved/execution_status.json', seal)
    (tmp_path / 'saved/summary.json').unlink()
    with pytest.raises(ValueError, match='already be sealed'):
        exporter.export(tmp_path, [('stage93', 'plan.json')])


def test_export_mismatch_between_per_geometry_and_summary_stops(tmp_path):
    export_fixture(tmp_path)
    result = stats.read_json(tmp_path / 'saved/cases/seed_1/result.json')
    result['streams'][0]['outcomes'][0]['metrics']['C'] = 0
    stats.write_json(tmp_path / 'saved/cases/seed_1/result.json', result)
    with pytest.raises(ValueError, match='differ'):
        exporter.export(tmp_path, [('stage93', 'plan.json')])


def test_historical_parity_adapter_handles_actual_three_summary_shapes():
    complete = stats.estimate([.5, 0.])
    compact = dict(n=2, registered=2, all_registered_mean=.25,
        conditional_mean=.25, conditional_ci95=complete['conditional_available']['descriptive_95pct_t_interval'])
    assert parity._normalized(complete) == parity._normalized(compact)
    old_control = dict(mean=.25, defined=2, available_mean=.25)
    assert all(parity._normalized(complete)[key] == value for key, value in parity._normalized(old_control).items())


def test_parity_compares_registered_support_nulls_and_intervals_without_aggregating_cells():
    result = stats.recompute(fixture(2))
    result['cohorts'].append(dict(deepcopy(result['cohorts'][0]), name='stage85'))
    references = {}
    for name in ('stage85', 'stage93'):
        cohort = next(row for row in result['cohorts'] if row['name'] == name)
        unit = next(row for row in cohort['units'] if row['name'] == 'independent')
        references[name] = dict(registered_terminal_cells=cohort['registered_cells'], independent=deepcopy(unit))
    references['stage103'] = dict(cohorts=[])
    assert parity.verify(result, references)['status'] == 'passed'
    references['stage93']['independent']['comparisons'][0]['delta']['G']['conditional_available']['mean'] = .2
    with pytest.raises(ValueError, match='Frozen table parity failed'):
        parity.verify(result, references)
