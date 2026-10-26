"""Small saved-metric substitutes; no source fitting, truth scoring or replay."""
from copy import deepcopy
import csv
import json

import pytest

from performance import analyze_mainline_frozen_backend_comparison as audit
from performance.summarize_mainline_two_comparison_development import metrics


def value(gain=0., *, baseline=.5, certified=True, feasible=True):
    return metrics(certified, dict(true_feasible=feasible, quality=.5 + gain),
                   None if baseline is None else dict(quality=baseline))


def fixture(stage=0, count=4):
    descriptor = audit.COHORTS[stage]
    tasks = [dict(geometry_task_seed=seed, domain=('Inventory', 'Queue', 'FactorShock')[(seed - 1) % 3])
             for seed in range(1, count + 1)]
    names = ['observation_1', 'observation_2'] if stage == 0 else ['stream1', 'stream2']
    plan = dict(tasks=tasks, streams=[dict(name=name) for name in names],
        terminal_arms=[dict(name=arm) for arm in descriptor['arms']], output_dir=descriptor['output'],
        budget=dict(registered_terminal_cells=count * 2 * len(descriptor['arms']), maximum_verification_calls_per_cell=352),
        historical_accounts=[dict(source='older_cohort/completion.json', cpu_seconds=93.)])
    seal = dict(status='sealed', truth_access_allowed=True, cases=[
        dict(stream=stream, seed=task['geometry_task_seed'], domain=task['domain'], arm=arm,
             old=dict(status='complete', input_file='saved/input.json'))
        for stream in names for task in tasks for arm in descriptor['arms']])
    science = dict(status='complete', registered_geometries=count, registered_terminal_cells=len(seal['cases']),
        streams=[dict(name=name, controls=[dict(arm=arm, protocol='old', outcome_counts=dict(complete=count),
            rows=[dict(seed=task['geometry_task_seed'], domain=task['domain'], values=value()) for task in tasks])
            for arm in descriptor['arms']]) for name in names],
        terminal_failures=[], scoring_failures=[], source_numerical_limits=['Saved Queue low ESS warning'])
    completion = dict(status='complete', registered_geometries=count, registered_terminal_cells=len(seal['cases']),
        costs=dict(physical_target_calls=None, known_recorded_physical_target_calls=123,
            known_recorded=dict(cpu_seconds=12.), retained_source_known_recorded=dict(cpu_seconds=50.),
            attempts=[dict(ledger='original/attempts.jsonl', cpu_seconds=12., new_verification_calls=None)],
            unknown_target_fee_records=[dict(calls=None)], accounting='physical and historical are separate'),
        historical_accounts=deepcopy(plan['historical_accounts']))
    data = (plan, science, seal, completion, descriptor)
    refresh(data)
    return data


def selected(data, stream=0, seed=1, arm=audit.AFTER):
    unit = data[1]['streams'][stream]
    return next(row for control in unit['controls'] if control['arm'] == arm for row in control['rows'] if row['seed'] == seed)


def refresh(data):
    plan, science, _, _, descriptor = data
    def comparison(rows):
        output = dict(protocol='old', rows=rows,
            delta={metric: audit.estimate([row['delta'][metric] for row in rows]) for metric in audit.METRICS})
        output.update(dict(left=audit.AFTER, right=audit.POINT) if descriptor['name'] == 'stage85'
                      else dict(name='real_AFTER_minus_POINT'))
        return output
    for i, unit in enumerate(science['streams']):
        for control in unit['controls']:
            control['metrics'] = {m: audit.estimate([row['values'][m] for row in control['rows']]) for m in audit.METRICS}
        unit['comparisons'] = [comparison([dict(seed=task['geometry_task_seed'], domain=task['domain'],
            delta=audit.difference(selected(data, i, task['geometry_task_seed'])['values'],
                selected(data, i, task['geometry_task_seed'], audit.POINT)['values'])) for task in plan['tasks']])]
    controls = []
    for arm in descriptor['arms']:
        rows = [dict(seed=task['geometry_task_seed'], domain=task['domain'], values=audit._average([
            selected(data, i, task['geometry_task_seed'], arm)['values'] for i in (0, 1)])) for task in plan['tasks']]
        controls.append(dict(arm=arm, protocol='old', rows=rows,
            metrics={m: audit.estimate([row['values'][m] for row in rows]) for m in audit.METRICS}))
    science['independent'] = dict(controls=controls, comparisons=[comparison([
        dict(seed=task['geometry_task_seed'], domain=task['domain'], delta=audit._average([
            next(row for row in unit['comparisons'][0]['rows'] if row['seed'] == task['geometry_task_seed'])['delta']
            for unit in science['streams']])) for task in plan['tasks']])])


def test_pairs_by_geometry_and_stream_preserving_signed_quality_and_original_controls():
    data = fixture(1)
    for i, gain in enumerate((.3, -.1)):
        selected(data, i)['values'] = value(gain)
        selected(data, i, arm=audit.GP)['values'] = value(.05)
    for i in (0, 1):
        selected(data, i, seed=2)['values'] = value(-.05)
    refresh(data)
    # Deliberately reorder every input list: pairing must use identifiers, not zip.
    for unit in data[1]['streams']:
        unit['controls'].reverse()
        for control in unit['controls']:
            control['rows'].reverse()
    data[1]['streams'].reverse()
    original = deepcopy(data)
    result = audit.summarize_cohort(*data)
    effect = result['independent']['comparisons'][0]
    assert effect['rows'][0]['delta']['G'] == pytest.approx(.05)
    assert effect['rows'][0]['delta']['P'] == pytest.approx(.1)
    assert effect['rows'][0]['delta']['L'] == pytest.approx(.05)
    assert effect['positive_gain_component']['all_registered_mean'] == pytest.approx(.025)
    assert effect['avoided_loss_component']['all_registered_mean'] == pytest.approx(-.025)
    assert effect['delta']['G']['all_registered_mean'] == pytest.approx(0.)
    assert effect['signed_geometry_seeds']['G'] == dict(positive=[1], zero=[3, 4], negative=[2], unknown=[])
    assert effect['delta']['G']['conditional_available']['largest_positive_share'] == 1.
    leave = effect['delta']['G']['leave_largest_positive_geometry_out']
    assert leave['excluded_seed'] == 1 and leave['estimate']['all_registered_mean'] == pytest.approx(-.05 / 3)
    assert len(result['independent']['controls']) == 11 and len(result['streams']) == 2
    assert result['costs'] == data[3]['costs'] and data == original


def test_one_failed_stream_is_unknown_not_single_stream_mean_and_baseline_missing_is_distinct():
    data = fixture()
    selected(data, 1, arm=audit.GP)['values'] = dict.fromkeys(audit.METRICS)
    case = next(row for row in data[2]['cases'] if row['seed'] == 1 and row['stream'] == 'observation_2' and row['arm'] == audit.GP)
    case['old'] = dict(status='failed', failure_file='saved/failed.json', failure=dict(new_verification_calls=None))
    for i in (0, 1):
        for arm in data[4]['arms']:
            selected(data, i, seed=2, arm=arm)['values'] = value(baseline=None)
    data[1]['status'] = data[3]['status'] = 'complete_with_retained_failures'
    refresh(data)
    result = audit.summarize_cohort(*data)
    effect = result['independent']['comparisons'][0]
    assert effect['rows'][0]['available_streams']['C'] == 1
    assert effect['rows'][0]['delta']['C'] is None and effect['rows'][0]['delta']['G'] is None
    assert effect['rows'][1]['delta']['C'] == 0 and effect['rows'][1]['delta']['R'] == 0
    assert effect['rows'][1]['delta']['G'] is None
    assert effect['delta']['C']['available_geometries'] == 3
    assert effect['delta']['G']['available_geometries'] == 2
    assert effect['delta']['G']['all_registered_mean'] is None
    assert effect['rows'][1]['stream_statuses'][0]['left_quality'] == 'undefined_initial_baseline'
    assert result['sealed_terminal_failures'][0]['old']['failure']['new_verification_calls'] is None
    assert len(result['independent']['controls']) == 9


def test_false_certificate_and_unrelated_failed_control_are_retained():
    data = fixture(1)
    for i in (0, 1):
        selected(data, i)['values'] = value(.2, feasible=False)
    arm = audit.PERMUTED_ARMS[0]
    selected(data, 0, arm=arm)['values'] = dict.fromkeys(audit.METRICS)
    next(row for row in data[2]['cases'] if row['seed'] == 1 and row['stream'] == 'stream1' and row['arm'] == arm)['old']['status'] = 'missing'
    refresh(data)
    result = audit.summarize_cohort(*data)
    effect = result['independent']['comparisons'][0]
    assert effect['rows'][0]['delta']['A'] == 0 and effect['rows'][0]['delta']['C'] == -1
    assert effect['rows'][0]['delta']['false_certificate'] == 1
    assert effect['rows'][0]['delta']['R'] == -.5
    assert effect['rows'][0]['delta']['G'] == 0
    assert result['sealed_terminal_counts']['missing'] == 1
    assert effect['delta']['C']['available_geometries'] == 4
    assert result['retained_source_metadata']['source_numerical_limits'] == ['Saved Queue low ESS warning']


@pytest.mark.parametrize('mutation', ['missing_cell', 'duplicate_cell', 'running', 'cap', 'missing_arm', 'ap_changed', 'failed_nonnull'])
def test_actual_grid_and_metric_contract_violations_stop(mutation):
    data = fixture()
    if mutation == 'missing_cell':
        data[2]['cases'].pop()
    elif mutation == 'duplicate_cell':
        data[2]['cases'][-1] = deepcopy(data[2]['cases'][0])
    elif mutation == 'running':
        data[2]['cases'][-1]['old']['status'] = 'running'
    elif mutation == 'cap':
        data[0]['budget']['maximum_verification_calls_per_cell'] = 353
    elif mutation == 'missing_arm':
        data[1]['streams'][0]['controls'].pop()
    elif mutation == 'ap_changed':
        data[1]['independent']['comparisons'][0]['rows'][0]['delta']['R'] += .1
    else:
        data[2]['cases'][-1]['old']['status'] = 'failed'
    with pytest.raises(ValueError):
        audit.summarize_cohort(*data)


def files(tmp_path, data):
    plan, science, seal, completion, descriptor = data
    for path, value_ in ((descriptor['manifest'], plan), (descriptor['output'] + '/summary.json', science),
                         (descriptor['output'] + '/execution_status.json', seal),
                         (descriptor['output'] + '/completion.json', completion)):
        audit.save(tmp_path / path, value_)


def test_file_entry_checks_both_seals_before_metrics_and_records_only_new_cpu(tmp_path, monkeypatch):
    for stage in (0, 1):
        files(tmp_path, fixture(stage))
    audit.save(tmp_path / 'checks.json', dict(status='passed'))
    original_read = audit.read
    opened = []
    def watched(path):
        opened.append(str(path))
        assert str(path).endswith(('.json',))
        return original_read(path)
    monkeypatch.setattr(audit, 'read', watched)
    completion = audit.run(root=tmp_path, checks='checks.json')
    summaries = [i for i, path in enumerate(opened) if path.endswith('/summary.json')]
    gates = [i for i, path in enumerate(opened) if path.endswith(('/execution_status.json', '/completion.json'))]
    assert min(summaries) > max(gates)
    assert len(opened) == 9  # check + (manifest, seal, completion, summary) for each cohort
    output = tmp_path / audit.OUTPUT
    details = original_read(output / 'summary.json')
    headline = original_read(output / 'headline.json')
    assert len(details['cohorts']) == 2 and len(headline['cohorts']) == 2
    assert (output / 'headline.json').stat().st_size < 100_000
    assert all(value_ == 0 for value_ in completion['new_work'].values())
    assert completion['analysis_attempt']['cpu_seconds'] >= 0
    assert details['cohorts'][0]['costs'] == fixture()[3]['costs']
    with (output / 'geometry_effects.csv').open() as stream:
        assert len(list(csv.DictReader(stream))) == 2 * 3 * 3 * 4
    assert audit.run(root=tmp_path, checks='checks.json') == completion
    assert len((output / 'attempts.jsonl').read_text().splitlines()) == 1


def test_unsealed_second_cohort_never_opens_either_science_summary(tmp_path, monkeypatch):
    for stage in (0, 1):
        data = fixture(stage)
        if stage:
            data[2]['status'] = 'running'
        files(tmp_path, data)
    audit.save(tmp_path / 'checks.json', dict(status='passed'))
    original = audit.read
    def watched(path):
        assert not str(path).endswith('/summary.json')
        return original(path)
    monkeypatch.setattr(audit, 'read', watched)
    with pytest.raises(ValueError, match='already be sealed'):
        audit.run(root=tmp_path, checks='checks.json')
    ledger = (tmp_path / audit.OUTPUT / 'attempts.jsonl').read_text().splitlines()
    assert json.loads(ledger[0])['status'] == 'failed'


def test_failed_checks_block_all_source_reads(tmp_path, monkeypatch):
    audit.save(tmp_path / 'checks.json', dict(status='failed'))
    original = audit.read
    def watched(path):
        assert path.name == 'checks.json'
        return original(path)
    monkeypatch.setattr(audit, 'read', watched)
    with pytest.raises(ValueError, match='checks must pass'):
        audit.run(root=tmp_path, checks='checks.json')
