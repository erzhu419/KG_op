"""Small saved-record substitutes for the registered source-alignment contrasts."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from performance import mainline_source_alignment_summary as summary
from performance.summarize_mainline_two_comparison_development import metrics


def fixture(count=3):
    plan = dict(output_dir='alignment',
        tasks=[dict(geometry_task_seed=index + 1, domain=('Inventory', 'Queue', 'FactorShock')[index % 3])
               for index in range(count)],
        streams=[dict(name=name) for name in ('stream1', 'stream2')],
        terminal_arms=[dict(name=arm, history_arm=arm.removesuffix('_native_terminal')) for arm in summary.ARMS],
        budget=dict(terminal_cells=count * 22, verification_cap_per_cell=352),
        reference_manifest='historical/confirmation.json', historical_accounts=['historical/fees.json'])
    tasks, cases = [], []
    for task in plan['tasks']:
        seed, domain = task['geometry_task_seed'], task['domain']
        streams = []
        for stream in plan['streams']:
            outcomes = []
            for arm in summary.ARMS:
                outcomes.append(dict(seed=seed, domain=domain, arm=arm, history_arm=arm.removesuffix('_native_terminal'),
                    protocol='old', status='complete', metrics=value(0.), best_initial_quality=.5,
                    quality_reference_status='defined', discovery_category='initial',
                    reported_policy_id='P0', deployed_policy_id='P0', actual_deployed_quality=.5,
                    verification_budget=80))
                cases.append(dict(stream=stream['name'], seed=seed, domain=domain, arm=arm,
                    old=dict(status='complete', input_file='saved/input.json', verification_file='saved/terminal.json')))
            streams.append(dict(name=stream['name'], status='complete', outcomes=outcomes, queries=[], policy_paths=[]))
        tasks.append(dict(seed=seed, domain=domain, status='complete', streams=streams))
    seal = dict(status='sealed', truth_access_allowed=True, cases=cases)
    accounts = dict(known_recorded_physical_target_calls=123, logical_calls=234,
        physical_target_calls=None, unknown_target_fee_records=[dict(stage='verify', calls=None)],
        failed_attempts=[dict(stage='verify', cpu_seconds=1.5)], historical_accounts=['historical/fees.json'])
    return plan, tasks, seal, accounts


def value(gain, *, certified=True, feasible=True, baseline=.5):
    return metrics(certified, dict(true_feasible=feasible, quality=.5 + gain),
                   None if baseline is None else dict(quality=baseline))


def selected(tasks, geometry=0, stream=0, arm=summary.AFTER):
    return next(row for row in tasks[geometry]['streams'][stream]['outcomes'] if row['arm'] == arm)


def analyze(data):
    plan, tasks, seal, accounts = data
    return summary.summarize_loaded(plan, tasks, seal=seal, accounts=accounts)


def contrast(result, name):
    return next(row for row in result['independent']['comparisons'] if row['name'] == name)


def test_factorial_preserves_all_eleven_arms_and_averages_signed_two_stream_effects_first():
    data = fixture()
    _, tasks, _, accounts = data
    gains = {summary.AFTER: (.6, -.2), summary.POINT: (.2, .1),
             summary.PERMUTED_AFTER: (.1, -.3), summary.PERMUTED_POINT: (.05, 0.)}
    for arm, pair in gains.items():
        for stream, gain in enumerate(pair):
            selected(tasks, stream=stream, arm=arm)['metrics'] = value(gain)
    # An opposing geometry ensures losses are retained, not selected away.
    for stream in (0, 1):
        selected(tasks, geometry=1, stream=stream)['metrics'] = value(-.15)
        selected(tasks, geometry=1, stream=stream, arm=summary.PERMUTED_AFTER)['metrics'] = value(.05)
    result = analyze(data)
    primary, interaction = result['primary'], result['interaction']
    assert len(result['independent']['controls']) == 11
    assert len(result['independent']['comparisons']) == 6
    assert primary['rows'][0]['delta']['G'] == pytest.approx(.3)
    assert primary['rows'][0]['positive_gain_component'] == pytest.approx(.25)
    assert primary['rows'][0]['avoided_loss_component'] == pytest.approx(.05)
    assert interaction['rows'][0]['delta']['G'] == pytest.approx(.175)
    assert contrast(result, 'real_POINT_minus_permuted_POINT')['rows'][0]['delta']['G'] == pytest.approx(.125)
    assert primary['delta']['G']['all_registered_mean'] == pytest.approx(.1 / 3)
    assert primary['delta']['G']['conditional_available']['degrees_of_freedom'] == 2
    assert primary['delta']['G']['all_registered_descriptive_95pct_t_interval'] is not None
    assert primary['signed_geometry_seeds']['G'] == dict(positive=[1], zero=[3], negative=[2], unknown=[])
    sensitivity = primary['delta']['G']['leave_largest_positive_geometry_out']
    assert sensitivity['excluded_seed'] == 1
    assert sensitivity['all_registered_leave_one_out_mean'] == pytest.approx(-.1)
    assert primary['delta']['G']['conditional_available']['largest_positive_share'] == 1
    assert len(primary['by_family']) == 3 and len(result['streams']) == 2
    assert result['costs'] == accounts and all(value == 0 for value in result['new_work'].values())


def test_interaction_uses_common_four_arm_subset_instead_of_difference_of_subset_means():
    data = fixture()
    _, tasks, seal, _ = data
    for stream in (0, 1):
        selected(tasks, stream=stream)['metrics'] = value(.4)
        selected(tasks, stream=stream, arm=summary.PERMUTED_AFTER)['metrics'] = value(.1)
    missing = selected(tasks, stream=1, arm=summary.PERMUTED_POINT)
    missing.update(status='failed', metrics=metrics(failed=True), verification_budget=None)
    case = next(row for row in seal['cases'] if row['seed'] == 1 and row['stream'] == 'stream2'
                and row['arm'] == summary.PERMUTED_POINT)
    case['old'] = dict(status='failed', failure=dict(message='actual retained fault', new_verification_calls=None))
    result = analyze(data)
    assert result['primary']['rows'][0]['delta']['G'] == pytest.approx(.3)
    assert result['interaction']['rows'][0]['delta']['G'] is None
    assert result['interaction']['rows'][0]['delta']['C'] is None
    assert result['interaction']['delta']['G']['all_registered_mean'] is None
    assert result['interaction']['delta']['G']['available_geometries'] == 2
    assert contrast(result, 'real_AFTER_minus_POINT')['rows'][0]['delta']['G'] == pytest.approx(.4)
    assert contrast(result, 'permuted_AFTER_minus_POINT')['rows'][0]['delta']['G'] is None
    assert result['terminal_failures'][0]['old']['failure']['new_verification_calls'] is None
    assert result['status'] == 'complete_with_retained_failures'


def test_missing_baseline_no_certificate_false_certificate_and_scoring_failure_keep_meanings():
    data = fixture()
    _, tasks, _, _ = data
    for stream in (0, 1):
        row = selected(tasks, stream=stream)
        row.update(metrics=value(0., certified=False, baseline=None), best_initial_quality=None,
                   actual_deployed_quality=None, deployed_policy_id=None)
        bad_certificate = selected(tasks, geometry=1, stream=stream)
        bad_certificate.update(metrics=value(.2, feasible=False), actual_deployed_quality=None, deployed_policy_id=None)
    tasks[2]['status'] = 'scoring_failed'
    stream = tasks[2]['streams'][1]
    stream.update(status='failed', error='saved scoring failure')
    for row in stream['outcomes']:
        row.update(status='scoring_failed', metrics=metrics(failed=True))
    result = analyze(data)
    assert result['primary']['rows'][0]['delta']['C'] == -1
    assert result['primary']['rows'][0]['delta']['G'] is None
    assert result['primary']['rows'][1]['delta']['false_certificate'] == 1
    assert result['primary']['rows'][1]['delta']['G'] == 0
    assert result['primary']['rows'][2]['delta']['C'] is None
    assert len(result['scoring_failures']) == 1 and len(result['scoring_failed_outcomes']) == 11
    assert result['terminal_failures'] == []
    assert result['costs']['physical_target_calls'] is None
    assert result['historical_accounts'] == ['historical/fees.json']
    assert all(row['metrics']['G']['registered_geometries'] == 3 for row in result['independent']['controls'])


@pytest.mark.parametrize('terminal_status', ['failed', 'missing'])
def test_whole_stream_scoring_failure_keeps_independently_sealed_terminal_failure(terminal_status):
    # Stage85 failed_stream writes scoring_failed for every arm, including arms
    # whose original verification had already failed or lacked terminal input.
    data = fixture(2)
    _, tasks, seal, _ = data
    case = next(row for row in seal['cases'] if row['stream'] == 'stream1' and row['seed'] == 1
                and row['arm'] == summary.PERMUTED_AFTER)
    case['old'] = dict(status=terminal_status, failure_file='retained_failure.json',
                       failure=dict(message='terminal-stage failure', new_verification_calls=None))
    tasks[0]['status'] = 'scoring_failed'
    stream = tasks[0]['streams'][0]
    stream.update(status='failed', error='subsequent stream-scoring fault')
    for row in stream['outcomes']:
        row.update(status='scoring_failed', metrics=metrics(failed=True), verification_budget=None)
    result = analyze(data)
    assert result['terminal_failures'][0]['old']['status'] == terminal_status
    assert result['terminal_failures'][0]['old']['failure']['new_verification_calls'] is None
    assert len(result['scoring_failed_outcomes']) == 11
    assert result['primary']['rows'][0]['delta']['C'] is None
    assert result['primary']['rows'][0]['delta']['G'] is None
    assert result['primary']['delta']['G']['registered_geometries'] == 2


@pytest.mark.parametrize('mutation', ['missing_last', 'duplicate', 'running', 'wrong_budget', 'omit_control'])
def test_full_terminal_seal_blocks_before_any_scored_record_read(tmp_path, monkeypatch, mutation):
    plan, _, seal, _ = fixture(2)
    if mutation == 'missing_last':
        seal['cases'].pop()
    elif mutation == 'duplicate':
        seal['cases'][-1] = deepcopy(seal['cases'][0])
    elif mutation == 'running':
        seal['cases'][-1]['old']['status'] = 'running'
    elif mutation == 'wrong_budget':
        plan['budget']['terminal_cells'] += 1
    else:
        plan['terminal_arms'].pop()
    output = tmp_path / plan['output_dir']
    output.mkdir()
    (output / 'execution_status.json').write_text(json.dumps(seal))
    original = Path.read_text
    opened = []
    def watched(path, *args, **kwargs):
        opened.append(path.name)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', watched)
    with pytest.raises(ValueError):
        summary.summarize_files(plan, root=tmp_path, accounts_file='absent_accounts.json')
    assert opened == ['execution_status.json']


def test_file_adapter_reuses_stage85_result_schema_without_scoring_or_accounting_recharge(tmp_path, monkeypatch):
    from performance import summarize_mainline_precision_confirmation as old
    plan, tasks, seal, accounts = fixture(2)
    def forbidden(*args, **kwargs):
        raise AssertionError('Saved-record summary invoked scientific work or recounted ledgers')
    for name in ('score_task', 'score_stream', 'freeze', 'accounting'):
        monkeypatch.setattr(old, name, forbidden)
    output = tmp_path / plan['output_dir']
    output.mkdir()
    (output / 'execution_status.json').write_text(json.dumps(seal))
    for task in tasks:
        folder = output / 'cases' / f"seed_{task['seed']}"
        folder.mkdir(parents=True)
        (folder / 'result.json').write_text(json.dumps(task))
    (tmp_path / 'saved_accounts.json').write_text(json.dumps(accounts))
    before = deepcopy(tasks)
    result = summary.summarize_files(plan, root=tmp_path, accounts_file='saved_accounts.json')
    assert result['registered_terminal_cells'] == 44
    assert result['costs'] == accounts
    assert result['costs_source'] == 'saved_accounts.json'
    assert tasks == before
    assert len(result['retained_scored_outcomes']) == 44


@pytest.mark.parametrize('mutation', ['failed_as_zero', 'missing_baseline_as_zero', 'over_budget', 'missing_cell'])
def test_reachable_input_errors_do_not_silently_change_registered_estimands(mutation):
    data = fixture(2)
    row = selected(data[1])
    if mutation == 'failed_as_zero':
        row['status'] = 'failed'
    elif mutation == 'missing_baseline_as_zero':
        row['best_initial_quality'] = None
    elif mutation == 'over_budget':
        row['verification_budget'] = 353
    else:
        data[1][0]['streams'][0]['outcomes'].pop()
    with pytest.raises(ValueError):
        analyze(data)
