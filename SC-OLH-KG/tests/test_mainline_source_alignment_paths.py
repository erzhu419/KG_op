"""Saved-record substitutes for source attribution, signed losses and missingness."""
from copy import deepcopy
import json
import pickle

import numpy as np
import pytest

from performance import analyze_mainline_source_alignment_paths as paths
from performance import mainline_source_alignment_summary as factorial
from test_mainline_precision_propagation_paths import fixture as old_fixture, save
from test_mainline_frozen_propagation_paths import selection_record


MULTI = 'multiple_safe_unobserved_competition'
PART_METRICS = ('C', 'positive_gain', 'avoided_loss', 'G')


def outcome(c=1, gain=0., *, category='initial', repeats=0, status='complete'):
    values = dict(A=c, C=c, false_certificate=0, R=.5 + gain if c else 0.,
                  G=gain, P=max(gain, 0.), L=max(-gain, 0.))
    if status != 'complete':
        values = dict.fromkeys(values)
    return dict(status=status, protocol='old', metrics=values,
                discovery_category=category, returned_repeat_count=repeats,
                best_initial_quality=.5, quality_reference_status='defined',
                deployed_policy_id='P0010' if c and status == 'complete' else None,
                actual_deployed_quality=.5 + gain if c and status == 'complete' else None)


def assert_partitions_add_to_saved_delta(result):
    expected = dict(C=result['delta']['C'], positive_gain=result['delta']['P'],
                    avoided_loss=None if result['delta']['L'] is None else -result['delta']['L'],
                    G=result['delta']['G'])
    for partition in result['partitions'].values():
        assert partition
        for metric, target in expected.items():
            values = [row[metric] for row in partition.values()]
            if target is None:
                assert all(value is None for value in values)
            else:
                assert all(value is not None for value in values)
                assert sum(values) == pytest.approx(target)


@pytest.mark.parametrize('left,right,label,expected', [
    (outcome(1, -.2), outcome(0), 'left_only_true_certified', (1., 0., -.2, -.2)),
    (outcome(1, .3), outcome(1, -.1), 'both_true_certified', (0., .3, .1, .4)),
    (outcome(0), outcome(1, -.2), 'right_only_true_certified', (-1., 0., .2, .2)),
    (outcome(0), outcome(0), 'neither_true_certified', (0., 0., 0., 0.)),
])
def test_certification_channels_keep_new_certified_losses_and_same_certificate_quality(left, right, label, expected):
    before = deepcopy((left, right))
    result = paths.pair_components(left, right)
    assert_partitions_add_to_saved_delta(result)
    assert tuple(result['partitions']['certification'][label][key] for key in PART_METRICS) == pytest.approx(expected)
    for other, values in result['partitions']['certification'].items():
        if other != label:
            assert tuple(values[key] for key in PART_METRICS) == pytest.approx((0., 0., 0., 0.))
    assert (left, right) == before


def test_four_arm_interaction_is_difference_of_additive_components_on_common_support():
    real_a, real_p = outcome(1, .6, category=MULTI, repeats=2), outcome(0)
    perm_a, perm_p = outcome(1, .3, category=MULTI), outcome(1, -.1)
    result = paths.interaction_components(real_a, real_p, perm_a, perm_p)
    assert_partitions_add_to_saved_delta(result)
    assert result['delta']['C'] == 1
    assert result['delta']['G'] == pytest.approx(.2)
    parts = result['partitions']['certification']
    assert parts['left_only_true_certified']['G'] == pytest.approx(.6)
    assert parts['both_true_certified']['G'] == pytest.approx(-.4)
    # Discovery and repeat views retain the POINT-side loss, rather than
    # counting only the newly discovered positive policy.
    assert result['partitions']['discovery'][MULTI]['positive_gain'] == pytest.approx(.3)
    assert result['partitions']['discovery']['initial']['avoided_loss'] == pytest.approx(-.1)


def test_missing_fourth_arm_invalidates_interaction_without_erasing_available_direct_pair():
    real_a, perm_a = outcome(1, .3), outcome(1, .1)
    direct = paths.pair_components(real_a, perm_a)
    result = paths.interaction_components(real_a, outcome(1, .05), perm_a,
                                         outcome(status='missing'))
    assert direct['delta']['G'] == pytest.approx(.2)
    assert all(value is None for value in result['delta'].values())
    assert_partitions_add_to_saved_delta(result)


def test_undefined_baseline_retains_C_but_never_zero_fills_quality_partitions():
    values = [outcome(1, .3), outcome(0), outcome(1, .1), outcome(0)]
    for row in values:
        row['best_initial_quality'] = None
        row['quality_reference_status'] = 'no_true_feasible_initial'
        row['metrics'].update(G=None, P=None, L=None)
    result = paths.interaction_components(*values)
    assert result['delta']['C'] == 0
    assert all(result['delta'][key] is None for key in ('G', 'P', 'L'))
    assert_partitions_add_to_saved_delta(result)


def test_two_stream_mean_averages_signed_P_L_and_requires_both_streams():
    first = paths.pair_components(outcome(1, .4, category=MULTI), outcome(0))
    second = paths.pair_components(outcome(1, -.1, category=MULTI), outcome(1, .2))
    result = paths.mean_components([first, second])
    assert result['delta']['G'] == pytest.approx(.05)
    assert result['delta']['P'] == pytest.approx(.1)
    assert result['delta']['L'] == pytest.approx(.05)
    assert_partitions_add_to_saved_delta(result)
    missing = paths.pair_components(outcome(status='missing'), outcome(0))
    unknown = paths.mean_components([first, missing])
    assert all(value is None for value in unknown['delta'].values())
    assert_partitions_add_to_saved_delta(unknown)


def test_false_certificate_remains_separate_from_true_certified_deployment():
    false = outcome(0)
    false['metrics'].update(A=1, false_certificate=1)
    false['reported_policy_id'] = 'P0010'
    result = paths.pair_components(false, outcome(0))
    assert result['delta']['A'] == result['delta']['false_certificate'] == 1
    assert result['delta']['C'] == result['delta']['G'] == 0
    assert result['partitions']['certification']['neither_true_certified']['C'] == 0
    assert_partitions_add_to_saved_delta(result)


@pytest.mark.parametrize('updates,expected', [
    ({}, 'initial_common_state'),
    ({'same_physical_pool': False}, 'initial_not_common'),
    ({'same_paid_history': False}, 'initial_not_common'),
    ({'status': 'unavailable'}, 'initial_unavailable'),
    ({'paid_prefix': 11}, 'later_adaptive_path'),
    ({'paid_prefix': 12, 'same_paid_history': False}, 'later_adaptive_path'),
])
def test_only_saved_common_prefix10_is_initialization_state(updates, expected):
    row = dict(paid_prefix=10, status='available', same_paid_history=True,
               same_physical_pool=True, query_changed=True, first_divergence=True)
    row.update(updates)
    assert paths.query_scope(row) == expected


@pytest.mark.parametrize('updates,expected', [
    ({}, 'later_common_state_first_divergence'),
    ({'same_paid_history': False}, 'later_adaptive_path'),
    ({'same_physical_pool': False}, 'later_adaptive_path'),
    ({'first_divergence': False}, 'later_adaptive_path'),
    ({'both_observations_returned': False}, 'later_adaptive_path'),
])
def test_later_first_divergence_requires_actual_shared_history_pool_and_returned_observations(updates, expected):
    row = dict(paid_prefix=11, status='available', same_paid_history=True,
               same_physical_pool=True, query_changed=True, first_divergence=True,
               both_observations_returned=True)
    row.update(updates)
    assert paths.query_scope(row) == expected


def fixture(root):
    """Extend the existing saved Stage85 fixture with both permuted histories."""
    plan = old_fixture(root)
    plan.update(output_dir='analysis', historical_accounts=['source/completion.json'], limits=[])
    plan['budget'] = dict(terminal_cells=22, verification_cap_per_cell=352)
    pairs = ((factorial.PERMUTED_AFTER, factorial.POINT), (factorial.PERMUTED_POINT, factorial.POINT))
    for arm, _ in pairs:
        plan['terminal_arms'].append(dict(name=arm, history_arm=arm, terminal_backend='role'))
    saved = json.loads((root/'source/cases/seed_7/result.json').read_text())
    index = json.loads((root/'source/mechanism/seed_7/index.json').read_text())
    sealed = []
    for stream, entry in zip(saved['streams'], index['streams']):
        name = stream['name']
        filename = f'source/cases/seed_7/{name}/analysis.json'
        raw = json.loads((root/filename).read_text())
        original_states, original_queries = deepcopy(raw['states']), deepcopy(stream['queries'])
        for arm, old_arm in pairs:
            original = next(row for row in entry['cases'] if row['arm'] == old_arm)
            case = deepcopy(original)
            case.update(arm=arm, history_arm=arm, source_case=f'search/{name}/{arm}')
            history = json.loads((root/(original['source_case']+'/search.json')).read_text())
            audit = json.loads((root/(original['source_case']+'/query_audit.json')).read_text())
            save(root, case['source_case']+'/search.json', history)
            save(root, case['source_case']+'/query_audit.json', audit)
            item = json.loads((root/original['old']['input_file']).read_text())
            item['terminal_recording'] = selection_record(history['history'], item['frozen_shortlist'], arm, 1)
            record = json.loads((root/original['old']['verification_file']).read_text())
            case['old'].update(input_file=f'inputs/{name}/{arm}.json',
                               verification_file=f'verification/{name}/{arm}.json')
            save(root, case['old']['input_file'], item)
            save(root, case['old']['verification_file'], record)
            entry['cases'].append(case)
            new = deepcopy(next(row for row in stream['outcomes'] if row['arm'] == old_arm))
            new.update(arm=arm, history_arm=arm)
            stream['outcomes'].append(new)
            raw['states'].extend(dict(deepcopy(row), arm=arm) for row in original_states if row['arm'] == old_arm)
            stream['queries'].extend(dict(deepcopy(row), arm=arm) for row in original_queries if row['arm'] == old_arm)
        for label, left, right in (
                ('source_AFTER', factorial.AFTER, factorial.PERMUTED_AFTER),
                ('source_POINT', factorial.POINT, factorial.PERMUTED_POINT),
                ('permuted_AP', factorial.PERMUTED_AFTER, factorial.PERMUTED_POINT)):
            for prefix in (10, 11, 12):
                selected = {arm: next(row for row in raw['states'] if row['arm'] == arm
                    and row['paid_prefix'] == prefix)['selected']['policy_id'] for arm in (left, right)}
                changed = selected[left] != selected[right]
                raw['comparisons'].append(dict(contrast=label, left=left, right=right, paid_prefix=prefix,
                    status='available', same_paid_history=prefix == 10 or not changed,
                    same_physical_pool=True, both_observations_returned=True, query_changed=changed,
                    first_divergence=prefix == 10 and changed, selected=selected))
        for row in stream['outcomes']:
            row.update(seed=7, domain='fixture', discovery_category=MULTI
                if row['deployed_policy_id'] == 'P0010' else 'initial', returned_repeat_count=0)
        for case in entry['cases']:
            case['stream'] = name
            sealed.append(deepcopy(case))
        save(root, filename, raw)
    save(root, 'source/mechanism/seed_7/index.json', index)
    save(root, 'source/cases/seed_7/result.json', saved)
    source = dict(plan, output_dir='source', status='complete')
    save(root, 'source.json', source)
    save(root, 'source/completion.json', dict(status='complete'))
    save(root, 'source/execution_status.json', dict(status='sealed', truth_access_allowed=True, cases=sealed))
    save(root, 'source/mechanism/inventory.json', dict(status='sealed', truth_access_allowed=True,
        inventory_complete=True, registered_terminal_cells=22, registered_query_states=30))
    science = dict(status='complete', registered_geometries=1, registered_terminal_cells=22,
        source_numerical_limits='Queue low ESS is retained; precision_certified=false',
        historical_accounts=['older/completion.json'], costs={'physical_target_calls': 1234},
        independent={'original_science': 'unchanged'})
    save(root, 'source/headline.json', science)
    return plan, science


def test_saved_end_to_end_five_roles_eleven_controls_and_no_new_science(tmp_path, monkeypatch):
    plan, _ = fixture(tmp_path)
    from performance import summarize_mainline_source_alignment as scorer
    from performance import summarize_mainline_precision_confirmation as original_scorer
    def forbidden(*args, **kwargs):
        pytest.fail('Saved attribution must not score, restore a model, or draw randomness')
    monkeypatch.setattr(scorer, 'score_task', forbidden)
    monkeypatch.setattr(original_scorer, 'score_task', forbidden)
    monkeypatch.setattr(original_scorer.audit, 'score_points', forbidden)
    monkeypatch.setattr(pickle, 'load', forbidden)
    monkeypatch.setattr(np.random, 'default_rng', forbidden)
    before = {str(p): p.stat().st_mtime_ns for p in tmp_path.rglob('*') if p.is_file()}
    paths.require_source(plan, root=tmp_path)
    result = paths.analyze_task(tmp_path, plan, 7)
    for stream in result['streams']:
        assert len(stream['outcomes']) == len(stream['policy_terminal_paths']) == 11
        assert {row['arm'] for row in stream['outcomes']} == set(factorial.ARMS)
        assert {row['contrast'] for row in stream['query_comparisons']} == {
            'AP', 'FA', 'source_AFTER', 'source_POINT', 'permuted_AP'}
        assert len(stream['query_comparisons']) == 15
        cell = next(row for row in stream['policy_terminal_paths'] if row['arm'] == factorial.AFTER)
        missing = next(row for row in cell['observed'] if row['policy_id'] == 'P0011')
        third = next(row for row in cell['observed'] if row['policy_id'] == 'P0000')
        assert missing['terminal_path']['terminal_reason'] == 'not_on_frozen_shortlist'
        assert third['terminal_path']['terminal_reason'] == 'not_verified_after_earlier_terminal_decision'
        assert third['terminal_path']['objective_comparison_coverage'] == []
        assert next(row for row in stream['policy_terminal_paths']
                    if row['arm'] == factorial.PERMUTED_AFTER)['selection_record_status'] == 'complete'
        measures = paths.stream_measures(stream)
        assert measures[factorial.AFTER+'.returned_repeat'] == 1
        assert stream['four_arm_initial_common_state'] is True
    assert before == {str(p): p.stat().st_mtime_ns for p in tmp_path.rglob('*') if p.is_file()}


def test_registration_accepts_retained_partial_inventory_but_requires_every_sealed_cell(tmp_path):
    plan, _ = fixture(tmp_path)
    path = tmp_path/'source/mechanism/inventory.json'
    inventory = json.loads(path.read_text())
    inventory['inventory_complete'] = False
    save(tmp_path, str(path.relative_to(tmp_path)), inventory)
    paths.require_source(plan, root=tmp_path)
    seal = json.loads((tmp_path/'source/execution_status.json').read_text())
    seal['cases'].pop()
    save(tmp_path, 'source/execution_status.json', seal)
    with pytest.raises(ValueError):
        paths.require_source(plan, root=tmp_path)


@pytest.mark.parametrize('executed', [False, True])
def test_saved_actual_objective_test_remains_distinct_from_unexecuted_branch(tmp_path, executed):
    plan, _ = fixture(tmp_path)
    filename = f'verification/observation_1/{factorial.AFTER}.json'
    record = json.loads((tmp_path/filename).read_text())
    record['deployed'] = [1]
    record['verification']['attempts'][0]['certified'] = executed
    record['verification']['objective_comparison'] = (dict(status='challenger_not_dominant', simulation_calls=128,
        challenger_dominates=False, sample_mean_difference=-.01, one_sided_upper_confidence_bound=.03)
        if executed else dict(status='challenger_not_safety_certified', simulation_calls=0))
    save(tmp_path, filename, record)
    saved = json.loads((tmp_path/'source/cases/seed_7/result.json').read_text())
    row = next(row for row in saved['streams'][0]['outcomes'] if row['arm'] == factorial.AFTER)
    row.update(reported_policy_id='P0001', deployed_policy_id='P0001')
    save(tmp_path, 'source/cases/seed_7/result.json', saved)
    result = paths.analyze_task(tmp_path, plan, 7)
    cell = next(row for row in result['streams'][0]['policy_terminal_paths'] if row['arm'] == factorial.AFTER)
    path = next(row for row in cell['observed'] if row['policy_id'] == 'P0010')['terminal_path']
    assert path['terminal_reason'] == ('actual_objective_comparison_retained_other_policy'
                                      if executed else 'safety_certificate_failed')
    assert path['objective_comparison_coverage'][0]['executed'] is executed


def test_path_parse_failure_preserves_all_eleven_saved_scientific_outcomes(tmp_path):
    plan, _ = fixture(tmp_path)
    original = json.loads((tmp_path/'source/cases/seed_7/result.json').read_text())
    (tmp_path/'source/cases/seed_7/observation_2/analysis.json').write_text('{invalid')
    result = paths.analyze_task(tmp_path, plan, 7)
    assert result['streams'][1]['status'] == 'path_analysis_failed'
    assert result['streams'][1]['outcomes'] == original['streams'][1]['outcomes']
    assert len(result['streams'][1]['outcomes']) == 11


def reference_summary(tasks, plan):
    seal = dict(status='sealed', truth_access_allowed=True, cases=[dict(stream=stream['name'],
        seed=task['seed'], domain=task['domain'], arm=row['arm'], old={'status': row['status']})
        for task in tasks for stream in task['streams'] for row in stream['outcomes']])
    result = factorial.summarize_loaded(plan, tasks, seal=seal,
        accounts={'physical_target_calls': 1234, 'retained_failure_cpu_seconds': 9.})
    result['source_numerical_limits'] = 'Queue low ESS retained; precision_certified=false'
    return result


def test_summary_uses_common_four_arm_two_stream_support_and_retains_every_control(tmp_path):
    plan, _ = fixture(tmp_path)
    first = paths.analyze_task(tmp_path, plan, 7)
    second = deepcopy(first)
    second['seed'] = 8
    plan['tasks'].append(dict(geometry_task_seed=8, domain='fixture'))
    plan['budget']['terminal_cells'] = 44
    for task in (first, second):
        for index, stream in enumerate(task['streams']):
            for row in stream['outcomes']:
                row.update(outcome(), seed=task['seed'], domain='fixture')
                if row['arm'] == factorial.AFTER:
                    row.update(outcome(gain=(.4 if index == 0 else -.1) if task['seed'] == 7 else .3))
                elif row['arm'] == factorial.POINT and task['seed'] == 7 and index == 1:
                    row.update(outcome(gain=.2))
    missing = next(row for row in second['streams'][1]['outcomes'] if row['arm'] == factorial.PERMUTED_AFTER)
    missing.update(outcome(status='missing'))
    science = reference_summary([first, second], plan)
    before = deepcopy(science)
    result = paths.summarize([first, second], plan, science)
    effects = {row['name']: row for row in result['independent']['comparisons']}
    interaction = effects['source_by_propagation_interaction']
    stat = interaction['delta']['G']
    assert stat['registered_geometries'] == 2 and stat['available_geometries'] == 1
    assert stat['all_registered_mean'] is None
    assert stat['conditional_available']['mean'] == pytest.approx(.05)
    assert interaction['rows'][1]['delta']['G'] is None
    assert effects['real_AFTER_minus_POINT']['delta']['G']['available_geometries'] == 2
    assert len(result['retained_scientific_outcomes']) == 44
    assert result['retained_scientific_status_counts'] == {'complete': 43, 'missing': 1}
    assert result['saved_science'] == science == before
    for row in interaction['rows']:
        assert_partitions_add_to_saved_delta(row)


def test_summary_path_failure_does_not_change_saved_science_fees_or_ESS_warning(tmp_path):
    plan, _ = fixture(tmp_path)
    saved = json.loads((tmp_path/'source/cases/seed_7/result.json').read_text())
    science = reference_summary([saved], plan)
    (tmp_path/'source/cases/seed_7/observation_2/analysis.json').write_text('{invalid')
    analyzed = paths.analyze_task(tmp_path, plan, 7)
    result = paths.summarize([analyzed], plan, science)
    assert result['saved_science'] == science
    assert len(result['analysis_failures']) == 1
    assert len(result['retained_scientific_outcomes']) == 22
    primary = result['independent']['comparisons'][0]
    assert primary['delta']['C']['available_geometries'] == 1
    query = result['paired_query_measures']['measures']['four_arm_initial_common_state']
    assert query['all_registered_mean'] is None and query['available_geometries'] == 0
    assert result['new_work'] and all(value == 0 for value in result['new_work'].values())
    compact = paths.compact_summary(result)
    assert len(compact['saved_scientific_controls']) == 11
    assert compact['saved_scientific_costs'] == science['costs']
    assert compact['source_numerical_limits'] == science['source_numerical_limits']


def test_missing_search_and_terminal_history_do_not_become_zero_known_opportunities(tmp_path):
    plan, _ = fixture(tmp_path)
    arm, name = factorial.PERMUTED_AFTER, 'observation_1'
    filename = 'source/cases/seed_7/result.json'
    saved = json.loads((tmp_path/filename).read_text())
    stream = saved['streams'][0]
    row = next(row for row in stream['outcomes'] if row['arm'] == arm)
    row.update(outcome(status='missing'), deployed_policy_id=None, actual_deployed_quality=None)
    for query in stream['queries']:
        if query['arm'] == arm:
            query.update(status='unavailable', observation_returned=False,
                         returned_policy_matches=False, query_action_kind=None,
                         selected_policy_id=None)
    save(tmp_path, filename, saved)
    index = json.loads((tmp_path/'source/mechanism/seed_7/index.json').read_text())
    case = next(row for row in index['streams'][0]['cases'] if row['arm'] == arm)
    case['old'].update(status='missing', input_file=None, verification_file=None)
    (tmp_path/(case['source_case']+'/search.json')).unlink()
    save(tmp_path, 'source/mechanism/seed_7/index.json', index)
    filename = f'source/cases/seed_7/{name}/analysis.json'
    raw = json.loads((tmp_path/filename).read_text())
    raw['states'] = [state for state in raw['states'] if state['arm'] != arm]
    for comparison in raw['comparisons']:
        if arm in (comparison['left'], comparison['right']):
            comparison.update(status='unavailable', same_paid_history=None,
                same_physical_pool=None, query_changed=None, both_observations_returned=False)
    save(tmp_path, filename, raw)
    result = paths.analyze_task(tmp_path, plan, 7)
    stream = result['streams'][0]
    values = paths.stream_measures(stream)
    assert values[arm+'.returned_new'] is None
    assert values[arm+'.returned_repeat'] is None
    assert values[arm+'.captured_query_states'] == 0
    assert values[arm+'.known_prefix.returned_new'] == 0
    assert values.get(arm+'.terminal.known_improving_safe') is None
    assert values.get(arm+'.terminal.known_safe_better_than_deployment') is None
    assert stream['four_arm_initial_common_state'] is None
    assert next(row for row in stream['outcomes'] if row['arm'] == arm)['metrics']['C'] is None
    coverage = next(row for row in stream['history_coverage'] if row['arm'] == arm)
    assert coverage['completed_observed_pool_available'] is False
    assert coverage['observed_pool_size'] is None
    assert coverage['captured_query_states'] == coverage['known_returned_query_prefix'] == 0
    assert coverage['unavailable_query_states'] == coverage['registered_query_rows'] == 3


def test_collect_preserves_failed_analysis_CPU_and_historical_science_without_recharging(tmp_path):
    plan, _ = fixture(tmp_path)
    saved = json.loads((tmp_path/'source/cases/seed_7/result.json').read_text())
    science = reference_summary([saved], plan)
    save(tmp_path, 'source/headline.json', science)
    save(tmp_path, 'checks.json', {'status': 'passed'})
    paths.register(plan, 'plan.json', 'checks.json', root=tmp_path)
    failed = paths.failed_task(plan, plan['tasks'][0], 'saved analysis failure', saved)
    save(tmp_path, 'analysis/cases/seed_7/result.json', failed)
    ledger = tmp_path/'analysis/cases/seed_7/attempts.jsonl'
    ledger.write_text(json.dumps(dict(stage='analyze-source-alignment-paths', status='failed',
        cpu_seconds=2., **paths.ZERO))+'\n')
    completion = paths.collect(plan, 'plan.json', root=tmp_path)
    assert completion['status'] == 'complete_with_analysis_failures'
    assert len(completion['failed_attempts']) == 1
    assert completion['recorded_analysis_cpu_seconds'] >= 2.
    assert all(value == 0 for value in completion['new_work'].values())
    assert json.loads((tmp_path/'source/headline.json').read_text()) == science
    headline = json.loads((tmp_path/'analysis/headline.json').read_text())
    assert headline['saved_scientific_costs'] == science['costs']
    assert len(headline['saved_scientific_controls']) == 11
