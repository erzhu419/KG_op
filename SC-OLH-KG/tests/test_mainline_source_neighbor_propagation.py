"""Saved-gap fixtures: no source raw, posterior reconstruction or random draws."""
import copy

import pytest

from performance import mainline_source_neighbor_propagation as propagation


def pair(i, j, label=1.):
    return dict(i=i, j=j, policy_ids=[100+i, 100+j], distance_squared=.1, selected_from=[i],
        observed={str(prefix): dict(pair_group='observed_observed' if j < prefix else
            'observed_unobserved' if i < prefix else 'unobserved_unobserved') for prefix in (10, 13)},
        unused_mean_difference=label, independent_mean_difference_standard_error=.2,
        leave_one_endpoint_repeat=dict(all_same_sign_as_full=True, minimum=.5, maximum=1.5),
        empirical_feasible=[True, True], empirical_both_feasible=True)


def states(values):
    return {name: {str(prefix): dict(status='complete', gap=gap, both_predicted_eligible=True,
        endpoint_eligible=[True, True], endpoint_probability_violation=[.1, .1])
        for prefix, gap in zip((10, 13), values.get(name, (1., 1.)))} for name in propagation.ARMS}


def test_sign_spoiling_can_improve_absolute_error_by_overshooting_label():
    overshoot = propagation.compare_gap(3., -.5, 1.)
    assert overshoot['transition'] == 'correct_to_wrong'
    assert overshoot['absolute_gap_error_change'] == -.5
    assert overshoot['new_wrong_mechanism'] == 'toward_label_overshoot_absolute_error_improved'
    away = propagation.compare_gap(.5, -.5, 1.)
    assert away['transition'] == 'correct_to_wrong' and away['absolute_gap_error_change'] == 1.
    assert away['new_wrong_mechanism'] == 'away_from_label'
    assert propagation.compare_gap(-1., 1., 0.)['transition'] == 'label_tie'
    assert propagation.compare_gap(None, 1., 1.)['transition'] == 'unknown'
    assert propagation.compare_gap(0., 1., 1.)['transition'] == 'wrong_to_correct'


def test_timeline_separates_initialization_self_update_interaction_and_continuation():
    result = propagation.decompose_pair(pair(10, 11), states(dict(
        POINT=(2., 3.), AFTER=(5., 9.), FULL=(5., 11.))))
    timeline = result['timeline']
    assert timeline['point_self_change_13_minus_10'] == 1.
    assert timeline['initialization_effect_after10_minus_point10'] == 3.
    assert timeline['initialization_self_update_interaction'] == 3.
    assert timeline['after_self_change_13_minus_10'] == 4.
    assert timeline['continued_full13_minus_after13'] == 2.
    assert timeline['reconstructed_full13'] == 11. and timeline['reconstruction_residual'] == 0.


def source_fixture():
    pairs = [pair(10, 11), pair(11, 14), pair(13, 14)]
    arms = []
    for name, arm in propagation.ARMS.items():
        snapshots = {}
        for prefix in (10, 13):
            gaps = ([1., -1., 1.] if name == 'POINT' else [-.5, 1., 1.])
            snapshots[str(prefix)] = dict(status='complete', failure=None, pairs=[dict(
                pair_index=index, predicted_difference=gap, both_predicted_eligible=True,
                endpoint_eligible=[True, True], endpoint_probability_violation=[.1, .1])
                for index, gap in enumerate(gaps)])
        arms.append(dict(arm=arm, status='complete', failures=[], snapshots=snapshots))
    case = dict(archive_heldout_family='archive', fold=0, family='fixture', d=2,
        held_policy_ids=list(range(16)), arms=arms, pairs=pairs)
    return dict(coverage=dict(source_groups=1, arm_cases=4, snapshots=8, completed_snapshots=8,
        fixed_pairs=3, failed_arm_cases=[]), cases=[case], prior_paid_accounting=dict(cost={'cpu_seconds': 1.}))


def test_initial_uu_population_follows_all_pairs_to_final_observation_states(monkeypatch, tmp_path):
    source = source_fixture()
    monkeypatch.setattr(propagation, 'read', lambda path: source)
    result = propagation.analyze_propagation(tmp_path, dict(inputs=dict(neighbor_analysis='saved.json')))
    initial = next(row for row in result['aggregates'] if
        (row['cohort'], row['contrast']) == ('initial_unobserved_unobserved', 'AP13'))
    final = next(row for row in result['aggregates'] if
        (row['cohort'], row['contrast']) == ('final_unobserved_unobserved', 'AP13'))
    assert initial['pairs'] == 3 and final['pairs'] == 1
    assert initial['transitions'] == dict(wrong_to_correct=1, correct_to_wrong=1,
        same_correct=1, same_wrong=0, label_tie=0, unknown=0)
    compact = propagation.compact_propagation(result)
    assert len(compact['original_uu_pairs']) == 3
    assert {row['final_group'] for row in compact['original_uu_pairs']} == set(propagation.COHORTS[1:4][i].removeprefix('initial_') for i in range(3))
    assert result['final_uu_saved_gap_changes']['POINT']['maximum_absolute_gap_change'] == 0
    assert result['cases'][0]['pairs'][0]['states']['POINT']['13']['point_reference_semantics'] == 'point_only_reference'
    assert result['cases'][0]['pairs'][2]['states']['POINT']['13']['point_reference_semantics'] == 'prior_only_unobserved_pair'


def test_saved_missing_snapshot_remains_unknown_without_erasing_later_gap(monkeypatch, tmp_path):
    source = copy.deepcopy(source_fixture())
    point = next(arm for arm in source['cases'][0]['arms'] if arm['arm'] == propagation.POINT)
    point.update(status='partial_failure', failures=['saved initial failure'])
    point['snapshots']['10'] = dict(status='failed', failure='saved initial failure', pairs=[])
    source['coverage'].update(completed_snapshots=7, failed_arm_cases=[dict(arm=propagation.POINT)])
    monkeypatch.setattr(propagation, 'read', lambda path: source)
    result = propagation.analyze_propagation(tmp_path, dict(inputs=dict(neighbor_analysis='saved.json')))
    row = result['cases'][0]['pairs'][0]
    assert row['contrasts']['AP10']['transition'] == 'unknown'
    assert row['contrasts']['AP13']['transition'] == 'correct_to_wrong'
    assert row['timeline']['reconstructed_full13'] is None and row['timeline']['full13'] == -.5
    assert result['coverage']['snapshots'] == 8 and result['coverage']['completed_snapshots'] == 7
    compact = propagation.compact_propagation(result)
    assert any(row[2:4] == ['AP10', 'unknown'] for row in compact['cases'][0]['changed_pairs'])
