"""Saved-statistic substitutes; no real cohort or verifier is executed."""
from copy import deepcopy

import pytest

from performance.mainline_saved_terminal_comparison import (
    aggregate_tasks, analyze_pair, compact_task, comparison_diagnostics,
)


def comparison(mean=-.1, se=.003, critical=2.5, dominates=True):
    return dict(simulation_calls=16, comparison_stream_tag=1329744452,
        challenger_dominates=dominates, sample_mean_difference=mean,
        sample_std_difference=.01, standard_error=se, critical_value=critical,
        one_sided_upper_confidence_bound=mean + critical * se,
        delta=.05/6, degrees_of_freedom=7, comparison_budget_per_policy=8)


def fixture():
    values = dict(A=1, C=1, R=.8, G=.1, P=.1, L=0., false_certificate=0)
    reference = dict(arm='after', history_arm='after', status='complete',
        quality_reference_status='defined', quality_baseline_defined=True,
        best_initial_quality=.7, shortlist=['initial:0', 'initial:1', 'neighbor:2'])
    reference.update({field: dict(values) for field in
        ('old', 'new', 'B', 'new_minus_old', 'third_challenge', 'prechange')})
    item = dict(source_case='saved/after', verification_seed=31, objective_incumbent_position=2,
        frozen_shortlist=[dict(point=[i]) for i in range(3)])
    attempts = [dict(candidate_index=i, certified=True, upper_margin=-.1,
        verification_budget=n) for i, n in enumerate((80, 120, 120))]
    first, third = comparison(), comparison(mean=-.005, dominates=False)
    old = dict(deployed=[0], new_verification_calls=224, verification=dict(
        certified=True, attempts=attempts[:2], objective_comparison=first, verification_budget=224))
    new = dict(deployed=[0], cumulative_new_verification_calls=136, verification=dict(
        certified=True, attempts=attempts, initial_stage_winner_index=0,
        objective_comparisons=[dict(first, block_index=0, challenger_index=0, incumbent_index=1),
            dict(third, block_index=1, challenger_index=2, incumbent_index=0)], verification_budget=352))
    history = [dict(point=[i]) for i in [0, 1, *range(3, 11), 2]]
    truth = {(0,): dict(objective=.2, quality=.8, true_feasible=True),
             (1,): dict(objective=.3, quality=.7, true_feasible=True),
             (2,): dict(objective=.1, quality=.9, true_feasible=True)}
    return (dict(old=dict(status='complete'), new=dict(status='complete')), reference,
        dict(old=item, new=deepcopy(item)), dict(old=old, new=new), history, truth)


def test_favorable_direction_is_distinct_from_passing_the_saved_bound():
    record = comparison(mean=-.00005, se=.00003, critical=2.64, dominates=False)
    result = comparison_diagnostics(record, -.00007)
    assert result['disposition'] == 'favorable_mean_blocked_by_uncertainty'
    assert result['uncertainty_penalty'] == pytest.approx(.0000792)
    assert result['saved_upper_bound'] == pytest.approx(.0000292)
    assert result['point_estimate_sign_matches_truth'] is True
    assert result['true_better_objective_challenger_not_selected'] is True


def test_noisy_wrong_direction_is_not_described_as_only_a_tight_threshold():
    result = comparison_diagnostics(comparison(mean=.005, dominates=False), -.1)
    assert result['disposition'] == 'point_estimate_favors_incumbent'
    assert result['point_estimate_sign_matches_truth'] is False
    assert result['true_better_objective_challenger_not_selected'] is True
    assert comparison_diagnostics(comparison(), None)['true_better_objective_challenger_not_selected'] is None


def test_actual_third_opponent_and_unattempted_old_third_stay_distinct():
    args = fixture()
    registry = {(0,): 'P0073'}
    row = analyze_pair(*args, policy_registry=registry)
    third = row['protocols']['new']['comparisons'][1]
    assert third['incumbent_index'] == 0
    assert third['incumbent_policy_id'] == 'P0073'
    assert third['seed_namespace'] == [31, 'replication', 1329744452, 1]
    assert third['statistics']['disposition'] == 'favorable_mean_blocked_by_uncertainty'
    assert row['candidates'][2]['old']['safety']['status'] == 'not_attempted'
    assert len(row['comparison_slots']['old']) == 1
    assert row['comparison_slots']['new'][1]['incumbent_policy_id'] == 'P0073'
    task = compact_task(dict(seed=24, domain='Fixture', status='complete', arms=[row],
        quality_reference_status='defined'))
    aggregate = aggregate_tasks([task])
    new = next(group for group in aggregate['comparison_groups'] if group['protocol'] == 'new')
    assert new['executed_comparisons'] == 2
    assert new['comparison_disposition_counts']['favorable_mean_blocked_by_uncertainty'] == 1
    assert len(new['true_better_objective_challenger_not_selected']) == 1


def test_fallback_without_opponent_is_not_a_failed_comparison():
    args = fixture()
    old, new = args[3]['old']['verification'], args[3]['new']['verification']
    for record in (old, new):
        record['attempts'] = [dict(candidate_index=i, certified=i == 2,
            upper_margin=-.1 if i == 2 else .1, verification_budget=120) for i in range(3)]
    old['objective_comparison'] = dict(status='neither_primary_nor_incumbent_safety_certified', simulation_calls=0)
    new.update(objective_comparisons=[], initial_stage_winner_index=None)
    args[3]['old']['deployed'] = args[3]['new']['deployed'] = [2]
    row = analyze_pair(*args)
    assert row['protocols']['old']['comparisons'] == []
    assert row['protocols']['new']['comparisons'] == []
    assert row['comparison_slots']['new'][1]['status'] == 'third_only_without_first_stage_opponent'
    assert row['comparison_slots']['new'][1]['incumbent_policy_id'] is None


def test_execution_failure_keeps_unknown_branches_costs_and_missing_baseline():
    args = fixture()
    args[0]['new'] = dict(status='failed', failure_file='failure.json')
    args[3]['new'] = dict(new_verification_calls=7, failed_simulation_calls=1)
    args[1].update(quality_baseline_defined=False, quality_reference_status='no_true_feasible_initial',
        best_initial_quality=None, status='incomplete_pair')
    row = analyze_pair(*args)
    assert all(slot['status'] == 'unknown_after_failed' for slot in row['comparison_slots']['new'])
    assert row['protocols']['new']['saved_costs']['new_verification_calls'] == 7
    assert row['protocols']['new']['saved_costs']['failed_simulation_calls'] == 1
    assert row['coverage']['old']['unknown_good_classification'] == 3
    assert all(item['good_beyond_initial'] is None for item in row['observed_pool'])
