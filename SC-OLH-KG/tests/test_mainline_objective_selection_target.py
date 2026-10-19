"""Prevent false objective replacements, truth claims and deployment-union entries."""
from performance import mainline_objective_selection_target as target


def state(mean, sampled, margin, actual, truth=None):
    bank = [(1,), (2,), (3,)]
    registry = {p: f'P{i:03d}' for i, p in enumerate(bank)}
    stage = dict(paid_prefix=1, action_ids=[0, 1, 2], selected_index=actual,
        selected_id=actual, score=dict(objective_mean=mean, sampled_objective=sampled,
        sampled_margin=margin, objective_epistemic=[.1, .1, .1],
        sampled_feasibility_mode='feasible_objective' if min(margin) <= 0 else 'minimum_violation'))
    return target.analyze_state(stage, bank, [dict(point=[1]), dict(point=list(bank[actual]))],
        {(1,)}, {(2,)}, registry, truth or {}, list(registry.values()),
        registry[bank[actual]], 'P001')


def test_actual_risk_set_is_fixed_and_empty_set_has_no_mean_replacement():
    # An excellent objective at an ineligible point cannot become a mean winner.
    result = state([.3, -99., .1], [.1, -99., .2], [-1., 1., -1.], 0)
    assert result['actual']['policy_id'] == 'P000'
    assert result['mean_choice']['policy_id'] == 'P002'
    assert result['mean_choice_changed'] and result['actual_repeated_query']
    assert result['union_candidates'][1]['mean_rank'] is None
    # Minimum-violation selection remains an actual repeat, without a replacement.
    empty = state([2., 0., 1.], [3., 1., 2.], [.1, .2, .3], 0)
    assert empty['actual_repeated_query']
    assert empty['mean_choice'] is None and empty['mean_choice_changed'] is None
    assert empty['mean_choice_status'] == 'not_applicable_no_eligible_policy'


def test_wrong_mean_order_and_draw_reversal_are_separate_and_truth_stays_unknown():
    truth = {(1,): dict(objective=2., true_feasible=True, quality=.2),
             (2,): dict(objective=1., true_feasible=True, quality=.3)}
    reversed_draw = state([2., 1., 4.], [0., 3., 4.], [-1., -1., -1.], 0, truth)
    candidate = reversed_draw['union_candidates'][1]
    pair = candidate['versus_actual']
    assert not candidate['previously_observed']
    assert pair['ordering'] == 'draw_reverses_mean_preference'
    assert pair['cached_truth_mean_order'] == 'mean_correct_order'
    assert pair['cached_truth_order_transition'] == 'mean_correct_draw_wrong'
    assert pair['mean_difference'] + pair['draw_difference'] == pair['sampled_difference']
    group = next(g for g in reversed_draw['groups'] if g['origin'] == 'shared_neighbor'
                 and not g['previously_observed'])
    assert group['cached_better_feasible_omission_by_cause'] == {'draw_reverses_correct_mean': 1}
    wrong_mean = state([1., 2., 4.], [0., 3., 4.], [-1., -1., -1.], 0, truth)
    assert wrong_mean['union_candidates'][1]['versus_actual']['cached_truth_mean_order'] == 'mean_wrong_order'
    unknown = wrong_mean['union_candidates'][2]['versus_actual']
    assert unknown['true_objective_difference'] is None
    assert unknown['cached_truth_mean_order'] == 'unknown'
    repaired_truth = {(1,): truth[(1,)],
                      (2,): dict(objective=3., true_feasible=True, quality=.1)}
    repaired = state([2., 1., 4.], [0., 3., 4.], [-1., -1., -1.], 0, repaired_truth)
    assert repaired['mean_choice_versus_actual']['cached_truth_order_transition'] == 'mean_wrong_draw_correct'


def test_actual_query_union_keeps_shared_physical_ids_and_excludes_uncertified_reports():
    def protocol(pid, certified):
        return dict(status='complete', certified=certified,
            deployed_policy_id=pid if certified else None, reported_policy_id=pid)
    def stream(name, arm, queried, deployed, certified):
        return dict(name=name, terminal=dict(arms=[dict(arm=arm,
            adaptive_queries=[dict(policy_id=queried, history_index=12)],
            protocols={'new': protocol(deployed, certified)})]))
    saved = dict(terminal=dict(streams=[stream('one', 'AFTER', 'P007', 'P008', False),
        stream('two', 'POINT', 'P007', 'P009', True)]))
    union, uncertified = target.policy_union(saved)
    assert set(union) == {'P007', 'P009'}
    assert len(union['P007']) == 2
    assert uncertified == [dict(stream='one', arm='AFTER', protocol='new', policy_id='P008')]
