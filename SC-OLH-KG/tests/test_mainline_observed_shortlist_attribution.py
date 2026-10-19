import json

from performance.mainline_observed_shortlist_attribution import (
    analyze_arm, analyze_task, challenger_reason, final_evidence,
)


def rows():
    return [dict(point=[i], raw=[float(i), -1.]) for i in range(4)]


def frozen(count=2):
    return dict(primary=[0], terminal=dict(objective_incumbent_guard_enabled=True,
        objective_challenger=dict(status='selected', eligible_count=count,
            maximum_violation_probability=.5, selected_posterior_objective=.1,
            selected_probability_violation=.2), frozen_shortlist=[dict(point=[0],
            selected_posterior_objective=.1, selected_probability_violation=.2,
            selection_contract='posterior_median_feasible_minimum_posterior_objective'),
            dict(point=[1], shortlist_role='frozen_initial_empirical_objective_incumbent'),
            dict(point=[2], shortlist_role='posterior_bayes_fallback')]))


def test_missing_default_scores_do_not_borrow_native_or_empirical_values():
    scores, challenger = final_evidence(frozen(), rows())
    assert scores[(3,)]['objective'] is None
    assert scores[(3,)]['eligible'] is None
    assert challenger_reason((3,), scores, challenger) == 'unknown_final_risk_eligibility'
    assert scores[(0,)]['objective'] == .1
    assert scores[(0,)]['objective_rank_all'] is None


def test_incumbent_overwritten_contract_retains_challenger_identity_from_role():
    saved = frozen()
    selected = saved['terminal']['frozen_shortlist'][0]
    selected.update(shortlist_role='posterior_objective_primary', objective_incumbent=True,
        selection_contract='minimum_empirical_objective_in_frozen_initial_design')
    scores, challenger = final_evidence(saved, rows())
    assert challenger['point'] == [0]
    assert challenger_reason((0,), scores, challenger) == 'selected_objective_challenger'
    assert scores[(0,)]['objective'] == .1 and scores[(0,)]['eligible'] is True
    assert scores[(3,)]['objective'] is None and scores[(3,)]['eligible'] is None


def test_full_pool_count_can_identify_risk_eligibility_without_point_scores():
    scores, challenger = final_evidence(frozen(count=4), rows())
    assert scores[(3,)]['eligible'] is True
    assert scores[(3,)]['eligibility_source'] == 'saved_full_pool_eligible_count'
    assert challenger_reason((3,), scores, challenger) == 'eligible_not_selected_objective_score_unavailable'
    native = frozen(count=3)
    native['observed'] = [dict(point=[i], objective=i/10, probability_violation=.2 if i < 3 else .8)
        for i in range(4)]
    scores, challenger = final_evidence(native, rows())
    assert challenger_reason((2,), scores, challenger) == 'eligible_higher_posterior_objective_than_challenger'
    assert challenger_reason((3,), scores, challenger) == 'excluded_from_objective_challenger_by_final_risk'


def test_missing_baseline_and_incomplete_saved_shortlist_do_not_invent_displacement():
    saved = frozen()
    saved['primary'] = [3]  # Missing guaranteed entry is inconsistent or incomplete evidence.
    case = dict(arm='a', history_arm='a', old=dict(status='missing', item=None),
                new=dict(status='missing', item=None))
    truth = {(i,):dict(quality=1-i/10, objective=i/10, true_feasible=True) for i in range(4)}
    result = analyze_arm(case, rows(), saved, truth, None, {}, {}, [])
    last = result['observed'][-1]
    assert last['improving_safe'] is None and last['quality_gain_to_initial'] is None
    assert last['shortlist_membership_reason'] == 'saved_guaranteed_candidate_missing_unresolved'
    assert last['protocols']['old']['disposition'] == 'terminal_input_unavailable'
    assert last['protocols']['old'].get('deployed') is None


def test_native_routing_uses_registered_arm_when_seal_omits_backend(tmp_path):
    def save(name, value):
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    case = dict(seed=1, domain='fixture', history_arm='a', source_case='task/a',
        old=dict(status='missing'), new=dict(status='missing'))
    save('cohort/execution_status.json', dict(status='sealed', cases=[
        dict(case, arm='a'), dict(case, arm='a_native')]))
    save('cohort/mechanism/seed_1/analysis.json', dict(query_chains=[], policies=[
        dict(point=[i], policy_id=f'P{i:04d}', origin='initial') for i in range(4)]))
    save('task/initialization.json', dict(history=rows()))
    save('task/truth.json', [])
    save('task/a/search.json', dict(frozen(), history=rows()))
    save('task/a/native_terminal.json', dict(frozen(count=4), observed=[
        dict(point=[i], objective=float(i), probability_violation=.2, eligible=True) for i in range(4)]))
    plan = dict(cohort_dir='cohort', terminal_arms=[dict(name='a', terminal_backend='role'),
        dict(name='a_native', terminal_backend='native')])
    result = analyze_task(tmp_path, plan, 1)
    original, native = result['arms']
    assert original['score_coverage']['full_table_rows'] == 0
    assert original['observed'][3]['final_scores']['objective'] is None
    assert native['score_coverage']['full_table_rows'] == 4
    assert native['observed'][3]['final_scores']['objective'] == 3.
    assert native['observed'][3]['policy_id'] == original['observed'][3]['policy_id'] == 'P0003'
