"""Saved-evidence boundaries: exact identity, selection reason, unknown tests."""
import json

from performance.mainline_confirmation_query_attribution import (
    AFTER, FULL, POINT, aggregate_queries, candidate_path, compact_task, terminal_path,
)


def bundle():
    score = dict(sampled_objective=[3.,2.,1.], objective_mean=[3.,2.,1.],
        sampled_margin=[-.1,-.1,.2], sampled_feasibility_mode='feasible_objective',
        objective_epistemic=[.1]*3,constraint_mean=[-.1,-.1,.2],
        constraint_aleatoric=[0.]*3,constraint_epistemic=[.1]*3,
        sampled_constraint=[-.1,-.1,.2],stochastic_margin_mean=[-.1,-.1,.2],
        probability_violation=[.1,.1,.8])
    return dict(bank=[(1,1),(2,2),(3,3)], history=[dict(point=[1,1])]*10,
        audit=dict(stages=[dict(paid_prefix=10,action_ids=[0,1,2],
            action_sources=['initial','shared','dynamic'],selected_index=1,score=score)]))


def test_saved_risk_qualification_and_objective_competition_are_separate():
    data = bundle()
    first = candidate_path((1,1),data)['states'][0]
    second = candidate_path((2,2),data)['states'][0]
    third = candidate_path((3,3),data)['states'][0]
    assert first['reason']=='objective_competition' and first['sampled_feasible']
    assert second['reason']=='selected'
    assert third['reason']=='risk_excluded' and third['sampled_objective_rank_all']==1


def test_policy_identity_requires_exact_physical_vector_and_absence_is_not_a_gate():
    data = bundle()
    present = candidate_path((2,2),data)
    absent = candidate_path((2,3),data)
    assert present['first_candidate_paid_prefix']==10
    assert absent['first_candidate_paid_prefix'] is None
    assert absent['observation_history_indices']==[]
    assert absent['states']==[dict(paid_prefix=10,present=False)]


def test_candidate_never_verified_has_unknown_certificate_not_failed_certificate():
    item = dict(frozen_shortlist=[dict(point=[1,1]),dict(point=[2,2]),dict(point=[3,3])])
    record = dict(deployed=[1,1],verification=dict(certified=True,attempts=[
        dict(candidate_index=0,certified=True,upper_margin=-.1)],
        objective_incumbent_position=2,objective_comparison=dict(status='not_required')))
    path = terminal_path((3,3),item,record,'complete')
    assert path['disposition']=='not_verified_after_earlier_terminal_decision'
    assert path['safety_passed'] is None
    assert path['counterfactual_certificate'] is None
    assert terminal_path((3,4),item,record,'complete')['disposition']=='not_on_frozen_shortlist'


def test_third_comparison_is_matched_by_physical_vector_and_actual_index():
    item = dict(frozen_shortlist=[dict(point=[1,1]),dict(point=[2,2]),dict(point=[3,3])])
    record = dict(deployed=[3,3],verification=dict(certified=True,attempts=[
        dict(candidate_index=2,certified=True,upper_margin=-.1)],objective_comparisons=[
        dict(block_index=0,challenger_index=0,incumbent_index=1,winner_index=1),
        dict(block_index=1,challenger_index=2,incumbent_index=1,winner_index=2)]))
    path = terminal_path((3,3),item,record,'complete')
    assert path['disposition']=='deployed'
    assert path['actual_objective_comparisons']==[
        dict(block_index=1,challenger_index=2,incumbent_index=1,winner_index=2)]


def test_compaction_keeps_all_choices_first_divergence_and_unknown_exposure():
    truth = dict(true_feasible=True,quality=.2,objective=.8)
    identity = dict(policy='neighbor:2',origin='shared_neighbor',truth=truth)
    selected = dict(identity=identity,truth=truth,new_good=True)
    states = [dict(arm=a,paid_prefix=p,candidate_count=3,cached_truth_action_count=1,
        unknown_truth_action_count=2,choices=dict(actual=dict(selected=selected,mode='feasible_objective')))
        for a in (FULL,AFTER,POINT) for p in (10,11,12)]
    pairs = []
    for left,right in ((AFTER,POINT),(FULL,AFTER)):
        for prefix in (10,11,12):
            swaps = [dict(receiver=receiver,donor=donor,substitution=component,
                donor_choice_recovered=component.startswith('risk_'),selection_changed=True,
                selected=dict(neighbor_index=2,left_bank_id=2))
                for receiver,donor in ((left,right),(right,left))
                for component in ('risk_risk_mean+noise_margin+risk_draw','all_objective')]
            pairs.append(dict(left=left,right=right,paid_prefix=prefix,query_changed=True,
                same_paid_history=prefix==10,same_physical_pool=True,first_divergence=prefix==10,
                selected={left:selected,right:selected},substitutions=swaps if prefix==10 else []))
    raw = dict(status='complete',seed=7,domain='InventorySupplyChain',states=states,
        paired_states=pairs,task=dict(quality_baseline_defined=True,cached_truth_count=10),
        reconstruction=dict(recorded_choices=9,max_score_error=0.),deployment_paths=[dict(
            protocol='new',after_policy=identity,after_certified=True,paths=[dict(arm=a,
                search=candidate_path((3,3),bundle()),terminals={p:dict(status='complete',
                    shortlist_position=None,disposition='not_on_frozen_shortlist') for p in ('old','new')})
                for a in (FULL,AFTER,POINT)])])
    compact = compact_task(raw)
    assert len(compact['actual_queries'])==9
    assert compact['truth_coverage']['unknown_truth_exposures']==18
    assert compact['after_new']['paths']['POINT']['search']['states'][0]['reason']=='risk_excluded'
    assert compact['after_new']['paths']['POINT']['terminals']['new']['safety_passed'] is None
    assert len(json.dumps(compact,separators=(',',':')).encode()) < 8192
    summary = aggregate_queries([compact])
    for comparison in summary['comparisons']:
        assert comparison['paired_states']==3 and comparison['changed_queries']==3
        assert comparison['first_divergences']==1
        assert comparison['swaps'][0]['counts']==dict(risk_only=1,objective_only=0,
            both_alone=0,neither_alone=0)
