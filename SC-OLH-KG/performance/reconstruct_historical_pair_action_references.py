"""Reconstruct guarded action references from paid P/I certificate controls."""
import argparse
import itertools
import json
import signal
import time

import numpy as np
from scipy.special import logsumexp

from core.shared_certificate_pair import guard_statistics_from_pair_states
from core.branch_coupled_certificate_bound import coupled_branch_quality_upper
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.diagnose_historical_component_guarded import compressed_read, compressed_write
from performance.validate_historical_component_bank import action_groups
from performance.refine_historical_primary_incumbent_caps import inherited_primary_rows,merge_primary_caps,unit_rule_config
from performance.refine_historical_query_certificate_caps import load_caps
from performance import screen_historical_observed_batch_risk as screen


def reward_precision_checks(mixed,omission,evidence):
    """Apply the existing reward/evidence gate to every head and action."""
    comparisons=[]
    for low,high in itertools.combinations(mixed,2):
        differences=abs(np.asarray(mixed[low])-np.asarray(mixed[high]))
        comparisons.append(dict(low=low,high=high,maximum_difference=float(differences.max()),
            action_differences=differences.max(axis=0).tolist(),head_action_differences=differences.tolist()))
    reward_difference=max(c['maximum_difference'] for c in comparisons)
    omission_bound=float(np.max(omission));evidence=float(abs(evidence))
    return dict(passed=reward_difference+omission_bound<=1e-4 and evidence<=1e-4,
        comparisons=comparisons,maximum_log_evidence_error=evidence,
        maximum_mixed_reward_rule_difference=reward_difference,fallback_reward_omission_bound=omission_bound)


def prepare(plan):
    started = time.process_time(); spec = read(plan); out = ROOT/spec['output']
    kind, = spec['selected_kinds']; assert kind in ('raw','projected')
    assert spec['estimated_cost_admitted'] and spec['estimated_phase_cpu_seconds']<=spec['maximum_estimated_cpu_seconds']
    out.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGPROF, cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF, spec['maximum_cached_analysis_cpu_seconds'])
    source = read(spec['source_bound_plan']); bank = read(source['bank_plan'])
    context = compressed_read(ROOT/bank['output']/'context.json.gz')
    old = read(source['output']+'/summary.json'); assert old['numerical_containment_passed']
    diagnosis = read(spec['diagnosis_path'])
    assert read(diagnosis['source_plan'])['followup_bound_plan'] == spec['source_bound_plan']
    shared = compressed_read(ROOT/'paper_artifacts/historical_component_bank_v1/marginals/063.json.gz')
    assert context['query'] == 63 and shared['points'] == [63, 45, 10]
    assert shared['prior_weights'] == context['prior_weights']
    assert [len(shared['fields'][h]['means']) for h in (4,5)] == [512,1024]
    live = {r['record']:r for r in old['records'] if r['kind']==kind}
    probes = {r['record']:r for r in diagnosis['records'] if r['kind']==kind}
    risk = {r['record']:(int(o),r) for o,v in context['origins'].items() for r in v['risk']}
    group_ids = {a:g['group'] for g in action_groups(context['actions']) for a in g['action_indices']}
    prior = np.asarray(context['prior_weights']); selected = []; groups = []
    for name in spec['selected_records']:
        (out/'parts'/name).mkdir(parents=True,exist_ok=True)
        (out/'marginals').mkdir(parents=True,exist_ok=True)
        origin_id, observation = risk[name]; origin = context['origins'][str(origin_id)]
        indices = live[name]['candidate_actions']; actions = [context['actions'][a] for a in indices]
        assert indices == probes[name]['old']['candidate_actions'] and indices
        assert all(group_ids[a]==63 for a in indices)
        assert all(a['kind']=='guarded' and a['shortlist']==[63,45,10] and a['incumbent_position']==1 for a in actions)
        scores = np.log(prior)+np.asarray(origin['objective_scores'])+observation['head_log_scores']
        posterior = np.exp(scores-logsumexp(scores,axis=-1)[:,None])
        objective = np.asarray(origin['objective'][kind])
        pair_plan = spec['source_pair_plans'][name]; pair_spec = read(pair_plan)
        pair_source = read(pair_spec['source_bound_plan'])
        assert pair_source['bank_plan']==source['bank_plan']
        assert pair_spec.get('primary_policy',63)==63 and pair_spec.get('integration_backend','pair')=='pair'
        assert pair_spec['source_components']=={'4':512,'5':1024}
        unit = next(u for u in read(pair_spec['output']+'/preparation.json')['units'] if (u['record'],u['kind'])==(name,kind))
        source_units = {h:read(pair_spec['output']+f'/units/{h}/{name}/{kind}.json') for h in (4,5)}
        source_rules = {}
        for h, accepted in source_units.items():
            assert accepted['complete'] and accepted['budgets']==unit['budgets']
            assert (accepted['record'],accepted['kind'],accepted['identity'],accepted['pair'])==(name,kind,h,[63,45])
            rules, selected_rule = unit_rule_config(pair_spec,unit,h)
            if 'rules' in accepted: assert accepted['rules']==rules
            if 'selected_rule' in accepted: assert accepted['selected_rule']==selected_rule
            source_rules[h] = (rules,selected_rule)
        rules,selected_rule = source_rules[4]
        assert len(rules)==4 and selected_rule in ('E9','E10')
        assert source_rules[5]==(rules,selected_rule)
        source_precision = {str(h):u['precision_passed'] for h,u in source_units.items()}
        positions = [unit['budgets'].index(a['budgets'][:2]) for a in actions]
        budgets = np.asarray([a['budgets'] for a in actions])
        mixed = {rule[0]:np.zeros((16,len(indices))) for rule in rules}
        mixed_costs = {rule[0]:np.zeros((16,len(indices))) for rule in rules}
        evidence = 0.; omission = np.zeros((16,len(indices))); identity_cost_difference = 0.
        for h in (4,5):
            accepted = source_units[h]
            rows = {}; costs_by_rule = {}
            for rule, order, power, offset in rules:
                path = pair_spec['output']+f'/parts/{h}/{name}/{kind}_{rule}.json'; profile = read(path)
                expected = dict(record=name,kind=kind,identity=h,pair=[63,45],budgets=unit['budgets'],
                    rule=rule,order=order,power=power,seed=pair_spec['seed']+offset,integrated_mean_position=1)
                assert all(profile[k]==v for k,v in expected.items())
                d = profile['diagnostics']
                assert d['components']==pair_spec['source_components'][str(h)] and d['all_positive_components_retained']
                assert d['source_covariance_retained'] and d['verifier_sd_order']==16
                assert (d['order'],d['power'],d['integrated_mean_position'])==(order,power,1)
                states = np.asarray(profile['states'])[positions]
                if rule==selected_rule: np.testing.assert_array_equal(states,np.asarray(accepted['states'])[positions])
                statistics, costs = guard_statistics_from_pair_states(states,budgets,context['protocol'])
                mixed[rule] += posterior[:,h,None]*np.sum(objective[:,h][:,indices]*statistics,axis=-1)
                mixed_costs[rule] += posterior[:,h,None]*costs[None,:]; costs_by_rule[rule]=costs
                evidence = max(evidence,abs(d['predictive_log_evidence_error']))
                rows[rule] = dict(statistics=statistics.tolist(),costs=costs.tolist(),diagnostics=d,
                    source_pair_profile=path,source_pair_configuration=expected,fallback_statistic_zero=True,
                    source_rule_cpu_seconds=profile['cpu_seconds'],new_joint_integrations=0)
            for a, idx in enumerate(indices):
                detail = next(v for v in probes[name]['details'] if (v['identity'],v['action'])==(h,idx))
                assert detail['budgets']==actions[a]['budgets']
                fallback_cap = detail['certificate_caps'][2]
                omission[:,a] += posterior[:,h]*objective[:,h,idx,2]*fallback_cap
            identity_cost_difference = max(identity_cost_difference,
                max(float(np.max(abs(costs_by_rule[a]-costs_by_rule[b]))) for a,b in itertools.combinations(rows,2)))
            save(out/'parts'/name/f'{h}_{kind}.json',dict(complete=True,record=name,identity=h,kind=kind,
                action_indices=indices,rows=rows,reused_pair_rule_evaluations=len(rows),new_joint_integrations=0,
                source_pair_precision_passed=accepted['precision_passed'],source_pair_selected_rule=selected_rule))
        checks = reward_precision_checks(mixed,omission,evidence)
        for comparison in checks['comparisons']:
            comparison['maximum_cost_difference'] = float(np.max(abs(
                mixed_costs[comparison['low']]-mixed_costs[comparison['high']])))
        passed = checks['passed']
        selected.append(dict(record=name,origin=origin_id,group=63,action_indices=indices,
            label=observation['label'],eta_label=observation['eta_label'],source_pair_plan=pair_plan))
        groups.append(dict(record=name,kind=kind,group=63,candidate_passed=passed,
            selected_candidate=selected_rule,selected_rules_by_identity={str(h):config[1] for h,config in source_rules.items()},
            validation_rule=selected_rule,checks=checks,
            maximum_mixed_reward_rule_difference=checks['maximum_mixed_reward_rule_difference'],
            fallback_reward_omission_bound=checks['fallback_reward_omission_bound'],
            fallback_omission_by_head_action=omission.tolist(),
            maximum_mixed_cost_rule_difference=max(c['maximum_cost_difference'] for c in checks['comparisons']),
            maximum_identity_cost_rule_difference=identity_cost_difference,cost_precision_threshold_applied=False,
            source_pair_precision_passed=all(source_precision.values()),source_pair_precision_by_identity=source_precision,
            source_pair_plan=pair_plan))
        compressed_write(out/'marginals'/f'{name}.json.gz',dict(name=name,record=name,origin=origin_id,query=63,
            label=observation['label'],eta_label=observation['eta_label'],head_log_scores=observation['head_log_scores'],
            prior_weights=context['prior_weights'],replicates=4,utility_identities=[4,5],protocol=context['protocol'],
            source_bound_plan=spec['source_bound_plan'],source_pair_plan=pair_plan,diagnosis_path=spec['diagnosis_path']))
    prep = dict(complete=True,selected=selected,source_bound_plan=spec['source_bound_plan'],
        source_phase_completion=spec['source_phase_completion'],new_joint_integrations=0,new_source_fits=0,
        new_physical_calls=0,new_verifier_calls=0,all_previous_phase_costs_retained=True)
    save(out/'preparation.json',prep)
    all_passed = all(g['candidate_passed'] for g in groups)
    result = dict(**prep,groups=groups,candidate_precision_passed=all_passed,all_requested_precision_passed=all_passed,
        selected_candidates_by_record={g['record']:g['selected_candidate'] for g in groups},
        analysis_cpu_seconds=time.process_time()-started,
        scope=f'Reconstruct paid P/I controls for {sum(len(r["action_indices"]) for r in selected)} live {kind} guarded actions; fallback omission is bounded per matched head. Reward/evidence use the original 1e-4 gate independently of source state precision; cost differences are recorded without a new cost precision gate.')
    choices = {g['selected_candidate'] for g in groups}
    if len(choices)==1: result['selected_candidate']=next(iter(choices))
    accepted_groups = [g for g in groups if g['candidate_passed']]
    accepted_records = {g['record'] for g in accepted_groups}
    compact = dict(result,groups=accepted_groups,
        selected=[r for r in selected if r['record'] in accepted_records],
        precision_hold_groups=[g for g in groups if not g['candidate_passed']],
        candidate_precision_passed=bool(accepted_groups))
    save(out/'compact_summary.json',compact); save(out/'summary.json',result)
    signal.setitimer(signal.ITIMER_PROF,0.)
    print('SCHEDULEURM_DONE '+json.dumps(result),flush=True)
    return result


def collect(plan):
    spec=read(plan);references=read(spec['output']+'/compact_summary.json')
    assert references['candidate_precision_passed']
    pilot=read(read(spec['pilot_plan'])['output']+'/pair_summary.json');assert pilot['precision_passed']
    primary={('raw',pilot['identity'],pilot['record'],tuple(pilot['pair']),tuple(b)):u
        for b,u in zip(pilot['budgets'],pilot['upper'])}
    inherited=inherited_primary_rows(spec);merge_primary_caps(primary,inherited)
    anchor=read(spec['anchor_plan'])
    anchors={(r['kind'],h,r['record'],(45,10),tuple(b)):u for h in (4,5)
        for r in read(anchor['output']+f'/summary_{h}.json')['rows'] if r['precision_passed']
        for b,u in zip(r['budgets'],r['upper'])}
    screen.branch_quality_upper=coupled_branch_quality_upper
    screen.collect(spec['followup_bound_plan'],certificate_caps=load_caps(spec['query_cap_plan']),
        anchor_caps=anchors,primary_incumbent_caps=primary,
        record_kinds=[(r['record'],r['kind']) for r in references['groups']])
    new_plan=read(spec['followup_bound_plan']);old=read(read(spec['source_bound_plan'])['output']+'/compact_summary.json')
    new=read(new_plan['output']+'/compact_summary.json');comparison={}
    for kind in ('raw','projected'):
        a=old['weighted_bank_envelope']['kinds'][kind];b=new['weighted_bank_envelope']['kinds'][kind]
        assert b['best_reference_contrast']>=a['best_reference_contrast']-1e-14
        assert b['unresolved_upper_gap']<=a['unresolved_upper_gap']+1e-12
        comparison[kind]=dict(old_gap=a['unresolved_upper_gap'],new_gap=b['unresolved_upper_gap'],
            old_reference_contrast=a['best_reference_contrast'],new_reference_contrast=b['best_reference_contrast'],
            old_totals=old['totals'][kind],new_totals=new['totals'][kind])
    for filename in ('summary.json','compact_summary.json'):
        row=read(new_plan['output']+'/'+filename)
        row.update(cached_collection_charged_to_certificate_cap_phase_reserves=False,
            cached_collection_charged_to_cached_pair_reference_phase_reserves=True,cached_pair_reference_plan=plan)
        save(ROOT/new_plan['output']/filename,row)
    result=dict(complete=True,state='bound_refined' if new['numerical_containment_passed'] else 'containment_hold',
        source_bound_plan=spec['source_bound_plan'],source_phase_completion=spec['source_phase_completion'],
        followup_bound_plan=spec['followup_bound_plan'],comparison=comparison,
        numerical_containment_passed=new['numerical_containment_passed'],
        maximum_reference_above_new_bound=new['maximum_validated_reference_above_bound'],
        all_conditional_actions_closed=all(t['record_group_candidates']==0 for t in new['totals'].values()),
        original_zero_candidate_gate_unchanged=True,inherited_accepted_units=len(inherited),
        new_admitted_joint_action_references=sum(len(r['action_indices']) for r in references['selected']),
        cached_reference_records=len(references['groups']),candidate_precision_passed=True,
        reference_reward_hold_records=len(references.get('precision_hold_groups',[])),
        all_requested_precision_passed=references.get('all_requested_precision_passed',True),
        recomputed_record_kinds=new['recomputed_record_kinds'],reused_record_kinds=new['reused_record_kinds'],
        maximum_mixed_reward_rule_difference=max(r['maximum_mixed_reward_rule_difference'] for r in references['groups']),
        maximum_fallback_reward_omission=max(r['fallback_reward_omission_bound'] for r in references['groups']),
        maximum_mixed_cost_rule_difference=max(r['maximum_mixed_cost_rule_difference'] for r in references['groups']),
        new_joint_integrations=0,new_source_fits=0,new_physical_calls=0,new_verifier_calls=0,
        independent_confirmation=False)
    save(ROOT/spec['output']/'summary.json',result)
    print('SCHEDULEURM_DONE '+json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','collect']);parser.add_argument('--plan',required=True)
    args=parser.parse_args()
    try: (prepare if args.stage=='prepare' else collect)(args.plan)
    finally: print('SCHEDULEURM_RESOURCE '+json.dumps(dict(process_cpu_seconds=time.process_time())),flush=True)
