"""Refine remaining63/45 or71/45 certificates using historical pair rules."""
import argparse
import copy
import json
import resource
import signal
import time

import numpy as np

from core.branch_coupled_certificate_bound import coupled_branch_quality_upper
from core.conditional_raw_reference import component_risk_allocations
from core.shared_certificate_pair import certificate_pair_states
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.diagnose_historical_component_guarded import compressed_read
from performance.refine_historical_observed_prediction import field_from_snapshot
from performance.refine_historical_query_certificate_caps import load_caps
from performance.screen_historical_observed_batch_risk import condition_risk
from performance.validate_historical_component_bank import action_groups
from performance import screen_historical_observed_batch_risk as screen


def candidate_units(records, actions, selected_records=None, selected_kinds=None, primary_policy=63):
    selected = None if selected_records is None else set(selected_records)
    kinds = None if selected_kinds is None else set(selected_kinds)
    groups = {a:g['group'] for g in action_groups(actions) for a in g['action_indices']}
    units = []
    for row in records:
        if selected is not None and row['record'] not in selected:
            continue
        if kinds is not None and row['kind'] not in kinds:
            continue
        eligible = [a for a in row['candidate_actions']
            if actions[a]['shortlist'][:2] == [primary_policy,45]
            and actions[a]['incumbent_position'] == 1 and len(actions[a]['shortlist']) == 3]
        budgets = sorted({tuple(actions[a]['budgets'][:2]) for a in eligible})
        if not budgets:
            continue
        weight = row['outer_grid_weight']
        gap = max(row['candidate_group_expected_positive_gap'][str(groups[a])] for a in eligible)
        units.append(dict(record=row['record'],kind=row['kind'],pair=[primary_policy,45],budgets=[list(b) for b in budgets],
            priority_weight=weight,priority_gap=gap,weighted_priority_gap=weight*gap,
            weighted_unresolved_upper_gap=row['weighted_unresolved_upper_gap']))
    return sorted(units,key=lambda u:(-u['weighted_priority_gap'],u['record'],u['kind']))


def prepare(plan):
    spec = read(plan);out = ROOT/spec['output'];out.mkdir(parents=True,exist_ok=True)
    source = read(spec['source_bound_plan']);old = read(source['output']+'/summary.json')
    assert old['numerical_containment_passed']
    context = compressed_read(ROOT/read(source['bank_plan'])['output']/'context.json.gz')
    units = candidate_units(old['records'],context['actions'],spec.get('selected_records'),spec.get('selected_kinds'),
        spec.get('primary_policy',63))
    jobs = spec.get('jobs',[])
    for job in jobs:
        work = [u for u in units if u['record'] in job['records']] if 'records' in job else units[job['shard']::spec['shards']]
        work = [u for u in work if u['kind'] in job.get('kinds',('raw','projected'))]
        path = out/'worklists'/f'{job["shard"]}.json';path.parent.mkdir(parents=True,exist_ok=True)
        save(path,dict(requests=work))
        job['worklist'] = str(path.relative_to(ROOT))
    result = dict(complete=True,units=units,jobs=jobs,observations=len({u['record'] for u in units}),
        pair_observation_kinds=len(units),pair_budget_kind_rows=sum(len(u['budgets']) for u in units),
        source_bound_plan=spec['source_bound_plan'],source_phase_completion=spec['source_phase_completion'],
        original_phase_costs_retained=True,new_physical_calls=0,new_source_fits=0,new_verifier_calls=0)
    if jobs:
        result.update(estimated_cost_admitted=spec['estimated_cost_admitted'],
            estimated_phase_cpu_seconds=spec['estimated_phase_cpu_seconds'])
        assert result['estimated_cost_admitted'] and result['estimated_phase_cpu_seconds']<=spec['maximum_estimated_cpu_seconds']
    save(out/'preparation.json',result)
    print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in result.items() if k!='units'}),flush=True)
    return result


def profile_summary(unit, identity, profiles, padding, selected_rule='E9'):
    arrays = [np.asarray(p['states']) for p in profiles]
    assert all(a.shape == (len(unit['budgets']),4) for a in arrays)
    difference = max(float(np.max(abs(a-b))) for a in arrays for b in arrays)
    evidence = max(abs(p['diagnostics']['predictive_log_evidence_error']) for p in profiles)
    return dict(record=unit['record'],kind=unit['kind'],identity=identity,pair=unit.get('pair',[63,45]),budgets=unit['budgets'],
        states=next(p['states'] for p in profiles if p['rule']==selected_rule),selected_rule=selected_rule,
        upper=(np.maximum.reduce(arrays)+padding).tolist(),maximum_rule_difference=difference,
        maximum_log_evidence_error=evidence,precision_passed=max(difference,evidence)<=1e-4,
        integrated_mean_position=1,all_positive_components_retained=True,
        new_rule_evaluations=sum(p['source']=='computed' for p in profiles),
        reused_pilot_rule_evaluations=sum(p['source']=='pilot' for p in profiles),
        new_rule_cpu_seconds=sum(p['cpu_seconds'] for p in profiles if p['source']=='computed'),
        reused_pilot_cpu_seconds=sum(p['cpu_seconds'] for p in profiles if p['source']=='pilot'))


def pilot_profiles(spec, unit, identity):
    if spec.get('primary_policy',63)!=63 or unit.get('pair',[63,45])!=[63,45]:
        return None
    pilot = read(spec['pilot_plan'])
    if (unit['record'],unit['kind'],identity) != (pilot['record'],'raw',4):
        return None
    assert pilot['integrated_position']==1 and pilot['seed']==spec['seed'] and pilot['rules']==spec['rules']
    summary = read(pilot['output']+'/pair_summary.json')
    assert summary['precision_passed'] and summary['pair']==[63,45]
    indices = [summary['budgets'].index(b) for b in unit['budgets']]
    result = []
    for name,order,power,offset in spec['rules']:
        profile = read(pilot['output']+f'/parts/{name}.json')
        d = profile['diagnostics']
        assert (d['order'],d['power'],d['integrated_mean_position'])==(order,power,1)
        result.append(dict(profile,states=np.asarray(profile['states'])[indices].tolist(),source='pilot'))
    return result


def unit_rule_config(spec, unit, identity):
    for refinement in spec.get('unit_rule_refinements',[]):
        if (refinement['record'],refinement['kind'],refinement['identity'])==(unit['record'],unit['kind'],identity):
            return refinement['rules'],refinement['selected_rule']
    return spec['rules'],'E9'


def historical_guard_actions(context, unit, points):
    actions=[]
    for budget in unit['budgets']:
        action=next(a for a in context['actions'] if a['kind']=='guarded'
            and a['shortlist']==unit['pair']+[10] and a['incumbent_position']==1
            and a['budgets'][:2]==budget)
        actions.append(dict(action,primary=points.index(action['primary']),
            shortlist=[points.index(p) for p in action['shortlist']]))
    return actions


def pair_states_from_guard_statistics(statistics):
    s=np.asarray(statistics)
    return np.column_stack([1-s[:,0]-s[:,1]-s[:,3],s[:,1],s[:,0],s[:,3]])


def run(plan, identity, shard=None, record=None, kind=None):
    started = time.process_time();spec = read(plan);out = ROOT/spec['output']
    prep = read(spec['output']+'/preparation.json')
    if shard is None:
        assert record is not None and kind is not None
        units = [u for u in prep['units'] if (u['record'],u['kind'])==(record,kind)]
        assert len(units)==1
        admission=spec['pricing_admission']
        assert admission['cost_admitted'] and admission['aggregate_estimated_phase_cpu_seconds']<=spec['maximum_estimated_cpu_seconds']
        limit = spec.get('maximum_worker_cpu_seconds',150.)
    else:
        job = next(j for j in prep['jobs'] if (j['identity'],j['shard'])==(identity,shard))
        assert prep['estimated_cost_admitted'] and prep['estimated_phase_cpu_seconds']<=spec['maximum_estimated_cpu_seconds']
        units = read(job['worklist'])['requests'];limit = job['maximum_worker_cpu_seconds']
    signal.signal(signal.SIGPROF,cpu_timeout);signal.setitimer(signal.ITIMER_PROF,limit)
    source = read(spec['source_bound_plan'])
    context = compressed_read(ROOT/read(source['bank_plan'])['output']/'context.json.gz')
    risks = {r['record']:r for o in context['origins'].values() for r in o['risk']}
    primary_policy=spec.get('primary_policy',63);pair=[primary_policy,45]
    shared = compressed_read(ROOT/'paper_artifacts/historical_component_bank_v1/marginals'/f'{primary_policy:03d}.json.gz')
    assert shared['points']==list(dict.fromkeys([context['query'],primary_policy,45,10]))
    assert shared['prior_weights']==context['prior_weights']
    pair_positions=[shared['points'].index(p) for p in pair];query_position=shared['points'].index(context['query'])
    before = field_from_snapshot(shared['fields'][identity])
    assert len(before.means)==spec['source_components'][str(identity)]
    rows=[];new_calls=0;resumed=0
    for unit in units:
        unit=dict(unit,pair=pair)
        name=unit['record'];kind=unit['kind'];unit_started=time.process_time()
        backend=spec.get('integration_backend','pair')
        guarded_actions=historical_guard_actions(context,unit,shared['points']) if backend=='historical_guard' else None
        rules,selected_rule=unit_rule_config(spec,unit,identity)
        profiles=pilot_profiles(dict(spec,rules=rules),unit,identity)
        if profiles is None:
            r=risks[name];future=copy.deepcopy(before);score=condition_risk(future,r,identity,context)
            assert abs(score-r['head_log_scores'][identity])<=1e-10
            profiles=[]
            for rule,order,power,offset in rules:
                path=out/'parts'/str(identity)/name/f'{kind}_{rule}.json';path.parent.mkdir(parents=True,exist_ok=True)
                configuration=dict(record=name,kind=kind,identity=identity,pair=pair,budgets=unit['budgets'],
                    rule=rule,order=order,power=power,seed=spec['seed']+offset,integrated_mean_position=1)
                if backend=='historical_guard':
                    configuration.update(integration_backend=backend,guarded_budgets=[a['budgets'] for a in guarded_actions])
                if path.exists():
                    profile=read(str(path));assert all(profile[k]==v for k,v in configuration.items());resumed+=1
                else:
                    tick=time.process_time()
                    reference={}
                    if backend=='historical_guard':
                        _,rank=np.unique(before.means,axis=0,return_inverse=True)
                        statistics,costs,d=component_risk_allocations(before,future,guarded_actions,r['label'],score,
                            context['protocol'],query=query_position,raw=kind=='raw',order=order,power=power,
                            seed=spec['seed']+offset+rank*104729,replicates=4,eta_label=r['eta_label'],integrated_position=1)
                        states=pair_states_from_guard_statistics(statistics)
                        reference=dict(statistics=statistics.tolist(),costs=costs.tolist())
                    else:
                        states,d=certificate_pair_states(before,future,pair_positions,unit['budgets'],r['label'],score,
                            context['protocol'],query=query_position,raw=kind=='raw',order=order,power=power,
                            seed=spec['seed']+offset,replicates=4,eta_label=r['eta_label'],integrated_position=1)
                    assert d['all_positive_components_retained'] and d['components']==len(before.means)
                    profile=dict(configuration,states=states.tolist(),diagnostics=d,
                        cpu_seconds=time.process_time()-tick,source='computed',**reference);save(path,profile);new_calls+=1
                profiles.append(profile)
                print('SCHEDULEURM_PROGRESS '+json.dumps(dict(record=name,kind=kind,identity=identity,
                    rule=rule,cpu_seconds=profile['cpu_seconds'])),flush=True)
        row=profile_summary(unit,identity,profiles,spec['numerical_padding'],selected_rule)
        row.update(complete=True,rules=rules,integration_backend=backend,worker_cpu_seconds=time.process_time()-unit_started)
        # Earlier failed controls remain paid profiles of this same unit.
        saved_profiles=[read(str(p)) for p in (out/'parts'/str(identity)/name).glob(f'{kind}_*.json')]
        if saved_profiles:
            row['new_rule_evaluations']=sum(p['source']=='computed' for p in saved_profiles)
            row['new_rule_cpu_seconds']=sum(p['cpu_seconds'] for p in saved_profiles if p['source']=='computed')
        path=out/'units'/str(identity)/name/f'{kind}.json';path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            previous=read(str(path))
            if not previous['precision_passed']:
                hold=out/'precision_holds'/str(identity)/name/f'{kind}_{previous.get("selected_rule","E9")}.json'
                hold.parent.mkdir(parents=True,exist_ok=True)
                if not hold.exists():
                    save(hold,previous)
        save(path,row)
        rows.append(row)
    signal.setitimer(signal.ITIMER_PROF,0.)
    result=dict(complete=True,identity=identity,shard=shard,units=len(rows),
        precision_passed=all(r['precision_passed'] for r in rows),precision_hold_units=sum(not r['precision_passed'] for r in rows),
        new_rule_evaluations=new_calls,resumed_rule_profiles=resumed,
        reused_pilot_rule_evaluations=sum(r['reused_pilot_rule_evaluations'] for r in rows),
        cpu_seconds=time.process_time()-started)
    if shard is not None:
        path=out/'workers'/f'{identity}_{shard}.json';path.parent.mkdir(parents=True,exist_ok=True);save(path,result)
    print('SCHEDULEURM_DONE '+json.dumps(result),flush=True)
    return result


def inherited_primary_rows(spec):
    """Retain accepted rows from earlier batches of this same runner."""
    rows=[]
    source=read(spec['source_bound_plan'])
    while 'certificate_cap_plan' in source:
        previous=read(source['certificate_cap_plan'])
        path=ROOT/previous['output']/'preparation.json'
        if not path.exists():
            break
        prep=read(str(path))
        if 'units' not in prep:
            break
        for h in (4,5):
            for unit in prep['units']:
                row=read(previous['output']+f'/units/{h}/{unit["record"]}/{unit["kind"]}.json')
                assert row['complete']
                if row['precision_passed']:
                    rows.append(row)
        source=read(previous['source_bound_plan'])
    return rows


def merge_primary_caps(caps, rows):
    for row in rows:
        if row['precision_passed']:
            for budget,upper in zip(row['budgets'],row['upper']):
                key=(row['kind'],row['identity'],row['record'],tuple(row['pair']),tuple(budget))
                caps[key]=np.minimum(caps.get(key,upper),upper)


def collect(plan):
    started=time.process_time();spec=read(plan);prep=read(spec['output']+'/preparation.json')
    rows=[read(spec['output']+f'/units/{h}/{u["record"]}/{u["kind"]}.json') for h in (4,5) for u in prep['units']]
    assert all(r['complete'] for r in rows)
    pilot=read(read(spec['pilot_plan'])['output']+'/pair_summary.json');assert pilot['precision_passed']
    primary={(pilot['kind'],pilot['identity'],pilot['record'],tuple(pilot['pair']),tuple(b)):u for b,u in zip(pilot['budgets'],pilot['upper'])}
    inherited=inherited_primary_rows(spec)
    merge_primary_caps(primary,inherited+rows)
    anchor=read(spec['anchor_plan'])
    anchors={(r['kind'],r['identity'],r['record'],(45,10),tuple(b)):u for h in (4,5)
        for r in read(anchor['output']+f'/summary_{h}.json')['rows'] if r['precision_passed']
        for b,u in zip(r['budgets'],r['upper'])}
    caps=load_caps(spec['query_cap_plan']);screen.branch_quality_upper=coupled_branch_quality_upper
    screen.collect(spec['followup_bound_plan'],certificate_caps=caps,anchor_caps=anchors,primary_incumbent_caps=primary)
    new=read(read(spec['followup_bound_plan'])['output']+'/compact_summary.json')
    old=read(read(spec['source_bound_plan'])['output']+'/compact_summary.json');comparison={}
    for kind in ('raw','projected'):
        a=old['weighted_bank_envelope']['kinds'][kind];b=new['weighted_bank_envelope']['kinds'][kind]
        assert abs(a['best_reference_contrast']-b['best_reference_contrast'])<1e-14
        assert b['unresolved_upper_gap']<=a['unresolved_upper_gap']+1e-12
        comparison[kind]=dict(old_gap=a['unresolved_upper_gap'],new_gap=b['unresolved_upper_gap'],
            old_totals=old['totals'][kind],new_totals=new['totals'][kind])
    computed_profiles=[p for path in (ROOT/spec['output']/'parts').glob('*/*/*.json')
        if (p:=read(str(path)))['source']=='computed']
    backend=spec.get('integration_backend','pair')
    guard_evaluations=sum(p.get('integration_backend')=='historical_guard' for p in computed_profiles)
    result=dict(complete=True,state='bound_refined' if new['numerical_containment_passed'] else 'containment_hold',
        units=len(rows),precision_passed=all(r['precision_passed'] for r in rows),
        inherited_accepted_units=len(inherited),
        precision_hold_units=sum(not r['precision_passed'] for r in rows),
        accepted_units=sum(r['precision_passed'] for r in rows),
        maximum_rule_difference=max((r['maximum_rule_difference'] for r in rows),default=0.),
        maximum_log_evidence_error=max((r['maximum_log_evidence_error'] for r in rows),default=0.),
        comparison=comparison,numerical_containment_passed=new['numerical_containment_passed'],
        maximum_reference_above_new_bound=new['maximum_validated_reference_above_bound'],
        all_conditional_actions_closed=all(v['record_group_candidates']==0 for v in new['totals'].values()),
        original_zero_candidate_gate_unchanged=True,source_bound_plan=spec['source_bound_plan'],
        source_phase_completion=spec['source_phase_completion'],original_phase_costs_retained=True,
        collection_cpu_seconds=time.process_time()-started,new_physical_calls=0,new_source_fits=0,
        new_verifier_calls=0,independent_confirmation=False,integration_backend=backend,
        computed_guard_statistic_rule_evaluations=guard_evaluations,new_admitted_joint_action_references=0,
        integration_scope=('Full guarded statistics and costs computed with historical component rules; validated pair caps applied to the existing action-reference set.'
            if backend=='historical_guard' else 'Two-policy certificate states computed for validated pair caps.'),
        new_rule_evaluations=len(computed_profiles),new_rule_cpu_seconds=sum(p['cpu_seconds'] for p in computed_profiles),
        **{k:sum(r[k] for r in rows) for k in ('reused_pilot_rule_evaluations','reused_pilot_cpu_seconds')})
    save(ROOT/spec['output']/'summary.json',result)
    print('SCHEDULEURM_DONE '+json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','collect']);p.add_argument('--plan',required=True)
    p.add_argument('--identity',type=int,choices=[4,5]);p.add_argument('--shard',type=int)
    p.add_argument('--record');p.add_argument('--kind',choices=['raw','projected']);a=p.parse_args()
    try:
        if a.stage=='prepare':prepare(a.plan)
        elif a.stage=='run':run(a.plan,a.identity,a.shard,a.record,a.kind)
        else:collect(a.plan)
    finally:
        print('SCHEDULEURM_RESOURCE '+json.dumps(dict(process_cpu_seconds=time.process_time(),
            peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)),flush=True)
