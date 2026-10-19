"""Reuse historical query=deploy integrals for the remaining group63 caps."""
import argparse
import copy
import json
import math
import resource
import signal
import time

import numpy as np

from core.direct_terminal_reference import fast_expected_certificates
from performance.complete_historical_source_heads import ROOT,read,save,cpu_timeout
from performance.diagnose_historical_component_guarded import compressed_read
from performance.diagnose_historical_predictive_tower import query_direct_value
from performance.refine_historical_observed_prediction import field_from_snapshot
from performance.screen_historical_observed_batch_risk import condition_risk
from core.branch_coupled_certificate_bound import coupled_branch_quality_upper
from performance import screen_historical_observed_batch_risk as screen


def query_caps(before,future,row,score,budgets,protocol,padding):
    """Budget/shortlist-specific certificates; retain the original SD16 rule."""
    result=[];maximum_difference=0.;maximum_evidence=0.;projected_error=0.
    for size in sorted({size for kind,budget,size in budgets}):
        matched=dict(protocol,familywise_delta=protocol['familywise_delta']/size)
        projected=[budget for kind,budget,n in budgets if kind=='projected' and n==size]
        if projected:
            values,d=fast_expected_certificates(future,[0]*len(projected),projected,matched)
            projected_error=max(projected_error,d['maximum_accepted_nested_difference'],d['adaptive_absolute_error_bound'])
            result.extend(dict(kind='projected',budget=budget,shortlist_size=size,point=float(v),
                upper=min(1.,float(v)+padding),difference64_128=0.) for budget,v in zip(projected,values))
        for kind,budget,n in budgets:
            if kind!='raw' or n!=size:continue
            values=[query_direct_value(before,future,row['label'],score,1,budget,matched,4,row['eta_label'],order)
                for order in (64,128)]
            difference=abs(values[0][0]-values[1][0]);maximum_difference=max(maximum_difference,difference)
            maximum_evidence=max(maximum_evidence,*(v[1] for v in values))
            result.append(dict(kind='raw',budget=budget,shortlist_size=size,point=values[1][0],
                upper=min(1.,max(v[0] for v in values)+padding),difference64_128=difference))
    return result,dict(maximum_raw64_128_difference=maximum_difference,
        maximum_raw_log_evidence_error=maximum_evidence,maximum_projected_error=projected_error,
        precision_passed=max(maximum_difference,maximum_evidence,projected_error)<=1e-4)


def prepare(plan):
    spec=read(plan);out=ROOT/spec['output'];out.mkdir(parents=True,exist_ok=True)
    source=read(spec['source_bound_plan']);old=read(source['output']+'/summary.json')
    assert old['numerical_containment_passed']
    context=compressed_read(ROOT/read(source['bank_plan'])['output']/'context.json.gz');query=context['query']
    requests={}
    for r in old['records']:
        for idx in r['candidate_actions']:
            a=context['actions'][idx]
            if query in a['shortlist']:
                j=a['shortlist'].index(query)
                requests.setdefault(r['record'],set()).add((r['kind'],a['budgets'][j],len(a['shortlist'])))
    units=[dict(record=name,budgets=[list(b) for b in sorted(v)]) for name,v in sorted(requests.items())]
    result=dict(complete=True,query=query,requests=units,observations=len(units),
        certificate_budget_kind_pairs=sum(len(u['budgets']) for u in units),source_bound_plan=spec['source_bound_plan'],
        scope='Only original candidate actions containing query63; exact budget and shortlist delta retained.')
    save(out/'preparation.json',result);print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in result.items() if k!='requests'}),flush=True)


def run(plan,pilot=False,identity=None):
    started=time.process_time();spec=read(plan);out=ROOT/spec['output'];prep=read(spec['output']+'/preparation.json')
    admission=spec['pilot_cost_admission'] if pilot else read(spec['output']+'/extension_cost_plan.json')
    assert admission['cost_admitted'] and admission['aggregate_estimated_phase_cpu_seconds']<=3600
    job=admission if pilot else next(j for j in admission['jobs'] if j['identity']==identity)
    signal.signal(signal.SIGPROF,cpu_timeout);signal.setitimer(signal.ITIMER_PROF,job['maximum_worker_cpu_seconds'])
    source=read(spec['source_bound_plan']);context=compressed_read(ROOT/read(source['bank_plan'])['output']/'context.json.gz')
    shared=compressed_read(ROOT/'paper_artifacts/historical_component_bank_v1/marginals/136.json.gz')
    assert shared['points']==[prep['query']] and shared['prior_weights']==context['prior_weights']
    risks={r['record']:r for origin in context['origins'].values() for r in origin['risk']}
    units=[u for u in prep['requests'] if u['record'] in spec['pilot_records']] if pilot else prep['requests']
    previous=read(spec['output']+'/pilot.json') if not pilot else None
    cached={(r['record'],r['identity']):[] for r in previous['rows']} if previous else {}
    if previous:
        assert previous['precision_passed'] and previous['source_bound_plan']==spec['source_bound_plan']
        assert previous['numerical_padding']==spec['numerical_padding']
        for r in previous['rows']:cached[r['record'],r['identity']].append(r)
    rows=[];diagnostics=[];timers=[];reused=0
    for h in ((4,5) if pilot else (identity,)):
        before=field_from_snapshot(shared['fields'][h]);assert len(before.means)==spec['source_components'][str(h)]
        for unit in units:
            name=unit['record']
            if (name,h) in cached:
                rows.extend(cached[name,h]);reused+=len(cached[name,h]);continue
            tick=time.process_time();r=risks[name];future=copy.deepcopy(before);score=condition_risk(future,r,h,context)
            assert abs(score-r['head_log_scores'][h])<=1e-10
            caps,d=query_caps(before,future,r,score,unit['budgets'],context['protocol'],spec['numerical_padding'])
            rows.extend(dict(c,record=name,identity=h,policy=prep['query']) for c in caps);diagnostics.append(d)
            timers.append(dict(record=name,identity=h,cpu_seconds=time.process_time()-tick))
            print('SCHEDULEURM_PROGRESS '+json.dumps(timers[-1]),flush=True)
    result=dict(complete=True,pilot=pilot,rows=rows,observations=len(units),timers=timers,reused_pilot_caps=reused,
        source_bound_plan=spec['source_bound_plan'],source_components=spec['source_components'],
        all_positive_components_retained=True,numerical_padding=spec['numerical_padding'],
        precision_passed=all(d['precision_passed'] for d in diagnostics),
        **{k:max([d[k] for d in diagnostics]+([previous[k]] if previous else [])) for k in
            ('maximum_raw64_128_difference','maximum_raw_log_evidence_error','maximum_projected_error')},
        cpu_seconds=time.process_time()-started,new_joint_reference_integrations=0,
        new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,independent_confirmation=False)
    save(out/('pilot.json' if pilot else f'caps_{identity}.json'),result);signal.setitimer(signal.ITIMER_PROF,0.)
    print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in result.items() if k not in ('rows','timers')}),flush=True)


def load_caps(plan):
    spec=read(plan);parts=[read(spec['output']+f'/caps_{h}.json') for h in (4,5)]
    assert all(d['precision_passed'] for d in parts)
    data=dict(rows=[r for d in parts for r in d['rows']],precision_passed=True)
    old_spec=read(spec['previous_cap_plan']);prep=read(old_spec['output']+'/preparation.json')
    caps={}
    for j in prep['jobs']:
        for r in read(old_spec['output']+f'/parts/{j["identity"]}_{j["shard"]}.json')['rows']:
            caps[r['kind'],r['identity'],r['record'],r['policy'],r['budget'],r['shortlist_size']]=r['upper']
    for r in data['rows']:
        key=(r['kind'],r['identity'],r['record'],r['policy'],r['budget'],r['shortlist_size'])
        caps[key]=min(caps.get(key,1.),r['upper'])
    return caps


def collect(plan):
    spec=read(plan);caps=load_caps(plan)
    screen.branch_quality_upper=coupled_branch_quality_upper
    screen.collect(spec['followup_bound_plan'],certificate_caps=caps)
    new=read(read(spec['followup_bound_plan'])['output']+'/compact_summary.json')
    old=read(read(spec['source_bound_plan'])['output']+'/compact_summary.json');comparison={}
    for kind in ('raw','projected'):
        a=old['weighted_bank_envelope']['kinds'][kind];b=new['weighted_bank_envelope']['kinds'][kind]
        assert abs(a['best_reference_contrast']-b['best_reference_contrast'])<1e-14
        assert b['unresolved_upper_gap']<=a['unresolved_upper_gap']+1e-12
        comparison[kind]=dict(old_gap=a['unresolved_upper_gap'],new_gap=b['unresolved_upper_gap'],
            reduction=1-b['unresolved_upper_gap']/a['unresolved_upper_gap'],old_totals=old['totals'][kind],new_totals=new['totals'][kind])
    result=dict(complete=True,state='bound_refined' if new['numerical_containment_passed'] else 'containment_hold',
        comparison=comparison,numerical_containment_passed=new['numerical_containment_passed'],
        maximum_reference_above_new_bound=new['maximum_validated_reference_above_bound'],
        all_conditional_actions_closed=all(v['record_group_candidates']==0 for v in new['totals'].values()),
        precision_passed=True,original_zero_candidate_gate_unchanged=True,
        new_joint_reference_integrations=0,new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,
        independent_confirmation=False,source_bound_plan=spec['source_bound_plan'])
    save(ROOT/spec['output']/'summary.json',result);print('SCHEDULEURM_DONE '+json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','pilot','run','collect']);p.add_argument('--plan',required=True)
    p.add_argument('--identity',type=int,choices=[4,5]);a=p.parse_args()
    try:
        if a.stage=='prepare':prepare(a.plan)
        elif a.stage in ('pilot','run'):run(a.plan,a.stage=='pilot',a.identity)
        else:collect(a.plan)
    finally:print('SCHEDULEURM_RESOURCE '+json.dumps(dict(process_cpu_seconds=time.process_time(),
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)),flush=True)
