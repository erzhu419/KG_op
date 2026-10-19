"""Refine certification caps only for the frozen bank's remaining candidates."""
import argparse
import copy
import json
import math
import resource
import signal
import time

import numpy as np

from core.branch_coupled_certificate_bound import coupled_branch_quality_upper
from core.marginal_certificate_bound import (multilevel_certificate_upper,
    projected_ratio_cdf_upper,raw_ratio_cdf_upper)
from performance.complete_historical_source_heads import ROOT,read,save,cpu_timeout
from performance.diagnose_historical_component_guarded import compressed_read
from performance.refine_historical_observed_prediction import field_from_snapshot
from performance.screen_historical_observed_batch_risk import condition_risk
from performance import screen_historical_observed_batch_risk as screen


def prepare(plan):
    spec=read(plan);out=ROOT/spec['output'];(out/'worklists').mkdir(parents=True,exist_ok=True)
    source=read(spec['source_bound_plan']);bank=ROOT/read(source['bank_plan'])['output']
    context=compressed_read(bank/'context.json.gz');selection=read(spec['selection_bound_plan'])
    candidates=read(selection['output']+'/summary.json');assert candidates['numerical_containment_passed']
    requests={}
    for r in candidates['records']:
        for idx in r['candidate_actions']:
            action=context['actions'][idx]
            for p,b in zip(action['shortlist'],action['budgets']):
                requests.setdefault((r['record'],p),set()).add((r['kind'],b,len(action['shortlist'])))
    units=[dict(record=name,policy=p,budgets=[list(k) for k in sorted(v)]) for (name,p),v in sorted(requests.items())]
    price=read(spec['pricing_plan']);assert len(units)==price['union_policy_observation_pairs']
    rate=price['maximum_observed_identity_policy_cpu_seconds'];jobs=[]
    for h in (4,5):
        for shard in range(4):
            work=units[shard::4];estimate=1.2*rate*len(work)+2
            jobs.append(dict(identity=h,shard=shard,requests=len(work),estimated_cpu_seconds=estimate,
                maximum_worker_cpu_seconds=math.ceil(estimate)))
            save(out/f'worklists/{shard}.json',dict(requests=work))
    estimate=sum(j['estimated_cpu_seconds'] for j in jobs)+600
    assert estimate<=spec['maximum_estimated_cpu_seconds']
    result=dict(complete=True,jobs=jobs,policy_observation_pairs=len(units),
        estimated_phase_cpu_seconds=estimate,shared_reserve_cpu_seconds=600,estimated_cost_admitted=True,
        source_bound_plan=spec['source_bound_plan'],source_phase_completion=spec['source_phase_completion'],
        original_phase_costs_retained=True,new_joint_reference_integrations=0,new_physical_calls=0,
        new_source_fits=0,new_verifier_calls=0,independent_confirmation=False)
    save(out/'preparation.json',result);print('SCHEDULEURM_DONE '+json.dumps(result),flush=True)
    return result


def run(plan,identity,shard):
    start=time.process_time();spec=read(plan);out=ROOT/spec['output'];(out/'parts').mkdir(parents=True,exist_ok=True)
    prep=read(spec['output']+'/preparation.json');assert prep['estimated_cost_admitted']
    job=next(j for j in prep['jobs'] if (j['identity'],j['shard'])==(identity,shard))
    signal.signal(signal.SIGPROF,cpu_timeout);signal.setitimer(signal.ITIMER_PROF,job['maximum_worker_cpu_seconds'])
    source=read(spec['source_bound_plan']);context=compressed_read(ROOT/read(source['bank_plan'])['output']/'context.json.gz')
    risks={r['record']:r for o in context['origins'].values() for r in o['risk']}
    cached={(r['record'],r['policy'],r['budget'],3):r for r in read(spec['pilot_output']+'/cap_values.json')['rows'] if r['identity']==identity}
    work=read(spec['output']+f'/worklists/{shard}.json')['requests'];rows=[];fields={}
    max_cdf=0.;max_cap=0.;max_evidence=0.;reused=0;raw_calls=0;projected_calls=0
    thresholds=np.asarray(spec['thresholds'])
    for number,unit in enumerate(work):
        p=unit['policy'];name=unit['record'];r=risks[name]
        if p not in fields:
            shared=compressed_read(ROOT/'paper_artifacts/historical_component_bank_v1/marginals'/f'{73+p:03d}.json.gz')
            assert shared['prior_weights']==context['prior_weights'] and shared['points'][0]==context['query']
            fields[p]=(field_from_snapshot(shared['fields'][identity]),shared['points'].index(p))
        before,position=fields[p];assert len(before.means)==spec['source_components'][str(identity)]
        future=copy.deepcopy(before);score=condition_risk(future,r,identity,context)
        assert abs(score-r['head_log_scores'][identity])<=1e-10
        needed=[tuple(b) for b in unit['budgets']];cdf={}
        if any(k=='raw' and (name,p,b,size) not in cached for k,b,size in needed):
            for order in (64,128):
                value,d=raw_ratio_cdf_upper(before,future,0,position,r['label'],score,thresholds,order,
                    replicates=4,eta_label=r['eta_label'])
                assert d['components']==len(before.means) and d['all_positive_components_retained']
                cdf['raw',order]=value;max_evidence=max(max_evidence,abs(d['log_evidence_error']))
            max_cdf=max(max_cdf,float(np.max(np.abs(cdf['raw',64]-cdf['raw',128]))));raw_calls+=2
        if any(k=='projected' for k,b,size in needed):
            cdf['projected',128]=projected_ratio_cdf_upper(future,position,thresholds);projected_calls+=1
        for kind,budget,size in needed:
            cache=cached.get((name,p,budget,size)) if kind=='raw' else None
            if cache is not None:
                value=cache['sd16_step_upper'];difference=cache['difference64_128'];reused+=1
            else:
                value=float(multilevel_certificate_upper(cdf[kind,128],thresholds,budget,size,context['protocol']))
                difference=(abs(value-float(multilevel_certificate_upper(cdf[kind,64],thresholds,budget,size,context['protocol']))) if kind=='raw' else 0.)
            max_cap=max(max_cap,difference)
            rows.append(dict(kind=kind,identity=identity,record=name,policy=p,budget=budget,
                shortlist_size=size,upper=value,difference64_128=difference))
        if number%20==0:print('SCHEDULEURM_PROGRESS '+json.dumps(dict(identity=identity,shard=shard,
            completed_requests=number+1,cpu_seconds=time.process_time()-start)),flush=True)
    result=dict(complete=True,identity=identity,shard=shard,requests=len(work),rows=rows,
        maximum_cdf64_128_difference=max_cdf,maximum_cap64_128_difference=max_cap,
        maximum_log_evidence_error=max_evidence,precision_passed=max(max_cdf,max_cap,max_evidence)<=1e-4,
        reused_pilot_caps=reused,new_raw_cdf_integrations=raw_calls,new_projected_cdf_integrations=projected_calls,
        all_positive_components_retained=True,components=len(before.means),cpu_seconds=time.process_time()-start)
    save(out/f'parts/{identity}_{shard}.json',result);signal.setitimer(signal.ITIMER_PROF,0.)
    print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
    return result


def collect(plan):
    spec=read(plan);prep=read(spec['output']+'/preparation.json')
    parts=[read(spec['output']+f'/parts/{j["identity"]}_{j["shard"]}.json') for j in prep['jobs']]
    assert all(p['complete'] and p['requests']==j['requests'] for p,j in zip(parts,prep['jobs']))
    summary=dict(complete=True,precision_passed=all(p['precision_passed'] for p in parts),
        maximum_cdf64_128_difference=max(p['maximum_cdf64_128_difference'] for p in parts),
        maximum_cap64_128_difference=max(p['maximum_cap64_128_difference'] for p in parts),
        maximum_log_evidence_error=max(p['maximum_log_evidence_error'] for p in parts),
        reused_pilot_caps=sum(p['reused_pilot_caps'] for p in parts),
        new_raw_cdf_integrations=sum(p['new_raw_cdf_integrations'] for p in parts),
        new_projected_cdf_integrations=sum(p['new_projected_cdf_integrations'] for p in parts),
        new_joint_reference_integrations=0,new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,
        all_positive_components_retained=True,independent_confirmation=False)
    if not summary['precision_passed']:
        summary['state']='precision_hold'
    else:
        caps={(r['kind'],r['identity'],r['record'],r['policy'],r['budget'],r['shortlist_size']):r['upper']
            for p in parts for r in p['rows']}
        screen.branch_quality_upper=coupled_branch_quality_upper
        screen.collect(spec['followup_bound_plan'],certificate_caps=caps)
        bound=read(spec['followup_bound_plan']);new=read(bound['output']+'/compact_summary.json')
        old=read(read(spec['selection_bound_plan'])['output']+'/compact_summary.json');comparison={}
        for kind in ('raw','projected'):
            before=old['weighted_bank_envelope']['kinds'][kind];after=new['weighted_bank_envelope']['kinds'][kind]
            assert abs(before['best_reference_contrast']-after['best_reference_contrast'])<1e-14
            assert after['unresolved_upper_gap']<=before['unresolved_upper_gap']+1e-12
            comparison[kind]=dict(old_gap=before['unresolved_upper_gap'],new_gap=after['unresolved_upper_gap'],
                reduction=1-after['unresolved_upper_gap']/before['unresolved_upper_gap'],
                old_totals=old['totals'][kind],new_totals=new['totals'][kind])
        summary.update(state='bound_refined' if new['numerical_containment_passed'] else 'containment_hold',
            numerical_containment_passed=new['numerical_containment_passed'],
            maximum_reference_above_new_bound=new['maximum_validated_reference_above_bound'],comparison=comparison,
            original_zero_candidate_gate_unchanged=True,
            all_conditional_actions_closed=all(v['record_group_candidates']==0 for v in new['totals'].values()))
    save(ROOT/spec['output']/'summary.json',summary)
    print('SCHEDULEURM_DONE '+json.dumps(summary),flush=True)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','collect']);p.add_argument('--plan',required=True)
    p.add_argument('--identity',type=int);p.add_argument('--shard',type=int);a=p.parse_args()
    try:
        if a.stage=='prepare':prepare(a.plan)
        elif a.stage=='run':run(a.plan,a.identity,a.shard)
        else:collect(a.plan)
    finally:print('SCHEDULEURM_RESOURCE '+json.dumps(dict(process_cpu_seconds=time.process_time(),
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)),flush=True)
