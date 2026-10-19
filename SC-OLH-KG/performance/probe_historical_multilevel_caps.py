"""Test a multi-level SD16 cap on cached, already verified guarded actions."""
import argparse
import copy
import json
import resource
import signal
import time

import numpy as np
from scipy.special import logsumexp

from core.branch_coupled_certificate_bound import coupled_branch_quality_upper
from core.marginal_certificate_bound import (branch_quality_upper, certificate_upper,
    multilevel_certificate_upper, raw_ratio_cdf_upper)
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.diagnose_historical_component_guarded import compressed_read
from performance.refine_historical_observed_prediction import field_from_snapshot
from performance.screen_historical_observed_batch_risk import condition_risk
from performance.validate_historical_component_bank import selected_reference_rule


def run(plan):
    start=time.process_time();spec=read(plan);admission=spec['analysis_reserve_admission']
    assert admission['cost_admitted'] and admission['total_reserved_support_cpu_seconds']<=600
    (ROOT/spec['output']).mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGPROF,cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF,spec['maximum_worker_cpu_seconds'])
    source=read(spec['source_bound_plan']);base=ROOT/read(source['bank_plan'])['output']
    context=compressed_read(base/'context.json.gz');prior=np.asarray(context['prior_weights'])
    risks={r['record']:(int(o),r) for o,data in context['origins'].items() for r in data['risk']}
    parts={(h,r['record'],r['policy']):r for h in (4,5)
        for shard in range(len(source['policy_shards']))
        for r in read(str(base/'tasks'/f'{h}_{shard}.json'))['rows']}
    thresholds=np.asarray(spec['thresholds']);rows=[];marginal_rows=[]
    maximum_evidence_error=0.;maximum_cdf_difference=0.;maximum_cap_difference=0.
    maximum_score_error=0.;maximum_primary_or_incumbent_above_cap=0.
    for case in spec['cases']:
        name=case['record'];ref=read(case['reference_plan']);origin,r=risks[name]
        check=next(c for c in read(ref['output']+'/compact_summary.json')['groups']
            if (c['record'],c['kind'],c['group'])==(name,'raw',case['group']))
        assert check['candidate_passed']
        selected=next(c for c in read(ref['output']+'/preparation.json')['selected'] if c['record']==name)
        assert selected['action_indices']==case['action_indices']
        bundle=compressed_read(ROOT/ref['output']/'marginals'/(name+'.json.gz'))
        assert bundle['prior_weights']==prior.tolist() and bundle['replicates']==4
        assert all(bundle[k]==r[k] for k in ('label','eta_label','head_log_scores'))
        ix=case['action_indices'];actions=[context['actions'][a] for a in ix]
        assert all(a['shortlist']==actions[0]['shortlist'] for a in actions)
        data=context['origins'][str(origin)]
        scores=np.log(prior)+np.asarray(data['objective_scores'])+r['head_log_scores']
        posterior=np.exp(scores-logsumexp(scores,axis=-1)[:,None])
        values={k:np.zeros((16,len(ix))) for k in ('old','coupled','multilevel','reference','validation')}
        for h in (4,5):
            before=field_from_snapshot(bundle['fields'][h]);future=copy.deepcopy(before)
            assert len(before.means)==spec['source_components'][str(h)]
            score=condition_risk(future,r,h,context)
            maximum_score_error=max(maximum_score_error,abs(score-r['head_log_scores'][h]))
            assert maximum_score_error<=1e-10
            cdfs={}
            for policy in actions[0]['shortlist']:
                position=bundle['points'].index(policy);cdfs[policy]={}
                for order in spec['orders']:
                    cdf,d=raw_ratio_cdf_upper(before,future,0,position,r['label'],score,thresholds,order,
                        replicates=4,eta_label=r['eta_label'])
                    assert d['all_positive_components_retained'] and d['components']==len(before.means)
                    cdfs[policy][order]=cdf
                    maximum_evidence_error=max(maximum_evidence_error,abs(d['log_evidence_error']))
                maximum_cdf_difference=max(maximum_cdf_difference,
                    float(np.max(np.abs(cdfs[policy][64]-cdfs[policy][128]))))
            saved=read(ref['output']+'/parts/'+name+f'/{h}_raw.json')['rows']
            statistics=np.asarray(saved[selected_reference_rule(check,h)]['statistics'])
            validation=np.asarray(saved[check.get('validation_rule','E9')]['statistics'])
            for j,(aidx,a) in enumerate(zip(ix,actions)):
                inc=a['incumbent_position'];assert inc in (1,2)
                quality=np.asarray(data['objective']['raw'])[:,h,aidx]
                old_caps=[];caps=[]
                for p,budget in zip(a['shortlist'],a['budgets']):
                    old=certificate_upper(parts[h,name,p]['raw']['128']['feasible_upper'],budget,3,context['protocol'])
                    levels={order:float(multilevel_certificate_upper(cdfs[p][order],thresholds,budget,3,context['protocol']))
                        for order in spec['orders']}
                    difference=abs(levels[64]-levels[128]);maximum_cap_difference=max(maximum_cap_difference,difference)
                    cap=min(old,levels[128]);old_caps.append(old);caps.append(cap)
                    point=(statistics[j,0]+statistics[j,3] if p==a['shortlist'][0] else
                        statistics[j,inc]+statistics[j,3] if p==a['shortlist'][inc] else None)
                    if point is not None:
                        maximum_primary_or_incumbent_above_cap=max(maximum_primary_or_incumbent_above_cap,float(point-cap))
                    marginal_rows.append(dict(record=name,identity=h,action=aidx,policy=p,budget=budget,
                        old_cap=old,multilevel_cap=cap,sd16_step_upper=levels[128],difference64_128=difference,
                        primary_or_incumbent_reference_point=point))
                weight=posterior[:,h]
                values['old'][:,j]+=weight*branch_quality_upper(quality,old_caps,inc)
                values['coupled'][:,j]+=weight*coupled_branch_quality_upper(quality,old_caps,inc)
                values['multilevel'][:,j]+=weight*coupled_branch_quality_upper(quality,caps,inc)
                values['reference'][:,j]+=weight*(quality@statistics[j])
                values['validation'][:,j]+=weight*(quality@validation[j])
            print('SCHEDULEURM_PROGRESS '+json.dumps(dict(record=name,identity=h,
                cpu_seconds=time.process_time()-start)),flush=True)
        for j,aidx in enumerate(ix):
            rows.append(dict(record=name,origin=origin,group=case['group'],action=aidx,cdf=r['cdf'],
                **{k+'_slack':float(np.mean(values[k][:,j]-values['reference'][:,j]))
                    for k in ('old','coupled','multilevel')},
                maximum_reference_above_multilevel=float(np.max(values['reference'][:,j]-values['multilevel'][:,j])),
                maximum_validation_above_multilevel=float(np.max(values['validation'][:,j]-values['multilevel'][:,j]))))
    maximum_above=max(0.,max(max(r['maximum_reference_above_multilevel'],r['maximum_validation_above_multilevel']) for r in rows))
    result=dict(complete=True,source_bound_plan=spec['source_bound_plan'],pilot_plan=plan,
        records=len(spec['cases']),action_comparisons=len(rows),rows=rows,
        means={k:float(np.mean([r[k] for r in rows])) for k in ('old_slack','coupled_slack','multilevel_slack')},
        maximum_cdf64_128_difference=maximum_cdf_difference,maximum_cap64_128_difference=maximum_cap_difference,
        maximum_log_evidence_error=maximum_evidence_error,maximum_score_error=maximum_score_error,
        maximum_reference_above_new_bound=maximum_above,
        maximum_primary_or_incumbent_above_cap=maximum_primary_or_incumbent_above_cap,
        numerical_precision_passed=max(maximum_evidence_error,maximum_cdf_difference,maximum_cap_difference)<=1e-4,
        numerical_containment_passed=max(maximum_above,maximum_primary_or_incumbent_above_cap)<=1e-4,
        thresholds=spec['thresholds'],source_components=spec['source_components'],all_positive_components_retained=True,
        original_zero_candidate_gate_unchanged=True,new_joint_reference_integrations=0,
        new_marginal_cdf_integrations=len(spec['cases'])*2*3*len(spec['orders']),
        new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,independent_confirmation=False,
        analysis_cpu_seconds=time.process_time()-start,analysis_reserve_admission=admission,
        scope='Four cached development observations, three guarded groups and both origins; local cap/slack diagnostic, not a global competition closure or independent deployment test.')
    save(ROOT/spec['output']/'cap_values.json',dict(rows=marginal_rows))
    save(ROOT/spec['output']/'summary.json',result)
    signal.setitimer(signal.ITIMER_PROF,0.)
    print('SCHEDULEURM_DONE '+json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args()
    try:run(a.plan)
    finally:print('SCHEDULEURM_RESOURCE '+json.dumps(dict(process_cpu_seconds=time.process_time(),
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)),flush=True)
