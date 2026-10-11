"""Screen saved n4 heads with the existing marginal certificate upper bound."""
import argparse
import copy
import json
import resource
import signal
import time

import numpy as np
from scipy.special import logsumexp

from core.direct_terminal_reference import fast_expected_certificates
from core.marginal_certificate_bound import raw_feasibility_upper, projected_feasibility_upper, certificate_upper, branch_quality_upper
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.diagnose_historical_component_guarded import compressed_read, compressed_write
from performance.diagnose_historical_predictive_tower import query_direct_value
from performance.refine_historical_achievable_precision import assemble_case, original_heads
from performance.refine_historical_observed_prediction import field_from_snapshot
from performance.validate_historical_component_bank import action_groups
from representation.vectorized_joint_function_risk import update_raw_observation

PLAN = 'performance/manifests/historical_observed_batch_bound_v1.json'


def prepare(plan=PLAN):
    start=time.process_time();spec=read(plan);out=ROOT/spec['output'];out.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGPROF,cpu_timeout);signal.setitimer(signal.ITIMER_PROF,120.)
    pilot=read(spec['pilot_plan']);base=ROOT/pilot['output']
    assert read(str(base/'compact_summary.json'))['local_risk_candidate_passed']
    parent=read(pilot['parent_plan']);precision=read(parent['precision_plan'])
    case=next(c for c in precision['cases'] if c['name']=='source_n4')
    heads,bank,_=assemble_case(precision,case,read(precision['output']+'/registered_jobs.json')['jobs'])
    context=read('paper_artifacts/historical_component_bank_v1/context.json');origins={}
    assert bank['prior_weights']==context['prior_weights']
    for origin in bank['origins']:
        f,g=heads[0,origin],heads[1,origin];actions=original_heads(precision,case,origin)[0]['actions']
        assert len(f)==len(g)==16 and len(actions)==365 and all(sum(a['budgets'])<=356 for a in actions)
        assert all(r['replicates']==4 and r['eta_label'] is not None for r in g)
        origins[str(origin)]=dict(objective_scores=[r['head_log_scores'] for r in f],
            objective={k:[r[k][str(max(map(int,r[k])))]['statistics'] for r in f] for k in ('raw','projected')},
            risk=[dict(record=f'source_n4_o{origin}_h1_p{r["position"]}',
                **{k:r[k] for k in ('position','label','eta_label','head_log_scores')}) for r in g])
    compressed_write(out/'context.json.gz',dict(origins=origins,actions=actions,prior_weights=bank['prior_weights'],
        protocol=context['protocol'],query=63,replicates=4,remaining_calls=356))
    jobs=[];total=read(str(base/'preparation.json'))['estimated_cpu_seconds_including_reserves']
    for identity in (4,5):
        units=[]
        for origin in (4,5):
            cost=0.
            for kind in ('raw','projected'):
                suffix='_before_resolution' if origin==4 and kind=='raw' else ''
                p=read(str(base/'parts'/f'source_n4_o{origin}_h1_p13'/f'{identity}_{kind}{suffix}.json'))
                cost+=p['cpu_seconds']-sum(r['cpu_seconds'] for r in p['rows'].values())
            units.append(cost/3.)
        for shard,policies in enumerate(spec['policy_shards']):
            estimate=1.2*max(units)*32*len(policies)+2.
            total+=estimate;jobs.append(dict(identity=identity,shard=shard,policies=policies,
                maximum_measured_nonrule_cpu_per_policy=max(units),estimated_cpu_seconds=estimate))
    result=dict(complete=True,jobs=jobs,cumulative_estimated_cpu_seconds=total,
        estimated_cost_admitted=total<=spec['maximum_estimated_cpu_seconds'],cpu_seconds=time.process_time()-start,
        reused_phase_reserves=600,risk_observations=32,objective_heads_per_origin=16,actions=365,
        full_posterior_restorations=0,new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,
        independent_confirmation=False)
    save(out/'preparation.json',result);signal.setitimer(signal.ITIMER_PROF,0.)
    print('SCHEDULEURM_DONE '+json.dumps(result),flush=True)


def run(identity,shard,plan=PLAN):
    start=time.process_time();spec=read(plan);out=ROOT/spec['output'];(out/'tasks').mkdir(parents=True,exist_ok=True)
    assert read(spec['output']+'/preparation.json')['estimated_cost_admitted']
    signal.signal(signal.SIGPROF,cpu_timeout);signal.setitimer(signal.ITIMER_PROF,spec['maximum_worker_cpu_seconds'])
    context=compressed_read(out/'context.json.gz');base=ROOT/'paper_artifacts/historical_component_bank_v1/marginals'
    rows=[];maximum_score_error=0.
    for policy in spec['policy_shards'][shard]:
        shared=compressed_read(base/f'{73+policy:03d}.json.gz');field=field_from_snapshot(shared['fields'][identity])
        assert shared['prior_weights']==context['prior_weights'] and shared['points'][0]==context['query']
        position=shared['points'].index(policy)
        for origin in context['origins'].values():
            for r in origin['risk']:
                future=copy.deepcopy(field);score=update_raw_observation(future,0,r['label'],r['eta_label'],4)
                error=abs(score-r['head_log_scores'][identity]);assert error<=1e-10
                maximum_score_error=max(maximum_score_error,error);raw={}
                for order in (64,128):
                    value,d=raw_feasibility_upper(field,future,0,position,r['label'],score,
                        context['protocol']['chance_alpha'],order,replicates=4,eta_label=r['eta_label'])
                    raw[str(order)]=dict(feasible_upper=value,log_evidence_error=d['log_evidence_error'])
                row=dict(record=r['record'],policy=policy,raw=raw,
                    projected_feasible_upper=projected_feasibility_upper(future,position,context['protocol']['chance_alpha']))
                if policy==63:
                    fixed=[query_direct_value(field,future,r['label'],score,1,356,context['protocol'],4,r['eta_label'],order) for order in (64,128)]
                    native,diagnostics=fast_expected_certificates(future,[0],[356],context['protocol'])
                    row['fixed']=dict(raw=fixed[1][0],raw64_128_difference=abs(fixed[1][0]-fixed[0][0]),
                        raw_evidence_error=max(v[1] for v in fixed),projected=float(native[0]),
                        projected_diagnostics=diagnostics)
                rows.append(row)
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(identity=identity,shard=shard,policy=policy,cpu_seconds=time.process_time()-start)),flush=True)
    result=dict(complete=True,identity=identity,shard=shard,rows=rows,components=len(field.means),
        maximum_score_error=maximum_score_error,cpu_seconds=time.process_time()-start)
    save(out/'tasks'/f'{identity}_{shard}.json',result);signal.setitimer(signal.ITIMER_PROF,0.)
    print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


def native_direct_references(spec,context,out):
    base=ROOT/read(spec['native_reference_plan'])['output'] if 'native_reference_plan' in spec else out
    path=base/'native_direct_references.json'
    if path.exists():return read(str(path))
    start=time.process_time();rows=[]
    groups=action_groups(context['actions']);coordinates={r['record']:r for o in context['origins'].values() for r in o['risk']}
    for selected in read(spec['native_direct_cost_plan'])['native_direct_cases']:
        r=coordinates[selected['record']]
        for gid in selected['groups']:
            group=groups[gid];assert group['kind']=='direct'
            shared=compressed_read(ROOT/'paper_artifacts/historical_component_bank_v1/marginals'/f'{gid:03d}.json.gz')
            assert shared['prior_weights']==context['prior_weights'] and shared['points'][0]==63
            ix=group['action_indices'];actions=[context['actions'][i] for i in ix]
            position=shared['points'].index(actions[0]['primary'])
            for h in (4,5):
                field=field_from_snapshot(shared['fields'][h]);score=update_raw_observation(field,0,r['label'],r['eta_label'],4)
                assert abs(score-r['head_log_scores'][h])<=1e-10
                values,d=fast_expected_certificates(field,[position]*len(ix),[a['budgets'][0] for a in actions],context['protocol'])
                error=max(d['maximum_accepted_nested_difference'],d['adaptive_absolute_error_bound'])
                rows.extend(dict(identity=h,record=r['record'],action=a,projected=float(v),numerical_error=error) for a,v in zip(ix,values))
    result=dict(complete=True,rows=rows,cpu_seconds=time.process_time()-start,new_joint_terminal_integrations=0)
    save(path,result);return result


def collect(plan=PLAN):
    start=time.process_time();spec=read(plan);out=ROOT/spec['output'];out.mkdir(parents=True,exist_ok=True)
    base=ROOT/read(spec['bank_plan'])['output'] if 'bank_plan' in spec else out
    context=compressed_read(base/'context.json.gz')
    signal.signal(signal.SIGPROF,cpu_timeout);signal.setitimer(signal.ITIMER_PROF,240.)
    tasks=[read(str(base/'tasks'/f'{h}_{s}.json')) for h in (4,5) for s in range(len(spec['policy_shards']))]
    assert all(t['complete'] for t in tasks)
    parts={(t['identity'],r['record'],r['policy']):r for t in tasks for r in t['rows']}
    difference=max(abs(r['raw']['128']['feasible_upper']-r['raw']['64']['feasible_upper']) for r in parts.values())
    evidence=max(abs(v['log_evidence_error']) for r in parts.values() for v in r['raw'].values())
    fixed_error=max(max(r['fixed']['raw64_128_difference'],r['fixed']['raw_evidence_error'],
        r['fixed']['projected_diagnostics']['maximum_accepted_nested_difference'],
        r['fixed']['projected_diagnostics']['adaptive_absolute_error_bound']) for r in parts.values() if 'fixed' in r)
    query_path=base/'query_references.json'
    if query_path.exists():query_references=read(str(query_path))
    else:
        tick=time.process_time();query_rows=[]
        ix=[a for a,action in enumerate(context['actions']) if action['kind']=='direct' and action['primary']==63 and a!=355]
        shared=compressed_read(ROOT/'paper_artifacts/historical_component_bank_v1/marginals/136.json.gz')
        assert shared['points']==[63] and shared['prior_weights']==context['prior_weights']
        for h in (4,5):
            before=field_from_snapshot(shared['fields'][h])
            for origin in context['origins'].values():
                for r in origin['risk']:
                    future=copy.deepcopy(before);score=update_raw_observation(future,0,r['label'],r['eta_label'],4)
                    assert abs(score-r['head_log_scores'][h])<=1e-10
                    budgets=[context['actions'][a]['budgets'][0] for a in ix]
                    native,diagnostics=fast_expected_certificates(future,[0]*len(ix),budgets,context['protocol'])
                    for j,a in enumerate(ix):
                        values=[query_direct_value(before,future,r['label'],score,1,budgets[j],context['protocol'],4,r['eta_label'],order) for order in (64,128)]
                        query_rows.append(dict(identity=h,record=r['record'],action=a,raw=values[1][0],projected=float(native[j]),
                            numerical_error=max(abs(values[1][0]-values[0][0]),max(v[1] for v in values),
                                diagnostics['maximum_accepted_nested_difference'],diagnostics['adaptive_absolute_error_bound'])))
        query_references=dict(complete=True,rows=query_rows,cpu_seconds=time.process_time()-tick,
            charged_to_existing_cached_analysis_reserve=True)
        save(query_path,query_references)
    assert query_references['complete']
    query_parts={(r['identity'],r['record'],r['action']):r for r in query_references['rows']}
    fixed_error=max(fixed_error,max(r['numerical_error'] for r in query_references['rows']))
    joint={}
    for source in spec.get('joint_reference_plans',[spec['pilot_plan']]):
        pilot=read(source);preparation=read(pilot['output']+'/preparation.json')
        indices={r['record']:r['action_indices'] for r in preparation['selected']}
        for check in read(pilot['output']+'/compact_summary.json')['groups']:
            if check['candidate_passed']:
                joint.setdefault((check['record'],check['kind']),[]).append(dict(check=check,indices=indices[check['record']],output=pilot['output']))
    native=native_direct_references(spec,context,out) if 'native_direct_cost_plan' in spec else dict(rows=[],cpu_seconds=0.)
    native_parts={(r['identity'],r['record'],r['action']):r for r in native['rows']}
    if native['rows']:fixed_error=max(fixed_error,max(r['numerical_error'] for r in native['rows']))
    containment=0.;records=[];prior=np.asarray(context['prior_weights']);actions=context['actions'];groups=action_groups(actions)
    for origin in context['origins'].values():
        for r in origin['risk']:
            scores=np.log(prior)+np.asarray(origin['objective_scores'])+r['head_log_scores']
            posterior=np.exp(scores-logsumexp(scores,axis=-1)[:,None]);missing=posterior[:,[0,1,2,3,6,7]].sum(axis=-1)
            for kind in ('raw','projected'):
                objective=np.asarray(origin['objective'][kind]);fixed=np.zeros(16);upper=np.zeros((16,365))
                for h in (4,5):
                    fixed+=posterior[:,h]*objective[:,h,355,0]*parts[h,r['record'],63]['fixed'][kind]
                    for a,action in enumerate(actions):
                        caps=[]
                        for policy,budget in zip(action['shortlist'],action['budgets']):
                            part=parts[h,r['record'],policy]
                            feasible=part['raw']['128']['feasible_upper'] if kind=='raw' else part['projected_feasible_upper']
                            caps.append(certificate_upper(feasible,budget,len(action['shortlist']),context['protocol']))
                        quality=objective[:,h,a]
                        bounded=quality[:,0]*caps[0] if len(caps)==1 else branch_quality_upper(quality,np.asarray(caps),action['incumbent_position'])
                        upper[:,a]+=posterior[:,h]*bounded
                upper+=missing[:,None];containment=max(containment,float(np.max(fixed-upper[:,355])))
                references={355:fixed.copy()}
                for a in sorted({v['action'] for v in query_references['rows']}):
                    value=sum(posterior[:,h]*objective[:,h,a,0]*query_parts[h,r['record'],a][kind] for h in (4,5))
                    references[a]=value;containment=max(containment,float(np.max(value-upper[:,a])))
                    fixed=np.maximum(fixed,value)
                if kind=='projected':
                    for a in sorted({v['action'] for v in native['rows'] if v['record']==r['record']}):
                        value=sum(posterior[:,h]*objective[:,h,a,0]*native_parts[h,r['record'],a]['projected'] for h in (4,5))
                        references[a]=value;containment=max(containment,float(np.max(value-upper[:,a])))
                        fixed=np.maximum(fixed,value)
                for reference in joint.get((r['record'],kind),[]):
                    check=reference['check'];ix=reference['indices'];value=np.zeros((16,len(ix)))
                    for h in (4,5):
                        part=read(reference['output']+'/parts/'+r['record']+'/'+str(h)+'_'+kind+'.json')
                        stats=np.asarray(part['rows'][check['selected_candidate']]['statistics'])
                        value+=posterior[:,h,None]*np.sum(objective[:,h][:,ix]*stats,axis=-1)
                    for j,a in enumerate(ix):
                        references[a]=value[:,j]
                        containment=max(containment,float(np.max(value[:,j]-upper[:,a])))
                    fixed=np.maximum(fixed,value.max(axis=-1))
                overlap=upper+2e-4>=fixed[:,None];overlap[:,list(references)]=False
                action_overlap=overlap.any(axis=0);gap=np.max(upper-fixed[:,None],axis=0)
                records.append(dict(record=r['record'],kind=kind,overlapping_action_head_pairs=int(overlap.sum()),
                    validated_reference_actions=list(references),
                    candidate_actions=np.flatnonzero(action_overlap).tolist(),
                    candidate_upper_minus_reference={str(a):float(gap[a]) for a in np.flatnonzero(action_overlap)},
                    candidate_groups=[g['group'] for g in groups if action_overlap[g['action_indices']].any()],
                    maximum_upper_minus_fixed=float(gap.max()),
                    closest=[dict(action=int(a),upper_minus_fixed=float(gap[a])) for a in np.argsort(-gap)[:6] if a!=355]))
    result=dict(complete=True,numerical_containment_passed=max(difference,evidence,fixed_error,containment)<=1e-4,
        maximum_raw64_128_bound_difference=difference,maximum_raw_log_evidence_error=evidence,
        maximum_fixed_reference_error=fixed_error,maximum_validated_reference_above_bound=containment,
        risk_observations=32,objective_heads_per_origin=16,action_count=365,remaining_calls=356,
        records=records,totals={k:dict(compared_action_head_pairs=32*16*365,
            overlapping_action_head_pairs=sum(r['overlapping_action_head_pairs'] for r in records if r['kind']==k),
            record_action_candidates=sum(len(r['candidate_actions']) for r in records if r['kind']==k),
            record_group_candidates=sum(len(r['candidate_groups']) for r in records if r['kind']==k)) for k in ('raw','projected')},
        successful_worker_cpu_seconds=sum(t['cpu_seconds'] for t in tasks),collection_cpu_seconds=time.process_time()-start,
        query_reference_cpu_seconds=query_references['cpu_seconds'],
        native_direct_reference_cpu_seconds=0. if 'native_reference_plan' in spec else native['cpu_seconds'],
        native_direct_reference_reused='native_reference_plan' in spec,
        cumulative_estimated_cpu_seconds=read(str(base/'preparation.json'))['cumulative_estimated_cpu_seconds'],
        reused_marginal_bank=str(base.relative_to(ROOT)),new_marginal_workers=0 if base!=out else len(tasks),
        components={str(t['identity']):t['components'] for t in tasks},new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,
        full_predictive_integration_admitted=False,independent_confirmation=False,
        scope='All365 n4 actions screened against independently integrated query and admitted joint action references,conditional on the32 saved risk observations and16 cached objective heads perorigin. All8 identity scores and512/1024 components retained; no full batch outer convergence,query selection or independent gain.')
    if 'native_direct_cost_plan' in spec:
        result['joint_reference_phase_estimated_cpu_seconds']=read(spec.get('reference_phase_cost_plan',spec['native_direct_cost_plan']))['expected_cpu_seconds_including600reserves']
        result['cached_collection_charged_to_joint_reference_phase_reserves']=True
    save(out/'summary.json',result);compact={k:v for k,v in result.items() if k!='records'}
    compact['closest_records']=sorted(records,key=lambda r:-r['maximum_upper_minus_fixed'])[:8]
    save(out/'compact_summary.json',compact);signal.setitimer(signal.ITIMER_PROF,0.)
    print('SCHEDULEURM_DONE '+json.dumps(compact),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','collect']);p.add_argument('--plan',default=PLAN)
    p.add_argument('--identity',type=int,choices=(4,5));p.add_argument('--shard',type=int);a=p.parse_args()
    try:
        if a.stage=='prepare':prepare(a.plan)
        elif a.stage=='collect':collect(a.plan)
        else:run(a.identity,a.shard,a.plan)
    finally:print('SCHEDULEURM_RESOURCE '+json.dumps(dict(process_cpu_seconds=time.process_time(),peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)),flush=True)
