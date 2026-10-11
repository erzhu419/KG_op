"""Screen saved n4 heads with the existing marginal certificate upper bound."""
import argparse
import copy
import json
import resource
import signal
import time
from collections import Counter

import numpy as np
from scipy.special import logsumexp

from core.direct_terminal_reference import fast_expected_certificates
from core.marginal_certificate_bound import raw_feasibility_upper, projected_feasibility_upper, certificate_upper, branch_quality_upper
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.diagnose_historical_component_guarded import compressed_read, compressed_write
from performance.diagnose_historical_predictive_tower import query_direct_value
from performance.refine_historical_achievable_precision import assemble_case, original_heads
from performance.refine_historical_observed_prediction import field_from_snapshot
from performance.validate_historical_component_bank import RULES, action_groups
from representation.vectorized_joint_function_risk import update_raw_observation, apply_raw_moments

PLAN = 'performance/manifests/historical_observed_batch_bound_v1.json'


def condition_risk(field,row,identity,context):
    if 'risk_local_moments' not in context:
        return update_raw_observation(field,0,row['label'],row['eta_label'],4)
    source=context['risk_query_fields'][str(identity)]
    np.testing.assert_allclose(field.means[:,:2],source['means'],atol=1e-12,rtol=0.)
    np.testing.assert_allclose(field.covariances[:,:2,:2],source['covariances'],atol=1e-12,rtol=0.)
    np.testing.assert_allclose(field.log_weights,source['log_weights'],atol=1e-12,rtol=0.)
    saved=context['risk_local_moments'][row['record']][str(identity)]
    local={k:np.asarray(v) if k in ('mean','covariance','log_evidence') else v for k,v in saved.items()}
    return apply_raw_moments(field,0,local,replicates=4)


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
                future=copy.deepcopy(field);score=condition_risk(future,r,identity,context)
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


def native_direct_references(spec,context,out,shard=None):
    base=ROOT/read(spec['native_reference_plan'])['output'] if 'native_reference_plan' in spec else out
    path=base/('native_direct_references.json' if shard is None else f'native_parts/{shard}.json')
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():return read(str(path))
    if shard is None and 'native_direct_jobs' in spec:
        chunks=[read(str(base/f'native_parts/{j["shard"]}.json')) for j in spec['native_direct_jobs']]
        assert all(c['complete'] for c in chunks)
        rows=[r for c in chunks for r in c['rows']]
        expected={r['record'] for o in context['origins'].values() for r in o['risk']}
        assert {r['record'] for r in rows}==expected
        assert len(rows)==spec['native_direct_pricing']['rows']==len({(r['record'],r['identity'],r['action']) for r in rows})
        result=dict(complete=True,rows=rows,cpu_seconds=sum(c['cpu_seconds'] for c in chunks),new_joint_terminal_integrations=0)
        save(path,result);return result
    start=time.process_time();rows=[]
    groups=action_groups(context['actions']);coordinates={r['record']:r for o in context['origins'].values() for r in o['risk']}
    cases=read(spec['native_direct_cost_plan'])['native_direct_cases']
    if shard is not None:
        job=next(j for j in read(spec['output']+'/preparation.json')['jobs'] if j['kind']=='native-direct' and j['shard']==shard)
        cases=[r for r in cases if r['record'] in job['records']]
        assert len(cases)==job['observation_count']
    for selected in cases:
        r=coordinates[selected['record']]
        for gid in selected['groups']:
            group=groups[gid];assert group['kind']=='direct'
            shared=compressed_read(ROOT/'paper_artifacts/historical_component_bank_v1/marginals'/f'{gid:03d}.json.gz')
            assert shared['prior_weights']==context['prior_weights'] and shared['points'][0]==63
            ix=group['action_indices'];actions=[context['actions'][i] for i in ix]
            position=shared['points'].index(actions[0]['primary'])
            for h in (4,5):
                field=field_from_snapshot(shared['fields'][h]);score=condition_risk(field,r,h,context)
                assert abs(score-r['head_log_scores'][h])<=1e-10
                values,d=fast_expected_certificates(field,[position]*len(ix),[a['budgets'][0] for a in actions],context['protocol'])
                error=max(d['maximum_accepted_nested_difference'],d['adaptive_absolute_error_bound'])
                rows.extend(dict(identity=h,record=r['record'],action=a,projected=float(v),numerical_error=error) for a,v in zip(ix,values))
    result=dict(complete=True,rows=rows,cpu_seconds=time.process_time()-start,new_joint_terminal_integrations=0)
    save(path,result);return result


def native_direct_worker(plan,shard):
    spec=read(plan);out=ROOT/spec['output'];context=compressed_read(out/'context.json.gz')
    assert read(str(out/'preparation.json'))['estimated_cost_admitted']
    signal.signal(signal.SIGPROF,cpu_timeout);signal.setitimer(signal.ITIMER_PROF,240.)
    result=native_direct_references(spec,context,out,shard);signal.setitimer(signal.ITIMER_PROF,0.)
    print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
    return result


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
                    future=copy.deepcopy(before);score=condition_risk(future,r,h,context)
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
    for origin_id,origin in context['origins'].items():
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
                fixed_action=fixed.copy();references={355:fixed.copy()}
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
                candidates=np.flatnonzero(action_overlap)
                unknown=upper.copy();unknown[:,list(references)]=-np.inf
                unresolved=np.maximum(unknown.max(axis=-1)-fixed,0.)
                weight=float(prior[int(origin_id)]*r['outer_weight']) if 'outer_weight' in r else None
                records.append(dict(record=r['record'],kind=kind,overlapping_action_head_pairs=int(overlap.sum()),
                    validated_reference_actions=list(references),
                    candidate_actions=candidates.tolist(),
                    candidate_upper_minus_reference={str(a):float(gap[a]) for a in candidates},
                    candidate_groups=[g['group'] for g in groups if action_overlap[g['action_indices']].any()],
                    maximum_upper_minus_fixed=float(gap[candidates].max()) if len(candidates) else 0.,
                    outer_grid_weight=weight,mean_fixed_action_reward=float(fixed_action.mean()),
                    mean_best_validated_reward=float(fixed.mean()),mean_unresolved_upper_gap=float(unresolved.mean()),
                    weighted_reference_contrast=None if weight is None else float(weight*np.mean(fixed-fixed_action)),
                    weighted_unresolved_upper_gap=None if weight is None else float(weight*unresolved.mean()),
                    closest=[dict(action=int(a),upper_minus_fixed=float(gap[a])) for a in sorted(candidates,key=lambda a:-gap[a])[:6]]))
    observation_count=sum(len(o['risk']) for o in context['origins'].values())
    result=dict(complete=True,numerical_containment_passed=max(difference,evidence,fixed_error,containment)<=1e-4,
        maximum_raw64_128_bound_difference=difference,maximum_raw_log_evidence_error=evidence,
        maximum_fixed_reference_error=fixed_error,maximum_validated_reference_above_bound=containment,
        risk_observations=observation_count,objective_heads_per_origin=16,action_count=365,remaining_calls=356,
        records=records,totals={k:dict(compared_action_head_pairs=observation_count*16*365,
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
        scope=f'All365 n4 actions screened against independently integrated query and admitted joint action references,conditional on the{observation_count} registered risk observations and16 cached objective heads perorigin. All8 identity scores and512/1024 components retained; no full batch outer convergence,query selection or independent gain.')
    if all(r['outer_grid_weight'] is not None for r in records):
        sums={o:sum(r['outer_weight'] for r in v['risk']) for o,v in context['origins'].items()}
        result['weighted_bank_envelope']=dict(positive_weight_sums_per_origin=sums,
            normalized_per_origin=all(abs(w-1.)<=1e-14 for w in sums.values()),
            unrepresented_origin_prior_mass=float(max(0.,1.-sum(prior[int(o)] for o in sums))),
            kinds={k:dict(best_reference_contrast=sum(r['weighted_reference_contrast'] for r in records if r['kind']==k),
                unresolved_upper_gap=sum(r['weighted_unresolved_upper_gap'] for r in records if r['kind']==k)) for k in ('raw','projected')},
            scope='Envelope within the registered finite grid and admitted numerical rules. Projected values are not raw deployment rewards;outer quadrature error is separate.')
    if 'native_direct_cost_plan' in spec:
        result['joint_reference_phase_estimated_cpu_seconds']=read(spec.get('reference_phase_cost_plan',spec['native_direct_cost_plan']))['expected_cpu_seconds_including600reserves']
        result['cached_collection_charged_to_joint_reference_phase_reserves']=True
    save(out/'summary.json',result);compact={k:v for k,v in result.items() if k!='records'}
    compact['closest_records']=[dict(record=r['record'],kind=r['kind'],
        overlapping_action_head_pairs=r['overlapping_action_head_pairs'],
        candidate_action_count=len(r['candidate_actions']),candidate_group_count=len(r['candidate_groups']),
        maximum_upper_minus_fixed=r['maximum_upper_minus_fixed'],closest=r['closest'])
        for r in sorted(records,key=lambda r:-r['maximum_upper_minus_fixed'])[:8]]
    save(out/'compact_summary.json',compact);signal.setitimer(signal.ITIMER_PROF,0.)
    print('SCHEDULEURM_DONE '+json.dumps(compact),flush=True)


def next_plan(plan=PLAN):
    start=time.process_time();spec=read(plan);out=ROOT/spec['output'];summary=read(str(out/'summary.json'))
    assert summary['numerical_containment_passed']
    base=ROOT/read(spec['bank_plan'])['output'] if 'bank_plan' in spec else out
    groups=action_groups(compressed_read(base/'context.json.gz')['actions']);remaining=[];measured=[];fine_groups=set()
    for row in summary['records']:
        for gid in row['candidate_groups']:
            gaps=[row['candidate_upper_minus_reference'][str(a)] for a in groups[gid]['action_indices']
                if str(a) in row['candidate_upper_minus_reference']]
            weight=row.get('outer_grid_weight');priority=max(gaps)*(1. if weight is None else weight)
            remaining.append(dict(record=row['record'],kind=row['kind'],group=gid,
                maximum_unresolved_upper_gap=max(gaps),weighted_unresolved_upper_gap=priority))
    remaining.sort(key=lambda r:(-r['weighted_unresolved_upper_gap'],r['record'],r['kind'],r['group']))
    for source in spec.get('pricing_reference_plans',spec.get('joint_reference_plans',[spec['pilot_plan']])):
        prior=read(source);preparation=read(prior['output']+'/preparation.json')
        for check in read(prior['output']+'/compact_summary.json')['groups']:
            if check['candidate_passed'] and check['selected_candidate']=='E9':fine_groups.add((check['group'],check['kind']))
        for row in preparation['selected']:
            for h in (4,5):
                for kind in ('raw','projected'):
                    path=ROOT/prior['output']/'parts'/row['record']/f'{h}_{kind}.json'
                    if path.exists():measured.append(dict(group=row['group'],identity=h,kind=kind,rows=read(str(path))['rows']))
    selected=[];jobs=[];record_groups={}
    total=read(spec['continued_phase_cost_plan'])['estimated_cpu_seconds_including_reserves'] if 'continued_phase_cost_plan' in spec else 600.
    phase_base=total
    for row in remaining:
        gid,kind=row['group'],row['kind']
        if row['record'] in record_groups and record_groups[row['record']]!=gid:continue
        candidate='E9' if (gid,kind) in fine_groups else 'E8'
        rule_names=['E8','E9','F9','K9'] if candidate=='E9' else ['E8','F8','H8','E9']
        planned=dict(row,candidate=candidate,rules=rule_names);priced=[]
        for h in (4,5):
            historical=read(str(ROOT/'paper_artifacts/historical_component_bank_v1/parts'/f'{gid:03d}_{h}_{kind}.json'))['rows']
            same=[r for r in measured if r['identity']==h and r['kind']==kind and r['group']==gid]
            cohort=same or [r for r in measured if r['identity']==h and r['kind']==kind
                and groups[r['group']]['kind']==groups[gid]['kind']]
            units={}
            for rule in rule_names:
                order,power,_=RULES[rule]
                actual=[v['cpu_seconds'] for r in cohort for name,v in r['rows'].items()
                    if name in RULES and RULES[name][:2]==(order,power)]
                scaled=[v['cpu_seconds']*2**(power-RULES[name][1]) for name,v in historical.items()
                    if name in RULES and RULES[name][0]==order and RULES[name][1]<=power]
                units[rule]=dict(cpu=max(actual+scaled),basis=('Maximum measured n4 same group' if same else
                    'Maximum measured n4 same action kind' if cohort else 'Measured n1 same group scaled to required point count'),
                    compared_with='Actual same-group n1 rule timers scaled by continuous point ratio;20percent margin and2setup seconds perworker')
            priced.append(dict(**row,identity=h,candidate=candidate,rules=units,estimated_cpu_seconds=1.2*sum(v['cpu'] for v in units.values())+2.))
        estimate=sum(j['estimated_cpu_seconds'] for j in priced)
        if total+estimate>spec['maximum_estimated_cpu_seconds']:break
        selected.append(planned);jobs.extend(priced);record_groups[row['record']]=gid;total+=estimate
        if len(selected)==6:break
    result=dict(complete=True,source=spec['output']+'/summary.json',remaining_groups={k:dict(Counter(r['group'] for r in remaining if r['kind']==k)) for k in ('raw','projected')},
        next_joint_cases=selected,next_joint_jobs=jobs,expected_cpu_seconds_including600reserves=total,cost_admitted=bool(jobs) and total<=spec['maximum_estimated_cpu_seconds'],
        phase_base_estimated_cpu_seconds=phase_base,
        maximum_live_workers=8,priority='low',dispatched=False,cpu_seconds=time.process_time()-start,
        preparation_required='Use the registered n4 observation coordinates,complete365-action/objective-head tables and original joint marginals;one guarded or direct group per observation in this phase.',
        native_direct_cases=[],remaining_direct_cases=[r for r in remaining if groups[r['group']]['kind']=='direct'],
        remaining='Finite n4 conditional references,adaptive outer convergence,query/batch choice and independent deployment gain remain open.')
    save(out/'next_cost_plan.json',result);print('SCHEDULEURM_DONE '+json.dumps(result),flush=True);return result


def prepare_next(plan):
    from performance.check_historical_observed_batch_risk import prepare_frozen
    current=read(plan);cost_path=current['output']+'/next_cost_plan.json';priced=read(cost_path)
    assert priced['cost_admitted']
    previous=current['joint_reference_plans'][-1];risk=read(previous)
    number=int((ROOT/previous).stem.rsplit('_v',1)[1])+1
    risk_path=f'performance/manifests/historical_observed_batch_risk_competition_v{number}.json'
    bound_path=f'performance/manifests/historical_observed_batch_bound_competition_v{number}.json'
    risk.update(output=f'paper_artifacts/historical_observed_batch_risk_competition_v{number}',
        frozen_cost_plan=cost_path,followup_bound_plan=bound_path,
        rules=['E8','F8','H8','E9'],
        refined_cases=[dict(record=r['record'],kind=r['kind'],candidate=r.get('candidate','E8'),
            rules=r.get('rules',['E8','F8','H8','E9'])) for r in priced['next_joint_cases']],
        selection='Largest unresolved action upper gaps,one guarded or direct group per observation within this frozen phase.',
        purpose='Reuse original n4 observations and complete action/objective-head tables on the saved joint marginals;E8 compared against F8,H8,E9,or E9 against E8,F9,K9 when this group/kind already required512points.',
        scope='Original source prior,all8 identity scores,512/1024 positive components andcovariance,365 actions and356 remaining calls. Selected finite n4 risk observations;full adaptive outer integration,query/batch choice andindependent gain remain open.')
    bound=copy.deepcopy(current);bound['joint_reference_plans'].append(risk_path)
    bound.update(output=f'paper_artifacts/historical_observed_batch_bound_competition_v{number}',
        reference_phase_cost_plan=cost_path,
        scope='Reuse the completed n4 bound bank and admitted direct references;add passed frozen joint references. All365 actions and356remainingcalls;no new marginal workers or physical calls.',
        pricing='Assembly and next pricing consume this joint-reference phase600 CPU second reserves;reuse admitted query and native direct references.')
    save(ROOT/risk_path,risk);save(ROOT/bound_path,bound);prepared=prepare_frozen(risk_path)
    result=dict(complete=True,risk_plan=risk_path,risk_manifest=risk,bound_plan=bound_path,bound_manifest=bound,preparation=prepared)
    print('SCHEDULEURM_DONE '+json.dumps(result),flush=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','native-direct','collect','next-plan','prepare-next']);p.add_argument('--plan',default=PLAN)
    p.add_argument('--identity',type=int,choices=(4,5));p.add_argument('--shard',type=int);a=p.parse_args()
    try:
        if a.stage=='prepare':prepare(a.plan)
        elif a.stage=='collect':collect(a.plan)
        elif a.stage=='next-plan':next_plan(a.plan)
        elif a.stage=='prepare-next':prepare_next(a.plan)
        elif a.stage=='native-direct':native_direct_worker(a.plan,a.shard)
        else:run(a.identity,a.shard,a.plan)
    finally:print('SCHEDULEURM_RESOURCE '+json.dumps(dict(process_cpu_seconds=time.process_time(),peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)),flush=True)
