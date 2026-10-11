"""Reuse saved four-call heads and joint marginals for a bounded risk pilot."""
import argparse
import copy
import json
import resource
import signal
import time

import numpy as np

from core.conditional_raw_reference import component_risk_allocations
from core.marginal_certificate_bound import raw_feasibility_upper, projected_feasibility_upper, certificate_upper
from core.raw_terminal_head_grid import compose_head_grid
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.diagnose_historical_component_guarded import compressed_read, compressed_write, mixed_values
from performance.refine_historical_achievable_precision import assemble_case, original_heads
from performance.refine_historical_observed_prediction import field_from_snapshot
from performance.validate_historical_component_bank import RULES, action_groups, group_checks
from representation.vectorized_joint_function_risk import update_raw_observation

PLAN = 'performance/manifests/historical_observed_batch_risk_v1.json'


def prepare(plan=PLAN):
    start=time.process_time();spec=read(plan);out=ROOT/spec['output'];out.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGPROF,cpu_timeout);signal.setitimer(signal.ITIMER_PROF,120.)
    parent=read(spec['parent_plan']);precision=read(parent['precision_plan'])
    case=next(c for c in precision['cases'] if c['name']=='source_n4')
    heads,bank,_=assemble_case(precision,case,read(precision['output']+'/registered_jobs.json')['jobs'])
    base=ROOT/read(spec['marginal_plan'])['output'];context=read(str(base/'context.json'))
    assert bank['prior_weights']==context['prior_weights'] and bank['utility_identities']==[4,5]
    selected=[];jobs=[];total=600.
    for origin in bank['origins']:
        f,g=heads[0,origin],heads[1,origin];actions=original_heads(precision,case,origin)[0]['actions']
        assert len(actions)==365 and all(sum(a['budgets'])<=356 for a in actions)
        groups=action_groups(actions);guarded=[i for i,a in enumerate(actions) if a['kind']=='guarded']
        fs=np.asarray([r['raw'][str(max(map(int,r['raw'])))]['statistics'] for r in f])
        gs=np.asarray([r['raw'][str(max(map(int,r['raw'])))]['statistics'] for r in g])
        costs=np.asarray([r['raw'][str(max(map(int,r['raw'])))]['costs'] for r in g])
        values=compose_head_grid(fs,gs,costs,[r['head_log_scores'] for r in f],[r['head_log_scores'] for r in g],np.asarray(bank['prior_weights']))[0]
        gains=values[:,:,guarded]-values[:,:,spec['fixed_action'],None]
        fi,gi,ai=np.unravel_index(np.argmax(gains),gains.shape);row=g[gi];action_index=guarded[ai]
        group=next(v for v in groups if action_index in v['action_indices']);gid=group['group'];ix=group['action_indices']
        shared=compressed_read(base/'marginals'/f'{gid:03d}.json.gz')
        assert shared['prior_weights']==bank['prior_weights'] and row['replicates']==4 and row['eta_label'] is not None
        original=[actions[i] for i in ix];points=shared['points']
        assert all(a['shortlist']==context['actions'][i]['shortlist'] for a,i in zip(original,ix))
        name=f'source_n4_o{origin}_h1_p{row["position"]}'
        bundle=dict(name=name,origin=origin,query=parent['query'],label=row['label'],eta_label=row['eta_label'],
            replicates=4,head_log_scores=row['head_log_scores'],prior_weights=bank['prior_weights'],
            utility_identities=bank['utility_identities'],protocol=context['protocol'],seed=parent['seed'],
            group=gid,kind='guarded',points=points,fields=shared['fields'],action_indices=ix,
            actions=[dict(a,primary=points.index(a['primary']),shortlist=[points.index(p) for p in a['shortlist']]) for a in original],
            opposite=[dict(position=r['position'],head_log_scores=r['head_log_scores'],
                **{k:np.asarray(r[k][str(max(map(int,r[k])))]['statistics'])[:,ix].tolist() for k in ('raw','projected')}) for r in f],
            baseline={k:dict(statistics=np.asarray(row[k][str(max(map(int,row[k])))]['statistics'])[:,ix].tolist(),
                costs=np.asarray(row[k][str(max(map(int,row[k])))]['costs'])[:,ix].tolist()) for k in ('raw','projected')})
        compressed_write(out/'marginals'/(name+'.json.gz'),bundle)
        selected.append(dict(record=name,origin=origin,position=row['position'],group=gid,action_indices=ix,
            cached_guarded_gain_over_fixed=float(gains[fi,gi,ai]),opposite_position=f[fi]['position'],
            label=row['label'],eta_label=row['eta_label'],marginal_source=str((base/'marginals'/f'{gid:03d}.json.gz').relative_to(ROOT))))
        for identity in bank['utility_identities']:
            for kind in ('raw','projected'):
                old=read(str(base/'parts'/f'{gid:03d}_{identity}_{kind}.json'))['rows'];units={}
                for rule in spec['rules']:
                    order,power,_=RULES[rule]
                    times=[v['cpu_seconds'] for key,v in old.items() if key in RULES and
                        RULES[key][:2]==(order,power) and v['cpu_seconds']>0]
                    if not times:
                        compatible=[key for key,v in old.items() if key in RULES and
                            RULES[key][0]>=order and RULES[key][1]>=power and v['cpu_seconds']>0]
                        size=min(RULES[key][0]*2**RULES[key][1] for key in compatible)
                        times=[old[key]['cpu_seconds'] for key in compatible if RULES[key][0]*2**RULES[key][1]==size]
                    assert times,(gid,identity,kind,rule)
                    units[rule]=max(times)
                estimate=1.2*sum(units.values())+2.
                total+=estimate;jobs.append(dict(record=name,identity=identity,kind=kind,
                    estimated_cpu_seconds=estimate,matched_n1_rule_cpu_seconds=units))
    result=dict(complete=True,selected=selected,jobs=jobs,action_count=365,remaining_calls=356,
        estimated_cpu_seconds_including_reserves=total,estimated_cost_admitted=total<=spec['maximum_estimated_cpu_seconds'],
        cpu_seconds=time.process_time()-start,full_posterior_restorations=0,new_physical_calls=0,
        new_source_fits=0,new_verifier_calls=0,independent_confirmation=False)
    save(out/'preparation.json',result);signal.setitimer(signal.ITIMER_PROF,0.)
    print('SCHEDULEURM_DONE '+json.dumps(result),flush=True);return result


def prepare_frozen(plan):
    start=time.process_time();spec=read(plan);out=ROOT/spec['output'];out.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGPROF,cpu_timeout);signal.setitimer(signal.ITIMER_PROF,120.)
    frozen=read(spec['frozen_cost_plan']);assert frozen['cost_admitted']
    pilot=read(spec['pilot_plan']);parent=read(pilot['parent_plan']);precision=read(parent['precision_plan'])
    case=next(c for c in precision['cases'] if c['name']=='source_n4')
    heads,bank,_=assemble_case(precision,case,read(precision['output']+'/registered_jobs.json')['jobs'])
    selected=[];maximum_score_error=0.
    actions_by_origin={o:original_heads(precision,case,o)[0]['actions'] for o in bank['origins']}
    marginal_base=ROOT/read(spec['marginal_plan'])['output']
    for record in dict.fromkeys(r['record'] for r in frozen['next_joint_cases']):
        origin=int(record.split('_o')[1].split('_')[0]);position=int(record.rsplit('_p',1)[1])
        template=compressed_read(ROOT/pilot['output']/'marginals'/f'source_n4_o{origin}_h1_p13.json.gz')
        gids={r['group'] for r in frozen['next_joint_cases'] if r['record']==record};assert len(gids)==1
        gid=gids.pop();actions=actions_by_origin[origin];group=action_groups(actions)[gid]
        assert group['kind']=='guarded' and template['prior_weights']==bank['prior_weights']
        assert [r['head_log_scores'] for r in template['opposite']]==[r['head_log_scores'] for r in heads[0,origin]]
        row=next(r for r in heads[1,origin] if r['position']==position)
        assert row['replicates']==4 and row['eta_label'] is not None
        ix=group['action_indices'];shared=compressed_read(marginal_base/'marginals'/f'{gid:03d}.json.gz')
        assert shared['prior_weights']==bank['prior_weights'] and shared['points'][0]==template['query']
        points=shared['points'];bundle=copy.deepcopy(template)
        bundle.update(group=gid,points=points,fields=shared['fields'],action_indices=ix,
            actions=[dict(actions[i],primary=points.index(actions[i]['primary']),
                shortlist=[points.index(p) for p in actions[i]['shortlist']]) for i in ix],
            opposite=[dict(position=r['position'],head_log_scores=r['head_log_scores'],
                **{k:np.asarray(r[k][str(max(map(int,r[k])))]['statistics'])[:,ix].tolist() for k in ('raw','projected')}) for r in heads[0,origin]])
        bundle.update(name=record,label=row['label'],eta_label=row['eta_label'],head_log_scores=row['head_log_scores'],
            baseline={k:dict(statistics=np.asarray(row[k][str(max(map(int,row[k])))]['statistics'])[:,ix].tolist(),
                costs=np.asarray(row[k][str(max(map(int,row[k])))]['costs'])[:,ix].tolist()) for k in ('raw','projected')})
        assert all(sum(a['budgets'])<=356 for a in bundle['actions'])
        for h in (4,5):
            field=field_from_snapshot(bundle['fields'][h]);score=update_raw_observation(field,0,row['label'],row['eta_label'],4)
            maximum_score_error=max(maximum_score_error,abs(score-row['head_log_scores'][h]))
        assert maximum_score_error<=1e-10
        compressed_write(out/'marginals'/(record+'.json.gz'),bundle)
        selected.append(dict(record=record,origin=origin,position=position,group=gid,action_indices=ix,
            label=row['label'],eta_label=row['eta_label'],marginal_source=str((marginal_base/'marginals'/f'{gid:03d}.json.gz').relative_to(ROOT))))
    jobs=[dict(r,shard=next(i for i,s in enumerate(selected) if s['record']==r['record'])) for r in frozen['next_joint_jobs']]
    total=600.+sum(r['estimated_cpu_seconds'] for r in jobs)
    assert abs(total-frozen['expected_cpu_seconds_including600reserves'])<=1e-8
    result=dict(complete=True,selected=selected,jobs=jobs,action_count=365,remaining_calls=356,
        source_components={str(h):len(template['fields'][h]['means']) for h in (4,5)},
        maximum_head_score_error=maximum_score_error,
        estimated_cpu_seconds_including_reserves=total,estimated_cost_admitted=total<=spec['maximum_estimated_cpu_seconds'],
        cpu_seconds=time.process_time()-start,full_posterior_restorations=0,new_physical_calls=0,
        new_source_fits=0,new_verifier_calls=0,independent_confirmation=False)
    save(out/'preparation.json',result);signal.setitimer(signal.ITIMER_PROF,0.)
    print('SCHEDULEURM_DONE '+json.dumps(result),flush=True);return result


def run(record,identity,kind,plan=PLAN):
    start=time.process_time();spec=read(plan);out=ROOT/spec['output'];prep=read(spec['output']+'/preparation.json')
    assert prep['estimated_cost_admitted']
    signal.signal(signal.SIGPROF,cpu_timeout);signal.setitimer(signal.ITIMER_PROF,spec['maximum_worker_cpu_seconds'])
    b=compressed_read(out/'marginals'/(record+'.json.gz'));before=field_from_snapshot(b['fields'][identity]);after=copy.deepcopy(before)
    score=update_raw_observation(after,0,b['label'],b['eta_label'],4)
    score_error=abs(score-b['head_log_scores'][identity]);assert score_error<=1e-10
    _,rank=np.unique(before.means,axis=0,return_inverse=True);rows={}
    path=out/'parts'/record/f'{identity}_{kind}.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    previous=read(str(path)) if path.exists() else {}
    rows.update(previous.get('rows',{}))
    if dict(record=record,kind=kind) in spec.get('refined_cases',[]) and previous.get('complete') and not set(spec['rules']).issubset(rows):
        save(path.with_name(path.stem+'_before_resolution.json'),previous)
    for rule in spec['rules']:
        if rule in rows:continue
        order,power,offset=RULES[rule];tick=time.process_time()
        s,c,d=component_risk_allocations(before,after,b['actions'],b['label'],score,b['protocol'],query=0,
            raw=kind=='raw',order=order,power=power,seed=b['seed']+offset+rank*104729,
            integrated_position=b['actions'][0]['incumbent_position'],replicates=4,eta_label=b['eta_label'])
        rows[rule]=dict(statistics=s.tolist(),costs=c.tolist(),diagnostics=d,cpu_seconds=time.process_time()-tick)
        save(path,dict(complete=False,record=record,identity=identity,kind=kind,rows=rows))
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(record=record,identity=identity,kind=kind,rule=rule,cpu_seconds=time.process_time()-start)),flush=True)
    marginals=previous.get('marginal_bounds',[])
    for policy in ([] if marginals else b['actions'][0]['shortlist']):
        if kind=='raw':
            low,ld=raw_feasibility_upper(before,after,0,policy,b['label'],score,b['protocol']['chance_alpha'],64,
                replicates=4,eta_label=b['eta_label'])
            upper,d=raw_feasibility_upper(before,after,0,policy,b['label'],score,b['protocol']['chance_alpha'],128,
                replicates=4,eta_label=b['eta_label'])
        else:
            upper=low=projected_feasibility_upper(after,policy,b['protocol']['chance_alpha']);ld=d=dict(log_evidence_error=0.)
        marginals.append(dict(policy=policy,feasibility_upper=upper,raw64_128_difference=abs(upper-low),
            maximum_log_evidence_error=max(abs(ld['log_evidence_error']),abs(d['log_evidence_error']))))
    violation=0.
    candidate='E8' if dict(record=record,kind=kind) in spec.get('refined_cases',[]) else 'E7'
    for action,values in zip(b['actions'],rows[candidate]['statistics']):
        caps=np.asarray([certificate_upper(m['feasibility_upper'],n,len(action['shortlist']),b['protocol']) for m,n in zip(marginals,action['budgets'])])
        caps=np.r_[caps,min(caps[0],caps[action['incumbent_position']])]
        violation=max(violation,float(np.max(np.asarray(values)-caps)))
    result=dict(complete=True,record=record,identity=identity,kind=kind,rows=rows,head_score_error=score_error,
        marginal_bounds=marginals,maximum_branch_above_marginal_bound=violation,cpu_seconds=time.process_time()-start,
        cumulative_cpu_seconds=previous.get('cumulative_cpu_seconds',previous.get('cpu_seconds',0.))+time.process_time()-start,
        new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,independent_confirmation=False)
    save(path,result);signal.setitimer(signal.ITIMER_PROF,0.)
    print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in result.items() if k not in ['rows','marginal_bounds']}),flush=True)


def collect(plan=PLAN):
    start=time.process_time();spec=read(plan);out=ROOT/spec['output'];prep=read(spec['output']+'/preparation.json');groups=[];cpu=0.
    for row in prep['selected']:
        b=compressed_read(out/'marginals'/(row['record']+'.json.gz'))
        for kind in ('raw','projected'):
            parts=[read(str(out/'parts'/row['record']/f'{h}_{kind}.json')) for h in (4,5)]
            assert all(p['complete'] for p in parts);cpu+=sum(p.get('cumulative_cpu_seconds',p['cpu_seconds']) for p in parts)
            candidate='E8' if dict(record=row['record'],kind=kind) in spec.get('refined_cases',[]) else 'E7'
            pairs=[('E8','F8'),('E8','H8'),('E8','E9')] if candidate=='E8' else [('E7','E8'),('E7','F7'),('E7','G7')]
            check=group_checks(b,parts,kind,pairs,1e-4)
            stats=np.zeros_like(np.asarray(b['baseline'][kind]['statistics']));costs=np.zeros_like(np.asarray(b['baseline'][kind]['costs']))
            for h,p in zip((4,5),parts):stats[h]=p['rows'][candidate]['statistics'];costs[h]=p['rows'][candidate]['costs']
            baseline=mixed_values(b,np.asarray(b['baseline'][kind]['statistics']),np.asarray(b['baseline'][kind]['costs']),kind)
            current=mixed_values(b,stats,costs,kind)
            marginal_difference=max(m['raw64_128_difference'] for p in parts for m in p['marginal_bounds'])
            marginal_evidence=max(m['maximum_log_evidence_error'] for p in parts for m in p['marginal_bounds'])
            containment=max(p['maximum_branch_above_marginal_bound'] for p in parts)
            marginal_passed=max(marginal_difference,marginal_evidence,containment)<=1e-4
            groups.append(dict(record=row['record'],kind=kind,group=row['group'],checks=check,
                cached_mixed_action_maximum_difference=float(np.max(abs(current-baseline))),
                marginal_numerical_containment_passed=marginal_passed,maximum_raw64_128_bound_difference=marginal_difference,
                maximum_branch_above_bound=containment,selected_candidate=candidate,
                candidate_passed=check['passed'] and marginal_passed))
    result=dict(complete=True,groups=groups,local_risk_candidate_passed=all(r['candidate_passed'] for r in groups),
        successful_worker_cpu_seconds=cpu,collection_cpu_seconds=time.process_time()-start,
        estimated_cpu_seconds_including_reserves=prep['estimated_cpu_seconds_including_reserves'],
        action_count=365,remaining_calls=356,new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,
        full_predictive_integration_admitted=False,independent_confirmation=False,
        scope='Frozen decision-relevant risk observations from the saved n4 bank,all three allocations of their guarded groups. All8 identity scores and512/1024 positive components retained. Candidates are conditional on16 cached objective heads; no full batch outer convergence or query ranking.')
    save(out/'compact_summary.json',result);print('SCHEDULEURM_DONE '+json.dumps(result),flush=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','prepare-frozen','run','collect']);p.add_argument('--record')
    p.add_argument('--identity',type=int);p.add_argument('--kind',choices=['raw','projected']);p.add_argument('--plan',default=PLAN);a=p.parse_args()
    try:
        if a.stage=='prepare':prepare(a.plan)
        elif a.stage=='prepare-frozen':prepare_frozen(a.plan)
        elif a.stage=='collect':collect(a.plan)
        else:run(a.record,a.identity,a.kind,a.plan)
    finally:
        print('SCHEDULEURM_RESOURCE '+json.dumps(dict(process_cpu_seconds=time.process_time(),
            peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)),flush=True)
