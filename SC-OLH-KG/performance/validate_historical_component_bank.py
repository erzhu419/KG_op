"""Continue one opened 365-action risk bank with shared component integrals."""

import argparse
import copy
import json
import resource
import signal
import time

import numpy as np

from core.conditional_raw_reference import component_risk_allocations
from core.raw_terminal_head_grid import compose_head_grid
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.diagnose_historical_component_guarded import compressed_read, compressed_write, mixed_values
from performance.diagnose_historical_predictive_tower import restrict_field
from performance.historical_full_bank_batch_kg import restore
from performance.refine_historical_observed_prediction import field_from_snapshot
from representation.vectorized_joint_function_risk import update_raw_observation

PLAN = 'performance/manifests/historical_component_bank_v1.json'
RULES = {'E5': (32,5,0), 'E7': (32,7,0), 'F7': (32,7,101), 'G7': (64,7,0),
    'H8': (64,8,0), 'I8': (64,8,101), 'J8': (128,8,0), 'K9': (64,9,0)}


def action_groups(actions):
    groups = {}
    for i, a in enumerate(actions):
        key = (a['kind'], tuple(a['shortlist']), a['incumbent_position'])
        groups.setdefault(key, []).append(i)
    return [dict(group=i, kind=key[0], action_indices=indices)
        for i,(key,indices) in enumerate(groups.items())]


def prepare(plan=PLAN):
    spec=read(plan);parent=read(spec['parent_plan']);precision=read(parent['precision_plan'])
    out=ROOT/spec['output'];out.mkdir(parents=True,exist_ok=True);started=time.process_time()
    (out/'parts').mkdir(exist_ok=True)
    signal.signal(signal.SIGPROF,cpu_timeout);signal.setitimer(signal.ITIMER_PROF,spec['preparation_cpu_seconds'])
    case=parent['cases'][0]
    data=read(precision['output']+'/shards/'+spec['record']+'/head_records.json');row=data['records'][0]
    assert row['head']==1 and row['position']==207 and row['replicates']==1 and row['eta_label'] is None
    opposite=read(precision['head_inputs'][case['name']][0].format(origin=5)+'/head_records.json')
    assert opposite['actions']==data['actions']
    model,terminal,_,_=restore(parent,case['state'])
    np.testing.assert_array_equal(model.weights,data['prior_weights'])
    groups=action_groups(data['actions']);assert len(groups)==146 and len(data['actions'])==365
    context=dict(name=spec['record'],query=parent['query'],label=row['label'],replicates=1,
        head_log_scores=row['head_log_scores'],protocol=terminal.protocol,prior_weights=data['prior_weights'],
        utility_identities=row['utility_identities'],seed=parent['seed'],actions=data['actions'],groups=groups,
        opposite=[dict(position=r['position'],head_log_scores=r['head_log_scores'],
            **{k:r[k][str(max(map(int,r[k])))]['statistics'] for k in ('raw','projected')})
            for r in opposite['records']],
        baseline={k:row[k]['12'] for k in ('raw','projected')})
    save(out/'context.json',context)
    selected=compressed_read(ROOT/spec['selected_marginal'])
    reused=[];total_bytes=0
    for group in groups:
        indices=group['action_indices'];original=[data['actions'][i] for i in indices]
        action=original[0];points=list(dict.fromkeys([parent['query']]+action['shortlist']))
        fields=[None]*len(model.weights)
        for h in row['utility_identities']:
            field=restrict_field(model.components[h][1],points)
            fields[h]=dict(means=field.means.tolist(),covariances=field.covariances.tolist(),
                log_weights=field.log_weights.tolist(),paid_batches=field.paid_batches,paid_episodes=field.paid_episodes)
        bundle={k:context[k] for k in ('name','query','label','replicates','head_log_scores','protocol','prior_weights','utility_identities','seed')}
        bundle.update(group=group['group'],kind=group['kind'],action_indices=indices,points=points,fields=fields,
            actions=[dict(a,primary=points.index(a['primary']),shortlist=[points.index(p) for p in a['shortlist']]) for a in original],
            opposite=[dict(r,**{k:np.asarray(r[k])[:,indices].tolist() for k in ('raw','projected')}) for r in context['opposite']])
        path=out/'marginals'/f'{group["group"]:03d}.json.gz';compressed_write(path,bundle);total_bytes+=path.stat().st_size
        match=next((a for a,b in enumerate(selected['blocks']) if b['action']['kind']=='guarded'
            and b['original_action']['shortlist']==action['shortlist']),None)
        if match is not None and group['kind']=='guarded':
            for h in row['utility_identities']:
                for key in ('means','covariances','log_weights'):
                    np.testing.assert_allclose(fields[h][key],selected['blocks'][match]['fields'][h][key],atol=1e-13,rtol=0.)
                for kind in ('raw','projected'):
                    base=ROOT/spec['allocation_cache']
                    cached=read(str(base/('compact_parts' if kind=='raw' else 'parts')/f'{h}_{match}_{kind}.json'))
                    rows={}
                    for name,cache_name in [('E7','coarse'),('H8','candidate')]:
                        r=cached['rows'][cache_name]
                        rows[name]=dict(statistics=r['statistics'],costs=r['costs'],diagnostics=r['diagnostics'],cpu_seconds=0.,reused=True)
                    save(out/'parts'/f'{group["group"]:03d}_{h}_{kind}.json',dict(group=group['group'],identity=h,kind=kind,rows=rows))
            reused.append(group['group'])
    direct=[i for i,a in enumerate(data['actions']) if a['kind']=='direct']
    cached_difference=float(np.max(abs(np.asarray(row['projected']['10']['statistics'])[:,direct]-
        np.asarray(row['projected']['12']['statistics'])[:,direct])))
    assert cached_difference<=1e-12
    result=dict(complete=True,record=spec['record'],actions=len(data['actions']),guarded_groups=sum(g['kind']=='guarded' for g in groups),
        direct_groups=sum(g['kind']=='direct' for g in groups),reused_guarded_groups=reused,native_direct_cached_difference=cached_difference,
        source_components={str(h):len(model.components[h][1].means) for h in row['utility_identities']},
        server_marginal_bytes=total_bytes,cpu_seconds=time.process_time()-started,
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,independent_confirmation=False)
    save(out/'preparation.json',result);signal.setitimer(signal.ITIMER_PROF,0.)
    print('SCHEDULEURM_DONE '+json.dumps(result),flush=True);return result


def run_group(group,identity,kind,rules,spec):
    out=ROOT/spec['output'];path=out/'parts'/f'{group:03d}_{identity}_{kind}.json'
    result=read(str(path)) if path.exists() else dict(group=group,identity=identity,kind=kind,rows={})
    pending=[n for n in rules if n not in result['rows']]
    if not pending:return
    b=compressed_read(out/'marginals'/f'{group:03d}.json.gz')
    before=field_from_snapshot(b['fields'][identity]);after=copy.deepcopy(before)
    score=update_raw_observation(after,0,b['label'],None,1)
    assert abs(score-b['head_log_scores'][identity])<=1e-10
    _,rank=np.unique(before.means,axis=0,return_inverse=True)
    for name in pending:
        order,power,offset=RULES[name];tick=time.process_time()
        if b['kind']=='direct' and len(b['points'])==1:power=0
        s,c,d=component_risk_allocations(before,after,b['actions'],b['label'],score,b['protocol'],query=0,
            raw=kind=='raw',order=order,power=power,seed=b['seed']+offset+rank*104729,
            integrated_position=b['actions'][0]['incumbent_position'])
        result['rows'][name]=dict(statistics=s.tolist(),costs=c.tolist(),diagnostics=d,cpu_seconds=time.process_time()-tick)
        save(path,result)


def selected_groups(context,stage,out,kind):
    if stage=='screen':return [g['group'] for g in context['groups'] if kind=='raw' or g['kind']=='guarded']
    previous=read(str(out/('screen_summary.json' if stage=='refine' else 'refined_summary.json')))
    return [g['group'] for g in previous['groups'] if not g['checks'][kind]['passed']]


def run(identity,kind,shard,stage,plan=PLAN):
    spec=read(plan);out=ROOT/spec['output'];context=read(str(out/'context.json'))
    (out/'tasks'/stage).mkdir(parents=True,exist_ok=True)
    groups=selected_groups(context,stage,out,kind)[shard::spec['shards_per_identity_kind']]
    rules={'screen':['E5','E7','F7','G7'],'refine':['H8','I8','J8'],'resolution':['K9']}[stage]
    started=time.process_time();handler=signal.signal(signal.SIGPROF,cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF,spec['maximum_job_cpu_seconds'])
    complete=0;result=dict(identity=identity,kind=kind,shard=shard,stage=stage,groups=groups,complete=False)
    try:
        for group in groups:
            run_group(group,identity,kind,rules,spec);complete+=1
            print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=complete,total=len(groups),unit='joint_policy_groups',
                identity=identity,kind=kind,shard=shard,stage=stage)),flush=True)
        result['complete']=True
    except TimeoutError:result['cpu_budget_stopped']=True
    finally:
        signal.setitimer(signal.ITIMER_PROF,0.);signal.signal(signal.SIGPROF,handler)
        result.update(completed_groups=complete,cpu_seconds=time.process_time()-started,
            peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            new_physical_calls=0,new_source_fits=0,new_verifier_calls=0)
        save(out/'tasks'/stage/f'{identity}_{kind}_{shard}.json',result)
    print('SCHEDULEURM_DONE '+json.dumps(result),flush=True);return result


def group_checks(bundle,parts,kind,pairs,tolerance):
    values={};checks=[];identity_count=len(bundle['prior_weights']);action_count=len(bundle['actions'])
    for name in set(n for pair in pairs for n in pair):
        statistics=np.zeros((identity_count,action_count,4));costs=np.zeros((identity_count,action_count))
        for part in parts:
            statistics[part['identity']]=part['rows'][name]['statistics'];costs[part['identity']]=part['rows'][name]['costs']
        values[name]=mixed_values(bundle,statistics,costs,kind)
    for low,high in pairs:
        delta=abs(values[high]-values[low]);checks.append(dict(low=low,high=high,maximum_difference=float(delta.max()),
            action_differences=np.max(delta,axis=0).tolist()))
    evidence=max(abs(p['rows'][n]['diagnostics']['predictive_log_evidence_error']) for p in parts for n in values)
    return dict(passed=evidence<=tolerance and all(c['maximum_difference']<=tolerance for c in checks),
        comparisons=checks,maximum_log_evidence_error=evidence)


def collect(stage,plan=PLAN):
    spec=read(plan);out=ROOT/spec['output'];context=read(str(out/'context.json'));groups=[]
    previous=read(str(out/('screen_summary.json' if stage=='refine' else 'refined_summary.json'))) if stage!='screen' else None
    tables={k:np.asarray(context['baseline'][k]['statistics']).copy() for k in ('raw','projected')}
    costs={k:np.asarray(context['baseline'][k]['costs']).copy() for k in ('raw','projected')}
    for group in context['groups']:
        gid=group['group'];bundle=compressed_read(out/'marginals'/f'{gid:03d}.json.gz');checks={};chosen={}
        for kind in ('raw','projected'):
            if kind=='projected' and group['kind']=='direct':
                checks[kind]=dict(passed=True,native_analytic_reference_reused=True);chosen[kind]='native_reference';continue
            old=previous['groups'][gid]['checks'][kind] if previous else None
            active=stage=='screen' or not old['passed']
            if not active:
                checks[kind]=old;chosen[kind]=previous['groups'][gid]['selected_rules'][kind]
            else:
                pairs={'screen':[('E5','E7'),('E7','F7'),('E7','G7')],
                    'refine':[('G7','H8'),('H8','I8'),('H8','J8')],
                    'resolution':[('H8','I8'),('H8','J8'),('H8','K9')]}[stage]
                parts=[read(str(out/'parts'/f'{gid:03d}_{h}_{kind}.json')) for h in context['utility_identities']]
                checks[kind]=group_checks(bundle,parts,kind,pairs,spec['maximum_action_difference'])
                chosen[kind]={'screen':'G7','refine':'H8','resolution':'H8'}[stage]
            for h in context['utility_identities']:
                r=read(str(out/'parts'/f'{gid:03d}_{h}_{kind}.json'))['rows'][chosen[kind]]
                tables[kind][h,group['action_indices']]=r['statistics'];costs[kind][h,group['action_indices']]=r['costs']
        groups.append(dict(**group,checks=checks,selected_rules=chosen))
    values={k:compose_head_grid(np.asarray([r[k] for r in context['opposite']]),tables[k][None],costs[k][None],
        [r['head_log_scores'] for r in context['opposite']],[context['head_log_scores']],context['prior_weights'])[0][:,0] for k in tables}
    old_values={k:compose_head_grid(np.asarray([r[k] for r in context['opposite']]),
        np.asarray(context['baseline'][k]['statistics'])[None],np.asarray(context['baseline'][k]['costs'])[None],
        [r['head_log_scores'] for r in context['opposite']],[context['head_log_scores']],context['prior_weights'])[0][:,0] for k in tables}
    failed={k:[g['group'] for g in groups if not g['checks'][k]['passed']] for k in tables}
    expected=[out/'tasks'/stage/f'{h}_{k}_{s}.json' for h in context['utility_identities']
        for k in ('raw','projected') for s in range(spec['shards_per_identity_kind'])
        if selected_groups(context,stage,out,k)[s::spec['shards_per_identity_kind']]]
    tasks=[read(str(p)) for p in expected]
    complete=all(t['complete'] for t in tasks)
    summary=dict(complete=complete,record=spec['record'],stage=stage,groups=groups,failed_groups=failed,
        actions=len(context['actions']),actions_checked={k:sum(len(g['action_indices']) for g in groups if g['checks'][k]['passed']) for k in tables},
        single_record_risk_bank_precision_passed=complete and not any(failed.values()),
        old_new_maximum_action_difference={k:float(abs(values[k]-old_values[k]).max()) for k in tables},
        changed_recommendation_count={k:int(np.count_nonzero(values[k].argmax(axis=1)!=old_values[k].argmax(axis=1))) for k in tables},
        maximum_best_value_change={k:float(abs(values[k].max(axis=1)-old_values[k].max(axis=1)).max()) for k in tables},
        cpu_seconds=sum(t['cpu_seconds'] for t in tasks),maximum_task_peak_rss_mb=max((t['peak_rss_mb'] for t in tasks),default=0.),
        full_predictive_integration_admitted=False,independent_confirmation=False,
        new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,
        scope='One opened development risk observation, all 365 actions, fixed cached opposite head. Rule differences are diagnostics, not strict error bounds.')
    save(out/({'screen':'screen_summary.json','refine':'refined_summary.json','resolution':'admitted_summary.json'}[stage]),summary)
    save(out/(stage+'_risk_bank.json'),dict(statistics={k:v.tolist() for k,v in tables.items()},costs={k:v.tolist() for k,v in costs.items()}))
    print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in summary.items() if k!='groups'}),flush=True);return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','collect'])
    parser.add_argument('--stage',choices=['screen','refine','resolution'],default='screen')
    parser.add_argument('--identity',type=int);parser.add_argument('--kind',choices=['raw','projected'])
    parser.add_argument('--shard',type=int);parser.add_argument('--plan',default=PLAN);args=parser.parse_args()
    if args.mode=='prepare':prepare(args.plan)
    elif args.mode=='run':run(args.identity,args.kind,args.shard,args.stage,args.plan)
    else:collect(args.stage,args.plan)
