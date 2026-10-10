"""Localize the admitted body bank's remaining joint-component integration error."""

import argparse
import copy
import gzip
import json
import time

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

from core.conditional_raw_reference import (conditional_raw_coordinates,
    conditional_risk_action_statistics, component_risk_statistics)
from core.conditional_terminal_kg import condition_eta_components
from core.raw_terminal_head_grid import compose_head_grid
from performance.complete_historical_source_heads import ROOT, read, save
from performance.diagnose_historical_predictive_tower import restrict_field
from performance.historical_full_bank_batch_kg import restore
from performance.refine_historical_observed_prediction import field_from_snapshot
from representation.vectorized_joint_function_risk import update_raw_observation

PLAN='performance/manifests/historical_component_guarded_v1.json'


def compressed_write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path,'wt',encoding='utf-8') as f:json.dump(value,f,separators=(',',':'))


def compressed_read(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:return json.load(f)


def export(plan=PLAN):
    spec=read(plan);parent=read(spec['parent_plan']);precision=read(parent['precision_plan'])
    case=parent['cases'][0];out=ROOT/spec['output'];started=time.process_time()
    if (out/'export_summary.json').exists():return read(str(out/'export_summary.json'))
    bank=read(parent['output']+'/'+case['state']+'/bank.json');prior=np.asarray(bank['prior_weights'])
    model,terminal,_,_=restore(parent,case['state']);np.testing.assert_array_equal(model.weights,prior)
    opposite={};summaries=[]
    for name in spec['records']:
        origin=int(name.split('_o')[1].split('_')[0])
        if origin not in opposite:
            opposite[origin]=read(precision['head_inputs'][case['name']][0].format(origin=origin)+'/head_records.json')
        data=read(precision['output']+'/shards/'+name+'/head_records.json');row=data['records'][0]
        f=opposite[origin]['records'];actions=data['actions'];assert actions==opposite[origin]['actions']
        guarded=np.array([i for i,a in enumerate(actions) if a['kind']=='guarded']);chosen=[];selection=[]
        for kind in ('raw','projected'):
            f_stats=np.asarray([r[kind][str(max(map(int,r[kind])))]['statistics'] for r in f])
            values=[]
            for power in (10,12):
                rule=row[kind][str(power)]
                values.append(compose_head_grid(f_stats,np.asarray(rule['statistics'])[None],
                    np.asarray(rule['costs'])[None],[r['head_log_scores'] for r in f],[row['head_log_scores']],prior)[0][:,0])
            delta=abs(values[1]-values[0]);where=np.unravel_index(delta[:,guarded].argmax(),delta[:,guarded].shape)
            action_index=int(guarded[where[1]]);chosen.append(action_index)
            selection.append(dict(kind=kind,action_index=action_index,opposite_position=f[where[0]]['position'],
                mixed_action_resolution_difference=float(delta[where[0],action_index])))
        chosen.append(next(i for i,a in enumerate(actions) if a['kind']=='direct' and a['primary']==parent['query'] and a['budgets']==[359]))
        chosen=list(dict.fromkeys(chosen));blocks=[]
        for a in chosen:
            action=actions[a];points=list(dict.fromkeys([parent['query']]+action['shortlist']));fields=[]
            for identity,pair in enumerate(model.components):
                field=restrict_field(pair[1],points)
                fields.append(dict(means=field.means.tolist(),covariances=field.covariances.tolist(),
                    log_weights=field.log_weights.tolist(),paid_batches=field.paid_batches,paid_episodes=field.paid_episodes))
            blocks.append(dict(action_index=a,original_action=action,points=points,
                action=dict(action,primary=points.index(action['primary']),shortlist=[points.index(i) for i in action['shortlist']]),
                fields=fields))
        reference={kind:{str(p):dict(statistics=np.asarray(row[kind][str(p)]['statistics'])[:,chosen].tolist(),
            costs=np.asarray(row[kind][str(p)]['costs'])[:,chosen].tolist()) for p in (10,12)} for kind in ('raw','projected')}
        f_rows=[]
        for r in f:
            f_rows.append(dict(position=r['position'],head_log_scores=r['head_log_scores'],
                **{kind:np.asarray(r[kind][str(max(map(int,r[kind])))]['statistics'])[:,chosen].tolist()
                    for kind in ('raw','projected')}))
        bundle=dict(name=name,origin=origin,query=parent['query'],label=row['label'],eta_label=row['eta_label'],
            replicates=row['replicates'],head_log_scores=row['head_log_scores'],protocol=terminal.protocol,
            prior_weights=prior.tolist(),utility_identities=row['utility_identities'],blocks=blocks,
            opposite=f_rows,reference=reference,selection=selection,seed=parent['seed'])
        path=out/'marginals'/(name+'.json.gz');compressed_write(path,bundle)
        summaries.append(dict(name=name,selection=selection,action_count=len(chosen),compressed_bytes=path.stat().st_size,
            maximum_joint_policy_count=max(len(b['points']) for b in blocks),
            source_components=[len(field['means']) for field in blocks[0]['fields']]))
    result=dict(complete=True,records=summaries,cpu_seconds=time.process_time()-started,
        full_posterior_restorations=1,formal_action_count=len(actions),all_source_identities_exported=True,
        new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,independent_confirmation=False)
    save(out/'export_summary.json',result);print('SCHEDULEURM_DONE '+json.dumps(result),flush=True);return result


def categorical_rule(prior,future,action,label,score,protocol,seed,power,*,raw,reweight):
    n=len(action['shortlist']);m=len(set([0]+action['shortlist']));dimensions=7 if raw else 6
    nodes=qmc.Sobol(dimensions,scramble=True,seed=seed).random_base2(power)
    component=np.minimum(np.searchsorted(np.cumsum(np.exp(future.log_weights)),nodes[:,0]),len(future.means)-1)
    counts=np.bincount(component,minlength=len(future.means));missing=float(np.exp(future.log_weights)[counts==0].sum())
    if raw:
        eta,mean,covariance,weights,diagnostics=conditional_raw_coordinates(prior,future,action['shortlist'],nodes,
            label,score,query=0,components=component,
            log_node_weights=-np.log(counts[component]) if reweight else None)
        position=np.arange(len(nodes))
    else:
        eta,mean,covariance,position=condition_eta_components(future,action['shortlist'],nodes,components=component)
        weights=np.exp(future.log_weights[component])/counts[component] if reweight else np.full(len(nodes),1/len(nodes))
        diagnostics=dict(predictive_log_evidence_error=0.)
    statistics,cost=conditional_risk_action_statistics(eta,mean,covariance,position,nodes,action,protocol,weights,
        raw=raw,query_points=m)
    diagnostics.update(unrepresented_component_mass=missing,represented_components=int(np.count_nonzero(counts)),
        diagnostic_only=bool(reweight))
    if raw and reweight:
        # Keep the lost mass visible; the normalized covered-component answer
        # is a diagnostic and is never substituted into the formal bank.
        diagnostics['covered_raw_evidence_mass']=float(np.exp(diagnostics['predictive_log_evidence_error']))
    return statistics,cost,diagnostics


def mixed_values(bundle,statistics,costs,kind):
    f=bundle['opposite']
    return compose_head_grid(np.asarray([r[kind] for r in f]),np.asarray(statistics)[None],
        np.asarray(costs)[None],[r['head_log_scores'] for r in f],[bundle['head_log_scores']],bundle['prior_weights'])[0][:,0]


def run(name,plan=PLAN):
    spec=read(plan);out=ROOT/spec['output'];bundle=compressed_read(out/'marginals'/(name+'.json.gz'))
    path=out/'results'/(name+'.json');path.parent.mkdir(parents=True,exist_ok=True)
    previous=read(str(path)) if path.exists() else {}
    if previous.get('complete') and all(any(r['name']==v['name'] for r in previous['rows'])
            for v in spec['component_rules']):return previous
    started=time.process_time();rows=list(previous.get('rows',[]));updates=[]
    for row in rows:
        if not row['rule'].get('categorical'):continue
        p=str(row['rule']['power']);native=np.asarray(row['statistics']['projected'])
        for a,b in enumerate(bundle['blocks']):
            if b['action']['kind']=='direct':
                native[:,a]=np.asarray(bundle['reference']['projected'][p]['statistics'])[:,a]
        row['statistics']['projected']=native.tolist();row['native_direct_reference_reused']=True
    for block in bundle['blocks']:
        pairs=[]
        for identity,payload in enumerate(block['fields']):
            if identity not in bundle['utility_identities']:pairs.append(None);continue
            field=field_from_snapshot(payload);future=copy.deepcopy(field)
            score=update_raw_observation(future,0,bundle['label'],None,1)
            assert abs(score-bundle['head_log_scores'][identity])<=1e-10
            pairs.append((field,future,score))
        updates.append(pairs)
    versions=[dict(name='A10',categorical=True,power=10,reweight=False),
        dict(name='A12',categorical=True,power=12,reweight=False),
        dict(name='B12',categorical=True,power=12,reweight=True)]+spec['component_rules']
    for version in versions:
        if any(r['name']==version['name'] for r in rows):continue
        tick=time.process_time();values={};costs={};diagnostics={}
        for kind,raw in [('raw',True),('projected',False)]:
            stats=np.zeros((len(bundle['prior_weights']),len(bundle['blocks']),4));c=np.zeros(stats.shape[:2]);checks=[]
            for a,(block,pairs) in enumerate(zip(bundle['blocks'],updates)):
                for identity,pair in enumerate(pairs):
                    if pair is None:continue
                    prior,future,score=pair;action=block['action']
                    if version.get('categorical'):
                        s,cc,d=categorical_rule(prior,future,action,bundle['label'],score,bundle['protocol'],
                            bundle['seed'],version['power'],raw=raw,reweight=version['reweight'])
                    else:
                        seed=bundle['seed']+version.get('seed_offset',0)
                        if version.get('scramble_scope')=='component':
                            # Canonical mean ranks give duplicated or reordered
                            # components the same rule without changing masses.
                            _,rank=np.unique(prior.means,axis=0,return_inverse=True)
                            seed=seed+rank*104729
                        s,cc,d=component_risk_statistics(prior,future,action,bundle['label'],score,bundle['protocol'],
                            query=0,raw=raw,order=version['order'],power=version['power'],
                            seed=seed)
                    stats[identity,a]=s;c[identity,a]=cc;checks.append(dict(identity=identity,action=a,**d))
            values[kind]=stats.tolist();costs[kind]=c.tolist();diagnostics[kind]=checks
        if version.get('categorical'):
            p=str(version['power']);native=np.asarray(values['projected'])
            for a,b in enumerate(bundle['blocks']):
                if b['action']['kind']=='direct':
                    native[:,a]=np.asarray(bundle['reference']['projected'][p]['statistics'])[:,a]
            values['projected']=native.tolist()
        result=dict(name=version['name'],rule=version,statistics=values,costs=costs,diagnostics=diagnostics,
            cpu_seconds=time.process_time()-tick)
        if version['name'] in ('A10','A12'):
            p=str(version['power']);result['maximum_cached_statistic_difference']=max(
                float(np.max(abs(np.asarray(values[kind])-bundle['reference'][kind][p]['statistics'])))
                for kind in ('raw','projected'))
            result['maximum_cached_statistic_difference_after_direct_reference']=max(
                float(np.max(abs(np.asarray(values[kind])-bundle['reference'][kind][p]['statistics'])))
                for kind in ('raw','projected'))
            assert result['maximum_cached_statistic_difference_after_direct_reference']<=1e-10
        rows.append(result)
        save(path,dict(complete=False,name=name,rows=rows,cpu_seconds=previous.get('cpu_seconds',0.)+time.process_time()-started,
            new_physical_calls=0,new_source_fits=0,new_verifier_calls=0))
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=len(rows),total=len(versions),unit='local_rules',name=name)),flush=True)
    summary=[]
    for previous,current in zip(rows,rows[1:]):
        summary.append(dict(low=previous['name'],high=current['name'],**{kind:float(np.max(abs(
            mixed_values(bundle,current['statistics'][kind],current['costs'][kind],kind)-
            mixed_values(bundle,previous['statistics'][kind],previous['costs'][kind],kind)))) for kind in ('raw','projected')}))
    by_name={r['name']:r for r in rows};local_checks=[]
    for low,high in [('E32_32','E32_128'),('F32_32','F32_128'),('E32_128','F32_128')]:
        if low not in by_name or high not in by_name:continue
        local_checks.append(dict(low=low,high=high,**{kind:float(np.max(abs(
            mixed_values(bundle,by_name[high]['statistics'][kind],by_name[high]['costs'][kind],kind)-
            mixed_values(bundle,by_name[low]['statistics'][kind],by_name[low]['costs'][kind],kind))))
            for kind in ('raw','projected')}))
    latest_evidence=max(abs(d['predictive_log_evidence_error']) for r in rows[-2:] for d in r['diagnostics']['raw'])
    local_passed=len(local_checks)==3 and all(c[k]<=spec['maximum_action_resolution_difference']
        for c in local_checks for k in ('raw','projected')) and latest_evidence<=1e-4
    result=dict(complete=True,name=name,rows=rows,comparisons=summary,cpu_seconds=previous.get('cpu_seconds',0.)+time.process_time()-started,
        component_scramble_checks=local_checks,selected_action_precision_passed=local_passed,
        new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,independent_confirmation=False,
        scope='Selected guarded actions and a direct anchor on six opened development observations; not full-bank KG admission.')
    save(path,result);print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True);return result


def collect(plan=PLAN):
    spec=read(plan);out=ROOT/spec['output'];records=[]
    for name in spec['records']:
        result=read(str(out/'results'/(name+'.json')))
        records.append(dict(name=name,complete=result['complete'],comparisons=result.get('comparisons',[]),
            component_scramble_checks=result.get('component_scramble_checks',[]),
            selected_action_precision_passed=result.get('selected_action_precision_passed',False),
            cpu_seconds=result['cpu_seconds'],rules=[dict(name=r['name'],cpu_seconds=r['cpu_seconds'],
                maximum_raw_log_evidence_error=max(abs(d['predictive_log_evidence_error']) for d in r['diagnostics']['raw']),
                maximum_unrepresented_component_mass=max(d.get('unrepresented_component_mass',0.) for d in r['diagnostics']['raw']),
                maximum_weighted_component_evidence_error=max(d.get('weighted_component_absolute_evidence_error',0.) for d in r['diagnostics']['raw']))
                for r in result['rows']]))
    result=dict(complete=all(r['complete'] for r in records),records=records,
        all_selected_action_precision_passed=all(r['selected_action_precision_passed'] for r in records),
        cpu_seconds=sum(r['cpu_seconds'] for r in records)+read(str(out/'export_summary.json'))['cpu_seconds'],
        new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,independent_confirmation=False,
        full_predictive_integration_admitted=False)
    save(out/'summary.json',result);print('SCHEDULEURM_DONE '+json.dumps(result),flush=True);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['export','run','collect'])
    parser.add_argument('--name');parser.add_argument('--plan',default=PLAN);args=parser.parse_args()
    if args.mode=='export':export(args.plan)
    elif args.mode=='run':run(args.name,args.plan)
    else:collect(args.plan)
