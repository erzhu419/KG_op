"""Shared-reference pilot on one already opened, frozen joint risk block."""

import argparse
import copy
import json
import time

import numpy as np

from core.conditional_raw_reference import shared_risk_reference,component_control_risk_statistics,component_risk_statistics
from performance.complete_historical_source_heads import ROOT,read,save
from performance.diagnose_historical_component_guarded import compressed_read,mixed_values
from performance.refine_historical_observed_prediction import field_from_snapshot
from representation.vectorized_joint_function_risk import update_raw_observation

BASE=ROOT/'paper_artifacts/historical_component_guarded_v1'
OUT=BASE/'shared_reference'
NAME='source_n1_o5_h1_p207'


def run(identity,action_index,kind,*,record=NAME,output=OUT):
    path=output/'parts'/f'{identity}_{action_index}_{kind}.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and read(str(path)).get('complete'):return read(str(path))
    started=time.process_time();bundle=compressed_read(BASE/'marginals'/(record+'.json.gz'))
    assert identity in bundle['utility_identities'] and bundle['replicates']==1
    block=bundle['blocks'][action_index];assert block['action']['kind']=='guarded'
    prior=field_from_snapshot(block['fields'][identity]);future=copy.deepcopy(prior)
    score=update_raw_observation(future,0,bundle['label'],None,1)
    assert abs(score-bundle['head_log_scores'][identity])<=1e-10
    raw=kind=='raw';kwargs=dict(query=0,raw=raw,order=32);refs={};rows={}
    for name,power,offset in [('R12',12,0),('R14',14,0),('S14',14,101)]:
        tick=time.process_time()
        refs[name]=shared_risk_reference(prior,future,block['action'],bundle['label'],score,bundle['protocol'],
            power=power,seed=bundle['seed']+7000001+offset,**kwargs)
        refs[name]['cpu_seconds']=time.process_time()-tick
    _,rank=np.unique(prior.means,axis=0,return_inverse=True)
    historical=read(str(BASE/'results'/(record+'.json'))) if (BASE/'results'/(record+'.json')).exists() else read(str(BASE/'reference_results'/(record+'.json')))
    old={r['name']:r for r in historical['rows']}
    for name,power,offset in [('E5',5,0),('E7',7,0),('F7',7,101)]:
        tick=time.process_time()
        s,c,d=component_control_risk_statistics(prior,future,block['action'],bundle['label'],score,bundle['protocol'],
            refs['R14'],power=power,seed=bundle['seed']+offset+rank*104729,**kwargs)
        old_name={'E5':'E32_32','E7':'E32_128','F7':'F32_128'}[name]
        expected=np.asarray(old[old_name]['statistics'][kind])[identity,action_index]
        difference=float(np.max(abs(np.asarray(d['ordinary_statistics'])-expected)))
        assert difference<=1e-10
        rows[name]=dict(statistics=s.tolist(),cost=c,diagnostics=d,cpu_seconds=time.process_time()-tick,
            maximum_cached_ordinary_difference=difference)
    # Reuse the residual integrals; only the shared reference changes.
    for name in ('R12','S14'):
        row=rows['E7'];mass=np.exp(row['diagnostics']['predictive_log_evidence_error'])
        correction=(np.asarray(refs[name]['numerator'])-np.asarray(refs['R14']['numerator']))/mass
        rows[name]=dict(statistics=(np.asarray(row['statistics'])+correction[:4]).tolist(),cost=float(row['cost']+correction[4]))
    result=dict(complete=True,name=record,identity=identity,action=action_index,kind=kind,references=refs,rows=rows,
        cpu_seconds=time.process_time()-started,new_physical_calls=0,new_source_fits=0,new_verifier_calls=0)
    save(path,result);print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in result.items() if k not in ('rows','references')}),flush=True)
    return result


def collect():
    bundle=compressed_read(BASE/'marginals'/(NAME+'.json.gz'))
    parts=[read(str(OUT/'parts'/f'{h}_{a}_{kind}.json')) for h in bundle['utility_identities']
        for a,b in enumerate(bundle['blocks']) if b['action']['kind']=='guarded' for kind in ('raw','projected')]
    historical=read(str(BASE/'results'/(NAME+'.json'))) if (BASE/'results'/(NAME+'.json')).exists() else read(str(BASE/'reference_results'/(NAME+'.json')))
    old={r['name']:r for r in historical['rows']};values={};ordinary={};tables={};costs={};comparisons=[]
    low32_complete=all('F5' in p['rows'] for p in parts)
    names=['E5','E7','F7','R12','S14']+(['F5'] if low32_complete else [])
    for name in names:
        baseline=old[{'E5':'E32_32','F5':'F32_32','F7':'F32_128'}.get(name,'E32_128')]
        tables[name]={kind:np.asarray(baseline['statistics'][kind]).copy() for kind in ('raw','projected')}
        costs[name]={kind:np.asarray(baseline['costs'][kind]).copy() for kind in ('raw','projected')}
        for part in parts:
            h,a,kind=part['identity'],part['action'],part['kind']
            tables[name][kind][h,a]=part['rows'][name]['statistics'];costs[name][kind][h,a]=part['rows'][name]['cost']
        values[name]={kind:mixed_values(bundle,tables[name][kind],costs[name][kind],kind) for kind in ('raw','projected')}
        if name in ('E5','E7','F7'):
            ordinary[name]={kind:mixed_values(bundle,baseline['statistics'][kind],baseline['costs'][kind],kind) for kind in ('raw','projected')}
    pairs=[('E5','E7'),('E7','F7'),('R12','E7'),('E7','S14')]+([('F5','F7'),('E5','F5')] if low32_complete else [])
    for low,high in pairs:
        comparisons.append(dict(low=low,high=high,**{kind:float(np.max(abs(values[high][kind]-values[low][kind])))
            for kind in ('raw','projected')}))
    ordinary_checks=[dict(low=low,high=high,**{kind:float(np.max(abs(ordinary[high][kind]-ordinary[low][kind])))
        for kind in ('raw','projected')}) for low,high in [('E5','E7'),('E7','F7')]]
    cached_cost_difference=max(abs(p['rows'][n]['diagnostics']['ordinary_cost']-
        old[{'E5':'E32_32','F5':'F32_32','E7':'E32_128','F7':'F32_128'}[n]]['costs'][p['kind']][p['identity']][p['action']])
        for p in parts for n in ('E5','E7','F7')+ (('F5',) if low32_complete else ()))
    assert cached_cost_difference<=1e-10
    evidence=max(abs(p['rows'][n]['diagnostics']['predictive_log_evidence_error']) for p in parts for n in ('E5','E7','F7'))
    passed=all(c[k]<=1e-4 for c in comparisons[:4] for k in ('raw','projected')) and evidence<=1e-4
    low32_passed=low32_complete and all(c[k]<=1e-4 for c in comparisons for k in ('raw','projected')) and evidence<=1e-4
    result=dict(complete=all(p['complete'] for p in parts),name=NAME,comparisons=comparisons,
        ordinary_comparisons=ordinary_checks,maximum_raw_log_evidence_error=evidence,
        selected_action_precision_passed=passed,full_predictive_integration_admitted=False,
        low32_precision_passed=bool(low32_passed),
        cpu_seconds=sum(p['cpu_seconds'] for p in parts)+(read(str(OUT/'low32_extension.json'))['cpu_seconds'] if low32_complete else 0.),part_count=len(parts),
        maximum_cached_ordinary_difference=max(p['rows'][n]['maximum_cached_ordinary_difference'] for p in parts for n in ('E5','E7','F7')),
        maximum_cached_ordinary_cost_difference=float(cached_cost_difference),
        references=[dict(identity=p['identity'],action=p['action'],kind=p['kind'],
            component=p['references']['R14']['component'],cpu_seconds=sum(r['cpu_seconds'] for r in p['references'].values())) for p in parts],
        corrected_statistics={n:{k:v.tolist() for k,v in tab.items()} for n,tab in tables.items()},
        corrected_costs={n:{k:v.tolist() for k,v in tab.items()} for n,tab in costs.items()},
        new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,independent_confirmation=False,
        scope='Two selected guarded actions and the reused direct anchor on one opened development observation; not full-bank admission.')
    save(OUT/'summary.json',result);print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in result.items() if k not in ('corrected_statistics','corrected_costs')}),flush=True)
    return result


def extend_low():
    """Test a second 32-point rule using the already computed high reference."""
    path=OUT/'low32_extension.json'
    if path.exists() and read(str(path)).get('complete'):return read(str(path))
    started=time.process_time();bundle=compressed_read(BASE/'marginals'/(NAME+'.json.gz'));timers=[]
    for h in bundle['utility_identities']:
        for a,block in enumerate(bundle['blocks']):
            if block['action']['kind']!='guarded':continue
            prior=field_from_snapshot(block['fields'][h]);future=copy.deepcopy(prior)
            score=update_raw_observation(future,0,bundle['label'],None,1)
            _,rank=np.unique(prior.means,axis=0,return_inverse=True)
            for kind in ('raw','projected'):
                part_path=OUT/'parts'/f'{h}_{a}_{kind}.json';part=read(str(part_path))
                tick=time.process_time()
                s,c,d=component_control_risk_statistics(prior,future,block['action'],bundle['label'],score,bundle['protocol'],
                    part['references']['R14'],query=0,raw=kind=='raw',order=32,power=5,
                    seed=bundle['seed']+101+rank*104729)
                part['rows']['F5']=dict(statistics=s.tolist(),cost=c,diagnostics=d,cpu_seconds=time.process_time()-tick)
                timers.append(part['rows']['F5']['cpu_seconds']);save(part_path,part)
    result=dict(complete=True,cpu_seconds=time.process_time()-started,rule_cpu_seconds=sum(timers),new_reference_solves=0)
    save(path,result);print('SCHEDULEURM_DONE '+json.dumps(result),flush=True);return result


def coordinate(identity,action_index,kind,*,record=NAME,output=OUT):
    path=output/'coordinate_parts'/f'{identity}_{action_index}_{kind}.json';path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and read(str(path)).get('complete'):return read(str(path))
    started=time.process_time();bundle=compressed_read(BASE/'marginals'/(record+'.json.gz'))
    assert identity in bundle['utility_identities'] and bundle['replicates']==1
    block=bundle['blocks'][action_index];assert block['action']['kind']=='guarded'
    prior=field_from_snapshot(block['fields'][identity]);future=copy.deepcopy(prior)
    score=update_raw_observation(future,0,bundle['label'],None,1)
    assert abs(score-bundle['head_log_scores'][identity])<=1e-10
    _,rank=np.unique(prior.means,axis=0,return_inverse=True);rows={}
    for name,order,power,offset in [('E5',32,5,0),('E7',32,7,0),('F7',32,7,101),('G7',64,7,0)]:
        tick=time.process_time()
        s,c,d=component_risk_statistics(prior,future,block['action'],bundle['label'],score,bundle['protocol'],
            query=0,raw=kind=='raw',order=order,power=power,seed=bundle['seed']+offset+rank*104729,
            integrated_position=block['action']['incumbent_position'])
        rows[name]=dict(statistics=s.tolist(),cost=c,diagnostics=d,cpu_seconds=time.process_time()-tick)
    result=dict(complete=True,name=record,identity=identity,action=action_index,kind=kind,rows=rows,
        cpu_seconds=time.process_time()-started,new_physical_calls=0,new_source_fits=0,new_verifier_calls=0)
    save(path,result);return result


def collect_coordinate():
    bundle=compressed_read(BASE/'marginals'/(NAME+'.json.gz'))
    parts=[read(str(OUT/'coordinate_parts'/f'{h}_{a}_{kind}.json')) for h in bundle['utility_identities']
        for a,b in enumerate(bundle['blocks']) if b['action']['kind']=='guarded' for kind in ('raw','projected')]
    control=read(str(OUT/'summary.json'));values={};tables={};costs={}
    for name in ('E5','E7','F7','G7'):
        # The direct anchor and all original identity weights/scores are unchanged.
        tables[name]={kind:np.asarray(control['corrected_statistics']['E7'][kind]).copy() for kind in ('raw','projected')}
        costs[name]={kind:np.asarray(control['corrected_costs']['E7'][kind]).copy() for kind in ('raw','projected')}
        for part in parts:
            h,a,kind=part['identity'],part['action'],part['kind'];row=part['rows'][name]
            tables[name][kind][h,a]=row['statistics'];costs[name][kind][h,a]=row['cost']
        values[name]={kind:mixed_values(bundle,tables[name][kind],costs[name][kind],kind) for kind in ('raw','projected')}
    comparisons=[dict(low=low,high=high,**{kind:float(np.max(abs(values[high][kind]-values[low][kind])))
        for kind in ('raw','projected')}) for low,high in [('E5','E7'),('E7','F7'),('E7','G7')]]
    evidence=max(abs(p['rows'][n]['diagnostics']['predictive_log_evidence_error']) for p in parts for n in ('E5','E7','F7','G7'))
    cross=[]
    for cv in ('E5','F5','E7','F7'):
        if cv not in control['corrected_statistics']:continue
        for coordinate_name in ('E7','G7'):
            cross.append(dict(control=cv,coordinate=coordinate_name,**{kind:float(np.max(abs(values[coordinate_name][kind]-
                mixed_values(bundle,control['corrected_statistics'][cv][kind],control['corrected_costs'][cv][kind],kind))))
                for kind in ('raw','projected')}))
    result=dict(complete=all(p['complete'] for p in parts),comparisons=comparisons,
        maximum_raw_log_evidence_error=evidence,selected_action_precision_passed=evidence<=1e-4 and
            all(c[k]<=1e-4 for c in comparisons for k in ('raw','projected')),
        full_predictive_integration_admitted=False,cpu_seconds=sum(p['cpu_seconds'] for p in parts),
        cross_rule_checks=cross,cross_rule_128_passed=all(c[k]<=1e-4 for c in cross if c['control'] in ('E7','F7') for k in ('raw','projected')),
        cross_rule_32_passed=control.get('low32_precision_passed',False) and all(c[k]<=1e-4
            for c in cross if c['control'] in ('E5','F5') for k in ('raw','projected')),
        original_control_difference={kind:float(np.max(abs(values['E7'][kind]-mixed_values(bundle,
            control['corrected_statistics']['E7'][kind],control['corrected_costs']['E7'][kind],kind)))) for kind in ('raw','projected')},
        statistics={n:{k:v.tolist() for k,v in tab.items()} for n,tab in tables.items()},
        costs={n:{k:v.tolist() for k,v in tab.items()} for n,tab in costs.items()},
        independent_confirmation=False,new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,
        scope='Original joint distribution; only the analytically integrated mean changes to the incumbent on two selected actions.')
    save(OUT/'coordinate_summary.json',result);print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in result.items() if k not in ('statistics','costs')}),flush=True);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['run','collect','extend-low','coordinate','collect-coordinate'])
    parser.add_argument('--identity',type=int);parser.add_argument('--action',type=int)
    parser.add_argument('--kind',choices=['raw','projected']);args=parser.parse_args()
    if args.mode=='run':run(args.identity,args.action,args.kind)
    elif args.mode=='collect':collect()
    elif args.mode=='extend-low':extend_low()
    elif args.mode=='coordinate':
        result=coordinate(args.identity,args.action,args.kind)
        print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
    else:collect_coordinate()
