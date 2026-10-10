"""Reuse historical conditional integration on opened joint-risk failures."""

import argparse
import copy
import json
import time

import numpy as np

from performance.complete_historical_source_heads import ROOT,read,save
from performance.diagnose_historical_component_guarded import compressed_read,mixed_values
from performance.diagnose_historical_shared_risk import BASE,OUT,coordinate,run as control
from core.conditional_raw_reference import component_risk_statistics,component_control_risk_statistics,shared_risk_reference
from performance.refine_historical_observed_prediction import field_from_snapshot
from representation.vectorized_joint_function_risk import update_raw_observation

PLAN='performance/manifests/historical_incumbent_validation_v1.json'


def run_record(name,identity,kind,plan=PLAN):
    spec=read(plan);assert name in spec['records'];output=ROOT/spec['output']/name
    path=output/'tasks'/f'{identity}_{kind}.json';path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and read(str(path)).get('complete'):return read(str(path))
    started=time.process_time();bundle=compressed_read(BASE/'marginals'/(name+'.json.gz'));count=0
    for a,block in enumerate(bundle['blocks']):
        if block['action']['kind']!='guarded':continue
        coordinate(identity,a,kind,record=name,output=output);count+=1
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(name=name,identity=identity,kind=kind,
            current=count,total=sum(b['action']['kind']=='guarded' for b in bundle['blocks']),unit='guarded_actions')),flush=True)
    result=dict(complete=True,name=name,identity=identity,kind=kind,guarded_actions=count,
        cpu_seconds=time.process_time()-started,new_physical_calls=0,new_source_fits=0,new_verifier_calls=0)
    save(path,result);print('SCHEDULEURM_DONE '+json.dumps(result),flush=True);return result


def coordinate_result(bundle,parts,baseline,tolerance=1e-4):
    tables={};costs={};values={}
    for name in ('E5','E7','F7','G7'):
        tables[name]={kind:np.asarray(baseline['statistics'][kind]).copy() for kind in ('raw','projected')}
        costs[name]={kind:np.asarray(baseline['costs'][kind]).copy() for kind in ('raw','projected')}
        for part in parts:
            h,a,kind=part['identity'],part['action'],part['kind'];row=part['rows'][name]
            tables[name][kind][h,a]=row['statistics'];costs[name][kind][h,a]=row['cost']
        values[name]={kind:mixed_values(bundle,tables[name][kind],costs[name][kind],kind) for kind in ('raw','projected')}
    comparisons=[]
    for low,high in [('E5','E7'),('E7','F7'),('E7','G7')]:
        delta={kind:abs(values[high][kind]-values[low][kind]) for kind in ('raw','projected')}
        comparisons.append(dict(low=low,high=high,**{kind:float(delta[kind].max()) for kind in delta},
            actions=[dict(action=a,action_index=block['action_index'],
                **{kind:float(delta[kind][:,a].max()) for kind in delta})
                for a,block in enumerate(bundle['blocks']) if block['action']['kind']=='guarded']))
    evidence=max(abs(p['rows'][n]['diagnostics']['predictive_log_evidence_error']) for p in parts for n in ('E5','E7','F7','G7'))
    failures=[dict(low=c['low'],high=c['high'],kind=kind,**a) for c in comparisons for a in c['actions']
        for kind in ('raw','projected') if a[kind]>tolerance]
    return dict(complete=all(p['complete'] for p in parts),name=bundle['name'],comparisons=comparisons,
        maximum_raw_log_evidence_error=evidence,selected_action_precision_passed=not failures and evidence<=tolerance,
        failed_action_checks=failures,full_predictive_integration_admitted=False,
        statistics={n:{k:v.tolist() for k,v in tab.items()} for n,tab in tables.items()},
        costs={n:{k:v.tolist() for k,v in tab.items()} for n,tab in costs.items()},
        independent_confirmation=False,new_physical_calls=0,new_source_fits=0,new_verifier_calls=0)


def collect(plan=PLAN):
    spec=read(plan);output=ROOT/spec['output'];records=[]
    for name in spec['records']:
        bundle=compressed_read(BASE/'marginals'/(name+'.json.gz'));case=output/name
        mixing=dict(bundle);mixing['blocks']=[{k:v for k,v in b.items() if k!='fields'} for b in bundle['blocks']]
        (output/'mixing_inputs').mkdir(exist_ok=True);save(output/'mixing_inputs'/(name+'.json'),mixing)
        parts=[read(str(case/'coordinate_parts'/f'{h}_{a}_{kind}.json')) for h in bundle['utility_identities']
            for a,b in enumerate(bundle['blocks']) if b['action']['kind']=='guarded' for kind in ('raw','projected')]
        historical=read(str(BASE/'results'/(name+'.json')))
        baseline=next(r for r in historical['rows'] if r['name']=='E32_128')
        result=coordinate_result(bundle,parts,baseline,spec['maximum_action_resolution_difference'])
        result['cpu_seconds']=sum(read(str(case/'tasks'/f'{h}_{kind}.json'))['cpu_seconds']
            for h in bundle['utility_identities'] for kind in ('raw','projected'))
        save(case/'summary.json',result)
        records.append({k:v for k,v in result.items() if k not in ('statistics','costs')})
    reused=read(str(OUT/'coordinate_summary.json'))
    result=dict(complete=all(r['complete'] for r in records),records=records,reused_record=spec['reused_record'],
        reused_record_precision_passed=reused['selected_action_precision_passed'],
        all_six_selected_action_precision_passed=all(r['selected_action_precision_passed'] for r in records)
            and reused['selected_action_precision_passed'],cpu_seconds=sum(r['cpu_seconds'] for r in records),
        full_predictive_integration_admitted=False,independent_confirmation=False,
        new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,scope=spec['scope'])
    save(output/'summary.json',result);print('SCHEDULEURM_DONE '+json.dumps(result),flush=True);return result


def control_result(bundle,coordinate_summary,parts,tolerance=1e-4):
    """Replace only failed action/kind pairs, retaining the incumbent results."""
    selected={(p['action'],p['kind']) for p in parts};tables={};costs={};values={}
    for name in ('E5','E7','F7','R12','S14'):
        tables[name]={k:np.asarray(coordinate_summary['statistics']['E7'][k]).copy() for k in ('raw','projected')}
        costs[name]={k:np.asarray(coordinate_summary['costs']['E7'][k]).copy() for k in ('raw','projected')}
        for p in parts:
            h,a,k=p['identity'],p['action'],p['kind'];row=p['rows'][name]
            tables[name][k][h,a]=row['statistics'];costs[name][k][h,a]=row['cost']
        values[name]={k:mixed_values(bundle,tables[name][k],costs[name][k],k) for k in ('raw','projected')}
    comparisons=[]
    for low,high in [('E5','E7'),('E7','F7'),('R12','E7'),('E7','S14')]:
        comparisons.append(dict(low=low,high=high,**{k:float(abs(values[high][k]-values[low][k]).max())
            for k in ('raw','projected')}))
    cross=[]
    for coordinate_name in ('E7','G7'):
        for control_name in ('E7','F7'):
            for a,k in sorted(selected):
                reference=mixed_values(bundle,coordinate_summary['statistics'][coordinate_name][k],
                    coordinate_summary['costs'][coordinate_name][k],k)
                cross.append(dict(control=control_name,coordinate=coordinate_name,action=a,
                    action_index=bundle['blocks'][a]['action_index'],kind=k,
                    difference=float(abs(values[control_name][k][:,a]-reference[:,a]).max())))
    evidence=max(abs(p['rows'][n]['diagnostics']['predictive_log_evidence_error'])
        for p in parts for n in ('E5','E7','F7'))
    unresolved=[f for f in coordinate_summary['failed_action_checks'] if (f['action'],f['kind']) not in selected]
    control_passed=evidence<=tolerance and all(c[k]<=tolerance for c in comparisons for k in ('raw','projected'))
    # The GH32 incumbent values already failed; preserve that comparison as a
    # diagnostic and require agreement with the higher GH64 coordinate rule.
    higher_coordinate_passed=all(c['difference']<=tolerance for c in cross if c['coordinate']=='G7')
    return dict(complete=all(p['complete'] for p in parts),name=bundle['name'],comparisons=comparisons,
        cross_coordinate_checks=cross,maximum_raw_log_evidence_error=evidence,
        control_precision_passed=control_passed,higher_coordinate_agreement_passed=higher_coordinate_passed,
        selected_action_precision_passed=control_passed and higher_coordinate_passed and not unresolved,
        remaining_incumbent_failures=unresolved,original_incumbent_precision_passed=coordinate_summary['selected_action_precision_passed'],
        cpu_seconds=sum(p['cpu_seconds'] for p in parts),
        statistics={n:{k:v.tolist() for k,v in tab.items()} for n,tab in tables.items()},
        costs={n:{k:v.tolist() for k,v in tab.items()} for n,tab in costs.items()},
        full_predictive_integration_admitted=False,independent_confirmation=False,
        new_physical_calls=0,new_source_fits=0,new_verifier_calls=0)


def collect_control(plan=PLAN):
    spec=read(plan);output=ROOT/spec['output'];records=[]
    for name in spec['records']:
        bundle=read(str(output/'mixing_inputs'/(name+'.json')));case=output/name
        original=read(str(case/'summary.json'))
        selected=sorted({(f['action'],f['kind']) for f in original['failed_action_checks']})
        parts=[read(str(case/'parts'/f'{h}_{a}_{k}.json')) for a,k in selected for h in bundle['utility_identities']]
        result=control_result(bundle,original,parts,spec['maximum_action_resolution_difference']) if parts else original
        save(case/'control_summary.json',result)
        records.append({k:v for k,v in result.items() if k not in ('statistics','costs')})
    reused=read(str(OUT/'coordinate_summary.json'))
    result=dict(complete=all(r['complete'] for r in records),records=records,
        all_six_selected_action_precision_passed=all(r['selected_action_precision_passed'] for r in records)
            and reused['selected_action_precision_passed'],control_cpu_seconds=sum(r['cpu_seconds'] for r in records),
        incumbent_cpu_seconds=read(str(output/'summary.json'))['cpu_seconds'],
        reused_record=spec['reused_record'],full_predictive_integration_admitted=False,
        independent_confirmation=False,new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,scope=spec['scope'])
    save(output/'control_summary.json',result);print('SCHEDULEURM_DONE '+json.dumps(result),flush=True);return result


def refine(name,identity,action_index,kind,plan=PLAN):
    spec=read(plan);case=ROOT/spec['output']/name;path=case/'refined_parts'/f'{identity}_{action_index}_{kind}.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and read(str(path)).get('complete'):return read(str(path))
    started=time.process_time();bundle=compressed_read(BASE/'marginals'/(name+'.json.gz'))
    original=read(str(case/'summary.json'))
    assert (action_index,kind) in {(f['action'],f['kind']) for f in original['failed_action_checks']}
    block=bundle['blocks'][action_index];prior=field_from_snapshot(block['fields'][identity]);future=copy.deepcopy(prior)
    score=update_raw_observation(future,0,bundle['label'],None,1)
    assert abs(score-bundle['head_log_scores'][identity])<=1e-10
    _,rank=np.unique(prior.means,axis=0,return_inverse=True);rows={}
    for n,order,offset in [('H8',64,0),('I8',64,101),('J8',128,0)]:
        tick=time.process_time()
        s,c,d=component_risk_statistics(prior,future,block['action'],bundle['label'],score,bundle['protocol'],
            query=0,raw=kind=='raw',order=order,power=8,seed=bundle['seed']+offset+rank*104729,
            integrated_position=block['action']['incumbent_position'])
        rows[n]=dict(statistics=s.tolist(),cost=c,diagnostics=d,cpu_seconds=time.process_time()-tick)
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=len(rows),total=3,unit='local_rules',name=name)),flush=True)
    disagreement=read(str(case/'control_summary.json'))['cross_coordinate_checks']
    if any(c['coordinate']=='G7' and c['action']==action_index and c['kind']==kind
            and c['difference']>spec['maximum_action_resolution_difference'] for c in disagreement):
        tick=time.process_time();reference=shared_risk_reference(prior,future,block['action'],bundle['label'],score,
            bundle['protocol'],query=0,raw=kind=='raw',order=64,power=14,seed=bundle['seed']+7000001)
        s,c,d=component_control_risk_statistics(prior,future,block['action'],bundle['label'],score,bundle['protocol'],
            reference,query=0,raw=kind=='raw',order=64,power=7,seed=bundle['seed']+rank*104729)
        rows['K7']=dict(statistics=s.tolist(),cost=c,diagnostics=d,cpu_seconds=time.process_time()-tick)
    result=dict(complete=True,name=name,identity=identity,action=action_index,kind=kind,rows=rows,
        cpu_seconds=time.process_time()-started,new_physical_calls=0,new_source_fits=0,new_verifier_calls=0)
    save(path,result);print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True);return result


def refined_result(bundle,original,control_summary,parts,tolerance=1e-4):
    selected={(p['action'],p['kind']) for p in parts};tables={};costs={};values={}
    for n in ('G7','H8','I8','J8'):
        tables[n]={k:np.asarray(original['statistics']['G7'][k]).copy() for k in ('raw','projected')}
        costs[n]={k:np.asarray(original['costs']['G7'][k]).copy() for k in ('raw','projected')}
        if n!='G7':
            for p in parts:
                h,a,k=p['identity'],p['action'],p['kind'];tables[n][k][h,a]=p['rows'][n]['statistics'];costs[n][k][h,a]=p['rows'][n]['cost']
        values[n]={k:mixed_values(bundle,tables[n][k],costs[n][k],k) for k in ('raw','projected')}
    comparisons=[]
    for low,high in [('G7','H8'),('H8','I8'),('H8','J8')]:
        delta={k:abs(values[high][k]-values[low][k]) for k in ('raw','projected')}
        comparisons.append(dict(low=low,high=high,**{k:float(delta[k].max()) for k in delta},
            actions=[dict(action=a,kind=k,difference=float(delta[k][:,a].max())) for a,k in sorted(selected)]))
    cross=[]
    control_tables={k:np.asarray(control_summary['statistics']['E7'][k]).copy() for k in ('raw','projected')}
    control_costs={k:np.asarray(control_summary['costs']['E7'][k]).copy() for k in ('raw','projected')}
    for p in parts:
        if 'K7' not in p['rows']:continue
        h,a,k=p['identity'],p['action'],p['kind'];control_tables[k][h,a]=p['rows']['K7']['statistics'];control_costs[k][h,a]=p['rows']['K7']['cost']
    for a,k in sorted(selected):
        reference=mixed_values(bundle,control_tables[k],control_costs[k],k)
        order=64 if any(p['action']==a and p['kind']==k and 'K7' in p['rows'] for p in parts) else 32
        for n in ('H8','I8','J8'):
            cross.append(dict(coordinate=n,control_order=order,action=a,action_index=bundle['blocks'][a]['action_index'],
                kind=k,difference=float(abs(values[n][k][:,a]-reference[:,a]).max())))
    evidence=max(abs(r['diagnostics']['predictive_log_evidence_error']) for p in parts for r in p['rows'].values())
    unresolved=[f for f in original['failed_action_checks'] if (f['action'],f['kind']) not in selected]
    passed=evidence<=tolerance and not unresolved and all(c[k]<=tolerance for c in comparisons for k in ('raw','projected'))
    return dict(complete=all(p['complete'] for p in parts),name=bundle['name'],comparisons=comparisons,
        cross_coordinate_checks=cross,maximum_raw_log_evidence_error=evidence,
        refined_rule_precision_passed=passed,cross_coordinate_agreement_passed=all(c['difference']<=tolerance for c in cross),
        selected_action_precision_passed=passed and all(c['difference']<=tolerance for c in cross),
        remaining_incumbent_failures=unresolved,
        cpu_seconds=sum(p['cpu_seconds'] for p in parts),
        statistics={n:{k:v.tolist() for k,v in tab.items()} for n,tab in tables.items()},
        costs={n:{k:v.tolist() for k,v in tab.items()} for n,tab in costs.items()},
        full_predictive_integration_admitted=False,independent_confirmation=False,
        new_physical_calls=0,new_source_fits=0,new_verifier_calls=0)


def collect_refined(plan=PLAN):
    spec=read(plan);output=ROOT/spec['output'];records=[]
    for name in spec['records']:
        bundle=read(str(output/'mixing_inputs'/(name+'.json')));case=output/name;original=read(str(case/'summary.json'))
        selected=sorted({(f['action'],f['kind']) for f in original['failed_action_checks']})
        parts=[read(str(case/'refined_parts'/f'{h}_{a}_{k}.json')) for a,k in selected for h in bundle['utility_identities']]
        result=refined_result(bundle,original,read(str(case/'control_summary.json')),parts,spec['maximum_action_resolution_difference'])
        save(case/'refined_summary.json',result);records.append({k:v for k,v in result.items() if k not in ('statistics','costs')})
    result=dict(complete=all(r['complete'] for r in records),records=records,
        all_six_selected_action_precision_passed=all(r['selected_action_precision_passed'] for r in records)
            and read(str(OUT/'coordinate_summary.json'))['selected_action_precision_passed'],
        cpu_seconds=sum(r['cpu_seconds'] for r in records),full_predictive_integration_admitted=False,
        independent_confirmation=False,new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,scope=spec['scope'])
    save(output/'refined_summary.json',result);print('SCHEDULEURM_DONE '+json.dumps(result),flush=True);return result


def resolution_pairs(summary,tolerance):
    return sorted({(a['action'],a['kind']) for c in summary['comparisons'] if c['low']=='G7'
        for a in c['actions'] if a['difference']>tolerance})


def extend_resolution(name,identity,action_index,kind,plan=PLAN):
    spec=read(plan);case=ROOT/spec['output']/name;path=case/'resolution_parts'/f'{identity}_{action_index}_{kind}.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and read(str(path)).get('complete'):return read(str(path))
    assert (action_index,kind) in resolution_pairs(read(str(case/'refined_summary.json')),spec['maximum_action_resolution_difference'])
    started=time.process_time();bundle=compressed_read(BASE/'marginals'/(name+'.json.gz'));block=bundle['blocks'][action_index]
    prior=field_from_snapshot(block['fields'][identity]);future=copy.deepcopy(prior)
    score=update_raw_observation(future,0,bundle['label'],None,1)
    assert abs(score-bundle['head_log_scores'][identity])<=1e-10
    _,rank=np.unique(prior.means,axis=0,return_inverse=True)
    s,c,d=component_risk_statistics(prior,future,block['action'],bundle['label'],score,bundle['protocol'],
        query=0,raw=kind=='raw',order=64,power=9,seed=bundle['seed']+rank*104729,
        integrated_position=block['action']['incumbent_position'])
    result=dict(complete=True,name=name,identity=identity,action=action_index,kind=kind,
        statistics=s.tolist(),cost=c,diagnostics=d,cpu_seconds=time.process_time()-started,
        new_physical_calls=0,new_source_fits=0,new_verifier_calls=0)
    save(path,result);print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in result.items() if k not in ('statistics','diagnostics')}),flush=True);return result


def resolution_result(bundle,refined,parts,tolerance=1e-4):
    required=set(resolution_pairs(refined,tolerance));covered={(p['action'],p['kind']) for p in parts}
    statistics={k:np.asarray(refined['statistics']['H8'][k]).copy() for k in ('raw','projected')}
    costs={k:np.asarray(refined['costs']['H8'][k]).copy() for k in ('raw','projected')}
    for p in parts:
        h,a,k=p['identity'],p['action'],p['kind'];statistics[k][h,a]=p['statistics'];costs[k][h,a]=p['cost']
    checks=[]
    for a,k in sorted(covered):
        high=mixed_values(bundle,statistics[k],costs[k],k)
        low=mixed_values(bundle,refined['statistics']['H8'][k],refined['costs']['H8'][k],k)
        checks.append(dict(action=a,action_index=bundle['blocks'][a]['action_index'],kind=k,
            difference=float(abs(high[:,a]-low[:,a]).max())))
    evidence=max([refined['maximum_raw_log_evidence_error']]+[abs(p['diagnostics']['predictive_log_evidence_error']) for p in parts])
    stable=evidence<=tolerance and not refined['remaining_incumbent_failures'] and refined['cross_coordinate_agreement_passed']
    stable&=all(c[k]<=tolerance for c in refined['comparisons'] if c['low']=='H8' for k in ('raw','projected'))
    passed=stable and required<=covered and all(c['difference']<=tolerance for c in checks)
    return dict(complete=all(p['complete'] for p in parts),name=bundle['name'],
        selected_action_precision_passed=bool(passed),resolution_checks=checks,
        maximum_raw_log_evidence_error=evidence,cpu_seconds=sum(p['cpu_seconds'] for p in parts),
        candidate_rule='Retain GH64/256 on the seven refined action/kind pairs; GH64/512 is a resolution reference, not a changed decision.',
        original_128_to_256_precision_passed=refined['selected_action_precision_passed'],
        full_predictive_integration_admitted=False,independent_confirmation=False,
        new_physical_calls=0,new_source_fits=0,new_verifier_calls=0)


def collect_resolution(plan=PLAN):
    spec=read(plan);output=ROOT/spec['output'];records=[]
    for name in spec['records']:
        bundle=read(str(output/'mixing_inputs'/(name+'.json')));case=output/name;refined=read(str(case/'refined_summary.json'))
        parts=[read(str(case/'resolution_parts'/f'{h}_{a}_{k}.json')) for a,k in
            resolution_pairs(refined,spec['maximum_action_resolution_difference']) for h in bundle['utility_identities']]
        result=resolution_result(bundle,refined,parts,spec['maximum_action_resolution_difference']);save(case/'admitted_summary.json',result);records.append(result)
    result=dict(complete=all(r['complete'] for r in records),records=records,
        all_six_selected_action_precision_passed=all(r['selected_action_precision_passed'] for r in records)
            and read(str(OUT/'coordinate_summary.json'))['selected_action_precision_passed'],
        cpu_seconds=sum(r['cpu_seconds'] for r in records),full_predictive_integration_admitted=False,
        independent_confirmation=False,new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,scope=spec['scope'])
    save(output/'admitted_summary.json',result);print(json.dumps(result),flush=True);return result


def replay(plan=PLAN):
    """Replay the admission sequence without downloading covariances."""
    spec=read(plan);output=ROOT/spec['output'];maximum_difference=0.;status_matches=True
    for name in spec['records']:
        bundle=read(str(output/'mixing_inputs'/(name+'.json')));case=output/name
        saved=read(str(case/'summary.json'));baseline=dict(statistics=saved['statistics']['E7'],costs=saved['costs']['E7'])
        coordinate_parts=[read(str(case/'coordinate_parts'/f'{h}_{a}_{k}.json')) for h in bundle['utility_identities']
            for a,b in enumerate(bundle['blocks']) if b['action']['kind']=='guarded' for k in ('raw','projected')]
        initial=coordinate_result(bundle,coordinate_parts,baseline,spec['maximum_action_resolution_difference'])
        selected=sorted({(f['action'],f['kind']) for f in initial['failed_action_checks']})
        control_parts=[read(str(case/'parts'/f'{h}_{a}_{k}.json')) for a,k in selected for h in bundle['utility_identities']]
        corrected=control_result(bundle,initial,control_parts,spec['maximum_action_resolution_difference'])
        refined_parts=[read(str(case/'refined_parts'/f'{h}_{a}_{k}.json')) for a,k in selected for h in bundle['utility_identities']]
        refined=refined_result(bundle,initial,corrected,refined_parts,spec['maximum_action_resolution_difference'])
        resolution_parts=[read(str(case/'resolution_parts'/f'{h}_{a}_{k}.json')) for a,k in
            resolution_pairs(refined,spec['maximum_action_resolution_difference']) for h in bundle['utility_identities']]
        admitted=resolution_result(bundle,refined,resolution_parts,spec['maximum_action_resolution_difference'])
        for actual,filename in [(initial,'summary.json'),(corrected,'control_summary.json'),(refined,'refined_summary.json'),(admitted,'admitted_summary.json')]:
            expected=read(str(case/filename));status_matches&=actual['selected_action_precision_passed']==expected['selected_action_precision_passed']
            for a,b in zip(actual.get('comparisons',[]),expected.get('comparisons',[])):
                for k in ('raw','projected'):maximum_difference=max(maximum_difference,abs(a[k]-b[k]))
            for key in ('cross_coordinate_checks','resolution_checks'):
                for a,b in zip(actual.get(key,[]),expected.get(key,[])):
                    maximum_difference=max(maximum_difference,abs(a['difference']-b['difference']))
    result=dict(complete=True,records=len(spec['records']),maximum_difference=maximum_difference,
        admission_statuses_match=bool(status_matches),passed=bool(status_matches and maximum_difference<=1e-15),
        scope='Replays stored integration statistics and identity mixing; does not rerun continuous integrals.')
    save(output/'replay_results.json',result);print(json.dumps(result),flush=True);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['run','collect','control','collect-control','refine','collect-refined','extend-resolution','collect-resolution','replay'])
    parser.add_argument('--name');parser.add_argument('--identity',type=int)
    parser.add_argument('--action',type=int);parser.add_argument('--kind',choices=['raw','projected'])
    parser.add_argument('--plan',default=PLAN);args=parser.parse_args()
    if args.mode=='run':run_record(args.name,args.identity,args.kind,args.plan)
    elif args.mode=='collect':collect(args.plan)
    elif args.mode=='collect-control':collect_control(args.plan)
    elif args.mode=='refine':refine(args.name,args.identity,args.action,args.kind,args.plan)
    elif args.mode=='collect-refined':collect_refined(args.plan)
    elif args.mode=='extend-resolution':extend_resolution(args.name,args.identity,args.action,args.kind,args.plan)
    elif args.mode=='collect-resolution':collect_resolution(args.plan)
    elif args.mode=='replay':replay(args.plan)
    else:control(args.identity,args.action,args.kind,record=args.name,output=ROOT/read(args.plan)['output']/args.name)
