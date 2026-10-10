"""Refine the worst observed failure of each numerical comparison, without expanding the bank."""

import argparse
import json
import resource
import signal
import time

from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.diagnose_historical_component_guarded import compressed_read
from performance.validate_historical_component_bank import PLAN, run_group, group_checks

PAIRS=[('E5','E7'),('E7','F7'),('E7','G7')]
REFINED_PAIRS=[('G7','H8'),('H8','I8'),('H8','J8')]
RESOLUTION_PAIRS=[('H8','I8'),('H8','J8'),('H8','K9')]


def select_failures(summary,tolerance):
    selected={}
    for kind in ('raw','projected'):
        for low,high in PAIRS:
            candidates=[(c['maximum_difference'],g['group']) for g in summary['groups']
                for c in g['checks'][kind].get('comparisons',[]) if (c['low'],c['high'])==(low,high)]
            if not candidates:continue
            difference,group=max(candidates)
            if difference<=tolerance:continue
            row=selected.setdefault((kind,group),dict(group=group,kind=kind,reasons=[]))
            row['reasons'].append(dict(low=low,high=high,maximum_difference=difference))
    return list(selected.values())


def candidate_units(spec,identities):
    cache=ROOT/spec['allocation_cache'];unit={}
    for kind in ('raw','projected'):
        unit[kind]=max(sum(read(str(cache/('compact_parts' if kind=='raw' else 'parts')/f'{h}_{a}_{kind}.json'))
            ['rows']['candidate']['cpu_seconds' if kind=='raw' else 'shared_cpu_seconds'] for h in identities) for a in (0,1))
    return unit


def prepare():
    spec=read(PLAN);out=ROOT/spec['output'];summary=read(str(out/'screen_summary.json'))
    assert summary['complete'] and not summary['single_record_risk_bank_precision_passed']
    prep=read(str(out/'preparation.json'));identities=sorted(map(int,prep['source_components']))
    selected=select_failures(summary,spec['maximum_action_difference']);unit=candidate_units(spec,identities)
    estimate=5.*sum(unit[r['kind']] for r in selected)
    result=dict(record=spec['record'],identities=identities,selected=selected,rules=['H8','I8','J8'],
        estimated_cpu_seconds=estimate,estimated_cost_admitted=estimate<=spec['maximum_estimated_adaptive_cpu_seconds'],
        maximum_job_cpu_seconds=900,full_predictive_integration_admitted=False,independent_confirmation=False)
    (out/'failure_pilot/tasks').mkdir(parents=True,exist_ok=True)
    save(out/'failure_pilot/selection.json',result);print(json.dumps(result),flush=True);return result


def prepare_resolution():
    spec=read(PLAN);out=ROOT/spec['output'];plan=read(str(out/'failure_pilot/selection.json'))
    summary=read(str(out/'failure_pilot/summary.json'));assert summary['complete']
    failed={(r['kind'],r['group']) for r in summary['groups'] if not r['checks']['passed']}
    # K9 confirms H8; it cannot repair an already unstable H8 independent/GH rule.
    assert all(c['maximum_difference']<=spec['maximum_action_difference'] for r in summary['groups']
        if (r['kind'],r['group']) in failed for c in r['checks']['comparisons'][1:])
    selected=[i for i,r in enumerate(plan['selected']) if (r['kind'],r['group']) in failed]
    unit=candidate_units(spec,plan['identities']);estimate=2.5*sum(unit[plan['selected'][i]['kind']] for i in selected)
    result=dict(selected=selected,rules=['K9'],estimated_cpu_seconds=estimate,
        cumulative_estimated_cpu_seconds=plan['estimated_cpu_seconds']+estimate,
        estimated_cost_admitted=plan['estimated_cpu_seconds']+estimate<=spec['maximum_estimated_adaptive_cpu_seconds'])
    save(out/'failure_pilot/resolution_plan.json',result);print(json.dumps(result),flush=True);return result


def prepare_cheaper():
    spec=read(PLAN);out=ROOT/spec['output'];plan=read(str(out/'failure_pilot/selection.json'))
    summary=read(str(out/'failure_pilot/resolution_summary.json'))
    assert summary['complete'] and summary['selected_failure_precision_passed']
    resolution=read(str(out/'failure_pilot/resolution_plan.json'));unit=candidate_units(spec,plan['identities'])
    estimate=1.25*sum(unit[r['kind']] for r in plan['selected'])
    cumulative=resolution['cumulative_estimated_cpu_seconds']+estimate
    result=dict(rules=['E8','F8'],estimated_cpu_seconds=estimate,cumulative_estimated_cpu_seconds=cumulative,
        estimated_cost_admitted=cumulative<=spec['maximum_estimated_adaptive_cpu_seconds'])
    save(out/'failure_pilot/cheaper_plan.json',result);print(json.dumps(result),flush=True);return result


def run(identity,selection,stage='refine'):
    spec=read(PLAN);out=ROOT/spec['output'];plan=read(str(out/'failure_pilot/selection.json'))
    assert plan['estimated_cost_admitted'] and identity in plan['identities']
    (out/'failure_pilot/tasks').mkdir(parents=True,exist_ok=True)
    rules=plan['rules'];suffix=''
    if stage=='resolution':
        resolution=read(str(out/'failure_pilot/resolution_plan.json'))
        assert resolution['estimated_cost_admitted'] and selection in resolution['selected']
        rules=resolution['rules'];suffix='_resolution'
    elif stage=='cheaper':
        cheaper=read(str(out/'failure_pilot/cheaper_plan.json'));assert cheaper['estimated_cost_admitted']
        rules=cheaper['rules'];suffix='_cheaper'
    row=plan['selected'][selection];started=time.process_time();result=dict(identity=identity,selection=selection,
        group=row['group'],kind=row['kind'],stage=stage,complete=False)
    previous=signal.signal(signal.SIGPROF,cpu_timeout);signal.setitimer(signal.ITIMER_PROF,plan['maximum_job_cpu_seconds'])
    try:
        run_group(row['group'],identity,row['kind'],rules,spec);result['complete']=True
    except TimeoutError:result['cpu_budget_stopped']=True
    finally:
        signal.setitimer(signal.ITIMER_PROF,0.);signal.signal(signal.SIGPROF,previous)
        result.update(cpu_seconds=time.process_time()-started,peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            new_physical_calls=0,new_source_fits=0,new_verifier_calls=0)
        save(out/'failure_pilot/tasks'/f'{identity}_{selection}{suffix}.json',result)
    print('SCHEDULEURM_DONE '+json.dumps(result),flush=True);return result


def collect(stage='refine'):
    spec=read(PLAN);out=ROOT/spec['output'];plan=read(str(out/'failure_pilot/selection.json'));rows=[]
    selected=list(range(len(plan['selected'])));previous=None;suffix='';pairs=REFINED_PAIRS
    if stage=='resolution':
        selected=read(str(out/'failure_pilot/resolution_plan.json'))['selected']
        previous=read(str(out/'failure_pilot/summary.json'));suffix='_resolution';pairs=RESOLUTION_PAIRS
    elif stage=='cheaper':
        previous=read(str(out/'failure_pilot/resolution_summary.json'));suffix='_cheaper'
        assert previous['selected_failure_precision_passed']
        pairs=[('E8','F8'),('E8','H8'),('E8','I8'),('E8','J8')]
    tasks=[read(str(out/'failure_pilot/tasks'/f'{h}_{i}{suffix}.json')) for i in selected for h in plan['identities']]
    complete=all(t['complete'] for t in tasks)
    if complete:
        for i,row in enumerate(plan['selected']):
            if previous and i not in selected:
                rows.append(previous['groups'][i]);continue
            group,kind=row['group'],row['kind'];bundle=compressed_read(out/'marginals'/f'{group:03d}.json.gz')
            parts=[read(str(out/'parts'/f'{group:03d}_{h}_{kind}.json')) for h in plan['identities']]
            group_pairs=pairs+[('E8','K9')] if stage=='cheaper' and all('K9' in p['rows'] for p in parts) else pairs
            checks=group_checks(bundle,parts,kind,group_pairs,spec['maximum_action_difference'])
            rows.append(dict(**row,action_indices=bundle['action_indices'],checks=checks,
                cpu_seconds_by_rule={n:sum(p['rows'][n]['cpu_seconds'] for p in parts)
                    for n in set(v for pair in group_pairs for v in pair)}))
    prior_cpu=read(str(out/'failure_pilot/summary.json'))['cpu_seconds'] if previous else 0.
    if stage=='cheaper':prior_cpu+=previous['cpu_seconds']
    estimate=plan['estimated_cpu_seconds'];cumulative=estimate
    if stage!='refine':
        budget=read(str(out/'failure_pilot'/('resolution_plan.json' if stage=='resolution' else 'cheaper_plan.json')))
        estimate=budget['estimated_cpu_seconds'];cumulative=budget['cumulative_estimated_cpu_seconds']
    result=dict(complete=complete,record=plan['record'],stage=stage,groups=rows,
        selected_failure_precision_passed=complete and all(r['checks']['passed'] for r in rows),
        cpu_seconds=sum(t['cpu_seconds'] for t in tasks),estimated_cpu_seconds=estimate,cumulative_estimated_cpu_seconds=cumulative,
        total_cpu_seconds=sum(t['cpu_seconds'] for t in tasks)+prior_cpu,
        maximum_task_peak_rss_mb=max((t['peak_rss_mb'] for t in tasks),default=0.),
        full_predictive_integration_admitted=False,independent_confirmation=False,
        new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,
        scope='Worst observed failure of each comparison on one opened risk record; the full-bank screen and its failures remain unchanged.')
    if stage=='cheaper':result['previous_high_rule_precision_passed']=previous['selected_failure_precision_passed']
    save(out/'failure_pilot'/dict(refine='summary.json',resolution='resolution_summary.json',cheaper='cheaper_summary.json')[stage],result)
    print('SCHEDULEURM_DONE '+json.dumps(result),flush=True);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','prepare-resolution','prepare-cheaper','run','collect'])
    parser.add_argument('--stage',choices=['refine','resolution','cheaper'],default='refine')
    parser.add_argument('--identity',type=int);parser.add_argument('--selection',type=int);args=parser.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode=='prepare-resolution':prepare_resolution()
    elif args.mode=='prepare-cheaper':prepare_cheaper()
    elif args.mode=='run':run(args.identity,args.selection,args.stage)
    else:collect(args.stage)
