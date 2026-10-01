#!/usr/bin/env python3
"""Outcome-free constant design and exact finite-window cost diagnostic."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from core.profile_atlas import farthest_first_indices
from performance.inspect_public_battery_announced_library import decode_library
from performance.inspect_public_battery_announced_population import read, lines, key
from performance.inspect_public_battery_source_pilot import ordered_coordinates
from performance.replay_public_battery_target_search import evaluate, roster
from performance.run_public_battery_population_parallel import write_json

PROTOCOL=ROOT/'performance/manifests/public_battery_simple_controls_v1_20261002.json'
OUTPUT=ROOT/'paper_artifacts/public_battery_simple_controls_v1_20261002'


def constant_design(ids,nodes,pairs,n0):
    indices=[i for i,p in enumerate(ids) if p.startswith('constant_pair_')]
    constant_ids=[ids[i] for i in indices]
    coordinates=ordered_coordinates(nodes,pairs[indices])
    scale=coordinates.std(axis=0)
    z=(coordinates-coordinates.mean(axis=0))/np.where(scale>1e-10,scale,1.)
    medoid=int(np.argmin(np.linalg.norm(z[:,None]-z[None,:],axis=2).mean(axis=1)))
    chosen=farthest_first_indices(z,n0,initial_index=medoid)
    return constant_ids,[constant_ids[i] for i in chosen]


def enumerate_population(cells,ids,period,starts,rule,required):
    groups=[]
    for profile in ids:
        rows=[cells[(period,minute,profile,rule)] for minute in starts]
        successes=sum(r['window_success'] for r in rows)
        # Chance-feasible profiles need not have an unconditional full objective.
        groups.append(dict(profile_id=profile,successful_windows=successes,finite_population_feasible=successes>=required,
                           full_population_mean_cost_GBP=float(np.mean([r['cost_GBP'] for r in rows])) if successes==len(starts) else None))
    complete=[r for r in groups if r['full_population_mean_cost_GBP'] is not None]
    best=min(complete,key=lambda r:(r['full_population_mean_cost_GBP'],r['profile_id'])) if complete else None
    return dict(period=period,enumerated_profiles=len(ids),represented_window_queries=len(ids)*len(starts),
                feasible_candidates=sum(r['finite_population_feasible'] for r in groups),
                complete_cost_candidates=len(complete),selected_complete_cost_candidate=best,
                certification='exact census of fixed finite-window law; no statistical or calendar-period generalization',
                candidate_groups=groups)


def prepare(protocol,out):
    if (out/'plan.json').exists():raise ValueError('simple controls already frozen')
    source=read(ROOT/protocol['source_designs'])
    population=read(ROOT/protocol['population_protocol'])
    roster_rows,nodes,pairs,_=decode_library(read(ROOT/population['library_protocol']))
    ids=[r['profile_id'] for r in roster_rows]
    if ids!=source['full_library_profiles']:raise ValueError('library order changed')
    constants,initial=constant_design(ids,nodes,pairs,10)
    out.mkdir(parents=True,exist_ok=True)
    write_json(out,'designs.json',dict(constant_library_profiles=constants,constant_structural_initial=initial,
        source_first_center=source['source_first_center'],target_outcomes_used_for_constant_design=False,
        fixed_at_utc=datetime.now(timezone.utc).isoformat()))
    write_json(out,'plan.json',dict(tasks=roster(protocol),protocol_id=protocol['protocol_id']))


def run(protocol,out,index):
    task=read(out/'plan.json')['tasks'][index]
    folder=out/'tasks'/f'{index:03d}'
    if (folder/'result.json').exists():return
    frozen=read(out/'designs.json')
    original=read(ROOT/protocol['source_designs'])
    base=read(ROOT/protocol['replay_protocol'])
    source=dict(full_library_profiles=frozen['constant_library_profiles'] if task['arm']=='constant_structural_13' else original['full_library_profiles'],
                designs={task['arm']:frozen['constant_structural_initial'] if task['arm']=='constant_structural_13' else [frozen['source_first_center']]})
    base['search_budget']=13 if task['arm']=='constant_structural_13' else 1
    rows=lines(ROOT/base['population_outputs']/'cells.jsonl')
    cells={key(r):r for r in rows}
    if len(cells)!=68766 or len(cells)!=len(rows):raise ValueError('complete unique server population required')
    result=evaluate(task,base,source,cells)
    result['source_queries_paid_once']=44968 if task['arm']=='source_first_only' else 0
    folder.mkdir(parents=True,exist_ok=True)
    write_json(folder,'result.json',result)
    print(json.dumps(dict(index=index,certified=result['certified'],deployed=result['deployed_profile_id'])),flush=True)


def collect(protocol,out):
    tasks=read(out/'plan.json')['tasks']
    rows=[read(out/'tasks'/f'{i:03d}'/'result.json') for i in range(len(tasks))]
    if [tuple(r[k] for k in ('period','algorithm_seed','arm')) for r in rows]!=[tuple(r[k] for k in ('period','algorithm_seed','arm')) for r in tasks]:raise ValueError('paired controls missing or reordered')
    previous=read(ROOT/protocol['previous_results'])
    all_rows=previous+rows
    groups=[]
    for arm in ['source_complete_cost_best_z','standardized_generic_dct_maximin']+protocol['arms']:
        chosen=[r for r in all_rows if r['arm']==arm]
        paid=44968 if arm in ('source_complete_cost_best_z','source_first_only') else 0
        target=sum(np.mean([r['represented_search_queries']+r['represented_verification_queries'] for r in chosen if r['period']==day]) for day in protocol['periods'])
        certificates=sum(np.mean([r['certified_finite_population_feasible'] for r in chosen if r['period']==day]) for day in protocol['periods'])
        groups.append(dict(arm=arm,certified_runs=sum(r['certified'] for r in chosen),runs=len(chosen),false_certificates=sum(r['false_certificate'] for r in chosen),
            represented_target_queries=sum(r['represented_search_queries']+r['represented_verification_queries'] for r in chosen),
            expected_three_task_target_queries=float(target),expected_three_task_certificates=float(certificates),source_queries_paid_once=paid,
            all_in_queries_per_expected_certificate=float((paid+target)/certificates) if certificates else None))
    source=read(ROOT/protocol['source_designs'])
    base=read(ROOT/protocol['replay_protocol'])
    saved=lines(ROOT/base['population_outputs']/'cells.jsonl');cells={key(r):r for r in saved}
    if len(cells)!=68766 or len(cells)!=len(saved):raise ValueError('complete population census required')
    census=[enumerate_population(cells,source['full_library_profiles'],day,base['window_start_minutes'],base['dispatch_rule'],70) for day in protocol['periods']]
    constant_ids=read(out/'designs.json')['constant_library_profiles']
    constant_census=[enumerate_population(cells,constant_ids,day,base['window_start_minutes'],base['dispatch_rule'],70) for day in protocol['periods']]
    source_budget=next(g for g in groups if g['arm']=='source_complete_cost_best_z')
    result=dict(protocol_id=protocol['protocol_id'],status='simple_controls_and_cost_diagnostic_complete',new_runs=len(rows),groups=groups,
        full_library_census=census,constant_library_census=constant_census,
        full_census_queries_for_three_tasks=sum(r['represented_window_queries'] for r in census),
        constant_census_queries_for_three_tasks=sum(r['represented_window_queries'] for r in constant_census),
        source_all_in_queries_for_three_tasks=source_budget['source_queries_paid_once']+source_budget['expected_three_task_target_queries'],
        exhaustive_finite_population_fits_source_all_in_budget=sum(r['represented_window_queries'] for r in census)<=source_budget['source_queries_paid_once']+source_budget['expected_three_task_target_queries'],
        new_controller_window_evaluations=0,confirmation_year_access=False,independent_confirmation=False,source_comparison_gate='HOLD',
        decision='close_development_efficiency_claim; retain simple baselines before any separate confirmation protocol',limitations=protocol['limitations'])
    write_json(out,'rows.json',rows);write_json(out,'summary.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('full_library_census','constant_library_census')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['prepare','run','collect']);parser.add_argument('--index',type=int)
    args=parser.parse_args();protocol=read(PROTOCOL)
    if args.mode=='prepare':prepare(protocol,OUTPUT)
    elif args.mode=='run':run(protocol,OUTPUT,args.index)
    else:collect(protocol,OUTPUT)
