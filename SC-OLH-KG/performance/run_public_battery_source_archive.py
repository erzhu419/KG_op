#!/usr/bin/env python3
"""Complete fixed 2023 archive and freeze source-only first-center designs."""
import argparse
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from core.profile_atlas import farthest_first_indices, percentile_ranks
from performance.assess_public_battery_source_archive import assess
from performance import inspect_public_battery_announced_population as p
from performance import run_public_battery_population_parallel as par
from performance.inspect_public_battery_source_pilot import ordered_coordinates, pilot_work, source_windows

PROTOCOL=ROOT/"performance/manifests/public_battery_source_archive_population_v1_20261002.json"
OUTPUT=ROOT/"paper_artifacts/public_battery_source_archive_population_v1_20261002"


def inputs(protocol):
    pilot=p.read(ROOT/protocol["pilot_protocol"])
    data,contract,roster,nodes,pairs,windows=source_windows(pilot,ROOT/protocol["pilot_outputs"],
                                                        protocol["population"]["window_start_minutes"])
    return data,contract,[(r["profile_id"],r["family"],pair) for r,pair in zip(roster,pairs)],nodes,windows


def prepare(protocol,out,shards):
    campaign=out/"parallel"
    if (campaign/"work_plan.json").exists():
        raise ValueError("source archive already registered; use the saved shard plan")
    pilot_out=ROOT/protocol["pilot_outputs"]
    gate=p.read(pilot_out/"summary.json")
    if gate["status"]!="source_interface_pilot_complete" or gate["completed_controller_window_evaluations"]!=32:
        raise ValueError("complete bounded source prerequisite is required")
    if not p.read(ROOT/protocol["stopping_equivalence"])["gate_pass"]:
        raise ValueError("saved stopping-equivalence prerequisite is required")
    baseline=campaign/"baseline"
    baseline.mkdir(parents=True)
    pilot=p.read(ROOT/protocol["pilot_protocol"])
    rows=[]
    for i,identity in enumerate(pilot_work(pilot)):
        row=p.read(pilot_out/"pilot_cells"/f"{i:03d}"/"validated.json")
        if p.key(row)!=p.key(identity): raise ValueError("source pilot identity changed")
        rows.append({**row,"evaluation_origin":"reused_source_pilot","provenance":str(pilot_out/"pilot_cells"/f"{i:03d}"/"validated.json")})
    par.write_journal(baseline,"cells.jsonl",rows)
    _,_,profiles,_,_=inputs(protocol)
    work=par.remaining_work(protocol,profiles,{p.key(r):r for r in rows})
    if len(work)+len(rows)!=protocol["population"]["reported_unique_cells"]:
        raise ValueError("source work does not cover the original library exactly once")
    par.write_json(campaign,"work_items.json",work)
    par.write_json(campaign,"work_plan.json",dict(protocol_id=protocol["protocol_id"],shard_count=shards,
        shards=par.split_work(work,shards),baseline_assessed_unique_cells=len(rows),remaining_new_cells=len(work),
        registered_at_utc=datetime.now(timezone.utc).isoformat(),source_comparison_gate="HOLD",confirmation_year_access=False))
    print(json.dumps(dict(reused_pilot_cells=len(rows),new_cells=len(work),logical_shards=shards)),flush=True)


def load(protocol,out):
    campaign=out/"parallel"
    data,contract,profiles,nodes,windows=inputs(protocol)
    rows,_,_=par.saved_rows(campaign/"baseline")
    par.STATE.clear()
    par.STATE.update(campaign=campaign,protocol=protocol,data=data,contract=contract,profiles=profiles,nodes=nodes,
        windows=windows,baseline=rows,work=p.read(campaign/"work_items.json"),plan=p.read(campaign/"work_plan.json"))


def run(protocol,out,start,stop,workers):
    load(protocol,out)
    campaign=out/"parallel"
    if not 0<=start<stop<=par.STATE["plan"]["shard_count"]: raise ValueError("unregistered shard range")
    cpus=par.physical_cpu_ids()
    os.sched_setaffinity(0,set(cpus))
    workers=min(workers,len(cpus),stop-start)
    name=f"node_task_{start:05d}_{stop:05d}.json"
    status=dict(status="running",task_id=os.environ.get("SCHEDULEURM_TASK_ID"),workers=workers,
                shard_start=start,shard_stop=stop,started_at_utc=datetime.now(timezone.utc).isoformat())
    par.write_json(campaign,name,status)
    completed=0
    try:
        with mp.get_context("fork").Pool(workers) as pool:
            for _ in pool.imap_unordered(par.run_shard,range(start,stop),chunksize=1):
                completed+=1
                if completed%10==0 or completed==stop-start:
                    print(json.dumps(dict(completed_shards=completed,total_shards=stop-start)),flush=True)
        par.write_json(campaign,name,{**status,"status":"complete","completed_shards":completed,
                                    "finished_at_utc":datetime.now(timezone.utc).isoformat()})
    except Exception as error:
        par.write_json(campaign,name,{**status,"status":"discrepancy","error":str(error),"completed_shards":completed})
        raise


def designs(rows,profile_ids,nodes,pairs,periods,starts,rule,n0=10):
    """First center uses only fully supported source costs; geometry uses all profiles."""
    cells={p.key(r):r for r in rows}
    support=[profile for profile in profile_ids if all(cells[(day,minute,profile,rule)]["window_success"]
             for day in periods for minute in starts)]
    if not support:
        return dict(status="no_common_complete_source_cost_support",designs={},full_library_profiles=profile_ids)
    ranks={profile:0. for profile in support}
    cost_means={}
    for day in periods:
        means=np.array([np.mean([cells[(day,minute,profile,rule)]["cost_GBP"] for minute in starts]) for profile in support])
        if not np.all(np.isfinite(means)): raise ValueError("supported source costs are not finite")
        for profile,cost,rank in zip(support,means,percentile_ranks(means)):
            ranks[profile]+=float(rank)/len(periods)
            cost_means.setdefault(profile,{})[day]=float(cost)
    best=min(support,key=lambda profile:(ranks[profile],profile))
    coordinate=ordered_coordinates(nodes,pairs)
    scale=coordinate.std(axis=0)
    z=(coordinate-coordinate.mean(axis=0))/np.where(scale>1e-10,scale,1.)
    medoid=int(np.argmin(np.linalg.norm(z[:,None]-z[None,:],axis=2).mean(axis=1)))
    result={}
    for arm,first in (("source_complete_cost_best_z",profile_ids.index(best)),("standardized_generic_dct_maximin",medoid)):
        chosen=farthest_first_indices(z,n0,initial_index=first)
        result[arm]=[profile_ids[i] for i in chosen]
    return dict(status="source_first_center_designs_frozen",designs=result,source_first_center=best,
        structural_medoid=profile_ids[medoid],source_complete_cost_support=support,source_period_mean_cost_GBP=cost_means,
        source_mean_cost_percentile=ranks,coordinate_mean=coordinate.mean(axis=0).tolist(),coordinate_scale=scale.tolist(),
        structural_coordinate_dimension=int(z.shape[1]),full_library_profiles=profile_ids,
        source_selection_uses_target_outcomes=False,target_candidate_domain_pruned=False,
        rule="equal-period average percentile of full-window source mean cost among common complete source candidates",
        full_augmented_atlas_connected=False,source_comparison_gate="HOLD")


def collect(protocol,out,wait):
    campaign=out/"parallel"
    plan=p.read(campaign/"work_plan.json")
    while True:
        jobs=[p.read(path) for path in campaign.glob("node_task_*.json")]
        if any(j["status"]=="discrepancy" for j in jobs): raise ValueError("source node task reported a discrepancy")
        count=sum(1 for _ in (campaign/"shards").glob("*/summary.json"))
        if count==plan["shard_count"] and jobs and all(j["status"]=="complete" for j in jobs): break
        if not wait: raise ValueError(f"only {count}/{plan['shard_count']} source shards complete")
        time.sleep(10)
    rows,replays,executions=par.saved_rows(campaign/"baseline")
    failures,seen=[],set()
    work=p.read(campaign/"work_items.json")
    for shard in plan["shards"]:
        par.merge_shard(campaign,shard["shard_index"],work[shard["start"]:shard["stop"]],rows,replays,executions,failures,seen)
    if len(seen)!=len(work) or len(rows)!=protocol["population"]["reported_unique_cells"] or replays:
        raise ValueError("source archive coverage or replay count changed")
    data,contract,profiles,nodes,windows=inputs(protocol)
    ids=[r[0] for r in profiles]
    records=sorted(rows.values(),key=lambda r:(ids.index(r["profile_id"]),r["period"],r["window_start_minute"],r["dispatch_rule"]))
    for name,saved in (("cells.jsonl",records),("executions.jsonl",executions),("first_failures.jsonl",failures)):
        par.write_journal(out,name,saved)
    population=protocol["population"]
    assessment=assess(records,ids,[],population["development_samples"],population["window_start_minutes"],
                      population["dispatch_rules"],70)
    par.write_json(out,"population_support.json",assessment)
    selected=designs(records,ids,nodes,np.array([r[2] for r in profiles]),population["development_samples"],
                     population["window_start_minutes"],protocol["selection"]["dispatch_rule"])
    selected.update(protocol_id=protocol["protocol_id"],frozen_at_utc=datetime.now(timezone.utc).isoformat())
    par.write_json(out,"source_designs.json",selected)
    pilot=p.read(ROOT/protocol["pilot_outputs"]/"summary.json")
    summary=dict(protocol_id=protocol["protocol_id"],status="source_archive_complete",registered_unique_cells=len(rows),
        completed_unique_cells=len(rows),new_controller_window_evaluations=len(executions),reused_source_pilot_cells=32,
        cumulative_source_controller_window_evaluations=len(executions)+32,
        new_execution_calls=p.call_totals(executions),reused_pilot_calls=pilot["calls"],
        known_full_costs=sum(r["cost_GBP"] is not None for r in records),undefined_failed_full_costs=sum(not r["window_success"] for r in records),
        reliable_joint_candidates=assessment["reliable_joint_candidates"],complete_cost_support_candidates=assessment["full_cost_support_candidates"],
        source_design_status=selected["status"],designs=selected["designs"],logical_shards=plan["shard_count"],node_tasks=jobs,
        source_comparison_gate="HOLD",algorithm_optimizer_calls=0,terminal_verifier_calls=0,confirmation_year_access=False,
        new_public_requests=0,decision="register_retrospective_development_comparison_of_frozen_source_first_center_and_identical_structural_geometry",
        limitations=protocol["limitations"])
    par.write_json(out,"summary.json",summary)
    par.write_json(campaign,"collection_status.json",dict(status="complete",merged_new_cells=len(seen)))
    print(json.dumps({k:v for k,v in summary.items() if k not in ("node_tasks","reliable_joint_candidates","complete_cost_support_candidates")}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol",type=Path,default=PROTOCOL)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    modes=parser.add_subparsers(dest="mode",required=True)
    prep=modes.add_parser("prepare");prep.add_argument("--shards",type=int,default=4096)
    worker=modes.add_parser("run")
    worker.add_argument("--shard-start",type=int,required=True);worker.add_argument("--shard-stop",type=int,required=True)
    worker.add_argument("--workers",type=int,required=True)
    aggregator=modes.add_parser("collect");aggregator.add_argument("--wait",action="store_true")
    args=parser.parse_args();protocol=p.read(args.protocol)
    if args.mode=="prepare":prepare(protocol,args.output,args.shards)
    elif args.mode=="run":run(protocol,args.output,args.shard_start,args.shard_stop,args.workers)
    else:collect(protocol,args.output,args.wait)
