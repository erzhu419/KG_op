"""Dispatch the actual bank shards with at most eight live scheduler jobs."""

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CLI='/home/erzhu419/mine_code/scheduleurm/skill/scheduler.py'
DEPLOY='/home/erzhu419/mine_code/KG_op_independent_kg_parallel_deploy/SC-OLH-KG'
PYTHON='/home/zhengliang01/scheduleurm_work/conda_envs/scomp-py310/bin/python'
QUEUE=Path('/home/erzhu419/.claude/scheduler/queue.json')
REMOTE='/home/zhengliang01/scheduleurm_work/KG_op_independent_kg_parallel_deploy/SC-OLH-KG'
PILOT_MODES={'pilot':'refine','pilot-resolution':'resolution','pilot-cheaper':'cheaper'}


def launch(stage):
    spec=json.loads((ROOT/'performance/manifests/historical_component_bank_v1.json').read_text())
    out=ROOT/spec['output'];prep=json.loads((out/'preparation.json').read_text());assert prep['complete']
    identities=sorted(map(int,prep['source_components']))
    if stage in PILOT_MODES:
        out=out/'failure_pilot';pilot=json.loads((out/'selection.json').read_text());assert pilot['estimated_cost_admitted']
        selected=list(range(len(pilot['selected'])))
        if stage=='pilot-resolution':
            resolution=json.loads((out/'resolution_plan.json').read_text());assert resolution['estimated_cost_admitted']
            selected=resolution['selected']
        elif stage=='pilot-cheaper':assert json.loads((out/'cheaper_plan.json').read_text())['estimated_cost_admitted']
        pending=[dict(identity=h,kind=r['kind'],shard=i) for i,r in enumerate(pilot['selected']) if i in selected for h in identities]
    elif stage=='screen':counts=dict(raw=prep['guarded_groups']+prep['direct_groups'],projected=prep['guarded_groups'])
    elif stage=='adaptive':
        old=json.loads((out/'next_bank_plan.json').read_text());counts={k:len(v) for k,v in old['remaining_groups'].items()}
    elif stage=='adaptive-resolution':
        old=json.loads((out/'adaptive_summary.json').read_text());assert old['complete']
        counts={k:len(v) for k,v in old['failed_groups'].items()}
    else:
        old=json.loads((out/('screen_summary.json' if stage=='refine' else 'refined_summary.json')).read_text())
        assert old['complete'];counts={k:len(v) for k,v in old['failed_groups'].items()}
    if stage not in PILOT_MODES:
        pending=[dict(identity=h,kind=k,shard=s) for s in range(spec['shards_per_identity_kind'])
            for h in identities for k in ('raw','projected') if s<counts[k]]
    jobs=[];specs=[];last=None
    jobs_path=out/(stage+'_jobs.json');specs_path=out/(stage+'_specs.jsonl')
    if jobs_path.exists():
        jobs=json.loads(jobs_path.read_text());submitted={(j['identity'],j['kind'],j['shard']) for j in jobs}
        pending=[j for j in pending if (j['identity'],j['kind'],j['shard']) not in submitted]
        specs=[json.loads(line) for line in specs_path.read_text().splitlines()]
    while True:
        ids={j['id'] for j in jobs}
        state={t['id']:t for t in json.loads(QUEUE.read_text())['tasks'] if t.get('id') in ids}
        active=sum(t['status'] in ('queued','launching','running') for t in state.values())
        failures=[t['id'] for t in state.values() if t['status'] not in ('queued','launching','running','done')]
        if failures:raise RuntimeError('Scheduler tasks stopped: '+','.join(failures))
        slots=8-active
        if pending and slots:
            new_ids=[]
            for _ in range(min(slots,len(pending))):
                job=pending.pop(0);h,k,s=job['identity'],job['kind'],job['shard']
                worker=(['performance.diagnose_historical_bank_failures','run','--identity',str(h),'--selection',str(s),
                    '--stage',PILOT_MODES[stage]] if stage in PILOT_MODES
                    else ['performance.validate_historical_component_bank','run','--stage',stage,'--identity',str(h),'--kind',k,'--shard',str(s)])
                cmd=shlex.join(['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','PYTHONPATH=.',PYTHON,'-u','-m']+worker)
                args=[sys.executable,CLI,'submit','--description',f'KG one record bank {stage} H{h} {k} shard{s}',
                    '--cmd',cmd,'--cwd',DEPLOY,'--signature',f'KG_op/component_bank_v1/{stage}/{h}/{k}/{s}',
                    '--project','KG_op','--vram','0','--ram-mb','2000','--cpu','1','--priority','low',
                    '--skip-launch-staging','--env-spec','none']
                for node in range(1,7):args+=['--allowed-node',f'node{node:03d}']
                p=subprocess.run(args,text=True,capture_output=True)
                specs.append(dict(**job,command=cmd,args=args,stdout=p.stdout,stderr=p.stderr,returncode=p.returncode))
                specs_path.write_text(''.join(json.dumps(r)+'\n' for r in specs))
                if p.returncode:raise RuntimeError(p.stderr or p.stdout)
                job['id']=re.search(r'submitted (t\d+)',p.stdout).group(1);jobs.append(job);new_ids.append(job['id'])
                jobs_path.write_text(json.dumps(jobs,indent=2)+'\n')
                print('SUBMITTED '+json.dumps(job),flush=True)
            args=[sys.executable,CLI,'dispatch','--no-bulk-window','--lock-timeout','20']
            for tid in new_ids:args+=['--task-id',tid]
            p=subprocess.run(args,text=True,capture_output=True)
            with (out/(stage+'_dispatch.jsonl')).open('a') as f:f.write(json.dumps(dict(args=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))+'\n')
            continue
        progress=dict(stage=stage,submitted=len(jobs),pending=len(pending),live=active,
            done=sum(t['status']=='done' for t in state.values()))
        if progress!=last:print('PROGRESS '+json.dumps(progress),flush=True);last=progress
        if not pending and len(state)==len(jobs) and all(t['status']=='done' for t in state.values()):break
        time.sleep(20)
    print('COMPLETE '+json.dumps(progress),flush=True)


def collect_remote(stage,out):
    sys.path.insert(0,str(Path(CLI).parent))
    import scheduler
    worker=(['performance.diagnose_historical_bank_failures','collect','--stage',PILOT_MODES[stage]]
        if stage in PILOT_MODES
        else ['performance.validate_historical_component_bank','collect','--stage',stage])
    command='cd '+shlex.quote(REMOTE)+' && '+shlex.join(['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1',
        'PYTHONPATH=.',PYTHON,'-u','-m']+worker)
    rc,body,err=scheduler.run_on('node005',command,timeout=35,check=False)
    (out/(stage+'_collection.log')).write_text(body+err)
    if rc:raise RuntimeError(err or body)
    name={'screen':'screen_summary.json','refine':'refined_summary.json','resolution':'admitted_summary.json',
        'adaptive':'adaptive_summary.json','adaptive-resolution':'adaptive_resolution_summary.json',
        'pilot':'summary.json','pilot-resolution':'resolution_summary.json','pilot-cheaper':'cheaper_summary.json'}[stage]
    relative='paper_artifacts/historical_component_bank_v1/'+('failure_pilot/' if stage in PILOT_MODES else '')+name
    code='from pathlib import Path;print(Path('+repr(relative)+').read_text())'
    rc,body,err=scheduler.run_on('node005','cd '+shlex.quote(REMOTE)+' && '+shlex.join([PYTHON,'-c',code]),timeout=35,check=False)
    if rc:raise RuntimeError(err or body)
    result=json.loads(body);(out/name).write_text(json.dumps(result,indent=2)+'\n')
    print('COLLECTED '+json.dumps({k:v for k,v in result.items() if k!='groups'}),flush=True)
    return result


def continue_adaptive():
    spec=json.loads((ROOT/'performance/manifests/historical_component_bank_v1.json').read_text());out=ROOT/spec['output']
    plan=json.loads((out/'next_bank_plan.json').read_text())
    status=dict(state='running_adaptive',maximum_live_workers=8,
        estimated_initial_cpu_seconds=plan['estimated_initial_refinement_cpu_seconds'],
        maximum_estimated_resolution_cpu_seconds=plan['maximum_estimated_resolution_cpu_seconds'],
        full_predictive_integration_admitted=False,independent_confirmation=False)
    path=out/'adaptive_status.json';path.write_text(json.dumps(status,indent=2)+'\n')
    launch('adaptive');summary=collect_remote('adaptive',out)
    status['measured_cpu_seconds_by_stage']={'adaptive':summary['cpu_seconds']}
    estimate=sum(r['estimated_cpu_seconds'] for rows in summary['next_resolution_estimates'].values() for r in rows.values())
    status['estimated_resolution_cpu_seconds']=estimate
    if not summary['single_record_risk_bank_precision_passed']:
        if estimate<=plan['maximum_estimated_resolution_cpu_seconds']:
            status['state']='running_adaptive_resolution';path.write_text(json.dumps(status,indent=2)+'\n')
            launch('adaptive-resolution');summary=collect_remote('adaptive-resolution',out)
            status['measured_cpu_seconds_by_stage']['adaptive-resolution']=summary['cpu_seconds']
        else:status['state']='resolution_cost_hold'
    if summary['single_record_risk_bank_precision_passed']:status['state']='single_record_risk_precision_passed'
    elif status['state']!='resolution_cost_hold':status['state']='precision_hold'
    status.update(last_stage=summary['stage'],failed_groups=summary['failed_groups'],actions_checked=summary['actions_checked'],
        single_record_risk_bank_precision_passed=summary['single_record_risk_bank_precision_passed'],
        measured_cpu_seconds=sum(status['measured_cpu_seconds_by_stage'].values()))
    path.write_text(json.dumps(status,indent=2)+'\n');print('ADAPTIVE_DONE '+json.dumps(status),flush=True)


def continue_pipeline():
    spec=json.loads((ROOT/'performance/manifests/historical_component_bank_v1.json').read_text());out=ROOT/spec['output']
    prep=json.loads((out/'preparation.json').read_text())
    identities=sorted(map(int,prep['source_components']));expected=len(identities)*2*spec['shards_per_identity_kind']
    status=dict(state='waiting_for_screen',maximum_live_workers=8,estimated_adaptive_cpu_seconds=0.)
    status_path=out/'pipeline_status.json';status_path.write_text(json.dumps(status,indent=2)+'\n')
    while True:
        jobs=json.loads((out/'screen_jobs.json').read_text());ids={j['id'] for j in jobs}
        state={t['id']:t['status'] for t in json.loads(QUEUE.read_text())['tasks'] if t.get('id') in ids}
        if len(jobs)==expected and len(state)==expected and all(s=='done' for s in state.values()):break
        failures=[tid for tid,s in state.items() if s not in ('queued','launching','running','done')]
        if failures:raise RuntimeError('Screen scheduler tasks stopped: '+','.join(failures))
        time.sleep(20)
    summary=collect_remote('screen',out)
    status['measured_cpu_seconds_by_stage']={'preparation':prep['cpu_seconds'],'screen':summary['cpu_seconds']}
    # Use the actual matched high-rule timers, retaining the slower shortlist.
    # Five high-rule equivalents cover H8, I8, J8 and setup; 2.5 cover K9.
    cache=ROOT/spec['allocation_cache'];unit={}
    for kind in ('raw','projected'):
        unit[kind]=max(sum(json.loads((cache/('compact_parts' if kind=='raw' else 'parts')/
            f'{h}_{a}_{kind}.json').read_text())['rows']['candidate']['cpu_seconds' if kind=='raw' else 'shared_cpu_seconds']
            for h in identities) for a in (0,1))
    for stage,factor in [('refine',5.),('resolution',2.5)]:
        if summary['single_record_risk_bank_precision_passed']:break
        estimated=factor*sum(unit[k]*len(v) for k,v in summary['failed_groups'].items())
        if status['estimated_adaptive_cpu_seconds']+estimated>spec['maximum_estimated_adaptive_cpu_seconds']:
            status.update(state='adaptive_cost_hold',next_stage=stage,next_stage_estimated_cpu_seconds=estimated,
                failed_groups=summary['failed_groups']);break
        status.update(state='running_'+stage);status['estimated_adaptive_cpu_seconds']+=estimated
        status_path.write_text(json.dumps(status,indent=2)+'\n')
        launch(stage);summary=collect_remote(stage,out)
        status['measured_cpu_seconds_by_stage'][stage]=summary['cpu_seconds']
    else:status['state']='precision_hold'
    if summary['single_record_risk_bank_precision_passed']:status['state']='single_record_risk_precision_passed'
    elif status['state']!='adaptive_cost_hold':status['state']='precision_hold'
    status.update(last_stage=summary['stage'],single_record_risk_bank_precision_passed=summary['single_record_risk_bank_precision_passed'],
        measured_cpu_seconds=sum(status['measured_cpu_seconds_by_stage'].values()),
        full_predictive_integration_admitted=False,independent_confirmation=False)
    status_path.write_text(json.dumps(status,indent=2)+'\n');print('PIPELINE_DONE '+json.dumps(status),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['screen','refine','resolution','continue',
        'adaptive','adaptive-resolution','continue-adaptive']+list(PILOT_MODES))
    args=parser.parse_args()
    if args.stage=='continue':continue_pipeline()
    elif args.stage=='continue-adaptive':continue_adaptive()
    else:launch(args.stage)
