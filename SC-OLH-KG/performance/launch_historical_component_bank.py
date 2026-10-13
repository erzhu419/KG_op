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
TRANSFER_STAGES={'transfer-screen':'screen','transfer-adaptive':'adaptive','transfer-resolution':'resolution'}
BANK_STAGES={'label-bank-coarse':'coarse','label-bank-fine':'fine','label-bank-independent':'independent',
    'label-bank-gh-pilot':'gh_pilot','label-bank-gh-full':'gh_full','cdf-bank-coarse':'coarse'}
LABEL_MODES={'label-control':('label_control','resolution'),
    'label-groups':('label_groups','resolution'),'label-groups-refine':('label_groups','adaptive'),
    'label-groups-direct':('label_groups','direct'),'cdf-bank-pilot':('cdf_bank','pilot'),
    'body-reference-pilot':('body_reference_pilot','resolution'),
    'nearby-anchor':('nearby_anchor','resolution'),'risk-competition':('risk_competition','resolution'),
    'risk-competition-batch':('risk_competition_batch','resolution')}
UPPER_MODES={'upper-decisions-pilot':'pilot','upper-decisions-full':'full'}


def refresh_jobs(ids):
    """Refresh these workers without waiting for a whole-cluster status scan."""
    sys.path.insert(0,str(Path(CLI).parent))
    import scheduler
    snapshot=json.loads(QUEUE.read_text())
    snapshot['tasks']=[t for t in snapshot['tasks'] if t['id'] in ids and t['status']=='running']
    if not snapshot['tasks']:return
    owned={t['id'] for t in snapshot['tasks']}
    results=scheduler._running_probe_snapshot_deps().backend_batch_probe(snapshot)
    diagnostics=scheduler._terminal_diagnostics_snapshot_outside_lock(results,owned,'KG:task-health')
    try:
        with scheduler.state_lock(timeout_s=20,purpose='KG:task-health'):
            state=scheduler.load_state()
            scheduler._batch_check_running(state,probe_results=results,probed_task_ids=owned,terminal_diagnostics=diagnostics)
            scheduler.save_state(state)
    except scheduler.SchedulerLockTimeout:
        print('TASK_HEALTH_DEFERRED scheduler state lock busy; retaining recorded worker states',flush=True)


def launch(stage,plan=None):
    manifest=Path(plan).name if plan else ('historical_upper_decision_refs_v1.json' if stage in UPPER_MODES else
        'historical_cdf_bank_v1.json' if stage=='cdf-bank-coarse' else 'historical_label_bank_v1.json' if stage in BANK_STAGES else
        'historical_'+LABEL_MODES[stage][0]+'_v1.json' if stage in LABEL_MODES else
        'historical_component_transfer_v1.json' if stage in TRANSFER_STAGES else 'historical_component_bank_v1.json')
    spec=json.loads((ROOT/'performance/manifests'/manifest).read_text())
    out=ROOT/spec['output'];prep=json.loads((out/'preparation.json').read_text());assert prep['complete']
    identities=sorted(map(int,prep['source_components']))
    if stage in ('observed-batch-risk','positive-risk-grid'):
        assert prep['estimated_cost_admitted']
        pending=prep['jobs'].copy()
    elif stage in UPPER_MODES:
        short=UPPER_MODES[stage];plan=json.loads((out/(short+'_plan.json')).read_text());assert plan['estimated_cost_admitted']
        pending=[dict(identity=j['identity'],kind='raw',shard=i) for i,j in enumerate(plan['jobs'])]
    elif stage in TRANSFER_STAGES or stage in LABEL_MODES or stage in BANK_STAGES:
        short=(BANK_STAGES[stage] if stage in BANK_STAGES else
            LABEL_MODES[stage][1] if stage in LABEL_MODES else TRANSFER_STAGES[stage])
        plan=json.loads((out/(short+'_plan.json')).read_text());assert plan['estimated_cost_admitted']
        counts={k:sum(r['kind']==k for r in plan['selected']) for k in ('raw','projected')}
    elif stage in PILOT_MODES:
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
    elif stage=='bank-resolution':
        plan=json.loads((out/'bank_resolution_plan.json').read_text());assert plan['estimated_cost_admitted']
        counts={k:sum(r['kind']==k for r in plan['selected']) for k in ('raw','projected')}
    elif stage=='bank-coarse':
        plan=json.loads((out/'bank_coarse_plan.json').read_text());assert plan['estimated_cost_admitted']
        counts={k:len(v) for k,v in plan['selected_groups'].items()}
    elif stage in ('bank-tail','bank-tail-resolution'):
        plan=json.loads((out/(stage.replace('-','_')+'_plan.json')).read_text());assert plan['estimated_cost_admitted']
        counts={k:sum(r['kind']==k for r in plan['selected']) for k in ('raw','projected')}
    else:
        old=json.loads((out/('screen_summary.json' if stage=='refine' else 'refined_summary.json')).read_text())
        assert old['complete'];counts={k:len(v) for k,v in old['failed_groups'].items()}
    if stage not in PILOT_MODES and stage not in UPPER_MODES and stage not in ('observed-batch-risk','positive-risk-grid'):
        shards=max(counts.values()) if stage in TRANSFER_STAGES or stage in LABEL_MODES else spec['shards_per_identity_kind']
        pending=[dict(identity=h,kind=k,shard=s) for s in range(shards)
            for h in identities for k in ('raw','projected') if s<counts[k]]
    jobs=[];specs=[];last=None
    jobs_path=out/(stage+'_jobs.json');specs_path=out/(stage+'_specs.jsonl')
    if jobs_path.exists():
        jobs=json.loads(jobs_path.read_text());submitted={(j['identity'],j['kind'],j['shard']) for j in jobs}
        pending=[j for j in pending if (j['identity'],j['kind'],j['shard']) not in submitted]
        specs=[json.loads(line) for line in specs_path.read_text().splitlines()]
    while True:
        ids={j['id'] for j in jobs}
        if stage in ('observed-batch-risk','positive-risk-grid') and ids:refresh_jobs(ids)
        state={t['id']:t for t in json.loads(QUEUE.read_text())['tasks'] if t.get('id') in ids}
        active=sum(t['status'] in ('queued','launching','running') for t in state.values())
        failures=[t['id'] for t in state.values() if t['status'] not in ('queued','launching','running','done')]
        if failures:raise RuntimeError('Scheduler tasks stopped: '+','.join(failures))
        slots=8-active
        if pending and slots:
            new_ids=[]
            for _ in range(min(slots,len(pending))):
                job=pending.pop(0);h,k,s=job['identity'],job['kind'],job['shard']
                worker=((['performance.screen_historical_observed_batch_risk','run','--identity',str(h),'--shard',str(s),
                    '--plan','performance/manifests/'+manifest] if k=='screen' else
                    ['performance.screen_historical_observed_batch_risk','native-direct','--shard',str(s),
                    '--plan','performance/manifests/'+manifest]) if stage=='positive-risk-grid' else
                    ['performance.check_historical_observed_batch_risk','run','--record',job['record'],
                    '--identity',str(h),'--kind',k,'--plan','performance/manifests/'+manifest] if stage=='observed-batch-risk' else
                    ['performance.validate_historical_upper_decisions','run','--phase',short,'--job',str(s)] if stage in UPPER_MODES else
                    ['performance.validate_historical_label_bank','run','--identity',str(h),'--kind',k,
                    '--shard',str(s),'--stage',short,'--plan','performance/manifests/'+manifest] if stage in BANK_STAGES else
                    ['performance.validate_historical_component_transfer','run','--identity',str(h),'--kind',k,
                    '--shard',str(s),'--stage',short,'--plan','performance/manifests/'+manifest]
                    if stage in TRANSFER_STAGES or stage in LABEL_MODES else
                    ['performance.diagnose_historical_bank_failures','run','--identity',str(h),'--selection',str(s),
                    '--stage',PILOT_MODES[stage]] if stage in PILOT_MODES
                    else ['performance.validate_historical_component_bank','run','--stage',stage,'--identity',str(h),'--kind',k,'--shard',str(s)])
                cmd=shlex.join(['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','PYTHONPATH=.',PYTHON,'-u','-m']+worker)
                args=[sys.executable,CLI,'submit','--description',f'KG risk integration {stage} H{h} {k} shard{s}',
                    '--cmd',cmd,'--cwd',DEPLOY,'--signature',
                    (f'KG_op/{Path(manifest).stem}/{stage}/{h}/{k}/{s}' if stage in ('observed-batch-risk','positive-risk-grid')
                        else f'KG_op/component_bank_v1/{stage}/{h}/{k}/{s}'),
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


def collect_remote(stage,out,plan=None):
    out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(Path(CLI).parent))
    import scheduler
    worker=(['performance.positive_historical_direct_references','collect','--plan',plan] if stage=='positive-direct-reference' else
        ['performance.check_historical_observed_batch_outer','collect','--plan',plan] if stage=='observed-batch-outer' else
        ['performance.assemble_historical_observed_batch_references','--plan',plan] if stage=='observed-batch-assembly' else
        ['performance.screen_historical_observed_batch_risk','next-plan','--plan',plan] if stage=='observed-batch-next-plan' else
        ['performance.screen_historical_observed_batch_risk','collect','--plan',plan] if stage=='observed-batch-bound' else
        ['performance.check_historical_observed_batch_risk','collect','--plan',plan] if stage=='observed-batch-risk' else
        ['performance.validate_historical_label_groups','collect','--stage','resolution',
        '--plan','performance/manifests/historical_body_reference_pilot_v1.json'] if stage=='body-reference-pilot' else
        ['performance.validate_historical_risk_competition','collect',
            '--plan',plan or 'performance/manifests/historical_risk_competition_batch_v1.json'] if stage=='risk-competition-batch' else
        ['performance.validate_historical_upper_decisions','collect','--phase',UPPER_MODES[stage]] if stage in UPPER_MODES else
        ['performance.probe_historical_label_gh','collect','--stage',BANK_STAGES[stage]] if stage in ('label-bank-gh-pilot','label-bank-gh-full') else
        ['performance.validate_historical_label_bank','collect','--stage',BANK_STAGES[stage],
            '--plan','performance/manifests/historical_cdf_bank_v1.json' if stage=='cdf-bank-coarse' else 'performance/manifests/historical_label_bank_v1.json'] if stage in BANK_STAGES else
        ['performance.validate_historical_'+LABEL_MODES[stage][0],'collect','--stage',LABEL_MODES[stage][1]]
        if stage in ('label-groups-refine','label-groups-direct') else
        ['performance.validate_historical_'+LABEL_MODES[stage][0],'collect'] if stage in LABEL_MODES else
        ['performance.validate_historical_component_transfer','collect','--stage',TRANSFER_STAGES[stage]]
        if stage in TRANSFER_STAGES else ['performance.diagnose_historical_bank_failures','collect','--stage',PILOT_MODES[stage]]
        if stage in PILOT_MODES
        else ['performance.validate_historical_component_bank','collect','--stage',stage])
    command='cd '+shlex.quote(REMOTE)+' && '+shlex.join(['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1',
        'PYTHONPATH=.',PYTHON,'-u','-m']+worker)
    # The 224-point all-action collection takes about34 CPU seconds before
    # interpreter startup and transport; its CPU limit remains240 seconds.
    rc,body,err=scheduler.run_on('node005',command,
        timeout=60 if stage=='observed-batch-bound' else 35,check=False)
    (out/(stage+'_collection.log')).write_text(body+err)
    if rc:raise RuntimeError(err or body)
    name='next_cost_plan.json' if stage=='observed-batch-next-plan' else 'compact_summary.json' if stage in ('positive-direct-reference','observed-batch-risk','observed-batch-bound','observed-batch-assembly','observed-batch-outer') else UPPER_MODES[stage]+'_summary.json' if stage in UPPER_MODES else BANK_STAGES[stage]+'_summary.json' if stage in BANK_STAGES else 'summary.json' if stage in LABEL_MODES else TRANSFER_STAGES[stage]+'_summary.json' if stage in TRANSFER_STAGES else {
        'screen':'screen_summary.json','refine':'refined_summary.json','resolution':'admitted_summary.json',
        'adaptive':'adaptive_summary.json','adaptive-resolution':'adaptive_resolution_summary.json',
        'bank-resolution':'bank_resolution_summary.json',
        'bank-coarse':'bank_coarse_summary.json',
        'bank-tail':'bank_tail_summary.json',
        'bank-tail-resolution':'bank_tail_resolution_summary.json',
        'pilot':'summary.json','pilot-resolution':'resolution_summary.json','pilot-cheaper':'cheaper_summary.json'}[stage]
    relative=(str((out/name).relative_to(ROOT)) if stage in ('positive-direct-reference','observed-batch-risk','observed-batch-bound','observed-batch-next-plan','observed-batch-assembly','observed-batch-outer') or stage in UPPER_MODES or stage in TRANSFER_STAGES or stage in LABEL_MODES or stage in BANK_STAGES else
        'paper_artifacts/historical_component_bank_v1/'+('failure_pilot/' if stage in PILOT_MODES else '')+name)
    code='from pathlib import Path;print(Path('+repr(relative)+').read_text())'
    rc,body,err=scheduler.run_on('node005','cd '+shlex.quote(REMOTE)+' && '+shlex.join([PYTHON,'-c',code]),timeout=35,check=False)
    if rc:raise RuntimeError(err or body)
    result=json.loads(body);(out/name).write_text(json.dumps(result,indent=2)+'\n')
    if stage in TRANSFER_STAGES and result['next_plan'] is not None:
        stages=list(TRANSFER_STAGES);next_stage=TRANSFER_STAGES[stages[stages.index(stage)+1]]
        (out/(next_stage+'_plan.json')).write_text(json.dumps(result['next_plan'],indent=2)+'\n')
    if stage in ('adaptive','bank-resolution','bank-coarse','bank-tail','bank-tail-resolution'):
        from performance.validate_historical_component_bank import save_progress
        save_progress(result,out)
    print('COLLECTED '+json.dumps({k:v for k,v in result.items() if k not in ('groups','checks','records')}),flush=True)
    return result


def continue_transfer():
    spec=json.loads((ROOT/'performance/manifests/historical_component_transfer_v1.json').read_text())
    out=ROOT/spec['output'];path=out/'status.json';measured={}
    status=dict(state='running',maximum_live_workers=8,full_predictive_integration_admitted=False,independent_confirmation=False)
    for stage,short in TRANSFER_STAGES.items():
        plan=json.loads((out/(short+'_plan.json')).read_text())
        if not plan['estimated_cost_admitted']:
            status.update(state='cost_hold',cumulative_estimated_cpu_seconds=plan['cumulative_estimated_cpu_seconds']);break
        status.update(state='running_'+short,cumulative_estimated_cpu_seconds=plan['cumulative_estimated_cpu_seconds'])
        path.write_text(json.dumps(status,indent=2)+'\n')
        launch(stage);summary=collect_remote(stage,out);measured[short]=summary['cpu_seconds']
        status.update(last_stage=short,failed_groups=summary['failed_groups'],
            selected_action_record_checks_passed=summary['selected_action_record_checks_passed'],
            all_selected_action_precision_passed=summary['all_selected_action_precision_passed'])
        if summary['all_selected_action_precision_passed']:
            status['state']='selected_action_precision_passed';break
        if summary['next_plan'] is None:status['state']='precision_hold';break
    status.update(measured_cpu_seconds_by_stage=measured,measured_cpu_seconds=sum(measured.values()))
    path.write_text(json.dumps(status,indent=2)+'\n');print('TRANSFER_DONE '+json.dumps(status),flush=True)


def continue_label_bank():
    spec=json.loads((ROOT/'performance/manifests/historical_label_bank_v1.json').read_text())
    out=ROOT/spec['output'];status=dict(state='running',maximum_live_workers=8,stages={})
    for stage in ('label-bank-coarse','label-bank-fine'):
        short=BANK_STAGES[stage]
        plan=json.loads((out/(short+'_plan.json')).read_text())
        if not plan['estimated_cost_admitted']:
            status.update(state='cost_hold',held_stage=short);break
        status['state']='running_'+short;(out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
        launch(stage);summary=collect_remote(stage,out);status['stages'][short]=summary
        if not summary['complete']:status['state']='incomplete';break
        status['state']='collected_'+short
    status.update(full_predictive_integration_admitted=False,independent_confirmation=False)
    (out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    print('LABEL_BANK_DONE '+json.dumps(status),flush=True)


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


def continue_resolution_pilot():
    spec=json.loads((ROOT/'performance/manifests/historical_component_bank_v1.json').read_text());out=ROOT/spec['output']
    plan=json.loads((out/'bank_resolution_plan.json').read_text());assert plan['estimated_cost_admitted']
    path=out/'bank_resolution_status.json'
    status=dict(state='running',estimated_cpu_seconds=plan['estimated_cpu_seconds'],maximum_live_workers=8,
        full_predictive_integration_admitted=False,independent_confirmation=False)
    path.write_text(json.dumps(status,indent=2)+'\n')
    launch('bank-resolution');summary=collect_remote('bank-resolution',out)
    passed=summary['complete'] and all(summary['groups'][r['group']]['checks'][r['kind']]['passed'] for r in plan['selected'])
    status.update(state='selected_resolution_passed' if passed else 'precision_hold',
        selected_resolution_precision_passed=passed,actions_checked=summary['actions_checked'],
        failed_groups=summary['failed_groups'],measured_cpu_seconds=summary['cpu_seconds'],
        single_record_risk_bank_precision_passed=summary['single_record_risk_bank_precision_passed'])
    path.write_text(json.dumps(status,indent=2)+'\n');print('BANK_RESOLUTION_DONE '+json.dumps(status),flush=True)


def continue_coarse_resolution():
    from performance.validate_historical_component_bank import adaptive_resolution_rules
    spec=json.loads((ROOT/'performance/manifests/historical_component_bank_v1.json').read_text());out=ROOT/spec['output']
    while json.loads((out/'bank_resolution_status.json').read_text())['state']=='running':time.sleep(20)
    summary=json.loads((out/'bank_resolution_summary.json').read_text())
    pilot=json.loads((out/'bank_resolution_plan.json').read_text())
    # Both observed worst coarse failures must pass the GH64/512 check first.
    admitted=summary['complete'] and all(summary['groups'][r['group']]['checks'][r['kind']]['passed']
        for r in pilot['selected'] if r['rules']==['K9'])
    selected={k:[g['group'] for g in summary['groups'] if not g['checks'][k]['passed']
        and adaptive_resolution_rules(g['checks'][k],spec['maximum_action_difference'])==['K9']]
        for k in ('raw','projected')}
    initial=json.loads((out/'adaptive_summary.json').read_text())
    estimate=.5*sum(initial['next_resolution_estimates'][k][str(g)]['estimated_cpu_seconds'] for k,ids in selected.items() for g in ids)
    total=pilot['estimated_cpu_seconds']+estimate
    plan=dict(selected_groups=selected,rules=['E9'],estimated_cpu_seconds=estimate,
        cumulative_estimated_cpu_seconds=total,maximum_estimated_cpu_seconds=pilot['maximum_estimated_cpu_seconds'],
        estimated_cost_admitted=admitted and total<=pilot['maximum_estimated_cpu_seconds'],
        scope='Only coarse-only failures: keep E8/F8 independent and E8/H8 GH checks, add E8/E9 continuous resolution. Worst coarse groups already checked against GH64/512.')
    (out/'bank_coarse_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    path=out/'bank_coarse_status.json'
    status=dict(state='running' if plan['estimated_cost_admitted'] else 'coarse_or_cost_hold',
        cumulative_estimated_cpu_seconds=total,maximum_live_workers=8,
        full_predictive_integration_admitted=False,independent_confirmation=False)
    path.write_text(json.dumps(status,indent=2)+'\n')
    if plan['estimated_cost_admitted']:
        launch('bank-coarse');summary=collect_remote('bank-coarse',out)
        status.update(state='collected',actions_checked=summary['actions_checked'],failed_groups=summary['failed_groups'],
            measured_cpu_seconds=summary['cpu_seconds'],single_record_risk_bank_precision_passed=summary['single_record_risk_bank_precision_passed'])
        path.write_text(json.dumps(status,indent=2)+'\n')
    print('BANK_COARSE_DONE '+json.dumps(status),flush=True)


def continue_tail(stage='bank-tail'):
    spec=json.loads((ROOT/'performance/manifests/historical_component_bank_v1.json').read_text());out=ROOT/spec['output']
    prefix=stage.replace('-','_')
    plan=json.loads((out/(prefix+'_plan.json')).read_text());assert plan['estimated_cost_admitted']
    path=out/(prefix+'_status.json')
    status=dict(state='running',estimated_cpu_seconds=plan['estimated_cpu_seconds'],maximum_live_workers=8,
        full_predictive_integration_admitted=False,independent_confirmation=False)
    path.write_text(json.dumps(status,indent=2)+'\n')
    launch(stage);summary=collect_remote(stage,out)
    if stage=='bank-tail-resolution':
        status.update(cumulative_estimated_cpu_seconds=plan['cumulative_estimated_cpu_seconds'],
            cumulative_measured_cpu_seconds=json.loads((out/'bank_tail_summary.json').read_text())['cpu_seconds']+summary['cpu_seconds'])
    status.update(state='single_record_risk_precision_passed' if summary['single_record_risk_bank_precision_passed'] else 'precision_hold',
        actions_checked=summary['actions_checked'],failed_groups=summary['failed_groups'],measured_cpu_seconds=summary['cpu_seconds'],
        single_record_risk_bank_precision_passed=summary['single_record_risk_bank_precision_passed'])
    path.write_text(json.dumps(status,indent=2)+'\n');print('BANK_TAIL_DONE '+json.dumps(status),flush=True)


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


def selected_reference_decisions(records, groups):
    rmap = {r['record']:r for r in records}; decisions = []
    for value in groups:
        row = {k:value[k] for k in ('record','kind','group')}
        row['admitted'] = value['hybrid_direct_admitted']
        if value['record'] not in rmap:
            # The final32-point grid does not contain the distinct16-point nodes.
            assert 176 <= int(value['record'].rsplit('_p',1)[1]) < 192
            row['outer_body_order'] = 16
        elif row['admitted']:
            group = next(g for g in rmap[value['record']][value['kind']+'_referenced_competitors'] if g['group']==value['group'])
            row.update(previous_minimum_gap=value['minimum_cached_gap'],**group)
        decisions.append(row)
    return decisions


def assemble_risk_phase(out,phase_plan):
    """Finish one campaign phase before allowing its next eight-worker window."""
    sys.path.insert(0,str(Path(CLI).parent))
    import scheduler
    path=out/'assembly_scheduler_spec.json';spec=json.loads(path.read_text())
    if 'task_id' not in spec:
        p=subprocess.run(spec['args'],text=True,capture_output=True)
        spec.update(stdout=p.stdout,stderr=p.stderr,returncode=p.returncode)
        if p.returncode:raise RuntimeError(p.stderr or p.stdout)
        spec['task_id']=re.search(r'submitted (t\d+)',p.stdout).group(1)
        path.write_text(json.dumps(spec,indent=2)+'\n')
        args=[sys.executable,CLI,'dispatch','--task-id',spec['task_id'],'--no-bulk-window','--lock-timeout','20']
        p=subprocess.run(args,text=True,capture_output=True)
        (out/'assembly_dispatch.json').write_text(json.dumps(dict(args=args,stdout=p.stdout,
            stderr=p.stderr,returncode=p.returncode),indent=2)+'\n')
        if p.returncode:raise RuntimeError(p.stderr or p.stdout)
    while True:
        state={t['id']:t for t in json.loads(QUEUE.read_text())['tasks']}
        task=state[spec['task_id']]
        if task['status']=='done':break
        if task['status'] not in ('queued','launching','running'):
            raise RuntimeError('Risk assembly stopped: '+task['id'])
        time.sleep(20)
    # Prior assemblies exposed DONE before the resource footer became readable.
    # Wait for that footer on the same job; never resubmit a finished assembly.
    for attempt in range(6):
        rc,body,err=scheduler.run_on(task['node'],shlex.join(['tail','-n','12',task['log_path']]),timeout=35,check=False)
        if rc:raise RuntimeError(err or body)
        (out/'assembly.log').write_text(body.rstrip()+'\n')
        resources=[json.loads(line.split(' ',1)[1]) for line in body.splitlines()
            if line.startswith('SCHEDULEURM_RESOURCE ')]
        if resources:break
        if attempt<5:time.sleep(2)
    assert resources,'Finished assembly has no CPU resource record'
    (out/'assembly_resources.json').write_text(json.dumps(resources[-1],indent=2)+'\n')
    jobs=json.loads((out/'risk-competition-batch_jobs.json').read_text())
    ids=[j['id'] for j in jobs]+[spec['task_id']]
    final={tid:{k:state[tid].get(k) for k in ('status','node','log_path','exit_code')} for tid in ids}
    (out/'scheduler_final_states.json').write_text(json.dumps(final,indent=2)+'\n')
    code="""from pathlib import Path
import json
from performance.launch_historical_component_bank import selected_reference_decisions
b=Path(%r);out=b/%r
s=json.loads((b/'paper_artifacts/historical_body_reference_assembly_v1/summary.json').read_text())
assert s['complete']
keys=['complete','reference_assembly','raw_model_net_preview','total','regions','prior_weights','action_count','fixed_action','computed_predictive_origins','omitted_predictive_origin_mass']
a={k:s[k] for k in keys}
a['records']=[{k:r[k] for k in ['record','raw_winners','native_winners']} for r in s['records']]
checked=json.loads((out/'summary.json').read_text())
a['selected_reference_decisions']=selected_reference_decisions(s['records'],checked['groups'])
print(json.dumps(a))
"""%(REMOTE,str(out.relative_to(ROOT)))
    command='cd '+shlex.quote(REMOTE)+' && '+shlex.join(['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1',
        'PYTHONPATH=.',PYTHON,'-c',code])
    rc,body,err=scheduler.run_on('node005',command,timeout=35,check=False)
    if rc:raise RuntimeError(err or body)
    result=json.loads(body);(out/'assembly_after.json').write_text(json.dumps(result,indent=2)+'\n')
    result['assembly_resources']=resources[-1]
    return result


def continue_risk_competition(plan):
    """Drain the frozen remaining comparisons through the existing dispatcher."""
    spec=json.loads((ROOT/plan).read_text());out=ROOT/spec['output'];path=out/'status.json'
    status=json.loads(path.read_text()) if path.exists() else dict(state='ready',completed_phases=[],
        completed_combinations=0,admitted_combinations=0,precision_holds=[],reported_worker_cpu_seconds=0.,
        maximum_live_workers=8,independent_confirmation=False,full_predictive_integration_admitted=False)
    if status['state'] in ('complete','complete_with_precision_holds'):return status
    baseline=json.loads((out/'baseline_assembly.json').read_text())
    original={r['record']:r for r in baseline['records']}
    try:
        for phase in spec['phases']:
            if phase['plan'] in status['completed_phases']:continue
            phase_out=ROOT/phase['output']
            prepared=json.loads((phase_out/'resolution_plan.json').read_text())
            assert prepared['estimated_cost_admitted'] and prepared['cumulative_estimated_cpu_seconds']<=3600
            status.update(state='running',current_phase=phase['plan'])
            path.write_text(json.dumps(status,indent=2)+'\n')
            launch('risk-competition-batch',phase['plan'])
            summary_path=phase_out/'summary.json'
            summary=json.loads(summary_path.read_text()) if summary_path.exists() else collect_remote(
                'risk-competition-batch',phase_out,phase['plan'])
            assert summary['complete']
            assembled=assemble_risk_phase(phase_out,phase['plan'])
            for k in ('prior_weights','action_count','fixed_action','computed_predictive_origins','omitted_predictive_origin_mass'):
                assert assembled[k]==baseline[k],k
            held=[{k:r[k] for k in ('record','kind','group')} for r in summary['groups'] if not r['hybrid_direct_admitted']]
            status['completed_phases'].append(phase['plan'])
            status['completed_combinations']+=len(summary['groups'])
            status['admitted_combinations']+=sum(r['hybrid_direct_admitted'] for r in summary['groups'])
            status['precision_holds'].extend(held)
            status['reported_worker_cpu_seconds']+=summary['successful_task_cpu_seconds']+assembled['assembly_resources']['process_cpu_seconds']
            status.update(reference_groups=assembled['reference_assembly']['groups'],model_net_preview=assembled['raw_model_net_preview'],
                changed_recommendation_records=[r['record'] for r in assembled['records'] if any(
                    r[k]!=original[r['record']][k] for k in ('raw_winners','native_winners'))])
            path.write_text(json.dumps(status,indent=2)+'\n')
            print('RISK_CAMPAIGN_PHASE_DONE '+json.dumps(status),flush=True)
        sys.path.insert(0,str(Path(CLI).parent))
        import scheduler
        command='cd '+shlex.quote(REMOTE)+' && '+shlex.join(['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1',
            'PYTHONPATH=.',PYTHON,'paper_artifacts/historical_body_reference_assembly_v1/price_remaining.py',
            '--plan',spec['phases'][-1]['plan']])
        rc,body,err=scheduler.run_on('node005',command,timeout=35,check=False)
        (out/'remaining_pricing.log').write_text(body.rstrip()+'\n')
        (out/'remaining_pricing_command.json').write_text(json.dumps(dict(command=command,returncode=rc),indent=2)+'\n')
        if rc:raise RuntimeError(err or body)
        code="from pathlib import Path;import json;p=Path(%r);print(json.dumps({n:json.loads((p/n).read_text()) for n in ['compact_summary.json','remaining_cost_plan.json']}))"%(REMOTE+'/paper_artifacts/historical_body_reference_assembly_v1')
        rc,body,err=scheduler.run_on('node005',shlex.join([PYTHON,'-c',code]),timeout=35,check=False)
        if rc:raise RuntimeError(err or body)
        final=json.loads(body)
        for name,value in final.items():(ROOT/'paper_artifacts/historical_body_reference_assembly_v1'/name).write_text(json.dumps(value,indent=2)+'\n')
        assert status['completed_combinations']==spec['combinations']
        status.update(state='complete_with_precision_holds' if status['precision_holds'] else 'complete',
            remaining_selected_reference_checks=spec['combinations']-status['admitted_combinations'],
            remaining_missing_group_kind_record_checks=final['remaining_cost_plan.json']['missing_group_kind_record_checks'])
        if 'certificate_bound_plan' in spec:
            bound=json.loads((ROOT/spec['certificate_bound_plan']).read_text())
            command='cd '+shlex.quote(REMOTE)+' && '+shlex.join(['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1',
                'PYTHONPATH=.',PYTHON,'-m','performance.probe_historical_certificate_bound','collect',
                '--plan',spec['certificate_bound_plan']])
            rc,body,err=scheduler.run_on('node005',command,timeout=35,check=False)
            (out/'bound_reassembly.log').write_text(body+err)
            if rc:raise RuntimeError(err or body)
            code='from pathlib import Path;print(Path('+repr(REMOTE+'/'+bound['output']+'/compact_summary.json')+').read_text())'
            rc,body,err=scheduler.run_on('node005',shlex.join([PYTHON,'-c',code]),timeout=35,check=False)
            if rc:raise RuntimeError(err or body)
            refreshed=json.loads(body)
            (ROOT/bound['output']/'compact_summary.json').write_text(json.dumps(refreshed,indent=2)+'\n')
            status.update(remaining_bound_competitor_checks=refreshed['unreferenced_group_record_checks_remaining'],
                bound_numerical_gate_passed=refreshed['raw_marginal_numerical_gate_passed'],
                bound_reference_containment_passed=refreshed['cached_reference_containment_passed'])
    except Exception as error:
        status.update(state='failed',error=str(error));path.write_text(json.dumps(status,indent=2)+'\n');raise
    path.write_text(json.dumps(status,indent=2)+'\n');print('RISK_CAMPAIGN_DONE '+json.dumps(status),flush=True)
    return status


def continue_observed_batch_risk(plan):
    """Continue the finite n4 action overlaps after each admitted assembly."""
    sys.path.insert(0,str(Path(CLI).parent))
    import scheduler
    spec=json.loads((ROOT/plan).read_text());out=ROOT/spec['output'];out.mkdir(parents=True,exist_ok=True);path=out/'status.json'
    status=json.loads(path.read_text()) if path.exists() else dict(state='ready',completed_phases=[],
        source_bound_plan=spec['initial_bound_plan'],reported_worker_cpu_seconds=0.,maximum_live_workers=8,
        full_predictive_integration_admitted=False,independent_confirmation=False)
    try:
        while True:
            if 'current_phase' not in status:
                source=status['source_bound_plan'];source_spec=json.loads((ROOT/source).read_text())
                priced=json.loads((ROOT/source_spec['output']/'next_cost_plan.json').read_text())
                if not priced['next_joint_jobs'] or not priced['cost_admitted']:
                    guarded=sum(sum(v for k,v in groups.items() if int(k)<73) for groups in priced['remaining_groups'].values())
                    status['state']='cost_hold' if guarded else 'guarded_complete_with_direct_pending' if priced['remaining_direct_cases'] else 'conditional_actions_closed'
                    break
                command='cd '+shlex.quote(REMOTE)+' && '+shlex.join(['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1',
                    'PYTHONPATH=.',PYTHON,'-u','-m','performance.screen_historical_observed_batch_risk','prepare-next','--plan',source])
                rc,body,err=scheduler.run_on('node005',command,timeout=35,check=False)
                (out/'preparation_latest.log').write_text(body+err)
                if rc:raise RuntimeError(err or body)
                prepared=json.loads([line[len('SCHEDULEURM_DONE '):] for line in body.splitlines() if line.startswith('SCHEDULEURM_DONE ')][-1])
                files={prepared['risk_plan']:prepared['risk_manifest'],prepared['bound_plan']:prepared['bound_manifest'],
                    prepared['risk_manifest']['output']+'/preparation.json':prepared['preparation']}
                for name,data in files.items():
                    for root in (ROOT,Path(DEPLOY)):
                        target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(data,indent=2)+'\n')
                status['current_phase']=prepared['risk_plan'];status['state']='running';path.write_text(json.dumps(status,indent=2)+'\n')
            current=status['current_phase'];phase=json.loads((ROOT/current).read_text());phase_out=ROOT/phase['output']
            status.update(state='running',current_phase_expected_cpu_seconds=json.loads((phase_out/'preparation.json').read_text())['estimated_cpu_seconds_including_reserves'])
            path.write_text(json.dumps(status,indent=2)+'\n')
            launch('observed-batch-risk',current)
            summary=collect_remote('observed-batch-risk',phase_out,current)
            bound=phase['followup_bound_plan'];bound_out=ROOT/json.loads((ROOT/bound).read_text())['output']
            screened=collect_remote('observed-batch-bound',bound_out,bound)
            status.update(remaining_totals=screened['totals'],current_phase_expected_cpu_seconds=summary['estimated_cpu_seconds_including_reserves'])
            if not summary['local_risk_candidate_passed'] or not screened['numerical_containment_passed']:
                status['state']='precision_hold';break
            collect_remote('observed-batch-next-plan',bound_out,bound)
            status['completed_phases'].append(current);status['reported_worker_cpu_seconds']+=summary['successful_worker_cpu_seconds']
            status.update(source_bound_plan=bound);status.pop('current_phase')
            path.write_text(json.dumps(status,indent=2)+'\n');print('OBSERVED_BATCH_PHASE_DONE '+json.dumps(status),flush=True)
    except Exception as error:
        status.update(state='failed',error=str(error));path.write_text(json.dumps(status,indent=2)+'\n');raise
    path.write_text(json.dumps(status,indent=2)+'\n')
    if status['state']=='conditional_actions_closed' and spec.get('assembly_plan'):
        assembly_plan=spec['assembly_plan'];assembly=json.loads((ROOT/assembly_plan).read_text())
        # The server needs the completed local controller state,not a bank download.
        code='from pathlib import Path;p=Path('+repr(REMOTE+'/'+spec['output']+'/status.json')+');p.parent.mkdir(parents=True,exist_ok=True);p.write_text('+repr(json.dumps(status,indent=2)+'\n')+')'
        rc,body,err=scheduler.run_on('node005',shlex.join([PYTHON,'-c',code]),timeout=35,check=False)
        if rc:raise RuntimeError(err or body)
        assembled=collect_remote('observed-batch-assembly',ROOT/assembly['output'],assembly_plan)
        status['finite_outer_diagnostics_passed']=assembled['finite_outer_diagnostics_passed']
        status['assembly_summary']=assembly['output']+'/compact_summary.json'
        path.write_text(json.dumps(status,indent=2)+'\n')
    print('OBSERVED_BATCH_CAMPAIGN_DONE '+json.dumps(status),flush=True)
    return status


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['screen','refine','resolution','continue',
        'adaptive','adaptive-resolution','continue-adaptive','bank-resolution','continue-bank-resolution',
        'bank-coarse','continue-bank-coarse','bank-tail','continue-bank-tail',
        'bank-tail-resolution','continue-bank-tail-resolution','continue-transfer','continue-label-bank',
        'continue-risk-competition','observed-batch-risk','continue-observed-batch-risk']+list(PILOT_MODES)+list(TRANSFER_STAGES)+list(BANK_STAGES)+sorted(LABEL_MODES)+list(UPPER_MODES))
    parser.add_argument('--plan',help='Manifest path for finite risk phases or campaigns')
    args=parser.parse_args()
    if args.plan and args.stage not in ('risk-competition-batch','continue-risk-competition','observed-batch-risk','continue-observed-batch-risk'):
        parser.error('--plan requires a finite risk phase or campaign stage')
    if args.stage=='continue':continue_pipeline()
    elif args.stage=='continue-adaptive':continue_adaptive()
    elif args.stage=='continue-bank-resolution':continue_resolution_pilot()
    elif args.stage=='continue-bank-coarse':continue_coarse_resolution()
    elif args.stage=='continue-bank-tail':continue_tail()
    elif args.stage=='continue-bank-tail-resolution':continue_tail('bank-tail-resolution')
    elif args.stage=='continue-transfer':continue_transfer()
    elif args.stage=='continue-label-bank':continue_label_bank()
    elif args.stage=='continue-risk-competition':continue_risk_competition(
        args.plan or 'performance/manifests/historical_risk_competition_remaining_v1.json')
    elif args.stage=='continue-observed-batch-risk':
        if not args.plan:parser.error('continue-observed-batch-risk requires --plan')
        continue_observed_batch_risk(args.plan)
    elif args.stage=='observed-batch-risk':
        if not args.plan:parser.error('observed-batch-risk requires --plan')
        spec=json.loads((ROOT/args.plan).read_text())
        launch(args.stage,args.plan)
        collect_remote(args.stage,ROOT/spec['output'],args.plan)
        if 'followup_bound_plan' in spec:
            bound=spec['followup_bound_plan']
            collect_remote('observed-batch-bound',ROOT/json.loads((ROOT/bound).read_text())['output'],bound)
    elif args.stage in UPPER_MODES:
        launch(args.stage);collect_remote(args.stage,ROOT/'paper_artifacts/historical_upper_decision_refs_v1')
    elif args.stage in BANK_STAGES:
        launch(args.stage);collect_remote(args.stage,ROOT/'paper_artifacts'/('historical_cdf_bank_v1' if args.stage=='cdf-bank-coarse' else 'historical_label_bank_v1'))
    elif args.stage in LABEL_MODES:
        launch(args.stage,args.plan)
        out=ROOT/json.loads((ROOT/args.plan).read_text())['output'] if args.plan else ROOT/'paper_artifacts'/('historical_'+LABEL_MODES[args.stage][0]+'_v1')
        collect_remote(args.stage,out,args.plan)
    else:launch(args.stage)
