"""Collect completed full-grid phase logs and replay its admitted choices."""
import argparse
import json
import shlex
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from performance.launch_historical_component_bank import ROOT,CLI,PYTHON,QUEUE,REMOTE
sys.path.insert(0,str(Path(CLI).parent))
import scheduler


def close(plan):
    bound=json.loads((ROOT/plan).read_text());out=ROOT/Path(bound['phase_completion']).parent
    initial='performance/manifests/'+out.name+'.json';number=int(out.name.rsplit('_v',1)[1])
    phase_plans=bound['joint_reference_plans'][bound['joint_reference_plans'].index(initial):]
    queue=json.loads(QUEUE.read_text());resources=[];checks=[];allocations=0;scientific=0.;logs=[];prep_cpu=0.
    for i,reference in enumerate(phase_plans):
        spec=json.loads((ROOT/reference).read_text());folder=ROOT/spec['output']
        jobs=json.loads((folder/'observed-batch-risk_jobs.json').read_text());ids={j['id'] for j in jobs}
        tasks=[t for t in queue['tasks'] if t['id'] in ids]
        assert len(tasks)==len(jobs) and all(t['status']=='done' for t in tasks)
        local=[]
        for node in sorted({t['node'] for t in tasks}):
            paths={t['id']:t['log_path'] for t in tasks if t['node']==node}
            code='import json;from pathlib import Path;paths='+repr(paths)+';print(json.dumps({k:json.loads([s[21:] for s in Path(v).read_text().splitlines() if s.startswith("SCHEDULEURM_RESOURCE ")][-1]) for k,v in paths.items()}))'
            rc,body,err=scheduler.run_on(node,shlex.join([PYTHON,'-c',code]),timeout=25,check=False)
            if rc:raise RuntimeError(err or body)
            rows=json.loads(body);local.extend(dict(id=t['id'],node=node,status=t['status'],resource=rows[t['id']]) for t in tasks if t['node']==node)
        (folder/'worker_resources.json').write_text(json.dumps(local,indent=2)+'\n');resources.extend(local)
        summary=json.loads((folder/'compact_summary.json').read_text());prep=json.loads((folder/'preparation.json').read_text())
        assert summary['local_risk_candidate_passed']
        sizes={r['record']:len(r['action_indices']) for r in prep['selected']}
        allocations+=sum(sizes[r['record']] for r in summary['groups']);checks.extend(summary['groups']);scientific+=summary['successful_worker_cpu_seconds']
        resource=(ROOT/'paper_artifacts/historical_positive_risk_full_grid_continuation_v1'/f'preparation_v{number}_result.json') if i==0 else folder/'preparation_resources.json'
        recorded=json.loads(resource.read_text());prep_cpu+=(recorded['resource'] if i==0 else recorded)['process_cpu_seconds']
        logs.append(folder/'observed-batch-risk_collection.log')
        following=json.loads((ROOT/spec['followup_bound_plan']).read_text())
        logs.extend(ROOT/following['output']/(stage+'_collection.log') for stage in ('observed-batch-bound','observed-batch-next-plan'))
    destination=str((out/'incumbent_replay_final.json').relative_to(ROOT))
    command='cd '+shlex.quote(REMOTE)+' && '+shlex.join(['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','PYTHONPATH=.',PYTHON,'-u','-m','performance.replay_historical_positive_risk_subset','--plan',plan,'--output',destination])
    rc,body,err=scheduler.run_on('node005',command,timeout=35,check=False);log=out/'incumbent_replay_final.log';log.write_text(body+err)
    if rc:raise RuntimeError(err or body)
    replay=json.loads([s[17:] for s in body.splitlines() if s.startswith('SCHEDULEURM_DONE ')][-1])
    assert replay['source_bound_plan']==plan and replay['native_choices_all_raw_referenced']
    (ROOT/destination).write_text(json.dumps(replay,indent=2)+'\n');logs.append(log)
    if (out/'incumbent_replay_initial.log').exists():logs.append(out/'incumbent_replay_initial.log')
    if (out/'incumbent_replay_failed_close.log').exists():logs.append(out/'incumbent_replay_failed_close.log')
    collection=sum(json.loads(s[21:])['process_cpu_seconds'] for p in logs for s in p.read_text().splitlines() if s.startswith('SCHEDULEURM_RESOURCE '))
    first=json.loads((ROOT/initial).read_text());frozen=json.loads((ROOT/first['frozen_cost_plan']).read_text())
    previous=json.loads((ROOT/frozen['previous_closed_phase_completion']).read_text())
    failure=json.loads((ROOT/first['failed_preparation_account']).read_text()) if 'failed_preparation_account' in first else None
    failure_charge=0. if failure is None else failure['conservative_reserve_charge_seconds']
    reserve_charge=prep_cpu+collection+failure_charge;assert reserve_charge<=600
    envelope=json.loads((ROOT/bound['output']/'compact_summary.json').read_text());price=json.loads((ROOT/bound['output']/'next_cost_plan.json').read_text())
    assert envelope['numerical_containment_passed']
    worker=sum(r['resource']['process_cpu_seconds'] for r in resources)
    result=dict(complete=True,state='phase_complete',reference_workers=len(resources),
        estimated_phase_cpu_seconds=prep['estimated_cpu_seconds_including_reserves'],shared_reserve_cpu_seconds=600,
        previous_closed_phase_completion=frozen['previous_closed_phase_completion'],
        previous_phase_estimated_cpu_seconds=previous['estimated_phase_cpu_seconds'],
        scientific_work_cpu_seconds=scientific,worker_process_cpu_seconds=worker,
        preparation_process_cpu_seconds=prep_cpu,collection_process_cpu_seconds=collection,
        total_recorded_process_cpu_seconds=worker+prep_cpu+collection,
        unmeasured_failed_preparation_conservative_charge_seconds=failure_charge,
        recorded_cpu_plus_conservative_failed_charge_seconds=worker+reserve_charge,
        shared_reserve_charged_cpu_seconds=reserve_charge,
        peak_worker_rss_mb=max(r['resource']['peak_rss_mb'] for r in resources),
        reference_group_checks=len(checks),allocation_checks=allocations,all_reference_precision_passed=True,
        maximum_reference_rule_difference=max(c['maximum_difference'] for r in checks for c in r['checks']['comparisons']),
        maximum_live_workers=8,native_incumbents_all_raw_referenced=True,
        raw_reward_of_native_incumbent_contrast=replay['weighted_raw_reward_of_native_incumbent_contrast'],
        raw_best_admitted_contrast=replay['weighted_raw_best_admitted_contrast'],
        projected_incumbent_contrast=replay['weighted_projected_incumbent_contrast'],
        weighted_bank_envelope=envelope['weighted_bank_envelope'],remaining_totals=envelope['totals'],
        next_cost_admitted=price['cost_admitted'],new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,
        full_predictive_integration_admitted=False,independent_confirmation=False)
    result['weighted_unresolved_gap_reduction']={k:
        1-envelope['weighted_bank_envelope']['kinds'][k]['unresolved_upper_gap']/previous['weighted_bank_envelope']['kinds'][k]['unresolved_upper_gap']
        if previous['weighted_bank_envelope']['kinds'][k]['unresolved_upper_gap'] else None
        for k in ('raw','projected')}
    (out/'completion.json').write_text(json.dumps(result,indent=2)+'\n')
    status=json.loads((out/'status.json').read_text());status.update(state='phase_complete',
        cumulative_phase_estimated_cpu_seconds=result['estimated_phase_cpu_seconds'],
        remaining_totals=envelope['totals'],weighted_bank_envelope=envelope['weighted_bank_envelope'],
        next_cost_admitted=price['cost_admitted'],final_bound_plan=plan,completion=bound['phase_completion'])
    (out/'status.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);close(p.parse_args().plan)
