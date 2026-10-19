"""Collect query caps and retain the original cap phase's single reserve."""
import json
import shlex
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from performance.launch_historical_component_bank import ROOT,CLI,PYTHON,REMOTE,QUEUE,refresh_jobs
sys.path.insert(0,str(Path(CLI).parent))
import scheduler

plan='performance/manifests/historical_positive_risk_uppervariance_axis_query_caps_v1.json'
spec=json.loads((ROOT/plan).read_text());out=ROOT/spec['output']
if (out/'completion.json').exists():
    print((out/'summary.json').read_text());raise SystemExit(0)
jobs=json.loads((out/'registered_jobs.json').read_text());ids={j['id'] for j in jobs};deadline=time.monotonic()+300
while True:
    refresh_jobs(ids);tasks=[t for t in json.loads(QUEUE.read_text())['tasks'] if t['id'] in ids]
    if all(t['status'] not in ('queued','launching','running') for t in tasks):break
    if time.monotonic()>deadline:raise TimeoutError('Query-cap workers still active; collect their existing outputs later.')
    time.sleep(15)
resources=[]
for node in sorted({t['node'] for t in tasks}):
    paths={t['id']:t.get('log_path') or '/tmp/sched_'+t['id']+'.log' for t in tasks if t['node']==node}
    code=('import json,time;from pathlib import Path;paths='+repr(paths)+';'
        'print(json.dumps(dict(resources={k:json.loads([s[21:] for s in Path(v).read_text().splitlines() '
        'if s.startswith("SCHEDULEURM_RESOURCE ")][-1]) for k,v in paths.items()},collection_cpu_seconds=time.process_time())))')
    rc,body,err=scheduler.run_on(node,shlex.join([PYTHON,'-c',code]),timeout=25,check=False)
    if rc:raise RuntimeError(err or body)
    r=json.loads(body)
    with (out/'support.log').open('a') as stream:stream.write('SCHEDULEURM_RESOURCE '+json.dumps(dict(process_cpu_seconds=r['collection_cpu_seconds']))+'\n')
    resources.extend(dict(id=t['id'],node=node,status=t['status'],resource=r['resources'][t['id']]) for t in tasks if t['node']==node)
(out/'worker_resources.json').write_text(json.dumps(resources,indent=2)+'\n')
assert all(t['status']=='done' for t in tasks),[(t['id'],t['status']) for t in tasks]
command='cd '+shlex.quote(REMOTE)+' && '+shlex.join(['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','PYTHONPATH=.',PYTHON,'-u','-m','performance.refine_historical_query_certificate_caps','collect','--plan',plan])
rc,body,err=scheduler.run_on('node004',command,timeout=60,check=False)
(out/'collection.log').write_text(body+err)
if rc:raise RuntimeError(err or body)
result=json.loads([s[17:] for s in body.splitlines() if s.startswith('SCHEDULEURM_DONE ')][-1])
(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
support=sum(json.loads(s[21:])['process_cpu_seconds'] for name in ('preparation.log','support.log','collection.log')
    for s in (out/name).read_text().splitlines() if s.startswith('SCHEDULEURM_RESOURCE '))
pilot=json.loads((out/'pilot_worker_resources.json').read_text())['resource']['process_cpu_seconds']
workers=sum(r['resource']['process_cpu_seconds'] for r in resources)
source=json.loads((ROOT/spec['source_phase_completion']).read_text());price=json.loads((out/'extension_cost_plan.json').read_text())
assert source['shared_reserve_charged_cpu_seconds']+support<=600
assert source['total_recorded_process_cpu_seconds']+pilot+workers+support<=3600
completion=dict(complete=True,state=result['state'],source_phase_completion=spec['source_phase_completion'],
    aggregate_estimated_phase_cpu_seconds=price['aggregate_estimated_phase_cpu_seconds'],
    worker_process_cpu_seconds=pilot+workers,collection_and_preparation_cpu_seconds=support,
    additional_recorded_process_cpu_seconds=pilot+workers+support,
    cumulative_cap_phase_recorded_process_cpu_seconds=source['total_recorded_process_cpu_seconds']+pilot+workers+support,
    cumulative_shared_reserve_charged_cpu_seconds=source['shared_reserve_charged_cpu_seconds']+support,
    shared_phase_reserve_cpu_seconds=600,new_phase_reserve_cpu_seconds=0,
    numerical_containment_passed=result['numerical_containment_passed'],precision_passed=result['precision_passed'],
    original_zero_candidate_gate_unchanged=True,all_conditional_actions_closed=result['all_conditional_actions_closed'],
    new_joint_reference_integrations=0,new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,independent_confirmation=False)
(out/'completion.json').write_text(json.dumps(completion,indent=2)+'\n')
print(json.dumps(dict(summary=result,completion=completion)),flush=True)
