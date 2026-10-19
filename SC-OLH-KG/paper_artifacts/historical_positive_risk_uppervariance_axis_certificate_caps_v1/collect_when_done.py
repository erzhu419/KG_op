"""Collect this eight-worker cap phase without retrieving component arrays."""
import json
import shlex
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from performance.launch_historical_component_bank import ROOT,CLI,PYTHON,REMOTE,QUEUE,refresh_jobs
sys.path.insert(0,str(Path(CLI).parent))
import scheduler

plan='performance/manifests/historical_positive_risk_uppervariance_axis_certificate_caps_v1.json'
spec=json.loads((ROOT/plan).read_text());out=ROOT/spec['output']
if (out/'completion.json').exists():
    print(json.dumps(dict(summary=json.loads((out/'summary.json').read_text()),
        completion=json.loads((out/'completion.json').read_text()))),flush=True)
    raise SystemExit(0)
jobs=json.loads((out/'registered_jobs.json').read_text());ids={j['id'] for j in jobs}
deadline=time.monotonic()+600
while True:
    refresh_jobs(ids)
    tasks=[t for t in json.loads(QUEUE.read_text())['tasks'] if t['id'] in ids]
    assert len(tasks)==len(jobs)
    if all(t['status'] not in ('queued','launching','running') for t in tasks):break
    if time.monotonic()>deadline:raise TimeoutError('Cap phase still active; collect existing jobs after completion.')
    time.sleep(20)
resources=[];support=[]
for node in sorted({t['node'] for t in tasks}):
    paths={t['id']:t.get('log_path') or '/tmp/sched_'+t['id']+'.log' for t in tasks if t['node']==node}
    code=('import json,time;from pathlib import Path;paths='+repr(paths)+';'
        'data={k:Path(v).read_text().splitlines() for k,v in paths.items()};'
        'print(json.dumps(dict(resources={k:json.loads([s[21:] for s in v if s.startswith("SCHEDULEURM_RESOURCE ")][-1]) '
        'for k,v in data.items()},collection_cpu_seconds=time.process_time())))')
    rc,body,err=scheduler.run_on(node,shlex.join([PYTHON,'-c',code]),timeout=25,check=False)
    if rc:raise RuntimeError(err or body)
    result=json.loads(body);support.append(dict(process_cpu_seconds=result['collection_cpu_seconds']))
    resources.extend(dict(id=t['id'],node=node,status=t['status'],resource=result['resources'][t['id']]) for t in tasks if t['node']==node)
(out/'worker_resources.json').write_text(json.dumps(resources,indent=2)+'\n')
(out/'resource_collection.log').write_text(''.join('SCHEDULEURM_RESOURCE '+json.dumps(r)+'\n' for r in support))
assert all(t['status']=='done' for t in tasks),[(t['id'],t['status']) for t in tasks]
command='cd '+shlex.quote(REMOTE)+' && '+shlex.join(['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','PYTHONPATH=.',PYTHON,'-u','-m','performance.refine_historical_certificate_caps','collect','--plan',plan])
rc,body,err=scheduler.run_on('node004',command,timeout=60,check=False)
(out/'collection.log').write_text(body+err)
if rc:raise RuntimeError(err or body)
result=json.loads([s[17:] for s in body.splitlines() if s.startswith('SCHEDULEURM_DONE ')][-1])
(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
support_cpu=sum(json.loads(s[21:])['process_cpu_seconds'] for name in ('preparation.log','resource_collection.log','collection.log')
    for s in (out/name).read_text().splitlines() if s.startswith('SCHEDULEURM_RESOURCE '))
worker_cpu=sum(r['resource']['process_cpu_seconds'] for r in resources)
assert support_cpu<=600 and worker_cpu+support_cpu<=3600
completion=dict(complete=True,state=result['state'],source_phase_completion=spec['source_phase_completion'],
    source_bound_plan=spec['source_bound_plan'],followup_bound_plan=spec['followup_bound_plan'],
    estimated_phase_cpu_seconds=json.loads((out/'preparation.json').read_text())['estimated_phase_cpu_seconds'],
    worker_process_cpu_seconds=worker_cpu,collection_and_preparation_cpu_seconds=support_cpu,
    total_recorded_process_cpu_seconds=worker_cpu+support_cpu,shared_reserve_charged_cpu_seconds=support_cpu,
    maximum_live_workers=8,precision_passed=result['precision_passed'],
    numerical_containment_passed=result.get('numerical_containment_passed',False),
    all_conditional_actions_closed=result.get('all_conditional_actions_closed',False),
    new_joint_reference_integrations=0,new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,
    all_previous_phase_costs_retained=True,original_zero_candidate_gate_unchanged=True,independent_confirmation=False)
(out/'completion.json').write_text(json.dumps(completion,indent=2)+'\n')
print(json.dumps(dict(summary=result,completion=completion)),flush=True)
