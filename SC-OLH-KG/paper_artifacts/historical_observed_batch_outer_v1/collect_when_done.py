"""Collect the registered fixed-action phase once its existing workers finish."""
import json
import time
from pathlib import Path
from performance.launch_historical_component_bank import ROOT,QUEUE,collect_remote

plan='performance/manifests/historical_observed_batch_outer_v1.json'
spec=json.loads((ROOT/plan).read_text());out=ROOT/spec['output']
ids={j['id'] for j in json.loads((out/'scheduler_jobs.json').read_text())}
while True:
    tasks={t['id']:t for t in json.loads(QUEUE.read_text())['tasks'] if t['id'] in ids}
    stopped=[tid for tid,t in tasks.items() if t['status'] not in ('queued','launching','running','done')]
    if stopped:raise RuntimeError('Fixed-action worker stopped: '+','.join(stopped))
    if len(tasks)==len(ids) and all(t['status']=='done' for t in tasks.values()):break
    time.sleep(20)
summary=collect_remote('observed-batch-outer',out,plan)
(out/'completed_worker_states.json').write_text(json.dumps({tid:{k:t.get(k) for k in
    ('status','node','log_path','exit_code')} for tid,t in tasks.items()},indent=2)+'\n')
print('FIXED_OUTER_PHASE_DONE '+json.dumps(summary),flush=True)
