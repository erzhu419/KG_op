"""Collect the eight registered switch-screen jobs; do not dispatch new work."""
import json
import subprocess
import sys
import time

from performance.launch_historical_component_bank import CLI, ROOT, QUEUE, collect_remote

plan='performance/manifests/historical_positive_risk_screen_v1.json'
out=ROOT/json.loads((ROOT/plan).read_text())['output']
ids={j['id'] for j in json.loads((out/'jobs.json').read_text())}
while True:
    tasks={t['id']:t for t in json.loads(QUEUE.read_text())['tasks'] if t['id'] in ids}
    assert len(tasks)==len(ids)
    assert all(t['status'] in ('queued','launching','running','done') for t in tasks.values()),tasks
    pending=[tid for tid,t in tasks.items() if t['status']!='done']
    if not pending:break
    subprocess.run([sys.executable,CLI,'status','--json','--brief','--lock-timeout','20','--ids']+pending,
        capture_output=True,text=True,check=True,timeout=45)
    time.sleep(20)
summary=collect_remote('observed-batch-bound',out,plan)
(out/'completed_worker_states.json').write_text(json.dumps({tid:{k:t.get(k) for k in
    ('status','node','log_path','exit_code')} for tid,t in tasks.items()},indent=2)+'\n')
print('POSITIVE_RISK_SCREEN_DONE '+json.dumps(summary),flush=True)
