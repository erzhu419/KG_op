"""Run the two frozen axis screens through the historical grid controller."""
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
out=Path(__file__).resolve().parent
frozen=json.loads((out/'frozen_plan.json').read_text())
status=dict(state='running',completed_plans=[],maximum_live_workers=8,
    total_recorded_process_cpu_seconds=0.,full_predictive_integration_admitted=False,independent_confirmation=False)
try:
    for plan in frozen['plans']:
        status['current_plan']=plan
        (out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
        subprocess.run([sys.executable,'-u',str(ROOT/'paper_artifacts/historical_positive_risk_full_grid_v1/continue_when_ready.py'),
            '--plan',plan],cwd=ROOT,check=True)
        spec=json.loads((ROOT/plan).read_text());folder=ROOT/spec['output']
        phase=json.loads((folder/'status.json').read_text())
        if phase['state']!='screened_coarse_grid':
            status.update(state=phase['state']);break
        account=json.loads((folder/'execution_account.json').read_text())
        status['total_recorded_process_cpu_seconds']+=sum(account[k] for k in
            ('worker_process_cpu_seconds','preparation_process_cpu_seconds','collection_process_cpu_seconds'))
        status['completed_plans'].append(plan)
    else:
        status.update(state='axis_screens_complete');status.pop('current_plan')
except Exception as error:
    status.update(state='failed',error=str(error));raise
finally:
    (out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
print('AXIS_SCREENS_DONE '+json.dumps(status),flush=True)
