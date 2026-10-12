"""Refine the body grid, then complete its native raw values using existing runners."""
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from performance.launch_historical_component_bank import CLI,DEPLOY,PYTHON,REMOTE
sys.path.insert(0,str(Path(CLI).parent))
import scheduler

out=Path(__file__).resolve().parent
helper=ROOT/'paper_artifacts/historical_positive_risk_outer_axes_v1'
status=dict(state='running',maximum_live_workers=8,priority='low',
    full_predictive_integration_admitted=False,independent_confirmation=False)
try:
    plan='performance/manifests/historical_positive_risk_body_axis_v1.json'
    subprocess.run([sys.executable,'-u',str(ROOT/'paper_artifacts/historical_positive_risk_full_grid_v1/continue_when_ready.py'),
        '--plan',plan],cwd=ROOT,check=True)
    screened=json.loads((out/'status.json').read_text());assert screened['state']=='screened_coarse_grid',screened
    frozen=json.loads((out/'incumbent_frozen_plan.json').read_text())
    risk_plan=frozen['plans'][0];risk=json.loads((ROOT/risk_plan).read_text());folder=ROOT/risk['output']
    folder.mkdir(parents=True,exist_ok=True)
    command='cd '+shlex.quote(REMOTE)+' && '+shlex.join(['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','PYTHONPATH=.','nice','-n','19',PYTHON,'-u',
        '-m','performance.positive_historical_direct_references','prepare-incumbents','--plan',risk_plan])
    rc,body,err=scheduler.run_on('node004',command,timeout=35,check=False)
    (folder/'preparation.log').write_text(body+err)
    if rc:raise RuntimeError(err or body)
    result=json.loads([line[17:] for line in body.splitlines() if line.startswith('SCHEDULEURM_DONE ')][-1])
    resource=json.loads([line[21:] for line in body.splitlines() if line.startswith('SCHEDULEURM_RESOURCE ')][-1])
    for path,data in ((risk_plan,result['manifest']),(risk['frozen_cost_plan'],result['frozen_cost_plan']),
        (risk['output']+'/preparation.json',result['preparation']),(risk['output']+'/preparation_resources.json',resource)):
        for root in (ROOT,Path(DEPLOY)):
            target=root/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(data,indent=2)+'\n')
    subprocess.run([sys.executable,str(helper/'validate_incumbents.py'),'--frozen-plan',str(out/'incumbent_frozen_plan.json')],cwd=ROOT,check=True)
    result=json.loads((out/'incumbent_status.json').read_text());assert result['state']=='incumbent_probes_complete',result
    subprocess.run([sys.executable,str(helper/'close_initial_phases.py'),'--axes','body'],cwd=ROOT,check=True)
    completion=json.loads((folder/'completion.json').read_text())
    if completion['native_choices_all_raw_referenced']:
        status.update(state='native_raw_coverage_complete',last_reference_plan=risk_plan)
    else:
        next_folder='paper_artifacts/historical_positive_risk_body_axis_followthrough_v2'
        subprocess.run([sys.executable,str(helper/'prepare_next_incumbents.py'),'--phase','2','--axes','body',
            '--frozen-output',next_folder],cwd=ROOT,check=True)
        runner=ROOT/'paper_artifacts/historical_positive_risk_outer_axes_incumbents_v2/finish_coverage.py'
        subprocess.run([sys.executable,str(runner),'--frozen-plan',str(ROOT/next_folder/'incumbent_frozen_plan.json')],cwd=ROOT,check=True)
        followup=json.loads((ROOT/next_folder/'coverage_status.json').read_text())
        status.update(state=followup['state'],latest_axes=followup['latest_axes'])
except Exception as error:
    status.update(state='failed',error=str(error));raise
finally:
    (out/'body_status.json').write_text(json.dumps(status,indent=2)+'\n')
    print('BODY_GRID_DONE '+json.dumps(status),flush=True)
