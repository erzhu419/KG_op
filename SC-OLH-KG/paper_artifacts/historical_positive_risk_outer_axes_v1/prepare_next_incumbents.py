"""Prepare the next bounded raw-coverage phase after registered work passes."""
import argparse
import base64
import copy
import json
import shlex
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from performance.launch_historical_component_bank import CLI,DEPLOY,PYTHON,REMOTE
sys.path.insert(0,str(Path(CLI).parent))
import scheduler

def read(path):
    return json.loads((ROOT/path).read_text())

def save(path,data):
    for root in (ROOT,Path(DEPLOY)):
        target=root/path;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(data,indent=2)+'\n')

def upload(path):
    data=(ROOT/path).read_bytes()
    for start in range(0,len(data),32000):
        code='from pathlib import Path;import base64;p=Path('+repr(REMOTE+'/'+path)+');p.parent.mkdir(parents=True,exist_ok=True);p.open('+repr('wb' if start==0 else 'ab')+').write(base64.b64decode('+repr(base64.b64encode(data[start:start+32000]).decode())+'))'
        rc,body,err=scheduler.run_on('node004',shlex.join([PYTHON,'-c',code]),timeout=25,check=False)
        if rc:raise RuntimeError(err or body)

parser=argparse.ArgumentParser();parser.add_argument('--phase',type=int,required=True)
parser.add_argument('--axes',nargs='+',choices=('mean','variance','body','shoulder'),default=['mean','variance'])
parser.add_argument('--frozen-output')
args=parser.parse_args();assert args.phase>=2
plans=[];rows=[]
for axis in args.axes:
    prior_name=f'historical_positive_risk_{axis}_axis_incumbents_v{args.phase-1}'
    previous_plan='performance/manifests/'+prior_name+'.json';previous=read(previous_plan)
    previous_completion=previous['output']+'/completion.json';closed=read(previous_completion)
    assert closed['state']=='phase_complete' and closed['all_reference_precision_passed']
    if closed['native_choices_all_raw_referenced']:continue
    name=f'historical_positive_risk_{axis}_axis_incumbents_v{args.phase}'
    risk_plan='performance/manifests/'+name+'.json'
    bound_name=f'historical_positive_risk_{axis}_axis_bound_incumbents_v{args.phase}'
    bound_plan='performance/manifests/'+bound_name+'.json'
    risk=copy.deepcopy(previous)
    risk.update(output='paper_artifacts/'+name,incumbent_bound_plan=previous['followup_bound_plan'],
        previous_phase_completion=previous_completion,frozen_cost_plan='paper_artifacts/'+name+'/next_cost_plan.json',
        followup_bound_plan=bound_plan,refined_cases=[],
        pricing='Same-group actual n4 timers,20 percent margin,2 setup per worker and one new600 reserve after the previous registered phase passed.',
        scope='Complete missing raw values of the existing native recommendations on the registered axis grid. Original source prior,all8 likelihoods,512/1024 components,covariance,365 actions and SD16 retained.')
    risk.pop('continued_phase_cost_plan',None)
    bound=copy.deepcopy(read(previous['followup_bound_plan']))
    bound.update(output='paper_artifacts/'+bound_name,
        joint_reference_plans=bound['joint_reference_plans']+[risk_plan],
        pricing_reference_plans=bound['pricing_reference_plans']+[risk_plan],
        reference_phase_cost_plan=risk['output']+'/preparation.json',continued_phase_cost_plan=risk['output']+'/preparation.json')
    for path,data in ((previous_completion,closed),(risk_plan,risk),(bound_plan,bound)):
        save(path,data);upload(path)
    folder=ROOT/risk['output'];folder.mkdir(parents=True,exist_ok=True)
    command='cd '+shlex.quote(REMOTE)+' && '+shlex.join(['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','PYTHONPATH=.','nice','-n','19',PYTHON,'-u','-m','performance.positive_historical_direct_references','prepare-incumbents','--plan',risk_plan])
    rc,body,err=scheduler.run_on('node004',command,timeout=35,check=False)
    (folder/'preparation.log').write_text(body+err)
    if rc:raise RuntimeError(err or body)
    result=json.loads([line[17:] for line in body.splitlines() if line.startswith('SCHEDULEURM_DONE ')][-1])
    resource=json.loads([line[21:] for line in body.splitlines() if line.startswith('SCHEDULEURM_RESOURCE ')][-1])
    save(risk['output']+'/preparation_resources.json',resource)
    for path,data in ((risk_plan,result['manifest']),(risk['frozen_cost_plan'],result['frozen_cost_plan']),
        (risk['output']+'/preparation.json',result['preparation'])):save(path,data)
    row=dict(axis=axis,plan=risk_plan,registered_workers=len(result['preparation']['jobs']),
        estimated_phase_cpu_seconds=result['preparation']['estimated_cpu_seconds_including_reserves'],
        selected_observations=result['frozen_cost_plan']['selected_observations'],
        remaining_missing_observations=len(result['frozen_cost_plan']['remaining_missing_observations']))
    rows.append(row);plans.append(risk_plan);print(json.dumps(row),flush=True)
folder=args.frozen_output or f'paper_artifacts/historical_positive_risk_outer_axes_incumbents_v{args.phase}'
save(folder+'/incumbent_frozen_plan.json',dict(plans=plans,rows=rows,maximum_live_workers=8,priority='low',
    shared_phase_reserve_cpu_seconds=600,maximum_estimated_phase_cpu_seconds=3600,
    scope='Sequential axis phases reuse cached observations and condition only missing raw recommendation values. Previous phases passed and retain their recorded costs. Outer convergence and independent deployment remain separate.'))
