"""Run local mean refinement, then a variance check after its value stabilizes."""
import argparse
import base64
import json
import shlex
import subprocess
import sys
from pathlib import Path

from performance.complete_historical_source_heads import ROOT, read, save
from performance.launch_historical_component_bank import CLI, DEPLOY, PYTHON, REMOTE
from performance.prepare_historical_raw_axis_refinement import prepare

sys.path.insert(0,str(Path(CLI).parent))
import scheduler


def sync(paths):
    for path in paths:
        data=(ROOT/path).read_bytes();target=Path(DEPLOY)/path
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        for start in range(0,len(data),32000):
            code='from pathlib import Path;import base64;p=Path('+repr(REMOTE+'/'+path)+');p.parent.mkdir(parents=True,exist_ok=True);p.open('+repr('wb' if start==0 else 'ab')+').write(base64.b64decode('+repr(base64.b64encode(data[start:start+32000]).decode())+'))'
            rc,body,err=scheduler.run_on('node004',shlex.join([PYTHON,'-c',code]),timeout=25,check=False)
            if rc:raise RuntimeError(err or body)


def replay_for(bound):
    spec=read(bound);ref=read(spec['joint_reference_plans'][-1])
    completion=read(spec['phase_completion'])
    return read(completion.get('replay_summary_path',ref['output']+'/incumbent_replay_final.json'))


def run(source_bound,mean_axis='midmean'):
    name='historical_raw_axis_followthrough_v1' if mean_axis=='midmean' else f'historical_raw_axis_{mean_axis}_followthrough_v1'
    out=ROOT/'paper_artifacts'/name;out.mkdir(parents=True,exist_ok=True)
    previous=read(out/'status.json') if (out/'status.json').exists() else {}
    completed=[r for r in previous.get('completed_axes',[]) if r['local_axis_difference_passed']]
    status=dict(state='running',source_bound_plan=source_bound,completed_axes=completed,maximum_live_workers=8,
        priority='low',outer_axis_tolerance=1e-4,full_predictive_integration_admitted=False,independent_confirmation=False)
    try:
        current=replay_for(source_bound)
        baseline=current['raw_recommendation']['policy_totals']['raw']['validation_contrast']
        sync(['performance/check_historical_observed_batch_outer.py','performance/prepare_historical_positive_risk_screen.py',
            'performance/prepare_historical_raw_axis_refinement.py'])
        for mode in (mean_axis,'uppervariance'):
            cached=next((r for r in completed if r['axis']==mode),None)
            if cached:
                source_bound=cached['final_bound_plan'];baseline=cached['raw_validation_contrast']
                continue
            folder=ROOT/'paper_artifacts'/f'historical_positive_risk_{mode}_axis_v1'
            if (folder/'preparation_job.json').exists():
                prepared=read(folder/'refinement_plan.json')
            else:
                prepared=prepare(mode,source_bound);sync(prepared['files'])
            status.update(current_axis=mode,current_plan=prepared['plan'],current_initial_estimated_cpu_seconds=prepared['estimated_initial_phase_cpu_seconds'])
            save(out/'status.json',status);print('LOCAL_AXIS_STARTED '+json.dumps(prepared),flush=True)
            frozen=ROOT/'paper_artifacts'/f'historical_positive_risk_{mode}_axis_followthrough_v2/incumbent_frozen_plan.json'
            if frozen.exists():
                coverage_path=frozen.parent/'coverage_status.json'
                done=read(coverage_path) if coverage_path.exists() else {}
                if done.get('state')!='native_raw_coverage_complete':
                    start=max(done.get('completed_phases',[])+[1])+1
                    runner=ROOT/'paper_artifacts/historical_positive_risk_outer_axes_incumbents_v2/finish_coverage.py'
                    subprocess.run([sys.executable,'-u',str(runner),'--frozen-plan',str(frozen),
                        '--start-phase',str(start),'--last-phase',str(max(4,start+1))],cwd=ROOT,check=True)
                    done=read(coverage_path)
                save(folder/'body_status.json',done)
            else:
                runner=ROOT/'paper_artifacts/historical_positive_risk_body_axis_v1/continue_when_ready.py'
                subprocess.run([sys.executable,'-u',str(runner),'--plan',prepared['plan']],cwd=ROOT,check=True)
                done=read(folder/'body_status.json')
            if done['state']!='native_raw_coverage_complete':
                status.update(state='coverage_hold',coverage=done);break
            ref_plan=done.get('last_reference_plan') or done['latest_axes'][mode]['plan']
            ref=read(ref_plan);source_bound=ref['followup_bound_plan'];current=replay_for(source_bound)
            value=current['raw_recommendation']['policy_totals']['raw']['validation_contrast']
            difference=value-baseline
            row=dict(axis=mode,source_bound_plan=prepared['source_bound_plan'],final_bound_plan=source_bound,
                previous_raw_validation_contrast=baseline,raw_validation_contrast=value,difference=difference,
                local_axis_difference_passed=abs(difference)<=1e-4,risk_observations=prepared['risk_observations'],
                reused_risk_observations=prepared['reused_risk_observations'],new_risk_observations=prepared['new_risk_observations'],
                recommendation=current['raw_recommendation']['policy_totals'],
                full_action_optimality=current['all_conditional_actions_closed'],
                scope='Same state andF16; raw recommendations on admitted candidate pools,checked with cached G E9. Local CDF-axis difference only;full action competition,other intervals,F16 convergence anddeployment remain open.')
            status['completed_axes'].append(row);save(out/(mode+'_analysis.json'),row)
            print('LOCAL_AXIS_COMPLETE '+json.dumps(row),flush=True)
            baseline=value
            if not row['local_axis_difference_passed']:
                status.update(state='outer_axis_hold');break
        else:status.update(state='local_mean_and_upper_variance_checked')
    except Exception as error:
        status.update(state='failed',error=str(error));raise
    finally:
        save(out/'status.json',status)
        print('LOCAL_AXIS_DONE '+json.dumps(status),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-bound',default='performance/manifests/historical_positive_risk_shoulder_axis_bound_incumbents_v4.json')
    parser.add_argument('--mean-axis',choices=('midmean','uppermidmean'),default='midmean')
    args=parser.parse_args();run(args.source_bound,args.mean_axis)
