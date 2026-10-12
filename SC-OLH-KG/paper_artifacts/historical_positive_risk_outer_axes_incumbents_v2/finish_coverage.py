"""Finish existing native recommendation coverage in successive bounded phases."""
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
out=Path(__file__).resolve().parent
helper=ROOT/'paper_artifacts/historical_positive_risk_outer_axes_v1'
status=dict(state='running',completed_phases=[],maximum_live_workers=8,priority='low',
    full_predictive_integration_admitted=False,independent_confirmation=False)
active=['mean','variance'];latest={}
try:
    for phase in range(2,5):
        if phase>2:
            subprocess.run([sys.executable,str(helper/'prepare_next_incumbents.py'),'--phase',str(phase),'--axes',*active],cwd=ROOT,check=True)
        folder=ROOT/f'paper_artifacts/historical_positive_risk_outer_axes_incumbents_v{phase}'
        frozen_path=folder/'incumbent_frozen_plan.json';frozen=json.loads(frozen_path.read_text())
        status.update(current_phase=phase)
        (out/'coverage_status.json').write_text(json.dumps(status,indent=2)+'\n')
        subprocess.run([sys.executable,str(helper/'validate_incumbents.py'),'--frozen-plan',str(frozen_path)],cwd=ROOT,check=True)
        result=json.loads((folder/'incumbent_status.json').read_text())
        assert result['state']=='incumbent_probes_complete',result
        active=[]
        for row in frozen['rows']:
            spec=json.loads((ROOT/row['plan']).read_text());ref=ROOT/spec['output']
            completion=json.loads((ref/'completion.json').read_text())
            replay=json.loads((ref/'incumbent_replay_final.json').read_text())
            latest[row['axis']]=dict(phase=phase,plan=row['plan'],missing_raw_observations=completion['missing_raw_observations'],
                raw_covered_grid_weight=completion['raw_covered_grid_weight'],
                weighted_raw_reward_of_native_incumbent_contrast=replay['weighted_raw_reward_of_native_incumbent_contrast'],
                weighted_raw_best_admitted_contrast=replay['weighted_raw_best_admitted_contrast'],
                all_conditional_actions_closed=replay['all_conditional_actions_closed'])
            if not completion['native_choices_all_raw_referenced']:active.append(row['axis'])
        status['completed_phases'].append(phase);status['latest_axes']=latest
        print('COVERAGE_PHASE_DONE '+json.dumps(status),flush=True)
        if not active:
            status.update(state='native_raw_coverage_complete');break
    else:status.update(state='partial_coverage',remaining_axes=active)
except Exception as error:
    status.update(state='failed',error=str(error));raise
finally:
    (out/'coverage_status.json').write_text(json.dumps(status,indent=2)+'\n')
    print('COVERAGE_DONE '+json.dumps(status),flush=True)
