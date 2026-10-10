"""Bounded integration check around the source-solver recommendation change."""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from performance.complete_legacy_related_inventory_joint_population import restore
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.diagnose_legacy_source_terminal import fixed_action_value
from performance.replay_legacy_mixture_budget import terminal_at_prefix
from performance.run_legacy_related_inventory_population_mean import fit_arms
from performance.run_legacy_related_inventory_source import problem_for,save,training_view
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.joint_response_mixture import JointResponseMixture
from representation.legacy_observable_prior import legacy_exposure_descriptors
from representation.source_joint_population import same_source_full_residual_prior


def run():
    started=time.process_time()
    spec=json.loads((ROOT/'performance/manifests/legacy_related_inventory_joint_precision_v1.json').read_text())
    folder=ROOT/spec['completion']
    source=json.loads((folder/'source_fit.json').read_text())
    current=json.loads((folder/'recommendations.json').read_text())
    pilot=json.loads((ROOT/spec['pilot']/'recommendations.json').read_text())
    raw_path=ROOT/spec['data']; data=json.loads(raw_path.read_text())
    frozen=json.loads((raw_path.parent/'frozen_inputs.json').read_text())
    plan=frozen['plan']; source_spec=json.loads((ROOT/frozen['execution']['legacy_replay']).read_text())['source_fit_contract']
    bank=[tuple(x) for x in data['bank']]
    tasks=training_view(data,range(6),plan['task_distribution']['L'])
    descriptors=np.vstack([legacy_exposure_descriptors(no_outcomes(problem_for(plan,data['tasks'][i])),bank,
                           source_spec['dynamics_grid_dimension']) for i in range(6)])
    coordinate,regularizers=fit_arms(tasks,descriptors,source_spec)
    joint=[restore(regularizers['zero_mean_structural'][h].basis,d)
           for h,d in enumerate(source['fit_diagnostics']['joint_population_structural'])]
    arms={'joint_population_structural':joint,'joint_population_full':[same_source_full_residual_prior(p) for p in joint]}
    task=data['tasks'][spec['task_index']]; provider=no_outcomes(problem_for(plan,task))
    psi=coordinate.transform(legacy_exposure_descriptors(provider,bank,source_spec['dynamics_grid_dimension']))
    observed=np.asarray([v[0] for v in task['Y_replicates'][:4]])
    terminal,_=terminal_at_prefix(provider,bank[:4],{'paid_observations':observed.tolist()},4,source_spec,len(bank))
    retained=next(f['retained_policy'] for f in current['folds'] if f['task_id']==task['task_id'])
    terminal.retain_recommendation(retained)
    selected=[next(r['action'] for r in p['rows'] if r['task_id']==task['task_id'] and
                   r['selector']=='joint_population_structural') for p in (current,pilot)]
    primaries=[a['primary'] for a in selected]
    panel=[]
    for kind,tables,incumbents,budgets in terminal.allocations(358):
        for indices,incumbent in zip(tables,incumbents):
            if int(indices[0]) in primaries:
                panel.append({'kind':kind,'primary':int(indices[0]),'shortlist':indices.tolist(),
                              'incumbent_position':int(incumbent),'budgets':budgets})
    output={'protocol':spec,'task_id':task['task_id'],'panel':panel,'rows':[]}
    for name in spec['arms']:
        p=arms[name]
        model=JointResponseMixture([name],[(JointFunctionRiskPosterior([p[0]],[p[3]],psi),
                                          JointFunctionRiskPosterior([p[1]],[p[2]],psi))])
        for i,value in enumerate(observed): model.update_raw(i,[value[0],value[1]-provider.tau])
        for seed in spec['seeds']:
            values=[fixed_action_value(model,action,terminal.protocol,spec['draw_power'],seed) for action in panel]
            r=np.array([v['reward'] for v in values]);c=np.array([v['verification_calls'] for v in values])
            best=int(np.lexsort((c,-r))[0])
            other=[i for i,a in enumerate(panel) if a['primary']!=panel[best]['primary']]
            row={'arm':name,'seed':seed,'values':values,'best_panel_action':panel[best],
                 'reward':float(r[best]),'gap_to_other_primary':float(r[best]-r[other].max())}
            output['rows'].append(row)
            print('precision',name,seed,'primary',panel[best]['primary'],'gap',row['gap_to_other_primary'],flush=True)
    output.update({'cpu_seconds':time.process_time()-started,'new_simulator_calls':0,'scoring_truth_queries':0})
    save(ROOT/spec['output'],output)


if __name__=='__main__':
    run()
