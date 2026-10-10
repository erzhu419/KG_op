"""Retain the pre-update recommendation; replay saved posteriors only."""

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from core.guarded_deployment_kg import deployment_values
from performance.replay_legacy_mixture_budget import build_models,terminal_at_prefix


def run():
    started=time.process_time()
    spec=json.loads((ROOT/'performance/manifests/legacy_recommendation_retention_v1.json').read_text())
    pilot=json.loads((ROOT/spec['input']).read_text()); p=pilot['protocol']
    replay=json.loads((ROOT/p['replay']).read_text()); old=json.loads((ROOT/replay['protocol']['paid_paths']).read_text())
    problem,initial,candidates,model,_=build_models(p['domain'],replay['source_fit_contract'],old['protocol'],replay['protocol']['hypotheses'])
    def no_simulation(*args,**kwargs):
        raise RuntimeError('retention replay cannot execute new simulations')
    problem.simulate=no_simulation
    paid=next(r for r in old['rows'] if r['domain']==p['domain'] and r['arm']=='source_orthogonal' and r['search_calls']==13)
    for index,value in zip(paid['paid_candidate_indices'][:p['prefix']],paid['paid_observations'][:p['prefix']]):
        model.update_raw(index,[value[0],value[1]-problem.tau])
    terminal,_=terminal_at_prefix(problem,initial,paid,p['prefix'],replay['source_fit_contract'],len(candidates))
    baseline=next(r for r in pilot['rows'] if r['arm']=='stop_now')
    retained=baseline['action']['primary']
    terminal.retain_recommendation(retained)
    rows=[]
    states=[baseline]+[r for r in pilot['rows'] if r['arm']=='observe_primary']
    for original in states:
        current=copy.deepcopy(model)
        if original['query'] is not None:
            value=original['raw_observation']
            current.update_raw(original['query'],[value[0],value[1]-problem.tau])
        np.testing.assert_allclose(current.weights,original['model_weights'],rtol=1e-12,atol=1e-12)
        r,c,actions=current.terminal_values(terminal,p['total_target_call_cap']-original['search_calls'],spec['draw_power'],spec['seed'])
        best=int(np.lexsort((c,-r))[0])
        rows.append({'arm':original['arm'],'group':original['group'],'search_calls':original['search_calls'],
            'action':actions[best],'predicted_reward':float(r[best]),'predicted_verification_calls':float(c[best]),
            'original_action':original['action'],'original_truth':original['truth']})
        print('retention recommendation frozen',original['arm'],original['group'],actions[best]['shortlist'],flush=True)
    # Every recommendation is fixed before exact conditional scoring truth.
    for row in rows:
        action=row['action']; points=[candidates[i] for i in action['shortlist']]
        truth=np.array([[problem.true_objective(point),np.log(problem.true_sigma(point)[0]**2),
            problem.true_constraint_mean(point)-problem.tau,np.log(problem.true_sigma(point)[1]**2)] for point in points])[None]
        reward,cost=deployment_values(truth,np.arange(len(points))[None],np.array([action['incumbent_position']]),
            dict(terminal.protocol,candidate_budgets=action['budgets']))
        bound=row['search_calls']+sum(action['budgets'])+(2*terminal.protocol['comparison_budget'] if action['kind']=='guarded' else 0)
        if bound>p['total_target_call_cap']:
            raise RuntimeError('retention action exceeded the shared call cap')
        row['truth_conditional']={'reward':float(reward[0]),'total_calls':float(cost[0]+row['search_calls']),
            'primary_feasible':bool(problem.is_truly_feasible(points[0])),'maximum_total_calls':bound}
    output={'protocol':spec,'retained_policy':retained,'rows':rows,
        'new_simulator_calls':0,'cpu_seconds':time.process_time()-started}
    (ROOT/spec['output']).write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'rows':len(rows),'cpu_seconds':output['cpu_seconds'],'new_simulator_calls':0}),flush=True)


if __name__=='__main__':
    run()
