"""Reuse saved observations across a frozen target-call budget frontier."""

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from performance.run_legacy_batch_closed_loop import saved_start,select_terminal


def run():
    started=time.process_time()
    spec=json.loads((ROOT/'performance/manifests/legacy_batch_budget_frontier_v1.json').read_text())
    closed=json.loads((ROOT/spec['closed_loop']).read_text()); p=closed['protocol']
    problem,candidates,model,terminal,_=saved_start(p)
    def forbidden(*args,**kwargs):
        raise RuntimeError('saved frontier cannot execute new simulations')
    problem.simulate=forbidden
    rows=[]; baseline={}
    for original in closed['rows']:
        current=model
        if original['new_search_calls']:
            current=copy.deepcopy(model)
            values=np.asarray(original['raw_observations']); values[:,1]-=problem.tau
            current.update_replicates(original['query'],values)
        np.testing.assert_allclose(current.weights,original['model_weights'],rtol=1e-12,atol=1e-12)
        if current.paid_episodes!=original['search_calls']:
            raise RuntimeError('frontier replay changed the original physical debit')
        for cap in spec['total_target_call_caps']:
            if original['arm']=='stop_now' and cap in baseline:
                decision=baseline[cap]
            else:
                decision=select_terminal(current,terminal,current.paid_episodes,dict(spec,total_target_call_cap=cap))
                if original['arm']=='stop_now': baseline[cap]=decision
            action=decision['action']; reserve=2*terminal.protocol['comparison_budget'] if action['kind']=='guarded' else 0
            if current.paid_episodes+sum(action['budgets'])+reserve>cap:
                raise RuntimeError('frontier action exceeds its frozen total cap')
            rows.append({'group':original['group'],'arm':original['arm'],'total_target_call_cap':cap,
                'search_calls':current.paid_episodes,**decision})
        print('frontier recommendations frozen',original['group'],original['arm'],flush=True)
    truth=np.array([[problem.true_objective(x),np.log(problem.true_sigma(x)[0]**2),
        problem.true_constraint_mean(x)-problem.tau,np.log(problem.true_sigma(x)[1]**2)] for x in candidates])[None]
    for row in rows:
        reward,cost,actions=terminal.evaluate(truth,row['total_target_call_cap']-row['search_calls'])
        position=actions.index(row['action'])
        row['truth']={'expected_terminal_reward':float(reward[position]),
            'expected_total_calls':float(cost[position]+row['search_calls']),
            'primary_feasible':bool(problem.is_truly_feasible(candidates[row['action']['primary']]))}
    paired=[]
    for row in rows:
        if row['arm']=='stop_now': continue
        stop=next(r for r in rows if r['arm']=='stop_now' and r['group']==row['group'] and
            r['total_target_call_cap']==row['total_target_call_cap'])
        paired.append({'group':row['group'],'arm':row['arm'],'total_target_call_cap':row['total_target_call_cap'],
            'expected_reward_difference':row['truth']['expected_terminal_reward']-stop['truth']['expected_terminal_reward'],
            'expected_total_call_difference':row['truth']['expected_total_calls']-stop['truth']['expected_total_calls']})
    output={'protocol':spec,'rows':rows,'paired_comparisons':paired,
        'new_simulator_calls':0,'new_independent_certificates':0,'cpu_seconds':time.process_time()-started}
    (ROOT/spec['output']).write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'rows':len(rows),'new_simulator_calls':0,'cpu_seconds':output['cpu_seconds']}),flush=True)


if __name__=='__main__':
    run()
