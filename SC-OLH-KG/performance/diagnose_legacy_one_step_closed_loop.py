"""Selected-posterior attribution; truth substitutions are diagnostic only."""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from core.guarded_deployment_kg import deployment_values
from performance.replay_legacy_mixture_budget import build_models,terminal_at_prefix


def run():
    started=time.process_time()
    spec=json.loads((ROOT/'performance/manifests/legacy_one_step_attribution_v1.json').read_text())
    pilot=json.loads((ROOT/spec['input']).read_text()); p=pilot['protocol']
    replay=json.loads((ROOT/p['replay']).read_text()); old=json.loads((ROOT/replay['protocol']['paid_paths']).read_text())
    row=next(r for r in pilot['rows'] if r['group']==spec['group'] and r['arm']==spec['arm'])
    problem,initial,candidates,model,_=build_models(p['domain'],replay['source_fit_contract'],old['protocol'],replay['protocol']['hypotheses'])
    def no_simulation(*args,**kwargs):
        raise RuntimeError('attribution cannot generate physical observations')
    problem.simulate=no_simulation
    paid=next(r for r in old['rows'] if r['domain']==p['domain'] and r['arm']=='source_orthogonal' and r['search_calls']==13)
    for index,value in zip(paid['paid_candidate_indices'][:p['prefix']],paid['paid_observations'][:p['prefix']]):
        model.update_raw(index,[value[0],value[1]-problem.tau])
    model.update_raw(row['query'],[row['raw_observation'][0],row['raw_observation'][1]-problem.tau])
    np.testing.assert_allclose(model.weights,row['model_weights'],rtol=1e-12,atol=1e-12)
    terminal,_=terminal_at_prefix(problem,initial,paid,p['prefix'],replay['source_fit_contract'],len(candidates))
    # The first two are pilot actions. The third only measures retaining the
    # pre-update recommendation, without changing the objective incumbent.
    tables=[[50,3,2],[63,3,2],[50,3,63]]
    z=ndtri(qmc.Sobol(12,scramble=True,seed=spec['seed']).random_base2(spec['draw_power']))
    frozen=[]
    for f,g in model.components:
        fm,fc=f.predict(full_cov=True); gm,gc=g.predict(full_cov=True); n=len(fm)
        fc=fc.reshape(2*n,2*n); gc=gc.reshape(2*n,2*n); local=[]
        for indices in tables:
            ix=(2*np.asarray(indices)[:,None]+np.arange(2)).reshape(-1)
            fd=fm[indices].reshape(-1)+z[:,:6]@np.linalg.cholesky(fc[np.ix_(ix,ix)]).T
            gd=gm[indices].reshape(-1)+z[:,6:]@np.linalg.cholesky(gc[np.ix_(ix,ix)]).T
            local.append(np.concatenate([fd.reshape(-1,3,2),gd.reshape(-1,3,2)],axis=-1))
        frozen.append(local)
    # All candidate roles and draws are fixed before reading scoring truth.
    points=list(dict.fromkeys(i for table in tables for i in table))
    truth={i:np.array([problem.true_objective(candidates[i]),np.log(problem.true_sigma(candidates[i])[0]**2),
        problem.true_constraint_mean(candidates[i])-problem.tau,np.log(problem.true_sigma(candidates[i])[1]**2)]) for i in points}
    substitutions={'unchanged':[], 'true_eta_g_63':[(63,3)],
        'true_mean_g_50':[(50,2)], 'true_mean_f_50':[(50,0)],
        'both_g_substitutions':[(63,3),(50,2)]}
    protocol=dict(terminal.protocol,candidate_budgets=row['action']['budgets'])
    scenario_results={}
    for name,changes in substitutions.items():
        rewards,costs=[],[]
        for local in frozen:
            r,c=[],[]
            for table,draws in zip(tables,local):
                draws=draws.copy()
                for candidate,head in changes:
                    if candidate in table:
                        draws[:,table.index(candidate),head]=truth[candidate][head]
                rv,cv=deployment_values(draws,np.arange(3)[None],np.array([1]),protocol)
                r.append(float(rv[0])); c.append(float(cv[0]))
            rewards.append(r); costs.append(c)
        scenario_results[name]={'component_rewards':rewards,'mixture_rewards':(model.weights@np.array(rewards)).tolist(),
            'expected_verification_calls':(model.weights@np.array(costs)).tolist()}
    exact=[]
    for table in tables:
        rv,cv=deployment_values(np.array([[truth[i] for i in table]]),np.arange(3)[None],np.array([1]),protocol)
        exact.append({'reward':float(rv[0]),'total_calls':float(cv[0]+row['search_calls'])})
    output={'protocol':spec,'shortlists':tables,'model_weights':model.weights.tolist(),'scenarios':scenario_results,
        'truth_conditional_reference':exact,'point_truth':{str(i):{'values':truth[i].tolist(),
            'truly_feasible':bool(problem.is_truly_feasible(candidates[i]))} for i in points},
        'new_simulator_calls':0,'cpu_seconds':time.process_time()-started}
    (ROOT/spec['output']).write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    run()
