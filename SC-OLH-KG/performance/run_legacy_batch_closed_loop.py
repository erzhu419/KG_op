"""Execute frozen stop/new/repeated-batch arms from the saved paid posterior."""

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from core.budgeted_guarded_kg import BudgetedTerminalActions
from performance.replay_legacy_mixture_budget import build_models,terminal_at_prefix
from performance.run_legacy_one_step_closed_loop import execute_terminal,predict_points


def paid_batch_update(model,problem,candidates,index,count,seed,stream_tag):
    updated=copy.deepcopy(model)
    rng=np.random.default_rng(np.random.SeedSequence([seed,stream_tag]))
    observations=np.asarray([problem.simulate(candidates[index],rng) for _ in range(count)],dtype=float)
    centered=observations.copy(); centered[:,1]-=problem.tau
    evidence=updated.update_replicates(index,centered)
    return updated,observations,evidence


def select_terminal(model,terminal,search_calls,spec):
    reward,cost,actions=model.terminal_values(terminal,spec['total_target_call_cap']-search_calls,
        spec['terminal_draw_power'],spec['integration_seed'])
    best=int(np.lexsort((cost,-reward))[0]); action=actions[best]
    other=[i for i,a in enumerate(actions) if a['primary']!=action['primary']]
    return {'action':action,'predicted_reward':float(reward[best]),
        'predicted_verification_calls':float(cost[best]),
        'reward_gap_to_other_primary':float(reward[best]-reward[other].max())}


def score_frozen_rows(problem,candidates,terminal,rows,cap):
    # All terminal observations have already been consumed, with no posterior updates.
    truth=np.array([[problem.true_objective(p),np.log(problem.true_sigma(p)[0]**2),
        problem.true_constraint_mean(p)-problem.tau,np.log(problem.true_sigma(p)[1]**2)]
        for p in candidates])[None]
    for row in rows:
        action=row['action']; primary=action['primary']; deployed=row['deployed']
        reward,cost,actions=terminal.evaluate(truth,cap-row['search_calls'])
        position=actions.index(action); certified=row['verification']['certified']
        feasible=bool(problem.is_truly_feasible(candidates[deployed]))
        quality=float(np.clip(1-truth[0,deployed,0]/terminal.protocol['objective_reference'],0,1))
        row['truth']={'primary_feasible':bool(problem.is_truly_feasible(candidates[primary])),
            'deployed_feasible':feasible,'false_certificate':bool(certified and not feasible),
            'certified_reward':float(certified*quality),'safe_reward':float(certified*feasible*quality),
            'expected_terminal_reward':float(reward[position]),
            'expected_total_calls':float(cost[position]+row['search_calls']),
            'primary_g_eta':truth[0,primary,2:].tolist()}


def paired_comparisons(rows):
    comparisons=[]
    for group in sorted(set(r['group'] for r in rows)):
        baseline=next(r for r in rows if r['group']==group and r['arm']=='stop_now')
        for row in [r for r in rows if r['group']==group and r['arm']!='stop_now']:
            comparisons.append({'group':group,'arm':row['arm'],
                'safe_reward_difference':row['truth']['safe_reward']-baseline['truth']['safe_reward'],
                'actual_total_call_difference':row['verification']['total_evaluation_count']-baseline['verification']['total_evaluation_count'],
                'expected_reward_difference':row['truth']['expected_terminal_reward']-baseline['truth']['expected_terminal_reward'],
                'expected_total_call_difference':row['truth']['expected_total_calls']-baseline['truth']['expected_total_calls']})
    return comparisons


def saved_start(spec):
    """Reconstruct the shared archived posterior and retained terminal class."""
    diagnosis=json.loads((ROOT/spec['diagnosis']).read_text()); pilot=json.loads((ROOT/spec['pilot']).read_text())
    p=pilot['protocol']; replay=json.loads((ROOT/p['replay']).read_text())
    old=json.loads((ROOT/replay['protocol']['paid_paths']).read_text()); source_spec=replay['source_fit_contract']
    problem,initial,candidates,model,sources=build_models(p['domain'],source_spec,old['protocol'],replay['protocol']['hypotheses'])
    paid=next(r for r in old['rows'] if r['domain']==p['domain'] and r['arm']=='source_orthogonal' and r['search_calls']==13)
    prefix=spec['initial_prefix']
    for index,value in zip(paid['paid_candidate_indices'][:prefix],paid['paid_observations'][:prefix]):
        model.update_raw(index,[value[0],value[1]-problem.tau])
    saved=next(r for r in pilot['rows'] if r['group']==spec['saved_group'] and r['arm']==spec['saved_arm'])
    model.update_raw(saved['query'],[saved['raw_observation'][0],saved['raw_observation'][1]-problem.tau])
    np.testing.assert_allclose(model.weights,diagnosis['target_model_weights'],rtol=1e-12,atol=1e-12)
    if model.paid_episodes!=spec['starting_paid_calls']:
        raise RuntimeError('saved posterior does not match the frozen starting call count')
    original,_=terminal_at_prefix(problem,initial,paid,prefix,source_spec,len(candidates))
    mask=np.isin(original.shortlists[:,0],diagnosis['target_primary_roster'])
    terminal=BudgetedTerminalActions(original.shortlists[mask],original.incumbents[mask],original.protocol)
    terminal.retain_recommendation(spec['retained_policy'])
    return problem,candidates,model,terminal,sources


def run():
    started=time.process_time()
    spec=json.loads((ROOT/'performance/manifests/legacy_batch_closed_loop_v1.json').read_text())
    problem,candidates,model,terminal,sources=saved_start(spec)
    baseline=select_terminal(model,terminal,model.paid_episodes,spec)
    rows=[]; total_simulations=0
    simulate=problem.simulate
    def counted_simulate(point,rng=None):
        nonlocal total_simulations
        total_simulations+=1
        return simulate(point,rng)
    problem.simulate=counted_simulate
    for group,seed in enumerate(spec['observation_seeds']):
        for arm,intervention in spec['arms'].items():
            current=model; observations=None; evidence=None; decision=baseline
            count=intervention['replicates']; query=intervention['query']
            if count:
                current,observations,evidence=paid_batch_update(model,problem,candidates,query,count,seed,spec['observation_stream_tag'])
                decision=select_terminal(current,terminal,current.paid_episodes,spec)
            rows.append({'group':group,'observation_seed':seed,'arm':arm,'query':query,
                'new_search_calls':count,'search_calls':current.paid_episodes,
                'raw_observations':None if observations is None else observations.tolist(),'evidence':evidence,
                'model_weights':current.weights.tolist(),'verification_seed':spec['verification_seeds'][group],
                'posterior_points':predict_points(current,list(dict.fromkeys([50,63,decision['action']['primary']]))),
                **decision})
            print('recommendation frozen',group,arm,decision['action']['primary'],decision['action']['shortlist'],flush=True)
    new_search_calls=total_simulations
    if new_search_calls!=sum(r['new_search_calls'] for r in rows):
        raise RuntimeError('fresh batch simulator calls do not match the registered branch debits')
    for row in rows:
        row['deployed'],row['verification']=execute_terminal(problem,candidates,row['action'],row['search_calls'],
            row['verification_seed'],terminal.protocol,spec['total_target_call_cap'])
        print('verification complete',row['group'],row['arm'],row['verification']['certified'],
            row['verification']['total_evaluation_count'],flush=True)
    terminal_calls=total_simulations-new_search_calls
    if terminal_calls!=sum(r['verification']['verification_budget'] for r in rows):
        raise RuntimeError('terminal simulator calls do not match verifier accounting')
    score_frozen_rows(problem,candidates,terminal,rows,spec['total_target_call_cap'])
    result={'protocol':spec,'rows':rows,'paired_comparisons':paired_comparisons(rows),'source_fit':sources,
        'initial_model_weights':model.weights.tolist(),'initial_posterior_points':predict_points(model,[50,63]),
        'new_calls':{'source':0,'search':new_search_calls,'terminal':terminal_calls},
        'cpu_seconds':time.process_time()-started}
    out=ROOT/spec['output']; out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'rows':len(rows),'new_calls':result['new_calls'],'cpu_seconds':result['cpu_seconds']}),flush=True)


if __name__=='__main__':
    run()
