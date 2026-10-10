"""One actual paid update and independent terminal verification per branch."""

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from core.terminal_verification import verify_frozen_shortlist
from performance.replay_legacy_mixture_budget import build_models,terminal_at_prefix


def paid_update(model,problem,candidates,index,seed,stream_tag):
    updated=copy.deepcopy(model)
    rng=np.random.default_rng(np.random.SeedSequence([seed,stream_tag]))
    observation=np.asarray(problem.simulate(candidates[index],rng),dtype=float)
    evidence=updated.update_raw(index,[observation[0],observation[1]-problem.tau])
    return updated,observation,evidence


def execute_terminal(problem,candidates,action,search_calls,seed,protocol,cap):
    guarded=action['kind']=='guarded'
    comparison=protocol['comparison_budget'] if guarded else 0
    bound=search_calls+sum(action['budgets'])+(2*comparison if guarded else 0)
    if bound>cap:
        raise ValueError('terminal action exceeds shared search/verification cap')
    rows=[{'point':list(candidates[i]),'shortlist_position':j+1,
        'target_oracle_used':False,'verification_samples_used':False}
        for j,i in enumerate(action['shortlist'])]
    deployed,result=verify_frozen_shortlist(problem,rows,seed=seed,search_evaluation_count=search_calls,
        candidate_budgets=action['budgets'],familywise_delta=protocol['familywise_delta'],
        method='normal_quantile_tolerance',objective_incumbent_position=action['incumbent_position']+1 if guarded else None,
        objective_comparison_budget=comparison,objective_comparison_delta=protocol['comparison_delta'])
    if result['total_evaluation_count']>cap:
        raise RuntimeError('actual verifier exceeded total call cap')
    keep=['status','certified','verification_budget','safety_verification_budget',
        'objective_comparison_budget','max_verification_budget','total_evaluation_count',
        'selected_shortlist_rank','search_samples_reused','posterior_updated_from_verification']
    compact={k:result[k] for k in keep}
    compact['attempts']=[{k:a[k] for k in ['candidate_index','certified','sample_count',
        'sample_mean','sample_std','upper_margin','delta','tolerance_factor']} for a in result['attempts']]
    compact['objective_comparison']={k:v for k,v in result['objective_comparison'].items() if 'fingerprint' not in k}
    return candidates.index(tuple(deployed)),compact


def predict_points(model,indices):
    mean,C=model.predict_joint()
    return {str(i):{'mean':mean[i].tolist(),'variance':np.diag(C[i,:,i,:]).tolist()} for i in indices}


def run():
    spec=json.loads((ROOT/'performance/manifests/legacy_one_step_closed_loop_v1.json').read_text())
    replay=json.loads((ROOT/spec['replay']).read_text()); protocol=replay['protocol']
    old=json.loads((ROOT/protocol['paid_paths']).read_text()); started=time.process_time()
    problem,initial,candidates,model,sources=build_models(spec['domain'],replay['source_fit_contract'],old['protocol'],protocol['hypotheses'])
    paid=next(r for r in old['rows'] if r['domain']==spec['domain'] and r['arm']=='source_orthogonal' and r['search_calls']==13)
    for index,value in zip(paid['paid_candidate_indices'][:spec['prefix']],paid['paid_observations'][:spec['prefix']]):
        model.update_raw(index,[value[0],value[1]-problem.tau])
    terminal,_=terminal_at_prefix(problem,initial,paid,spec['prefix'],replay['source_fit_contract'],len(candidates))
    remaining=spec['total_target_call_cap']-spec['prefix']
    baseline_reward,baseline_cost,baseline_actions=model.terminal_values(terminal,remaining,spec['terminal_draw_power'],spec['integration_seed'])
    baseline_best=int(np.lexsort((baseline_cost,-baseline_reward))[0])
    baseline_action=baseline_actions[baseline_best]
    rows=[]; total_simulations=0
    simulate=problem.simulate
    def counted_simulate(point,rng=None):
        nonlocal total_simulations
        total_simulations+=1
        return simulate(point,rng)
    problem.simulate=counted_simulate
    for group,observation_seed in enumerate(spec['observation_seeds']):
        for arm,index in spec['arms'].items():
            current=model; evidence=None; observation=None
            search_calls=spec['prefix']
            if index is not None:
                current,observation,evidence=paid_update(model,problem,candidates,index,observation_seed,spec['observation_stream_tag'])
                search_calls+=1
                r,c,actions=current.terminal_values(terminal,spec['total_target_call_cap']-search_calls,
                    spec['terminal_draw_power'],spec['integration_seed'])
                best=int(np.lexsort((c,-r))[0]); action=actions[best]
                predicted_reward=float(r[best]); predicted_cost=float(c[best])
            else:
                action=baseline_action; predicted_reward=float(baseline_reward[baseline_best]); predicted_cost=float(baseline_cost[baseline_best])
                r,c,actions,best=baseline_reward,baseline_cost,baseline_actions,baseline_best
            other=[i for i,a in enumerate(actions) if a['primary']!=action['primary']]
            rows.append({'group':group,'observation_seed':observation_seed,'arm':arm,'query':index,
                'search_calls':search_calls,'new_search_calls':int(index is not None),
                'raw_observation':None if observation is None else observation.tolist(),'evidence':evidence,
                'model_weights':current.weights.tolist(),'action':action,'predicted_reward':predicted_reward,
                'predicted_verification_calls':predicted_cost,'verification_seed':spec['verification_seeds'][group],
                'reward_gap_to_other_primary':float(r[best]-r[other].max()),
                'posterior_points':predict_points(current,list(dict.fromkeys([2,63,action['primary']]))),
                'initial_baseline_predicted_reward':float(baseline_reward[baseline_best])})
            print('recommendation frozen',group,arm,action['kind'],action['primary'],flush=True)
    new_search_calls=total_simulations
    # Every action has frozen before any terminal stream or scoring truth is read.
    for row in rows:
        deployed,verification=execute_terminal(problem,candidates,row['action'],row['search_calls'],
            row['verification_seed'],terminal.protocol,spec['total_target_call_cap'])
        row['deployed']=deployed; row['verification']=verification
        print('verification complete',row['group'],row['arm'],verification['certified'],
            verification['total_evaluation_count'],flush=True)
    terminal_calls=total_simulations-new_search_calls
    if new_search_calls!=8 or terminal_calls!=sum(r['verification']['verification_budget'] for r in rows):
        raise RuntimeError('actual simulator counts do not match branch accounting')
    truth=np.array([[problem.true_objective(p),np.log(problem.true_sigma(p)[0]**2),
        problem.true_constraint_mean(p)-problem.tau,np.log(problem.true_sigma(p)[1]**2)] for p in candidates])[None]
    for row in rows:
        r,c,actions=terminal.evaluate(truth,spec['total_target_call_cap']-row['search_calls'])
        a=actions.index(row['action']); deployed=row['deployed']; primary=row['action']['primary']
        certified=row['verification']['certified']; feasible=bool(problem.is_truly_feasible(candidates[deployed]))
        quality=float(np.clip(1-truth[0,deployed,0]/terminal.protocol['objective_reference'],0,1))
        row['truth']={'primary_feasible':bool(problem.is_truly_feasible(candidates[primary])),
            'deployed_feasible':feasible,'false_certificate':bool(certified and not feasible),
            'certified_reward':float(certified*quality),'safe_reward':float(certified*feasible*quality),
            'expected_terminal_reward':float(r[a]),'expected_total_calls':float(c[a]+row['search_calls']),
            'primary_g_eta':truth[0,primary,2:].tolist(),'query_g_eta':None if row['query'] is None else truth[0,row['query'],2:].tolist()}
    comparisons=[]
    for group in range(len(spec['observation_seeds'])):
        baseline=next(r for r in rows if r['group']==group and r['arm']=='stop_now')
        for row in [r for r in rows if r['group']==group and r['arm']!='stop_now']:
            comparisons.append({'group':group,'arm':row['arm'],
                'safe_reward_difference':row['truth']['safe_reward']-baseline['truth']['safe_reward'],
                'actual_total_call_difference':row['verification']['total_evaluation_count']-baseline['verification']['total_evaluation_count'],
                'expected_reward_difference':row['truth']['expected_terminal_reward']-baseline['truth']['expected_terminal_reward'],
                'expected_total_call_difference':row['truth']['expected_total_calls']-baseline['truth']['expected_total_calls']})
    result={'protocol':spec,'rows':rows,'paired_comparisons':comparisons,'sources':sources,
        'initial_model_weights':model.weights.tolist(),'new_calls':{'source':0,'search':new_search_calls,'terminal':terminal_calls},
        'cpu_seconds':time.process_time()-started}
    out=ROOT/spec['output']; out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'rows':len(rows),'new_calls':result['new_calls'],'cpu_seconds':result['cpu_seconds']}),flush=True)


if __name__=='__main__':
    run()
