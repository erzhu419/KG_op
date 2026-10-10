"""Source-task holdout calibration of joint terminal rewards and call costs."""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from baselines.transfer_archive import FrozenTransferArchive
from core.budgeted_guarded_kg import BudgetedTerminalActions
from core.guarded_deployment_kg import deployment_values
from performance.benchmark_lodo_meta_prior import build_scalarized_problem
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.replay_legacy_mixture_budget import terminal_at_prefix
from performance.replay_legacy_observable_budget import fit_arms
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.joint_response_mixture import JointResponseMixture
from representation.legacy_function_prior import legacy_source_labels
from representation.legacy_observable_prior import legacy_exposure_descriptors


def freeze_roster(model,original,paid_indices,cap,power,seed,size=6):
    reward,cost,actions=model.terminal_values(original,cap-model.paid_episodes,power,seed)
    ranking=np.lexsort((cost,-reward)); retained=actions[int(ranking[0])]['primary']
    roster=list(paid_indices)
    for index in ranking:
        candidate=actions[int(index)]['primary']
        if candidate not in roster: roster.append(candidate)
        if len(roster)==size: break
    mask=np.isin(original.shortlists[:,0],roster)
    terminal=BudgetedTerminalActions(original.shortlists[mask],original.incumbents[mask],original.protocol)
    terminal.retain_recommendation(retained)
    return terminal,retained,roster


def joint_action_draws(f,g,indices,z):
    parts=[]; m=len(indices)
    for offset,field in zip((0,6),(f,g)):
        mean,cov=field.predict(full_cov=True); n=len(mean)
        ix=(2*indices[:,None]+np.arange(2)).reshape(-1)
        C=cov.reshape(2*n,2*n)[np.ix_(ix,ix)]
        parts.append((mean[indices].reshape(-1)+z[:,offset:offset+2*m]@np.linalg.cholesky(C).T).reshape(-1,m,2))
    return np.concatenate(parts,axis=-1)


def fixed_action_value(model,action,protocol,power,seed):
    indices=np.asarray(action['shortlist']); m=len(indices)
    z=ndtri(qmc.Sobol(12,scramble=True,seed=seed).random_base2(power))
    value=np.zeros(2)
    for weight,(f,g) in zip(model.weights,model.components):
        draws=joint_action_draws(f,g,indices,z)
        reward,cost=deployment_values(draws,np.arange(m)[None],np.array([action['incumbent_position']]),
                                      dict(protocol,candidate_budgets=action['budgets']))
        value+=weight*np.array([reward[0],cost[0]])
    return {'reward':float(value[0]),'verification_calls':float(value[1])}


def action_attributions(model,action,protocol,truth,power,seed):
    """Substitute true heads at a frozen action, retaining all other draws."""
    modes={'posterior':[], 'g_mean_truth':[2], 'eta_g_truth':[3],
        'constraint_truth':[2,3], 'objective_truth':[0,1]}
    indices=np.asarray(action['shortlist']); m=len(indices)
    z=ndtri(qmc.Sobol(12,scramble=True,seed=seed).random_base2(power))
    value={mode:np.zeros(2) for mode in modes}
    local_protocol=dict(protocol,candidate_budgets=action['budgets'])
    for weight,(f,g) in zip(model.weights,model.components):
        draws=joint_action_draws(f,g,indices,z)
        for mode,heads in modes.items():
            substituted=draws.copy()
            for head in heads: substituted[:,:,head]=truth[indices,head]
            reward,cost=deployment_values(substituted,np.arange(m)[None],np.array([action['incumbent_position']]),local_protocol)
            value[mode]+=weight*np.array([reward[0],cost[0]])
    return {mode:{'reward':float(v[0]),'verification_calls':float(v[1])} for mode,v in value.items()}


def run(protocol_path='performance/manifests/legacy_source_terminal_calibration_v1.json'):
    started=time.process_time()
    spec=json.loads((ROOT/protocol_path).read_text())
    replay=json.loads((ROOT/spec['replay']).read_text()); source_spec=replay['source_fit_contract']
    base=json.loads((ROOT/replay['protocol']['paid_paths']).read_text())['protocol']
    names=replay['protocol']['hypotheses']; archive=FrozenTransferArchive.load(ROOT/spec['archive'])
    tasks=legacy_source_labels(archive); descriptors=[]; providers=[]
    for task in archive.tasks:
        provider=no_outcomes(build_scalarized_problem(task.name,task.X.shape[1],base['L'],base['sigma'],base['alpha'],base['weights']))
        providers.append(provider)
        descriptors.append(legacy_exposure_descriptors(provider,task.X_integer,source_spec['dynamics_grid_dimension']))
    rows=[]; folds=[]; jobs=[]
    for heldout,task in enumerate(archive.tasks):
        train=[i for i in range(len(tasks)) if i!=heldout]
        coordinate,arms=fit_arms([tasks[i] for i in train],np.vstack([descriptors[i] for i in train]),source_spec)
        observable=coordinate.transform(descriptors[heldout]); components=[]
        for name in names:
            psi=tasks[heldout]['psi'] if name=='legacy_source_orthogonal' else observable
            ps=arms[name]
            components.append((JointFunctionRiskPosterior([ps[0]],[ps[3]],psi),JointFunctionRiskPosterior([ps[1]],[ps[2]],psi)))
        model=JointResponseMixture(names,components)
        count=spec['adaptation_calls']; observed=np.array([values[0] for values in task.Y_replicates[:count]])
        for index,value in enumerate(observed): model.update_raw(index,[value[0],value[1]-task.tau])
        candidates=[tuple(map(int,x)) for x in task.X_integer]
        original,_=terminal_at_prefix(providers[heldout],candidates[:count],{'paid_observations':observed.tolist()},count,source_spec,len(candidates))
        terminal,retained,roster=freeze_roster(model,original,list(range(count)),max(spec['total_target_call_caps']),
            spec['roster_draw_power'],spec['integration_seed'],spec['primary_roster_size'])
        mean,cov=model.predict_joint()
        fold={'heldout_domain':task.name,'training_domains':[tasks[i]['name'] for i in train],
            'model_weights':model.weights.tolist(),'retained_policy':retained,'primary_roster':roster,
            'posterior_mean':mean.tolist(),
            'posterior_variance':[np.diag(cov[i,:,i,:]).tolist() for i in range(len(mean))],
            'coordinate':coordinate.diagnostics(),'fit_converged':{a:[p.diagnostics.get('optimizer_converged') for p in ps] for a,ps in arms.items()}}
        folds.append(fold); current_rows=[]
        for cap in spec['total_target_call_caps']:
            reward,cost,actions=model.terminal_components(terminal,cap-count,spec['terminal_draw_power'],spec['integration_seed'])
            views={name:(reward[i],cost[i]) for i,name in enumerate(names)}
            views['paid_evidence_mixture']=(model.weights@reward,model.weights@cost)
            direct=next(i for i,a in enumerate(actions) if a['kind']=='direct' and a['primary']==retained and a['budgets']==[min(128,cap-count)])
            views['fixed_retained_direct']=views['paid_evidence_mixture']
            for name in spec['selectors']:
                r,c=views[name]
                best=direct if name=='fixed_retained_direct' else int(np.lexsort((c,-r))[0])
                row={'heldout_domain':task.name,'selector':name,'total_target_call_cap':cap,'search_calls':count,
                    'action':actions[best],'action_index':best,'predicted_reward':float(r[best]),
                    'predicted_total_calls':float(c[best]+count)}
                current_rows.append(row); rows.append(row)
        jobs.append((task,model,terminal,current_rows))
        print('source fold recommendations frozen',task.name,roster,retained,flush=True)
    # Source simulator parameters are scoring-only, after all choices have frozen.
    for task,model,terminal,current_rows in jobs:
        provider=build_scalarized_problem(task.name,task.X.shape[1],base['L'],base['sigma'],base['alpha'],base['weights'])
        candidates=[tuple(map(int,x)) for x in task.X_integer]
        truth=np.array([[provider.true_objective(x),np.log(provider.true_sigma(x)[0]**2),
            provider.true_constraint_mean(x)-provider.tau,np.log(provider.true_sigma(x)[1]**2)] for x in candidates])
        fold=next(f for f in folds if f['heldout_domain']==task.name)
        feasible=[i for i,x in enumerate(candidates) if provider.is_truly_feasible(x)]
        fold['support']={'bank_feasible_policies':feasible,
            'roster_feasible_policies':[i for i in fold['primary_roster'] if i in feasible]}
        fold['scoring_truth']=truth.tolist()
        for cap in spec['total_target_call_caps']:
            reward,cost,actions=terminal.evaluate(truth[None],cap-spec['adaptation_calls'])
            best=int(np.lexsort((cost,-reward))[0])
            for row in [r for r in current_rows if r['total_target_call_cap']==cap]:
                a=row['action_index']; primary=row['action']['primary']
                row['truth']={'reward':float(reward[a]),'total_calls':float(cost[a]+row['search_calls']),
                    'primary_feasible':bool(provider.is_truly_feasible(candidates[primary])),
                    'reward_estimation_error':float(row['predicted_reward']-reward[a]),
                    'call_estimation_error':float(row['predicted_total_calls']-cost[a]-row['search_calls']),
                    'loss_to_best_frozen_action':float(reward[best]-reward[a]),'best_frozen_action':actions[best]}
                if row['selector'] in ('paid_evidence_mixture','fixed_retained_direct'):
                    row['attribution']=action_attributions(model,row['action'],terminal.protocol,truth,
                        spec['terminal_draw_power'],spec['integration_seed'])
                    np.testing.assert_allclose(row['attribution']['posterior']['reward'],row['predicted_reward'],rtol=1e-12,atol=1e-12)
                    np.testing.assert_allclose(row['attribution']['posterior']['verification_calls']+row['search_calls'],row['predicted_total_calls'],rtol=1e-12,atol=1e-12)
        print('source fold scoring complete',task.name,flush=True)
    output={'protocol':spec,'folds':folds,'rows':rows,'new_simulator_calls':0,'inventory_truth_queries':0,
        'new_independent_certificates':0,'cpu_seconds':time.process_time()-started}
    out=ROOT/spec['output']; out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'folds':len(folds),'rows':len(rows),'cpu_seconds':output['cpu_seconds'],'new_simulator_calls':0}),flush=True)


if __name__=='__main__':
    run()
