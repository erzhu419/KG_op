"""Cross-task source calibration and a saved-target repeat/new probe."""

import json
import sys
import time
from pathlib import Path

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import gammainc,logsumexp

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from baselines.transfer_archive import FrozenTransferArchive
from core.budgeted_guarded_kg import BudgetedTerminalActions
from core.raw_likelihood_lookahead import likelihood_query_lookahead
from performance.benchmark_lodo_meta_prior import build_scalarized_problem
from performance.replay_legacy_mixture_budget import build_models,terminal_at_prefix
from performance.replay_legacy_observable_budget import fit_arms
from representation.joint_function_risk import JointFunctionRiskPosterior,joint_batch_log_score
from representation.joint_response_mixture import JointResponseMixture
from representation.legacy_function_prior import legacy_source_labels
from representation.legacy_observable_prior import legacy_exposure_descriptors


def sample_variance_pit(eta_mean,eta_variance,sample_variance,replicates):
    """CDF of observed sample variance, integrating Gaussian latent eta."""
    nodes,weights=hermgauss(40)
    eta=np.asarray(eta_mean)[...,None]+np.sqrt(2*np.asarray(eta_variance))[...,None]*nodes
    k=(replicates-1)/2
    return gammainc(k,k*np.asarray(sample_variance)[...,None]*np.exp(-eta))@(weights/np.sqrt(np.pi))


def no_outcomes(problem):
    def forbidden(*args,**kwargs):
        raise RuntimeError('risk diagnosis cannot simulate or query scoring truth')
    for name in ('simulate','true_objective','true_constraint_mean','true_sigma','is_truly_feasible'):
        setattr(problem,name,forbidden)
    return problem


def source_folds(archive,source_spec,base,names):
    tasks=legacy_source_labels(archive); descriptors=[]
    for task in archive.tasks:
        provider=no_outcomes(build_scalarized_problem(task.name,task.X.shape[1],base['L'],base['sigma'],base['alpha'],base['weights']))
        descriptors.append(legacy_exposure_descriptors(provider,task.X_integer,source_spec['dynamics_grid_dimension']))
    records=[]
    for heldout in range(len(tasks)):
        train=[i for i in range(len(tasks)) if i!=heldout]
        coordinate,arms=fit_arms([tasks[i] for i in train],np.vstack([descriptors[i] for i in train]),source_spec)
        task=tasks[heldout]; raw=archive.tasks[heldout].Y_replicates
        observable=coordinate.transform(descriptors[heldout]); components=[]
        for name in names:
            psi=task['psi'] if name=='legacy_source_orthogonal' else observable
            ps=arms[name]
            components.append((JointFunctionRiskPosterior([ps[0]],[ps[3]],psi),
                JointFunctionRiskPosterior([ps[1]],[ps[2]],psi)))
        model=JointResponseMixture(names,components)
        for i in range(4):
            model.update_raw(i,[raw[i][0,0],raw[i][0,1]-archive.tasks[heldout].tau])
        indices=np.arange(4,len(raw)); component_records=[]; scores=[]; pits=[]; predictions=[]
        for name,(f,g) in zip(names,model.components):
            fm,fv=f.predict(); gm,gv=g.predict()
            predicted=np.column_stack([fm[:,0],gm[:,0],gm[:,1],fm[:,1]])
            local_scores=[]
            for i in indices:
                y=task['response'][i]; count=len(raw[i])
                local_scores.append([joint_batch_log_score(f.local_components(i),y[0],y[3],count),
                    joint_batch_log_score(g.local_components(i),y[1],y[2],count)])
            local_scores=np.asarray(local_scores)
            pit=np.asarray([sample_variance_pit(gm[i,1],gv[i,1],np.var(raw[i][:,1],ddof=1),len(raw[i])) for i in indices])
            component_records.append({'arm':name,'joint_log_score':float(local_scores.sum(axis=1).mean()),
                'head_log_scores':local_scores.mean(axis=0).tolist(),
                'head_bias':(predicted[indices]-task['response'][indices]).mean(axis=0).tolist(),
                'head_mse':np.mean((predicted[indices]-task['response'][indices])**2,axis=0).tolist(),
                'eta_g_pit_mean':float(pit.mean()),'eta_g_pit_low_fraction':float(np.mean(pit<.025)),
                'eta_g_pit_high_fraction':float(np.mean(pit>.975))})
            scores.append(local_scores.sum(axis=1)); pits.append(pit); predictions.append(predicted)
        mixture_pit=model.weights@np.asarray(pits)
        records.append({'heldout_domain':task['name'],'training_domains':[tasks[i]['name'] for i in train],
            'adaptation_calls':4,'scored_sites':len(indices),'model_weights':model.weights.tolist(),
            'components':component_records,
            'mixture_joint_log_score':float(logsumexp(model.log_weights[:,None]+np.asarray(scores),axis=0).mean()),
            'mixture_head_bias':((np.einsum('h,hij->ij',model.weights,np.asarray(predictions))[indices]-task['response'][indices]).mean(axis=0)).tolist(),
            'mixture_eta_g_pit_mean':float(mixture_pit.mean()),
            'mixture_eta_g_pit_low_fraction':float(np.mean(mixture_pit<.025)),
            'mixture_eta_g_pit_high_fraction':float(np.mean(mixture_pit>.975)),
            'fit_converged':{a:[p.diagnostics.get('optimizer_converged') for p in ps] for a,ps in arms.items()},
            'coordinate':coordinate.diagnostics()})
        print('source task heldout complete',task['name'],records[-1]['mixture_eta_g_pit_mean'],flush=True)
    return records


def run():
    started=time.process_time()
    spec=json.loads((ROOT/'performance/manifests/legacy_risk_transfer_diagnosis_v1.json').read_text())
    pilot=json.loads((ROOT/spec['pilot']).read_text()); retention=json.loads((ROOT/spec['retention']).read_text())
    p=pilot['protocol']; replay=json.loads((ROOT/p['replay']).read_text())
    old=json.loads((ROOT/replay['protocol']['paid_paths']).read_text()); base=old['protocol']; source_spec=replay['source_fit_contract']
    names=replay['protocol']['hypotheses']
    archive=FrozenTransferArchive.load(ROOT/'paper_artifacts/legacy_joint_bridge_v1/inputs'/spec['domain']/'archive.json')
    source=source_folds(archive,source_spec,base,names)
    problem,initial,candidates,model,_=build_models(spec['domain'],source_spec,base,names)
    no_outcomes(problem)
    paid=next(r for r in old['rows'] if r['domain']==spec['domain'] and r['arm']=='source_orthogonal' and r['search_calls']==13)
    for index,value in zip(paid['paid_candidate_indices'][:4],paid['paid_observations'][:4]):
        model.update_raw(index,[value[0],value[1]-problem.tau])
    row=next(r for r in pilot['rows'] if r['group']==2 and r['arm']=='observe_primary')
    model.update_raw(row['query'],[row['raw_observation'][0],row['raw_observation'][1]-problem.tau])
    np.testing.assert_allclose(model.weights,row['model_weights'],rtol=1e-12,atol=1e-12)
    original,support=terminal_at_prefix(problem,initial,paid,4,source_spec,len(candidates))
    roster=sorted(set([r['action']['primary'] for r in retention['rows']]+[r['action']['primary'] for r in pilot['rows']]+support))
    mask=np.isin(original.shortlists[:,0],roster)
    terminal=BudgetedTerminalActions(original.shortlists[mask],original.incumbents[mask],original.protocol)
    terminal.retain_recommendation(retention['retained_policy'])
    remaining=spec['total_target_call_cap']-row['search_calls']
    r,c,actions=model.terminal_values(terminal,remaining,spec['reference_draw_power'],spec['seeds'][0])
    best=int(np.lexsort((c,-r))[0]); reference={'action':actions[best],'reward':float(r[best]),'verification_calls':float(c[best])}
    probes=[]
    for seed in spec['seeds']:
        for query in spec['queries']:
            print('repeat/new likelihood probe starts',seed,query,flush=True)
            probe=likelihood_query_lookahead(model,terminal,remaining,query,draw_power=spec['draw_power'],fantasy_power=spec['fantasy_power'],seed=seed)
            probe['seed']=seed; probes.append(probe)
            print('repeat/new likelihood probe complete',seed,query,probe['net_reward_of_paid_episode'],probe['likelihood_tower_drift'],flush=True)
    output={'protocol':spec,'source_folds':source,'target_primary_roster':roster,'target_reference':reference,
        'target_model_weights':model.weights.tolist(),'target_probes':probes,
        'new_simulator_calls':0,'target_scoring_truth_queries':0,'cpu_seconds':time.process_time()-started}
    out=ROOT/spec['output']; out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'source_folds':len(source),'target_probes':len(probes),'cpu_seconds':output['cpu_seconds'],'new_simulator_calls':0}),flush=True)


if __name__=='__main__':
    run()
