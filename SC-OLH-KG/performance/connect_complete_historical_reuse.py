"""One cached replay of the complete restored source/raw-posterior/KG chain."""

import copy
import signal
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import digamma

from performance.complete_historical_source_heads import PLAN, ROOT, IDENTITY, read, save, source_case, cpu_timeout
from performance.connect_historical_prior_posterior import (
    BudgetedTerminalActions, JointFunctionRiskPosterior, JointResponseMixture,
    evaluate_or_replicate_action_set, posterior_mixture_weights, restore_selector, snapshot,
)
from performance.check_legacy_f_raw_source_predictive_mixture import project_source_components
from performance.prepare_legacy_f_variance_reference import VarianceReferencePosterior
from performance.check_legacy_f_original_source_raw_update import common_moments
from representation.joint_function_risk import local_joint_moments
from representation.vectorized_joint_function_risk import local_raw_moments
from core.mixture_guarded_kg import mixture_raw_lookahead


def design_for(law,coordinates,maps,selection):
    key='joint_f_g_eta_g_eta_f' if law['support']=='joint' else law['head']
    record=next(r for r in maps if (r['layout'],r['selector'],r['head'])==(law['layout'],law['selector'],key))
    psi=np.column_stack(coordinates) if law['support']=='joint' else coordinates[int(law['head'].startswith('eta'))]
    return np.column_stack([np.ones(len(psi)),restore_selector(record,selection).transform(psi)])


def source_component_weights(states):
    mixture=states['positive_eta_mixture']
    count=np.asarray(mixture['eta_states_by_chain']).shape[0]*np.asarray(mixture['eta_states_by_chain']).shape[1]
    weights=np.asarray(mixture.get('positive_component_weights',np.full(count,1./count)))
    assert weights.shape==(count,) and np.all(weights>0)
    np.testing.assert_allclose(weights.sum(),1.,atol=1e-12,rtol=0.)
    return weights


def restore_head(law,risk,source,states,coordinates,maps,selection):
    field=source_case(law,risk,source); D=design_for(law,coordinates,maps,selection)
    E=design_for(risk,coordinates,maps,selection); Ds=np.asarray(law['source_design']); Es=np.asarray(risk['source_design'])
    A=np.asarray(law['unconditioned_common_covariance']); B=np.asarray(risk['unconditioned_common_covariance'])
    fmean=D@law['unconditioned_common_mean']; qcov=D@(A+law['task_covariance'])@D.T+law['point_discrepancy_variance']*np.eye(len(D))
    cross=D@A@Ds.T; eta=np.asarray(states['positive_eta_mixture']['eta_states_by_chain']).reshape(-1,48)
    rcross=E@B@Es.T; gain=cho_solve(cho_factor(np.asarray(risk['source_covariance']),lower=True),rcross.T).T
    rmean=(E@risk['unconditioned_common_mean'])[None]+(eta-risk['source_mean'])@gain.T
    rcov=E@(B+risk['task_covariance'])@E.T+risk['point_discrepancy_variance']*np.eye(len(E))-gain@rcross.T
    rcov=.5*(rcov+rcov.T);np.linalg.cholesky(rcov)
    model=VarianceReferencePosterior(field['f_means'][0],field['f_covariances'][0],field['batch_means'],field['sample_variances'])
    projection=project_source_components(model,eta,fmean,qcov,cross,rmean,rcov)
    count,n=len(eta),len(D); covariance=np.zeros((count,n,2,n,2))
    covariance[:,:,0,:,0]=projection['f_component_covariances'];covariance[:,:,1,:,1]=projection['risk_component_covariances']
    posterior=JointFunctionRiskPosterior.from_moments(projection['component_means'][0],covariance[0])
    posterior.means=projection['component_means'].reshape(count,2*n).copy()
    posterior.covariances=covariance.reshape(count,2*n,2*n)
    posterior.log_weights=np.log(source_component_weights(states))
    posterior.component_pairs=[(0,0)]*count
    posterior.g_structural=posterior.eta_structural=[True]
    weights=source_component_weights(states);means=projection['component_means'].reshape(count,2*n)
    wm=weights@means;centered=means-wm
    wc=np.einsum('k,kij->ij',weights,posterior.covariances)+(centered.T*weights)@centered
    pm,pc=posterior.predict(full_cov=True)
    np.testing.assert_allclose(pm.reshape(-1),wm,atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(pc.reshape(2*n,2*n),wc,atol=1e-12,rtol=1e-12)
    # Independent old source-to-common reconstruction must recover the same
    # first component, including new-task discrepancy exactly once.
    _,factor,alpha=model.evaluate(eta[0]);K=np.asarray(law['source_covariance'])
    sm=np.asarray(law['source_mean'])+K@alpha;sc=K-K@cho_solve(factor,K)
    cm,cc=common_moments(field,0,sm,sc)
    direct_mean=D@cm;direct_cov=D@(cc+law['task_covariance'])@D.T+law['point_discrepancy_variance']*np.eye(n)
    error=max(float(np.max(abs(direct_mean-projection['component_means'][0,:,0]))),
        float(np.max(abs(direct_cov-projection['f_component_covariances'][0]))))
    assert error<1e-10
    return posterior,error


def check_scalar_batched(field,candidate,label):
    k=[0,len(field.means)//2,len(field.means)-1];ix=[2*candidate,2*candidate+1]
    means=field.means[k][:,ix];covs=field.covariances[k][:,ix][:,:,ix]
    batched=local_raw_moments(means,covs,label);error=0.
    for j,(mean,cov) in enumerate(zip(means,covs)):
        scalar=local_joint_moments(mean,cov,label,None,1)
        for key in ('mean','covariance','log_evidence'):
            error=max(error,float(np.max(abs(np.asarray(scalar[key])-np.asarray(batched[key][j])))))
    assert error<1e-8
    return error


def run(plan=PLAN):
    spec=read(plan);oldplan=read(spec['legacy_connection_plan']);selection=read(spec['selection_plan'])
    maps=read(spec['maps'])['rows'];laws=read(spec['source_laws'])['rows'];registry=read(spec['completed_source_registry'])['rows']
    source_fields=read(read(spec['covariance_plan'])['source_fields'])['rows'];saved_states=read(spec['development_states'])['rows']
    out=ROOT/spec['output']; rows=[];checks=[];models=[];started=time.process_time()
    handler=signal.signal(signal.SIGPROF,cpu_timeout);signal.setitimer(signal.ITIMER_PROF,spec['integration_maximum_single_thread_cpu_seconds'])
    try:
        for cached in saved_states:
            layout=cached['layout'];inputs=read(selection['source_inputs_by_layout'][layout])
            query=next(t for t in inputs['source_tasks'] if t['name']==oldplan['development_task'])
            riskinputs=read(oldplan['query_risk_inputs_by_layout'][layout])
            coordinates=(np.asarray(query['psi'])[:oldplan['candidate_count']],np.asarray(riskinputs['heldout_public_psi'])[cached['candidate_bank_indices']])
            tasks=[t for t in inputs['source_tasks'] if t['name'] in selection['source_tasks']]
            assert [t['name'] for t in tasks]==selection['source_tasks']
            response=np.vstack([t['response'] for t in tasks]);template=next(r for r in source_fields if r['layout']==layout)
            names=[];components=[];usable=[]
            observations=np.asarray(cached['paid_observations'])
            for law in (r for r in laws if r['layout']==layout and r['head']=='f'):
                name='/'.join(law[k] for k in IDENTITY[1:]);pair=[];flags=[]
                for head in ('f','g'):
                    meanlaw=next(r for r in laws if r['head']==head and all(r[k]==law[k] for k in IDENTITY))
                    risklaw=next(r for r in laws if r['head']==('eta_f' if head=='f' else 'eta_g') and all(r[k]==law[k] for k in IDENTITY))
                    case=next(r for r in registry if r['head']==head and all(r[k]==law[k] for k in IDENTITY));source=dict(template)
                    if head=='g':source['batch_means']=response[:,1].tolist();source['sample_variances']=np.exp(response[:,2]+digamma(1.)).tolist()
                    field,projection_error=restore_head(meanlaw,risklaw,source,read(case['component_file']),coordinates,maps,selection)
                    update_error=check_scalar_batched(field,0,observations[0,int(head=='g')])
                    pair.append(field);flags.append(case['source_posterior_usable'])
                    checks.append({**{k:law[k] for k in IDENTITY},'head':head,'source_case':case['component_file'],
                        'source_posterior_usable':case['source_posterior_usable'],'old_common_projection_max_error':projection_error,
                        'scalar_batched_raw_update_max_error':update_error,'retained_source_components':len(field.means),
                        'inverse_tilt_weights_retained':bool(case.get('variance_tilt_corrected',False)),
                        'source_component_weight_ESS':float(1./np.sum(np.exp(field.log_weights)**2))})
                names.append(name);components.append(tuple(pair));usable.append(all(flags))
            mixture=JointResponseMixture(names,components);prior=snapshot(mixture)
            for candidate,value in enumerate(observations):
                before=mixture.weights.copy();record=mixture.update_raw(candidate,value)
                _,expected=posterior_mixture_weights(before,record['joint_log_scores'])
                np.testing.assert_allclose(mixture.weights,expected,atol=1e-12,rtol=0.)
            assert mixture.paid_episodes==oldplan['paid_prefix']
            for pair in mixture.components:
                assert all(field.paid_episodes==oldplan['paid_prefix'] for field in pair)
            # Reuse the paid incumbent and complete shortlist of the earlier
            # cached bridge; only priorities are recomputed under the new law.
            guarded=cached['terminal_actions'][:8]
            terminal=BudgetedTerminalActions([a['shortlist'] for a in guarded],[a['incumbent_position'] for a in guarded],cached['terminal_protocol'])
            remaining=oldplan['total_target_call_cap']-mixture.paid_episodes;kg=spec['mixture_lookahead']
            values,costs,actions=mixture.terminal_values(terminal,remaining,kg['draw_power'],kg['seed'])
            debit,_,future_actions=mixture.terminal_values(terminal,remaining-1,kg['draw_power'],kg['seed'])
            priority=np.full(len(coordinates[0]),np.inf)
            for value,action in zip(values,actions):priority[action['primary']]=min(priority[action['primary']],-float(value))
            # The public problem is required only by the existing action
            # generator, which is never permitted to simulate here.
            from performance.run_legacy_related_inventory_source import problem_for
            raw=read(oldplan['source_data']);public=read(oldplan['public_inputs'])
            candidates=[tuple(raw['bank'][i]) for i in cached['candidate_bank_indices']]
            problem=problem_for(public['plan'],public['tasks'][oldplan['development_task_index']])
            observed=[(candidates[i],v) for i,v in enumerate(observations)]
            action_sets={str(k):evaluate_or_replicate_action_set(candidates,observed,problem,iteration=len(observations),seed=oldplan['seed'],
                new_action_count=k,new_action_policy='canonical_plus_posterior',new_action_priority=priority) for k in (1,4)}
            one,four=(action_sets[str(k)] for k in (1,4))
            assert set(one['active_indices'])<=set(four['active_indices'])
            assert np.array_equal(one['replicate_mask'],four['replicate_mask'])
            assert len(four['new_action_indices'])==4
            row={'layout':layout,'names':names,'source_usable_by_complete_identity':usable,
                'source_HOLD_identities':[name for name,flag in zip(names,usable) if not flag],
                'candidate_bank_indices':cached['candidate_bank_indices'],'paid_observations':observations.tolist(),
                'prior':prior,'posterior':snapshot(mixture),'terminal_protocol':cached['terminal_protocol'],
                'terminal_values':values.tolist(),'terminal_costs':costs.tolist(),'terminal_actions':actions,
                'action_sets':{k:{key:v.tolist() if isinstance(v,np.ndarray) else v for key,v in a.items()} for k,a in action_sets.items()},
                'complete_positive_mixture_used_for_terminal':True,'scientific_model_ranking_authorized':False}
            rows.append(row);models.append((row,mixture,terminal,remaining,(values,costs,actions,debit,future_actions)))
            save(out/'connected_states.json',{'execution_plan':plan,'rows':rows})
            print('complete connected',layout,'heads',2*len(names),'usable identities',sum(usable),'balanced4',four['active_indices'],flush=True)
    finally:
        signal.setitimer(signal.ITIMER_PROF,0.);signal.signal(signal.SIGPROF,handler)
    conditioning_cpu=time.process_time()-started
    # An interrupted replay has already saved completed actions. The source
    # laws, states and registered rule are fixed for this concrete batch.
    lookahead=read(str((out/'mixture_lookahead.json').relative_to(ROOT)))['rows'] if (out/'mixture_lookahead.json').exists() else []
    costs={};stopped=[]
    for row,model,terminal,remaining,baseline in models:
        phase=time.process_time();kg=spec['mixture_lookahead'];handler=signal.signal(signal.SIGPROF,cpu_timeout)
        signal.setitimer(signal.ITIMER_PROF,kg['maximum_single_thread_cpu_seconds_per_layout'])
        try:
            for candidate in row['action_sets']['4']['active_indices']:
                if any(r['layout']==row['layout'] and r['candidate']==candidate for r in lookahead):continue
                replay=mixture_raw_lookahead(model,terminal,remaining,[int(candidate)],draw_power=kg['draw_power'],
                    fantasy_power=kg['fantasy_power'],seed=kg['seed'],baseline=baseline)
                result={**replay['lookahead'][0],'remaining_calls':remaining,'future_remaining_calls':remaining-1,
                    'draw_count':replay['draw_count'],'fantasies_per_identity':replay['fantasies_per_component'],
                    'fantasy_count':replay['fantasy_count'],'numerically_certified_decision_ranking':False}
                lookahead.append({'layout':row['layout'],**result})
                save(out/'mixture_lookahead.json',{'rows':lookahead,'execution_plan':plan,'independent_confirmation':False})
                print('mixture KG',row['layout'],candidate,'net',round(result['net_reward_of_paid_episode'],6),flush=True)
        except TimeoutError:stopped.append(row['layout'])
        finally:
            signal.setitimer(signal.ITIMER_PROF,0.);signal.signal(signal.SIGPROF,handler)
        costs[row['layout']]=time.process_time()-phase
    expected=sum(len(r['action_sets']['4']['active_indices']) for r in rows)
    save(out/'integration_results.json',{'execution_plan':plan,'checks':checks,'all32_heads_connected':len(checks)==32,
        'all12_balanced4_actions_replayed':len(lookahead)==expected==12,'lookahead_actions_completed':len(lookahead),
        'complete_identities':sum(len(r['names']) for r in rows),'usable_complete_identities':sum(sum(r['source_usable_by_complete_identity']) for r in rows),
        'paid_target_episodes_per_layout':oldplan['paid_prefix'],'paid_variance_labels':0,
        'positive_source_mixture_retained':True,'same_paid_and_fantasy_filter':True,'same_terminal_objective_and_budget':True,
        'conditioning_cpu_seconds':conditioning_cpu,'lookahead_cpu_seconds_by_layout':costs,'stopped_layouts':stopped,
        'cost':spec['cost'],'no_model_adopted':True,'independent_confirmation':False,
        'limitations':([str(sum(not c['source_posterior_usable'] for c in checks))+' source heads still HOLD;none was dropped.']
            if any(not c['source_posterior_usable'] for c in checks) else [])+['Eight-policy opened development bank only.',
            'Within-component Gaussian moment projection remains after each raw episode.',
            'One fantasy per identity is an interface rule,not a numerically certified KG comparison.',
            'Source evidence normalizers,sparse hyperparameter convergence,physical factor recovery and independent decision advantage remain unestablished.']})
    return models


if __name__=='__main__':run()
