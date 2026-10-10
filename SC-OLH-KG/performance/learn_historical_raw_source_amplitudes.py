"""Reuse historical source Q optimization and retained raw posterior states."""

import signal
import time

import numpy as np
from scipy.linalg import block_diag,cho_factor,solve_triangular
from scipy.optimize import minimize
from scipy.special import digamma,logsumexp

from performance.complete_historical_source_heads import ROOT,IDENTITY,read,save
from performance.check_legacy_f_raw_covariance_mstep import expected_objective
from performance.check_legacy_f_collapsed_source_raw_update import aggregate,CollapsedRawSourcePosterior
from performance.diagnose_legacy_f_raw_source_covariance_score import projected
from performance.resolve_historical_source_variance import weighted_moments,weighted_function_moments
from performance.refine_historical_raw_lookahead import cpu_timeout

PLAN='performance/manifests/historical_raw_source_amplitude_learning_v1.json'


def covariance_builder(law):
    D=np.asarray(law['source_design']);A=np.asarray(law['unconditioned_common_covariance'])
    R=np.asarray(law['task_covariance'])
    directions=[D@A@D.T,block_diag(*[d@R@d.T for d in np.split(D,4)])]
    white=law['point_discrepancy_variance']*np.eye(len(D))
    def builder(theta):return white+sum(t*d for t,d in zip(theta,directions)),directions,{}
    np.testing.assert_allclose(builder([1.,1.])[0],law['source_covariance'],atol=1e-12,rtol=1e-12)
    return builder


def normal_logs(values,mean,covariance):
    root=np.linalg.cholesky(covariance)
    z=solve_triangular(root,(np.asarray(values)-mean).T,lower=True)
    return -.5*(len(mean)*np.log(2*np.pi)+2*np.log(np.diag(root)).sum()+np.sum(z*z,axis=0))


def run(plan=PLAN):
    spec=read(plan);parent=read(spec['source_plan']);out=ROOT/spec['output'];out.mkdir(parents=True,exist_ok=True)
    path=out/'results.json';previous=read(str(path.relative_to(ROOT))) if path.exists() else {}
    if previous.get('all_registered_cases_complete'):return previous
    rows=previous.get('rows',[]);evidence=previous.get('evidence_rows',[])
    laws=read(parent['source_laws'])['rows'];cases=read(parent['completed_source_registry'])['rows']
    selection=read(parent['selection_plan']);scores=read(spec['raw_scores'])['rows']
    started=time.process_time();handler=signal.signal(signal.SIGPROF,cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF,spec['maximum_single_thread_cpu_seconds']);stopped=False
    try:
        for case in cases:
            if any(all(r[k]==case[k] for k in (*IDENTITY,'head')) for r in evidence):continue
            phase=time.process_time();identity={k:case[k] for k in IDENTITY};head=int(case['head']=='g')
            law=next(r for r in laws if r['head']==case['head'] and all(r[k]==case[k] for k in IDENTITY))
            risk=next(r for r in laws if r['head']==('eta_g' if head else 'eta_f') and all(r[k]==case[k] for k in IDENTITY))
            source=read(selection['source_inputs_by_layout'][case['layout']])
            response=np.vstack([t['response'] for t in source['source_tasks'] if t['name'] in selection['source_tasks']])
            labels=response[:,head];s2=np.exp(response[:,2 if head else 3]+digamma(1.))
            model=CollapsedRawSourcePosterior(law['source_mean'],law['source_covariance'],labels,s2)
            states=read(case['component_file'])['positive_eta_mixture'];eta=np.asarray(states['eta_states_by_chain'])
            count=eta.shape[1];weights=(np.asarray(states['positive_component_weights']).reshape(eta.shape[:2])
                if 'positive_component_weights' in states else None)
            evaluations=[model.evaluate(x) for x in eta.reshape(-1,48)]
            moments=[model.moments(v) for v in evaluations]
            ds=np.stack([m[0] for m in moments]).reshape(*eta.shape)
            cs=np.stack([m[1] for m in moments]).reshape(*eta.shape[:2],48,48)
            builders=[];learned=[];gates=[]
            for current in (law,risk):
                builder=covariance_builder(current);builders.append(builder)
                mean=np.asarray(current['source_mean'])
                def second(part):
                    if current is law:
                        if weights is None:delta,cov,_,_=aggregate(ds[:,part],cs[:,part])
                        else:delta,cov=weighted_function_moments(ds[:,part],cs[:,part],weights[:,part])
                        pm=model.mean+delta
                    else:
                        if weights is None:
                            flat=eta[:,part].reshape(-1,48);pm=flat.mean(axis=0);cov=np.cov(flat,rowvar=False)
                        else:pm,cov=weighted_moments(eta[:,part],weights[:,part])
                    return cov+np.outer(pm-mean,pm-mean)
                full=second(slice(None));halves=[second(slice(0,count//2)),second(slice(count//2,None))]
                old,gradient=expected_objective(np.ones(2),builder,full)
                recorded=next(r for r in scores if r['head']==current['head'] and all(r[k]==case[k] for k in IDENTITY))
                error=float(np.max(abs(gradient-recorded['common_task_log_amplitude_NLL_gradient_per_batch'])))
                if error>1e-8:raise RuntimeError('Historical Q gradient does not reproduce the current raw source score')
                settings=spec['optimizer']
                fit=minimize(lambda t:expected_objective(t,builder,full),np.ones(2),method='L-BFGS-B',jac=True,
                    bounds=[(0.,None)]*2,options={'maxiter':settings['max_iterations'],
                        'ftol':settings['ftol'],'gtol':settings['gtol']})
                value,gradient=expected_objective(fit.x,builder,full)
                improvements=[old-value,*[expected_objective(np.ones(2),builder,s)[0]
                    -expected_objective(fit.x,builder,s)[0] for s in halves]]
                gate=bool(fit.success and min(improvements)>0.);gates.append(gate);learned.append(fit.x)
                rows.append({**identity,'head':current['head'],'source_case':case['component_file'],
                    'common_task_covariance_scales':fit.x.tolist(),'old_expected_complete_NLL_per_batch':old,
                    'learned_expected_complete_NLL_per_batch':value,'full_and_split_Q_improvements':improvements,
                    'optimizer_converged':bool(fit.success),'optimizer_iterations':int(fit.nit),
                    'optimizer_message':str(fit.message),'initial_raw_score_identity_max_error':error,
                    'projected_gradient':projected(fit.x,gradient,np.full(2,np.inf)).tolist(),
                    'source_Mstep_gate_passed':gate,'sparse_support_and_basis_unchanged':True})
            # The new law is conditioned at each retained eta. Old f moments
            # are used for Q only and never relabelled as the new posterior.
            Kf=builders[0](learned[0])[0];Ke=builders[1](learned[1])[0]
            new=CollapsedRawSourcePosterior(law['source_mean'],Kf,labels,s2)
            flat=eta.reshape(-1,48);new_evaluations=[new.evaluate(x) for x in flat]
            logratio=np.array([v[0]-u[0] for v,u in zip(new_evaluations,evaluations)])
            logratio+=normal_logs(flat,np.asarray(risk['source_mean']),Ke)
            logratio-=normal_logs(flat,np.asarray(risk['source_mean']),np.asarray(risk['source_covariance']))
            base=np.full(eta.shape[:2],1/len(flat)) if weights is None else weights
            logbase=np.log(base.reshape(-1));logZ=float(logsumexp(logbase+logratio))
            posterior_weights=np.exp(logbase+logratio-logZ)
            split=[]
            ratio=logratio.reshape(eta.shape[:2])
            for part in (slice(0,count//2),slice(count//2,None)):
                w=base[:,part];split.append(float(logsumexp(np.log(w/w.sum())+ratio[:,part])))
            ess=float(1/(posterior_weights@posterior_weights));gap=abs(split[0]-split[1])
            precision=spec['evidence_precision']
            adequate=bool(ess>=precision['minimum_importance_ESS'] and gap<=precision['maximum_split_log_ratio_difference'])
            evidence.append({**identity,'head':case['head'],'source_case':case['component_file'],
                'f_common_task_scales':learned[0].tolist(),'eta_common_task_scales':learned[1].tolist(),
                'log_source_evidence_ratio_new_vs_old':logZ,'split_log_evidence_ratios':split,
                'split_log_ratio_difference':gap,'importance_ESS':ess,
                'source_Mstep_pair_passed':all(gates),'evidence_ratio_precision_passed':adequate,
                'new_positive_eta_weights':posterior_weights.tolist(),
                'new_collapsed_f_likelihood_recomputed':True,'posterior_precision_certified':False,
                'cpu_seconds':time.process_time()-phase,
                'normalizer_scope':'Ratio Z(new)/Z(old) within one original raw-source model;not absolute evidence across identities.'})
            save(path,{'execution_plan':plan,'rows':rows,'evidence_rows':evidence,'all_registered_cases_complete':len(evidence)==32})
    except TimeoutError:stopped=True
    finally:
        signal.setitimer(signal.ITIMER_PROF,0.);signal.signal(signal.SIGPROF,handler)
    result={'execution_plan':plan,'rows':rows,'evidence_rows':evidence,'all_registered_cases_complete':len(evidence)==32,
        'source_Mstep_gates_passed':sum(r['source_Mstep_gate_passed'] for r in rows),
        'evidence_ratio_precision_cases_passed':sum(r['evidence_ratio_precision_passed'] for r in evidence),
        'stopped_at_cpu_budget':stopped,'cpu_seconds':time.process_time()-started,'cost':spec['cost'],
        'no_model_adopted':True,'independent_confirmation':False,'scientific_model_ranking_authorized':False,
        'limitations':['One raw-source MCEM M step;no iterative convergence or changed spectral exponent/support.',
            'Importance ESS and split normalizer checks do not certify all posterior functional precision.',
            'Within-model source evidence ratios do not supply absolute evidence for model-identity ranking.',
            'All candidate laws require conditioned-posterior precision checks before target deployment.']}
    save(path,result)
    print('raw source Mstep',len(rows),'Q passes',result['source_Mstep_gates_passed'],
        'evidence ratio passes',result['evidence_ratio_precision_cases_passed'],'/',len(evidence),flush=True)
    return result


if __name__=='__main__':run()
