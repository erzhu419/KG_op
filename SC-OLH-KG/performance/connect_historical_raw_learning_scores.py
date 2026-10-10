"""Connect the historical raw covariance score to every restored source head."""

import signal
import time

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve
from scipy.special import digamma

from performance.complete_historical_source_heads import PLAN, ROOT, IDENTITY, read, save, cpu_timeout
from performance.check_legacy_f_collapsed_source_raw_update import aggregate
from performance.prepare_legacy_f_variance_reference import VarianceReferencePosterior
from performance.diagnose_legacy_f_raw_source_covariance_score import score
from performance.resolve_historical_source_variance import weighted_moments, weighted_function_moments


def run(plan=PLAN):
    spec=read(plan);execution=spec['raw_learning_diagnostic'];laws=read(spec['source_laws'])['rows']
    cases=read(spec['completed_source_registry'])['rows'];selection=read(spec['selection_plan'])
    rows=[];started=time.process_time();handler=signal.signal(signal.SIGPROF,cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF,execution['maximum_single_thread_cpu_seconds'])
    try:
        for case in cases:
            law=next(r for r in laws if r['head']==case['head'] and all(r[k]==case[k] for k in IDENTITY))
            risk=next(r for r in laws if r['head']==('eta_f' if case['head']=='f' else 'eta_g') and all(r[k]==case[k] for k in IDENTITY))
            inputs=read(selection['source_inputs_by_layout'][case['layout']])
            tasks=[t for t in inputs['source_tasks'] if t['name'] in selection['source_tasks']]
            response=np.vstack([t['response'] for t in tasks]);head=int(case['head']=='g')
            labels=response[:,head];s2=np.exp(response[:,2 if head else 3]+digamma(1.))
            model=VarianceReferencePosterior(law['source_mean'],law['source_covariance'],labels,s2)
            mixture=read(case['component_file'])['positive_eta_mixture']
            states=np.asarray(mixture['eta_states_by_chain']);count=states.shape[1]
            weights=(np.asarray(mixture['positive_component_weights']).reshape(states.shape[:2])
                if 'positive_component_weights' in mixture else None)
            moments=[[model.moments(model.evaluate(eta)) for eta in chain] for chain in states]
            ds=np.asarray([[m[0] for m in chain] for chain in moments]);cs=np.asarray([[m[1] for m in chain] for chain in moments])
            for current in (law,risk):
                D=np.asarray(current['source_design']);A=np.asarray(current['unconditioned_common_covariance']);R=np.asarray(current['task_covariance'])
                directions=[D@A@D.T,block_diag(*[d@R@d.T for d in np.split(D,4)])]
                K=np.asarray(current['source_covariance']);mean=np.asarray(current['source_mean'])
                np.testing.assert_allclose(sum(directions)+current['point_discrepancy_variance']*np.eye(48),K,atol=1e-12,rtol=1e-12)
                if current is law:
                    def summary(part):
                        if weights is None:delta,cov,_,_=aggregate(ds[:,part],cs[:,part])
                        else:delta,cov=weighted_function_moments(ds[:,part],cs[:,part],weights[:,part])
                        return model.mean+delta,cov
                    half_moments=[summary(slice(0,count//2)),summary(slice(count//2,None))];pm,pc=summary(slice(None))
                    factor=cho_factor(K+np.diag(s2/3),lower=True);alpha=cho_solve(factor,labels-mean)
                    gm=mean+K@alpha;gc=K-K@cho_solve(factor,K)
                    plugin=score(mean,K,directions,gm,gc)
                    direct=np.asarray([.5*(np.trace(cho_solve(factor,d))-alpha@d@alpha)/48 for d in directions])
                    np.testing.assert_allclose(plugin,direct,atol=1e-9,rtol=1e-9)
                    identity_error=float(np.max(abs(plugin-direct)))
                else:
                    def summary(part):
                        if weights is not None:return weighted_moments(states[:,part],weights[:,part])
                        flat=states[:,part].reshape(-1,48);return flat.mean(axis=0),np.cov(flat,rowvar=False)
                    pm,pc=summary(slice(None))
                    half_moments=[summary(slice(0,count//2)),summary(slice(count//2,None))];identity_error=None
                gradient=score(mean,K,directions,pm,pc)
                halves=[score(mean,K,directions,m,c) for m,c in half_moments]
                rows.append({**{k:case[k] for k in IDENTITY},'head':current['head'],
                    'common_task_log_amplitude_NLL_gradient_per_batch':gradient.tolist(),
                    'first_second_half_gradients':[h.tolist() for h in halves],
                    'split_direction_agreement':(halves[0]*halves[1]>0).tolist(),
                    'Gaussian_plugin_score_identity_max_error':identity_error,
                    'source_posterior_usable':case['source_posterior_usable'],
                    'moment_scope':str(states.shape[0]*count)+' retained components;inverse-tilt weights retained;separate from full-chain diagnostics',
                    'source_case':case['component_file'],'optimizer_run':False})
    finally:
        signal.setitimer(signal.ITIMER_PROF,0.);signal.signal(signal.SIGPROF,handler)
    save(ROOT/spec['output']/'raw_learning_scores.json',{'rows':rows,'all64_raw_learning_heads_connected':len(rows)==64,
        'source_case_precision_flags_preserved':True,'cpu_seconds':time.process_time()-started,'cost':spec['cost'],
        'no_model_adopted':True,'independent_confirmation':False,
        'interpretation':'Historical source covariance score now applies to both response and risk laws. Directions scale existing common/task covariance blocks;they are not physical risk factors or spectral exponent fits. Fixed-mixture split stability is a diagnostic,not a convergence certificate or permission to select a law.'})


if __name__=='__main__':run()
