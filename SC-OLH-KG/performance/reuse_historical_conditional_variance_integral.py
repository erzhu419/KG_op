"""Use the existing conditional integral on the three unresolved variance sites."""

import signal
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import digamma

from performance.complete_historical_source_heads import PLAN, ROOT, IDENTITY, read, save, cpu_timeout
from performance.prepare_legacy_f_variance_reference import VarianceReferencePosterior, variance_reference
from performance.diagnose_legacy_f_inner_conditional_calibration import integrate
from performance.check_legacy_f_correlated_source_raw_update import diagnostics


def run():
    spec=read(PLAN);execution=spec['remaining_variance_integral'];selection=read(spec['selection_plan'])
    cases=read(spec['completed_source_registry'])['rows'];laws=read(spec['source_laws'])['rows']
    rows=[];started=time.process_time();handler=signal.signal(signal.SIGPROF,cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF,execution['maximum_single_thread_cpu_seconds'])
    try:
        for case in (r for r in cases if not r['source_posterior_usable']):
            saved=read(case['component_file']);first,second=map(np.asarray,(saved['first_half_posterior_variance'],saved['second_half_posterior_variance']))
            site=int(np.argmax(abs(first-second)/((first+second)/2)))
            law=next(r for r in laws if r['head']==case['head'] and all(r[k]==case[k] for k in IDENTITY))
            risk=next(r for r in laws if r['head']==('eta_f' if case['head']=='f' else 'eta_g') and all(r[k]==case[k] for k in IDENTITY))
            source=read(selection['source_inputs_by_layout'][case['layout']]);tasks=[t for t in source['source_tasks'] if t['name'] in selection['source_tasks']]
            response=np.vstack([t['response'] for t in tasks]);h=int(case['head']=='g')
            labels=response[:,h];s2=np.exp(response[:,2 if h else 3]+digamma(1.))
            model=VarianceReferencePosterior(law['source_mean'],law['source_covariance'],labels,s2)
            mean,cov=variance_reference(risk['source_mean'],risk['source_covariance'],s2)
            Q=cho_solve(cho_factor(cov,lower=True),np.eye(48));sd=np.sqrt(1/Q[site,site])
            states=np.asarray(saved['positive_eta_mixture']['eta_states_by_chain']).reshape(-1,48)
            conditional=mean[site]-(states-mean)@Q[site]/Q[site,site]+states[:,site]-mean[site]
            K=np.asarray(law['source_covariance']);kr=K[site];center=saved['source_f_posterior_mean'][site]-model.mean[site]
            cache={}
            def evaluate(points):
                evaluations=[model.evaluate(p) for p in points]
                displacement=np.asarray([kr@e[2] for e in evaluations])
                variance=np.asarray([K[site,site]-kr@cho_solve(e[1],kr) for e in evaluations])
                return np.asarray([e[0] for e in evaluations]),displacement,variance
            baseline,ds,vs=evaluate(states)
            def at(t,index=slice(None)):
                points=np.atleast_2d(states[index]).copy();points[:,site]=np.atleast_1d(conditional[index])+sd*t
                logs,delta,var=evaluate(points)
                if isinstance(index,slice):cache['statistics']=np.column_stack([np.ones(len(points)),var+(delta-center)**2,delta])
                result=logs-baseline[index]-.5*(np.log(2*np.pi)+t*t)
                return result if isinstance(index,slice) else float(result[0])
            def statistics(t):return cache['statistics']
            values,_,error=integrate(at,statistics)
            before=(vs+(ds-center)**2).reshape(4,128,1);after=values[:,1].reshape(4,128,1)
            d0,d1=diagnostics(before),diagnostics(after)
            def half_error(v,delta):
                v=v.reshape(4,128);delta=delta.reshape(4,128)
                variances=[v[:,part].mean()-(delta[:,part].mean()-center)**2 for part in (slice(0,64),slice(64,128))]
                return abs(variances[0]-variances[1])/np.mean(variances)
            rows.append({**{k:case[k] for k in IDENTITY},'head':case['head'],'site_index':site,
                'saved_direct_half_variance_relative_difference':float(half_error(before,ds)),
                'conditional_integrated_half_variance_relative_difference':float(half_error(after,values[:,2])),
                'direct_variance_functional_ESS':d0['autocorrelation_ess'][0],
                'conditional_integrated_variance_functional_ESS':d1['autocorrelation_ess'][0],
                'conditional_integrated_variance_functional_Rhat':d1['split_rhat'][0],
                'conditional_integral_relative_error_bound':error,'full_source_precision_HOLD_unchanged':True})
    finally:
        signal.setitimer(signal.ITIMER_PROF,0.);signal.signal(signal.SIGPROF,handler)
    save(ROOT/spec['output']/'conditional_variance_diagnostic.json',{'rows':rows,'all_three_saved_failures_integrated':len(rows)==3,
        'cpu_seconds':time.process_time()-started,'new_source_draws':0,'no_model_adopted':True,
        'interpretation':'Historical conditional Rao-Blackwell integration on each worst saved full-chain half-variance site. Diagnostics refer to512 retained states and one coordinate,not a replacement for the full48-site gate or retained-mixture validation.'})
    print('Historical conditional variance integrals complete',len(rows),'CPU',round(time.process_time()-started,3),flush=True)


if __name__=='__main__':run()
