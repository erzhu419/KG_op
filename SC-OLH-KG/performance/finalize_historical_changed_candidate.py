"""Reuse the historical site8 repair on the sole remaining new-law variance."""

import signal
import time

import numpy as np
from scipy.linalg import cho_solve

from performance.complete_historical_source_heads import ROOT,IDENTITY,read,save
from performance.recondition_historical_changed_candidates import candidate_inputs
from performance.resolve_historical_changed_candidate_tails import PrecisionTiltedPosterior
from performance.prepare_legacy_f_variance_reference import VarianceReferencePosterior
from performance.check_legacy_f_correlated_source_raw_update import diagnostics
from performance.diagnose_legacy_f_inner_conditional_calibration import integrate
from performance.refine_historical_raw_lookahead import cpu_timeout

PLAN='performance/manifests/historical_changed_candidate_functional_resolution_v4.json'


def integrate_site_variance(sampler,target,states,reference_mean,reference_covariance,site,bound=32.):
    """One integral of the actual tilted law supplies inverse-h and site moments."""
    states=np.asarray(states);sd=np.sqrt(reference_covariance[site,site])
    direction=reference_covariance[:,site]/sd
    origins=states-((states[:,site]-reference_mean[site])/sd)[:,None]*direction
    position=int(np.where(sampler.sites==site)[0][0]);cache={}

    def evaluate(point):
        base=target.evaluate(point);_,factor,alpha=base
        ds=sampler.rows@alpha
        variances=np.diag(target.covariance)[sampler.sites]-np.sum(
            sampler.rows*cho_solve(factor,sampler.rows.T).T,axis=1)
        hvar=np.log1p(np.mean((variances+(target.mean[sampler.sites]+ds-sampler.center[sampler.sites])**2)
            /sampler.scale[sampler.sites]))
        old=sampler.old.evaluate(point);risk=point-sampler.relative.risk_mean
        risk_ratio=-.5*(sampler.relative.log_determinant_difference
            +risk@sampler.relative.precision_difference@risk)
        reverse=old[0]-sampler.old_log_tilt(old)-base[0]-risk_ratio
        logh=float(np.logaddexp(hvar,reverse-sampler.log_ratio_scale))
        return base[0]+logh,logh,ds[position],variances[position]

    baseline=np.asarray([evaluate(p)[0] for p in states])

    def at(t,index=slice(None)):
        points=np.atleast_2d(origins[index])+np.asarray(t)[...,None]*direction
        evaluations=np.asarray([evaluate(p) for p in points])
        logs=evaluations[:,0]-baseline[index]-.5*(np.log(2*np.pi)+np.asarray(t)**2)
        if isinstance(index,slice):
            inverse=np.exp(-evaluations[:,1]);ds=evaluations[:,2];vs=evaluations[:,3]
            cache['logh']=evaluations[:,1]
            cache['statistics']=np.column_stack([np.ones(len(points)),inverse,
                inverse*ds,inverse*(vs+ds**2)])
        return logs if isinstance(index,slice) else float(logs[0])

    values,logs,error=integrate(at,lambda t:cache['statistics'],coordinate_bound=bound)
    inverse=values[:,1];ds=values[:,2]/inverse;second=values[:,3]/inverse
    endpoint=-np.inf
    for t in (-bound,bound):
        edge=at(t)
        endpoint=max(endpoint,float(np.max(edge-logs)),
            float(np.max(edge-cache['logh']-logs-np.log(inverse))))
    if endpoint>=np.log(1e-10):raise RuntimeError('Fixed-site source integral reaches its registered boundary')
    return {'displacements':ds,'second_moments':second,'inverse_weights':inverse,
        'maximum_quadrature_relative_error':error,'maximum_endpoint_log_density':endpoint}


def site_summary(displacements,second,weights):
    count=weights.shape[1];mean=float(np.sum(weights*displacements)/weights.sum())
    variance=float(np.sum(weights*second)/weights.sum()-mean**2)
    halves=[]
    for s in (slice(0,count//2),slice(count//2,None)):
        w=weights[:,s];d=displacements[:,s];sq=second[:,s]
        halves.append(float(np.sum(w*sq)/w.sum()-(np.sum(w*d)/w.sum())**2))
    central=second-2*mean*displacements+mean**2
    coordinates=[weights*(second-(variance+mean**2))/weights.mean(),
        weights*(central-variance)/weights.mean()]
    normalizer=weights/weights.mean()-1.
    if np.any(normalizer!=0.):coordinates.append(normalizer)
    influences=np.stack(coordinates,axis=-1)
    diagnostic=diagnostics(influences)
    return {'variance':variance,'half_variance_relative_difference':abs(halves[0]-halves[1])/np.mean(halves),
        'maximum_split_rhat':max(diagnostic['split_rhat']),
        'minimum_autocorrelation_ess':min(diagnostic['autocorrelation_ess'])}


def run():
    if not (ROOT/PLAN).exists():
        save(ROOT/PLAN,{'status':'registered before fused scalar site8 functional repair',
            'parent_plan':'performance/manifests/historical_changed_candidate_tail_resolution_v2.json',
            'parent_results':'paper_artifacts/historical_changed_candidate_tail_resolution_v2/results.json',
            'integrated_source_site':8,'batch_size':16,'gaussian_coordinate_bound':32.,
            'maximum_single_thread_cpu_seconds':150,
            'previous_cpu_cap_result':'paper_artifacts/historical_changed_candidate_functional_resolution_v3/results.json',
            'scope':'Same historical final site8 case. Full32768-chain Rhat,ESS and all other variance splits pass. Integrate only the fixed site8 reference using1024 saved new-law tilted residuals;retain every passed full-chain estimate elsewhere. Exact inverse-h includes the variance and old/new bridge. Original candidate parameters,new-law1024 mixture and all thresholds unchanged. No further sampling or model adoption.',
            'output':'paper_artifacts/historical_changed_candidate_functional_resolution_v4'})
    spec=read(PLAN);out=ROOT/spec['output'];out.mkdir(parents=True,exist_ok=True);path=out/'results.json'
    if path.exists():return read(str(path.relative_to(ROOT)))
    parent=read(spec['parent_plan']);result=read(spec['parent_results'])
    failed=[r for r in result['rows'] if not r['new_source_posterior_precision_passed']]
    if len(failed)!=1:raise RuntimeError('Expected the single archived functional failure')
    row=failed[0];site=spec['integrated_source_site'];gate=parent['monte_carlo_gate']
    differences=np.asarray(row['half_variance_relative_differences'])
    if (np.where(differences>gate['maximum_first_second_half_f_variance_relative_difference'])[0].tolist()!=[site]
        or row['maximum_split_rhat']>gate['maximum_split_rhat_eta_and_rao_blackwell_f_moments']
        or row['minimum_autocorrelation_ess']<gate['minimum_autocorrelation_ess_eta_and_rao_blackwell_f_moments']):
        raise RuntimeError('The historical single-site repair does not apply to other failures')
    source=read(parent['source_plan']);laws=read(source['source_laws'])['rows'];selection=read(source['selection_plan'])
    case=next(r for r in read(parent['learning_results'])['evidence_rows'] if all(r[k]==row[k] for k in (*IDENTITY,'head')))
    registered=next(r for r in parent['cases'] if all(r[k]==row[k] for k in (*IDENTITY,'head')))
    old,relative,old_tilt,field,reference=candidate_inputs(case,laws,selection)
    sampler=PrecisionTiltedPosterior(old,relative,old_tilt,registered['tilt_sites'],registered['tilt_center'],
        registered['tilt_scale'],registered['log_ratio_scale'])
    target=VarianceReferencePosterior(relative.mean,relative.covariance,relative.labels,relative.s2)
    saved=read(row['candidate_component_file']);mixture=saved['positive_eta_mixture']
    states=np.asarray(mixture['eta_states_by_chain']);flat=states.reshape(-1,48)
    started=time.process_time();handler=signal.signal(signal.SIGPROF,cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF,spec['maximum_single_thread_cpu_seconds']);chunks=[];stopped=False
    try:
        for begin in range(0,len(flat),spec['batch_size']):
            chunks.append(integrate_site_variance(sampler,target,flat[begin:begin+spec['batch_size']],
                reference['reference_mean'],reference['reference_covariance'],site,spec['gaussian_coordinate_bound']))
    except TimeoutError:stopped=True
    finally:
        signal.setitimer(signal.ITIMER_PROF,0.);signal.signal(signal.SIGPROF,handler)
    if stopped:
        output={'execution_plan':PLAN,'functional_precision_passed':False,'stopped_at_cpu_budget':True,
            'cpu_seconds':time.process_time()-started,'no_model_adopted':True}
    else:
        ds,second,weights=[np.concatenate([b[k] for b in chunks]).reshape(states.shape[:2])
            for k in ('displacements','second_moments','inverse_weights')]
        summary=site_summary(ds,second,weights)
        # Preserve all original full-chain coordinates except the fixed
        # variance functional,as in finalize_historical_source_variance.
        rhat=summary['maximum_split_rhat'];ess=summary['minimum_autocorrelation_ess']
        gap=summary['half_variance_relative_difference'];variance=summary['variance']
        full=float(row['posterior_covariance'][site][site])
        corrected=np.diag(row['posterior_covariance']).copy();corrected[site]=variance
        # The saved mixture's moments must be recomputed under its positive
        # weights;the source law and1024 states themselves remain unchanged.
        moments=[target.moments(target.evaluate(p)) for p in flat]
        from performance.resolve_historical_source_variance import weighted_function_moments
        dm,cm=weighted_function_moments(np.stack([v[0] for v in moments]).reshape(states.shape),
            np.stack([v[1] for v in moments]).reshape(*states.shape[:2],48,48),
            np.asarray(mixture['positive_component_weights']).reshape(states.shape[:2]))
        retention=float(np.max(abs(np.diag(cm)-corrected)/corrected))
        mean_error=float(np.sqrt(np.mean((target.mean+dm-row['posterior_mean'])**2/corrected)))
        passed=bool(rhat<=gate['maximum_split_rhat_eta_and_rao_blackwell_f_moments']
            and ess>=gate['minimum_autocorrelation_ess_eta_and_rao_blackwell_f_moments']
            and gap<=gate['maximum_first_second_half_f_variance_relative_difference']
            and retention<=parent['retained_mixture']['maximum_source_f_diagonal_variance_relative_difference_vs_all_draws']
            and mean_error<=parent['retained_mixture']['maximum_source_f_mean_rms_difference_in_full_posterior_sd'])
        output={'execution_plan':PLAN,'identity':{k:row[k] for k in (*IDENTITY,'head')},'functional_precision_passed':passed,
            'full_chain_variance_reference':full,'conditional_site8_variance_reference':variance,
            'fixed_site8_variance_split':gap,'fixed_site8_rhat':rhat,'fixed_site8_ESS':ess,
            'retained_mixture_variance_error':retention,'retained_mixture_mean_error':mean_error,
            'original_mean_and_other47_variance_estimates_preserved':True,
            'posterior_variance_reference':corrected.tolist(),'candidate_component_file':row['candidate_component_file'],
            'maximum_quadrature_relative_error':max(b['maximum_quadrature_relative_error'] for b in chunks),
            'maximum_endpoint_log_density':max(b['maximum_endpoint_log_density'] for b in chunks),
            'source_evidence_ratio_precision_preserved':row['evidence_ratio_precision_passed'],
            'completed_source_posterior_cases':32 if passed else 31,'cpu_seconds':time.process_time()-started,
            'new_candidate_MCMC_sweeps':0,'new_source_labels':0,'server_jobs':0,'no_model_adopted':True,'independent_confirmation':False}
    save(path,output);print('fixed new-law site8',output.get('functional_precision_passed'),
        'gap',output.get('fixed_site8_variance_split'),'cpu',round(output['cpu_seconds'],2),flush=True)
    return output


if __name__=='__main__':run()
