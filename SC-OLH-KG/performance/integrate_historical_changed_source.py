"""Condition changed source laws by the historical Gaussian-line integral."""

import signal
import time

import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.special import digamma,logsumexp

from performance.complete_historical_source_heads import ROOT,IDENTITY,read,save
from performance.prepare_legacy_f_variance_reference import VarianceReferencePosterior,variance_reference
from performance.resolve_historical_source_variance import (
    VarianceTiltedPosterior,original_posterior_summary,weighted_function_moments,
)
from performance.learn_historical_raw_source_amplitudes import covariance_builder
from performance.diagnose_legacy_f_inner_conditional_calibration import integrate
from performance.refine_historical_raw_lookahead import cpu_timeout

PLAN='performance/manifests/historical_changed_source_conditional_precision_v1.json'


class ChangedSourcePosterior(VarianceReferencePosterior):
    """Exact new raw source density relative to the unchanged Gaussian reference."""

    def __init__(self,mean,covariance,labels,s2,risk_mean,old_risk_covariance,new_risk_covariance):
        super().__init__(mean,covariance,labels,s2)
        self.risk_mean=np.asarray(risk_mean)
        old=cho_factor(old_risk_covariance,lower=True);new=cho_factor(new_risk_covariance,lower=True)
        eye=np.eye(len(mean))
        self.precision_difference=cho_solve(new,eye)-cho_solve(old,eye)
        self.log_determinant_difference=2*(np.log(np.diag(new[0])).sum()-np.log(np.diag(old[0])).sum())

    def evaluate(self,eta):
        score,factor,alpha=super().evaluate(eta);d=eta-self.risk_mean
        ratio=-.5*(self.log_determinant_difference+d@self.precision_difference@d)
        return float(score+ratio),factor,alpha


def integrate_change(old,new,states,reference_mean,reference_covariance,gradient,old_log_tilt,bound=32.):
    """New conditional moments and Z(new)/Z(old*h) weights per old residual.

    Residuals retain the original sampling distribution. Both densities are
    integrated on the same Gaussian-reference line;the denominator uses the
    old sampling law,including its actual h. The old inverse-h conditional
    expectation supplies Z(old)/Z(old*h) for the source evidence ratio.
    """
    states=np.asarray(states);n=len(new.mean)
    gradient=np.asarray(gradient)
    sd=np.sqrt(gradient@reference_covariance@gradient);direction=reference_covariance@gradient/sd
    origins=states-((states-reference_mean)@gradient/sd)[:,None]*direction
    baseline=np.asarray([old.evaluate(p)[0] for p in states]);cache={}

    def old_at(t,index=slice(None)):
        point=np.atleast_2d(origins[index])+np.asarray(t)[...,None]*direction
        evaluations=[old.evaluate(p) for p in point]
        logs=np.asarray([v[0] for v in evaluations])-baseline[index]-.5*(np.log(2*np.pi)+np.asarray(t)**2)
        if isinstance(index,slice):
            cache['old']=np.column_stack([np.ones(len(point)),np.exp(-np.asarray([
                old_log_tilt(p,v) for p,v in zip(point,evaluations)]))])
        return logs if isinstance(index,slice) else float(logs[0])

    old_values,old_logs,old_error=integrate(old_at,lambda t:cache['old'],coordinate_bound=bound)

    def new_at(t,index=slice(None)):
        point=np.atleast_2d(origins[index])+np.asarray(t)[...,None]*direction
        evaluations=[new.evaluate(p) for p in point]
        logs=np.asarray([v[0] for v in evaluations])-baseline[index]-.5*(np.log(2*np.pi)+np.asarray(t)**2)
        if isinstance(index,slice):
            moments=[new.moments(v) for v in evaluations]
            ds=np.asarray([v[0] for v in moments]);cs=np.asarray([v[1] for v in moments])
            second=cs+ds[:,:,None]*ds[:,None,:]
            cache['new']=np.column_stack([np.ones(len(point)),ds,second.reshape(-1,n*n),
                point,(point[:,:,None]*point[:,None,:]).reshape(-1,n*n)])
        return logs if isinstance(index,slice) else float(logs[0])

    values,new_logs,new_error=integrate(new_at,lambda t:cache['new'],coordinate_bound=bound)
    endpoint=max(float(np.max(at(t)-logs)) for at,logs in ((old_at,old_logs),(new_at,new_logs)) for t in (-bound,bound))
    if endpoint>=np.log(1e-10):raise RuntimeError('Conditional source density reaches the registered integration boundary')
    j=1;ds=values[:,j:j+n];j+=n
    cs=values[:,j:j+n*n].reshape(-1,n,n)-ds[:,:,None]*ds[:,None,:];j+=n*n
    eta=values[:,j:j+n];j+=n
    ec=values[:,j:j+n*n].reshape(-1,n,n)-eta[:,:,None]*eta[:,None,:]
    return {'eta':eta,'displacements':ds,'covariances':cs,'eta_covariances':ec,
        'log_change_weights':new_logs-old_logs,'old_inverse_tilt_weights':old_values[:,1],
        'maximum_quadrature_relative_error':max(old_error,new_error),'maximum_endpoint_log_density':endpoint}


def load_case(case,laws,selection):
    law=next(r for r in laws if r['head']==case['head'] and all(r[k]==case[k] for k in IDENTITY))
    risk=next(r for r in laws if r['head']==('eta_g' if case['head']=='g' else 'eta_f') and all(r[k]==case[k] for k in IDENTITY))
    source=read(selection['source_inputs_by_layout'][case['layout']])
    response=np.vstack([t['response'] for t in source['source_tasks'] if t['name'] in selection['source_tasks']])
    head=int(case['head']=='g');s2=np.exp(response[:,2 if head else 3]+digamma(1.))
    saved=read(case['source_case']);tilt=saved.get('tilt')
    if tilt:
        old=VarianceTiltedPosterior(law['source_mean'],law['source_covariance'],response[:,head],s2,
            tilt['sites'],tilt['center'],tilt['scale']);old_log_tilt=old.log_tilt
    else:
        old=VarianceReferencePosterior(law['source_mean'],law['source_covariance'],response[:,head],s2)
        old_log_tilt=lambda evaluation:0.
    new=ChangedSourcePosterior(law['source_mean'],covariance_builder(law)(case['f_common_task_scales'])[0],
        response[:,head],s2,risk['source_mean'],risk['source_covariance'],covariance_builder(risk)(case['eta_common_task_scales'])[0])
    rm,rc=variance_reference(risk['source_mean'],risk['source_covariance'],s2)
    states=np.asarray(saved['positive_eta_mixture']['eta_states_by_chain'])
    return old,new,states,rm,rc,old_log_tilt


def variance_gradient(model,states,weights,site,center):
    """Sensitivity of the SAME centered f second moment to all source risks."""
    result=np.zeros(len(model.mean));flat=states.reshape(-1,len(model.mean))
    w=np.asarray(weights).reshape(-1);w=w/w.sum()
    for mass,eta in zip(w,flat):
        _,factor,alpha=model.evaluate(eta)
        gain=cho_solve(factor,model.covariance[site])
        displacement=model.covariance[site]@alpha
        result+=mass*(gain**2-2*(displacement-center)*gain*alpha)*np.exp(eta)/3
    return result


def register():
    path=ROOT/PLAN
    if path.exists():return read(PLAN)
    learning=read('paper_artifacts/historical_raw_source_amplitude_learning_v1/results.json')
    previous=read('paper_artifacts/historical_raw_source_amplitude_learning_v1/posterior_precision.json')
    parent=read('performance/manifests/historical_source_variance_resolution_v4.json')
    laws=read(parent['source_laws'])['rows'];selection=read(parent['selection_plan']);cases=[]
    for case in learning['evidence_rows']:
        row=next(r for r in previous['rows'] if all(r[k]==case[k] for k in (*IDENTITY,'head')))
        if row['new_source_posterior_precision_passed']:continue
        _,new,states,*_=load_case(case,laws,selection)
        moments=[new.moments(new.evaluate(p)) for p in states.reshape(-1,48)]
        ds=np.stack([m[0] for m in moments]).reshape(states.shape)
        cs=np.stack([m[1] for m in moments]).reshape(*states.shape[:2],48,48)
        weights=np.asarray(case['new_positive_eta_weights']).reshape(states.shape[:2]);count=states.shape[1]
        halves=[np.diag(weighted_function_moments(ds[:,s],cs[:,s],weights[:,s])[1])
            for s in (slice(0,count//2),slice(count//2,None))]
        difference=abs(halves[0]-halves[1])/np.mean(halves,axis=0)
        cases.append({**{k:case[k] for k in (*IDENTITY,'head')},'source_case':case['source_case'],
            'integrated_source_site':int(difference.argmax()),'previous_maximum_variance_split':float(difference.max()),
            'direction_gradient':np.eye(48)[difference.argmax()].tolist(),
            'retained_component_count':int(np.prod(states.shape[:2]))})
    spec={'status':'registered before changed-law source conditional integration','cases':cases,
        'source_plan':'performance/manifests/historical_source_variance_resolution_v4.json',
        'learning_results':'paper_artifacts/historical_raw_source_amplitude_learning_v1/results.json',
        'previous_posterior_precision':'paper_artifacts/historical_raw_source_amplitude_learning_v1/posterior_precision.json',
        'point_mixture_moments':'paper_artifacts/historical_raw_source_amplitude_learning_v1/posterior_precision.json',
        'evidence_precision':read('performance/manifests/historical_raw_source_amplitude_learning_v1.json')['evidence_precision'],
        'integration_batch_size':16,'gaussian_coordinate_bound':32.,'maximum_single_thread_cpu_seconds_per512_components':45,
        'scope':'All27 failed changed-law cases;reuse the five combined PASS cases. Freeze each source-only largest variance-split site before integration. Historical Gaussian line and adaptive integrator,exact old/new raw density ratio,inverse-tilt normalization,full f and eta conditional covariance. Original gates and unchanged candidate parameters. No new source chains,target labels,model adoption or across-identity evidence claim.',
        'cost':{'new_source_labels':0,'new_source_MCMC_sweeps':0,'new_prior_fits':0,'new_target_simulator_calls':0,'new_verifier_calls':0,'server_jobs':0},
        'output':'paper_artifacts/historical_changed_source_conditional_precision_v1'}
    save(path,spec);return spec


def register_gradient():
    plan='performance/manifests/historical_changed_source_conditional_precision_v2.json'
    if (ROOT/plan).exists():return read(plan)
    parent=read(PLAN);precision=read(parent['output']+'/results.json')
    if not precision['all_registered_cases_complete']:
        raise RuntimeError('Finish all registered single-site cases before freezing new directions')
    latest=precision['rows']+precision['reused_passed_cases']
    combined=ROOT/parent['output']/'complete_posterior_precision.json'
    save(combined,{'rows':latest,'scope':'All32 fixed candidate laws;single-site refinements and five reused PASS cases.'})
    source=read(parent['source_plan']);laws=read(source['source_laws'])['rows'];selection=read(source['selection_plan'])
    learning=read(parent['learning_results']);point=read(parent['point_mixture_moments']);cases=[]
    for row in precision['rows']:
        if row['new_source_posterior_precision_passed']:continue
        case=next(r for r in learning['evidence_rows'] if all(r[k]==row[k] for k in (*IDENTITY,'head')))
        moment=next(r for r in point['rows'] if all(r[k]==row[k] for k in (*IDENTITY,'head')))
        _,new,states,*_=load_case(case,laws,selection)
        site=(int(np.argmax(row['half_variance_relative_differences'])) if 'half_variance_relative_differences' in row
            else row['integrated_source_site'])
        gradient=variance_gradient(new,states,case['new_positive_eta_weights'],site,
            moment['posterior_mean'][site]-new.mean[site])
        cases.append({**{k:case[k] for k in (*IDENTITY,'head')},'source_case':case['source_case'],
            'integrated_source_site':site,'direction_gradient':gradient.tolist(),
            'retained_component_count':int(np.prod(states.shape[:2])),
            'previous_maximum_variance_split':row.get('maximum_first_second_half_f_variance_relative_difference')})
    spec={**parent,'status':'registered before joint variance-sensitive directions','cases':cases,
        'previous_posterior_precision':str(combined.relative_to(ROOT)),
        'output':'paper_artifacts/historical_changed_source_conditional_precision_v2',
        'scope':'Only single-site HOLD cases. Freeze the weighted gradient of the centered f second moment at the remaining worst source site. Integrate its q-independent Gaussian coordinate Sigma_q*g/sqrt(g*Sigma_q*g),which moves the correlated risk sites jointly. Same candidate parameters,raw likelihood,gates and point-mixture reference. Reuse all prior PASS cases;preserve all previous failures. No new chains,labels,server work or model adoption.'}
    save(ROOT/plan,spec);return spec


def run(plan=PLAN):
    spec=read(plan)
    out=ROOT/spec['output'];out.mkdir(parents=True,exist_ok=True);path=out/'results.json'
    previous=read(str(path.relative_to(ROOT))) if path.exists() else {}
    if previous.get('all_registered_cases_complete'):return previous
    rows=previous.get('rows',[]);started=time.process_time()
    parent=read(spec['source_plan']);laws=read(parent['source_laws'])['rows'];selection=read(parent['selection_plan'])
    learning=read(spec['learning_results']);old_precision=read(spec['previous_posterior_precision'])
    point_precision=read(spec['point_mixture_moments'])
    for registered in spec['cases']:
        if any(all(r[k]==registered[k] for k in (*IDENTITY,'head')) for r in rows):continue
        case=next(r for r in learning['evidence_rows'] if all(r[k]==registered[k] for k in (*IDENTITY,'head')))
        one=time.process_time();handler=signal.signal(signal.SIGPROF,cpu_timeout)
        signal.setitimer(signal.ITIMER_PROF,spec['maximum_single_thread_cpu_seconds_per512_components']
            *registered['retained_component_count']/512)
        row=dict(registered)
        try:
            old,new,states,rm,rc,tilt=load_case(case,laws,selection);flat=states.reshape(-1,48)
            chunks=[integrate_change(old,new,flat[i:i+spec['integration_batch_size']],rm,rc,
                registered['direction_gradient'],lambda p,e:tilt(e),spec['gaussian_coordinate_bound'])
                for i in range(0,len(flat),spec['integration_batch_size'])]
            eta,ds,cs,ec,lw,iw=[np.concatenate([b[k] for b in chunks]) for k in
                ('eta','displacements','covariances','eta_covariances','log_change_weights','old_inverse_tilt_weights')]
            lw=lw.reshape(states.shape[:2]);iw=iw.reshape(states.shape[:2]);weights=np.exp(lw-logsumexp(lw))
            eta=eta.reshape(states.shape);ds=ds.reshape(states.shape)
            cs=cs.reshape(*states.shape[:2],48,48);ec=ec.reshape(cs.shape)
            summary=original_posterior_summary(new,eta,ds,cs,weights,parent)
            summary['eta_posterior_covariance']=(np.asarray(summary['eta_posterior_covariance'])
                +np.einsum('ct,ctij->ij',weights,ec)).tolist()
            ratio=float(logsumexp(lw)-np.log(iw.sum()))
            halves=[float(logsumexp(lw[:,s])-np.log(iw[:,s].sum()))
                for s in (slice(0,states.shape[1]//2),slice(states.shape[1]//2,None))]
            gap=abs(halves[0]-halves[1]);ess=summary['normalized_inverse_weight_ESS']
            evidence=bool(ess>=spec['evidence_precision']['minimum_importance_ESS']
                and gap<=spec['evidence_precision']['maximum_split_log_ratio_difference'])
            point=next(r for r in point_precision['rows'] if all(r[k]==case[k] for k in (*IDENTITY,'head')))
            variance=np.diag(summary['posterior_covariance']);pv=np.diag(point['posterior_covariance'])
            mean_error=float(np.sqrt(np.mean((np.asarray(point['posterior_mean'])-summary['posterior_mean'])**2/variance)))
            variance_error=float(np.max(abs(pv-variance)/variance));retained=parent['retained_mixture']
            retention=bool(mean_error<=retained['maximum_source_f_mean_rms_difference_in_full_posterior_sd']
                and variance_error<=retained['maximum_source_f_diagonal_variance_relative_difference_vs_all_draws'])
            passed=bool(summary['monte_carlo_gate_passed'] and evidence)
            row.update(**summary,evidence_ratio_precision_passed=evidence,
                log_source_evidence_ratio_new_vs_old=ratio,split_log_evidence_ratios=halves,split_log_ratio_difference=gap,
                new_source_posterior_precision_passed=passed,point_mixture_mean_error=mean_error,
                point_mixture_maximum_f_variance_relative_error=variance_error,point_mixture_retention_passed=retention,
                source_posterior_and_retention_passed=passed and retention,
                eta_within_conditional_covariance_retained=True,
                maximum_quadrature_relative_error=max(b['maximum_quadrature_relative_error'] for b in chunks),
                maximum_endpoint_log_density=max(b['maximum_endpoint_log_density'] for b in chunks))
        except (TimeoutError,RuntimeError,AssertionError) as exc:
            row.update(new_source_posterior_precision_passed=False,source_posterior_and_retention_passed=False,
                numerical_failure='registered source conditional CPU cap' if isinstance(exc,TimeoutError) else str(exc),
                stopped_at_cpu_budget=isinstance(exc,TimeoutError))
        finally:
            signal.setitimer(signal.ITIMER_PROF,0.);signal.signal(signal.SIGPROF,handler)
        row['cpu_seconds']=time.process_time()-one;rows.append(row)
        save(path,{'execution_plan':plan,'rows':rows,'all_registered_cases_complete':False})
        print('source conditional',*[row[k] for k in (*IDENTITY,'head')],
            'pass',row['new_source_posterior_precision_passed'],'variance split',
            round(row.get('maximum_first_second_half_f_variance_relative_difference',-1),4),
            'cpu',round(row['cpu_seconds'],2),row.get('numerical_failure',''),flush=True)
    reused=[r for r in old_precision['rows'] if r['new_source_posterior_precision_passed']]
    result={'execution_plan':plan,'rows':rows,'reused_passed_cases':reused,
        'all_registered_cases_complete':len(rows)==len(spec['cases']),'case_count':len(rows)+len(reused),
        'posterior_precision_cases_passed':len(reused)+sum(r['new_source_posterior_precision_passed'] for r in rows),
        'refined_posterior_and_retention_cases_passed':sum(r['source_posterior_and_retention_passed'] for r in rows),
        'cpu_seconds':time.process_time()-started,'cost':spec['cost'],'original32_source_cases_preserved':True,
        'no_model_adopted':True,'independent_confirmation':False,
        'limitations':['Source-only numerical conditioning of one fixed MCEM candidate,not converged learning.',
            'Conditional source evidence ratios remain within each original identity.',
            'Original point mixtures require a retention PASS against the integrated posterior before target use.']}
    save(path,result);return result


if __name__=='__main__':register();run()
