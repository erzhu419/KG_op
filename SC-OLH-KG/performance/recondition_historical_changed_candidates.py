"""Reuse the archived sampler for fixed new candidates,not passed old laws."""

import signal
import time

import numpy as np
from scipy.special import digamma,logsumexp,polygamma

from performance.complete_historical_source_heads import ROOT,IDENTITY,read,save
from performance.integrate_historical_changed_source import load_case
from performance.check_legacy_f_collapsed_source_raw_update import sample
from performance.check_legacy_f_learned_covariance_source_update import retained_mixture,verify_learned
from performance.prepare_legacy_f_variance_reference import VarianceReferencePosterior,variance_reference
from performance.resolve_historical_source_variance import original_posterior_summary
from performance.check_legacy_f_correlated_source_raw_update import diagnostics
from performance.diagnose_legacy_f_source_incremental_calibration import log_density
from performance.refine_historical_raw_lookahead import cpu_timeout

PLAN='performance/manifests/historical_changed_candidate_reconditioning_v1.json'


def candidate_inputs(case,laws,selection):
    old,relative,_,_,_,tilt=load_case(case,laws,selection)
    law=next(r for r in laws if r['head']==case['head'] and all(r[k]==case[k] for k in IDENTITY))
    risk=next(r for r in laws if r['head']==('eta_g' if case['head']=='g' else 'eta_f') and all(r[k]==case[k] for k in IDENTITY))
    from performance.learn_historical_raw_source_amplitudes import covariance_builder
    risk_cov=covariance_builder(risk)(case['eta_common_task_scales'])[0]
    common=np.asarray(law['unconditioned_common_covariance'])*case['f_common_task_scales'][0]
    field={'layout':case['layout'],'arms':['/'.join(case[k] for k in (*IDENTITY,'head'))],
        'f_means':[relative.mean],'f_covariances':[relative.covariance],
        'eta_mean':risk['source_mean'],'eta_covariance':risk_cov,
        'batch_means':relative.labels,'sample_variances':relative.s2,
        'common_latent_prior_means':[law['unconditioned_common_mean']],
        'common_latent_prior_covariances':[common],
        'common_latent_to_source_f_covariances':[common@np.asarray(law['source_design']).T]}
    rm,rc=variance_reference(field['eta_mean'],risk_cov,relative.s2)
    reference={'reference_mean':rm,'reference_covariance':rc,
        'approximate_Gaussian_variance_evidence':log_density(np.log(relative.s2)-digamma(1.)-np.asarray(field['eta_mean']),
            risk_cov+float(polygamma(1,1.))*np.eye(48))}
    return old,relative,tilt,field,reference


def register():
    if (ROOT/PLAN).exists():return read(PLAN)
    previous=read('paper_artifacts/historical_changed_source_conditional_precision_v2/results.json')
    if not previous['all_registered_cases_complete']:raise RuntimeError('Finish the two conditional integration batches first')
    latest=previous['rows']+previous['reused_passed_cases']
    combined=ROOT/'paper_artifacts/historical_changed_source_conditional_precision_v2/complete_posterior_precision.json'
    save(combined,{'rows':latest})
    parent=read('performance/manifests/historical_source_variance_resolution_v4.json')
    spec={'status':'registered before fresh conditioning of the remaining fixed candidates',
        'cases':[{k:r[k] for k in (*IDENTITY,'head','source_case')} for r in latest if not r['new_source_posterior_precision_passed']],
        'source_plan':'performance/manifests/historical_source_variance_resolution_v4.json',
        'learning_results':'paper_artifacts/historical_raw_source_amplitude_learning_v1/results.json',
        'previous_posterior_precision':str(combined.relative_to(ROOT)),
        'sampler':{'chains':4,'warmup_sweeps_per_chain':512,'retained_sweeps_per_chain':4096,'seed':202610089},
        'retained_mixture':{**parent['retained_mixture'],'stride_per_chain':16,'components_per_arm':1024},
        'monte_carlo_gate':parent['monte_carlo_gate'],
        'evidence_precision':read('performance/manifests/historical_raw_source_amplitude_learning_v1.json')['evidence_precision'],
        'maximum_single_thread_cpu_seconds':45,
        'scope':'Only the15 remaining changed-law HOLD cases;reuse all17 passed candidates. Fixed four fresh chains and1024 positive components. Reuse the Stage203/206 compensated raw sampler and central-variance influence diagnostics. A new Gaussian variance reference belongs to each new risk prior;the old-reference prior-density ratio is not used as the new likelihood. Original32 passed laws/files/chains and all previous failures remain untouched. No seed search,source labels,prior optimization,target reads or server jobs.',
        'cost':{'new_source_labels':0,'new_prior_fits':0,'new_target_simulator_calls':0,'new_verifier_calls':0,'server_jobs':0},
        'output':'paper_artifacts/historical_changed_candidate_reconditioning_v1'}
    save(ROOT/PLAN,spec);return spec


def run(plan=PLAN):
    spec=register() if plan==PLAN else read(plan)
    out=ROOT/spec['output'];out.mkdir(parents=True,exist_ok=True);path=out/'results.json'
    previous=read(str(path.relative_to(ROOT))) if path.exists() else {}
    if previous.get('all_registered_cases_complete'):return previous
    parent=read(spec['source_plan']);laws=read(parent['source_laws'])['rows'];selection=read(parent['selection_plan'])
    learning=read(spec['learning_results']);baseline=read(spec['previous_posterior_precision'])
    rows=previous.get('rows',[]);started=time.process_time()
    for item in spec['cases']:
        if any(all(r[k]==item[k] for k in (*IDENTITY,'head')) for r in rows):continue
        case=next(r for r in learning['evidence_rows'] if all(r[k]==item[k] for k in (*IDENTITY,'head')))
        one=time.process_time();handler=signal.signal(signal.SIGPROF,cpu_timeout)
        signal.setitimer(signal.ITIMER_PROF,spec['maximum_single_thread_cpu_seconds']);row=dict(item)
        try:
            old,relative,tilt,field,reference=candidate_inputs(case,laws,selection)
            rm,rc=reference['reference_mean'],reference['reference_covariance']
            check=verify_learned(field,0,reference)
            model=VarianceReferencePosterior(relative.mean,relative.covariance,relative.labels,relative.s2)
            seed=spec['sampler']['seed']+1000*(case['selector']=='additive')+10*(case['support']=='joint')+10000*(case['head']=='g')
            paired={**spec,'sampler':{**spec['sampler'],'seed':seed}};eta=[];moments=[]
            sample(field,0,paired,one,model,rm,rc,eta,retained_moments=moments)
            ds,cs=moments[0];states=eta[0]
            summary=original_posterior_summary(model,states,ds,cs,np.ones(states.shape[:2]),spec)
            mixture=retained_mixture(field,0,model,states,summary,spec)
            reverse=[]
            for point in states.reshape(-1,48):
                evaluation=old.evaluate(point)
                reverse.append(evaluation[0]-tilt(evaluation)-relative.evaluate(point)[0])
            reverse=np.asarray(reverse).reshape(states.shape[:2]);scaled=np.exp(reverse-reverse.max())
            diagnostic=diagnostics(scaled[...,None]);weights=np.exp(reverse-logsumexp(reverse))
            ess=float(1/np.sum(weights**2));logratio=float(-logsumexp(reverse)+np.log(reverse.size))
            halves=[float(-logsumexp(reverse[:,s])+np.log(reverse[:,s].size))
                for s in (slice(0,states.shape[1]//2),slice(states.shape[1]//2,None))]
            gap=abs(halves[0]-halves[1]);gate=spec['monte_carlo_gate']
            evidence=bool(ess>=spec['evidence_precision']['minimum_importance_ESS']
                and gap<=spec['evidence_precision']['maximum_split_log_ratio_difference']
                and diagnostic['split_rhat'][0]<=gate['maximum_split_rhat_eta_and_rao_blackwell_f_moments']
                and diagnostic['autocorrelation_ess'][0]>=gate['minimum_autocorrelation_ess_eta_and_rao_blackwell_f_moments'])
            passed=bool(summary['monte_carlo_gate_passed'] and mixture['mixture_gate_passed'] and evidence)
            component=out/('candidate_'+'_'.join(case[k] for k in (*IDENTITY,'head'))+'.json')
            save(component,{'execution_plan':plan,'fixed_candidate':{k:case[k] for k in (*IDENTITY,'head',
                'f_common_task_scales','eta_common_task_scales')},'posterior':summary,'positive_eta_mixture':mixture})
            row.update(**summary,verification=check,new_source_posterior_precision_passed=passed,
                posterior_functional_precision_passed=summary['monte_carlo_gate_passed'],point_mixture_retention_passed=mixture['mixture_gate_passed'],
                source_posterior_and_retention_passed=bool(summary['monte_carlo_gate_passed'] and mixture['mixture_gate_passed']),evidence_ratio_precision_passed=evidence,
                log_source_evidence_ratio_new_vs_old=logratio,reverse_importance_ESS=ess,
                reverse_ratio_rhat=diagnostic['split_rhat'][0],reverse_ratio_autocorrelation_ESS=diagnostic['autocorrelation_ess'][0],
                split_log_ratio_difference=gap,point_mixture_maximum_f_variance_relative_error=mixture['maximum_source_f_variance_relative_difference'],
                point_mixture_mean_error=mixture['source_f_mean_rms_difference_in_full_posterior_sd'],
                candidate_component_file=str(component.relative_to(ROOT)),
                new_candidate_MCMC_sweeps=4*(512+4096))
        except TimeoutError:
            row.update(new_source_posterior_precision_passed=False,source_posterior_and_retention_passed=False,
                numerical_failure='registered fresh-candidate CPU cap')
        finally:
            signal.setitimer(signal.ITIMER_PROF,0.);signal.signal(signal.SIGPROF,handler)
        row['cpu_seconds']=time.process_time()-one;rows.append(row)
        save(path,{'execution_plan':plan,'rows':rows,'all_registered_cases_complete':False})
        print('fresh candidate',*[row[k] for k in (*IDENTITY,'head')],'pass',row['new_source_posterior_precision_passed'],
            'variance split',round(row.get('maximum_first_second_half_f_variance_relative_difference',-1),4),
            'reverse evidence',row.get('evidence_ratio_precision_passed'), 'cpu',round(row['cpu_seconds'],2),flush=True)
    reused=[r for r in baseline['rows'] if r['new_source_posterior_precision_passed']]
    result={'execution_plan':plan,'rows':rows,'reused_passed_cases':reused,'all_registered_cases_complete':len(rows)==len(spec['cases']),
        'posterior_precision_cases_passed':len(reused)+sum(r['new_source_posterior_precision_passed'] for r in rows),
        'cpu_seconds':time.process_time()-started,'cost':spec['cost'],'new_candidate_MCMC_sweeps':sum(r.get('new_candidate_MCMC_sweeps',0) for r in rows),
        'original32_passed_source_cases_unchanged':True,'no_model_adopted':True,'independent_confirmation':False}
    save(path,result);return result


if __name__=='__main__':run()
