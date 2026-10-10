"""Complete only unresolved source cases using the historical conditional kernel."""

import signal
import time

import numpy as np
from scipy.special import digamma, polygamma

from performance.complete_historical_source_heads import (
    PLAN, ROOT, IDENTITY, common_moments, cpu_timeout, log_density, read,
    retained_mixture, sample, save, source_case, variance_reference,
    VarianceReferencePosterior, verify_learned,
)
from performance.historical_source_block_transition import source_block_transition


def run():
    spec=read(PLAN); block=spec['unresolved_block_sampling']; previous=read(block['source_results'])
    parent=read(spec['covariance_plan']); selection=read(spec['selection_plan'])
    laws=read(spec['source_laws'])['rows']; fields=read(parent['source_fields'])['rows']
    out=ROOT/spec['output']; (out/'blocked_cases').mkdir(exist_ok=True)
    rows=[dict(r) for r in previous['rows']]; checks=[]; costs=[]
    g_summaries={}
    for layout,path in selection['source_inputs_by_layout'].items():
        tasks=[t for t in read(path)['source_tasks'] if t['name'] in selection['source_tasks']]
        response=np.vstack([t['response'] for t in tasks]); noise=np.vstack([t['noise'] for t in tasks])
        s2=np.exp(response[:,2]+digamma(1.))
        np.testing.assert_allclose(s2/3,noise[:,1],atol=1e-14,rtol=1e-10)
        g_summaries[layout]=(response[:,1].tolist(),s2.tolist())
    started=time.process_time(); stopped=False
    handler=signal.signal(signal.SIGPROF,cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF,block['maximum_single_thread_cpu_seconds'])
    try:
        for old in previous['rows']:
            if old['source_posterior_usable']: continue
            one=time.process_time(); identity={k:old[k] for k in IDENTITY}; head=old['head']
            law=next(r for r in laws if r['head']==head and all(r[k]==old[k] for k in IDENTITY))
            risk=next(r for r in laws if r['head']==('eta_f' if head=='f' else 'eta_g') and all(r[k]==old[k] for k in IDENTITY))
            source=dict(next(r for r in fields if (r['layout'],r['active_arm'])==(old['layout'],parent['proposal_arm'])))
            if head=='g':source['batch_means'],source['sample_variances']=g_summaries[old['layout']]
            field=source_case(law,risk,source)
            mean,covariance=variance_reference(field['eta_mean'],field['eta_covariance'],field['sample_variances'])
            reference={'reference_mean':mean,'reference_covariance':covariance,
                'approximate_Gaussian_variance_evidence':log_density(np.log(field['sample_variances'])-digamma(1.)-field['eta_mean'],
                    np.asarray(field['eta_covariance'])+float(polygamma(1,1.))*np.eye(48))}
            transition,check=source_block_transition(mean,covariance)
            checks.append({**identity,'head':head,**check,**verify_learned(field,0,reference)})
            execution={**spec,'maximum_single_thread_cpu_seconds':block['maximum_single_thread_cpu_seconds'],
                'sampler':{**block['sampler'],'seed':block['sampler']['seed']+1000*(old['selector']=='additive')
                    +10*(old['support']=='joint')+(10000 if head=='g' else 0)}}
            model=VarianceReferencePosterior(field['f_means'][0],field['f_covariances'][0],field['batch_means'],field['sample_variances'])
            captured=[]; posterior=sample(field,0,execution,started,model,mean,covariance,captured,transition=transition)
            mixture=retained_mixture(field,0,model,captured[0],posterior,execution)
            cm,cc=common_moments(field,0,np.asarray(posterior['posterior_mean']),np.asarray(posterior['posterior_covariance']))
            path=out/'blocked_cases'/(head+'_layout'+'_'.join(identity.values())+'.json')
            row={**identity,'head':head,'reused_completed_case':False,'conditional_task_blocks_used':True,
                **{k:posterior[k] for k in ('monte_carlo_gate_passed','maximum_split_rhat','minimum_autocorrelation_ess',
                    'maximum_first_second_half_f_variance_relative_difference','elliptical_likelihood_evaluations_by_chain')},
                'mixture_gate_passed':mixture['mixture_gate_passed'],
                'source_posterior_usable':bool(posterior['monte_carlo_gate_passed'] and mixture['mixture_gate_passed']),
                'retained_sweeps_per_chain':execution['sampler']['retained_sweeps_per_chain'],
                'component_file':str(path.relative_to(ROOT)), 'previous_component_file':old['component_file']}
            save(path,{'execution_plan':PLAN,**identity,'head':head,'gates':row,'source_laws':spec['source_laws'],
                'positive_eta_mixture':mixture,'common_coefficient_posterior_mean':cm.tolist(),'common_coefficient_posterior_covariance':cc.tolist(),
                'source_f_posterior_mean':posterior['posterior_mean'],'source_f_posterior_variance':np.diag(posterior['posterior_covariance']).tolist(),
                'full_source_eta_posterior_mean':posterior['eta_posterior_mean'],'full_source_eta_posterior_covariance':posterior['eta_posterior_covariance'],
                'diagnostics':posterior['diagnostics'],'first_half_posterior_mean':posterior['first_half_posterior_mean'],
                'second_half_posterior_mean':posterior['second_half_posterior_mean'],'first_half_posterior_variance':posterior['first_half_posterior_variance'],
                'second_half_posterior_variance':posterior['second_half_posterior_variance']})
            rows[rows.index(next(r for r in rows if r['head']==head and all(r[k]==old[k] for k in IDENTITY)))]=row
            costs.append({**identity,'head':head,'cpu_seconds':time.process_time()-one})
            print('blocked source',head,*identity.values(),'usable',row['source_posterior_usable'],
                'ESS',round(row['minimum_autocorrelation_ess'],1),'halfvar',round(row['maximum_first_second_half_f_variance_relative_difference'],3),flush=True)
    except TimeoutError:stopped=True
    finally:
        signal.setitimer(signal.ITIMER_PROF,0.);signal.signal(signal.SIGPROF,handler)
    save(out/'blocked_source_verification.json',{'checks':checks,'execution_plan':PLAN})
    save(out/'blocked_source_results.json',{'rows':rows,'usable_cases':sum(r['source_posterior_usable'] for r in rows),
        'reused_usable_cases':sum(r['source_posterior_usable'] for r in previous['rows']),
        'block_cases_completed':len(costs),'cpu_seconds':time.process_time()-started,'case_costs':costs,
        'stopped_at_cpu_budget':stopped,'all_source_posteriors_usable':all(r['source_posterior_usable'] for r in rows),
        'no_model_adopted':True,'independent_confirmation':False,'cost':spec['cost']})


if __name__=='__main__':run()
