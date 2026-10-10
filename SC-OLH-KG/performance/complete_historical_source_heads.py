"""Reuse completed source cases and complete both response/risk head pairs."""

import signal
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import digamma, polygamma

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from performance.check_legacy_f_collapsed_source_raw_update import sample
from performance.check_legacy_f_learned_covariance_source_update import retained_mixture, verify_learned
from performance.check_legacy_f_original_source_raw_update import common_moments
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_source_incremental_calibration import log_density
from performance.prepare_legacy_f_variance_reference import VarianceReferencePosterior, variance_reference
from performance.recondition_historical_source_laws import IDENTITY, source_case
from performance.refine_historical_raw_lookahead import cpu_timeout
from performance.run_legacy_related_inventory_source import save

PLAN='performance/manifests/historical_complete_reuse_v1.json'


def run():
    spec=read(PLAN);parent=read(spec['covariance_plan']);selection=read(spec['selection_plan'])
    laws=read(spec['source_laws'])['rows'];old=read(spec['existing_source_results'])['rows']
    fields=read(parent['source_fields'])['rows'];out=ROOT/spec['output'];(out/'source_cases').mkdir(parents=True,exist_ok=True)
    g_summaries={}
    for layout,path in selection['source_inputs_by_layout'].items():
        tasks=[t for t in read(path)['source_tasks'] if t['name'] in selection['source_tasks']]
        assert [t['name'] for t in tasks]==selection['source_tasks']
        response=np.vstack([t['response'] for t in tasks]);noise=np.vstack([t['noise'] for t in tasks])
        s2=np.exp(response[:,2]+digamma(1.))
        np.testing.assert_allclose(s2/3,noise[:,1],atol=1e-14,rtol=1e-10)
        g_summaries[layout]=(response[:,1].tolist(),s2.tolist())
    rows=[];checks=[];costs={};stops=[]
    for head in ('f','g'):
        started=time.process_time();cap=spec['unresolved_f_maximum_single_thread_cpu_seconds' if head=='f' else 'new_g_maximum_single_thread_cpu_seconds']
        handler=signal.signal(signal.SIGPROF,cpu_timeout);signal.setitimer(signal.ITIMER_PROF,cap)
        try:
            for law in (r for r in laws if r['head']==head):
                identity={k:law[k] for k in IDENTITY}
                previous=next(r for r in old if all(r[k]==law[k] for k in IDENTITY)) if head=='f' else None
                if previous is not None and previous['source_posterior_usable']:
                    rows.append({**previous,'head':head,'reused_completed_case':True});continue
                risk=next(r for r in laws if r['head']==('eta_f' if head=='f' else 'eta_g') and all(r[k]==law[k] for k in IDENTITY))
                original=next(r for r in fields if (r['layout'],r['active_arm'])==(law['layout'],parent['proposal_arm']))
                source=dict(original)
                if head=='g':source['batch_means'],source['sample_variances']=g_summaries[law['layout']]
                field=source_case(law,risk,source)
                mean,covariance=variance_reference(field['eta_mean'],field['eta_covariance'],field['sample_variances'])
                approximate=np.log(field['sample_variances'])-digamma(1.)
                reference={'reference_mean':mean,'reference_covariance':covariance,
                    'approximate_Gaussian_variance_evidence':log_density(approximate-np.asarray(field['eta_mean']),
                        np.asarray(field['eta_covariance'])+float(polygamma(1,1.))*np.eye(48))}
                checks.append({**identity,'head':head,**verify_learned(field,0,reference)})
                count=spec['unresolved_f_retained_sweeps_per_chain'] if head=='f' else spec['sampler']['retained_sweeps_per_chain']
                paired={**spec,'maximum_single_thread_cpu_seconds':cap,'sampler':{**spec['sampler'],
                    'retained_sweeps_per_chain':count,'seed':spec['sampler']['seed']
                    +1000*(law['selector']=='additive')+10*(law['support']=='joint')+(10000 if head=='g' else 0)},
                    'retained_mixture':{**spec['retained_mixture'],'stride_per_chain':count//128}}
                model=VarianceReferencePosterior(field['f_means'][0],field['f_covariances'][0],
                    field['batch_means'],field['sample_variances'])
                captured=[];posterior=sample(field,0,paired,started,model,mean,covariance,captured)
                states=captured[0]
                if previous is not None:
                    saved=read(previous['component_file'])
                    np.testing.assert_allclose(states[:,:1024:8],saved['positive_eta_mixture']['eta_states_by_chain'],atol=1e-10,rtol=0.)
                    checks.append({**identity,'head':head,'original1024_prefix_and512_states_preserved':True})
                mixture=retained_mixture(field,0,model,states,posterior,paired)
                cm,cc=common_moments(field,0,np.asarray(posterior['posterior_mean']),np.asarray(posterior['posterior_covariance']))
                name=head+'_layout'+'_'.join(identity.values())+'.json';path=out/'source_cases'/name
                row={**identity,'head':head,'reused_completed_case':False,
                    **{k:posterior[k] for k in ('monte_carlo_gate_passed','maximum_split_rhat','minimum_autocorrelation_ess',
                        'maximum_first_second_half_f_variance_relative_difference','elliptical_likelihood_evaluations_by_chain')},
                    'mixture_gate_passed':mixture['mixture_gate_passed'],'source_posterior_usable':bool(posterior['monte_carlo_gate_passed'] and mixture['mixture_gate_passed']),
                    'retained_sweeps_per_chain':count,'component_file':str(path.relative_to(ROOT))}
                save(path,{'execution_plan':PLAN,**identity,'head':head,'gates':row,'source_laws':spec['source_laws'],
                    'positive_eta_mixture':mixture,'common_coefficient_posterior_mean':cm.tolist(),'common_coefficient_posterior_covariance':cc.tolist(),
                    'source_f_posterior_mean':posterior['posterior_mean'],'source_f_posterior_variance':np.diag(posterior['posterior_covariance']).tolist(),
                    'full_source_eta_posterior_mean':posterior['eta_posterior_mean'],'full_source_eta_posterior_covariance':posterior['eta_posterior_covariance'],
                    'diagnostics':posterior['diagnostics'],'first_half_posterior_mean':posterior['first_half_posterior_mean'],
                    'second_half_posterior_mean':posterior['second_half_posterior_mean'],'first_half_posterior_variance':posterior['first_half_posterior_variance'],
                    'second_half_posterior_variance':posterior['second_half_posterior_variance']})
                rows.append(row)
                print('complete source',head,*identity.values(),'usable',row['source_posterior_usable'],
                    'ESS',round(row['minimum_autocorrelation_ess'],1),'halfvar',round(row['maximum_first_second_half_f_variance_relative_difference'],3),flush=True)
                save(out/'source_results.json',{'execution':spec,'rows':rows,'all32_cases_complete':len(rows)==32,'independent_confirmation':False})
        except TimeoutError:
            signal.setitimer(signal.ITIMER_PROF,0.);stops.append(head)
        finally:
            signal.setitimer(signal.ITIMER_PROF,0.);signal.signal(signal.SIGPROF,handler)
        costs[head]=time.process_time()-started
    save(out/'source_verification.json',{'execution':spec,'checks':checks,'all_completed_pre_sampling_checks_passed':True})
    save(out/'source_results.json',{'execution':spec,'rows':rows,'all32_cases_complete':len(rows)==32,
        'usable_cases':sum(r['source_posterior_usable'] for r in rows),'reused_completed_cases':sum(r['reused_completed_case'] for r in rows),
        'all_source_posteriors_usable':len(rows)==32 and all(r['source_posterior_usable'] for r in rows),
        'stopped_head_phases':stops,'cpu_seconds_by_phase':costs,'no_model_adopted':True,'independent_confirmation':False,
        'cost':spec['cost'],'interpretation':'Existing usable f cases reused;unresolved f chains extended with their exact original prefix;g uses the same Normal/chi-square raw summaries and historical sampler. Numerical source conditioning is separate from structure-selection uncertainty,hyperparameter convergence and decision advantage.'})


if __name__=='__main__':run()
