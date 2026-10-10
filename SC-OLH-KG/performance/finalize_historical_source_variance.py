"""Reuse full-chain precision except the prespecified integrated variance site."""

import copy
import time

import numpy as np

from performance.resolve_historical_source_variance import (
    ROOT, IDENTITY, VarianceTiltedPosterior, weighted_function_moments, read, save,
)

PLAN = 'performance/manifests/historical_source_variance_resolution_v4.json'


def run(plan=PLAN):
    spec=read(plan);out=ROOT/spec['output'];result_path=out/'source_results.json'
    if result_path.exists():return read(str(result_path.relative_to(ROOT)))
    (out/'source_cases').mkdir(parents=True,exist_ok=True);started=time.process_time()
    previous=read(spec['previous_source_registry']);rows=[dict(r) for r in previous['rows']]
    integrated=read(spec['conditional_source_registry'])['rows'];laws=read(spec['source_laws'])['rows'];checks=[]
    for i,case in enumerate(previous['rows']):
        if case['source_posterior_usable']:continue
        raw=read(case['component_file']);icase=next(c for c in integrated if c['head']==case['head']
            and all(c[k]==case[k] for k in IDENTITY));rb=read(icase['component_file']);site=spec['integrated_source_site']
        assert rb['conditional_integration']['joint_source_site']==site
        assert raw['positive_eta_mixture']['eta_states_by_chain']==rb['positive_eta_mixture']['eta_states_by_chain']
        first=np.array(raw['first_half_posterior_variance']);second=np.array(raw['second_half_posterior_variance'])
        reference=np.array(raw['source_f_posterior_variance'])
        first[site]=rb['first_half_posterior_variance'][site];second[site]=rb['second_half_posterior_variance'][site]
        reference[site]=rb['source_f_posterior_variance'][site]
        diagnostic=copy.deepcopy(raw['diagnostics']);coordinate=144+site
        for key in ('split_rhat','autocorrelation_ess'):
            diagnostic[key][coordinate]=rb['diagnostics'][key][coordinate]
        rhat=max(diagnostic['split_rhat']);ess=min(diagnostic['autocorrelation_ess'])
        half=float(np.max(abs(first-second)/((first+second)/2)));gate=spec['monte_carlo_gate']
        mc=bool(rhat<=gate['maximum_split_rhat_eta_and_rao_blackwell_f_moments']
            and ess>=gate['minimum_autocorrelation_ess_eta_and_rao_blackwell_f_moments']
            and half<=gate['maximum_first_second_half_f_variance_relative_difference'])
        law=next(l for l in laws if l['head']==case['head'] and all(l[k]==case[k] for k in IDENTITY))
        parent=read(spec['covariance_plan']);source=next(s for s in read(parent['source_fields'])['rows']
            if s['layout']==case['layout'] and s['active_arm']==parent['proposal_arm'])
        model=VarianceTiltedPosterior(law['source_mean'],law['source_covariance'],source['batch_means'],
            source['sample_variances'],raw['tilt']['sites'],raw['tilt']['center'],raw['tilt']['scale'])
        mixture=copy.deepcopy(raw['positive_eta_mixture']);eta=np.array(mixture['eta_states_by_chain'])
        moments=[[model.moments(model.evaluate(p)) for p in chain] for chain in eta]
        ds=np.array([[m[0] for m in chain] for chain in moments]);cs=np.array([[m[1] for m in chain] for chain in moments])
        weights=np.array(mixture['positive_component_weights']).reshape(eta.shape[:2])
        delta,cov=weighted_function_moments(ds,cs,weights)
        mean_error=float(np.sqrt(np.mean((model.mean+delta-np.array(raw['source_f_posterior_mean']))**2/reference)))
        variance_error=float(np.max(abs(np.diag(cov)-reference)/reference));retention=spec['retained_mixture']
        mixture.update(source_f_mean_rms_difference_in_full_posterior_sd=mean_error,
            maximum_source_f_diagonal_variance_relative_difference_vs_all_draws=variance_error,
            variance_precision_reference='full-chain all sites;exact joint conditional integral at fixed source site8',
            mixture_gate_passed=bool(mean_error<=retention['maximum_source_f_mean_rms_difference_in_full_posterior_sd']
                and variance_error<=retention['maximum_source_f_diagonal_variance_relative_difference_vs_all_draws']))
        path=out/'source_cases'/('f_layout'+'_'.join(case[k] for k in IDENTITY)+'.json')
        row={**case,'monte_carlo_gate_passed':mc,'maximum_split_rhat':rhat,'minimum_autocorrelation_ess':ess,
            'maximum_first_second_half_f_variance_relative_difference':half,'mixture_gate_passed':mixture['mixture_gate_passed'],
            'source_posterior_usable':bool(mc and mixture['mixture_gate_passed']),
            'component_file':str(path.relative_to(ROOT)),'previous_component_file':case['component_file'],
            'joint_direction_conditionally_integrated':True}
        save(path,{**raw,'execution_plan':plan,'gates':row,'positive_eta_mixture':mixture,
            'source_f_precision_reference_variance':reference.tolist(),'first_half_posterior_variance':first.tolist(),
            'second_half_posterior_variance':second.tolist(),'diagnostics':diagnostic,
            'conditional_integration':rb['conditional_integration'],
            'variance_precision_scope':{'integrated_site':site,'conditional_reference':icase['component_file'],
                'all_other_full_chain_estimates_reused':True,
                'all_target_components_and_weights_unchanged_from_full_chain':True,
                'conditional_reference_Rhat':rb['diagnostics']['split_rhat'][coordinate],
                'conditional_reference_ESS':rb['diagnostics']['autocorrelation_ess'][coordinate]}})
        rows[i]=row;checks.append({**{k:case[k] for k in IDENTITY},'head':case['head'],
            'integrated_site':site,'variance_reference_relative_difference_vs_full_chain':float(
                abs(reference[site]/raw['source_f_posterior_variance'][site]-1.)),
            'retained_variance_error_vs_combined_reference':variance_error,'retained_mean_error':mean_error})
        print('final source usable',row['source_posterior_usable'],'ESS',round(ess,1),'halfvar',round(half,3),
            'retention',round(variance_error,3),flush=True)
    save(out/'source_verification.json',{'execution_plan':plan,'checks':checks})
    result={'execution_plan':plan,'rows':rows,'usable_cases':sum(r['source_posterior_usable'] for r in rows),
        'reused_usable_cases':sum(r['source_posterior_usable'] for r in previous['rows']),
        'all_source_posteriors_usable':all(r['source_posterior_usable'] for r in rows),
        'cpu_seconds':time.process_time()-started,'new_source_draws':0,'cost':spec['cost'],
        'original_population_prior_and_raw_likelihood_unchanged':True,'no_model_adopted':True,'independent_confirmation':False}
    save(result_path,result);return result


if __name__=='__main__':run()
