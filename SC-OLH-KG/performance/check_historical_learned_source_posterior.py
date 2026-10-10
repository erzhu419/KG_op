"""Condition changed source laws without resampling the original passed cases."""

import time

import numpy as np
from scipy.special import digamma

from performance.complete_historical_source_heads import ROOT,IDENTITY,read,save
from performance.check_legacy_f_collapsed_source_raw_update import CollapsedRawSourcePosterior
from performance.learn_historical_raw_source_amplitudes import covariance_builder
from performance.resolve_historical_source_variance import original_posterior_summary

PLAN='performance/manifests/historical_raw_source_amplitude_learning_v1.json'


def run(plan=PLAN):
    spec=read(plan);parent=read(spec['source_plan']);out=ROOT/spec['output']
    path=out/'posterior_precision.json'
    if path.exists():return read(str(path.relative_to(ROOT)))
    learning=read(str((out/'results.json').relative_to(ROOT)));laws=read(parent['source_laws'])['rows']
    selection=read(parent['selection_plan']);rows=[];started=time.process_time()
    for case in learning['evidence_rows']:
        law=next(r for r in laws if r['head']==case['head'] and all(r[k]==case[k] for k in IDENTITY))
        source=read(selection['source_inputs_by_layout'][case['layout']])
        response=np.vstack([t['response'] for t in source['source_tasks'] if t['name'] in selection['source_tasks']])
        head=int(case['head']=='g')
        s2=np.exp(response[:,2 if head else 3]+digamma(1.))
        model=CollapsedRawSourcePosterior(law['source_mean'],covariance_builder(law)(case['f_common_task_scales'])[0],response[:,head],s2)
        eta=np.asarray(read(case['source_case'])['positive_eta_mixture']['eta_states_by_chain'])
        conditional=[model.moments(model.evaluate(x)) for x in eta.reshape(-1,48)]
        ds=np.stack([v[0] for v in conditional]).reshape(*eta.shape)
        cs=np.stack([v[1] for v in conditional]).reshape(*eta.shape[:2],48,48)
        weights=np.asarray(case['new_positive_eta_weights']).reshape(eta.shape[:2])
        summary=original_posterior_summary(model,eta,ds,cs,weights,parent)
        passed=bool(case['source_Mstep_pair_passed'] and case['evidence_ratio_precision_passed'] and summary['monte_carlo_gate_passed'])
        rows.append({**{k:case[k] for k in (*IDENTITY,'head')},'source_case':case['source_case'],
            'source_Mstep_pair_passed':case['source_Mstep_pair_passed'],
            'evidence_ratio_precision_passed':case['evidence_ratio_precision_passed'],
            'new_source_posterior_precision_passed':passed,
            **{k:summary[k] for k in ('maximum_split_rhat','minimum_autocorrelation_ess',
                'maximum_first_second_half_f_variance_relative_difference','normalized_inverse_weight_ESS')},
            'posterior_mean':summary['posterior_mean'],'posterior_covariance':summary['posterior_covariance'],
            'eta_posterior_mean':summary['eta_posterior_mean'],'eta_posterior_covariance':summary['eta_posterior_covariance']})
    result={'execution_plan':plan,'rows':rows,'cases_passed':sum(r['new_source_posterior_precision_passed'] for r in rows),
        'case_count':len(rows),'all_new_posteriors_precision_passed':all(r['new_source_posterior_precision_passed'] for r in rows),
        'original32_source_cases_preserved':True,'new_source_MCMC_sweeps':0,'cpu_seconds':time.process_time()-started,
        'no_model_adopted':True,'independent_confirmation':False,
        'scope':'Changed-law importance conditioning with freshly computed conditional f moments. Same original Rhat,ESS and variance-split gates on ratio-estimator influences;not iterative hyperparameter convergence.'}
    save(path,result)
    print('changed source posterior precision',result['cases_passed'],'/',len(rows),flush=True)
    return result


def run_retained_baseline(plan=PLAN):
    """Separate retention error from a covariance move on identical eta states."""
    spec=read(plan);parent=read(spec['source_plan']);out=ROOT/spec['output']
    path=out/'retained_baseline_precision.json'
    if path.exists():return read(str(path.relative_to(ROOT)))
    learned=run(plan);laws=read(parent['source_laws'])['rows']
    selection=read(parent['selection_plan']);rows=[];started=time.process_time()
    gate=parent['monte_carlo_gate']
    def failed_checks(summary):
        return [name for name,bad in (
            ('Rhat',summary['maximum_split_rhat']>gate['maximum_split_rhat_eta_and_rao_blackwell_f_moments']),
            ('ESS',summary['minimum_autocorrelation_ess']<gate['minimum_autocorrelation_ess_eta_and_rao_blackwell_f_moments']),
            ('variance_split',summary['maximum_first_second_half_f_variance_relative_difference']
                >gate['maximum_first_second_half_f_variance_relative_difference'])) if bad]
    for case in learned['rows']:
        law=next(r for r in laws if r['head']==case['head'] and all(r[k]==case[k] for k in IDENTITY))
        source=read(selection['source_inputs_by_layout'][case['layout']])
        response=np.vstack([t['response'] for t in source['source_tasks'] if t['name'] in selection['source_tasks']])
        head=int(case['head']=='g');s2=np.exp(response[:,2 if head else 3]+digamma(1.))
        model=CollapsedRawSourcePosterior(law['source_mean'],law['source_covariance'],response[:,head],s2)
        mixture=read(case['source_case'])['positive_eta_mixture'];eta=np.asarray(mixture['eta_states_by_chain'])
        conditional=[model.moments(model.evaluate(x)) for x in eta.reshape(-1,48)]
        ds=np.stack([v[0] for v in conditional]).reshape(*eta.shape)
        cs=np.stack([v[1] for v in conditional]).reshape(*eta.shape[:2],48,48)
        weights=(np.asarray(mixture['positive_component_weights']).reshape(eta.shape[:2])
            if 'positive_component_weights' in mixture else np.ones(eta.shape[:2]))
        old=original_posterior_summary(model,eta,ds,cs,weights,parent)
        old_fail=failed_checks(old);new_fail=failed_checks(case)
        rows.append({**{k:case[k] for k in (*IDENTITY,'head')},'source_case':case['source_case'],
            'retained_component_count':int(np.prod(eta.shape[:2])),
            'old_retained_precision_passed':not old_fail,'new_retained_precision_passed':not new_fail,
            'old_retained_failed_checks':old_fail,'new_retained_failed_checks':new_fail,
            'old_retained_diagnostics':{k:old[k] for k in ('maximum_split_rhat','minimum_autocorrelation_ess',
                'maximum_first_second_half_f_variance_relative_difference')},
            'new_retained_diagnostics':{k:case[k] for k in ('maximum_split_rhat','minimum_autocorrelation_ess',
                'maximum_first_second_half_f_variance_relative_difference')},
            'evidence_ratio_precision_passed':case['evidence_ratio_precision_passed'],
            'learned_combined_precision_passed':case['new_source_posterior_precision_passed']})
    result={'execution_plan':plan,'rows':rows,'case_count':len(rows),
        'old_retained_precision_cases_passed':sum(r['old_retained_precision_passed'] for r in rows),
        'new_retained_precision_cases_passed':sum(r['new_retained_precision_passed'] for r in rows),
        'old_pass_new_fail_cases':sum(r['old_retained_precision_passed'] and not r['new_retained_precision_passed'] for r in rows),
        'old_fail_new_fail_cases':sum(not r['old_retained_precision_passed'] and not r['new_retained_precision_passed'] for r in rows),
        'old_fail_new_pass_cases':sum(not r['old_retained_precision_passed'] and r['new_retained_precision_passed'] for r in rows),
        'original_full_chain32_PASS_preserved':True,'new_source_MCMC_sweeps':0,
        'cpu_seconds':time.process_time()-started,'no_model_adopted':True,
        'scope':'Paired old/new numerical diagnostic on identical retained eta states and original precision gates. '
            'Does not replace the original full-chain certificate or certify changed-law posterior tails.'}
    save(path,result)
    print('paired retained precision old/new',result['old_retained_precision_cases_passed'],
        result['new_retained_precision_cases_passed'],'/',len(rows),flush=True)
    return result


if __name__=='__main__':run()
