"""Replay completed CDF/identity/KG precision without source resampling."""

from performance.complete_historical_source_heads import ROOT,read,save
from performance.check_historical_predictive_cdf import run as cdf_check
from performance.check_historical_predictive_martingale import run as density_check
from performance.refine_historical_mixture_lookahead import run as kg_check
from performance.check_historical_positive_likelihood_reference import run as reference_check
from performance.refine_historical_factorized_tower import run as factorized_check


def run():
    cdf=cdf_check('performance/manifests/historical_predictive_cdf_v2.json')
    if not cdf['all_precision_gates_passed']:raise RuntimeError('Predictive CDF precision remains HOLD')
    density=density_check('performance/manifests/historical_predictive_martingale_v2.json')
    if not density['all_layouts_complete']:raise RuntimeError('Predictive identity precision remains HOLD')
    kg=kg_check('performance/manifests/historical_mixture_lookahead_precision_v4.json')
    reference=(reference_check('performance/manifests/historical_positive_likelihood_reference_v3.json')
        if not kg['all_latest_precision_gates_passed'] else None)
    factorized=factorized_check('performance/manifests/historical_factorized_tower_v1.json')
    latest=[next(r for r in reversed(factorized['rows']) if r['layout']==layout) for layout in ('5','6')]
    out=ROOT/'paper_artifacts/historical_source_variance_resolution_v4'
    source=read(str((out/'batch_results.json').relative_to(ROOT)))
    tests=read('paper_artifacts/historical_factorized_tower_v1/unit_tests.json')
    fixed_passed=all(r['likelihood_tower_precision_passed'] and r['projected_tower_precision_passed'] for r in latest)
    result={'status':'source,CDF,identity and two-query fixed-action precision closed;complete ranking open'
            if fixed_passed else 'source,CDF and identity precision closed;fixed-action precision HOLD',
        'source_batch':'paper_artifacts/historical_source_variance_resolution_v4/batch_results.json',
        'source_usable_cases':source['source_usable_cases'],'complete_source_identities':16,
        'source_cases_resampled_in_this_phase':0,
        'predictive_CDF_precision':'paper_artifacts/historical_predictive_cdf_v2/results.json',
        'all32_predictive_CDF_heads_passed':cdf['all_precision_gates_passed'],
        'density_only_precision':'paper_artifacts/historical_predictive_martingale_v2/results.json',
        'density_only_identity_precision_passed':density['all_layouts_complete'],
        'positive_mixture_KG_precision':'paper_artifacts/historical_mixture_lookahead_precision_v4/results.json',
        'raw_joint_KG_precision_passed':kg['all_latest_precision_gates_passed'],
        'latest_query_precision':[{k:r[k] for k in ('layout','candidate','net_reward_of_paid_episode',
            'adaptation_value','filter_and_quadrature_baseline_drift',
            'maximum_identity_martingale_residual','precision_gate_passed')} for r in kg['rows']],
        'tests_passed':tests['passed'],'tests_run':tests['tests_run'],
        'test_record':'paper_artifacts/historical_factorized_tower_v1/unit_tests.json',
        'cost':kg['cost'],'scientific_model_ranking_authorized':False,'independent_confirmation':False,
        'no_model_adopted':True,
        'non_Gaussian_reference':'paper_artifacts/historical_positive_likelihood_reference_v3/results.json' if reference else None,
        'reference_adequate':reference['all_references_adequate'] if reference else None,
        'reference_projection_attribution':[{k:r[k] for k in ('layout','likelihood_tower_drift',
            'projected_baseline_drift','projection_minus_likelihood_baseline','predictive_log_evidence_max_error',
            'effective_sample_size_min','tower_precision_gate_passed')} for r in reference['rows']] if reference else [],
        'factorized_terminal_precision':'paper_artifacts/historical_factorized_tower_v1/results.json',
        'fixed_action_precision_passed':fixed_passed,
        'latest_fixed_action_precision':[{k:r[k] for k in ('layout','fantasy_power','head_raw_updates',
            'likelihood_tower_drift','projected_baseline_drift','control_variate_net_diagnostic',
            'likelihood_tower_precision_passed','projected_tower_precision_passed')} for r in latest],
        'next_work':{'numerical_integration':'Connect the independently integrated fixed-action baseline to the existing adaptive contrast as a control variate for complete action comparison. The adaptive maximum remains joint;two opened queries do not certify a full ranking.',
            'source_learning':'Reuse performance/check_legacy_f_raw_covariance_mstep.py and the current64-head raw covariance scores for source-only learning and evidence normalization;then matched completeGP controls. No source law selected by opened target decision outcomes.'},
        'legacy_evidence_retained':['performance/baselines/v51_observed_terminal_closure_evidence.json',
            'paper_artifacts/legacy_frontend_evidence.json']}
    save(out/'phase_results.json',result)
    print(result['status'],flush=True);return result


if __name__=='__main__':run()
