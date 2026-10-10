"""Certify an existing current-law weighted mixture against a saved same-law reference."""

import time

import numpy as np

from performance.complete_historical_source_heads import ROOT, IDENTITY, read, save
from performance.recondition_historical_changed_candidates import candidate_inputs
from performance.prepare_legacy_f_variance_reference import VarianceReferencePosterior
from performance.resolve_historical_source_variance import weighted_function_moments

PLAN = 'performance/manifests/historical_source_learning_round6_mixture_reuse_v1.json'
KEY = (*IDENTITY, 'head')


def run(plan=PLAN):
    spec = read(plan); out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    path = out / 'results.json'
    if path.exists():
        return read(str(path.relative_to(ROOT)))
    started = time.process_time()
    same = lambda r: all(r[k] == spec['case'][k] for k in KEY)
    case = next(r for r in read(spec['learning_results'])['evidence_rows'] if same(r))
    post = next(r for r in read(spec['initial_posterior'])['rows'] if same(r))
    full = next(r for r in read(spec['fresh_results'])['rows'] if same(r))
    reference = next(r for r in read(spec['functional_results'])['functional_repairs'] if same(r))
    source = read(spec['source_plan'])
    _, relative, _, _, _ = candidate_inputs(case, read(source['source_laws'])['rows'], read(source['selection_plan']))
    model = VarianceReferencePosterior(relative.mean, relative.covariance, relative.labels, relative.s2)
    saved = read(post['candidate_component_file']); mixture = saved['positive_eta_mixture']
    for name in ('f_common_task_scales', 'eta_common_task_scales'):
        np.testing.assert_allclose(saved['fixed_candidate'][name], case[name], atol=0., rtol=0.)
    states = np.asarray(mixture['eta_states_by_chain'])
    weights = np.asarray(mixture['positive_component_weights']).reshape(states.shape[:2])
    moments = [model.moments(model.evaluate(p)) for p in states.reshape(-1, 48)]
    delta, covariance = weighted_function_moments(np.stack([m[0] for m in moments]).reshape(states.shape),
        np.stack([m[1] for m in moments]).reshape(*states.shape[:2], 48, 48), weights)
    variances = np.asarray(reference['posterior_variance_reference'])
    ve = float(np.max(abs(np.diag(covariance) - variances) / variances))
    me = float(np.sqrt(np.mean((model.mean + delta - full['posterior_mean']) ** 2 / variances)))
    rhat = max(post['maximum_split_rhat'], reference['maximum_split_rhat'])
    ess = min(post['minimum_autocorrelation_ess'], reference['minimum_autocorrelation_ess'])
    gate, retain = source['monte_carlo_gate'], source['retained_mixture']
    passed = bool(case['source_Mstep_pair_passed'] and case['evidence_ratio_precision_passed']
        and rhat <= gate['maximum_split_rhat_eta_and_rao_blackwell_f_moments']
        and ess >= gate['minimum_autocorrelation_ess_eta_and_rao_blackwell_f_moments']
        and reference['maximum_first_second_half_f_variance_relative_difference']
            <= gate['maximum_first_second_half_f_variance_relative_difference']
        and ve <= retain['maximum_source_f_diagonal_variance_relative_difference_vs_all_draws']
        and me <= retain['maximum_source_f_mean_rms_difference_in_full_posterior_sd'])
    result = {'execution_plan': plan, **{k: case[k] for k in KEY}, 'functional_precision_passed': passed,
        'maximum_split_rhat': rhat, 'minimum_autocorrelation_ess': ess,
        'maximum_first_second_half_f_variance_relative_difference': reference['maximum_first_second_half_f_variance_relative_difference'],
        'retained_mixture_variance_error': ve, 'retained_mixture_mean_error': me,
        'posterior_variance_reference': variances.tolist(), 'candidate_component_file': post['candidate_component_file'],
        'current_law_parameters_match_exactly': True, 'old_weights_not_reused': True,
        'functional_reference': spec['functional_results'], 'mean_reference': spec['fresh_results'],
        'initial_precision_reference': spec['initial_posterior'], 'cpu_seconds': time.process_time() - started,
        'new_candidate_MCMC_sweeps': 0, 'new_source_labels': 0, 'server_jobs': 0,
        'no_model_adopted': True, 'independent_confirmation': False}
    save(path, result)
    print('same-law weighted mixture reuse', passed, 'variance error', round(ve, 5),
        'mean error', round(me, 5), flush=True)
    return result


if __name__ == '__main__':
    run()
