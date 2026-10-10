"""Condition only the actual final risk pilot after parent overlap fails."""

import time

import numpy as np

from performance.complete_historical_source_heads import ROOT, IDENTITY, read, save
from performance.recondition_historical_changed_candidates import candidate_inputs
from performance.prepare_legacy_f_variance_reference import VarianceReferencePosterior
from performance.check_legacy_f_collapsed_source_raw_update import sample
from performance.check_legacy_f_learned_covariance_source_update import retained_mixture, verify_learned
from performance.resolve_historical_source_variance import original_posterior_summary
from performance.certify_historical_reused_risk_candidate import response_stationarity

PLAN = 'performance/manifests/historical_source_risk_state_reuse_conditioning_v1.json'
KEY = (*IDENTITY, 'head')


def run(plan=PLAN):
    spec = read(plan); out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    path = out / 'results.json'
    if path.exists():
        return read(str(path.relative_to(ROOT)))
    started = time.process_time(); pilot = read(spec['risk_pilot'])
    previous = read(spec['failed_parent_reweighting'])
    assert not previous['final_candidate_posterior_certified']
    parent = read(spec['source_plan']); laws = read(parent['source_laws'])['rows']
    selection = read(parent['selection_plan'])
    current = next(r for r in read(spec['parent_learning'])['evidence_rows']
        if all(r[k] == pilot[k] for k in KEY))
    case = {**current, 'eta_common_task_scales': pilot['final_risk_scales']}
    _, relative, _, field, reference = candidate_inputs(case, laws, selection)
    model = VarianceReferencePosterior(relative.mean, relative.covariance, relative.labels, relative.s2)
    verification = verify_learned(field, 0, reference)
    states = []; moments = []
    seed = spec['sampler']['seed'] + 1000 * (case['selector'] == 'additive') \
        + 10 * (case['support'] == 'joint') + 10000 * (case['head'] == 'g')
    execution = {**spec, 'sampler': {**spec['sampler'], 'seed': seed}}
    sample(field, 0, execution, started, model, reference['reference_mean'], reference['reference_covariance'],
        states, retained_moments=moments)
    eta = states[0]; ds, cs = moments[0]; weights = np.ones(eta.shape[:2])
    summary = original_posterior_summary(model, eta, ds, cs, weights, spec)
    mixture = retained_mixture(field, 0, model, eta, summary, spec)
    law = next(r for r in laws if all(r[k] == case[k] for k in KEY))
    response = response_stationarity(model, eta, ds, cs, weights, law, case['f_common_task_scales'], spec)
    component = out / ('candidate_' + '_'.join(case[k] for k in KEY) + '.json')
    save(component, {'execution_plan': plan, 'fixed_candidate': {k: case[k] for k in
        (*KEY, 'f_common_task_scales', 'eta_common_task_scales')}, 'posterior': summary,
        'positive_eta_mixture': mixture})
    result = {'execution_plan': plan, **{k: case[k] for k in KEY},
        'final_risk_scales': case['eta_common_task_scales'], 'fixed_response_scales': case['f_common_task_scales'],
        'conditional_risk_stationarity_passed': pilot['conditional_risk_stationarity_passed'],
        'verification': verification, 'candidate_component_file': str(component.relative_to(ROOT)),
        'posterior_functional_precision_passed': summary['monte_carlo_gate_passed'],
        'point_mixture_retention_passed': mixture['mixture_gate_passed'],
        'final_posterior_and_retention_certified': bool(summary['monte_carlo_gate_passed'] and mixture['mixture_gate_passed']),
        'maximum_split_rhat': summary['maximum_split_rhat'],
        'minimum_autocorrelation_ess': summary['minimum_autocorrelation_ess'],
        'maximum_first_second_half_f_variance_relative_difference':
            summary['maximum_first_second_half_f_variance_relative_difference'],
        'mixture_variance_error': mixture['maximum_source_f_variance_relative_difference'],
        'mixture_mean_error_in_posterior_sd': mixture['source_f_mean_rms_difference_in_full_posterior_sd'],
        'mixture_components': spec['retained_mixture']['components_per_arm'],
        'response_stationarity_at_final_risk': response,
        'new_candidate_MCMC_sweeps': spec['sampler']['chains'] * (spec['sampler']['warmup_sweeps_per_chain']
            + spec['sampler']['retained_sweeps_per_chain']),
        'source_registry_replaced': False, 'joint_source_learning_converged': False,
        'final_vs_parent_evidence_precision_passed': False,
        'cpu_seconds': time.process_time() - started, 'new_source_labels': 0,
        'new_target_simulator_calls': 0, 'new_verifier_calls': 0, 'server_jobs': 0,
        'no_model_adopted': True, 'independent_confirmation': False,
        'scope': 'One fixed final conditional-risk candidate. Original response posterior and retained-mixture gates;no optimizer adoption. The failed direct parent evidence estimate remains unresolved. Original32 source laws and current registry are unchanged.'}
    save(path, result)
    print('Final risk fresh conditioning', result['final_posterior_and_retention_certified'],
        'response stationary', response['single_iteration_stationarity_passed'],
        'CPU', round(result['cpu_seconds'], 2), flush=True)
    return result


if __name__ == '__main__':
    run()
