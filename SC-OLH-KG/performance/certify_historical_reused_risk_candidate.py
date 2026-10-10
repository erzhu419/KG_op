"""Check the final conditional-risk pilot using its actual parent components."""

import time

import numpy as np
from scipy.optimize import minimize

from core.gpr import posterior_mixture_weights
from performance.complete_historical_source_heads import ROOT, IDENTITY, read, save
from performance.continue_historical_raw_source_learning import load_mixture, sufficient_moments
from performance.recondition_historical_changed_candidates import candidate_inputs
from performance.check_legacy_f_collapsed_source_raw_update import CollapsedRawSourcePosterior
from performance.learn_historical_raw_source_amplitudes import covariance_builder
from performance.check_legacy_f_risk_amplitude_importance import log_ratios
from performance.probe_historical_source_risk_plateau import ratio_summary
from performance.resolve_historical_source_variance import original_posterior_summary
from performance.check_legacy_f_raw_covariance_mstep import expected_objective
from performance.diagnose_legacy_f_raw_source_covariance_score import projected

PLAN = 'performance/manifests/historical_source_risk_state_reuse_certification_v1.json'
KEY = (*IDENTITY, 'head')


def response_stationarity(model, eta, ds, cs, weights, law, theta, spec):
    fb = covariance_builder(law); theta = np.asarray(theta)
    scale = np.where(theta > 0., theta, 1.)
    seconds = [sufficient_moments(model, eta, ds, cs, weights, p)[0]
        for p in (slice(None), slice(0, eta.shape[1] // 2), slice(eta.shape[1] // 2, None))]
    def objective(z):
        value, gradient = expected_objective(z * scale, fb, seconds[0])
        return value, gradient * scale
    fit = minimize(objective, theta / scale, method='L-BFGS-B', jac=True, bounds=[(0., None)] * 2,
        options=spec['optimizer'])
    proposed = fit.x * scale; gradient = expected_objective(theta, fb, seconds[0])[1]
    pg = projected(theta, gradient, np.full(2, np.inf)) * scale
    change = float(np.linalg.norm(fb(proposed)[0] - fb(theta)[0]) / np.linalg.norm(fb(theta)[0]))
    return {'current_scales': theta.tolist(), 'proposed_scales': proposed.tolist(),
        'scaled_projected_raw_score': pg.tolist(), 'relative_covariance_change': change,
        'optimizer_converged': bool(fit.success), 'full_and_split_Q_improvements': [
            expected_objective(theta, fb, s)[0] - expected_objective(proposed, fb, s)[0] for s in seconds],
        'single_iteration_stationarity_passed': bool(fit.success
            and max(abs(pg)) <= spec['convergence']['maximum_scaled_projected_raw_score']
            and change <= spec['convergence']['maximum_relative_covariance_change']),
        'response_parameters_changed': False}


def run(plan=PLAN):
    spec = read(plan); out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    path = out / 'results.json'
    if path.exists():
        return read(str(path.relative_to(ROOT)))
    started = time.process_time(); pilot = read(spec['risk_pilot'])
    source = read(spec['source_plan']); laws = read(source['source_laws'])['rows']
    selection = read(source['selection_plan'])
    current = next(r for r in read(spec['parent_learning'])['evidence_rows']
        if all(r[k] == pilot[k] for k in KEY))
    record = next(r for r in read(spec['parent_precision'])['rows']
        if all(r[k] == pilot[k] for k in KEY))
    case = {**current, 'eta_common_task_scales': pilot['final_risk_scales']}
    assert case['f_common_task_scales'] == pilot['fixed_response_scales']
    law = next(r for r in laws if all(r[k] == case[k] for k in KEY))
    risk = next(r for r in laws if r['head'] == ('eta_g' if case['head'] == 'g' else 'eta_f')
        and all(r[k] == case[k] for k in IDENTITY))
    _, relative, _, _, _ = candidate_inputs(current, laws, selection)
    model = CollapsedRawSourcePosterior(relative.mean, relative.covariance, relative.labels, relative.s2)
    eta, weights = load_mixture(record, current)
    eb = covariance_builder(risk)
    lr = log_ratios(eta.reshape(-1, 48) - np.asarray(risk['source_mean']),
        eb(current['eta_common_task_scales'])[0], eb(case['eta_common_task_scales'])[0]).reshape(weights.shape)
    evidence = ratio_summary(lr, weights, spec['evidence_precision'])
    _, nw = posterior_mixture_weights(weights.ravel(), lr.ravel()); nw = nw.reshape(weights.shape)
    moments = [model.moments(model.evaluate(p)) for p in eta.reshape(-1, 48)]
    ds = np.stack([m[0] for m in moments]).reshape(eta.shape)
    cs = np.stack([m[1] for m in moments]).reshape(*eta.shape[:2], 48, 48)
    summary = original_posterior_summary(model, eta, ds, cs, nw, source)
    response = response_stationarity(model, eta, ds, cs, nw, law, case['f_common_task_scales'], spec)
    passed = bool(evidence['precision_gate_passed'] and evidence['eligible_improvement_excludes_zero']
        and summary['monte_carlo_gate_passed'])
    component = out / ('candidate_' + '_'.join(case[k] for k in KEY) + '.json')
    save(component, {'execution_plan': plan, 'fixed_candidate': {k: case[k] for k in
        (*KEY, 'f_common_task_scales', 'eta_common_task_scales')}, 'posterior': summary,
        'positive_eta_mixture': {'eta_states_by_chain': eta.tolist(),
            'positive_component_weights': nw.ravel().tolist()},
        'previous_candidate_component_file': record['positive_mixture_component_file']})
    result = {'execution_plan': plan, **{k: case[k] for k in KEY},
        'final_risk_scales': case['eta_common_task_scales'], 'fixed_response_scales': case['f_common_task_scales'],
        'candidate_component_file': str(component.relative_to(ROOT)),
        'conditional_risk_stationarity_passed': pilot['conditional_risk_stationarity_passed'], 'source_evidence': evidence,
        'posterior_functional_precision_passed': summary['monte_carlo_gate_passed'],
        'maximum_split_rhat': summary['maximum_split_rhat'],
        'minimum_autocorrelation_ess': summary['minimum_autocorrelation_ess'],
        'maximum_first_second_half_f_variance_relative_difference':
            summary['maximum_first_second_half_f_variance_relative_difference'],
        'final_candidate_posterior_certified': passed,
        'mixture_components': int(nw.size), 'components_reduced': False,
        'response_stationarity_at_final_risk': response,
        'joint_source_learning_converged': False, 'source_registry_replaced': False,
        'new_candidate_MCMC_sweeps': 0, 'new_source_labels': 0, 'new_target_simulator_calls': 0,
        'new_verifier_calls': 0, 'server_jobs': 0, 'cpu_seconds': time.process_time() - started,
        'no_model_adopted': True, 'independent_confirmation': False,
        'scope': 'Actual final conditional-risk law reweighted from the certified parent. Directly certify all retained states;no further component reduction. Response M-step is diagnostic only. A passing conditional-risk pilot does not establish joint or32-case convergence.'}
    save(path, result)
    print('Final reused risk candidate posterior', passed, 'response stationary',
        response['single_iteration_stationarity_passed'], 'CPU', round(result['cpu_seconds'], 3), flush=True)
    return result


if __name__ == '__main__':
    run()
