"""Audit both source heads on consecutive actual risk-learning laws."""

import time

import numpy as np
from scipy.optimize import minimize

from core.gpr import posterior_mixture_weights
from performance.complete_historical_source_heads import ROOT, IDENTITY, read, save
from performance.continue_historical_raw_source_learning import load_mixture
from performance.recondition_historical_changed_candidates import candidate_inputs
from performance.prepare_legacy_f_variance_reference import VarianceReferencePosterior, variance_reference
from performance.check_legacy_f_collapsed_source_raw_update import CollapsedRawSourcePosterior
from performance.extend_historical_source_risk_score_bank import draw_risk_states
from performance.reuse_historical_source_risk_states import risk_statistics
from performance.learn_historical_raw_source_amplitudes import covariance_builder
from performance.check_legacy_f_risk_amplitude_importance import log_ratios
from performance.check_legacy_f_raw_covariance_mstep import expected_objective
from performance.diagnose_legacy_f_raw_source_covariance_score import projected
from performance.certify_historical_reused_risk_candidate import response_stationarity
from performance.resolve_historical_source_variance import original_posterior_summary

PLAN = 'performance/manifests/historical_joint_source_stationarity_v1.json'
KEY = (*IDENTITY, 'head')


def precise(summaries, spec):
    limits = spec['score_precision']
    return bool(all(max(s['split_rhat']) <= limits['maximum_split_Rhat']
        and min(s['autocorrelation_ess']) >= limits['minimum_functional_ESS'] for s in summaries)
        and max(summaries[0]['scaled_95_MC_halfwidth']) <= limits['maximum_scaled_95_MC_halfwidth'])


def stored_diagnostic(attempt, spec):
    theta = np.asarray(attempt['current_theta']); summaries = attempt['old_score_summaries']
    pg = np.asarray(summaries[0]['scaled_raw_score'])
    pg = np.where((theta == 0.) & (pg > 0.), 0., pg)
    return {'score_summaries': summaries, 'score_precision_passed': precise(summaries, spec),
        'scaled_projected_raw_score': pg.tolist(), 'relative_diagnostic_covariance_change': attempt['relative_covariance_change'],
        'stationarity_diagnostic_passed': bool(precise(summaries, spec) and attempt['optimizer_converged']
            and max(abs(pg)) <= spec['convergence']['maximum_scaled_projected_raw_score']
            and attempt['relative_covariance_change'] <= spec['convergence']['maximum_relative_covariance_change']),
        'parameters_changed': False}


def fixed_diagnostic(result, spec):
    summaries = result['current_score_summaries']
    return {'score_summaries': summaries, 'score_precision_passed': precise(summaries, spec),
        'scaled_projected_raw_score': result['scaled_projected_current_score'],
        'relative_diagnostic_covariance_change': result['new_diagnostic_covariance_change'],
        'stationarity_diagnostic_passed': result['new_current_stationarity_diagnostic_passed'], 'parameters_changed': False}


def fitted_diagnostic(seconds, summaries, is_precise, builder, theta, spec):
    scale = np.where(theta > 0., theta, 1.)
    def objective(z):
        v, g = expected_objective(z * scale, builder, seconds[0]); return v, g * scale
    fit = minimize(objective, theta / scale, method='L-BFGS-B', jac=True, bounds=[(0., None)] * 2,
        options=spec['optimizer'])
    change = float(np.linalg.norm(builder(fit.x * scale)[0] - builder(theta)[0]) / np.linalg.norm(builder(theta)[0]))
    pg = projected(theta, expected_objective(theta, builder, seconds[0])[1], np.full(2, np.inf)) * scale
    return {'score_summaries': summaries, 'score_precision_passed': is_precise,
        'scaled_projected_raw_score': pg.tolist(), 'relative_diagnostic_covariance_change': change,
        'diagnostic_proposed_scales': (fit.x * scale).tolist(), 'optimizer_converged': bool(fit.success),
        'stationarity_diagnostic_passed': bool(is_precise and fit.success
            and change <= spec['convergence']['maximum_relative_covariance_change']
            and max(abs(pg)) <= spec['convergence']['maximum_scaled_projected_raw_score']), 'parameters_changed': False}


def joint_confirmation(previous_theta, current_theta, previous_risk, current_risk,
        previous_response, current_response, previous_posterior_precise, accepted_transition):
    return bool(previous_theta != current_theta and accepted_transition and previous_posterior_precise
        and previous_risk['score_precision_passed'] and previous_risk['stationarity_diagnostic_passed']
        and current_risk['score_precision_passed'] and current_risk['stationarity_diagnostic_passed']
        and previous_response['single_iteration_stationarity_passed']
        and current_response['single_iteration_stationarity_passed'])


def case_run(item, spec, cases, laws, selection):
    suffix = '_'.join(item[k] for k in KEY); out = ROOT / spec['output'] / 'cases' / suffix
    out.mkdir(parents=True, exist_ok=True); path = out / 'results.json'
    if path.exists():
        return read(str(path.relative_to(ROOT)))
    started = time.process_time(); candidate = read(item['candidate_results'])
    actual = read(candidate['actual_risk_results'])
    case = next(c for c in cases if all(c[k] == item[k] for k in KEY))
    for name, field in [('f_common_task_scales', 'fixed_response_scales'), ('eta_common_task_scales', 'final_risk_scales')]:
        assert case[name] == item[field] == actual[field]
    risk = next(r for r in laws if r['head'] == ('eta_g' if case['head'] == 'g' else 'eta_f')
        and all(r[k] == case[k] for k in IDENTITY))
    law = next(r for r in laws if all(r[k] == case[k] for k in KEY))
    builder = covariance_builder(risk); mean = np.asarray(risk['source_mean']); theta = np.asarray(case['eta_common_task_scales'])
    _, relative, _, _, _ = candidate_inputs(case, laws, selection)
    driver = actual if 'attempts' in actual else read(actual['previous_risk_results'])
    transition = None
    if 'attempts' not in actual and actual['accepted_fixed_risk_update']:
        previous_theta = actual['checked_current_scales']; previous = fixed_diagnostic(actual, spec)
        accepted = actual['fixed_proposal_qualified']; transition = candidate['actual_risk_results']
    else:
        prior = next((a for a in reversed(driver['attempts'])
            if a['update_accepted'] and a['proposed_theta'] == theta.tolist() and a['current_theta'] != theta.tolist()), None)
        previous_theta = prior['current_theta'] if prior else None
        previous = stored_diagnostic(prior, spec) if prior else None
        accepted = bool(prior and min(prior['full_and_split_Q_improvements']) > 0.
            and prior['evidence']['precision_gate_passed'] and prior['evidence']['eligible_improvement_excludes_zero'])
        if prior:
            transition = {'risk_results': actual.get('previous_risk_results', candidate['actual_risk_results']),
                'attempt': prior['attempt'], 'full_and_split_Q_improvements': prior['full_and_split_Q_improvements'],
                'source_evidence': prior['evidence']}
    parent = read(item['evidence_results']); current = parent['final_risk_diagnostic_from_parent_bank']
    assert parent['final_risk_scales'] == theta.tolist()
    assert parent['fixed_response_scales'] == case['f_common_task_scales']
    method = 'certified final-law diagnostic from parent bank'; sweeps = 0; prefix_error = None; seconds_path = None
    if current is None or not current['score_precision_passed']:
        if 'attempts' not in actual and not actual['accepted_fixed_risk_update']:
            assert actual['checked_current_scales'] == theta.tolist()
            current = fixed_diagnostic(actual, spec); method = 'existing same-law prefix precision diagnostic'
        elif driver['attempts'][-1]['current_theta'] == theta.tolist():
            current = stored_diagnostic(driver['attempts'][-1], spec); method = 'existing actual final-law risk attempt'
        else:
            assert item['replay_nearest_risk_bank']
            original = read(driver['execution_plan']); last = driver['attempts'][-1]
            bank = driver['sampling_batches'][last['bank'] - 1]; anchor = np.asarray(bank['anchor_theta'])
            rm, rc = variance_reference(mean, builder(anchor)[0], relative.s2)
            model = VarianceReferencePosterior(relative.mean, relative.covariance, relative.labels, relative.s2)
            sampler = {**original['sampler'], 'seed': bank['seed']}
            states = draw_risk_states(model, rm, rc, case, sampler, started, spec['maximum_case_single_thread_cpu_seconds'])
            errors = states - mean
            _, anchor_summaries, _ = risk_statistics(errors, np.ones(states.shape[:2]), builder, anchor, spec)
            first = next(a for a in driver['attempts'] if a['bank'] == last['bank'])
            assert first['current_theta'] == bank['anchor_theta']
            prefix_error = float(np.max(abs(np.asarray(anchor_summaries[0]['scaled_raw_score'])
                - first['old_score_summaries'][0]['scaled_raw_score'])))
            assert prefix_error < 1e-9
            lr = log_ratios(errors.reshape(-1, 48), builder(anchor)[0], builder(theta)[0]).reshape(states.shape[:2])
            _, weights = posterior_mixture_weights(np.ones(lr.size) / lr.size, lr.ravel())
            seconds, summaries, is_precise = risk_statistics(errors, weights.reshape(lr.shape), builder, theta, spec)
            current = fitted_diagnostic(seconds, summaries, is_precise, builder, theta, spec)
            seconds_path = str((out / 'current_risk_sufficient_statistics.json').relative_to(ROOT))
            save(ROOT / seconds_path, {'current_candidate_parameters': {k: case[k] for k in (*KEY,
                'f_common_task_scales', 'eta_common_task_scales')}, 'risk_prior_mean': mean.tolist(),
                'full_and_split_second_moments_about_prior_mean': [s.tolist() for s in seconds],
                'score_diagnostic': current, 'sampling_anchor_scales': anchor.tolist(), 'seed': bank['seed']})
            sweeps = sampler['chains'] * (sampler['warmup_sweeps_per_chain'] + sampler['retained_sweeps_per_chain'])
            method = 'nearest actual risk bank replay at final law'
    old_response = None; old_post_precise = False; old_post_summary = None
    if previous_theta is not None:
        record = {'positive_mixture_component_file': item['candidate_component_file'],
            'positive_weights_results': item['candidate_component_file']}
        eta, weights = load_mixture(record, case)
        lr = log_ratios(eta.reshape(-1, 48) - mean, builder(theta)[0], builder(previous_theta)[0])
        _, weights = posterior_mixture_weights(weights.ravel(), lr); weights = weights.reshape(eta.shape[:2])
        model = CollapsedRawSourcePosterior(relative.mean, relative.covariance, relative.labels, relative.s2)
        moments = [model.moments(model.evaluate(p)) for p in eta.reshape(-1, 48)]
        ds = np.stack([v[0] for v in moments]).reshape(eta.shape)
        cs = np.stack([v[1] for v in moments]).reshape(*eta.shape[:2], 48, 48)
        old_post_summary = original_posterior_summary(model, eta, ds, cs, weights, spec)
        old_post_precise = old_post_summary['monte_carlo_gate_passed']
        old_response = response_stationarity(model, eta, ds, cs, weights, law, case['f_common_task_scales'], spec)
    confirmed = bool(previous is not None and joint_confirmation(previous_theta, theta.tolist(), previous, current,
        old_response, item['response_stationarity_at_final_risk'], old_post_precise, accepted))
    current_passed = bool(current['score_precision_passed'] and current['stationarity_diagnostic_passed']
        and item['response_stationarity_at_final_risk']['single_iteration_stationarity_passed'])
    result = {**{k: case[k] for k in KEY}, 'execution_plan': PLAN,
        'current_f_common_task_scales': case['f_common_task_scales'], 'current_eta_common_task_scales': theta.tolist(),
        'previous_actual_eta_scales': previous_theta, 'accepted_actual_risk_transition': transition,
        'previous_risk_diagnostic': previous, 'current_risk_diagnostic': current, 'current_risk_method': method,
        'previous_response_diagnostic': old_response, 'previous_response_posterior_summary': old_post_summary,
        'current_response_diagnostic': item['response_stationarity_at_final_risk'],
        'current_joint_stationarity_diagnostic_passed': current_passed,
        'two_consecutive_actual_law_joint_stationarity_confirmed': confirmed,
        'new_joint_optimization_iterations': 0, 'parameters_changed': False,
        'nearest_bank_score_prefix_maximum_error': prefix_error, 'current_risk_sufficient_statistics': seconds_path,
        'completed_candidate_MCMC_sweeps': sweeps, 'cpu_seconds': time.process_time() - started,
        'scope': 'Retrospective joint source stationarity on a qualified accepted risk transition at fixed response law. Both actual laws must pass risk score precision and response posterior/functionals and covariance diagnostics. Same-law prefix extensions and unaccepted proposals are not a second law. No new optimizer iteration or target benefit is inferred.'}
    save(path, result)
    print('joint source audit', suffix, 'current', current_passed, 'consecutive actual laws', confirmed,
        'risk method', method, 'CPU', round(result['cpu_seconds'], 2), flush=True)
    return result


def run(plan=PLAN):
    spec = read(plan); out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    source = read(spec['source_plan']); laws = read(source['source_laws'])['rows']; selection = read(source['selection_plan'])
    cases = read(spec['source_learning'])['evidence_rows']; rows = []; started = time.process_time()
    for item in spec['cases']:
        if time.process_time() - started > spec['maximum_batch_single_thread_cpu_seconds']:
            break
        result = case_run(item, spec, cases, laws, selection); rows.append(result)
        phase = {'execution_plan': plan, 'rows': rows, 'registered_cases': len(spec['cases']),
            'all_registered_cases_complete': len(rows) == len(spec['cases']),
            'current_joint_stationarity_cases_passed': sum(r['current_joint_stationarity_diagnostic_passed'] for r in rows),
            'consecutive_actual_law_joint_stationarity_cases_confirmed': sum(
                r['two_consecutive_actual_law_joint_stationarity_confirmed'] for r in rows),
            'joint_source_learning_converged': len(rows) == 32 and all(
                r['two_consecutive_actual_law_joint_stationarity_confirmed'] for r in rows),
            'new_joint_optimization_iterations': 0, 'recorded_CPU_seconds': sum(r['cpu_seconds'] for r in rows),
            'completed_candidate_MCMC_sweeps': sum(r['completed_candidate_MCMC_sweeps'] for r in rows),
            'original_source_cases_resampled': 0, 'new_source_labels': 0, 'new_target_simulator_calls': 0,
            'new_verifier_calls': 0, 'server_jobs': 0, 'source_registry_replaced': False,
            'independent_confirmation': False}
        save(out / 'results.json', phase)
    return phase


if __name__ == '__main__':
    run()
