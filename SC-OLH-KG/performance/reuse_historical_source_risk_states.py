"""Reuse full risk chains in memory for source-only fixed-response MCEM."""

import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from core.gpr import posterior_mixture_weights
from performance.complete_historical_source_heads import ROOT, IDENTITY, read, save
from performance.extend_historical_source_risk_score_bank import draw_risk_states
from performance.recondition_historical_changed_candidates import candidate_inputs
from performance.prepare_legacy_f_variance_reference import VarianceReferencePosterior, variance_reference
from performance.learn_historical_raw_source_amplitudes import covariance_builder
from performance.check_legacy_f_raw_covariance_mstep import expected_objective
from performance.check_legacy_f_risk_amplitude_importance import log_ratios
from performance.diagnose_historical_source_risk_score_precision import summarize
from performance.diagnose_legacy_f_raw_source_covariance_score import projected
from performance.probe_historical_source_risk_plateau import ratio_summary

PLAN = 'performance/manifests/historical_source_risk_state_reuse_v1.json'
KEY = (*IDENTITY, 'head')


def risk_statistics(errors, weights, builder, theta, spec):
    n = errors.shape[-1]; count = errors.shape[1]; flat = errors.reshape(-1, n)
    covariance, directions = builder(theta)[:2]; factor = cho_factor(covariance, lower=True)
    pe = cho_solve(factor, flat.T).T
    series = np.column_stack([.5 * (np.trace(cho_solve(factor, d)) - np.sum((pe @ d) * pe, axis=1)) / n
        for d in directions]).reshape(*errors.shape[:2], 2)
    seconds = []; summaries = []
    scale = np.where(theta > 0., theta, 1.)
    for part in (slice(None), slice(0, count // 2), slice(count // 2, None)):
        e = errors[:, part].reshape(-1, n); w = weights[:, part].ravel(); w = w / w.sum()
        seconds.append((e.T * w) @ e)
        mean, se, diag = summarize(series[:, part], weights[:, part])
        summaries.append({'scaled_raw_score': (mean * scale).tolist(), 'scaled_MCSE': (se * scale).tolist(),
            'scaled_95_MC_halfwidth': (1.96 * se * scale).tolist(), **diag})
    gradient = expected_objective(theta, builder, seconds[0])[1]
    np.testing.assert_allclose(gradient * scale, summaries[0]['scaled_raw_score'], atol=1e-9, rtol=1e-9)
    precise = all(max(s['split_rhat']) <= spec['score_precision']['maximum_split_Rhat']
        and min(s['autocorrelation_ess']) >= spec['score_precision']['minimum_functional_ESS'] for s in summaries)
    precise = precise and max(summaries[0]['scaled_95_MC_halfwidth']) <= spec['score_precision']['maximum_scaled_95_MC_halfwidth']
    return seconds, summaries, bool(precise)


def run(plan=PLAN):
    spec = read(plan); out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    path = out / 'results.json'
    if path.exists():
        return read(str(path.relative_to(ROOT)))
    started = time.process_time(); key = spec['case']
    parent = read(spec['source_plan']); laws = read(parent['source_laws'])['rows']; selection = read(parent['selection_plan'])
    case = next(r for r in read(spec['source_learning'])['evidence_rows'] if all(r[k] == key[k] for k in KEY))
    risk = next(r for r in laws if r['head'] == ('eta_g' if key['head'] == 'g' else 'eta_f')
        and all(r[k] == key[k] for k in IDENTITY))
    _, relative, _, _, _ = candidate_inputs(case, laws, selection)
    model = VarianceReferencePosterior(relative.mean, relative.covariance, relative.labels, relative.s2)
    builder = covariance_builder(risk); mean = np.asarray(risk['source_mean'])
    initial = np.asarray(case['eta_common_task_scales']); theta = initial.copy()
    batches = []; attempts = []; updates = []; streak = 0; states = None; refresh = 0
    status = 'registered update limit reached'; final_summaries = []
    try:
        while len(updates) < spec['maximum_updates']:
            if time.process_time() - started >= spec['maximum_single_thread_cpu_seconds']:
                raise TimeoutError('registered memory-reuse CPU cap')
            if states is None:
                anchor = theta.copy(); covariance = builder(anchor)[0]
                rm, rc = variance_reference(mean, covariance, relative.s2)
                seed = spec['sampler']['seed'] if not batches else spec['refresh_seeds'][refresh - 1]
                sampler = {**spec['sampler'], 'seed': seed}; one = time.process_time()
                states = draw_risk_states(model, rm, rc, key, sampler, started, spec['maximum_single_thread_cpu_seconds'])
                errors = states - mean; weights = np.ones(states.shape[:2]) / np.prod(states.shape[:2])
                batches.append({'anchor_theta': anchor.tolist(), 'seed': seed, 'completed_sweeps': sampler['chains'] * (
                    sampler['warmup_sweeps_per_chain'] + sampler['retained_sweeps_per_chain']),
                    'sampler_CPU_seconds': time.process_time() - one})
                print('risk memory bank', len(batches), 'anchor', anchor.tolist(), flush=True)
            seconds, summaries, precise = risk_statistics(errors, weights, builder, theta, spec)
            final_summaries = summaries
            if not precise:
                status = 'current-law score precision unresolved'; break
            scale = np.where(theta > 0., theta, 1.)
            def objective(z):
                value, gradient = expected_objective(z * scale, builder, seconds[0])
                return value, gradient * scale
            fit = minimize(objective, theta / scale, method='L-BFGS-B', jac=True, bounds=[(0., None)] * 2,
                options={'maxiter':spec['optimizer']['max_iterations'], 'ftol':spec['optimizer']['ftol'], 'gtol':spec['optimizer']['gtol']})
            proposed = fit.x * scale
            improvements = [expected_objective(theta, builder, s)[0] - expected_objective(proposed, builder, s)[0] for s in seconds]
            change = float(np.linalg.norm(builder(proposed)[0] - builder(theta)[0]) / np.linalg.norm(builder(theta)[0]))
            gradient = expected_objective(theta, builder, seconds[0])[1]
            stationary = bool(change <= spec['convergence']['maximum_relative_covariance_change']
                and max(abs(projected(theta, gradient, np.full(2, np.inf)) * scale))
                    <= spec['convergence']['maximum_scaled_projected_raw_score'])
            row = {'attempt': len(attempts) + 1, 'bank': len(batches), 'current_theta': theta.tolist(),
                'proposed_theta': proposed.tolist(), 'full_and_split_Q_improvements': improvements,
                'optimizer_converged': bool(fit.success), 'relative_covariance_change': change,
                'stationarity_diagnostic_passed': stationary, 'old_score_summaries': summaries,
                'update_accepted': False}
            if not fit.success or min(improvements) <= 0.:
                attempts.append(row); status = 'full/split Q requires more precision'; break
            lr = log_ratios(errors.reshape(-1, 48), builder(theta)[0], builder(proposed)[0]).reshape(weights.shape)
            evidence = ratio_summary(lr, weights, spec['evidence_precision'])
            _, nw = posterior_mixture_weights(weights.ravel(), lr.ravel()); nw = nw.reshape(weights.shape)
            _, next_summary, next_precise = risk_statistics(errors, nw, builder, proposed, spec)
            row.update(evidence=evidence, new_score_summaries=next_summary, new_score_precision_passed=next_precise)
            if not evidence['precision_gate_passed'] or not next_precise:
                attempts.append(row)
                if refresh < len(spec['refresh_seeds']) and not np.array_equal(anchor, theta):
                    refresh += 1; states = None
                    print('risk memory precision refresh', refresh, 'at accepted theta', theta.tolist(), flush=True)
                    continue
                status = 'reweighted-law precision requires additional sampling'; break
            if not evidence['eligible_improvement_excludes_zero']:
                attempts.append(row); status = 'source evidence improvement unresolved'; break
            streak = streak + 1 if stationary else 0
            row.update(update_accepted=True, consecutive_stationarity_iterations=streak)
            attempts.append(row); updates.append(row); theta = proposed; weights = nw; final_summaries = next_summary
            print('risk reused update', len(updates), 'theta', theta.tolist(), 'ESS', round(evidence['importance_ESS']),
                'covariance change', round(change, 4), 'stationary', streak, flush=True)
            if streak >= spec['convergence']['required_consecutive_iterations']:
                status = 'fixed-response risk stationarity reached'; break
    except TimeoutError as exc:
        status = str(exc)
    result = {'execution_plan': plan, **key, 'status': status, 'initial_risk_scales': initial.tolist(),
        'final_risk_scales': theta.tolist(), 'fixed_response_scales': case['f_common_task_scales'],
        'accepted_risk_updates': len(updates), 'attempts': attempts, 'sampling_batches': batches,
        'final_score_summaries': final_summaries, 'consecutive_risk_stationarity_iterations': streak,
        'conditional_risk_stationarity_passed': streak >= spec['convergence']['required_consecutive_iterations'],
        'completed_candidate_MCMC_sweeps': sum(b['completed_sweeps'] for b in batches),
        'risk_states_written_to_disk': 0, 'KG_components_changed': 0, 'source_registry_replaced': False,
        'original_source_cases_resampled': 0, 'new_source_labels': 0, 'new_target_simulator_calls': 0,
        'new_verifier_calls': 0, 'server_jobs': 0, 'cpu_seconds': time.process_time() - started,
        'no_model_adopted': True, 'independent_confirmation': False,
        'scope': 'One fixed-response source-risk learning pilot. Every accepted update has positive full/split Q,precise current/new risk scores and source evidence improvement outside95percent numerical error. Final response posterior precision,retention and independent decisions are not certified here.'}
    save(path, result); print(status, 'accepted updates', len(updates), 'CPU', round(result['cpu_seconds'], 2), flush=True)
    return result


if __name__ == '__main__':
    run()
