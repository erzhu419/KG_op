"""Extend the original bank once and check its fixed rejected risk proposal."""

import time

import numpy as np
from scipy.optimize import minimize

from core.gpr import posterior_mixture_weights
from performance.complete_historical_source_heads import ROOT, IDENTITY, read, save
from performance.recondition_historical_changed_candidates import candidate_inputs
from performance.prepare_legacy_f_variance_reference import VarianceReferencePosterior, variance_reference
from performance.extend_historical_source_risk_score_bank import draw_risk_states
from performance.reuse_historical_source_risk_states import risk_statistics
from performance.learn_historical_raw_source_amplitudes import covariance_builder
from performance.check_legacy_f_risk_amplitude_importance import log_ratios
from performance.check_legacy_f_raw_covariance_mstep import expected_objective
from performance.diagnose_legacy_f_raw_source_covariance_score import projected
from performance.probe_historical_source_risk_plateau import ratio_summary

PLAN = 'performance/manifests/historical_stopped_risk_precision_v1.json'
KEY = (*IDENTITY, 'head')


def fixed_update(errors, weights, builder, theta, proposed, spec):
    seconds, summaries, precise = risk_statistics(errors, weights, builder, theta, spec)
    q = [expected_objective(theta, builder, s)[0] - expected_objective(proposed, builder, s)[0]
        for s in seconds]
    lr = log_ratios(errors.reshape(-1, errors.shape[-1]), builder(theta)[0], builder(proposed)[0]).reshape(weights.shape)
    evidence = ratio_summary(lr, weights, spec['evidence_precision'])
    _, nw = posterior_mixture_weights(weights.ravel(), lr.ravel())
    _, next_summary, next_precise = risk_statistics(errors, nw.reshape(weights.shape), builder, proposed, spec)
    return seconds, {'full_and_split_Q_improvements': q, 'current_score_summaries': summaries,
        'current_score_precision_passed': precise, 'proposed_score_summaries': next_summary,
        'proposed_score_precision_passed': next_precise, 'source_evidence': evidence,
        'fixed_proposal_qualified': bool(precise and next_precise and min(q) > 0.
            and evidence['precision_gate_passed'] and evidence['eligible_improvement_excludes_zero'])}


def case_run(item, spec):
    out = ROOT / spec['output'] / 'cases' / '_'.join(item[k] for k in KEY)
    out.mkdir(parents=True, exist_ok=True); path = out / 'results.json'
    if path.exists():
        return read(str(path.relative_to(ROOT)))
    started = time.process_time(); old = read(item['risk_results']); plan = read(old['execution_plan'])
    last = old['attempts'][-1]; bank = old['sampling_batches'][last['bank'] - 1]
    parent = read(plan['source_plan']); laws = read(parent['source_laws'])['rows']
    selection = read(parent['selection_plan'])
    case = next(r for r in read(plan['source_learning'])['evidence_rows'] if all(r[k] == item[k] for k in KEY))
    _, relative, _, _, _ = candidate_inputs(case, laws, selection)
    risk = next(r for r in laws if r['head'] == ('eta_g' if item['head'] == 'g' else 'eta_f')
        and all(r[k] == item[k] for k in IDENTITY))
    builder = covariance_builder(risk); mean = np.asarray(risk['source_mean'])
    anchor = np.asarray(bank['anchor_theta']); theta = np.asarray(last['current_theta'])
    proposed = np.asarray(last['proposed_theta'])
    assert last['current_theta'] == old['final_risk_scales'] and not last['update_accepted']
    rm, rc = variance_reference(mean, builder(anchor)[0], relative.s2)
    model = VarianceReferencePosterior(relative.mean, relative.covariance, relative.labels, relative.s2)
    original_count = plan['sampler']['retained_sweeps_per_chain']
    sampler = {**plan['sampler'], 'seed': bank['seed'],
        'retained_sweeps_per_chain': spec['retained_sweeps_multiplier'] * original_count}
    states = draw_risk_states(model, rm, rc, item, sampler, started, spec['maximum_single_thread_cpu_seconds_per_case'])
    errors = states - mean
    lr = log_ratios(errors.reshape(-1, 48), builder(anchor)[0], builder(theta)[0]).reshape(states.shape[:2])
    _, weights = posterior_mixture_weights(np.ones(lr.size) / lr.size, lr.ravel())
    weights = weights.reshape(lr.shape)
    _, prefix, _ = risk_statistics(errors[:, :original_count], weights[:, :original_count], builder, theta, spec)
    prefix_error = float(np.max(abs(np.asarray(prefix[0]['scaled_raw_score'])
        - last['old_score_summaries'][0]['scaled_raw_score'])))
    assert prefix_error < 1e-9
    seconds, check = fixed_update(errors, weights, builder, theta, proposed, spec)
    scale = np.where(theta > 0., theta, 1.)
    def objective(z):
        value, gradient = expected_objective(z * scale, builder, seconds[0])
        return value, gradient * scale
    fit = minimize(objective, theta / scale, method='L-BFGS-B', jac=True, bounds=[(0., None)] * 2,
        options=spec['optimizer'])
    diagnostic_theta = fit.x * scale
    change = float(np.linalg.norm(builder(diagnostic_theta)[0] - builder(theta)[0]) / np.linalg.norm(builder(theta)[0]))
    pg = projected(theta, expected_objective(theta, builder, seconds[0])[1], np.full(2, np.inf)) * scale
    stationary = bool(check['current_score_precision_passed'] and fit.success
        and change <= spec['convergence']['maximum_relative_covariance_change']
        and max(abs(pg)) <= spec['convergence']['maximum_scaled_projected_raw_score'])
    held = bool(last['stationarity_diagnostic_passed'] and stationary)
    accepted = bool(not held and check['fixed_proposal_qualified'])
    final = proposed if accepted else theta
    result = {**{k: item[k] for k in KEY}, 'previous_risk_results': item['risk_results'],
        'same_seed': bank['seed'], 'sampling_anchor_scales': anchor.tolist(),
        'old_retained_sweeps_per_chain': original_count,
        'extended_retained_sweeps_per_chain': sampler['retained_sweeps_per_chain'],
        'prefix_current_score_maximum_error': prefix_error,
        'checked_current_scales': theta.tolist(), 'fixed_rejected_proposal_scales': proposed.tolist(), **check,
        'new_current_stationarity_diagnostic_passed': stationary,
        'new_diagnostic_covariance_change': change, 'scaled_projected_current_score': pg.tolist(),
        # Extending one overlapping prefix verifies precision at one law;
        # it does not create a second optimization iteration.
        'conditional_risk_stationarity_passed': False,
        'same_law_precision_confirmation_passed': held, 'stationary_current_law_held': held,
        'accepted_fixed_risk_update': accepted, 'final_risk_scales': final.tolist(),
        'fixed_response_scales': old['fixed_response_scales'],
        'held_law_source_evidence_ratio': 1. if held else None,
        'completed_candidate_MCMC_sweeps': sampler['chains'] * (sampler['warmup_sweeps_per_chain']
            + sampler['retained_sweeps_per_chain']),
        'additional_retained_sweeps_beyond_prefix': sampler['chains'] * original_count,
        'cpu_seconds': time.process_time() - started, 'source_registry_replaced': False,
        'joint_source_learning_converged': False, 'independent_confirmation': False}
    save(path, result)
    print('fixed risk extension', *[item[k] for k in KEY], 'stationary hold', held,
        'fixed update accepted', accepted, 'CPU', round(result['cpu_seconds'], 2), flush=True)
    return result


def run(plan=PLAN):
    spec = read(plan); out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    path = out / 'results.json'
    if path.exists() and read(str(path.relative_to(ROOT))).get('all_registered_cases_complete'):
        return read(str(path.relative_to(ROOT)))
    rows = []
    for item in spec['cases']:
        result = case_run(item, spec); rows.append(result)
        phase = {'execution_plan': plan, 'rows': rows, 'all_registered_cases_complete': len(rows) == len(spec['cases']),
            'registered_cases': len(spec['cases']), 'additional_stationary_risk_laws_held':
                sum(r['stationary_current_law_held'] for r in rows),
            'fixed_rejected_proposals_qualified': sum(r['fixed_proposal_qualified'] for r in rows),
            'fixed_risk_updates_accepted': sum(r['accepted_fixed_risk_update'] for r in rows),
            'conditional_risk_stationarity_cases_passed': spec['previous_conditional_risk_stationarity_cases_passed']
                + sum(r['conditional_risk_stationarity_passed'] for r in rows),
            'recorded_CPU_seconds': sum(r['cpu_seconds'] for r in rows),
            'completed_candidate_MCMC_sweeps': sum(r['completed_candidate_MCMC_sweeps'] for r in rows),
            'source_registry_replaced': False, 'joint_source_learning_converged': False,
            'new_source_labels': 0, 'new_target_simulator_calls': 0, 'new_verifier_calls': 0, 'server_jobs': 0,
            'independent_confirmation': False}
        save(path, phase)
    return phase


if __name__ == '__main__':
    run()
