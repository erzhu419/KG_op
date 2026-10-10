"""Final/parent source normalizer from the parent's actual full risk bank."""

import time

import numpy as np
from scipy.optimize import minimize

from performance.complete_historical_source_heads import ROOT, IDENTITY, read, save
from performance.recondition_historical_changed_candidates import candidate_inputs
from performance.prepare_legacy_f_variance_reference import VarianceReferencePosterior, variance_reference
from performance.extend_historical_source_risk_score_bank import draw_risk_states
from performance.reuse_historical_source_risk_states import risk_statistics
from performance.learn_historical_raw_source_amplitudes import covariance_builder
from performance.check_legacy_f_risk_amplitude_importance import log_ratios
from performance.probe_historical_source_risk_plateau import ratio_summary
from core.gpr import posterior_mixture_weights
from performance.check_legacy_f_raw_covariance_mstep import expected_objective
from performance.diagnose_legacy_f_raw_source_covariance_score import projected

KEY = (*IDENTITY, 'head')


def run(plan):
    spec = read(plan); out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    path = out / 'results.json'
    if path.exists():
        return read(str(path.relative_to(ROOT)))
    started = time.process_time(); candidate = read(spec['candidate']); first = read(spec['initial_risk_results'])
    initial_plan = read(first['execution_plan']); bank = first['sampling_batches'][0]
    source = read(spec['source_plan']); laws = read(source['source_laws'])['rows']
    selection = read(source['selection_plan'])
    current = next(r for r in read(spec['parent_learning'])['evidence_rows']
        if all(r[k] == candidate[k] for k in KEY))
    assert candidate['fixed_response_scales'] == current['f_common_task_scales'] == first['fixed_response_scales']
    assert bank['anchor_theta'] == current['eta_common_task_scales'] == first['initial_risk_scales']
    theta = np.asarray(candidate['final_risk_scales']); anchor = np.asarray(bank['anchor_theta'])
    count = initial_plan['sampler']['retained_sweeps_per_chain']
    unchanged = bool(np.array_equal(theta, anchor)); prefix_error = 0.; sweeps = 0; diagnostic = None
    if unchanged:
        evidence = ratio_summary(np.zeros((4, count)), np.ones((4, count)), spec['evidence_precision'])
    else:
        risk = next(r for r in laws if r['head'] == ('eta_g' if candidate['head'] == 'g' else 'eta_f')
            and all(r[k] == candidate[k] for k in IDENTITY))
        builder = covariance_builder(risk); mean = np.asarray(risk['source_mean'])
        _, relative, _, _, _ = candidate_inputs(current, laws, selection)
        model = VarianceReferencePosterior(relative.mean, relative.covariance, relative.labels, relative.s2)
        rm, rc = variance_reference(mean, builder(anchor)[0], relative.s2)
        sampler = {**initial_plan['sampler'], 'seed': bank['seed']}
        states = draw_risk_states(model, rm, rc, candidate, sampler, started,
            spec['maximum_single_thread_cpu_seconds'])
        errors = states - mean
        _, summary, _ = risk_statistics(errors, np.ones(states.shape[:2]), builder, anchor, spec)
        prefix_error = float(np.max(abs(np.asarray(summary[0]['scaled_raw_score'])
            - first['attempts'][0]['old_score_summaries'][0]['scaled_raw_score'])))
        assert prefix_error < 1e-9
        lr = log_ratios(errors.reshape(-1, 48), builder(anchor)[0], builder(theta)[0]).reshape(states.shape[:2])
        evidence = ratio_summary(lr, np.ones(states.shape[:2]), spec['evidence_precision'])
        _, weights = posterior_mixture_weights(np.ones(lr.size) / lr.size, lr.ravel())
        seconds, summaries, precise = risk_statistics(errors, weights.reshape(lr.shape), builder, theta, spec)
        scale = np.where(theta > 0., theta, 1.)
        def objective(z):
            value, gradient = expected_objective(z * scale, builder, seconds[0])
            return value, gradient * scale
        fit = minimize(objective, theta / scale, method='L-BFGS-B', jac=True, bounds=[(0., None)] * 2,
            options=spec['optimizer'])
        change = float(np.linalg.norm(builder(fit.x * scale)[0] - builder(theta)[0]) / np.linalg.norm(builder(theta)[0]))
        pg = projected(theta, expected_objective(theta, builder, seconds[0])[1], np.full(2, np.inf)) * scale
        diagnostic = {'score_summaries': summaries, 'score_precision_passed': precise,
            'relative_diagnostic_covariance_change': change, 'scaled_projected_raw_score': pg.tolist(),
            'stationarity_diagnostic_passed': bool(precise and fit.success
                and change <= spec['convergence']['maximum_relative_covariance_change']
                and max(abs(pg)) <= spec['convergence']['maximum_scaled_projected_raw_score']),
            'parameters_changed': False}
        sweeps = sampler['chains'] * (sampler['warmup_sweeps_per_chain'] + count)
    result = {'execution_plan': plan, **{k: candidate[k] for k in KEY},
        'initial_risk_scales': anchor.tolist(), 'final_risk_scales': theta.tolist(),
        'fixed_response_scales': candidate['fixed_response_scales'], 'unchanged_source_law': unchanged,
        'initial_bank_score_replay_maximum_error': prefix_error, 'source_evidence': evidence,
        'final_risk_diagnostic_from_parent_bank': diagnostic,
        'final_vs_parent_evidence_certified': bool(evidence['precision_gate_passed']
            and (unchanged or evidence['eligible_improvement_excludes_zero'])),
        'completed_candidate_MCMC_sweeps': sweeps, 'cpu_seconds': time.process_time() - started,
        'source_registry_replaced': False, 'joint_source_learning_converged': False,
        'new_source_labels': 0, 'new_target_simulator_calls': 0, 'new_verifier_calls': 0, 'server_jobs': 0,
        'independent_confirmation': False,
        'scope': 'One fixed final conditional-risk candidate versus its selected parent,at unchanged response covariance. Exact normalized Gaussian risk-prior ratio on the actual parent full bank,not a product of update intervals. Within-identity evidence only;no cross-identity normalizer or decision claim.'}
    save(path, result)
    print('final risk evidence', *[candidate[k] for k in KEY], result['final_vs_parent_evidence_certified'],
        'ESS', round(evidence['importance_ESS']), 'CPU', round(result['cpu_seconds'], 2), flush=True)
    return result
