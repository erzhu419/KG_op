"""Certify one fixed risk proposal and its actual source law before adoption."""

import time

import numpy as np

from core.gpr import posterior_mixture_weights
from performance.complete_historical_source_heads import ROOT, read, save
from performance.confirm_historical_joint_source_stationarity import KEY, fitted_diagnostic, joint_confirmation
from performance.continue_historical_raw_source_learning import load_mixture
from performance.recondition_historical_changed_candidates import candidate_inputs
from performance.prepare_legacy_f_variance_reference import VarianceReferencePosterior, variance_reference
from performance.check_legacy_f_collapsed_source_raw_update import CollapsedRawSourcePosterior
from performance.extend_historical_source_risk_score_bank import draw_risk_states
from performance.learn_historical_raw_source_amplitudes import covariance_builder
from performance.check_legacy_f_risk_amplitude_importance import log_ratios
from performance.refine_historical_stopped_risk_updates import fixed_update
from performance.reuse_historical_source_risk_states import risk_statistics
from performance.certify_historical_reused_risk_candidate import response_stationarity
from performance.resolve_historical_source_variance import original_posterior_summary
from performance.bridge_historical_final_risk_evidence import compose_intervals

PLAN = 'performance/manifests/historical_pending_joint_update_v1.json'


def same(a, b):
    return all(a[k] == b[k] for k in KEY)


def run(plan=PLAN):
    spec = read(plan); out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    path = out / 'results.json'
    if path.exists():
        return read(str(path.relative_to(ROOT)))
    started = time.process_time(); proposal = read(spec['qualified_fixed_proposal'])
    assert not proposal['accepted_fixed_risk_update']
    assert min(proposal['full_and_split_Q_improvements']) > 0.
    base = read(spec['source_closure']); item = next(r for r in base['rows'] if same(r, proposal))
    assert item['posterior_functional_and_retention_certified']
    learning = read(base['qualified_source_learning_results']); case = next(r for r in learning['evidence_rows'] if same(r, item))
    source = read(spec['source_plan']); laws = read(source['source_laws'])['rows']; selection = read(source['selection_plan'])
    risk = next(r for r in laws if r['head'] == 'eta_' + item['head'] and all(r[k] == item[k] for k in KEY[:-1]))
    law = next(r for r in laws if same(r, item)); builder = covariance_builder(risk); mean = np.asarray(risk['source_mean'])
    driver = read(proposal['previous_risk_results']); original = read(driver['execution_plan'])
    last = driver['attempts'][-1]; bank = driver['sampling_batches'][last['bank'] - 1]
    theta = np.asarray(proposal['checked_current_scales']); proposed = np.asarray(proposal['fixed_rejected_proposal_scales'])
    assert case['eta_common_task_scales'] == theta.tolist() == proposal['final_risk_scales']
    assert case['f_common_task_scales'] == proposal['fixed_response_scales'] == driver['fixed_response_scales']
    _, relative, _, _, _ = candidate_inputs(case, laws, selection)
    model = VarianceReferencePosterior(relative.mean, relative.covariance, relative.labels, relative.s2)
    anchor = np.asarray(bank['anchor_theta']); rm, rc = variance_reference(mean, builder(anchor)[0], relative.s2)
    sampler = {**original['sampler'], 'seed': bank['seed'],
        'retained_sweeps_per_chain': spec.get('retained_sweeps_per_chain', proposal['extended_retained_sweeps_per_chain'])}
    states = draw_risk_states(model, rm, rc, case, sampler, started, spec['maximum_single_thread_cpu_seconds'])
    errors = states - mean
    lr = log_ratios(errors.reshape(-1, 48), builder(anchor)[0], builder(theta)[0]).reshape(states.shape[:2])
    _, weights = posterior_mixture_weights(np.ones(lr.size) / lr.size, lr.ravel()); weights = weights.reshape(lr.shape)
    seconds, check = fixed_update(errors, weights, builder, theta, proposed, spec)
    prefix_count = proposal['extended_retained_sweeps_per_chain']
    _, prefix_summaries, _ = risk_statistics(errors[:, :prefix_count], weights[:, :prefix_count], builder, theta, spec)
    replay_error = float(np.max(abs(np.asarray(prefix_summaries[0]['scaled_raw_score'])
        - proposal['current_score_summaries'][0]['scaled_raw_score'])))
    assert replay_error < 1e-9
    old_risk = fitted_diagnostic(seconds, check['current_score_summaries'], check['current_score_precision_passed'], builder, theta, spec)
    lr = log_ratios(errors.reshape(-1, 48), builder(theta)[0], builder(proposed)[0]).reshape(weights.shape)
    _, nw = posterior_mixture_weights(weights.ravel(), lr.ravel())
    new_seconds, summaries, is_precise = risk_statistics(errors, nw.reshape(weights.shape), builder, proposed, spec)
    new_risk = fitted_diagnostic(new_seconds, summaries, is_precise, builder, proposed, spec)
    record = {'positive_mixture_component_file': item['candidate_component_file'],
        'positive_weights_results': item['candidate_component_file']}
    eta, weights = load_mixture(record, case)
    lr = log_ratios(eta.reshape(-1, 48) - mean, builder(theta)[0], builder(proposed)[0])
    _, nw = posterior_mixture_weights(weights.ravel(), lr); nw = nw.reshape(eta.shape[:2])
    model = CollapsedRawSourcePosterior(relative.mean, relative.covariance, relative.labels, relative.s2)
    moments = [model.moments(model.evaluate(p)) for p in eta.reshape(-1, 48)]
    ds = np.stack([v[0] for v in moments]).reshape(eta.shape)
    cs = np.stack([v[1] for v in moments]).reshape(*eta.shape[:2], 48, 48)
    posterior = original_posterior_summary(model, eta, ds, cs, nw, spec)
    response = response_stationarity(model, eta, ds, cs, nw, law, case['f_common_task_scales'], spec)
    composed = compose_intervals([{'source_evidence': item['source_evidence']}, {'source_evidence': check['source_evidence']}])
    confirmed = joint_confirmation(theta.tolist(), proposed.tolist(), old_risk, new_risk,
        item['response_stationarity_at_final_risk'], response, posterior['monte_carlo_gate_passed'], check['fixed_proposal_qualified'])
    accepted = bool(confirmed and composed['precision_gate_passed'] and composed['eligible_improvement_excludes_zero'])
    component_path = str((out / ('candidate_' + '_'.join(case[k] for k in KEY) + '.json')).relative_to(ROOT))
    if accepted:
        save(ROOT / component_path, {'execution_plan': plan,
            'fixed_candidate': {**{k: case[k] for k in KEY}, 'f_common_task_scales': case['f_common_task_scales'],
                'eta_common_task_scales': proposed.tolist()}, 'posterior': posterior,
            'positive_eta_mixture': {'eta_states_by_chain': eta.tolist(), 'positive_component_weights': nw.ravel().tolist()},
            'previous_candidate_component_file': item['candidate_component_file']})
        case.update(eta_common_task_scales=proposed.tolist(), candidate_component_file=component_path,
            new_positive_eta_weights=nw.ravel().tolist(), conditional_risk_stationarity_passed=True,
            evidence_certificate_results=str(path.relative_to(ROOT)), posterior_certificate_results=str(path.relative_to(ROOT)),
            log_source_evidence_ratio_new_vs_previous=composed['log_source_evidence_ratio'])
        for head in learning['rows']:
            if head['head'] == 'eta_' + case['head'] and all(head[k] == case[k] for k in KEY[:-1]):
                head['common_task_covariance_scales'] = proposed.tolist()
                head['source_case_certificate'] = str(path.relative_to(ROOT))
        learning['predecessor_source_candidates'] = base['qualified_source_learning_results']
        learning['execution_plan'] = plan
        save(out / 'source_candidates.json', learning)
        precision = read(base['qualified_source_precision_results'])
        precision['execution_plan'] = plan
        certificate = next(r for r in precision['rows'] if same(r, case))
        certificate.update(positive_mixture_component_file=component_path, positive_weights_results=component_path,
            within_identity_evidence_results=str(path.relative_to(ROOT)),
            posterior_functional_and_retention_results=str(path.relative_to(ROOT)))
        save(out / 'source_precision.json', precision)
        # Verify the branch through the existing loader at its actual parameters.
        for c, r in zip(learning['evidence_rows'], precision['rows']):
            assert same(c, r); load_mixture(r, c)
    save(out / 'risk_sufficient_statistics.json', {'current_before_update_scales': theta.tolist(),
        'current_after_update_scales': proposed.tolist() if accepted else theta.tolist(),
        'fixed_response_scales': case['f_common_task_scales'], 'risk_prior_mean': mean.tolist(),
        'full_and_split_second_moments_before_update': [s.tolist() for s in seconds],
        'full_and_split_second_moments_at_proposed_law': [s.tolist() for s in new_seconds],
        'proposal_adopted': accepted})
    result = {'execution_plan': plan, **{k: case[k] for k in KEY},
        'previous_risk_results': spec['qualified_fixed_proposal'],
        'previous_fixed_proposal_results': spec['qualified_fixed_proposal'], 'fixed_proposal_qualified': check['fixed_proposal_qualified'],
        'accepted_fixed_risk_update': accepted, 'checked_current_scales': theta.tolist(),
        'final_risk_scales': proposed.tolist() if accepted else theta.tolist(), 'fixed_response_scales': case['f_common_task_scales'],
        'old_risk_diagnostic': old_risk, 'new_risk_diagnostic': new_risk, 'new_response_diagnostic': response,
        'final_risk_diagnostic_from_parent_bank': new_risk,
        'conditional_risk_stationarity_passed': accepted,
        'current_score_summaries': check['current_score_summaries'],
        'current_score_precision_passed': check['current_score_precision_passed'],
        'scaled_projected_current_score': old_risk['scaled_projected_raw_score'],
        'new_diagnostic_covariance_change': old_risk['relative_diagnostic_covariance_change'],
        'new_current_stationarity_diagnostic_passed': old_risk['stationarity_diagnostic_passed'],
        'response_posterior_summary': posterior, 'source_evidence': check['source_evidence'],
        'final_vs_selected_parent_evidence': composed, 'full_and_split_Q_improvements': check['full_and_split_Q_improvements'],
        'two_consecutive_actual_law_joint_stationarity_confirmed': accepted,
        'same_seed_extended_current_score_replay_maximum_error': replay_error,
        'candidate_component_file': component_path if accepted else None,
        'qualified_source_learning_results': str((out / 'source_candidates.json').relative_to(ROOT)) if accepted else None,
        'qualified_source_precision_results': str((out / 'source_precision.json').relative_to(ROOT)) if accepted else None,
        'completed_candidate_MCMC_sweeps': sampler['chains'] * (sampler['warmup_sweeps_per_chain'] + sampler['retained_sweeps_per_chain']),
        'retained_sweeps_per_chain': sampler['retained_sweeps_per_chain'],
        'additional_retained_sweeps_beyond_previous_prefix': sampler['chains'] * (sampler['retained_sweeps_per_chain'] - prefix_count),
        'cpu_seconds': time.process_time() - started, 'original_source_cases_resampled': 0,
        'new_source_labels': 0, 'new_target_simulator_calls': 0, 'new_verifier_calls': 0, 'server_jobs': 0,
        'joint_source_learning_converged': False, 'source_registry_replaced': False, 'independent_confirmation': False,
        'scope': 'One fixed unadopted risk proposal. Original seed,registered bank length and normalized source ratios. Adopt only after source Q/evidence/precision and both actual laws satisfy joint stationarity,final response posterior precision and retained-mixture loading,and total selected-parent evidence certification. No rejected or same-law proposal is counted as a second actual law.'}
    save(path, result)
    print('Fixed joint update', accepted, 'CPU', round(result['cpu_seconds'], 2), flush=True)
    return result


if __name__ == '__main__':
    run()
