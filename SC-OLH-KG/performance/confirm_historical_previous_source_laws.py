"""Recheck four actual predecessors using already registered source banks."""

import time

import numpy as np

from core.gpr import posterior_mixture_weights
from performance.complete_historical_source_heads import ROOT, IDENTITY, read, save
from performance.confirm_historical_joint_source_stationarity import (
    KEY, fitted_diagnostic, joint_confirmation,
)
from performance.continue_historical_raw_source_learning import load_mixture
from performance.recondition_historical_changed_candidates import candidate_inputs
from performance.prepare_legacy_f_variance_reference import VarianceReferencePosterior, variance_reference
from performance.check_legacy_f_collapsed_source_raw_update import CollapsedRawSourcePosterior
from performance.extend_historical_source_risk_score_bank import draw_risk_states
from performance.reuse_historical_source_risk_states import risk_statistics
from performance.learn_historical_raw_source_amplitudes import covariance_builder
from performance.check_legacy_f_risk_amplitude_importance import log_ratios
from performance.certify_historical_reused_risk_candidate import response_stationarity
from performance.resolve_historical_source_variance import original_posterior_summary

PLAN = 'performance/manifests/historical_previous_source_laws_v1.json'


def same_case(row, item, fields=KEY):
    return all(row[k] == item[k] for k in fields)


def round_transition(before, after, response_update, risk_update, fixed_response, current_risk):
    """Recognize the actual round6->7 transition, preserving its original gate."""
    assert before['f_common_task_scales'] == after['f_common_task_scales'] == fixed_response
    assert response_update['old_common_task_covariance_scales'] == fixed_response
    assert response_update['common_task_covariance_scales'] == fixed_response
    assert risk_update['old_common_task_covariance_scales'] == before['eta_common_task_scales']
    assert risk_update['common_task_covariance_scales'] == after['eta_common_task_scales'] == current_risk
    return bool(before['eta_common_task_scales'] != current_risk
        and response_update['source_update_accepted'] and risk_update['source_update_accepted']
        and risk_update['source_Mstep_gate_passed'] and after['source_Mstep_pair_passed']
        and after['evidence_ratio_precision_passed']
        and min(risk_update['full_and_split_Q_improvements']) > 0.)


def predecessor(item, audit, spec):
    if audit['previous_actual_eta_scales'] is not None:
        proof = audit['accepted_actual_risk_transition']
        driver = read(proof['risk_results'])
        update = next(a for a in driver['attempts'] if a['attempt'] == proof['attempt'])
        assert update['current_theta'] == audit['previous_actual_eta_scales']
        assert update['proposed_theta'] == item['final_risk_scales']
        accepted = bool(update['update_accepted'] and min(update['full_and_split_Q_improvements']) > 0.
            and update['evidence']['precision_gate_passed']
            and update['evidence']['eligible_improvement_excludes_zero'])
        return audit['previous_actual_eta_scales'], accepted, proof
    before = next(r for r in read(spec['previous_learning_round'])['evidence_rows'] if same_case(r, item))
    after_round = read(spec['current_learning_round'])
    after = next(r for r in after_round['evidence_rows'] if same_case(r, item))
    response = next(r for r in after_round['rows'] if same_case(r, item))
    risk = next(r for r in after_round['rows'] if same_case(r, item, IDENTITY)
        and r['head'] == 'eta_' + item['head'])
    accepted = round_transition(before, after, response, risk,
        item['fixed_response_scales'], item['final_risk_scales'])
    proof = {'before_results': spec['previous_learning_round'], 'after_results': spec['current_learning_round'],
        'full_and_split_Q_improvements': risk['full_and_split_Q_improvements'],
        'historical_log_source_evidence_ratio': after['log_source_evidence_ratio_new_vs_previous'],
        'acceptance_contract': 'Actual historical full/split M-step and evidence-precision gates;no new update adopted.'}
    return before['eta_common_task_scales'], accepted, proof


def case_run(item, spec, cases, laws, selection):
    suffix = '_'.join(item[k] for k in KEY)
    out = ROOT / spec['output'] / 'cases' / suffix
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'results.json').exists():
        return read(str((out / 'results.json').relative_to(ROOT)))
    started = time.process_time()
    audit = read(item['joint_stationarity_results'])
    previous, accepted, proof = predecessor(item, audit, spec)
    assert accepted
    refinement = read(item['risk_stationarity_certificate'])
    driver = read(refinement['previous_risk_results'])
    original = read(driver['execution_plan'])
    bank = driver['sampling_batches'][driver['attempts'][-1]['bank'] - 1]
    case = next(r for r in cases if same_case(r, item))
    assert case['f_common_task_scales'] == driver['fixed_response_scales'] == item['fixed_response_scales']
    assert case['eta_common_task_scales'] == refinement['checked_current_scales'] == item['final_risk_scales']
    original_case = next(r for r in read(original['source_learning'])['evidence_rows'] if same_case(r, item))
    assert original_case['f_common_task_scales'] == case['f_common_task_scales']
    assert bank['anchor_theta'] == refinement['sampling_anchor_scales']
    assert bank['seed'] == refinement['same_seed'] == item['replay_seed']
    _, relative, _, _, _ = candidate_inputs(case, laws, selection)
    risk = next(r for r in laws if same_case(r, item, IDENTITY) and r['head'] == 'eta_' + item['head'])
    law = next(r for r in laws if same_case(r, item))
    builder = covariance_builder(risk)
    mean = np.asarray(risk['source_mean'])
    anchor = np.asarray(bank['anchor_theta'])
    current = np.asarray(item['final_risk_scales'])
    theta = np.asarray(previous)
    rm, rc = variance_reference(mean, builder(anchor)[0], relative.s2)
    model = VarianceReferencePosterior(relative.mean, relative.covariance, relative.labels, relative.s2)
    sampler = {**original['sampler'], 'seed': item['replay_seed'],
        'retained_sweeps_per_chain': item['replay_retained_sweeps_per_chain']}
    assert sampler['retained_sweeps_per_chain'] == refinement['extended_retained_sweeps_per_chain']
    states = draw_risk_states(model, rm, rc, case, sampler, started, spec['maximum_case_single_thread_cpu_seconds'])
    errors = states - mean
    def statistics(at):
        lr = log_ratios(errors.reshape(-1, 48), builder(anchor)[0], builder(at)[0])
        _, mass = posterior_mixture_weights(np.ones(lr.size) / lr.size, lr)
        return risk_statistics(errors, mass.reshape(states.shape[:2]), builder, at, spec)
    current_seconds, current_summaries, _ = statistics(current)
    prefix_error = float(np.max(abs(np.asarray([s['scaled_raw_score'] for s in current_summaries])
        - np.asarray([s['scaled_raw_score'] for s in refinement['current_score_summaries']]))))
    assert prefix_error < 1e-9
    seconds, summaries, is_precise = statistics(theta)
    previous_risk = fitted_diagnostic(seconds, summaries, is_precise, builder, theta, spec)
    old_response = audit['previous_response_diagnostic']
    old_summary = audit['previous_response_posterior_summary']
    if old_response is None:
        record = {'positive_mixture_component_file': item['candidate_component_file'],
            'positive_weights_results': item['candidate_component_file']}
        eta, weights = load_mixture(record, case)
        lr = log_ratios(eta.reshape(-1, 48) - mean, builder(current)[0], builder(theta)[0])
        _, weights = posterior_mixture_weights(weights.ravel(), lr)
        weights = weights.reshape(eta.shape[:2])
        response_model = CollapsedRawSourcePosterior(relative.mean, relative.covariance, relative.labels, relative.s2)
        moments = [response_model.moments(response_model.evaluate(p)) for p in eta.reshape(-1, 48)]
        ds = np.stack([v[0] for v in moments]).reshape(eta.shape)
        cs = np.stack([v[1] for v in moments]).reshape(*eta.shape[:2], 48, 48)
        old_summary = original_posterior_summary(response_model, eta, ds, cs, weights, spec)
        old_response = response_stationarity(response_model, eta, ds, cs, weights, law,
            case['f_common_task_scales'], spec)
    confirmed = joint_confirmation(previous, current.tolist(), previous_risk, audit['current_risk_diagnostic'],
        old_response, audit['current_response_diagnostic'], old_summary['monte_carlo_gate_passed'], accepted)
    statistics_path = str((out / 'risk_sufficient_statistics.json').relative_to(ROOT))
    save(ROOT / statistics_path, {'previous_actual_risk_scales': previous, 'current_actual_risk_scales': current.tolist(),
        'fixed_response_scales': case['f_common_task_scales'], 'risk_prior_mean': mean.tolist(),
        'previous_full_and_split_second_moments_about_prior_mean': [s.tolist() for s in seconds],
        'current_full_and_split_second_moments_about_prior_mean': [s.tolist() for s in current_seconds],
        'sampling_anchor_scales': anchor.tolist(), 'sampler': sampler})
    result = {**{k: item[k] for k in KEY}, 'execution_plan': PLAN,
        'previous_actual_eta_scales': previous, 'current_eta_common_task_scales': current.tolist(),
        'current_f_common_task_scales': case['f_common_task_scales'], 'accepted_actual_risk_transition': proof,
        'previous_risk_diagnostic': previous_risk, 'original_previous_risk_diagnostic': audit['previous_risk_diagnostic'],
        'current_risk_diagnostic': audit['current_risk_diagnostic'], 'previous_response_diagnostic': old_response,
        'previous_response_posterior_summary': old_summary, 'current_response_diagnostic': audit['current_response_diagnostic'],
        'current_joint_stationarity_diagnostic_passed': audit['current_joint_stationarity_diagnostic_passed'],
        'two_consecutive_actual_law_joint_stationarity_confirmed': confirmed,
        'registered_current_bank_score_maximum_error': prefix_error, 'risk_sufficient_statistics': statistics_path,
        'completed_candidate_MCMC_sweeps': sampler['chains'] * (sampler['warmup_sweeps_per_chain']
            + sampler['retained_sweeps_per_chain']), 'additional_sweeps_beyond_registered_bank': 0,
        'cpu_seconds': time.process_time() - started, 'parameters_changed': False, 'new_joint_optimization_iterations': 0,
        'independent_confirmation': False}
    save(out / 'results.json', result)
    print('actual predecessor', suffix, 'joint', confirmed, 'risk covariance change',
        round(previous_risk['relative_diagnostic_covariance_change'], 6), 'CPU', round(result['cpu_seconds'], 2), flush=True)
    return result


def run(plan=PLAN):
    spec = read(plan)
    base = read(spec['current_phase'])
    source = read(spec['source_plan'])
    laws = read(source['source_laws'])['rows']
    selection = read(source['selection_plan'])
    cases = read(base['qualified_source_learning_results'])['evidence_rows']
    out = ROOT / spec['output']
    out.mkdir(parents=True, exist_ok=True)
    results = []
    started = time.process_time()
    for item in spec['cases']:
        if time.process_time() - started > spec['maximum_batch_single_thread_cpu_seconds']:
            break
        results.append(case_run(item, spec, cases, laws, selection))
        save(out / 'results.json', {'execution_plan': plan, 'rows': results,
            'all_registered_cases_complete': len(results) == len(spec['cases']),
            'additional_actual_law_joint_confirmations': sum(r['two_consecutive_actual_law_joint_stationarity_confirmed'] for r in results),
            'recorded_CPU_seconds': sum(r['cpu_seconds'] for r in results),
            'completed_candidate_MCMC_sweeps': sum(r['completed_candidate_MCMC_sweeps'] for r in results)})
    if len(results) == len(spec['cases']):
        assemble_phase(spec, base, results)
    return results


def assemble_phase(spec, base, results):
    """Expose the same candidate parameters with the verified learning metadata."""
    checks = {tuple(r[k] for k in KEY): r for r in results}
    rows = []
    for original in base['rows']:
        row = dict(original)
        case_key = tuple(row[k] for k in KEY)
        if case_key in checks:
            check = checks[case_key]
            assert row['final_risk_scales'] == check['current_eta_common_task_scales']
            assert row['fixed_response_scales'] == check['current_f_common_task_scales']
            row['two_consecutive_actual_law_joint_stationarity_confirmed'] = check['two_consecutive_actual_law_joint_stationarity_confirmed']
            row['joint_stationarity_results'] = spec['output'] + '/cases/' + '_'.join(case_key) + '/results.json'
            row['conditional_risk_stationarity_passed'] = row['two_consecutive_actual_law_joint_stationarity_confirmed']
        rows.append(row)
    remaining = [{k: r[k] for k in (*KEY, 'joint_stationarity_results')}
        for r in rows if not r['two_consecutive_actual_law_joint_stationarity_confirmed']]
    converged = not remaining
    identities = {tuple(r[k] for k in IDENTITY) for r in rows}
    complete = sum(all(r['two_consecutive_actual_law_joint_stationarity_confirmed']
        for r in rows if tuple(r[k] for k in IDENTITY) == identity) for identity in identities)
    learning = read(base['qualified_source_learning_results'])
    precision = read(base['qualified_source_precision_results'])
    learning = {**learning, 'execution_plan': PLAN, 'predecessor_source_candidates': base['qualified_source_learning_results'],
        'source_learning_converged': converged, 'joint_stationarity_iterations_counted': 2 if converged else 0,
        'new_joint_optimization_iterations': 0, 'scope': 'Retrospectively certified consecutive distinct actual-law joint stationarity;candidate parameters and posterior mixtures unchanged.'}
    for case, certificate, row in zip(learning['evidence_rows'], precision['rows'], rows):
        assert same_case(case, row) and same_case(certificate, row)
        assert case['f_common_task_scales'] == row['fixed_response_scales']
        assert case['eta_common_task_scales'] == row['final_risk_scales']
        eta, _ = load_mixture(certificate, case)
        assert eta.shape[0] * eta.shape[1] == row['mixture_components']
        case.update(joint_stationarity_certificate_results=row['joint_stationarity_results'],
            conditional_risk_stationarity_passed=row['conditional_risk_stationarity_passed'],
            two_consecutive_actual_law_joint_stationarity_confirmed=row['two_consecutive_actual_law_joint_stationarity_confirmed'])
        for head in learning['rows']:
            if same_case(head, case, IDENTITY) and head['head'] in (case['head'], 'eta_' + case['head']):
                head.update(consecutive_source_stationarity_iterations=2 if row['two_consecutive_actual_law_joint_stationarity_confirmed'] else 1,
                    joint_stationarity_iteration_count_reset=False, joint_stationarity_certificate_results=row['joint_stationarity_results'])
    precision = {**precision, 'execution_plan': PLAN, 'source_learning_converged': converged}
    out = ROOT / spec['output']
    save(out / 'source_candidates.json', learning)
    save(out / 'source_precision.json', precision)
    phase = {**base, 'rows': rows, 'predecessor_phase_results': spec['current_phase'], 'execution_plan': PLAN,
        'consecutive_actual_law_joint_stationarity_cases_confirmed': len(rows) - len(remaining),
        'complete_joint_stationary_source_identities': complete, 'remaining_joint_stationarity_cases': remaining,
        'joint_source_learning_converged': converged, 'new_accepted_fixed_risk_updates': 0,
        'new_joint_optimization_iterations': 0, 'parameters_changed': False,
        'qualified_source_learning_results': spec['output'] + '/source_candidates.json',
        'qualified_source_precision_results': spec['output'] + '/source_precision.json',
        'recorded_CPU_seconds': sum(r['cpu_seconds'] for r in results),
        'completed_candidate_MCMC_sweeps': sum(r['completed_candidate_MCMC_sweeps'] for r in results),
        'recorded_CPU_is_lower_bound': False, 'execution_incident': None,
        'compact_replayed_risk_statistics_files': len(results), 'additional_sweeps_beyond_registered_banks': 0,
        'unit_test_results': spec['output'] + '/unit_tests.json',
        'limitations': ['Four predecessor checks reuse overlapping historical risk banks;they are numerical trajectory certification,not independent replications.',
            'Registered local score/covariance tolerances establish numerical stationarity,not global optimality or physical-factor recovery.',
            'Cross-identity absolute evidence and independent matched-control target decisions remain open.'],
        'next_work': 'Complete cross-identity source evidence,then compare the actual learned priors and matched controls on cached target decisions.'}
    save(out / 'phase_results.json', phase)
    print('Joint source confirmation', len(rows) - len(remaining), '/32;complete identities', complete, '/16', flush=True)
    return phase


if __name__ == '__main__':
    run()
