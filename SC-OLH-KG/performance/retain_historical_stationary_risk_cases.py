"""Retain actual risk laws after two checks when a further proposal fails."""

import time

import numpy as np

from performance.complete_historical_source_heads import ROOT, read, save

PLAN = 'performance/manifests/historical_source_risk_stationary_retention_v1.json'


def retain_risk(result, spec):
    last = result['attempts'][-1]
    previous = result['attempts'][-2]
    assert previous['update_accepted']
    assert not last['update_accepted'] and last['stationarity_diagnostic_passed']
    assert previous['stationarity_diagnostic_passed']
    assert result['consecutive_risk_stationarity_iterations'] + 1 >= spec['required_consecutive_iterations']
    assert previous['proposed_theta'] == last['current_theta'] == result['final_risk_scales']
    assert previous['current_theta'] != last['current_theta']
    summaries = last['old_score_summaries']; limits = spec['score_precision']
    assert all(max(r['split_rhat']) <= limits['maximum_split_Rhat']
        and min(r['autocorrelation_ess']) >= limits['minimum_functional_ESS'] for r in summaries)
    assert max(summaries[0]['scaled_95_MC_halfwidth']) <= limits['maximum_scaled_95_MC_halfwidth']
    theta = np.asarray(last['current_theta']); gradient = np.asarray(summaries[0]['scaled_raw_score'])
    # Stored scores already include parameter scaling; at zero use the KKT projection.
    gradient = np.where((theta == 0.) & (gradient > 0.), 0., gradient)
    assert max(abs(gradient)) <= spec['maximum_scaled_projected_raw_score']
    assert last['relative_covariance_change'] <= spec['maximum_relative_covariance_change']
    return {'held_risk_scales': result['final_risk_scales'], 'conditional_risk_stationarity_passed': True,
        'consecutive_stationarity_checks': result['consecutive_risk_stationarity_iterations'] + 1,
        'stationary_current_law_held': True, 'further_proposal_accepted': False,
        'held_law_relative_covariance_change': 0., 'held_law_full_and_split_Q_improvements': [0., 0., 0.],
        'held_law_source_evidence_ratio': 1., 'additional_accepted_risk_updates': 0,
        'rejected_proposal': last,
        'scope': 'Two successive precise stationarity diagnostics at actual distinct current laws. Keep the last accepted law exactly;the failed next proposal is not adopted. This is conditional risk stationarity,not joint source convergence.'}


def retain_followup_risk(previous, evidence, spec):
    """Hold a fixed accepted update after its final-law source diagnostic."""
    assert previous['accepted_fixed_risk_update'] and previous['fixed_proposal_qualified']
    assert previous['new_current_stationarity_diagnostic_passed']
    assert previous['checked_current_scales'] != previous['final_risk_scales']
    assert previous['final_risk_scales'] == evidence['final_risk_scales']
    assert previous['fixed_response_scales'] == evidence['fixed_response_scales']
    final = evidence['final_risk_diagnostic_from_parent_bank']
    assert final['stationarity_diagnostic_passed'] and not final['parameters_changed']
    limits = spec['score_precision']
    for summaries, precise, gradient, change in [
        (previous['current_score_summaries'], previous['current_score_precision_passed'],
         previous['scaled_projected_current_score'], previous['new_diagnostic_covariance_change']),
        (final['score_summaries'], final['score_precision_passed'],
         final['scaled_projected_raw_score'], final['relative_diagnostic_covariance_change'])]:
        assert precise
        assert all(max(r['split_rhat']) <= limits['maximum_split_Rhat']
            and min(r['autocorrelation_ess']) >= limits['minimum_functional_ESS'] for r in summaries)
        assert max(summaries[0]['scaled_95_MC_halfwidth']) <= limits['maximum_scaled_95_MC_halfwidth']
        assert max(abs(np.asarray(gradient))) <= spec['maximum_scaled_projected_raw_score']
        assert change <= spec['maximum_relative_covariance_change']
    return {'held_risk_scales': previous['final_risk_scales'],
        'conditional_risk_stationarity_passed': True, 'consecutive_stationarity_checks': 2,
        'stationary_current_law_held': True, 'additional_accepted_risk_updates': 0,
        'held_law_relative_covariance_change': 0., 'held_law_full_and_split_Q_improvements': [0., 0., 0.],
        'held_law_source_evidence_ratio': 1., 'new_candidate_MCMC_sweeps': 0,
        'previous_actual_scales': previous['checked_current_scales'],
        'previous_scaled_projected_raw_score': previous['scaled_projected_current_score'],
        'final_scaled_projected_raw_score': final['scaled_projected_raw_score'],
        'scope': 'Two successive precise source diagnostics at distinct actual laws, before and after one already qualified fixed update. Keep the accepted final law exactly. Conditional risk stationarity at fixed response covariance.'}


def run(plan=PLAN):
    spec = read(plan); out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    path = out / 'results.json'
    if path.exists():
        return read(str(path.relative_to(ROOT)))
    started = time.process_time(); batch = read(spec['risk_batch']); rows = []
    for item in spec['cases']:
        result = read(item['risk_results'])
        rows.append({**item, **retain_risk(result, spec)})
    result = {'execution_plan': plan, 'rows': rows, 'stationary_current_risk_laws_held': len(rows),
        'conditional_risk_stationarity_cases_passed': batch['conditional_risk_stationarity_cases_passed'] + len(rows),
        'registered_source_cases': batch['registered_cases'], 'additional_accepted_risk_updates': 0,
        'new_candidate_MCMC_sweeps': 0, 'source_registry_replaced': False,
        'joint_source_learning_converged': False, 'new_source_labels': 0,
        'new_target_simulator_calls': 0, 'new_verifier_calls': 0, 'server_jobs': 0,
        'cpu_seconds': time.process_time() - started, 'no_model_adopted': True, 'independent_confirmation': False}
    save(path, result)
    print('Stationary actual risk laws held', len(rows), '; conditional stationarity',
        result['conditional_risk_stationarity_cases_passed'], '/32', flush=True)
    return result


if __name__ == '__main__':
    run()
