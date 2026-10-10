"""Test a fixed larger source-only risk step against Q and marginal evidence."""

import time

import numpy as np

from performance.complete_historical_source_heads import ROOT, IDENTITY, read, save
from performance.continue_historical_raw_source_learning import load_mixture, risk_bank_moments
from performance.learn_historical_raw_source_amplitudes import covariance_builder
from performance.check_legacy_f_raw_covariance_mstep import expected_objective
from performance.check_legacy_f_risk_amplitude_importance import log_ratios
from performance.probe_historical_source_risk_plateau import ratio_summary

PLAN = 'performance/manifests/historical_source_risk_acceleration_probe_v1.json'
KEY = (*IDENTITY, 'head')


def run():
    spec = read(PLAN); started = time.process_time()
    (ROOT / spec['output']).mkdir(parents=True, exist_ok=True)
    original = read(spec['previous_learning']); step = read(spec['learning'])
    precision = read(spec['previous_precision']); registry = read(spec['risk_statistics'])
    source = read(spec['source_plan']); laws = read(source['source_laws'])['rows']; rows = []
    assert len(precision['rows']) == 32 and all(r['positive_mixture_precision_passed'] for r in precision['rows'])
    for record in precision['rows']:
        case = next(r for r in original['evidence_rows'] if all(r[k] == record[k] for k in KEY))
        learned = next(r for r in step['evidence_rows'] if all(r[k] == record[k] for k in KEY))
        bank = read(next(r['results'] for r in registry['rows'] if all(r[k] == record[k] for k in KEY)))
        risk = next(r for r in laws if r['head'] == ('eta_g' if case['head'] == 'g' else 'eta_f')
            and all(r[k] == case[k] for k in IDENTITY))
        mean = np.asarray(risk['source_mean']); builder = covariance_builder(risk)
        current = np.asarray(case['eta_common_task_scales']); proposed = current + spec['step_multiplier'] * (
            np.asarray(learned['eta_common_task_scales']) - current)
        assert np.min(proposed) >= 0.
        _, _, seconds = risk_bank_moments(bank, case, mean)
        improvements = [expected_objective(current, builder, s)[0] - expected_objective(proposed, builder, s)[0]
            for s in seconds]
        eta, weights = load_mixture(record, case)
        lr = log_ratios(eta.reshape(-1, 48) - mean, builder(current)[0], builder(proposed)[0]).reshape(weights.shape)
        evidence = ratio_summary(lr, weights, spec['precision_gates'])
        rows.append({**{k: case[k] for k in KEY}, 'current_eta_common_task_scales': current.tolist(),
            'proposed_eta_common_task_scales': proposed.tolist(), 'full_and_split_Q_improvements': improvements,
            'Q_gate_passed': bool(min(improvements) > 0.), **evidence,
            'joint_acceleration_probe_passed': bool(min(improvements) > 0. and evidence['eligible_improvement_excludes_zero'])})
    result = {'execution_plan': PLAN, 'rows': rows, 'all_registered_cases_complete': len(rows) == 32,
        'Q_cases_passed': sum(r['Q_gate_passed'] for r in rows),
        'evidence_precision_cases_passed': sum(r['precision_gate_passed'] for r in rows),
        'Q_and_evidence_improvement_cases_passed': sum(r['joint_acceleration_probe_passed'] for r in rows),
        'cpu_seconds': time.process_time() - started, 'new_candidate_MCMC_sweeps': 0,
        'new_source_labels': 0, 'new_target_simulator_calls': 0, 'new_verifier_calls': 0, 'server_jobs': 0,
        'no_model_adopted': True, 'independent_confirmation': False,
        'scope': 'Fixed sixth-law response parameters;1.5 times the sixth-to-seventh risk step. Positive full/split Q uses certified long-chain sixth-law moments;marginal evidence uses the existing sixth-law weighted mixture. No posterior-retention certificate or accepted replacement parameters.'}
    save(ROOT / spec['output'] / 'results.json', result)
    print('fixed acceleration probe:Q', result['Q_cases_passed'], 'evidence precision', result['evidence_precision_cases_passed'],
        'both improving', result['Q_and_evidence_improvement_cases_passed'], flush=True)
    return result


if __name__ == '__main__':
    run()
