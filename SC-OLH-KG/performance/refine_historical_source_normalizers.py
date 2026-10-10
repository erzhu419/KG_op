"""Replace four imprecise bridge banks at fixed current source laws."""

import time

import numpy as np
from scipy.special import digamma, polygamma

from performance.complete_historical_source_heads import ROOT, read, save
from performance.compare_historical_source_structures import KEY, key, prepare, scientific_log_density
from performance.bridge_historical_source_normalizers import bridge_summary, assemble
from performance.adapt_historical_source_covariance import gaussian_rows_log_density
from performance.prepare_legacy_f_variance_reference import VarianceReferencePosterior, variance_reference
from performance.extend_historical_source_risk_score_bank import draw_risk_states
from performance.diagnose_legacy_f_source_incremental_calibration import log_density

PLAN = 'performance/manifests/historical_source_normalizer_precision_v1.json'


def case_run(entry, previous, spec):
    case = entry['case']; suffix = '_'.join(key(case))
    out = ROOT / spec['output'] / 'cases' / suffix
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'results.json').exists():
        return read(str((out / 'results.json').relative_to(ROOT)))
    started = time.process_time()
    assert previous['fixed_response_scales'] == case['f_common_task_scales']
    assert previous['fixed_risk_scales'] == case['eta_common_task_scales']
    mean = np.asarray(previous['Gaussian_reference_mean'])
    covariance = np.asarray(previous['Gaussian_reference_covariance'])
    original_count = previous['Gaussian_reference_draws'] // 4
    count = spec['Gaussian_draws_per_chain']
    assert count >= original_count
    rng = np.random.default_rng(previous['Gaussian_reference_seed'])
    draws = rng.normal(size=(4, original_count, 48))
    if count > original_count:
        draws = np.concatenate([draws, rng.normal(size=(4, count - original_count, 48))], axis=1)
    draws = mean + draws @ np.linalg.cholesky(covariance).T
    reference_mean, reference_covariance = variance_reference(entry['risk_mean'], entry['risk_covariance'], entry['model'].s2)
    model = VarianceReferencePosterior(entry['model'].mean, entry['model'].covariance, entry['model'].labels, entry['model'].s2)
    # Check the actual reference compensation before sampling this new bank.
    labels = np.log(model.s2) - digamma(1.)
    constant = log_density(labels - entry['risk_mean'], entry['risk_covariance'] + float(polygamma(1, 1.)) * np.eye(48))
    points = np.stack([reference_mean, mean]).reshape(1, 2, 48)
    left = scientific_log_density(entry['model'], entry['risk_mean'], entry['risk_covariance'], points)
    right = gaussian_rows_log_density(points.reshape(-1, 48), reference_mean, reference_covariance)
    right += np.asarray([model.evaluate(p)[0] for p in points[0]]) + constant
    error = float(np.max(abs(left.ravel() - right)))
    assert error < 1e-8
    eta = draw_risk_states(model, reference_mean, reference_covariance, case, spec['sampler'], started,
        spec['maximum_case_single_thread_cpu_seconds'])
    gaussian_proposal = gaussian_rows_log_density(draws.reshape(-1, 48), mean, covariance).reshape(draws.shape[:2])
    gaussian_posterior = gaussian_rows_log_density(eta.reshape(-1, 48), mean, covariance).reshape(eta.shape[:2])
    raw_proposal = scientific_log_density(entry['model'], entry['risk_mean'], entry['risk_covariance'], draws)
    raw_posterior = scientific_log_density(entry['model'], entry['risk_mean'], entry['risk_covariance'], eta)
    summary = bridge_summary(raw_proposal - gaussian_proposal, raw_posterior - gaussian_posterior,
        np.ones(draws.shape[:2]), np.ones(eta.shape[:2]), spec['evidence_precision'])
    # These scalar ratios suffice for another precision calculation at the
    # frozen reference;keeping full48-dimensional chains would be wasteful.
    np.savez_compressed(out / 'bridge_density_ratios.npz',
        proposal=raw_proposal - gaussian_proposal, posterior=raw_posterior - gaussian_posterior)
    result = {**previous, **summary, 'execution_plan': PLAN, 'previous_bridge_certified': previous['precision_gate_passed'],
        'previous_log_source_evidence': previous['log_source_evidence'],
        'previous_source_normalizer_results': spec['previous_results'],
        'posterior_components': int(np.prod(eta.shape[:2])), 'posterior_bank_method': 'fresh fixed-current-law full elliptical chains',
        'posterior_component_file': case['candidate_component_file'], 'posterior_component_file_replaced': False,
        'fresh_source_MCMC_sweeps': spec['sampler']['chains'] * (spec['sampler']['warmup_sweeps_per_chain']
            + spec['sampler']['retained_sweeps_per_chain']),
        'raw_source_reference_density_identity_maximum_error': error,
        'Gaussian_reference_draws': int(np.prod(draws.shape[:2])),
        'additional_Gaussian_integration_draws': 4 * (count - original_count),
        'Gaussian_reference_original_prefix_preserved': True,
        'bridge_density_ratio_statistics': str((out / 'bridge_density_ratios.npz').relative_to(ROOT)),
        'Gaussian_reference_replayed_without_new_points': count == original_count,
        'sampler': spec['sampler'], 'cpu_seconds': time.process_time() - started}
    save(out / 'results.json', result)
    print('source normalizer precision', suffix, 'certified', summary['precision_gate_passed'],
        'split gap', round(summary['split_log_evidence_difference'], 4), 'CPU', round(result['cpu_seconds'], 2), flush=True)
    return result


def run(plan=PLAN):
    spec = read(plan); out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    if (out / 'results.json').exists() and read(spec['output'] + '/results.json')['all_registered_cases_complete']:
        return read(spec['output'] + '/results.json')
    previous = read(spec['previous_results'])
    by_key = {key(r): r for r in previous['rows']}
    started = time.process_time(); models = prepare(spec)
    results = []
    for case in spec['cases']:
        if time.process_time() - started >= spec['maximum_single_thread_cpu_seconds']:
            break
        case_spec = {**spec, 'sampler': {**spec['sampler'],
            'retained_sweeps_per_chain': case.get('retained_sweeps_per_chain', spec['sampler']['retained_sweeps_per_chain'])}}
        results.append(case_run(models[key(case)], by_key[key(case)], case_spec))
    completed = {key(r): r for r in results}
    reused = [read(p) for p in spec.get('reused_normalizer_results', [])]
    for row in reused:
        assert row['precision_gate_passed'] and row['posterior_bank_method'] == 'fresh fixed-current-law full elliptical chains'
        completed[key(row)] = row
    rows = [completed.get(key(r), r) for r in previous['rows']]
    phase = assemble(spec, rows, time.process_time() - started)
    phase.update(execution_plan=plan, all_registered_precision_cases_complete=len(results) == len(spec['cases']),
        all_registered_cases_complete=len(results) == len(spec['cases']),
        reused_certified_normalizers=len(previous['rows']) - len(spec['cases']),
        new_Gaussian_integration_draws=sum(r.get('additional_Gaussian_integration_draws', 0) for r in results),
        reused_Gaussian_integration_draws=sum(r['Gaussian_reference_draws'] for r in previous['rows']),
        new_source_MCMC_sweeps=sum(r['fresh_source_MCMC_sweeps'] for r in results),
        recorded_new_case_CPU_seconds=sum(r['cpu_seconds'] for r in results),
        previous_results=spec['previous_results'], original_prelearning_source_cases_resampled=0,
        unit_test_results=spec['output'] + '/unit_tests.json')
    if reused:
        phase['reference_fit_and_bridge_posterior_banks_separated'] = True
        phase['next_work'] = 'Use the complete within-layout source weights only after every normalizer passes;reconnect actual learned priors to matched cached target KG/GP decisions.'
    save(out / 'results.json', phase)
    print('Source normalizer closure', phase['absolute_source_normalizers_certified'], '/32;identities',
        phase['joint_source_identities_certified'], '/16;CPU', round(phase['cpu_seconds'], 2), flush=True)
    return phase


if __name__ == '__main__':
    run()
