"""Extend the last fixed-source posterior prefix;reuse scalar Gaussian ratios."""

import time

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from performance.complete_historical_source_heads import ROOT, read, save
from performance.compare_historical_source_structures import KEY, key, prepare, scientific_log_density
from performance.bridge_historical_source_normalizers import bridge_summary, assemble
from performance.adapt_historical_source_covariance import gaussian_rows_log_density
from performance.prepare_legacy_f_variance_reference import VarianceReferencePosterior, variance_reference
from performance.extend_historical_source_risk_score_bank import draw_risk_states

PLAN = 'performance/manifests/historical_last_source_normalizer_v1.json'


def export_source_weights(phase, spec):
    """Freeze complete source weights and simultaneous numerical-error bounds."""
    if not phase['all_cross_structure_evidence_certified']:
        return
    heads = {key(r): r for r in phase['rows']}
    blocks = {}
    for layout, block in phase['source_structure_weights_by_layout'].items():
        quantile = float(norm.ppf(1. - .05 / (2 * 4 * len(block['identities']))))
        lower, upper = [], []
        for identity in block['identities']:
            pair = [heads[(*identity, h)] for h in ('f', 'g')]
            center = sum(r['log_source_evidence'] for r in pair)
            width = quantile * sum(sum(r['proposal_and_posterior_log_evidence_MCSE']) for r in pair)
            lower.append(center - width); upper.append(center + width)
        bounds = []
        for i in range(len(lower)):
            low_den = upper.copy(); low_den[i] = lower[i]
            high_den = lower.copy(); high_den[i] = upper[i]
            bounds.append([float(np.exp(lower[i] - logsumexp(low_den))),
                float(np.exp(upper[i] - logsumexp(high_den)))])
        blocks[layout] = {**block, 'conditional_simultaneous_log_evidence_intervals_95': list(zip(lower, upper)),
            'conditional_source_weight_bounds_95': bounds, 'interval_normal_quantile': quantile}
    path = ROOT / spec['output'] / 'source_structure_weights.json'
    save(path, {'source_normalizer_results': spec['output'] + '/results.json',
        'source_learning': spec['source_learning'], 'source_precision': spec['source_precision'], 'by_layout': blocks,
        'source_only': True, 'new_target_observations_used': 0,
        'scope': 'Equal-prior empirical-Bayes structure weights,complete eight-identity blocks on their own source data. Simultaneous bounds propagate conditional Gaussian/posterior bridge Monte Carlo errors by a union bound;they do not represent model or physical-factor uncertainty.'})
    phase['qualified_source_structure_weights'] = str(path.relative_to(ROOT))


def run(plan=PLAN):
    spec = read(plan); out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    if (out / 'results.json').exists():
        return read(spec['output'] + '/results.json')
    started = time.process_time(); base = read(spec['previous_results'])
    previous = next(r for r in base['rows'] if key(r) == key(spec['case']))
    assert not previous['precision_gate_passed']
    entry = prepare(spec)[key(spec['case'])]
    for name, field in [('f_common_task_scales', 'fixed_response_scales'), ('eta_common_task_scales', 'fixed_risk_scales')]:
        assert entry['case'][name] == previous[field]
    saved = np.load(ROOT / previous['bridge_density_ratio_statistics'])
    proposal, old_posterior = saved['proposal'], saved['posterior']
    mean = np.asarray(previous['Gaussian_reference_mean'])
    covariance = np.asarray(previous['Gaussian_reference_covariance'])
    rm, rc = variance_reference(entry['risk_mean'], entry['risk_covariance'], entry['model'].s2)
    model = VarianceReferencePosterior(entry['model'].mean, entry['model'].covariance, entry['model'].labels, entry['model'].s2)
    eta = draw_risk_states(model, rm, rc, entry['case'], spec['sampler'], started, spec['maximum_single_thread_cpu_seconds'])
    gaussian = gaussian_rows_log_density(eta.reshape(-1, 48), mean, covariance).reshape(eta.shape[:2])
    posterior = scientific_log_density(entry['model'], entry['risk_mean'], entry['risk_covariance'], eta) - gaussian
    prefix_error = float(np.max(abs(posterior[:, :old_posterior.shape[1]] - old_posterior)))
    assert prefix_error < 1e-9
    summary = bridge_summary(proposal, posterior, np.ones(proposal.shape), np.ones(posterior.shape), spec['evidence_precision'])
    np.savez_compressed(out / 'bridge_density_ratios.npz', proposal=proposal, posterior=posterior)
    updated = {**previous, **summary, 'execution_plan': plan, 'sampler': spec['sampler'],
        'posterior_components': int(posterior.size), 'posterior_bank_method': 'original fixed-current-law posterior prefix extended once',
        'posterior_component_file_replaced': False, 'new_Gaussian_integration_draws': 0,
        'source_density_ratio_prefix_maximum_error': prefix_error,
        'previous_separate_bank_normalizer_results': spec['previous_results'],
        'bridge_density_ratio_statistics': str((out / 'bridge_density_ratios.npz').relative_to(ROOT)),
        'cpu_seconds': time.process_time() - started}
    rows = [updated if key(r) == key(previous) else r for r in base['rows']]
    phase = assemble(spec, rows, time.process_time() - started)
    phase.update(execution_plan=plan, previous_results=spec['previous_results'],
        reference_fit_and_bridge_posterior_banks_separated=True, previous_same_bank_certification_withdrawn=True,
        reused_certified_normalizers=31, new_Gaussian_integration_draws=0,
        new_source_MCMC_sweeps=spec['sampler']['chains'] * (spec['sampler']['warmup_sweeps_per_chain']
            + spec['sampler']['retained_sweeps_per_chain']),
        additional_retained_source_sweeps=posterior.size - old_posterior.size,
        original_prelearning_source_cases_resampled=0,
        final_precision_case=updated, unit_test_results=spec['output'] + '/unit_tests.json')
    export_source_weights(phase, spec)
    save(out / 'results.json', phase)
    print('Final source normalizers', phase['absolute_source_normalizers_certified'], '/32;identities',
        phase['joint_source_identities_certified'], '/16;last split gap', round(summary['split_log_evidence_difference'], 4),
        'CPU', round(phase['cpu_seconds'], 2), flush=True)
    return phase


if __name__ == '__main__':
    run()
