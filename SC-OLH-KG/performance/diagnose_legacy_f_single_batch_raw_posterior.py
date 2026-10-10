"""Deterministic single-batch posterior moments,with uncertain episode variance."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_inner_conditional_calibration import integrate
from performance.diagnose_legacy_f_source_latent_noise_feasibility import ARMS
from performance.run_legacy_related_inventory_source import save
from representation.physical_risk_prior import log_sample_variance_density

PLAN = 'performance/manifests/legacy_joint_population_f_single_batch_raw_posterior_plan_v1.json'


def posterior_pair(mean, variance, em, ev, label, s2):
    mean, variance = np.asarray(mean), np.asarray(variance); residual = label - mean; sd = np.sqrt(ev)
    def at(t, index=slice(None)):
        eta = em + sd * t; total = variance[index] + np.exp(eta) / 3
        return (-.5 * (np.log(2 * np.pi) + t * t) + log_sample_variance_density(eta, s2, 3)
            - .5 * (np.log(2 * np.pi * total) + residual[index] ** 2 / total))
    def statistics(t):
        noise = np.exp(em + sd * t) / 3; gain = variance / (variance + noise)
        displacement = gain * residual; conditional_variance = variance * (1 - gain)
        return np.column_stack([np.ones(2), displacement, conditional_variance + displacement ** 2,
            conditional_variance, displacement ** 2, np.full(2, noise)])
    moments, logs, error = integrate(at, statistics); rows = []
    for index, arm in enumerate(ARMS):
        displacement = moments[index, 1]; total = moments[index, 2] - displacement ** 2
        within = moments[index, 3]; between = moments[index, 4] - displacement ** 2
        np.testing.assert_allclose(total, within + between, atol=1e-12, rtol=1e-10)
        plugin_noise = s2 / 3; plugin_gain = variance[index] / (variance[index] + plugin_noise)
        plugin_variance = variance[index] * plugin_noise / (variance[index] + plugin_noise)
        plugin_mean = mean[index] + plugin_gain * residual[index]
        assert total > 0 and plugin_variance > 0
        rows.append({'arm': arm, 'posterior_mean': float(mean[index] + displacement),
            'posterior_variance': float(total), 'posterior_within_eta_variance': float(within),
            'posterior_between_eta_mean_variance': float(between), 'posterior_expected_episode_noise_over_three': float(moments[index, 5]),
            'plug_in_posterior_mean': float(plugin_mean), 'plug_in_posterior_variance': float(plugin_variance),
            'posterior_variance_ratio': float(total / plugin_variance),
            'posterior_mean_shift_in_plugin_sd': float((mean[index] + displacement - plugin_mean) / np.sqrt(plugin_variance)),
            'sample_noise_to_latent_f_prior_variance_ratio': float(plugin_noise / variance[index]),
            'joint_batch_log_density': float(logs[index]), 'integration_relative_error_bound': error})
    return rows


def run():
    started = time.process_time(); spec = read(PLAN); baseline = read(spec['known_source_results'])
    assert baseline['diagnostic_interpreted']; joint = read(spec['joint_source_results'])
    assert joint['all_four_conditionals_complete'] and not joint['all_monte_carlo_gates_passed']
    priors = read(spec['known_source_inputs'])['rows']; observations = read(spec['joint_source_inputs'])['rows']; rows = []
    for prior in priors:
        observation = next(r for r in observations if r['layout'] == prior['layout']); assert prior['sites'] == observation['sites']
        for j, site in enumerate(prior['sites']):
            assert time.process_time() - started < spec['maximum_single_thread_cpu_seconds']
            results = posterior_pair(prior['f_means'][j], prior['f_variances'][j], prior['eta_mean'][j], prior['eta_variance'][j],
                observation['added_means'][j], observation['added_sample_variances'][j])
            for result in results:
                previous = next(r for r in baseline['rows'] if (r['layout'], r['arm'], r['source_task'], r['policy']) ==
                    (prior['layout'], result['arm'], site['source_task'], site['policy']))
                difference = abs(result['joint_batch_log_density'] - previous['joint_batch_density']); assert difference <= 1e-8
                result.update(layout=prior['layout'], **site, previous_joint_density_absolute_error=difference); rows.append(result)
    summaries = []
    for layout in spec['layouts']:
        for arm in ARMS:
            selected = [r for r in rows if (r['layout'], r['arm']) == (layout, arm)]
            ratio = np.asarray([r['posterior_variance_ratio'] for r in selected])
            prior_noise = np.asarray([r['sample_noise_to_latent_f_prior_variance_ratio'] for r in selected])
            maximum = selected[int(np.argmax(ratio))]
            summaries.append({'layout': layout, 'arm': arm, 'posterior_variance_ratio_mean': float(ratio.mean()),
                'posterior_variance_ratio_quantiles_0_50_100': np.quantile(ratio, [0, .5, 1]).tolist(),
                'posterior_mean_shift_rms_in_plugin_sd': float(np.sqrt(np.mean([r['posterior_mean_shift_in_plugin_sd'] ** 2 for r in selected]))),
                'variance_ratio_to_sample_noise_spearman': float(spearmanr(ratio, prior_noise).statistic),
                'maximum_ratio_source_task': maximum['source_task'], 'maximum_ratio_policy': maximum['policy'],
                'maximum_ratio_sample_noise_to_latent_f_variance': maximum['sample_noise_to_latent_f_prior_variance_ratio']})
    out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    save(out / 'results.json', {'execution': spec, 'rows': rows, 'summary': summaries,
        'diagnostic_interpreted': True, 'all_joint_density_and_variance_decomposition_checks_passed': True,
        'joint_GP_precision_gate_still_failed': True, 'new_MCMC_sweeps': 0, 'cpu_seconds': time.process_time() - started,
        'cost': {**spec['cost'], 'actual_two_arm_integrations': len(rows) // 2},
        'no_model_adopted': True, 'independent_confirmation': False,
        'interpretation': 'Deterministic single-batch latent-f posterior under the original Gaussian source marginals and proper joint mean/variance likelihood. Not the full joint48-batch posterior or a repair of the failed joint GP precision gate. No extra training or held-task forecast.'})
    print('single-batch raw posterior', summaries, 'CPU', time.process_time() - started, flush=True)


if __name__ == '__main__':
    run()
