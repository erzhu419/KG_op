"""Raw conditioning of the original source batches,and common-law moments."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_f_collapsed_source_raw_update import CollapsedRawSourcePosterior, sample, verify
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.run_legacy_related_inventory_source import save

PLAN = 'performance/manifests/legacy_joint_population_f_original_source_raw_update_plan_v1.json'


def common_moments(field, index, source_mean, source_covariance):
    prior_source_mean = np.asarray(field['f_means'][index]); prior_source_cov = np.asarray(field['f_covariances'][index])
    mean = np.asarray(field['common_latent_prior_means'][index]); covariance = np.asarray(field['common_latent_prior_covariances'][index])
    cross = np.asarray(field['common_latent_to_source_f_covariances'][index])
    gain = cho_solve(cho_factor(prior_source_cov, lower=True), cross.T).T
    residual = covariance - gain @ cross.T
    posterior_mean = mean + gain @ (source_mean - prior_source_mean)
    posterior_cov = residual + gain @ source_covariance @ gain.T
    posterior_cov = .5 * (posterior_cov + posterior_cov.T)
    assert np.linalg.eigvalsh(posterior_cov).min() >= -1e-10
    return posterior_mean, posterior_cov


def run():
    started = time.process_time(); spec = read(PLAN); contract = read(spec['source_contract_results'])
    assert contract['all_original48_raw_contracts_complete'] and contract['diagnostic_interpreted']
    assert read(spec['collapsed_results'])['all_monte_carlo_gates_passed']
    source = read(spec['source_contract_inputs']); assert source['original_variances_not_conditioned_in_prior']
    fields = source['rows']; out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    checks = [verify(field, index, None) for field in fields for index in range(2)]
    save(out / 'verification.json', {'execution': spec, 'checks': checks, 'all_checks_passed_before_sampling': True})
    rows = []
    try:
        for field in fields:
            for index in range(2):
                model = CollapsedRawSourcePosterior(field['f_means'][index], field['f_covariances'][index],
                    field['batch_means'], field['sample_variances'])
                row = sample(field, index, spec, started, model, field['eta_mean'], field['eta_covariance'])
                common_mean, common_cov = common_moments(field, index, np.asarray(row['posterior_mean']), np.asarray(row['posterior_covariance']))
                plugin_mean, plugin_cov = common_moments(field, index, np.asarray(row['plug_in_posterior_mean']), np.asarray(row['plug_in_posterior_covariance']))
                row.update(common_coordinate_system=field['source_common_coordinate_systems'][index],
                    common_posterior_mean=common_mean.tolist(), common_posterior_covariance=common_cov.tolist(),
                    plug_in_common_posterior_mean=plugin_mean.tolist(), plug_in_common_posterior_covariance=plugin_cov.tolist(),
                    common_posterior_variance_ratio_mean=float(np.mean(np.diag(common_cov) / np.diag(plugin_cov))),
                    common_mean_shift_rms_in_plugin_sd=float(np.sqrt(np.mean((common_mean - plugin_mean) ** 2 / np.diag(plugin_cov)))))
                rows.append(row)
                print('original raw source', row['layout'], row['arm'], 'gate', row['monte_carlo_gate_passed'],
                    'Rhat', row['maximum_split_rhat'], 'ESS', row['minimum_autocorrelation_ess'],
                    'half variance', row['maximum_first_second_half_f_variance_relative_difference'],
                    'source variance ratio', row['posterior_variance_ratio_mean'], flush=True)
                save(out / 'results.json', {'execution': spec, 'rows': rows, 'diagnostic_interpreted': False,
                    'all_four_source_updates_complete': len(rows) == 4, 'cpu_seconds': time.process_time() - started})
    except TimeoutError:
        save(out / 'results.json', {'execution': spec, 'rows': rows, 'diagnostic_interpreted': False,
            'all_four_source_updates_complete': False, 'stopped_at_cpu_budget': True, 'cpu_seconds': time.process_time() - started})
        return
    passed = all(r['monte_carlo_gate_passed'] for r in rows)
    save(out / 'results.json', {'execution': spec, 'rows': rows, 'all_four_source_updates_complete': True,
        'all_monte_carlo_gates_passed': passed, 'diagnostic_interpreted': passed, 'all_pre_sampling_checks_passed': True,
        'cpu_seconds': time.process_time() - started, 'cost': {**spec['cost'], 'actual_chains': 16,
            'actual_sweeps': 16 * (spec['sampler']['warmup_sweeps_per_chain'] + spec['sampler']['retained_sweeps_per_chain'])},
        'raw_original_source_likelihood_factors_per_arm': {'batch_means': 48, 'sample_variances': 48},
        'additional_old32_Gaussian_risk_likelihood_factors': 0,
        'validation_task0_or_task1_outcomes_indexed': False, 'no_model_adopted': True, 'independent_confirmation': False,
        'interpretation': 'Original48 source raw likelihood update from unconditioned empirical population priors. Common and function moments retain variance uncertainty;they are not a Gaussian replacement for the source posterior. Covariance templates/hyperparameters and encoder remain old empirical estimates;no held-source or deployment gain measured.'})


if __name__ == '__main__':
    run()
