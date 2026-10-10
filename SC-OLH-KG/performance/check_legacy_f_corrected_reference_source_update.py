"""Unchanged original source posterior using a corrected variance reference."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_f_collapsed_source_raw_update import sample
from performance.check_legacy_f_original_source_raw_update import common_moments
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.prepare_legacy_f_variance_reference import VarianceReferencePosterior
from performance.run_legacy_related_inventory_source import save

PLAN = 'performance/manifests/legacy_joint_population_f_corrected_reference_source_update_plan_v1.json'


def run():
    started = time.process_time(); spec = read(PLAN); verified = read(spec['reference_results'])
    assert verified['diagnostic_interpreted'] and verified['original_population_priors_and_exact_likelihood_unchanged']
    references = read(spec['references']); assert references['all_reference_density_identities_verified']
    fields = read(spec['source_inputs'])['rows']; out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True); rows = []
    try:
        for field in fields:
            reference = next(r for r in references['rows'] if r['layout'] == field['layout'])
            for index in range(2):
                model = VarianceReferencePosterior(field['f_means'][index], field['f_covariances'][index],
                    field['batch_means'], field['sample_variances'])
                row = sample(field, index, spec, started, model, reference['reference_mean'], reference['reference_covariance'])
                mean, covariance = common_moments(field, index, np.asarray(row['posterior_mean']), np.asarray(row['posterior_covariance']))
                pm, pc = common_moments(field, index, np.asarray(row['plug_in_posterior_mean']), np.asarray(row['plug_in_posterior_covariance']))
                row.update(common_coordinate_system=field['source_common_coordinate_systems'][index],
                    common_posterior_mean=mean.tolist(), common_posterior_covariance=covariance.tolist(),
                    plug_in_common_posterior_mean=pm.tolist(), plug_in_common_posterior_covariance=pc.tolist(),
                    common_posterior_variance_ratio_mean=float(np.mean(np.diag(covariance) / np.diag(pc))),
                    common_mean_shift_rms_in_plugin_sd=float(np.sqrt(np.mean((mean - pm) ** 2 / np.diag(pc)))))
                rows.append(row)
                print('corrected reference', row['layout'], row['arm'], 'gate', row['monte_carlo_gate_passed'],
                    'Rhat', row['maximum_split_rhat'], 'ESS', row['minimum_autocorrelation_ess'],
                    'half variance', row['maximum_first_second_half_f_variance_relative_difference'], flush=True)
                save(out / 'results.json', {'execution': spec, 'rows': rows, 'diagnostic_interpreted': False,
                    'all_four_source_updates_complete': len(rows) == 4, 'cpu_seconds': time.process_time() - started})
    except TimeoutError:
        save(out / 'results.json', {'execution': spec, 'rows': rows, 'diagnostic_interpreted': False,
            'all_four_source_updates_complete': False, 'stopped_at_cpu_budget': True, 'cpu_seconds': time.process_time() - started})
        return
    passed = all(r['monte_carlo_gate_passed'] for r in rows)
    save(out / 'results.json', {'execution': spec, 'rows': rows, 'all_four_source_updates_complete': True,
        'all_monte_carlo_gates_passed': passed, 'diagnostic_interpreted': passed, 'cpu_seconds': time.process_time() - started,
        'cost': {**spec['cost'], 'actual_chains': 16,
            'actual_sweeps': 16 * (spec['sampler']['warmup_sweeps_per_chain'] + spec['sampler']['retained_sweeps_per_chain'])},
        'raw_original_source_likelihood_factors_per_arm': {'batch_means': 48, 'sample_variances': 48},
        'additional_old32_Gaussian_risk_likelihood_factors': 0,
        'original_empirical_population_priors_and_raw_likelihood_unchanged': True,
        'no_model_adopted': True, 'independent_confirmation': False,
        'interpretation': 'Density-corrected variance reference estimates the unchanged original48-source raw posterior. Reference variance likelihood cancels exactly;the scientific eta prior is still Stage200 population law. First/second-half sufficient moments retained for covariance-score diagnostics,not Gaussian paid-update replacement.'})


if __name__ == '__main__':
    run()
