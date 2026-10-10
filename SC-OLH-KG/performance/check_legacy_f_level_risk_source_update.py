"""Fresh raw source conditioning under learned function and risk levels."""

import copy
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import digamma, polygamma

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_f_collapsed_source_raw_update import sample
from performance.check_legacy_f_learned_covariance_source_update import retained_mixture, verify_learned
from performance.check_legacy_f_original_source_raw_update import common_moments
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_source_incremental_calibration import log_density
from performance.prepare_legacy_f_variance_reference import VarianceReferencePosterior
from performance.run_legacy_related_inventory_source import save

PLAN = 'performance/manifests/legacy_joint_population_f_level_risk_source_update_plan_v1.json'


def updated_fields(spec):
    assert read(spec['Mstep_results'])['all_source_Mstep_gates_passed']
    laws = read(spec['source_laws'])['rows']; risks = read(spec['risk_laws'])['rows']; result = []
    for original in read(spec['source_inputs'])['rows']:
        base = copy.deepcopy(original)
        for index, arm in enumerate(base['arms']):
            law = next(r for r in laws if (r['layout'], r['arm']) == (base['layout'], arm))
            assert law['no_new_posterior_attached']
            displacement = law['learned_f_center'] - base['f_means'][index][0]
            base['f_means'][index] = (np.asarray(base['f_means'][index]) + displacement).tolist()
            common_mean = np.asarray(base['common_latent_prior_means'][index])
            if index == 0:
                common_mean[0] += displacement
            else:
                common_mean += displacement
            base['common_latent_prior_means'][index] = common_mean.tolist()
            for target, origin in [('f_covariances', 'source_f_population_covariance'),
                ('common_latent_prior_covariances', 'common_latent_prior_covariance'),
                ('common_latent_to_source_f_covariances', 'common_latent_to_source_f_covariance')]:
                base[target][index] = law[origin]
        # The scientific eta prior is learned separately in each matched arm.
        for index, arm in enumerate(base['arms']):
            field = copy.deepcopy(base); risk = next(r for r in risks if (r['layout'], r['arm']) == (base['layout'], arm))
            field.update(active_arm=arm, active_arm_index=index, eta_mean=risk['source_eta_prior_mean'],
                eta_covariance=risk['source_eta_prior_covariance'], risk_population_prior_center_and_covariance_used=True)
            d = np.asarray(risk['source_design']); a = np.asarray(risk['common_prior_covariance'])
            np.testing.assert_allclose(d @ risk['prior_mean'], field['eta_mean'], atol=1e-8, rtol=1e-8)
            np.testing.assert_allclose(np.asarray(risk['source_eta_gain']) @ field['eta_covariance'], a @ d.T, atol=1e-8, rtol=1e-8)
            result.append(field)
    return result


def variance_reference(field):
    mean = np.asarray(field['eta_mean']); covariance = np.asarray(field['eta_covariance'])
    labels = np.log(field['sample_variances']) - digamma(1.); noise = float(polygamma(1, 1.))
    factor = cho_factor(covariance + noise * np.eye(48), lower=True)
    posterior_mean = mean + covariance @ cho_solve(factor, labels - mean)
    posterior_covariance = covariance - covariance @ cho_solve(factor, covariance)
    posterior_covariance = .5 * (posterior_covariance + posterior_covariance.T)
    np.linalg.cholesky(posterior_covariance)
    return {'layout': field['layout'], 'arm': field['active_arm'], 'reference_mean': posterior_mean.tolist(),
        'reference_covariance': posterior_covariance.tolist(),
        'approximate_Gaussian_variance_evidence': log_density(labels - mean, covariance + noise * np.eye(48)),
        'reference_is_not_population_prior': True}


def run():
    started = time.process_time(); spec = read(PLAN); fields = updated_fields(spec)
    references = [variance_reference(f) for f in fields]; checks = []
    for field, reference in zip(fields, references):
        checks.append(verify_learned(field, field['active_arm_index'], reference))
    out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    save(out / 'verification.json', {'execution': spec, 'checks': checks, 'all_checks_passed_before_sampling': True})
    save(out / 'frozen_inputs.json', {'execution': spec, 'rows': fields,
        'row_active_arm_selects_its_scientific_risk_prior': True, 'new_level_and_risk_laws_conditioned_here': True})
    save(out / 'references.json', {'execution': spec, 'rows': references, 'all_reference_density_identities_verified': True})
    old = read(spec['old_posterior_results'])['rows']; rows, mixtures, partitions = [], [], []
    try:
        for field, reference in zip(fields, references):
            index = field['active_arm_index']; arm = field['active_arm']; key = (field['layout'], arm)
            model = VarianceReferencePosterior(field['f_means'][index], field['f_covariances'][index],
                field['batch_means'], field['sample_variances'])
            retained = []; row = sample(field, index, spec, started, model,
                reference['reference_mean'], reference['reference_covariance'], retained)
            cm, cc = common_moments(field, index, np.asarray(row['posterior_mean']), np.asarray(row['posterior_covariance']))
            previous = next(r for r in old if (r['layout'], r['arm']) == key)
            old_cc = np.asarray(previous['common_posterior_covariance'])
            row.update(common_coordinate_system=field['source_common_coordinate_systems'][index],
                common_posterior_mean=cm.tolist(), common_posterior_covariance=cc.tolist(),
                common_variance_ratio_mean_vs_old_raw_posterior=float(np.mean(np.diag(cc) / np.diag(old_cc))),
                common_mean_shift_rms_in_old_raw_posterior_sd=float(np.sqrt(np.mean((cm - previous['common_posterior_mean']) ** 2 / np.diag(old_cc)))))
            states = retained[0]; mixture = retained_mixture(field, index, model, states, row, spec)
            selected = states[:, ::spec['retained_mixture']['stride_per_chain']].reshape(-1, 48)
            full_covariance = np.asarray(row['eta_posterior_covariance'])
            eta_mean_error = float(np.sqrt(np.mean((selected.mean(axis=0) - row['eta_posterior_mean']) ** 2 / np.diag(full_covariance))))
            eta_variance_error = float(np.max(np.abs(selected.var(axis=0, ddof=1) / np.diag(full_covariance) - 1)))
            limits = spec['retained_mixture']
            mixture.update(source_eta_mean_rms_difference_in_full_posterior_sd=eta_mean_error,
                maximum_source_eta_variance_relative_difference=eta_variance_error)
            mixture['mixture_gate_passed'] &= bool(eta_mean_error <= limits['maximum_source_eta_mean_rms_difference_in_full_posterior_sd']
                and eta_variance_error <= limits['maximum_source_eta_diagonal_variance_relative_difference_vs_all_draws'])
            partition = {'layout': field['layout'], 'arm': arm}
            for name, part in [('first_half', states[:, :512]), ('second_half', states[:, 512:])]:
                flat = part.reshape(-1, 48); partition[name + '_eta_mean'] = flat.mean(axis=0).tolist()
                partition[name + '_eta_covariance'] = np.cov(flat, rowvar=False).tolist()
            rows.append(row); mixtures.append(mixture); partitions.append(partition)
            print('new level/risk source', *key, 'posterior pass', row['monte_carlo_gate_passed'],
                'Rhat', row['maximum_split_rhat'], 'ESS', row['minimum_autocorrelation_ess'],
                'half f variance', row['maximum_first_second_half_f_variance_relative_difference'],
                'mixture pass', mixture['mixture_gate_passed'], flush=True)
            save(out / 'results.json', {'execution': spec, 'rows': rows, 'all_four_source_updates_complete': len(rows) == 4,
                'diagnostic_interpreted': False, 'cpu_seconds': time.process_time() - started})
            save(out / 'mixtures.json', {'execution': spec, 'rows': mixtures, 'source_posterior_precision_pending': len(rows) < 4})
    except TimeoutError:
        save(out / 'results.json', {'execution': spec, 'rows': rows, 'all_four_source_updates_complete': False,
            'stopped_at_cpu_budget': True, 'diagnostic_interpreted': False, 'cpu_seconds': time.process_time() - started})
        return
    posterior_passed = all(r['monte_carlo_gate_passed'] for r in rows)
    mixture_passed = all(r['mixture_gate_passed'] for r in mixtures)
    save(out / 'source_eta_partitions.json', {'execution': spec, 'rows': partitions})
    save(out / 'mixtures.json', {'execution': spec, 'rows': mixtures, 'all_mixture_retention_gates_passed': mixture_passed,
        'all_source_posterior_precision_gates_passed': posterior_passed})
    save(out / 'results.json', {'execution': spec, 'rows': rows, 'all_four_source_updates_complete': True,
        'all_monte_carlo_gates_passed': posterior_passed, 'all_mixture_retention_gates_passed': mixture_passed,
        'diagnostic_interpreted': posterior_passed and mixture_passed, 'cpu_seconds': time.process_time() - started,
        'cost': {**spec['cost'], 'actual_sweeps': 24576, 'retained_positive_mixture_components_per_arm': 512},
        'raw_source_likelihood_factors_per_arm': {'batch_means': 48, 'sample_variances': 48},
        'additional_old32_Gaussian_risk_likelihood_factors': 0,
        'no_model_adopted': True, 'independent_confirmation': False,
        'interpretation': 'Fresh source posterior under the Stage213 function and arm-specific risk law,using exactly compensated recomputed references. No old moments attached to the new prior. Positive mixture retained;one M step is not a converged fit or independent decision result.'})


if __name__ == '__main__':
    run()
