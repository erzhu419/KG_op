"""Fresh original-source posterior after the single raw covariance M step."""

import copy
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_f_collapsed_source_raw_update import aggregate, sample, verify
from performance.check_legacy_f_original_source_raw_update import common_moments
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_source_incremental_calibration import log_density
from performance.prepare_legacy_f_variance_reference import VarianceReferencePosterior
from performance.run_legacy_related_inventory_source import save

PLAN = 'performance/manifests/legacy_joint_population_f_learned_covariance_source_update_plan_v1.json'


def learned_fields(spec):
    assert read(spec['Mstep_results'])['all_source_Mstep_gates_passed']
    laws = read(spec['source_laws']); assert laws['all_four_source_Mstep_gates_passed']
    fields = copy.deepcopy(read(spec['source_inputs'])['rows'])
    for field in fields:
        for index, arm in enumerate(field['arms']):
            law = next(r for r in laws['rows'] if (r['layout'], r['arm']) == (field['layout'], arm))
            assert law['no_new_posterior_attached']
            field['f_covariances'][index] = law['source_f_population_covariance']
            field['common_latent_prior_covariances'][index] = law['common_latent_prior_covariance']
            field['common_latent_to_source_f_covariances'][index] = law['common_latent_to_source_f_covariance']
    return fields


def verify_learned(field, index, reference):
    check = verify(field, index, None)
    model = VarianceReferencePosterior(field['f_means'][index], field['f_covariances'][index],
        field['batch_means'], field['sample_variances'])
    point = np.asarray(reference['reference_mean'])
    corrected = model.evaluate(point)
    approximate = -.5 * np.sum(np.log(2 * np.pi * model.approximate_noise)
        + (model.approximate_labels - point) ** 2 / model.approximate_noise)
    left = log_density(point - reference['reference_mean'], reference['reference_covariance']) + corrected[0]
    right = log_density(point - field['eta_mean'], field['eta_covariance']) + corrected[0] + approximate
    right -= reference['approximate_Gaussian_variance_evidence']
    error = abs(left - right); assert error <= 1e-8
    delta, covariance = model.moments(corrected)
    mean = model.mean + delta
    cm, cc = common_moments(field, index, mean, covariance)
    cross = np.asarray(field['common_latent_to_source_f_covariances'][index])
    direct_mean = np.asarray(field['common_latent_prior_means'][index]) + cross @ corrected[2]
    direct_covariance = np.asarray(field['common_latent_prior_covariances'][index]) - cross @ cho_solve(corrected[1], cross.T)
    np.testing.assert_allclose(cm, direct_mean, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(cc, direct_covariance, atol=1e-9, rtol=1e-8)
    check.update(compensated_density_identity_error=error,
        common_conditional_mean_identity_error=float(np.max(np.abs(cm - direct_mean))),
        common_conditional_covariance_identity_error=float(np.max(np.abs(cc - direct_covariance))))
    return check


def retained_mixture(field, index, model, states, posterior, spec):
    limits = spec['retained_mixture']; selected = states[:, ::limits['stride_per_chain']]
    assert selected.shape[0] * selected.shape[1] == limits['components_per_arm']
    ds, cs = [], []
    for chain in selected:
        moments = [model.moments(model.evaluate(eta)) for eta in chain]
        ds.append([m[0] for m in moments]); cs.append([m[1] for m in moments])
    delta, covariance, _, _ = aggregate(np.asarray(ds), np.asarray(cs))
    full_covariance = np.asarray(posterior['posterior_covariance'])
    full_mean = np.asarray(posterior['posterior_mean'])
    mean_error = float(np.sqrt(np.mean((model.mean + delta - full_mean) ** 2 / np.diag(full_covariance))))
    variance_error = float(np.max(np.abs(np.diag(covariance) / np.diag(full_covariance) - 1)))
    passed = (variance_error <= limits['maximum_source_f_diagonal_variance_relative_difference_vs_all_draws']
        and mean_error <= limits['maximum_source_f_mean_rms_difference_in_full_posterior_sd'])
    return {'layout': field['layout'], 'arm': field['arms'][index], 'mixture_gate_passed': bool(passed),
        'maximum_source_f_variance_relative_difference': variance_error,
        'source_f_mean_rms_difference_in_full_posterior_sd': mean_error,
        'eta_states_by_chain': selected.tolist(), 'equal_positive_component_weight': 1 / limits['components_per_arm'],
        'single_Gaussian_not_substituted': True}


def run():
    started = time.process_time(); spec = read(PLAN); fields = learned_fields(spec)
    references = read(spec['references']); assert references['all_reference_density_identities_verified']
    assert read(spec['reference_results'])['original_population_priors_and_exact_likelihood_unchanged']
    old = read(spec['old_posterior_results']); assert old['all_monte_carlo_gates_passed']
    reference_by_layout = {r['layout']: r for r in references['rows']}
    out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    checks = [verify_learned(f, i, reference_by_layout[f['layout']]) for f in fields for i in range(2)]
    save(out / 'verification.json', {'execution': spec, 'checks': checks, 'all_checks_passed_before_sampling': True})
    save(out / 'frozen_inputs.json', {'execution': spec, 'rows': fields, 'new_covariance_reconditioned_here': True})
    rows, mixtures = [], []
    try:
        for field in fields:
            reference = reference_by_layout[field['layout']]
            for index in range(2):
                model = VarianceReferencePosterior(field['f_means'][index], field['f_covariances'][index],
                    field['batch_means'], field['sample_variances'])
                retained = []
                row = sample(field, index, spec, started, model, reference['reference_mean'], reference['reference_covariance'], retained)
                cm, cc = common_moments(field, index, np.asarray(row['posterior_mean']), np.asarray(row['posterior_covariance']))
                previous = next(r for r in old['rows'] if (r['layout'], r['arm']) == (row['layout'], row['arm']))
                old_cc = np.asarray(previous['common_posterior_covariance'])
                row.update(common_coordinate_system=field['source_common_coordinate_systems'][index],
                    common_posterior_mean=cm.tolist(), common_posterior_covariance=cc.tolist(),
                    common_variance_ratio_mean_vs_old_raw_posterior=float(np.mean(np.diag(cc) / np.diag(old_cc))),
                    common_mean_shift_rms_in_old_raw_posterior_sd=float(np.sqrt(np.mean((cm - previous['common_posterior_mean']) ** 2 / np.diag(old_cc)))))
                mixture = retained_mixture(field, index, model, retained[0], row, spec)
                rows.append(row); mixtures.append(mixture)
                print('learned covariance', row['layout'], row['arm'], 'posterior gate', row['monte_carlo_gate_passed'],
                    'Rhat', row['maximum_split_rhat'], 'ESS', row['minimum_autocorrelation_ess'],
                    'half variance', row['maximum_first_second_half_f_variance_relative_difference'],
                    'mixture gate', mixture['mixture_gate_passed'], flush=True)
                save(out / 'results.json', {'execution': spec, 'rows': rows, 'diagnostic_interpreted': False,
                    'all_four_source_updates_complete': len(rows) == 4, 'cpu_seconds': time.process_time() - started})
                save(out / 'mixtures.json', {'execution': spec, 'rows': mixtures, 'source_posterior_precision_pending': len(rows) < 4})
    except TimeoutError:
        save(out / 'results.json', {'execution': spec, 'rows': rows, 'diagnostic_interpreted': False,
            'all_four_source_updates_complete': False, 'stopped_at_cpu_budget': True, 'cpu_seconds': time.process_time() - started})
        return
    posterior_passed = all(r['monte_carlo_gate_passed'] for r in rows)
    mixture_passed = all(r['mixture_gate_passed'] for r in mixtures)
    save(out / 'mixtures.json', {'execution': spec, 'rows': mixtures, 'all_mixture_retention_gates_passed': mixture_passed,
        'all_source_posterior_precision_gates_passed': posterior_passed})
    save(out / 'results.json', {'execution': spec, 'rows': rows, 'all_four_source_updates_complete': True,
        'all_monte_carlo_gates_passed': posterior_passed, 'all_mixture_retention_gates_passed': mixture_passed,
        'diagnostic_interpreted': posterior_passed and mixture_passed, 'cpu_seconds': time.process_time() - started,
        'cost': {**spec['cost'], 'actual_chains': 16,
            'actual_sweeps': 16 * (spec['sampler']['warmup_sweeps_per_chain'] + spec['sampler']['retained_sweeps_per_chain']),
            'retained_positive_mixture_components_per_arm': spec['retained_mixture']['components_per_arm']},
        'raw_original_source_likelihood_factors_per_arm': {'batch_means': 48, 'sample_variances': 48},
        'additional_old32_Gaussian_risk_likelihood_factors': 0,
        'no_model_adopted': True, 'independent_confirmation': False,
        'interpretation': 'Fresh raw posterior under Stage205 covariance,with compensated variance reference. Retained positive eta components support conditional shared-latent reconstruction;aggregate moments are not a Gaussian paid-update substitute. One M step is not convergence or a held-task improvement.'})


if __name__ == '__main__':
    run()
