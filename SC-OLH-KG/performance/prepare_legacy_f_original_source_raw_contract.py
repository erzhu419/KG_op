"""Restore original source population priors before raw-batch conditioning."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_diverse_source_design import encode_diverse_source
from performance.check_legacy_f_source_spectral_uncertainty import source_model
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_source_incremental_calibration import log_density
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.run_legacy_related_inventory_source import problem_for, save
from representation.legacy_observable_prior import legacy_exposure_descriptors
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.source_kernel_population import rbf_cross, shared_kernel

PLAN = 'performance/manifests/legacy_joint_population_f_original_source_raw_contract_plan_v1.json'


def common_replay(mean, a, d, residual, labels, noise):
    covariance = d @ a @ d.T + residual + np.diag(noise); factor = cho_factor(covariance, lower=True)
    error = labels - d @ mean; cross = a @ d.T
    beta = mean + cross @ cho_solve(factor, error)
    h = a - cross @ cho_solve(factor, cross.T)
    return beta, .5 * (h + h.T), -log_density(error, covariance) / len(labels)


def run():
    started = time.process_time(); spec = read(PLAN); collapsed = read(spec['collapsed_results'])
    assert collapsed['all_monte_carlo_gates_passed'] and collapsed['diagnostic_interpreted']
    known = read(spec['known_source_plan']); source_spec = read(spec['source_mean_plan'])
    coverage, _ = source_paths(read(known['source_context_execution']))
    public = read(Path(coverage['source_data']).with_name('frozen_inputs.json')); bank = [tuple(p) for p in public['bank']]
    raw = read(coverage['source_data']); data = {'bank': raw['bank'], 'tasks': {i: raw['tasks'][i] for i in spec['source_tasks']}}; del raw
    descriptors = {i: legacy_exposure_descriptors(no_outcomes(problem_for(public['plan'], public['tasks'][i])), bank, 50)
        for i in spec['source_tasks']}
    registry = read(source_spec['source_registry'])['rows']; risk_execution = read(known['risk_source_execution'])
    fields, checks = [], []
    for layout in spec['layouts']:
        model, law, psi = source_model(source_spec, layout, 0)
        assert law['source_task_names'] == [f'related_inventory_{i:02d}' for i in spec['source_tasks']]
        mean_inputs = read(Path(coverage['output']) / f'layout{layout}' / 'task1' / 'inputs.json')
        mean_tasks = [t for t in mean_inputs['source_tasks'] if t['name'] in law['source_task_names']]
        sites = [coverage['layouts'][layout][i] for i in spec['source_tasks']]
        values = np.concatenate([np.asarray([[v[0] for v in data['tasks'][i]['Y_replicates'][j]] for j in s])
            for i, s in zip(spec['source_tasks'], sites)])
        assert values.shape == (48, 3); labels, s2 = values.mean(axis=1), values.var(axis=1, ddof=1)
        np.testing.assert_allclose(labels, np.concatenate([np.asarray(t['response'])[:, 0] for t in mean_tasks]), atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(s2 / 3, np.concatenate([np.asarray(t['noise'])[:, 0] for t in mean_tasks]), atol=1e-8, rtol=1e-8)
        a = np.asarray(law['common_prior_coefficient_covariance']); r = np.asarray(law['task_coefficient_covariance']); d = model.design
        residual = block_diag(*[dt @ r @ dt.T + model.white * np.eye(12) for dt in model.designs])
        f_prior = d @ a @ d.T + residual; m0 = np.r_[model.center, np.zeros(model.basis.feature_dim)]
        beta, h, nll = common_replay(m0, a, d, residual, labels, s2 / 3)
        np.testing.assert_allclose(beta, law['source_posterior_coefficient_mean'], atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(h, law['source_posterior_common_covariance'], atol=1e-8, rtol=1e-8)
        expected_nll = model.evaluate(np.asarray(law['parameters']))[0]
        np.testing.assert_allclose(nll, expected_nll, atol=1e-8, rtol=1e-8)
        gp = read(next(row['source_fit'] for row in registry if (row['layout'], row['inner_query_task']) ==
            (layout, 'related_inventory_00')))['fits']['gp']
        anchor = gp['anchor']; scale2 = anchor['normalization']['scale'] ** 2
        amplitude, intercept, white = gp['residual_parameters']; xs = [np.asarray(t['features']) for t in mean_tasks]
        all_x = np.vstack(xs); gp_common = scale2 * shared_kernel(all_x, all_x, anchor)
        gp_prior = gp_common + block_diag(*[scale2 * (amplitude * rbf_cross(x, x, anchor['lengthscale'])
            + intercept + white * np.eye(12)) for x in xs])
        gp_nll = -log_density(labels - model.center, gp_prior + np.diag(s2 / 3)) / 48
        gp_physical_nll = gp['source_negative_log_likelihood_per_label'] + np.log(anchor['normalization']['scale'])
        np.testing.assert_allclose(gp_nll, gp_physical_nll, atol=1e-8, rtol=1e-8)
        case = next(c for c in risk_execution['cases'] if (c['layout'], c['query_task']) == (layout, 0))
        folder = Path(risk_execution['output']) / case['folder']; risk_inputs = read(folder / 'risk_inputs.json')
        fits = read(folder / 'risk_source_fit.json')['fits']; risk = fits['joint_structural']
        assert risk_inputs['training_task_ids'] == spec['source_tasks']
        coordinate, tasks, _, _ = encode_diverse_source(data, spec['source_tasks'], risk_inputs['source_site_indices_by_task'],
            descriptors, public['plan']['task_distribution']['L'])
        reference = coordinate.transform(descriptors[spec['source_tasks'][0]])
        np.testing.assert_allclose(reference, risk_inputs['public_reference_psi'], atol=1e-8, rtol=1e-8)
        basis = OrthogonalLowOrderBasis(reference); ds = []; original_ds = []
        for i, t, s, saved in zip(spec['source_tasks'], tasks, sites, risk_inputs['source_tasks']):
            for key in ('psi', 'response', 'noise'): np.testing.assert_allclose(t[key], saved[key], atol=1e-8, rtol=1e-8)
            ds.append(np.column_stack([np.ones(12), basis.transform(coordinate.transform(descriptors[i][s]))]))
            original_ds.append(np.column_stack([np.ones(8), basis.transform(t['psi'])]))
        eta_m0 = np.asarray(fits['regularizer']['coefficient_mean']); eta_a = np.diag(risk['common_prior_coefficient_diagonal'])
        eta_r = np.diag(risk['residual_coefficient_diagonal']); eta_white = risk['discrepancy_variance']
        eta_d = np.vstack(ds); eta_prior = eta_d @ eta_a @ eta_d.T + block_diag(*[
            dt @ eta_r @ dt.T + eta_white * np.eye(12) for dt in ds])
        # This replay checks the parameter restoration;its posterior is not the raw prior.
        old_d = np.vstack(original_ds); old_residual = block_diag(*[
            dt @ eta_r @ dt.T + eta_white * np.eye(8) for dt in original_ds])
        eb, eh, enll = common_replay(eta_m0, eta_a, old_d, old_residual,
            np.concatenate([t['response'][:, 3] for t in tasks]), np.concatenate([t['noise'][:, 3] for t in tasks]))
        np.testing.assert_allclose(eb, risk['coefficient_mean'], atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(eh, risk['population_mean_covariance'], atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(enll, risk['source_negative_log_likelihood_per_label'], atol=1e-8, rtol=1e-8)
        for covariance in (f_prior, gp_prior, eta_prior): np.linalg.cholesky(covariance)
        fields.append({'layout': layout, 'sites': [{'source_task': i, 'policy': j} for i, s in zip(spec['source_tasks'], sites) for j in s],
            'arms': ['joint_spectral_geometry', 'source_gp'], 'f_means': [np.full(48, model.center).tolist()] * 2,
            'f_covariances': [f_prior.tolist(), gp_prior.tolist()], 'eta_mean': (eta_d @ eta_m0).tolist(),
            'eta_covariance': eta_prior.tolist(), 'batch_means': labels.tolist(), 'sample_variances': s2.tolist(),
            'common_latent_prior_means': [m0.tolist(), np.full(48, model.center).tolist()],
            'common_latent_prior_covariances': [a.tolist(), gp_common.tolist()],
            'common_latent_to_source_f_covariances': [(a @ d.T).tolist(), gp_common.tolist()],
            'source_common_coordinate_systems': ['orthogonal_coefficients_intercept_and_modes', 'GP_common_values_at48_source_sites'],
            'risk_population_prior_center_and_covariance_used': True, 'no_original_risk_posterior_as_prior': True})
        checks.append({'layout': layout, 'source_roster': spec['source_tasks'], 'source_label_and_map_reconstruction_passed': True,
            'f_plugin_common_beta_H_and_NLL_replayed': True, 'GP_plugin_NLL_replayed': True,
            'GP_stored_NLL_measure': 'normalized f units', 'GP_replayed_NLL_measure': 'physical f units',
            'GP_NLL_log_scale_Jacobian': float(np.log(anchor['normalization']['scale'])),
            'risk_original32_Gaussian_beta_H_and_NLL_replayed_for_verification_only': True,
            'all_original48_priors_positive': True, 'passed': True})
        print('original source priors', layout, 'eta SD range', np.sqrt(np.diag(eta_prior)).min(), np.sqrt(np.diag(eta_prior)).max(), flush=True)
        assert time.process_time() - started < spec['maximum_single_thread_cpu_seconds']
    out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    save(out / 'frozen_inputs.json', {'execution': spec, 'rows': fields, 'all_population_priors_restored': True,
        'original_variances_not_conditioned_in_prior': True, 'covariance_parameters_and_encoder_are_source_empirical_estimates': True})
    save(out / 'results.json', {'execution': spec, 'checks': checks, 'diagnostic_interpreted': True,
        'all_original48_raw_contracts_complete': True, 'cpu_seconds': time.process_time() - started, 'cost': spec['cost'],
        'original32_risk_Gaussian_label_likelihood_added_to_raw_update': False,
        'validation_task0_or_task1_outcomes_indexed': False, 'no_model_adopted': True, 'independent_confirmation': False,
        'interpretation': 'Unconditioned population priors at original48 source sites,with parameters/coordinate fixed at old source empirical estimates. Raw batch likelihood consumes each original mean and S^2 once;no old risk posterior reused as prior. Hyperparameter/encoder uncertainty and covariance relearning remain outside this repair.'})


if __name__ == '__main__':
    run()
