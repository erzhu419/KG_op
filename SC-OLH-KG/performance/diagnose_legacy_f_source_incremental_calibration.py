"""Conditional added-source innovations and effective estimated-noise influence."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_f_source_spectral_uncertainty import source_model
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.run_legacy_related_inventory_source import save
from representation.source_joint_spectral_geometry import JointSpectralGeometryLikelihood
from representation.source_kernel_population import rbf_cross, shared_kernel

PLAN = 'performance/manifests/legacy_joint_population_f_source_incremental_calibration_plan_v1.json'


def log_density(error, covariance):
    factor = cho_factor(covariance, lower=True); alpha = cho_solve(factor, error)
    return float(-.5 * (len(error) * np.log(2 * np.pi) + 2 * np.log(np.diag(factor[0])).sum() + error @ alpha))


def innovation(covariance, error, noise):
    old = np.concatenate([np.arange(24 * t, 24 * t + 12) for t in range(4)])
    new = np.concatenate([np.arange(24 * t + 12, 24 * t + 24) for t in range(4)])
    koo = covariance[np.ix_(old, old)]; factor = cho_factor(koo, lower=True)
    cross = covariance[np.ix_(new, old)]; shift = cross @ cho_solve(factor, error[old])
    residual = error[new] - shift; conditional = covariance[np.ix_(new, new)] - cross @ cho_solve(factor, cross.T)
    conditional = .5 * (conditional + conditional.T); factor_new = cho_factor(conditional, lower=True)
    inverse = cho_solve(factor_new, np.eye(48)); quadratic = float(residual @ inverse @ residual)
    conditional_density = log_density(residual, conditional)
    joint_difference = log_density(error, covariance) - log_density(error[old], koo)
    np.testing.assert_allclose(conditional_density, joint_difference, atol=1e-8, rtol=1e-8)
    noise_share = float(np.diag(inverse) @ noise[new] / 48)
    assert -1e-10 <= noise_share <= 1 + 1e-10
    standardized = residual / np.sqrt(np.diag(conditional))
    return {'conditional_log_density_per_added_label': conditional_density / 48,
        'joint_quadratic_per_added_label': quadratic / 48, 'individual_z_squared_mean': float(np.mean(standardized ** 2)),
        'individual_abs_z_max': float(np.max(np.abs(standardized))),
        'effective_measurement_noise_information_share': noise_share,
        'diagonal_noise_fraction_median': float(np.median(noise[new] / np.diag(conditional))),
        'diagonal_noise_fraction_maximum': float(np.max(noise[new] / np.diag(conditional))),
        'added_source_mean_residuals': residual.tolist(), 'individual_standardized_residuals': standardized.tolist(),
        'conditional_variance_diagonal': np.diag(conditional).tolist(),
        'conditional_joint_density_identity_error': abs(conditional_density - joint_difference)}


def run():
    started = time.process_time(); spec = read(PLAN); coverage_result = read(spec['coverage_results'])
    assert coverage_result['diagnostic_interpreted'] and not coverage_result['source_coverage_screening_passed']
    coverage_spec = read(spec['coverage_plan']); rosters = read(Path(coverage_spec['output']) / 'source_rosters.json')['rows']
    source_spec = read(spec['source_probe_plan']); registry = read(spec['source_gp_registry'])['rows']
    coverage, _ = source_paths(read(spec['source_context_execution'])); raw = read(coverage['source_data']); rows = []
    query = spec['source_validation_task']
    for roster in rosters:
        layout = roster['layout']; model, law, public = source_model(source_spec, layout, query); tasks = []
        assert roster['source_task_names'] == law['source_task_names'] and query not in roster['source_ids'] and 1 not in roster['source_ids']
        for q, sites, name in zip(roster['source_ids'], roster['policy_indices_by_source'], law['source_task_names']):
            values = np.asarray([[v[0] for v in raw['tasks'][q]['Y_replicates'][j]] for j in sites]); assert values.shape == (24, 3)
            tasks.append({'name': name, 'psi': public[sites], 'features': public[sites], 'response': values.mean(axis=1)[:, None],
                'noise': (values.var(axis=1, ddof=1) / 3)[:, None]})
        expanded = JointSpectralGeometryLikelihood(model.basis, tasks, *model.templates, model.columns, model.center, model.white, .01)
        nll, gradient, state = expanded.evaluate(np.asarray(law['parameters']))
        for k in ('common_prior_coefficient_covariance', 'task_coefficient_covariance'):
            np.testing.assert_allclose(state[k], law[k], atol=1e-10, rtol=1e-10)
        d = expanded.design; a, r = state['common_prior_coefficient_covariance'], state['task_coefficient_covariance']
        structure_covariance = d @ a @ d.T + block_diag(*[z @ r @ z.T for z in expanded.designs]) + np.diag(expanded.noise)
        noise = np.concatenate([t['noise'][:, 0] for t in tasks])
        structural = innovation(structure_covariance, expanded.error, noise)
        gp = read(next(r['source_fit'] for r in registry if (r['layout'], r['inner_query_task']) == (layout, f'related_inventory_{query:02d}')))['fits']['gp']
        anchor = gp['anchor']; scale2 = anchor['normalization']['scale'] ** 2
        center = anchor['normalization']['mean'] + anchor['normalization']['scale'] * anchor['constant']
        amplitude, intercept, white = gp['residual_parameters']; xs = np.vstack([t['features'] for t in tasks])
        gp_covariance = scale2 * shared_kernel(xs, xs, anchor) + block_diag(*[
            scale2 * (amplitude * rbf_cross(t['features'], t['features'], anchor['lengthscale']) + intercept)
            + np.diag(t['noise'][:, 0] + scale2 * white) for t in tasks])
        gp_error = np.concatenate([t['response'][:, 0] for t in tasks]) - center
        generic = innovation(gp_covariance, gp_error, noise)
        np.testing.assert_allclose(nll * 96, -log_density(expanded.error, structure_covariance), atol=1e-8, rtol=1e-8)
        rows.append({'layout': layout, 'source_task_names': law['source_task_names'], 'sparse_joint': structural, 'same_source_GP': generic,
            'old_parameters': law['parameters'], 'source96_spectral_gradient_per_label_at_old_parameters': gradient.tolist(),
            'source96_total_gradient_at_old_parameters': (96 * gradient).tolist()})
        print('added source innovations', layout, [(name, value['joint_quadratic_per_added_label'], value['effective_measurement_noise_information_share'])
            for name, value in (('structure', structural), ('GP', generic))], 'source gradient', gradient.tolist(), flush=True)
        assert time.process_time() - started < spec['maximum_single_thread_cpu_seconds']
    out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    save(out / 'results.json', {'execution': spec, 'rows': rows, 'diagnostic_interpreted': True, 'cpu_seconds': time.process_time() - started,
        'cost': spec['cost'], 'no_model_adopted': True, 'validation_task0_or_task1_outcomes_indexed': False, 'source6_7_evaluated': False,
        'independent_confirmation': False, 'interpretation': 'Added-source conditional Gaussian innovations under old empirical laws and frozen plug-in noises. Joint calibration/noise information summaries are not chi-square significance tests or proof of a specific variance error. New gradients motivate source training design only;no optimization or held-source decision.'})


if __name__ == '__main__':
    run()
