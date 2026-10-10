"""Raw source covariance score from posterior sufficient moments."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve, solve_triangular

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_f_source_spectral_uncertainty import source_model
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.run_legacy_related_inventory_source import save
from representation.source_kernel_population import rbf_cross

PLAN = 'performance/manifests/legacy_joint_population_f_raw_source_covariance_score_plan_v1.json'


def whiten(root, matrix):
    left = solve_triangular(root, matrix, lower=True)
    return solve_triangular(root, left.T, lower=True).T


def score(prior_mean, prior_covariance, directions, posterior_mean, posterior_covariance):
    root = np.linalg.cholesky(prior_covariance)
    delta = solve_triangular(root, posterior_mean - prior_mean, lower=True)
    second = whiten(root, posterior_covariance) + np.outer(delta, delta)
    contrast = .5 * (np.eye(len(delta)) - second) / len(delta)
    return np.asarray([np.sum(contrast * whiten(root, direction).T) for direction in directions])


def projected(parameters, gradient, upper):
    return np.where(((parameters <= 0) & (gradient > 0)) | ((parameters >= upper) & (gradient < 0)), 0., gradient)


def run():
    started = time.process_time(); spec = read(PLAN); source = read(spec['source_results'])
    assert source['diagnostic_interpreted'] and source['all_monte_carlo_gates_passed']
    fields = read(spec['source_inputs'])['rows']; mean_spec = read(spec['source_mean_plan']); registry = read(mean_spec['source_registry'])['rows']; rows = []
    for field in fields:
        layout = field['layout']; model, law, _ = source_model(mean_spec, layout, 0); parameters = np.asarray(law['parameters'])
        _, ca, cp = model.covariance(0, parameters[0], parameters[1])
        _, ra, rp = model.covariance(1, parameters[2], parameters[3]); d = model.design
        structure_directions = [d @ ca @ d.T, d @ cp @ d.T,
            block_diag(*[dt @ ra @ dt.T for dt in model.designs]), block_diag(*[dt @ rp @ dt.T for dt in model.designs])]
        gp = read(next(r['source_fit'] for r in registry if (r['layout'], r['inner_query_task']) ==
            (layout, 'related_inventory_00')))['fits']['gp']
        xs = np.asarray(gp['source_features']); anchor = gp['anchor']; scale2 = anchor['normalization']['scale'] ** 2
        local = [xs[i * 12:(i + 1) * 12] for i in range(4)]
        generic_directions = [scale2 * rbf_cross(xs, xs, anchor['lengthscale']),
            scale2 * block_diag(*[rbf_cross(x, x, anchor['lengthscale']) for x in local]),
            scale2 * block_diag(*[np.ones((12, 12)) for _ in local]), scale2 * np.eye(48)]
        for index, directions in enumerate((structure_directions, generic_directions)):
            row = next(r for r in source['rows'] if (r['layout'], r['arm']) == (layout, field['arms'][index]))
            prior_mean = np.asarray(field['f_means'][index]); covariance = np.asarray(field['f_covariances'][index])
            def from_row(prefix):
                return score(prior_mean, covariance, directions, np.asarray(row[prefix + 'posterior_mean']),
                    np.asarray(row[prefix + 'posterior_covariance']))
            plugin = from_row('plug_in_'); raw = from_row(''); first = from_row('first_half_'); second = from_row('second_half_')
            if index == 0:
                direct = model.evaluate(parameters)[1]; theta = parameters; upper = np.asarray([np.inf, 4., np.inf, 4.])
                names = ['common_root_amplitude', 'common_spectral_exponent', 'task_root_amplitude', 'task_spectral_exponent']
            else:
                factor = cho_factor(covariance + np.diag(np.asarray(field['sample_variances']) / 3), lower=True)
                alpha = cho_solve(factor, np.asarray(field['batch_means']) - prior_mean)
                direct = np.asarray([.5 * (np.trace(cho_solve(factor, direction)) - alpha @ direction @ alpha) / 48 for direction in directions])
                theta = np.r_[anchor['signal_variance'], gp['residual_parameters']]; upper = np.full(4, np.inf)
                names = ['common_RBF_amplitude', 'task_RBF_amplitude', 'task_intercept_variance', 'white_discrepancy']
            np.testing.assert_allclose(plugin, direct, atol=1e-8, rtol=1e-8)
            raw_projected, first_projected, second_projected, old_projected = [projected(theta, value, upper) for value in (raw, first, second, plugin)]
            rows.append({'layout': layout, 'arm': row['arm'], 'parameter_names': names, 'frozen_parameters': theta.tolist(),
                'plug_in_source_NLL_gradient_per_batch': plugin.tolist(), 'raw_source_NLL_gradient_per_batch': raw.tolist(),
                'first_half_raw_gradient': first.tolist(), 'second_half_raw_gradient': second.tolist(),
                'plug_in_projected_gradient': old_projected.tolist(), 'raw_projected_gradient': raw_projected.tolist(),
                'first_half_raw_projected_gradient': first_projected.tolist(), 'second_half_raw_projected_gradient': second_projected.tolist(),
                'projected_split_direction_agreement': ((first_projected * second_projected > 0) | ((first_projected == 0) & (second_projected == 0))).tolist(),
                'plugin_score_identity_max_absolute_error': float(np.max(np.abs(plugin - direct))),
                'raw_projected_gradient_max_absolute': float(np.max(np.abs(raw_projected))),
                'old_projected_gradient_max_absolute': float(np.max(np.abs(old_projected)))})
            print('raw covariance score', layout, row['arm'], 'old', old_projected.tolist(), 'raw', raw_projected.tolist(),
                'split', first_projected.tolist(), second_projected.tolist(), flush=True)
        assert time.process_time() - started < spec['maximum_single_thread_cpu_seconds']
    out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    save(out / 'results.json', {'execution': spec, 'rows': rows, 'diagnostic_interpreted': True,
        'all_analytic_plugin_score_identities_passed': True, 'cpu_seconds': time.process_time() - started, 'cost': spec['cost'],
        'new_covariance_optimizations': 0, 'no_model_adopted': True, 'independent_confirmation': False,
        'interpretation': 'Raw marginal source covariance NLL gradients from verified posterior moment scores,at the original empirical parameters. Split summaries assess direction stability,not statistical significance. Covariance templates/means/risk priors unchanged;no held-source or decision scores.'})


if __name__ == '__main__':
    run()
