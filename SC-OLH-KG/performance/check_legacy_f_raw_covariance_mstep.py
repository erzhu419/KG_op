"""One source covariance M step from verified raw posterior moments."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_f_source_spectral_uncertainty import source_model
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_raw_source_covariance_score import projected
from performance.run_legacy_related_inventory_source import save
from representation.source_kernel_population import rbf_cross

PLAN = 'performance/manifests/legacy_joint_population_f_raw_covariance_mstep_plan_v1.json'


def covariance_systems(spec, field):
    mean_spec = read(spec['source_mean_plan']); layout = field['layout']
    model, law, _ = source_model(mean_spec, layout, 0)
    registry = read(mean_spec['source_registry'])['rows']
    gp = read(next(r['source_fit'] for r in registry if (r['layout'], r['inner_query_task']) ==
        (layout, 'related_inventory_00')))['fits']['gp']
    xs = np.asarray(gp['source_features']); anchor = gp['anchor']; scale2 = anchor['normalization']['scale'] ** 2
    local = [xs[12 * i:12 * (i + 1)] for i in range(4)]
    gp_directions = [scale2 * rbf_cross(xs, xs, anchor['lengthscale']),
        scale2 * block_diag(*[rbf_cross(x, x, anchor['lengthscale']) for x in local]),
        scale2 * block_diag(*[np.ones((12, 12)) for _ in local]), scale2 * np.eye(48)]
    gp_fixed = scale2 * anchor['constant_variance'] * np.ones((48, 48))

    def structure(theta):
        a, ca, cp = model.covariance(0, theta[0], theta[1]); r, ra, rp = model.covariance(1, theta[2], theta[3])
        d = model.design
        covariance = d @ a @ d.T + block_diag(*[dt @ r @ dt.T + model.white * np.eye(12) for dt in model.designs])
        directions = [d @ ca @ d.T, d @ cp @ d.T,
            block_diag(*[dt @ ra @ dt.T for dt in model.designs]), block_diag(*[dt @ rp @ dt.T for dt in model.designs])]
        state = {'common_coefficient_covariance': a, 'task_coefficient_covariance': r,
            'common_latent_prior_covariance': a, 'common_latent_to_source_f_covariance': a @ d.T,
            'original_positive_tail_per_coefficient': model.tails, 'original_support_columns': model.columns.tolist(),
            'orthogonal_basis_and_order_unchanged': True, 'original_intercept_and_white_unchanged': True}
        return covariance, directions, state

    def generic(theta):
        common = gp_fixed + theta[0] * gp_directions[0]
        covariance = common + sum(theta[j] * gp_directions[j] for j in range(1, 4))
        return covariance, gp_directions, {'common_latent_prior_covariance': common,
            'common_latent_to_source_f_covariance': common, 'original_lengthscale_center_and_common_intercept_unchanged': True}

    return [(structure, np.asarray(law['parameters']), np.asarray([np.inf, 4., np.inf, 4.])),
        (generic, np.r_[anchor['signal_variance'], gp['residual_parameters']], np.full(4, np.inf))]


def sufficient_moment(row, prior_mean, prefix=''):
    mean = np.asarray(row[prefix + 'posterior_mean']); covariance = np.asarray(row[prefix + 'posterior_covariance'])
    delta = mean - prior_mean
    return covariance + np.outer(delta, delta)


def expected_objective(theta, builder, second_moment):
    covariance, directions, _ = builder(theta)
    try:
        factor = cho_factor(covariance, lower=True)
    except np.linalg.LinAlgError:
        # Allowed GP zero-amplitude boundaries can have singular latent laws.
        return np.inf, np.zeros(4)
    inverse = cho_solve(factor, np.eye(48)); solved = inverse @ second_moment
    contrast = .5 * (inverse - solved @ inverse) / 48
    objective = .5 * (48 * np.log(2 * np.pi) + 2 * np.log(np.diag(factor[0])).sum() + np.trace(solved)) / 48
    gradient = np.asarray([np.sum(contrast * direction.T) for direction in directions])
    return float(objective), gradient


def serialize(state):
    return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in state.items()}


def run():
    started = time.process_time(); spec = read(PLAN); source = read(spec['source_posterior_results']); scores = read(spec['score_results'])
    assert source['all_monte_carlo_gates_passed'] and scores['all_analytic_plugin_score_identities_passed']
    fields = read(spec['source_inputs'])['rows']; out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    rows, states = [], []
    for field in fields:
        systems = covariance_systems(spec, field)
        for index, (builder, theta, upper) in enumerate(systems):
            arm = field['arms'][index]; row = next(r for r in source['rows'] if (r['layout'], r['arm']) == (field['layout'], arm))
            score_row = next(r for r in scores['rows'] if (r['layout'], r['arm']) == (field['layout'], arm))
            prior_mean = np.asarray(field['f_means'][index]); second = sufficient_moment(row, prior_mean)
            halves = [sufficient_moment(row, prior_mean, prefix) for prefix in ('first_half_', 'second_half_')]
            old_covariance, _, _ = builder(theta)
            np.testing.assert_allclose(old_covariance, field['f_covariances'][index], atol=1e-10, rtol=1e-10)
            old_value, old_gradient = expected_objective(theta, builder, second)
            np.testing.assert_allclose(old_gradient, score_row['raw_source_NLL_gradient_per_batch'], atol=1e-8, rtol=1e-8)
            scale = np.where(theta > 0, theta, 1.)
            def objective(z):
                if time.process_time() - started >= spec['maximum_single_thread_cpu_seconds']: raise TimeoutError('registered CPU budget reached')
                value, gradient = expected_objective(z * scale, builder, second)
                return value, gradient * scale
            settings = spec['optimizer']
            result = minimize(objective, theta / scale, method=settings['method'], jac=True,
                bounds=list(zip(np.zeros(4), upper / scale)), options={'maxiter': settings['max_iterations'],
                    'ftol': settings['ftol'], 'gtol': settings['gtol']})
            learned = result.x * scale; value, gradient = expected_objective(learned, builder, second)
            covariance, _, state = builder(learned); np.linalg.cholesky(covariance)
            old_half = [expected_objective(theta, builder, s)[0] for s in halves]
            learned_half = [expected_objective(learned, builder, s)[0] for s in halves]
            improvements = [old_value - value, *[a - b for a, b in zip(old_half, learned_half)]]
            passed = bool(result.success and min(improvements) > 0)
            if index == 0:
                assert np.all(learned[[1, 3]] <= 4.) and np.all(learned >= 0)
                assert all(t > 0 for t in state['original_positive_tail_per_coefficient'])
                for key in ('common_coefficient_covariance', 'task_coefficient_covariance'):
                    assert np.linalg.eigvalsh(state[key]).min() >= -1e-12
            rows.append({'layout': field['layout'], 'arm': arm, 'parameter_names': score_row['parameter_names'],
                'old_parameters': theta.tolist(), 'learned_parameters': learned.tolist(), 'optimization_parameter_scales': scale.tolist(),
                'optimizer_converged': bool(result.success), 'optimizer_message': str(result.message),
                'optimizer_iterations': int(result.nit), 'optimizer_function_evaluations': int(result.nfev),
                'old_expected_complete_NLL_per_batch': old_value, 'learned_expected_complete_NLL_per_batch': value,
                'old_split_expected_NLL': old_half, 'learned_split_expected_NLL': learned_half,
                'full_and_split_Q_improvements': improvements, 'candidate_gate_passed': passed,
                'original_parameter_projected_gradient': projected(learned, gradient, upper).tolist(),
                'optimization_coordinate_projected_gradient': projected(result.x, gradient * scale, upper / scale).tolist(),
                'initial_raw_score_identity_max_error': float(np.max(np.abs(old_gradient - score_row['raw_source_NLL_gradient_per_batch']))),
                'source_prior_trace_ratio': float(np.trace(covariance) / np.trace(old_covariance))})
            states.append({'layout': field['layout'], 'arm': arm, 'learned_parameters': learned.tolist(),
                'source_f_population_covariance': covariance.tolist(), **serialize(state), 'no_new_posterior_attached': True})
            print('source covariance M step', field['layout'], arm, 'pass', passed, 'iterations', result.nit,
                'parameters', learned.tolist(), 'Q improvements', improvements, flush=True)
            save(out / 'results.json', {'execution': spec, 'rows': rows, 'all_four_optimizations_complete': len(rows) == 4,
                'diagnostic_interpreted': False, 'cpu_seconds': time.process_time() - started})
    passed = all(r['candidate_gate_passed'] for r in rows)
    save(out / 'source_laws.json', {'execution': spec, 'rows': states, 'all_four_source_Mstep_gates_passed': passed,
        'posterior_moments_not_reused_as_new_covariance_posterior': True})
    save(out / 'results.json', {'execution': spec, 'rows': rows, 'all_four_optimizations_complete': True,
        'all_source_Mstep_gates_passed': passed, 'diagnostic_interpreted': True, 'cpu_seconds': time.process_time() - started,
        'cost': {**spec['cost'], 'actual_new_covariance_Mstep_optimizations': 4},
        'no_model_adopted': True, 'independent_confirmation': False,
        'interpretation': 'One Monte Carlo EM source covariance M step using original48-source raw posterior sufficient moments. Q decreases do not prove marginal raw likelihood or held-source decision improvement. Original structural constraints and matched source information retained;new covariance requires fresh source raw conditioning.'})


if __name__ == '__main__':
    run()
