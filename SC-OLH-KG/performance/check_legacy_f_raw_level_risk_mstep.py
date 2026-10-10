"""One constrained raw-source M step for function level and risk parameters."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.special import digamma, polygamma

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_raw_source_covariance_score import projected, whiten
from performance.diagnose_legacy_f_raw_source_level_risk_score import source_systems, moment_gradient, verify_gaussian
from performance.run_legacy_related_inventory_source import save

PLAN = 'performance/manifests/legacy_joint_population_f_raw_level_risk_mstep_plan_v1.json'


def risk_system(field, risk, tail_budget):
    a = np.asarray(risk['common_prior_covariance']); d = np.asarray(risk['source_design'])
    np.testing.assert_allclose(a, np.diag(np.diag(a)), atol=0, rtol=0)
    tail = tail_budget / (len(a) - 1)
    strong = np.diag(np.r_[0., np.diag(a)[1:] - tail])
    assert np.min(np.diag(strong)) >= 0 and tail > 0
    fixed = a - strong; direction = d @ strong @ d.T
    base = np.asarray(field['eta_covariance']); center = np.asarray(field['eta_mean'])

    def builder(theta):
        return center + theta[0], base + (theta[1] - 1) * direction, [direction]

    return builder, strong, fixed, tail


def expected(theta, builder, mean, covariance):
    prior_mean, prior_covariance, directions = builder(theta)
    try:
        factor = cho_factor(prior_covariance, lower=True)
    except np.linalg.LinAlgError:
        return np.inf, np.zeros(len(theta))
    root = np.tril(factor[0]); delta = solve_triangular(root, mean - prior_mean, lower=True)
    second = whiten(root, covariance) + np.outer(delta, delta)
    q = .5 * (48 * np.log(2 * np.pi) + 2 * np.log(np.diag(root)).sum() + np.trace(second)) / 48
    contrast = .5 * (np.eye(48) - second) / 48
    center = solve_triangular(root, np.ones(48), lower=True)
    grad = np.r_[-center @ delta / 48, [np.sum(contrast * whiten(root, dk).T) for dk in directions]]
    return float(q), grad


def source_moments(source, partition):
    fm = [(np.asarray(source[p + 'posterior_mean']), np.asarray(source[p + 'posterior_covariance']))
        for p in ('', 'first_half_', 'second_half_')]
    rm = [(np.asarray(source['eta_posterior_mean']), np.asarray(source['eta_posterior_covariance']))]
    rm += [(np.asarray(partition[p + '_eta_mean']), np.asarray(partition[p + '_eta_covariance']))
        for p in ('first_half', 'second_half')]
    return fm, rm


def run():
    started = time.process_time(); spec = read(PLAN); level = read(spec['level_risk_plan'])
    source = read(spec['source_posterior_results']); assert source['all_monte_carlo_gates_passed']
    score_check = read(spec['risk_score_results'])
    assert score_check['all_derivative_checks_passed'] and score_check['all_original_source_replays_matched']
    fields = read(spec['source_inputs'])['rows']; partitions = read(spec['risk_partitions'])['rows']
    old_laws = read(level['source_laws'])['rows']; rows, laws, risks, checks = [], [], [], []
    out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    for field in fields:
        systems, _, risk = source_systems(level, field)
        rb, strong, fixed, tail = risk_system(field, risk, spec['risk_original_shape_tail_budget'])
        rc = verify_gaussian(rb, np.asarray([0., 1.]), np.log(field['sample_variances']) - digamma(1.),
            np.full(48, float(polygamma(1, 1.))))
        for index, (full_builder, old_f, _) in enumerate(systems):
            arm = field['arms'][index]; key = (field['layout'], arm)
            previous = next(r for r in source['rows'] if (r['layout'], r['arm']) == key)
            partition = next(r for r in partitions if (r['layout'], r['arm']) == key)
            fm, rm = source_moments(previous, partition)

            def fb(theta):
                m, k, ds = full_builder(np.asarray([theta[0], old_f[1], theta[1]]))
                return m, k, [ds[1]]

            theta = np.asarray([old_f[0], old_f[2], 0., 1.])
            np.testing.assert_allclose(fb(theta[:2])[1], field['f_covariances'][index], atol=1e-10, rtol=1e-10)
            np.testing.assert_allclose(rb(theta[2:])[1], field['eta_covariance'], atol=1e-10, rtol=1e-10)
            fc = verify_gaussian(fb, theta[:2], np.asarray(field['batch_means']), np.asarray(field['sample_variances']) / 3)
            risk_scores = [moment_gradient(*rb(theta[2:]), *p) for p in rm]
            assert all(g[0] > 0 and g[1] > 0 for g in risk_scores)

            def q(t, half=0):
                f, fg = expected(t[:2], fb, *fm[half]); r, rg = expected(t[2:], rb, *rm[half])
                return f + r, np.r_[fg, rg], [f, r]

            old_value, old_gradient, old_components = q(theta)
            sd = np.sqrt(np.mean(np.diag(field['f_covariances'][index])))
            scale = np.asarray([sd, 1. if index == 0 else sd ** 2, 1., 1.])
            bounds = [(None, None), (0., None), (None, None), (0., None)]

            def objective(z):
                if time.process_time() - started >= spec['maximum_single_thread_cpu_seconds']:
                    raise TimeoutError('registered CPU budget reached')
                v, g, _ = q(z * scale); return v, g * scale

            opt = spec['optimizer']
            result = minimize(objective, theta / scale, method=opt['method'], jac=True, bounds=bounds,
                options={'maxiter': opt['max_iterations'], 'ftol': opt['ftol'], 'gtol': opt['gtol']})
            learned = result.x * scale; value, gradient, components = q(learned)
            derivative_errors = []
            for j in range(4):
                h = 1e-5 * scale[j]; t = learned.copy()
                if j in (1, 3) and learned[j] < h:
                    t[j] += h; first = q(t)[0]; t[j] += h
                    finite = (-3 * value + 4 * first - q(t)[0]) / (2 * h)
                else:
                    t[j] += h; upper = q(t)[0]; t[j] -= 2 * h
                    finite = (upper - q(t)[0]) / (2 * h)
                np.testing.assert_allclose(finite, gradient[j], atol=1e-6, rtol=1e-4)
                derivative_errors.append(abs(finite - gradient[j]))
            old_splits = [q(theta, h)[2] for h in range(3)]
            new_splits = [q(learned, h)[2] for h in range(3)]
            improvements = np.asarray(old_splits) - np.asarray(new_splits)
            passed = bool(result.success and np.all(improvements.sum(axis=1) > 0) and improvements.min() >= -1e-10)
            kf = fb(learned[:2])[1]; kr = rb(learned[2:])[1]
            np.linalg.cholesky(kf); np.linalg.cholesky(kr)
            old_law = next(r for r in old_laws if (r['layout'], r['arm']) == key)
            law = dict(old_law)
            if index == 0:
                a = np.asarray(old_law['common_coefficient_covariance']); r = np.asarray(old_law['task_coefficient_covariance'])
                b = np.eye(len(r)); b[0, 0] = learned[1]; r = b @ r @ b
                law.update(task_coefficient_covariance=r.tolist(), original_intercept_and_white_unchanged=False,
                    common_intercept_variance_unchanged=True, task_intercept_root_multiplier=float(learned[1]))
            else:
                # Retain the existing GP prediction interface's normalized variance units.
                old_gp_task = systems[1][1][2]
                mean_spec = read(level['source_mean_plan']); registry = read(mean_spec['source_registry'])['rows']
                gp = read(next(r['source_fit'] for r in registry if (r['layout'], r['inner_query_task']) ==
                    (field['layout'], 'related_inventory_00')))['fits']['gp']
                gp_parameters = list(old_law['learned_parameters']); gp_parameters[2] = float(learned[1] / gp['anchor']['normalization']['scale'] ** 2)
                law.update(learned_parameters=gp_parameters, original_lengthscale_center_and_common_intercept_unchanged=False,
                    common_intercept_variance_unchanged=True, old_physical_task_constant_variance=float(old_gp_task))
            law.update(source_f_population_covariance=kf.tolist(), learned_f_center=float(learned[0]),
                no_new_posterior_attached=True, level_risk_Mstep_parameters=learned.tolist())
            a = fixed + learned[3] * strong
            np.testing.assert_allclose(a[0, 0], risk['common_prior_covariance'][0][0], atol=0, rtol=0)
            assert np.all(np.diag(a)[1:] >= tail)
            prior_mean = np.asarray(risk['prior_mean']); prior_mean[0] += learned[2]
            new_risk = {**risk, 'arm': arm, 'prior_mean': prior_mean.tolist(), 'common_prior_covariance': a.tolist(),
                'source_eta_prior_mean': (np.asarray(field['eta_mean']) + learned[2]).tolist(),
                'source_eta_prior_covariance': kr.tolist(), 'original_positive_shape_tail_per_coefficient': tail,
                'common_strong_shape_covariance_multiplier': float(learned[3]), 'common_intercept_and_task_weak_tail_unchanged': True,
                'no_new_posterior_attached': True}
            # Changed covariance requires new conditional coefficient maps,not the old Stage207 gain.
            d = np.asarray(risk['source_design']); cross = a @ d.T
            gain = cho_solve(cho_factor(kr, lower=True), cross.T).T
            conditional = a - gain @ cross.T
            new_risk.update(source_eta_gain=gain.tolist(), conditional_covariance=(.5 * (conditional + conditional.T)).tolist())
            rows.append({'layout': field['layout'], 'arm': arm, 'old_parameters': theta.tolist(),
                'learned_parameters': learned.tolist(), 'parameter_names': ['f_center',
                    'task_intercept_root_multiplier' if index == 0 else 'task_constant_variance', 'eta_mean_offset', 'eta_common_strong_shape_covariance_multiplier'],
                'optimizer_converged': bool(result.success), 'optimizer_message': str(result.message),
                'optimizer_iterations': int(result.nit), 'optimizer_function_evaluations': int(result.nfev),
                'old_Q_components_full_first_second': old_splits, 'new_Q_components_full_first_second': new_splits,
                'Q_component_improvements_full_first_second': improvements.tolist(),
                'joint_Q_improvements_full_first_second': improvements.sum(axis=1).tolist(),
                'initial_risk_scores_full_first_second': [g.tolist() for g in risk_scores],
                'initial_joint_score': old_gradient.tolist(), 'final_joint_score': gradient.tolist(),
                'candidate_gate_passed': passed, 'final_Q_derivative_max_error': float(max(derivative_errors))})
            checks.append({'layout': field['layout'], 'arm': arm, 'f_Gaussian': fc, 'risk_Gaussian': rc})
            laws.append(law); risks.append(new_risk)
            print('level/risk M step', *key, 'pass', passed, 'parameters', learned.tolist(),
                'joint Q improvements', improvements.sum(axis=1).tolist(), flush=True)
    passed = all(r['candidate_gate_passed'] for r in rows)
    save(out / 'source_laws.json', {'execution': spec, 'rows': laws, 'all_source_Mstep_gates_passed': passed})
    save(out / 'risk_reconstruction.json', {'execution': spec, 'rows': risks, 'all_source_Mstep_gates_passed': passed})
    save(out / 'results.json', {'execution': spec, 'rows': rows, 'Gaussian_checks': checks,
        'all_source_Mstep_gates_passed': passed, 'cpu_seconds': time.process_time() - started,
        'cost': spec['cost'], 'no_model_adopted': True, 'independent_confirmation': False,
        'interpretation': 'One source-only constrained joint M step;Q improvement does not establish marginal-likelihood or independent decision gain. Function common intercept and risk common intercept/positive tails stay fixed. Fresh raw conditioning is required.'})


if __name__ == '__main__':
    run()
