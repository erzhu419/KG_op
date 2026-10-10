"""Reproduce the actual source optimizer failure and verify whitened Q."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_f_raw_level_risk_mstep import expected, source_moments
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.fit_legacy_f_bounded_level_risk_source import make_systems, objective, project
from performance.run_legacy_related_inventory_source import save

PLAN = 'performance/manifests/legacy_joint_population_f_level_risk_whitened_Q_plan_v1.json'


def inverse_expected(theta, builder, mean, covariance):
    prior_mean, k, directions = builder(theta); factor = cho_factor(k, lower=True)
    delta = mean - prior_mean; second = covariance + np.outer(delta, delta)
    inverse = cho_solve(factor, np.eye(48)); solved = inverse @ second
    q = .5 * (48 * np.log(2 * np.pi) + 2 * np.log(np.diag(factor[0])).sum() + np.trace(solved)) / 48
    contrast = .5 * (inverse - solved @ inverse) / 48
    gradient = np.r_[-(inverse @ delta).sum() / 48, [np.sum(contrast * d.T) for d in directions]]
    return float(q), gradient


def run():
    started = time.process_time(); spec = read(PLAN); bounded = read(spec['bounded_learning_plan'])
    failed = read(spec['failed_learning_results'])['rounds'][-1]['fits']
    moments = read(spec['accepted_source_moments']); checks = []
    systems = make_systems(bounded)
    for system in systems:
        key = system['key']; failure = next(r for r in failed if (r['layout'], r['arm']) == key)
        post = next(r for r in moments['rows'] if (r['layout'], r['arm']) == key)
        part = next(r for r in moments['partitions'] if (r['layout'], r['arm']) == key)
        fm, rm = source_moments(post, part); theta = np.asarray(failure['old_parameters'])
        candidate = np.asarray(failure['candidate_parameters']); old_q, new_q, value_errors, gradient_errors = [], [], [], []
        for t in (theta, candidate):
            components = []
            for builder, params, posterior in [(system['f_builder'], t[:2], fm[0]), (system['risk_builder'], t[2:], rm[0])]:
                value, gradient = inverse_expected(params, builder, *posterior)
                stable_value, stable_gradient = expected(params, builder, *posterior)
                np.testing.assert_allclose(stable_value, value, atol=1e-7, rtol=1e-8)
                np.testing.assert_allclose(stable_gradient, gradient, atol=1e-5, rtol=1e-6)
                value_errors.append(abs(value - stable_value)); gradient_errors.append(float(np.max(abs(gradient - stable_gradient))))
            for h in range(3):
                components.append([inverse_expected(t[:2], system['f_builder'], *fm[h])[0],
                    inverse_expected(t[2:], system['risk_builder'], *rm[h])[0]])
            if np.array_equal(t, theta): old_q = components
            else: new_q = components
        np.testing.assert_allclose(np.asarray(old_q) - new_q, failure['Q_component_improvements_full_first_second'], atol=1e-10, rtol=1e-8)
        checks.append({'layout': key[0], 'arm': key[1], 'inverse_cached_Q_improvements_reproduced': True,
            'maximum_inverse_whitened_Q_difference': max(value_errors), 'maximum_inverse_whitened_gradient_difference': max(gradient_errors)})
    system = next(s for s in systems if s['key'] == ('6', 'source_gp')); failure = next(r for r in failed if (r['layout'], r['arm']) == system['key'])
    post = next(r for r in moments['rows'] if (r['layout'], r['arm']) == system['key'])
    part = next(r for r in moments['partitions'] if (r['layout'], r['arm']) == system['key']); sm = source_moments(post, part)
    theta = np.asarray(failure['old_parameters']); scale = system['scale']; candidate = np.asarray(failure['candidate_parameters'])
    value, gradient, _ = objective(system, candidate, sm); derivative_errors = []
    for j in range(4):
        h = 1e-5 * scale[j]; t = candidate.copy()
        if j in (1, 3) and candidate[j] < h:
            t[j] += h; first = objective(system, t, sm)[0]; t[j] += h
            finite = (-3 * value + 4 * first - objective(system, t, sm)[0]) / (2 * h)
        else:
            t[j] += h; upper = objective(system, t, sm)[0]; t[j] -= 2 * h
            finite = (upper - objective(system, t, sm)[0]) / (2 * h)
        np.testing.assert_allclose(finite, gradient[j], atol=1e-6, rtol=1e-4); derivative_errors.append(abs(finite - gradient[j]))
    def q(z):
        v, g, _ = objective(system, z * scale, sm); return v, g * scale
    settings = bounded['optimizer']; opt = minimize(q, theta / scale, method=settings['method'], jac=True,
        bounds=[(None, None), (0., None), (None, None), (0., None)],
        options={'maxiter': settings['max_iterations'], 'ftol': settings['ftol'], 'gtol': settings['gtol']})
    learned = opt.x * scale; new_value, new_gradient, _ = objective(system, learned, sm)
    assert time.process_time() - started < spec['maximum_single_thread_cpu_seconds']
    out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    save(out / 'results.json', {'execution': spec, 'checks': checks, 'same_objective_and_cached_moments_verified': True,
        'failed_GP6_finite_gradient_max_error': max(derivative_errors), 'whitened_optimizer_converged': bool(opt.success),
        'whitened_optimizer_message': str(opt.message), 'whitened_optimizer_iterations': int(opt.nit),
        'whitened_candidate_parameters': learned.tolist(), 'maximum_scaled_projected_gradient': float(np.abs(project(learned, new_gradient) * scale).max()),
        'old_inverse_failure_preserved': True, 'cpu_seconds': time.process_time() - started, 'cost': spec['cost'],
        'scientific_model_changed': False, 'independent_confirmation': False})
    print('same-Q numerical repair', 'converged', opt.success, 'message', opt.message,
        'maximum scaled projected gradient', np.abs(project(learned, new_gradient) * scale).max(), flush=True)


if __name__ == '__main__':
    run()
