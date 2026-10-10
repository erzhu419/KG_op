"""Predictive relevance of source parameter information, not a posterior fit."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_f_source_spectral_uncertainty import source_model
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_information_ceiling import variance_parts
from performance.run_legacy_related_inventory_source import save

PLAN = 'performance/manifests/legacy_joint_population_f_source_predictive_information_plan_v1.json'


def predictive_derivatives(model, theta):
    a, aa, ap = model.covariance(0, theta[0], theta[1])
    r, ra, _ = model.covariance(1, theta[2], theta[3]); d = model.design; b = a @ d.T
    covariance = d @ a @ d.T + block_diag(*[z @ r @ z.T for z in model.designs]) + np.diag(model.noise)
    factor = cho_factor(covariance, lower=True); alpha = cho_solve(factor, model.error); solved_b = cho_solve(factor, b.T)
    beta_derivatives, covariance_derivatives = [], []
    zero = np.zeros_like(a)
    for da, dr in ((aa, zero), (ap, zero), (zero, ra)):
        dk = d @ da @ d.T + block_diag(*[z @ dr @ z.T for z in model.designs]); db = da @ d.T
        beta_derivatives.append(db @ alpha - b @ cho_solve(factor, dk @ alpha))
        dh = da - db @ solved_b - solved_b.T @ db.T + solved_b.T @ dk @ solved_b
        covariance_derivatives.append(.5 * (dh + dh.T) + dr)
    return np.stack(beta_derivatives, axis=1), np.asarray(covariance_derivatives)


def run():
    started = time.process_time(); spec = read(PLAN); probe = read(spec['source_probe_results']); assert probe['diagnostic_interpreted']
    source_spec = read(spec['source_probe_plan']); information = read(spec['information_results']); rows = []
    out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    for layout in spec['layouts']:
        model, law, public = source_model(source_spec, layout, spec['query_task']); theta = np.asarray(law['parameters'])
        nll, _, state = model.evaluate(theta)
        np.testing.assert_allclose(nll, law['source_nll_per_label'], atol=1e-10, rtol=1e-10)
        for k in state: np.testing.assert_allclose(state[k], law[k], atol=1e-10, rtol=1e-10)
        jacobian, directions = predictive_derivatives(model, theta); verification = None
        if not rows:
            checks = []; step = spec['derivative_step']; tolerance = spec['derivative_tolerances']
            for j in range(3):
                lower, upper = theta.copy(), theta.copy(); lower[j] -= step; upper[j] += step
                a, b = model.evaluate(lower)[2], model.evaluate(upper)[2]
                beta_fd = (b['source_posterior_coefficient_mean'] - a['source_posterior_coefficient_mean']) / (2 * step)
                covariance_fd = ((b['source_posterior_common_covariance'] + b['task_coefficient_covariance'])
                    - (a['source_posterior_common_covariance'] + a['task_coefficient_covariance'])) / (2 * step)
                np.testing.assert_allclose(jacobian[:, j], beta_fd, atol=tolerance['absolute'], rtol=tolerance['relative'])
                np.testing.assert_allclose(directions[j], covariance_fd, atol=tolerance['absolute'], rtol=tolerance['relative'])
                checks.append({'parameter': j, 'beta_max_absolute_error': float(np.max(np.abs(jacobian[:, j] - beta_fd))),
                    'covariance_max_absolute_error': float(np.max(np.abs(directions[j] - covariance_fd)))})
            verification = {'new_predictive_derivatives_verified': True, 'checks': checks}
        record = next(r for r in information['rows'] if (r['source_task_count'], r['layout'], r['query_task']) == (4, layout, spec['query_task']))
        fisher = np.asarray(record['expected_fisher_information'])[:3, :3]; inverse = np.linalg.solve(fisher, np.eye(3))
        design = np.column_stack([np.ones(len(public)), model.basis.transform(public)])
        prior = design @ (state['source_posterior_common_covariance'] + state['task_coefficient_covariance']) @ design.T + model.white * np.eye(len(public))
        joint = jacobian @ inverse @ jacobian.T; assert np.linalg.eigvalsh(joint).min() >= -1e-10
        public_jacobian = design @ jacobian; public_joint = public_jacobian @ inverse @ public_jacobian.T
        np.testing.assert_allclose(public_joint, design @ joint @ design.T, atol=1e-12, rtol=1e-10)
        tangents = []
        for j, name in enumerate(('common_root_amplitude', 'common_spectral_exponent', 'task_root_amplitude')):
            local_sd = float(1 / np.sqrt(fisher[j, j])); mean_direction = public_jacobian[:, j] * local_sd
            covariance_direction = design @ directions[j] @ design.T * local_sd
            tangents.append({'parameter': name, 'fixed_other_parameters_local_sd': local_sd,
                'predictive_mean_variance_parts': variance_parts(np.outer(mean_direction, mean_direction)),
                'predictive_covariance_relative_frobenius_tangent': float(np.linalg.norm(covariance_direction) / np.linalg.norm(prior)),
                'centered_shape_covariance_tangent': variance_parts(covariance_direction)['centered_shape_variance']})
        row = {'layout': layout, 'query_task': spec['query_task'], 'source_task_names': law['source_task_names'],
            'task_exponent_fixed_at_4': True, 'interior_parameter_inverse_fisher': inverse.tolist(), 'point_prior_variance_parts': variance_parts(prior),
            'joint_local_mean_uncertainty_parts': variance_parts(public_joint), 'fixed_other_parameter_tangents': tangents,
            'verification': verification, 'source_point_law_reproduced': True}; rows.append(row)
        print('predictive source information', layout, 'joint local mean uncertainty', row['joint_local_mean_uncertainty_parts'], flush=True)
        assert time.process_time() - started < spec['maximum_single_thread_cpu_seconds']
    save(out / 'results.json', {'execution': spec, 'rows': rows, 'diagnostic_interpreted': True, 'cpu_seconds': time.process_time() - started,
        'cost': {**spec['cost'], 'actual_source_conditionings': 8}, 'query_outcomes_indexed': False, 'no_model_adopted': True,
        'point_candidate_remains_stopped': True, 'independent_confirmation': False,
        'interpretation': 'Expected Fisher local sensitivity at interior parameters,task exponent fixed at4. J I^-1 J^T is NOT a posterior variance to add to a deployed prior. Template,center,support,tail,white and estimation uncertainty excluded.'})


if __name__ == '__main__':
    run()
