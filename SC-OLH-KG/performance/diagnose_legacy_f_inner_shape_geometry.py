"""Honest inner-source predictive derivative for native shape correlation."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.run_legacy_related_inventory_source import save
from performance.validate_legacy_paired_taskout import gaussian_score
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis, spectral_weights
from representation.source_kernel_population import rbf_cross

EXECUTION = 'performance/manifests/legacy_joint_population_f_inner_shape_geometry_execution_v1.json'


def prepare(inputs, fits, query_name):
    tasks = [t for t in inputs['source_tasks'] if t['name'] != query_name]
    query = next(t for t in inputs['source_tasks'] if t['name'] == query_name)
    basis = OrthogonalLowOrderBasis(inputs['public_mean_reference'])
    def design(psi):
        return np.column_stack([np.ones(len(psi)), basis.transform(np.asarray(psi))])
    public = np.asarray(inputs['public_mean_reference']); public_design = design(public)
    operator = np.linalg.lstsq(public_design, np.eye(len(public)), rcond=None)[0]
    gp, native = fits['gp'], fits['structural']; anchor = gp['anchor']
    amplitude, intercept, _ = gp['residual_parameters']; scale2 = anchor['normalization']['scale']**2
    projected = operator @ (scale2*(amplitude*rbf_cross(public, public, anchor['lengthscale'])+intercept)) @ operator.T
    weights = spectral_weights(basis, native['spectral_exponent'])[0]
    excess = np.r_[0., np.asarray(native['source_group_variances'])[basis.group_index]*weights]
    active = np.flatnonzero(excess > 0)
    diagonal = np.diag(projected)[active]; assert np.all(diagonal > 0)
    correlation = projected[np.ix_(active, active)]/np.sqrt(np.outer(diagonal, diagonal))
    template = np.zeros_like(projected)
    template[np.ix_(active, active)] = correlation*np.sqrt(np.outer(excess[active], excess[active]))
    direction = template-np.diag(excess)
    np.testing.assert_allclose(np.diag(direction), 0., atol=1e-14, rtol=0.)
    np.testing.assert_array_equal(direction[0], 0.)
    assert np.linalg.eigvalsh(template).min() >= -1e-12
    residual = np.asarray(native['residual_coefficient_diagonal'])
    np.testing.assert_allclose(residual[1:], excess[1:]+native['tail_variance_budget']/basis.feature_dim,
                               atol=1e-12, rtol=1e-10)
    local = [design(t['psi']) for t in tasks]; d = np.vstack(local)
    a = np.diag(native['common_prior_coefficient_diagonal']); m0 = np.asarray(fits['regularizer']['coefficient_mean'])
    y = np.concatenate([np.asarray(t['response'])[:, 0] for t in tasks]); delta = native['discrepancy_variance']
    c = d @ a @ d.T+block_diag(*[(z*residual) @ z.T+np.diag(np.asarray(t['noise'])[:, 0]+delta)
                               for z, t in zip(local, tasks)])
    return {'d': d, 'a': a, 'm0': m0, 'error': y-d @ m0, 'c': c,
        'cq': block_diag(*[z @ direction @ z.T for z in local]), 'direction': direction, 'residual': residual,
        'query_design': design(query['psi']), 'query_labels': np.asarray(query['response'])[:, 0],
        'query_noise': np.asarray(query['noise'])[:, 0]+delta, 'native': native,
        'active_shape_coefficient_indices': active.tolist(), 'template_minimum_eigenvalue': float(np.linalg.eigvalsh(template).min())}


def condition(system, q):
    d, a, m0, error, direction = (system[k] for k in ('d', 'a', 'm0', 'error', 'cq'))
    covariance = system['c']+q*direction; factor = cho_factor(covariance, lower=True)
    alpha = cho_solve(factor, error); solved = cho_solve(factor, d @ a)
    beta = m0+a @ d.T @ alpha; h = a-a @ d.T @ solved; h = (h+h.T)/2
    beta_derivative = -solved.T @ direction @ alpha; h_derivative = solved.T @ direction @ solved
    z = system['query_design']; r = np.diag(system['residual'])+q*system['direction']
    mean = z @ beta; query_covariance = z @ (h+r) @ z.T+np.diag(system['query_noise'])
    predictive_factor = cho_factor(query_covariance, lower=True)
    predictive_alpha = cho_solve(predictive_factor, system['query_labels']-mean)
    query_derivative = z @ (h_derivative+system['direction']) @ z.T
    gradient = .5*(np.trace(cho_solve(predictive_factor, query_derivative))
        -predictive_alpha @ query_derivative @ predictive_alpha)-(z @ beta_derivative) @ predictive_alpha
    n = len(system['query_labels']); train_n = len(error)
    predictive_loss = -gaussian_score(mean, query_covariance, system['query_labels'])/n
    inverse = cho_solve(factor, np.eye(train_n))
    train_gradient = .5*(np.sum(inverse*direction)-alpha @ direction @ alpha)/train_n
    train_loss = .5*(train_n*np.log(2*np.pi)+2*np.log(np.diag(factor[0])).sum()+error @ alpha)/train_n
    return {'predictive_nll_per_label': float(predictive_loss), 'predictive_q_gradient_per_label': float(gradient/n),
        'training_nll_per_label': float(train_loss), 'training_q_gradient_per_label': float(train_gradient),
        'coefficient_mean': beta.tolist(), 'population_mean_covariance': h.tolist(),
        'predictive_mean': mean.tolist(), 'predictive_covariance_including_known_label_noise': query_covariance.tolist()}


def run():
    started = time.process_time(); spec = read(EXECUTION); inner = read(spec['inner_results'])
    assert inner['all_inner_fits_converged'] and inner['all_ten_inner_forecasts_frozen_before_scores']
    parent = read(inner['execution']['source_context_execution']); coverage, _ = source_paths(parent)
    rows = []
    for record in inner['rows']:
        layout, query = record['layout'], record['inner_query_task']
        inputs = read(Path(coverage['output'])/f'layout{layout}'/f"task{spec['outer_deleted_task']}"/'inputs.json')
        saved = read(record['source_fit']); assert query not in saved['inner_training_task_names']
        system = prepare(inputs, saved['fits'], query); native = system['native']; baseline = condition(system, 0.)
        for key in ('coefficient_mean', 'population_mean_covariance'):
            np.testing.assert_allclose(baseline[key], native[key], atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(baseline['training_nll_per_label'], native['source_negative_log_likelihood_per_label'], atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(baseline['predictive_mean'], record['prediction']['structural']['mean'], atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(-baseline['predictive_nll_per_label'], record['scores']['structural']['gaussian_batch_log_density_per_label'], atol=1e-8, rtol=1e-8)
        probe = condition(system, spec['fixed_probe_q']); verification = {'native_inner_source_law_and_forecast_reproduced': True,
            'native_active_support_frequency_marginals_and_independent_tail_preserved': True}
        if query == spec['gradient_verification_task']:
            point, step = spec['gradient_point'], spec['gradient_step']
            finite = (condition(system, point+step)['predictive_nll_per_label']-condition(system, point-step)['predictive_nll_per_label'])/(2*step)
            expected = condition(system, point)['predictive_q_gradient_per_label']
            np.testing.assert_allclose(finite, expected, atol=1e-7, rtol=1e-4)
            verification['predictive_gradient_finite_difference_error'] = float(abs(finite-expected))
        def scalar(value):
            return {k: v for k, v in value.items() if isinstance(v, float)}
        rows.append({'layout': layout, 'inner_query_task': query, 'inner_training_task_names': saved['inner_training_task_names'],
            'baseline': scalar(baseline), 'fixed_probe': scalar(probe), 'active_shape_coefficient_indices': system['active_shape_coefficient_indices'],
            'template_minimum_eigenvalue': system['template_minimum_eigenvalue'],
            'fixed_probe_predictive_density_gain_per_label': baseline['predictive_nll_per_label']-probe['predictive_nll_per_label'],
            'verification': verification})
    summary = {layout: {'mean_predictive_q_gradient': float(np.mean([r['baseline']['predictive_q_gradient_per_label'] for r in rows if r['layout'] == layout])),
        'mean_training_q_gradient': float(np.mean([r['baseline']['training_q_gradient_per_label'] for r in rows if r['layout'] == layout])),
        'mean_fixed_probe_predictive_density_gain': float(np.mean([r['fixed_probe_predictive_density_gain_per_label'] for r in rows if r['layout'] == layout])),
        'fixed_probe_improved_inner_tasks': sum(r['fixed_probe_predictive_density_gain_per_label'] > 0 for r in rows if r['layout'] == layout)}
        for layout in spec['layouts']}
    supported = all(v['mean_predictive_q_gradient'] < 0 and v['mean_fixed_probe_predictive_density_gain'] > 0 for v in summary.values())
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    result = {'execution': spec, 'rows': rows, 'summary': summary, 'source_predictive_shape_direction_supported': supported,
        'cpu_seconds': time.process_time()-started, 'cost': spec['cost'], 'no_model_adopted': True,
        'outer_query_paid_or_score_outcomes_read': False, 'source6_7_evaluated': False, 'target_promoted': False, 'independent_confirmation': False}
    save(out/'results.json', result); print({'summary': summary, 'direction_supported': supported, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
