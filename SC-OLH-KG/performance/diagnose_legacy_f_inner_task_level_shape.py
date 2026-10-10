"""Honest source-task prediction for joint level/shape variance allocation."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_inner_shape_geometry import condition
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.run_legacy_related_inventory_source import save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis, spectral_weights
from representation.source_kernel_population import rbf_cross

EXECUTION = 'performance/manifests/legacy_joint_population_f_inner_task_level_shape_execution_v1.json'


def joint_template(basis, design, public, fits):
    native, gp = fits['structural'], fits['gp']; anchor = gp['anchor']
    amplitude, intercept, _ = gp['residual_parameters']; scale2 = anchor['normalization']['scale']**2
    operator = np.linalg.lstsq(design, np.eye(len(public)), rcond=None)[0]
    projected = operator @ (scale2*(amplitude*rbf_cross(public, public, anchor['lengthscale'])+intercept)) @ operator.T
    projected = .5*(projected+projected.T)
    active = (np.asarray(native['source_group_variances']) > 0) | (np.asarray(native['common_group_variances']) > 0)
    columns = np.r_[0, 1+np.flatnonzero(active[basis.group_index])]
    diagonal = np.diag(projected)[columns]; assert np.all(diagonal > 0)
    correlation = projected[np.ix_(columns, columns)]/np.sqrt(np.outer(diagonal, diagonal))
    budgets = np.bincount(basis.group_index, weights=np.diag(projected)[1:])*active
    weights = spectral_weights(basis, native['spectral_exponent'])[0]
    marginal = np.r_[projected[0, 0], budgets[basis.group_index]*weights]
    template = np.zeros_like(projected)
    template[np.ix_(columns, columns)] = correlation*np.sqrt(np.outer(marginal[columns], marginal[columns]))
    excess = np.r_[0., np.asarray(native['source_group_variances'])[basis.group_index]*weights]
    tail = native['tail_variance_budget']/basis.feature_dim
    np.testing.assert_allclose(excess[1:]+tail, native['residual_coefficient_diagonal'][1:], atol=1e-12, rtol=1e-10)
    np.testing.assert_allclose(np.diag(template), marginal, atol=1e-12, rtol=1e-10)
    inactive = 1+np.flatnonzero(~active[basis.group_index]); np.testing.assert_array_equal(template[inactive], 0.)
    minimum = float(np.linalg.eigvalsh(template).min()); assert minimum >= -1e-12
    return template-np.diag(excess), {'active_coefficient_indices': columns.tolist(),
        'active_group_names': [n for n, enabled in zip(basis.group_names, active) if enabled],
        'template_group_variance_budgets': budgets.tolist(), 'template_level_variance': float(template[0, 0]),
        'template_minimum_eigenvalue': minimum, 'native_task_spectral_exponent': native['spectral_exponent'],
        'independent_tail_per_shape_coefficient': tail, 'native_independent_task_level_variance': native['residual_coefficient_diagonal'][0],
        'native_support_union_spectral_marginals_and_independent_tail_preserved': True}


def prepare(inputs, fits, query_name):
    tasks = [t for t in inputs['source_tasks'] if t['name'] != query_name]
    query = next(t for t in inputs['source_tasks'] if t['name'] == query_name)
    assert len(tasks) == 4 and len(inputs['source_tasks']) == 5
    basis = OrthogonalLowOrderBasis(inputs['public_mean_reference']); public = np.asarray(inputs['public_mean_reference'])
    def design(psi):
        return np.column_stack([np.ones(len(psi)), basis.transform(np.asarray(psi))])
    direction, geometry = joint_template(basis, design(public), public, fits)
    local = [design(t['psi']) for t in tasks]; d = np.vstack(local)
    native = fits['structural']; a = np.diag(native['common_prior_coefficient_diagonal'])
    m0 = np.asarray(fits['regularizer']['coefficient_mean']); residual = np.asarray(native['residual_coefficient_diagonal'])
    error = np.concatenate([np.asarray(t['response'])[:, 0] for t in tasks])-d @ m0
    delta = native['discrepancy_variance']
    covariance = d @ a @ d.T+block_diag(*[(z*residual) @ z.T+np.diag(np.asarray(t['noise'])[:, 0]+delta)
                                         for z, t in zip(local, tasks)])
    return {'d': d, 'a': a, 'm0': m0, 'error': error, 'c': covariance,
        'cq': block_diag(*[z @ direction @ z.T for z in local]), 'direction': direction, 'residual': residual,
        'query_design': design(query['psi']), 'query_labels': np.asarray(query['response'])[:, 0],
        'query_noise': np.asarray(query['noise'])[:, 0]+delta, 'native': native, 'geometry': geometry}


def run():
    started = time.process_time(); spec = read(EXECUTION); inner = read(spec['inner_results'])
    assert inner['all_inner_fits_converged'] and inner['all_ten_inner_forecasts_frozen_before_scores']
    coverage, _ = source_paths(read(inner['execution']['source_context_execution'])); rows = []
    for record in inner['rows']:
        layout, query = record['layout'], record['inner_query_task']
        inputs = read(Path(coverage['output'])/f'layout{layout}'/f"task{spec['outer_deleted_task']}"/'inputs.json')
        assert {t['name'] for t in inputs['source_tasks']} == {f'related_inventory_{i:02d}' for i in range(6) if i != spec['outer_deleted_task']}
        saved = read(record['source_fit']); assert query not in saved['inner_training_task_names']
        assert set(saved['inner_training_task_names']) == {t['name'] for t in inputs['source_tasks'] if t['name'] != query}
        system = prepare(inputs, saved['fits'], query); native = system['native']; baseline = condition(system, 0.)
        for key in ('coefficient_mean', 'population_mean_covariance'):
            np.testing.assert_allclose(baseline[key], native[key], atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(baseline['training_nll_per_label'], native['source_negative_log_likelihood_per_label'], atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(baseline['predictive_mean'], record['prediction']['structural']['mean'], atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(-baseline['predictive_nll_per_label'], record['scores']['structural']['gaussian_batch_log_density_per_label'], atol=1e-8, rtol=1e-8)
        probe = condition(system, spec['fixed_probe_q']); verification = {'native_source_law_and_inner_forecast_reproduced': True}
        if query == spec['gradient_verification_task']:
            point, step = spec['gradient_point'], spec['gradient_step']
            finite = (condition(system, point+step)['predictive_nll_per_label']-condition(system, point-step)['predictive_nll_per_label'])/(2*step)
            expected = condition(system, point)['predictive_q_gradient_per_label']
            np.testing.assert_allclose(finite, expected, atol=1e-7, rtol=1e-4)
            verification['predictive_gradient_finite_difference_error'] = float(abs(finite-expected))
        def scalar(value):
            return {k: v for k, v in value.items() if isinstance(v, float)}
        rows.append({'layout': layout, 'inner_query_task': query, 'inner_training_task_names': saved['inner_training_task_names'],
            'baseline': scalar(baseline), 'fixed_probe': scalar(probe), 'geometry': system['geometry'], 'verification': verification,
            'fixed_probe_predictive_density_gain_per_label': baseline['predictive_nll_per_label']-probe['predictive_nll_per_label']})
    summary = {layout: {'mean_predictive_q_gradient': float(np.mean([r['baseline']['predictive_q_gradient_per_label'] for r in rows if r['layout'] == layout])),
        'mean_training_q_gradient': float(np.mean([r['baseline']['training_q_gradient_per_label'] for r in rows if r['layout'] == layout])),
        'mean_fixed_probe_predictive_density_gain': float(np.mean([r['fixed_probe_predictive_density_gain_per_label'] for r in rows if r['layout'] == layout])),
        'fixed_probe_improved_inner_tasks': sum(r['fixed_probe_predictive_density_gain_per_label'] > 0 for r in rows if r['layout'] == layout)}
        for layout in spec['layouts']}
    supported = all(v['mean_predictive_q_gradient'] < 0 and v['mean_fixed_probe_predictive_density_gain'] > 0 for v in summary.values())
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    result = {'execution': spec, 'rows': rows, 'summary': summary, 'source_predictive_joint_direction_supported': supported,
        'cpu_seconds': time.process_time()-started, 'cost': spec['cost'], 'no_model_adopted': True,
        'outer_query_paid_or_score_outcomes_read': False, 'source6_7_evaluated': False, 'target_promoted': False, 'independent_confirmation': False}
    save(out/'results.json', result); print({'summary': summary, 'direction_supported': supported, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
