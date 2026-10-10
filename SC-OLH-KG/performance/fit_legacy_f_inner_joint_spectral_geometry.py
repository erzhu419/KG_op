"""Source4 fits for a sparse joint prior with learned spectra and positive tails."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.run_legacy_related_inventory_source import save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis, spectral_weights
from representation.source_joint_spectral_geometry import JointSpectralGeometryLikelihood

EXECUTION = 'performance/manifests/legacy_joint_population_f_inner_joint_spectral_geometry_execution_v1.json'


def verify(model, parameters, state, value):
    common, task, h = (state[k] for k in ('common_prior_coefficient_covariance', 'task_coefficient_covariance', 'source_posterior_common_covariance'))
    for component, covariance in enumerate((common, task)):
        assert np.linalg.eigvalsh(covariance).min() >= -1e-10
        weights = spectral_weights(model.basis, parameters[2 * component + 1])[0]
        expected = parameters[2 * component] ** 2 * model.budgets[component][model.basis.group_index] * weights + model.tails[component]
        np.testing.assert_allclose(np.diag(covariance)[1:], expected, atol=1e-14, rtol=1e-10)
        inactive = np.setdiff1d(np.arange(len(covariance)), model.columns)
        np.testing.assert_array_equal(covariance[inactive], np.diag(np.diag(covariance))[inactive])
    assert np.linalg.eigvalsh(h).min() >= -1e-10
    # A coefficient solve independently detects incorrect source conditioning.
    precision = cho_solve(cho_factor(common, lower=True), np.eye(len(common))); rhs = np.zeros(len(common))
    logdet = np.linalg.slogdet(common)[1]
    offset = 0
    for d in model.designs:
        error = model.error[offset:offset + len(d)]; noise = model.noise[offset:offset + len(d)]; offset += len(d)
        factor = cho_factor(d @ task @ d.T + np.diag(noise), lower=True)
        precision += d.T @ cho_solve(factor, d); rhs += d.T @ cho_solve(factor, error)
        logdet += 2 * np.log(np.diag(factor[0])).sum()
    factor = cho_factor(precision, lower=True); checked_h = cho_solve(factor, np.eye(len(common)))
    displacement = checked_h @ rhs
    np.testing.assert_allclose(checked_h, h, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(np.r_[model.center, np.zeros(model.basis.feature_dim)] + displacement,
        state['source_posterior_coefficient_mean'], atol=1e-8, rtol=1e-8)
    quadratic = displacement @ cho_solve(cho_factor(common, lower=True), displacement); offset = 0
    for d in model.designs:
        error = model.error[offset:offset + len(d)]; noise = model.noise[offset:offset + len(d)]; offset += len(d)
        residual = error - d @ displacement
        quadratic += residual @ cho_solve(cho_factor(d @ task @ d.T + np.diag(noise), lower=True), residual)
    checked_value = .5 * (model.label_count * np.log(2 * np.pi) + logdet + 2 * np.log(np.diag(factor[0])).sum() + quadratic) / model.label_count
    np.testing.assert_allclose(checked_value, value, atol=1e-8, rtol=1e-8)
    return {'psd_support_spectra_positive_tails_verified': True, 'independent_coefficient_conditioning_verified': True,
        'dense_coefficient_nll_error': float(abs(checked_value - value))}


def run():
    started = time.process_time(); spec = read(EXECUTION)
    laws = read(spec['source_laws'])['rows']; spectra = read(spec['spectral_laws'])['rows']
    registry = read(spec['source_fit_registry'])['rows']; coverage, _ = source_paths(read(spec['source_context_execution']))
    out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True); rows, gradient_check = [], None
    base = {'execution': spec, 'query_outcomes_indexed': False, 'source6_7_evaluated': False, 'target_promoted': False,
        'independent_confirmation': False, 'no_model_adopted': True}
    for old in [r for r in laws if r['arm'] == 'joint_native_support']:
        layout, query = old['layout'], old['query_task']
        inputs = read(Path(coverage['output']) / f'layout{layout}' / 'task1' / 'inputs.json')
        tasks = [t for t in inputs['source_tasks'] if t['name'] in old['source_task_names']]
        assert len(tasks) == 4 and [t['name'] for t in tasks] == old['source_task_names']
        assert 'related_inventory_01' not in old['source_task_names'] and f'related_inventory_{query:02d}' not in old['source_task_names']
        record = next(r for r in registry if r['layout'] == layout and r['inner_query_task'] == f'related_inventory_{query:02d}')
        fitted = read(record['source_fit']); assert fitted['inner_training_task_names'] == old['source_task_names']
        anchor = fitted['fits']['gp']['anchor']; center = anchor['normalization']['mean'] + anchor['normalization']['scale'] * anchor['constant']
        basis = OrthogonalLowOrderBasis(inputs['public_mean_reference'])
        model = JointSpectralGeometryLikelihood(basis, tasks, old['common_prior_coefficient_covariance'],
            old['task_coefficient_covariance'], old['columns'], center, old['white_discrepancy'], spec['positive_tail_fraction'])
        p = [next(r['spectral_exponent'] for r in spectra if (r['layout'], r['query_task'], r['arm'], r['component'])
            == (layout, query, 'joint_native_support', component)) for component in ('common', 'task')]
        initial = np.asarray([1., p[0], 1., p[1]])
        if gradient_check is None:
            analytic = model.evaluate(initial)[1]; finite = []
            for j in range(4):
                lower, upper = initial.copy(), initial.copy(); lower[j] -= spec['gradient_step']; upper[j] += spec['gradient_step']
                finite.append((model.evaluate(upper)[0] - model.evaluate(lower)[0]) / (2 * spec['gradient_step']))
            np.testing.assert_allclose(analytic, finite, atol=1e-6, rtol=1e-5)
            gradient_check = {'analytic': analytic.tolist(), 'finite': finite, 'max_absolute_error': float(np.max(np.abs(analytic - finite))), 'passed': True}
        initial_value = model.evaluate(initial)[0]
        def objective(parameters):
            if time.process_time() - started >= spec['maximum_single_thread_cpu_seconds']: raise TimeoutError('source fit CPU cap')
            return model.evaluate(parameters)[:2]
        try:
            fitted_result = minimize(objective, initial, jac=True, method='L-BFGS-B', bounds=[(0., None), (0., 4.)] * 2,
                options={'maxiter': 512, 'ftol': 1e-9, 'gtol': 1e-5})
        except TimeoutError:
            save(out / 'results.json', {**base, 'rows': rows, 'all_ten_source_fits_complete': False,
                'stopped_at_cpu_budget': True, 'cpu_seconds': time.process_time() - started}); return
        value, gradient, state = model.evaluate(fitted_result.x)
        row = {'layout': layout, 'query_task': query, 'source_task_names': old['source_task_names'], 'columns': old['columns'],
            'parameters': fitted_result.x.tolist(), 'initial_parameters': initial.tolist(), 'initial_nll_per_label': initial_value,
            'source_nll_per_label': value, 'nll_change_vs_gp_per_label': value - old['source_nll_per_label'] + old['source_total_nll_change_vs_gp'] / model.label_count,
            'common_tail_per_coefficient': float(model.tails[0]), 'task_tail_per_coefficient': float(model.tails[1]),
            'white_discrepancy': old['white_discrepancy'], 'optimizer_converged': bool(fitted_result.success),
            'optimizer_message': str(fitted_result.message), 'optimizer_iterations': int(fitted_result.nit),
            **{k: v.tolist() for k, v in state.items()}}
        rows.append(row)
        if not fitted_result.success:
            save(out / 'results.json', {**base, 'rows': rows, 'all_ten_source_fits_complete': False,
                'gradient_check': gradient_check, 'cpu_seconds': time.process_time() - started}); return
        row['verification'] = verify(model, fitted_result.x, state, value)
        print('source joint spectral fit', layout, query, 'parameters', row['parameters'], 'NLL change vs GP', row['nll_change_vs_gp_per_label'], flush=True)
    assert len(rows) == 10
    save(out / 'source_laws.json', {**base, 'rows': rows, 'all_ten_source_laws_frozen_without_query_values': True})
    save(out / 'results.json', {**base, 'rows': rows, 'all_ten_source_fits_complete': True,
        'gradient_check': gradient_check, 'cpu_seconds': time.process_time() - started, 'cost': spec['cost']})


if __name__ == '__main__':
    run()
