"""One source task-shape correlation fit and conditional level integration."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_sparse_task_template import condition, source_paths, source_system
from performance.diagnose_legacy_f_task_template_attribution import native_task_shape_correlation
from performance.run_legacy_related_inventory_source import save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.task_level_reference import TaskLevelReferencePosterior

EXECUTION = 'performance/manifests/legacy_joint_population_f_task_shape_correlation_reference_execution_v1.json'


def fit(inputs, head, saved, deleted, spec):
    excess = np.asarray(saved['original_task_excess_diagonal'])
    template = native_task_shape_correlation(np.asarray(saved['template_coefficient_covariance']), excess)
    system = source_system(inputs, head, template, excess, saved['independent_tail_per_coefficient'], deleted)
    native = system['native']; v = native['residual_coefficient_diagonal'][0]; baseline = condition(system, 0., v)
    for key in ('source_negative_log_likelihood_per_label', 'coefficient_mean', 'population_mean_covariance'):
        np.testing.assert_allclose(baseline[key], native[key], atol=1e-8, rtol=1e-8)
    optimizer = inputs['execution']['optimizer']
    def objective(x):
        value = condition(system, float(x[0]), v, derivative=True)
        return value['source_negative_log_likelihood_per_label'], np.asarray([value['q_gradient_per_label']])
    fitted = minimize(objective, [0.], jac=True, bounds=[(0., 1.)], method=optimizer['method'],
        options={'maxiter': optimizer['max_iterations'], 'ftol': optimizer['ftol'], 'gtol': optimizer['gtol']})
    if not fitted.success:
        raise RuntimeError(f'Original-contract task correlation optimizer failed: {fitted.message}')
    q = float(fitted.x[0]); value = condition(system, q, v, derivative=True)
    assert value['source_negative_log_likelihood_per_label'] <= baseline['source_negative_log_likelihood_per_label']+1e-10
    base = system['base']+q*system['direction']
    np.testing.assert_allclose(np.diag(base), np.diag(system['base']), atol=1e-14, rtol=1e-12)
    np.testing.assert_array_equal(base[0], 0.)
    assert np.linalg.eigvalsh(base).min() >= -1e-12
    verification = {'native_zero_correlation_law_reproduced': True, 'fitted_task_shape_covariance_psd': True,
        'native_support_spectrum_marginals_tail_and_independent_v_preserved': True}
    if [saved['layout'], deleted] in spec['derivative_verification_folds']:
        point, step = spec['derivative_point'], spec['derivative_step']
        finite = (objective([point+step])[0]-objective([point-step])[0])/(2*step)
        np.testing.assert_allclose(finite, objective([point])[1][0], atol=1e-7, rtol=1e-4)
        verification.update(interior_q_gradient_error=float(abs(finite-objective([point])[1][0])))
    law = {**value, 'fitted_q': q, 'task_shape_covariance_at_zero_independent_intercept': base.tolist(),
        'source_task_names': [t['name'] for t in system['tasks']], 'common_prior_coefficient_diagonal': native['common_prior_coefficient_diagonal'],
        'empirical_center_coefficient_mean': system['m0'].tolist(), 'native_task_intercept_variance': v,
        'source_signal_scale': native['source_signal_scale'], 'discrepancy_variance': native['discrepancy_variance'],
        'independent_tail_per_coefficient': saved['independent_tail_per_coefficient'], 'strong_task_group_variances': native['source_group_variances'],
        'task_spectral_exponent': native['spectral_exponent'], 'optimizer': optimizer, 'optimizer_converged': bool(fitted.success),
        'optimizer_message': str(fitted.message), 'optimizer_iterations': int(fitted.nit), 'optimizer_evaluations': int(fitted.nfev)}
    record = {'fitted_q': q, 'source_log_likelihood_gain_per_label': baseline['source_negative_log_likelihood_per_label']-value['source_negative_log_likelihood_per_label'],
        'optimizer_converged': bool(fitted.success), 'optimizer_iterations': int(fitted.nit), 'verification': verification}
    return record, law, system


def integrate(inputs, law, system, deleted, spec):
    q = law['fitted_q']; c0 = system['c0']+q*system['cq']; u = system['u']; d, a = system['d'], system['a']
    error = system['error']; factor = cho_factor(c0, lower=True)
    inverse_u = cho_solve(factor, u); alpha0 = cho_solve(factor, error)
    b = u.T@inverse_u; eigenvalues, vectors = np.linalg.eigh(.5*(b+b.T)); assert np.all(eigenvalues > 0)
    rhs = vectors.T@u.T@alpha0; update = a@d.T@inverse_u@vectors
    beta0 = system['m0']+a@d.T@alpha0; h0 = a-a@d.T@cho_solve(factor, d)@a; h0 = .5*(h0+h0.T)
    nll0 = .5*(len(error)*np.log(2*np.pi)+2*np.log(np.diag(factor[0])).sum()+error@alpha0)
    v, signal = law['native_task_intercept_variance'], law['source_signal_scale']
    posterior = TaskLevelReferencePosterior(eigenvalues, rhs, signal, v, spec['quadrature'], signal*spec['quadrature_split_variance_over_signal'])
    def conditional(variance):
        weight = variance/(1+variance*eigenvalues)
        return beta0-update@(weight*rhs), h0+(update*weight)@update.T
    beta, h = conditional(v)
    np.testing.assert_allclose((nll0+posterior.nll_difference(v))/len(error), law['source_negative_log_likelihood_per_label'], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(beta, law['coefficient_mean'], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(h, law['population_mean_covariance'], atol=1e-8, rtol=1e-8)
    verification = {'fitted_full_source_nll_beta_H_reproduced': True}
    if deleted == spec['dense_verification_deleted_task']:
        interior = signal*spec['dense_verification_variance_over_source_signal']
        dense_factor = cho_factor(c0+interior*u@u.T, lower=True); dense_alpha = cho_solve(dense_factor, error)
        dense_nll = .5*(len(error)*np.log(2*np.pi)+2*np.log(np.diag(dense_factor[0])).sum()+error@dense_alpha)
        dense_b = u.T@cho_solve(dense_factor, u); expected_beta, expected_h = conditional(interior)
        np.testing.assert_allclose(nll0+posterior.nll_difference(interior), dense_nll, atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(posterior.information(interior), .5*np.trace(dense_b@dense_b), atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(expected_beta, system['m0']+a@d.T@dense_alpha, atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(expected_h, a-a@d.T@cho_solve(dense_factor, d)@a, atol=1e-8, rtol=1e-8)
        verification.update(interior_dense_likelihood_Fisher_beta_H_verified=True)
    moments = posterior.moments(spec['posterior_quantile_probabilities'])
    average = moments.pop('expected_shrinkage'); shrinkage_covariance = moments.pop('shrinkage_covariance')
    weights = average/eigenvalues; mean_update = update*(rhs/eigenvalues)
    integrated_beta = beta0-update@(weights*rhs)
    average_h = h0+(update*weights)@update.T
    integrated_h = average_h+mean_update@shrinkage_covariance@mean_update.T; integrated_h = .5*(integrated_h+integrated_h.T)
    assert np.linalg.eigvalsh(integrated_h).min() >= -1e-10
    basis = OrthogonalLowOrderBasis(inputs['public_mean_reference'])
    public = np.column_stack([np.ones(len(inputs['public_mean_reference'])), basis.transform(np.asarray(inputs['public_mean_reference']))])
    reference_law = {'fitted_source_law': law, 'task_variance_eigenvalues': eigenvalues.tolist(), 'task_variance_rhs': rhs.tolist(),
        'coefficient_mean_at_zero_task_intercept_variance': beta0.tolist(), 'common_covariance_at_zero_task_intercept_variance': h0.tolist(),
        'common_coefficient_update_columns': update.tolist(), 'source_signal': signal, 'native_task_intercept_variance': v,
        'reference_log_normalizer_relative_to_fitted_nll': float(posterior.log_scale+np.log(posterior.normalization)),
        'integrated_common_coefficient_mean': integrated_beta.tolist(), 'integrated_common_coefficient_covariance': integrated_h.tolist(),
        'integrated_task_intercept_variance': moments['mean'], 'full_task_shape_covariance_at_zero_intercept': law['task_shape_covariance_at_zero_independent_intercept'],
        'point_predictions_not_a_coherent_mixture_replacement': True}
    record = {'native_task_intercept_variance': v, 'reference_posterior': moments,
        'common_mean_bank_rms_change_from_fitted_plugin': float(np.sqrt(np.mean((public@(integrated_beta-beta))**2))),
        'scalar_quadrature_evaluations': posterior.scalar_evaluations, 'maximum_scalar_quadrature_error': posterior.maximum_scalar_error,
        'verification': verification, 'conditional_v_integrated_without_floor_or_cutoff': True}
    return record, reference_law


def run():
    started = time.process_time(); spec = read(EXECUTION); parent = read(spec['template_execution'])
    attribution = read(spec['attribution_results'])['summary']
    assert all(r['probes']['native_task_shape_correlation']['mean_source_log_likelihood_gain_per_label'] > 0 for r in attribution.values())
    coverage, profile = source_paths(parent); templates = read(spec['source_templates'])['rows']
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True); records, laws, inputs_list, systems = [], [], [], []
    for saved in templates:
        layout, deleted = saved['layout'], saved['deleted_task']; suffix = Path(f'layout{layout}')/f'task{deleted}'
        inputs = read(Path(coverage['output'])/suffix/'inputs.json'); inputs_list.append(inputs)
        head = read(Path(profile['output'])/suffix/'selected_source_fit.json')['heads'][0]
        record, law, system = fit(inputs, head, saved, deleted, spec)
        records.append({'layout': layout, 'deleted_task': deleted, **record}); laws.append({'layout': layout, 'deleted_task': deleted, **law}); systems.append(system)
    save(out/'fitted_source_laws.json', {'execution': spec, 'all12_source_fits_frozen_before_v_integration': True, 'rows': laws})
    integrated = []; rows = []
    for inputs, system, record, law in zip(inputs_list, systems, records, laws):
        result, reference_law = integrate(inputs, law, system, law['deleted_task'], spec)
        rows.append({**record, 'integration': result}); integrated.append({'layout': law['layout'], 'deleted_task': law['deleted_task'], **reference_law})
    save(out/'source_reference_laws.json', {'execution': spec, 'all12_conditional_v_source_laws_saved': True, 'rows': integrated})
    summary = {layout: {'fitted_q': [r['fitted_q'] for r in rows if r['layout'] == layout],
        'mean_source_log_likelihood_gain_per_label': float(np.mean([r['source_log_likelihood_gain_per_label'] for r in rows if r['layout'] == layout])),
        'total_source_log_likelihood_gain_over_six_folds': float(sum(r['source_log_likelihood_gain_per_label']*60 for r in rows if r['layout'] == layout)),
        'task1_independent_intercept_variance_posterior': next(r['integration']['reference_posterior'] for r in rows if r['layout'] == layout and r['deleted_task'] == 1)}
        for layout in parent['layouts']}
    result = {'execution': spec, 'rows': rows, 'summary': summary, 'all12_source_fits_and_integrations_verified': True,
        'cost': spec['cost'], 'cpu_seconds': time.process_time()-started, 'no_model_adopted': True,
        'query_paid_outcomes_read': False, 'source6_7_evaluated': False, 'target_promoted': False, 'independent_confirmation': False}
    save(out/'results.json', result); print({'summary': summary, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
