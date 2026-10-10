"""Source-only joint common/task geometry fits and conditional v posteriors."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.run_legacy_related_inventory_source import save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis, spectral_weights
from representation.source_function_prior import SourceFunctionPrior, head_batches
from representation.source_joint_geometry import JointCommonTaskGeometryLikelihood
from representation.source_joint_population import GaussianPopulationLikelihood
from representation.task_level_reference import TaskLevelReferencePosterior

EXECUTION = 'performance/manifests/legacy_joint_population_f_joint_common_task_geometry_execution_v1.json'


def gradient_check(likelihood, spec):
    point = likelihood.initial.copy(); count = likelihood.common_count
    point[:count-2] *= spec['gradient_common_amplitude_multiplier']; point[count-2:count] = [2., .5]
    point[count:-3] += spec['gradient_task_group_variance_offset']; point[-3:] = [2., .1, .5]
    analytic = likelihood.evaluate(point)[1]; step = spec['gradient_step']; finite = []
    for j in range(len(point)):
        lower, upper = point.copy(), point.copy(); lower[j] -= step; upper[j] += step
        finite.append((likelihood.evaluate(upper)[0]-likelihood.evaluate(lower)[0])/(2*step))
    np.testing.assert_allclose(analytic, finite, **spec['gradient_tolerance'])
    native_value, native_gradient, _ = likelihood.evaluate(likelihood.initial)
    boundary = []
    for j in np.flatnonzero(likelihood.initial == 0):
        one, two = likelihood.initial.copy(), likelihood.initial.copy(); one[j] += step; two[j] += 2*step
        estimate = (4*likelihood.evaluate(one)[0]-likelihood.evaluate(two)[0]-3*native_value)/(2*step)
        np.testing.assert_allclose(estimate, native_gradient[j], **spec['boundary_gradient_tolerance'])
        boundary.append({'parameter': likelihood.names[j], 'analytic': float(native_gradient[j]),
            'one_sided_finite_difference': float(estimate)})
    return {'interior_maximum_absolute_error': float(np.max(np.abs(analytic-finite))),
        'interior_parameters': point.tolist(), 'native_zero_parameter_one_sided_checks': boundary}


def verify(likelihood, parameters):
    value, _, state = likelihood.evaluate(parameters); basis = likelihood.source.basis
    common = state['common_prior_coefficient_covariance']; task = state['residual_coefficient_covariance']
    for covariance in (common, task, state['population_mean_covariance']):
        assert np.linalg.eigvalsh(covariance).min() >= -1e-10
    assert common[0, 0] == likelihood.native['common_prior_coefficient_diagonal'][0]
    np.testing.assert_array_equal(common[0, 1:], 0.)
    common_weights = spectral_weights(basis, state['common_spectral_exponent'])[0]
    task_weights = spectral_weights(basis, state['spectral_exponent'])[0]; q = state['task_template_strength']
    np.testing.assert_allclose(np.diag(common)[1:], state['common_group_variances'][basis.group_index]*common_weights+likelihood.source.tail, atol=1e-14, rtol=1e-12)
    effective_groups = (1-q)*state['source_group_variances']+q*likelihood.task_budgets
    np.testing.assert_allclose(np.diag(task)[1:], effective_groups[basis.group_index]*task_weights+likelihood.source.tail, atol=1e-14, rtol=1e-12)
    for covariance, groups in ((common, likelihood.common.groups), (task, likelihood.task_groups)):
        inactive = 1+np.flatnonzero(~np.isin(basis.group_index, groups))
        np.testing.assert_array_equal(covariance[inactive], np.diag(np.diag(covariance))[inactive])
        np.testing.assert_allclose(np.diag(covariance)[inactive], likelihood.source.tail, atol=1e-14, rtol=1e-12)
    # Independent coefficient solves check dense conditioning with full A and R.
    precision0 = cho_solve(cho_factor(common, lower=True), np.eye(len(common)))
    precision, rhs, logdet = precision0.copy(), np.zeros(len(common)), np.linalg.slogdet(common)[1]
    for d, error, noise in likelihood.source.blocks:
        variance = d@task@d.T+np.diag(noise+likelihood.native['discrepancy_variance'])
        factor = cho_factor(variance, lower=True); solved = cho_solve(factor, np.column_stack([d, error]))
        precision += d.T@solved[:, :-1]; rhs += d.T@solved[:, -1]; logdet += 2*np.log(np.diag(factor[0])).sum()
    factor = cho_factor(precision, lower=True); h = cho_solve(factor, np.eye(len(common)))
    displacement = h@rhs; beta = likelihood.source.prior.mean+displacement
    quadratic = displacement@precision0@displacement
    for d, error, noise in likelihood.source.blocks:
        residual = error-d@displacement; variance = d@task@d.T+np.diag(noise+likelihood.native['discrepancy_variance'])
        quadratic += residual@cho_solve(cho_factor(variance, lower=True), residual)
    logdet += 2*np.log(np.diag(factor[0])).sum()
    component_value = .5*(len(likelihood.error)*np.log(2*np.pi)+logdet+quadratic)/len(likelihood.error)
    np.testing.assert_allclose(value, component_value, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(state['coefficient_mean'], beta, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(state['population_mean_covariance'], h, atol=1e-8, rtol=1e-8)
    return {'psd_sparse_support_frequency_marginals_independent_tail_and_intercept_verified': True,
        'dense_and_component_conditioning_verified': True, 'dense_component_nll_error': float(abs(value-component_value))}


def integrate(likelihood, law, deleted, spec):
    a = np.asarray(law['common_prior_coefficient_covariance']); base = np.asarray(law['task_covariance_at_zero_independent_intercept'])
    c0 = likelihood.source_covariance(a, base); d, error = likelihood.design, likelihood.error
    u = block_diag(*[np.ones((len(local), 1)) for local, _, _ in likelihood.source.blocks])
    factor = cho_factor(c0, lower=True); inverse_u = cho_solve(factor, u); alpha0 = cho_solve(factor, error)
    b = u.T@inverse_u; eigenvalues, vectors = np.linalg.eigh(.5*(b+b.T)); assert np.all(eigenvalues > 0)
    rhs = vectors.T@u.T@alpha0; update = a@d.T@inverse_u@vectors
    beta0 = likelihood.source.prior.mean+a@d.T@alpha0; h0 = a-a@d.T@cho_solve(factor, d)@a; h0 = .5*(h0+h0.T)
    nll0 = .5*(len(error)*np.log(2*np.pi)+2*np.log(np.diag(factor[0])).sum()+error@alpha0)
    v, signal = law['independent_task_intercept_variance'], likelihood.source.signal
    posterior = TaskLevelReferencePosterior(eigenvalues, rhs, signal, v, spec['quadrature'], signal*spec['quadrature_split_variance_over_signal'])
    def conditional(variance):
        weights = variance/(1+variance*eigenvalues)
        return beta0-update@(weights*rhs), h0+(update*weights)@update.T
    beta, h = conditional(v)
    np.testing.assert_allclose((nll0+posterior.nll_difference(v))/len(error), law['source_negative_log_likelihood_per_label'], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(beta, law['coefficient_mean'], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(h, law['population_mean_covariance'], atol=1e-8, rtol=1e-8)
    verification = {'fitted_source_nll_beta_H_reproduced_under_new_v_law': True}
    if deleted == spec['dense_v_verification_deleted_task']:
        interior = signal*spec['dense_v_verification_variance_over_source_signal']
        dense_factor = cho_factor(c0+interior*u@u.T, lower=True); dense_alpha = cho_solve(dense_factor, error)
        dense_nll = .5*(len(error)*np.log(2*np.pi)+2*np.log(np.diag(dense_factor[0])).sum()+error@dense_alpha)
        dense_b = u.T@cho_solve(dense_factor, u); expected_beta, expected_h = conditional(interior)
        np.testing.assert_allclose(nll0+posterior.nll_difference(interior), dense_nll, atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(posterior.information(interior), .5*np.trace(dense_b@dense_b), atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(expected_beta, likelihood.source.prior.mean+a@d.T@dense_alpha, atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(expected_h, a-a@d.T@cho_solve(dense_factor, d)@a, atol=1e-8, rtol=1e-8)
        verification.update(interior_dense_v_likelihood_Fisher_beta_H_verified=True)
    moments = posterior.moments(spec['posterior_quantile_probabilities'])
    average = moments.pop('expected_shrinkage'); shrinkage_covariance = moments.pop('shrinkage_covariance')
    integrated_beta = beta0-update@(average/eigenvalues*rhs)
    mean_update = update*(rhs/eigenvalues)
    integrated_h = h0+(update*(average/eigenvalues))@update.T+mean_update@shrinkage_covariance@mean_update.T
    integrated_h = .5*(integrated_h+integrated_h.T); assert np.linalg.eigvalsh(integrated_h).min() >= -1e-10
    reference = {'fitted_source_law': law, 'task_variance_eigenvalues': eigenvalues.tolist(), 'task_variance_rhs': rhs.tolist(),
        'coefficient_mean_at_zero_task_intercept_variance': beta0.tolist(), 'common_covariance_at_zero_task_intercept_variance': h0.tolist(),
        'common_coefficient_update_columns': update.tolist(), 'source_signal': signal,
        'fitted_independent_task_intercept_variance': v, 'task_covariance_at_zero_independent_intercept': base.tolist(),
        'reference_log_normalizer_relative_to_fitted_nll': float(posterior.log_scale+np.log(posterior.normalization)),
        'integrated_common_coefficient_mean': integrated_beta.tolist(), 'integrated_common_coefficient_covariance': integrated_h.tolist(),
        'integrated_independent_task_intercept_variance': moments['mean'],
        'integrated_total_task_level_coefficient_variance': float(base[0, 0]+moments['mean']),
        'point_predictions_not_a_coherent_mixture_replacement': True}
    record = {'reference_posterior': moments, 'coupled_template_level_variance': float(base[0, 0]),
        'scalar_quadrature_evaluations': posterior.scalar_evaluations, 'maximum_scalar_quadrature_error': posterior.maximum_scalar_error,
        'verification': verification, 'conditional_independent_v_integrated_without_floor_or_cutoff': True}
    return record, reference


def run():
    started = time.process_time(); spec = read(EXECUTION); coverage, profile = source_paths(read(spec['template_execution']))
    geometry = read(read(spec['common_execution'])['geometry_results'])['rows']; templates = read(spec['source_templates'])['rows']
    controls = read(spec['shape_reference_results'])['rows']; branches = []
    for saved in templates:
        layout, deleted = saved['layout'], saved['deleted_task']; suffix = Path(f'layout{layout}')/f'task{deleted}'
        inputs = read(Path(coverage['output'])/suffix/'inputs.json'); head = read(Path(profile['output'])/suffix/'selected_source_fit.json')['heads'][0]
        tasks = [{**t, **{key: np.asarray(t[key]) for key in ('psi', 'response', 'noise')}} for t in inputs['source_tasks']]
        assert {t['name'] for t in tasks} == {f'related_inventory_{i:02d}' for i in range(6) if i != deleted}
        native = head['joint_structural']; basis = OrthogonalLowOrderBasis(inputs['public_mean_reference'])
        prior = SourceFunctionPrior(basis, np.asarray(head['regularizer']['coefficient_mean']),
            np.diag(native['common_prior_coefficient_diagonal']), native['discrepancy_variance'], {'source_signal_scale': native['source_signal_scale']})
        source = GaussianPopulationLikelihood(prior, head_batches(tasks, 0), [t['noise'][:, 0] for t in tasks])
        assert source.label_count == 60 and len(tasks) == 5
        common_geometry = next(r for r in geometry if r['layout'] == layout and r['deleted_task'] == deleted and r['head'] == 'f')
        likelihood = JointCommonTaskGeometryLikelihood(source, native, common_geometry, saved)
        value, _, initial = likelihood.evaluate(likelihood.initial)
        for key in ('source_negative_log_likelihood_per_label', 'coefficient_mean', 'population_mean_covariance'):
            np.testing.assert_allclose(initial[key], native[key], atol=1e-8, rtol=1e-8)
        checks = {'native_endpoint_nll_beta_H_reproduced': True}
        if deleted == spec['gradient_verification_deleted_task']:
            checks['gradient'] = gradient_check(likelihood, spec)
        branches.append((layout, deleted, inputs, likelihood, checks))
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True); laws, rows = [], []
    for layout, deleted, inputs, likelihood, checks in branches:
        optimizer = inputs['execution']['optimizer']
        fitted = minimize(lambda p: likelihood.evaluate(p)[:2], likelihood.initial, jac=True,
            method=optimizer['method'], bounds=likelihood.bounds,
            options={'maxiter': optimizer['max_iterations'], 'ftol': optimizer['ftol'], 'gtol': optimizer['gtol']})
        value, gradient, state = likelihood.evaluate(fitted.x); projected = gradient.copy()
        for j, (lo, hi) in enumerate(likelihood.bounds):
            if (fitted.x[j] <= lo and gradient[j] > 0) or (hi is not None and fitted.x[j] >= hi and gradient[j] < 0): projected[j] = 0.
        checks['fitted'] = verify(likelihood, fitted.x)
        law = {key: val.tolist() if isinstance(val, np.ndarray) else val for key, val in state.items()}
        law.update(source_task_names=[t['name'] for t in inputs['source_tasks']], source_label_count=60, source_task_count=5,
            empirical_center_coefficient_mean=likelihood.source.prior.mean.tolist(), source_signal_scale=likelihood.source.signal,
            independent_tail_per_coefficient=likelihood.source.tail, discrepancy_variance=likelihood.native['discrepancy_variance'],
            optimizer=optimizer, optimizer_parameter_names=likelihood.names, optimizer_parameters=fitted.x.tolist(),
            optimizer_initial_parameters=likelihood.initial.tolist(), optimizer_converged=bool(fitted.success),
            optimizer_message=str(fitted.message), optimizer_iterations=int(fitted.nit), optimizer_evaluations=int(fitted.nfev),
            optimizer_projected_gradient_max=float(np.max(np.abs(projected))))
        gain = likelihood.native['source_negative_log_likelihood_per_label']-value
        assert gain >= -1e-10
        control = next(r for r in controls if r['layout'] == layout and r['deleted_task'] == deleted)
        row = {'layout': layout, 'deleted_task': deleted, 'converged': bool(fitted.success), 'iterations': int(fitted.nit),
            'p': state['common_correlation_strength'], 'q': state['task_template_strength'],
            'common_spectral_exponent': state['common_spectral_exponent'], 'task_spectral_exponent': state['spectral_exponent'],
            'fitted_independent_task_intercept_variance': state['independent_task_intercept_variance'],
            'total_source_log_likelihood_gain_over_native': 60*gain,
            'total_source_log_likelihood_gain_over_stage129': 60*(gain-control['source_log_likelihood_gain_per_label']),
            'verification': checks}
        rows.append(row); laws.append({'layout': layout, 'deleted_task': deleted, **law})
        print('joint common/task source fit', layout, deleted, {'converged':row['converged'], 'iterations':row['iterations'], 'p':row['p'], 'q':row['q'], 'gain':row['total_source_log_likelihood_gain_over_native']}, flush=True)
        if not fitted.success:
            save(out/'fitted_source_laws.json', {'execution':spec, 'all12_source_fits_frozen':False, 'rows':laws})
            save(out/'results.json', {'execution':spec, 'rows':rows, 'all_fits_complete':False, 'stopped_on_optimizer_failure':True,
                'cpu_seconds':time.process_time()-started, 'no_model_adopted':True, 'query_paid_outcomes_read':False})
            raise RuntimeError(f'Original-contract joint optimizer failed on {layout}/{deleted}: {fitted.message}')
    save(out/'fitted_source_laws.json', {'execution':spec, 'all12_source_fits_frozen_before_v_integration_or_queries':True, 'rows':laws})
    references = []
    for branch, law, row in zip(branches, laws, rows):
        layout, deleted, inputs, likelihood, _ = branch
        record, reference = integrate(likelihood, law, deleted, spec); row['integration'] = record
        references.append({'layout':layout, 'deleted_task':deleted, **reference})
    save(out/'source_reference_laws.json', {'execution':spec, 'all12_conditional_source_v_laws_frozen':True, 'rows':references})
    summary = {layout:{'mean_total_source_log_likelihood_gain_over_native':float(np.mean([r['total_source_log_likelihood_gain_over_native'] for r in rows if r['layout']==layout])),
        'mean_total_source_log_likelihood_gain_over_stage129':float(np.mean([r['total_source_log_likelihood_gain_over_stage129'] for r in rows if r['layout']==layout])),
        'fitted_q':[r['q'] for r in rows if r['layout']==layout],
        'task1':next(r for r in rows if r['layout']==layout and r['deleted_task']==1)} for layout in ('5','6')}
    result = {'execution':spec, 'rows':rows, 'summary':summary, 'all12_source_fits_and_v_integrations_verified':True,
        'cost':{'new_f_source_fits':12, 'new_source_v_integrations':12, 'new_forecasts':0, 'new_simulator_calls':0, 'new_verification_calls':0, 'new_terminal_integrations':0},
        'cpu_seconds':time.process_time()-started, 'no_model_adopted':True, 'query_paid_outcomes_read':False,
        'source6_7_evaluated':False, 'target_promoted':False, 'independent_confirmation':False}
    save(out/'results.json', result)
    print({'summary':{k:{key:val for key,val in v.items() if key!='task1'} for k,v in summary.items()}, 'cpu_seconds':result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
