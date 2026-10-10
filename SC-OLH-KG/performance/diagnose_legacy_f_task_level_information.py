"""Source-only f task-level variance slice and covariance nuisance information."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve, solve_triangular
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_joint_common_covariance import restore_regularizer
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.run_legacy_related_inventory_source import save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis, spectral_weights
from representation.source_function_prior import head_batches
from representation.source_joint_population import JointCovariancePopulationLikelihood

EXECUTION = 'performance/manifests/legacy_joint_population_f_task_level_information_execution_v1.json'


def diagnose(likelihood, law, reference, spec):
    parameters = np.asarray(law['optimizer_parameters']); value, gradient, state = likelihood.evaluate(parameters)
    for key, saved in (('mean', 'coefficient_mean'), ('mean_covariance', 'population_mean_covariance'),
        ('common_diagonal', 'common_prior_coefficient_diagonal'), ('residual_diagonal', 'residual_coefficient_diagonal')):
        np.testing.assert_allclose(state[key], law[saved], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(value, law['source_negative_log_likelihood_per_label'], atol=1e-8, rtol=1e-8)
    designs = [b[0] for b in likelihood.blocks]; design = np.vstack(designs)
    error = np.concatenate([b[1] for b in likelihood.blocks]); n = len(error)
    a = np.diag(state['common_diagonal']); r = state['residual_diagonal']; v = float(r[0])
    covariance = design@a@design.T+block_diag(*[(d*r)@d.T+np.diag(b[2]+state['discrepancy'])
        for d, b in zip(designs, likelihood.blocks)])
    factor = cho_factor(covariance, lower=True); inverse = cho_solve(factor, np.eye(n)); alpha = inverse@error
    dense_value = .5*(n*np.log(2*np.pi)+2*np.log(np.diag(factor[0])).sum()+error@alpha)/n
    dense_mean = likelihood.prior.mean+a@design.T@alpha
    dense_h = a-a@design.T@inverse@design@a
    np.testing.assert_allclose(value, dense_value, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(state['mean'], dense_mean, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(state['mean_covariance'], dense_h, atol=1e-8, rtol=1e-8)
    membership = block_diag(*[np.ones((len(d), 1)) for d in designs]); direction = membership@membership.T
    score = .5*np.sum((inverse-np.outer(alpha, alpha))*direction)
    product = inverse@direction
    curvature = -.5*np.trace(product@product)+alpha@direction@inverse@direction@alpha
    index = len(likelihood.basis.group_names)+1
    np.testing.assert_allclose(score*likelihood.signal/n, gradient[index], atol=1e-8, rtol=1e-7)

    # Whiten covariance derivatives so nuisance projection uses physical spans,
    # without ill-conditioning caused solely by log-variance parameter units.
    lower = np.tril(factor[0])
    def whiten(matrix):
        left = solve_triangular(lower, matrix, lower=True)
        return solve_triangular(lower, left.T, lower=True).T.ravel()

    common_weights, common_derivative = spectral_weights(likelihood.basis, state['common_spectral_exponent'])
    task_weights, task_derivative = spectral_weights(likelihood.basis, state['spectral_exponent'])
    names, matrices = [], []
    def common_matrix(diagonal): return (design*diagonal)@design.T
    def task_matrix(diagonal): return block_diag(*[(d*diagonal)@d.T for d in designs])
    for kind, groups, weights, derivative, matrix in (
        ('common', state['common_group_variances'], common_weights, common_derivative, common_matrix),
        ('task', state['group_variances'], task_weights, task_derivative, task_matrix)):
        for group, amplitude in enumerate(groups):
            if amplitude > 0:
                names.append(f'{kind}:log_group_variance:{likelihood.basis.group_names[group]}')
                matrices.append(matrix(np.r_[0., amplitude*weights*(likelihood.basis.group_index == group)]))
        diagonal = np.r_[0., groups[likelihood.basis.group_index]*derivative]
        if np.any(diagonal != 0):
            names.append(f'{kind}:spectral_exponent'); matrices.append(matrix(diagonal))
    names.append('task:physical_discrepancy_variance'); matrices.append(np.eye(n))
    flat = np.asarray([whiten(m) for m in matrices]); norms = np.linalg.norm(flat, axis=1)
    assert np.all(norms > 0)
    unit = flat/norms[:, None]; intercept_flat = whiten(direction)
    coefficients, _, rank, singular_values = np.linalg.lstsq(unit.T, intercept_flat, rcond=None)
    residual = intercept_flat-unit.T@coefficients
    conditional_information = float(.5*intercept_flat@intercept_flat)
    efficient_information = float(.5*residual@residual)
    normalized = np.vstack([intercept_flat/np.linalg.norm(intercept_flat), unit])
    standardized_information = normalized@normalized.T

    # C(v)=C(0)+v UU': exact likelihood differences need only T source directions.
    zero_covariance = covariance-v*direction; zero_factor = cho_factor(zero_covariance, lower=True)
    s = membership.T@cho_solve(zero_factor, membership); eigenvalues, vectors = np.linalg.eigh(s)
    assert np.all(eigenvalues > 0)
    rhs = vectors.T@membership.T@cho_solve(zero_factor, error); evaluations = 0
    def loss(amplitude):
        nonlocal evaluations
        evaluations += 1; variance = amplitude*likelihood.signal
        return float(.5*np.sum(np.log1p(variance*eigenvalues)-np.log1p(v*eigenvalues)
            -(variance/(1+variance*eigenvalues)-v/(1+v*eigenvalues))*rhs**2))

    point = spec['derivative_verification_interior_signal_scaled_variance']
    epsilon = spec['derivative_verification_signal_scaled_epsilon']
    interior = parameters.copy(); interior[index] = point
    actual, interior_gradient, _ = likelihood.evaluate(interior)
    expected_loss = n*(actual-value); low_rank_loss = loss(point)
    np.testing.assert_allclose(low_rank_loss, expected_loss, atol=1e-7, rtol=1e-8)
    dense_factor = cho_factor(covariance+(point*likelihood.signal-v)*direction, lower=True)
    dense_interior = .5*(n*np.log(2*np.pi)+2*np.log(np.diag(dense_factor[0])).sum()+error@cho_solve(dense_factor, error))/n
    np.testing.assert_allclose(actual, dense_interior, atol=1e-8, rtol=1e-8)
    finite = (loss(point+epsilon)-loss(point-epsilon))/(2*epsilon*n)
    np.testing.assert_allclose(finite, interior_gradient[index], atol=1e-7, rtol=1e-5)

    fitted = v/likelihood.signal; drop = spec['likelihood_drop']; at_zero = loss(0.)
    tolerance = spec['root_absolute_tolerance_on_signal_scaled_amplitude']
    lo = 0. if at_zero <= drop else brentq(lambda q: loss(q)-drop, 0., fitted, xtol=tolerance)
    bracket = fitted+1.
    while loss(bracket) < drop:
        bracket = 2*bracket+1.
    hi = brentq(lambda q: loss(q)-drop, fitted, bracket, xtol=tolerance)
    phi = likelihood.basis.transform(reference); public_mean = np.r_[1., phi.mean(axis=0)]
    task_level = float((public_mean**2)@r)
    common_level = float(public_mean@state['mean_covariance']@public_mean)
    return {'source_label_count': n, 'source_task_count': len(designs), 'source_negative_log_likelihood_per_label': value,
        'native_task_intercept_variance': v, 'source_signal_scale': likelihood.signal,
        'conditional_total_nll_derivative_at_native_variance': float(score),
        'conditional_total_nll_observed_curvature_at_native_variance': float(curvature),
        'conditional_fisher_information_per_physical_variance_squared': conditional_information,
        'nuisance_adjusted_fisher_information_per_physical_variance_squared': efficient_information,
        'nuisance_adjusted_information_fraction': efficient_information/conditional_information,
        'conditional_local_variance_scale_from_information': 1/np.sqrt(conditional_information),
        'nuisance_adjusted_local_variance_scale_from_information': 1/np.sqrt(efficient_information) if efficient_information > 0 else None,
        'nuisance_parameter_names': names, 'nuisance_whitened_derivative_norms': norms.tolist(),
        'nuisance_derivative_rank': int(rank), 'nuisance_normalized_derivative_singular_values': singular_values.tolist(),
        'standardized_information_eigenvalues': np.linalg.eigvalsh(standardized_information).tolist(),
        'standardized_information_rank': int(np.linalg.matrix_rank(standardized_information)),
        'native_common_exponent': state['common_spectral_exponent'], 'native_task_exponent': state['spectral_exponent'],
        'native_discrepancy_variance': state['discrepancy'],
        'conditional_source_likelihood_support_variances': [float(lo*likelihood.signal), float(hi*likelihood.signal)],
        'conditional_source_likelihood_support_standard_deviations': [float(np.sqrt(lo*likelihood.signal)), float(np.sqrt(hi*likelihood.signal))],
        'zero_inside_conditional_likelihood_support': bool(at_zero <= drop), 'conditional_total_nll_increase_at_zero': at_zero,
        'conditional_support_width_over_native_intercept_variance': (hi-lo)*likelihood.signal/v if v > 0 else None,
        'public_bank_average_task_variance': task_level,
        'public_bank_average_shape_variance': task_level-v,
        'public_bank_average_common_posterior_variance': common_level,
        'public_bank_average_white_variance': state['discrepancy']/len(reference),
        'source_likelihood_evaluations': evaluations,
        'verification': {'native_dense_nll_mean_covariance_and_intercept_score_passed': True,
            'native_dense_nll_error': float(abs(value-dense_value)),
            'low_rank_slice_interior_total_loss_error': float(abs(low_rank_loss-expected_loss)),
            'signal_scaled_derivative_finite_difference_error': float(abs(finite-interior_gradient[index]))}}


def run():
    started = time.process_time(); spec = read(EXECUTION); centering = read(spec['centering_execution'])
    coverage = read(centering['coverage_execution']); profile = read(centering['profile_execution'])
    native_full = read(spec['native_full_execution']); rows = []
    for layout in spec['layouts']:
        folders = [(None, Path(coverage['output'])/f'layout{layout}', Path(native_full['output'])/f'layout{layout}')]
        folders += [(i, Path(coverage['output'])/f'layout{layout}'/f'task{i}', Path(profile['output'])/f'layout{layout}'/f'task{i}')
            for i in spec['training_tasks']]
        for deleted, inputs_folder, fit_folder in folders:
            inputs = read(inputs_folder/'inputs.json'); means = read(fit_folder/'selected_source_fit.json')
            native = means['heads'][0]['joint_structural']; reference = np.asarray(inputs['public_mean_reference'])
            basis = OrthogonalLowOrderBasis(reference)
            tasks = [{**t, **{k: np.asarray(t[k]) for k in ('psi', 'features', 'response', 'noise')}} for t in inputs['source_tasks']]
            prior = restore_regularizer(basis, means['heads'][0]['regularizer'], tasks, 0)
            likelihood = JointCovariancePopulationLikelihood(prior, head_batches(tasks, 0), [t['noise'][:, 0] for t in tasks])
            rows.append({'layout': layout, 'deleted_task': deleted, 'head': 'f', **diagnose(likelihood, native, reference, spec)})
        print('f task-level source identification complete', layout, flush=True)
    result = {'execution': spec, 'rows': rows, 'all14_native_laws_and_slice_derivatives_verified': len(rows) == 14,
        'zero_inside_conditional_support_count': sum(r['zero_inside_conditional_likelihood_support'] for r in rows),
        'cost': {**spec['cost'], 'source_likelihood_evaluations': sum(r['source_likelihood_evaluations'] for r in rows)},
        'cpu_seconds': time.process_time()-started, 'no_prior_or_model_selected': True,
        'query_paid_outcomes_read': False, 'source_holdouts_evaluated': False, 'target_promoted': False, 'independent_confirmation': False}
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True); save(out/'results.json', result)
    print({'task1_rows': [r for r in rows if r['deleted_task'] == 1],
        'zero_inside_conditional_support_count': result['zero_inside_conditional_support_count'],
        'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
