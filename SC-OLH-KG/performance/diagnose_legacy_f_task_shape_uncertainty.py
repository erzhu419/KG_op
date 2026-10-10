"""Conditional source uncertainty for one native task-shape variance group."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve, solve_triangular
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.run_legacy_related_inventory_source import save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis, spectral_weights
from representation.task_level_reference import TaskLevelReferencePosterior

EXECUTION = 'performance/manifests/legacy_joint_population_f_task_shape_uncertainty_execution_v1.json'


def source_system(inputs, head, deleted):
    basis = OrthogonalLowOrderBasis(inputs['public_mean_reference']); native = head['joint_structural']
    tasks = inputs['source_tasks']
    assert {t['name'] for t in tasks} == {f'related_inventory_{i:02d}' for i in range(6) if i != deleted}
    designs = [np.column_stack([np.ones(len(t['psi'])), basis.transform(np.asarray(t['psi']))]) for t in tasks]
    design = np.vstack(designs); m0 = np.asarray(head['regularizer']['coefficient_mean'])
    error = np.concatenate([np.asarray(t['response'])[:, 0] for t in tasks])-design@m0
    assert len(tasks) == 5 and len(error) == 60
    common = np.diag(native['common_prior_coefficient_diagonal'])
    residual = np.asarray(native['residual_coefficient_diagonal'])
    group = int(np.argmax(native['source_group_variances']))
    variance = float(native['source_group_variances'][group]); assert variance > 0
    features = np.flatnonzero(basis.group_index == group); columns = 1+features
    weights = spectral_weights(basis, native['spectral_exponent'])[0][features]
    unit = np.zeros(len(residual)); unit[columns] = weights
    r0 = residual-variance*unit
    tail = native['tail_variance_budget']/basis.feature_dim
    np.testing.assert_allclose(r0[columns], tail, atol=1e-14, rtol=1e-12)
    np.testing.assert_array_equal(r0[np.setdiff1d(np.arange(len(r0)), columns)],
        residual[np.setdiff1d(np.arange(len(r0)), columns)])
    u = block_diag(*[d[:, columns]*np.sqrt(weights) for d in designs])
    c0 = design@common@design.T+block_diag(*[(d*r0)@d.T+np.diag(np.asarray(t['noise'])[:, 0]+native['discrepancy_variance'])
        for d, t in zip(designs, tasks)])
    factor = cho_factor(c0, lower=True); inverse_u = cho_solve(factor, u); alpha0 = cho_solve(factor, error)
    whitened = solve_triangular(np.tril(factor[0]), u, lower=True)
    _, singular, right = np.linalg.svd(whitened, full_matrices=False)
    rank = int(np.sum(singular > max(whitened.shape)*np.finfo(float).eps*singular[0]))
    assert rank > 4, 'The full-tail second variance moment is not integrable at this source rank.'
    vectors = right[:rank].T; eigenvalues = singular[:rank]**2
    reconstruction = u@vectors@vectors.T@u.T
    np.testing.assert_allclose(reconstruction, u@u.T, atol=1e-11, rtol=1e-11)
    rhs = vectors.T@u.T@alpha0; update = common@design.T@inverse_u@vectors
    beta0 = m0+common@design.T@alpha0
    h0 = common-common@design.T@cho_solve(factor, design)@common; h0 = .5*(h0+h0.T)
    nll0 = .5*(len(error)*np.log(2*np.pi)+2*np.log(np.diag(factor[0])).sum()+error@alpha0)
    public = np.column_stack([np.ones(len(inputs['public_mean_reference'])), basis.transform(np.asarray(inputs['public_mean_reference']))])
    return dict(basis=basis, native=native, tasks=tasks, design=design, common=common, m0=m0, error=error,
        group=group, variance=variance, columns=columns, weights=weights, unit=unit, r0=r0, u=u, c0=c0,
        eigenvalues=eigenvalues, rhs=rhs, update=update, beta0=beta0, h0=h0, nll0=nll0, rank=rank, public=public)


def likelihood_slice(system, spec):
    eigenvalues, rhs = system['eigenvalues'], system['rhs']
    def nll(v):
        return float(.5*np.sum(np.log1p(v*eigenvalues)-v/(1+v*eigenvalues)*rhs**2))
    variance = system['variance']; signal = system['native']['source_signal_scale']
    def boundary(x):
        return nll(signal*x)-nll(variance)-spec['source_likelihood_slice_drop']
    upper = variance/signal+1.
    while boundary(upper) < 0:
        upper *= 2
    upper = signal*brentq(boundary, variance/signal, upper, xtol=spec['slice_root_tolerance_in_signal_units'])
    lower = 0. if boundary(0.) <= 0 else signal*brentq(boundary, 0., variance/signal,
        xtol=spec['slice_root_tolerance_in_signal_units'])
    return [lower, upper]


def diagnose(inputs, head, intercept_record, deleted, spec):
    s = source_system(inputs, head, deleted); native = s['native']; signal = native['source_signal_scale']
    support = likelihood_slice(s, spec)
    posterior = TaskLevelReferencePosterior(s['eigenvalues'], s['rhs'], signal, s['variance'], spec['quadrature'], support[1])
    def conditional(v):
        weight = v/(1+v*s['eigenvalues'])
        return s['beta0']-s['update']@(weight*s['rhs']), s['h0']+(s['update']*weight)@s['update'].T
    native_beta, native_h = conditional(s['variance'])
    np.testing.assert_allclose((s['nll0']+posterior.nll_difference(s['variance']))/len(s['error']),
        native['source_negative_log_likelihood_per_label'], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(native_beta, native['coefficient_mean'], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(native_h, native['population_mean_covariance'], atol=1e-8, rtol=1e-8)
    verification = {'native_nll_beta_h_and_unchanged_group_spectrum_tail_reproduced': True,
        'identifiable_source_rank': s['rank'], 'full_tail_first_and_second_moments_integrable': s['rank'] > 4}
    if deleted == spec['dense_verification_deleted_task']:
        interior = signal*spec['dense_verification_variance_over_source_signal']
        factor = cho_factor(s['c0']+interior*s['u']@s['u'].T, lower=True); alpha = cho_solve(factor, s['error'])
        nll = .5*(len(alpha)*np.log(2*np.pi)+2*np.log(np.diag(factor[0])).sum()+s['error']@alpha)
        beta = s['m0']+s['common']@s['design'].T@alpha
        h = s['common']-s['common']@s['design'].T@cho_solve(factor, s['design'])@s['common']
        b = s['u'].T@cho_solve(factor, s['u']); information = .5*np.trace(b@b)
        expected_beta, expected_h = conditional(interior)
        np.testing.assert_allclose(s['nll0']+posterior.nll_difference(interior), nll, atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(posterior.information(interior), information, atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(expected_beta, beta, atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(expected_h, h, atol=1e-8, rtol=1e-8)
        verification.update(interior_total_nll_error=float(abs(s['nll0']+posterior.nll_difference(interior)-nll)),
            interior_fisher_relative_error=float(abs(posterior.information(interior)/information-1.)))
    moments = posterior.moments(spec['posterior_quantile_probabilities'])
    average = moments.pop('expected_shrinkage'); covariance = moments.pop('shrinkage_covariance')
    mean_update = s['update']*(s['rhs']/s['eigenvalues'])
    integrated_beta = s['beta0']-mean_update@average
    mean_uncertainty = mean_update@covariance@mean_update.T
    integrated_h = s['h0']+(s['update']*(average/s['eigenvalues']))@s['update'].T+mean_uncertainty
    integrated_h = .5*(integrated_h+integrated_h.T); assert np.linalg.eigvalsh(integrated_h).min() >= -1e-10
    public = s['public']; centered = public-public.mean(axis=0)
    shape_unit = float(np.mean((centered**2)@s['unit']))
    def bank_variance(matrix, design):
        return float(np.mean(np.einsum('ij,jk,ik->i', design, matrix, design)))
    native_shape = float(np.mean((centered**2)@np.asarray(native['residual_coefficient_diagonal'])))
    added_shape = (moments['mean']-s['variance'])*shape_unit
    mass_above_slice = 1-posterior.cdf(support[1]/(signal+support[1]))
    record = {'source_task_count': 5, 'source_label_count': 60, 'selected_group': s['basis'].group_names[s['group']],
        'selected_group_index': s['group'], 'native_group_variance': s['variance'], 'source_signal_scale': signal,
        'reference_posterior': moments, 'posterior_mean_over_native_variance': moments['mean']/s['variance'],
        'posterior_sd_over_mean': moments['standard_deviation']/moments['mean'],
        'conditional_source_likelihood_support_variances': support, 'posterior_mass_above_slice_upper': float(mass_above_slice),
        'new_task_centered_shape_variance': {'native': native_shape, 'integrated': native_shape+added_shape,
            'relative_change': added_shape/native_shape, 'selected_group_unit_budget': shape_unit},
        'common_mean_bank_rms_change': float(np.sqrt(np.mean((public@(integrated_beta-native_beta))**2))),
        'common_posterior_average_bank_variance': {'native': bank_variance(native_h, public), 'integrated': bank_variance(integrated_h, public)},
        'common_posterior_centered_shape_variance': {'native': bank_variance(native_h, centered), 'integrated': bank_variance(integrated_h, centered)},
        'common_posterior_bank_variance_from_conditional_mean_uncertainty': bank_variance(mean_uncertainty, public),
        'existing_conditional_intercept_diagnostic': {'native_variance': intercept_record['native_task_intercept_variance'],
            'posterior_mean': intercept_record['reference_posterior']['mean'],
            'posterior_sd_over_mean': intercept_record['reference_posterior']['standard_deviation']/intercept_record['reference_posterior']['mean']},
        'verification': verification, 'scalar_quadrature_evaluations': posterior.scalar_evaluations,
        'conditional_weights_not_combined_with_intercept_weights': True}
    law = {'source_task_names': [t['name'] for t in s['tasks']], 'native_law': native,
        'selected_group_index': s['group'], 'task_group_unit_coefficient_diagonal': s['unit'].tolist(),
        'residual_diagonal_without_selected_group_excess': s['r0'].tolist(),
        'shape_variance_eigenvalues': s['eigenvalues'].tolist(), 'shape_variance_rhs': s['rhs'].tolist(),
        'coefficient_mean_at_zero_selected_group_variance': s['beta0'].tolist(),
        'common_covariance_at_zero_selected_group_variance': s['h0'].tolist(),
        'common_coefficient_update_columns': s['update'].tolist(), 'source_signal': signal,
        'native_group_variance': s['variance'], 'integrated_group_variance': moments['mean'],
        'reference_log_normalizer_relative_to_native_nll': float(posterior.log_scale+np.log(posterior.normalization)),
        'integrated_common_coefficient_mean': integrated_beta.tolist(), 'integrated_common_coefficient_covariance': integrated_h.tolist(),
        'conditional_information_diagnostic_only': True}
    return record, law


def run():
    started = time.process_time(); spec = read(EXECUTION); coverage, profile = source_paths(read(spec['source_context_execution']))
    intercept = read(spec['intercept_uncertainty_results'])['rows']; rows, laws = [], []
    for layout in spec['layouts']:
        for deleted in spec['training_tasks']:
            suffix = Path(f'layout{layout}')/f'task{deleted}'
            inputs = read(Path(coverage['output'])/suffix/'inputs.json')
            head = read(Path(profile['output'])/suffix/'selected_source_fit.json')['heads'][0]
            old = next(r for r in intercept if r['layout'] == layout and r['deleted_task'] == deleted)
            record, law = diagnose(inputs, head, old, deleted, spec)
            rows.append({'layout': layout, 'deleted_task': deleted, **record}); laws.append({'layout': layout, 'deleted_task': deleted, **law})
            print('conditional source shape uncertainty complete', layout, deleted, flush=True)
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    save(out/'source_laws.json', {'execution': spec, 'rows': laws})
    summary = {layout: {'mean_posterior_variance_over_native': float(np.mean([r['posterior_mean_over_native_variance'] for r in rows if r['layout'] == layout])),
        'mean_new_task_shape_variance_relative_change': float(np.mean([r['new_task_centered_shape_variance']['relative_change'] for r in rows if r['layout'] == layout]))}
        for layout in spec['layouts']}
    result = {'execution': spec, 'rows': rows, 'summary': summary, 'all12_source_integrations_verified': len(rows) == 12,
        'cost': spec['cost'], 'cpu_seconds': time.process_time()-started, 'no_prior_or_model_adopted': True,
        'query_paid_outcomes_read': False, 'source6_7_evaluated': False, 'target_promoted': False, 'independent_confirmation': False}
    save(out/'results.json', result)
    print({'summary': summary, 'task1': [r for r in rows if r['deleted_task'] == 1], 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
