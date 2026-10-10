"""Source-only integration of native f task-level variance uncertainty."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.run_legacy_related_inventory_source import save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.task_level_reference import TaskLevelReferencePosterior

EXECUTION = 'performance/manifests/legacy_joint_population_f_task_level_reference_integration_execution_v1.json'


def diagnose(inputs, head, previous, deleted, spec):
    basis = OrthogonalLowOrderBasis(inputs['public_mean_reference']); native = head['joint_structural']
    tasks = inputs['source_tasks']; m0 = np.asarray(head['regularizer']['coefficient_mean'])
    designs = [np.column_stack([np.ones(len(t['psi'])), basis.transform(np.asarray(t['psi']))]) for t in tasks]
    design = np.vstack(designs); error = np.concatenate([np.asarray(t['response'])[:, 0] for t in tasks])-design@m0
    a = np.diag(native['common_prior_coefficient_diagonal']); r = np.asarray(native['residual_coefficient_diagonal'])
    variance, signal = float(r[0]), native['source_signal_scale']; r0 = r.copy(); r0[0] = 0.
    c0 = design@a@design.T+block_diag(*[(d*r0)@d.T+np.diag(np.asarray(t['noise'])[:, 0]+native['discrepancy_variance'])
        for d, t in zip(designs, tasks)])
    u = block_diag(*[np.ones((len(d), 1)) for d in designs]); factor = cho_factor(c0, lower=True)
    inverse_u = cho_solve(factor, u); alpha0 = cho_solve(factor, error)
    b = u.T@inverse_u; eigenvalues, vectors = np.linalg.eigh(.5*(b+b.T)); assert np.all(eigenvalues > 0)
    rhs = vectors.T@u.T@alpha0; update = a@design.T@inverse_u@vectors
    beta0 = m0+a@design.T@alpha0; h0 = a-a@design.T@cho_solve(factor, design)@a; h0 = .5*(h0+h0.T)
    nll0 = .5*(len(error)*np.log(2*np.pi)+2*np.log(np.diag(factor[0])).sum()+error@alpha0)
    slice_upper = previous['conditional_source_likelihood_support_variances'][1]
    posterior = TaskLevelReferencePosterior(eigenvalues, rhs, signal, variance, spec['quadrature'], slice_upper)

    def conditional(v):
        weight = v/(1+v*eigenvalues)
        return beta0-update@(weight*rhs), h0+(update*weight)@update.T

    native_beta, native_h = conditional(variance)
    np.testing.assert_allclose((nll0+posterior.nll_difference(variance))/len(error),
        native['source_negative_log_likelihood_per_label'], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(native_beta, native['coefficient_mean'], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(native_h, native['population_mean_covariance'], atol=1e-8, rtol=1e-8)
    verification = {'native_source_nll_and_common_moments_reproduced': True}
    if deleted == spec['dense_verification_deleted_task']:
        interior = signal*spec['dense_verification_variance_over_source_signal']
        dense_factor = cho_factor(c0+interior*u@u.T, lower=True); dense_alpha = cho_solve(dense_factor, error)
        dense_nll = .5*(len(error)*np.log(2*np.pi)+2*np.log(np.diag(dense_factor[0])).sum()+error@dense_alpha)
        dense_beta = m0+a@design.T@dense_alpha
        dense_h = a-a@design.T@cho_solve(dense_factor, design)@a
        dense_b = u.T@cho_solve(dense_factor, u); dense_information = .5*np.trace(dense_b@dense_b)
        expected_beta, expected_h = conditional(interior)
        np.testing.assert_allclose(nll0+posterior.nll_difference(interior), dense_nll, atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(posterior.information(interior), dense_information, atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(expected_beta, dense_beta, atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(expected_h, dense_h, atol=1e-8, rtol=1e-8)
        verification.update(interior_physical_variance=interior, interior_total_nll_error=float(abs(nll0+posterior.nll_difference(interior)-dense_nll)),
            interior_fisher_relative_error=float(abs(posterior.information(interior)/dense_information-1.)))
    moments = posterior.moments(spec['posterior_quantile_probabilities'])
    probability_above_slice = 1-posterior.cdf(slice_upper/(signal+slice_upper))
    average_shrinkage = moments.pop('expected_shrinkage'); shrinkage_covariance = moments.pop('shrinkage_covariance')
    native_shrinkage = variance*eigenvalues/(1+variance*eigenvalues)
    mean_update = update*(rhs/eigenvalues)
    h_update = update/np.sqrt(eigenvalues)
    integrated_beta = native_beta-mean_update@(average_shrinkage-native_shrinkage)
    mean_uncertainty = mean_update@shrinkage_covariance@mean_update.T
    average_h = native_h+(h_update*(average_shrinkage-native_shrinkage))@h_update.T
    integrated_h = average_h+mean_uncertainty; integrated_h = .5*(integrated_h+integrated_h.T)
    assert np.linalg.eigvalsh(integrated_h).min() >= -1e-10
    public_design = np.column_stack([np.ones(len(inputs['public_mean_reference'])), basis.transform(np.asarray(inputs['public_mean_reference']))])
    level = public_design.mean(axis=0)
    record = {'source_task_count': len(tasks), 'source_label_count': len(error), 'native_task_intercept_variance': variance,
        'source_signal_scale': signal, 'reference_posterior': moments,
        'posterior_task_intercept_standard_deviation': float(np.sqrt(moments['mean'])),
        'mean_variance_over_native': moments['mean']/variance if variance > 0 else None,
        'stage105_slice_upper_variance': slice_upper, 'posterior_mass_above_stage105_slice_upper': float(probability_above_slice),
        'common_mean_bank_rms_change': float(np.sqrt(np.mean((public_design@(integrated_beta-native_beta))**2))),
        'common_posterior_average_bank_variance': {'native': float(np.trace(public_design@native_h@public_design.T)/len(public_design)),
            'integrated': float(np.trace(public_design@integrated_h@public_design.T)/len(public_design))},
        'public_average_common_level_variance': {'native': float(level@native_h@level), 'integrated': float(level@integrated_h@level)},
        'public_average_common_level_variance_from_conditional_mean_uncertainty': float(level@mean_uncertainty@level),
        'scalar_quadrature_evaluations': posterior.scalar_evaluations, 'maximum_scalar_quadrature_error': posterior.maximum_scalar_error,
        'verification': verification, 'conditional_posterior_integrated_without_variance_floor_or_cutoff': True}
    law = {'source_task_names': [t['name'] for t in tasks], 'native_law': native,
        'task_variance_eigenvalues': eigenvalues.tolist(), 'task_variance_rhs': rhs.tolist(),
        'coefficient_mean_at_zero_task_intercept_variance': beta0.tolist(), 'common_covariance_at_zero_task_intercept_variance': h0.tolist(),
        'common_coefficient_update_columns': update.tolist(), 'source_signal': signal, 'native_task_intercept_variance': variance,
        'reference_log_normalizer_relative_to_native_nll': float(posterior.log_scale+np.log(posterior.normalization)),
        'integrated_common_coefficient_mean': integrated_beta.tolist(), 'integrated_common_coefficient_covariance': integrated_h.tolist(),
        'integrated_task_intercept_variance': moments['mean'], 'point_predictions_not_a_coherent_mixture_replacement': True}
    return record, law


def run():
    started = time.process_time(); spec = read(EXECUTION); centering = read(spec['centering_execution'])
    coverage = read(centering['coverage_execution']); profile = read(centering['profile_execution'])
    full = read(spec['native_full_execution']); slices = read(spec['level_information_results'])['rows']; rows, laws = [], []
    for layout in spec['layouts']:
        for deleted in [None]+spec['training_tasks']:
            suffix = Path(f'layout{layout}') if deleted is None else Path(f'layout{layout}')/f'task{deleted}'
            inputs = read(Path(coverage['output'])/suffix/'inputs.json')
            head = read(Path(full['output'] if deleted is None else profile['output'])/suffix/'selected_source_fit.json')['heads'][0]
            previous = next(r for r in slices if r['layout'] == layout and r['deleted_task'] == deleted)
            record, law = diagnose(inputs, head, previous, deleted, spec)
            rows.append({'layout': layout, 'deleted_task': deleted, **record}); laws.append({'layout': layout, 'deleted_task': deleted, **law})
            print('conditional source task-level integration complete', layout, deleted, flush=True)
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    save(out/'source_laws.json', {'execution': spec, 'all14_conditional_source_laws_saved_without_query_data': True, 'rows': laws})
    result = {'execution': spec, 'rows': rows, 'all14_source_integrations_verified': len(rows) == 14,
        'cost': spec['cost'], 'cpu_seconds': time.process_time()-started, 'no_prior_or_model_adopted': True,
        'query_paid_outcomes_read': False, 'source6_7_evaluated': False, 'target_promoted': False, 'independent_confirmation': False}
    save(out/'results.json', result)
    print({'task1': [r for r in rows if r['deleted_task'] == 1], 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
