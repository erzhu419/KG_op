"""Source-only common geometry fitting; freeze all laws before development use."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_joint_common_covariance import restore_regularizer
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.run_legacy_related_inventory_source import save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis, spectral_weights
from representation.source_common_geometry import CommonGeometryLikelihood, fit_common_geometry
from representation.source_function_prior import head_batches
from representation.source_joint_population import GaussianPopulationLikelihood

EXECUTION = 'performance/manifests/legacy_joint_population_f_common_scale_correlation_execution_v1.json'


def verify(likelihood, parameters):
    value, _, state = likelihood.evaluate(parameters); source = likelihood.source
    a = state['common_prior_covariance']; native = likelihood.native; basis = source.basis
    weights = spectral_weights(basis, state['common_spectral_exponent'])[0]
    np.testing.assert_allclose(np.diag(a)[1:], state['common_group_variances'][basis.group_index]*weights+source.tail, atol=1e-14)
    inactive = np.flatnonzero(~np.isin(basis.group_index, likelihood.groups))+1
    np.testing.assert_array_equal(a[inactive], np.diag(np.diag(a))[inactive])
    np.testing.assert_array_equal(a[0], np.diag(np.diag(a))[0])
    np.testing.assert_equal(a[0, 0], native['common_prior_coefficient_diagonal'][0])
    np.testing.assert_allclose(state['residual_diagonal'], native['residual_coefficient_diagonal'], atol=1e-14, rtol=0.)
    np.testing.assert_allclose(state['discrepancy'], native['discrepancy_variance'], atol=1e-14, rtol=0.)
    np.testing.assert_array_equal(state['common_group_variances'][~np.isin(np.arange(len(basis.group_names)), likelihood.groups)], 0.)
    assert np.linalg.eigvalsh(a).min() > 0
    design = np.vstack([b[0] for b in source.blocks]); error = np.concatenate([b[1] for b in source.blocks])
    covariance = design@a@design.T+block_diag(*[(d*state['residual_diagonal'])@d.T+np.diag(noise+state['discrepancy'])
        for d, _, noise in source.blocks])
    factor = cho_factor(covariance, lower=True); alpha = cho_solve(factor, error); n = len(error)
    dense_value = .5*(n*np.log(2*np.pi)+2*np.log(np.diag(factor[0])).sum()+error@alpha)/n
    dense_mean = source.prior.mean+a@design.T@alpha
    dense_h = a-a@design.T@cho_solve(factor, design)@a
    np.testing.assert_allclose(value, dense_value, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(state['mean'], dense_mean, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(state['mean_covariance'], dense_h, atol=1e-8, rtol=1e-8)
    assert np.linalg.eigvalsh(state['mean_covariance']).min() >= -1e-12
    return {'dense_source_nll_error': float(abs(value-dense_value)),
        'dense_common_mean_max_error': float(np.max(np.abs(state['mean']-dense_mean))),
        'dense_common_covariance_max_error': float(np.max(np.abs(state['mean_covariance']-dense_h))),
        'native_support_intercept_tail_task_and_coherent_conditioning_verified': True}


def gradient_check(likelihood, spec):
    point = likelihood.initial.copy(); point[:-2] *= spec['gradient_verification_amplitude_multiplier']
    point[-2:] = [spec['gradient_verification_gamma'], spec['gradient_verification_p']]
    analytic = likelihood.evaluate(point)[1]; epsilon = spec['gradient_verification_epsilon']; finite = []
    for j in range(len(point)):
        lower, upper = point.copy(), point.copy(); lower[j] -= epsilon; upper[j] += epsilon
        finite.append((likelihood.evaluate(upper)[0]-likelihood.evaluate(lower)[0])/(2*epsilon))
    np.testing.assert_allclose(analytic, finite, atol=1e-7, rtol=1e-5)
    return {'interior_parameters': point.tolist(), 'analytic_gradient': analytic.tolist(),
        'finite_difference_gradient': finite, 'maximum_absolute_error': float(np.max(np.abs(analytic-finite)))}


def run():
    started = time.process_time(); spec = read(EXECUTION); centering = read(spec['centering_execution'])
    coverage = read(centering['coverage_execution']); profile = read(centering['profile_execution'])
    geometry = read(spec['geometry_results']); laws, checks, branches = [], [], []
    for layout in spec['layouts']:
        for deleted in spec['training_tasks']:
            suffix = Path(f'layout{layout}')/f'task{deleted}'
            inputs = read(Path(coverage['output'])/suffix/'inputs.json')
            means = read(Path(profile['output'])/suffix/'selected_source_fit.json')['heads'][0]
            tasks = [{**t, **{key: np.asarray(t[key]) for key in ('psi', 'features', 'response', 'noise')}} for t in inputs['source_tasks']]
            basis = OrthogonalLowOrderBasis(inputs['public_mean_reference'])
            prior = restore_regularizer(basis, means['regularizer'], tasks, 0)
            source = GaussianPopulationLikelihood(prior, head_batches(tasks, 0), [t['noise'][:, 0] for t in tasks])
            template = next(r for r in geometry['rows'] if r['layout'] == layout and r['deleted_task'] == deleted and r['head'] == 'f')
            likelihood = CommonGeometryLikelihood(source, means['joint_structural'], template)
            value, _, state = likelihood.evaluate(likelihood.initial)
            np.testing.assert_allclose(value, likelihood.native['source_negative_log_likelihood_per_label'], atol=1e-8, rtol=1e-8)
            np.testing.assert_allclose(state['mean'], likelihood.native['coefficient_mean'], atol=1e-8, rtol=1e-8)
            np.testing.assert_allclose(state['mean_covariance'], likelihood.native['population_mean_covariance'], atol=1e-8, rtol=1e-8)
            record = {'layout': layout, 'deleted_task': deleted, 'native_endpoint_reproduced': True}
            if deleted == spec['gradient_verification_deleted_task']:
                record['gradient'] = gradient_check(likelihood, spec)
            branches.append((layout, deleted, likelihood, record))
    for layout, deleted, likelihood, record in branches:
        law = fit_common_geometry(likelihood, spec['optimizer'])
        record['fitted'] = verify(likelihood, np.asarray(law['optimizer_parameters'])); checks.append(record)
        laws.append({'layout': layout, 'deleted_task': deleted, 'head': 'f', 'law': law})
        print('joint common geometry source fit', layout, deleted, law['optimizer_converged'], law['optimizer_iterations'], flush=True)
    assert len(laws) == spec['fit_cap']
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    save(out/'source_laws.json', {'execution': spec, 'all12_laws_frozen_before_development_queries': True, 'rows': laws, 'verification': checks})
    rows = [{'layout': r['layout'], 'deleted_task': r['deleted_task'],
        'converged': r['law']['optimizer_converged'], 'iterations': r['law']['optimizer_iterations'],
        'p': r['law']['common_correlation_strength'], 'gamma': r['law']['common_spectral_exponent'],
        'common_group_variances': r['law']['common_group_variances'],
        'projected_gradient_max': r['law']['optimizer_projected_gradient_max'],
        'total_source_log_likelihood_gain': r['law']['source_label_count']*(r['law']['initial_source_negative_log_likelihood_per_label']-r['law']['source_negative_log_likelihood_per_label'])}
        for r in laws]
    result = {'execution': spec, 'rows': rows, 'all_fits_complete': all(r['converged'] for r in rows),
        'all12_source_laws_verified_and_frozen': len(checks) == 12, 'verification': checks,
        'forecasts_scored': False, 'no_model_adopted': True, 'cost': {**spec['cost'], 'actual_new_f_source_fits': len(laws)},
        'cpu_seconds': time.process_time()-started, 'query_paid_outcomes_read': False,
        'source6_7_evaluated': False, 'target_promoted': False, 'independent_confirmation': False}
    save(out/'results.json', result); print({'rows': rows, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
