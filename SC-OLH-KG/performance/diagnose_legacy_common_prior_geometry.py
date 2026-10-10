"""Source-only GP common-prior correlations at native sparse spectral marginals."""

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
from representation.source_function_prior import head_batches
from representation.source_kernel_population import shared_kernel
from representation.task_bank_centering import CenteredTaskPopulationLikelihood

EXECUTION = 'performance/manifests/legacy_joint_population_common_prior_geometry_execution_v1.json'


def diagnose(likelihood, law, gp, reference, spec):
    value, _, state = likelihood.evaluate(np.asarray(law['optimizer_parameters']))
    for key, saved in (('mean', 'coefficient_mean'), ('mean_covariance', 'population_mean_covariance'),
        ('common_diagonal', 'common_prior_coefficient_diagonal'),
        ('residual_diagonal', 'residual_coefficient_diagonal'), ('discrepancy', 'discrepancy_variance')):
        np.testing.assert_allclose(state[key], law[saved], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(value, law['source_negative_log_likelihood_per_label'], atol=1e-8, rtol=1e-8)
    a = np.diag(state['common_diagonal'])
    design = np.vstack([b[0] for b in likelihood.blocks]); error = np.concatenate([b[1] for b in likelihood.blocks])
    residual_blocks = [(z*state['residual_diagonal'])@z.T+np.diag(b[2]+state['discrepancy'])
        for b, z in zip(likelihood.blocks, likelihood.task_designs)]
    covariance = design@a@design.T+block_diag(*residual_blocks)
    factor = cho_factor(covariance, lower=True); alpha = cho_solve(factor, error)
    n = likelihood.label_count
    dense_value = .5*(n*np.log(2*np.pi)+2*np.log(np.diag(factor[0])).sum()+error@alpha)/n
    dense_mean = likelihood.prior.mean+a@design.T@alpha
    dense_h = a-a@design.T@cho_solve(factor, design)@a
    np.testing.assert_allclose(value, dense_value, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(state['mean'], dense_mean, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(state['mean_covariance'], dense_h, atol=1e-8, rtol=1e-8)

    public_design = np.column_stack([np.ones(len(reference)), likelihood.basis.transform(reference)])
    operator, _, rank, _ = np.linalg.lstsq(public_design, np.eye(len(reference)), rcond=None)
    assert rank == public_design.shape[1]
    kernel = gp['anchor']['normalization']['scale']**2*shared_kernel(reference, reference, gp['anchor'])
    projected = operator@kernel@operator.T; projected = .5*(projected+projected.T)
    weights = spectral_weights(likelihood.basis, state['common_spectral_exponent'])[0]
    excess = state['common_group_variances'][likelihood.basis.group_index]*weights
    np.testing.assert_allclose(state['common_diagonal'][1:], excess+likelihood.tail, atol=1e-14, rtol=1e-12)
    active = np.flatnonzero(excess > 0); projected_shape = np.diag(projected)[1:]
    groups = likelihood.basis.group_names
    record = {'source_negative_log_likelihood_per_label': value, 'source_label_count': n,
        'native_source_law_dense_nll_mean_covariance_verified': True,
        'public_reference_point_count': len(reference), 'public_design_rank': int(rank),
        'common_function_draw_count': 1, 'source_task_count': len(likelihood.blocks),
        'active_common_excess_coefficient_count': len(active), 'active_common_shape_indices': active.tolist(),
        'active_common_excess_group_count': int(np.count_nonzero(state['common_group_variances'])),
        'native_common_group_variances': state['common_group_variances'].tolist(),
        'native_common_spectral_exponent': state['common_spectral_exponent'],
        'native_common_diagonal': state['common_diagonal'].tolist(),
        'native_common_excess_diagonal': excess.tolist(), 'independent_tail_variance_per_shape_coefficient': likelihood.tail,
        'projected_gp_common_prior_diagonal': np.diag(projected).tolist(),
        'projected_gp_common_prior_eigenvalues': np.linalg.eigvalsh(projected).tolist(),
        'common_shape_group_marginal_comparison': {name: {
            'native_excess_variance': float(excess[likelihood.basis.group_index == j].sum()),
            'gp_projected_variance': float(projected_shape[likelihood.basis.group_index == j].sum())}
            for j, name in enumerate(groups)},
        'projected_prior_kernel_relative_frobenius_error': float(np.linalg.norm(public_design@projected@public_design.T-kernel)/np.linalg.norm(kernel)),
        'prior_not_posterior_projected': True}
    if np.any(projected_shape[active] <= 0) or not np.all(np.isfinite(projected_shape[active])):
        return {**record, 'direction_defined': False, 'failure': 'actual active GP common prior variances are not positive finite'}
    scale = np.sqrt(projected_shape[active])
    correlation = projected[np.ix_(active+1, active+1)]/np.outer(scale, scale)
    eigenvalues = np.linalg.eigvalsh(correlation)
    record.update(projected_gp_active_shape_correlation=correlation.tolist(),
        projected_gp_active_shape_correlation_eigenvalues=eigenvalues.tolist())
    if not np.all(np.isfinite(correlation)) or (len(active) and eigenvalues.min() < -1e-10):
        return {**record, 'direction_defined': False, 'failure': 'actual projected active GP correlation is not numerically PSD'}
    root = np.sqrt(excess[active]); direction = np.zeros_like(a)
    direction[np.ix_(active+1, active+1)] = np.outer(root, root)*(correlation-np.eye(len(active)))
    np.fill_diagonal(direction, 0.)
    marginal_checks = []
    for p in (-spec['finite_difference_epsilon'], spec['finite_difference_epsilon'], spec['fixed_source_probe'], 1.):
        varied = a+p*direction
        np.testing.assert_array_equal(np.diag(varied), state['common_diagonal'])
        minimum = float(np.linalg.eigvalsh(varied).min()); assert minimum >= -1e-12
        marginal_checks.append({'p': p, 'minimum_common_covariance_eigenvalue': minimum})
    displacement = state['mean']-likelihood.prior.mean
    inverse = np.diag(1/state['common_diagonal'])
    gradient = .5*(inverse-inverse@(state['mean_covariance']+np.outer(displacement, displacement))@inverse)/n
    analytic = float(np.sum(gradient*direction))
    covariance_direction = design@direction@design.T
    dense_derivative = float(.5*np.sum((cho_solve(factor, np.eye(n))-np.outer(alpha, alpha))*covariance_direction.T)/n)
    np.testing.assert_allclose(analytic, dense_derivative, atol=1e-9, rtol=1e-7)

    def dense_nll(p):
        varied_factor = cho_factor(covariance+p*covariance_direction, lower=True)
        return float(.5*(n*np.log(2*np.pi)+2*np.log(np.diag(varied_factor[0])).sum()+error@cho_solve(varied_factor, error))/n)

    epsilon = spec['finite_difference_epsilon']
    finite_difference = (dense_nll(epsilon)-dense_nll(-epsilon))/(2*epsilon)
    np.testing.assert_allclose(analytic, finite_difference, atol=1e-7, rtol=1e-5)
    product = cho_solve(factor, covariance_direction); information = float(.5*np.trace(product@product))
    total_score = -n*analytic
    return {**record, 'direction_defined': True, 'coefficient_covariance_direction': direction.tolist(),
        'common_marginals_intercept_tail_support_preserved': True, 'covariance_eigenvalue_checks': marginal_checks,
        'nll_directional_derivative_per_label': analytic, 'dense_nll_directional_derivative_per_label': dense_derivative,
        'directional_derivative_finite_difference': float(finite_difference),
        'directional_derivative_absolute_error': float(abs(analytic-finite_difference)),
        'total_source_log_likelihood_directional_score': total_score,
        'fixed_direction_expected_fisher_information': information,
        'source_adapted_score_over_sqrt_information': total_score/np.sqrt(information) if information > 0 else None,
        'fixed_probe_total_log_likelihood_difference': float(-n*(dense_nll(spec['fixed_source_probe'])-dense_value)),
        'all_source_direction_derivative_and_marginal_checks_passed': True}


def run():
    started = time.process_time(); spec = read(EXECUTION); centering = read(spec['centering_execution'])
    coverage = read(centering['coverage_execution']); profile = read(centering['profile_execution']); rows = []
    for layout in spec['layouts']:
        for deleted in spec['training_tasks']:
            folder = Path(coverage['output'])/f'layout{layout}'/f'task{deleted}'
            inputs = read(folder/'inputs.json'); gp = read(folder/'gp_source_fit.json')
            means = read(Path(profile['output'])/f'layout{layout}'/f'task{deleted}'/'selected_source_fit.json')
            reference = np.asarray(inputs['public_mean_reference']); basis = OrthogonalLowOrderBasis(reference)
            tasks = [{**t, **{k: np.asarray(t[k]) for k in ('psi', 'features', 'response', 'noise')}} for t in inputs['source_tasks']]
            for head, name in enumerate(spec['heads']):
                if name == 'g':
                    law = read(Path(centering['output'])/f'layout{layout}'/f'task{deleted}'/'source_fit.json')
                    centers = [np.asarray(law['source_bank_centers'][t['name']]) for t in tasks]
                else:
                    law = means['heads'][head]['joint_structural']; centers = [np.zeros(basis.feature_dim) for _ in tasks]
                prior = restore_regularizer(basis, means['heads'][head]['regularizer'], tasks, head)
                likelihood = CenteredTaskPopulationLikelihood(prior, head_batches(tasks, head), [t['noise'][:, head] for t in tasks], centers)
                rows.append({'layout': layout, 'deleted_task': deleted, 'head': name,
                    **diagnose(likelihood, law, gp['heads'][head]['joint_gp'], reference, spec)})
        print('common-prior source geometry complete', layout, flush=True)
    summary = {}
    for layout in spec['layouts']:
        summary[layout] = {}
        for head in spec['heads']:
            group = [r for r in rows if r['layout'] == layout and r['head'] == head]
            valid = [r for r in group if r['direction_defined']]
            summary[layout][head] = {'defined_direction_count': len(valid),
                'active_excess_dimensions': [r['active_common_excess_coefficient_count'] for r in group],
                'source_directional_scores': [r['total_source_log_likelihood_directional_score'] for r in valid],
                'fixed_probe_log_likelihood_gains': [r['fixed_probe_total_log_likelihood_difference'] for r in valid],
                'fixed_direction_fisher_information': [r['fixed_direction_expected_fisher_information'] for r in valid]}
    result = {'execution': spec, 'summary': summary, 'rows': rows, 'no_covariance_model_fitted_or_selected': True,
        'all24_source_laws_reproduced': len(rows) == 24 and all(r['native_source_law_dense_nll_mean_covariance_verified'] for r in rows),
        'all24_directions_defined_and_verified': len(rows) == 24 and all(r['direction_defined'] for r in rows),
        'cost': spec['cost'], 'cpu_seconds': time.process_time()-started,
        'query_paid_outcomes_read': False, 'source_holdouts_evaluated': False,
        'target_promoted': False, 'independent_confirmation': False}
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True); save(out/'results.json', result)
    print({'summary': summary, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
