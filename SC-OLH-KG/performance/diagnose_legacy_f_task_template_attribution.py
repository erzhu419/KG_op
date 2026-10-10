"""Attribute a failed task template using training-source likelihood only."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve, solve_triangular

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_sparse_task_template import condition, source_paths, source_system
from performance.run_legacy_related_inventory_source import save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis, spectral_weights

EXECUTION = 'performance/manifests/legacy_joint_population_f_task_template_attribution_execution_v1.json'


def native_task_shape_correlation(template, excess):
    shape = template[1:, 1:]; marginal = np.diag(shape); columns = np.flatnonzero(excess[1:] > 0)
    correlation = shape[np.ix_(columns, columns)]/np.sqrt(np.outer(marginal[columns], marginal[columns]))
    correlated = np.zeros_like(template)
    correlated[np.ix_(columns+1, columns+1)] = correlation*np.sqrt(np.outer(excess[columns+1], excess[columns+1]))
    np.testing.assert_allclose(np.diag(correlated), excess, atol=1e-14, rtol=1e-12)
    assert np.linalg.eigvalsh(correlated).min() >= -1e-12
    return correlated


def diagnose(inputs, head, saved, deleted, spec):
    template = np.asarray(saved['template_coefficient_covariance']); excess = np.asarray(saved['original_task_excess_diagonal'])
    system = source_system(inputs, head, template, excess, saved['independent_tail_per_coefficient'], deleted)
    native = system['native']; v = native['residual_coefficient_diagonal'][0]; q = spec['fixed_probe_q']
    basis = OrthogonalLowOrderBasis(inputs['public_mean_reference']); d = system['d']; designs = system['designs']
    level = np.zeros_like(template); level[0, 0] = template[0, 0]
    diagonal_change = np.diag(np.diag(template)-excess)-level
    active = np.r_[False, np.asarray(native['source_group_variances'])[basis.group_index] > 0]
    existing = np.diag(np.diag(diagonal_change)*active)
    new = diagonal_change-existing
    level_shape = np.zeros_like(template); level_shape[0, 1:] = template[0, 1:]; level_shape[1:, 0] = template[1:, 0]
    shape_correlation = template-np.diag(np.diag(template))-level_shape
    parts = {'level_variance': level, 'existing_task_shape_marginals': existing,
        'new_task_group_marginals': new, 'level_shape_correlation': level_shape, 'shape_correlation': shape_correlation}
    np.testing.assert_allclose(sum(parts.values()), system['direction'], atol=1e-14, rtol=1e-12)
    c = system['c0']+v*system['u']@system['u'].T; factor = cho_factor(c, lower=True)
    inverse = cho_solve(factor, np.eye(len(c))); alpha = cho_solve(factor, system['error'])
    score = inverse-np.outer(alpha, alpha); lower = np.tril(factor[0])
    def task_matrix(value): return block_diag(*[local@value@local.T for local in designs])
    def whiten(value):
        left = solve_triangular(lower, value, lower=True)
        return solve_triangular(lower, left.T, lower=True).T.ravel()
    component_matrices = {name: task_matrix(value) for name, value in parts.items()}
    gradients = {name: float(.5*np.sum(score*value)/len(c)) for name, value in component_matrices.items()}
    baseline = condition(system, 0., v, derivative=True)
    np.testing.assert_allclose(sum(gradients.values()), baseline['q_gradient_per_label'], atol=1e-10, rtol=1e-9)

    # Only locally free native parameters enter this information projection.
    # Bounds and inactive groups are not treated as unconstrained nuisance moves.
    nuisance_names, nuisance = [], []
    for kind, groups, exponent in (('common', native['common_group_variances'], native['common_spectral_exponent']),
        ('task', native['source_group_variances'], native['spectral_exponent'])):
        weights, derivative = spectral_weights(basis, exponent); groups = np.asarray(groups)
        def covariance(diagonal):
            return (d*diagonal)@d.T if kind == 'common' else task_matrix(np.diag(diagonal))
        for group in np.flatnonzero(groups > 0):
            nuisance_names.append(f'{kind}:log_group_variance:{basis.group_names[group]}')
            nuisance.append(whiten(covariance(np.r_[0., groups[group]*weights*(basis.group_index == group)])))
        if 0 < exponent < 4:
            nuisance_names.append(f'{kind}:spectral_exponent')
            nuisance.append(whiten(covariance(np.r_[0., groups[basis.group_index]*derivative])))
    if v > 0:
        nuisance_names.append('task:log_intercept_variance'); nuisance.append(whiten(v*system['u']@system['u'].T))
    if native['discrepancy_variance'] > 0:
        nuisance_names.append('task:log_discrepancy_variance'); nuisance.append(whiten(native['discrepancy_variance']*np.eye(len(c))))
    flat = np.asarray(nuisance); unit = flat/np.linalg.norm(flat, axis=1)[:, None]
    direction = whiten(system['cq']); coefficients, _, rank, _ = np.linalg.lstsq(unit.T, direction, rcond=None)
    residual = direction-unit.T@coefficients
    conditional_info = .5*direction@direction; efficient_info = .5*residual@residual
    np.testing.assert_allclose(conditional_info, baseline['q_total_fisher_information'], atol=1e-8, rtol=1e-9)

    # Correlate existing task shape excess only; retain independent v and tail.
    correlated = native_task_shape_correlation(template, excess)
    probes = {'full_template': system['direction'], 'diagonal_template': np.diag(np.diag(template))-np.diag(excess),
        'shape_diagonal_template': diagonal_change, 'native_task_shape_correlation': correlated-np.diag(excess)}
    records = {}
    for name, coefficient_direction in probes.items():
        r = system['base']+q*coefficient_direction; r[0, 0] += v
        assert np.linalg.eigvalsh(r).min() >= -1e-12
        value = condition({**system, 'cq': task_matrix(coefficient_direction)}, q, v, derivative=True)
        at_zero = condition({**system, 'cq': task_matrix(coefficient_direction)}, 0., v, derivative=True)
        records[name] = {'source_log_likelihood_gain_per_label': baseline['source_negative_log_likelihood_per_label']-value['source_negative_log_likelihood_per_label'],
            'q_gradient_at_zero_per_label': at_zero['q_gradient_per_label'], 'q_total_fisher_information_at_zero': at_zero['q_total_fisher_information']}
    return {'component_q_gradients_per_label': gradients, 'probes': records, 'nuisance_parameter_names': nuisance_names,
        'nuisance_derivative_rank': int(rank), 'conditional_q_fisher_information': float(conditional_info),
        'nuisance_adjusted_q_fisher_information': float(efficient_info), 'nuisance_adjusted_information_fraction': float(efficient_info/conditional_info),
        'verification': {'template_direction_and_score_decomposition_match': True, 'dense_and_whitened_fisher_match': True,
            'all_four_fixed_probe_covariances_psd': True, 'native_shape_correlation_preserves_marginals_support_independent_v_and_tail': True}}


def run():
    started = time.process_time(); spec = read(EXECUTION); parent = read(spec['template_execution'])
    coverage, profile = source_paths(parent); templates = read(spec['source_templates'])['rows']; rows = []
    assert not read(spec['template_results'])['source_direction_gate_passed']
    for saved in templates:
        layout, deleted = saved['layout'], saved['deleted_task']; suffix = Path(f'layout{layout}')/f'task{deleted}'
        inputs = read(Path(coverage['output'])/suffix/'inputs.json')
        head = read(Path(profile['output'])/suffix/'selected_source_fit.json')['heads'][0]
        rows.append({'layout': layout, 'deleted_task': deleted, **diagnose(inputs, head, saved, deleted, spec)})
    summary = {layout: {'mean_component_q_gradients_per_label': {name: float(np.mean([r['component_q_gradients_per_label'][name] for r in rows if r['layout'] == layout]))
        for name in rows[0]['component_q_gradients_per_label']}, 'probes': {name: {
            'mean_source_log_likelihood_gain_per_label': float(np.mean([r['probes'][name]['source_log_likelihood_gain_per_label'] for r in rows if r['layout'] == layout])),
            'improving_folds': sum(r['probes'][name]['source_log_likelihood_gain_per_label'] > 0 for r in rows if r['layout'] == layout)} for name in rows[0]['probes']},
        'mean_nuisance_adjusted_q_information_fraction': float(np.mean([r['nuisance_adjusted_information_fraction'] for r in rows if r['layout'] == layout]))}
        for layout in parent['layouts']}
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    result = {'execution': spec, 'rows': rows, 'summary': summary, 'all12_source_attributions_verified': True,
        'cost': spec['cost'], 'cpu_seconds': time.process_time()-started, 'no_model_adopted': True,
        'query_paid_outcomes_read': False, 'source6_7_evaluated': False, 'target_promoted': False, 'independent_confirmation': False}
    save(out/'results.json', result); print({'summary': summary, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
