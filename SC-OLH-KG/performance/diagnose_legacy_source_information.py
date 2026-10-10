"""Source-cohort influence and local covariance information, without fitting."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve, solve_triangular

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_joint_common_covariance import restore_regularizer
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.run_legacy_related_inventory_source import problem_for, save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis, spectral_weights
from representation.public_control_mean import public_control_mean_coordinates
from representation.source_function_prior import head_batches
from representation.source_joint_population import JointCovariancePopulationLikelihood

EXECUTION = 'performance/manifests/legacy_joint_population_source_information_execution_v1.json'


def common_curves(inputs, law, physical):
    basis = OrthogonalLowOrderBasis(inputs['public_mean_reference']); means, variances = [], []
    for query in physical:
        psi = (query-inputs['mean_coordinate_center'])/inputs['mean_coordinate_scale']
        design = np.column_stack([np.ones(len(psi)), basis.transform(psi)])
        means.append(design@np.asarray(law['coefficient_mean']))
        variances.append(np.einsum('ij,jk,ik->i', design, np.asarray(law['population_mean_covariance']), design))
    return np.asarray(means), np.asarray(variances)


def covariance_information(inputs, means):
    law = means['heads'][1]['joint_structural']; basis = OrthogonalLowOrderBasis(inputs['public_mean_reference'])
    tasks = [{**t, **{k: np.asarray(t[k]) for k in ('psi', 'features', 'response', 'noise')}} for t in inputs['source_tasks']]
    prior = restore_regularizer(basis, means['heads'][1]['regularizer'], tasks, 1)
    likelihood = JointCovariancePopulationLikelihood(prior, head_batches(tasks, 1), [t['noise'][:, 1] for t in tasks])
    designs = [np.column_stack([np.ones(len(t['psi'])), basis.transform(t['psi'])]) for t in tasks]
    design = np.vstack(designs); noise = np.concatenate([t['noise'][:, 1] for t in tasks])
    common = np.asarray(law['common_prior_coefficient_diagonal']); residual = np.asarray(law['residual_coefficient_diagonal'])
    covariance = (design*common)@design.T+block_diag(*[(d*residual)@d.T for d in designs])
    covariance += np.diag(noise+law['discrepancy_variance'])
    factor = cho_factor(covariance, lower=True); error = np.concatenate([t['response'][:, 1] for t in tasks])-design@prior.mean
    dense_nll = .5*(len(error)*np.log(2*np.pi)+2*np.log(np.diag(factor[0])).sum()+error@cho_solve(factor, error))/len(error)
    np.testing.assert_allclose(dense_nll, law['source_negative_log_likelihood_per_label'], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(dense_nll, likelihood.evaluate(np.asarray(law['optimizer_parameters']))[0], atol=1e-8, rtol=1e-8)
    common_weights, common_derivative = spectral_weights(basis, law['common_spectral_exponent'])
    task_weights, task_derivative = spectral_weights(basis, law['spectral_exponent'])
    derivatives, names = [], []
    def common_matrix(diagonal): return (design*diagonal)@design.T
    def task_matrix(diagonal): return block_diag(*[(d*diagonal)@d.T for d in designs])
    for kind, key, weights, derivative, matrix in (
            ('common', 'common_group_variances', common_weights, common_derivative, common_matrix),
            ('task', 'source_group_variances', task_weights, task_derivative, task_matrix)):
        groups = np.asarray(law[key])
        for group, value in enumerate(groups):
            if value > 0:
                diagonal = np.r_[0., np.where(basis.group_index == group, value*weights, 0.)]
                derivatives.append(matrix(diagonal)); names.append(f'{kind}:log_variance:{basis.group_names[group]}')
        derivatives.append(matrix(np.r_[0., groups[basis.group_index]*derivative])); names.append(f'{kind}:spectral_exponent')
    if residual[0] > 0:
        derivatives.append(task_matrix(np.r_[residual[0], np.zeros(basis.feature_dim)])); names.append('task:log_intercept_variance')
    if law['discrepancy_variance'] > 0:
        derivatives.append(law['discrepancy_variance']*np.eye(len(error))); names.append('task:log_discrepancy_variance')
    lower = np.tril(factor[0]); whitened = []
    for derivative in derivatives:
        left = solve_triangular(lower, derivative, lower=True)
        whitened.append(solve_triangular(lower, left.T, lower=True).T)
    flat = np.asarray(whitened).reshape(len(whitened), -1); fisher = .5*flat@flat.T
    sensitivity = np.sqrt(np.diag(fisher)); nonzero = sensitivity > 0
    correlation = fisher[np.ix_(nonzero, nonzero)]/np.outer(sensitivity[nonzero], sensitivity[nonzero])
    eigenvalues = np.linalg.eigvalsh(correlation); active = np.asarray(names)[nonzero].tolist()
    pairs = sorted(({'parameters': [active[i], active[j]], 'correlation': float(correlation[i, j])}
        for i in range(len(active)) for j in range(i)), key=lambda r: abs(r['correlation']), reverse=True)[:4]
    return {'dense_source_nll_reproduced': float(dense_nll), 'parameter_names': names,
        'expected_covariance_information': fisher.tolist(), 'standardized_information_eigenvalues': eigenvalues.tolist(),
        'active_direction_count': len(active), 'standardized_information_numerical_rank': int(np.linalg.matrix_rank(correlation)),
        'zero_sensitivity_directions': np.asarray(names)[~nonzero].tolist(), 'largest_derivative_correlations': pairs,
        'zero_common_groups': [g for g, v in zip(basis.group_names, law['common_group_variances']) if v == 0],
        'zero_task_groups': [g for g, v in zip(basis.group_names, law['source_group_variances']) if v == 0],
        'task_spectral_exponent': law['spectral_exponent'], 'discrepancy_variance': law['discrepancy_variance'],
        'interpretation': 'Local fixed-mean covariance information at fitted support; boundary/support uncertainty and task-population uncertainty are not quantified.'}


def run():
    started = time.process_time(); spec = read(EXECUTION); native_spec = read(spec['native_execution'])
    profile = read(spec['profile_execution']); coverage = read(spec['coverage_execution'])
    public = read(Path(coverage['source_data']).with_name('frozen_inputs.json')); bank = [tuple(p) for p in public['bank']]
    physical = [public_control_mean_coordinates(no_outcomes(problem_for(public['plan'], public['tasks'][i])), bank)
        for i in spec['training_tasks']]
    cached = read(Path(profile['output'])/'calibration_profiles.json')['rows']
    rows, information = [], {}
    for layout in spec['layouts']:
        inputs = read(Path(coverage['output'])/f'layout{layout}'/'inputs.json')
        full = read(Path(native_spec['output'])/f'layout{layout}'/'selected_source_fit.json')
        native = full['heads'][1]['joint_structural']; mean, variance = common_curves(inputs, native, physical)
        information[layout] = covariance_information(inputs, full)
        for deleted in spec['training_tasks']:
            branch = read(Path(coverage['output'])/f'layout{layout}'/f'task{deleted}'/'inputs.json')
            fitted = read(Path(profile['output'])/f'layout{layout}'/f'task{deleted}'/'selected_source_fit.json')['heads'][1]['joint_structural']
            branch_mean, _ = common_curves(branch, fitted, physical)
            old = next(r for r in cached if r['layout'] == layout and r['deleted_task'] == deleted and r['case'] == 'native_source_nll_selection')
            np.testing.assert_allclose(branch_mean[deleted, coverage['layouts'][layout][deleted]], old['predictions']['prior']['mean'], atol=1e-8)
            shift = branch_mean-mean; offsets = shift.mean(axis=1, keepdims=True); basis = OrthogonalLowOrderBasis(branch['public_mean_reference'])
            rows.append({'layout': layout, 'deleted_task': deleted,
                'common_curve_change_rms': float(np.sqrt(np.mean(shift**2))),
                'constant_curve_change_rms': float(np.sqrt(np.mean(offsets**2))),
                'centered_shape_change_rms': float(np.sqrt(np.mean((shift-offsets)**2))),
                'change_over_full_conditional_common_sd_rms': float(np.sqrt(np.mean(shift**2/variance))),
                'native_full_task_intercept_variance': native['residual_coefficient_diagonal'][0],
                'deleted_fit_task_intercept_variance': fitted['residual_coefficient_diagonal'][0],
                'active_common_groups': [g for g, v in zip(basis.group_names, fitted['common_group_variances']) if v > 0],
                'active_task_groups': [g for g, v in zip(basis.group_names, fitted['source_group_variances']) if v > 0]})
        print('source influence and dense marginal information reproduced', layout, flush=True)
    result = {'execution': spec, 'influence_rows': rows, 'covariance_information': information,
        'all_deleted_curves_and_dense_objectives_reproduced': True, 'cost': spec['cost'],
        'cpu_seconds': time.process_time()-started, 'target_promoted': False, 'independent_confirmation': False}
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True); save(out/'results.json', result)
    print({'largest_influence_by_layout': {layout: max((r for r in rows if r['layout'] == layout), key=lambda r: r['common_curve_change_rms'])
        for layout in spec['layouts']}, 'covariance_information': {layout: {k: info[k] for k in
        ('active_direction_count', 'standardized_information_numerical_rank', 'standardized_information_eigenvalues', 'largest_derivative_correlations')}
        for layout, info in information.items()}, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
