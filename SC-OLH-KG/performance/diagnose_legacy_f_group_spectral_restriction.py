"""Source-only information for group-specific f frequency decay rates."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve, solve_triangular

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_joint_common_covariance import restore_regularizer
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.run_legacy_related_inventory_source import save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis, spectral_weights
from representation.source_function_prior import head_batches
from representation.source_joint_population import JointCovariancePopulationLikelihood

EXECUTION = 'performance/manifests/legacy_joint_population_f_group_spectral_restriction_execution_v1.json'


def diagnose(likelihood, law, spec):
    parameters = np.asarray(law['optimizer_parameters']); native, gradient, state = likelihood.evaluate(parameters)
    designs = [b[0] for b in likelihood.blocks]; design = np.vstack(designs)
    error = np.concatenate([b[1] for b in likelihood.blocks]); n = len(error); basis = likelihood.basis
    covariance = (design*state['common_diagonal'])@design.T+block_diag(*[
        (d*state['residual_diagonal'])@d.T+np.diag(b[2]+state['discrepancy']) for d, b in zip(designs, likelihood.blocks)])
    factor = cho_factor(covariance, lower=True); lower = np.tril(factor[0])
    inverse = cho_solve(factor, np.eye(n)); alpha = inverse@error
    influence = .5*(inverse-np.outer(alpha, alpha))
    def common_matrix(diagonal): return (design*diagonal)@design.T
    def task_matrix(diagonal): return block_diag(*[(d*diagonal)@d.T for d in designs])
    def whiten(matrix):
        left = solve_triangular(lower, matrix, lower=True)
        return solve_triangular(lower, left.T, lower=True).T.ravel()
    def dense_nll(matrix):
        cf = cho_factor(matrix, lower=True)
        return float(.5*(n*np.log(2*np.pi)+2*np.log(np.diag(cf[0])).sum()+error@cho_solve(cf, error)))
    dense_native = dense_nll(covariance)
    np.testing.assert_allclose(dense_native/n, native, atol=1e-8, rtol=1e-8)
    nuisances, nuisance_names, directions, records = [], [], [], []
    components = (
        ('common', state['common_group_variances'], state['common_spectral_exponent'], common_matrix),
        ('task', state['group_variances'], state['spectral_exponent'], task_matrix))
    for kind, amplitudes, exponent, matrix in components:
        weights, derivative = spectral_weights(basis, exponent); total_gradient = 0.
        for group, amplitude in enumerate(amplitudes):
            if amplitude <= 0: continue
            mask = basis.group_index == group
            nuisance_names.append(f'{kind}:log_group_variance:{basis.group_names[group]}')
            nuisances.append(matrix(np.r_[0., amplitude*weights*mask]))
            delta = amplitude*derivative*mask
            if not np.any(delta != 0): continue
            direction = matrix(np.r_[0., delta]); analytic = float(np.sum(influence*direction)); total_gradient += analytic
            def changed(q):
                new_weights, new_derivative = spectral_weights(basis, q)
                difference = amplitude*(new_weights-weights)*mask
                np.testing.assert_allclose(difference.sum(), 0., atol=1e-14, rtol=0.)
                return covariance+matrix(np.r_[0., difference]), matrix(np.r_[0., amplitude*new_derivative*mask])
            q = spec['gradient_verification_exponent']; eps = spec['gradient_verification_epsilon']
            interior_covariance, interior_direction = changed(q)
            interior_factor = cho_factor(interior_covariance, lower=True)
            interior_alpha = cho_solve(interior_factor, error)
            interior_score = float(.5*np.sum((cho_solve(interior_factor, np.eye(n))-np.outer(interior_alpha, interior_alpha))*interior_direction))
            finite = (dense_nll(changed(q+eps)[0])-dense_nll(changed(q-eps)[0]))/(2*eps)
            np.testing.assert_allclose(interior_score, finite, atol=1e-6, rtol=1e-5)
            step = -np.sign(analytic)*spec['fixed_source_probe']; varied = exponent+step
            feasible = bool(0 <= varied <= 4 and step != 0)
            probe_gain = dense_native-dense_nll(changed(varied)[0]) if feasible else None
            records.append({'component': kind, 'group': basis.group_names[group], 'native_shared_exponent': exponent,
                'group_variance_mass': float(amplitude), 'total_source_nll_group_exponent_derivative': analytic,
                'source_selected_step': float(step), 'fixed_probe_feasible_within_original_bounds': feasible,
                'fixed_probe_total_log_likelihood_gain': probe_gain,
                'interior_derivative_finite_difference_error': float(abs(interior_score-finite))})
            directions.append(direction)
        expected = n*gradient[len(basis.group_names) if kind == 'common' else -2]
        np.testing.assert_allclose(total_gradient, expected, atol=1e-7, rtol=1e-6)
    nuisance_names += ['task:physical_intercept_variance', 'task:physical_discrepancy_variance']
    nuisances += [block_diag(*[np.ones((len(d), len(d))) for d in designs]), np.eye(n)]
    nuisance_flat = np.asarray([whiten(m) for m in nuisances]); norms = np.linalg.norm(nuisance_flat, axis=1)
    unit = nuisance_flat/norms[:, None]; spectral_flat = np.asarray([whiten(m) for m in directions])
    coefficients, _, nuisance_rank, _ = np.linalg.lstsq(unit.T, spectral_flat.T, rcond=None)
    residual = spectral_flat-(unit.T@coefficients).T
    conditional = .5*spectral_flat@spectral_flat.T; efficient = .5*residual@residual.T
    sensitivity = np.sqrt(np.diag(conditional))
    normalized = conditional/np.outer(sensitivity, sensitivity)
    for j, record in enumerate(records):
        record.update(conditional_fisher_information=float(conditional[j, j]),
            nuisance_adjusted_fisher_information=float(efficient[j, j]),
            nuisance_adjusted_information_fraction=float(efficient[j, j]/conditional[j, j]))
    return {'source_label_count': n, 'source_task_count': len(designs), 'group_directions': records,
        'conditional_group_exponent_fisher_information': conditional.tolist(),
        'nuisance_adjusted_group_exponent_fisher_information': efficient.tolist(),
        'standardized_group_exponent_information_eigenvalues': np.linalg.eigvalsh(normalized).tolist(),
        'nuisance_parameter_names': nuisance_names, 'nuisance_derivative_rank': int(nuisance_rank),
        'all_group_derivatives_budgets_bounds_and_shared_gradient_verified': True}


def run():
    started = time.process_time(); spec = read(EXECUTION); centering = read(spec['centering_execution'])
    assert read(spec['level_information_results'])['all14_native_laws_and_slice_derivatives_verified']
    coverage = read(centering['coverage_execution']); profile = read(centering['profile_execution'])
    native_full = read(spec['native_full_execution']); rows = []
    for layout in spec['layouts']:
        folders = [(None, Path(coverage['output'])/f'layout{layout}', Path(native_full['output'])/f'layout{layout}')]
        folders += [(i, Path(coverage['output'])/f'layout{layout}'/f'task{i}', Path(profile['output'])/f'layout{layout}'/f'task{i}')
            for i in spec['training_tasks']]
        for deleted, inputs_folder, fit_folder in folders:
            inputs = read(inputs_folder/'inputs.json'); means = read(fit_folder/'selected_source_fit.json')
            basis = OrthogonalLowOrderBasis(inputs['public_mean_reference'])
            tasks = [{**t, **{k: np.asarray(t[k]) for k in ('psi', 'features', 'response', 'noise')}} for t in inputs['source_tasks']]
            prior = restore_regularizer(basis, means['heads'][0]['regularizer'], tasks, 0)
            likelihood = JointCovariancePopulationLikelihood(prior, head_batches(tasks, 0), [t['noise'][:, 0] for t in tasks])
            rows.append({'layout': layout, 'deleted_task': deleted, **diagnose(likelihood, means['heads'][0]['joint_structural'], spec)})
        print('source f group-frequency restriction complete', layout, flush=True)
    summary = {layout: {kind: {field: [d[field] for row in rows if row['layout'] == layout for d in row['group_directions'] if d['component'] == kind]
        for field in ('total_source_nll_group_exponent_derivative', 'nuisance_adjusted_fisher_information', 'fixed_probe_total_log_likelihood_gain')}
        for kind in ('common', 'task')} for layout in spec['layouts']}
    result = {'execution': spec, 'summary': summary, 'rows': rows, 'all14_source_group_spectral_checks_passed': len(rows) == 14,
        'cost': spec['cost'], 'cpu_seconds': time.process_time()-started, 'no_model_selected': True,
        'query_paid_outcomes_read': False, 'source_holdouts_evaluated': False, 'target_promoted': False, 'independent_confirmation': False}
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True); save(out/'results.json', result)
    print({'full_source': [r for r in rows if r['deleted_task'] is None], 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
