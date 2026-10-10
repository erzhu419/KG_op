"""Source-only fixed-marginal residual coefficient covariance directions."""

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
from representation.task_bank_centering import CenteredTaskPopulationLikelihood

EXECUTION = 'performance/manifests/legacy_joint_population_coefficient_correlation_score_execution_v1.json'


def diagnose(likelihood, law, spec):
    parameters = np.asarray(law['optimizer_parameters'])
    value, _, state = likelihood.evaluate(parameters)
    for key, saved in (('mean', 'coefficient_mean'), ('mean_covariance', 'population_mean_covariance'),
        ('residual_diagonal', 'residual_coefficient_diagonal'), ('discrepancy', 'discrepancy_variance')):
        np.testing.assert_allclose(state[key], law[saved], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(value, law['source_negative_log_likelihood_per_label'], atol=1e-8, rtol=1e-8)
    size = likelihood.basis.feature_dim+1; coefficient_score = np.zeros((size, size)); source_covariances = []
    displacement = state['mean']-likelihood.prior.mean
    for (design, error, noise), z in zip(likelihood.blocks, likelihood.task_designs):
        variance = (z*state['residual_diagonal'])@z.T+np.diag(noise+state['discrepancy'])
        factor = cho_factor(variance, lower=True); W = cho_solve(factor, np.eye(len(error)))
        cross = design.T@W@z; alpha = W@(error-design@displacement)
        coefficient_score += .5*(z.T@W@z-cross.T@state['mean_covariance']@cross-np.outer(z.T@alpha, z.T@alpha))/likelihood.label_count
        source_covariances.append(variance)
    weights = spectral_weights(likelihood.basis, state['spectral_exponent'])[0]
    excess = state['group_variances'][likelihood.basis.group_index]*weights
    root = np.sqrt(excess); standardized = coefficient_score[1:, 1:]*np.outer(root, root)
    off_diagonal = standardized-np.diag(np.diag(standardized))
    eigenvalues, vectors = np.linalg.eigh(off_diagonal); vector = vectors[:, 0]
    shape_direction = np.outer(root, root)*(np.outer(vector, vector)-np.diag(vector**2))
    direction = np.zeros((size, size)); direction[1:, 1:] = shape_direction
    np.testing.assert_allclose(np.diag(direction), 0., atol=1e-14)
    for p in (-spec['finite_difference_epsilon'], spec['finite_difference_epsilon'], spec['fixed_source_probe']):
        covariance = np.diag(state['residual_diagonal'])+p*direction
        np.testing.assert_array_equal(np.diag(covariance), state['residual_diagonal'])
        assert np.linalg.eigvalsh(covariance).min() >= -1e-12
    design = np.vstack([b[0] for b in likelihood.blocks]); error = np.concatenate([b[1] for b in likelihood.blocks])
    covariance = (design*state['common_diagonal'])@design.T+block_diag(*source_covariances)
    covariance_direction = block_diag(*[z@direction@z.T for z in likelihood.task_designs])
    n = likelihood.label_count
    def dense_nll(p):
        factor = cho_factor(covariance+p*covariance_direction, lower=True)
        return float(.5*(n*np.log(2*np.pi)+2*np.log(np.diag(factor[0])).sum()+error@cho_solve(factor, error))/n)
    np.testing.assert_allclose(dense_nll(0.), value, atol=1e-8, rtol=1e-8)
    analytic = float(np.sum(coefficient_score*direction))
    np.testing.assert_allclose(analytic, eigenvalues[0], atol=1e-12, rtol=1e-10)
    epsilon = spec['finite_difference_epsilon']
    finite_difference = (dense_nll(epsilon)-dense_nll(-epsilon))/(2*epsilon)
    np.testing.assert_allclose(analytic, finite_difference, atol=1e-7, rtol=1e-5)
    factor = cho_factor(covariance, lower=True); product = cho_solve(factor, covariance_direction)
    information = float(.5*np.trace(product@product))
    total_score = -n*analytic
    return {'source_negative_log_likelihood_per_label': value, 'source_label_count': n,
        'active_task_excess_coefficient_count': int(np.count_nonzero(excess)),
        'active_task_excess_group_count': int(np.count_nonzero(state['group_variances'])),
        'offdiagonal_standardized_score_eigenvalues': eigenvalues.tolist(),
        'source_direction_vector': vector.tolist(), 'coefficient_covariance_direction': direction.tolist(),
        'nll_directional_derivative_per_label': analytic,
        'directional_derivative_finite_difference': float(finite_difference),
        'directional_derivative_absolute_error': float(abs(analytic-finite_difference)),
        'total_source_log_likelihood_directional_score': float(total_score),
        'fixed_direction_expected_fisher_information': information,
        'adaptively_selected_source_score_over_sqrt_information': float(total_score/np.sqrt(information)) if information > 0 else None,
        'fixed_probe_total_log_likelihood_difference': float(-n*(dense_nll(spec['fixed_source_probe'])-value)),
        'native_source_law_dense_nll_direction_derivative_marginals_psd_verified': True}


def run():
    started = time.process_time(); spec = read(EXECUTION); centering = read(spec['centering_execution'])
    coverage = read(centering['coverage_execution']); profile = read(centering['profile_execution']); rows = []
    for layout in spec['layouts']:
        for deleted in spec['training_tasks']:
            inputs = read(Path(coverage['output'])/f'layout{layout}'/f'task{deleted}'/'inputs.json')
            means = read(Path(profile['output'])/f'layout{layout}'/f'task{deleted}'/'selected_source_fit.json')
            basis = OrthogonalLowOrderBasis(inputs['public_mean_reference'])
            tasks = [{**t, **{k: np.asarray(t[k]) for k in ('psi', 'features', 'response', 'noise')}} for t in inputs['source_tasks']]
            for head, name in enumerate(spec['heads']):
                if name == 'g':
                    law = read(Path(centering['output'])/f'layout{layout}'/f'task{deleted}'/'source_fit.json')
                    centers = [np.asarray(law['source_bank_centers'][t['name']]) for t in tasks]
                else:
                    law = means['heads'][head]['joint_structural']; centers = [np.zeros(basis.feature_dim) for _ in tasks]
                prior = restore_regularizer(basis, means['heads'][head]['regularizer'], tasks, head)
                likelihood = CenteredTaskPopulationLikelihood(prior, head_batches(tasks, head), [t['noise'][:, head] for t in tasks], centers)
                rows.append({'layout': layout, 'deleted_task': deleted, 'head': name, **diagnose(likelihood, law, spec)})
        print('both-head source coefficient directions complete', layout, flush=True)
    summary = {layout: {head: {
        'active_excess_dimensions': [r['active_task_excess_coefficient_count'] for r in rows if r['layout'] == layout and r['head'] == head],
        'source_directional_scores': [r['total_source_log_likelihood_directional_score'] for r in rows if r['layout'] == layout and r['head'] == head],
        'fixed_probe_log_likelihood_gains': [r['fixed_probe_total_log_likelihood_difference'] for r in rows if r['layout'] == layout and r['head'] == head],
        'fixed_direction_fisher_information': [r['fixed_direction_expected_fisher_information'] for r in rows if r['layout'] == layout and r['head'] == head]}
        for head in spec['heads']} for layout in spec['layouts']}
    result = {'execution': spec, 'summary': summary, 'rows': rows, 'all24_source_law_and_direction_checks_passed': True,
        'no_covariance_family_selected': True, 'cost': spec['cost'], 'cpu_seconds': time.process_time()-started,
        'source_holdouts_indexed': False, 'target_promoted': False, 'independent_confirmation': False}
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True); save(out/'results.json', result)
    print({'summary': summary, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
