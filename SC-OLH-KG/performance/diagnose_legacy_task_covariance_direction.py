"""Source-only residual coefficient moments and covariance likelihood directions."""

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
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.source_function_prior import head_batches
from representation.source_joint_population import JointCovariancePopulationLikelihood

EXECUTION = 'performance/manifests/legacy_joint_population_task_covariance_direction_v1.json'


def dense_nll(blocks, common, residual, discrepancy):
    x = np.vstack([a for a, _, _ in blocks]); error = np.concatenate([b for _, b, _ in blocks])
    within = block_diag(*[a@residual@a.T+np.diag(c+discrepancy) for a, _, c in blocks])
    covariance = (x*common)@x.T+within; factor = cho_factor(covariance, lower=True)
    return float(.5*(len(error)*np.log(2*np.pi)+2*np.log(np.diag(factor[0])).sum()+error@cho_solve(factor, error))/len(error))


def run():
    started = time.process_time(); spec = read(EXECUTION); coverage = read(spec['coverage_execution']); rows = []
    for layout in spec['layouts']:
        folder = Path(coverage['output'])/f'layout{layout}'
        inputs, fit = read(folder/'inputs.json'), read(folder/'source_fit.json')
        tasks = [{**t, **{k: np.asarray(t[k]) for k in ('psi', 'features', 'response', 'noise')}} for t in inputs['source_tasks']]
        basis = OrthogonalLowOrderBasis(inputs['public_mean_reference'])
        for h, head_name in enumerate(spec['heads']):
            native = fit['heads'][h]; regularizer = restore_regularizer(basis, native['regularizer'], tasks, h)
            likelihood = JointCovariancePopulationLikelihood(regularizer, head_batches(tasks, h), [t['noise'][:, h] for t in tasks])
            value, _, state = likelihood.evaluate(np.asarray(native['joint_structural']['optimizer_parameters']))
            r = np.diag(state['residual_diagonal']); posterior_means, posterior_covariances = [], []
            for x, error, noise in likelihood.blocks:
                variance = x@r@x.T+np.diag(noise+state['discrepancy'])
                gain = r@cho_solve(cho_factor(variance, lower=True), x).T
                conditional_mean = gain@(error-x@(state['mean']-regularizer.mean))
                common_gain = gain@x
                conditional_covariance = r-gain@x@r+common_gain@state['mean_covariance']@common_gain.T
                posterior_means.append(conditional_mean); posterior_covariances.append(conditional_covariance)
            moment = sum(np.outer(m, m)+c for m, c in zip(posterior_means, posterior_covariances))/len(tasks)
            support = state['residual_diagonal'] > 0
            scale = np.sqrt(state['residual_diagonal'][support]); whitened = moment[np.ix_(support, support)]/np.outer(scale, scale)
            values, vectors = np.linalg.eigh((whitened+whitened.T)/2)
            direction = np.zeros(len(r)); direction[support] = scale*vectors[:, -1]
            if direction[np.argmax(np.abs(direction))] < 0:
                direction *= -1
            derivative = .5*len(tasks)/likelihood.label_count*(1-values[-1])
            base = dense_nll(likelihood.blocks, state['common_diagonal'], r, state['discrepancy'])
            # Independent marginal construction checks the common/task separation and origin.
            np.testing.assert_allclose(base, value, atol=1e-8, rtol=1e-8)
            step = 1e-5
            actual = (dense_nll(likelihood.blocks, state['common_diagonal'], r+step*np.outer(direction, direction), state['discrepancy'])-base)/step
            np.testing.assert_allclose(actual, derivative, atol=2e-5, rtol=2e-4)
            names = ['intercept']+basis.names
            group_energy = {'intercept': float(direction[0]**2), **{g: float(np.sum(direction[1:][basis.group_index == i]**2))
                for i, g in enumerate(basis.group_names)}}
            rows.append({'layout': layout, 'head': head_name, 'source_label_count': likelihood.label_count,
                'posterior_task_coefficient_means': [m.tolist() for m in posterior_means],
                'whitened_posterior_second_moment_eigenvalues': values.tolist(),
                'leading_eigenvalue': float(values[-1]), 'rank_one_variance_source_nll_derivative': float(derivative),
                'finite_difference_derivative': float(actual), 'direction_coefficient_names': names,
                'direction': direction.tolist(), 'direction_group_squared_norm': group_energy,
                'native_source_nll': value, 'dense_source_nll': base})
    result = {'execution': spec, 'rows': rows,
        'cost': {'new_source_fits': 0, 'new_simulator_calls': 0, 'new_verification_calls': 0, 'new_terminal_integrations': 0},
        'cpu_seconds': time.process_time()-started}
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True); save(out/'results.json', result)
    print({'directions': [{k: r[k] for k in ('layout', 'head', 'leading_eigenvalue', 'rank_one_variance_source_nll_derivative', 'finite_difference_derivative')}
        for r in rows], 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
