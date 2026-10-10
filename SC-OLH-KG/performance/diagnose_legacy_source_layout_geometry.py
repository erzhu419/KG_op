"""Cached source design and fixed-hyperparameter observation-weight attribution."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.run_legacy_related_inventory_source import save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis


def fixed_common_mean(tasks, designs, info, origin, *, median_noise=False):
    precision = np.diag(1/np.asarray(info['common_prior_coefficient_diagonal']))
    rhs = np.zeros(len(origin))
    residual = np.asarray(info['residual_coefficient_diagonal'])
    for task, x in zip(tasks, designs):
        noise = np.asarray(task['noise'])[:, 1]
        if median_noise:
            noise = np.full(len(noise), np.median(noise))
        covariance = (x*residual)@x.T+np.diag(noise+info['discrepancy_variance'])
        factor = cho_factor(covariance, lower=True)
        precision += x.T@cho_solve(factor, x)
        rhs += x.T@cho_solve(factor, np.asarray(task['response'])[:, 1]-x@origin)
    return origin+cho_solve(cho_factor(precision, lower=True), rhs)


def run():
    started = time.process_time()
    spec = read('performance/manifests/legacy_joint_population_source_layout_robustness_execution_v1.json')
    base = read(spec['reference_source_execution'])
    folders = {'0': ROOT/base['output'], **{n: ROOT/spec['output']/f'layout{n}' for n in spec['layouts']}}
    rows = []
    for name, folder in folders.items():
        inputs, fit = read(folder/'inputs.json'), read(folder/'source_fit.json')
        tasks = inputs['source_tasks']; basis = OrthogonalLowOrderBasis(inputs['public_mean_reference'])
        designs = [np.column_stack([np.ones(len(t['psi'])), basis.transform(t['psi'])]) for t in tasks]
        x = np.vstack(designs)
        q = np.column_stack([np.ones(len(inputs['public_mean_reference'])),
                             basis.transform(inputs['public_mean_reference'])])
        u, singular, v = np.linalg.svd(x, full_matrices=False)
        if np.linalg.matrix_rank(x) != x.shape[1]:
            raise ValueError('full-rank source geometry required for unregularized leverage attribution')
        influence = ((q@v.T)/singular)@u.T
        leverage = np.sum(influence**2, axis=1)
        noise = np.vstack([t['noise'] for t in tasks])[:, 1]
        origin = np.asarray(fit['heads'][1]['regularizer']['coefficient_mean'])
        attributions = []
        for arm in ('joint_structural', 'flat'):
            info = fit['heads'][1][arm]
            original = fixed_common_mean(tasks, designs, info, origin)
            # This checks the proposed source-weight attribution against the actual cached fit.
            np.testing.assert_allclose(original, info['coefficient_mean'], rtol=1e-8, atol=1e-8)
            median = fixed_common_mean(tasks, designs, info, origin, median_noise=True)
            attributions.append({'mean_arm': arm,
                'source_negative_log_likelihood_per_label': info['source_negative_log_likelihood_per_label'],
                'common_spectral_exponent': info.get('common_spectral_exponent'),
                'common_zero_group_count': info.get('common_zero_group_count'),
                'discrepancy_variance': info['discrepancy_variance'],
                'candidates': [{'candidate': j, 'original_source_common_g_mean': float(q[j]@original),
                    'within_task_median_noise_common_g_mean': float(q[j]@median),
                    'fixed_hyperparameter_noise_change': float(q[j]@(median-original))}
                    for j in spec['diagnostic_candidates']]})
        sites = inputs['execution']['source_site_indices_by_task']
        rows.append({'layout': name, 'source_design_rank': int(np.linalg.matrix_rank(x)),
            'coefficient_count': x.shape[1], 'distinct_source_policies': len(set(j for s in sites for j in s)),
            'unweighted_design_condition_number': float(singular[0]/singular[-1]),
            'public_catalog_ols_prediction_leverage_quantiles': np.quantile(leverage, [0, .5, .9, 1]).tolist(),
            'critical_geometry': [{'candidate': j, 'source_task_coverage': sum(j in s for s in sites),
                'ols_prediction_leverage': float(leverage[j]),
                'largest_absolute_ols_observation_weight': float(np.max(np.abs(influence[j])))}
                for j in spec['diagnostic_candidates']],
            'g_observed_label_variance_quantiles': np.quantile(noise, [0, .5, .9, 1]).tolist(),
            'conditional_mean_attributions': attributions})
    result = {'rows': rows, 'original_source_common_means_reproduced': True,
        'interpretation': 'source-only cached input geometry and conditional weighting attribution; median-noise substitution fixes all fitted hyperparameters and is not a refit or an adopted estimator. OLS leverage concerns the full unregularized function span, not actual Bayesian predictive uncertainty.',
        'cost': {'new_source_fits': 0, 'new_simulator_calls': 0, 'new_verification_calls': 0,
                 'new_terminal_integrations': 0}, 'cpu_seconds': time.process_time()-started}
    out = ROOT/'paper_artifacts/legacy_joint_population_source_layout_geometry_v1'
    out.mkdir(parents=True, exist_ok=True); save(out/'results.json', result)
    for row in rows:
        print({k: row[k] for k in ('layout', 'critical_geometry', 'conditional_mean_attributions')}, flush=True)


if __name__ == '__main__':
    run()
