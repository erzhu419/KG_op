"""Fit a source-only paired f/g task covariance scale; no paid forecast yet."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_joint_common_covariance import restore_regularizer
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.run_legacy_related_inventory_source import save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.source_function_prior import head_batches
from representation.source_task_covariance import SourceTaskDirectionLikelihood

EXECUTION = 'performance/manifests/legacy_joint_population_paired_task_direction_fit_v1.json'


def run():
    started = time.process_time(); spec = read(EXECUTION); coverage = read(spec['coverage_execution'])
    directions = read(spec['direction_results'])['paired_source_directions']; rows = []
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    for layout in spec['layouts']:
        folder = Path(coverage['output'])/f'layout{layout}'
        inputs, native = read(folder/'inputs.json'), read(folder/'source_fit.json')
        direction = next(r for r in directions if r['layout'] == layout)
        tasks = [{**t, **{k: np.asarray(t[k]) for k in ('psi', 'features', 'response', 'noise')}} for t in inputs['source_tasks']]
        basis = OrthogonalLowOrderBasis(inputs['public_mean_reference']); heads = native['heads']
        priors = [restore_regularizer(basis, h['regularizer'], tasks, i) for i, h in enumerate(heads)]
        original = [h['joint_structural'] for h in heads]
        model = SourceTaskDirectionLikelihood.from_paired_heads(priors, [head_batches(tasks, h) for h in range(2)],
            [[t['noise'][:, h] for t in tasks] for h in range(2)],
            [h['common_prior_coefficient_diagonal'] for h in original], [h['residual_coefficient_diagonal'] for h in original],
            [h['discrepancy_variance'] for h in original], direction['direction'])
        value, gradient, state = model.evaluate(0.)
        np.testing.assert_allclose(value, direction['dense_joint_source_nll'], atol=1e-8)
        np.testing.assert_allclose(gradient[0], direction['rank_one_variance_source_nll_derivative'], atol=1e-8)
        np.testing.assert_allclose(state['mean'], np.concatenate([h['coefficient_mean'] for h in original]), atol=1e-8)
        opt = inputs['execution']['optimizer']; fit_started = time.process_time()
        result = minimize(lambda p: model.evaluate(p[0])[:2], [0.], jac=True, method=opt['method'], bounds=[(0., None)],
            options={'maxiter': opt['max_iterations'], 'ftol': opt['ftol'], 'gtol': opt['gtol']})
        value, gradient, state = model.evaluate(result.x[0])
        row = {'layout': layout, 'source_paired_records': 72, 'source_label_count': model.label_count,
            'task_direction_scale': float(result.x[0]), 'source_negative_log_likelihood_per_label': value,
            'native_independent_head_source_nll': direction['dense_joint_source_nll'],
            'optimizer_converged': bool(result.success), 'optimizer_iterations': int(result.nit),
            'optimizer_gradient': float(gradient[0]), 'optimizer_message': str(result.message),
            'source_fit_cpu_seconds': time.process_time()-fit_started,
            'coefficient_names': direction['coefficient_names'], 'coefficient_mean': state['mean'].tolist(),
            'population_mean_covariance': state['mean_covariance'].tolist(),
            'residual_coefficient_covariance': state['residual_covariance'].tolist(),
            'discrepancy_by_head': [h['discrepancy_variance'] for h in original],
            'basis_reference': str(folder/'inputs.json'), 'task_direction': direction['direction']}
        local = out/f'layout{layout}'; local.mkdir(parents=True, exist_ok=True)
        save(local/'source_fit.json', row)
        if not result.success:
            raise RuntimeError('paired source covariance fit incomplete; do not advance')
        rows.append({k: v for k, v in row.items() if k not in ('coefficient_names', 'coefficient_mean',
            'population_mean_covariance', 'residual_coefficient_covariance', 'task_direction')})
    result = {'execution': spec, 'fits': rows,
        'cost': {'new_scalar_source_fits': 2, 'new_risk_fits': 0, 'new_simulator_calls': 0,
            'new_verification_calls': 0, 'new_terminal_integrations': 0}, 'cpu_seconds': time.process_time()-started}
    save(out/'results.json', result); print(result, flush=True)


if __name__ == '__main__':
    run()
