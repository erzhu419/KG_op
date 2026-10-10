"""Select fixed-marginal f/g correlation using whole training tasks only."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_joint_common_covariance import restore_regularizer
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_counterbalanced_source_dependence import paired_direction
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.diagnose_legacy_task_covariance_direction import dense_nll
from performance.run_legacy_related_inventory_source import problem_for, save
from performance.validate_legacy_paired_taskout import gaussian_score, predictions
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.public_control_mean import public_control_mean_coordinates
from representation.source_function_prior import head_batches
from representation.source_task_correlation import SourceFixedMarginalCorrelationLikelihood
from representation.source_task_covariance import SourceTaskDirectionLikelihood

EXECUTION = 'performance/manifests/legacy_joint_population_fixed_marginal_correlation_execution_v1.json'


def fit_correlation(inputs, native, direction):
    tasks = [{**t, **{k: np.asarray(t[k]) for k in ('psi', 'features', 'response', 'noise')}} for t in inputs['source_tasks']]
    basis = OrthogonalLowOrderBasis(inputs['public_mean_reference'])
    priors = [restore_regularizer(basis, h['regularizer'], tasks, i) for i, h in enumerate(native['heads'])]
    original = [h['joint_structural'] for h in native['heads']]
    base = SourceTaskDirectionLikelihood.from_paired_heads(priors, [head_batches(tasks, h) for h in range(2)],
        [[t['noise'][:, h] for t in tasks] for h in range(2)], [h['common_prior_coefficient_diagonal'] for h in original],
        [h['residual_coefficient_diagonal'] for h in original], [h['discrepancy_variance'] for h in original], direction['direction'])
    model = SourceFixedMarginalCorrelationLikelihood(base)
    zero, gradient, baseline = model.evaluate(0.)
    np.testing.assert_allclose(zero, direction['dense_joint_source_nll'], atol=1e-8)
    np.testing.assert_allclose(baseline['mean'], np.concatenate([h['coefficient_mean'] for h in original]), atol=1e-8)
    width = model.width
    for h in (0, 1):
        section = slice(h*width, (h+1)*width)
        np.testing.assert_allclose(baseline['mean_covariance'][section, section], original[h]['population_mean_covariance'], atol=1e-8)
    dense = lambda gamma: dense_nll(base.blocks, base.common_diagonal,
        np.diag(base.residual_diagonal)+gamma*model.derivative, 0.)
    numerical = (dense(1e-5)-dense(-1e-5))/2e-5
    np.testing.assert_allclose(gradient[0], numerical, atol=1e-7, rtol=1e-5)
    opt = inputs['execution']['optimizer']
    result = minimize(lambda p: model.evaluate(p[0])[:2], [0.], jac=True, method=opt['method'], bounds=[(-1., 1.)],
        options={'maxiter': opt['max_iterations'], 'ftol': opt['ftol'], 'gtol': opt['gtol']})
    if not result.success:
        raise RuntimeError('correlation fit did not converge; no selection')
    value, fitted_gradient, state = model.evaluate(result.x[0])
    for h in (0, 1):
        section = slice(h*width, (h+1)*width)
        np.testing.assert_array_equal(state['residual_covariance'][section, section], np.diag(original[h]['residual_coefficient_diagonal']))
    support = base.residual_diagonal > 0; scale = np.sqrt(base.residual_diagonal[support])
    whitened = state['residual_covariance'][np.ix_(support, support)]/np.outer(scale, scale)
    minimum_eigenvalue = float(np.linalg.eigvalsh(whitened).min())
    if minimum_eigenvalue < -1e-12:
        raise RuntimeError('fitted correlation violates native task covariance PSD')
    return {'coefficient_mean': state['mean'].tolist(), 'population_mean_covariance': state['mean_covariance'].tolist(),
        'residual_coefficient_covariance': state['residual_covariance'].tolist(),
        'discrepancy_by_head': [h['discrepancy_variance'] for h in original],
        'task_correlation': float(result.x[0]), 'optimizer_converged': True,
        'optimizer_iterations': int(result.nit), 'source_nll': value, 'independent_source_nll': zero,
        'fitted_nll_derivative': float(fitted_gradient[0]), 'task_direction': direction['direction'],
        'normalized_direction': model.normalized_direction.tolist(),
        'checks': {'native_zero_correlation_nll_mean_and_common_covariance_reproduced': True,
            'native_task_marginal_covariances_unchanged': True, 'whitened_minimum_eigenvalue': minimum_eigenvalue,
            'zero_correlation_gradient': float(gradient[0]), 'dense_finite_difference_gradient': float(numerical),
            'common_posterior_cross_head_covariance_norm': float(np.linalg.norm(state['mean_covariance'][:width, width:]))}}


def run():
    started = time.process_time(); spec = read(EXECUTION); coverage = read(spec['coverage_execution'])
    old_mixture = read(spec['deleted_mean_execution'])
    public = read(Path(coverage['source_data']).with_name('frozen_inputs.json'))
    bank = [tuple(p) for p in public['bank']]; out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    rows = []
    for layout in spec['layouts']:
        for deleted in spec['training_tasks']:
            inputs = read(Path(coverage['output'])/f'layout{layout}'/f'task{deleted}'/'inputs.json')
            native = read(Path(old_mixture['output'])/f'layout{layout}'/f'task{deleted}'/'source_fit.json')
            direction = paired_direction(inputs, native, f'{layout}/drop{deleted}')
            fit = fit_correlation(inputs, native, direction)
            provider = no_outcomes(problem_for(public['plan'], public['tasks'][deleted]))
            sites = coverage['layouts'][layout][deleted]
            raw = public_control_mean_coordinates(provider, bank)
            psi = (raw[sites]-inputs['mean_coordinate_center'])/inputs['mean_coordinate_scale']
            forecast = predictions(inputs, native, fit, psi)
            forecast['correlation_only'] = forecast.pop('paired')
            rows.append({'layout': layout, 'deleted_task': deleted, 'sites': sites, 'predictions': forecast,
                'task_correlation': fit['task_correlation'], 'optimizer_iterations': fit['optimizer_iterations'],
                'optimizer_converged': fit['optimizer_converged'], 'checks': fit['checks']})
            local = out/f'layout{layout}'/f'task{deleted}'; local.mkdir(parents=True, exist_ok=True)
            save(local/'source_fit.json', fit)
    save(out/'validation_predictions.json', {'execution': spec, 'all12_fits_and24_predictions_frozen': True, 'rows': rows})
    # Deleted outcomes enter scoring only after all fits and forecasts are saved.
    full = {layout: read(Path(coverage['output'])/f'layout{layout}'/'inputs.json') for layout in spec['layouts']}
    for row in rows:
        task = full[row['layout']]['source_tasks'][row['deleted_task']]
        labels = np.asarray(task['response'])[:, :2]; noise = np.asarray(task['noise'])[:, :2]
        joint_labels, joint_noise = labels.T.reshape(-1), noise.T.reshape(-1); n = len(labels)
        row['scores'] = {}
        for name, prediction in row['predictions'].items():
            mean = np.asarray(prediction['mean']); covariance = np.asarray(prediction['covariance'])+np.diag(joint_noise)
            row['scores'][name] = {'joint_log_density_per_pair': gaussian_score(mean, covariance, joint_labels)/n,
                'head_log_density_per_label': [gaussian_score(mean[h*n:(h+1)*n], covariance[h*n:(h+1)*n, h*n:(h+1)*n], joint_labels[h*n:(h+1)*n])/n for h in range(2)],
                'mean_mse_by_head': np.mean((mean.reshape(2, n).T-labels)**2, axis=0).tolist()}
    summary = {layout: {name: {k: np.mean([r['scores'][name][k] for r in rows if r['layout'] == layout], axis=0).tolist()
        for k in rows[0]['scores'][name]} for name in ('independent', 'correlation_only')} for layout in spec['layouts']}
    differences = {layout: {'joint_density_difference': s['correlation_only']['joint_log_density_per_pair']-s['independent']['joint_log_density_per_pair'],
        'g_density_difference': s['correlation_only']['head_log_density_per_label'][1]-s['independent']['head_log_density_per_label'][1]} for layout, s in summary.items()}
    selected = 'correlation_only' if all(d['joint_density_difference'] > 0 and d['g_density_difference'] >= 0 for d in differences.values()) else 'independent'
    full_fits = []
    if selected == 'correlation_only':
        for layout in spec['layouts']:
            folder = Path(coverage['output'])/f'layout{layout}'
            inputs, native = read(folder/'inputs.json'), read(folder/'source_fit.json')
            fit = fit_correlation(inputs, native, paired_direction(inputs, native, layout))
            local = out/f'layout{layout}'; save(local/'source_fit.json', fit)
            full_fits.append({'layout': layout, 'task_correlation': fit['task_correlation'], 'source_nll': fit['source_nll'], 'checks': fit['checks']})
    result = {'execution': spec, 'rows': rows, 'summary': summary, 'differences': differences,
        'selected_by_training_taskout': selected, 'full_source_fits': full_fits,
        'paid_holdout_forecasts_executed': False,
        'cost': {'new_scalar_source_fits': 12+len(full_fits), 'reused_native_deleted_mean_fits': 96,
            'new_risk_fits': 0, 'new_simulator_calls': 0, 'new_verification_calls': 0, 'new_terminal_integrations': 0},
        'cpu_seconds': time.process_time()-started}
    save(out/'results.json', result)
    print({k: result[k] for k in ('summary', 'differences', 'selected_by_training_taskout', 'full_source_fits', 'cpu_seconds')}, flush=True)


if __name__ == '__main__':
    run()
