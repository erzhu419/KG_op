"""Whole-source-task-out selection of paired versus independent mean laws."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_joint_common_covariance import restore_regularizer
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_counterbalanced_source_dependence import paired_direction
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.run_legacy_related_inventory_source import problem_for, save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.public_control_mean import public_control_mean_coordinates
from representation.source_function_prior import head_batches
from representation.source_task_covariance import SourceTaskDirectionLikelihood

EXECUTION = 'performance/manifests/legacy_joint_population_paired_taskout_validation_v1.json'


def fit_pair(inputs, native, direction):
    tasks = [{**t, **{k: np.asarray(t[k]) for k in ('psi', 'features', 'response', 'noise')}} for t in inputs['source_tasks']]
    basis = OrthogonalLowOrderBasis(inputs['public_mean_reference'])
    priors = [restore_regularizer(basis, h['regularizer'], tasks, i) for i, h in enumerate(native['heads'])]
    original = [h['joint_structural'] for h in native['heads']]
    model = SourceTaskDirectionLikelihood.from_paired_heads(priors, [head_batches(tasks, h) for h in range(2)],
        [[t['noise'][:, h] for t in tasks] for h in range(2)], [h['common_prior_coefficient_diagonal'] for h in original],
        [h['residual_coefficient_diagonal'] for h in original], [h['discrepancy_variance'] for h in original], direction['direction'])
    value, gradient, _ = model.evaluate(0.)
    np.testing.assert_allclose(value, direction['dense_joint_source_nll'], atol=1e-8)
    np.testing.assert_allclose(gradient[0], direction['rank_one_variance_source_nll_derivative'], atol=1e-8)
    opt = inputs['execution']['optimizer']
    result = minimize(lambda p: model.evaluate(p[0])[:2], [0.], jac=True, method=opt['method'], bounds=[(0., None)],
        options={'maxiter': opt['max_iterations'], 'ftol': opt['ftol'], 'gtol': opt['gtol']})
    value, _, state = model.evaluate(result.x[0])
    if not result.success:
        raise RuntimeError('task-out paired fit did not converge; no selection')
    return {'coefficient_mean': state['mean'].tolist(), 'population_mean_covariance': state['mean_covariance'].tolist(),
        'residual_coefficient_covariance': state['residual_covariance'].tolist(),
        'discrepancy_by_head': [h['discrepancy_variance'] for h in original],
        'task_direction_scale': float(result.x[0]), 'optimizer_converged': True,
        'optimizer_iterations': int(result.nit), 'source_nll': value, 'task_direction': direction['direction']}


def predictions(inputs, native, fit, psi):
    basis = OrthogonalLowOrderBasis(inputs['public_mean_reference'])
    design = np.column_stack([np.ones(len(psi)), basis.transform(psi)]); joint_design = block_diag(design, design)
    independent_mean = np.concatenate([h['joint_structural']['coefficient_mean'] for h in native['heads']])
    independent_covariance = block_diag(*[np.asarray(h['joint_structural']['population_mean_covariance'])+
        np.diag(h['joint_structural']['residual_coefficient_diagonal']) for h in native['heads']])
    paired_covariance = np.asarray(fit['population_mean_covariance'])+np.asarray(fit['residual_coefficient_covariance'])
    discrepancy = np.diag(np.repeat(fit['discrepancy_by_head'], len(psi)))
    return {name: {'mean': (joint_design@np.asarray(m)).tolist(),
        'covariance': (joint_design@c@joint_design.T+discrepancy).tolist()} for name, m, c in
        [('independent', independent_mean, independent_covariance), ('paired', fit['coefficient_mean'], paired_covariance)]}


def gaussian_score(mean, covariance, labels):
    factor = cho_factor(covariance, lower=True); residual = labels-mean
    return float(-.5*(len(labels)*np.log(2*np.pi)+2*np.log(np.diag(factor[0])).sum()+residual@cho_solve(factor, residual)))


def run():
    started = time.process_time(); spec = read(EXECUTION); coverage = read(spec['coverage_execution'])
    old_mixture = read(spec['deleted_mean_execution']); public = read(Path(coverage['source_data']).with_name('frozen_inputs.json'))
    bank = [tuple(p) for p in public['bank']]; out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    rows = []
    for layout in spec['layouts']:
        for deleted in spec['training_tasks']:
            branch = Path(coverage['output'])/f'layout{layout}'/f'task{deleted}'
            inputs = read(branch/'inputs.json')
            native = read(Path(old_mixture['output'])/f'layout{layout}'/f'task{deleted}'/'source_fit.json')
            direction = paired_direction(inputs, native, f'{layout}/drop{deleted}')
            fit = fit_pair(inputs, native, direction)
            provider = no_outcomes(problem_for(public['plan'], public['tasks'][deleted]))
            sites = coverage['layouts'][layout][deleted]
            raw = public_control_mean_coordinates(provider, bank)
            psi = (raw[sites]-inputs['mean_coordinate_center'])/inputs['mean_coordinate_scale']
            row = {'layout': layout, 'deleted_task': deleted, 'sites': sites, 'predictions': predictions(inputs, native, fit, psi),
                'task_direction_scale': fit['task_direction_scale'], 'leading_eigenvalue': direction['paired_leading_eigenvalue'],
                'optimizer_iterations': fit['optimizer_iterations'], 'optimizer_converged': fit['optimizer_converged']}
            local = out/f'layout{layout}'/f'task{deleted}'; local.mkdir(parents=True, exist_ok=True)
            save(local/'source_fit.json', fit); rows.append(row)
    save(out/'validation_predictions.json', {'execution': spec, 'all12_fits_and24_predictions_frozen': True, 'rows': rows})
    full = {layout: read(Path(coverage['output'])/f'layout{layout}'/'inputs.json') for layout in spec['layouts']}
    for row in rows:
        task = full[row['layout']]['source_tasks'][row['deleted_task']]
        labels = np.asarray(task['response'])[:, :2]; noise = np.asarray(task['noise'])[:, :2]
        joint_labels, joint_noise = labels.T.reshape(-1), noise.T.reshape(-1); n = len(labels)
        scores = {}
        for name, prediction in row['predictions'].items():
            mean = np.asarray(prediction['mean']); covariance = np.asarray(prediction['covariance'])+np.diag(joint_noise)
            scores[name] = {'joint_log_density_per_pair': gaussian_score(mean, covariance, joint_labels)/n,
                'head_log_density_per_label': [gaussian_score(mean[h*n:(h+1)*n], covariance[h*n:(h+1)*n, h*n:(h+1)*n], joint_labels[h*n:(h+1)*n])/n for h in range(2)],
                'mean_mse_by_head': np.mean((mean.reshape(2, n).T-labels)**2, axis=0).tolist()}
        row['scores'] = scores
    summary = {layout: {name: {k: np.mean([r['scores'][name][k] for r in rows if r['layout'] == layout], axis=0).tolist()
        for k in rows[0]['scores'][name]} for name in ('independent', 'paired')} for layout in spec['layouts']}
    differences = {layout: {'joint_density_difference': s['paired']['joint_log_density_per_pair']-s['independent']['joint_log_density_per_pair'],
        'g_density_difference': s['paired']['head_log_density_per_label'][1]-s['independent']['head_log_density_per_label'][1]} for layout, s in summary.items()}
    selected = 'paired' if all(d['joint_density_difference'] > 0 and d['g_density_difference'] >= 0 for d in differences.values()) else 'independent'
    result = {'execution': spec, 'rows': rows, 'summary': summary, 'differences': differences,
        'selected_by_training_taskout': selected, 'opened_source_gate_remains_failed': True,
        'cost': {'new_scalar_source_fits': 12, 'reused_native_deleted_mean_fits': 96, 'new_risk_fits': 0,
            'new_simulator_calls': 0, 'new_verification_calls': 0, 'new_terminal_integrations': 0}, 'cpu_seconds': time.process_time()-started}
    save(out/'results.json', result); print({k: result[k] for k in ('summary', 'differences', 'selected_by_training_taskout', 'cpu_seconds')}, flush=True)


if __name__ == '__main__':
    run()
