"""Refit source f learners after deleting each inner source task."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.run_legacy_related_inventory_source import save
from performance.validate_legacy_paired_taskout import gaussian_score
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis, fit_orthogonal_function_prior
from representation.public_control_mean import source_prior_moments
from representation.source_function_prior import head_batches
from representation.source_joint_population import fit_joint_source_population, fit_joint_common_source_population
from representation.source_kernel_population import (fit_unequal_site_kernel_population,
    fit_joint_common_unequal_site_kernel_population, predict_unequal_site_kernel_population)
from representation.source_known_noise_gp import fit_known_noise_gp

EXECUTION = 'performance/manifests/legacy_joint_population_f_inner_task_learning_execution_v1.json'


def score(prediction, task):
    mean, covariance = np.asarray(prediction['mean']), np.asarray(prediction['covariance'])
    observed, noise = np.asarray(task['response'])[:, 0], np.asarray(task['noise'])[:, 0]
    error = mean-observed; level = float(error.mean()); shape_mse = float(np.mean((error-level)**2))
    level_variance = float((covariance.sum()+noise.sum())/len(mean)**2)
    return {'mse': float(np.mean(error**2)), 'signed_level_error': level, 'level_mse': level**2,
        'centered_shape_mse': shape_mse, 'predictive_observed_level_variance': level_variance,
        'level_error_in_predictive_standard_deviations': level/np.sqrt(level_variance),
        'gaussian_batch_log_density_per_label': gaussian_score(mean, covariance+np.diag(noise), observed)/len(mean)}


def run():
    started = time.process_time(); spec = read(EXECUTION); context = read(spec['source_context_execution'])
    coverage, _ = source_paths(context); out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    rows, heldouts = [], {}; fit_count = 0
    for layout in spec['layouts']:
        inputs = read(Path(coverage['output'])/f'layout{layout}'/f"task{spec['outer_deleted_task']}"/'inputs.json')
        source = inputs['source_tasks']; names = [task['name'] for task in source]
        assert set(names) == {f'related_inventory_{i:02d}' for i in range(6) if i != spec['outer_deleted_task']}
        basis = OrthogonalLowOrderBasis(inputs['public_mean_reference'])
        for query in source:
            keep = [{**task, **{key: np.asarray(task[key]) for key in ('psi', 'features', 'response', 'noise')}}
                    for task in source if task['name'] != query['name']]
            batches, noises = head_batches(keep, 0), [task['noise'][:, 0] for task in keep]
            assert len(keep) == 4 and sum(len(t['psi']) for t in keep) == 48
            regularizer = fit_orthogonal_function_prior(batches, noises, basis,
                max_iterations=spec['regularizer_max_iterations']); fit_count += 1
            fixed = fit_joint_source_population(regularizer, batches, noises,
                max_iterations=spec['fixed_population_max_iterations']); fit_count += 1
            structural = fit_joint_common_source_population(regularizer, batches, noises,
                fixed.diagnostics, optimizer=spec['optimizer']); fit_count += 1
            anchor = fit_known_noise_gp(keep, 0, spec['gp_anchor_max_iterations']); fit_count += 1
            gp_fixed = fit_unequal_site_kernel_population(keep, 0, anchor, optimizer=spec['optimizer']); fit_count += 1
            gp = fit_joint_common_unequal_site_kernel_population(keep, 0, gp_fixed, optimizer=spec['optimizer']); fit_count += 1
            fits = {'regularizer': regularizer.diagnostics, 'fixed_structural': fixed.diagnostics,
                    'structural': structural.diagnostics, 'gp_anchor': anchor, 'gp_fixed': gp_fixed, 'gp': gp}
            folder = out/f'layout{layout}'/query['name']; folder.mkdir(parents=True, exist_ok=True)
            save(folder/'source_fit.json', {'inner_training_task_names': [t['name'] for t in keep], 'fits': fits})
            if not all(fit['optimizer_converged'] for fit in fits.values()):
                save(out/'results.json', {'execution': spec, 'all_inner_fits_converged': False,
                    'failed_fold': {'layout': layout, 'query': query['name']}, 'new_f_source_fits': fit_count,
                    'cpu_seconds': time.process_time()-started, 'forecasts_scored': False})
                raise RuntimeError('inner source fit incomplete;no completion or scoring')
            predictions = {}
            for arm, (mean, covariance) in {'structural': source_prior_moments(structural, np.asarray(query['psi'])),
                    'source_gp': predict_unequal_site_kernel_population(np.asarray(query['psi']), gp)}.items():
                predictions[arm] = {'mean': mean.tolist(), 'covariance': covariance.tolist()}
            row = {'layout': layout, 'outer_deleted_task': spec['outer_deleted_task'], 'inner_query_task': query['name'],
                'inner_training_task_names': [t['name'] for t in keep], 'source_fit': str((folder/'source_fit.json').relative_to(ROOT)),
                'prediction': predictions}
            rows.append(row); heldouts[(layout, query['name'])] = query
            print('inner source fits and predictions frozen', layout, query['name'], flush=True)
    save(out/'predictions.json', {'execution': spec, 'all_ten_inner_forecasts_frozen_before_scores': True, 'rows': rows})
    for row in rows:
        row['scores'] = {arm: score(prediction, heldouts[(row['layout'], row['inner_query_task'])])
                         for arm, prediction in row['prediction'].items()}
    summary = {layout: {arm: {key: float(np.mean([row['scores'][arm][key] for row in rows if row['layout'] == layout]))
                for key in rows[0]['scores'][arm]} for arm in ('structural', 'source_gp')} for layout in spec['layouts']}
    result = {'execution': spec, 'rows': rows, 'summary': summary, 'all_inner_fits_converged': True,
        'all_ten_inner_forecasts_frozen_before_scores': True, 'cpu_seconds': time.process_time()-started,
        'cost': {**spec['cost'], 'actual_new_f_source_fits': fit_count, 'actual_source_validation_forecasts': 20},
        'outer_query_paid_or_score_outcomes_read': False, 'source6_7_evaluated': False, 'target_promoted': False,
        'independent_confirmation': False, 'no_new_prior_adopted': True,
        'validation_contract': 'Each inner held task is absent from every response-based fit,empirical center and noise statistic. Public mean coordinates/reference remain fixed and outcome-free. Overlapping training-only diagnostic;not independent confirmation.'}
    save(out/'results.json', result); print({'summary': summary, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
