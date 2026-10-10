"""Frozen mean-space capacity and source-assignment estimation attribution."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.check_legacy_source_layout_robustness import models_at_query, ARMS
from performance.run_legacy_related_inventory_source import save, training_view
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis

EXECUTION = 'performance/manifests/legacy_joint_population_covered_mean_attribution_execution_v1.json'


def design(basis, psi):
    return np.column_stack([np.ones(len(psi)), basis.transform(psi)])


def project(design_matrix, truth):
    u, _, _ = np.linalg.svd(design_matrix, full_matrices=False)
    rank = int(np.linalg.matrix_rank(design_matrix))
    projector = u[:, :rank]@u[:, :rank].T
    fitted = projector@truth
    return fitted, projector, rank


def run():
    started = time.process_time(); spec = read(EXECUTION); source = read(spec['source_execution'])
    snapshots = read(spec['target_snapshot'])['snapshots']; targets = read(spec['target_profiles'])['rows']
    source_rows = read(spec['source_profiles'])['rows']; truth_folds = read(spec['target_scoring'])['folds']
    members, projections, target_rows, source_points = {}, {}, [], []
    for name in spec['layouts']:
        folder = Path(source['output'])/f'layout{name}'
        group = [(read(folder/'inputs.json'), read(folder/'source_fit.json'), read(folder/'risk_source_fit.json'))]
        group += [(read(folder/f'task{i}'/'inputs.json'), read(folder/f'task{i}'/'gp_source_fit.json'),
                   read(folder/f'task{i}'/'risk_source_fit.json')) for i in source['training_tasks']]
        members[name] = group; inp, means, _ = group[0]
        basis = OrthogonalLowOrderBasis(inp['public_mean_reference'])
        for frozen in (s for s in snapshots if s['layout'] == name):
            task = frozen['query']['task_id']; x = design(basis, frozen['mean_queries'][0])
            fold = next(f for f in truth_folds if f['task_id'] == task)
            truth = np.asarray(fold['scoring_truth'])[:, [0, 2]]
            fitted, projector, rank = project(x, truth); residual = fitted-truth
            projections[(name, task)] = projector
            for arm in ARMS:
                row = next(r for r in targets if r['layout'] == name and r['task_id'] == task and r['mean_arm'] == arm)
                prior, posterior = np.asarray(row['prior']['mean'])[:, :2], np.asarray(row['mean'])[:, :2]
                if arm != 'joint_gp':
                    reconstructed = np.column_stack([x@np.asarray(h[arm]['coefficient_mean']) for h in means['heads']])
                    # Stop if a cached source fit/query does not actually produce the attributed prior.
                    np.testing.assert_allclose(reconstructed, prior, rtol=1e-8, atol=1e-8)
                error = posterior-truth
                np.testing.assert_allclose(np.mean(error[2:]**2, axis=0), row['scores']['truth_mse'][:2])
                target_rows.append({'layout': name, 'task_id': task, 'mean_arm': arm, 'basis_rank': rank,
                    'oracle_full_catalog_projection_mse': np.mean(residual**2, axis=0).tolist(),
                    'oracle_projection_unpaid_mse': np.mean(residual[2:]**2, axis=0).tolist(),
                    'prior_unpaid_mse': np.mean((prior[2:]-truth[2:])**2, axis=0).tolist(),
                    'posterior_unpaid_mse': np.mean(error[2:]**2, axis=0).tolist(),
                    'points': [{'candidate': j, 'oracle_projection_residual': residual[j].tolist(),
                        'prior_error': (prior[j]-truth[j]).tolist(), 'paid_mean_change': (posterior[j]-prior[j]).tolist(),
                        'posterior_error': error[j].tolist(), 'prior_mean_sd': np.sqrt(row['prior']['variance'][j][:2]).tolist(),
                        'posterior_mean_sd': np.sqrt(row['variance'][j][:2]).tolist()} for j in spec['diagnostic_candidates']]})
    comparisons = []
    for task in dict.fromkeys(s['query']['task_id'] for s in snapshots):
        left, right = [next(r for r in target_rows if r['layout'] == n and r['task_id'] == task
                           and r['mean_arm'] == 'joint_structural') for n in spec['layouts']]
        comparisons.append({'task_id': task,
            'function_projector_max_difference': float(np.max(np.abs(projections[(spec['layouts'][0], task)]-projections[(spec['layouts'][1], task)]))),
            'critical_g_assignment_changes': [{'candidate': a['candidate'],
                'prior_g_mean_change': b['prior_error'][1]-a['prior_error'][1],
                'posterior_g_mean_change': b['posterior_error'][1]-a['posterior_error'][1]} for a, b in zip(left['points'], right['points'])]})
    data = read(source['source_data']); public = read(Path(source['source_data']).with_name('frozen_inputs.json'))
    for name, group in members.items():
        for i in source['heldout_tasks']:
            labels = training_view(data, [i], public['plan']['task_distribution']['L'])[0]
            task = data['tasks'][i]
            for arm, model in models_at_query(group, [g[0]['heldout_mean_psi'][str(i)] for g in group],
                    [g[0]['heldout_risk_psi'][str(i)] for g in group]):
                for j in (0, 1):
                    value = task['Y_replicates'][j][0]; model.update_raw(j, [value[0], value[1]-task['tau']])
                means = sum(w*np.column_stack([f.predict()[0][:, 0], g.predict()[0][:, 0]])
                            for w, (f, g) in zip(model.weights, model.components))
                old = next(r for r in source_rows if r['layout'] == name and r['task_index'] == i and r['mean_arm'] == arm)
                np.testing.assert_allclose(means, np.asarray(old['mean'])[:, :2], rtol=1e-8, atol=1e-8)
                for j in spec['diagnostic_candidates']:
                    y = labels['response'][j]; density = model.batch_log_scores(j, y[:2], y[[3, 2]], 3)
                    source_points.append({'layout': name, 'task_id': task['task_id'], 'mean_arm': arm,
                        'candidate': j, 'observed_g_error': float(means[j, 1]-y[1]),
                        'g_three_repeat_log_density': density['heads'][1], 'joint_three_repeat_log_density': density['joint']})
    summary = {name: {arm: {k: np.mean([r[k] for r in target_rows if r['layout'] == name and r['mean_arm'] == arm], axis=0).tolist()
        for k in ('oracle_full_catalog_projection_mse', 'oracle_projection_unpaid_mse', 'prior_unpaid_mse', 'posterior_unpaid_mse')}
        for arm in ARMS} for name in spec['layouts']}
    result = {'execution': spec, 'head_order': ['f', 'g'], 'summary': summary, 'target_rows': target_rows,
        'assignment_comparisons': comparisons, 'source_points': source_points,
        'interpretation': 'full-catalog cached-truth projection tests finite dictionary capacity only; no projected value enters a prior, paid posterior, recommendation or verification. The unpaid residual is evaluated after a full-catalog projection, not a separately optimized unpaid lower bound.',
        'cost': {**spec['cost'], 'source_point_batch_density_evaluations': len(source_points)},
        'cpu_seconds': time.process_time()-started}
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True); save(out/'results.json', result)
    print({'summary': summary, 'maximum_assignment_projector_difference': max(c['function_projector_max_difference'] for c in comparisons),
           'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
