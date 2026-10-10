"""Attribute already-scored correlation losses without fitting another model."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.run_legacy_related_inventory_source import save


def run():
    started = time.process_time()
    execution = read('performance/manifests/legacy_joint_population_fixed_marginal_correlation_execution_v1.json')
    coverage = read(execution['coverage_execution']); out = ROOT/execution['output']; result = read(out/'results.json')
    source = {layout: read(Path(coverage['output'])/f'layout{layout}'/'inputs.json') for layout in execution['layouts']}
    rows = []
    for row in result['rows']:
        fit = read(out/f"layout{row['layout']}"/f"task{row['deleted_task']}"/'source_fit.json')
        native, correlated = row['scores']['independent'], row['scores']['correlation_only']
        head_changes = np.asarray(correlated['head_log_density_per_label'])-native['head_log_density_per_label']
        joint_change = correlated['joint_log_density_per_pair']-native['joint_log_density_per_pair']
        task = source[row['layout']]['source_tasks'][row['deleted_task']]
        labels, noise = np.asarray(task['response'])[:, :2], np.asarray(task['noise'])[:, :2]; n = len(labels)
        calibration = {}
        for name, prediction in row['predictions'].items():
            mean, covariance = np.asarray(prediction['mean']), np.asarray(prediction['covariance'])
            values = []
            for h in (0, 1):
                s = slice(h*n, (h+1)*n); residual = labels[:, h]-mean[s]
                factor = cho_factor(covariance[s, s]+np.diag(noise[:, h]), lower=True)
                values.append(float(residual@cho_solve(factor, residual)/n))
            calibration[name] = values
        rows.append({'layout': row['layout'], 'deleted_task': row['deleted_task'],
            'task_correlation': fit['task_correlation'], 'whitened_task_minimum_eigenvalue': 1-abs(fit['task_correlation']),
            'training_nll_gain': fit['independent_source_nll']-fit['source_nll'],
            'joint_density_change': joint_change, 'head_density_changes': head_changes.tolist(),
            'joint_dependence_contribution': joint_change-float(head_changes.sum()),
            'head_mahalanobis_per_label': calibration})
    summary = {}
    for layout in execution['layouts']:
        group = [r for r in rows if r['layout'] == layout]
        summary[layout] = {'joint_improving_tasks': sum(r['joint_density_change'] > 0 for r in group),
            'g_improving_tasks': sum(r['head_density_changes'][1] > 0 for r in group),
            'training_nll_gain_range': [min(r['training_nll_gain'] for r in group), max(r['training_nll_gain'] for r in group)],
            'mean_joint_dependence_contribution': float(np.mean([r['joint_dependence_contribution'] for r in group])),
            'mean_head_density_changes': np.mean([r['head_density_changes'] for r in group], axis=0).tolist(),
            'mean_head_mahalanobis_per_label': {name: np.mean([r['head_mahalanobis_per_label'][name] for r in group], axis=0).tolist()
                for name in ('independent', 'correlation_only')}}
    diagnostic = {'execution': execution, 'rows': rows, 'summary': summary,
        'interpretation': 'Frozen-score algebra: joint density change equals f marginal change plus g marginal change plus dependence contribution. Native task marginal covariance is unchanged, but source conditioning changes common means/covariances. Mahalanobis ratios are descriptive under fitted Gaussian sample-mean noise, not hypothesis tests or raw-risk calibration. No new model is selected here.',
        'cost': {'new_source_fits': 0, 'new_forecasts': 0, 'new_simulator_calls': 0, 'new_verification_calls': 0},
        'cpu_seconds': time.process_time()-started}
    save(out/'diagnostics.json', diagnostic)
    print({'summary': summary, 'rows': rows, 'cpu_seconds': diagnostic['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
