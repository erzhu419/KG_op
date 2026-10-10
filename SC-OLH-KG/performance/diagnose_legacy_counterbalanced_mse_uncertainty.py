"""Conditional three-repeat measurement uncertainty of paired mean MSE."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.run_legacy_related_inventory_source import save, training_view

EXECUTION = 'performance/manifests/legacy_joint_population_counterbalanced_mse_uncertainty_v1.json'


def paired_interval(new, old, observed, mean_noise, replicates):
    delta = np.asarray(new)-old; count = len(delta)
    point_difference = (np.asarray(new)-observed)**2-(np.asarray(old)-observed)**2
    difference = float(np.mean(point_difference)); contributions = 4*delta**2*np.asarray(mean_noise)/count**2
    variance = float(np.sum(contributions)); se = np.sqrt(variance)
    degrees = variance**2/float(np.sum(contributions**2/(replicates-1)))
    width = float(t.ppf(.975, degrees)*se)
    return {'paired_mse_difference': difference, 'measurement_standard_error': float(se),
        'welch_degrees_of_freedom': degrees, 'approximate_95_interval': [difference-width, difference+width]}, point_difference


def run():
    started = time.process_time(); spec = read(EXECUTION); data = read(spec['source_data'])
    public = read(Path(spec['source_data']).with_name('frozen_inputs.json')); references = read(spec['reference_profiles'])['rows']
    labels = {i: training_view(data, [i], public['plan']['task_distribution']['L'])[0] for i in (6, 7)}
    rows = []
    for variant, path in spec['variants'].items():
        for row in read(path)['rows']:
            old = next(r for r in references if r['layout'] == row['parent_layout'] and r['task_index'] == row['task_index'] and r['mean_arm'] == row['mean_arm'])
            label = labels[row['task_index']]
            for h, head in enumerate(('f', 'g')):
                result, points = paired_interval(np.asarray(row['mean'])[2:, h], np.asarray(old['mean'])[2:, h],
                    label['response'][2:, h], label['noise'][2:, h], 3)
                np.testing.assert_allclose(result['paired_mse_difference'],
                    row['scores']['observed_head_mse'][h]-old['scores']['observed_head_mse'][h], atol=1e-12)
                greatest = np.argsort(np.abs(points))[-5:][::-1]
                rows.append({'variant': variant, 'layout': row['layout'], 'task_id': row['task_id'], 'mean_arm': row['mean_arm'],
                    'head': head, **result, 'largest_absolute_policy_contributions': [
                        {'candidate': int(j+2), 'mse_difference_contribution': float(points[j]/len(points))} for j in greatest]})
    result = {'execution': spec, 'rows': rows,
        'source_improvement_gates_remain_failed': ['refit216', 'frozen144_covariance216'],
        'cost': {'new_source_fits': 0, 'new_simulator_calls': 0, 'new_verification_calls': 0, 'new_terminal_integrations': 0},
        'cpu_seconds': time.process_time()-started}
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True); save(out/'results.json', result)
    print({'g_structural_intervals': [r for r in rows if r['head'] == 'g' and r['mean_arm'] == 'joint_structural'],
        'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
