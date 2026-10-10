"""Cached f level/shape residuals after the failed complete original144 gate."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.run_legacy_related_inventory_source import save, training_view

EXECUTION = 'performance/manifests/legacy_joint_population_f_reference_complete_attribution_execution_v1.json'


def residual_summary(row, labels):
    scores = {}
    for stage, predicted in (('prior', row['prior']), ('raw_paid', row)):
        error = np.asarray(predicted['mean'])[2:, 0]-labels['response'][2:, 0]
        level, shape = float(error.mean()**2), float(np.mean((error-error.mean())**2))
        scores[stage] = {'level_bias': float(error.mean()), 'level_mse': level, 'shape_mse': shape,
            'total_f_mse': float(np.mean(error**2))}
        np.testing.assert_allclose(level+shape, scores[stage]['total_f_mse'], atol=1e-15, rtol=1e-12)
    np.testing.assert_allclose(scores['raw_paid']['total_f_mse'], row['scores']['observed_head_mse'][0], atol=1e-15, rtol=1e-12)
    scores['raw_paid'].update(f_raw_density=row['scores']['response_batch_log_scores'][0],
        g_raw_density=row['scores']['response_batch_log_scores'][1], eta_f_mse=row['scores']['observed_head_mse'][3])
    return scores


def run():
    started = time.process_time(); spec = read(EXECUTION); result = read(spec['complete_results'])
    assert result['scientific_gate_interpreted'] and not result['original144_source_gate_passed']
    complete = read(spec['complete_profiles']); parent_spec = complete['execution']
    coverage = read(parent_spec['coverage_execution']); raw = read(coverage['source_data'])
    public = read(Path(coverage['source_data']).with_name('frozen_inputs.json'))
    old, previous = read(parent_spec['parent_reference_profiles'])['rows'], read(parent_spec['g_profiles'])['rows']
    rows = []
    for row in complete['rows']:
        layout, i = row['layout'], row['task_index']
        older = next(r for r in old if r['layout'] == parent_spec['parent_layouts'][layout] and r['task_index'] == i and r['mean_arm'] == 'joint_structural')
        middle = next(r for r in previous if r['layout'] == layout and r['task_index'] == i)
        labels = training_view(raw, [i], public['plan']['task_distribution']['L'])[0]
        cases = {name: residual_summary(r, labels) for name, r in (('original144', older), ('stage104', middle), ('mixture216', row))}
        actual, base = cases['mixture216']['raw_paid'], cases['original144']['raw_paid']
        gap = {key: actual[key]-base[key] for key in ('level_mse', 'shape_mse', 'total_f_mse', 'f_raw_density', 'g_raw_density', 'eta_f_mse')}
        a, b = np.asarray(row['mean'])[2:, 0], np.asarray(older['mean'])[2:, 0]
        standard_error = float(np.sqrt(4*np.sum((a-b)**2*labels['noise'][2:, 0])/len(a)**2))
        interval = [gap['total_f_mse']-1.96*standard_error, gap['total_f_mse']+1.96*standard_error]
        np.testing.assert_allclose(gap['level_mse']+gap['shape_mse'], gap['total_f_mse'], atol=1e-15, rtol=1e-12)
        rows.append({'layout': layout, 'task_index': i, 'cases': cases, 'mixture216_minus_original144': gap,
            'fixed_prediction_f_mse_difference_measurement_se': standard_error,
            'approximate95percent_measurement_interval': interval,
            'f_mse_increase_fraction': actual['total_f_mse']/base['total_f_mse']-1})
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    save(out/'results.json', {'execution': spec, 'rows': rows, 'cost': spec['cost'],
        'cpu_seconds': time.process_time()-started, 'new_candidate_evaluated': False,
        'target_promoted': False, 'independent_confirmation': False})
    print([(r['layout'], r['task_index'], r['f_mse_increase_fraction'], r['approximate95percent_measurement_interval']) for r in rows], flush=True)


if __name__ == '__main__':
    run()
