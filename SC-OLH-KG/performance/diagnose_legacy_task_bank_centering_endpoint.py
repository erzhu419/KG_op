"""Cached development robustness and source prior/paid residual attribution."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.calibrate_legacy_task_transfer import NATIVE
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.check_legacy_task_bank_centering import CENTERED
from performance.run_legacy_related_inventory_source import save, training_view

EXECUTION = 'performance/manifests/legacy_joint_population_task_bank_centering_endpoint_execution_v1.json'


def residual_parts(predicted, observed):
    residual = np.asarray(predicted)-observed
    level = float(residual.mean()); shape = float(np.mean((residual-level)**2))
    return {'g_mse': float(np.mean(residual**2)), 'mean_residual': level,
        'level_residual_squared': level**2, 'centered_shape_residual_mse': shape}


def run():
    started = time.process_time(); spec = read(EXECUTION)
    training = read(spec['training_profiles'])['rows']; holdout = read(spec['holdout_execution'])
    native_spec = read(holdout['native_execution']); native = read(Path(native_spec['output'])/'calibration_profiles.json')['rows']
    candidates = read(spec['holdout_profiles'])['rows']; coverage = read(holdout['coverage_execution'])
    raw = read(coverage['source_data']); public = read(Path(coverage['source_data']).with_name('frozen_inputs.json'))
    labels = {i: training_view(raw, [i], public['plan']['task_distribution']['L'])[0] for i in spec['heldout_tasks']}; del raw
    gp_execution = read(spec['training_gp_execution']); gp = read(Path(gp_execution['output'])/'results.json')
    training_summary = {}; training_contrasts = []
    for layout in spec['layouts']:
        for i in spec['training_tasks']:
            new, old = (next(r for r in training if r['layout'] == layout and r['deleted_task'] == i and r['case'] == case)
                for case in (CENTERED, NATIVE))
            training_contrasts.append({'layout': layout, 'deleted_task': i,
                'paid_differences': {k: new['scores']['raw_paid'][k]-old['scores']['raw_paid'][k] for k in new['scores']['raw_paid']}})
        local = [r for r in training_contrasts if r['layout'] == layout]
        all_mse = sum(r['paid_differences']['g_mse'] for r in local)
        one = next(r['paid_differences']['g_mse'] for r in local if r['deleted_task'] == 1)
        training_summary[layout] = {'all6_average_paid_mse_difference': all_mse/6,
            'other5_average_paid_mse_difference': (all_mse-one)/5,
            'task1_fraction_of_summed_mse_improvement': one/all_mse,
            'other5_average_raw_density_difference': float(np.mean([r['paid_differences']['g_raw_batch_density'] for r in local if r['deleted_task'] != 1])),
            'mse_improved_task_count': sum(r['paid_differences']['g_mse'] < 0 for r in local),
            'raw_density_improved_task_count': sum(r['paid_differences']['g_raw_batch_density'] > 0 for r in local)}
    rows = []
    for row in candidates:
        layout, i = row['layout'], row['task_index']
        old = next(r for r in native if r['layout'] == layout and r['task_index'] == i)
        observed = np.asarray(labels[i]['response'])[2:, 1]; parts = {}
        for stage in ('prior', 'raw_paid'):
            new_prediction, old_prediction = (row['prior'], old['prior']) if stage == 'prior' else (row, old)
            new_mean, old_mean = np.asarray(new_prediction['mean'])[2:], np.asarray(old_prediction['mean'])[2:]
            new, original = residual_parts(new_mean[:, 1], observed), residual_parts(old_mean[:, 1], observed)
            np.testing.assert_allclose(new['level_residual_squared']+new['centered_shape_residual_mse'], new['g_mse'], atol=1e-12)
            np.testing.assert_allclose(original['level_residual_squared']+original['centered_shape_residual_mse'], original['g_mse'], atol=1e-12)
            if stage == 'raw_paid':
                for values, expected in ((new, row), (original, old)):
                    np.testing.assert_allclose(values['g_mse'], expected['scores']['observed_head_mse'][1], atol=1e-12)
            parts[stage] = {'centered': new, 'native': original,
                'centered_minus_native': {k: new[k]-original[k] for k in new},
                'g_mean_shift_rms': float(np.sqrt(np.mean((new_mean[:, 1]-old_mean[:, 1])**2))),
                'mean_g_latent_variance_ratio': float(np.mean(np.asarray(new_prediction['variance'])[2:, 1])/np.mean(np.asarray(old_prediction['variance'])[2:, 1])),
                'eta_g_mean_shift_rms': float(np.sqrt(np.mean((new_mean[:, 2]-old_mean[:, 2])**2))),
                'mean_g_episode_variance_ratio': float(np.mean(np.asarray(new_prediction['expected_episode_variance'])[2:, 1])/np.mean(np.asarray(old_prediction['expected_episode_variance'])[2:, 1])),
                'identity_weight_l1_change': float(np.sum(np.abs(np.asarray(new_prediction['identity_weights'])-np.asarray(old_prediction['identity_weights']))))}
        rows.append({'layout': layout, 'task_index': i, 'stages': parts})
    result = {'execution': spec, 'training_summary': training_summary, 'training_contrasts': training_contrasts,
        'training_gp_control_summary': gp['summary'], 'rows': rows, 'cached_paid_mse_reproduced': True,
        'cost': spec['cost'], 'cpu_seconds': time.process_time()-started,
        'target_promoted': False, 'independent_confirmation': False}
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True); save(out/'results.json', result)
    print({'training_summary': training_summary, 'source_endpoint_mse_changes': [{'layout': r['layout'], 'task_index': r['task_index'],
        'changes': {stage: r['stages'][stage]['centered_minus_native'] for stage in ('prior', 'raw_paid')}} for r in rows],
        'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
