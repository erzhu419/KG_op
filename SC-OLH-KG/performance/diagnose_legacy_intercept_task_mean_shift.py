"""Post-hoc mean-shift attribution of the frozen source holdout contrast."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.run_legacy_related_inventory_source import save, training_view

EXECUTION = 'performance/manifests/legacy_joint_population_intercept_task_holdout_execution_v1.json'


def run():
    started = time.process_time(); spec = read(EXECUTION); baseline = read(spec['profile_holdout_execution'])
    coverage = read(spec['coverage_execution'])
    native = read(Path(baseline['output'])/'calibration_profiles.json')['rows']
    restricted = read(Path(spec['output'])/'calibration_profiles.json')['rows']
    raw = read(coverage['source_data']); public = read(Path(coverage['source_data']).with_name('frozen_inputs.json'))
    labels = {i: training_view(raw, [i], public['plan']['task_distribution']['L'])[0] for i in spec['heldout_tasks']}
    rows = []
    for candidate in restricted:
        original = next(r for r in native if r['layout'] == candidate['layout'] and r['task_index'] == candidate['task_index'])
        observed = np.asarray(labels[candidate['task_index']]['response'])[2:, 1]
        for stage in ('prior', 'raw_paid'):
            a, b = (original['prior'], candidate['prior']) if stage == 'prior' else (original, candidate)
            old = np.asarray(a['mean'])[2:, 1]; new = np.asarray(b['mean'])[2:, 1]
            error, shift = old-observed, new-old; offset = shift.mean(); shape = shift-offset
            constant_effect = 2*offset*error.mean()+offset**2
            shape_effect = 2*np.mean(error*shape)+np.mean(shape**2)
            total = np.mean((new-observed)**2)-np.mean(error**2)
            np.testing.assert_allclose(constant_effect+shape_effect, total, atol=1e-13)
            variance = np.asarray(a['variance'])[2:, 1]+np.asarray(a['expected_episode_variance'])[2:, 1]/3
            weights = (1/variance)/np.sum(1/variance)
            weighted = float(np.sum(weights*((new-observed)**2-error**2)))
            if stage == 'raw_paid':
                np.testing.assert_allclose(total, candidate['scores']['observed_head_mse'][1]-original['scores']['observed_head_mse'][1], atol=1e-13)
            prior_shift = np.asarray(candidate['prior']['mean'])[2:, 1]-np.asarray(original['prior']['mean'])[2:, 1]
            rows.append({'layout': candidate['layout'], 'task_index': candidate['task_index'], 'stage': stage,
                'uniform_g_mse_change': float(total), 'constant_shift_mse_contribution': float(constant_effect),
                'centered_shape_mse_contribution': float(shape_effect), 'mean_g_shift': float(offset),
                'shape_shift_rms': float(np.sqrt(np.mean(shape**2))),
                'prior_mean_shift_rms': float(np.sqrt(np.mean(prior_shift**2))),
                'incremental_paid_mean_shift_rms': float(np.sqrt(np.mean((shift-prior_shift)**2))),
                'native_observed_mean_bias': float(error.mean()), 'restricted_observed_mean_bias': float(np.mean(new-observed)),
                'fixed_native_predictive_variance_weighted_mse_change': weighted,
                'mean_native_g_variance': float(np.mean(np.asarray(a['variance'])[2:, 1])),
                'mean_restricted_g_variance': float(np.mean(np.asarray(b['variance'])[2:, 1]))})
    result = {'interpretation': 'Post-hoc algebra on saved forecasts and already-scored noisy source means; no causal or independent claim. Weighted loss fixes native predictive variance before labels and is not the proper raw batch density or a replacement gate.',
        'candidate_execution': EXECUTION, 'baseline_execution': spec['profile_holdout_execution'],
        'rows': rows, 'mean_shift_decomposition_reproduced': True, 'cpu_seconds': time.process_time()-started,
        'cost': {'new_source_fits': 0, 'new_risk_fits': 0, 'new_simulator_calls': 0, 'new_verification_calls': 0}}
    save(ROOT/spec['output']/'mean_shift_attribution.json', result)
    print([r for r in rows if r['stage'] == 'raw_paid'], flush=True)


if __name__ == '__main__':
    run()
