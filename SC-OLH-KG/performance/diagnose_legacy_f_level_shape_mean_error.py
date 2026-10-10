"""Exact level/shape attribution from already-frozen,already-scored forecasts."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_f_level_shape_vectorized_filter import prepare_branch
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.run_legacy_related_inventory_source import save

EXECUTION = 'performance/manifests/legacy_joint_population_f_level_shape_mean_error_execution_v1.json'


def decompose(mean, observed):
    error = np.asarray(mean)-observed; level = float(np.mean(error)); shape = error-level
    mse = float(np.mean(error**2)); level_mse, shape_mse = level**2, float(np.mean(shape**2))
    np.testing.assert_allclose(mse, level_mse+shape_mse, atol=1e-15, rtol=1e-12)
    return {'mse': mse, 'signed_mean_error': level, 'level_mse': level_mse,
        'centered_shape_mse': shape_mse, 'level_fraction_of_mse': level_mse/mse}


def run():
    started = time.process_time(); spec = read(EXECUTION); development = read(spec['development_execution'])
    result = read(spec['development_results']); assert result['scientific_gate_interpreted']
    old = read(development['mixture_execution']); controls = read(old['controls_profiles'])['rows']
    mixtures = read(Path(old['output'])/'calibration_profiles.json')['rows']; rows = []
    for candidate in result['rows']:
        layout, deleted = candidate['layout'], candidate['deleted_task']; branch = prepare_branch(development, layout, deleted)
        labels = read(Path(branch['coverage']['output'])/f'layout{layout}'/'inputs.json')['source_tasks'][deleted]
        observed = np.asarray(labels['response'])[:, 0]; predictions = {'joint_level_shape_variance_mixture': candidate['prediction']}
        for arm in ['native_f', 'flat', 'source_gp', 'source_variance_mixture']:
            members = mixtures if arm == 'source_variance_mixture' else controls
            matched = next(r for r in members if r['layout'] == layout and r['deleted_task'] == deleted
                and r['episode_count'] == 4 and r['arm'] == arm)
            assert matched['paid_prefix'] == candidate['paid_prefix'] and matched['scoring_sites'] == candidate['scoring_sites']
            predictions[arm] = matched['prediction']
        breakdown = {arm: {**decompose(p['mean'], observed), 'average_posterior_f_variance': float(np.trace(p['covariance'])/len(observed))}
            for arm, p in predictions.items()}
        old_mean = np.asarray(predictions['source_variance_mixture']['mean']); new_mean = np.asarray(candidate['prediction']['mean'])
        old_error, change = old_mean-observed, new_mean-old_mean
        cross = float(2*np.mean(old_error*change)); square = float(np.mean(change**2))
        level_contrast = float(2*np.mean(old_error)*np.mean(change)+np.mean(change)**2)
        centered_contrast = float(2*np.mean((old_error-old_error.mean())*(change-change.mean()))+np.mean((change-change.mean())**2))
        difference = breakdown['joint_level_shape_variance_mixture']['mse']-breakdown['source_variance_mixture']['mse']
        np.testing.assert_allclose(difference, cross+square, atol=1e-15, rtol=1e-12)
        np.testing.assert_allclose(difference, level_contrast+centered_contrast, atol=1e-15, rtol=1e-12)
        rows.append({'layout': layout, 'deleted_task': deleted, 'error_breakdown': breakdown,
            'joint_minus_intercept_mean_change': {'bank_rms_change': float(np.sqrt(square)), 'mse_difference': difference,
                'old_error_cross_term': cross, 'mean_change_squared_term': square,
                'level_mse_difference': level_contrast, 'centered_shape_mse_difference': centered_contrast},
            'mean_known_label_noise_variance': float(np.mean(np.asarray(labels['noise'])[:, 0]))})
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    save(out/'results.json', {'execution': spec, 'rows': rows, 'cpu_seconds': time.process_time()-started,
        'cost': spec['cost'], 'no_prior_or_model_adopted': True, 'source6_7_evaluated': False, 'target_promoted': False,
        'independent_confirmation': False, 'diagnostic_is_not_oracle_mean_recovery': True})
    for row in rows:
        print({'layout': row['layout'], 'breakdown': row['error_breakdown'], 'mean_change': row['joint_minus_intercept_mean_change']}, flush=True)


if __name__ == '__main__':
    run()
