"""Cached mean-error decomposition and fresh-posterior source stationarity."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_f_correlated_source_raw_update import diagnostics
from performance.check_legacy_f_raw_level_risk_mstep import risk_system, source_moments
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_raw_source_level_risk_score import source_systems, moment_gradient
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.run_legacy_related_inventory_source import save

PLAN = 'performance/manifests/legacy_joint_population_f_level_risk_learning_attribution_plan_v1.json'


def run():
    started = time.process_time(); spec = read(PLAN); predictions = read(spec['predictions'])
    scored = read(spec['prediction_results']); assert scored['all_four_forecasts_scored']
    mstep = read(spec['Mstep_plan']); fitted = read(spec['Mstep_results'])['rows']; level = read(mstep['level_risk_plan'])
    original = read(mstep['source_inputs'])['rows']; source = read(spec['source_results'])['rows']
    fresh = read(Path(read(spec['source_results'])['execution']['output']) / 'frozen_inputs.json')['rows']
    partitions = read(spec['source_partitions'])['rows']; previous = read(spec['previous_attribution'])['rows']
    mean_spec = read(level['source_mean_plan']); coverage, _ = source_paths(read(mean_spec['source_context_execution']))
    rows, comparisons = [], []
    for field in original:
        systems, _, risk = source_systems(level, field); rb, _, _, _ = risk_system(field, risk, mstep['risk_original_shape_tail_budget'])
        inputs = read(Path(coverage['output']) / f"layout{field['layout']}" / 'task1' / 'inputs.json')
        labels = np.asarray(next(t['response'] for t in inputs['source_tasks'] if t['name'] == 'related_inventory_00'))[:, 0]
        for index, (builder, old_f, _) in enumerate(systems):
            key = (field['layout'], field['arms'][index])
            forecast = next(r for r in predictions['rows'] if (r['layout'], r['arm']) == key)
            result = next(r for r in scored['rows'] if (r['layout'], r['arm']) == key)
            fit = next(r for r in fitted if (r['layout'], r['arm']) == key); theta = np.asarray(fit['learned_parameters'])
            post = next(r for r in source if (r['layout'], r['arm']) == key)
            partition = next(r for r in partitions if (r['layout'], r['arm']) == key)
            fm, rm = source_moments(post, partition)
            mean, covariance, derivatives = builder(np.asarray([theta[0], old_f[1], theta[1]]))
            current = next(r for r in fresh if (r['layout'], r['active_arm']) == key)
            np.testing.assert_allclose(mean, current['f_means'][index], atol=1e-10, rtol=1e-10)
            np.testing.assert_allclose(covariance, current['f_covariances'][index], atol=1e-10, rtol=1e-10)
            risk_mean, risk_covariance, _ = rb(theta[2:])
            np.testing.assert_allclose(risk_mean, current['eta_mean'], atol=1e-10, rtol=1e-10)
            np.testing.assert_allclose(risk_covariance, current['eta_covariance'], atol=1e-10, rtol=1e-10)
            # The scores use each current scientific covariance,not the old posterior's law.
            fg = [moment_gradient(mean, covariance, [derivatives[1]], *p) for p in fm]
            rg = [moment_gradient(*rb(theta[2:]), *p) for p in rm]
            gradients = np.asarray([np.r_[a, b] for a, b in zip(fg, rg)])
            projected = gradients.copy()
            for j in (1, 3):
                if theta[j] == 0:
                    projected[:, j] = np.minimum(projected[:, j], 0.)
            center = np.asarray(forecast['prediction']['mean']); error = center - labels
            level_error = float(error.mean()); shape = error - level_error
            component = np.asarray(forecast['component_means'])[:, :, 0]
            influence = ((component - center) @ (2 * error / len(error))).reshape(4, -1, 1)
            diagnostic = diagnostics(influence); ess = diagnostic['autocorrelation_ess'][0]; rhat = diagnostic['split_rhat'][0]
            se = float(np.sqrt(np.var(influence, ddof=1) / ess))
            old = next(r for r in previous if (r['layout'], r['arm']) == key)
            mean_gate = (ess >= 100 and rhat <= 1.05 and all(e['mean_rms_difference_in_full_sd'] <= .1
                and e['maximum_variance_relative_difference'] <= .15 for e in result['split_prediction_errors']))
            rows.append({'layout': key[0], 'arm': key[1], 'mean_error': level_error, 'level_MSE': level_error ** 2,
                'centered_shape_MSE': float(np.mean(shape ** 2)), 'f_MSE': float(np.mean(error ** 2)),
                'level_MSE_change_vs_previous': level_error ** 2 - old['level_MSE'],
                'shape_MSE_change_vs_previous': float(np.mean(shape ** 2)) - old['centered_shape_MSE'],
                'MSE_influence_Rhat': rhat, 'MSE_influence_ESS': ess, 'conditional_MSE_MCSE': se,
                'mean_diagnostic_precision_gate_passed': bool(mean_gate), 'joint_density_precision_gate_passed': result['mixture_precision_gate_passed'],
                'fresh_source_raw_scores_full_first_second': gradients.tolist(),
                'fresh_source_projected_scores_full_first_second': projected.tolist(),
                'fresh_source_stable_nonzero_projected_directions': ((projected[0] * projected[1] > 0) & (projected[0] * projected[2] > 0)).tolist()})
            print('cached learning attribution', *key, 'MSE', np.mean(error ** 2), 'level', level_error ** 2,
                'shape', np.mean(shape ** 2), 'fresh source score', gradients[0].tolist(), flush=True)
    for layout in ('5', '6'):
        a, b = [next(r for r in rows if (r['layout'], r['arm']) == (layout, arm))
            for arm in ('joint_spectral_geometry', 'source_gp')]
        delta = a['f_MSE'] - b['f_MSE']; se = float(np.hypot(a['conditional_MSE_MCSE'], b['conditional_MSE_MCSE']))
        comparisons.append({'layout': layout, 'both_mean_diagnostic_gates_passed': a['mean_diagnostic_precision_gate_passed'] and b['mean_diagnostic_precision_gate_passed'],
            'structure_minus_GP_MSE': delta, 'level_MSE_difference': a['level_MSE'] - b['level_MSE'],
            'centered_shape_MSE_difference': a['centered_shape_MSE'] - b['centered_shape_MSE'],
            'conditional_MSE_difference_MCSE': se, 'nominal_conditional_MC_interval': [delta - 1.96 * se, delta + 1.96 * se],
            'joint_predictive_gate_remains_unpassed': True})
        print('conditional mean diagnostic', comparisons[-1], flush=True)
    assert time.process_time() - started < spec['maximum_single_thread_cpu_seconds']
    out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    save(out / 'results.json', {'execution': spec, 'rows': rows, 'comparisons': comparisons,
        'cpu_seconds': time.process_time() - started, 'cost': spec['cost'], 'no_model_adopted': True,
        'independent_confirmation': False,
        'interpretation': 'Source stationarity and cached mean-error diagnosis. MSE intervals describe computational uncertainty at fixed source empirical laws and12 development labels;not task uncertainty or joint-density precision. Matching source score signs indicate remaining learning directions;no additional fit performed.'})


if __name__ == '__main__':
    run()
