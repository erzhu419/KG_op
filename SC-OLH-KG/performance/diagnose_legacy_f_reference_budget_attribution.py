"""Cached eight-episode f mean/shape and Gaussian score accounting."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.run_legacy_related_inventory_source import save
from performance.validate_legacy_paired_taskout import gaussian_score

EXECUTION = 'performance/manifests/legacy_joint_population_f_reference_budget_attribution_execution_v1.json'


def run():
    started = time.process_time(); spec = read(EXECUTION); execution = read(spec['mixture_execution'])
    reference = read(execution['reference_execution'])
    centering = read(read(reference['source_execution'])['centering_execution'])
    coverage = read(centering['coverage_execution'])
    checked = read(spec['mixture_results']); assert checked['scientific_gate_interpreted']
    mixture = [r for r in read(spec['mixture_profiles'])['rows'] if r['episode_count'] == spec['episode_count']]
    gp = [r for r in read(spec['control_profiles'])['rows'] if r['episode_count'] == spec['episode_count'] and r['arm'] == 'source_gp']
    rows = []
    for layout in spec['layouts']:
        labels = read(Path(coverage['output'])/f'layout{layout}'/'inputs.json')['source_tasks']
        for deleted in spec['training_tasks']:
            a = next(r for r in mixture if r['layout'] == layout and r['deleted_task'] == deleted)
            b = next(r for r in gp if r['layout'] == layout and r['deleted_task'] == deleted)
            assert a['scoring_sites'] == b['scoring_sites'] and a['paid_prefix'] == b['paid_prefix']
            y = np.asarray(labels[deleted]['response'])[:, 0]; noise = np.asarray(labels[deleted]['noise'])[:, 0]; n = len(y)
            means = [np.asarray(r['prediction']['mean']) for r in (a, b)]
            covariances = [np.asarray(r['prediction']['covariance'])+np.diag(noise) for r in (a, b)]
            parts = []
            for mean, covariance, cached in zip(means, covariances, (a, b)):
                error = mean-y; total = float(np.mean(error**2)); level = float(error.mean()**2)
                shape = float(np.mean((error-error.mean())**2))
                np.testing.assert_allclose(total, level+shape, atol=1e-14, rtol=1e-10)
                np.testing.assert_allclose(total, cached['scores']['f_mse'], atol=1e-14, rtol=1e-10)
                logscore = gaussian_score(mean, covariance, y)/n
                np.testing.assert_allclose(logscore, cached['scores']['f_gaussian_mean_density'], atol=1e-10, rtol=1e-10)
                sign, logdet = np.linalg.slogdet(covariance); assert sign > 0
                parts.append({'mse': total, 'squared_average_residual': level, 'centered_shape_mse': shape,
                    'gaussian_mean_density': logscore, 'logdet_penalty': float(logdet/(2*n)),
                    'mahalanobis_penalty': float(error@np.linalg.solve(covariance, error)/(2*n))})
            differences = {key: parts[0][key]-parts[1][key] for key in parts[0]}
            np.testing.assert_allclose(differences['gaussian_mean_density'],
                -differences['logdet_penalty']-differences['mahalanobis_penalty'], atol=1e-10, rtol=1e-10)
            variance = float(4*np.sum((means[0]-means[1])**2*noise)/n**2)
            gp_mean_mix_cov = gaussian_score(means[1], covariances[0], y)/n
            mix_mean_gp_cov = gaussian_score(means[0], covariances[1], y)/n
            paths = {'mean_first_mean_gain': gp_mean_mix_cov-parts[0]['gaussian_mean_density'],
                'mean_first_covariance_gain': parts[1]['gaussian_mean_density']-gp_mean_mix_cov,
                'covariance_first_covariance_gain': mix_mean_gp_cov-parts[0]['gaussian_mean_density'],
                'covariance_first_mean_gain': parts[1]['gaussian_mean_density']-mix_mean_gp_cov}
            gain = parts[1]['gaussian_mean_density']-parts[0]['gaussian_mean_density']
            np.testing.assert_allclose(paths['mean_first_mean_gain']+paths['mean_first_covariance_gain'], gain, atol=1e-12)
            np.testing.assert_allclose(paths['covariance_first_covariance_gain']+paths['covariance_first_mean_gain'], gain, atol=1e-12)
            rows.append({'layout': layout, 'deleted_task': deleted, 'mixture': parts[0], 'source_gp': parts[1],
                'mixture_minus_gp': differences, 'paired_mse_measurement_variance': variance,
                'fixed_prediction_mse_gap_measurement_interval_95': [differences['mse']-1.96*np.sqrt(variance), differences['mse']+1.96*np.sqrt(variance)],
                'gaussian_posterior_scoring_paths': paths})
    summary = {}
    for layout in spec['layouts']:
        summary[layout] = {}
        for group, tasks in (('all6', range(6)), ('task1', [1]), ('other5', [0, 2, 3, 4, 5])):
            members = [r for r in rows if r['layout'] == layout and r['deleted_task'] in tasks]
            differences = {key: float(np.mean([r['mixture_minus_gp'][key] for r in members])) for key in members[0]['mixture_minus_gp']}
            measurement_variance = sum(r['paired_mse_measurement_variance'] for r in members)/len(members)**2
            summary[layout][group] = {'mixture_minus_gp': differences,
                'signed_level_share_of_mse_gap': differences['squared_average_residual']/differences['mse'],
                'signed_shape_share_of_mse_gap': differences['centered_shape_mse']/differences['mse'],
                'fixed_prediction_mse_gap_measurement_interval_95': [differences['mse']-1.96*np.sqrt(measurement_variance), differences['mse']+1.96*np.sqrt(measurement_variance)],
                'gaussian_posterior_scoring_paths': {key: float(np.mean([r['gaussian_posterior_scoring_paths'][key] for r in members]))
                    for key in members[0]['gaussian_posterior_scoring_paths']}}
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    result = {'execution': spec, 'rows': rows, 'summary': summary, 'cached_mse_and_gaussian_scores_reproduced': True,
        'no_model_adopted': True, 'cost': spec['cost'], 'cpu_seconds': time.process_time()-started,
        'source6_7_evaluated': False, 'target_promoted': False, 'independent_confirmation': False}
    save(out/'results.json', result); print({'summary': summary, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
