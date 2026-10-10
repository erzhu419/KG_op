"""Conditional-mean calibration and local covariance score tangents."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import quad_vec
from scipy.optimize import minimize_scalar
from scipy.special import digamma, logsumexp, ndtr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_inner_raw_density_attribution import component_variance_density
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.run_legacy_related_inventory_source import save
from representation.physical_risk_prior import log_sample_variance_density

EXECUTION = 'performance/manifests/legacy_joint_population_f_inner_conditional_calibration_execution_v1.json'


def integrate(at, statistics, coordinate_bound=12.):
    count = len(at(0.)); modes = np.asarray([minimize_scalar(lambda t: -at(t, i), bounds=(-coordinate_bound, coordinate_bound), method='bounded').x for i in range(count)])
    shifts = at(modes)
    values, error, info = quad_vec(lambda t: np.exp(at(t)-shifts)[:, None]*statistics(t), -coordinate_bound, coordinate_bound,
        points=np.unique([modes.min(), np.median(modes), modes.max()]), norm='max', full_output=True,
        epsabs=1e-10, epsrel=1e-10, workers=1)
    if not info.success: raise RuntimeError(info.message)
    assert np.all(values[:, 0] > 0)
    relative = float(error/values[:, 0].min()); assert relative <= 1e-7
    return values/values[:, :1], shifts+np.log(values[:, 0]), relative


def local_calibration(means, covariance, log_weights, label, eta_label):
    gm, em = means.T; gv, ev = covariance[:, 0, 0], covariance[:, 1, 1]
    sd = np.sqrt(ev); cross = covariance[:, 0, 1]
    c = np.maximum(gv-cross**2/ev, 0.); loading = cross/sd
    s2 = float(np.exp(eta_label+digamma(1.)))
    def risk_at(t, index=slice(None)):
        return -.5*(np.log(2*np.pi)+t*t)+log_sample_variance_density(em[index]+sd[index]*t, s2, 3)
    def moment_statistics(t):
        displacement = loading*t; noise = np.exp(em+sd*t)/3; total = c+noise
        return np.column_stack([np.ones(len(gm)), displacement, total+displacement**2, noise,
            ndtr((label-gm-displacement)/np.sqrt(total))])
    moments, risk_logs, risk_error = integrate(risk_at, moment_statistics)
    risk_log_density = float(logsumexp(log_weights+risk_logs)); posterior_weights = np.exp(log_weights+risk_logs-risk_log_density)
    component_mean = gm+moments[:, 1]; component_variance = moments[:, 2]-moments[:, 1]**2
    mean = float(posterior_weights@component_mean)
    variance = float(posterior_weights@(component_variance+(component_mean-mean)**2))
    noise_variance = float(posterior_weights@moments[:, 3]); cdf = float(posterior_weights@moments[:, 4])
    def joint_at(t, index=slice(None), log_scale=0.):
        eta = em[index]+sd[index]*t; total = np.exp(log_scale)*c[index]+np.exp(eta)/3
        error = label-gm[index]-loading[index]*t
        return risk_at(t, index)-.5*(np.log(2*np.pi*total)+error**2/total)
    def score_statistics(t):
        total = c+np.exp(em+sd*t)/3; error = label-gm-loading*t
        return np.column_stack([np.ones(len(gm)), -.5*c/total, .5*c*error**2/total**2])
    score_moments, joint_logs, joint_error = integrate(joint_at, score_statistics)
    joint_density = float(logsumexp(log_weights+joint_logs)); joint_weights = np.exp(log_weights+joint_logs-joint_density)
    volume, residual = joint_weights@score_moments[:, 1:3]
    record = {'conditional_mean': mean, 'conditional_batch_mean_variance': variance,
        'expected_episode_noise_over_three': noise_variance, 'latent_f_conditional_variance': variance-noise_variance,
        'conditional_squared_error': (mean-label)**2, 'squared_standardized_error': (mean-label)**2/variance,
        'conditional_cdf': cdf, 'covered_50': bool(.25 <= cdf <= .75), 'covered_90': bool(.05 <= cdf <= .95),
        'covered_95': bool(.025 <= cdf <= .975), 'joint_batch_density': joint_density,
        'variance_label_density': risk_log_density, 'conditional_mean_density': joint_density-risk_log_density,
        'log_conditional_f_variance_score_gradient': float(volume+residual),
        'gradient_volume_term': float(volume), 'gradient_residual_term': float(residual),
        'maximum_integration_relative_error_bound': max(risk_error, joint_error)}
    return record, joint_at


def run():
    started = time.process_time(); spec = read(EXECUTION); attributed = read(spec['attribution_results'])
    assert attributed['diagnostic_interpreted'] and attributed['decomposition_numerically_verified']
    plan = read(attributed['execution']['plan']); source = read(plan['source_results'])
    coverage, _ = source_paths(read(plan['source_context_execution'])); folder = ROOT/source['execution']['output']
    rows, refinement_checks, derivative_checks = [], [], []
    for row in source['rows']:
        if row['design'] != 'fixed_public_prefix' or row['arm'] not in spec['arms']: continue
        inputs = read(Path(coverage['output'])/f"layout{row['layout']}"/'task1'/'inputs.json')
        labels = np.asarray(next(t['response'] for t in inputs['source_tasks'] if t['name'] == f"related_inventory_{row['query_task']:02d}"))[4:]
        predictions = []
        for resolution in ('selected', 'refined'):
            if f'{resolution}_component_file' not in row: continue
            sites = []
            with np.load(folder/row[f'{resolution}_component_file']) as state:
                for j, label in enumerate(labels):
                    values, joint = local_calibration(state['component_means'][:, j], state['component_local_covariances'][:, j], state['log_weights'], label[0], label[3])
                    sites.append(values)
                    if resolution == 'selected' and row['query_task'] == 0 and j == 0:
                        h = spec['derivative_step']; densities = []
                        for point in (-h, h):
                            _, logs, _ = integrate(lambda t, index=slice(None): joint(t, index, point), lambda t: np.ones((len(state['log_weights']), 1)))
                            densities.append(float(logsumexp(state['log_weights']+logs)))
                        finite = (densities[1]-densities[0])/(2*h)
                        expected = values['log_conditional_f_variance_score_gradient']
                        np.testing.assert_allclose(finite, expected, atol=1e-7, rtol=1e-4)
                        derivative_checks.append({'layout': row['layout'], 'arm': row['arm'], 'finite_difference_error': float(abs(finite-expected))})
            predictions.append(sites)
        sites = predictions[0]; average = {k: float(np.mean([s[k] for s in sites])) for k in sites[0]}
        assert abs(average['joint_batch_density']-row['scores']['f_raw_batch_density']) <= 1e-8
        previous = next(r for r in attributed['rows'] if r['layout'] == row['layout'] and r['query_task'] == row['query_task'] and r['arm'] == row['arm'])
        assert abs(average['variance_label_density']-previous['means']['bias_corrected_log_sample_variance_density']) <= 1e-8
        if len(predictions) == 2:
            refined = predictions[1]
            errors = {'conditional_mean_max_absolute': max(abs(a['conditional_mean']-b['conditional_mean']) for a, b in zip(sites, refined)),
                'variance_relative': max(abs(a['conditional_batch_mean_variance']/b['conditional_batch_mean_variance']-1) for a, b in zip(sites, refined)),
                'cdf_max_absolute': max(abs(a['conditional_cdf']-b['conditional_cdf']) for a, b in zip(sites, refined)),
                'gradient_max_absolute': max(abs(a['log_conditional_f_variance_score_gradient']-b['log_conditional_f_variance_score_gradient']) for a, b in zip(sites, refined))}
            refinement_checks.append({'layout': row['layout'], 'query_task': row['query_task'], 'errors': errors,
                'passed': all(errors[k] <= limit for k, limit in spec['refinement_tolerances'].items())})
        rows.append({'layout': row['layout'], 'query_task': row['query_task'], 'arm': row['arm'], 'sites': sites, 'summary': average})
    summary = {layout: {arm: {k: float(np.mean([r['summary'][k] for r in rows if r['layout'] == layout and r['arm'] == arm]))
        for k in rows[0]['summary']} for arm in spec['arms']} for layout in ('5', '6')}
    verified = all(r['passed'] for r in refinement_checks)
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    result = {'execution': spec, 'rows': rows, 'summary': summary, 'refinement_checks': refinement_checks, 'derivative_checks': derivative_checks,
        'diagnostic_interpreted': verified, 'no_model_adopted': True, 'source6_7_evaluated': False,
        'target_promoted': False, 'independent_confirmation': False, 'no_new_unscored_outcomes_read': True,
        'cost': spec['cost'], 'cpu_seconds': time.process_time()-started,
        'interpretation': 'Conditional on the already-scored future variance label;not deployable forecasts. Local score tangent scales only within-component f|eta variance,fixing means,eta law,cross loading and mixture weights. PSD local diagnostic,not a new source law or fitted posterior.'}
    save(out/'results.json', result)
    print('conditional calibration', {l: {a: {k: s[k] for k in ('conditional_squared_error','conditional_batch_mean_variance','squared_standardized_error','covered_90','log_conditional_f_variance_score_gradient')}
        for a, s in g.items()} for l, g in summary.items()}, 'verified', verified, 'CPU', result['cpu_seconds'], flush=True)


if __name__ == '__main__':
    run()
