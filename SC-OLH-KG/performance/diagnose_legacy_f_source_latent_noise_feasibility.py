"""Held-source batch inference with estimated versus integrated episode noise."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import digamma, ndtr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_inner_conditional_calibration import integrate, local_calibration
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.run_legacy_related_inventory_source import save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.physical_risk_prior import log_sample_variance_density

PLAN = 'performance/manifests/legacy_joint_population_f_source_latent_noise_feasibility_plan_v1.json'
ARMS = ('joint_spectral_geometry', 'source_gp')


def conditional_pair(f_means, f_variances, eta_mean, eta_variance, label, eta_label):
    """One shared variance-label integral and a batched integral for both f arms."""
    f_means, f_variances = np.asarray(f_means), np.asarray(f_variances)
    sd = np.sqrt(eta_variance); s2 = float(np.exp(eta_label + digamma(1.)))

    def risk_at(t, index=slice(None)):
        value = -.5 * (np.log(2 * np.pi) + t * t) + log_sample_variance_density(eta_mean + sd * t, s2, 3)
        return np.atleast_1d(value)[index]

    def risk_statistics(t):
        eta = eta_mean + sd * t; noise = np.exp(eta) / 3
        cdfs = ndtr((label - f_means) / np.sqrt(f_variances + noise))
        return np.asarray([[1., eta, eta * eta, noise, *cdfs]])

    moments, risk_logs, risk_error = integrate(risk_at, risk_statistics)
    risk_log = float(risk_logs[0]); expected_noise = float(moments[0, 3])

    def joint_at(t, index=slice(None)):
        variance = f_variances[index] + np.exp(eta_mean + sd * t) / 3
        residual = label - f_means[index]
        return risk_at(t)[0] - .5 * (np.log(2 * np.pi * variance) + residual ** 2 / variance)

    def score_statistics(t):
        variance = f_variances + np.exp(eta_mean + sd * t) / 3; residual = label - f_means
        return np.column_stack([np.ones(2), -.5 * f_variances / variance, .5 * f_variances * residual ** 2 / variance ** 2])

    scores, joint_logs, joint_error = integrate(joint_at, score_statistics)
    rows = []
    for j, arm in enumerate(ARMS):
        rows.append({'arm': arm, 'mean': float(f_means[j]), 'latent_f_variance': float(f_variances[j]),
            'conditional_batch_mean_variance': float(f_variances[j] + expected_noise),
            'expected_episode_noise_over_three': expected_noise, 'conditional_cdf': float(moments[0, 4 + j]),
            'covered_90': bool(.05 <= moments[0, 4 + j] <= .95), 'variance_label_density': risk_log,
            'joint_batch_density': float(joint_logs[j]), 'conditional_mean_density': float(joint_logs[j] - risk_log),
            'log_conditional_f_variance_score_gradient': float(scores[j, 1] + scores[j, 2]),
            'maximum_integration_relative_error_bound': max(risk_error, joint_error)})
    return rows


def run():
    started = time.process_time(); spec = read(PLAN)
    assert read(spec['source_noise_results'])['diagnostic_interpreted']
    coverage, _ = source_paths(read(spec['source_context_execution']))
    point = read(spec['source_point_laws']); assert point['all_ten_source_laws_frozen_without_query_values']
    registry = read(spec['source_gp_registry']); assert registry['all_inner_fits_converged']
    controls = read(spec['risk_control_results']); assert controls['diagnostic_interpreted']
    branches = sum([read(Path(p) / 'source_branches.json')['rows'] for p in controls['execution']['source_packages']], [])
    query = spec['source_validation_task']; fields, frozen = {}, []
    for layout in spec['layouts']:
        law = next(r for r in point['rows'] if (r['layout'], r['query_task']) == (layout, query))
        branch = next(r for r in branches if (r['layout'], r['query_task']) == (layout, query))
        record = next(r for r in registry['rows'] if r['layout'] == layout and r['inner_query_task'] == f'related_inventory_{query:02d}')
        names = law['source_task_names']; assert names == branch['source_task_names'] == record['inner_training_task_names']
        assert len(names) == 4 and f'related_inventory_{query:02d}' not in names and 'related_inventory_01' not in names
        basis = OrthogonalLowOrderBasis(branch['public_mean_reference'])
        design = np.column_stack([np.ones(12), basis.transform(branch['query_mean_psi'])])
        f_mean = design @ np.asarray(law['source_posterior_coefficient_mean'])
        f_covariance = design @ (np.asarray(law['source_posterior_common_covariance']) + np.asarray(law['task_coefficient_covariance'])) @ design.T + law['white_discrepancy'] * np.eye(12)
        gp = record['prediction']['source_gp']; sites = coverage['layouts'][layout][query]
        assert sites[4:] == branch['scoring_sites'] and len(sites) == 12
        eta_mean = np.asarray(branch['eta_mean'])[sites]; eta_variance = np.diag(branch['eta_covariance'])[sites]
        f_means = np.stack([f_mean, gp['mean']], axis=1)
        f_variances = np.stack([np.diag(f_covariance), np.diag(gp['covariance'])], axis=1)
        assert np.all(f_variances > 0) and np.all(eta_variance > 0)
        fields[layout] = (f_means, f_variances, eta_mean, eta_variance)
        frozen.append({'layout': layout, 'source_validation_task': query, 'source_task_names': names, 'sites': sites, 'arms': ARMS,
            'f_means': f_means.tolist(), 'f_variances': f_variances.tolist(), 'eta_mean': eta_mean.tolist(), 'eta_variance': eta_variance.tolist(),
            'source_risk_package': next(p for p in controls['execution']['source_packages'] if any(b['layout'] == layout and b['query_task'] == query
                for b in read(Path(p) / 'source_branches.json')['rows'])), 'f_eta_prior_cross_covariance': 0.})
    out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    save(out / 'frozen_inputs.json', {'execution': spec, 'rows': frozen, 'all_zero_shot_marginals_frozen_before_validation_labels': True})
    rows, checks = [], []
    base = {'execution': spec, 'no_model_adopted': True, 'source6_7_evaluated': False, 'target_promoted': False,
        'outer_task1_outcomes_indexed': False, 'independent_confirmation': False, 'new_raw_paid_forecasts': 0}
    for layout in spec['layouts']:
        f_means, f_variances, eta_mean, eta_variance = fields[layout]
        inputs = read(Path(coverage['output']) / f'layout{layout}' / 'task1' / 'inputs.json')
        task = next(t for t in inputs['source_tasks'] if t['name'] == f'related_inventory_{query:02d}')
        labels, noise = np.asarray(task['response']), np.asarray(task['noise'])[:, 0]
        reconstructed = np.exp(labels[:, 3] + digamma(1.)) / 3
        np.testing.assert_allclose(reconstructed, noise, atol=1e-14, rtol=1e-10)
        for j, (label, plug_noise) in enumerate(zip(labels, noise)):
            if time.process_time() - started >= spec['maximum_single_thread_cpu_seconds']:
                save(out / 'results.json', {**base, 'rows': rows, 'diagnostic_interpreted': False, 'stopped_at_cpu_budget': True,
                    'cpu_seconds': time.process_time() - started}); return
            result = conditional_pair(f_means[j], f_variances[j], eta_mean[j], eta_variance[j], label[0], label[3])
            for arm_index, values in enumerate(result):
                variance = f_variances[j, arm_index] + plug_noise; error = label[0] - f_means[j, arm_index]
                plug_density = float(-.5 * (np.log(2 * np.pi * variance) + error ** 2 / variance))
                plug_gradient = float(.5 * f_variances[j, arm_index] * (error ** 2 / variance ** 2 - 1 / variance))
                plug_cdf = float(ndtr(error / np.sqrt(variance)))
                values.update(layout=layout, source_validation_task=query, policy=coverage['layouts'][layout][query][j],
                    source_plug_in_noise=float(plug_noise), plug_in_batch_mean_variance=float(variance),
                    plug_in_mean_density=plug_density, plug_in_log_f_variance_score_gradient=plug_gradient,
                    plug_in_cdf=plug_cdf, plug_in_covered_90=bool(.05 <= plug_cdf <= .95),
                    conditional_minus_plug_in_mean_density=values['conditional_mean_density'] - plug_density,
                    conditional_to_plug_in_variance_ratio=values['conditional_batch_mean_variance'] / variance,
                    gradient_difference=values['log_conditional_f_variance_score_gradient'] - plug_gradient)
                rows.append(values)
                if layout == spec['layouts'][0] and j == 0:
                    means = np.asarray([[f_means[j, arm_index], eta_mean[j]]]); covariance = np.asarray([np.diag([f_variances[j, arm_index], eta_variance[j]])])
                    reference, _ = local_calibration(means, covariance, np.zeros(1), label[0], label[3])
                    differences = {k: abs(values[k] - reference[k]) for k in ('conditional_batch_mean_variance', 'conditional_cdf',
                        'joint_batch_density', 'variance_label_density', 'conditional_mean_density', 'log_conditional_f_variance_score_gradient')}
                    assert max(differences.values()) <= 1e-8
                    checks.append({'arm': values['arm'], 'errors': differences, 'passed': True})
    summary = {layout: {arm: {k: float(np.mean([r[k] for r in rows if r['layout'] == layout and r['arm'] == arm])) for k in
        ('conditional_mean_density', 'plug_in_mean_density', 'conditional_minus_plug_in_mean_density', 'conditional_to_plug_in_variance_ratio',
            'log_conditional_f_variance_score_gradient', 'plug_in_log_f_variance_score_gradient', 'gradient_difference', 'covered_90', 'plug_in_covered_90')}
        for arm in ARMS} for layout in spec['layouts']}
    rankings = {}
    for layout in spec['layouts']:
        a, b = (summary[layout][arm] for arm in ARMS)
        rankings[layout] = {'structure_minus_gp_plug_in_mean_density': a['plug_in_mean_density'] - b['plug_in_mean_density'],
            'structure_minus_gp_integrated_conditional_mean_density': a['conditional_mean_density'] - b['conditional_mean_density']}
    save(out / 'results.json', {**base, 'rows': rows, 'summary': summary, 'arm_rankings': rankings, 'cross_implementation_checks': checks,
        'sample_variance_bias_and_noise_measure_verified': True, 'diagnostic_interpreted': True, 'cpu_seconds': time.process_time() - started,
        'cost': {**spec['cost'], 'actual_joint_batch_components_integrated': 48, 'actual_shared_variance_integrations': 24},
        'interpretation': 'Held-source single-site predictive densities conditional on its observed variance label,with a shared source4 Gaussian eta prior. No paid updates or source refits. This is not the joint48-label source likelihood,an exact eta source posterior,or deployable use of a future variance label.'})
    print('source latent noise feasibility', summary, 'rankings', rankings, 'CPU', time.process_time() - started, flush=True)


if __name__ == '__main__':
    run()
