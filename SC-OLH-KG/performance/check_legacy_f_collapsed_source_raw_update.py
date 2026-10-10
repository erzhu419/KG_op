"""Collapsed source-variance sampling with analytic Gaussian function moments."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.check_legacy_f_correlated_source_raw_update import diagnostics
from performance.diagnose_legacy_f_inner_conditional_calibration import integrate
from performance.diagnose_legacy_f_source_incremental_calibration import log_density
from performance.run_legacy_related_inventory_source import save
from representation.physical_risk_prior import log_sample_variance_density

PLAN = 'performance/manifests/legacy_joint_population_f_collapsed_source_raw_update_plan_v1.json'


class CollapsedRawSourcePosterior:
    """The same frozen f/eta law,with f integrated out of the batch likelihood."""

    def __init__(self, mean, covariance, labels, sample_variances):
        self.mean = np.asarray(mean); self.covariance = np.asarray(covariance)
        self.labels = np.asarray(labels); self.s2 = np.asarray(sample_variances)
        self.error = self.labels - self.mean

    def evaluate(self, eta):
        noise = np.exp(eta) / 3
        factor = cho_factor(self.covariance + np.diag(noise), lower=True)
        alpha = cho_solve(factor, self.error)
        score = -.5 * (len(eta) * np.log(2 * np.pi) + 2 * np.log(np.diag(factor[0])).sum() + self.error @ alpha)
        score += np.sum(log_sample_variance_density(eta, self.s2, 3))
        return float(score), factor, alpha

    def moments(self, evaluation):
        _, factor, alpha = evaluation
        displacement = self.covariance @ alpha
        covariance = self.covariance - self.covariance @ cho_solve(factor, self.covariance)
        return displacement, .5 * (covariance + covariance.T)


def elliptical_step(eta, prior_mean, prior_factor, evaluation, model, rng):
    residual = eta - prior_mean; direction = prior_factor @ rng.normal(size=len(eta))
    threshold = evaluation[0] + np.log(rng.uniform()); angle = rng.uniform(0, 2 * np.pi)
    lower, upper = angle - 2 * np.pi, angle; evaluations = 0
    while True:
        proposal = prior_mean + residual * np.cos(angle) + direction * np.sin(angle)
        proposed = model.evaluate(proposal); evaluations += 1
        if proposed[0] >= threshold: return proposal, proposed, evaluations
        if angle < 0: lower = angle
        else: upper = angle
        angle = rng.uniform(lower, upper)


def verify(field, index, singleton):
    em = np.asarray(field['eta_mean']); covariance = np.asarray(field['f_covariances'][index])
    mean, labels, s2 = (np.asarray(field[k]) for k in ('f_means', 'batch_means', 'sample_variances'))
    mean = mean[index]; model = CollapsedRawSourcePosterior(mean, covariance, labels, s2)
    evaluation = model.evaluate(em); displacement, post_covariance = model.moments(evaluation)
    reference = log_density(labels - mean, covariance + np.diag(np.exp(em) / 3))
    reference += np.sum(log_sample_variance_density(em, s2, 3))
    assert abs(evaluation[0] - reference) <= 1e-8
    prior_precision = cho_solve(cho_factor(covariance, lower=True), np.eye(len(mean)))
    precision_factor = cho_factor(prior_precision + np.diag(3 * np.exp(-em)), lower=True)
    covariance_precision = cho_solve(precision_factor, np.eye(len(mean)))
    mean_precision = cho_solve(precision_factor, prior_precision @ mean + 3 * np.exp(-em) * labels)
    np.testing.assert_allclose(post_covariance, covariance_precision, atol=1e-10, rtol=1e-8)
    np.testing.assert_allclose(mean + displacement, mean_precision, atol=1e-8, rtol=1e-8)
    checks = {'layout': field['layout'], 'arm': field['arms'][index],
        'dense_marginal_density_error': abs(evaluation[0] - reference),
        'precision_posterior_mean_error': float(np.max(np.abs(mean + displacement - mean_precision))),
        'precision_posterior_covariance_error': float(np.max(np.abs(post_covariance - covariance_precision))), 'passed': True}
    if singleton is not None and field['layout'] == '5':
        local = CollapsedRawSourcePosterior(mean[:1], covariance[:1, :1], labels[:1], s2[:1])
        eta_mean = em[0]; eta_sd = np.sqrt(field['eta_covariance'][0][0])
        def at(t, index=slice(None)):
            return np.atleast_1d(-.5 * (np.log(2 * np.pi) + t * t)
                + local.evaluate(np.atleast_1d(eta_mean + eta_sd * t))[0])[index]
        def statistics(t):
            delta, cov = local.moments(local.evaluate(np.atleast_1d(eta_mean + eta_sd * t)))
            return np.asarray([[1., delta[0], cov[0, 0] + delta[0] ** 2]])
        moments, logs, error = integrate(at, statistics)
        site = field['sites'][0]
        previous = next(r for r in singleton['rows'] if (r['layout'], r['arm'], r['source_task'], r['policy']) ==
            (field['layout'], field['arms'][index], site['source_task'], site['policy']))
        local_mean = mean[0] + moments[0, 1]; local_variance = moments[0, 2] - moments[0, 1] ** 2
        np.testing.assert_allclose([local_mean, local_variance, logs[0]],
            [previous['posterior_mean'], previous['posterior_variance'], previous['joint_batch_log_density']], atol=1e-8, rtol=1e-8)
        checks['singleton_posterior_mean_error'] = abs(float(local_mean) - previous['posterior_mean'])
        checks['singleton_posterior_variance_error'] = abs(float(local_variance) - previous['posterior_variance'])
        checks['singleton_integral_relative_error_bound'] = error
    return checks


def aggregate(displacements, conditional_covariances):
    flat = displacements.reshape(-1, displacements.shape[-1]); mean = flat.mean(axis=0)
    within = conditional_covariances.mean(axis=(0, 1)); between = np.cov(flat, rowvar=False)
    return mean, within + between, within, between


def sample(field, index, spec, started, model, reference_mean, reference_covariance, retained_eta=None,
           transition=elliptical_step, retained_moments=None):
    em = np.asarray(reference_mean); factor = np.linalg.cholesky(reference_covariance)
    sampler = spec['sampler']; count = sampler['retained_sweeps_per_chain']
    eta_draws, displacements, conditional_covariances, evaluations = [], [], [], []
    for chain in range(sampler['chains']):
        rng = np.random.default_rng(sampler['seed'] + 100 * int(field['layout']) + 10 * index + chain)
        eta = em + factor @ rng.normal(size=len(em)); evaluation = model.evaluate(eta)
        es, ds, cs = [], [], []; evaluations_chain = 1
        for iteration in range(sampler['warmup_sweeps_per_chain'] + count):
            if time.process_time() - started >= spec['maximum_single_thread_cpu_seconds']: raise TimeoutError('registered CPU budget reached')
            eta, evaluation, n = transition(eta, em, factor, evaluation, model, rng); evaluations_chain += n
            if iteration >= sampler['warmup_sweeps_per_chain']:
                displacement, covariance = model.moments(evaluation)
                es.append(eta); ds.append(displacement); cs.append(covariance)
        eta_draws.append(es); displacements.append(ds); conditional_covariances.append(cs); evaluations.append(evaluations_chain)
    eta_draws, displacements, conditional_covariances = map(np.asarray, (eta_draws, displacements, conditional_covariances))
    if retained_eta is not None:
        retained_eta.append(eta_draws)
    if retained_moments is not None:
        retained_moments.append((displacements, conditional_covariances))
    second_moments = np.diagonal(conditional_covariances, axis1=2, axis2=3) + displacements ** 2
    diagnostic = diagnostics(np.concatenate([eta_draws, displacements, second_moments], axis=2))
    delta, covariance, within, between = aggregate(displacements, conditional_covariances)
    first_delta, first, _, _ = aggregate(displacements[:, :count // 2], conditional_covariances[:, :count // 2])
    second_delta, second, _, _ = aggregate(displacements[:, count // 2:], conditional_covariances[:, count // 2:])
    first_diagonal, second_diagonal = np.diag(first), np.diag(second)
    half_difference = np.abs(first_diagonal - second_diagonal) / ((first_diagonal + second_diagonal) / 2)
    max_rhat = max(diagnostic['split_rhat']); min_ess = min(diagnostic['autocorrelation_ess'])
    gate = spec['monte_carlo_gate']; passed = (max_rhat <= gate['maximum_split_rhat_eta_and_rao_blackwell_f_moments']
        and min_ess >= gate['minimum_autocorrelation_ess_eta_and_rao_blackwell_f_moments']
        and half_difference.max() <= gate['maximum_first_second_half_f_variance_relative_difference'])
    plugin_displacement, plugin_covariance = model.moments(model.evaluate(np.log(field['sample_variances'])))
    ratios = np.diag(covariance) / np.diag(plugin_covariance)
    return {'layout': field['layout'], 'arm': field['arms'][index], 'monte_carlo_gate_passed': bool(passed),
        'maximum_split_rhat': max_rhat, 'minimum_autocorrelation_ess': min_ess,
        'maximum_first_second_half_f_variance_relative_difference': float(half_difference.max()),
        'half_variance_relative_differences': half_difference.tolist(), 'first_half_posterior_variance': first_diagonal.tolist(),
        'second_half_posterior_variance': second_diagonal.tolist(),
        'first_half_posterior_mean': (model.mean + first_delta).tolist(), 'first_half_posterior_covariance': first.tolist(),
        'second_half_posterior_mean': (model.mean + second_delta).tolist(), 'second_half_posterior_covariance': second.tolist(),
        'eta_posterior_mean': eta_draws.mean(axis=(0, 1)).tolist(),
        'eta_posterior_covariance': np.cov(eta_draws.reshape(-1, len(em)), rowvar=False).tolist(),
        'diagnostic_coordinate_order': 'eta48,conditional_f_mean_displacement48,conditional_f_second_moment_about_prior_mean48',
        'diagnostics': diagnostic, 'posterior_mean': (model.mean + delta).tolist(), 'posterior_covariance': covariance.tolist(),
        'within_eta_covariance': within.tolist(), 'between_eta_mean_covariance': between.tolist(),
        'plug_in_posterior_mean': (model.mean + plugin_displacement).tolist(), 'plug_in_posterior_covariance': plugin_covariance.tolist(),
        'posterior_variance_ratio_mean': float(ratios.mean()), 'posterior_variance_ratio_quantiles_0_50_100': np.quantile(ratios, [0, .5, 1]).tolist(),
        'posterior_mean_shift_rms_in_plugin_posterior_sd': float(np.sqrt(np.mean((delta - plugin_displacement) ** 2 / np.diag(plugin_covariance)))),
        'elliptical_likelihood_evaluations_by_chain': evaluations}


def run():
    started = time.process_time(); spec = read(PLAN); source = read(spec['joint_source_inputs'])
    previous = read(spec['joint_source_results']); singleton = read(spec['singleton_results'])
    assert previous['all_four_conditionals_complete'] and not previous['all_monte_carlo_gates_passed']
    assert source['marginal_reproduction_and_positive_covariance_checks_passed'] and singleton['diagnostic_interpreted']
    fields = [{**f, 'batch_means': f['added_means'], 'sample_variances': f['added_sample_variances']} for f in source['rows']]
    out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    checks = [verify(field, index, singleton) for field in fields for index in range(2)]
    save(out / 'verification.json', {'execution': spec, 'checks': checks, 'all_checks_passed_before_sampling': True})
    rows = []
    try:
        for field in fields:
            for index in range(2):
                model = CollapsedRawSourcePosterior(field['f_means'][index], field['f_covariances'][index],
                    field['batch_means'], field['sample_variances'])
                row = sample(field, index, spec, started, model, field['eta_mean'], field['eta_covariance']); rows.append(row)
                print('collapsed source', row['layout'], row['arm'], 'gate', row['monte_carlo_gate_passed'],
                    'Rhat', row['maximum_split_rhat'], 'ESS', row['minimum_autocorrelation_ess'],
                    'half variance', row['maximum_first_second_half_f_variance_relative_difference'],
                    'variance ratio', row['posterior_variance_ratio_mean'], flush=True)
                save(out / 'results.json', {'execution': spec, 'rows': rows, 'diagnostic_interpreted': False,
                    'all_four_conditionals_complete': len(rows) == 4, 'cpu_seconds': time.process_time() - started})
    except TimeoutError:
        save(out / 'results.json', {'execution': spec, 'rows': rows, 'diagnostic_interpreted': False,
            'all_four_conditionals_complete': False, 'stopped_at_cpu_budget': True, 'cpu_seconds': time.process_time() - started})
        return
    passed = all(r['monte_carlo_gate_passed'] for r in rows)
    save(out / 'results.json', {'execution': spec, 'rows': rows, 'all_four_conditionals_complete': True,
        'all_monte_carlo_gates_passed': passed, 'diagnostic_interpreted': passed, 'all_pre_sampling_checks_passed': True,
        'cpu_seconds': time.process_time() - started, 'cost': {**spec['cost'], 'actual_chains': 16,
            'actual_sweeps': 16 * (spec['sampler']['warmup_sweeps_per_chain'] + spec['sampler']['retained_sweeps_per_chain'])},
        'no_model_adopted': True, 'independent_confirmation': False,
        'interpretation': 'Collapsed eta sampling of the same Stage197 conditional raw posterior,with Rao-Blackwell function moments and between-eta mean covariance retained. No original source replay,refit or held-task prediction. Old GP precision failure remains in its original package.'})


if __name__ == '__main__':
    run()
