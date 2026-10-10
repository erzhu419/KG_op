"""Source-only function level and risk covariance scores at the learned law."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve
from scipy.special import digamma, polygamma

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_f_source_spectral_uncertainty import source_model
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_raw_source_covariance_score import projected, score
from performance.diagnose_legacy_f_source_incremental_calibration import log_density
from performance.run_legacy_related_inventory_source import save

PLAN = 'performance/manifests/legacy_joint_population_f_raw_source_level_risk_score_plan_v1.json'


def source_systems(spec, field):
    laws = read(spec['source_laws'])['rows']; layout = field['layout']
    law = next(r for r in laws if (r['layout'], r['arm']) == (layout, 'joint_spectral_geometry'))
    model, _, _ = source_model(read(spec['source_mean_plan']), layout, 0)
    a, r = [np.asarray(law[k]) for k in ('common_coefficient_covariance', 'task_coefficient_covariance')]
    e = np.zeros_like(a); e[0, 0] = 1; center = field['f_means'][0][0]

    def structural(theta):
        ba = np.eye(len(a)); ba[0, 0] = theta[1]; br = np.eye(len(r)); br[0, 0] = theta[2]
        ac, rt = ba @ a @ ba, br @ r @ br
        da, dr = e @ a @ ba + ba @ a @ e, e @ r @ br + br @ r @ e
        d = model.design
        covariance = d @ ac @ d.T + block_diag(*[dt @ rt @ dt.T + model.white * np.eye(12) for dt in model.designs])
        directions = [d @ da @ d.T, block_diag(*[dt @ dr @ dt.T for dt in model.designs])]
        return np.full(48, theta[0]), covariance, directions

    mean_spec = read(spec['source_mean_plan']); registry = read(mean_spec['source_registry'])['rows']
    gp = read(next(row['source_fit'] for row in registry if (row['layout'], row['inner_query_task']) == (layout, 'related_inventory_00')))['fits']['gp']
    learned_gp = next(row for row in laws if (row['layout'], row['arm']) == (layout, 'source_gp'))
    scale2 = gp['anchor']['normalization']['scale'] ** 2
    common = scale2 * gp['anchor']['constant_variance']; task = scale2 * learned_gp['learned_parameters'][2]
    constants = [np.ones((48, 48)), block_diag(*[np.ones((12, 12)) for _ in range(4)])]
    fixed = np.asarray(field['f_covariances'][1]) - common * constants[0] - task * constants[1]

    def generic(theta):
        return np.full(48, theta[0]), fixed + theta[1] * constants[0] + theta[2] * constants[1], constants

    risk = next(row for row in read(spec['risk_reconstruction'])['rows'] if row['layout'] == layout)
    d = np.asarray(risk['source_design']); blocks = np.split(d, 4)
    common_risk = d @ np.asarray(risk['common_prior_covariance']) @ d.T
    task_risk = block_diag(*[dt @ np.asarray(risk['task_coefficient_covariance']) @ dt.T for dt in blocks])
    risk_directions = [common_risk, task_risk, constants[1], np.eye(48)]
    baseline = common_risk + task_risk + risk['white_discrepancy'] * np.eye(48)
    np.testing.assert_allclose(baseline, field['eta_covariance'], atol=1e-8, rtol=1e-8)
    return [(structural, np.asarray([center, 1., 1.]), ['f_center', 'common_intercept_root_multiplier', 'task_intercept_root_multiplier']),
        (generic, np.asarray([center, common, task]), ['f_center', 'common_constant_variance', 'task_constant_variance'])], risk_directions, risk


def moment_gradient(mean, covariance, directions, post_mean, post_covariance):
    inverse_delta = cho_solve(cho_factor(covariance, lower=True), post_mean - mean)
    return np.r_[-inverse_delta.sum() / 48, score(mean, covariance, directions, post_mean, post_covariance)]


def gaussian(mean, covariance, labels, noise, directions):
    factor = cho_factor(covariance + np.diag(noise), lower=True); error = labels - mean
    alpha = cho_solve(factor, error); post_mean = mean + covariance @ alpha
    post_covariance = covariance - covariance @ cho_solve(factor, covariance)
    direct = np.r_[-alpha.sum() / 48, [.5 * (np.trace(cho_solve(factor, d)) - alpha @ d @ alpha) / 48 for d in directions]]
    return direct, post_mean, .5 * (post_covariance + post_covariance.T)


def verify_gaussian(builder, parameters, labels, noise):
    mean, covariance, directions = builder(parameters)
    direct, post_mean, post_covariance = gaussian(mean, covariance, labels, noise, directions)
    moment = moment_gradient(mean, covariance, directions, post_mean, post_covariance)
    np.testing.assert_allclose(moment, direct, atol=1e-8, rtol=1e-8)
    differences = []
    for j in range(len(parameters)):
        scale = max(abs(parameters[j]), 1.) if j == 0 else max(abs(parameters[j]), np.mean(np.diag(covariance)))
        h = 1e-5 * scale
        def value(t):
            m, k, _ = builder(t); return -log_density(labels - m, k + np.diag(noise)) / 48
        t = parameters.copy()
        if j and parameters[j] == 0:
            t[j] += h; first = value(t); t[j] += h
            finite = (-3 * value(parameters) + 4 * first - value(t)) / (2 * h)
        else:
            t[j] += h; upper = value(t); t[j] -= 2 * h
            finite = (upper - value(t)) / (2 * h)
        np.testing.assert_allclose(finite, direct[j], atol=1e-6, rtol=1e-5); differences.append(abs(finite - direct[j]))
    return {'moment_identity_error': float(np.max(np.abs(moment - direct))), 'finite_difference_max_error': float(max(differences))}


def run():
    started = time.process_time(); spec = read(PLAN); source = read(spec['source_posterior_results'])
    assert source['all_monte_carlo_gates_passed']
    fields = read(spec['source_inputs'])['rows']; mixture = read(Path(source['execution']['output']) / 'mixtures.json')['rows']
    rows, checks = [], []
    for field in fields:
        systems, risk_directions, risk = source_systems(spec, field)
        rm = np.asarray(field['eta_mean']); rk = np.asarray(field['eta_covariance'])
        rp = np.asarray([0., 1., 1., 0., risk['white_discrepancy']])
        def risk_builder(p):
            return rm + p[0], sum(p[j + 1] * d for j, d in enumerate(risk_directions)), risk_directions
        z = np.log(field['sample_variances']) - digamma(1.); v = np.full(48, float(polygamma(1, 1.)))
        risk_check = verify_gaussian(risk_builder, rp, z, v)
        for index, (builder, parameters, names) in enumerate(systems):
            arm = field['arms'][index]; row = next(r for r in source['rows'] if (r['layout'], r['arm']) == (field['layout'], arm))
            mean, covariance, directions = builder(parameters)
            np.testing.assert_allclose(covariance, field['f_covariances'][index], atol=1e-10, rtol=1e-10)
            check = verify_gaussian(builder, parameters, np.asarray(field['batch_means']), np.asarray(field['sample_variances']) / 3)
            f_scores = [moment_gradient(mean, covariance, directions, np.asarray(row[p + 'posterior_mean']), np.asarray(row[p + 'posterior_covariance']))
                for p in ('', 'first_half_', 'second_half_')]
            states = np.asarray(next(r['eta_states_by_chain'] for r in mixture if (r['layout'], r['arm']) == (field['layout'], arm)))
            risk_scores = [moment_gradient(rm, rk, risk_directions, np.asarray(row['eta_posterior_mean']), np.asarray(row['eta_posterior_covariance']))]
            for half in (states[:, :64], states[:, 64:]):
                flat = half.reshape(-1, 48); risk_scores.append(moment_gradient(rm, rk, risk_directions, flat.mean(axis=0), np.cov(flat, rowvar=False)))
            fp = [np.r_[g[0], projected(parameters[1:], g[1:], np.full(2, np.inf))] for g in f_scores]
            ep = [np.r_[g[0], projected(rp[1:], g[1:], np.full(4, np.inf))] for g in risk_scores]
            rows.append({'layout': field['layout'], 'arm': arm, 'f_parameter_names': names, 'f_parameters': parameters.tolist(),
                'f_raw_scores_full_first_second': [g.tolist() for g in f_scores],
                'f_projected_scores_full_first_second': [g.tolist() for g in fp],
                'f_split_sign_agreement': ((fp[1] * fp[2] > 0) | ((fp[1] == 0) & (fp[2] == 0))).tolist(),
                'risk_parameter_names': ['eta_mean_offset', 'common_covariance_multiplier', 'task_covariance_multiplier', 'task_intercept_variance', 'white_variance'],
                'risk_parameters': rp.tolist(), 'risk_raw_scores_full_thinned_first_second': [g.tolist() for g in risk_scores],
                'risk_projected_scores_full_thinned_first_second': [g.tolist() for g in ep],
                'risk_split_sign_agreement': ((ep[1] * ep[2] > 0) | ((ep[1] == 0) & (ep[2] == 0))).tolist(),
                'risk_original_task_diagonal_max': float(np.max(np.diag(risk['task_coefficient_covariance']))),
                'risk_original_white_variance': risk['white_discrepancy']})
            checks.append({'layout': field['layout'], 'arm': arm, 'f': check, 'eta': risk_check})
            print('level/risk raw score', field['layout'], arm, 'f', fp[0].tolist(), 'risk', ep[0].tolist(),
                'risk halves', ep[1].tolist(), ep[2].tolist(), flush=True)
        assert time.process_time() - started < spec['maximum_single_thread_cpu_seconds']
    out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    save(out / 'results.json', {'execution': spec, 'rows': rows, 'Gaussian_verification_checks': checks,
        'all_Gaussian_score_and_derivative_checks_passed': True, 'diagnostic_interpreted': True,
        'cpu_seconds': time.process_time() - started, 'cost': spec['cost'],
        'no_model_adopted': True, 'independent_confirmation': False,
        'interpretation': 'Raw source Fisher moment scores at fixed learned shape law. Small risk prior eigenvalues can make absent-variance moment gradients Monte Carlo unstable;thinned half disagreement cannot authorize optimization. Zero task/white directions use their actual variance derivatives. No held-task outcomes or new sampling.'})


if __name__ == '__main__':
    run()
