"""Risk source scores from likelihood derivatives,using original source states."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_solve
from scipy.special import digamma, polygamma

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_f_collapsed_source_raw_update import CollapsedRawSourcePosterior, sample
from performance.check_legacy_f_correlated_source_raw_update import diagnostics
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_raw_source_covariance_score import projected
from performance.diagnose_legacy_f_raw_source_level_risk_score import gaussian, source_systems
from performance.prepare_legacy_f_variance_reference import VarianceReferencePosterior
from performance.run_legacy_related_inventory_source import save

PLAN = 'performance/manifests/legacy_joint_population_f_raw_source_risk_stein_score_plan_v1.json'


def raw_derivatives(model, eta):
    _, factor, alpha = model.evaluate(eta); w = cho_solve(factor, np.eye(len(eta)))
    variance = np.exp(eta) / 3; mean_gradient = .5 * variance * (alpha ** 2 - np.diag(w))
    variance_ratio = model.s2 * np.exp(-eta)
    gradient = mean_gradient + variance_ratio - 1
    hessian = np.diag(mean_gradient - variance_ratio)
    hessian += .5 * np.outer(variance, variance) * (w ** 2 - 2 * np.outer(alpha, alpha) * w)
    return gradient, hessian


def verify(field, index, point, directions):
    model = CollapsedRawSourcePosterior(field['f_means'][index], field['f_covariances'][index], field['batch_means'], field['sample_variances'])
    gradient, hessian = raw_derivatives(model, point); gerrors, herrors = [], []
    for j in (0, 13, 47):
        step = np.zeros(48); step[j] = 1e-5
        finite = (model.evaluate(point + step)[0] - model.evaluate(point - step)[0]) / 2e-5
        column = (raw_derivatives(model, point + step)[0] - raw_derivatives(model, point - step)[0]) / 2e-5
        np.testing.assert_allclose(finite, gradient[j], atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(column, hessian[:, j], atol=1e-6, rtol=1e-5)
        gerrors.append(abs(finite - gradient[j])); herrors.append(float(np.max(np.abs(column - hessian[:, j]))))
    labels = np.log(field['sample_variances']) - digamma(1.); variance = float(polygamma(1, 1.))
    direct, mean, covariance = gaussian(np.asarray(field['eta_mean']), np.asarray(field['eta_covariance']), labels, np.full(48, variance), directions)
    g = (labels - mean) / variance
    matrix = -np.eye(48) / variance + covariance / variance ** 2 + np.outer(g, g)
    stein = np.r_[-g.sum() / 48, [-.5 * np.sum(matrix * d.T) / 48 for d in directions]]
    np.testing.assert_allclose(stein, direct, atol=1e-8, rtol=1e-8)
    return {'layout': field['layout'], 'arm': field['arms'][index], 'raw_gradient_max_error': max(gerrors),
        'raw_Hessian_column_max_error': max(herrors), 'Gaussian_Stein_identity_error': float(np.max(np.abs(stein - direct))), 'passed': True}


def run():
    started = time.process_time(); spec = read(PLAN); level = read(spec['level_risk_plan'])
    assert read(spec['moment_score_results'])['all_Gaussian_score_and_derivative_checks_passed']
    original = read(spec['original_source_execution']); old = read(spec['original_source_results']); assert old['all_monte_carlo_gates_passed']
    fields = read(spec['source_inputs'])['rows']; references = read(original['references'])['rows']
    thin = read(Path(original['output']) / 'mixtures.json')['rows']; out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    systems = {f['layout']: source_systems(level, f) for f in fields}
    checks = [verify(f, i, np.asarray(next(r['reference_mean'] for r in references if r['layout'] == f['layout'])), systems[f['layout']][1]) for f in fields for i in range(2)]
    save(out / 'verification.json', {'execution': spec, 'checks': checks, 'all_checks_passed_before_replay': True})
    rows, records, partitions = [], [], []
    for field in fields:
        reference = next(r for r in references if r['layout'] == field['layout']); directions = systems[field['layout']][1]
        for index, arm in enumerate(field['arms']):
            model = VarianceReferencePosterior(field['f_means'][index], field['f_covariances'][index], field['batch_means'], field['sample_variances'])
            captured = []; replay = sample(field, index, original, started, model, reference['reference_mean'], reference['reference_covariance'], captured)
            previous = next(r for r in old['rows'] if (r['layout'], r['arm']) == (field['layout'], arm))
            for key in ('posterior_mean', 'posterior_covariance', 'eta_posterior_mean', 'eta_posterior_covariance'):
                np.testing.assert_allclose(replay[key], previous[key], atol=1e-10, rtol=0)
            expected = next(r for r in thin if (r['layout'], r['arm']) == (field['layout'], arm))
            states = captured[0]; np.testing.assert_allclose(states[:, ::8], expected['eta_states_by_chain'], atol=1e-10, rtol=0)
            raw_model = CollapsedRawSourcePosterior(field['f_means'][index], field['f_covariances'][index], field['batch_means'], field['sample_variances'])
            scores = []
            for eta in states.reshape(-1, 48):
                if time.process_time() - started >= spec['maximum_single_thread_cpu_seconds']: raise TimeoutError('registered CPU budget reached')
                g, h = raw_derivatives(raw_model, eta); matrix = h + np.outer(g, g)
                scores.append(np.r_[-g.sum() / 48, [-.5 * np.sum(matrix * d.T) / 48 for d in directions]])
            series = np.asarray(scores).reshape(4, 1024, 5); diagnostic = diagnostics(series)
            means = [s.mean(axis=(0, 1)) for s in (series, series[:, :512], series[:, 512:])]
            se = np.sqrt(series.var(axis=(0, 1), ddof=1) / np.asarray(diagnostic['autocorrelation_ess']))
            theta = np.asarray([1., 1., 0., 0.])
            projections = [np.r_[g[0], projected(theta, g[1:], np.full(4, np.inf))] for g in means]
            stable = (np.asarray(diagnostic['split_rhat']) <= 1.05) & (np.asarray(diagnostic['autocorrelation_ess']) >= 100)
            stable &= (projections[0] * projections[1] > 0) & (projections[0] * projections[2] > 0)
            boundary = (np.asarray([False, False, False, True, True]) & (means[0] > 1.96 * se))
            rows.append({'layout': field['layout'], 'arm': arm, 'parameter_names': ['eta_mean_offset', 'common_covariance_multiplier', 'task_covariance_multiplier', 'task_intercept_variance', 'white_variance'],
                'raw_Stein_scores_full_first_second': [g.tolist() for g in means],
                'projected_scores_full_first_second': [g.tolist() for g in projections],
                'conditional_MC_standard_errors': se.tolist(), 'diagnostics': diagnostic,
                'stable_nonzero_projected_directions': stable.tolist(), 'positive_boundary_scores_excluding_zero': boundary.tolist(),
                'original_source_replay_matched': True, 'source_posterior_unchanged': True})
            records.append({'layout': field['layout'], 'arm': arm, 'raw_Stein_score_series_by_chain': series.tolist()})
            partition = {'layout': field['layout'], 'arm': arm}
            for name, part in [('first_half', states[:, :512]), ('second_half', states[:, 512:])]:
                flat = part.reshape(-1, 48); partition[name + '_eta_mean'] = flat.mean(axis=0).tolist()
                partition[name + '_eta_covariance'] = np.cov(flat, rowvar=False).tolist()
            partitions.append(partition)
            print('raw risk Stein score', field['layout'], arm, 'scores', means[0].tolist(), 'MCSE', se.tolist(),
                'stable', stable.tolist(), 'positive zero-boundary', boundary.tolist(), flush=True)
    save(out / 'score_records.json', {'execution': spec, 'rows': records})
    save(out / 'source_eta_partitions.json', {'execution': spec, 'rows': partitions})
    save(out / 'results.json', {'execution': spec, 'rows': rows, 'all_original_source_replays_matched': True,
        'all_derivative_checks_passed': True, 'diagnostic_interpreted': True, 'cpu_seconds': time.process_time() - started,
        'cost': spec['cost'], 'no_model_adopted': True, 'independent_confirmation': False,
        'interpretation': 'Gaussian integration-by-parts score of the same raw source likelihood,using original retained states. Stable derivative directions are source learning evidence only. Posterior law unchanged;moments/gradient records retained for a later M step. No future scoring or extra independent samples.'})


if __name__ == '__main__':
    run()
