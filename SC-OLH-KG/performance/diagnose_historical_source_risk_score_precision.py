"""Reuse historical raw-likelihood Stein scores on certified current source mixtures."""

import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from performance.complete_historical_source_heads import ROOT, IDENTITY, read, save
from performance.continue_historical_raw_source_learning import load_mixture
from performance.learn_historical_raw_source_amplitudes import covariance_builder
from performance.recondition_historical_changed_candidates import candidate_inputs
from performance.check_legacy_f_collapsed_source_raw_update import CollapsedRawSourcePosterior
from performance.diagnose_legacy_f_raw_source_risk_stein_score import raw_derivatives, verify
from performance.check_legacy_f_correlated_source_raw_update import diagnostics

PLAN = 'performance/manifests/historical_source_risk_score_precision_v1.json'
KEY = (*IDENTITY, 'head')


def summarize(series, weights):
    weights = weights / weights.sum()
    mean = np.einsum('ct,ctj->j', weights, series)
    influence = weights[..., None] * (series - mean) / weights.mean()
    diagnostic = diagnostics(influence)
    se = np.sqrt(influence.var(axis=(0, 1), ddof=1) / np.asarray(diagnostic['autocorrelation_ess']))
    return mean, se, diagnostic


def run(plan=PLAN):
    spec = read(plan); out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    path = out / 'results.json'
    if path.exists():
        return read(str(path.relative_to(ROOT)))
    started = time.process_time()
    closure = read(spec['source_precision']); learned = read(spec['source_learning'])
    assert len(closure['rows']) == 32 and all(r['positive_mixture_precision_passed'] for r in closure['rows'])
    source = read(spec['source_plan']); laws = read(source['source_laws'])['rows']
    selection = read(source['selection_plan']); rows = []; checks = []
    for record in closure['rows']:
        case = next(r for r in learned['evidence_rows'] if all(r[k] == record[k] for k in KEY))
        risk = next(r for r in laws if r['head'] == ('eta_g' if case['head'] == 'g' else 'eta_f')
            and all(r[k] == case[k] for k in IDENTITY))
        theta = np.asarray(case['eta_common_task_scales']); covariance, directions = covariance_builder(risk)(theta)[:2]
        _, relative, _, field, _ = candidate_inputs(case, laws, selection)
        model = CollapsedRawSourcePosterior(relative.mean, relative.covariance, relative.labels, relative.s2)
        eta, weights = load_mixture(record, case)
        checks.append(verify(field, 0, eta[0, 0], directions))
        factor = cho_factor(covariance, lower=True)
        inverse_d = [cho_solve(factor, d) for d in directions]
        traces = np.asarray([np.trace(d) for d in inverse_d])
        error = eta.reshape(-1, 48) - np.asarray(risk['source_mean'])
        precision_error = cho_solve(factor, error.T).T
        moment = np.column_stack([.5 * (traces[i] - np.einsum('ni,ij,nj->n', precision_error, d, precision_error)) / 48
            for i, d in enumerate(directions)])
        stein = []
        for point in eta.reshape(-1, 48):
            if time.process_time() - started >= spec['maximum_single_thread_cpu_seconds']:
                raise TimeoutError('registered cached risk-score CPU budget reached')
            gradient, hessian = raw_derivatives(model, point)
            matrix = hessian + np.outer(gradient, gradient)
            stein.append([-.5 * np.sum(matrix * d.T) / 48 for d in directions])
        moment = moment.reshape(*eta.shape[:2], 2); stein = np.asarray(stein).reshape(moment.shape)
        joint = np.concatenate([moment, stein, moment - stein], axis=-1)
        mean, se, diag = summarize(joint, weights)
        for coordinate in range(2):
            indexes = [coordinate, 2 + coordinate, 4 + coordinate]
            precise = all(diag['split_rhat'][i] <= spec['maximum_split_Rhat']
                and diag['autocorrelation_ess'][i] >= spec['minimum_functional_ESS'] for i in indexes)
            rows.append({**{k:case[k] for k in KEY}, 'coordinate': coordinate,
                'risk_head': risk['head'], 'current_amplitude': float(theta[coordinate]),
                'moment_scaled_score': float(mean[coordinate] * theta[coordinate]),
                'Stein_scaled_score': float(mean[2 + coordinate] * theta[coordinate]),
                'moment_scaled_MCSE': float(se[coordinate] * theta[coordinate]),
                'Stein_scaled_MCSE': float(se[2 + coordinate] * theta[coordinate]),
                'paired_scaled_difference': float(mean[4 + coordinate] * theta[coordinate]),
                'paired_scaled_difference_MCSE': float(se[4 + coordinate] * theta[coordinate]),
                'diagnostic_precision_passed': bool(precise),
                'moment_Stein_difference_inside_conditional_95_MC_interval': bool(precise
                    and abs(mean[4 + coordinate]) <= 1.96 * se[4 + coordinate]),
                'Stein_95_MC_halfwidth_below_stopping_threshold': bool(precise
                    and 1.96 * se[2 + coordinate] * theta[coordinate] <= spec['stationarity_threshold']),
                'Stein_score_and_95_MC_interval_inside_stopping_threshold': bool(precise
                    and (abs(mean[2 + coordinate]) + 1.96 * se[2 + coordinate]) * theta[coordinate]
                        <= spec['stationarity_threshold']),
                'functional_Rhat': [diag['split_rhat'][i] for i in indexes],
                'functional_ESS': [diag['autocorrelation_ess'][i] for i in indexes]})
    result = {'execution_plan': plan, 'rows': rows, 'checks': checks,
        'all_registered_cases_complete': len(rows) == 64, 'all_derivative_identity_checks_passed': True,
        'cpu_seconds': time.process_time() - started, 'new_candidate_MCMC_sweeps': 0,
        'new_source_labels': 0, 'new_target_simulator_calls': 0, 'new_verifier_calls': 0, 'server_jobs': 0,
        'no_model_adopted': True, 'independent_confirmation': False,
        'interval_scope': 'Conditional numerical error on current weighted source states;not independent task uncertainty.',
        'historical_implementation_reused': 'diagnose_legacy_f_raw_source_risk_stein_score.raw_derivatives and verify'}
    save(path, result)
    print('cached risk scores', len(rows), 'CPU', round(result['cpu_seconds'], 2), flush=True)
    return result


if __name__ == '__main__':
    run()
