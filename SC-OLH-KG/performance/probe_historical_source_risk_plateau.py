"""Source-only risk covariance probes on certified retained posterior states."""

import time

import numpy as np
from scipy.special import logsumexp

from core.gpr import posterior_mixture_weights
from performance.complete_historical_source_heads import ROOT, IDENTITY, read, save
from performance.continue_historical_raw_source_learning import load_mixture
from performance.learn_historical_raw_source_amplitudes import covariance_builder
from performance.check_legacy_f_risk_amplitude_importance import log_ratios, verify_gaussian_identity
from performance.check_legacy_f_correlated_source_raw_update import diagnostics
from performance.diagnose_legacy_f_raw_source_covariance_score import score
from performance.resolve_historical_source_variance import weighted_moments

PLAN = 'performance/manifests/historical_source_risk_plateau_v1.json'


def ratio_summary(logratio, weights, gate):
    """SNIS ratio influence retains the actual current-law positive weights."""
    logratio, weights = np.asarray(logratio), np.asarray(weights)
    weights = weights / weights.sum()
    log_evidence = float(logsumexp(np.log(weights) + logratio))
    _, updated = posterior_mixture_weights(weights.ravel(), logratio.ravel())
    weight_ess = float(1. / np.sum(updated ** 2))
    if np.max(abs(logratio)) == 0.:
        return {'log_source_evidence_ratio': 0., 'conditional_MC_interval_95': [0., 0.],
            'importance_ESS': weight_ess, 'split_Rhat': 1.,
            'chain_functional_ESS': float(weights.size), 'split_log_ratio_difference': 0.,
            'precision_gate_passed': True, 'eligible_improvement_excludes_zero': False}
    scaled = np.exp(logratio - logratio.max())
    total = weights * scaled
    influence = total / total.mean() - weights / weights.mean()
    diag = diagnostics(influence[..., None])
    ess = float(min(diag['autocorrelation_ess'])); rhat = float(max(diag['split_rhat']))
    count = weights.shape[1]
    halves = [float(logsumexp(np.log(weights[:, s] / weights[:, s].sum()) + logratio[:, s]))
        for s in (slice(0, count // 2), slice(count // 2, None))]
    gap = abs(halves[0] - halves[1])
    passed = bool(weight_ess >= gate['minimum_importance_ESS']
        and ess >= gate['minimum_chain_functional_ESS']
        and rhat <= gate['maximum_split_Rhat']
        and gap <= gate['maximum_split_log_ratio_difference'])
    se = float(influence.std(ddof=1) / np.sqrt(ess))
    interval = [log_evidence - 1.96 * se, log_evidence + 1.96 * se] if passed else None
    return {'log_source_evidence_ratio': log_evidence, 'conditional_MC_interval_95': interval,
        'importance_ESS': weight_ess, 'split_Rhat': rhat, 'chain_functional_ESS': ess,
        'split_log_ratio_difference': gap, 'precision_gate_passed': passed,
        'eligible_improvement_excludes_zero': bool(passed and interval[0] > 0.)}


def run(plan=PLAN):
    spec = read(plan); out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    path = out / 'results.json'
    if path.exists():
        return read(str(path.relative_to(ROOT)))
    certified = read(spec['source_precision']); learned = read(spec['source_learning'])
    source = read(spec['source_plan']); laws = read(source['source_laws'])['rows']
    assert len(certified['rows']) == 32 and all(r['positive_mixture_precision_passed'] for r in certified['rows'])
    started = time.process_time(); checks = [verify_gaussian_identity()]; rows = []
    for record in certified['rows']:
        case = next(r for r in learned['evidence_rows'] if all(r[k] == record[k] for k in (*IDENTITY, 'head')))
        risk = next(r for r in laws if r['head'] == ('eta_g' if case['head'] == 'g' else 'eta_f')
            and all(r[k] == case[k] for k in IDENTITY))
        theta = np.asarray(case['eta_common_task_scales']); builder = covariance_builder(risk)
        covariance, derivatives = builder(theta)[:2]; mean = np.asarray(risk['source_mean'])
        eta, weights = load_mixture(record, case); errors = eta.reshape(-1, 48) - mean
        em, ec = weighted_moments(eta, weights)
        gradient = score(mean, covariance, derivatives, em, ec)
        for coordinate in range(2):
            delta = 1e-5 * theta[coordinate]
            assert delta > 0.
            offset = np.eye(2)[coordinate] * delta
            log_plus = log_ratios(errors, covariance, builder(theta + offset)[0]).reshape(weights.shape)
            log_minus = log_ratios(errors, covariance, builder(theta - offset)[0]).reshape(weights.shape)
            finite = -(logsumexp(np.log(weights) + log_plus) - logsumexp(np.log(weights) + log_minus)) / (96 * delta)
            np.testing.assert_allclose(gradient[coordinate], finite, atol=2e-7, rtol=2e-5)
            checks.append({**{k: case[k] for k in (*IDENTITY, 'head')}, 'coordinate': coordinate,
                'same_state_Fisher_score': float(gradient[coordinate]), 'finite_evidence_derivative': float(finite),
                'absolute_error': float(abs(finite - gradient[coordinate])), 'passed': True})
        for relative in spec['relative_common_task_points']:
            if time.process_time() - started >= spec['maximum_single_thread_cpu_seconds']:
                raise TimeoutError('registered source risk probe CPU budget reached')
            proposed = theta * relative
            logratio = log_ratios(errors, covariance, builder(proposed)[0]).reshape(weights.shape)
            if relative == [1., 1.]:
                np.testing.assert_allclose(logratio, 0., atol=1e-12, rtol=0.)
                logratio[:] = 0.
            summary = ratio_summary(logratio, weights, spec['precision_gates'])
            rows.append({**{k: case[k] for k in (*IDENTITY, 'head')},
                'current_eta_common_task_scales': theta.tolist(), 'relative_common_task_scales': relative,
                'eta_common_task_scales': proposed.tolist(), **summary,
                'normalizer_scope': 'same identity, fixed current f covariance, probe/current round2 risk law'})
    result = {'execution_plan': plan, 'rows': rows, 'checks': checks,
        'all_registered_cases_complete': len(rows) == 32 * len(spec['relative_common_task_points']),
        'all_identity_checks_passed': True,
        'precision_points_passed': sum(r['precision_gate_passed'] for r in rows),
        'eligible_improving_points': sum(r['eligible_improvement_excludes_zero'] for r in rows),
        'cpu_seconds': time.process_time() - started, 'cost': spec['cost'],
        'historical_implementation_reused': 'check_legacy_f_risk_amplitude_importance.log_ratios; core.gpr.posterior_mixture_weights',
        'no_model_adopted': True, 'independent_confirmation': False}
    save(path, result)
    print('source risk probes', len(rows), 'qualified improvements', result['eligible_improving_points'],
        'CPU', round(result['cpu_seconds'], 3), flush=True)
    return result


if __name__ == '__main__':
    run()
