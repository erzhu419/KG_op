"""Cached source evidence ratios using the historical mixture-weight engine."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.gpr import posterior_mixture_weights
from performance.check_legacy_f_correlated_source_raw_update import diagnostics
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_raw_source_covariance_score import score
from performance.diagnose_legacy_f_source_incremental_calibration import log_density
from performance.run_legacy_related_inventory_source import save

PLAN = 'performance/manifests/legacy_joint_population_f_risk_amplitude_importance_plan_v1.json'


def log_ratios(error, old_covariance, new_covariance):
    old, new = [cho_factor(c, lower=True) for c in (old_covariance, new_covariance)]
    delta = cho_solve(old, error.T) - cho_solve(new, error.T)
    determinant = 2 * (np.log(np.diag(old[0])).sum() - np.log(np.diag(new[0])).sum())
    return .5 * (determinant + np.einsum('ij,ji->i', error, delta))


def verify_gaussian_identity():
    """Analytic posterior expectation of a prior ratio; no Monte Carlo."""
    old = np.array([[.9, .2], [.2, .6]]); new = np.array([[.6, .12], [.12, .8]])
    noise = np.diag([.3, .4]); labels = np.array([.7, -.5]); identity = np.eye(2)
    factor = cho_factor(old + noise, lower=True)
    mean = old @ cho_solve(factor, labels)
    covariance = old - old @ cho_solve(factor, old)
    inverse = cho_solve(cho_factor(covariance, lower=True), identity)
    precision = inverse + cho_solve(cho_factor(new, lower=True), identity) - cho_solve(cho_factor(old, lower=True), identity)
    linear = inverse @ mean
    expectation = .5 * (np.linalg.slogdet(old)[1] - np.linalg.slogdet(new)[1]
        - np.linalg.slogdet(covariance)[1] - np.linalg.slogdet(precision)[1]
        - mean @ inverse @ mean + linear @ cho_solve(cho_factor(precision, lower=True), linear))
    evidence = log_density(labels, new + noise) - log_density(labels, old + noise)
    np.testing.assert_allclose(expectation, evidence, atol=1e-12, rtol=0.)
    return {'analytic_Gaussian_identity_error': abs(expectation - evidence), 'passed': True}


def run():
    started = time.process_time(); spec = read(PLAN)
    mixtures = read(spec['eta_components']); fields = read(spec['source_inputs'])['rows']
    assert mixtures['all_source_posterior_precision_gates_passed'] and mixtures['all_mixture_retention_gates_passed']
    laws = read(spec['risk_laws'])['rows']; learned = read(spec['source_learning_results'])
    checks = [verify_gaussian_identity()]; rows = []
    for field in fields:
        key = (field['layout'], field['active_arm'])
        law = next(r for r in laws if (r['layout'], r['arm']) == key)
        mixture = next(r for r in mixtures['rows'] if (r['layout'], r['arm']) == key)
        amplitude = next(r['parameters'][3] for r in learned['final_parameters'] if (r['layout'], r['arm']) == key)
        covariance = np.asarray(field['eta_covariance']); mean = np.asarray(field['eta_mean'])
        draws = np.asarray(mixture['eta_states_by_chain']); assert draws.shape == (4, 128, 48)
        errors = draws.reshape(-1, 48) - mean
        common = np.asarray(law['common_prior_covariance']); design = np.asarray(law['source_design'])
        # Original weak tails and the intercept remain fixed even at zero amplitude.
        tail = .0002 / (len(common) - 1)
        excess = (np.diag(common)[1:] - tail) / amplitude
        assert excess.min() >= -1e-14
        strong = np.diag(np.r_[0., np.maximum(excess, 0.)])
        direction = design @ strong @ design.T
        def ratios(a):
            return log_ratios(errors, covariance, covariance + (a - amplitude) * direction)
        baseline = ratios(amplitude); np.testing.assert_allclose(baseline, 0., atol=1e-12, rtol=0.)
        fisher = float(score(mean, covariance, [direction], draws.reshape(-1, 48).mean(axis=0),
            np.cov(draws.reshape(-1, 48), rowvar=False, ddof=0))[0])
        step = 1e-5 * amplitude
        finite = -(logsumexp(ratios(amplitude + step)) - logsumexp(ratios(amplitude - step))) / (2 * step * 48)
        np.testing.assert_allclose(fisher, finite, atol=2e-7, rtol=2e-5)
        checks.append({'layout': key[0], 'arm': key[1], 'same_retained_state_Fisher_score': fisher,
            'finite_evidence_derivative': float(finite), 'derivative_error': abs(fisher - finite), 'passed': True})
        for relative in spec['relative_amplitude_points']:
            if time.process_time() - started >= spec['maximum_single_thread_cpu_seconds']:
                raise TimeoutError('registered source importance CPU budget reached')
            ratio = ratios(relative * amplitude)
            _, weights = posterior_mixture_weights(np.full(512, 1 / 512), ratio)
            log_evidence = float(logsumexp(ratio) - np.log(512))
            if relative == 1:
                ess = 512.; rhat = 1.; se = 0.; halves = [0., 0.]
            else:
                values = np.exp(ratio - ratio.max()).reshape(4, 128, 1)
                diagnostic = diagnostics(values)
                ess = diagnostic['autocorrelation_ess'][0]; rhat = diagnostic['split_rhat'][0]
                se = float(values.std(ddof=1) / np.sqrt(ess) / values.mean() / 48)
                halves = [float(-(logsumexp(part) - np.log(part.size)) / 48)
                          for part in np.split(ratio.reshape(4, 128), 2, axis=1)]
            weight_ess = float(1 / (weights @ weights)); difference = -log_evidence / 48
            limits = spec['precision_gates']
            passed = bool(weight_ess >= limits['minimum_weight_ESS']
                and ess >= limits['minimum_chain_functional_ESS'] and rhat <= limits['maximum_chain_Rhat']
                and abs(halves[0] - halves[1]) <= limits['maximum_half_NLL_difference_per_batch'])
            rows.append({'layout': key[0], 'arm': key[1], 'current_amplitude': amplitude,
                'relative_amplitude': relative, 'source_NLL_difference_per_batch': difference,
                'conditional_MC_interval_95': [difference - 1.96 * se, difference + 1.96 * se] if passed else None,
                'weight_ESS': weight_ess, 'chain_functional_ESS': ess, 'split_Rhat': rhat,
                'half_NLL_differences_per_batch': halves, 'precision_gate_passed': passed,
                'eligible_improvement_excludes_zero': bool(passed and difference + 1.96 * se < 0.)})
    out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    save(out / 'results.json', {'execution': spec, 'checks': checks, 'rows': rows,
        'all_identity_checks_passed': True, 'eligible_improving_points': sum(r['eligible_improvement_excludes_zero'] for r in rows),
        'cpu_seconds': time.process_time() - started, 'cost': spec['cost'],
        'historical_implementation_reused': 'core.gpr.posterior_mixture_weights (d092d982,2026-07-19)',
        'no_model_adopted': True, 'independent_confirmation': False,
        'interpretation': 'Cached raw-source evidence ratios in one risk covariance coordinate. Intervals describe conditional Monte Carlo error;failed weight/chain endpoints cannot select a law. Any covariance adoption requires fresh source conditioning.'})
    print('cached source amplitude points', len(rows), 'eligible improvements', sum(r['eligible_improvement_excludes_zero'] for r in rows),
          'CPU', time.process_time() - started, flush=True)


if __name__ == '__main__':
    run()
