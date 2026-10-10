import unittest

import numpy as np

from performance.complete_historical_source_learning_round2 import integrate_sites, reverse_adjacent
from performance.integrate_historical_changed_source import integrate_change
from performance.finalize_historical_changed_candidate import site_summary
from performance.prepare_legacy_f_variance_reference import VarianceReferencePosterior, variance_reference
from performance.continue_historical_raw_source_learning import risk_bank_moments
from performance.diagnose_historical_source_risk_em_rate import expected_complete_curvature


class ContinuedSourceLearningTests(unittest.TestCase):
    def test_louis_curvature_matches_independent_gaussian_marginal_likelihood(self):
        directions = [np.diag([1., .4, .7]), np.ones((3, 3))]
        theta = np.array([.6, .2]); noise = np.diag([.5, .3, .4]); residual = np.array([.3, -.2, .5])
        def covariance(value): return .1 * np.eye(3) + sum(t * d for t, d in zip(value, directions))
        current = covariance(theta); inverse = np.linalg.inv(current)
        posterior = np.linalg.inv(inverse + np.linalg.inv(noise))
        delta = posterior @ np.linalg.solve(noise, residual)
        second = posterior + np.outer(delta, delta)
        complete = expected_complete_curvature(current, directions, second)
        def marginal(value):
            total = covariance(value) + noise
            return .5 * (np.linalg.slogdet(total)[1] + residual @ np.linalg.solve(total, residual)) / 3
        for i, direction in enumerate(directions):
            b = inverse @ direction @ inverse
            score_variance = .5 * np.trace(b @ posterior @ b @ posterior) + delta @ b @ posterior @ b @ delta
            observed = complete[i, i] - score_variance / 3
            step = np.zeros(2); step[i] = 1e-4
            finite = (marginal(theta + step) - 2 * marginal(theta) + marginal(theta - step)) / step[i] ** 2
            self.assertAlmostEqual(observed, finite, places=7)

    def test_risk_bank_second_moments_center_on_the_actual_prior(self):
        case = {'layout': '6', 'selector': 'spectral', 'support': 'joint', 'treatment': 'deconvolved_task',
            'head': 'g', 'f_common_task_scales': [.2, .5], 'eta_common_task_scales': [.2, .6]}
        prior = np.array([-7., -6.]); draws = np.random.default_rng(19).normal(size=(4, 100, 2)) + prior + [.4, -.2]
        moments = {}
        parts = (slice(None), slice(0, 50), slice(50, None))
        for name, part in zip(('full', 'first_half', 'second_half'), parts):
            values = draws[:, part].reshape(-1, 2)
            moments[name] = {'eta_posterior_mean': values.mean(axis=0).tolist(),
                'eta_posterior_covariance': np.cov(values, rowvar=False, ddof=0).tolist()}
        bank = {'functional_precision_passed': True, 'score_95_MC_halfwidth_below_stationarity_threshold': True,
            'current_candidate_parameters': case, 'risk_sufficient_moments': moments}
        mean, covariance, seconds = risk_bank_moments(bank, case, prior)
        for second, part in zip(seconds, parts):
            error = draws[:, part].reshape(-1, 2) - prior
            np.testing.assert_allclose(second, error.T @ error / len(error), atol=1e-14)
        np.testing.assert_allclose(mean, draws.mean(axis=(0, 1)))
        np.testing.assert_allclose(covariance, np.cov(draws.reshape(-1, 2), rowvar=False, ddof=0))
        different = {**case, 'eta_common_task_scales': [.2, .7]}
        with self.assertRaises(AssertionError):
            risk_bank_moments(bank, different, prior)

    def test_local_unweighted_variance_integral_matches_full_conditional_moments(self):
        model = VarianceReferencePosterior([.2, -.4], [[.5, .15], [.15, .3]], [.6, -.1], [.01, .04])
        rm, rc = variance_reference(np.array([-2., -1.]), np.array([[.4, .1], [.1, .5]]), [.01, .04])
        states = np.array([[-3., -2.], [-1., -.2], [-2., -.4]])
        full = integrate_change(model, model, states, rm, rc, [1., 0.], lambda p, e: 0.)
        local = integrate_sites(model, states, rm, rc, 0, [0, 1])
        np.testing.assert_allclose(local['displacements'], full['displacements'], atol=1e-10)
        np.testing.assert_allclose(local['second_moments'],
            np.diagonal(full['covariances'], axis1=1, axis2=2) + full['displacements'] ** 2, atol=1e-10)

    def test_equal_weight_site_summary_has_no_constant_normalizer_coordinate(self):
        values = np.random.default_rng(3).normal(size=(4, 256))
        summary = site_summary(values, 1. + values ** 2, np.ones_like(values))
        self.assertAlmostEqual(summary['variance'], 1. + values.var(), places=12)
        self.assertTrue(np.isfinite(summary['maximum_split_rhat']))
        self.assertTrue(np.isfinite(summary['minimum_autocorrelation_ess']))

    def test_adjacent_reverse_ratio_restores_prior_normalization(self):
        class Constant:
            def __init__(self, value): self.value = value
            def evaluate(self, point): return self.value, None, None

        nodes, weights = np.polynomial.hermite.hermgauss(64)
        states = np.tile((np.sqrt(2 * .6) * nodes)[None, :, None], (4, 1, 1))
        weights = np.tile(weights[None, :], (4, 1)) / (4 * np.sqrt(np.pi))
        gate = {'maximum_split_rhat_eta_and_rao_blackwell_f_moments': 1.05,
            'minimum_autocorrelation_ess_eta_and_rao_blackwell_f_moments': 100}
        evidence = {'minimum_importance_ESS': 100, 'maximum_split_log_ratio_difference': .1}
        value = reverse_adjacent(Constant(0.), Constant(.7), np.zeros(1), np.eye(1),
            np.array([[.6]]), states, weights, gate, evidence)
        self.assertAlmostEqual(value['log_source_evidence_ratio_new_vs_previous'], .7, places=10)


if __name__ == '__main__':
    unittest.main()
