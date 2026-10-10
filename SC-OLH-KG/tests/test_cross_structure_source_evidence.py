import unittest

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, norm

from performance.check_legacy_f_collapsed_source_raw_update import CollapsedRawSourcePosterior
from performance.compare_historical_source_structures import scientific_log_density, combine_directions
from performance.bridge_historical_source_normalizers import bridge_solution, bridge_summary
from representation.physical_risk_prior import log_sample_variance_density


class CrossStructureSourceEvidenceTests(unittest.TestCase):
    def test_bridge_recovers_known_normalizer_from_two_weighted_gaussian_integrals(self):
        nodes, weights = np.polynomial.hermite.hermgauss(64)
        proposal = -.1 + 1.2 * np.sqrt(2.) * nodes
        posterior = .2 + .9 * np.sqrt(2.) * nodes
        def difference(x):
            return 1.3 + norm.logpdf(x, .2, .9) - norm.logpdf(x, -.1, 1.2)
        value, passed, _ = bridge_solution(difference(proposal), difference(posterior), weights, weights)
        self.assertTrue(passed)
        self.assertAlmostEqual(value, 1.3, places=10)
        scaled, passed, _ = bridge_solution(difference(proposal), difference(posterior), 3 * weights, 7 * weights)
        self.assertTrue(passed)
        self.assertAlmostEqual(scaled, value, places=12)

    def test_bridge_precision_interval_contains_known_density_normalizer(self):
        rng = np.random.default_rng(419)
        proposal = rng.normal(-.1, 1.2, size=(4, 2048))
        posterior = rng.normal(.2, .9, size=(4, 2048))
        def difference(x):
            return .4 + norm.logpdf(x, .2, .9) - norm.logpdf(x, -.1, 1.2)
        gate = {'minimum_importance_ESS': 100, 'minimum_chain_functional_ESS': 100,
            'maximum_split_Rhat': 1.05, 'maximum_split_log_ratio_difference': .1}
        result = bridge_summary(difference(proposal), difference(posterior), np.ones_like(proposal), np.ones_like(posterior), gate)
        self.assertTrue(result['precision_gate_passed'])
        self.assertLess(result['conditional_MC_interval_95'][0], .4)
        self.assertGreater(result['conditional_MC_interval_95'][1], .4)

    def test_raw_density_retains_both_gaussian_normalizers_and_mean_shifts(self):
        fmean = np.array([.2, -.3])
        covariance = np.array([[.8, .1], [.1, .4]])
        labels = np.array([.4, -.1])
        s2 = np.array([.1, .3])
        risk_mean = np.array([-1., -2.])
        risk_cov = np.array([[.7, .2], [.2, .9]])
        eta = np.array([[[-1.2, -1.8], [-.4, -2.3]]])
        model = CollapsedRawSourcePosterior(fmean, covariance, labels, s2)
        actual = scientific_log_density(model, risk_mean, risk_cov, eta)
        expected = [multivariate_normal.logpdf(labels, fmean, covariance + np.diag(np.exp(p) / 3))
            + np.sum(log_sample_variance_density(p, s2, 3))
            + multivariate_normal.logpdf(p, risk_mean, risk_cov) for p in eta[0]]
        np.testing.assert_allclose(actual[0], expected, atol=1e-12, rtol=0.)

    def test_task_policy_alignment_preserves_complete_raw_density(self):
        order = np.array([1, 0])
        mean = np.array([.2, -.3]); covariance = np.array([[.8, .1], [.1, .4]])
        labels = np.array([.4, -.1]); s2 = np.array([.1, .3])
        risk_mean = np.array([-1., -2.]); risk_cov = np.array([[.7, .2], [.2, .9]])
        eta = np.array([[[-1.2, -1.8], [-.4, -2.3]]])
        original = scientific_log_density(CollapsedRawSourcePosterior(mean, covariance, labels, s2), risk_mean, risk_cov, eta)
        aligned = scientific_log_density(CollapsedRawSourcePosterior(mean[order], covariance[np.ix_(order, order)],
            labels[order], s2[order]), risk_mean[order], risk_cov[np.ix_(order, order)], eta[..., order])
        np.testing.assert_allclose(aligned, original, atol=1e-12, rtol=0.)

    def test_forward_and_reverse_recover_known_gaussian_evidence_ratio(self):
        nodes, weights = np.polynomial.hermite.hermgauss(64)
        reference = np.sqrt(2.) * nodes
        target = .2 + 1.1 * reference
        def logratio(x):
            return .3 + norm.logpdf(x, .2, 1.1) - norm.logpdf(x)
        forward_logz = float(logsumexp(np.log(weights / np.sqrt(np.pi)) + logratio(reference)))
        reverse_logz = float(logsumexp(np.log(weights / np.sqrt(np.pi)) - logratio(target)))
        def summary(value):
            return {'log_source_evidence_ratio': value, 'precision_gate_passed': True,
                'conditional_MC_interval_95': [value - .001, value + .001]}
        result = combine_directions(summary(forward_logz), summary(reverse_logz))
        self.assertTrue(result['cross_structure_evidence_certified'])
        self.assertAlmostEqual(result['forward_log_evidence_ratio'], .3, places=12)
        self.assertAlmostEqual(result['reverse_log_evidence_ratio'], .3, places=12)

    def test_failed_overlap_or_disagreeing_banks_remain_uncertified(self):
        forward = {'log_source_evidence_ratio': .3, 'precision_gate_passed': True,
            'conditional_MC_interval_95': [.29, .31]}
        reverse = {'log_source_evidence_ratio': -.5, 'precision_gate_passed': True,
            'conditional_MC_interval_95': [-.51, -.49]}
        self.assertFalse(combine_directions(forward, reverse)['cross_structure_evidence_certified'])
        reverse.update(precision_gate_passed=False, conditional_MC_interval_95=None)
        self.assertFalse(combine_directions(forward, reverse)['cross_structure_evidence_certified'])


if __name__ == '__main__':
    unittest.main()
