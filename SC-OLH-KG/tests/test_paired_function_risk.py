import copy
import json
import unittest
from pathlib import Path

import numpy as np
from scipy.integrate import dblquad
from scipy.special import digamma, logsumexp
from scipy.stats import multivariate_normal

from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.joint_response_mixture import JointResponseMixture
from representation.paired_function_risk import PairedFunctionRiskMixture, paired_local_moments
from representation.physical_risk_prior import log_sample_variance_density


class PairedFunctionRiskTests(unittest.TestCase):
    def test_real_source_density_reaches_fixed_tolerance_beyond_first_order_cap(self):
        fixture = json.loads((Path(__file__).parent/'fixtures/paired_function_risk_quadrature_case.json').read_text())
        result = paired_local_moments(fixture['mean'], fixture['covariance'], fixture['labels'], fixture['eta_labels'], fixture['replicates'])
        self.assertLess(result['integration_error'], 1e-7)
        self.assertGreater(np.linalg.eigvalsh(result['covariance']).min(), 0.)

    def test_zero_coupling_reproduces_paid_fields_identity_weights_and_batch_scores(self):
        pairs = []
        for offset in (0., .12):
            fields = []
            for mean, local in (([-.2+offset, -1.4], [[.05, .02], [.02, .5]]),
                                ([.1-offset, -1.1], [[.06, -.035], [-.035, .4]])):
                m = np.tile(mean, (2, 1)); covariance = np.kron([[1., .25], [.25, 1.]], local)
                fields.append(JointFunctionRiskPosterior.from_moments(m, covariance.reshape(2, 2, 2, 2)))
            pairs.append(tuple(fields))
        old = JointResponseMixture(['a', 'b'], pairs)
        moments = [JointResponseMixture(['one'], [pair]).predict_joint() for pair in pairs]
        new = PairedFunctionRiskMixture(['a', 'b'], [m[0] for m in moments], [m[1] for m in moments])
        for candidate, values in ((0, [-.12, .25]), (1, [-.3, .05])):
            a, b = old.update_raw(candidate, values), new.update_raw(candidate, values)
            self.assertAlmostEqual(a['mixture_log_score'], b['mixture_log_score'], places=11)
        batch = [[-.18, .2], [-.23, -.04], [.04, .15]]
        old.update_replicates(0, batch); new.update_replicates(0, batch)
        for a, b in zip(old.predict_joint(), new.predict_joint()):
            np.testing.assert_allclose(a, b, atol=1e-11)
        np.testing.assert_allclose(old.weights, new.weights, atol=1e-12)
        before = copy.deepcopy((new.means, new.covariances, new.log_weights, new.history))
        a, b = old.batch_log_scores(1, [-.1, .2], [-1.8, -1.5], 3), new.batch_log_scores(1, [-.1, .2], [-1.8, -1.5], 3)
        self.assertAlmostEqual(a['joint'], b['joint'], places=11)
        np.testing.assert_allclose(a['heads'], b['heads'], atol=1e-11)
        for a, b in zip(before[:3], (new.means, new.covariances, new.log_weights)):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(before[3], new.history)

    def test_correlated_batch_evidence_matches_continuous_integral_and_raw_moments(self):
        mean = np.array([.2, -1.1, -.3, -.8])
        covariance = np.array([[.05, -.035, .025, .01], [-.035, .5, .018, .12],
            [.025, .018, .06, -.025], [.01, .12, -.025, .4]])
        labels, eta, count = np.array([.35, -.15]), np.array([-1.8, -1.3]), 3
        result = paired_local_moments(mean, covariance, labels, eta, count)
        mu, risk = [0, 2], [1, 3]; ce = covariance[np.ix_(risk, risk)]
        loading = covariance[np.ix_(mu, risk)]@np.linalg.inv(ce)
        conditional = covariance[np.ix_(mu, mu)]-loading@covariance[np.ix_(risk, mu)]
        s2 = np.exp(eta+digamma((count-1)/2)-np.log((count-1)/2))

        def integrand(e1, e0):
            e = np.array([e0, e1]); m = mean[mu]+loading@(e-mean[risk])
            value = multivariate_normal.pdf(e, mean[risk], ce)
            value *= multivariate_normal.pdf(labels, m, conditional+np.diag(np.exp(e)/count))
            return value*np.exp(log_sample_variance_density(e, s2, count).sum())

        bound = 9*np.sqrt(np.diag(ce))
        evidence, _ = dblquad(integrand, mean[1]-bound[0], mean[1]+bound[0],
            lambda _: mean[3]-bound[1], lambda _: mean[3]+bound[1], epsabs=1e-9, epsrel=1e-8)
        self.assertAlmostEqual(result['log_evidence'], np.log(evidence), places=6)
        draws = np.random.default_rng(62206).multivariate_normal(mean, covariance, 250000)
        logs = -.5*count*draws[:, risk].sum(axis=1)
        logs -= .5*np.sum((count*(labels-draws[:, mu])**2+(count-1)*s2)*np.exp(-draws[:, risk]), axis=1)
        weights = np.exp(logs-logsumexp(logs)); expected = weights@draws; centered = draws-expected
        np.testing.assert_allclose(result['mean'], expected, atol=.005)
        np.testing.assert_allclose(result['covariance'], (centered.T*weights)@centered, atol=.004)
        self.assertLess(result['integration_error'], 1e-7)

    def test_raw_f_changes_correlated_g_and_keeps_mean_risk_covariance(self):
        local = np.diag([.05, .2, .06, .25]); local[0, 2] = local[2, 0] = .03
        covariance = np.kron([[1., .4], [.4, 1.]], local)
        mean = np.array([[0., -2., 0., -2.], [.2, -2.1, .1, -2.1]])
        model = PairedFunctionRiskMixture(['a'], [mean], [covariance.reshape(2, 4, 2, 4)])
        model.update_raw(0, [.6, 0.])
        actual_mean, actual_covariance = model.predict_joint()
        self.assertGreater(actual_mean[0, 2], .02)
        self.assertGreater(abs(actual_mean[1, 2]-mean[1, 2]), .005)
        self.assertGreater(abs(actual_covariance[0, 0, 0, 1]), 1e-5)
        self.assertGreater(np.linalg.eigvalsh(actual_covariance.reshape(8, 8)).min(), 0.)
        self.assertEqual(model.paid_episodes, 1)


if __name__ == '__main__':
    unittest.main()
