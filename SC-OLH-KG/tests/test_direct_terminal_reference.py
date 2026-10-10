import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtri
from scipy.stats import norm, qmc

from core.direct_terminal_reference import (
    expected_quality, expected_certificate, expected_certificates, fast_expected_certificates,
    direct_terminal_reference)
from core.guarded_deployment_kg import conditional_safety_power
from representation.joint_response_mixture import JointResponseMixture
from test_positive_source_mixture import field, terminal


class DirectTerminalReferenceTests(unittest.TestCase):
    def test_fast_rule_keeps_sharp_uncertain_eta_certificate_adaptive(self):
        g = field([-.5], [1.])
        g.covariances[:, 1, 1] = 4.
        actual, diagnostics = fast_expected_certificates(g, [0], [359], terminal().protocol)
        expected, _ = expected_certificate(g, 0, 359, terminal().protocol)
        self.assertEqual(diagnostics['adaptive_fallback_actions'], 1)
        self.assertAlmostEqual(actual[0], expected, places=11)

    def test_batched_certificates_match_scalar_for_reordered_policies_and_budgets(self):
        g = field([-.18, -.3], [.4, .6])
        g.means[:, ::2] += [0., .08, -.04]
        for candidate in range(3):
            ix = [2*candidate, 2*candidate+1]
            g.covariances[:, ix[0], ix[0]] = .0004
            g.covariances[:, ix[1], ix[1]] = .25
            g.covariances[:, ix[0], ix[1]] = g.covariances[:, ix[1], ix[0]] = .004
        candidates = [2, 0, 1, 0]; budgets = [128, 359, 83, 128]
        actual, bound = expected_certificates(g, candidates, budgets, terminal().protocol)
        expected = [expected_certificate(g, i, n, terminal().protocol)[0]
            for i, n in zip(candidates, budgets)]
        np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=0.)
        self.assertLess(bound, 1e-10)

    def test_clipped_objective_integral_matches_piecewise_density_integral(self):
        f = field([-.15, .7, 1.7], [.2, .5, .3]); f.covariances[:, 0, 0] = .09
        values = []
        for mean in [-.15, .7, 1.7]:
            value = norm.cdf(0., mean, .3) + quad(
                lambda x: (1 - x / 1.5) * norm.pdf(x, mean, .3), 0., 1.5)[0]
            values.append(value)
        self.assertAlmostEqual(expected_quality(f, 0, 1.5), np.array([.2, .5, .3]) @ values, places=12)

    def test_certificate_integral_matches_joint_gaussian_draws_with_correlation(self):
        g = field([-.18, -.3], [.4, .6])
        g.covariances[:, :2, :2] = [[.0004, .004], [.004, .25]]
        nodes = ndtri(qmc.Sobol(2, scramble=True, seed=202610077).random_base2(18))
        values = []
        for mean in g.means[:, :2]:
            draws = mean + nodes @ np.linalg.cholesky(g.covariances[0, :2, :2]).T
            values.append(conditional_safety_power(draws[:, 0], np.exp(.5 * draws[:, 1]), 128, .05, .05).mean())
        expected = np.array([.4, .6]) @ values
        actual, error = expected_certificate(g, 0, 128, terminal().protocol)
        self.assertAlmostEqual(actual, expected, delta=1e-5)
        self.assertLess(error, 1e-10)

    def test_shared_identity_weights_do_not_independently_pool_heads(self):
        a = (field([.1, .7], [.3, .7]), field([-.5, .15], [.8, .2]))
        b = (field([.8], [1.]), field([-.55], [1.]))
        model = JointResponseMixture(['A', 'B'], [a, b], prior_weights=[.2, .8])
        action = {'kind': 'direct', 'primary': 0, 'shortlist': [0], 'incumbent_position': 0, 'budgets': [360]}
        result = direct_terminal_reference(model, action, terminal().protocol)
        quality = np.array([expected_quality(f, 0, 1.5) for f, _ in model.components])
        probability = np.array([expected_certificate(g, 0, 360, terminal().protocol)[0] for _, g in model.components])
        self.assertAlmostEqual(result['expected_reward'], model.weights @ (quality * probability), places=12)
        self.assertGreater(abs(result['expected_reward'] - (model.weights @ quality) * (model.weights @ probability)), .001)
        self.assertEqual(model.paid_episodes, 0)
        self.assertEqual(model.history, [])


if __name__ == '__main__':
    unittest.main()
