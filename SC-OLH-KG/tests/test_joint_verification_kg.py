import itertools
import sys
import unittest
from pathlib import Path

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.stats import chi2, norm, t

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.joint_verification_kg import budgeted_joint_kg, joint_verification_kg, joint_verification_utility
from core.verification_kg import verification_utility


class JointVerificationKGTests(unittest.TestCase):
    def setUp(self):
        self.protocol = {"verification_replicates": 928, "chance_alpha": .05}

    def test_zero_cross_covariance_recovers_original_terminal_utility(self):
        mean = np.array([[1., -.4, -3.], [-1., -.2, -4.]])
        variance = np.array([[.04, .01, .15], [.01, .02, .05]])
        cov = np.array([np.diag(row[1:]) for row in variance])
        actual = joint_verification_utility(mean[:, 0], variance[:, 0], mean[:, 1:], cov, self.protocol)
        expected = verification_utility(mean, variance, self.protocol)
        np.testing.assert_allclose(actual, expected, atol=1e-14)

    def test_correlated_power_matches_independent_gaussian_verifier_simulation(self):
        rng = np.random.default_rng(105070)
        mean, cov = np.array([[-.45, -3.]]), np.array([[[.01, -.027], [-.027, .12]]])
        _, power = joint_verification_utility(np.zeros(1), np.zeros(1), mean, cov, self.protocol)
        draws = rng.multivariate_normal(mean[0], cov[0], size=200000)
        sigma = np.exp(.5 * draws[:, 1])
        sample_mean = draws[:, 0] + sigma / np.sqrt(928) * rng.normal(size=len(draws))
        sample_sd = sigma * np.sqrt(rng.chisquare(927, size=len(draws)) / 927)
        upper = sample_mean + sample_sd * (t.ppf(.975, 927) / np.sqrt(928) + norm.ppf(.95) * np.sqrt(927 / chi2.ppf(.025, 927)))
        self.assertAlmostEqual(float(power[0]), float(np.mean(upper <= 0)), delta=.006)

    def test_blocked_kg_matches_dense_gaussian_conditioning(self):
        rng = np.random.default_rng(105071)
        n = 3
        A = rng.normal(size=(2*n, 5)) * .08
        pair_cov = A @ A.T + np.eye(2*n) * .01
        B = rng.normal(size=(n, 2)) * .1
        f_cov = B @ B.T + np.eye(n) * .01
        f_mean, pair_mean = np.array([0., -.3, .2]), np.column_stack([[-.45, -.35, -.4], [-3., -3.1, -2.9]])
        noise = np.tile([.0012, .004, .2], (n, 1))
        actual = joint_verification_kg(f_mean, f_cov, pair_mean, pair_cov.reshape(n, 2, n, 2), noise, self.protocol)
        # Build one joint 3K-variable Gaussian and condition by the standard
        # multivariate-normal formula, independently of the optimized layout.
        mean = np.r_[f_mean, pair_mean.reshape(-1)]
        C = np.zeros((3*n, 3*n))
        C[:n, :n], C[n:, n:] = f_cov, pair_cov
        point_cov = np.array([pair_cov[2*i:2*i+2, 2*i:2*i+2] for i in range(n)])
        before, _ = joint_verification_utility(f_mean, np.diag(f_cov), pair_mean, point_cov, self.protocol)
        incumbent = int(np.argmax(before))
        nodes, weights = hermgauss(3)
        expected = []
        for action in range(n):
            observed = [action, n + 2*action, n + 2*action + 1]
            S = C[np.ix_(observed, observed)] + np.diag(noise[action])
            gain = C[:, observed] @ np.linalg.inv(S)
            after_cov = C - gain @ C[observed, :]
            after_point_cov = np.array([after_cov[n+2*i:n+2*i+2, n+2*i:n+2*i+2] for i in range(n)])
            value = 0.
            for node_ids in itertools.product(range(3), repeat=3):
                z = np.sqrt(2) * nodes[list(node_ids)]
                after = mean + gain @ np.linalg.cholesky(S) @ z
                utility, _ = joint_verification_utility(after[:n], np.diag(after_cov)[:n], after[n:].reshape(n, 2), after_point_cov, self.protocol)
                value += np.prod(weights[list(node_ids)] / np.sqrt(np.pi)) * (utility.max() - utility[incumbent])
            expected.append(value)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
        self.assertTrue(np.all(actual >= 0.))

    def test_one_candidate_has_exactly_zero_information_value(self):
        pair = np.array([[-.4, -3.]])
        covariance = np.array([[.02, -.02], [-.02, .1]]).reshape(1, 2, 1, 2)
        result = joint_verification_kg(np.array([0.]), np.array([[.03]]), pair, covariance,
                                      np.array([[.0012, .003, .2]]), self.protocol)
        np.testing.assert_array_equal(result, [0.])

    def test_budget_rule_stops_when_sampling_only_reduces_verifier_power(self):
        protocol = {**self.protocol, "replicates": 12}
        pair = np.array([[-.8, -2.]])
        scores = budgeted_joint_kg(np.array([0.]), np.zeros((1, 1)), pair,
            np.zeros((1, 2, 1, 2)), np.array([[.0012, .003, .2]]), protocol, 100)
        self.assertLess(scores.max(), 0.)

    def test_budget_rule_continues_for_learnable_objective_ranking(self):
        protocol = {**self.protocol, "replicates": 12}
        pair = np.tile([-2., -2.], (2, 1))
        scores = budgeted_joint_kg(np.zeros(2), np.eye(2) * .4, pair,
            np.zeros((2, 2, 2, 2)), np.tile([.0012, .003, .2], (2, 1)), protocol, 100)
        self.assertGreater(scores.max(), .01)


if __name__ == "__main__":
    unittest.main()
