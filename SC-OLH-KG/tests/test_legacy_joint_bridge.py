import itertools
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import digamma
from scipy.stats import chi2, nct, norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.legacy_joint_kg import legacy_joint_utility, legacy_raw_joint_kg
from representation.legacy_function_prior import legacy_source_labels


class LegacyJointBridgeTests(unittest.TestCase):
    def setUp(self):
        self.protocol = {"objective_reference": 1.5, "verification_replicates": 80,
                         "chance_alpha": .05, "candidate_delta": .05 / 3}

    def test_source_columns_are_scalarized_f_and_g_and_use_observed_variance(self):
        values = np.array([[.5, -.3], [.6, -.5], [.4, -.4]])
        class ObservedTask:
            name = "source"
            X = np.array([[.1, .3, .5, .7]])
            Y_replicates = (values,)
            tau = -.1
            @property
            def constraint_sigma(self):
                raise AssertionError("oracle/plug-in sigma field must not be a training label")
        task = legacy_source_labels(SimpleNamespace(tasks=[ObservedTask()]))[0]
        np.testing.assert_allclose(task["response"][0, :2], [.5, -.3])
        self.assertAlmostEqual(task["response"][0, 2], np.log(.01) - digamma(1))
        self.assertAlmostEqual(task["response"][0, 3], np.log(.01) - digamma(1))
        np.testing.assert_allclose(task["noise"][0, :2], [.01 / 3, .01 / 3])
        self.assertEqual(task["psi"].shape, (1, 10))

    def test_primary_power_matches_actual_noncentral_t_certificate_samples(self):
        mean, cov = np.array([[-.23, -5.]]), np.array([[[.002, -.008], [-.008, .12]]])
        _, probability = legacy_joint_utility(np.array([.5]), np.zeros(1), mean, cov, self.protocol)
        rng = np.random.default_rng(105109)
        draws = rng.multivariate_normal(mean[0], cov[0], 200000)
        sigma = np.exp(.5 * draws[:, 1])
        sample_mean = draws[:, 0] + sigma * rng.normal(size=len(draws)) / np.sqrt(80)
        sample_sd = sigma * np.sqrt(rng.chisquare(79, len(draws)) / 79)
        factor = nct.ppf(1 - .05 / 3, 79, norm.ppf(.95) * np.sqrt(80)) / np.sqrt(80)
        self.assertAlmostEqual(float(probability[0]), float(np.mean(sample_mean + factor * sample_sd <= 0)), delta=.006)

    def test_raw_kg_matches_dense_conditioning_observing_only_f_and_g(self):
        rng = np.random.default_rng(105110)
        n = 3
        A = rng.normal(size=(2 * n, 4)) * .08
        pair_cov = A @ A.T + np.eye(2 * n) * .005
        B = rng.normal(size=(n, 2)) * .1
        f_cov = B @ B.T + np.eye(n) * .01
        f_mean = np.array([.4, .5, .6])
        pair_mean = np.column_stack([[-.4, -.35, -.45], [-3.5, -3.8, -3.2]])
        noise = np.tile([.002, .03], (n, 1))
        actual = legacy_raw_joint_kg(f_mean, f_cov, pair_mean, pair_cov.reshape(n, 2, n, 2), noise, self.protocol)
        mean = np.r_[f_mean, pair_mean.reshape(-1)]
        covariance = np.zeros((3 * n, 3 * n))
        covariance[:n, :n], covariance[n:, n:] = f_cov, pair_cov
        point_cov = np.array([pair_cov[2*i:2*i+2, 2*i:2*i+2] for i in range(n)])
        incumbent = int(legacy_joint_utility(f_mean, np.diag(f_cov), pair_mean, point_cov, self.protocol)[0].argmax())
        nodes, weights = hermgauss(3)
        expected = []
        for action in range(n):
            observed = [action, n + 2 * action]  # eta is NOT observed
            S = covariance[np.ix_(observed, observed)] + np.diag(noise[action])
            gain = covariance[:, observed] @ np.linalg.inv(S)
            future_cov = covariance - gain @ covariance[observed, :]
            future_point = np.array([future_cov[n+2*i:n+2*i+2, n+2*i:n+2*i+2] for i in range(n)])
            score = 0.
            for a, b in itertools.product(range(3), repeat=2):
                future = mean + gain @ np.linalg.cholesky(S) @ (np.sqrt(2) * nodes[[a, b]])
                utility, _ = legacy_joint_utility(future[:n], np.diag(future_cov)[:n],
                                                  future[n:].reshape(n, 2), future_point, self.protocol)
                score += weights[a] * weights[b] / np.pi * (utility.max() - utility[incumbent])
            expected.append(score)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
