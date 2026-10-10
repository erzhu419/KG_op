"""Check local inference against analytic Gaussian and raw-batch likelihoods."""
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.special import digamma, logsumexp, polygamma
from scipy.stats import chi2, norm, t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.local_risk_likelihood import local_update

P = {"replicates": 12, "verification_replicates": 928, "chance_alpha": .05}
Y = np.array([0., -.65, -2.25])
N = np.array([.0012, np.exp(Y[2] + digamma(5.5) - np.log(5.5)) / 12, polygamma(1, 5.5)])


class LocalRiskLikelihoodTests(unittest.TestCase):
    def test_gaussian_plugin_recovers_analytic_update(self):
        g, eta = [(1., -.5, .03)], [(1., -2.5, .18)]
        r = local_update(g, eta, Y, N, P, "gaussian", "sample_variance")
        for h, (mean, variance), prefix in ((1, g[0][1:], "g"), (2, eta[0][1:], "eta")):
            wanted_mean = mean + variance / (variance + N[h]) * (Y[h] - mean)
            wanted_variance = variance * N[h] / (variance + N[h])
            self.assertAlmostEqual(r[prefix + "_mean"], wanted_mean, places=10)
            self.assertAlmostEqual(r[prefix + "_variance"], wanted_variance, places=10)
        self.assertAlmostEqual(r["g_eta_covariance"], 0., places=10)

    def test_joint_chi_matches_raw_gaussian_batch_posterior(self):
        g = [(.65, -.5, .03), (.35, -.72, .06)]
        eta = [(.7, -2.5, .18), (.3, -1.9, .3)]
        r = local_update(g, eta, Y, N, P, "log_chi_square", "latent_variance")
        rng = np.random.default_rng(20261004)
        count = 400000
        gi = rng.choice(2, count, p=[c[0] for c in g])
        ei = rng.choice(2, count, p=[c[0] for c in eta])
        g_draw = np.array([c[1] for c in g])[gi] + np.sqrt(np.array([c[2] for c in g])[gi]) * rng.normal(size=count)
        eta_draw = np.array([c[1] for c in eta])[ei] + np.sqrt(np.array([c[2] for c in eta])[ei]) * rng.normal(size=count)
        sigma2 = np.exp(eta_draw)
        # A raw r-sample Normal likelihood uses residual sum of squares.
        # Its summary is (r-1)S² + r*(batch_mean-g)², independent of our quadrature code.
        residual = 11 * (N[1] * 12) + 12 * (Y[1] - g_draw) ** 2
        log_weights = -6 * eta_draw - residual / (2 * sigma2)
        weights = np.exp(log_weights - logsumexp(log_weights))
        for draw, prefix in ((g_draw, "g"), (eta_draw, "eta")):
            mean = float(weights @ draw)
            variance = float(weights @ (draw - mean) ** 2)
            self.assertAlmostEqual(r[prefix + "_mean"], mean, delta=.003)
            self.assertAlmostEqual(r[prefix + "_variance"], variance, delta=.002)
        covariance = float(weights @ ((g_draw - weights @ g_draw) * (eta_draw - weights @ eta_draw)))
        self.assertAlmostEqual(r["g_eta_covariance"], covariance, delta=.001)
        for labels, name in ((gi, "g_component_weights"), (ei, "eta_component_weights")):
            self.assertAlmostEqual(r[name][0], float(weights[labels == 0].sum()), delta=.004)
        df, reps = 927, 928
        sample_mean = g_draw + np.sqrt(sigma2 / reps) * rng.normal(size=count)
        sample_sd = np.sqrt(sigma2 * rng.chisquare(df, count) / df)
        upper = sample_mean + sample_sd * (t.ppf(.975, df) / np.sqrt(reps) +
                                           norm.ppf(.95) * np.sqrt(df / chi2.ppf(.025, df)))
        self.assertAlmostEqual(r["predicted_verifier_power"], float(weights @ (upper <= 0)), delta=.008)


if __name__ == "__main__":
    unittest.main()
