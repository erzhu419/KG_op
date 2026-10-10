import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import chi2, norm, t

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.verification_kg import particle_pass_probability, predictive_pass_probability


class ParticleVerificationUtilityTests(unittest.TestCase):
    def test_single_particle_matches_existing_known_risk_power(self):
        mean = np.array([[0., -.4, np.log(.06)], [0., -.2, np.log(.03)]])
        variance = np.array([[0., .02, 0.], [0., .01, 0.]])
        for reps in (256, 928):
            protocol = {"verification_replicates": reps, "chance_alpha": .05}
            actual = particle_pass_probability(mean[:, 1], variance[:, 1],
                np.exp(mean[:, 2, None]), np.ones(1), protocol)
            np.testing.assert_allclose(actual, predictive_pass_probability(mean, variance, protocol), atol=1e-13)

    def test_risk_mixture_power_matches_independent_gaussian_verifier_draws(self):
        rng = np.random.default_rng(105044)
        protocol = {"verification_replicates": 928, "chance_alpha": .05}
        risks, weights = np.array([.02, .2]), np.array([.3, .7])
        predicted = particle_pass_probability(np.array([-.5]), np.array([.015]),
            risks[None], weights, protocol)[0]
        draws = 100000
        sigma = np.sqrt(rng.choice(risks, size=draws, p=weights))
        g = -.5 + np.sqrt(.015) * rng.normal(size=draws)
        df = 927
        sample_mean = g + sigma / np.sqrt(928) * rng.normal(size=draws)
        sample_sd = sigma * np.sqrt(rng.chisquare(df, draws) / df)
        upper = sample_mean + sample_sd * (t.ppf(.975, df) / np.sqrt(928)
                + norm.ppf(.95) * np.sqrt(df / chi2.ppf(.025, df)))
        self.assertAlmostEqual(float(predicted), float(np.mean(upper <= 0)), delta=.005)


if __name__ == "__main__":
    unittest.main()
