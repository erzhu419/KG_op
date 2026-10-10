import unittest

import numpy as np

from performance.diagnose_legacy_counterbalanced_mse_uncertainty import paired_interval


class PairedMseMeasurementTests(unittest.TestCase):
    def test_independent_three_repeat_sampling_matches_linear_noise_variance(self):
        rng = np.random.default_rng(853)
        truth = np.array([.12, -.02, .31, .05])
        new = np.array([.08, .02, .24, .07]); old = np.array([.13, -.06, .27, .01])
        episode_variance = np.array([.008, .014, .003, .020])
        draws = truth+rng.normal(size=(30000, 3, 4))*np.sqrt(episode_variance)
        means = draws.mean(axis=1)
        paired = np.mean((new-means)**2-(old-means)**2, axis=1)
        result, _ = paired_interval(new, old, truth, episode_variance/3, 3)
        self.assertAlmostEqual(paired.var()/result['measurement_standard_error']**2, 1., delta=.03)
        self.assertAlmostEqual(paired.mean(), result['paired_mse_difference'], delta=1e-4)


if __name__ == '__main__':
    unittest.main()
