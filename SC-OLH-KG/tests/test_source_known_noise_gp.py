import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from representation.source_known_noise_gp import fit_known_noise_gp


class KnownNoiseGPTests(unittest.TestCase):
    def test_integrated_common_mean_likelihood_gradient(self):
        rng = np.random.default_rng(80821)
        x = rng.normal(size=(10, 2))
        tasks = [{'features': x, 'response': (-5.+np.sin(x[:, 0])+.2*i*x[:, 1])[:, None],
                  'noise': np.linspace(.05, .1, len(x))[:, None]} for i in range(3)]
        def checked_minimize(likelihood, initial, **kwargs):
            _, analytical = likelihood(initial)
            numerical = []
            for i in range(len(initial)):
                plus, minus = initial.copy(), initial.copy()
                plus[i] += 1e-6; minus[i] -= 1e-6
                numerical.append((likelihood(plus)[0]-likelihood(minus)[0])/2e-6)
            np.testing.assert_allclose(analytical, numerical, atol=2e-7, rtol=2e-5)
            return minimize(likelihood, initial, **kwargs)
        with patch('representation.source_known_noise_gp.minimize', side_effect=checked_minimize):
            fit_known_noise_gp(tasks, 0)

    def test_known_label_error_is_not_recovered_as_function_variance(self):
        x = np.linspace(-1., 1., 12)[:, None]
        y = .3*np.sin(x[:, 0])
        def tasks(noise):
            return [{'features': x, 'response': y[:, None], 'noise': np.full((12, 1), noise)} for _ in range(3)]
        uncertain = fit_known_noise_gp(tasks(1.), 0)
        precise = fit_known_noise_gp(tasks(.0001), 0)
        self.assertTrue(uncertain['optimizer_converged'])
        self.assertTrue(precise['optimizer_converged'])
        self.assertLess(uncertain['signal_variance'], 1e-8)
        self.assertLess(uncertain['discrepancy_variance'], 1e-8)
        self.assertGreater(precise['signal_variance'], .1)
        variance = uncertain['constant_variance']*uncertain['normalization']['scale']**2
        self.assertAlmostEqual(variance, 1/36, places=10)


if __name__ == '__main__':
    unittest.main()
