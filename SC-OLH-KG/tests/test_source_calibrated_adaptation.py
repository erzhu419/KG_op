"""Check prior-odds updates and source/target calibration separation."""
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.source_calibrated_adaptation import (prior_adjusted_log_score,
    fit_source_calibration, calibrate_existing_paths)


class SourceCalibrationTests(unittest.TestCase):
    def test_probability_endpoints_and_original_half_prior(self):
        source, adapted, null = -.4, -1.2, .7
        self.assertAlmostEqual(prior_adjusted_log_score(source, adapted, null, 0.), source)
        self.assertAlmostEqual(prior_adjusted_log_score(source, adapted, null, 1.), adapted)
        expected = np.log(null * np.exp(source) + (1. - null) * np.exp(adapted))
        self.assertAlmostEqual(prior_adjusted_log_score(source, adapted, null, .5), expected)

    def test_formula_matches_direct_reweighting_of_existing_branches(self):
        weights = np.log([.13, .25, .32, .30])
        scores = np.array([-.1, -1.3, -.7, -.9])
        null = np.array([True, False, False, True])
        p0 = float(np.exp(weights[null]).sum())
        source = float(logsumexp(weights[null] + scores[null]) - logsumexp(weights[null]))
        adapted = float(logsumexp(weights[~null] + scores[~null]) - logsumexp(weights[~null]))
        for q in (.1, .4, .9):
            changed = weights + np.where(null, np.log1p(-q), np.log(q))
            changed -= logsumexp(changed)
            self.assertAlmostEqual(prior_adjusted_log_score(source, adapted, p0, q),
                                   float(logsumexp(changed + scores)), places=12)

    def test_identical_predictors_preserve_uniform_calibration_density(self):
        forecasts = [{"source": -.6, "adapted": -.6, "null_posterior": p} for p in (.2, .5, .8)]
        fit = fit_source_calibration(forecasts)
        self.assertAlmostEqual(fit["q_mean"], .5, places=12)
        self.assertAlmostEqual(fit["q_sd"], np.sqrt(1. / 12.), places=12)

    def test_target_scores_cannot_change_source_fitted_probability(self):
        source = [{"seed": 48, "archive_group": "shared", "arm": "risk_ard_mixture",
            "forecasts": [{"source": -.2, "adapted": -1., "null_posterior": .6} for _ in range(16)]}
            for _ in range(8)]
        targets = [{"seed": 48, "regime": "shared_structure", "arm": "risk_ard_mixture",
            "forecasts": [{"step": 9, "candidate": 7, "source": -.4, "adapted": -.8,
                           "a9_null_posterior": .7, "a9": -.5}]}]
        fits_a, rows_a = calibrate_existing_paths(source, targets)
        changed = [{**targets[0], "forecasts": [{**targets[0]["forecasts"][0], "source": -3., "adapted": -.2}]}]
        fits_b, rows_b = calibrate_existing_paths(source, changed)
        self.assertEqual(fits_a, fits_b)
        self.assertLess(fits_a[0]["q_mean"], .5)
        self.assertNotEqual(rows_a[0]["mean_calibrated_minus_source"], rows_b[0]["mean_calibrated_minus_source"])


if __name__ == "__main__":
    unittest.main()
