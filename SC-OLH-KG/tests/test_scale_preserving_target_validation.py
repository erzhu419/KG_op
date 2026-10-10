import unittest

import numpy as np

from performance.scale_preserving_source_covariance import selected_path_forecasts
from performance.target_risk_residual import predictive_group_score
from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior


class ScalePreservingTargetValidationTests(unittest.TestCase):
    def setUp(self):
        self.psi = np.random.default_rng(21).uniform(-2., 2., (32, 8))
        basis = FixedFunctionBasis(self.psi, [0, 19, 67])
        self.priors = [SourceFunctionPrior(basis, np.r_[mean, np.zeros(3)],
            np.diag([.25, .08, .01, .02]), discrepancy, {})
            for mean, discrepancy in ((-2.3, .02), (-2.6, .06))]
        self.ids = np.array([5, 6, 25, 21, 23, 19, 7, 20, 26, 1, 27, 15,
                             12, 22, 30, 4, 15, 11, 26, 1, 9, 16, 14, 31])
        self.labels = np.random.default_rng(31).normal(-2.4, .3, 24)
        self.noise = np.full(24, .2)

    def test_query_identity_and_source_density_with_repeat_visits(self):
        forecasts = selected_path_forecasts(self.priors, self.psi, self.ids,
            self.labels, self.noise, [0, 1], 6)
        self.assertEqual([(f["step"], f["candidate"]) for f in forecasts],
                         [(t + 1, int(self.ids[t])) for t in range(8, 24)])
        for t in (8, 16, 18, 19, 23):
            expected = predictive_group_score(self.priors, self.psi, self.ids[:t],
                self.labels[:t], self.noise[:t], 0., int(self.ids[t]),
                self.labels[t:t + 1], self.noise[t:t + 1])
            self.assertAlmostEqual(forecasts[t - 8]["source"], expected, places=12)

    def test_current_repeat_visit_label_only_affects_future_branch_odds(self):
        first = selected_path_forecasts(self.priors, self.psi, self.ids,
            self.labels, self.noise, [0, 1], 6)
        changed = self.labels.copy()
        changed[16] += 4.
        second = selected_path_forecasts(self.priors, self.psi, self.ids,
            changed, self.noise, [0, 1], 6)
        self.assertEqual(first[8]["relative_log_evidence"], second[8]["relative_log_evidence"])
        self.assertNotEqual(first[8]["adapted"], second[8]["adapted"])
        self.assertNotEqual(first[9]["relative_log_evidence"], second[9]["relative_log_evidence"])


if __name__ == "__main__":
    unittest.main()
