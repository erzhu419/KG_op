import unittest

import numpy as np

from performance.function_covariance_source import predictive_scores
from performance.scale_preserving_source_covariance import selected_scale_forecasts
from performance.source_cross_archive_transfer import fold_forecasts
from performance.source_holdout_calibration import forecast_held_source
from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior


class SourceCrossArchiveTransferTests(unittest.TestCase):
    def setUp(self):
        source_reference = np.random.default_rng(23).uniform(-2., 2., (32, 8))
        # Scored archive has different coordinates; its labels do not fit the basis.
        psi = np.random.default_rng(24).uniform(-2., 2., (24, 8))
        basis = FixedFunctionBasis(source_reference, [0, 19, 67])
        self.priors = [SourceFunctionPrior(basis, np.r_[mean, np.zeros(3)],
            np.diag([.25, .08, .01, .02]), discrepancy, {})
            for mean, discrepancy in ((-2.3, .02), (-2.6, .06))]
        self.task = {"psi": psi, "response": np.random.default_rng(25).normal(-2.4, .3, (24, 3)),
                     "noise": np.full((24, 3), .2)}
        self.selected = {"f": [0, 1], "g": [2]}
        self.families = {"a": [0], "b": [1], "c": [2]}

    def test_two_arms_and_r_evidence_match_separate_predictors(self):
        result = fold_forecasts(self.priors, self.task, self.selected, self.families, (6,))
        for arm, priors in (("risk_ard_mixture", self.priors), ("fixed_source", self.priors[1:])):
            expected = forecast_held_source(priors, self.task, self.families, 6)
            actual = result[arm]["6"]
            for e, p, old in zip(actual["relative_log_evidence"], actual["component_log_density"], expected):
                self.assertAlmostEqual(p[0], old["source"], places=11)
                self.assertAlmostEqual(p[3], old["adapted"], places=11)
                self.assertAlmostEqual(e[3], np.log1p(-old["null_posterior"]) - np.log(old["null_posterior"]), places=11)
        generic = result["fixed_source"]["6"]
        e, p = np.array(generic["relative_log_evidence"]), np.array(generic["component_log_density"])
        np.testing.assert_array_equal(p[:, 0], p[:, 1])
        np.testing.assert_array_equal(p[:, 0], p[:, 2])
        np.testing.assert_array_equal(e[:, :3], np.zeros((16, 3)))
        for i in range(4):
            np.testing.assert_allclose(predictive_scores(e, p, np.eye(4)[i]), p[:, i], atol=1e-14)

    def test_f_g_aggregation_matches_independent_selected_predictors(self):
        actual = fold_forecasts(self.priors, self.task, self.selected, self.families, (6,))["risk_ard_mixture"]["6"]
        for index, head in enumerate(("f", "g"), 1):
            expected = selected_scale_forecasts(self.priors, self.task, self.selected[head], 6)
            np.testing.assert_allclose(np.array(actual["component_log_density"])[:, index], [f["adapted"] for f in expected], atol=1e-11)
            np.testing.assert_allclose(np.array(actual["relative_log_evidence"])[:, index], [f["relative_log_evidence"] for f in expected], atol=1e-11)

    def test_current_label_does_not_change_issued_branch_odds(self):
        first = fold_forecasts(self.priors, self.task, self.selected, self.families, (6,))
        self.task["response"][8, 2] += 4.
        second = fold_forecasts(self.priors, self.task, self.selected, self.families, (6,))
        for arm in first:
            self.assertEqual(first[arm]["6"]["relative_log_evidence"][0], second[arm]["6"]["relative_log_evidence"][0])
            self.assertNotEqual(first[arm]["6"]["component_log_density"][0], second[arm]["6"]["component_log_density"][0])
            self.assertNotEqual(first[arm]["6"]["relative_log_evidence"][1], second[arm]["6"]["relative_log_evidence"][1])


if __name__ == "__main__":
    unittest.main()
