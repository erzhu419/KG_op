import unittest

import numpy as np

from performance.function_covariance_attribution import decompose


class FunctionCovarianceAttributionTests(unittest.TestCase):
    def setUp(self):
        self.evidence = np.array([[0., .5, -.8, .2], [0., -.6, .9, -1.]])
        self.predictive = np.array([[-.3, -.7, -.1, -.5], [-.4, -.2, -.9, -.1]])
        self.prior = np.array([.5, .15, .25, .1])
        self.baseline = np.array([-.2, -.6])

    def test_exact_two_part_and_two_head_accounting(self):
        result = decompose(self.evidence, self.predictive, self.prior, self.baseline)
        np.testing.assert_allclose(result["fg_effect"] + result["q_effect"], result["log_score"] - self.baseline, atol=1e-14)
        np.testing.assert_allclose(result["f_accounting_effect"] + result["g_accounting_effect"], result["fg_effect"], atol=1e-14)
        manual_weights = self.prior * np.exp(self.evidence)
        manual_weights /= manual_weights.sum(axis=1)[:, None]
        np.testing.assert_allclose(result["posterior_macro_mass"], manual_weights, atol=1e-14)

    def test_future_density_changes_do_not_change_issued_weights(self):
        old = decompose(self.evidence, self.predictive, self.prior, self.baseline)
        changed = decompose(self.evidence, self.predictive + np.array([0., 3., -2., .4]), self.prior, self.baseline)
        np.testing.assert_array_equal(old["posterior_macro_mass"], changed["posterior_macro_mass"])
        self.assertFalse(np.allclose(old["fg_effect"], changed["fg_effect"]))

    def test_duplicate_branches_have_zero_fg_effect(self):
        evidence = self.evidence.copy()
        predictive = self.predictive.copy()
        evidence[:, 1:3] = evidence[:, 0:1]
        predictive[:, 1:3] = predictive[:, 0:1]
        result = decompose(evidence, predictive, self.prior, self.baseline)
        for name in ("fg_effect", "f_accounting_effect", "g_accounting_effect"):
            np.testing.assert_allclose(result[name], 0., atol=1e-14)


if __name__ == "__main__":
    unittest.main()
