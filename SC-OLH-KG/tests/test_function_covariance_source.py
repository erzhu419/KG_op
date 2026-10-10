import unittest

import numpy as np

from performance.function_covariance_source import node_grid, outer_priors, predictive_scores, selected_function_forecasts
from performance.source_calibrated_adaptation import fit_source_calibration
from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior


class FunctionCovarianceSourceTests(unittest.TestCase):
    def test_macro_reference_preserves_uniform_q_and_simplex(self):
        masses, log_weights, theta, q = node_grid(8)
        weights = np.exp(log_weights)
        np.testing.assert_allclose(masses.sum(axis=1), 1., atol=1e-14)
        np.testing.assert_allclose(weights @ masses, [1. / 6.] * 3 + [.5], atol=1e-14)
        np.testing.assert_allclose(weights @ theta, [1. / 3.] * 3, atol=1e-14)
        self.assertAlmostEqual(float(weights @ ((q - .5) ** 2)), 1. / 12., places=13)

    def test_pure_branches_reproduce_densities(self):
        evidence = np.array([[0., 1., -2., .6], [0., -.4, .8, -1.]])
        scores = np.array([[-.3, -.7, -.9, -1.], [-1.8, -.7, -.2, -.5]])
        for i in range(4):
            np.testing.assert_allclose(predictive_scores(evidence, scores, np.eye(4)[i]), scores[:, i], atol=1e-14)

    def test_issued_prior_excludes_held_seed_scores(self):
        grid = node_grid(4)
        scores = {s: np.sin(np.arange(len(grid[0])) + s) for s in (48, 49, 50)}
        old = outer_priors(scores, grid)[48]
        scores[48] += np.arange(len(grid[0])) * 10.
        new = outer_priors(scores, grid)[48]
        self.assertEqual(old, new)
        self.assertEqual(new["training_seeds"], [49, 50])

    def test_generic_duplicate_branches_reduce_to_same_data_a12(self):
        grid = node_grid(16)
        original = [dict(source=-.2 - i / 50., adapted=-.7 + i / 80., null_posterior=.4 + i / 100.) for i in range(16)]
        evidence = np.array([[0., 0., 0., np.log1p(-f["null_posterior"]) - np.log(f["null_posterior"])] for f in original])
        scores = np.array([[f["source"]] * 3 + [f["adapted"]] for f in original])
        nodes = predictive_scores(evidence, scores, grid[0]).sum(axis=0)
        fit = outer_priors({48: nodes * 7., 49: nodes}, grid)[48]
        expected = fit_source_calibration(original, 16)
        self.assertAlmostEqual(fit["q_mean"], expected["q_mean"], places=13)
        np.testing.assert_allclose(fit["theta_mean"], [1. / 3.] * 3, atol=1e-13)
        np.testing.assert_allclose(fit["theta_sd"], [np.sqrt(1. / 18.)] * 3, atol=1e-13)

    def test_selected_branch_weights_precede_current_held_label(self):
        psi = np.random.default_rng(3).uniform(-2., 2., (24, 8))
        basis = FixedFunctionBasis(psi, [0, 1])
        prior = SourceFunctionPrior(basis, np.zeros(3), np.diag([.25, .1, .2]), .02, {})
        task = {"psi": psi, "response": np.zeros((24, 3)), "noise": np.full((24, 3), .2)}
        original = selected_function_forecasts([prior], task, [0], 4)
        task["response"][8, 2] = 4.
        changed = selected_function_forecasts([prior], task, [0], 4)
        self.assertEqual(original[0]["relative_log_evidence"], changed[0]["relative_log_evidence"])
        self.assertNotEqual(original[0]["adapted"], changed[0]["adapted"])
        self.assertNotEqual(original[1]["relative_log_evidence"], changed[1]["relative_log_evidence"])


if __name__ == "__main__":
    unittest.main()
