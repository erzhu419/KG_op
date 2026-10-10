import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.cumulative_risk import cumulative_feature_vector
from problems.inventory_cost_plan import InventoryPlan
from representation.physical_risk_prior import (PhysicalRiskPrior, RiskBatchData,
    fit_source_physical_risk, fit_source_risk_families, physical_particle_bank, variance_log_likelihood)


class PhysicalRiskPriorTests(unittest.TestCase):
    def test_chi_square_density_includes_log_label_jacobian(self):
        coefficients = np.array([[.2], [.8]])
        task = RiskBatchData(np.ones((2, 1)), np.array([.1, .7]), 12)
        expected = []
        for variance in coefficients[:, 0]:
            u = 11 * task.sample_variance / variance
            expected.append(np.sum(chi2.logpdf(u, 11) + np.log(u)))
        np.testing.assert_allclose(variance_log_likelihood(task, coefficients), expected, atol=1e-12)

    def test_exact_update_matches_bayes_and_keeps_source_prior_frozen(self):
        prior = PhysicalRiskPrior(np.array([[.2], [.8]]), np.array([.6, .4]),
                                  np.array([0, 1]), ["a", "b"], {})
        posterior = prior.target_model()
        task = RiskBatchData(np.ones((1, 1)), np.array([.7]), 12)
        expected = np.log(prior.weights) + variance_log_likelihood(task, prior.coefficients)
        expected -= logsumexp(expected)
        posterior.update_batch(task)
        np.testing.assert_allclose(posterior.log_weights, expected)
        np.testing.assert_array_equal(prior.weights, [.6, .4])
        self.assertEqual(posterior.diagnostics()["paid_episodes"], 12)
        mean, covariance = posterior.predict(np.ones((3, 1)), full_cov=True)
        np.testing.assert_allclose(covariance, np.full((3, 3), covariance[0, 0]))
        np.testing.assert_allclose(posterior.predict(np.ones((3, 1)))[1], np.diag(covariance))
        self.assertGreater(mean[0], np.log(.2))

    def test_physical_bank_is_positive_and_shared_covariance_is_psd(self):
        plan = InventoryPlan()
        policies = np.random.default_rng(407).uniform(size=(24, 4))
        X = np.array([cumulative_feature_vector(plan.trajectory(x).risk_exposures()) for x in policies])
        coefficients, groups, names = physical_particle_bank(X, .1, bits=5)
        self.assertEqual(len(names), 8)
        self.assertTrue((X @ coefficients.T > 0).all())
        shared = np.array([[[b[4], b[5]], [b[5], b[6]]] for b in coefficients])
        self.assertGreaterEqual(np.linalg.eigvalsh(shared).min(), -1e-10)
        for group in range(8):
            self.assertEqual(int(np.sum(groups == group)), 32)

    def test_source_fit_and_log_label_conversion_preserve_noise_and_mixture_mass(self):
        plan = InventoryPlan()
        policies = np.random.default_rng(408).uniform(size=(16, 4))
        X = np.array([cumulative_feature_vector(plan.trajectory(x).risk_exposures()) for x in policies])
        tasks = [RiskBatchData(X, np.full(len(X), value), 12) for value in (.07, .11, .16)]
        learned, pooled = fit_source_physical_risk(tasks, X, bits=5)
        history = learned.diagnostics["objective_history"]
        self.assertTrue(np.all(np.diff(history) >= -1e-10))
        self.assertAlmostEqual(float(learned.weights.sum()), 1.)
        self.assertTrue(np.all(learned.weights >= pooled.weights / 4 - 1e-14))
        self.assertFalse(np.allclose(learned.weights, pooled.weights))
        from scipy.special import digamma
        eta = np.log(tasks[0].sample_variance) - digamma(5.5) - np.log(2 / 11)
        np.testing.assert_allclose(RiskBatchData.from_log_labels(X, eta, 12).sample_variance,
                                   tasks[0].sample_variance)

    def test_family_learning_preserves_conditional_parameter_uncertainty(self):
        plan = InventoryPlan()
        policies = np.random.default_rng(409).uniform(size=(16, 4))
        X = np.array([cumulative_feature_vector(plan.trajectory(x).risk_exposures()) for x in policies])
        tasks = [RiskBatchData(X, np.full(len(X), value), 12) for value in (.07, .11, .16)]
        prior = fit_source_risk_families(tasks, X, bits=5)
        self.assertAlmostEqual(prior.weights.sum(), 1.)
        self.assertEqual(prior.diagnostics["source_distribution_free_weights"], 7)
        self.assertTrue(np.all(np.diff(prior.diagnostics["objective_history"]) >= -1e-10))
        for group in range(8):
            within = prior.weights[prior.families == group]
            np.testing.assert_array_equal(within, np.full(32, within[0]))
        initial = prior.weights.copy()
        posterior = prior.target_model()
        posterior.update_batch(tasks[0])
        np.testing.assert_array_equal(prior.weights, initial)


if __name__ == "__main__":
    unittest.main()
