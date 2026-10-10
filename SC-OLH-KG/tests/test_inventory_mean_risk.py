import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import chi2, multivariate_normal, norm, t

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.verification_kg import particle_pass_probability
from problems.inventory_cost_plan import InventoryPlan
from representation.inventory_mean_prior import InventoryFunctionBasis
from representation.joint_mean_risk_prior import JointMeanRiskPosterior
from representation.physical_risk_prior import PhysicalRiskPrior, RiskBatchData
from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior
from representation.transfer_mixture import TransferMixtureGPR


class InventoryMeanRiskTests(unittest.TestCase):
    def setUp(self):
        coefficients = np.zeros((2, 9))
        coefficients[:, 0] = [.04, .16]
        self.risk = PhysicalRiskPrior(coefficients, np.array([.4, .6]), np.array([0, 1]), ["a", "b"], {})
        self.mean = np.array([.1, .3])
        self.covariance = np.array([[.1, .03], [.03, .2]])
        self.features = np.array([[1., 0.], [1., 1.]])
        X = np.zeros((2, 9))
        X[:, 0] = 1.
        self.batch = RiskBatchData(X, np.array([.05, .12]), 12)
        self.labels = np.array([.25, .4])

    def test_public_billing_quantities_and_mixture_preserve_existing_dictionary(self):
        policies = np.random.default_rng(105060).uniform(size=(16, 4))
        plan = InventoryPlan()
        psi = plan.coordinates_many(policies)
        augmented = InventoryFunctionBasis(psi)
        np.testing.assert_allclose(augmented.transform(psi)[:, :68], FixedFunctionBasis(psi).transform(psi), atol=1e-14)
        billing = InventoryFunctionBasis(psi, InventoryFunctionBasis.billing_indices)
        recovered = billing.transform(psi) * billing.scale + billing.mean
        np.testing.assert_allclose(recovered, [plan.trajectory(x).cost_features for x in policies], atol=1e-14)
        hard = SourceFunctionPrior(billing, np.arange(5) * .1, np.eye(5) * .2, .02, {})
        full = SourceFunctionPrior(augmented, np.zeros(73), np.eye(73) * .1, .02, {})
        mixture = TransferMixtureGPR([hard, full], psi)
        direct = [p.target_model(psi) for p in (hard, full)]
        indices = np.arange(len(psi))[:, None]
        expected_mean = sum(m.posterior_mean_many(indices) for m in direct) / 2
        expected_variance = sum(m.posterior_var_many(indices) + (m.posterior_mean_many(indices) - expected_mean) ** 2 for m in direct) / 2
        np.testing.assert_allclose(mixture.posterior_mean_many(indices), expected_mean, atol=1e-13)
        np.testing.assert_allclose(mixture.posterior_var_many(indices), expected_variance, atol=1e-13)

    def test_joint_evidence_and_coefficients_match_batch_normal_chi_square_reference(self):
        model = JointMeanRiskPosterior(self.mean, self.covariance, self.risk)
        evidence = model.update_batch(self.features, self.labels, self.batch)
        log_joint, expected_mean, expected_covariance = [], [], []
        for prior_weight, variance in zip(self.risk.weights, self.risk.coefficients[:, 0]):
            V = self.features @ self.covariance @ self.features.T + np.eye(2) * variance / 12
            logp = multivariate_normal.logpdf(self.labels, self.features @ self.mean, V)
            z = 11 * self.batch.sample_variance / variance
            logp += float(np.sum(chi2.logpdf(z, 11) + np.log(z)))
            log_joint.append(np.log(prior_weight) + logp)
            gain = self.covariance @ self.features.T @ np.linalg.inv(V)
            expected_mean.append(self.mean + gain @ (self.labels - self.features @ self.mean))
            expected_covariance.append(self.covariance - gain @ self.features @ self.covariance)
        self.assertAlmostEqual(evidence, float(logsumexp(log_joint)), places=12)
        np.testing.assert_allclose(np.exp(model.log_weights), np.exp(log_joint - logsumexp(log_joint)), atol=1e-13)
        np.testing.assert_allclose(model.coefficient_mean, expected_mean, atol=1e-13)
        np.testing.assert_allclose(model.coefficient_covariance, expected_covariance, atol=1e-13)
        self.assertEqual(model.paid_episodes, 24)
        sequential = JointMeanRiskPosterior(self.mean, self.covariance, self.risk)
        sequence_evidence = sum(sequential.update_batch(self.features[i:i+1], self.labels[i:i+1],
            RiskBatchData(self.batch.X[i:i+1], self.batch.sample_variance[i:i+1], 12)) for i in range(2))
        self.assertAlmostEqual(sequence_evidence, evidence, places=12)
        np.testing.assert_allclose(sequential.coefficient_mean, model.coefficient_mean, atol=1e-13)
        np.testing.assert_array_equal(self.risk.weights, [.4, .6])

    def test_full_covariance_keeps_mean_risk_dependence_and_matches_independent_draws(self):
        model = JointMeanRiskPosterior(self.mean, self.covariance, self.risk)
        model.update_batch(self.features, self.labels, self.batch)
        mean, covariance = model.predict(self.features, self.batch.X, full_cov=True)
        _, diagonal = model.predict(self.features, self.batch.X)
        np.testing.assert_allclose(np.diag(covariance.reshape(4, 4)).reshape(2, 2), diagonal, atol=1e-13)
        self.assertGreater(abs(covariance[0, 0, 0, 1]), 1e-5)
        self.assertGreaterEqual(np.linalg.eigvalsh(covariance.reshape(4, 4)).min(), -1e-12)
        rng = np.random.default_rng(105061)
        groups = rng.choice(2, size=100000, p=np.exp(model.log_weights))
        beta = np.empty((len(groups), 2))
        for group in range(2):
            mask = groups == group
            beta[mask] = rng.multivariate_normal(model.coefficient_mean[group], model.coefficient_covariance[group], mask.sum())
        samples = np.empty((len(groups), 2, 2))
        samples[:, :, 0] = beta @ self.features.T
        samples[:, :, 1] = np.log(self.risk.coefficients[groups, 0])[:, None]
        np.testing.assert_allclose(samples.mean(axis=0), mean, atol=.004)
        np.testing.assert_allclose(np.cov(samples.reshape(len(groups), 4), rowvar=False), covariance.reshape(4, 4), atol=.005)

    def test_conditional_particle_power_matches_joint_independent_verifier_draws(self):
        protocol = {"verification_replicates": 928, "chance_alpha": .05}
        means, variances, risks, weights = np.array([[-.15, -.7]]), np.array([[.003, .01]]), np.array([[.01, .2]]), np.array([.4, .6])
        predicted = particle_pass_probability(means, variances, risks, weights, protocol)[0]
        rng = np.random.default_rng(105062)
        group = rng.choice(2, size=100000, p=weights)
        sigma = np.sqrt(risks[0, group])
        g = means[0, group] + np.sqrt(variances[0, group]) * rng.normal(size=len(group))
        sample_mean = g + sigma / np.sqrt(928) * rng.normal(size=len(group))
        sample_sd = sigma * np.sqrt(rng.chisquare(927, size=len(group)) / 927)
        upper = sample_mean + sample_sd * (t.ppf(.975, 927) / np.sqrt(928) + norm.ppf(.95) * np.sqrt(927 / chi2.ppf(.025, 927)))
        self.assertAlmostEqual(float(predicted), float(np.mean(upper <= 0)), delta=.005)


if __name__ == "__main__":
    unittest.main()
