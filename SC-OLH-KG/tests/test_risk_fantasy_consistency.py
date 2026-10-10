"""Controls that distinguish mixture conditioning from integration artifacts."""
import copy
import unittest

import numpy as np
from scipy.special import polygamma
from scipy.stats import norm

from core.verification_kg import posterior_covariances, risk_only_kg_scores
from performance.risk_fantasy_consistency import condition_fantasies, probe_scores
from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior
from representation.transfer_mixture import TransferMixtureGPR

PROTOCOL = {"verification_replicates": 928, "chance_alpha": .05, "replicates": 12}


class RiskFantasyControls(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(9845)
        self.psi = rng.uniform(-2, 2, (7, 8))
        basis = FixedFunctionBasis(self.psi, [0, 8, 32])
        priors = [SourceFunctionPrior(basis, np.r_[value, np.zeros(3)],
                  np.eye(4) * scale, .02, {}) for value, scale in [(-2.6, .1), (-1.7, .3)]]
        self.model = TransferMixtureGPR(priors, self.psi)
        self.model.update([1], -2.4, .2)
        self.candidates = np.arange(7)[:, None]
        self.means = np.column_stack([m.posterior_mean_many(self.candidates) for m in self.model.models])
        self.covariance = np.moveaxis(posterior_covariances(self.model.models, self.candidates), -1, 0)
        self.weights = np.exp(self.model.log_weights)
        self.fixed = np.column_stack([np.linspace(-1, 1, 7), np.linspace(-.65, -.2, 7), np.zeros(7)])
        self.variance = np.tile([.03, .02, 0.], (7, 1))
        self.noise = float(polygamma(1, 5.5))

    def test_conditioning_matches_actual_paid_model_update(self):
        # A previously observed candidate and a new candidate exercise both
        # candidate-shared discrepancy and coefficient covariance updates.
        probes = [1, 4]
        labels, _, probabilities, means, variances = condition_fantasies(
            self.means, self.covariance, self.weights, probes, self.noise, 3)
        for s, candidate in enumerate(probes):
            for origin in range(2):
                y = float(labels[s, origin, 2])
                changed = copy.deepcopy(self.model)
                direct = self.weights * norm.pdf(y, self.means[candidate],
                    np.sqrt(self.covariance[:, candidate, candidate] + self.noise))
                np.testing.assert_allclose(probabilities[s, origin, 2], direct / direct.sum(), atol=1e-13)
                changed.update([candidate], y, self.noise)
                np.testing.assert_allclose(np.exp(changed.log_weights), probabilities[s, origin, 2], atol=1e-13)
                for b, component in enumerate(changed.models):
                    np.testing.assert_allclose(component.posterior_mean_many(self.candidates), means[:, s, origin, 2, b], atol=1e-12)
                    np.testing.assert_allclose(component.posterior_var_many(self.candidates), variances[:, s, b], atol=1e-12)

    def test_identical_components_reduce_to_gaussian(self):
        means = np.repeat(self.means[:, :1], 2, axis=1)
        covariance = np.repeat(self.covariance[:1], 2, axis=0)
        result = probe_scores(means, covariance, [.2, .8], [1, 4], self.noise,
                              self.fixed, self.variance, PROTOCOL)
        np.testing.assert_allclose(result["B"], result["F"], atol=1e-13)
        np.testing.assert_allclose(result["B"], result["G"], atol=1e-13)
        self.assertAlmostEqual(result["current_max"], result["G_current_max"], places=13)

    def test_zero_covariance_and_identical_label_law_give_zero_value(self):
        means = self.means.copy()
        means[0] = -2.
        covariance = self.covariance.copy()
        covariance[:, 0] = 0.
        covariance[:, :, 0] = 0.
        result = probe_scores(means, covariance, self.weights, [0], self.noise,
                              self.fixed, self.variance, PROTOCOL)
        for arm in ("B", "F", "G"):
            np.testing.assert_allclose(result[arm], 0., atol=1e-13)

    def test_g3_matches_core_and_does_not_mutate_paid_posterior(self):
        state = copy.deepcopy(self.model)
        arrays = [a.copy() for a in (self.means, self.covariance, self.weights)]
        result = probe_scores(self.means, self.covariance, self.weights,
                              list(range(7)), self.noise, self.fixed, self.variance, PROTOCOL, 3)
        expected = risk_only_kg_scores(self.model, self.candidates, self.fixed, self.variance, PROTOCOL)
        np.testing.assert_allclose(result["G"], expected, atol=1e-12)
        for before, after in zip(arrays, (self.means, self.covariance, self.weights)):
            np.testing.assert_array_equal(before, after)
        np.testing.assert_array_equal(self.model.a, state.a)
        np.testing.assert_array_equal(self.model.C, state.C)
        np.testing.assert_array_equal(self.model.log_weights, state.log_weights)


if __name__ == "__main__":
    unittest.main()
