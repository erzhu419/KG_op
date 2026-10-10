import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from representation.inventory_mean_prior import InventoryFunctionBasis, fit_inventory_posterior_group_prior
from representation.orthogonal_inventory_prior import OrthogonalInventoryBasis, embed_inventory_prior
from representation.source_function_prior import SourceFunctionPrior
from representation.transferable_spectral import SourceDomainBatch


class OrthogonalInventoryPriorTests(unittest.TestCase):
    def test_public_charges_remain_in_product_orthogonal_function_class(self):
        class TwoBillingBasis(OrthogonalInventoryBasis):
            billing_axes = {0: "quantity_a", 1: "quantity_b"}
        axis = np.linspace(-1.7, 1.7, 20)
        reference = np.column_stack([axis, axis])
        basis = TwoBillingBasis(reference)
        product = np.array([(a, b) for a in axis for b in axis])
        Z = basis.transform(product)
        np.testing.assert_allclose(Z.T @ Z / len(Z), np.eye(basis.feature_dim), atol=2e-8)
        np.testing.assert_allclose(Z.mean(axis=0), 0, atol=1e-9)
        raw = basis._library(reference)[0]
        # Use separate marginal fits: the correlated reference is not a joint
        # full-rank design and must not determine the coordinate transport.
        for j, (indices, center, transform, _) in enumerate(basis.axes):
            features = (raw[:, indices] - center) @ transform
            centered_quantity = reference[:, j] - reference[:, j].mean()
            coefficients = features.T @ centered_quantity / len(reference)
            np.testing.assert_allclose(features @ coefficients, centered_quantity, atol=1e-9)
            # Raw linear functions have unit derivative energy in this metric.
            np.testing.assert_allclose(coefficients ** 2 @ basis.roughness[j * 6:(j + 1) * 6]
                / np.var(reference[:, j]), 1., atol=1e-8)

    def test_transport_preserves_gaussian_functions_on_unseen_coordinates(self):
        rng = np.random.default_rng(1005)
        reference, heldout = rng.uniform(-2, 2, (100, 8)), rng.uniform(-1.8, 1.8, (24, 8))
        original = InventoryFunctionBasis(reference, [68, 69, 32, 26])
        A = rng.normal(size=(5, 5))
        prior = SourceFunctionPrior(original, np.r_[.3, np.zeros(4)], A @ A.T, .02, {})
        basis = OrthogonalInventoryBasis(reference)
        transported = embed_inventory_prior(prior, basis, reference)
        old_X = np.column_stack([np.ones(len(heldout)), original.transform(heldout)])
        new_X = np.column_stack([np.ones(len(heldout)), basis.transform(heldout)])
        np.testing.assert_allclose(new_X @ transported.mean, old_X @ prior.mean, atol=1e-8)
        np.testing.assert_allclose(new_X @ transported.covariance @ new_X.T,
                                   old_X @ prior.covariance @ old_X.T, atol=2e-7)
        old_model, new_model = prior.target_model(heldout), transported.target_model(heldout)
        for i in range(8):
            for model in (old_model, new_model):
                model.update([i], np.sin(i), .03)
        candidates = np.arange(len(heldout))[:, None]
        np.testing.assert_allclose(new_model.posterior_mean_many(candidates),
                                   old_model.posterior_mean_many(candidates), atol=2e-7)
        np.testing.assert_allclose(new_model.posterior_var_many(candidates),
                                   old_model.posterior_var_many(candidates), atol=2e-7)

    def test_inner_held_source_cannot_choose_greedy_support(self):
        rng = np.random.default_rng(510)
        reference = rng.uniform(.1, 1, (16, 8))
        batches = [SourceDomainBatch(str(i), reference, reference[:, 3] * (i + 1)) for i in range(4)]
        noises = [np.full(16, .01)] * 4
        domains = []
        def selection(training, *args):
            domains.append([b.domain for b in training])
            return [68, 69, 70, 71], []
        with patch("representation.inventory_mean_prior.conditional_selection", selection):
            prior = fit_inventory_posterior_group_prior(batches, noises, reference,
                                                      list(range(0, 16, 2)), list(range(1, 16, 2)))
        self.assertEqual(domains[0], ["0", "1", "2", "3"])
        for held, used in enumerate(domains[1:]):
            self.assertNotIn(str(held), used)
            self.assertEqual(len(used), 3)
        np.testing.assert_array_equal(prior.mean[1:], 0)


if __name__ == "__main__":
    unittest.main()
