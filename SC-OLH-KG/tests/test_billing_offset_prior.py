import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from problems.inventory_cost_plan import InventoryPlan
from representation.billing_offset_prior import fit_billing_offset_prior, link_billing_coefficients, pooled_linked_prior
from representation.inventory_mean_prior import InventoryFunctionBasis
from representation.orthogonal_inventory_prior import OrthogonalInventoryBasis, embed_inventory_prior
from representation.transferable_spectral import SourceDomainBatch


class BillingOffsetPriorTests(unittest.TestCase):
    def test_linkage_preserves_raw_price_function_law_on_actual_public_geometry(self):
        rng = np.random.default_rng(10051)
        reference = InventoryPlan().coordinates_many(rng.uniform(.05, .95, (96, 4)))
        held = InventoryPlan(initial_stock=5., cost_budget=1.3).coordinates_many(rng.uniform(.05, .95, (24, 4)))
        basis = InventoryFunctionBasis(reference, InventoryFunctionBasis.billing_indices)
        A = rng.normal(size=(4, 4))
        covariance = A @ A.T
        prior = link_billing_coefficients(basis, covariance, 0., {}, offset=-1.3)
        embedded = embed_inventory_prior(prior, OrthogonalInventoryBasis(reference), reference)
        X = np.column_stack([np.ones(len(held)), embedded.basis.transform(held)])
        raw = held[:, [3, 5, 4, 6]] / basis.scale
        np.testing.assert_allclose(X @ embedded.mean, -1.3, atol=1e-10)
        np.testing.assert_allclose(X @ embedded.covariance @ X.T, raw @ covariance @ raw.T, atol=1e-7)
        null = np.r_[1., -basis.mean / basis.scale]
        np.testing.assert_allclose(null @ prior.covariance, 0., atol=1e-12)
        np.testing.assert_array_equal(prior.mean[1:], 0)
        model = prior.target_model(held)
        for i in range(8):
            model.update([i], -.5 + i * .01, .02)
        self.assertAlmostEqual(float(null @ model.a[:5]), -1.3, places=8)
        np.testing.assert_allclose(null @ model.C[:5, :5], 0., atol=1e-9)
        pooled = pooled_linked_prior(prior)
        self.assertAlmostEqual(float(np.trace(pooled.covariance[1:, 1:])), float(np.trace(prior.covariance[1:, 1:])))
        np.testing.assert_allclose(null @ pooled.covariance, 0., atol=1e-12)

    def test_source_fitting_removes_varying_public_offsets(self):
        rng = np.random.default_rng(10052)
        psi = rng.uniform(.05, .9, (16, 8))
        offset = np.tile([-1., -1.2, -.8, -1.4], 4)
        batches = [SourceDomainBatch(str(i), psi, psi[:, [3, 5, 4, 6]] @ rng.uniform(.3, 1., 4))
                   for i in range(4)]
        shifted = [SourceDomainBatch(b.domain, psi, b.signals + offset) for b in batches]
        noise = [np.full(16, .003)] * 4
        adapt, test = list(range(0, 16, 2)), list(range(1, 16, 2))
        centered = fit_billing_offset_prior(batches, noise, psi, np.zeros(16), adapt, test)
        normalized = fit_billing_offset_prior(shifted, noise, psi, offset, adapt, test)
        np.testing.assert_allclose(normalized.covariance, centered.covariance, atol=1e-10)
        np.testing.assert_array_equal(normalized.mean, np.zeros(5))
        self.assertEqual(normalized.discrepancy_variance, centered.discrepancy_variance)

    def test_public_offset_does_not_depend_on_prices(self):
        np.testing.assert_array_equal(InventoryPlan(cost_budget=1.3).mean_offsets, [-5., -1.3])
        np.testing.assert_array_equal(InventoryPlan(cost_budget=.8).mean_offsets, [-5., -.8])


if __name__ == "__main__":
    unittest.main()
