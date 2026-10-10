import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.verification_kg import verification_kg_scores
from representation.orthogonal_function_prior import (
    OrthogonalLowOrderBasis, fit_orthogonal_function_prior,
    same_source_orthogonal_full_prior, spectral_weights,
)
from representation.transfer_mixture import TransferMixtureGPR
from representation.transferable_spectral import SourceDomainBatch


class OrthogonalFunctionPriorTests(unittest.TestCase):
    def test_one_source_task_uses_existing_intercept_floor(self):
        reference=np.linspace(-1.5,1.5,24)[:,None]
        basis=OrthogonalLowOrderBasis(reference)
        batch=SourceDomainBatch('only_source',reference,np.tanh(reference[:,0])+2.)
        prior=fit_orthogonal_function_prior([batch],[np.full(24,.01)],basis)
        self.assertEqual(prior.covariance[0,0],.25)
        self.assertTrue(np.isfinite(prior.mean).all())
        self.assertTrue(np.isfinite(prior.covariance).all())
        np.testing.assert_array_equal(prior.mean[1:],0.)

    def test_product_orthogonality_does_not_claim_joint_coordinate_independence(self):
        axis = np.linspace(-1.8, 1.8, 18)
        reference = np.column_stack([axis, axis])
        basis = OrthogonalLowOrderBasis(reference)
        product = np.array([(a, b) for a in axis for b in axis])
        features = basis.transform(product)
        np.testing.assert_allclose(features.mean(axis=0), 0, atol=1e-10)
        np.testing.assert_allclose(features.T @ features / len(product),
                                   np.eye(basis.feature_dim), atol=1e-9)
        self.assertGreater(basis.joint_gram_error, 0.9)
        pair = features[:, -1].reshape(len(axis), len(axis))
        np.testing.assert_allclose(pair.mean(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(pair.mean(axis=1), 0, atol=1e-12)

    def test_dictionary_span_preserved_on_new_coordinates(self):
        rng = np.random.default_rng(1004)
        reference, heldout = rng.uniform(-2, 2, (80, 3)), rng.uniform(-2, 2, (24, 3))
        basis = OrthogonalLowOrderBasis(reference)
        original = basis.dictionary._library(reference)[0]
        matrix = np.column_stack([np.ones(len(reference)), basis.transform(reference)])
        conversion = np.linalg.lstsq(matrix, original, rcond=None)[0]
        reconstructed = np.column_stack([np.ones(len(heldout)), basis.transform(heldout)]) @ conversion
        np.testing.assert_allclose(reconstructed, basis.dictionary._library(heldout)[0], atol=1e-9)

    def test_constant_public_coordinates_are_removed(self):
        reference = np.column_stack([np.linspace(-2, 2, 24), np.full(24, 0.3)])
        basis = OrthogonalLowOrderBasis(reference)
        self.assertEqual(basis.group_names, ["main:0"])
        self.assertLess(basis.product_centering_error, 1e-9)

    def test_spectral_shrinkage_and_its_learning_gradient(self):
        basis = OrthogonalLowOrderBasis(np.random.default_rng(41).uniform(-2, 2, (64, 2)))
        weights, derivative = spectral_weights(basis, 1.3)
        finite_difference = (spectral_weights(basis, 1.30001)[0]
                             - spectral_weights(basis, 1.29999)[0]) / 0.00002
        np.testing.assert_allclose(derivative, finite_difference, atol=1e-9)
        for group in range(len(basis.group_names)):
            indices = np.flatnonzero(basis.group_index == group)
            self.assertAlmostEqual(float(weights[indices].sum()), 1.0)
            ordered = indices[np.argsort(basis.roughness[indices])]
            self.assertTrue(np.all(np.diff(weights[ordered]) <= 1e-12))

    def test_source_structure_changes_posterior_and_kg_with_identical_paid_data(self):
        rng = np.random.default_rng(104)
        reference = rng.uniform(-2, 2, (32, 2))
        basis = OrthogonalLowOrderBasis(reference)
        designs = [rng.uniform(-2, 2, (32, 2)) for _ in range(4)]
        priors = []
        for coordinate in (0, 1):
            batches = [SourceDomainBatch(str(i), X,
                (i + 1) * np.tanh(0.5 * X[:, coordinate])) for i, X in enumerate(designs)]
            priors.append(fit_orthogonal_function_prior(batches, [np.full(32, .001)] * 4, basis))
        masses = [p.diagnostics["source_group_variances"] for p in priors]
        self.assertGreater(masses[0][0], masses[0][1])
        self.assertGreater(masses[1][1], masses[1][0])
        candidates = np.arange(len(reference))[:, None]
        models = []
        for prior in priors:
            np.testing.assert_array_equal(prior.mean[1:], 0)
            heads = [TransferMixtureGPR([prior, same_source_orthogonal_full_prior(prior)], reference)
                     for _ in range(3)]
            for i in range(4):
                for h, model in enumerate(heads):
                    model.update([i], [float(np.tanh(reference[i, 0])), -.8, -2.5][h], .03)
                    self.assertEqual(model.observations, i + 1)
            models.append(heads)
        self.assertGreater(float(np.max(np.abs(models[0][0].posterior_mean_many(candidates)
                      - models[1][0].posterior_mean_many(candidates)))), .01)
        protocol = {"replicates": 12, "verification_replicates": 256, "chance_alpha": .05}
        scores = [verification_kg_scores(heads, candidates, protocol) for heads in models]
        self.assertTrue(all(np.isfinite(score).all() for score in scores))
        self.assertGreater(float(np.max(np.abs(scores[0] - scores[1]))), 1e-5)


if __name__ == "__main__":
    unittest.main()
