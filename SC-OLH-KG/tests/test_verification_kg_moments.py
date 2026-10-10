import itertools
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from numpy.polynomial.hermite import hermgauss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.transfer_external_models import OfficialHyperBOSurrogate
from baselines.transfer_models import ExactGPSurrogate
from core.verification_kg import (verification_kg_from_moments, verification_kg_surrogates,
                                  verification_utility)

PROTOCOL = {"verification_replicates": 256, "chance_alpha": .05}


class VerificationMomentTests(unittest.TestCase):
    def test_kernel_bridge_matches_dense_conditioning_and_counts_noise_once(self):
        rng = np.random.default_rng(407)
        X = rng.uniform(size=(35, 2))
        models = []
        for value in (4.6, -.4, -2.5):
            model = ExactGPSurrogate(kernel="rbf", outputscale=.2, normalization=(value, 1.))
            model.fit(X[:5], value + rng.normal(0, .1, 5), np.full(5, .05))
            models.append(model)
        predictions = [m.predict(X, full_cov=True) for m in models]
        mean = np.column_stack([m for m, _ in predictions])
        covariance = np.stack([c for _, c in predictions], axis=-1)
        variance = np.einsum('iih->ih', covariance)
        noise = np.tile([.0012, .01, .2], (len(X), 1))
        for model, (_, c) in zip(models, predictions):
            np.testing.assert_allclose(model.predict(X)[1], np.diag(c))
        nodes, weights = hermgauss(3)
        triples = list(itertools.product(range(3), repeat=3))
        normals = np.array([[np.sqrt(2) * nodes[j] for j in triple] for triple in triples])
        joint_weights = np.array([np.prod([weights[j] / np.sqrt(np.pi) for j in triple]) for triple in triples])
        dense = []
        current, _ = verification_utility(mean, variance, PROTOCOL)
        for i in range(len(X)):
            slope = covariance[:, i] / np.sqrt(variance[i] + noise[i])
            future_mean = mean[:, None] + slope[:, None] * normals[None]
            future_var = np.maximum(variance - slope ** 2, 0)
            future, _ = verification_utility(future_mean, future_var[:, None], PROTOCOL)
            dense.append(np.max(future, axis=0) @ joint_weights - current.max())
        snapshots = (mean.copy(), covariance.copy(), noise.copy(), [m.y.copy() for m in models])
        np.testing.assert_allclose(verification_kg_surrogates(models, X, noise, PROTOCOL), dense, atol=1e-12)
        np.testing.assert_allclose(verification_kg_from_moments(mean, covariance, noise, PROTOCOL), dense, atol=1e-12)
        for value, before in zip((mean, covariance, noise), snapshots[:3]):
            np.testing.assert_array_equal(value, before)
        for model, before in zip(models, snapshots[3]):
            np.testing.assert_array_equal(model.y, before)

    def test_official_hyperbo_noise_contract_reads_warped_label_variance(self):
        # A contract test without importing JAX. It does not train or validate
        # the official runtime; the external call is checked at its boundary.
        model = OfficialHyperBOSurrogate()
        model.params = object()
        warp = object()
        calls = []
        def retrieve(params, names, warp_func):
            calls.append((params, names, warp_func))
            return [.037]
        modules = (None, None, None, SimpleNamespace(retrieve_params=retrieve),
                   None, None, SimpleNamespace(DEFAULT_WARP_FUNC=warp))
        with patch.object(model, "_modules", return_value=modules):
            self.assertEqual(model.observation_noise_variance(), .037)
        self.assertEqual(calls, [(model.params, ["noise_variance"], warp)])
        self.assertFalse(model.diagnostics()["supplied_per_point_noise_consumed"])


if __name__ == "__main__":
    unittest.main()
