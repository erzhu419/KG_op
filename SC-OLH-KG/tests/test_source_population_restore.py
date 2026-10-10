import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from representation.orthogonal_function_prior import OrthogonalLowOrderBasis, fit_orthogonal_function_prior
from representation.source_population_mean import fit_source_population_mean
from representation.source_population_restore import restore_common_mean_prior
from representation.transferable_spectral import SourceDomainBatch


class FrozenPopulationRestorationTests(unittest.TestCase):
    def test_restores_fitted_covariance_and_common_mean_without_optimization(self):
        rng = np.random.default_rng(80795)
        x = rng.normal(size=(20, 2))
        basis = OrthogonalLowOrderBasis(x)
        # The large inter-task shift exercises the non-floor intercept variance.
        batches = [SourceDomainBatch(str(i), x, 4*i+.3*np.sin(x[:, 0])) for i in range(3)]
        noise = [np.full(len(x), .01) for _ in batches]
        original = fit_orthogonal_function_prior(batches, noise, basis, max_iterations=12)
        common = fit_source_population_mean(original, batches, noise)
        restored = restore_common_mean_prior(basis, common.diagnostics, batches, noise)
        self.assertGreater(original.covariance[0, 0], .25)
        np.testing.assert_allclose(restored.covariance, common.covariance, atol=1e-12, rtol=1e-12)
        np.testing.assert_array_equal(restored.mean, common.mean)
        self.assertEqual(restored.discrepancy_variance, common.discrepancy_variance)


if __name__ == '__main__':
    unittest.main()
