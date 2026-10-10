import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from representation.orthogonal_function_prior import same_source_orthogonal_full_prior
from representation.source_function_prior import SourceFunctionPrior
from representation.source_population_mean import fit_source_population_mean
from representation.transferable_spectral import SourceDomainBatch


class LinearBasis:
    feature_dim = 1

    def transform(self, psi):
        return np.asarray(psi)

    def diagnostics(self):
        return {'active_dim': 1}


def prior():
    return SourceFunctionPrior(LinearBasis(), np.array([.2, 0.]),
        np.array([[.25, .03], [.03, .15]]), .02,
        {'mode': 'source_orthogonal_group_spectral', 'source_signal_scale': .3, 'source_label_count': 6})


class PopulationMeanTests(unittest.TestCase):
    def test_common_mean_matches_joint_gaussian_conditioning(self):
        p = prior()
        batches = [SourceDomainBatch('a', np.array([[-1.], [0.], [1.]]), np.array([-1., .1, 1.1])),
                   SourceDomainBatch('b', np.array([[-.8], [.2], [.9]]), np.array([-.6, .4, 1.3]))]
        noise = [np.array([.03, .01, .02]), np.array([.04, .02, .03])]
        designs = [np.column_stack([np.ones(3), b.psi]) for b in batches]
        D = np.vstack(designs); y = np.concatenate([b.signals for b in batches])
        V = block_diag(*[d @ p.covariance @ d.T + np.diag(n+p.discrepancy_variance)
                          for d, n in zip(designs, noise)])
        expected = p.mean+p.covariance @ D.T @ np.linalg.solve(V+D @ p.covariance @ D.T, y-D @ p.mean)
        got = fit_source_population_mean(p, batches, noise)
        np.testing.assert_allclose(got.mean, expected, atol=1e-12)

    def test_many_sites_in_one_task_do_not_become_independent_tasks(self):
        p = prior(); p.mean[:] = 0.; p.covariance = np.diag([.25, .15])
        n = 100
        batch = SourceDomainBatch('one_task', np.zeros((n, 1)), np.ones(n))
        got = fit_source_population_mean(p, [batch], [np.full(n, .01)])
        expected = n*.25/(2*n*.25+.03)
        self.assertAlmostEqual(got.mean[0], expected, places=12)
        self.assertLess(got.mean[0], .5)

    def test_mean_attribution_preserves_priors_and_shared_full_control(self):
        p = prior(); before = (p.mean.copy(), p.covariance.copy(), dict(p.diagnostics))
        batch = SourceDomainBatch('a', np.array([[-1.], [0.], [1.]]), np.array([-1., 0., 1.]))
        common = fit_source_population_mean(p, [batch], [np.full(3, .01)])
        full = same_source_orthogonal_full_prior(common)
        np.testing.assert_array_equal(common.covariance, p.covariance)
        self.assertEqual(common.discrepancy_variance, p.discrepancy_variance)
        np.testing.assert_array_equal(full.mean, common.mean)
        np.testing.assert_array_equal(p.mean, before[0])
        np.testing.assert_array_equal(p.covariance, before[1])
        self.assertEqual(p.diagnostics, before[2])
        self.assertFalse(common.diagnostics['new_task_covariance_includes_population_mean_uncertainty'])


if __name__ == '__main__':
    unittest.main()
