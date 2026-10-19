import copy
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.special import logsumexp

from representation.vectorized_joint_function_risk import apply_raw_moments, local_raw_moments


def original_apply(field, candidate, result, gain=None, replicates=1):
    indices = [2*candidate, 2*candidate+1]
    local_mean = field.means[:, indices].copy()
    local_covariance = field.covariances[:, indices][:, :, indices].copy()
    if gain is None:
        gain = np.linalg.solve(local_covariance, field.covariances[:, indices, :]).transpose(0, 2, 1)
    field.means += np.einsum('kij,kj->ki', gain, result['mean']-local_mean)
    field.covariances += gain@(result['covariance']-local_covariance)@gain.transpose(0, 2, 1)
    field.covariances = .5*(field.covariances+field.covariances.transpose(0, 2, 1))
    field.log_weights += result['log_evidence']
    evidence = float(logsumexp(field.log_weights))
    field.log_weights -= evidence
    field.paid_batches += 1
    field.paid_episodes += replicates
    field.max_integration_relative_error = max(field.max_integration_relative_error, result['integration_relative_error'])
    return evidence


class ChunkedRawUpdateTests(unittest.TestCase):
    def test_multiple_chunks_and_tail_preserve_raw_filter_and_cached_gain(self):
        rng = np.random.default_rng(818)
        loading = rng.normal(size=(129, 12, 12)) / 15
        covariance = loading @ loading.transpose(0, 2, 1) + .1*np.eye(12)
        means = rng.normal(0, .1, (129, 12))
        means[:, 1::2] -= 3
        field = SimpleNamespace(means=means, covariances=covariance,
            log_weights=np.log(np.arange(1, 130)/np.arange(1, 130).sum()),
            paid_batches=2, paid_episodes=3, max_integration_relative_error=0.)
        indices = [6, 7]
        for replicates, eta_label in ((1, None), (3, -3.)):
            result = local_raw_moments(field.means[:, indices],
                field.covariances[:, indices][:, :, indices], .2, eta_label, replicates)
            gain = np.linalg.solve(field.covariances[:, indices][:, :, indices],
                                   field.covariances[:, indices, :]).transpose(0, 2, 1)
            for supplied in (None, gain):
                old, new = copy.deepcopy(field), copy.deepcopy(field)
                before_covariance = new.covariances
                expected = original_apply(old, 3, result, supplied, replicates)
                actual = apply_raw_moments(new, 3, result, supplied, replicates=replicates)
                self.assertEqual(actual, expected)
                np.testing.assert_array_equal(new.means, old.means)
                np.testing.assert_array_equal(new.covariances, old.covariances)
                np.testing.assert_array_equal(new.log_weights, old.log_weights)
                self.assertIs(new.covariances, before_covariance)
                self.assertEqual(new.paid_batches, 3)
                self.assertEqual(new.paid_episodes, 3+replicates)
                self.assertEqual(new.max_integration_relative_error, old.max_integration_relative_error)


if __name__ == '__main__':
    unittest.main()
