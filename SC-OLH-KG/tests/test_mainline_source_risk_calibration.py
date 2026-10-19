import unittest

import numpy as np
from scipy.linalg import block_diag

from performance.mainline_joint_risk_source import joint_prior_moments, law_with_original_task_scale
from performance.calibrate_mainline_source_risk import TaskScaleEvidence
from tests.test_mainline_joint_risk_source import JointSourceRiskTests


class SourceTaskScaleTest(unittest.TestCase):
    def test_prior_scaling_and_evidence_match_zero_change_and_finite_difference(self):
        law, H, Q, J = JointSourceRiskTests().setup_prior()
        self.assertIs(law_with_original_task_scale(law, 1.), law)
        old = joint_prior_moments(law, H, Q, J, [.07, .03])
        new = joint_prior_moments(law_with_original_task_scale(law, .3), H, Q, J, [.07, .03])
        D, T = law['source_design'], law['task_covariance']
        np.testing.assert_allclose(new['source_covariance'], old['source_covariance']-.7*block_diag(*[
            d @ T @ d.T for d in np.split(D, 4)]), atol=1e-14)
        np.testing.assert_allclose(new['query_covariance'], old['query_covariance']-.7*Q @ T @ Q.T, atol=1e-14)
        np.testing.assert_array_equal(new['cross_covariance'], old['cross_covariance'])
        rng = np.random.default_rng(20261015)
        eta = rng.normal(-4., .2, (4, 8, 12))
        weights = np.full(32, 1/32)
        labels = law['source_mean']+rng.normal(0, .1, 12)
        evidence = TaskScaleEvidence(law, H, labels, eta, weights, [.07, .03])
        zero = evidence.summarize(1.)
        self.assertAlmostEqual(zero['log_source_evidence_ratio_from_joint'], 0., places=13)
        np.testing.assert_allclose(zero['new_component_weights'], weights, atol=1e-15)
        actual = evidence.objective([.6])[1][0]
        finite = (evidence.objective([.60001])[0]-evidence.objective([.59999])[0])/.00002
        self.assertAlmostEqual(actual, finite, places=8)


if __name__ == '__main__':
    unittest.main()
