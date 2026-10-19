import unittest

import numpy as np

from performance.diagnose_mainline_objective_transfer import decompose_mean_shift, internal_decomposition
from performance.diagnose_mainline_risk_transfer import capture_risk, native_transport
from performance.mainline_moment_policy import moment_state
from performance.replay_mainline_risk_correction import risk_moments
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.joint_response_mixture import JointResponseMixture


class RiskTransferTests(unittest.TestCase):
    def test_native_risk_transport_retains_eta_and_both_mass_layers(self):
        factor = np.array([[.4, 0, 0, 0], [.1, .5, 0, 0], [.2, .1, .3, 0], [.1, .2, .1, .4]])
        covariance = factor@factor.T
        def field(offset):
            mean = np.array([[.2+offset, -3.], [.4+offset, -2.]])
            result = JointFunctionRiskPosterior.from_moments(mean, covariance.reshape(2, 2, 2, 2))
            result.means = np.stack([mean.reshape(-1), mean.reshape(-1)+[.1, .2, -.1, -.2]])
            result.covariances = np.stack([covariance, covariance*1.2])
            result.log_weights = np.log([.4, .6])
            return result
        model = JointResponseMixture(['a', 'b'], [(field(3), field(0)), (field(4), field(.3))], [.7, .3])
        before = capture_risk(model, 0)
        model.update_raw(0, [3.7, .8])
        after = capture_risk(model, 0)
        terms = decompose_mean_shift(before['weights'], before['identity_mean'], after['weights'], after['identity_mean'])
        inner = internal_decomposition(before, after)
        native = native_transport(before, after)
        np.testing.assert_allclose(sum(inner.values()), terms['within_identity'], atol=1e-12)
        np.testing.assert_allclose(native['from_query_g_displacement']+native['from_query_eta_displacement'], inner['component_location'], atol=1e-12)
        self.assertGreater(abs(native['from_query_eta_displacement'][1]), 1e-5)
        np.testing.assert_allclose(after['weights']@after['identity_mean'], model.predict_joint()[0][:, 2])
        full = moment_state(model)
        for query in range(2):
            marginal = risk_moments(model, query)
            np.testing.assert_allclose(marginal['mean'], full['mean'][:, 2], atol=1e-12)
            np.testing.assert_allclose(marginal['variance'], np.diag(full['cov'][:, 2, :, 2]), atol=1e-12)
            np.testing.assert_allclose(marginal['covariance_to_query'], full['cov'][:, 2, query, 2], atol=1e-12)
            np.testing.assert_allclose(marginal['episode_noise'], full['noise'][query, 1], atol=1e-12)


if __name__ == '__main__':
    unittest.main()
