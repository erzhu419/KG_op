import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from performance.mainline_moment_policy import moment_state, query_action, terminal_action
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.joint_response_mixture import JointResponseMixture


def field(mean_value, eta_value, coupled=False):
    mean = np.column_stack([np.full(3, mean_value), np.full(3, eta_value)])
    covariance = np.diag(np.tile([.09, .2], 3)).reshape(3, 2, 3, 2)
    if coupled:
        covariance[0, 0, 1, 0] = covariance[1, 0, 0, 0] = .06
    return JointFunctionRiskPosterior.from_moments(mean, covariance)


class MainlineMomentPolicyTests(unittest.TestCase):
    def test_episode_noise_retains_both_mixture_levels(self):
        first = field(.2, -3.)
        second = field(.2, -1.)
        first.means = np.vstack([first.means, second.means])
        first.covariances = np.concatenate([first.covariances, second.covariances])
        first.log_weights = np.log([.25, .75])
        model = JointResponseMixture(["a", "b"],
            [(first, field(-.4, -2.)), (field(.2, -4.), field(-.4, -5.))],
            prior_weights=[.3, .7])
        state = moment_state(model)
        expected_f = .3 * (.25 * np.exp(-2.9) + .75 * np.exp(-.9)) + .7 * np.exp(-3.9)
        expected_g = .3 * np.exp(-1.9) + .7 * np.exp(-4.9)
        np.testing.assert_allclose(state["noise"], np.tile([expected_f, expected_g], (3, 1)))
        gaussian_noise = np.exp(state["mean"][:, 1] +
            np.array([state["cov"][i, 1, i, 1] for i in range(3)]) / 2)
        self.assertGreater(np.max(abs(state["noise"][:, 0] - gaussian_noise)), .01)

    def test_raw_episode_transmits_only_through_available_policy_covariance(self):
        coupled = JointResponseMixture(["one"], [(field(.2, -3.), field(-.4, -3., True))])
        independent = JointResponseMixture(["one"], [(field(.2, -3.), field(-.4, -3.))])
        before = moment_state(coupled)
        control_before = moment_state(independent)
        protocol = {"objective_reference": 1., "chance_alpha": .05, "familywise_delta": .05}
        np.testing.assert_allclose(terminal_action(before, 100, protocol)["utilities"],
                                   terminal_action(control_before, 100, protocol)["utilities"])
        choice = query_action(before, 100, protocol)
        self.assertEqual(len(choice["scores73"]), 3)
        self.assertTrue(np.isfinite(choice["scores73"]).all())
        self.assertEqual(choice["order"], 3)
        self.assertEqual(coupled.paid_episodes, 0)
        np.testing.assert_array_equal(before["mean"], moment_state(coupled)["mean"])
        coupled.update_raw(0, [.2, -.9])
        independent.update_raw(0, [.2, -.9])
        self.assertGreater(abs(moment_state(coupled)["mean"][1, 2] - before["mean"][1, 2]), .01)
        np.testing.assert_allclose(moment_state(independent)["mean"][1:], control_before["mean"][1:], atol=1e-12)
        self.assertEqual(coupled.paid_episodes, 1)


if __name__ == "__main__":
    unittest.main()
