import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.budgeted_guarded_kg import BudgetedTerminalActions, terminal_covariance_root
from core.guarded_deployment_kg import posterior_draws
from representation.joint_function_risk import JointFunctionRiskPosterior


class ConstantRiskTerminalTests(unittest.TestCase):
    def test_exact_constant_risk_field_matches_dense_terminal_expectations(self):
        def field(means):
            covariance = np.zeros((3, 2, 3, 2))
            covariance[:, 0, :, 0] = .001*np.eye(3)
            covariance[:, 1, :, 1] = .2*np.ones((3, 3))
            return JointFunctionRiskPosterior.from_moments(np.column_stack([means, np.full(3, -7.)]), covariance)
        f, g = field([.4, .5, .6]), field([-.08, -.1, -.03])
        protocol = {'objective_reference': 1., 'familywise_delta': .05, 'chance_alpha': .05,
                    'comparison_budget': 8, 'comparison_delta': .05}
        terminal = BudgetedTerminalActions([[0, 1, 2], [1, 0, 2], [2, 0, 1]], [1, 0, 1], protocol)
        dense = terminal.evaluate(posterior_draws(f, g, 14, 80843), 360)
        marginal = terminal.evaluate_posterior(f, g, 360, 14, 80844)
        np.testing.assert_allclose(marginal[0], dense[0], atol=.003)
        np.testing.assert_allclose(marginal[1], dense[1], atol=.6)
        self.assertEqual(marginal[2], dense[2])

    def test_full_rank_root_keeps_frozen_gaussian_coordinates(self):
        covariance = np.array([[1., .3], [.3, .2]])
        np.testing.assert_array_equal(terminal_covariance_root(covariance), np.linalg.cholesky(covariance))


if __name__ == '__main__':
    unittest.main()
