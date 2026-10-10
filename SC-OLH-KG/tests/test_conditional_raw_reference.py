import copy
import unittest

import numpy as np
from scipy.stats import qmc

from core.conditional_raw_reference import conditional_raw_coordinates, raw_guarded_head_statistics
from core.positive_mixture_lookahead import condition_positive_raw
from core.raw_likelihood_lookahead import objective_terminal_statistics, risk_terminal_statistics
from test_conditional_terminal_kg import correlated_fields
from test_positive_source_mixture import terminal


class ConditionalRawReferenceTests(unittest.TestCase):
    def test_conditional_statistics_match_historical_joint_transport(self):
        action = {'kind': 'guarded', 'primary': 0, 'shortlist': [0, 1, 2],
            'incumbent_position': 1, 'budgets': [130, 130, 83]}
        protocol = terminal().protocol
        nodes = qmc.Sobol(6, scramble=True, seed=202610077).random_base2(16)
        full_nodes = qmc.Sobol(7, scramble=True, seed=202610077).random_base2(18)
        for head, field in enumerate(correlated_fields()):
            future = copy.deepcopy(field); label = [.4, -.28][head]
            evidence = future.update_batch(0, label, None, 1)
            actual, diagnostics = raw_guarded_head_statistics(field, future, action, nodes, label,
                evidence, head, protocol)
            draws, weights, _ = condition_positive_raw(field, future, 0, [0, 1, 2], full_nodes, label, evidence)
            draws = draws.reshape(1, len(full_nodes), 3, 2)
            expected = (objective_terminal_statistics(draws, weights[None], 1, protocol)[0] if head == 0
                else risk_terminal_statistics(draws, weights[None], 1, action['budgets'], protocol)[0][0])
            np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=0.)
            self.assertLess(abs(diagnostics['predictive_log_evidence_error']), 1e-4)
            self.assertGreater(diagnostics['effective_sample_size'], len(nodes)*.95)
            self.assertEqual(field.paid_episodes, 0)

    def test_exactly_shared_eta_is_preserved_after_raw_conditioning(self):
        field = correlated_fields(shared_eta=True)[1]
        future = copy.deepcopy(field); evidence = future.update_batch(0, -.1, None, 1)
        nodes = qmc.Sobol(6, scramble=True, seed=202610077).random_base2(10)
        eta, mean, covariance, weights, _ = conditional_raw_coordinates(field, future, [0, 1, 2],
            nodes, -.1, evidence)
        np.testing.assert_allclose(eta[:, 1:], np.repeat(eta[:, :1], 2, axis=1), atol=1e-12, rtol=0.)
        self.assertAlmostEqual(weights.sum(), 1., places=14)
        self.assertGreater(np.abs(covariance[:, 0, 1]).max(), 1e-5)
        self.assertTrue(np.all(np.isfinite(mean)))


if __name__ == '__main__':
    unittest.main()
