import copy
import unittest

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import qmc

from core.conditional_terminal_kg import ConditionalTerminalActions, risk_coordinates_from_moments, guarded_risk_statistics
from core.guarded_deployment_kg import _sd_quadrature, _tolerance_factor
from core.positive_mixture_lookahead import condition_positive_raw
from core.positive_raw_terminal import raw_terminal_component
from core.raw_likelihood_lookahead import risk_terminal_statistics, objective_terminal_statistics
from test_conditional_terminal_kg import correlated_fields
from test_positive_source_mixture import terminal


class PrimaryMeanTerminalTests(unittest.TestCase):
    def test_stopping_primary_integral_matches_native_positive_joint_reference(self):
        from core.budgeted_guarded_kg import BudgetedTerminalActions
        f, g = correlated_fields()
        for incumbent in (0, 1):
            native = BudgetedTerminalActions([[0, 1, 2]], [incumbent], terminal().protocol)
            reference, cost, actions = native.evaluate_posterior(f, g, 360, 18, 202610077)
            current = ConditionalTerminalActions(native.shortlists, native.incumbents, native.protocol)
            actual, actual_cost, actual_actions = current.evaluate_posterior(f, g, 360, 16, 202610077,
                guarded_mean_position=0)
            self.assertEqual(actions, actual_actions)
            np.testing.assert_allclose(actual, reference, atol=2e-4, rtol=0.)
            np.testing.assert_allclose(actual_cost, cost, atol=.03, rtol=0.)

    def test_independent_mean_integrals_have_known_branch_probabilities(self):
        protocol = terminal().protocol; budgets = [81, 130, 132]
        mean = np.array([-.15, -.16, -.08]); sigma = np.array([.06, .08, .05]); variance = np.array([.07, .05, .04])**2
        nodes = qmc.Sobol(7, scramble=True, seed=202610077).random_base2(16)
        eta = np.tile(2*np.log(sigma), (len(nodes), 1)); means = np.tile(mean, (len(nodes), 1))
        covariance = np.diag(variance)[None]; position = np.zeros(len(nodes), dtype=int)
        p = []
        for i, n in enumerate(budgets):
            sd, weights = _sd_quadrature(n)
            threshold = _tolerance_factor(n, protocol['chance_alpha'], protocol['familywise_delta']/3)*sd
            p.append(float(ndtr(-(mean[i]+sigma[i]*threshold)/np.sqrt(variance[i]+sigma[i]**2/n)) @ weights))
        for incumbent in (0, 1):
            expected = (np.array([p[0], (1-p[0])*p[1], (1-p[0])*(1-p[1])*p[2], 0.]) if incumbent == 0 else
                np.array([p[0]*(1-p[1]), (1-p[0])*p[1], (1-p[0])*(1-p[1])*p[2], p[0]*p[1]]))
            expected_cost = (budgets[0]+(1-p[0])*budgets[1]+(1-p[0])*(1-p[1])*budgets[2] if incumbent == 0 else
                budgets[0]+budgets[1]+p[0]*p[1]*2*protocol['comparison_budget']+(1-p[0])*(1-p[1])*budgets[2])
            for integrated in (0, 1, 2):
                coordinates = risk_coordinates_from_moments(eta, means, covariance, position, nodes,
                    mean_normals=ndtri(nodes[:, 4:6]), integrated_position=integrated)
                actual, cost = guarded_risk_statistics(coordinates, incumbent, budgets, protocol,
                    integrated_position=integrated)
                np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=0.)
                self.assertAlmostEqual(cost, expected_cost, delta=.02)

    def test_correlated_means_match_full_joint_transport(self):
        protocol = terminal().protocol; budgets = [130, 130, 83]
        mean = np.array([-.15, -.16, -.08]); sigma = np.array([.06, .08, .05])
        root = np.diag([.07, .05, .04]); root[1, 0] = .035; root[2, 0] = -.025
        covariance = (root @ root.T)[None]
        full = qmc.Sobol(3, scramble=True, seed=202610077).random_base2(18)
        means = mean+ndtri(full) @ root.T
        draws = np.stack([means, np.tile(2*np.log(sigma), (len(full), 1))], axis=-1)[None]
        nodes = qmc.Sobol(7, scramble=True, seed=202610077).random_base2(16)
        for incumbent in (0, 1):
            expected, expected_cost = risk_terminal_statistics(draws, np.full((1, len(full)), 1/len(full)), incumbent, budgets, protocol)
            for integrated in (0, 2):
                coordinates = risk_coordinates_from_moments(np.tile(2*np.log(sigma), (len(nodes), 1)),
                    np.tile(mean, (len(nodes), 1)), covariance, np.zeros(len(nodes), dtype=int), nodes,
                    mean_normals=ndtri(nodes[:, 4:6]), integrated_position=integrated)
                actual, cost = guarded_risk_statistics(coordinates, incumbent, budgets, protocol, integrated_position=integrated)
                np.testing.assert_allclose(actual, expected[0], atol=2e-4, rtol=0.)
                self.assertAlmostEqual(cost, expected_cost[0], delta=.02)

    def test_raw_primary_integral_matches_historical_joint_reference(self):
        prior = correlated_fields(); updated = copy.deepcopy(prior); labels = [.4, -.15]
        scores = [f.update_batch(0, label, None, 1) for f, label in zip(updated, labels)]
        current = ConditionalTerminalActions([[0, 1, 2]], [1], terminal().protocol)
        nodes = qmc.Sobol(7, scramble=True, seed=202610077).random_base2(16)
        new, _, actions, _ = raw_terminal_component(prior, updated, current, 359, 0, labels, scores, nodes, guarded_mean_position=0)
        old, _, _, _ = raw_terminal_component(prior, updated, current, 359, 0, labels, scores, nodes)
        full_nodes = qmc.Sobol(7, scramble=True, seed=202610077).random_base2(18)
        transport = [condition_positive_raw(p, u, 0, [0, 1, 2], full_nodes, labels[h], scores[h]) for h, (p, u) in enumerate(zip(prior, updated))]
        for index, action in enumerate(actions):
            if action['kind'] == 'direct':
                self.assertEqual(new[index], old[index])
                continue
            f, fw, _ = transport[0]; g, gw, _ = transport[1]
            objective = objective_terminal_statistics(f.reshape(1, len(full_nodes), 3, 2), fw[None], 1, current.protocol)[0]
            safety = risk_terminal_statistics(g.reshape(1, len(full_nodes), 3, 2), gw[None], 1, action['budgets'], current.protocol)[0][0]
            self.assertAlmostEqual(new[index], float(objective @ safety), delta=2e-4)


if __name__ == '__main__':
    unittest.main()
