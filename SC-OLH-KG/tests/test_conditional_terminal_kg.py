import unittest

import numpy as np

from core.budgeted_guarded_kg import BudgetedTerminalActions
from core.conditional_terminal_kg import ConditionalTerminalActions
from core.direct_terminal_reference import direct_terminal_reference
from core.mixture_guarded_kg import mixture_raw_lookahead
from representation.joint_response_mixture import JointResponseMixture
from test_positive_source_mixture import field, terminal


def correlated_fields(shared_eta=False):
    f, g = field([.3, .8], [.25, .75]), field([-.32, -.2], [.65, .35])
    for current in (f, g):
        current.means[:, ::2] += np.array([0., .04, -.02])
        current.means[:, 1::2] = -4. + np.array([0., .4, -.3])
        root = np.diag([.055, .32, .06, .26, .05, .3])
        root[2, 0] = .03; root[4, 0] = -.02; root[3, 1] = .12
        root[0, 1] = .015; root[4, 3] = .02
        current.covariances[:] = root @ root.T
        if shared_eta:
            current.means[:, 1::2] = -4.
            current.covariances[:, ::2, 1::2] = 0.
            current.covariances[:, 1::2, ::2] = 0.
            current.covariances[:, 1::2, 1::2] = .09
    return f, g


class ConditionalTerminalTests(unittest.TestCase):
    def test_completed_fantasy_prefix_matches_full_rule_without_repeating_terminals(self):
        first, second = correlated_fields(), correlated_fields()
        second[0].means[:, 0] += .2; second[0].means[:, 2] -= .2
        second[1].means[:, 0] += .1; second[1].means[:, 2] -= .1
        model = JointResponseMixture(['A', 'B'], [first, second], prior_weights=[.4, .6])
        calls = []
        class RecordingTerminal(ConditionalTerminalActions):
            def evaluate_posterior(self, f, g, remaining, power, seed):
                if f.paid_episodes:
                    calls.append(remaining)
                return super().evaluate_posterior(f, g, remaining, power, seed)
        terminal_model = RecordingTerminal([[0, 1, 2], [1, 0, 2]], [1, 1], terminal().protocol)
        kwargs = {'draw_power': 5, 'seed': 202610077}
        full = mixture_raw_lookahead(model, terminal_model, 360, [1], fantasy_power=3, **kwargs)['lookahead'][0]
        for power in (1, 2):
            prefix = mixture_raw_lookahead(model, terminal_model, 360, [1], fantasy_power=power, **kwargs)['lookahead'][0]
            calls.clear()
            resumed = mixture_raw_lookahead(model, terminal_model, 360, [1], fantasy_power=3,
                completed_prefix=prefix, **kwargs)['lookahead'][0]
            self.assertEqual(len(calls), (8-2**power)*2*2)
            for key in full:
                if key in ('reused_fantasies', 'new_terminal_fantasies'):
                    continue
                np.testing.assert_allclose(resumed[key], full[key], atol=1e-12, rtol=0., err_msg=key)
        self.assertEqual(model.paid_episodes, 0)

    def test_guarded_integrals_match_joint_reference_for_both_branch_types(self):
        f, g = correlated_fields()
        for incumbent in (0, 1):
            old = BudgetedTerminalActions([[0, 1, 2]], [incumbent], terminal().protocol)
            new = ConditionalTerminalActions(old.shortlists, old.incumbents, old.protocol)
            reference, cost, actions = old.evaluate_posterior(f, g, 360, 18, 202610077)
            actual, actual_cost, actual_actions = new.evaluate_posterior(f, g, 360, 16, 202610077)
            self.assertEqual(actual_actions, actions)
            np.testing.assert_allclose(actual[:3], reference[:3], atol=2e-4, rtol=0.)
            np.testing.assert_allclose(actual_cost[:3], cost[:3], atol=.02, rtol=0.)

    def test_exactly_shared_eta_retains_correlated_policy_means(self):
        f, g = correlated_fields(shared_eta=True)
        old = terminal(); new = ConditionalTerminalActions(old.shortlists, old.incumbents, old.protocol)
        reference, cost, actions = old.evaluate_posterior(f, g, 360, 18, 202610077)
        actual, actual_cost, actual_actions = new.evaluate_posterior(f, g, 360, 14, 202610077)
        self.assertEqual(actions, actual_actions)
        np.testing.assert_allclose(actual[:3], reference[:3], atol=2e-4, rtol=0.)
        np.testing.assert_allclose(actual_cost[:3], cost[:3], atol=.03, rtol=0.)

    def test_direct_values_match_analytic_reference_and_are_sobol_independent(self):
        model = JointResponseMixture(['A', 'B'], [correlated_fields(),
            (field([.8], [1.]), field([-.55], [1.]))], prior_weights=[.2, .8])
        old = terminal(); new = ConditionalTerminalActions(old.shortlists, old.incumbents, old.protocol)
        first, _, actions = model.terminal_values(new, 359, 4, 202610077)
        second, _, _ = model.terminal_values(new, 359, 6, 80530)
        for i, action in enumerate(actions):
            if action['kind'] == 'direct':
                expected = direct_terminal_reference(model, action, new.protocol)['expected_reward']
                self.assertAlmostEqual(first[i], expected, places=12)
                self.assertEqual(first[i], second[i])

    def test_actual_lookahead_uses_new_terminal_after_each_paid_fantasy(self):
        calls = []
        class RecordingTerminal(ConditionalTerminalActions):
            def evaluate_posterior(self, f, g, remaining, power, seed):
                calls.append((f.paid_episodes, g.paid_episodes, remaining))
                return super().evaluate_posterior(f, g, remaining, power, seed)
        model = JointResponseMixture(['source'], [correlated_fields()])
        old = terminal(); new = RecordingTerminal(old.shortlists, old.incumbents, old.protocol)
        result = mixture_raw_lookahead(model, new, 360, [1], draw_power=4,
            fantasy_power=1, seed=202610077)
        self.assertEqual(calls, [(0, 0, 360), (0, 0, 359), (1, 1, 359), (1, 1, 359)])
        self.assertTrue(result['lookahead'][0]['actual_paid_filter_used_for_every_fantasy'])
        self.assertEqual(model.paid_episodes, 0)
        self.assertEqual(model.history, [])


if __name__ == '__main__':
    unittest.main()
