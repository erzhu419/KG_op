import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.special import digamma, polygamma

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from performance.run_legacy_related_inventory_source import problem_for, task_configs, training_view
from representation.legacy_function_prior import legacy_source_labels, replicated_source_labels
from representation.legacy_observable_prior import legacy_exposure_descriptors
from core.budgeted_guarded_kg import BudgetedTerminalActions
from performance.score_legacy_related_inventory_source import safe_terminal_values


class RelatedInventorySourceTests(unittest.TestCase):
    def test_observed_label_refactor_preserves_original_four_head_contract(self):
        raw = np.array([[[1., 4.], [2., 2.], [3., 3.]],
                        [[2., 5.], [4., 1.], [6., 3.]]])
        profiles = np.array([[.2, .3, .4, .5], [.6, .4, .3, .2]])
        task = SimpleNamespace(name='source', X=profiles, Y_replicates=raw, tau=2.)
        old = legacy_source_labels(SimpleNamespace(tasks=[task]))[0]
        new = replicated_source_labels('source', profiles, raw, 2.)
        expected = np.array([[2., 1., -digamma(1.), -digamma(1.)],
                             [4., 1., np.log(4.)-digamma(1.), np.log(4.)-digamma(1.)]])
        np.testing.assert_allclose(new['response'], expected)
        np.testing.assert_allclose(new['noise'], [[1/3, 1/3, polygamma(1, 1), polygamma(1, 1)],
                                                  [4/3, 4/3, polygamma(1, 1), polygamma(1, 1)]])
        for field in ('psi', 'response', 'noise'):
            np.testing.assert_array_equal(old[field], new[field])

    def test_fit_view_keeps_task_identity_and_cannot_read_heldout_records(self):
        class Heldout:
            def __getitem__(self, key):
                raise AssertionError('heldout source data entered training')
        bank = [[20, 30, 40, 50], [60, 40, 30, 20]]
        raw = [[[1., 3.], [2., 2.], [3., 4.]], [[2., 1.], [4., 3.], [6., 2.]]]
        tasks = [{'task_id': f'source_{i}', 'tau': 0., 'Y_replicates': raw,
                  'problem_kwargs': Heldout(), 'true_sigma': Heldout()} for i in range(6)]
        data = {'bank': bank, 'tasks': tasks + [Heldout(), Heldout()]}
        view = training_view(data, range(6), 100)
        self.assertEqual([t['name'] for t in view], [f'source_{i}' for i in range(6)])
        self.assertTrue(all(set(t) == {'name', 'psi', 'response', 'noise'} for t in view))

    def test_parameterized_tasks_change_response_without_changing_public_descriptor(self):
        plan = {'task_seed': 80751, 'source_task_count': 8, 'family': 'InventorySupplyChain',
                'source_dimension': 50, 'task_distribution': {'L': 100, 'sigma': .04,
                                                             'alpha': .05, 'weights': [.5, .5]}}
        configs = task_configs(plan)
        self.assertEqual(configs, task_configs(plan))
        bank = [[60]*17 + [35]*17 + [43]*16, [40]*17 + [20]*17 + [30]*16]
        providers = [problem_for(plan, c) for c in configs]
        descriptors = [legacy_exposure_descriptors(p, bank) for p in providers]
        for descriptor in descriptors[1:]:
            np.testing.assert_array_equal(descriptor, descriptors[0])
        means = [p.true_constraint_mean(bank[0]) for p in providers]
        self.assertGreater(np.ptp(means), .005)

    def test_safe_score_excludes_false_certificates_without_changing_calls(self):
        protocol = {'objective_reference': 1., 'familywise_delta': .05, 'chance_alpha': .05,
                    'comparison_budget': 8, 'comparison_delta': .05}
        terminal = BudgetedTerminalActions([[0, 1, 2]], [1], protocol)
        # Policy0 is just outside the chance constraint: certification has
        # positive power, but deploying it must contribute no safe utility.
        truth = np.array([[.2, -4., -.15, -4.], [.4, -4., -.5, -4.], [.5, -4., -.6, -4.]])
        raw, costs, actions = terminal.evaluate(truth[None], 358)
        safe, safe_costs = safe_terminal_values(terminal, truth, np.array([False, True, True]), 358)
        direct = next(i for i, a in enumerate(actions) if a['kind'] == 'direct' and a['budgets'] == [128])
        self.assertGreater(raw[direct], 0.)
        self.assertEqual(safe[direct], 0.)
        np.testing.assert_allclose(safe_costs, costs)
        all_safe, all_safe_costs = safe_terminal_values(terminal, truth, np.ones(3, dtype=bool), 358)
        np.testing.assert_allclose(all_safe, raw)
        np.testing.assert_allclose(all_safe_costs, costs)


if __name__ == '__main__':
    unittest.main()
