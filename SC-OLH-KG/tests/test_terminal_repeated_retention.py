import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.budgeted_guarded_kg import BudgetedTerminalActions


class RepeatedRetentionTests(unittest.TestCase):
    def test_two_preceding_recommendations_retain_original_actions_without_duplicates(self):
        original = np.array([[0, 1, 2], [3, 1, 2], [4, 1, 2]])
        terminal = BudgetedTerminalActions(original, [1, 1, 1], {'comparison_budget': 8})
        terminal.retain_recommendation(3)
        terminal.retain_recommendation(4)
        table = terminal.allocations(358)[0][1].copy()
        np.testing.assert_array_equal(table[:3], original)
        self.assertEqual(len(table), 7)
        self.assertEqual(len(np.unique(table, axis=0)), len(table))
        # A stopped recommendation may remain unchanged across paid steps.
        terminal.retain_recommendation(4)
        terminal.retain_recommendation(3)
        np.testing.assert_array_equal(terminal.allocations(358)[0][1], table)
        self.assertTrue(np.all(terminal.incumbents == 1))
        for kind, _, _, budgets in terminal.allocations(358):
            self.assertLessEqual(sum(budgets)+(16 if kind == 'guarded' else 0), 358)


if __name__ == '__main__':
    unittest.main()
