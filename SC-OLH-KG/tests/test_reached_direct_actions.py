import unittest

import numpy as np

from core.budgeted_guarded_kg import BudgetedTerminalActions
from performance.compare_legacy_source_layout_reached_actions import ReachedDirectActions
from test_joint_response_mixture import mixture
from test_guarded_deployment_kg import PROTOCOL


class ReachedDirectActionsTests(unittest.TestCase):
    def test_native_mixture_utilities_equal_full_class_direct_entries(self):
        model = mixture(((.1,.2),(1.2,-.5)),mean_variance=.02)
        model.update_raw(1,[1.18,-.49])
        full = BudgetedTerminalActions(np.array([[0,1,2],[1,0,2],[2,0,1]]),np.array([1,0,1]),PROTOCOL)
        probe = ReachedDirectActions(PROTOCOL,[0,1,2])
        r,c,actions = model.terminal_values(full,352,7,80771)
        dr,dc,direct = model.terminal_values(probe,352,7,80771)
        ix = [actions.index(a) for a in direct]
        np.testing.assert_allclose(dr,r[ix],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(dc,c[ix],rtol=1e-12,atol=1e-12)


if __name__=='__main__':
    unittest.main()
