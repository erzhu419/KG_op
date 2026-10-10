import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from baselines.transfer_archive import resample_normalized_profiles
from core.budgeted_guarded_kg import BudgetedTerminalActions, budgeted_raw_lookahead, budgeted_marginal_lookahead
from core.guarded_deployment_kg import posterior_draws
from performance.benchmark_lodo_meta_prior import build_scalarized_problem
from representation.legacy_observable_prior import legacy_exposure_descriptors
from test_guarded_deployment_kg import models, PROTOCOL


class ObservableBudgetTests(unittest.TestCase):
    def test_public_descriptor_queries_no_outcome_or_hidden_risk(self):
        p=build_scalarized_problem("InventorySupplyChain",50,100,.04,.05,[.5,.5])
        def forbidden(*args,**kwargs): raise AssertionError("oracle/outcome query")
        p.simulate=p.true_objective=p.true_constraint_mean=p.true_sigma=p.risk_exposures=forbidden
        p.base.simulate=p.base.true_objectives=p.base.true_sigma=p.base.risk_exposures=forbidden
        result=legacy_exposure_descriptors(p,[tuple(range(50))])
        self.assertTrue(np.isfinite(result).all())

    def test_curve_spectral_units_remain_constant_across_supported_dimensions(self):
        z=.5+.1*np.cos(np.pi*np.linspace(0,1,50))
        lifted=resample_normalized_profiles(z,1000)[0]
        a=build_scalarized_problem("InventorySupplyChain",50,100000,.04,.05,[.5,.5])
        b=build_scalarized_problem("InventorySupplyChain",1000,100000,.04,.05,[.5,.5])
        da=legacy_exposure_descriptors(a,[tuple(np.rint(z*100000).astype(int))])
        db=legacy_exposure_descriptors(b,[tuple(np.rint(lifted*100000).astype(int))])
        # Ordered descriptor: 12 channel slots + 13 occupancy entries,
        # followed by 12 dynamics entries, the final six being the DCT.
        np.testing.assert_allclose(da[:,31:37],db[:,31:37],atol=.0001)

    def test_every_terminal_action_respects_paid_plus_verification_cap(self):
        terminal=BudgetedTerminalActions(np.array([[0,1,2],[1,0,2],[2,0,1]]),np.array([1,0,1]),PROTOCOL)
        draws=np.array([[[.6,-4.,-.2,-4.],[.65,-4.,-.3,-4.],[.7,-4.,-.4,-4.]]])
        reward,cost,actions=terminal.evaluate(draws,351)
        self.assertTrue(np.all(cost<=351))
        for a in actions:
            worst=sum(a["budgets"])+(16 if a["kind"]=="guarded" else 0)
            self.assertLessEqual(worst,351)
        self.assertEqual(len(reward),15)

    def test_two_response_heads_debit_one_episode_and_leave_paid_state_unchanged(self):
        f,g=models()
        class RecordedTerminal(BudgetedTerminalActions):
            def evaluate(self,draws,remaining):
                seen.append(remaining)
                return super().evaluate(draws,remaining)
        seen=[]
        terminal=RecordedTerminal(np.array([[0,1,2],[1,0,2],[2,0,1]]),np.array([1,0,1]),PROTOCOL)
        result=budgeted_raw_lookahead(f,g,terminal,352,[0,1],draw_power=4,fantasy_power=2,seed=105219)
        self.assertEqual(seen[0],352)
        self.assertEqual(seen[1:],[351]*9)
        self.assertEqual(f.paid_episodes,0); self.assertEqual(g.paid_episodes,0)

    def test_three_policy_marginal_retains_full_joint_reward_and_cost(self):
        f,g=models()
        terminal=BudgetedTerminalActions(np.array([[0,1,2],[1,0,2],[2,0,1]]),np.array([1,0,1]),PROTOCOL)
        dense=terminal.evaluate(posterior_draws(f,g,15,105220),352)
        marginal=terminal.evaluate_posterior(f,g,352,15,105221)
        np.testing.assert_allclose(marginal[0],dense[0],atol=.002)
        np.testing.assert_allclose(marginal[1],dense[1],atol=.5)
        self.assertEqual(marginal[2],dense[2])

    def test_marginal_fantasy_debits_one_episode_without_mutating_original_posteriors(self):
        f,g=models(); before=g.predict(full_cov=True)
        class RecordedTerminal(BudgetedTerminalActions):
            def evaluate_posterior(self,f,g,remaining,power,seed):
                seen.append(remaining)
                return super().evaluate_posterior(f,g,remaining,power,seed)
        seen=[]
        terminal=RecordedTerminal(np.array([[0,1,2],[1,0,2],[2,0,1]]),np.array([1,0,1]),PROTOCOL)
        result=budgeted_marginal_lookahead(f,g,terminal,352,[1],draw_power=4,fantasy_power=2,seed=105222)
        self.assertEqual(seen,[352]+[351]*5)
        self.assertTrue(np.isfinite(result["lookahead"][0]["net_reward_of_paid_episode"]))
        self.assertEqual(f.paid_episodes,0); self.assertEqual(g.paid_episodes,0)
        np.testing.assert_array_equal(before[0],g.predict(full_cov=True)[0])
        np.testing.assert_array_equal(before[1],g.predict(full_cov=True)[1])


if __name__=="__main__":
    unittest.main()
