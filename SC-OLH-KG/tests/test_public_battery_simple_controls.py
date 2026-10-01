import unittest
import numpy as np
from performance.replay_public_battery_simple_controls import constant_design,enumerate_population

class ControlsTests(unittest.TestCase):
    def test_constant_design_excludes_functional_family_without_outcomes(self):
        ids=['constant_pair_a','constant_pair_b','constant_pair_c','functional_pair_x']
        nodes=(np.arange(1000)+.5)/1000
        pairs=np.array([[np.full(1000,a),np.full(1000,b)] for a,b in ((.1,.1),(.5,.5),(.9,.9),(.3,.7))])
        domain,chosen=constant_design(ids,nodes,pairs,3)
        self.assertEqual(domain,ids[:3]);self.assertEqual(set(chosen),set(ids[:3]));self.assertEqual(chosen[0],ids[1])

    def test_census_keeps_chance_feasible_null_cost_and_charges_all_windows(self):
        cells={('day',m,p,'rule'):dict(window_success=p=='safe' or m==0,cost_GBP=10 if p=='safe' else 1 if m==0 else None)
               for p in ['safe','partial'] for m in [0,60]}
        result=enumerate_population(cells,['safe','partial'],'day',[0,60],'rule',1)
        self.assertEqual(result['represented_window_queries'],4)
        self.assertEqual(result['feasible_candidates'],2)
        self.assertEqual(result['complete_cost_candidates'],1)
        self.assertEqual(result['selected_complete_cost_candidate']['profile_id'],'safe')
