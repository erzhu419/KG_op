import unittest

import numpy as np

from performance.run_public_battery_source_archive import designs


class SourceDesignTests(unittest.TestCase):
    def fixture(self):
        ids=["constant_pair_a","constant_pair_b","constant_pair_unsafe","functional_pair_x"]
        nodes=(np.arange(1000)+.5)/1000
        pairs=np.array([[np.full(1000,a),np.full(1000,b)] for a,b in ((.35,.75),(.65,.95),(.05,.05),(.85,.15))])
        rows=[dict(period=day,window_start_minute=minute,profile_id=profile,dispatch_rule="conservative_admission",
                   window_success=profile!="constant_pair_unsafe",cost_GBP=None if profile=="constant_pair_unsafe" else cost)
              for day in ("2023-07-15","2023-10-15") for minute in (0,60)
              for profile,cost in zip(ids,(30,10,0,50))]
        return rows,ids,nodes,pairs

    def test_missing_failure_cost_never_becomes_an_objective_or_prunes_geometry(self):
        rows,ids,nodes,pairs=self.fixture()
        result=designs(rows,ids,nodes,pairs,["2023-07-15","2023-10-15"],[0,60],"conservative_admission",n0=4)
        self.assertEqual(result["source_first_center"],"constant_pair_b")
        self.assertNotIn("constant_pair_unsafe",result["source_complete_cost_support"])
        self.assertEqual(result["designs"]["source_complete_cost_best_z"][0],"constant_pair_b")
        self.assertEqual(set(result["designs"]["source_complete_cost_best_z"]),set(ids))
        self.assertFalse(result["target_candidate_domain_pruned"])
        self.assertEqual(result["structural_coordinate_dimension"],36)

    def test_unavailable_common_source_support_freezes_no_design(self):
        rows,ids,nodes,pairs=self.fixture()
        for r in rows:
            r.update(window_success=False,cost_GBP=None)
        result=designs(rows,ids,nodes,pairs,["2023-07-15","2023-10-15"],[0,60],"conservative_admission",n0=2)
        self.assertEqual(result["designs"],{})
        self.assertEqual(result["status"],"no_common_complete_source_cost_support")

    def test_source_periods_receive_equal_rank_weight_despite_price_scale(self):
        rows,ids,nodes,pairs=self.fixture()
        for r in rows:
            if r["period"]=="2023-10-15":
                r["cost_GBP"]={ids[0]:10.,ids[1]:300000.,ids[2]:None,ids[3]:100.}[r["profile_id"]]
        result=designs(rows,ids,nodes,pairs,["2023-07-15","2023-10-15"],[0,60],"conservative_admission",n0=2)
        self.assertEqual(result["source_first_center"],ids[0])
        self.assertEqual(result["source_mean_cost_percentile"][ids[0]],.25)
        self.assertEqual(result["source_mean_cost_percentile"][ids[1]],.5)
