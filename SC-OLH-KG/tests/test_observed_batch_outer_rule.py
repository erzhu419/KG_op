import unittest

import numpy as np

from performance.check_historical_observed_batch_outer import outer_rule


class ObservedBatchOuterRuleTest(unittest.TestCase):
    def test_local_variance_refinement_preserves_cached_coordinates_and_probability_mass(self):
        intervals=[[0.,.8],[.8,.95],[.95,.99],[.99,1.]]
        coarse,cw=outer_rule(4,4,intervals)
        fine,fw=outer_rule(4,4,intervals,variance_orders=[4,4,8,4])
        saved={tuple(p):w for p,w in zip(coarse,cw) if not .95<p[0]<.99}
        reused={tuple(p):w for p,w in zip(fine,fw) if not .95<p[0]<.99}
        self.assertEqual(saved,reused)
        self.assertEqual(len(fine),80)
        self.assertTrue(np.all(fw>0))
        self.assertAlmostEqual(fw[(fine[:,0]>.95)&(fine[:,0]<.99)].sum(),.04,places=14)
        for a,b in [(0,0),(1,1),(2,3),(4,6)]:
            self.assertAlmostEqual(float(fw@(fine[:,0]**a*fine[:,1]**b)),1./((a+1)*(b+1)),places=13)

    def test_explicit_uniform_variance_orders_reproduce_historical_rule(self):
        intervals=[[0.,.8],[.8,.95],[.95,.99],[.99,1.]]
        old,ow=outer_rule(4,4,intervals)
        current,cw=outer_rule(4,4,intervals,variance_orders=[4]*4)
        np.testing.assert_array_equal(old,current)
        np.testing.assert_array_equal(ow,cw)

    def test_positive_shards_preserve_joint_moments_without_renormalizing(self):
        intervals=[[0.,.9],[.9,1.]]
        full,weights=outer_rule(16,64,intervals)
        shards=[outer_rule(16,64,intervals,shard,2) for shard in range(2)]
        self.assertEqual(len(full),2048)
        self.assertTrue(all(len(p)==1024 and np.all(w>0) for p,w in shards))
        self.assertTrue(all(abs(w.sum()-.5)<1e-14 for _,w in shards))
        self.assertEqual(set(map(tuple,full)),set(map(tuple,np.concatenate([p for p,_ in shards]))))
        for a,b in [(0,0),(1,1),(2,3),(4,6)]:
            exact=1./((a+1)*(b+1))
            joint=float(weights@(full[:,0]**a*full[:,1]**b))
            split=sum(float(w@(p[:,0]**a*p[:,1]**b)) for p,w in shards)
            self.assertAlmostEqual(joint,exact,places=13)
            self.assertAlmostEqual(split,exact,places=13)


if __name__=='__main__':
    unittest.main()
