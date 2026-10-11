import unittest

import numpy as np

from performance.check_historical_observed_batch_outer import outer_rule


class ObservedBatchOuterRuleTest(unittest.TestCase):
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
