import unittest
import numpy as np
from performance.compare_source_first_remainders import random_remainder, fixed_first_portfolio
from performance.submission_revision_controls import greedy_portfolio_indices

class RemainderControls(unittest.TestCase):
    def test_random_keeps_same_first_and_samples_distinct_library_members(self):
        ids=[f'profile_{i:04d}' for i in range(64)];first=ids[17]
        chosen=random_remainder(ids,first,20260930,800000000)
        self.assertEqual(chosen[0],first);self.assertEqual(len(chosen),10)
        self.assertEqual(len(set(chosen)),10);self.assertTrue(set(chosen)<=set(ids))
        self.assertEqual(chosen,random_remainder(ids,first,20260930,800000000))
        self.assertNotEqual(chosen,random_remainder(ids,first,20260930,800000001))

    def test_portfolio_preserves_original_when_first_matches_and_forces_tied_first(self):
        scores=np.array([[0.,0.,.5,.4],[0.,0.,.5,.4]])
        original,_=greedy_portfolio_indices(scores,3)
        self.assertEqual(fixed_first_portfolio(scores,original[0],3),list(original))
        self.assertEqual(fixed_first_portfolio(scores,1,3),[1,0,3])

if __name__=='__main__':unittest.main()
