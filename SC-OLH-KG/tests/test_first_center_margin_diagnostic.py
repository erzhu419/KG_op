import unittest
import numpy as np
from performance.diagnose_first_center_margin_transfer import margin_bound

class MarginDiagnosticTests(unittest.TestCase):
    def test_safe_near_tie_and_unsafe_case(self):
        z=np.array([[0.],[.1],[1.]])
        result=margin_bound(z,np.array([-1.,-.9,0.]),np.array([.05,0.,1.]),np.array([0.,.05,1.]))
        self.assertEqual(result['compatible_set_size'],2)
        self.assertTrue(result['sufficient_condition_holds'])
        result=margin_bound(z,np.array([.1,.2,1.]),np.array([.05,0.,1.]),np.array([0.,.05,1.]))
        self.assertFalse(result['sufficient_condition_holds'])

    def test_coordinate_collision_precludes_finite_bound(self):
        result=margin_bound(np.zeros((2,1)),np.array([-1.,1.]),np.array([0.,.1]),np.array([0.,.1]))
        self.assertTrue(result['coordinate_collision'])
        self.assertIsNone(result['finite_library_lipschitz'])
        self.assertFalse(result['sufficient_condition_holds'])

if __name__=='__main__': unittest.main()
