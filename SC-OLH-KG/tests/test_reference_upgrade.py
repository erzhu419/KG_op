import copy
import unittest

import numpy as np

from performance.diagnose_historical_component_guarded import mixed_values
from performance.validate_historical_label_control import tables
from performance.validate_historical_label_groups import check
from performance.validate_historical_reference_upgrade import upgrade
from tests import test_label_groups


class ReferenceUpgradeTests(unittest.TestCase):
    def fixture(self):
        bundle,anchor,target=test_label_groups.LabelGroupsTests().fixture()
        for bank in (anchor,target):
            for rows in bank.values():
                for row in rows.values():row['cpu_seconds']=1.
        return bundle,anchor,target

    def test_finer_independent_reference_resolves_failed_residual_without_erasing_it(self):
        b,a,t=self.fixture();a[4]['F7']['statistics'][0][0]+=.1
        legacy=check(b,a,t,'raw',1e-4,True);saved=copy.deepcopy(legacy)
        r=upgrade(b,a,t,'raw',1e-4,legacy,'E8')
        self.assertEqual(legacy,saved)
        self.assertEqual(legacy['selected_method'],'direct256')
        self.assertEqual(r['selected_method'],'control32')
        self.assertGreater(r['legacy_check']['comparisons']['independent'],1e-4)
        self.assertTrue(r['hybrid_direct_admitted'])
        self.assertEqual(np.asarray(r['selected_statistics']).shape,(8,3,4))

    def test_candidate_error_against_finer_rule_still_requires_direct_fallback(self):
        b,a,t=self.fixture();a[4]['E8']['statistics'][0][0]+=.1
        r=upgrade(b,a,t,'raw',1e-4,check(b,a,t,'raw',1e-4,True),'E8')
        self.assertEqual(r['selected_method'],'direct256')
        self.assertGreater(r['candidate_checks']['control128']['direct_differences']['E9'],1e-4)

    def test_disagreeing_direct_reference_bank_cannot_admit_a_control_or_fallback(self):
        b,a,t=self.fixture();t[5]['H8']['statistics'][2][0]+=.05
        r=upgrade(b,a,t,'projected',1e-4,check(b,a,t,'projected',1e-4,True),'E8')
        self.assertEqual(r['selected_method'],'HOLD')
        self.assertFalse(r['hybrid_direct_admitted'])
        self.assertIsNone(r['selected_statistics'])

    def test_negative_coarse_probability_is_not_clipped(self):
        b,a,t=self.fixture();t[4]['E5']['statistics'][1][0]=-1.
        r=upgrade(b,a,t,'raw',1e-4,check(b,a,t,'raw',1e-4,True),'E8')
        self.assertFalse(r['candidate_checks']['control32']['valid_unclipped_ranges'])
        self.assertEqual(r['selected_method'],'control128')
        self.assertEqual(t[4]['E5']['statistics'][1][0],-1.)

    def test_normalization_is_checked_for_anchor_and_each_finer_reference(self):
        b,a,t=self.fixture();a[4]['E8']['diagnostics']['predictive_log_evidence_error']=.01
        r=upgrade(b,a,t,'raw',1e-4,check(b,a,t,'raw',1e-4,True),'E8')
        self.assertEqual(r['selected_method'],'direct256')
        a[4]['E8']['diagnostics']['predictive_log_evidence_error']=0.
        t[5]['F8']['diagnostics']['predictive_log_evidence_error']=.01
        r=upgrade(b,a,t,'raw',1e-4,check(b,a,t,'raw',1e-4,True),'E8')
        self.assertEqual(r['selected_method'],'HOLD')

    def test_pairwise_agreement_via_E8_is_insufficient_for_candidate_F8_agreement(self):
        b,a,t=self.fixture();tol=1e-4
        original=mixed_values(b,*tables(b,{h:t[h]['E8'] for h in (4,5)}),'raw')
        shifted=copy.deepcopy(t)
        shifted[4]['E8']['statistics']=(np.asarray(t[4]['E8']['statistics'])+1e-4).tolist()
        delta=abs(mixed_values(b,*tables(b,{h:shifted[h]['E8'] for h in (4,5)}),'raw')-original).max()
        step=1e-4*(.75*tol/delta)
        a[4]['E8']['statistics']=(np.asarray(a[4]['E8']['statistics'])+step).tolist()
        t[4]['F8']['statistics']=(np.asarray(t[4]['F8']['statistics'])-step).tolist()
        legacy=check(b,a,t,'raw',tol,True)
        self.assertEqual(legacy['selected_method'],'control32')
        r=upgrade(b,a,t,'raw',tol,legacy,'E8')
        self.assertTrue(r['direct_reference_checks']['passed'])
        self.assertLess(r['candidate_checks']['control32']['direct_differences']['E8'],tol)
        self.assertGreater(r['candidate_checks']['control32']['direct_differences']['F8'],tol)
        self.assertEqual(r['selected_method'],'direct256')


if __name__=='__main__':unittest.main()
