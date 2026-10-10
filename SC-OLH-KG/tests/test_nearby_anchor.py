import unittest

from performance.validate_historical_nearby_anchor import nearest_record, priced_row, finer_reference_differences
from performance.validate_historical_label_groups import check
from tests import test_label_groups


class NearbyAnchorTests(unittest.TestCase):
    def test_nearest_uses_observed_label_not_cdf_or_origin(self):
        target=dict(label=-.25,predictive_cdf=.05)
        candidates=dict(old=dict(label=-.16,predictive_cdf=.06),
            nearby=dict(label=-.20,predictive_cdf=.22),low=dict(label=-.36,predictive_cdf=.001))
        self.assertEqual(nearest_record(target,candidates),'nearby')

    def test_price_excludes_reuse_and_uses_slower_matching_anchor(self):
        row=dict(rules=['E5','E8'])
        cached={4:dict(E5=dict(cpu_seconds=200.)),5:{}}
        old={h:dict(E5=dict(cpu_seconds=2.),E8=dict(cpu_seconds=20.)) for h in (4,5)}
        new={h:dict(E5=dict(cpu_seconds=3.),E8=dict(cpu_seconds=10.)) for h in (4,5)}
        price=priced_row(row,[4,5],cached,[old,new])
        self.assertEqual(price['cached_cpu_seconds'],200.)
        self.assertAlmostEqual(price['estimated_new_cpu_seconds'],1.2*(20+3+20+2))

    def test_two_anchors_share_direct_target_but_can_require_different_fallbacks(self):
        bundle,old,target=test_label_groups.LabelGroupsTests().fixture()
        _,near,_=test_label_groups.LabelGroupsTests().fixture()
        old[4]['E7']['statistics'][0][0]+=.1
        a=check(bundle,old,target,'raw',1e-4,True)
        b=check(bundle,near,target,'raw',1e-4,True)
        self.assertEqual(a['direct_reference_checks'],b['direct_reference_checks'])
        self.assertEqual(a['selected_method'],'direct256')
        self.assertEqual(b['selected_method'],'control32')

    def test_independent_finer_reference_is_compared_to_candidate_itself(self):
        bundle,anchor,target=test_label_groups.LabelGroupsTests().fixture()
        target[4]['F8']['statistics'][0][0]+=.1
        differences=finer_reference_differences(bundle,anchor,target,'raw')
        self.assertLess(differences['E8'],1e-12)
        self.assertLess(differences['H8'],1e-12)
        self.assertGreater(differences['F8'],1e-4)


if __name__=='__main__':unittest.main()
