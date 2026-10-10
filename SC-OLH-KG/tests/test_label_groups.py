import unittest

import numpy as np

from performance.validate_historical_label_groups import check


def row(value):
    return dict(statistics=np.full((3,4),value).tolist(),costs=[300.]*3,
        diagnostics=dict(predictive_log_evidence_error=0.))


class LabelGroupsTests(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(84)
        bundle=dict(utility_identities=[4,5],prior_weights=[.125]*8,actions=[{},{},{}],
            head_log_scores=rng.normal(size=8).tolist(),opposite=[dict(
                raw=rng.uniform(size=(8,3,4)).tolist(),projected=rng.uniform(size=(8,3,4)).tolist(),
                head_log_scores=rng.normal(size=8).tolist()) for _ in range(4)])
        names=['E5','E7','F7','G7','E8','F8','H8','E9']
        anchor={h:{n:row(.1) for n in names} for h in (4,5)}
        target={h:{n:row(.2) for n in names} for h in (4,5)}
        return bundle,anchor,target

    def test_comparative_pass_does_not_imply_direct_admission(self):
        b,a,t=self.fixture();r=check(b,a,t,'raw',1e-4,False)
        self.assertTrue(r['control_rule_diagnostics_passed'])
        self.assertFalse(r['direct_control_admitted'])
        self.assertIsNone(r['direct_reference_checks'])

    def test_direct_reference_supersedes_a_failed_coarse_rule(self):
        b,a,t=self.fixture();t[4]['E5']['statistics'][1][0]+=.1
        self.assertFalse(check(b,a,t,'raw',1e-4,False)['control_rule_diagnostics_passed'])
        r=check(b,a,t,'raw',1e-4,True)
        self.assertGreater(r['comparisons']['coarse'],1e-4)
        self.assertTrue(r['direct_control_admitted'])
        self.assertFalse(r['coarse_direct_control_admitted'])
        t[4]['E5']['statistics'][1][0]=-1.
        self.assertTrue(check(b,a,t,'raw',1e-4,True)['direct_control_admitted'])

    def test_one_native_identity_budget_failure_blocks_reference_admission(self):
        b,a,t=self.fixture();t[5]['H8']['statistics'][2][0]+=.05
        r=check(b,a,t,'projected',1e-4,True)
        self.assertTrue(r['control_rule_diagnostics_passed'])
        self.assertFalse(r['direct_reference_checks']['passed'])
        self.assertFalse(r['direct_control_admitted'])

    def test_coarse_rule_requires_its_own_direct_comparison(self):
        b,a,t=self.fixture();r=check(b,a,t,'raw',1e-4,True)
        self.assertTrue(r['coarse_direct_control_admitted'])
        self.assertEqual(set(r['coarse_direct_differences']),{'E8','E9'})

    def test_direct_disagreement_or_negative_probability_is_not_admitted(self):
        b,a,t=self.fixture();t[5]['E9']['statistics'][0][0]+=.05
        self.assertFalse(check(b,a,t,'raw',1e-4,True)['direct_control_admitted'])
        t[4]['E7']['statistics'][0][0]=-1.
        r=check(b,a,t,'raw',1e-4,True)
        self.assertFalse(r['valid_unclipped_ranges'])
        self.assertFalse(r['direct_control_admitted'])

    def test_failed_control_selects_validated_direct_fallback(self):
        b,a,t=self.fixture();t[4]['E7']['statistics'][1][0]+=.1
        r=check(b,a,t,'raw',1e-4,True)
        self.assertFalse(r['control_rule_diagnostics_passed'])
        self.assertFalse(r['direct_control_admitted'])
        self.assertEqual(r['selected_method'],'direct256')
        self.assertTrue(r['hybrid_direct_admitted'])
        self.assertEqual(np.asarray(r['selected_statistics']).shape,(8,3,4))
        np.testing.assert_array_equal(r['selected_statistics'][4],t[4]['E8']['statistics'])
        np.testing.assert_array_equal(r['selected_costs'][5],t[5]['E8']['costs'])

    def test_finer_only_resolves_coarse_check_without_direct_admission(self):
        b,a,t=self.fixture();t[4]['E5']['statistics'][1][0]+=.1
        r=check(b,a,t,'raw',1e-4,False,True)
        self.assertTrue(r['control_rule_diagnostics_passed'])
        self.assertEqual(r['selected_method'],'control128')
        self.assertFalse(r['hybrid_direct_admitted'])
        self.assertEqual(set(r['direct_differences']),{'E8'})
        t[4]['E8']['diagnostics']['predictive_log_evidence_error']=.01
        self.assertFalse(check(b,a,t,'raw',1e-4,False,True)['control_rule_diagnostics_passed'])


if __name__=='__main__':unittest.main()
