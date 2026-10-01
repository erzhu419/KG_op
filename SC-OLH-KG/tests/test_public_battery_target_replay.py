import unittest
from performance.replay_public_battery_target_search import evaluate, sample_starts, shortlist

class ReplayTests(unittest.TestCase):
    def test_null_failures_are_not_cheap_objectives(self):
        records=[dict(profile_id='fail',window_success=False,cost_GBP=None,evaluation_index=0),
                 dict(profile_id='safe',window_success=True,cost_GBP=100,evaluation_index=1)]
        self.assertEqual(shortlist(records,2)[0]['profile_id'],'safe')
        records[0]['cost_GBP']=0
        with self.assertRaises(ValueError):shortlist(records,2)

    def test_verification_is_separate_stream_with_replacement(self):
        a=sample_starts([0,60,120],7,3211,0,80)
        b=sample_starts([0,60,120],7,1112100431,0,80)
        self.assertNotEqual(a,b)
        self.assertEqual(a,sample_starts([0,60,120],7,3211,0,80))
        self.assertLess(len(set(b)),len(b))

    def test_safe_certificate_and_failed_abstention_keep_exact_budget(self):
        source=dict(full_library_profiles=['a','b','c','d'],designs={'source':['a','b']})
        p=dict(seed_base=7,periods=['2024'],continuation_stream=330107,search_budget=3,
               window_start_minutes=[0,60],dispatch_rule='rule',search_stream=3211,
               verification_stream=1112100431,shortlist_size=3,verification_trials=80,
               familywise_delta=.05,failure_probability=.05)
        task=dict(period='2024',algorithm_seed=0,arm='source')
        for success in (True,False):
            cells={('2024',minute,profile,'rule'):dict(window_success=success,cost_GBP=10 if success else None)
                   for minute in [0,60] for profile in source['full_library_profiles']}
            result=evaluate(task,p,source,cells)
            self.assertEqual(result['represented_search_queries'],3)
            self.assertEqual(result['represented_verification_queries'],80 if success else 240)
            self.assertEqual(result['certified'],success)
            self.assertFalse(result['false_certificate'])
            self.assertEqual(result['new_controller_window_evaluations'],0)
            self.assertEqual(len(set(r['profile_id'] for r in result['observations'])),3)
