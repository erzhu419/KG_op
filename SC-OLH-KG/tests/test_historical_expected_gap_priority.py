import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from performance import screen_historical_observed_batch_risk as screen


class ExpectedGapPriorityTest(unittest.TestCase):
    def test_group_score_averages_heads_and_excludes_admitted_actions(self):
        upper=np.zeros((16,3));upper[0,0]=.9;upper[:,1]=.1;upper[:,2]=.99
        groups=[dict(group=0,action_indices=[0,2]),dict(group=1,action_indices=[1])]
        scores=screen.expected_group_upper_gaps(upper,np.zeros(16),groups,[0,1])
        self.assertAlmostEqual(scores[0],.9/16)
        self.assertAlmostEqual(scores[1],.1)
        self.assertGreater(scores[1],scores[0])
        clipped=screen.expected_group_upper_gaps(upper,np.full(16,.10001),groups,[0,1])
        self.assertEqual(set(clipped),{0,1})
        self.assertEqual(clipped[1],0.)
        self.assertAlmostEqual(clipped[0],(.9-.10001)/16)

    def check_plan(self,old_summary):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def read(path):return json.loads((root/path).read_text())
            def save(path,data):
                p=root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data))
            rows=[]
            for record,gid,maximum,expected,weight in [('rare_head',0,.9,.9/16,1.),
                ('consistent_heads',1,.1,.1,1.),('small_probability',2,.5,.5,1e-4)]:
                rows.append(dict(record=record,kind='raw',candidate_groups=[gid],
                    candidate_upper_minus_reference={str(gid):maximum},outer_grid_weight=weight,
                    candidate_group_expected_positive_gap={str(gid):expected}))
            summary=dict(numerical_containment_passed=True,records=rows)
            save('plan.json',dict(output='out',maximum_estimated_cpu_seconds=602))
            initial=json.loads(json.dumps(summary))
            if old_summary:
                for row in initial['records']:row.pop('candidate_group_expected_positive_gap')
            save('out/summary.json',initial)
            groups=[dict(group=g,kind='guarded',action_indices=[g]) for g in range(3)]
            def price(row,*args):
                return row,[dict(identity=h,estimated_cpu_seconds=1) for h in [4,5]]
            with patch.object(screen,'ROOT',root),patch.object(screen,'read',read),\
                patch.object(screen,'save',save),patch.object(screen,'compressed_read',return_value=dict(actions=[])),\
                patch.object(screen,'action_groups',return_value=groups),\
                patch.object(screen,'reference_rule_timers',return_value=([],set())),\
                patch.object(screen,'price_reference_group',side_effect=price),\
                patch.object(screen,'collect',side_effect=lambda plan:save('out/summary.json',summary)) as collect,\
                contextlib.redirect_stdout(io.StringIO()):
                result=screen.next_plan('plan.json')
            self.assertEqual(result['next_joint_cases'][0]['record'],'consistent_heads')
            self.assertEqual(len(result['next_joint_cases']),1)
            self.assertEqual(result['expected_cpu_seconds_including600reserves'],602)
            self.assertEqual(sum(result['remaining_groups']['raw'].values()),3)
            self.assertEqual(result['next_joint_cases'][0]['weighted_expected_group_upper_gap'],.1)
            self.assertEqual(collect.call_count,1 if old_summary else 0)

    def test_pricing_prioritizes_expected_gap_with_original_probability_weights(self):
        self.check_plan(False)

    def test_existing_summary_is_rebuilt_from_cached_values_for_new_priority(self):
        self.check_plan(True)


if __name__=='__main__':
    unittest.main()
