"""Fresh confirmation interface checks; established metric math is not rerun."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from performance import summarize_mainline_two_comparison_confirmation as summary


class FreshConfirmationSummaryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.plan = dict(output_dir='confirmation',
            tasks=[dict(geometry_task_seed=i,domain='synthetic') for i in range(24)],
            terminal_arms=[dict(name=f'arm_{i}') for i in range(9)])
        self.cases = [dict(seed=seed,arm=f'arm_{arm}',domain='synthetic',
            old=dict(status='missing',failure=dict(reason='synthetic search failure')),
            new=dict(status='missing',failure=dict(reason='synthetic search failure')))
            for seed in range(24) for arm in range(9)]

    def seal(self, status='sealed'):
        (self.root/'confirmation').mkdir(exist_ok=True)
        (self.root/'confirmation/execution_status.json').write_text(json.dumps(dict(status=status,cases=self.cases)))

    def test_not_globally_sealed_blocks_all_truth_collection(self):
        self.seal('running')
        with patch.object(summary,'ROOT',self.root), patch.object(summary,'collect_task_truth') as collect:
            with self.assertRaisesRegex(RuntimeError,'globally sealed'):
                summary.score(self.plan)
            collect.assert_not_called()

    def test_new_missing_complete_record_blocks_even_with_old_sealed(self):
        self.cases[-1]['new'] = dict(status='complete',input_file='missing_input.json',
                                    verification_file='missing_new_verification.json')
        self.seal()
        with patch.object(summary,'ROOT',self.root), patch.object(summary,'collect_task_truth') as collect:
            with self.assertRaisesRegex(RuntimeError,'absent'):
                summary.score(self.plan)
            collect.assert_not_called()

    def test_all432_outcomes_remain_and_completed_old_is_not_deleted(self):
        item = dict(seed=0,arm='arm_0',source_case='search/0/arm_0',
                    frozen_shortlist=[dict(point=[i]) for i in range(3)])
        saved = dict(deployed=[0],new_verification_calls=224,
            verification=dict(certified=True,verification_budget=224,total_evaluation_count=237))
        (self.root/'input.json').write_text(json.dumps(item))
        (self.root/'old.json').write_text(json.dumps(saved))
        self.cases[0]['old'] = dict(status='complete',input_file='input.json',verification_file='old.json')
        self.seal()
        with patch.object(summary,'ROOT',self.root):
            _, cases = summary.load_sealed(self.plan)
        context = dict(cache={(0,):dict(true_feasible=True,quality=.8)},best=dict(quality=.5),
            reference_status='defined',histories={'search/0/arm_0':{(i,):f'initial:{i}' for i in range(3)}})
        rows = [summary.score_pair(case,context) for case in cases]
        outcomes = [summary.outcome_record(case,p,row[p]) for case,row in zip(cases,rows)
                    for p in ('old','new')]
        self.assertEqual((len(rows),len(outcomes)),(216,432))
        self.assertEqual(rows[0]['old']['C'],1)
        self.assertAlmostEqual(rows[0]['old']['G'],.3)
        self.assertIsNone(rows[0]['new']['C'])
        self.assertIsNone(rows[0]['new_minus_old']['G'])
        self.assertEqual(sum(row['status']=='complete' for row in outcomes),1)

    def test_fresh_labels_and_fresh_initial_feasibility(self):
        initial = [dict(point=[i]) for i in range(10)]
        history = initial+[dict(point=[10]),dict(point=[11])]
        labels = summary.policy_labels(initial,[[0],[10]],history,'history_arm')
        self.assertEqual(labels[(0,)],'initial:0')
        self.assertEqual(labels[(10,)],'neighbor:1')
        self.assertEqual(labels[(11,)],'history_arm:history:11')
        cache = {(i,):dict(true_feasible=False,quality=.9,objective=.1) for i in range(10)}
        self.assertEqual(summary.initial_reference(initial,cache),(None,'no_true_feasible_initial'))
        cache[(4,)]['true_feasible'] = True
        self.assertEqual(summary.initial_reference(initial,cache),(cache[(4,)],'defined'))

    def test_largest_contribution_sensitivity_preserves_primary_rows(self):
        rows = [dict(seed=i,new_minus_old={'G':v}) for i,v in enumerate((3.,-1.,1.))]
        original = copy.deepcopy(rows)
        result = summary.leave_largest_positive(rows,'new_minus_old','G')
        self.assertEqual(result['excluded_seed'],0)
        self.assertEqual(result['all_registered_leave_one_out_mean'],0)
        self.assertEqual(result['conditional_direction'],'zero')
        self.assertFalse(result['primary_sample_changed'])
        self.assertEqual(rows,original)
        rows.append(dict(seed=3,new_minus_old={'G':None}))
        self.assertIsNone(summary.leave_largest_positive(rows,'new_minus_old','G')['all_registered_leave_one_out_mean'])

    def test_dependent_new_missing_does_not_duplicate_old_failure_cost(self):
        case = dict(seed=0,domain='synthetic',arm='arm_0',new=dict(status='missing',
            failure_file='old_failure.json',record=dict(new_verification_calls=80,error='synthetic')))
        result = summary.outcome_record(case,'new',summary.metrics(failed=True))
        self.assertEqual(result['costs']['newly_charged_physical_calls'],0)
        self.assertNotIn('new_verification_calls',result['costs'])
        self.assertEqual(result['failure']['new_verification_calls'],80)


if __name__ == '__main__':
    unittest.main()
