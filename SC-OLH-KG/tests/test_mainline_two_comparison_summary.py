"""Synthetic checks for registered denominators, attribution, and the truth barrier."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from performance import summarize_mainline_two_comparison_development as summary


class MetricsTests(unittest.TestCase):
    def test_undefined_initial_baseline_stays_null_even_without_certificate(self):
        result = summary.metrics(False, None, None)
        self.assertEqual((result['A'], result['C'], result['R']), (0, 0, 0))
        self.assertEqual([result[k] for k in ('G', 'P', 'L')], [None]*3)

    def test_false_certificate_is_not_safe_reward(self):
        result = summary.metrics(True, {'true_feasible':False, 'quality':.9}, {'quality':.5})
        self.assertEqual((result['A'], result['C'], result['false_certificate'], result['R'], result['G']),
                         (1, 0, 1, 0, 0))

    def test_missing_execution_is_not_zero(self):
        self.assertTrue(all(value is None for value in summary.metrics(failed=True).values()))

    def test_signed_regression_and_two_terms_add_exactly(self):
        old = summary.metrics(True, {'true_feasible':True, 'quality':.7}, {'quality':.6})
        baseline = summary.metrics(True, {'true_feasible':True, 'quality':.5}, {'quality':.6})
        new = summary.metrics(True, {'true_feasible':True, 'quality':.8}, {'quality':.6})
        result = summary.decompose(old, new, baseline)
        self.assertAlmostEqual(baseline['G'], -.1)
        self.assertAlmostEqual(baseline['L'], .1)
        self.assertAlmostEqual(result['new_minus_old']['G'], .1)
        self.assertAlmostEqual(result['third_challenge']['G'], .3)
        self.assertAlmostEqual(result['prechange']['G'], -.2)
        for metric in summary.METRICS:
            self.assertAlmostEqual(result['new_minus_old'][metric],
                result['third_challenge'][metric]+result['prechange'][metric])

    def test_original_third_only_fallback_is_not_extra_exposure(self):
        verification = dict(initial_stage_winner_index=None, selected_shortlist_rank=3, certified=True)
        self.assertEqual(summary.intermediate_index(verification), (2,True,'original_third_only_fallback'))
        old = summary.metrics(False, None, {'quality':.6})
        new = summary.metrics(True, {'true_feasible':True, 'quality':.8}, {'quality':.6})
        result = summary.decompose(old, new, new)
        self.assertEqual(result['third_challenge']['G'], 0)
        self.assertAlmostEqual(result['prechange']['G'], .2)

    def test_existing_first_stage_winner_is_B_even_after_third_wins(self):
        verification = dict(initial_stage_winner_index=1, selected_shortlist_rank=3, certified=True)
        self.assertEqual(summary.intermediate_index(verification), (1,True,'first_stage_winner'))

    def test_all24_missing_means_and_conditional23_or22(self):
        for missing in (1,2):
            result = summary.estimate([.1]*(24-missing)+[None]*missing)
            self.assertEqual(result['registered_geometries'], 24)
            self.assertIsNone(result['all_registered_mean'])
            self.assertIsNone(result['all_registered_descriptive_95pct_t_interval'])
            self.assertEqual(result['conditional_available']['n'], 24-missing)
            self.assertAlmostEqual(result['conditional_available']['mean'], .1)

    def test_binary_bound_does_not_impute_shared_missing_failure(self):
        rows = [dict(old={'C':0}, new={'C':0}) for _ in range(23)]
        rows[0]['new']['C'] = 1
        rows.append(dict(old={'C':None}, new={'C':None}))
        self.assertEqual(summary.certification_bounds(rows), [0.,2/24])


class BarrierTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.plan = dict(output_dir='output', block_inventory='inventory.json')
        self.inventory = dict(rows=[dict(seed=1,arm='arm')], available_cells=1, registered_cells=2)
        (self.root/'inventory.json').write_text(json.dumps(self.inventory))

    def test_absent_seal_blocks_before_any_label_file_read(self):
        calls = []
        real_read = summary.read
        def read(path):
            calls.append(Path(path).name)
            return real_read(path)
        with patch.object(summary, 'ROOT', self.root), patch.object(summary, 'read', side_effect=read), \
             patch.object(summary, 'load_cached_truth') as truth:
            with self.assertRaisesRegex(RuntimeError, 'sealed'):
                summary.score(self.plan)
            truth.assert_not_called()
        self.assertEqual(calls, ['inventory.json'])

    def test_sealed_missing_record_still_blocks_labels(self):
        (self.root/'output').mkdir()
        seal = dict(status='sealed', available_cases=1, registered_cases=2,
                    cases=[dict(seed=1,arm='arm',status='complete',verification_file='absent.json')])
        (self.root/'output/execution_status.json').write_text(json.dumps(seal))
        with patch.object(summary, 'ROOT', self.root), patch.object(summary, 'load_cached_truth') as truth:
            with self.assertRaisesRegex(RuntimeError, 'absent'):
                summary.score(self.plan)
            truth.assert_not_called()

    def test_recorded_failure_counts_as_sealed_not_as_success(self):
        (self.root/'output').mkdir()
        (self.root/'failure.json').write_text(json.dumps(dict(status='failed',error='synthetic')))
        seal = dict(status='sealed',available_cases=1,registered_cases=2,
                    cases=[dict(seed=1,arm='arm',status='failed',failure_file='failure.json')])
        (self.root/'output/execution_status.json').write_text(json.dumps(seal))
        with patch.object(summary, 'ROOT', self.root):
            _, records = summary.sealed_execution(self.plan,self.inventory)
        self.assertEqual(records[1,'arm'][0]['status'],'failed')


if __name__ == '__main__':
    unittest.main()
