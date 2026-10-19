"""Small saved-record substitutes for two-stream pairing and the truth barrier."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from performance import summarize_mainline_confirmation_seed_replication as summary


def metric(gain):
    return dict(A=1, C=1, false_certificate=0, R=.5 + gain, G=gain,
                P=max(gain, 0.), L=max(-gain, 0.))


def row(seed, gain, arm=summary.AFTER):
    old, new = metric(0.), metric(gain)
    delta, zero = summary.difference(new, old), summary.difference(old, old)
    return dict(seed=seed, domain='fixture', arm=arm, history_arm=arm, status='complete',
        quality_baseline_defined=True, quality_reference_status='defined', best_initial_quality=.5,
        old=old, new=new, B=old, new_minus_old=delta, third_challenge=delta, prechange=zero,
        old_reported_deployed='initial:0', new_reported_deployed=f'{arm}:history:10', B_deployed='initial:0')


class SeedReplicationSummaryTests(unittest.TestCase):
    def test_second_unsealed_stream_blocks_all_scoring_and_cache_preload(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(summary, 'ROOT', Path(directory)), \
                patch.object(summary, 'children', return_value=[({}, {}), ({}, {})]), \
                patch.object(summary.single, 'load_sealed', side_effect=[None, RuntimeError('unsealed')]), \
                patch.object(summary.single, 'score') as score, patch.object(summary, 'preload_truth') as preload:
            with self.assertRaisesRegex(RuntimeError, 'unsealed'):
                summary.score_streams(dict(output_dir='audit'))
            score.assert_not_called()
            preload.assert_not_called()

    def test_streams_are_averaged_before_geometry_estimation(self):
        streams = [summary.origin_rows([row(1, 1.), row(2, 2.)]),
                   summary.origin_rows([row(1, 3.), row(2, 4.)])]
        averaged = summary.geometry_average(streams)
        self.assertEqual([r['new_minus_old']['G'] for r in averaged], [2., 3.])
        result = summary.summarize_unit(averaged, [summary.AFTER])
        estimate = result['primary']['new_minus_old']['G']
        self.assertEqual(estimate['registered_geometries'], 2)
        self.assertEqual(estimate['conditional_available']['degrees_of_freedom'], 1)
        self.assertEqual(estimate['conditional_available']['mean'], 2.5)

    def test_missing_one_stream_is_not_replaced_by_other_stream(self):
        a, b = row(1, 1.), row(1, 3.)
        b['new_minus_old']['G'] = None
        b['new']['G'] = None
        result = summary.geometry_average([summary.origin_rows([a]), summary.origin_rows([b])])[0]
        self.assertIsNone(result['new_minus_old']['G'])
        self.assertEqual(result['new_minus_old']['C'], 0.)

    def test_origin_allocation_keeps_false_certificate_and_signed_loss(self):
        loss = row(1, -.1)
        false = row(2, 0.)
        false['new'].update(A=1, C=0, false_certificate=1, R=0., G=0., P=0., L=0.)
        enriched = summary.origin_rows([loss, false])
        self.assertEqual(enriched[0]['origin_contributions']['dynamic']['new']['G'], -.1)
        self.assertEqual(enriched[0]['origin_contributions']['dynamic']['new']['L'], .1)
        self.assertEqual(enriched[1]['origin_contributions']['dynamic']['new']['false_certificate'], 1)
        self.assertEqual(enriched[1]['origin_contributions']['dynamic']['new']['G'], 0.)
        self.assertEqual(enriched[1]['origin_contributions']['initial']['new']['G'], 0.)

    def test_compact_unit_retains_primary_secondary_and_origin_counts(self):
        rows = summary.origin_rows([row(seed, gain, arm) for seed, gain in ((1, .1), (2, .2))
            for arm in (summary.AFTER, summary.POINT, summary.FULL)])
        unit = summary.summarize_unit(rows, [summary.AFTER, summary.POINT, summary.FULL])
        compact = summary.compact_unit(unit)
        self.assertEqual(compact['arms'][0]['new_minus_old']['G']['n'], 2)
        self.assertIn('third_challenge', compact['arms'][0])
        self.assertEqual(compact['tasks'][0]['AFTER_minus_POINT']['G'], 0.)
        self.assertEqual(compact['origins'][0]['dynamic']['new_true_feasible_deployments']['total'], 2)
        self.assertEqual(compact['tasks'][1]['primary']['origin_G']['dynamic'], .2)

    def test_truth_reuse_checks_geometry_and_exact_initial_points(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(summary, 'ROOT', Path(directory)):
            root = Path(directory)
            task = dict(geometry_task_seed=1, domain='fixture', geometry={'shift': .1}, optimizer_seed=1)
            plans = [dict(output_dir=name, tasks=[copy.deepcopy(task)]) for name in ('old', 'new')]
            for plan in plans:
                out = root / plan['output_dir']
                (out / 'seed_1' / 'arm').mkdir(parents=True)
                (out / 'execution_status.json').write_text(json.dumps(dict(cases=[dict(seed=1,
                    source_case=str(Path(plan['output_dir']) / 'seed_1' / 'arm'))])))
                (out / 'seed_1' / 'initialization.json').write_text(json.dumps(dict(history=[dict(point=[0])])) )
            old_truth = root / 'old/seed_1/truth.json'
            old_truth.write_text(json.dumps([dict(point=[0], truth={'quality': .5}),
                                             dict(point=[2], truth={'quality': .7})]))
            result = summary.preload_truth(plans[1], [plans[0]])
            self.assertEqual(result['added_policies'], 2)
            self.assertEqual(result['new_truth_function_calls'], 0)
            (root / 'new/seed_1/initialization.json').write_text(json.dumps(dict(history=[dict(point=[9])])))
            with self.assertRaisesRegex(ValueError, 'initial physical'):
                summary.preload_truth(plans[1], [plans[0]])


if __name__ == '__main__':
    unittest.main()
