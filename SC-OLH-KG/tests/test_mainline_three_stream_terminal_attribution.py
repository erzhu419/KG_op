"""Tiny identity and reuse fixtures; no saved scientific cohort is analyzed."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from performance import mainline_three_stream_terminal_attribution as three
from tests.test_mainline_confirmation_terminal_attribution import fixture


def identified(third_point, registry):
    case, reference, items, saved, history, truth = fixture()
    reference.update(arm=three.AFTER, history_arm=three.AFTER)
    reference['shortlist'][2] = three.AFTER + ':history:10'
    for item in items.values():
        item['frozen_shortlist'][2]['point'] = [third_point]
    saved['new']['deployed'] = [third_point]
    history[-1]['point'] = [third_point]
    truth[(third_point,)] = truth.pop((2,))
    arm = three.terminal.analyze_pair(case, reference, items, saved, history, truth)
    task = dict(seed=1, domain='fixture', status='complete', quality_reference_status='defined', arms=[arm])
    return three.attach_policy_ids(task, {three.AFTER: history},
        {three.AFTER: [[0], [1], [third_point]]}, registry)


class ThreeStreamTerminalTests(unittest.TestCase):
    def test_dynamic_same_label_different_physical_policy_does_not_merge(self):
        registry = {}
        first, second, third = (identified(point, registry) for point in (2, 99, 2))
        a, b, c = (task['arms'][0]['candidates'][2] for task in (first, second, third))
        self.assertEqual(a['policy'], b['policy'])
        self.assertNotEqual(a['policy_id'], b['policy_id'])
        self.assertEqual(a['policy_id'], c['policy_id'])
        streams = [dict(name=name, terminal=task) for name, task in (('old', first), ('new', second))]
        paths = three.exposure_paths(streams)
        tracked = next(row for row in paths['policies'] if row['policy_id'] == a['policy_id'])
        self.assertEqual(tracked['paths'][0]['observed_at'], [10])
        self.assertEqual(tracked['paths'][1]['observed_at'], [])
        self.assertIsNone(tracked['paths'][1]['shortlisted_slot'])
        self.assertIsNone(tracked['paths'][1]['new']['safety']['passed'])

    def test_actual_third_opponent_identity_survives_compact(self):
        terminal = identified(2, {})
        result = dict(seed=1, domain='fixture', status='complete', streams=[dict(
            name='original', reused=True, reused_file='old.json', source_output_dir='old', terminal=terminal)],
            exposure=dict(policies=[]), registered_negatives=[], new_work=dict(three.terminal.ZERO))
        compact = three.compact_task(result)
        arm = compact['streams'][0]['terminal']['arms'][0]
        comparison = arm['protocols']['new']['comparisons'][1]
        self.assertEqual(comparison['incumbent_policy_id'], arm['candidates'][0]['policy_id'])
        self.assertNotEqual(comparison['incumbent_policy_id'], arm['candidates'][1]['policy_id'])
        self.assertEqual(arm['protocols']['new']['deployed_policy_id'], arm['candidates'][2]['policy_id'])
        self.assertIsNone(arm['candidates'][2]['old']['passed'])

    def test_original_stream_is_read_from_existing_attribution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'stage66/cases/seed_1/analysis.json'
            source.parent.mkdir(parents=True)
            source.write_text(json.dumps(dict(terminal=dict(seed=1, arms=[]))))
            with patch.object(three.terminal, 'analyze_task') as analyze:
                result, reused, file = three.load_terminal(root, dict(reuse_attribution_dir='stage66'), 1)
            self.assertEqual(result['seed'], 1)
            self.assertTrue(reused)
            self.assertEqual(file, 'stage66/cases/seed_1/analysis.json')
            analyze.assert_not_called()


if __name__ == '__main__':
    unittest.main()
