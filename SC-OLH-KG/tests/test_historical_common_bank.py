"""Index/raw-unit/budget checks; no numerical backend or integration runs."""

import json
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from performance.historical_common_bank import (common_bank_mapping, initialization_batches,
    legacy_search_batch, remaining_target_calls, replay_raw_batches, restore_zero_paid_prior)

ROOT = Path(__file__).resolve().parents[1]


class RecordingModel:
    def __init__(self, count=73):
        self.paid_episodes = 0
        self.history = []
        self.components = [(SimpleNamespace(candidate_count=count, paid_episodes=0),)*2]

    def update_raw(self, index, row):
        self.history.append(('single', index, row))
        self.paid_episodes += 1

    def update_replicates(self, index, rows):
        self.history.append(('batch', index, rows))
        self.paid_episodes += len(rows)


class CommonBankTests(unittest.TestCase):
    def test_saved_layout_queries_preserve_physical_policies(self):
        data = json.loads((ROOT/'paper_artifacts/historical_independent_kg_preflight_v1/frozen_inputs.json').read_text())
        public = json.loads((ROOT/'paper_artifacts/legacy_related_inventory_source_v1/frozen_inputs.json').read_text())['bank']
        expected = {'5': [24, 52, 11, 42, 45, 10], '6': [53, 22, 51, 33, 45, 10]}
        for bank in data['banks']:
            mapped = common_bank_mapping(public, bank)
            self.assertEqual(mapped['candidate_bank_indices'], list(range(73)))
            self.assertEqual(mapped['queries'], expected[bank['layout']])

    def test_saved_initialization_is_two_single_updates_at45_and10(self):
        record = json.loads((ROOT/'paper_artifacts/historical_independent_kg_preflight_v1/initializations.json').read_text())['rows'][0]
        model = RecordingModel()
        result = replay_raw_batches(model, initialization_batches(record), record['tau'])
        self.assertEqual([(kind, index) for kind, index, _ in model.history], [('single', 45), ('single', 10)])
        self.assertEqual(result, {'paid_episodes': 2, 'remaining_calls': 360,
                                  'replayed_batches': 2, 'new_physical_calls': 0})
        with self.assertRaises(ValueError):
            replay_raw_batches(model, initialization_batches(record), record['tau'])

    def test_saved_search_uses_raw_g_and_maps_old_local_index(self):
        saved = {'decision': {'query': 0}, 'raw_observation': [.3, 1.2],
                 'centered_observation': [.3, .7], 'physical_calls': 1}
        model = RecordingModel()
        replay_raw_batches(model, [legacy_search_batch(saved, [45, 10])], tau=.5)
        self.assertEqual(model.history, [('single', 45, [.3, .7])])

    def test_four_repeats_keep_one_batch_and_debit_four_not_one(self):
        rows = [[.1, .6], [.2, .7], [.3, .8], [.4, .9]]
        batches = [{'public_policy_index': 45, 'raw_observations': [[.2, .6]]},
                   {'public_policy_index': 10, 'raw_observations': [[.4, .8]]},
                   {'public_policy_index': 63, 'raw_observations': rows}]
        model = RecordingModel()
        result = replay_raw_batches(model, batches, tau=.5)
        self.assertEqual([kind for kind, _, _ in model.history], ['single', 'single', 'batch'])
        self.assertEqual(model.history[-1][2], [[f, g-.5] for f, g in rows])
        self.assertEqual((result['paid_episodes'], result['remaining_calls']), (6, 356))
        self.assertEqual(remaining_target_calls(2, prospective_batch=4), 356)
        self.assertEqual(remaining_target_calls(10), 352)

    def test_over_budget_is_rejected_before_mutating_model(self):
        model = RecordingModel()
        with self.assertRaises(ValueError):
            replay_raw_batches(model, [{'public_policy_index': 45,
                'raw_observations': [[.1, .2]]*363}], tau=0.)
        self.assertEqual((model.paid_episodes, model.history), (0, []))

    def test_restore_requires_full_zero_paid_fields_and_preserves_identity_object(self):
        module = ModuleType('performance.historical_full_bank_development')
        model = RecordingModel()
        calls = []
        def restore(*args):
            calls.append(args)
            return model, {'source_fit': 'unchanged'}
        module.restore_full_prior = restore
        with patch.dict(sys.modules, {module.__name__: module}):
            result, diagnostics = restore_zero_paid_prior({}, {}, {}, {}, {})
            self.assertIs(result, model)
            self.assertEqual(diagnostics, {'source_fit': 'unchanged'})
            self.assertEqual(calls[0][-1], 73)
            model.paid_episodes = 2
            with self.assertRaises(ValueError):
                restore_zero_paid_prior({}, {}, {}, {}, {})
            model.paid_episodes = 0
            model.components[0][0].candidate_count = 8
            with self.assertRaises(ValueError):
                restore_zero_paid_prior({}, {}, {}, {}, {})


if __name__ == '__main__':
    unittest.main()
