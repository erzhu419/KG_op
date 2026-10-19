"""Cached paired observations keep logical debits without new simulation."""

from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from performance import run_mainline_two_round73 as runner
from performance.historical_common_bank import initialization_batches, replay_raw_batches


class ReplayModel:
    def __init__(self, candidate_count):
        self.components = [(SimpleNamespace(candidate_count=candidate_count, paid_episodes=0),
                            SimpleNamespace(candidate_count=candidate_count, paid_episodes=0))]
        self.paid_episodes = 0
        self.history = []

    def update_raw(self, index, values):
        self.history.append((index, values))
        self.paid_episodes += 1
        for pair in self.components:
            for field in pair:
                field.paid_episodes += 1


class MainlineInitializationReuseTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.public = {'source_dimension': 2, 'L': 100, 'tau': .25}
        self.bank = [[10, 20], [30, 40], [50, 60], [70, 80]]
        self.tasks = [{'task_id': f'new_{i}', 'family': 'InventorySupplyChain',
                       'problem_kwargs': {'geometry': [1., 2.]}} for i in range(4)]
        self.spec = {'output': 'new', 'candidate_count': 4, 'layout': '5',
                     'models': ['source_structural_mixture', 'same_source_gp'],
                     'task_seed': 999, 'task_count': 4, 'target_tasks': self.tasks,
                     'initialization_indices': [0, 2], 'initialization_cache': 'old'}
        self.cached = {'problem_plan': deepcopy(self.public),
                       'tasks': [dict(task, task_id=f'old_{i}') for i, task in enumerate(self.tasks)],
                       'bank': deepcopy(self.bank[:3])}
        self.write('old/frozen_inputs.json', self.cached)
        self.old_records = []
        for i in range(4):
            record = {'task_id': f'old_{i}', 'candidate_bank_indices': [0, 2],
                      'paired_observations': [[.125+i, -.375-i], [.625+i, .875-i]],
                      'physical_calls': 2, 'tau': .25}
            self.old_records.append(record)
            self.write(f'old/task{i:02d}/initialization.json', record)
        states = [{'layout': '5', 'arm': arm, 'terminal_protocol': {'objective_reference': 1.}}
                  for arm in self.spec['models']]
        self.stack.enter_context(patch.object(runner, 'ROOT', self.root))
        self.stack.enter_context(patch.object(runner, 'read', side_effect=self.read))
        self.stack.enter_context(patch.object(runner, 'load',
                                             return_value=(self.spec, self.root/'new', {})))
        self.stack.enter_context(patch.object(runner, 'context', return_value=(
            {'training_tasks': [2, 3, 4, 5]}, None, states, self.public, None, None, self.bank)))
        self.problem = self.stack.enter_context(patch.object(
            runner, 'problem_for', side_effect=AssertionError('Cached initialization must not simulate')))

    def write(self, relative, value):
        path = self.root/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def read(self, path):
        return json.loads((self.root/path).read_text())

    def assert_no_initializations(self):
        self.assertEqual(list((self.root/'new').glob('task*/initialization.json')), [])
        self.problem.assert_not_called()

    def test_reuse_preserves_raw_episodes_and_logical_budget(self):
        runner.prepare()
        self.problem.assert_not_called()
        for i, old in enumerate(self.old_records):
            copied = self.read(f'new/task{i:02d}/initialization.json')
            self.assertEqual(copied['paired_observations'], old['paired_observations'])
            self.assertEqual(copied['candidate_bank_indices'], old['candidate_bank_indices'])
            self.assertEqual(copied['tau'], old['tau'])
            self.assertEqual(copied['physical_calls'], 0)
            self.assertEqual(copied['reused_initialization_calls'], 2)
            self.assertEqual(copied['task_id'], f'new_{i}')
            self.assertEqual(self.read(f'old/task{i:02d}/initialization.json'), old)
            reused, original = ReplayModel(4), ReplayModel(4)
            accounting = replay_raw_batches(reused, initialization_batches(copied), copied['tau'])
            replay_raw_batches(original, initialization_batches(old), old['tau'])
            self.assertEqual(reused.history, original.history)
            self.assertEqual(reused.history, [(q, [raw[0], raw[1]-.25])
                                             for q, raw in zip([0, 2], old['paired_observations'])])
            self.assertEqual(reused.paid_episodes, 2)
            self.assertEqual(accounting['remaining_calls'], 360)
            self.assertEqual(accounting['new_physical_calls'], 0)

    def test_changed_physical_initial_policy_is_rejected_before_copy(self):
        self.bank[2][1] += 1
        with self.assertRaises(AssertionError):
            runner.prepare()
        self.assert_no_initializations()

    def test_changed_task_geometry_is_rejected_before_copy(self):
        self.tasks[0]['problem_kwargs'] = {'geometry': [1., 3.]}
        with self.assertRaises(AssertionError):
            runner.prepare()
        self.assert_no_initializations()

    def test_changed_problem_plan_is_rejected_before_copy(self):
        self.public['L'] = 101
        with self.assertRaises(AssertionError):
            runner.prepare()
        self.assert_no_initializations()


if __name__ == '__main__':
    unittest.main()
