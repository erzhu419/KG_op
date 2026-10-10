import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from performance.run_legacy_joint_population_source_budget import encode_subset, source_subset
from performance.run_legacy_related_inventory_source import problem_for, task_configs
from representation.legacy_observable_prior import legacy_exposure_descriptors


class SourceBudgetInformationTests(unittest.TestCase):
    def test_excluded_labels_cannot_change_representation_or_normalization(self):
        rng = np.random.default_rng(80841)
        plan = {'task_seed': 80841, 'source_task_count': 6, 'family': 'InventorySupplyChain',
                'source_dimension': 50, 'task_distribution': {'L': 100, 'sigma': .04,
                                                             'alpha': .05, 'weights': [.5, .5]}}
        bank = rng.integers(10, 80, size=(20, 50)).tolist()
        provider = problem_for(plan, task_configs(plan)[0])
        descriptor = legacy_exposure_descriptors(provider, bank)
        base = np.column_stack([.35+.1*np.asarray(bank).mean(axis=1)/100,
                                -.1+.3*np.asarray(bank)[:, :17].mean(axis=1)/100])
        tasks = [{'task_id': f'source_{i}', 'tau': 0.,
                  'Y_replicates': (base[:, None, :]+.02*i+rng.normal(0, .03, (20, 3, 2))).tolist()}
                 for i in range(6)]
        data = {'bank': bank, 'tasks': tasks}
        descriptors = {i: descriptor for i in range(6)}
        a, view_a, center_a, scale_a = encode_subset(data, range(6), 16, descriptors, 100)
        changed = copy.deepcopy(data)
        for task in changed['tasks']:
            task['Y_replicates'][16:] = rng.normal(1e6, 1e3, (4, 3, 2)).tolist()
        b, view_b, center_b, scale_b = encode_subset(changed, range(6), 16, descriptors, 100)
        np.testing.assert_array_equal(a.transform(descriptor), b.transform(descriptor))
        np.testing.assert_array_equal(center_a, center_b)
        np.testing.assert_array_equal(scale_a, scale_b)
        self.assertEqual(a.diagnostics()['source_record_count'], 96)
        for left, right in zip(view_a, view_b):
            for key in ('psi', 'features', 'response', 'noise'):
                np.testing.assert_array_equal(left[key], right[key])
        selected = source_subset(data, range(6), 16)
        self.assertEqual(sum(len(t['Y_replicates'])*3 for t in selected['tasks']), 288)


if __name__ == '__main__':
    unittest.main()
