import copy
import json
import unittest
from pathlib import Path

import numpy as np

from performance.check_legacy_public_mean_source_half import source_views

ROOT = Path(__file__).resolve().parents[1]


class PublicMeanSourceHalfTests(unittest.TestCase):
    def test_selected_labels_alone_determine_both_head_views_and_normalization(self):
        spec = json.loads((ROOT/'performance/manifests/legacy_joint_population_public_mean_source_half_plan_v1.json').read_text())
        data = json.loads((ROOT/spec['source_data']).read_text())
        public = json.loads((ROOT/spec['source_data']).with_name('frozen_inputs.json').read_text())
        original = source_views(data, public, spec)
        changed = copy.deepcopy(data)
        layouts = dict(zip(spec['training_tasks'], spec['source_site_indices_by_task']))
        for i, task in enumerate(changed['tasks']):
            for j in range(len(data['bank'])):
                if j not in layouts.get(i, []):
                    task['Y_replicates'][j] = [[1000.+j, -1000.-j], [3000.+j, 7000.+j], [-2000.-j, 9000.+j]]
        replay = source_views(changed, public, spec)
        self.assertEqual(original, replay)
        self.assertEqual(original['coordinate']['source_record_count'], 48)
        self.assertEqual(sum(len(t['response']) for t in original['source_tasks'])*3, 144)
        for mean, risk, sites in zip(original['source_tasks'], original['risk_source_tasks'],
                                     spec['source_site_indices_by_task']):
            np.testing.assert_array_equal(mean['response'], risk['response'])
            np.testing.assert_array_equal(mean['noise'], risk['noise'])
            self.assertEqual(len(mean['psi']), len(sites))
        self.assertEqual(np.asarray(original['source_tasks'][0]['psi']).shape[1], 4)
        self.assertEqual(np.asarray(original['risk_source_tasks'][0]['psi']).shape[1], 10)


if __name__ == '__main__':
    unittest.main()
