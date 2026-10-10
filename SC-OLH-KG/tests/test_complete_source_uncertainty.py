import copy
import unittest
from pathlib import Path

from performance.check_legacy_complete_source_uncertainty import mean_views, read


class CompleteSourceUncertaintyTests(unittest.TestCase):
    def test_deleted_and_unselected_labels_do_not_enter_mean_views(self):
        source = read('performance/manifests/legacy_joint_population_public_mean_source_half_execution_v1.json')
        data = read(source['source_data'])
        public = read(Path(source['source_data']).with_name('frozen_inputs.json'))
        for deleted in source['training_tasks']:
            original = mean_views(data, public, source, deleted)
            changed = copy.deepcopy(data)
            selected = {i: set(s) for i, s in zip(source['training_tasks'], source['source_site_indices_by_task']) if i != deleted}
            for i, task in enumerate(changed['tasks']):
                for j in range(len(data['bank'])):
                    if j not in selected.get(i, set()):
                        task['Y_replicates'][j] = [[1000.+j, -2000.-j], [-3000.-j, 5000.+j], [7000.+j, 9000.+j]]
            self.assertEqual(original, mean_views(changed, public, source, deleted))
            self.assertEqual(sum(len(t['response']) for t in original['source_tasks']), 40)


if __name__ == '__main__':
    unittest.main()
