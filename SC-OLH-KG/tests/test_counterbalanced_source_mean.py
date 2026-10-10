import copy
import unittest
from pathlib import Path

import numpy as np

from performance.check_legacy_counterbalanced_source_mean import mean_view
from performance.check_legacy_public_mean_risk_uncertainty import read


class CounterbalancedSourceMeanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = read('performance/manifests/legacy_joint_population_counterbalanced_source_mean_execution_v1.json')
        cls.old = read(cls.spec['source_execution'])
        cls.data = read(cls.spec['source_data'])
        cls.public = read(Path(cls.spec['source_data']).with_name('frozen_inputs.json'))

    def test_unselected_and_deleted_labels_cannot_enter_frozen_views(self):
        for layout, sites in self.spec['layouts'].items():
            folder = Path(self.old['output'])/f"layout{self.spec['parent_layouts'][layout]}"
            source = dict(self.spec, source_site_indices_by_task=sites)
            for deleted in (None, 0):
                parent = read(folder/'inputs.json' if deleted is None else folder/f'task{deleted}'/'inputs.json')
                view = mean_view(self.data, self.public, source, parent, deleted)
                changed = copy.deepcopy(self.data)
                selected = {i: set(s) for i, s in zip(source['training_tasks'], sites) if i != deleted}
                for i, task in enumerate(changed['tasks']):
                    for j in range(len(self.data['bank'])):
                        if j not in selected.get(i, set()):
                            task['Y_replicates'][j] = [[1000., 2000.], [-3000., 4000.], [5000., -6000.]]
                self.assertEqual(view, mean_view(changed, self.public, source, parent, deleted))
                self.assertEqual(view['mean_source_record_count'], 72 if deleted is None else 60)
                self.assertEqual(len(view['risk_source_tasks'])*8, 48 if deleted is None else 40)
                self.assertEqual(view['public_mean_reference'], parent['public_mean_reference'])

    def test_extra_source_variance_changes_mean_noise_only(self):
        layout = '5'; sites = self.spec['layouts'][layout]
        source = dict(self.spec, source_site_indices_by_task=sites)
        parent = read(Path(self.old['output'])/'layout3'/'inputs.json')
        view = mean_view(self.data, self.public, source, parent)
        changed = copy.deepcopy(self.data)
        for offset, raw in zip((-1, 0, 1), changed['tasks'][0]['Y_replicates'][sites[0][8]]):
            raw[0] += 9*offset; raw[1] += 7*offset
        replay = mean_view(changed, self.public, source, parent)
        np.testing.assert_allclose(np.asarray(view['source_tasks'][0]['response'])[:, :2],
            np.asarray(replay['source_tasks'][0]['response'])[:, :2], atol=1e-12)
        self.assertNotEqual(view['source_tasks'][0]['noise'][8][:2], replay['source_tasks'][0]['noise'][8][:2])
        for key in ('risk_source_tasks', 'public_risk_reference', 'coordinate', 'risk_coordinate_center',
                    'risk_coordinate_scale', 'heldout_risk_psi'):
            self.assertEqual(view[key], replay[key])


if __name__ == '__main__':
    unittest.main()
