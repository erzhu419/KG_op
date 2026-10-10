import copy
import unittest
from pathlib import Path

import numpy as np

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.check_legacy_public_mean_source_half import source_views
from performance.check_legacy_source_layout_robustness import branch_views, models_at_query


class SourceLayoutRobustnessTests(unittest.TestCase):
    def test_general_query_builder_reproduces_frozen_source_forecasts(self):
        execution = read('performance/manifests/legacy_joint_population_source_layout_robustness_execution_v1.json')
        out = Path(execution['output'])/'layout2'
        members = [(read(out/'inputs.json'), read(out/'source_fit.json'), read(out/'risk_source_fit.json'))]
        members += [(read(out/f'task{i}'/'inputs.json'), read(out/f'task{i}'/'gp_source_fit.json'),
                     read(out/f'task{i}'/'risk_source_fit.json')) for i in execution['training_tasks']]
        task = read(execution['source_data'])['tasks'][6]
        reference = read(Path(execution['output'])/'source_predictions.json')['rows']
        for arm, model in models_at_query(members, [m[0]['heldout_mean_psi']['6'] for m in members],
                                         [m[0]['heldout_risk_psi']['6'] for m in members]):
            for j in (0,1):
                raw = task['Y_replicates'][j][0]; model.update_raw(j, [raw[0], raw[1]-task['tau']])
            mean, covariance = model.predict_joint()
            variance = np.diag(covariance.reshape(292,292)).reshape(73,4)
            old = next(r for r in reference if r['layout']=='2' and r['task_index']==6 and r['mean_arm']==arm)
            np.testing.assert_allclose(mean[:,[0,2,3,1]],old['mean'],rtol=1e-8,atol=1e-9)
            np.testing.assert_allclose(variance[:,[0,2,3,1]],old['variance'],rtol=1e-8,atol=1e-9)

    def test_new_layouts_and_deleted_reference_task_use_only_selected_labels(self):
        plan = read('performance/manifests/legacy_joint_population_source_layout_robustness_plan_v1.json')
        source = read(plan['reference_source_execution']); data = read(plan['source_data'])
        public = read(Path(plan['source_data']).with_name('frozen_inputs.json'))
        for sites in plan['layouts'].values():
            spec = dict(source, source_site_indices_by_task=sites)
            for deleted in (None, 0):
                view = source_views(data, public, spec) if deleted is None else branch_views(data, public, spec, deleted)
                changed = copy.deepcopy(data)
                selected = {i: set(s) for i, s in zip(spec['training_tasks'], sites) if i != deleted}
                for i, task in enumerate(changed['tasks']):
                    for j in range(len(data['bank'])):
                        if j not in selected.get(i, set()):
                            task['Y_replicates'][j] = [[1000.+j, -2000.-j], [-3000.-j, 5000.+j], [7000.+j, 9000.+j]]
                replay = source_views(changed, public, spec) if deleted is None else branch_views(changed, public, spec, deleted)
                self.assertEqual(view, replay)
                self.assertEqual(sum(len(t['response']) for t in view['risk_source_tasks']), 48 if deleted is None else 40)


if __name__ == '__main__':
    unittest.main()
