import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from performance import positive_historical_direct_references as reference


class CompetitionNamespaceTest(unittest.TestCase):
    def test_new_grid_preserves_old_artifact_names_and_links_its_closed_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def save(path,data):
                p=root/path;p.parent.mkdir(parents=True,exist_ok=True)
                p.write_text(json.dumps(data))
            def read(path):return json.loads((root/path).read_text())
            closed='paper_artifacts/historical_positive_risk_uppervariance_axis_incumbents_v6/completion.json'
            prefix='historical_positive_risk_uppervariance_axis'
            current=dict(output='screen',risk_mean_order=4,competition_prefix=prefix,
                phase_completion=closed,joint_reference_plans=['old_reference'],
                pricing_reference_plans=['old_reference'],continued_phase_cost_plan='old_budget')
            save('start',current)
            save('screen/summary.json',{})
            save('screen/compact_summary.json',dict(numerical_containment_passed=True))
            save('old_reference',dict(output='old_ref',previous_phase_completion='older_phase',
                maximum_worker_cpu_seconds=421,maximum_estimated_cpu_seconds=3600))
            save('old_ref/compact_summary.json',dict(local_risk_candidate_passed=True))
            save(closed,dict(state='phase_complete',all_reference_precision_passed=True,estimated_phase_cpu_seconds=1000))
            legacy=root/'performance/manifests/historical_positive_risk_full_grid_competition_v1.json'
            save(legacy,dict(legacy_result='preserve'))
            save(f'performance/manifests/{prefix}_competition_v1.json',
                dict(failed_preparation_account='same_phase_failed_cpu.json'))
            priced=dict(cost_admitted=True,next_joint_cases=[dict(record='record',kind='raw',candidate='E8',rules=[])],
                next_joint_jobs=[dict(estimated_cpu_seconds=421.5778667636)],
                expected_cpu_seconds_including600reserves=1021.5778667636)
            with patch.object(reference,'ROOT',root),patch.object(reference,'read',read),\
                patch('performance.screen_historical_observed_batch_risk.next_plan',return_value=priced),\
                patch.object(reference,'prepare_frozen',return_value=dict(complete=True)),contextlib.redirect_stdout(io.StringIO()):
                result=reference.prepare_next('start')
            self.assertEqual(result['risk_plan'],f'performance/manifests/{prefix}_competition_v1.json')
            self.assertEqual(result['risk_manifest']['previous_phase_completion'],closed)
            self.assertEqual(result['priced']['previous_closed_phase_completion'],closed)
            self.assertNotIn('continued_phase_cost_plan',result['pricing_manifest'])
            self.assertEqual(read(legacy),dict(legacy_result='preserve'))
            self.assertEqual(read('old_reference')['previous_phase_completion'],'older_phase')
            self.assertEqual(result['risk_manifest']['maximum_worker_cpu_seconds'],422)
            self.assertEqual(result['bound_manifest']['maximum_worker_cpu_seconds'],422)
            self.assertEqual(result['risk_manifest']['maximum_estimated_cpu_seconds'],3600)
            self.assertEqual(result['priced']['expected_cpu_seconds_including600reserves'],1021.5778667636)
            self.assertEqual(result['risk_manifest']['failed_preparation_account'],'same_phase_failed_cpu.json')


if __name__=='__main__':
    unittest.main()
