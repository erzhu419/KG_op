import contextlib
import io
import json
import runpy
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from performance import continue_historical_raw_axis_refinement as controller


class RawAxisResumeTest(unittest.TestCase):
    def check_resume(self, coverage_state):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def read(path):
                return json.loads((root/path).read_text())
            def save(path,data):
                p=root/path;p.parent.mkdir(parents=True,exist_ok=True)
                p.write_text(json.dumps(data))
            out=root/'paper_artifacts/historical_raw_axis_uppermidmean_followthrough_v1'
            mean=dict(axis='uppermidmean',final_bound_plan='mean_bound',
                raw_validation_contrast=.2,local_axis_difference_passed=True)
            save(out/'status.json',dict(completed_axes=[mean]))
            folder=root/'paper_artifacts/historical_positive_risk_uppervariance_axis_v1'
            save(folder/'preparation_job.json',dict(id='existing'))
            save(folder/'refinement_plan.json',dict(plan='variance_plan',source_bound_plan='mean_bound',
                estimated_initial_phase_cpu_seconds=3000,risk_observations=224,
                reused_risk_observations=160,new_risk_observations=64))
            follow=root/'paper_artifacts/historical_positive_risk_uppervariance_axis_followthrough_v2'
            save(follow/'incumbent_frozen_plan.json',{})
            save(follow/'coverage_status.json',dict(state='partial_coverage',completed_phases=[2,3,4]))
            save('next_reference',dict(followup_bound_plan='variance_bound'))
            def finish(args,**kwargs):
                save(follow/'coverage_status.json',dict(state=coverage_state,
                    completed_phases=[2,3,4,5,6],latest_axes=dict(uppervariance=dict(plan='next_reference'))))
            replay=lambda value: dict(raw_recommendation=dict(policy_totals=dict(raw=dict(validation_contrast=value))),
                all_conditional_actions_closed=False)
            with patch.object(controller,'ROOT',root),patch.object(controller,'read',read),patch.object(controller,'save',save),\
                patch.object(controller,'sync'),patch.object(controller,'prepare') as prepare,\
                patch.object(controller,'replay_for',side_effect=[replay(.1),replay(.20003)]) as replay_for,\
                patch.object(controller.subprocess,'run',side_effect=finish) as run,contextlib.redirect_stdout(io.StringIO()):
                controller.run('initial_bound','uppermidmean')
            prepare.assert_not_called()
            self.assertEqual(run.call_count,1)
            args=run.call_args.args[0]
            self.assertEqual(args[args.index('--start-phase')+1],'5')
            self.assertEqual(args[args.index('--last-phase')+1],'6')
            result=read(out/'status.json')
            self.assertEqual(result['completed_axes'][0],mean)
            return result,(out/'uppervariance_analysis.json').exists()

    def test_resume_skips_completed_mean_and_starts_after_closed_phase_four(self):
        result,analyzed=self.check_resume('native_raw_coverage_complete')
        self.assertEqual(result['state'],'local_mean_and_upper_variance_checked')
        self.assertTrue(analyzed)
        self.assertAlmostEqual(result['completed_axes'][1]['difference'],.00003)

    def test_partial_coverage_does_not_admit_an_axis_comparison(self):
        result,analyzed=self.check_resume('partial_coverage')
        self.assertEqual(result['state'],'coverage_hold')
        self.assertFalse(analyzed)
        self.assertEqual(len(result['completed_axes']),1)

    def test_coverage_extension_preserves_closed_phases_and_stops_when_complete(self):
        script=controller.ROOT/'paper_artifacts/historical_positive_risk_outer_axes_incumbents_v2/finish_coverage.py'
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'paper_artifacts/test_followthrough_v2'
            folder.mkdir(parents=True)
            copied=folder/'finish_coverage.py';shutil.copyfile(script,copied)
            frozen=folder/'incumbent_frozen_plan.json';frozen.write_text(json.dumps(dict(rows=[dict(axis='uppervariance')])))
            (folder/'coverage_status.json').write_text(json.dumps(dict(completed_phases=[2,3,4])))
            def step(args,**kwargs):
                new=root/'paper_artifacts/test_followthrough_v5';new.mkdir(exist_ok=True)
                if args[1].endswith('prepare_next_incumbents.py'):
                    self.assertEqual(args[args.index('--phase')+1],'5')
                    (new/'incumbent_frozen_plan.json').write_text(json.dumps(dict(rows=[dict(axis='uppervariance',plan='plan.json')])))
                    (root/'plan.json').write_text(json.dumps(dict(output='reference')))
                else:
                    (new/'incumbent_status.json').write_text(json.dumps(dict(state='incumbent_probes_complete')))
                    ref=root/'reference';ref.mkdir()
                    (ref/'completion.json').write_text(json.dumps(dict(native_choices_all_raw_referenced=True,
                        missing_raw_observations=0,raw_covered_grid_weight=1)))
                    (ref/'incumbent_replay_final.json').write_text(json.dumps(dict(
                        weighted_raw_reward_of_native_incumbent_contrast=.2,
                        weighted_raw_best_admitted_contrast=.2,all_conditional_actions_closed=False)))
            argv=[str(copied),'--frozen-plan',str(frozen),'--start-phase','5','--last-phase','6']
            with patch.object(sys,'argv',argv),patch('subprocess.run',side_effect=step) as run,contextlib.redirect_stdout(io.StringIO()):
                runpy.run_path(str(copied),run_name='__main__')
            self.assertEqual(run.call_count,2)
            result=json.loads((folder/'coverage_status.json').read_text())
            self.assertEqual(result['completed_phases'],[2,3,4,5])
            self.assertEqual(result['state'],'native_raw_coverage_complete')


if __name__=='__main__':
    unittest.main()
