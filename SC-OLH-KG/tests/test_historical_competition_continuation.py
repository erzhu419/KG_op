import contextlib
import importlib.util
import io
import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from performance import launch_historical_component_bank as bank

FOLDER=bank.ROOT/'paper_artifacts/historical_positive_risk_full_grid_continuation_v1'


def save(root,path,data):
    p=root/path;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data))


class CompetitionContinuationTest(unittest.TestCase):
    def check_start(self,remaining,maximum_new_phases=0,run_phase=False,resume_reference=False,preparation_log='/tmp/preparation',dispatch_error=None):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);phase='paper_artifacts/new_grid_competition_v1'
            closed='performance/manifests/new_grid_competition_v1.json'
            source='performance/manifests/new_grid_screen_v1.json'
            save(root,closed,dict(output=phase))
            save(root,source,dict(phase_completion=phase+'/completion.json',output='screen'))
            save(root,phase+'/compact_summary.json',dict(local_risk_candidate_passed=True,
                estimated_cpu_seconds_including_reserves=3500,successful_worker_cpu_seconds=2000))
            save(root,phase+'/completion.json',dict(state='phase_complete',all_reference_precision_passed=True,
                recorded_process_cpu_seconds=2088))
            save(root,'screen/compact_summary.json',dict(numerical_containment_passed=True,
                totals=dict(raw=dict(record_group_candidates=remaining),projected=dict(record_group_candidates=remaining))))
            legacy='paper_artifacts/historical_positive_risk_full_grid_continuation_v1/status.json'
            save(root,legacy,dict(legacy='preserve'))
            next_plan='performance/manifests/new_grid_competition_v2.json'
            next_bound='performance/manifests/new_grid_screen_v2.json'
            next_phase='paper_artifacts/new_grid_competition_v2'
            preparation=dict(result=dict(prepared=True,
                pricing_plan='pricing.json',pricing_manifest=dict(output='pricing'),priced={},
                risk_plan=next_plan,risk_manifest=dict(output=next_phase),
                bound_plan=next_bound,bound_manifest=dict(output='next_screen',phase_completion=next_phase+'/completion.json'),
                preparation=dict(jobs=[{}],estimated_cpu_seconds_including_reserves=1000)),
                resource=dict(process_cpu_seconds=1))
            if resume_reference:
                save(root,next_plan,preparation['result']['risk_manifest'])
                save(root,next_bound,preparation['result']['bound_manifest'])
                save(root,next_phase+'/preparation.json',preparation['result']['preparation'])
                save(root,'paper_artifacts/new_campaign/preparation_v2_result.json',preparation)
                save(root,'paper_artifacts/new_campaign/status.json',dict(state='precision_hold',
                    source_bound_plan=source,completed_phases=[closed],phase_estimates={closed:3500},
                    continuation_worker_cpu_seconds=2020,recorded_process_cpu_seconds=2088,current_phase=next_plan,
                    current_reference_worker_cpu_plan=next_plan,current_reference_worker_cpu_seconds=20))
            scheduler=Mock()
            scheduler.run_on.side_effect=lambda node,cmd,**kw:(0,json.dumps(preparation) if 'lines=Path' in cmd else '','')
            save(root,'queue.json',dict(tasks=[dict(id='t99',status='done',node='node004',log_path=preparation_log)]))
            def command(cmd,**kw):
                if str(cmd[1]).endswith('close_phase.py'):
                    save(root,next_phase+'/completion.json',dict(state='phase_complete',total_recorded_process_cpu_seconds=50))
                if len(cmd)>2 and cmd[2]=='dispatch' and dispatch_error:
                    return SimpleNamespace(returncode=1,stdout='',stderr=dispatch_error)
                return SimpleNamespace(returncode=0,stdout='submitted t99',stderr='')
            def collect(stage,folder,plan):
                if stage=='observed-batch-risk':
                    return dict(local_risk_candidate_passed=True,successful_worker_cpu_seconds=25)
                if stage=='observed-batch-bound':
                    result=dict(numerical_containment_passed=True,
                        totals=dict(raw=dict(record_group_candidates=5),projected=dict(record_group_candidates=5)))
                    save(root,folder/'compact_summary.json',result)
                    return result
                return dict(cost_admitted=False,next_joint_cases=[])
            argv=['continue_when_ready.py','--source-bound',source,'--output','paper_artifacts/new_campaign',
                '--maximum-new-phases',str(maximum_new_phases)]
            with patch.object(bank,'ROOT',root),patch.object(bank,'DEPLOY',str(root/'deploy')),\
                patch.object(bank,'QUEUE',root/'queue.json'),patch.object(bank,'refresh_jobs'),\
                patch.object(bank,'launch') as launch,patch.object(bank,'collect_remote',side_effect=collect),\
                patch('subprocess.run',side_effect=command),patch.object(sys,'argv',argv),\
                patch.dict(sys.modules,{'scheduler':scheduler}),contextlib.redirect_stdout(io.StringIO()):
                runpy.run_path(str(FOLDER/'continue_when_ready.py'),run_name='__main__')
            if preparation_log is None and run_phase:
                self.assertTrue(any('lines=Path' in call.args[1] and '/tmp/sched_t99.log' in call.args[1]
                    for call in scheduler.run_on.call_args_list))
            self.assertEqual(json.loads((root/legacy).read_text()),dict(legacy='preserve'))
            result=json.loads((root/'paper_artifacts/new_campaign/status.json').read_text())
            self.assertEqual(result['recorded_process_cpu_seconds'],2138 if run_phase else 2088)
            self.assertEqual(result['completed_phases'],[closed,next_plan] if run_phase else [closed])
            self.assertEqual(result['continuation_worker_cpu_seconds'],2025 if run_phase else 2000)
            if run_phase:launch.assert_called_once_with('observed-batch-risk',next_plan)
            else:launch.assert_not_called()
            return result

    def test_new_campaign_uses_current_completion_and_respects_phase_limit(self):
        self.assertEqual(self.check_start(5)['state'],'phase_limit_hold')

    def test_closed_actions_take_precedence_over_phase_limit(self):
        self.assertEqual(self.check_start(0)['state'],'conditional_actions_closed')

    def test_campaign_options_survive_submitting_and_closing_a_new_phase(self):
        self.assertEqual(self.check_start(5,maximum_new_phases=1,run_phase=True)['state'],'phase_limit_hold')

    def test_resumed_reference_adds_only_new_cpu_to_campaign_summary(self):
        self.assertEqual(self.check_start(5,maximum_new_phases=1,run_phase=True,resume_reference=True)['state'],'phase_limit_hold')

    def test_preparation_recovered_by_exit_sentinel_uses_existing_scheduler_log(self):
        self.assertEqual(self.check_start(5,maximum_new_phases=1,run_phase=True,preparation_log=None)['state'],'phase_limit_hold')

    def test_preparation_dispatch_lock_timeout_keeps_submitted_task(self):
        self.assertEqual(self.check_start(5,maximum_new_phases=1,run_phase=True,
            dispatch_error='scheduler_state.locks.SchedulerLockTimeout: timed out after20s')['state'],'phase_limit_hold')

    def test_other_preparation_dispatch_error_still_stops(self):
        with self.assertRaisesRegex(RuntimeError,'dispatch transport failed'):
            self.check_start(5,maximum_new_phases=1,run_phase=True,dispatch_error='dispatch transport failed')

    def check_close(self,failure=None,refinement_cpu=0,collection_failure=None,worker_log='/tmp/log',probe_cpu=0,multilevel_cpu=0,prior_replay_cpu=0):
        spec=importlib.util.spec_from_file_location('competition_close',FOLDER/'close_phase.py')
        close=importlib.util.module_from_spec(spec);spec.loader.exec_module(close)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);phase='paper_artifacts/new_grid_competition_v2'
            plan='performance/manifests/new_grid_competition_v2.json';bound='bound.json'
            previous='paper_artifacts/new_grid_competition_v1/completion.json'
            initial=dict(output=phase,followup_bound_plan=bound,frozen_cost_plan='cost.json')
            if failure is not None:
                initial['failed_preparation_account']='failed_preparation.json'
                save(root,'failed_preparation.json',failure)
            save(root,plan,initial)
            save(root,bound,dict(output='screen',phase_completion=phase+'/completion.json',joint_reference_plans=[plan]))
            save(root,phase+'/observed-batch-risk_jobs.json',[dict(id='t1')])
            save(root,'queue.json',dict(tasks=[dict(id='t1',status='done',node='node004',log_path=worker_log)]))
            save(root,phase+'/preparation.json',dict(selected=[dict(record='r',action_indices=[1])],estimated_cpu_seconds_including_reserves=1000))
            save(root,phase+'/compact_summary.json',dict(local_risk_candidate_passed=True,successful_worker_cpu_seconds=7,
                groups=[dict(record='r',checks=dict(comparisons=[dict(maximum_difference=.00001)]))]))
            save(root,'new_campaign/preparation_v2_result.json',dict(resource=dict(process_cpu_seconds=2)))
            save(root,'cost.json',dict(previous_closed_phase_completion=previous))
            save(root,previous,dict(estimated_phase_cpu_seconds=3500))
            old_envelope=dict(kinds={k:dict(unresolved_upper_gap=.02) for k in ['raw','projected']})
            save(root,Path(previous).with_name('execution_account.json'),dict(weighted_bank_envelope=old_envelope))
            envelope=dict(kinds={k:dict(unresolved_upper_gap=.01) for k in ['raw','projected']})
            save(root,'screen/compact_summary.json',dict(numerical_containment_passed=True,weighted_bank_envelope=envelope,totals={}))
            save(root,'screen/next_cost_plan.json',dict(cost_admitted=False))
            save(root,phase+'/status.json',dict(state='joint_references_admitted'))
            if collection_failure is not None:
                save(root,phase+'/collection_failure_account.json',collection_failure)
            marker='SCHEDULEURM_RESOURCE '+json.dumps(dict(process_cpu_seconds=1))+'\n'
            for rel in [phase+'/observed-batch-risk_collection.log','screen/observed-batch-bound_collection.log','screen/observed-batch-next-plan_collection.log']:
                (root/rel).write_text(marker)
            if refinement_cpu:
                for name in ('observed-batch-risk_collection_before_refinement.log','refinement_preparation.log'):
                    (root/phase/name).write_text('SCHEDULEURM_RESOURCE '+json.dumps(dict(process_cpu_seconds=refinement_cpu/2))+'\n')
            if probe_cpu:
                (root/phase/'branch_coupling_probe.log').write_text('SCHEDULEURM_RESOURCE '+json.dumps(dict(process_cpu_seconds=probe_cpu))+'\n')
            for name,cpu in [('multilevel_caps_probe.log',multilevel_cpu),('incumbent_replay_final.log',prior_replay_cpu)]:
                if cpu:(root/phase/name).write_text('SCHEDULEURM_RESOURCE '+json.dumps(dict(process_cpu_seconds=cpu))+'\n')
            replay=dict(source_bound_plan=bound,native_choices_all_raw_referenced=True,
                weighted_raw_reward_of_native_incumbent_contrast=.2,weighted_raw_best_admitted_contrast=.2,
                weighted_projected_incumbent_contrast=.19)
            def remote(node,cmd,**kwargs):
                if 'performance.replay_historical_positive_risk_subset' in cmd:
                    return 0,'SCHEDULEURM_DONE '+json.dumps(replay)+'\n'+marker,''
                if worker_log is None:
                    self.assertIn('/tmp/sched_t1.log',cmd)
                    self.assertNotIn("'t1': None",cmd)
                return 0,json.dumps(dict(t1=dict(process_cpu_seconds=10,peak_rss_mb=50))),''
            with patch.object(close,'ROOT',root),patch.object(close,'QUEUE',root/'queue.json'),\
                patch.object(close.scheduler,'run_on',side_effect=remote),contextlib.redirect_stdout(io.StringIO()):
                close.close(bound,'new_campaign')
            result=json.loads((root/phase/'completion.json').read_text())
            self.assertEqual(result['preparation_process_cpu_seconds'],2)
            recorded=0 if failure is None else (failure.get('actual_process_cpu_seconds') or 0)
            charged=0 if failure is None else failure['conservative_reserve_charge_seconds']
            collection_recorded=0 if collection_failure is None else (collection_failure.get('actual_process_cpu_seconds') or 0)
            collection_charged=0 if collection_failure is None else collection_failure['conservative_reserve_charge_seconds']
            self.assertEqual(result['total_recorded_process_cpu_seconds'],16+recorded+refinement_cpu+collection_recorded+probe_cpu+multilevel_cpu+prior_replay_cpu)
            self.assertEqual(result['shared_reserve_charged_cpu_seconds'],6+charged+refinement_cpu+collection_charged+probe_cpu+multilevel_cpu+prior_replay_cpu)
            self.assertEqual(result['unmeasured_failed_preparation_conservative_charge_seconds'],charged-recorded)
            self.assertEqual(result['unmeasured_failed_collection_conservative_charge_seconds'],collection_charged-collection_recorded)
            self.assertEqual(result['weighted_unresolved_gap_reduction'],dict(raw=.5,projected=.5))
            self.assertEqual(json.loads((root/result['replay_summary_path']).read_text()),replay)
            self.assertFalse((root/phase/'incumbent_replay_final.json').exists())
            resources=json.loads((root/phase/'worker_resources.json').read_text())
            self.assertEqual(resources[0]['log_path'],worker_log or '/tmp/sched_t1.log')

    def test_saved_replay_summary_preserves_decisions_without_rerunning_or_pulling_rows(self):
        from performance import replay_historical_positive_risk_subset as replay
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path='phase/incumbent_replay_final.json'
            saved=dict(complete=True,source_bound_plan='bound.json',rows=[dict(raw_actions=[1])],
                native_incumbents=[dict(native_action_counts={'1':16})],maximum_omitted_utility_identity_mass=1e-8,
                raw_recommendation=dict(policy_totals={'raw':dict(validation_contrast=.2)},rows=[dict(actions=[1])]),
                weighted_raw_best_admitted_contrast=.2)
            save(root,path,saved)
            with patch.object(replay,'ROOT',root),patch.object(replay,'read',side_effect=lambda p:json.loads((root/p).read_text())),\
                patch.object(replay,'run') as run:
                summary=replay.collect_saved_replay('bound.json',path)
            run.assert_not_called()
            self.assertNotIn('rows',summary)
            self.assertNotIn('native_incumbents',summary)
            self.assertNotIn('rows',summary['raw_recommendation'])
            self.assertEqual(summary['raw_recommendation']['policy_totals'],saved['raw_recommendation']['policy_totals'])
            self.assertEqual(summary['full_replay_path'],path)

    def test_cap_pilot_and_previous_summary_collection_remain_in_original_reserve(self):
        self.check_close(probe_cpu=41,multilevel_cpu=70,prior_replay_cpu=1,
            collection_failure=dict(actual_process_cpu_seconds=None,conservative_reserve_charge_seconds=35))

    def test_close_uses_new_preparation_resources_and_original_account_envelope(self):
        self.check_close()

    def test_refinement_preparation_and_original_failed_collection_are_charged_once(self):
        self.check_close(refinement_cpu=4)

    def test_failed_preparation_stays_in_original_reserve_and_recorded_cpu(self):
        for recorded,charged in [(41.034681312,41.034681312),(None,122.685705369)]:
            with self.subTest(recorded=recorded):
                self.check_close(dict(actual_process_cpu_seconds=recorded,conservative_reserve_charge_seconds=charged))

    def test_collection_timeout_stays_in_same_phase_without_inventing_measured_cpu(self):
        self.check_close(collection_failure=dict(actual_process_cpu_seconds=None,
            conservative_reserve_charge_seconds=35))

    def test_worker_recovered_by_exit_sentinel_preserves_measured_resources(self):
        self.check_close(worker_log=None)

    def test_cached_branch_probe_stays_in_original_phase_reserve(self):
        self.check_close(probe_cpu=9.5)


if __name__=='__main__':
    unittest.main()
