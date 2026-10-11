import unittest
import contextlib
import io
import json
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

from performance.validate_historical_risk_competition import select_competitors,selection_rows,prepare,PLAN
from performance import launch_historical_component_bank as launcher


class RiskCompetitionTests(unittest.TestCase):
    def test_campaign_waits_for_delayed_resource_footer_without_resubmitting(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'phase';out.mkdir()
            (out/'assembly_scheduler_spec.json').write_text(json.dumps(dict(task_id='assembly')))
            (out/'risk-competition-batch_jobs.json').write_text(json.dumps([dict(id='worker')]))
            queue=root/'queue.json';queue.write_text(json.dumps(dict(tasks=[dict(id=tid,status='done',
                node='node006',log_path='/tmp/log',exit_code=0) for tid in ('worker','assembly')])))
            responses=iter(['SCHEDULEURM_DONE {}',
                'SCHEDULEURM_RESOURCE {"process_cpu_seconds":70,"peak_rss_mb":4600}',
                '{"complete":true}'])
            with (patch.object(launcher,'ROOT',root),patch.object(launcher,'QUEUE',queue),
                patch.dict('sys.modules',scheduler=types.SimpleNamespace(run_on=lambda *a,**kw:(0,next(responses),''))),
                patch.object(launcher.time,'sleep') as sleep,patch.object(launcher.subprocess,'run') as submit):
                result=launcher.assemble_risk_phase(out,'phase.json')
            submit.assert_not_called();sleep.assert_called_once_with(2)
            self.assertEqual(result['assembly_resources']['process_cpu_seconds'],70)

    def campaign_fixture(self,root):
        phases=[dict(plan=f'phase{i}.json',output=f'phase{i}') for i in (1,2)]
        out=root/'campaign';out.mkdir()
        (root/'campaign.json').write_text(json.dumps(dict(output='campaign',phases=phases,combinations=2)))
        baseline=dict(records=[dict(record='a',raw_winners=[355],native_winners=[355])],
            prior_weights=[.7,.3],action_count=365,fixed_action=355,computed_predictive_origins=[4,5],
            omitted_predictive_origin_mass=0.)
        (out/'baseline_assembly.json').write_text(json.dumps(baseline))
        (root/'paper_artifacts/historical_body_reference_assembly_v1').mkdir(parents=True)
        for p in phases:
            d=root/p['output'];d.mkdir()
            (d/'resolution_plan.json').write_text(json.dumps(dict(estimated_cost_admitted=True,
                cumulative_estimated_cpu_seconds=3200)))
        return out,baseline

    def test_campaign_runs_phases_in_order_and_keeps_numerical_holds_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out,baseline=self.campaign_fixture(root);events=[]
            def collect(stage,phase_out,plan):
                events.append(('collect',plan))
                return dict(complete=True,groups=[dict(record=plan,kind='raw',group=34,
                    hybrid_direct_admitted=plan=='phase2.json')],successful_task_cpu_seconds=100)
            def assemble(phase_out,plan):
                events.append(('assemble',plan))
                return dict(baseline,reference_assembly=dict(groups=10),raw_model_net_preview=.01,
                    assembly_resources=dict(process_cpu_seconds=10))
            def run_on(node,command,**kwargs):
                if 'price_remaining.py' in command:return 0,'priced',''
                return 0,json.dumps({'compact_summary.json':{},'remaining_cost_plan.json':{
                    'missing_group_kind_record_checks':{'raw':1}}}),''
            with (patch.object(launcher,'ROOT',root),patch.object(launcher,'launch',side_effect=lambda s,p:events.append(('launch',p))),
                patch.object(launcher,'collect_remote',side_effect=collect),patch.object(launcher,'assemble_risk_phase',side_effect=assemble),
                patch.dict('sys.modules',scheduler=types.SimpleNamespace(run_on=run_on)),contextlib.redirect_stdout(io.StringIO())):
                result=launcher.continue_risk_competition('campaign.json')
            self.assertEqual(events,[(s,p) for p in ('phase1.json','phase2.json') for s in ('launch','collect','assemble')])
            self.assertEqual(result['state'],'complete_with_precision_holds')
            self.assertEqual(result['completed_combinations'],2)
            self.assertEqual(result['remaining_selected_reference_checks'],1)
            self.assertEqual(result['precision_holds'],[dict(record='phase1.json',kind='raw',group=34)])
            self.assertEqual(json.loads((out/'status.json').read_text()),result)

    def test_bound_campaign_refreshes_coverage_after_actual_reference_assembly(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out,baseline=self.campaign_fixture(root)
            campaign=json.loads((root/'campaign.json').read_text());campaign['certificate_bound_plan']='bound.json'
            (root/'campaign.json').write_text(json.dumps(campaign));(root/'bound.json').write_text(json.dumps(dict(output='bound')))
            (root/'bound').mkdir();calls=[]
            compact=dict(unreferenced_group_record_checks_remaining=0,raw_marginal_numerical_gate_passed=True,cached_reference_containment_passed=True)
            def run_on(node,command,**kwargs):
                calls.append(command)
                if 'probe_historical_certificate_bound' in command:return 0,'reassembled',''
                if '/bound/compact_summary.json' in command:return 0,json.dumps(compact),''
                if 'price_remaining.py' in command:return 0,'priced',''
                return 0,json.dumps({'compact_summary.json':{},'remaining_cost_plan.json':{'missing_group_kind_record_checks':{}}}),''
            assembled=dict(baseline,reference_assembly=dict(groups=10),raw_model_net_preview=.01,assembly_resources=dict(process_cpu_seconds=10))
            summary=dict(complete=True,groups=[dict(record='r',kind='projected',group=71,hybrid_direct_admitted=True)],successful_task_cpu_seconds=100)
            with (patch.object(launcher,'ROOT',root),patch.object(launcher,'launch'),
                patch.object(launcher,'collect_remote',return_value=summary),patch.object(launcher,'assemble_risk_phase',return_value=assembled),
                patch.dict('sys.modules',scheduler=types.SimpleNamespace(run_on=run_on)),contextlib.redirect_stdout(io.StringIO())):
                result=launcher.continue_risk_competition('campaign.json')
            self.assertEqual(result['remaining_bound_competitor_checks'],0)
            self.assertTrue(result['bound_numerical_gate_passed'])
            self.assertEqual(json.loads((root/'bound/compact_summary.json').read_text()),compact)
            self.assertIn('--plan bound.json',calls[-2])

    def test_campaign_stops_after_scheduler_failure_without_launching_next_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out,_=self.campaign_fixture(root)
            with (patch.object(launcher,'ROOT',root),patch.object(launcher,'launch',side_effect=RuntimeError('worker stopped')) as launch):
                with self.assertRaisesRegex(RuntimeError,'worker stopped'):launcher.continue_risk_competition('campaign.json')
            launch.assert_called_once_with('risk-competition-batch','phase1.json')
            self.assertEqual(json.loads((out/'status.json').read_text())['state'],'failed')

    def test_campaign_rejects_phase_over_budget_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out,_=self.campaign_fixture(root)
            (root/'phase1/resolution_plan.json').write_text(json.dumps(dict(estimated_cost_admitted=True,
                cumulative_estimated_cpu_seconds=3601)))
            with patch.object(launcher,'ROOT',root),patch.object(launcher,'launch') as launch:
                with self.assertRaises(AssertionError):launcher.continue_risk_competition('campaign.json')
            launch.assert_not_called()
            self.assertEqual(json.loads((out/'status.json').read_text())['state'],'failed')

    def test_batch_collection_uses_the_requested_manifest_and_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'paper_artifacts/batch_v2';out.mkdir(parents=True)
            plan='performance/manifests/historical_risk_competition_batch_v2.json'
            result=dict(complete=True,groups=[]);commands=[]
            def run_on(node,command,**kwargs):
                commands.append(command)
                return (0,'collected','') if len(commands)==1 else (0,json.dumps(result),'')
            with patch.object(launcher,'ROOT',root),patch.dict('sys.modules',scheduler=types.SimpleNamespace(run_on=run_on)),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(launcher.collect_remote('risk-competition-batch',out,plan),result)
            self.assertIn('--plan '+plan,commands[0])
            self.assertIn('paper_artifacts/batch_v2/summary.json',commands[1])
            self.assertEqual(json.loads((out/'summary.json').read_text()),result)

    def test_batch_reuses_the_frozen_order_and_records_without_reranking(self):
        priced=dict(budgeted_batch_estimated_cpu_seconds=3284.,budgeted_batch=[
            dict(record='b',group=63,kind='projected',minimum_gap=.002),
            dict(record='a',group=63,kind='projected',minimum_gap=.001)])
        with patch('performance.validate_historical_risk_competition.read',return_value=priced) as reader:
            rows=selection_rows(dict(batch_plan='frozen.json',maximum_estimated_cpu_seconds=3600),Path('out'))
        reader.assert_called_once_with('frozen.json')
        self.assertEqual([r['record'] for r in rows],['b','a'])
        self.assertEqual(rows[0]['selection_reasons'],['frozen_gap_ordered_batch'])
        self.assertNotIn('selection_reasons',priced['budgeted_batch'][0])

    def test_repeated_preparation_preserves_completed_parts_and_the_original_cost_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'out';out.mkdir()
            saved=dict(complete=True,cumulative_estimated_cpu_seconds=1612.)
            (out/'preparation.json').write_text(json.dumps(saved))
            part=out/'completed_part.json';part.write_text('{"E9":1}')
            def read(path):
                return dict(output='out') if path==PLAN else json.loads(Path(path).read_text())
            with (patch('performance.validate_historical_risk_competition.ROOT',root),patch(
                'performance.validate_historical_risk_competition.read',side_effect=read),patch(
                'performance.validate_historical_risk_competition.prepare_contexts',side_effect=AssertionError('recomputed preparation')),
                contextlib.redirect_stdout(io.StringIO())):
                self.assertEqual(prepare(),saved)
            self.assertEqual(part.read_text(),'{"E9":1}')

    def test_selects_closest_in_each_kind_and_more_distant_overlap_without_duplicates(self):
        def group(gid,gap,overlap=0):
            return dict(group=gid,minimum_gap=gap,empirical_overlap_action_head_pairs=overlap)
        def report(rows):
            return dict(closest_groups=rows,empirical_overlap_groups=sum(r['empirical_overlap_action_head_pairs']>0 for r in rows))
        records=[dict(record='a',cdf=.1,raw_competitors=report([group(0,.01),group(1,.1,1)]),
            projected_competitors=report([group(2,.02,1)])),
            dict(record='b',cdf=.2,raw_competitors=report([group(3,.03)]),
            projected_competitors=report([group(4,.04)]))]
        selected=select_competitors(dict(records=records))
        self.assertEqual([(r['record'],r['kind'],r['group']) for r in selected],
            [('a','raw',0),('a','raw',1),('a','projected',2)])
        self.assertEqual(selected[-1]['selection_reasons'],['closest_uncovered_competitor','empirical_gap_rule_overlap'])

    def test_does_not_silently_omit_overlaps_outside_the_compact_report(self):
        summary=dict(closest_groups=[dict(group=0,minimum_gap=.1,empirical_overlap_action_head_pairs=0)],
            empirical_overlap_groups=1)
        with self.assertRaises(AssertionError):
            select_competitors(dict(records=[dict(record='a',cdf=.1,raw_competitors=summary)]))


if __name__=='__main__':unittest.main()
