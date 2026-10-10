import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from performance import launch_historical_component_bank as launcher
from performance import validate_historical_component_transfer as transfer
from performance.diagnose_historical_component_guarded import compressed_write
from performance.refine_historical_observed_prediction import field_from_snapshot
from performance.validate_historical_component_bank import evaluate_group_rules
from representation.vectorized_joint_function_risk import update_raw_observation


class ComponentTransferTests(unittest.TestCase):
    def test_record_metadata_changes_without_copying_or_mutating_joint_prior(self):
        shared=dict(group=52,name='old',label=-.1,head_log_scores=[0.]*8,prior_weights=[.125]*8,
            utility_identities=[4,5],fields=[None]*8,opposite=[],actions=[dict(budgets=[1,2,3])],protocol={'x':1})
        context=dict(name='new',head=1,replicates=1,eta_label=None,label=.2,head_log_scores=list(range(8)),
            utility_identities=[4,5],prior_weights=[.125]*8,groups={'52':dict(opposite=[{'position':3}])})
        bundle=transfer.record_bundle(shared,context)
        self.assertEqual(bundle['label'],.2)
        self.assertEqual(bundle['head_log_scores'],list(range(8)))
        self.assertIs(bundle['fields'],shared['fields'])
        self.assertIs(bundle['actions'],shared['actions'])
        self.assertEqual(shared['label'],-.1)
        self.assertEqual(shared['opposite'],[])
        context['prior_weights'][0]=.2
        with self.assertRaises(AssertionError):transfer.record_bundle(shared,context)

    def test_shared_evaluator_conditions_on_new_raw_label(self):
        snapshot=dict(means=[[0.,-1.],[0.,.1]],covariances=[np.diag([.1,.01]).tolist()]*2,
            log_weights=np.log([.4,.6]).tolist(),paid_batches=2,paid_episodes=2)
        values=[]
        for label in (-1.,1.):
            field=field_from_snapshot(snapshot);score=update_raw_observation(field,0,label,None,1)
            bundle=dict(fields=[snapshot],head_log_scores=[score],label=label,kind='direct',points=[63],
                actions=[dict(kind='direct',shortlist=[0],incumbent_position=0,budgets=[359])],seed=3,
                protocol=dict(chance_alpha=.05,familywise_delta=.05))
            _,row=next(evaluate_group_rules(bundle,0,'raw',['E5']))
            self.assertEqual(row['diagnostics']['components'],2)
            self.assertTrue(row['diagnostics']['all_positive_components_retained'])
            values.append(row['statistics'][0][0])
        self.assertGreater(values[0],values[1])

    def test_worker_creates_completion_directory_and_reuses_finished_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);plan=out/'plan.json';parent=out/'parent.json'
            plan.write_text(json.dumps(dict(output=str(out),parent_plan=str(parent),maximum_job_cpu_seconds=5.)))
            parent.write_text(json.dumps(dict(output=str(out/'base'))))
            row=dict(record='low',group=0,kind='raw',rules=['E5'])
            (out/'screen_plan.json').write_text(json.dumps(dict(estimated_cost_admitted=True,selected=[row])))
            (out/'contexts.json').write_text(json.dumps({'low':{}}))
            value=dict(statistics=[[1.,0.,0.,0.]],costs=[359.],cpu_seconds=.1)
            path=transfer.part_path(out,row,4);path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(identity=4,rows={'E5':value})))
            with patch.object(transfer,'compressed_read',return_value={}),patch.object(transfer,'record_bundle',return_value={}),\
                patch.object(transfer,'evaluate_group_rules',return_value=iter(())) as evaluate,contextlib.redirect_stdout(io.StringIO()):
                result=transfer.run(4,'raw',0,'screen',str(plan))
            self.assertTrue(result['complete'])
            self.assertEqual(evaluate.call_args.args[-1],[])
            self.assertEqual(json.loads(path.read_text())['rows']['E5'],value)
            self.assertTrue(json.loads((out/'tasks/screen/4_raw_0.json').read_text())['complete'])

    def test_collector_preserves_other_records_and_all_budget_variants(self):
        rng=np.random.default_rng(71);prior=[.125]*8;heads=rng.normal(size=8).tolist()
        opposite=[dict(position=i,head_log_scores=rng.normal(size=8).tolist(),
            raw=rng.uniform(size=(8,3,4)).tolist(),projected=rng.uniform(size=(8,3,4)).tolist()) for i in range(2)]
        s=np.zeros((8,3,4));s[4:6]=rng.uniform(size=(2,3,4));cost=np.zeros((8,3));cost[4:6]=200.
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);base=out/'base';plan=out/'plan.json';parent=out/'parent.json'
            parent.write_text(json.dumps(dict(output=str(base))))
            plan.write_text(json.dumps(dict(output=str(out),parent_plan=str(parent),records=['low','high'],groups=[0],
                maximum_action_difference=1e-4,maximum_estimated_cpu_seconds=1000.)))
            shared=dict(group=0,prior_weights=prior,utility_identities=[4,5],actions=[{},{},{}])
            compressed_write(base/'marginals/000.json.gz',shared)
            contexts={n:dict(name=n,head=1,replicates=1,eta_label=None,label=y,head_log_scores=heads,
                prior_weights=prior,utility_identities=[4,5],groups={'0':dict(action_indices=[2,5,8],opposite=opposite,
                baseline={k:dict(statistics=s.tolist(),costs=cost.tolist()) for k in ('raw','projected')})}) for n,y in [('low',-.3),('high',-.05)]}
            (out/'contexts.json').write_text(json.dumps(contexts))
            selected=[dict(record=n,group=0,kind=k,rules=['E5','E7','F7','G7'],
                pairs=[['E5','E7'],['E7','F7'],['E7','G7']],selected_rule='E7') for n in contexts for k in ('raw','projected')]
            (out/'screen_plan.json').write_text(json.dumps(dict(selected=selected,cumulative_estimated_cpu_seconds=20.)))
            probe=dict(groups={'0':dict(times={k:{str(h):{r:1. for r in ['E8','F8','H8']} for h in (4,5)} for k in ('raw','projected')})},
                resolution_units={k+'_'+str(h):dict(E9=2.,F9=2.,K9=4.) for k in ('raw','projected') for h in (4,5)})
            (out/'cost_probe.json').write_text(json.dumps(probe))
            for row in selected:
                shard=0 if row['record']=='low' else 1
                for h in (4,5):
                    path=transfer.part_path(out,row,h);path.parent.mkdir(parents=True,exist_ok=True)
                    part=dict(identity=h,rows={r:dict(statistics=s[h].tolist(),costs=cost[h].tolist(),cpu_seconds=.1,
                        diagnostics=dict(predictive_log_evidence_error=0.)) for r in ['E5','E7','F7','G7','E8','F8','H8']})
                    if row['record']=='low' and row['kind']=='raw' and h==5:part['rows']['E5']['statistics'][1][0]+=.01
                    path.write_text(json.dumps(part))
                    task=out/'tasks/screen'/f'{h}_{row["kind"]}_{shard}.json';task.parent.mkdir(parents=True,exist_ok=True)
                    task.write_text(json.dumps(dict(complete=True,cpu_seconds=1.,peak_rss_mb=1.)))
            with contextlib.redirect_stdout(io.StringIO()):screen=transfer.collect('screen',str(plan))
            self.assertEqual(screen['failed_groups'],[dict(record='low',group=0,kind='raw')])
            self.assertAlmostEqual(screen['cpu_seconds'],3.2)
            self.assertEqual(screen['successful_task_cpu_seconds'],8.)
            self.assertEqual(screen['selected_action_record_checks_passed'],dict(raw=3,projected=6))
            self.assertEqual(screen['selected_action_record_checks_total'],dict(raw=6,projected=6))
            self.assertEqual(len(screen['next_plan']['selected']),1)
            for h in (4,5):
                task=out/'tasks/adaptive'/f'{h}_raw_0.json';task.parent.mkdir(parents=True,exist_ok=True)
                task.write_text(json.dumps(dict(complete=True,cpu_seconds=1.,peak_rss_mb=1.)))
            with contextlib.redirect_stdout(io.StringIO()):admitted=transfer.collect('adaptive',str(plan))
            self.assertTrue(admitted['all_selected_action_precision_passed'])
            self.assertFalse(admitted['full_predictive_integration_admitted'])
            self.assertEqual(admitted['selected_action_record_checks_passed'],dict(raw=6,projected=6))
            row=dict(record='low',group=0,kind='raw');path=transfer.part_path(out,row,5)
            part=json.loads(path.read_text());part['rows']['F8']['statistics'][2][0]+=.01;path.write_text(json.dumps(part))
            with contextlib.redirect_stdout(io.StringIO()):failed=transfer.collect('adaptive',str(plan))
            self.assertEqual(failed['next_plan']['selected'][0]['rules'],['E9','F9','L9'])
            self.assertEqual(failed['next_plan']['selected'][0]['selected_rule'],'E9')
            self.assertEqual(len(failed['groups']),4)
            part['rows']['F8']['statistics'][2][0]-=.01
            part['rows']['H8']['statistics'][1][0]+=.01;path.write_text(json.dumps(part))
            with contextlib.redirect_stdout(io.StringIO()):gh_failed=transfer.collect('adaptive',str(plan))
            row=gh_failed['next_plan']['selected'][0]
            self.assertEqual(row['rules'],['I8','J8','K9'])
            self.assertEqual(row['selected_rule'],'H8')
            self.assertIn(['H8','J8'],row['pairs'])

    def test_controller_stops_before_over_budget_adaptation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'out';out.mkdir();(root/'performance/manifests').mkdir(parents=True)
            (root/'performance/manifests/historical_component_transfer_v1.json').write_text(json.dumps(dict(output='out')))
            (out/'screen_plan.json').write_text(json.dumps(dict(estimated_cost_admitted=True,cumulative_estimated_cpu_seconds=10.)))
            (out/'adaptive_plan.json').write_text(json.dumps(dict(estimated_cost_admitted=False,cumulative_estimated_cpu_seconds=101.)))
            summary=dict(cpu_seconds=8.,failed_groups=[dict(record='low',group=0,kind='raw')],
                selected_action_record_checks_passed=dict(raw=3,projected=6),all_selected_action_precision_passed=False,next_plan={})
            with patch.object(launcher,'ROOT',root),patch.object(launcher,'launch') as launch,\
                patch.object(launcher,'collect_remote',return_value=summary),contextlib.redirect_stdout(io.StringIO()):
                launcher.continue_transfer()
            launch.assert_called_once_with('transfer-screen')
            status=json.loads((out/'status.json').read_text())
            self.assertEqual(status['state'],'cost_hold')
            self.assertEqual(status['measured_cpu_seconds'],8.)
            self.assertFalse(status['independent_confirmation'])


if __name__=='__main__':unittest.main()
