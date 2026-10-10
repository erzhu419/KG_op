import copy
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from core.budgeted_guarded_kg import BudgetedTerminalActions
from core.raw_terminal_head_grid import compose_head_grid
from performance.validate_historical_component_bank import action_groups, group_checks, collect
from performance.diagnose_historical_component_guarded import compressed_write
from performance import launch_historical_component_bank as launcher


class ComponentBankTests(unittest.TestCase):
    def test_allocation_groups_keep_original_action_indices(self):
        terminal=BudgetedTerminalActions([[0,2,3],[1,2,3]],[1,1],dict(comparison_budget=8))
        actions=[dict(kind=k,shortlist=s.tolist(),incumbent_position=int(i),budgets=b)
            for k,table,inc,b in terminal.allocations(359) for s,i in zip(table,inc)]
        groups=action_groups(actions)
        self.assertEqual([g['action_indices'] for g in groups],[[0,2,4],[1,3,5],[6,8],[7,9]])
        self.assertEqual(sorted(i for g in groups for i in g['action_indices']),list(range(10)))

    def test_mixed_gate_catches_one_budget_and_one_identity(self):
        rng=np.random.default_rng(10)
        f=rng.uniform(size=(2,3,3,4));g=rng.uniform(size=(3,3,4));c=np.full((3,3),200.)
        fs=np.array([[.1,.4,-.2],[.3,-.2,.5]]);gs=[.5,-.1,.3];prior=[.2,.5,.3]
        bundle=dict(prior_weights=prior,head_log_scores=gs,actions=[{}, {}, {}],
            opposite=[dict(head_log_scores=s.tolist(),raw=r.tolist()) for s,r in zip(fs,f)])
        parts=[dict(identity=h,rows={n:dict(statistics=g[h].tolist(),costs=c[h].tolist(),
            diagnostics=dict(predictive_log_evidence_error=0.)) for n in ('E7','F7','G7')}) for h in range(3)]
        self.assertTrue(group_checks(bundle,parts,'raw',[('E7','F7'),('E7','G7')],1e-4)['passed'])
        changed=copy.deepcopy(parts);changed[2]['rows']['F7']['statistics'][1][0]+=.01
        result=group_checks(bundle,changed,'raw',[('E7','F7'),('E7','G7')],1e-4)
        new=g.copy();new[2,1,0]+=.01
        before=compose_head_grid(f,g[None],c[None],fs,[gs],prior)[0]
        after=compose_head_grid(f,new[None],c[None],fs,[gs],prior)[0]
        self.assertFalse(result['passed'])
        self.assertAlmostEqual(result['comparisons'][0]['maximum_difference'],float(abs(after-before).max()),places=14)
        self.assertGreater(result['comparisons'][0]['action_differences'][1],1e-4)
        self.assertEqual(result['comparisons'][0]['action_differences'][0],0.)
        changed=copy.deepcopy(parts);changed[1]['rows']['G7']['diagnostics']['predictive_log_evidence_error']=2e-4
        self.assertFalse(group_checks(bundle,changed,'raw',[('E7','F7'),('E7','G7')],1e-4)['passed'])

    def test_full_collector_preserves_noncontiguous_budgets_and_native_direct(self):
        terminal=BudgetedTerminalActions([[0,2,3],[1,2,3]],[1,1],dict(comparison_budget=8))
        actions=[dict(kind=k,shortlist=s.tolist(),incumbent_position=int(i),budgets=b)
            for k,table,inc,b in terminal.allocations(359) for s,i in zip(table,inc)]
        groups=action_groups(actions);rng=np.random.default_rng(22)
        f=rng.uniform(size=(2,3,10,4));g=rng.uniform(size=(3,10,4));cost=np.full((3,10),200.)
        baseline=np.zeros_like(g);baseline[:,6:]=g[:,6:]
        opposite=[dict(position=i,head_log_scores=[.1,.2,.3],raw=f[i].tolist(),projected=f[i].tolist()) for i in range(2)]
        context=dict(actions=actions,groups=groups,head_log_scores=[.3,.1,.2],prior_weights=[.2,.5,.3],
            utility_identities=[0,1,2],opposite=opposite,
            baseline={k:dict(statistics=baseline.tolist(),costs=cost.tolist()) for k in ('raw','projected')})
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);(out/'parts').mkdir();(out/'tasks/screen').mkdir(parents=True)
            (out/'context.json').write_text(json.dumps(context))
            plan=out/'plan.json';plan.write_text(json.dumps(dict(record='toy',output=str(out),
                shards_per_identity_kind=4,maximum_action_difference=1e-4)))
            for group in groups:
                gid=group['group'];ix=group['action_indices']
                bundle=dict(prior_weights=context['prior_weights'],head_log_scores=context['head_log_scores'],
                    actions=[actions[i] for i in ix],opposite=[dict(r,**{k:f[i,:,ix,:].transpose(1,0,2).tolist()
                        for k in ('raw','projected')}) for i,r in enumerate(opposite)])
                compressed_write(out/'marginals'/f'{gid:03d}.json.gz',bundle)
                for h in range(3):
                    for kind in ('raw','projected'):
                        if kind=='projected' and group['kind']=='direct':continue
                        rows={n:dict(statistics=g[h,ix].tolist(),costs=cost[h,ix].tolist(),
                            diagnostics=dict(predictive_log_evidence_error=0.)) for n in ('E5','E7','F7','G7')}
                        (out/'parts'/f'{gid:03d}_{h}_{kind}.json').write_text(json.dumps(dict(identity=h,rows=rows)))
            for h in range(3):
                for kind,count in [('raw',4),('projected',2)]:
                    for s in range(count):
                        (out/'tasks/screen'/f'{h}_{kind}_{s}.json').write_text(json.dumps(dict(complete=True,cpu_seconds=0.,peak_rss_mb=0.)))
            with contextlib.redirect_stdout(io.StringIO()):result=collect('screen',str(plan))
            self.assertTrue(result['single_record_risk_bank_precision_passed'])
            self.assertEqual(result['actions_checked'],dict(raw=10,projected=10))
            table=json.loads((out/'screen_risk_bank.json').read_text())
            for kind in ('raw','projected'):
                np.testing.assert_array_equal(table['statistics'][kind],g)
                np.testing.assert_array_equal(table['costs'][kind],cost)
            path=out/'parts/000_2_raw.json';part=json.loads(path.read_text())
            part['rows']['E5']['statistics'][1][0]+=.01;path.write_text(json.dumps(part))
            with contextlib.redirect_stdout(io.StringIO()):result=collect('screen',str(plan))
            self.assertFalse(result['single_record_risk_bank_precision_passed'])
            self.assertEqual(result['failed_groups']['raw'],[0])
            self.assertEqual(result['actions_checked']['raw'],7)
            part['rows'].update({n:copy.deepcopy(part['rows']['G7']) for n in ('H8','I8','J8')})
            path.write_text(json.dumps(part))
            for h in (0,1):
                path=out/f'parts/000_{h}_raw.json';part=json.loads(path.read_text())
                part['rows'].update({n:copy.deepcopy(part['rows']['G7']) for n in ('H8','I8','J8')})
                path.write_text(json.dumps(part))
            (out/'tasks/refine').mkdir()
            for h in range(3):
                (out/f'tasks/refine/{h}_raw_0.json').write_text(json.dumps(dict(complete=True,cpu_seconds=1.,peak_rss_mb=0.)))
            with contextlib.redirect_stdout(io.StringIO()):result=collect('refine',str(plan))
            self.assertTrue(result['single_record_risk_bank_precision_passed'])
            self.assertEqual(result['groups'][0]['selected_rules']['raw'],'H8')
            self.assertEqual(result['groups'][1]['selected_rules']['raw'],'G7')
            table=json.loads((out/'refine_risk_bank.json').read_text())
            for kind in ('raw','projected'):np.testing.assert_array_equal(table['statistics'][kind],g)

    def test_pipeline_stops_without_submitting_passed_or_over_budget_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'performance/manifests').mkdir(parents=True);out=root/'out';out.mkdir()
            (root/'performance/manifests/historical_component_bank_v1.json').write_text(json.dumps(dict(
                output='out',allocation_cache='cache',shards_per_identity_kind=1,maximum_estimated_adaptive_cpu_seconds=10.)))
            (out/'preparation.json').write_text(json.dumps(dict(source_components={'4':512,'5':1024},cpu_seconds=2.)))
            (out/'screen_jobs.json').write_text(json.dumps([dict(id=str(i)) for i in range(4)]))
            queue=root/'queue.json';queue.write_text(json.dumps(dict(tasks=[dict(id=str(i),status='done') for i in range(4)])))
            for folder,key,kind in [('compact_parts','cpu_seconds','raw'),('parts','shared_cpu_seconds','projected')]:
                (root/'cache'/folder).mkdir(parents=True)
                for h in (4,5):
                    for a in (0,1):
                        (root/'cache'/folder/f'{h}_{a}_{kind}.json').write_text(json.dumps(dict(rows=dict(candidate={key:1.}))))
            for passed,expected in [(True,'single_record_risk_precision_passed'),(False,'adaptive_cost_hold')]:
                summary=dict(complete=True,stage='screen',cpu_seconds=3.,single_record_risk_bank_precision_passed=passed,
                    failed_groups=dict(raw=[] if passed else [0,1],projected=[]))
                with patch.object(launcher,'ROOT',root),patch.object(launcher,'QUEUE',queue),\
                    patch.object(launcher,'collect_remote',return_value=summary),patch.object(launcher,'launch') as launch,\
                    contextlib.redirect_stdout(io.StringIO()):launcher.continue_pipeline()
                launch.assert_not_called();status=json.loads((out/'pipeline_status.json').read_text())
                self.assertEqual(status['state'],expected)
                self.assertEqual(status['measured_cpu_seconds'],5.)
                self.assertFalse(status['full_predictive_integration_admitted'])
                self.assertFalse(status['independent_confirmation'])


if __name__=='__main__':unittest.main()
