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
from performance.validate_historical_component_bank import (action_groups, group_checks, collect,
    adaptive_resolution_rules, prepare_resolution_pilot, prepare_tail, prepare_tail_resolution, RULES)
from performance.diagnose_historical_component_guarded import compressed_write
from performance import launch_historical_component_bank as launcher
from performance.diagnose_historical_bank_failures import select_failures


class ComponentBankTests(unittest.TestCase):
    def test_failure_pilot_selects_observed_worst_cases_and_deduplicates(self):
        groups=[]
        for gid,(coarse,independent,gh) in enumerate([(2e-4,3e-4,0.),(4e-4,5e-4,2e-4),(1e-5,1e-5,1e-5)]):
            checks=dict(comparisons=[dict(low=a,high=b,maximum_difference=d)
                for (a,b),d in zip([('E5','E7'),('E7','F7'),('E7','G7')],(coarse,independent,gh))])
            groups.append(dict(group=gid,checks=dict(raw=checks,projected=dict(comparisons=[]))))
        selected=select_failures(dict(groups=groups),1e-4)
        self.assertEqual(len(selected),1)
        self.assertEqual((selected[0]['kind'],selected[0]['group']),('raw',1))
        self.assertEqual(len(selected[0]['reasons']),3)
        self.assertEqual(select_failures(dict(groups=groups),1e-3),[])

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

            # Resume from the original failed screen, merge an already confirmed
            # pilot, and refine only the other failed group (three separated budgets).
            path=out/'parts/001_2_raw.json';part=json.loads(path.read_text())
            part['rows']['E5']['statistics'][1][0]+=.01;path.write_text(json.dumps(part))
            with contextlib.redirect_stdout(io.StringIO()):screen=collect('screen',str(plan))
            self.assertEqual(screen['failed_groups']['raw'],[0,1])
            (out/'failure_pilot').mkdir()
            (out/'failure_pilot/cheaper_summary.json').write_text(json.dumps(dict(complete=True,
                selected_failure_precision_passed=True,groups=[dict(group=0,kind='raw',checks=result['groups'][0]['checks']['raw'])])))
            for gid in (0,1):
                for h in range(3):
                    path=out/f'parts/{gid:03d}_{h}_raw.json';part=json.loads(path.read_text())
                    part['rows'].update({n:dict(copy.deepcopy(part['rows']['G7']),cpu_seconds=1.) for n in ('E8','F8','H8')})
                    if gid==1 and h==2:part['rows']['E7']['statistics'][1][0]+=.01
                    path.write_text(json.dumps(part))
            (out/'tasks/adaptive').mkdir()
            for h in range(3):
                (out/f'tasks/adaptive/{h}_raw_0.json').write_text(json.dumps(dict(complete=True,cpu_seconds=1.,peak_rss_mb=0.)))
            with contextlib.redirect_stdout(io.StringIO()):adaptive=collect('adaptive',str(plan))
            self.assertEqual(adaptive['failed_groups'],dict(raw=[1],projected=[]))
            self.assertEqual(adaptive['groups'][0]['selected_rules']['raw'],'E8')
            self.assertEqual(adaptive['next_resolution_estimates']['raw']['1']['rules'],['K9'])
            self.assertEqual(adaptive['next_resolution_estimates']['raw']['1']['estimated_cpu_seconds'],7.5)
            self.assertEqual(json.loads((out/'screen_summary.json').read_text())['failed_groups']['raw'],[0,1])
            for h in range(3):
                path=out/f'parts/001_{h}_raw.json';part=json.loads(path.read_text())
                part['rows']['K9']=copy.deepcopy(part['rows']['H8']);path.write_text(json.dumps(part))
            (out/'tasks/adaptive-resolution').mkdir()
            for h in range(3):
                (out/f'tasks/adaptive-resolution/{h}_raw_0.json').write_text(json.dumps(dict(complete=True,cpu_seconds=1.,peak_rss_mb=0.)))
            with contextlib.redirect_stdout(io.StringIO()):resolved=collect('adaptive-resolution',str(plan))
            self.assertTrue(resolved['single_record_risk_bank_precision_passed'])
            self.assertEqual(resolved['actions_checked'],dict(raw=10,projected=10))
            table=json.loads((out/'adaptive-resolution_risk_bank.json').read_text())
            for kind in ('raw','projected'):np.testing.assert_array_equal(table['statistics'][kind],g)

            # A GH increase cannot fix an unstable independent continuous rule.
            # Check both independent 512-point rules and preserve unselected failures.
            hard=copy.deepcopy(adaptive)
            hard['groups'][0]['checks']['raw']['passed']=False
            hard['groups'][1]['checks']['raw']['comparisons'][1]['maximum_difference']=2e-4
            (out/'adaptive_summary.json').write_text(json.dumps(hard))
            (out/'bank_resolution_plan.json').write_text(json.dumps(dict(estimated_cost_admitted=True,
                selected=[dict(group=1,kind='raw')])))
            for h in range(3):
                path=out/f'parts/001_{h}_raw.json';part=json.loads(path.read_text())
                part['rows'].update({n:copy.deepcopy(part['rows']['H8']) for n in ('E9','F9')})
                path.write_text(json.dumps(part))
            (out/'tasks/bank-resolution').mkdir()
            for h in range(3):
                (out/f'tasks/bank-resolution/{h}_raw_0.json').write_text(json.dumps(dict(complete=True,cpu_seconds=1.,peak_rss_mb=0.)))
            with contextlib.redirect_stdout(io.StringIO()):partial=collect('bank-resolution',str(plan))
            self.assertEqual(partial['failed_groups'],dict(raw=[0],projected=[]))
            self.assertEqual(partial['groups'][1]['selected_rules']['raw'],'E9')
            self.assertFalse(partial['single_record_risk_bank_precision_passed'])
            path=out/'parts/001_2_raw.json';part=json.loads(path.read_text())
            part['rows']['F9']['statistics'][1][0]+=.01;path.write_text(json.dumps(part))
            with contextlib.redirect_stdout(io.StringIO()):partial=collect('bank-resolution',str(plan))
            self.assertEqual(partial['failed_groups']['raw'],[0,1])

            coarse=copy.deepcopy(adaptive)
            coarse['groups'][0]['checks']['raw']=copy.deepcopy(coarse['groups'][1]['checks']['raw'])
            coarse['groups'][0]['checks']['raw']['comparisons'][0]['maximum_difference']=0.
            coarse['groups'][0]['checks']['raw']['comparisons'][1]['maximum_difference']=2e-4
            (out/'bank_resolution_summary.json').write_text(json.dumps(coarse))
            (out/'tasks/bank-coarse').mkdir()
            for h in range(3):
                (out/f'tasks/bank-coarse/{h}_raw_0.json').write_text(json.dumps(dict(complete=True,cpu_seconds=1.,peak_rss_mb=0.)))
            with contextlib.redirect_stdout(io.StringIO()):partial=collect('bank-coarse',str(plan))
            self.assertEqual(partial['failed_groups'],dict(raw=[0],projected=[]))
            self.assertEqual(partial['groups'][1]['selected_rules']['raw'],'E8')
            self.assertEqual([(c['low'],c['high']) for c in partial['groups'][1]['checks']['raw']['comparisons']],
                [('E8','F8'),('E8','H8'),('E8','E9')])
            progress=json.loads((out/'partial_bank_progress.json').read_text())
            self.assertEqual(progress['stage'],'bank-coarse')
            self.assertEqual(progress['remaining_failed_groups'],dict(raw=[0],projected=[]))

            tail=copy.deepcopy(partial);tail['groups'][1]['checks']['raw']['passed']=False
            (out/'bank_coarse_summary.json').write_text(json.dumps(tail))
            (out/'bank_tail_plan.json').write_text(json.dumps(dict(estimated_cost_admitted=True,selected=[
                dict(group=gid,kind='raw',continuous_reference='E8' if gid==0 else 'E10') for gid in (0,1)])))
            for gid in (0,1):
                for h in range(3):
                    path=out/f'parts/{gid:03d}_{h}_raw.json';part=json.loads(path.read_text())
                    part['rows'].update({n:copy.deepcopy(part['rows']['H8']) for n in ('E9','F9','L9','E10')})
                    if gid==1 and h==2:part['rows']['E8']['statistics'][1][0]+=.01
                    path.write_text(json.dumps(part))
            (out/'tasks/bank-tail').mkdir()
            for h in range(3):
                for shard in (0,1):
                    (out/f'tasks/bank-tail/{h}_raw_{shard}.json').write_text(json.dumps(dict(complete=True,cpu_seconds=1.,peak_rss_mb=0.)))
            with contextlib.redirect_stdout(io.StringIO()):admitted=collect('bank-tail',str(plan))
            self.assertTrue(admitted['single_record_risk_bank_precision_passed'])
            self.assertEqual(admitted['actions_checked'],dict(raw=10,projected=10))
            self.assertEqual(admitted['groups'][1]['checks']['raw']['comparisons'][0]['high'],'E10')
            table=json.loads((out/'bank-tail_risk_bank.json').read_text())
            for kind in ('raw','projected'):np.testing.assert_array_equal(table['statistics'][kind],g)
            for gid,name in [(0,'L9'),(1,'E10')]:
                path=out/f'parts/{gid:03d}_2_raw.json';part=json.loads(path.read_text())
                part['rows'][name]['statistics'][1][0]+=.01;path.write_text(json.dumps(part))
                with contextlib.redirect_stdout(io.StringIO()):failed=collect('bank-tail',str(plan))
                self.assertEqual(failed['failed_groups']['raw'],[gid])
                part['rows'][name]['statistics'][1][0]-=.01;path.write_text(json.dumps(part))

            # Replace only the failed coarse comparison, retain the independently
            # checked selected rule and all other admitted groups.
            held=copy.deepcopy(admitted);held['groups'][0]['checks']['raw']['passed']=False
            (out/'bank_tail_summary.json').write_text(json.dumps(held))
            (out/'bank_tail_resolution_plan.json').write_text(json.dumps(dict(estimated_cost_admitted=True,
                selected=[dict(group=0,kind='raw')])))
            (out/'tasks/bank-tail-resolution').mkdir()
            for h in range(3):
                path=out/f'parts/000_{h}_raw.json';part=json.loads(path.read_text())
                part['rows']['E10']=copy.deepcopy(part['rows']['E9']);path.write_text(json.dumps(part))
                (out/f'tasks/bank-tail-resolution/{h}_raw_0.json').write_text(json.dumps(dict(complete=True,cpu_seconds=1.,peak_rss_mb=0.)))
            with contextlib.redirect_stdout(io.StringIO()):resolved=collect('bank-tail-resolution',str(plan))
            self.assertTrue(resolved['single_record_risk_bank_precision_passed'])
            self.assertEqual(resolved['groups'][0]['selected_rules']['raw'],'E9')
            self.assertEqual(resolved['groups'][1],held['groups'][1])
            self.assertFalse(resolved['full_predictive_integration_admitted'])
            path=out/'parts/000_2_raw.json';part=json.loads(path.read_text())
            part['rows']['E10']['statistics'][1][0]+=.01;path.write_text(json.dumps(part))
            with contextlib.redirect_stdout(io.StringIO()):failed=collect('bank-tail-resolution',str(plan))
            self.assertEqual(failed['failed_groups']['raw'],[0])

    def test_tail_resolution_counts_previous_estimate_and_rejects_independent_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);plan=out/'plan.json'
            plan.write_text(json.dumps(dict(output=str(out),record='toy',maximum_action_difference=1e-4)))
            check=dict(maximum_log_evidence_error=0.,comparisons=[dict(low=a,high=b,maximum_difference=d)
                for a,b,d in [('E8','E9',2e-4),('E9','F9',2e-5),('L9','E9',1e-9)]])
            summary=dict(complete=True,failed_groups=dict(raw=[0],projected=[]),
                groups=[dict(group=0,checks=dict(raw=check))])
            (out/'bank_tail_summary.json').write_text(json.dumps(summary))
            (out/'tail_cost_probe.json').write_text(json.dumps(dict(units={f'raw_{h}':dict(E9=10.) for h in (4,5)})))
            for limit in (100.,90.):
                (out/'bank_tail_plan.json').write_text(json.dumps(dict(estimated_cpu_seconds=40.,maximum_estimated_cpu_seconds=limit)))
                with contextlib.redirect_stdout(io.StringIO()):selected=prepare_tail_resolution(str(plan))
                self.assertEqual(selected['estimated_cpu_seconds'],60.)
                self.assertEqual(selected['cumulative_estimated_cpu_seconds'],100.)
                self.assertEqual(selected['estimated_cost_admitted'],limit>=100.)
                self.assertEqual(selected['selected'][0]['rules'],['E10'])
            check['comparisons'][1]['maximum_difference']=2e-4
            (out/'bank_tail_summary.json').write_text(json.dumps(summary))
            with self.assertRaises(AssertionError):prepare_tail_resolution(str(plan))

    def test_tail_plan_keeps_just_over_threshold_failure_and_accounts_for_cached_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);plan=out/'plan.json'
            checks=[dict(comparisons=[dict(low='E8',high='F8',maximum_difference=2e-4)]),
                dict(comparisons=[dict(low='E8',high='E9',maximum_difference=1.00245e-4)])]
            summary=dict(complete=True,failed_groups=dict(raw=[0,1],projected=[]),
                groups=[dict(group=gid,checks=dict(raw=check)) for gid,check in enumerate(checks)])
            (out/'bank_coarse_summary.json').write_text(json.dumps(summary))
            probe=dict(selected=[dict(group=gid,kind='raw',missing={str(h):rules for h in (4,5)}) for gid,rules in
                [(0,['E9','F9','K9']),(1,['F9','K9','E10'])]],
                units={f'raw_{h}':dict(E9=10.,F9=10.,K9=20.) for h in (4,5)})
            (out/'tail_cost_probe.json').write_text(json.dumps(probe))
            for limit in (200.,150.):
                plan.write_text(json.dumps(dict(output=str(out),record='toy',maximum_action_difference=1e-4,
                    maximum_estimated_adaptive_cpu_seconds=limit)))
                with contextlib.redirect_stdout(io.StringIO()):selected=prepare_tail(str(plan))
                self.assertAlmostEqual(selected['estimated_cpu_seconds'],168.)
                self.assertEqual(selected['estimated_cost_admitted'],limit>=168.)
                self.assertEqual(selected['selected'][1]['continuous_reference'],'E10')
                self.assertIn('E10',selected['selected'][1]['rules'])
                self.assertEqual(selected['selected'][0]['rules'],['E9','F9','L9'])

    def test_resolution_keeps_failed_independent_and_gh_checks(self):
        checks=dict(maximum_log_evidence_error=0.,comparisons=[dict(low=a,high=b,maximum_difference=d)
            for a,b,d in [('E7','E8',2e-4),('E8','F8',1e-5),('E8','H8',1e-5)]])
        self.assertEqual(adaptive_resolution_rules(checks,1e-4),['K9'])
        for i in (1,2):
            failed=copy.deepcopy(checks);failed['comparisons'][i]['maximum_difference']=2e-4
            self.assertEqual(adaptive_resolution_rules(failed,1e-4),['E9','F9','K9'])
        checks['maximum_log_evidence_error']=2e-4
        self.assertEqual(adaptive_resolution_rules(checks,1e-4),['E9','F9','K9'])
        self.assertEqual(RULES['E9'][:2],(32,9))
        self.assertEqual(RULES['F9'][:2],(32,9))
        self.assertNotEqual(RULES['E9'][2],RULES['F9'][2])

    def test_resolution_pilot_uses_current_worst_comparisons_and_budget(self):
        groups=[];estimates={k:{} for k in ('raw','projected')}
        for gid in range(2):
            checks={}
            for kind in estimates:
                differences=[3e-4,5e-5,0.] if gid==0 else [5e-5,2e-4,0.]
                checks[kind]=dict(passed=False,maximum_log_evidence_error=0.,comparisons=[
                    dict(low=a,high=b,maximum_difference=d) for (a,b),d in
                    zip([('E7','E8'),('E8','F8'),('E8','H8')],differences)])
                estimates[kind][str(gid)]=dict(rules=['K9'] if gid==0 else ['I8','J8','K9'],
                    estimated_cpu_seconds=250. if gid==0 else 550.)
            groups.append(dict(group=gid,checks=checks))
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);plan=out/'plan.json'
            plan.write_text(json.dumps(dict(output=str(out),record='toy',maximum_action_difference=1e-4)))
            (out/'adaptive_summary.json').write_text(json.dumps(dict(complete=True,groups=groups,next_resolution_estimates=estimates)))
            for limit in (1600.,1400.):
                (out/'next_bank_plan.json').write_text(json.dumps(dict(maximum_estimated_resolution_cpu_seconds=limit)))
                with contextlib.redirect_stdout(io.StringIO()):selected=prepare_resolution_pilot(str(plan))
                self.assertEqual(len(selected['selected']),4)
                self.assertEqual(selected['estimated_cpu_seconds'],1500.)
                self.assertEqual(selected['estimated_cost_admitted'],limit>=1500.)
                self.assertEqual(selected['selected'][1]['rules'],['E9','F9','K9'])

    def test_adaptive_controller_respects_resolution_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'performance/manifests').mkdir(parents=True);out=root/'out';out.mkdir()
            (root/'performance/manifests/historical_component_bank_v1.json').write_text(json.dumps(dict(output='out')))
            (out/'next_bank_plan.json').write_text(json.dumps(dict(estimated_initial_refinement_cpu_seconds=20.,
                maximum_estimated_resolution_cpu_seconds=10.)))
            for estimate,expected,stages in [(11.,'resolution_cost_hold',['adaptive']),
                    (9.,'single_record_risk_precision_passed',['adaptive','adaptive-resolution'])]:
                summary=dict(stage='adaptive',cpu_seconds=3.,single_record_risk_bank_precision_passed=False,
                    failed_groups=dict(raw=[1],projected=[]),actions_checked=dict(raw=7,projected=10),
                    next_resolution_estimates=dict(raw={'1':dict(estimated_cpu_seconds=estimate)}))
                resolved=dict(summary,stage='adaptive-resolution',cpu_seconds=2.,single_record_risk_bank_precision_passed=True,
                    failed_groups=dict(raw=[],projected=[]),actions_checked=dict(raw=10,projected=10))
                with patch.object(launcher,'ROOT',root),patch.object(launcher,'launch') as launch,\
                    patch.object(launcher,'collect_remote',side_effect=[summary,resolved]),\
                    contextlib.redirect_stdout(io.StringIO()):launcher.continue_adaptive()
                self.assertEqual([c.args[0] for c in launch.call_args_list],stages)
                status=json.loads((out/'adaptive_status.json').read_text());self.assertEqual(status['state'],expected)
                self.assertEqual(status['measured_cpu_seconds'],3. if estimate>10. else 5.)
                self.assertFalse(status['full_predictive_integration_admitted'])

    def test_coarse_controller_counts_the_already_budgeted_pilot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'performance/manifests').mkdir(parents=True);out=root/'out';out.mkdir()
            (root/'performance/manifests/historical_component_bank_v1.json').write_text(json.dumps(dict(
                output='out',maximum_action_difference=1e-4)))
            (out/'bank_resolution_status.json').write_text(json.dumps(dict(state='selected_resolution_passed')))
            groups=[dict(group=0,checks={k:dict(passed=True) for k in ('raw','projected')}),
                dict(group=1,checks={k:dict(passed=False,maximum_log_evidence_error=0.,comparisons=[
                    dict(low=a,high=b,maximum_difference=d) for a,b,d in
                    [('E7','E8',2e-4),('E8','F8',1e-5),('E8','H8',1e-5)]]) for k in ('raw','projected')})]
            (out/'bank_resolution_summary.json').write_text(json.dumps(dict(complete=True,groups=groups)))
            (out/'adaptive_summary.json').write_text(json.dumps(dict(next_resolution_estimates={
                k:{'1':dict(estimated_cpu_seconds=250.)} for k in ('raw','projected')})))
            for limit in (1200.,1300.):
                (out/'bank_resolution_plan.json').write_text(json.dumps(dict(estimated_cpu_seconds=1000.,
                    maximum_estimated_cpu_seconds=limit,selected=[dict(group=0,kind=k,rules=['K9']) for k in ('raw','projected')])))
                collected=dict(actions_checked=dict(raw=10,projected=10),failed_groups=dict(raw=[],projected=[]),
                    cpu_seconds=240.,single_record_risk_bank_precision_passed=True)
                with patch.object(launcher,'ROOT',root),patch.object(launcher,'launch') as launch,\
                    patch.object(launcher,'collect_remote',return_value=collected),\
                    contextlib.redirect_stdout(io.StringIO()):launcher.continue_coarse_resolution()
                plan=json.loads((out/'bank_coarse_plan.json').read_text())
                self.assertEqual(plan['cumulative_estimated_cpu_seconds'],1250.)
                self.assertEqual(launch.call_count,int(limit>=1250.))

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
