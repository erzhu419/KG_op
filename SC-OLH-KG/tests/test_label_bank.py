import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from core.budgeted_guarded_kg import BudgetedTerminalActions
from performance.validate_historical_component_bank import action_groups
from performance.validate_historical_component_transfer import part_path
from performance.validate_historical_label_bank import anchor_tables, collect, make_plan, prepare, run, validated_selections, independent_check, anchor_part_path
from performance.probe_historical_label_gh import select_pilot, check as gh_check, prepare_full, collect as gh_collect


def row(value,actions):
    return dict(statistics=np.full((actions,4),value).tolist(),costs=[200.]*actions,
        diagnostics=dict(predictive_log_evidence_error=0.),cpu_seconds=3.)


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data))


class LabelBankTests(unittest.TestCase):
    def test_full_gh_prices_both_roles_and_excludes_cached_integrals(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'out';base=root/'base'
            entry=dict(record='near',group=0,kind='raw',anchor_rule='G7')
            write(root/'parent.json',dict(output='base'));plan=root/'plan.json'
            write(plan,dict(output='out',parent_plan=str(root/'parent.json'),shards_per_identity_kind=8,
                maximum_estimated_cpu_seconds_per_stage=20.))
            write(out/'contexts.json',dict(near=dict(utility_identities=[4,5])))
            write(out/'bank_layout.json',dict(groups=[dict(kind='guarded')]))
            write(out/'independent_groups.json',[dict(entry,target_precision_admitted=False)])
            write(out/'fine_plan.json',dict(selected=[entry]))
            write(out/'gh_pilot_summary.json',dict(pilot_passed=True,groups=[],
                measured_maximum_G5_E5_ratios={f'raw_guarded_{h}_{role}':2. for h in (4,5) for role in ('anchor','target')}))
            for h in (4,5):
                write(base/'parts'/f'000_{h}_raw.json',dict(rows=dict(E5=row(.1,3))))
                write(part_path(out,entry,h),dict(rows=dict(E5=row(.2,3))))
            with patch('performance.probe_historical_label_gh.ROOT',root),contextlib.redirect_stdout(io.StringIO()):
                held=prepare_full(str(plan));self.assertFalse(held['estimated_cost_admitted'])
                self.assertAlmostEqual(held['estimated_cpu_seconds'],31.2)
                write(anchor_part_path(out,entry,4),dict(rows=dict(G5=row(.1,3))))
                write(part_path(out,entry,4),dict(rows=dict(E5=row(.2,3),G5=row(.2,3))))
                admitted=prepare_full(str(plan));self.assertTrue(admitted['estimated_cost_admitted'])
                self.assertAlmostEqual(admitted['estimated_cpu_seconds'],16.8)

    def test_negative_control_uses_valid_direct_independent_reference_without_clipping(self):
        rng=np.random.default_rng(81)
        bundle=dict(prior_weights=[.125]*8,utility_identities=[4,5],actions=[dict(kind='guarded')]*3,
            head_log_scores=rng.normal(size=8).tolist(),opposite=[dict(raw=rng.uniform(size=(8,3,4)).tolist(),
                head_log_scores=rng.normal(size=8).tolist()) for _ in range(4)])
        anchor={h:{r:row(.2,3) for r in ('E5','E7','F7','G5','G7')} for h in (4,5)}
        target={h:{r:row(.2,3) for r in ('E5','E7','F7','G5')} for h in (4,5)}
        for rows in (anchor,target):
            for value in rows[5].values():value['statistics'][1][0]=1e-10
        anchor[5]['F7']['statistics'][1][0]=2.4e-9
        actual=anchor_tables(bundle,anchor,target,'E5','G7')
        result=independent_check(bundle,anchor,target,'G7','raw',1e-4)
        self.assertTrue(result['passed']);self.assertEqual(result['reference']['method'],'direct_F7')
        self.assertFalse(result['reference']['controlled_valid_unclipped_ranges'])
        self.assertLess(result['reference']['controlled_minimum_statistic'],-1e-9)
        self.assertEqual(actual[0][5,1,0],1e-10)
        self.assertTrue(gh_check(bundle,anchor,target,'G7',actual,'raw',1e-4)['passed'])
        target[5]['F7']['statistics'][2][0]+=.1
        self.assertFalse(independent_check(bundle,anchor,target,'G7','raw',1e-4)['passed'])
        self.assertFalse(gh_check(bundle,anchor,target,'G7',actual,'raw',1e-4)['passed'])

    def test_gh_pilot_includes_each_invalid_control_reference_substitution(self):
        groups=[dict(record='near',group=0,kind='projected',target_precision_admitted=True,
            independent_check=dict(reference=dict(method='direct_F7')))]
        self.assertEqual(select_pilot(groups,[dict(kind='guarded')],[]),{('near',0,'projected')})

    def test_gh_pilot_retains_history_and_observed_worst_of_each_action_kind(self):
        layout=[dict(kind='direct' if g in (2,3) else 'guarded') for g in range(8)]
        groups=[dict(record='near',group=g,kind='raw' if g<4 else 'projected',
            target_precision_admitted=False,coarse_fine_maximum_difference=float(g),
            independent_check=dict(differences={'E5':float(6-g),'E7':0.})) for g in range(6)]
        selected=select_pilot(groups,layout,[dict(record='near',group=7,kind='raw')])
        self.assertEqual(selected,{('near',g,'raw' if g<4 or g==7 else 'projected') for g in (0,1,2,3,4,5,7)})

    def test_gh_residual_disagreement_in_one_identity_budget_is_not_admitted(self):
        rng=np.random.default_rng(77)
        bundle=dict(prior_weights=[.125]*8,utility_identities=[4,5],actions=[dict(kind='guarded')]*3,
            head_log_scores=rng.normal(size=8).tolist(),opposite=[dict(raw=rng.uniform(size=(8,3,4)).tolist(),
                head_log_scores=rng.normal(size=8).tolist()) for _ in range(4)])
        anchor={h:{r:row(.2,3) for r in ('E5','E7','F7','G5','G7')} for h in (4,5)}
        target={h:{r:row(.3,3) for r in ('E5','E7','F7','G5')} for h in (4,5)}
        actual=anchor_tables(bundle,anchor,target,'E5','G7')
        self.assertTrue(gh_check(bundle,anchor,target,'G7',actual,'raw',1e-4)['passed'])
        target[5]['G5']['statistics'][1][0]+=.1
        self.assertFalse(gh_check(bundle,anchor,target,'G7',actual,'raw',1e-4)['passed'])

    def test_independent_rule_detects_one_identity_budget_and_normalization_failure(self):
        rng=np.random.default_rng(73)
        bundle=dict(prior_weights=[.125]*8,utility_identities=[4,5],actions=[dict(kind='guarded')]*3,
            head_log_scores=rng.normal(size=8).tolist(),opposite=[dict(raw=rng.uniform(size=(8,3,4)).tolist(),
                head_log_scores=rng.normal(size=8).tolist()) for _ in range(4)])
        anchor={h:{r:row(.2,3) for r in ('E5','E7','F7','G7')} for h in (4,5)}
        target={h:{r:row(.3,3) for r in ('E5','E7','F7')} for h in (4,5)}
        self.assertTrue(independent_check(bundle,anchor,target,'G7','raw',1e-4)['passed'])
        target[5]['F7']['statistics'][2][0]+=.1
        self.assertFalse(independent_check(bundle,anchor,target,'G7','raw',1e-4)['passed'])
        target[5]['F7']=row(.3,3);target[5]['F7']['diagnostics']['predictive_log_evidence_error']=.01
        self.assertFalse(independent_check(bundle,anchor,target,'G7','raw',1e-4)['passed'])

    def test_first_preparation_and_worker_create_output_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);base=root/'base';out=root/'out'
            group=dict(group=0,kind='guarded',action_indices=[0,1,2])
            original=dict(groups=[group],actions=[{},{},{}],utility_identities=[4,5])
            write(base/'context.json',original)
            write(base/'bank_tail_resolution_summary.json',dict(groups=[dict(
                selected_rules=dict(raw='G7',projected='G7'),checks={k:dict(passed=True) for k in ('raw','projected')})]))
            for h in (4,5):
                for kind in ('raw','projected'):write(base/'parts'/f'000_{h}_{kind}.json',dict(
                    rows={n:row(.1,3) for n in ('E5','E7','G7')}))
            write(root/'parent.json',dict(output='base'));write(root/'validated.json',dict(complete=True,groups=[]))
            plan=root/'plan.json';write(plan,dict(output='out',parent_plan=str(root/'parent.json'),cache_outputs=[],
                validated_groups=str(root/'validated.json'),validated_direct_groups=str(root/'validated.json'),shards_per_identity_kind=8,
                maximum_estimated_cpu_seconds_per_stage=3600.,maximum_job_cpu_seconds=600.))
            context=dict(near=dict(utility_identities=[4,5]));shared=dict(fields={h:dict(means=[0,1]) for h in (4,5)})
            with patch('performance.validate_historical_label_bank.ROOT',root),contextlib.redirect_stdout(io.StringIO()):
                with patch('performance.validate_historical_label_bank.prepare_contexts',return_value=(context,shared,original)):
                    self.assertTrue(prepare(str(plan))['complete'])
                with patch('performance.validate_historical_label_bank.compressed_read',return_value={}),patch(
                    'performance.validate_historical_label_bank.record_bundle',return_value={}),patch(
                    'performance.validate_historical_label_bank.evaluate_group_rules',return_value=iter([('E5',row(.1,3))])):
                    self.assertTrue(run(4,'raw',0,'coarse',str(plan))['complete'])
                write(out/'independent_summary.json',dict(complete=True))
                write(out/'gh_pilot_plan.json',json.loads((out/'coarse_plan.json').read_text()))
                labels=[]
                def evaluate(bundle,identity,kind,rules):
                    labels.append(bundle['label']);return iter([('G5',row(.2,3))])
                with patch('performance.validate_historical_label_bank.compressed_read',return_value=dict(label='anchor')),patch(
                    'performance.validate_historical_label_bank.record_bundle',return_value=dict(label='target')),patch(
                    'performance.validate_historical_label_bank.evaluate_group_rules',side_effect=evaluate) as evaluator:
                    result=run(4,'raw',0,'gh_pilot',str(plan))
                    self.assertTrue(result['complete']);self.assertEqual(result['new_integral_cpu_seconds'],6.)
                    self.assertEqual(labels,['anchor','target'])
                    evaluator.reset_mock()
                    self.assertEqual(run(4,'raw',0,'gh_pilot',str(plan))['new_integral_cpu_seconds'],0.)
                    evaluator.assert_not_called()
            self.assertTrue((out/'tasks/coarse/4_raw_0.json').exists())
            self.assertIn('E5',json.loads((out/'parts/near/000_4_raw.json').read_text())['rows'])
            self.assertIn('G5',json.loads(anchor_part_path(out,dict(group=0,kind='raw'),4).read_text())['rows'])

    def test_earlier_direct_reference_is_reused_as_actual_selected_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'out';entry=dict(record='near',group=52,kind='raw',
                selected_rule='E9',checks=dict(passed=True))
            write(root/'direct.json',dict(complete=True,groups=[entry]));write(root/'controlled.json',dict(complete=True,groups=[]))
            for h in (4,5):write(part_path(out,entry,h),dict(rows=dict(E9=row(.3,3))))
            context=dict(near=dict(prior_weights=[.125]*8,utility_identities=[4,5],groups={'52':dict(action_indices=[0,2,4])}))
            known=validated_selections(dict(validated_groups=str(root/'controlled.json'),
                validated_direct_groups=str(root/'direct.json')),context,out)
            self.assertEqual(len(known),1);self.assertEqual(known[0]['selected_method'],'direct_E9')
            np.testing.assert_array_equal(np.asarray(known[0]['selected_statistics'])[4:6],.3)

    def test_direct_actions_keep_all_identity_slots_and_unnormalized_control(self):
        bundle=dict(prior_weights=[.125]*8,actions=[dict(kind='direct',budgets=[b]) for b in (128,359)],utility_identities=[4,5])
        anchor={h:dict(E5=row(.1,2),G7=row(.2,2),E8=row(.3,2),E9=row(.4,2)) for h in (4,5)}
        target={h:dict(E5=row(.15,2)) for h in (4,5)}
        anchor[4]['E9']['diagnostics']['predictive_log_evidence_error']=np.log(2.)
        for rule in ('G7','E8','E9'):
            s,c=anchor_tables(bundle,anchor,target,'E5',rule)
            self.assertEqual(s.shape,(8,2,4));self.assertEqual(c.shape,(8,2))
            expected=(2*.4+.15-.1) if rule=='E9' else {'G7':.25,'E8':.35}[rule]
            np.testing.assert_allclose(s[4],expected)
            np.testing.assert_array_equal(c[4:6],[[128.,359.],[128.,359.]])
            np.testing.assert_array_equal(s[:4],0.)

    def test_pricing_excludes_caches_and_does_not_relax_stage_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'out';base=root/'base'
            entry=dict(record='near',group=0,kind='raw',anchor_rule='E9')
            for h in (4,5):
                write(base/'parts'/f'000_{h}_raw.json',dict(rows=dict(E5=dict(cpu_seconds=5.))))
                write(part_path(out,entry,h),dict(rows={'E5':row(.1,3)} if h==4 else {}))
            spec=dict(shards_per_identity_kind=8,maximum_estimated_cpu_seconds_per_stage=8.)
            context=dict(near=dict(utility_identities=[4,5]))
            work=make_plan([entry],context,out,base,'coarse',spec)
            self.assertAlmostEqual(work['estimated_cpu_seconds'],8.4)
            self.assertFalse(work['estimated_cost_admitted'])
            spec['maximum_estimated_cpu_seconds_per_stage']=8.5
            self.assertTrue(make_plan([entry],context,out,base,'coarse',spec)['estimated_cost_admitted'])

    def test_bank_assembly_preserves_budgets_native_reference_and_precision_scope(self):
        terminal=BudgetedTerminalActions([[0,2,3],[1,2,3]],[1,1],dict(comparison_budget=8))
        actions=[dict(kind=k,shortlist=s.tolist(),incumbent_position=int(i),budgets=b)
            for k,table,inc,b in terminal.allocations(359) for s,i in zip(table,inc)]
        groups=action_groups(actions);rng=np.random.default_rng(9)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'out';base=root/'base';contexts={};selected=[];expected={}
            context=dict(utility_identities=[4,5],prior_weights=[.125]*8,head_log_scores=rng.normal(size=8).tolist(),groups={})
            for group in groups:
                gid=group['group'];a=len(group['action_indices']);chosen=('G7','E8','E9','G7')[gid]
                high={'G7':.2,'E8':.3,'E9':.4}[chosen]+.05
                s=np.zeros((8,a,4));s[4:6]=high;c=np.zeros((8,a));c[4:6]=200.
                if group['kind']=='direct':c[4:6]=[actions[i]['budgets'][0] for i in group['action_indices']]
                expected[gid]=(s.copy(),c.copy())
                context['groups'][str(gid)]=dict(action_indices=group['action_indices'],
                    baseline={k:dict(statistics=s.tolist(),costs=c.tolist()) for k in ('raw','projected')},
                    opposite=[dict(head_log_scores=rng.normal(size=8).tolist(),
                        **{k:rng.uniform(size=(8,a,4)).tolist() for k in ('raw','projected')}) for _ in range(2)])
                for kind in ('raw','projected'):
                    if kind=='projected' and group['kind']=='direct':continue
                    entry=dict(record='near',group=gid,kind=kind,anchor_rule=chosen);selected.append(entry)
                    for h in (4,5):
                        write(base/'parts'/f'{gid:03d}_{h}_{kind}.json',dict(rows={n:row(v,a) for n,v in
                            [('E5',.1),('E7',.1),('F7',.1),('G5',.1),('G7',.2),('E8',.3),('E9',.4)]}))
                        write(part_path(out,entry,h),dict(rows={n:row(.15,a) for n in ('E5','E7','F7','G5')}))
                        write(anchor_part_path(out,entry,h),dict(rows=dict(G5=row(.1,a))))
            contexts['near']=context
            write(out/'contexts.json',contexts);write(out/'bank_layout.json',dict(actions=actions,groups=groups))
            known=dict(record='near',group=1,kind='raw',selected_method='control32',hybrid_direct_admitted=True,
                selected_statistics=(expected[1][0]+np.eye(8,1)[:,None,:]*1e-5).tolist(),selected_costs=expected[1][1].tolist())
            write(out/'validated_groups.json',[known])
            write(out/'gh_pilot_plan.json',dict(selected=[]))
            for stage in ('coarse','fine','independent','gh_full'):
                work=[r for r in selected if (r['group'],r['kind'])!=(1,'raw')] if stage in ('independent','gh_full') else selected
                write(out/(stage+'_plan.json'),dict(selected=work,estimated_cpu_seconds=10.))
                for h in (4,5):
                    for kind in ('raw','projected'):
                        for shard in range(2):write(out/'tasks'/stage/f'{h}_{kind}_{shard}.json',dict(
                            complete=True,cpu_seconds=1.,new_integral_cpu_seconds=.5,peak_rss_mb=100.))
            write(root/'parent.json',dict(output='base'))
            plan=root/'plan.json';write(plan,dict(output='out',parent_plan=str(root/'parent.json'),
                shards_per_identity_kind=2,maximum_action_difference=1e-4))
            with patch('performance.validate_historical_label_bank.ROOT',root),patch(
                'performance.validate_historical_label_bank.validated_selections',return_value=[known]),contextlib.redirect_stdout(io.StringIO()):
                collect('coarse',str(plan));result=collect('fine',str(plan))
                self.assertEqual(result['coarse_fine_actions_passed'],dict(raw=10,projected=10))
                self.assertEqual(result['target_precision_actions_admitted'],dict(raw=3,projected=4))
                self.assertFalse(result['full_predictive_integration_admitted'])
                bank=json.loads((out/'fine_risk_bank.json').read_text())['near']
                for group in groups:
                    gid=group['group'];ix=group['action_indices']
                    for kind in ('raw','projected'):
                        want=known['selected_statistics'] if gid==1 and kind=='raw' else expected[gid][0]
                        np.testing.assert_allclose(np.asarray(bank['statistics'][kind])[:,ix],want,atol=1e-15)
                independent=collect('independent',str(plan))
                self.assertEqual(independent['independent_actions_passed'],dict(raw=10,projected=10))
                self.assertEqual(independent['target_precision_actions_admitted'],dict(raw=3,projected=4))
                self.assertFalse(independent['full_predictive_integration_admitted'])
                with patch('performance.probe_historical_label_gh.ROOT',root):
                    gh=gh_collect(str(plan),stage='gh_full')
                    self.assertTrue(gh['all_label_comparisons_passed'])
                    self.assertEqual(gh['gh_comparison_actions_passed'],dict(raw=10,projected=10))
                    self.assertEqual(gh['target_precision_actions_admitted'],dict(raw=3,projected=4))
                    self.assertFalse(gh['full_predictive_integration_admitted'])
                    entry=selected[0];path=part_path(out,entry,5);part=json.loads(path.read_text())
                    part['rows']['G5']['statistics'][1][0]+=.1;write(path,part)
                    failed_gh=gh_collect(str(plan),stage='gh_full')
                    self.assertFalse(failed_gh['all_label_comparisons_passed'])
                    self.assertEqual(failed_gh['gh_comparison_actions_passed']['raw'],7)
                failed_entry=selected[0];path=part_path(out,failed_entry,5);part=json.loads(path.read_text())
                part['rows']['F7']['statistics'][1][0]+=.1;write(path,part)
                failed=collect('independent',str(plan))
                self.assertEqual(failed['independent_failed_groups']['raw'],[0])
                self.assertEqual(failed['target_precision_actions_admitted'],dict(raw=3,projected=4))
                entry=selected[0];path=part_path(out,entry,5);part=json.loads(path.read_text())
                part['rows']['E7']['statistics'][1][0]+=.1;write(path,part)
                result=collect('fine',str(plan))
                self.assertEqual(result['coarse_fine_actions_passed']['raw'],7)
                self.assertEqual(result['coarse_fine_failed_groups']['raw'],[0])
                self.assertEqual(result['target_precision_actions_admitted']['raw'],3)


if __name__=='__main__':unittest.main()
