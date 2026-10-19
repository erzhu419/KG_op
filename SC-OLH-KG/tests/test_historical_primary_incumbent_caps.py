import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from performance.refine_historical_primary_incumbent_caps import candidate_units,inherited_primary_rows,merge_primary_caps,pilot_profiles,prepare,profile_summary,run,unit_rule_config
from performance.validate_historical_component_bank import RULES


class HistoricalPrimaryIncumbentCapsTest(unittest.TestCase):
    def test_requests_only_live63_45_budgets_and_actual_group_priority(self):
        def action(shortlist,budgets,incumbent=1):
            return dict(kind='guarded',shortlist=shortlist,budgets=budgets,incumbent_position=incumbent)
        actions=[action([71,45,10],[80,130,140]),action([63,45,10],[80,130,140]),
            action([63,45,10],[80,130,146]),action([63,45,10],[82,128,140]),
            action([63,45],[90,150]),action([63,45,10],[91,150,110],0)]
        row=dict(record='r',kind='raw',candidate_actions=[0,1,2,3,4,5],outer_grid_weight=.02,
            candidate_group_expected_positive_gap={'0':.9,'1':.03,'2':.8,'3':.7},weighted_unresolved_upper_gap=.001)
        units=candidate_units([row],actions)
        self.assertEqual(units[0]['budgets'],[[80,130],[82,128]])
        self.assertAlmostEqual(units[0]['weighted_priority_gap'],.0006)
        self.assertEqual(candidate_units([row],actions,['other']),[])

    def test_rules_fail_locally_and_preserve_computed_and_reused_costs(self):
        unit=dict(record='r',kind='raw',budgets=[[80,130]])
        profiles=[dict(rule=n,states=[[.7,.1,.1,.1]],diagnostics=dict(predictive_log_evidence_error=0.),
            source='pilot' if n=='E8' else 'computed',cpu_seconds=2.) for n in ('E8','E9','F9','L9')]
        result=profile_summary(unit,4,profiles,1e-4)
        self.assertTrue(result['precision_passed'])
        self.assertEqual(result['new_rule_evaluations'],3)
        self.assertEqual(result['reused_pilot_rule_evaluations'],1)
        self.assertEqual(result['new_rule_cpu_seconds'],6.)
        profiles[2]['states']=[[.70011,.09989,.1,.1]]
        self.assertFalse(profile_summary(unit,4,profiles,1e-4)['precision_passed'])
        profiles[2]['states']=[[.7,.1,.1,.1]]
        profiles[3]['diagnostics']['predictive_log_evidence_error']=.00011
        self.assertFalse(profile_summary(unit,4,profiles,1e-4)['precision_passed'])

    def test_pilot_profiles_map_requested_budget_order_and_reuse_only_exact_case(self):
        rules=[['E8',32,8,0],['E9',32,9,0],['F9',32,9,101],['L9',16,9,0]]
        spec=dict(pilot_plan='pilot.json',seed=202610147,rules=rules)
        pilot=dict(record='r',output='old',integrated_position=1,seed=spec['seed'],rules=rules)
        stored={'pilot.json':pilot,'old/pair_summary.json':dict(precision_passed=True,pair=[63,45],budgets=[[80,130],[82,128]])}
        for name,order,power,_ in rules:
            stored[f'old/parts/{name}.json']=dict(rule=name,states=[[.7,.1,.1,.1],[.6,.2,.1,.1]],cpu_seconds=4.,
                diagnostics=dict(order=order,power=power,integrated_mean_position=1,predictive_log_evidence_error=0.))
        unit=dict(record='r',kind='raw',budgets=[[82,128],[80,130]])
        with patch('performance.refine_historical_primary_incumbent_caps.read',side_effect=lambda p:stored[p]):
            profiles=pilot_profiles(spec,unit,4)
            np.testing.assert_array_equal(profiles[0]['states'],[[.6,.2,.1,.1],[.7,.1,.1,.1]])
            self.assertTrue(all(p['source']=='pilot' for p in profiles))
            self.assertIsNone(pilot_profiles(spec,unit,5))
            self.assertIsNone(pilot_profiles(spec,dict(unit,kind='projected'),4))

    def test_formal_prepare_retains_cost_admission_for_shard_run(self):
        with tempfile.TemporaryDirectory() as directory:
            spec=dict(output='new',source_bound_plan='source',source_phase_completion='completion',
                jobs=[dict(identity=4,shard=0,records=[])],estimated_cost_admitted=True,
                estimated_phase_cpu_seconds=1234.,maximum_estimated_cpu_seconds=3600)
            stored={'plan':spec,'source':dict(output='source_output',bank_plan='bank'),
                'source_output/summary.json':dict(numerical_containment_passed=True,records=[]),
                'bank':dict(output='bank_output')}
            with patch('performance.refine_historical_primary_incumbent_caps.ROOT',Path(directory)),\
                    patch('performance.refine_historical_primary_incumbent_caps.read',side_effect=lambda p:stored[p]),\
                    patch('performance.refine_historical_primary_incumbent_caps.compressed_read',return_value=dict(actions=[])):
                result=prepare('plan')
            self.assertTrue(result['estimated_cost_admitted'])
            self.assertEqual(result['estimated_phase_cpu_seconds'],1234.)

    def test_raw_only_prepare_omits_projected_from_collect_units_and_worklist(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            spec=dict(output='new',source_bound_plan='source',source_phase_completion='completion',
                selected_records=['r'],selected_kinds=['raw'],
                jobs=[dict(identity=4,shard=0,records=['r'],kinds=['raw'])],estimated_cost_admitted=True,
                estimated_phase_cpu_seconds=1000.,maximum_estimated_cpu_seconds=3600)
            raw=dict(record='r',kind='raw',candidate_actions=[0],outer_grid_weight=.02,
                candidate_group_expected_positive_gap={'0':.03},weighted_unresolved_upper_gap=.001)
            stored={'plan':spec,'source':dict(output='source_output',bank_plan='bank'),
                'source_output/summary.json':dict(numerical_containment_passed=True,records=[raw,dict(raw,kind='projected')]),
                'bank':dict(output='bank_output')}
            actions=[dict(kind='guarded',shortlist=[63,45,10],budgets=[129,129,98],incumbent_position=1)]
            with patch('performance.refine_historical_primary_incumbent_caps.ROOT',root),\
                    patch('performance.refine_historical_primary_incumbent_caps.read',side_effect=lambda p:stored[p]),\
                    patch('performance.refine_historical_primary_incumbent_caps.compressed_read',return_value=dict(actions=actions)):
                result=prepare('plan')
            self.assertEqual([(u['record'],u['kind']) for u in result['units']],[('r','raw')])
            self.assertEqual(json.loads((root/'new/worklists/0.json').read_text())['requests'],result['units'])

    def test_71_requests_keep_only_the_real71_45_budget_rows(self):
        actions=[dict(kind='guarded',shortlist=[63,45,10],budgets=[81,128,140],incumbent_position=1),
            dict(kind='guarded',shortlist=[71,45,10],budgets=[80,129,131],incumbent_position=1),
            dict(kind='guarded',shortlist=[71,45,10],budgets=[129,80,131],incumbent_position=1)]
        row=dict(record='r',kind='raw',candidate_actions=[0,1,2],outer_grid_weight=.02,
            candidate_group_expected_positive_gap={'0':.09,'1':.03},weighted_unresolved_upper_gap=.001)
        unit=candidate_units([row],actions,primary_policy=71)[0]
        self.assertEqual(unit['pair'],[71,45])
        self.assertEqual(unit['budgets'],[[80,129],[129,80]])
        self.assertAlmostEqual(unit['priority_gap'],.03)

    def test_71_cannot_reuse63_pilot_even_when_record_and_budgets_match(self):
        with patch('performance.refine_historical_primary_incumbent_caps.read') as reader:
            self.assertIsNone(pilot_profiles(dict(primary_policy=71),dict(record='r',kind='raw',budgets=[[129,129]]),4))
            reader.assert_not_called()

    def test_distinct_pairs_keep_separate_caps_for_the_same_record_and_budget(self):
        row=dict(record='r',kind='raw',identity=4,budgets=[[129,129]],precision_passed=True)
        caps={}
        merge_primary_caps(caps,[dict(row,pair=[63,45],upper=[[.6,.1,.2,.1]])])
        merge_primary_caps(caps,[dict(row,pair=[71,45],upper=[[.7,.1,.1,.1]])])
        self.assertEqual(len(caps),2)
        np.testing.assert_array_equal(caps['raw',4,'r',(63,45),(129,129)],[.6,.1,.2,.1])
        np.testing.assert_array_equal(caps['raw',4,'r',(71,45),(129,129)],[.7,.1,.1,.1])

    def test_71_run_uses_full071_field_and_raw_pair_coordinates(self):
        rules=[['E8',32,8,0],['E9',32,9,0],['F9',32,9,101],['L9',16,9,0]]
        unit=dict(record='r',kind='raw',pair=[71,45],budgets=[[129,129]])
        spec=dict(output='new',source_bound_plan='source',primary_policy=71,rules=rules,seed=202610147,
            numerical_padding=1e-4,source_components={'4':1},maximum_estimated_cpu_seconds=3600,
            pricing_admission=dict(cost_admitted=True,aggregate_estimated_phase_cpu_seconds=1))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            stored={'plan':spec,'new/preparation.json':dict(units=[unit]),'source':dict(bank_plan='bank'),
                'bank':dict(output='bank_output')}
            context=dict(query=63,prior_weights=[1.],protocol={},
                origins={'4':dict(risk=[dict(record='r',label=-1.,eta_label=0.,head_log_scores=[0.]*8)])})
            shared=dict(points=[63,71,45,10],prior_weights=[1.],fields={4:{}})
            diagnostics=dict(predictive_log_evidence_error=0.,components=1,all_positive_components_retained=True)
            def reader(path):
                return stored[path] if path in stored else json.loads((root/path).read_text())
            with patch('performance.refine_historical_primary_incumbent_caps.ROOT',root),\
                    patch('performance.refine_historical_primary_incumbent_caps.read',side_effect=reader),\
                    patch('performance.refine_historical_primary_incumbent_caps.compressed_read',side_effect=[context,shared]) as compressed,\
                    patch('performance.refine_historical_primary_incumbent_caps.field_from_snapshot',return_value=SimpleNamespace(means=np.zeros((1,2)))),\
                    patch('performance.refine_historical_primary_incumbent_caps.condition_risk',return_value=0.),\
                    patch('performance.refine_historical_primary_incumbent_caps.signal.signal'),\
                    patch('performance.refine_historical_primary_incumbent_caps.signal.setitimer'),\
                    patch('performance.refine_historical_primary_incumbent_caps.certificate_pair_states',return_value=(np.array([[.7,.1,.1,.1]]),diagnostics)) as integrate:
                run('plan',4,record='r',kind='raw')
            self.assertEqual(compressed.call_args_list[1].args[0].name,'071.json.gz')
            self.assertEqual(integrate.call_count,4)
            for call in integrate.call_args_list:
                self.assertEqual(call.args[2],[1,2])
                self.assertEqual(call.kwargs['query'],0)
                self.assertTrue(call.kwargs['raw'])
                self.assertEqual(call.kwargs['integrated_position'],1)
            row=json.loads((root/'new/units/4/r/raw.json').read_text())
            self.assertEqual(row['pair'],[71,45])
            self.assertTrue(all(json.loads(p.read_text())['pair']==[71,45] for p in (root/'new/parts/4/r').glob('*.json')))

    def test_historical_guard_backend_preserves_budget_order_covariance_coordinates_and_rule(self):
        rules=[[name,*RULES[name]] for name in ('E8','F8','L8','E9')]
        unit=dict(record='r',kind='raw',pair=[71,45],budgets=[[129,129],[80,129]])
        spec=dict(output='recovery',source_bound_plan='source',primary_policy=71,integration_backend='historical_guard',
            rules=rules,seed=202610147,numerical_padding=1e-4,source_components={'4':3},
            maximum_estimated_cpu_seconds=3600,pricing_admission=dict(cost_admitted=True,aggregate_estimated_phase_cpu_seconds=1))
        statistics=np.array([[.6,.02,.03,.04],[.7,.01,.02,.05]])
        means=np.zeros((3,8));means[1,0]=1.
        before=SimpleNamespace(means=means)
        protocol=dict(chance_alpha=.1,familywise_delta=.05)
        context=dict(query=63,prior_weights=[1.],protocol=protocol,
            origins={'4':dict(risk=[dict(record='r',label=-1.,eta_label=-8.,head_log_scores=[0.]*8)])},
            actions=[dict(kind='guarded',primary=63,shortlist=[63,45,10],budgets=[129,129,90],incumbent_position=1),
                dict(kind='guarded',primary=71,shortlist=[71,45,10],budgets=[80,129,131],incumbent_position=1),
                dict(kind='guarded',primary=71,shortlist=[71,45,10],budgets=[129,129,82],incumbent_position=1)])
        shared=dict(points=[63,71,45,10],prior_weights=[1.],fields={4:{}})
        diagnostics=dict(predictive_log_evidence_error=0.,components=3,all_positive_components_retained=True)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            stored={'plan':spec,'recovery/preparation.json':dict(units=[unit]),'source':dict(bank_plan='bank'),
                'bank':dict(output='bank_output')}
            def reader(path):
                return stored[path] if path in stored else json.loads((root/path).read_text())
            with patch('performance.refine_historical_primary_incumbent_caps.ROOT',root),\
                    patch('performance.refine_historical_primary_incumbent_caps.read',side_effect=reader),\
                    patch('performance.refine_historical_primary_incumbent_caps.compressed_read',side_effect=[context,shared]),\
                    patch('performance.refine_historical_primary_incumbent_caps.field_from_snapshot',return_value=before),\
                    patch('performance.refine_historical_primary_incumbent_caps.condition_risk',return_value=0.),\
                    patch('performance.refine_historical_primary_incumbent_caps.signal.signal'),\
                    patch('performance.refine_historical_primary_incumbent_caps.signal.setitimer'),\
                    patch('performance.refine_historical_primary_incumbent_caps.certificate_pair_states') as pair_integrate,\
                    patch('performance.refine_historical_primary_incumbent_caps.component_risk_allocations',return_value=(statistics,np.array([340.,340.]),diagnostics)) as full_integrate:
                run('plan',4,record='r',kind='raw')
            pair_integrate.assert_not_called()
            self.assertEqual(full_integrate.call_count,4)
            for call,rule in zip(full_integrate.call_args_list,rules):
                _,order,power,offset=rule
                self.assertIs(call.args[0],before)
                self.assertEqual([a['budgets'] for a in call.args[2]],[[129,129,82],[80,129,131]])
                self.assertTrue(all(a['shortlist']==[1,2,3] and a['primary']==1 and a['incumbent_position']==1 for a in call.args[2]))
                self.assertIs(call.args[5],protocol)
                self.assertEqual(call.args[5]['familywise_delta']/len(call.args[2][0]['shortlist']),.05/3)
                self.assertEqual((call.kwargs['order'],call.kwargs['power']),(order,power))
                np.testing.assert_array_equal(call.kwargs['seed'],[202610147+offset,202714876+offset,202610147+offset])
                self.assertEqual((call.kwargs['query'],call.kwargs['replicates'],call.kwargs['eta_label'],call.kwargs['integrated_position']),(0,4,-8.,1))
                self.assertTrue(call.kwargs['raw'])
            row=json.loads((root/'recovery/units/4/r/raw.json').read_text())
            self.assertEqual(row['integration_backend'],'historical_guard')
            np.testing.assert_allclose(row['states'],[[.34,.02,.6,.04],[.24,.01,.7,.05]],rtol=0.,atol=1e-15)
            saved=json.loads((root/'recovery/parts/4/r/raw_E9.json').read_text())
            np.testing.assert_array_equal(saved['statistics'],statistics)
            self.assertEqual(saved['guarded_budgets'],[[129,129,82],[80,129,131]])

    def test_inherits_accepted_batch_rows_across_two_phases_without_relabeling_costs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('batch1','batch2'):
                (root/name).mkdir();(root/name/'preparation.json').touch()
            stored={'screen2':dict(certificate_cap_plan='plan2'),
                'plan2':dict(output='batch2',source_bound_plan='screen1'),
                'screen1':dict(certificate_cap_plan='plan1'),
                'plan1':dict(output='batch1',source_bound_plan='legacy_screen'),
                'legacy_screen':dict(certificate_cap_plan='legacy_plan'),'legacy_plan':dict(output='legacy')}
            for name in ('batch1','batch2'):
                stored[str(root/name/'preparation.json')]=dict(units=[dict(record=name,kind='raw')])
                for h in (4,5):
                    stored[f'{name}/units/{h}/{name}/raw.json']=dict(complete=True,record=name,kind='raw',
                        identity=h,precision_passed=(name=='batch1' or h==4),new_rule_evaluations=4,new_rule_cpu_seconds=20.)
            with patch('performance.refine_historical_primary_incumbent_caps.ROOT',root),\
                    patch('performance.refine_historical_primary_incumbent_caps.read',side_effect=lambda p:stored[p]):
                rows=inherited_primary_rows(dict(source_bound_plan='screen2'))
            self.assertEqual([(r['record'],r['identity']) for r in rows],[('batch2',4),('batch1',4),('batch1',5)])
            self.assertTrue(all(r['new_rule_cpu_seconds']==20. for r in rows))

    def test_one_unit_promotion_reuses_e9_adds_three_rules_and_retains_failed_evidence(self):
        old_rules=[['E8',32,8,0],['E9',32,9,0],['F9',32,9,101],['L9',16,9,0]]
        new_rules=[['E9',32,9,0],['E10',32,10,0],['F10',32,10,101],['L10',16,10,0]]
        unit=dict(record='r',kind='projected',budgets=[[129,129]])
        spec=dict(output='new',source_bound_plan='source',pilot_plan='pilot',rules=old_rules,seed=202610147,
            numerical_padding=1e-4,source_components={'4':1},maximum_estimated_cpu_seconds=3600,
            pricing_admission=dict(cost_admitted=True,aggregate_estimated_phase_cpu_seconds=1),
            unit_rule_refinements=[dict(record='r',identity=4,kind='projected',rules=new_rules,selected_rule='E10')])
        self.assertEqual(unit_rule_config(spec,unit,5),(old_rules,'E9'))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);parts=root/'new/parts/4/r';parts.mkdir(parents=True)
            diagnostics=dict(predictive_log_evidence_error=0.,components=1,all_positive_components_retained=True)
            for rule,order,power,offset in old_rules:
                profile=dict(record='r',kind='projected',identity=4,pair=[63,45],budgets=unit['budgets'],
                    rule=rule,order=order,power=power,seed=spec['seed']+offset,integrated_mean_position=1,
                    states=[[.7,.1,.1,.1]],diagnostics=diagnostics,source='computed',cpu_seconds=2.)
                (parts/f'projected_{rule}.json').write_text(json.dumps(profile))
            old=dict(complete=True,precision_passed=False,selected_rule='E9',rules=old_rules,
                maximum_rule_difference=.00011723529898655105,original_evidence='kept')
            summary=root/'new/units/4/r/projected.json';summary.parent.mkdir(parents=True);summary.write_text(json.dumps(old))
            stored={'plan':spec,'new/preparation.json':dict(units=[unit]),'source':dict(bank_plan='bank'),
                'bank':dict(output='bank_output'),'pilot':dict(record='other')}
            context=dict(query=63,prior_weights=[1.],origins={'4':dict(risk=[dict(record='r',label=-1.,eta_label=0.,head_log_scores=[0.]*8)])})
            shared=dict(points=[63,45,10],prior_weights=[1.],fields={4:{}})
            def reader(path):
                return stored[path] if path in stored else json.loads((root/path).read_text())
            with patch('performance.refine_historical_primary_incumbent_caps.ROOT',root),\
                    patch('performance.refine_historical_primary_incumbent_caps.read',side_effect=reader),\
                    patch('performance.refine_historical_primary_incumbent_caps.compressed_read',side_effect=[context,shared]),\
                    patch('performance.refine_historical_primary_incumbent_caps.field_from_snapshot',return_value=SimpleNamespace(means=np.zeros((1,2)))),\
                    patch('performance.refine_historical_primary_incumbent_caps.condition_risk',return_value=0.),\
                    patch('performance.refine_historical_primary_incumbent_caps.signal.signal'),\
                    patch('performance.refine_historical_primary_incumbent_caps.signal.setitimer'),\
                    patch('performance.refine_historical_primary_incumbent_caps.certificate_pair_states',return_value=(np.array([[.7,.1,.1,.1]]),diagnostics)) as integrate:
                context['protocol']={}
                result=run('plan',4,record='r',kind='projected')
            self.assertEqual(integrate.call_count,3)
            self.assertEqual(result['new_rule_evaluations'],3)
            self.assertEqual(result['resumed_rule_profiles'],1)
            saved=json.loads(summary.read_text())
            self.assertEqual(saved['rules'],new_rules)
            self.assertEqual(saved['selected_rule'],'E10')
            self.assertEqual(saved['new_rule_evaluations'],7)
            self.assertEqual(json.loads((root/'new/precision_holds/4/r/projected_E9.json').read_text()),old)
            self.assertEqual(len(list(parts.glob('*.json'))),7)


if __name__=='__main__':
    unittest.main()
