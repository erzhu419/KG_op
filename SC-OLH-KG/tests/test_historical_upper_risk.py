import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import performance.extend_historical_upper_risk as upper
import performance.refine_historical_achievable_precision as precision
import performance.refine_historical_achievable_policies as policies
from performance.refine_historical_achievable_precision import grid,head_checks,all_head_checks
from performance.refine_historical_remaining_query_failures import positive_risk_tail,positive_cdf_interval
from core.conditional_terminal_kg import ConditionalTerminalActions
from core.raw_terminal_head_grid import evaluate_head_records
from representation.joint_response_mixture import JointResponseMixture
from performance.run_legacy_joint_population_matched_baselines import paired_fields


class UpperRiskTests(unittest.TestCase):
    def test_body_extension_reuses_admitted_labels_and_keeps_old_tail_rules(self):
        body={'0':{'2':dict(positions=[10,11],labels=[-.4,-.2],weights=[.45,.45],maximum_inverse_cdf_error=1e-12),
            '3':dict(positions=[12,13,14],labels=[-.5,-.3,-.1],weights=[.3,.3,.3],maximum_inverse_cdf_error=1e-12)}}
        tails={'0':{'3':dict(positions=[1,2,3],weights=[.03,.03,.04])}}
        case=dict(name='source_n1',state='source',head_counts=[1,9])
        spec=dict(output='body',reuse_precision_plan='old',cases=[case],query=63,reused_risk_count=4,
            risk_tail_orders=[2,3],risk_rule_input='screen',retained_tail_rule_input='tails')
        cached=dict(fixed_action_only_admitted=True,selected_body_orders=[2,3],query=63,prior_weights=[1.],body_rules=body)
        old_heads={(0,0):[dict(position=0)],(1,0):[dict(position=p) for p in range(4)]}
        bank=dict(origins=[0],prior_weights=[1.]);before=copy.deepcopy(old_heads)
        payload={'plan':spec,'old':dict(output='old',cases=[case]),'old/outcome_summary.json':
            dict(cases=[dict(terminal_diagnostics_passed=True,gates=dict(baseline=True))]),
            'old/registered_jobs.json':dict(jobs=[]),'screen':cached,'tails':tails}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(upper,'ROOT',root),patch.object(upper,'read',side_effect=payload.__getitem__), \
                    patch.object(upper,'assemble_case',return_value=(old_heads,bank,[])),patch.object(upper,'restore') as restore:
                upper.prepare('plan')
                new_rules=json.loads((root/'body/risk_rules.json').read_text())
                rules=upper.head_rules(spec,new_rules,[0],1,3,2)
            restore.assert_not_called();self.assertEqual(old_heads,before)
            prep=json.loads((root/'body/preparation.json').read_text())
            self.assertEqual(prep['new_risk_states'],5);self.assertEqual(prep['reused_head_states'],5)
            self.assertEqual([j['position'] for j in prep['jobs']],list(range(4,9)))
            self.assertEqual(rules[1,0]['positions'],[4,5,1,2,3])
            self.assertAlmostEqual(sum(rules[1,0]['weights']),1.)
            self.assertEqual(rules[0,0]['positions'],[0])

    def test_body_rule_uses_saved_field_and_joins_tail_without_losing_probability_mass(self):
        field=object()
        class CDF:
            def inverse(self,q):return 2*q-1
            def adaptive(self,y):return (y+1)/2,0.
        with patch('performance.refine_historical_remaining_query_failures.RawPredictiveCDF',return_value=CDF()) as cdf:
            body=positive_cdf_interval(field,0,dict(maximum_inverse_cdf_error=1e-8),order=16,position=176,interval=(0.,.9))
        cdf.assert_called_once_with(field,0)
        self.assertAlmostEqual(sum(body['weights']),.9)
        self.assertGreater(min(body['weights']),0.)
        self.assertAlmostEqual(np.dot(body['weights'],body['labels']),-.09)
        state=dict(prior_weights=[.6,.4],baseline={'1':dict(reward=.312)})
        cached={(o,0):dict(values=np.full(512,f).tolist()) for o,f in [(0,.8),(1,.5)]}
        tail_rules={str(o):{'32':dict(weights=np.full(32,.1/32).tolist())} for o in [0,1]}
        body_rules={str(o):{'16':body} for o in [0,1]}
        tail_values={str(o):{'32':np.full(32,g).tolist()} for o,g in [(0,.4),(1,.6)]}
        body_values={str(o):{'16':np.full(16,g).tolist()} for o,g in [(0,.4),(1,.6)]}
        for row in upper.fixed_body_rows(state,cached,tail_rules,tail_values,body_rules,body_values,[16]):
            self.assertAlmostEqual(row['factorized_fixed_action_tower_residual'],0.,places=14)

    def test_fixed_axis_diagnostic_reuses_cached_body_and_only_evaluates_new_tail(self):
        spec=dict(output='upper',cases=[dict(state='source')],query=63,predictive_seed=202610092,
            risk_tail_interval=[.9,1.],risk_tail_orders=[16,32],accepted_baseline_inputs={'source':'base'})
        action=dict(kind='direct',primary=63,budgets=[359]);state=dict(query=63,predictive_seed=202610092,
            prior_weights=[0.,0.,0.,0.,.6,.4],baseline={'1':dict(action=action,reward=.312)},protocol={},
            fields={str(o):[None,dict(value=value)] for o,value in [(4,.4),(5,.6)]})
        rules={str(o):{str(t):dict(weights=np.full(t,.1/t).tolist(),labels=np.full(t,g).tolist())
            for t in [16,32]} for o,g in [(4,.4),(5,.6)]}
        previous=dict(output='old');payload={'plan':spec,
            'performance/manifests/historical_observed_prediction_v1.json':previous,'old/state.json':state,
            'upper/risk_rules.json':rules,'upper/source/bank.json':dict(prior_weights=state['prior_weights'],origins=[4,5]),
            'base/source/n1/baseline/terminal_rules.json':dict(rows=[dict(best_action=action,best_reward=.312)])}
        cached={}
        for o,f,g in [(4,.8,.4),(5,.5,.6)]:
            for h,value in [(0,f),(1,g)]:
                old=dict(complete=True,all_inner_rules_passed=True,predictive_cdf_passed=True,
                    maximum_evidence_error=1e-12,values=np.full(512,value).tolist())
                payload['old/n1/'+str(o)+'/'+str(h)+'/results.json']=old;cached[o,h]=old
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'upper').mkdir()
            with patch.object(upper,'ROOT',root),patch.object(upper,'read',side_effect=payload.__getitem__), \
                    patch('performance.refine_historical_observed_prediction.field_from_snapshot',side_effect=lambda p:p['value']), \
                    patch('performance.refine_historical_observed_prediction.conditioned_value',side_effect=lambda field,*a:(field,1e-12,True,32,0.)) as evaluate:
                upper.diagnose_fixed('plan')
            got=json.loads((root/'upper/fixed_axis_diagnostic.json').read_text())
            self.assertEqual(evaluate.call_count,96);self.assertEqual(got['new_fixed_action_evaluations'],96)
            self.assertEqual(got['reused_fixed_action_values'],2048);self.assertTrue(got['all_inner_rules_passed'])
            self.assertFalse(got['independent_confirmation'])
            for row in got['rows']:self.assertAlmostEqual(row['factorized_fixed_action_tower_residual'],0.,places=14)
        # An objective-axis shift must remain visible with the same risk rule.
        cached[4,0]['values'][32:]=np.full(480,.4).tolist()
        rows=upper.fixed_rule_rows(spec,rules,state,cached,{str(o):{str(t):np.full(t,g).tolist()
            for t in [16,32]} for o,g in [(4,.4),(5,.6)]})
        self.assertAlmostEqual(rows[5]['expected_reward']-rows[4]['expected_reward'],-.09)

    def test_bulk_admission_matches_individual_checks_with_heterogeneous_rules_and_joint_identities(self):
        rng=np.random.default_rng(42);data=[]
        for h,count in enumerate([3,4]):
            rows=[]
            for p in range(count):
                low=rng.uniform(.1,.2,(2,3,4));high=low.copy();high[0,1,0]+=3e-3 if p==1 else 1e-5
                raw={str(power):dict(statistics=value.tolist(),costs=rng.uniform(10,20,(2,3)).tolist(),
                    predictive_log_evidence_max_error=2e-4 if p==2 else 1e-6)
                    for power,value in [(8,low*.9),(12+p*2,low),(14+p*2,high)]}
                native={str(power):dict(rule) for power,rule in raw.items()}
                rows.append(dict(position=p,head_log_scores=rng.normal(size=2).tolist(),raw=raw,projected=native))
            data.append(dict(records=rows))
        bulk=all_head_checks(data,[.7,.3],1e-4)
        for h,part in enumerate(data):
            for p,row in enumerate(part['records']):
                expected=head_checks(row,data[1-h]['records'],[.7,.3],h,1e-4)
                for key,value in expected.items():
                    if isinstance(value,float):self.assertAlmostEqual(bulk[h][p][key],value,places=13)
                    else:self.assertEqual(bulk[h][p][key],value)

    def test_merge_grades_both_heads_against_the_completed_joint_bank(self):
        actions=[dict(kind='direct',primary=i,shortlist=[i],incumbent_position=0,budgets=[10]) for i in [0,1]]
        def record(value,powers):
            rules={str(p):dict(statistics=[[[value,0.,0.,0.]]*2],costs=[[10.,10.]],
                predictive_log_evidence_max_error=0.,effective_sample_size_min=32.,
                diagnostics_by_identity=[dict(predictive_log_evidence_max_error=0.,effective_sample_size_min=32.)]) for p in powers}
            return dict(position=0,label=.1,eta_label=None,head_log_scores=[0.],utility_identities=[0],
                raw=rules,projected=copy.deepcopy(rules))
        data=[dict(actions=actions,records=[record(v,[8,10])]) for v in [.8,.5]]
        jobs=[dict(name='head'+str(h),case='source_n1',origin=0,head=h,ordinal=0,raw=True,native=True,target_power=14) for h in [0,1]]
        payload={'plan':dict(parent_plan='parent',output='precision'),
            'parent':dict(maximum_action_resolution_difference=1e-4),
            'precision/policy14/registered_jobs.json':dict(jobs=jobs,shards=1)}
        for job,value in zip(jobs,[.80015,1.]):
            base='precision/policy14/'+job['name']+'/shard0/'
            payload[base+'head_records.json']=dict(actions=actions,records=[record(value,[14])],prior_weights=[1.])
            payload[base+'results.json']=dict(complete=True,cpu_seconds=.5)
            payload['precision/shards/'+job['name']+'/results.json']=dict(complete=True,cpu_seconds=1.)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for job in jobs:
                (root/'precision/policy14'/job['name']).mkdir(parents=True)
                (root/'precision/shards'/job['name']).mkdir(parents=True)
            def read(path):return payload[str(Path(path).relative_to(root))] if Path(path).is_absolute() else payload[path]
            with patch.object(policies,'ROOT',root),patch.object(policies,'read',side_effect=read), \
                    patch.object(policies,'current_head',return_value=(dict(),data)):
                policies.merge(14,'plan')
            for h,job in enumerate(jobs):
                result=json.loads((root/'precision/shards'/job['name']/'results.json').read_text())
                expected=head_checks(data[h]['records'][0],data[1-h]['records'],[1.],h,1e-4)
                self.assertEqual(result['checks'],expected)
                self.assertAlmostEqual(result['cpu_seconds'],1.5)
            self.assertFalse(json.loads((root/'precision/shards/head0/results.json').read_text())['checks']['raw_passed'])

    def test_upper_tail_inverts_original_identity_cdf_and_retains_positive_domain_mass(self):
        fields=[[object(),object()] for _ in range(6)];model=SimpleNamespace(components=fields)
        class CDF:
            def inverse(self,q):return 2*q-1
            def adaptive(self,y):return (y+1)/2,0.
        with patch('performance.refine_historical_remaining_query_failures.RawPredictiveCDF',return_value=CDF()) as cdf:
            rule=positive_risk_tail(model,63,5,dict(maximum_inverse_cdf_error=1e-8),
                interval=(.9,1.),order=16,position=128)
        cdf.assert_called_once_with(fields[5][1],63)
        self.assertEqual(rule['positions'],list(range(128,144)))
        self.assertGreater(min(rule['weights']),0.)
        self.assertAlmostEqual(sum(rule['weights']),.1)
        self.assertTrue(all(.8<y<1. for y in rule['labels']))
        # A positive Gauss rule integrates a linear CDF-coordinate function exactly.
        self.assertAlmostEqual(np.dot(rule['weights'],rule['labels']),.09)

    def test_body_tail_weights_use_domain_mass_without_double_counting_cached_upper_points(self):
        spec=dict(predictive_seed=202610092,risk_tail_interval=[.9,1.])
        tails={str(o):{'16':dict(positions=list(range(128,144)),weights=np.full(16,.1/16).tolist())} for o in [4,5]}
        for prefix in [64,128]:
            rules=upper.head_rules(spec,tails,[4,5],32,prefix,16)
            for o in [4,5]:
                risk=rules[1,o];body=risk['positions'][:-16]
                self.assertEqual(len(body),prefix-(7 if prefix==64 else 13))
                self.assertEqual(len(set(risk['positions'])),len(risk['positions']))
                self.assertAlmostEqual(sum(risk['weights'][:-16]),.9)
                self.assertAlmostEqual(sum(risk['weights']),1.)
                self.assertAlmostEqual(sum(rules[0,o]['weights']),1.)

    def test_weighted_composition_prices_native_choice_under_raw_law(self):
        def record(position,raw,native):
            def rules(value):return {str(p):dict(statistics=np.asarray(value).reshape(1,2,1).tolist(),
                costs=[[10.,10.]],predictive_log_evidence_max_error=0.) for p in [8,10]}
            return dict(position=position,head_log_scores=[0.],utility_identities=[0],raw=rules(raw),projected=rules(native))
        heads={(0,0):[record(0,[1.,1.],[1.,1.])],
            (1,0):[record(0,[.4,.2],[.4,.2]),record(128,[.4,.6],[.4,.7])]}
        rules={(0,0):dict(positions=[0],weights=[1.]),(1,0):dict(positions=[0,128],weights=[.9,.1])}
        before=copy.deepcopy(heads)
        result=grid(heads,np.array([1.]),[0],[1,2],[.4,.4,0],1,1,head_rules=rules)
        self.assertAlmostEqual(result['raw_net_reward_of_projected_choice_control_variate'],.02)
        self.assertAlmostEqual(result['projected_value_bias_at_selected_action'],.01)
        self.assertAlmostEqual(result['raw_fixed_action_tower_residual'],0.)
        self.assertAlmostEqual(result['quadrature_joint_probability_mass'],1.)
        self.assertEqual(heads,before)
        uniform=grid(heads,np.array([1.]),[0],[1,2],[.4,.4,0],1,1)
        rules[1,0]['weights']=[.5,.5]
        explicit=grid(heads,np.array([1.]),[0],[1,2],[.4,.4,0],1,1,head_rules=rules)
        self.assertEqual(uniform,explicit)

    def test_shared_low_order_group_matches_serial_raw_and_native_records(self):
        fields=[(np.full(3,v),np.eye(3)*.02+.003) for v in [1.,-.2,-4.,-4.]]
        model=JointResponseMixture(['one'],[paired_fields(fields)])
        terminal=ConditionalTerminalActions([[1,0,2]],[1],dict(objective_reference=2.,chance_alpha=.05,
            familywise_delta=.05,comparison_budget=8,comparison_delta=.05))
        labels={128:-.13,129:-.25};jobs=[dict(name='r'+str(p),position=p,label=y) for p,y in labels.items()]
        spec=dict(output='tail',reuse_precision_plan='old',cases=[dict(state='source')],query=0,
            powers=[4,5],seed=714,maximum_case_cpu_seconds=420)
        payload={'plan':spec,'tail/preparation.json':dict(groups=[dict(name='group0',jobs=jobs)]),
            'tail/source/bank.json':dict(utility_identities=[0])}
        expected=evaluate_head_records(model,terminal,360,0,1,labels,list(labels),powers=[4,5],
            projected_powers=[4,5],seed=714,guarded_mean_position=0)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(upper,'ROOT',root),patch.object(upper,'read',side_effect=payload.__getitem__), \
                    patch.object(upper,'restore',return_value=(model,terminal,None,None)):
                upper.run_group('group0',2,'plan')
            for row in expected:
                got=json.loads((root/'tail/new_heads'/('r'+str(row['position']))/'head_records.json').read_text())['records'][0]
                self.assertEqual(got['head_log_scores'],row['head_log_scores'])
                for kind in ['raw','projected']:
                    for power in ['4','5']:
                        np.testing.assert_allclose(got[kind][power]['statistics'],row[kind][power]['statistics'],atol=1e-13,rtol=0.)
            self.assertTrue(json.loads((root/'tail/groups/group0/results.json').read_text())['complete'])
        self.assertEqual(model.paid_episodes,0)

    def test_staging_keeps_old_inputs_and_starts_group_refinement_from_real_low_rules(self):
        def record(position,changed=False):
            rules={str(p):dict(statistics=[[[.5+(.2 if changed and p==10 else 0.)]]],
                costs=[[10.]],predictive_log_evidence_max_error=0.) for p in [8,10]}
            return dict(position=position,head_log_scores=[0.],raw=rules,projected=copy.deepcopy(rules))
        case=dict(name='source_n1',state='source',head_counts=[1,3],objective_prefixes=[1,1])
        spec=dict(output='tail',reuse_precision_plan='old',precision_plan='precision_plan',cases=[case],
            accepted_baseline_inputs={'source':'baseline'},predictive_prefixes=[1,3],maximum_action_resolution_difference=1e-4)
        precision_spec=dict(parent_plan='plan',output='precision',cases=[case],
            head_inputs={'source_n1':['tail/inputs/f_{origin}','tail/inputs/g_{origin}']})
        heads={(0,0):[record(0)],(1,0):[record(0)]};bank=dict(origins=[0],prior_weights=[1.])
        actions=[dict(kind='direct',primary=0)]
        jobs=[dict(name='tail'+str(p),origin=0,position=p) for p in [1,2]]
        payload={'plan':spec,'precision_plan':precision_spec,'old':dict(output='old',cases=[case]),
            'old/registered_jobs.json':dict(jobs=[]),'tail/source/bank.json':bank,
            'tail/preparation.json':dict(jobs=jobs),
            'baseline/source/n1/baseline/terminal_rules.json':dict(rows=[dict(actions=actions)])}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for job in jobs:
                path=root/'tail/new_heads'/job['name'];path.mkdir(parents=True)
                (path/'results.json').write_text(json.dumps(dict(complete=True)))
                (path/'head_records.json').write_text(json.dumps(dict(actions=actions,records=[record(job['position'],True)])))
            before=copy.deepcopy(heads)
            def read(path):return payload[path] if path in payload else json.loads((root/path).read_text())
            with patch.object(upper,'ROOT',root),patch.object(upper,'read',side_effect=read), \
                    patch.object(precision,'ROOT',root),patch.object(precision,'read',side_effect=read), \
                    patch.object(upper,'assemble_case',return_value=(heads,bank,[])):
                upper.stage('plan')
            self.assertEqual(heads,before)
            registry=json.loads((root/'precision/registered_jobs.json').read_text())
            self.assertEqual(len(registry['jobs']),2)
            for job in registry['jobs']:
                result=json.loads((root/'precision/shards'/job['name']/'results.json').read_text())
                self.assertTrue(result['complete']);self.assertFalse(result['numerical_diagnostics_passed'])
                self.assertTrue(result['low_rules_reused_for_group_refinement']);self.assertEqual(result['cpu_seconds'],0.)


if __name__=='__main__':unittest.main()
