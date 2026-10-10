import unittest
import copy

import numpy as np

from performance.refine_historical_achievable_precision import compose_records, head_checks
from performance.refine_historical_achievable_policies import partition_record, merge_rule
from performance.diagnose_historical_predictive_tower import restrict_field, direct_value, query_direct_value, summarize_tower
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.vectorized_joint_function_risk import update_raw_observation
from core.conditional_terminal_kg import ConditionalTerminalActions
from core.raw_terminal_head_grid import raw_terminal_head
from scipy.stats import qmc


def record(qualities, *, evidence=0.):
    raw={}
    for power,value in qualities.items():
        raw[str(power)]={'statistics':[[[value,0.,0.,0.]]], 'costs':[[356.]],
            'predictive_log_evidence_max_error':evidence}
    return {'raw':raw,'projected':copy.deepcopy(raw),'head_log_scores':[0.], 'utility_identities':[0]}


class AchievablePrecisionTests(unittest.TestCase):
    def test_head_resolution_contrast_holds_the_opposite_head_fixed(self):
        f=[record({14:.6,16:.7})];g=[record({14:.8,16:.9})]
        f_low=compose_records(f,g,[1.],'raw',(0,1))[0][0,0,0]
        g_low=compose_records(f,g,[1.],'raw',(1,0))[0][0,0,0]
        self.assertAlmostEqual(f_low,.6*.9);self.assertAlmostEqual(g_low,.7*.8)

    def test_mixed_record_precision_uses_each_records_last_two_rules(self):
        f=[record({8:.1,10:.5,12:.6,14:.61}),record({8:.6,10:.62})]
        g=[record({8:.7,10:.8,12:.82})]
        low=compose_records(f,g,[1.],'raw',0)[0][:,0,0]
        high=compose_records(f,g,[1.],'raw',1)[0][:,0,0]
        np.testing.assert_allclose(low,[.6*.8,.6*.8],atol=1e-14,rtol=0.)
        np.testing.assert_allclose(high,[.61*.82,.62*.82],atol=1e-14,rtol=0.)
        self.assertLess(float(np.max(abs(high-low))),.03)

    def test_head_admission_prices_reachable_branches_and_keeps_evidence_gate(self):
        f=record({8:.6,10:.6});g=record({8:.9,10:.9})
        f['raw']['8']['statistics'][0][0][3]=.1
        f['raw']['10']['statistics'][0][0][3]=.9
        checks=head_checks(f,[g],[1.],0,1e-4)
        self.assertTrue(checks['raw_passed'])
        self.assertTrue(checks['projected_passed'])
        self.assertEqual(checks['raw_maximum_action_difference'],0.)
        f['raw']['10']['predictive_log_evidence_max_error']=1.2e-4
        checks=head_checks(f,[g],[1.],0,1e-4)
        self.assertFalse(checks['evidence_passed'])

    def test_policy_merge_restores_action_order_and_keeps_worst_diagnostic(self):
        actions=[dict(kind='direct',shortlist=[i],incumbent_position=0,budgets=[359]) for i in range(3)]
        row=record({18:.6})
        values=np.arange(12,dtype=float).reshape(1,3,4)
        row['raw']['18'].update(statistics=values.tolist(),costs=[[1.,2.,3.]],
            diagnostics_by_identity=[dict(predictive_log_evidence_error=0.,
                predictive_log_evidence_max_error=0.,effective_sample_size_min=100.)],
            effective_sample_size_min=100.)
        row['projected']=copy.deepcopy(row['raw'])
        pieces=[]
        for indices,error,ess in [([2,0],2e-5,90.),([1],1.3e-4,70.)]:
            subset=partition_record(row,indices)
            rule=subset['raw']['18'];rule['predictive_log_evidence_max_error']=error
            rule['effective_sample_size_min']=ess
            rule['diagnostics_by_identity'][0].update(predictive_log_evidence_max_error=error,
                effective_sample_size_min=ess)
            pieces.append(dict(records=[subset],actions=[actions[i] for i in indices]))
        merged=merge_rule(actions,pieces,'raw',18)
        np.testing.assert_array_equal(merged['statistics'],values)
        self.assertEqual(merged['costs'],[[1.,2.,3.]])
        self.assertEqual(merged['predictive_log_evidence_max_error'],1.3e-4)
        self.assertEqual(merged['diagnostics_by_identity'][0]['effective_sample_size_min'],70.)
        self.assertEqual(row['raw']['18']['predictive_log_evidence_max_error'],0.)
        with self.assertRaises(AssertionError):merge_rule(actions,pieces[:1],'raw',18)

    def test_fixed_action_marginal_matches_full_batch_conditioning(self):
        protocol=dict(objective_reference=2.,chance_alpha=.05,familywise_delta=.05,
            comparison_budget=8,comparison_delta=.05)
        terminal=ConditionalTerminalActions([[1,0,2]],[1],protocol)
        nodes=qmc.Sobol(7,scramble=True,seed=713).random_base2(7)
        for head,level,label in [(0,1.,1.1),(1,-.2,-.13)]:
            means=np.tile([level,-3.],(3,1));C=np.eye(6)*.05+.01
            field=JointFunctionRiskPosterior.from_moments(means,C.reshape(3,2,3,2))
            full=copy.deepcopy(field);score=update_raw_observation(full,0,label,-3.2,4)
            marginal=restrict_field(field,[0,1]);future=copy.deepcopy(marginal)
            ms=update_raw_observation(future,0,label,-3.2,4)
            self.assertAlmostEqual(score,ms,places=14)
            reference,_,actions,_=raw_terminal_head(field,full,terminal,356,0,label,score,nodes,head,
                guarded_mean_position=0,replicates=4,eta_label=-3.2)
            fixed=next(i for i,a in enumerate(actions) if a['kind']=='direct' and a['primary']==1 and a['budgets']==[356])
            value,_=direct_value(marginal,future,1,nodes,label,ms,head,356,protocol,4,-3.2)
            self.assertAlmostEqual(value,reference[fixed,0],places=13)
            self.assertEqual(field.paid_episodes,0)

    def test_query_direct_quadrature_keeps_each_positive_component_mass(self):
        protocol=dict(objective_reference=2.,chance_alpha=.05,familywise_delta=.05)
        for head,level,label in [(0,1.,1.1),(1,-.2,-.13)]:
            field=JointFunctionRiskPosterior.from_moments(np.array([[level,-3.]]),
                np.array([[.06,.01],[.01,.15]]).reshape(1,2,1,2))
            field.means=np.array([[level,-3.],[level+.04,-2.9]])
            field.covariances=np.repeat(field.covariances,2,axis=0);field.log_weights=np.log([.4,.6])
            future=copy.deepcopy(field);score=update_raw_observation(future,0,label,-3.2,4)
            a,ea=query_direct_value(field,future,label,score,head,356,protocol,4,-3.2,32)
            b,eb=query_direct_value(field,future,label,score,head,356,protocol,4,-3.2,64)
            self.assertLess(max(ea,eb),1e-7);self.assertLess(abs(a-b),1e-9)
            nodes=qmc.Sobol(7,scramble=True,seed=713).random_base2(16)
            reference,_=direct_value(field,future,0,nodes,label,score,head,356,protocol,4,-3.2)
            self.assertLess(abs(reference-b),1e-4)

    def test_tower_summary_multiplies_heads_within_the_same_identity(self):
        heads=[]
        for origin,mass,f,g in [(4,.7,.8,.9),(5,.3,.3,.2)]:
            for head,value in [(0,f),(1,g)]:
                heads.append(dict(origin=origin,head=head,origin_prior_mass=mass,complete=True,
                    cpu_seconds=1.,all_inner_rules_passed=True,baseline_reward=.522,
                    prefixes={str(n):value for n in (8,64,512)}))
        result=summarize_tower(heads,'source_n4')
        self.assertAlmostEqual(result['prefixes']['512']['expected_reward'],.522,places=14)
        self.assertTrue(result['fixed_action_tower_diagnostic_passed'])
        self.assertFalse(summarize_tower(heads[:-1],'source_n4')['all_complete'])


if __name__=='__main__':unittest.main()
