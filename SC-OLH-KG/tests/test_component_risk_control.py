import copy
import unittest

import numpy as np

from core.conditional_raw_reference import (component_control_risk_statistics,
    component_risk_statistics, component_risk_allocations, shared_risk_reference,conditional_risk_action_statistics)
from test_conditional_terminal_kg import correlated_fields
from test_positive_source_mixture import terminal


class ComponentRiskControlTests(unittest.TestCase):
    def setUp(self):
        self.field=correlated_fields()[1];self.future=copy.deepcopy(self.field);self.label=-.28
        self.score=self.future.update_batch(0,self.label,None,1);self.protocol=terminal().protocol
        self.action=dict(kind='guarded',primary=0,shortlist=[0,1,2],incumbent_position=1,budgets=[130,130,83])

    def seeds(self,field):
        _,rank=np.unique(field.means,axis=0,return_inverse=True)
        return 202610077+rank*104729

    def test_budget_group_retains_each_joint_value_and_cost(self):
        from core.budgeted_guarded_kg import BudgetedTerminalActions
        terminal=BudgetedTerminalActions([self.action['shortlist']],[1],self.protocol)
        actions=[dict(self.action,budgets=b) for kind,_,_,b in terminal.allocations(359) if kind=='guarded']
        for raw in (False,True):
            kwargs=dict(query=0,raw=raw,order=16,power=4,seed=self.seeds(self.field),integrated_position=1)
            values,costs,d=component_risk_allocations(self.field,self.future,actions,self.label,self.score,self.protocol,**kwargs)
            singles=[component_risk_statistics(self.field,self.future,a,self.label,self.score,self.protocol,**kwargs) for a in actions]
            np.testing.assert_allclose(values,[s[0] for s in singles],atol=1e-13,rtol=0.)
            np.testing.assert_allclose(costs,[s[1] for s in singles],atol=1e-10,rtol=0.)
            self.assertEqual(d['components'],len(self.field.means))
            self.assertEqual(d['certificate_power_pairs'],6)
            self.assertGreater(np.ptp(values[:,0]),1e-3)
            self.assertTrue(np.all(costs<=359+1e-10))
            split_before=copy.deepcopy(self.field);split_after=copy.deepcopy(self.future)
            for field in (split_before,split_after):
                indices=np.array([0,0,1]);field.means=field.means[indices];field.covariances=field.covariances[indices]
                field.log_weights=field.log_weights[indices];field.log_weights[:2]-=np.log(2.)
            new_values,new_costs,_=component_risk_allocations(split_before,split_after,actions,self.label,self.score,
                self.protocol,**dict(kwargs,seed=self.seeds(split_before)))
            np.testing.assert_allclose(new_values,values,atol=1e-13,rtol=0.)
            np.testing.assert_allclose(new_costs,costs,atol=1e-10,rtol=0.)

    def test_ordinary_integral_matches_existing_rule_and_control_is_split_invariant(self):
        for raw in (False,True):
            kwargs=dict(query=0,raw=raw,order=32,power=3,seed=self.seeds(self.field))
            reference=shared_risk_reference(self.field,self.future,self.action,self.label,self.score,self.protocol,
                query=0,raw=raw,order=32,power=6,seed=54321)
            value,cost,d=component_control_risk_statistics(self.field,self.future,self.action,self.label,self.score,
                self.protocol,reference,**kwargs)
            ordinary,ordinary_cost,_=component_risk_statistics(self.field,self.future,self.action,self.label,
                self.score,self.protocol,**kwargs)
            np.testing.assert_allclose(d['ordinary_statistics'],ordinary,atol=1e-12,rtol=0.)
            self.assertAlmostEqual(d['ordinary_cost'],ordinary_cost,places=10)
            for indices,split in [(np.array([1,0]),False),(np.array([0,0,1]),True)]:
                before=copy.deepcopy(self.field);after=copy.deepcopy(self.future)
                for field in (before,after):
                    field.means=field.means[indices];field.covariances=field.covariances[indices]
                    field.log_weights=field.log_weights[indices]
                    if split:field.log_weights[:2]-=np.log(2.)
                new_reference=shared_risk_reference(before,after,self.action,self.label,self.score,self.protocol,
                    query=0,raw=raw,order=32,power=6,seed=54321)
                new_value,new_cost,_=component_control_risk_statistics(before,after,self.action,self.label,self.score,
                    self.protocol,new_reference,query=0,raw=raw,order=32,power=3,seed=self.seeds(before))
                np.testing.assert_allclose(new_value,value,atol=1e-12,rtol=0.)
                self.assertAlmostEqual(new_cost,cost,places=10)

    def test_identical_components_have_zero_residual_without_dropping_weights(self):
        for field in (self.field,self.future):
            field.means=np.repeat(field.means[:1],3,axis=0)
            field.covariances=np.repeat(field.covariances[:1],3,axis=0)
            field.log_weights=np.log([.2,.3,.5])
        # Recompute the actual observation evidence rather than reusing a
        # different mixture's evidence in the raw importance ratio.
        self.future=copy.deepcopy(self.field);self.score=self.future.update_batch(0,self.label,None,1)
        for raw in (False,True):
            reference=shared_risk_reference(self.field,self.future,self.action,self.label,self.score,self.protocol,
                query=0,raw=raw,order=32,power=7,seed=54321)
            value,cost,d=component_control_risk_statistics(self.field,self.future,self.action,self.label,self.score,
                self.protocol,reference,query=0,raw=raw,order=32,power=2,seed=self.seeds(self.field))
            np.testing.assert_allclose(d['difference_integral'],0.,atol=1e-13,rtol=0.)
            expected=np.asarray(reference['numerator'])/reference['mass']
            np.testing.assert_allclose(np.r_[value,cost],expected,atol=1e-11,rtol=0.)
            self.assertEqual(d['components'],3)

    def test_raw_direct_matches_independent_historical_query_reference(self):
        from performance.diagnose_historical_predictive_tower import query_direct_value
        action=dict(kind='direct',primary=0,shortlist=[0],incumbent_position=0,budgets=[359])
        reference=shared_risk_reference(self.field,self.future,action,self.label,self.score,self.protocol,
            query=0,raw=True,order=64,power=0,seed=54321)
        value,cost,_=component_control_risk_statistics(self.field,self.future,action,self.label,self.score,self.protocol,
            reference,query=0,raw=True,order=64,power=0,seed=self.seeds(self.field))
        expected,_=query_direct_value(self.field,self.future,self.label,self.score,1,359,self.protocol,1,None,64)
        self.assertAlmostEqual(value[0],expected,places=12)
        self.assertAlmostEqual(cost,359.,places=10)

    def test_integrating_incumbent_retains_the_full_joint_raw_reference(self):
        from scipy.stats import qmc
        from core.positive_mixture_lookahead import condition_positive_raw
        from core.raw_likelihood_lookahead import risk_terminal_statistics
        nodes=qmc.Sobol(7,scramble=True,seed=202610077).random_base2(18)
        draws,weights,_=condition_positive_raw(self.field,self.future,0,[0,1,2],nodes,self.label,self.score)
        expected,cost=risk_terminal_statistics(draws.reshape(1,len(nodes),3,2),weights[None],1,
            self.action['budgets'],self.protocol)
        for position in (0,1,2):
            value,actual_cost,_=component_risk_statistics(self.field,self.future,self.action,self.label,
                self.score,self.protocol,query=0,raw=True,order=64,power=12,seed=self.seeds(self.field),
                integrated_position=position)
            np.testing.assert_allclose(value,expected[0],atol=2e-4,rtol=0.)
            self.assertLess(abs(actual_cost-cost[0]),.03)

    def test_partial_mass_diagnostic_keeps_initial_certificate_cost(self):
        from scipy.stats import qmc
        from core.conditional_terminal_kg import condition_eta_components
        nodes=qmc.Sobol(6,scramble=True,seed=202610077).random_base2(6)
        eta,mean,covariance,position=condition_eta_components(self.future,[0,1,2],nodes)
        args=(eta,mean,covariance,position,nodes,self.action,self.protocol)
        weights=np.full(len(nodes),1/len(nodes));kwargs=dict(raw=False,query_points=3)
        full,cost=conditional_risk_action_statistics(*args,weights,**kwargs)
        partial,partial_cost=conditional_risk_action_statistics(*args,.9*weights,**kwargs)
        np.testing.assert_allclose(partial,.9*full,atol=1e-14,rtol=0.)
        initial=sum(self.action['budgets'][:2])
        self.assertAlmostEqual(partial_cost,initial+.9*(cost-initial),places=11)

    def test_multi_record_collector_replays_the_frozen_worst_block(self):
        from performance.complete_historical_source_heads import read
        from performance.diagnose_historical_component_guarded import compressed_read
        from performance.diagnose_historical_shared_risk import BASE,OUT,NAME
        from performance.validate_historical_incumbent import coordinate_result
        bundle=compressed_read(BASE/'marginals'/(NAME+'.json.gz'))
        parts=[read(str(OUT/'coordinate_parts'/f'{h}_{a}_{kind}.json')) for h in bundle['utility_identities']
            for a,b in enumerate(bundle['blocks']) if b['action']['kind']=='guarded' for kind in ('raw','projected')]
        historical=read(str(BASE/'reference_results'/(NAME+'.json')))
        baseline=next(r for r in historical['rows'] if r['name']=='E32_128')
        result=coordinate_result(bundle,parts,baseline);expected=read(str(OUT/'coordinate_summary.json'))
        for actual,reference in zip(result['comparisons'],expected['comparisons']):
            self.assertEqual((actual['low'],actual['high']),(reference['low'],reference['high']))
            np.testing.assert_allclose([actual['raw'],actual['projected']],
                [reference['raw'],reference['projected']],atol=1e-15,rtol=0.)
        self.assertEqual(result['selected_action_precision_passed'],expected['selected_action_precision_passed'])
        changed=copy.deepcopy(parts)
        changed[0]['rows']['E7']['statistics'][0]+=.01
        self.assertFalse(coordinate_result(bundle,changed,baseline)['selected_action_precision_passed'])

    def test_partial_control_collector_preserves_unrepaired_failures(self):
        from performance.complete_historical_source_heads import read
        from performance.diagnose_historical_component_guarded import compressed_read
        from performance.diagnose_historical_shared_risk import BASE,OUT,NAME
        from performance.validate_historical_incumbent import control_result,refined_result,resolution_result
        bundle=compressed_read(BASE/'marginals'/(NAME+'.json.gz'))
        coordinate=read(str(OUT/'coordinate_summary.json'))
        coordinate['failed_action_checks']=[dict(action=0,kind='raw'),dict(action=1,kind='projected')]
        coordinate['selected_action_precision_passed']=False
        parts=[read(str(OUT/'parts'/f'{h}_{a}_{k}.json')) for h in bundle['utility_identities']
            for a,b in enumerate(bundle['blocks']) if b['action']['kind']=='guarded' for k in ('raw','projected')]
        result=control_result(bundle,coordinate,parts)
        expected=read(str(OUT/'summary.json'))
        for actual,reference in zip(result['comparisons'],expected['comparisons'][:4]):
            np.testing.assert_allclose([actual['raw'],actual['projected']],
                [reference['raw'],reference['projected']],atol=1e-15,rtol=0.)
        self.assertTrue(result['selected_action_precision_passed'])
        raw_only=[p for p in parts if p['action']==0 and p['kind']=='raw']
        partial=control_result(bundle,coordinate,raw_only)
        self.assertEqual(partial['remaining_incumbent_failures'],[dict(action=1,kind='projected')])
        self.assertFalse(partial['selected_action_precision_passed'])
        coordinate_parts=[read(str(OUT/'coordinate_parts'/f'{p["identity"]}_{p["action"]}_{p["kind"]}.json')) for p in parts]
        refined=[dict(p,rows={n:copy.deepcopy(p['rows']['G7']) for n in ('H8','I8','J8')}) for p in coordinate_parts]
        high=refined_result(bundle,coordinate,result,refined)
        self.assertTrue(high['selected_action_precision_passed'])
        self.assertTrue(resolution_result(bundle,high,[])['selected_action_precision_passed'])
        required=copy.deepcopy(high);required['comparisons'][0]['actions'][0]['difference']=.01
        self.assertFalse(resolution_result(bundle,required,[])['selected_action_precision_passed'])
        altered=copy.deepcopy(refined);altered[0]['rows']['J8']['statistics'][0]+=.01
        self.assertFalse(refined_result(bundle,coordinate,result,altered)['selected_action_precision_passed'])


if __name__=='__main__':unittest.main()
