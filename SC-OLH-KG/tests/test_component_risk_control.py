import copy
import unittest

import numpy as np

from core.conditional_raw_reference import (component_control_risk_statistics,
    component_risk_statistics, shared_risk_reference,conditional_risk_action_statistics)
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


if __name__=='__main__':unittest.main()
