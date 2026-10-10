import copy
import unittest

import numpy as np
from scipy.stats import qmc

from core.conditional_raw_reference import conditional_raw_coordinates, raw_guarded_head_statistics, component_risk_statistics
from core.positive_mixture_lookahead import condition_positive_raw
from core.raw_likelihood_lookahead import objective_terminal_statistics, risk_terminal_statistics
from test_conditional_terminal_kg import correlated_fields
from test_positive_source_mixture import terminal


class ConditionalRawReferenceTests(unittest.TestCase):
    def test_exported_joint_snapshot_restores_all_policies_and_covariances(self):
        from performance.refine_historical_observed_prediction import field_from_snapshot
        original=correlated_fields()[1]
        payload=dict(means=original.means.tolist(),covariances=original.covariances.tolist(),
            log_weights=original.log_weights.tolist(),paid_batches=original.paid_batches,paid_episodes=original.paid_episodes)
        restored=field_from_snapshot(payload)
        np.testing.assert_array_equal(restored.means,original.means)
        np.testing.assert_array_equal(restored.covariances,original.covariances)
        self.assertEqual(restored.candidate_count,3)

    def test_component_rule_retains_joint_law_and_is_split_and_order_invariant(self):
        field=correlated_fields()[1];future=copy.deepcopy(field);label=-.28
        evidence=future.update_batch(0,label,None,1)
        action=dict(kind='guarded',primary=0,shortlist=[0,1,2],incumbent_position=1,budgets=[130,130,83])
        for separate in (False,True):
            def rule_seed(current):
                _,rank=np.unique(current.means,axis=0,return_inverse=True)
                return 202610077+rank*104729 if separate else 202610077
            kwargs=dict(query=0,raw=True,order=32,power=5)
            actual,cost,diagnostic=component_risk_statistics(field,future,action,label,evidence,terminal().protocol,
                block_size=1,seed=rule_seed(field),**kwargs)
            self.assertLess(abs(diagnostic['predictive_log_evidence_error']),1e-8)
            for indices,split in [(np.array([1,0]),False),(np.array([0,0,1]),True)]:
                before=copy.deepcopy(field);after=copy.deepcopy(future)
                for current in (before,after):
                    current.means=current.means[indices];current.covariances=current.covariances[indices]
                    current.log_weights=current.log_weights[indices]
                    if split:current.log_weights[:2]-=np.log(2.)
                result,new_cost,_=component_risk_statistics(before,after,action,label,evidence,terminal().protocol,
                    seed=rule_seed(before),**kwargs)
                np.testing.assert_allclose(result,actual,atol=1e-12,rtol=0.)
                self.assertAlmostEqual(new_cost,cost,places=10)

    def test_high_order_direct_gh_avoids_normal_cdf_round_trip(self):
        from performance.diagnose_historical_predictive_tower import query_direct_value
        field=correlated_fields()[1];future=copy.deepcopy(field);label=-.28
        evidence=future.update_batch(0,label,None,1)
        action=dict(kind='direct',primary=0,shortlist=[0],incumbent_position=0,budgets=[359])
        actual,_,d=component_risk_statistics(field,future,action,label,evidence,terminal().protocol,
            query=0,raw=True,order=64,power=0,seed=202610077,block_size=1)
        # Historical query=deploy GH code uses the normal nodes directly and
        # conditions the scalar mean through a separate implementation.
        expected,_=query_direct_value(field,future,label,evidence,1,359,terminal().protocol,1,None,64)
        self.assertTrue(np.isfinite(actual).all())
        self.assertAlmostEqual(actual[0],expected,places=12)
        self.assertLess(abs(d['predictive_log_evidence_error']),1e-8)

    def test_conditional_statistics_match_historical_joint_transport(self):
        action = {'kind': 'guarded', 'primary': 0, 'shortlist': [0, 1, 2],
            'incumbent_position': 1, 'budgets': [130, 130, 83]}
        protocol = terminal().protocol
        nodes = qmc.Sobol(6, scramble=True, seed=202610077).random_base2(16)
        full_nodes = qmc.Sobol(7, scramble=True, seed=202610077).random_base2(18)
        for head, field in enumerate(correlated_fields()):
            future = copy.deepcopy(field); label = [.4, -.28][head]
            evidence = future.update_batch(0, label, None, 1)
            actual, diagnostics = raw_guarded_head_statistics(field, future, action, nodes, label,
                evidence, head, protocol)
            draws, weights, _ = condition_positive_raw(field, future, 0, [0, 1, 2], full_nodes, label, evidence)
            draws = draws.reshape(1, len(full_nodes), 3, 2)
            expected = (objective_terminal_statistics(draws, weights[None], 1, protocol)[0] if head == 0
                else risk_terminal_statistics(draws, weights[None], 1, action['budgets'], protocol)[0][0])
            np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=0.)
            self.assertLess(abs(diagnostics['predictive_log_evidence_error']), 1e-4)
            self.assertGreater(diagnostics['effective_sample_size'], len(nodes)*.95)
            self.assertEqual(field.paid_episodes, 0)

    def test_exactly_shared_eta_is_preserved_after_raw_conditioning(self):
        field = correlated_fields(shared_eta=True)[1]
        future = copy.deepcopy(field); evidence = future.update_batch(0, -.1, None, 1)
        nodes = qmc.Sobol(6, scramble=True, seed=202610077).random_base2(10)
        eta, mean, covariance, weights, _ = conditional_raw_coordinates(field, future, [0, 1, 2],
            nodes, -.1, evidence)
        np.testing.assert_allclose(eta[:, 1:], np.repeat(eta[:, :1], 2, axis=1), atol=1e-12, rtol=0.)
        self.assertAlmostEqual(weights.sum(), 1., places=14)
        self.assertGreater(np.abs(covariance[:, 0, 1]).max(), 1e-5)
        self.assertTrue(np.all(np.isfinite(mean)))


if __name__ == '__main__':
    unittest.main()
