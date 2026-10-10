import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import performance.historical_full_bank_development as execution
from performance.prepare_historical_independent_kg import check_replayed_snapshot, terminal_from_initialization
from performance.connect_historical_prior_posterior import snapshot
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.joint_response_mixture import JointResponseMixture
from core.conditional_terminal_kg import ConditionalTerminalActions, fast_expected_certificates
from test_positive_source_mixture import field


PROTOCOL = {'objective_reference': 1.5, 'familywise_delta': .05, 'comparison_budget': 8,
            'comparison_delta': .05, 'chance_alpha': .05}


class FullBankTests(unittest.TestCase):
    def test_paid_policy_ids_survive_full_bank_expansion(self):
        def forbidden(*args):
            raise AssertionError('observations and public metadata suffice')
        provider = SimpleNamespace(alpha=.05, tau=0., sigma_level=.04, simulate=forbidden,
            true_objective=forbidden, true_constraint_mean=forbidden, true_sigma=forbidden)
        candidates = [(i,) for i in range(73)]
        observations = [[.3, .1], [.7, .4]]
        terminal = terminal_from_initialization(provider, candidates, observations, PROTOCOL,
                                                paid_indices=[45, 10])
        np.testing.assert_array_equal(terminal.shortlists[:, 0], np.arange(73))
        np.testing.assert_array_equal(terminal.shortlists[np.arange(73), terminal.incumbents], np.full(73, 45))
        self.assertEqual(terminal.shortlists[63].tolist(), [63, 45, 10])
        count = sum(len(table) for _, table, _, _ in terminal.allocations(360))
        self.assertEqual(count, 365)
        original = terminal_from_initialization(provider, candidates[:8], observations, PROTOCOL)
        self.assertEqual(original.shortlists[3].tolist(), [3, 0, 1])

    def test_full_bank_updates_reproduce_reordered_subset_evidence(self):
        rng = np.random.default_rng(631)
        components = []
        for offset in (.1, -.2):
            mean = np.column_stack([np.arange(5)*.03+offset, np.full(5, -4.)])
            matrix = rng.normal(size=(10, 10))*.01
            covariance = (matrix@matrix.T+np.eye(10)*.025).reshape(5, 2, 5, 2)
            components.append(JointFunctionRiskPosterior.from_moments(mean, covariance))
        full = JointResponseMixture(['GP'], [tuple(components)])
        indices = [3, 1, 4]
        prior = execution.restricted_snapshot(full, indices)['components'][0]
        fields = [JointFunctionRiskPosterior.from_moments(prior[h+'_mean'], prior[h+'_covariance']) for h in ('f', 'g')]
        subset = JointResponseMixture(['GP'], [tuple(fields)])
        for local, value in enumerate(([.25, -.15], [.15, -.1])):
            full.update_raw(indices[local], value); subset.update_raw(local, value)
        before = copy.deepcopy(full.history)
        check_replayed_snapshot(execution.restricted_snapshot(full, indices), snapshot(subset))
        self.assertEqual(full.history, before)
        self.assertEqual(full.paid_episodes, 2)
        wrong = execution.restricted_snapshot(full, indices)
        wrong['history'][0]['candidate'] = 1
        with self.assertRaises(AssertionError):
            check_replayed_snapshot(wrong, snapshot(subset), roundoff_units=146)
        wrong = execution.restricted_snapshot(full, indices)
        wrong['components'][0]['g_mean'][0][0] += 1e-10
        with self.assertRaises(AssertionError):
            check_replayed_snapshot(wrong, snapshot(subset), roundoff_units=146)

    def test_certificate_cache_preserves_both_native_terminal_rules(self):
        model = JointResponseMixture(['source'], [(field([.1, .8], [.4, .6]), field([-.4, .1], [.7, .3]))])
        terminal = ConditionalTerminalActions([[0, 1, 2], [1, 0, 2], [2, 0, 1]], [1, 1, 1], PROTOCOL)
        reference = [execution.terminal_rule(model, terminal, 360, [0], power, 731) for power in (5, 7)]
        with patch('core.conditional_terminal_kg.fast_expected_certificates', wraps=fast_expected_certificates) as direct:
            cache = {}
            actual = [execution.terminal_rule(model, terminal, 360, [0], power, 731, cache) for power in (5, 7)]
            self.assertEqual(direct.call_count, 1)
        self.assertEqual(actual, reference)
        self.assertEqual(model.paid_episodes, 0)

    def test_unfrozen_full_bank_case_blocks_truth_scoring(self):
        spec = {'output': 'out'}
        states = [{'name': 'source'}, {'name': 'gp'}]
        deployment = {'shards': [{'output': 'source'}, {'output': 'gp'}]}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = {execution.PLAN: spec, str(root/'out/deployment.json'): deployment,
                    'source/results.json': {'name': 'source', 'recommendation_frozen': True},
                    'gp/results.json': {'name': 'gp', 'recommendation_frozen': False}}
            with patch.object(execution, 'ROOT', root), patch.object(execution, 'read', side_effect=data.__getitem__), \
                 patch.object(execution, 'context', return_value=({}, {}, states, {}, {}, {}, [])), \
                 patch.object(execution, 'problem_for') as problem:
                with self.assertRaises(ValueError): execution.collect()
                problem.assert_not_called()
                self.assertFalse((root/'out/recommendations.json').exists())


if __name__ == '__main__':
    unittest.main()
