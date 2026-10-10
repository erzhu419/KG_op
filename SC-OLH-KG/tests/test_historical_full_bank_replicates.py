import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from scipy.special import digamma, logsumexp

import performance.historical_full_bank_replicates as experiment
from core.conditional_terminal_kg import ConditionalTerminalActions
from performance.historical_full_bank_development import terminal_rule
from representation.joint_function_risk import local_joint_moments
from representation.joint_response_mixture import JointResponseMixture
from representation.vectorized_joint_function_risk import local_raw_moments
from test_positive_source_mixture import field
from test_historical_full_bank_development import PROTOCOL


class ReplicatedFullBankTests(unittest.TestCase):
    def test_vectorized_replicates_match_actual_source_scalar_likelihood(self):
        data = json.loads((Path(__file__).parent/'fixtures/historical_mixture_raw_fantasy.json').read_text())
        means = np.asarray(data['means'])[:12]; cov = np.asarray(data['covariances'])[:12]
        values = np.array([data['label']-.02, data['label']+.04, data['label'], data['label']+.01])
        k = (len(values)-1)/2
        eta = float(np.log(values.var(ddof=1))-digamma(k)+np.log(k))
        fast = local_raw_moments(means, cov, values.mean(), eta, len(values))
        for i in range(len(means)):
            reference = local_joint_moments(means[i], cov[i], values.mean(), eta, len(values))
            for key in ('mean', 'covariance', 'log_evidence'):
                np.testing.assert_allclose(fast[key][i], reference[key], atol=1e-8, rtol=1e-8)
        self.assertLessEqual(fast['integration_relative_error'], 1e-8)

    def test_paid_batch_preserves_full_covariance_mixture_evidence_and_debits(self):
        model = JointResponseMixture(['A', 'B'], [(field([.2, .4], [.3, .7]), field([-.3, -.1], [.6, .4])),
                                                 (field([.5, .6], [.8, .2]), field([-.4, -.2], [.2, .8]))])
        for pair in model.components:
            for current in pair:
                current.covariances[:, 1::2, 1::2] += .5*np.eye(3)
        model.update_raw(0, [.3, -.2]); before = copy.deepcopy(model)
        values = np.array([[.32, -.18], [.35, -.23], [.28, -.21], [.3, -.17]])
        k = (len(values)-1)/2
        eta = np.log(values.var(axis=0, ddof=1))-digamma(k)+np.log(k)
        references = []; scores = []
        for pair in before.components:
            local_scores = []
            for head, current in enumerate(pair):
                ix = [2, 3]; expected_means = []; expected_cov = []; evidence = []
                for mean, cov in zip(current.means, current.covariances):
                    local = local_joint_moments(mean[ix], cov[np.ix_(ix, ix)], values[:, head].mean(), eta[head], 4)
                    gain = np.linalg.solve(cov[np.ix_(ix, ix)], cov[ix, :]).T
                    expected_means.append(mean+gain@(local['mean']-mean[ix]))
                    expected_cov.append(cov+gain@(local['covariance']-cov[np.ix_(ix, ix)])@gain.T)
                    evidence.append(local['log_evidence'])
                normalization = float(logsumexp(current.log_weights+evidence))
                references.append((expected_means, expected_cov, current.log_weights+evidence-normalization))
                local_scores.append(normalization)
            scores.append(local_scores)
        record = model.update_replicates(1, values)
        np.testing.assert_allclose(record['head_log_scores'], scores, atol=1e-8, rtol=1e-8)
        for current, reference in zip([f for pair in model.components for f in pair], references):
            for actual, expected in zip((current.means, current.covariances, current.log_weights), reference):
                np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-8)
            self.assertEqual(current.paid_episodes, 5)
        self.assertEqual(model.paid_episodes, 5)
        self.assertEqual(model.history[-1]['replicates'], 4)

    def test_direct_restriction_matches_existing_terminal_integrator(self):
        model = JointResponseMixture(['source'], [(field([.1, .8], [.4, .6]), field([-.4, .1], [.7, .3]))])
        terminal = ConditionalTerminalActions([[0, 1, 2], [1, 0, 2], [2, 0, 1]], [1, 1, 1], PROTOCOL)
        for remaining in (360, 359, 356):
            full = terminal_rule(model, terminal, remaining, [0], 5, 731)
            direct = [i for i, a in enumerate(full['actions']) if a['kind'] == 'direct']
            best = sorted(direct, key=lambda i: (-full['values'][i], full['costs'][i]))[0]
            actual = experiment.direct_rule(model, terminal, remaining, 'source_common_task_rbf', 2e-6)
            self.assertEqual(actual['action'], full['actions'][best])
            self.assertAlmostEqual(actual['predicted_reward'], full['values'][best], places=14)

    def test_missing_recommendation_blocks_truth_scoring(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = {experiment.PLAN: {'output': 'out'}, str(root/'out/deployment.json'): {'shards': [{'output': 'missing'}]},
                    'missing/results.json': {'recommendation_frozen': False}}
            with patch.object(experiment, 'ROOT', root), patch.object(experiment, 'read', side_effect=data.__getitem__), \
                 patch.object(experiment, 'problem_for') as problem:
                with self.assertRaises(ValueError): experiment.collect()
                problem.assert_not_called()

    def test_json_bank_reaches_native_shortlist_in_its_tuple_format(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); state = {'name': 'source', 'task_id': 'task',
                'candidate_bank_indices': [0, 1], 'posterior_moment_diagnostic': {}, 'terminal_protocol': PROTOCOL}
            base = {'initializations': 'init'}
            data = {'case': {'output': 'out', 'full_bank_plan': 'base', 'state_name': 'source',
                             'maximum_case_cpu_seconds': 30}, 'base': base,
                    'init': {'rows': [{'task_id': 'task', 'candidate_bank_indices': [0, 1],
                                      'paired_observations': [[.3, -.2], [.4, -.1]], 'tau': 0.}]}}
            bank = [[i] for i in range(73)]
            def reached(provider, candidates, *args, **kwargs):
                self.assertEqual(candidates, [(i,) for i in range(73)])
                raise RuntimeError('native shortlist reached')
            with patch.object(experiment, 'ROOT', root), patch.object(experiment, 'read', side_effect=data.__getitem__), \
                 patch.object(experiment, 'context', return_value=({}, {'tasks': [{'task_id': 'task'}]}, [state], {}, {}, {}, bank)), \
                 patch.object(experiment, 'restore_full_prior', return_value=(SimpleNamespace(update_raw=lambda *args: None), {})), \
                 patch.object(experiment, 'restricted_snapshot', return_value={}), \
                 patch.object(experiment, 'check_replayed_snapshot'), patch.object(experiment, 'problem_for'), \
                 patch.object(experiment, 'no_outcomes'), patch.object(experiment, 'terminal_from_initialization', side_effect=reached):
                with self.assertRaisesRegex(RuntimeError, 'native shortlist reached'): experiment.run_case('case')


if __name__ == '__main__': unittest.main()
