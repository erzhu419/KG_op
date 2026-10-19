import unittest

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve
from scipy.special import logsumexp

from performance.benchmark_lodo_meta_prior import build_scalarized_problem
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.mainline_legacy_source_models import (
    DEFAULTS, _condition_pair, _joint_risk_fit, _stream_head, restore_model, restore_moment_state,
)
from performance.mainline_moment_policy import moment_state
from performance.prepare_legacy_f_variance_reference import VarianceReferencePosterior


class LinearBasis:
    def transform(self, values):
        return np.asarray(values)[:, :2]


class PublicEncoder:
    def transform(self, values):
        return np.asarray(values)[:, :10]


def law(mean, design, common, task, white=.02):
    return dict(mean=mean, common=common, task=task, white=white,
        source_design=design, source_mean=design@mean,
        source_covariance=design@common@design.T
            + block_diag(*[d@task@d.T for d in np.split(design, 2)])
            + white*np.eye(len(design)))


class LegacySourceModelTests(unittest.TestCase):
    def setUp(self):
        self.design = np.array([[1., -.2, .3], [1., .1, .4],
                                [1., -.1, -.2], [1., .2, -.3]])
        self.f = law(np.array([.3, .1, -.05]), self.design, np.diag([.1, .05, .02]), np.eye(3)*.01)
        self.eta = law(np.array([-4., .1, .1]), self.design, np.eye(3)*.03, np.eye(3)*.01)
        self.states = self.eta['source_mean']+np.array([[-.2, .1, -.1, .2],
            [.2, -.1, .1, -.2], [.1, .2, -.2, -.1], [-.1, -.2, .2, .1]])
        self.labels = np.array([.26, .29, .35, .30]); self.variances = np.full(4, .02)

    def test_native_compact_batch64_and8_preserve_evidence_moments_and_noise(self):
        record, _ = _condition_pair(self.f, self.eta, self.labels, self.variances,
                                    self.states, -1.37)
        # 68 nonuniform components cross both chunk boundaries, with a tail.
        for key in ('mean', 'covariance', 'eta_mean', 'log_weights'):
            record[key] = np.concatenate([record[key]]*17, axis=0)
        record['log_weights'] -= np.log(17)
        design = np.array([[1., .2, .1], [1., -.1, .3], [1., .4, -.2]])
        history = [(0, [.27, .31]), (1, [.29, .28]), (0, [.33, .32])]
        small = _stream_head(record, design, design, history, 0, 8)
        large = _stream_head(record, design, design, history, 0, 64)
        for actual, expected in zip(large[:4], small[:4]):
            np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-11)

    def test_joint_public_basis_uses_one_source_likelihood_and_task_cross_is_absent(self):
        public = np.arange(16, dtype=float).reshape(4, 4)/20
        public[:, 2] += np.array([.2, -.1, .3, -.2])
        augmented, diagnostic = _joint_risk_fit(self.f, public, [2, 2], self.states,
            self.labels, self.variances, {**DEFAULTS, 'risk_max_iterations': 12})
        direct = VarianceReferencePosterior(augmented['source_mean'], augmented['source_covariance'],
                                           self.labels, self.variances)
        expected = -(logsumexp([direct.evaluate(e)[0] for e in self.states])-np.log(len(self.states)))
        self.assertAlmostEqual(expected, diagnostic['final_negative_log_integral'], places=9)
        D = augmented['source_design']
        np.testing.assert_allclose(augmented['source_covariance'][:2, 2:],
                                  D[:2]@augmented['common']@D[2:].T, atol=1e-13)

    def test_coefficient_projection_preserves_raw_source_evidence_and_replay_marginals(self):
        reference_logz = -1.37
        record, diagnostic = _condition_pair(self.f, self.eta, self.labels, self.variances,
                                             self.states, reference_logz)
        raw = VarianceReferencePosterior(self.f['source_mean'], self.f['source_covariance'],
                                        self.labels, self.variances)
        logs = np.array([raw.evaluate(e)[0] for e in self.states])
        np.testing.assert_allclose(record['log_weights'], logs-logsumexp(logs), atol=1e-13)
        self.assertAlmostEqual(diagnostic['log_source_evidence'],
                               reference_logz+logsumexp(logs)-np.log(4), places=12)
        D = np.array([[1., .2, .1], [1., -.1, .3]])
        cross = D@self.f['common']@self.design.T
        _, factor, alpha = raw.evaluate(self.states[0])
        np.testing.assert_allclose(D@record['mean'][0], D@self.f['mean']+cross@alpha, atol=1e-12)
        np.testing.assert_allclose(D@record['covariance'][0]@D.T,
            D@self.f['common']@D.T-cross@cho_solve(factor, cross.T), atol=1e-12)
        identities = ['/'.join((kind, support, treatment)) for kind in ('spectral', 'additive')
                      for support in ('head_specific', 'joint')
                      for treatment in ('observed_task', 'deconvolved_task')]
        bundle = dict(mean_center=np.zeros(10), mean_scale=np.ones(10),
            risk_center=np.zeros(10), risk_scale=np.ones(10), encoder=PublicEncoder(),
            maps={(kind, support, head): LinearBasis() for kind in ('spectral', 'additive')
                  for support in ('head_specific', 'joint') for head in (-1, 0, 1, 2, 3)},
            identities=identities,
            components=[(dict(record, mean=record['mean']+.02*j), record) for j in range(8)],
            log_weights=np.full(8, -np.log(8)))
        problem = no_outcomes(build_scalarized_problem('InventorySupplyChain', 6, 100, .04, .05, [.5, .5]))
        problem.tau = .37
        bank = [tuple([v]*6) for v in (20, 40, 60)]
        history = [{'point': bank[0], 'raw': [.27, .37+.31]},
                   {'point': bank[1], 'raw': [.29, .37+.28]},
                   {'point': bank[0], 'raw': [.33, .37+.32]}]
        short = restore_model(bundle, 'source_structural_mixture', problem, bank[:2])
        for candidate, value in [(0, [.27, .31]), (1, [.29, .28]), (0, [.33, .32])]:
            short.update_raw(candidate, value)
        full = restore_model(bundle, 'source_structural_mixture', problem, bank, history)
        a, ac = short.predict_joint(); b, bc = full.predict_joint()
        np.testing.assert_allclose(a, b[:2], atol=2e-11, rtol=2e-11)
        np.testing.assert_allclose(ac, bc[:2, :, :2, :], atol=2e-11, rtol=2e-11)
        np.testing.assert_allclose(short.weights, full.weights, atol=1e-13)
        self.assertEqual(full.paid_episodes, 3)
        self.assertEqual(len(full.components), 8)
        self.assertTrue(all(len(field.means) == 4 for pair in full.components for field in pair))
        for observations in ([], history):
            native = restore_model(bundle, 'source_structural_mixture', problem, bank, observations)
            expected = moment_state(native)
            streamed = restore_moment_state(bundle, 'source_structural_mixture', problem, bank,
                                             observations, component_batch_size=2)
            for key in ('mean', 'cov', 'noise'):
                np.testing.assert_allclose(streamed[key], expected[key], atol=2e-10, rtol=2e-10,
                                           err_msg=key)
            np.testing.assert_allclose(streamed['weights'], native.weights, atol=1e-12, rtol=1e-12)
            self.assertEqual(streamed['paid_episodes'], len(observations))


if __name__ == '__main__':
    unittest.main()
