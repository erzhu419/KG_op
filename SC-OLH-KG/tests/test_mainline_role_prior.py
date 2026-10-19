"""Compact role-prior replay must equal a direct finite-policy raw filter."""

import unittest
from types import SimpleNamespace

import numpy as np

from algorithms.single_olhkg import SingleOLHKGAlgorithm
from performance.benchmark_lodo_meta_prior import build_scalarized_problem
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.diagnose_mainline_initialization import marginal
from performance.mainline_moment_policy import moment_state
from performance.mainline_role_prior import (
    role_inputs, role_moment_state, role_prior, role_trajectory,
)
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.joint_response_mixture import JointResponseMixture


class LinearBasis:
    def transform(self, values):
        return np.asarray(values)[:, :2]


class PublicEncoder:
    def transform(self, values):
        return np.asarray(values)[:, :10]


class RoleBasis:
    def __init__(self, tau):
        self.tau = tau

    def source_parametric_prior(self):
        return dict(mean=np.array([self.tau+.3, .12, -.08]),
                    covariance=np.diag([.08, .04, .03]),
                    deviation_variance=.012,
                    diagnostics=dict(source_record_count=12))

    def matrix(self, bank):
        points = np.asarray(bank)/100
        return np.column_stack([np.ones(len(bank)), points[:, 0], points[:, -1]])


def source_pair():
    law = dict(task=np.eye(3)*.01, white=.02)
    risk = dict(task=np.eye(3)*.005, white=.01)
    return dict(law=law, risk=risk,
                mean=np.array([[.3, .1, -.05], [.32, .08, -.03], [.28, .12, -.02]]),
                covariance=np.tile(np.diag([.06, .02, .03]), (3, 1, 1)),
                eta_mean=np.array([[-4., .1, .1], [-3.8, .08, .1], [-4.2, .12, .08]]),
                eta_covariance=np.diag([.03, .02, .01]),
                log_weights=np.log([.2, .5, .3]))


def full_field(record, D, E, role=None, tau=0.):
    """Build the finite-bank law directly, without compact discrepancy slots."""
    count, n = len(record['mean']), len(D)
    if role is None:
        function_mean = record['mean']@D.T
        function_cov = np.einsum('np,kpq,mq->knm', D,
            record['covariance']+record['law']['task'], D)
        function_white = record['law']['white']
    else:
        # Center raw g after projection, independently of role_inputs' intercept.
        function_mean = np.broadcast_to(D@role['mean']-tau, (count, n))
        function_cov = np.broadcast_to(D@role['covariance']@D.T, (count, n, n))
        function_white = role['deviation_variance']
    means = np.stack([function_mean, record['eta_mean']@E.T], axis=2)
    covariance = np.zeros((count, n, 2, n, 2))
    covariance[:, :, 0, :, 0] = function_cov+function_white*np.eye(n)
    covariance[:, :, 1, :, 1] = E@(record['eta_covariance']+record['risk']['task'])@E.T
    covariance[:, :, 1, :, 1] += record['risk']['white']*np.eye(n)
    field = JointFunctionRiskPosterior.from_moments(means[0], covariance[0])
    field.means = means.reshape(count, 2*n).copy()
    field.covariances = covariance.reshape(count, 2*n, 2*n).copy()
    field.log_weights = record['log_weights'].copy()
    field.component_pairs = [(0, 0)]*count
    return field


class RolePriorProjectionTests(unittest.TestCase):
    def test_nonzero_tau_raw_updates_match_full_policy_posterior(self):
        problem = no_outcomes(build_scalarized_problem(
            'InventorySupplyChain', 6, 100, .04, .05, [.5, .5]))
        problem.tau = .37
        basis = RoleBasis(problem.tau)
        engine = SingleOLHKGAlgorithm.__new__(SingleOLHKGAlgorithm)
        engine.problem = problem
        engine.config = SimpleNamespace(source_constraint_mean_coefficient_prior=True,
            source_constraint_mean_hyperlaw_mode='single_gaussian_draw',
            source_constraint_mean_deviation_mode='latent_shared')
        engine.gpr = [None, SimpleNamespace(p=3, basis_map=basis, basis_matrix=basis.matrix)]
        identity = 'spectral/head_specific/observed_task'
        bundle = dict(mean_center=np.zeros(10), mean_scale=np.ones(10),
            risk_center=np.zeros(10), risk_scale=np.ones(10), encoder=PublicEncoder(),
            maps={('spectral', 'head_specific', head): LinearBasis() for head in (0, 2, 3)},
            identities=[identity], components=[(source_pair(), source_pair())],
            log_weights=np.zeros(1))
        bank = [tuple([value]*6) for value in (20, 40, 60)]
        rows = [dict(point=bank[0], raw=[.27, problem.tau+.31]),
                dict(point=bank[1], raw=[.29, problem.tau+.28])]
        replacement, designs = role_inputs(bundle, engine, bank)
        source_f, source_g = bundle['components'][0]
        role = role_prior(engine)
        (D, E), (_, G) = designs[0]
        direct = JointResponseMixture([identity], [(
            full_field(source_f, D, E),
            full_field(source_g, basis.matrix(bank), G, role, problem.tau),
        )])
        self.assertIs(replacement['components'][0][0], source_f)
        for key in ('eta_mean', 'eta_covariance', 'risk', 'log_weights'):
            self.assertIs(replacement['components'][0][1][key], source_g[key])
        snapshots, scores, _ = role_trajectory(bundle, engine, bank, rows, batch_size=2)
        for prefix in range(3):
            expected = moment_state(direct)
            actual = role_moment_state(bundle, engine, bank, rows[:prefix], component_batch_size=2)
            for key in ('mean', 'cov', 'noise'):
                np.testing.assert_allclose(actual[key], expected[key], atol=2e-11, rtol=2e-11)
            for key, value in marginal(expected).items():
                np.testing.assert_allclose(snapshots[prefix][key], value, atol=2e-11, rtol=2e-11)
            self.assertEqual(actual['paid_episodes'], prefix)
            if prefix < 2:
                raw = rows[prefix]['raw']
                update = direct.update_raw(prefix, [raw[0], raw[1]-problem.tau])
                np.testing.assert_allclose(scores[prefix]['heads'], update['head_log_scores'][0],
                                           atol=2e-11, rtol=2e-11)
                self.assertAlmostEqual(scores[prefix]['joint'], update['mixture_log_score'], places=10)


if __name__ == '__main__':
    unittest.main()
