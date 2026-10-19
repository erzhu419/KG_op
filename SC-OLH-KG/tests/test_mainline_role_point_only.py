"""The control must suppress cross-policy learning without changing own updates."""

import unittest
from types import SimpleNamespace

import numpy as np

from algorithms.single_olhkg import SingleOLHKGAlgorithm
from performance.benchmark_lodo_meta_prior import build_scalarized_problem
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.mainline_role_point_only import point_only_moment_state
from performance.mainline_role_prior import role_moment_state
from tests.test_mainline_role_prior import LinearBasis, PublicEncoder, RoleBasis, source_pair


class RolePointOnlyTests(unittest.TestCase):
    def test_local_raw_parity_cross_suppression_and_incremental_cache(self):
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
        alternative = (source_pair(), source_pair())
        alternative[0]['mean'] = alternative[0]['mean'] + .6
        alternative[1]['eta_mean'] = alternative[1]['eta_mean'] + .8
        bundle = dict(mean_center=np.zeros(10), mean_scale=np.ones(10),
            risk_center=np.zeros(10), risk_scale=np.ones(10), encoder=PublicEncoder(),
            maps={('spectral', 'head_specific', head): LinearBasis() for head in (0, 2, 3)},
            identities=['spectral/head_specific/observed_task',
                        'spectral/head_specific/deconvolved_task'],
            components=[(source_pair(), source_pair()), alternative],
            log_weights=np.log([.6, .4]))
        bank = [tuple([value] * 6) for value in (20, 40, 60)]
        rows = [dict(point=bank[0], raw=[.27, problem.tau + .31]),
                dict(point=bank[1], raw=[1.1, problem.tau - .4]),
                dict(point=bank[0], raw=[.28, problem.tau + .32])]
        cache = {}
        source = role_moment_state(bundle, engine, bank, component_batch_size=2)
        initial = point_only_moment_state(bundle, engine, bank, cache=cache,
                                          component_batch_size=2)
        for key in ('mean', 'cov', 'noise'):
            np.testing.assert_allclose(initial[key], source[key], rtol=0, atol=0)
        after_one = point_only_moment_state(bundle, engine, bank, rows[:1],
                                            cache=cache, component_batch_size=2)
        after_two = point_only_moment_state(bundle, engine, bank, rows[:2],
                                            cache=cache, component_batch_size=2)
        # The unrelated second observation cannot affect point zero at any head.
        for key in ('mean', 'noise', 'policy_weights'):
            np.testing.assert_array_equal(after_one[key][0], after_two[key][0])
        np.testing.assert_array_equal(after_one['cov'][0, :, 0, :],
                                      after_two['cov'][0, :, 0, :])
        final = point_only_moment_state(bundle, engine, bank, rows, cache=cache,
                                       component_batch_size=2)
        self.assertEqual(final['cost'], dict(source0_projections=0,
            own_model_restorations=0, own_raw_updates=1))
        for index in (0, 1):
            own = [row for row in rows if row['point'] == bank[index]]
            expected = role_moment_state(bundle, engine, [bank[index]], own,
                                         component_batch_size=2)
            for key in ('mean', 'noise'):
                np.testing.assert_allclose(final[key][index], expected[key][0],
                                           atol=2e-11, rtol=2e-11)
            np.testing.assert_allclose(final['policy_weights'][index], expected['weights'],
                                       atol=2e-11, rtol=2e-11)
            np.testing.assert_allclose(np.diag(final['cov'][index, :, index, :]),
                np.diag(expected['cov'].reshape(4, 4)), atol=2e-11, rtol=2e-11)
        # No mean, noise, variance or identity evidence reaches the third point.
        for key in ('mean', 'noise'):
            np.testing.assert_array_equal(final[key][2], source[key][2])
        np.testing.assert_array_equal(final['cov'][2, :, 2, :], source['cov'][2, :, 2, :])
        np.testing.assert_array_equal(final['policy_weights'][2], source['weights'])
        self.assertGreater(abs(final['policy_weights'][0, 0]
                               - final['policy_weights'][1, 0]), .01)
        # The TS control keeps the original correlations and remains PSD.
        matrices = [state['cov'].reshape(12, 12) for state in (source, final)]
        correlations = [matrix / np.sqrt(np.outer(np.diag(matrix), np.diag(matrix)))
                        for matrix in matrices]
        np.testing.assert_allclose(correlations[0], correlations[1], atol=2e-12)
        self.assertGreater(np.linalg.eigvalsh(matrices[1]).min(), -2e-12)
        repeated = point_only_moment_state(bundle, engine, bank, rows, cache=cache,
                                           component_batch_size=2)
        self.assertEqual(repeated['cost'], dict(source0_projections=0,
            own_model_restorations=0, own_raw_updates=0))
        self.assertEqual(repeated['roundoff_events'], [])
        reordered = point_only_moment_state(bundle, engine, list(reversed(bank)), rows,
                                            cache=cache, component_batch_size=2)
        self.assertEqual(reordered['cost'], dict(source0_projections=1,
            own_model_restorations=0, own_raw_updates=0))
        np.testing.assert_allclose(reordered['mean'][::-1], final['mean'], atol=2e-12)


if __name__ == '__main__':
    unittest.main()
