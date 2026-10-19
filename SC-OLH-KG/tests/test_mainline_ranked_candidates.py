"""Check shared source-only nominations and chunked marginal scoring."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from performance import mainline_ranked_candidates as ranked
from performance.mainline_dynamic_candidates import _source_templates
from performance.mainline_moment_policy import terminal_action
from representation.meta_prior import LearnedMetaPrior


class PublicBox:
    d, L = 1000, 100

    def continuous_to_int(self, values):
        return tuple(np.clip(np.round(np.asarray(values) * self.L), 0, self.L).astype(int))

    def sample_random(self, rng):
        return tuple(rng.integers(0, self.L + 1, self.d))

    def __getattr__(self, name):
        raise AssertionError(f'Non-public target API requested: {name}')


def synthetic_state(coordinates):
    x = np.asarray(coordinates['mean'])[:, 0]
    count = len(x)
    mean = np.column_stack([.2 + .003 * x, np.full(count, np.log(.0016)),
                            -.2 + .002 * x, np.full(count, np.log(.0016))])
    cov = np.zeros((count, 4, count, 4))
    for i in range(count):
        cov[i, :, i, :] = np.diag([.002, .01, .003, .01])
    return {'mean': mean, 'cov': cov, 'noise': np.full((count, 2), .0016)}


class RankedCandidateTests(unittest.TestCase):
    def test_balanced_unique_quota_skips_existing_and_other_arm_picks(self):
        source, gp = ranked.ARMS
        scores = {source: [100, 90, 80, 70, 60, 50, 40, 30, 20, 10],
                  gp: [100, 10, 90, 80, 80, 60, 50, 40, 30, 20]}
        nominations = ranked._balanced_nominations([(i,) for i in range(10)], scores, {(0,), (9,)})
        self.assertEqual([i for i, _, _ in nominations], [1, 2, 3, 4, 5, 6])
        self.assertEqual([arm for _, arm, _ in nominations], list(ranked.ARMS) * 3)
        self.assertEqual(scores, {source: [100, 90, 80, 70, 60, 50, 40, 30, 20, 10],
                                  gp: [100, 10, 90, 80, 80, 60, 50, 40, 30, 20]})

    def test_marginal_chunks_match_joint_utility_and_reuse_only_matching_cache(self):
        coordinates = {'mean': [[float(i), 0, 0, 0] for i in range(21)],
                       'risk': [[0.] * 10 for _ in range(21)]}
        protocol = {'objective_reference': 1., 'chance_alpha': .05, 'familywise_delta': .05}
        reuse = {'gp_fit': 'frozen_gp.json'}
        pilot = {'source_connection_plan': 'frozen_source.json', 'training_tasks': [2, 3, 4, 5]}
        sizes = []

        def restore(_reuse, _pilot, layout, arm, block):
            sizes.append(len(block['mean']))
            self.assertEqual(layout, '5')
            return synthetic_state(block), {'paid_episodes': 0, 'source_law_optimizations': 0}

        expected = terminal_action(synthetic_state(coordinates), 358, protocol)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(ranked, 'restore_cross_dimension_prior', side_effect=restore) as restored, \
                patch.object(ranked, 'moment_state', side_effect=lambda state: state):
            kwargs = dict(reuse=reuse, pilot=pilot, protocol=protocol, cache_dir=directory,
                          round_number=1, source_seed=123)
            actual = ranked._score_pool(coordinates, **kwargs)
            self.assertEqual(sizes, [16, 16, 5, 5])
            for arm in ranked.ARMS:
                np.testing.assert_allclose(actual['scores'][arm], expected['utilities'], atol=1e-14)
                np.testing.assert_allclose(actual['powers'][arm], expected['certificate_powers'], atol=1e-14)
            restored.reset_mock()
            replay = ranked._score_pool(coordinates, **kwargs)
            restored.assert_not_called()
            self.assertEqual(replay['scores'], actual['scores'])
            self.assertTrue(all(c['reused'] for c in replay['chunks']))
            changed = dict(coordinates, mean=[[42., 0, 0, 0]] + coordinates['mean'][1:])
            with self.assertRaisesRegex(ValueError, 'cache inputs differ'):
                ranked._score_pool(changed, **kwargs)

    def test_real_public_pools_keep_prefix_seeds_and_source_only_ranked_provenance(self):
        root = Path(__file__).resolve().parents[1]
        original = json.loads((root / 'paper_artifacts/mainline_d1000_two_round_v1/original_bank.json').read_text())['bank']

        def coordinates(layout, problem, bank, plan):
            self.assertEqual(len(bank), 512)
            return {'mean': [[float(i), 0, 0, 0] for i in range(len(bank))],
                    'risk': [[0.] * 10 for _ in bank], 'diagnostics': {'new_simulator_calls': 0}}

        def restore(_reuse, _pilot, layout, arm, block):
            return synthetic_state(block), {'paid_episodes': 0, 'source_law_optimizations': 0}

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(LearnedMetaPrior, 'fit_from_source_problems', side_effect=AssertionError('new fit')), \
                patch.object(ranked, 'query_coordinates', side_effect=coordinates), \
                patch.object(ranked, 'restore_cross_dimension_prior', side_effect=restore) as restored, \
                patch.object(ranked, 'moment_state', side_effect=lambda state: state):
            kwargs = dict(reuse={'gp_fit': 'frozen_gp.json'},
                          pilot={'source_connection_plan': 'frozen_source.json', 'training_tasks': [2, 3, 4, 5]},
                          protocol={'objective_reference': 1., 'chance_alpha': .05, 'familywise_delta': .05},
                          cache_dir=directory)
            result = ranked.build_ranked_shared_bank(PublicBox(), original, **kwargs)
            self.assertEqual(restored.call_count, 64)
            restored.reset_mock()
            repeated = ranked.build_ranked_shared_bank(PublicBox(), original, **kwargs)
            restored.assert_not_called()
            self.assertEqual(result, repeated)
        bank, meta = result['bank'], result['dynamic_candidates']
        self.assertEqual(bank[:73], original)
        self.assertEqual(meta['round_end_indices'], [89, 104])
        self.assertEqual(len(set(map(tuple, bank))), 104)
        self.assertTrue(np.all((np.asarray(bank) >= 0) & (np.asarray(bank) <= 100)))
        self.assertEqual([r['source_pool_seed'] for r in meta['rounds']], [1158395989, 4194981999])
        templates, _, _ = _source_templates()
        for r in meta['rounds']:
            prior = LearnedMetaPrior(seed=r['source_pool_seed'])
            prior.alignment_profile_templates = templates
            pool = prior._alignment_inverse_pool(PublicBox(), np.random.default_rng(r['source_pool_seed']), 256)
            selected = [(p, o) for p, o in zip(bank, meta['origins'])
                        if o['birth_round'] == r['round'] and o['route'] == 'source_ranked_inverse_pool']
            self.assertEqual([o['nominated_by'] for _, o in selected], list(ranked.ARMS) * 3)
            self.assertEqual(len(r['ranking_scores']), 256)
            for point, origin in selected:
                self.assertEqual(tuple(point), tuple(pool[origin['source_pool_index']]))


if __name__ == '__main__':
    unittest.main()
