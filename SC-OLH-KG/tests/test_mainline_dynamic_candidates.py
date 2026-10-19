"""Validate real source/public-only proposal routes without any simulator."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from performance.mainline_dynamic_candidates import build_shared_dynamic_bank, _source_templates
from representation.meta_prior import LearnedMetaPrior


class PublicBox:
    d, L = 1000, 100

    def continuous_to_int(self, values):
        return tuple(np.clip(np.round(np.asarray(values) * self.L), 0, self.L).astype(int))

    def sample_random(self, rng):
        return tuple(rng.integers(0, self.L + 1, self.d))

    def __getattr__(self, name):
        raise AssertionError(f'Generator requested non-public target API: {name}')


class DynamicCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.original = json.loads((root / 'paper_artifacts/mainline_d1000_two_round_v1/original_bank.json').read_text())['bank']

    def test_real_routes_preserve_prefix_and_never_fit_or_query_target(self):
        with patch.object(LearnedMetaPrior, 'fit_from_source_problems', side_effect=AssertionError('new fit')):
            result = build_shared_dynamic_bank(PublicBox(), self.original, seed=2026101509)
        bank, meta = result['bank'], result['dynamic_candidates']
        self.assertEqual(bank[:73], self.original)
        self.assertEqual(meta['round_end_indices'], [89, 104])
        self.assertEqual(len({tuple(p) for p in bank}), len(bank))
        self.assertTrue(np.all((np.asarray(bank) >= 0) & (np.asarray(bank) <= 100)))
        self.assertEqual(meta['source']['proposal_source_rows'], 32)
        for r in meta['rounds']:
            self.assertEqual(r['added_by_route']['source_profile_refinement'], 6)
        self.assertEqual(meta['rounds'][1]['duplicates_by_route']['canonical_sobol'], 1)
        for i, origin in enumerate(meta['origins'][73:], 73):
            self.assertEqual(origin['birth_round'], 1 if i < 89 else 2)

    def test_source_points_are_actual_original_inverse_pool_outputs(self):
        result = build_shared_dynamic_bank(PublicBox(), self.original, seed=2026101509)
        again = build_shared_dynamic_bank(PublicBox(), self.original, seed=2026101509)
        self.assertEqual(result, again)
        templates, origins, _ = _source_templates()
        for trace in result['dynamic_candidates']['rounds']:
            prior = LearnedMetaPrior(seed=trace['source_pool_seed'])
            prior.alignment_profile_templates = templates
            pool = prior._alignment_inverse_pool(PublicBox(), np.random.default_rng(trace['source_pool_seed']), 256)
            for point, origin in zip(result['bank'], result['dynamic_candidates']['origins']):
                if origin['birth_round'] == trace['round'] and origin['route'] == 'source_profile_refinement':
                    index = origin['source_pool_index']
                    self.assertEqual(tuple(point), tuple(pool[index]))
                    self.assertEqual(origin['source_task'], origins[index % 32]['source_task'])
                    self.assertIn(origin['source_task'], ['related_inventory_02', 'related_inventory_03', 'related_inventory_04', 'related_inventory_05'])


if __name__ == '__main__':
    unittest.main()
