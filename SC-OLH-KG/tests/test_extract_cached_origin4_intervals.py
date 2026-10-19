"""Local software fixtures only; no certificate or predictive evaluation."""
import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from performance import extract_cached_origin4_intervals as extractor


class CachedOrigin4IntervalsTest(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.spec = json.loads((root / extractor.PLAN).read_text())
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        folder = Path(self.temp.name)
        sources = dict(current_fixed=str(folder / 'current.json'),
                       gauss_shards=[str(folder / f'gauss{s}.json') for s in (0, 1)],
                       r2048_risk_orders=str(folder / 'r2048.json'))
        self.spec['missing_sources'] = sources
        self.spec['expected_current_risk_sum'] = 0.73
        self.spec['expected_gauss_risk_sum'] = 0.74
        rows = []
        for row in self.spec['current_metadata']:
            control = 0.95 <= row['cdf'][0] < 0.99
            rows.append(dict(record=row['record'], policy=63,
                             fixed=dict(raw=0.25 if control else 0.75,
                                        raw64_128_difference=1e-6, raw_evidence_error=2e-12)))
        # The real task also contains other policies and the other origin.
        rows.extend([dict(record='other_policy', policy=62),
                     dict(record='other_origin', policy=63, fixed=dict(raw=0.9))])
        self.write(sources['current_fixed'], dict(complete=True, rows=rows))
        for shard, path in enumerate(sources['gauss_shards']):
            bank = dict(probabilities=[], weights=[], values=[], inner_orders=[])
            for i in range(1024):
                early = i < 512
                bank['probabilities'].append([0.1 if early else 0.94, (shard * 1024 + i + 0.5) / 2048])
                bank['weights'].append((0.2 if early else 0.8) / 1024)
                bank['values'].append(0.1 if early else 0.9)
                bank['inner_orders'].append(64 if early else 128)
            self.write(path, bank)
        self.write(sources['r2048_risk_orders'], dict(complete=True,
                   orders=[64] * 1536 + [128] * 512, all_inner_rules_passed=True))

    @staticmethod
    def write(path, value):
        Path(path).write_text(json.dumps(value))

    def test_weights_orders_and_empty_gauss_buckets(self):
        result = extractor.summarize(self.spec)
        groups = result['intervals']
        self.assertEqual([r['current']['nodes'] for r in groups], [16, 16, 16, 16, 32, 16])
        self.assertEqual(sum(r['current']['nodes'] for r in groups if r['target']), 80)
        for row, mass in zip(groups, [0.8, 0.075, 0.0375, 0.0375, 0.04, 0.01]):
            self.assertAlmostEqual(row['current']['weight_sum'], mass)
        # Old node-mask masses deliberately differ from registered interval widths.
        self.assertAlmostEqual(groups[0]['old_gauss_node_mask']['weight_sum'], 0.2)
        self.assertAlmostEqual(groups[3]['old_gauss_node_mask']['weight_sum'], 0.8)
        self.assertEqual(groups[2]['old_gauss_node_mask']['nodes'], 0)
        self.assertEqual(groups[2]['old_gauss_node_mask']['risk_weighted_sum'], 0)
        self.assertEqual(groups[2]['old_gauss_node_mask']['order_counts'], {})
        self.assertAlmostEqual(result['current_risk_sum'], 0.96 * 0.75 + 0.04 * 0.25)
        self.assertAlmostEqual(result['gauss_risk_sum'], 0.2 * 0.1 + 0.8 * 0.9)
        self.assertEqual(groups[0]['old_gauss_node_mask']['order_counts'], {'64': 1024})
        self.assertAlmostEqual(groups[0]['old_gauss_node_mask']['order_weight_sums']['64'], 0.2)
        self.assertEqual(result['r2048_eta_order_counts'], {64: 1536, 128: 512})
        self.assertEqual(result['r2048_eta_order_weight_sums'], {'64': 0.75, '128': 0.25})
        self.assertEqual(result['eta_order_semantics']['current'], 'fixed128_with64_128_control')
        self.assertEqual(result['eta_order_semantics']['old_gauss_node_mask'], 'adaptive_first_passing')
        self.assertFalse(result['matched_interval_integrals_available'])
        self.assertFalse(result['same_node_adaptive_vs128_differences_available'])
        self.assertFalse(result['weights_renormalized'])

    def test_internal_boundaries_belong_to_the_right_bucket(self):
        path = self.spec['missing_sources']['gauss_shards'][0]
        bank = json.loads(Path(path).read_text())
        for row, boundary in zip(bank['probabilities'], [0.8, 0.875, 0.9125, 0.95, 0.99]):
            row[0] = boundary
        self.write(path, bank)
        result = extractor.summarize(self.spec)
        self.assertEqual([r['old_gauss_node_mask']['nodes'] for r in result['intervals']],
                         [1019, 1, 1, 1025, 1, 1])

    def test_current_interval_mass_is_checked_beyond_global_closure(self):
        spec = copy.deepcopy(self.spec)
        for row in spec['current_metadata']:
            if row['cdf'][0] < 0.8:
                row['outer_weight'] = 0.075 / 16
            elif row['cdf'][0] < 0.875:
                row['outer_weight'] = 0.8 / 16
        # Total mass and total synthetic risk stay unchanged; interval masses do not.
        with self.assertRaises(AssertionError):
            extractor.summarize(spec)

    def test_missing_shard_reports_incomplete_and_cpu(self):
        Path(self.spec['missing_sources']['gauss_shards'][1]).unlink()
        plan = Path(self.temp.name) / 'plan.json'
        self.write(plan, self.spec)
        output = io.StringIO()
        with patch('sys.argv', ['extractor', '--plan', str(plan)]), redirect_stdout(output):
            extractor.main()
        result = json.loads(output.getvalue())
        self.assertFalse(result['complete'])
        self.assertEqual(result['error_type'], 'FileNotFoundError')
        self.assertGreaterEqual(result['reader_process_cpu_seconds'], 0)


if __name__ == '__main__':
    unittest.main()
