import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from performance.check_historical_gauss_risk_tail import summarize
from performance.check_historical_independent_queries import registered_queries, cohort_status
from performance.diagnose_historical_independent_query_axes import terminal_axes, refinement_axes
import performance.refine_historical_independent_query_axes as refinement
from performance.refine_historical_remaining_query_failures import positive_risk_tail


def fixture():
    rule = dict(seed=714, raw_powers=[10, 12], objective_prefixes=[16, 32],
        risk_complement_prefixes=[2, 4], risk_tail_orders=[16, 32], low_constraint_probability=.125,
        maximum_numerical_adaptation_difference=1e-4, maximum_fixed_action_tower_residual=1e-4,
        proposal_origin=0, predictive_origins=[0])
    objective = {str(n): {'positions': [512], 'weights': [1.]} for n in [16, 32]}
    risk = {str(n): {'positions': [256 if n == 16 else 272], 'weights': [.125]} for n in [16, 32]}
    def record(head, position):
        statistics = {10: [.9, .8] if head == 0 else [.3, .7],
            12: [.9, .8] if head == 0 else [.31, .72],
            15: [.9, .8] if head == 0 else [.3, .7],
            16: [.9, .8] if head == 0 else [.31, .72]}
        return {'position': position, 'head_log_scores': [0.], 'raw': {str(p): {
            'statistics': np.asarray(v).reshape(1, 2, 1).tolist(), 'costs': [[10., 10.]]}
            for p, v in statistics.items()}}
    heads = [[record(0, 512)], [record(1, p) for p in [0, 1, 2, 3, 256, 272]]]
    baseline = [.2, .19, 0]
    stage = summarize(heads, risk, [1.], rule, objective_rules=objective, baseline=baseline)
    stage['raw_powers'] = [10, 12]
    case = {'arm': 'source_common_task_rbf', 'source_weights': [1.], 'baseline': baseline, 'stages': [stage]}
    return case, {'objective_rules': objective, 'risk_rules': risk}, heads, rule


class CachedAxisTests(unittest.TestCase):
    def test_cached_composition_localizes_risk_without_native_integration(self):
        case, bank, heads, rule = fixture(); before = copy.deepcopy(heads)
        split = terminal_axes(case, bank, heads, rule)
        self.assertEqual(split['objective_only_difference'], 0.)
        self.assertAlmostEqual(split['risk_only_difference'], .007)
        self.assertEqual(refinement_axes(case['stages'][0]['numerical_changes'], split, 1e-4), ['terminal_risk'])
        self.assertEqual(heads, before)

    def test_changed_cached_resolution_stops_before_choosing_a_refinement(self):
        case, bank, heads, rule = fixture()
        case['stages'][0]['numerical_changes']['terminal_absolute_contrast'] += .01
        with self.assertRaises(ValueError):
            terminal_axes(case, bank, heads, rule)

    def test_objective_and_body_failures_are_retained_as_other_axes(self):
        changes = dict(terminal_absolute_contrast=.003, objective_prefix=.002,
            risk_complement_prefix=.002, risk_tail_order=0.)
        split = {'objective_only_difference': .002, 'risk_only_difference': 1e-9}
        self.assertEqual(refinement_axes(changes, split, 1e-4),
            ['terminal_objective', 'objective_prefix', 'risk_complement_prefix'])

    def test_risk_only_extension_still_checks_the_complete_objective_risk_contrast(self):
        case, bank, heads, rule = fixture()
        heads[0][0]['raw']['12']['statistics'] = np.array([.85, .82]).reshape(1, 2, 1).tolist()
        # Objective has only its original two powers; no new objective integration exists.
        heads[0][0]['raw'] = {k: v for k, v in heads[0][0]['raw'].items() if k in ['10', '12']}
        result = summarize(heads, bank['risk_rules'], [1.], dict(rule, raw_powers=[15, 16],
            objective_raw_powers=[10, 12]), objective_rules=bank['objective_rules'], baseline=case['baseline'])
        fine = result['rows'][-1]
        # (.82*.72-.85*.31) - (.8*.7-.9*.3) = .0369.
        self.assertAlmostEqual(fine['weighted_absolute_terminal_contrast_difference'], .0369)
        self.assertEqual(fine['objective_raw_draw_power'], 12)
        self.assertEqual(fine['raw_draw_power'], 16)
        self.assertFalse(result['numerical_gates_passed'])


class HistoricalReuseTests(unittest.TestCase):
    def test_positive_tail_keeps_model_query_original_mass_and_unused_positions(self):
        fields = [[object(), object()] for _ in range(8)]
        model = SimpleNamespace(components=fields)
        class CDF:
            def inverse(self, values):
                return values+3
            def adaptive(self, values):
                return values-3, 0.
        with patch('performance.refine_historical_remaining_query_failures.RawPredictiveCDF', return_value=CDF()) as cdf:
            added = positive_risk_tail(model, 6, 4, {'low_constraint_probability': .125,
                'maximum_inverse_cdf_error': 1e-8})
            cdf.assert_called_once_with(fields[4][1], 6)
        self.assertEqual(added['positions'], list(range(304, 368)))
        self.assertAlmostEqual(sum(added['weights']), .125)
        self.assertGreater(min(added['weights']), 0.)

    def test_merged_refinement_preserves_every_task_and_accepted_query(self):
        states = [{'name': 'fresh', 'task_id': 'new_task', 'layout': '5', 'arm': 'source', 'queries': [5, 4]}]
        old = [dict(d, processing_complete=True, adaptation_diagnostics_passed=d['query'] == 5)
               for d in registered_queries(states)]
        refined = dict(old[1], adaptation_diagnostics_passed=True)
        merged = refinement.merge_cases(states, old, [refined])
        self.assertIs(merged[0], old[0]); self.assertIs(merged[1], refined)
        self.assertTrue(cohort_status(states, merged)['all_model_states_ready'])
        with self.assertRaises(ValueError):
            refinement.merge_cases(states, old, [dict(refined, task_id='development')])
        with self.assertRaises(ValueError):
            refinement.merge_cases(states, old[:1], [])

    def test_live_initial_sweep_blocks_selective_native_work(self):
        data = {'plan': {'parent_plan': 'parent'}, 'parent': {'pilot_plan': 'pilot',
            'numerical_overrides': {}, 'output': 'cohort', 'preflight': 'preflight'},
            'pilot': {'numerical_plan': 'numerical'}, 'numerical': {},
            'cohort/results.json': {'all_initial_cases_complete': False}, 'preflight': {'states': []}}
        with patch.object(refinement, 'read', side_effect=data.__getitem__), \
                patch.object(refinement, 'restore_target_models') as restore, \
                patch.object(refinement, 'evaluate_head_records') as integrate:
            with self.assertRaises(ValueError):
                refinement.run('plan')
            restore.assert_not_called(); integrate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
