"""Tiny saved-record fixtures: no simulator, RNG, models, or real cohort reads."""
import copy
import unittest

from performance.mainline_confirmation_terminal_attribution import (
    aggregate_terminals, analyze_pair, compact_task,
)


def fixture():
    metrics = dict(A=1, C=1, false_certificate=0, R=.8, G=.1, P=.1, L=0.)
    ref = dict(arm='test', history_arm='test', quality_reference_status='defined',
        quality_baseline_defined=True, best_initial_quality=.7, status='complete',
        shortlist=['initial:0', 'initial:1', 'neighbor:2'])
    ref.update({key: dict(metrics) for key in ('old', 'new', 'B', 'new_minus_old', 'third_challenge', 'prechange')})
    item = dict(source_case='saved/search', verification_seed=31,
        objective_incumbent_position=2,
        frozen_shortlist=[dict(point=[i], shortlist_role=role) for i, role in enumerate(
            ('primary', 'initial_incumbent', 'remaining'))])
    safety = [dict(candidate_index=i, certified=True, upper_margin=-.1,
                   verification_budget=n) for i, n in enumerate((80, 120, 120))]
    comparison = dict(simulation_calls=16, comparison_stream_tag=1329744452,
        challenger_dominates=True, sample_mean_difference=-.1,
        sample_std_difference=.01, standard_error=.003,
        one_sided_upper_confidence_bound=-.09)
    old = dict(deployed=[0], new_verification_calls=224, verification=dict(
        certified=True, attempts=safety[:2], objective_comparison=comparison,
        verification_budget=224))
    new = dict(deployed=[2], new_verification_calls=136, verification=dict(
        certified=True, attempts=safety, initial_stage_winner_index=0,
        objective_comparisons=[dict(comparison, block_index=0, challenger_index=0,
            incumbent_index=1), dict(comparison, block_index=1, challenger_index=2,
            incumbent_index=0)], verification_budget=352))
    truth = {(i,): dict(objective=f, quality=q, true_feasible=True)
             for i, f, q in [(0, .2, .8), (1, .3, .7), (2, .1, .9)]}
    history = [dict(point=[i]) for i in [0, 1, *range(3, 11), 2]]
    return (dict(old=dict(status='complete'), new=dict(status='complete')), ref,
            dict(old=item, new=copy.deepcopy(item)), dict(old=old, new=new), history, truth)


class TerminalAttributionTests(unittest.TestCase):
    def test_actual_first_winner_and_unattempted_third(self):
        row = analyze_pair(*fixture())
        third = row['third_comparison']['actual']
        self.assertEqual(third['incumbent'], 'initial:0')
        self.assertEqual(third['seed_namespace'], [31, 'replication', 1329744452, 1])
        self.assertAlmostEqual(third['true_objective_difference'], -.1)
        self.assertEqual(row['candidates'][2]['old']['safety']['status'], 'not_attempted')
        self.assertIsNone(row['candidates'][2]['old']['safety']['passed'])
        self.assertEqual(row['coverage']['old']['untested_good_new'], ['neighbor:2'])
        self.assertEqual(row['coverage']['new']['deployed_good_new'], 1)
        self.assertEqual(row['logical_terminal_calls_delta'], 128)

    def test_false_certificate_and_missing_quality_reference(self):
        args = fixture()
        args[-1][(2,)]['true_feasible'] = False
        args[1]['quality_baseline_defined'] = False
        args[1]['best_initial_quality'] = None
        row = analyze_pair(*args)
        self.assertEqual(row['false_certificates']['new'], ['neighbor:2'])
        self.assertEqual(row['coverage']['new']['unknown_good_classification'], 3)
        self.assertEqual(row['coverage']['new']['deployed_good_new'], 0)

    def test_failed_terminal_is_unknown_not_safety_failure(self):
        args = fixture()
        args[0]['new'] = dict(status='failed', failure_file='failure.json')
        args[3].pop('new')
        row = analyze_pair(*args)
        self.assertEqual(row['third_comparison']['state'], 'unknown_terminal_failed')
        self.assertEqual(row['candidates'][2]['new']['safety']['status'], 'unknown_after_failed')
        self.assertIsNone(row['candidates'][2]['new']['safety']['passed'])
        self.assertIsNone(row['logical_terminal_calls_delta'])

    def test_pairing_mismatch_stops_attribution(self):
        args = fixture()
        args[2]['new']['frozen_shortlist'][2]['point'] = [42]
        with self.assertRaises(AssertionError):
            analyze_pair(*args)

    def test_compact_and_aggregate_keep_unknown_and_false_certificate(self):
        args = fixture()
        args[-1][(2,)]['true_feasible'] = False
        args[1]['new_minus_old']['G'] = None
        row = analyze_pair(*args)
        task = compact_task(dict(seed=42, domain='Fixture', status='complete',
            quality_reference_status='defined', arms=[row]))
        self.assertFalse(task['arms'][0]['candidates'][2]['truth']['true_feasible'])
        self.assertEqual(task['arms'][0]['third_comparison']['actual_opponent'], 'initial:0')
        aggregate = aggregate_terminals([task])['arms'][0]
        self.assertEqual(aggregate['changes']['new_minus_old']['G']['counts']['unknown'], 1)
        self.assertIsNone(aggregate['changes']['new_minus_old']['G']['conditional_mean'])
        self.assertEqual(aggregate['false_certificate_cases'][0]['protocol'], 'new')
        self.assertEqual(aggregate['costs']['new']['logical_terminal_calls']['known_sum'], 352)


if __name__ == '__main__':
    unittest.main()
