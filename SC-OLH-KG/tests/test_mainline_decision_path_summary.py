"""Saved-value fixtures for denominators, signed effects and inherited evidence."""
from copy import deepcopy
import unittest

from performance import summarize_mainline_decision_path_attribution as summary


ARMS = [summary.AFTER, summary.POINT, summary.FULL, 'same_source_gp', 'legacy_old',
        'structured', 'native_after', 'native_full', 'stop']
SEEDS = (2026107013, 2026107017)
STREAMS = ('observation_1', 'observation_2')


def metrics(gain):
    return dict(A=1, C=1, false_certificate=0, R=.5+gain, G=gain,
                P=max(gain, 0.), L=max(-gain, 0.))


def fixture():
    plan = dict(tasks=[dict(geometry_task_seed=seed, domain='family') for seed in SEEDS],
        streams=[dict(name=name) for name in STREAMS], terminal_arms=ARMS,
        reference_output_dir='sealed-science', historical_accounts=['earlier/completion.json'])
    tasks, streams, outcomes = [], [], []
    for stream_index, stream in enumerate(STREAMS):
        scientific_rows = []
        for seed in SEEDS:
            arms = []
            for arm in ARMS:
                gain = ((.04 if stream_index == 0 else .02) if seed == SEEDS[1] else -.01) if arm == summary.AFTER else (
                    .01 if arm == summary.POINT and seed == SEEDS[1] else 0.)
                scientific = dict(seed=seed, domain='family', arm=arm)
                protocols = {}
                for protocol in ('old', 'new'):
                    values = metrics(gain if protocol == 'old' else -.02 if arm == summary.AFTER else .01)
                    gaps = dict(search=.03 if seed == SEEDS[1] else .04,
                        shortlist=.01, coverage=.01, safety=.02, comparison=.02,
                        terminal_total=.06, total=.09 if seed == SEEDS[1] else .10)
                    if arm not in (summary.AFTER, summary.POINT, summary.FULL):
                        gaps.update(search=None, total=None)
                    protocols[protocol] = dict(status='complete', metrics=deepcopy(values),
                        opportunity_gaps=gaps, deployed_policy_id='observed:10',
                        no_deployment_reason=None, counts={}, sets={})
                    scientific[protocol] = values
                    scientific[protocol+'_status'] = 'complete'
                    scientific[protocol+'_deployed'] = 'observed:10'
                    outcomes.append(dict(seed=seed, stream=stream, domain='family', arm=arm,
                        protocol=protocol, status='complete', costs=dict(new_verification_calls=7)))
                scientific_rows.append(scientific)
                arms.append(dict(arm=arm, history_arm=arm, terminal_backend='role',
                    search_status='complete', sets={}, discovery=dict(quality_gain=.03), protocols=protocols))
            tasks.append(dict(seed=seed, stream=stream, domain='family', status='complete',
                baseline=dict(status='defined', truth=dict(quality=.5)), arms=arms))
        streams.append(dict(name=stream, rows=scientific_rows))
    scientific_summary = dict(streams=streams, outcomes=outcomes)
    return tasks, plan, scientific_summary


def after(task):
    return next(arm for arm in task['arms'] if arm['arm'] == summary.AFTER)


class DecisionPathSummaryTests(unittest.TestCase):
    def test_all_controls_and_signed_ap_effects_use_two_geometry_units(self):
        tasks, plan, science = fixture()
        result = summary.aggregate(tasks, plan, scientific_summary=science)
        self.assertEqual(result['registered_outcomes'], 72)
        self.assertEqual(len(result['groups']), 18)
        self.assertEqual(len(result['focal_cases']), 4)
        primary = result['primary']['metrics']['G']
        self.assertEqual(result['primary']['protocol'], 'old')
        self.assertEqual(primary['task_stream_description']['registered_task_streams'], 4)
        self.assertEqual(primary['geometry_statistics']['registered_geometries'], 2)
        self.assertAlmostEqual(primary['geometry_statistics']['all_registered_mean'], .005)
        self.assertEqual(primary['geometry_statistics']['positive'], 1)
        self.assertEqual(primary['geometry_statistics']['negative'], 1)
        self.assertEqual(primary['geometry_statistics']['largest_positive']['seed'], SEEDS[1])
        self.assertAlmostEqual(primary['geometry_statistics']['leave_largest_positive_out']['all_registered_mean'], -.01)
        self.assertLess(result['secondary'][0]['metrics']['G']['geometry_statistics']['all_registered_mean'], 0.)
        compact = summary.compact_summary(result)
        self.assertEqual(len(compact['AFTER_old_task_streams']), 4)
        self.assertEqual(len(compact['focal_cells']), 72)
        self.assertNotIn('focal_cases', compact)
        self.assertEqual(compact['costs']['historical_accounts'], ['earlier/completion.json'])

    def test_common_support_and_no_certificate_do_not_impute_quality(self):
        tasks, plan, science = fixture()
        row = next(task for task in tasks if task['seed']==SEEDS[0] and task['stream']==STREAMS[1])
        saved = after(row)['protocols']['old']
        saved['opportunity_gaps'].update(search=None, comparison=None, terminal_total=None, total=None)
        saved['no_deployment_reason'] = 'no_safety_certificate'
        no_certificate = dict(A=0, C=0, false_certificate=0, R=0., G=0., P=0., L=0.)
        saved['metrics'] = no_certificate
        original = next(r for r in science['streams'][1]['rows'] if r['seed']==SEEDS[0] and r['arm']==summary.AFTER)
        original['old'] = no_certificate
        result = summary.aggregate(tasks, plan, scientific_summary=science)
        group = next(g for g in result['groups'] if g['arm']==summary.AFTER and g['protocol']=='old')
        common = group['search_vs_terminal_common_support']
        self.assertEqual(common['search']['task_stream_description']['defined_task_streams'], 3)
        self.assertEqual(common['terminal']['task_stream_description']['defined_task_streams'], 3)
        self.assertEqual(common['search_minus_terminal']['geometry_statistics']['defined_geometries'], 1)
        self.assertIsNone(common['search_minus_terminal']['geometry_statistics']['all_registered_mean'])
        self.assertAlmostEqual(common['search_minus_terminal']['geometry_statistics']['defined_subset_mean'], -.03)
        self.assertEqual(group['certificate_categories']['counts']['no_certificate'], 1)
        self.assertIsNone(group['opportunity_gaps']['comparison']['geometry_statistics']['all_registered_mean'])
        gp = next(g for g in result['groups'] if g['arm']=='same_source_gp' and g['protocol']=='old')
        self.assertEqual(gp['opportunity_gaps']['search']['task_stream_description']['defined_task_streams'], 0)
        self.assertIsNone(gp['opportunity_gaps']['search']['geometry_statistics']['all_registered_mean'])

    def test_attribution_failure_retains_scientific_success_failures_and_fees(self):
        tasks, plan, science = fixture()
        failed = tasks[0]
        failed.update(status='failed', arms=[], error='saved attribution input unavailable')
        scientific = next(row for row in science['streams'][1]['rows'] if row['seed']==SEEDS[1] and row['arm']=='same_source_gp')
        scientific.update(new_status='failed', new=dict.fromkeys(summary.METRICS))
        outcome = next(row for row in science['outcomes'] if row['seed']==SEEDS[1]
            and row['stream']==STREAMS[1] and row['arm']=='same_source_gp' and row['protocol']=='new')
        outcome['status'] = 'failed'
        false = next(row for row in science['streams'][1]['rows'] if row['seed']==SEEDS[1] and row['arm']==summary.FULL)
        false['old'] = dict(A=1, C=0, false_certificate=1, R=0., G=0., P=0., L=0.)
        result = summary.aggregate(tasks, plan, scientific_summary=science)
        self.assertEqual(len(result['cells']), 72)
        self.assertEqual(len(result['analysis_failures']), 1)
        self.assertEqual(len(result['outcome_failures']), 1)
        self.assertEqual(len(result['false_certificates']), 1)
        cell = next(row for row in result['cells'] if row['seed']==SEEDS[0]
            and row['stream']==STREAMS[0] and row['arm']==summary.AFTER and row['protocol']=='old')
        self.assertEqual(cell['status'], 'complete')
        self.assertEqual(cell['analysis_status'], 'failed')
        self.assertEqual(cell['metrics']['G'], -.01)
        self.assertEqual(cell['inherited_scientific_costs']['new_verification_calls'], 7)
        self.assertTrue(all(value is None for value in cell['opportunity_gaps'].values()))
        self.assertAlmostEqual(result['primary']['metrics']['G']['geometry_statistics']['all_registered_mean'], .005)
        with self.assertRaisesRegex(ValueError, 'both stream analyses'):
            summary.aggregate(tasks[:-1], plan, scientific_summary=science)


if __name__ == '__main__':
    unittest.main()
