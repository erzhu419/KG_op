"""Fixtures for paired geometry units, overlapping references and saved failures."""
from copy import deepcopy
import unittest

from performance import summarize_mainline_reference_retention as summary


ARMS = [summary.AFTER, summary.AFTER_NATIVE, summary.FULL, summary.FULL_NATIVE,
        summary.POINT, 'same_source_gp', 'legacy_native_neighbors',
        'current_structural_mixture', 'initialization_only']
SEEDS = (2026107013, 2026107017)
STREAMS = ('observation_1', 'observation_2')


def metrics(gain):
    return dict(A=1, C=1, false_certificate=0, R=.5+gain, G=gain,
                P=max(gain, 0.), L=max(-gain, 0.))


def fixture():
    plan = dict(tasks=[dict(geometry_task_seed=seed, domain='family') for seed in SEEDS],
        streams=[dict(name=name) for name in STREAMS], terminal_arms=ARMS,
        reference_output_dir='stage77', scientific_output_dir='stage76', historical_accounts=['older/completion.json'])
    tasks, streams, outcomes = [], [], []
    for stream_index, stream in enumerate(STREAMS):
        science_rows = []
        for seed in SEEDS:
            references = [dict(reference_id='initial_00', policy_id='P0',
                roles=['empirical_incumbent', 'stop_primary', 'stop_final_slot1']),
                dict(reference_id='initial_01', policy_id='P1', roles=['stop_safe_fallback', 'stop_final_slot2'])]
            arms = []
            for arm in ARMS:
                native = arm in (summary.AFTER_NATIVE, summary.FULL_NATIVE)
                history = summary.AFTER if arm==summary.AFTER_NATIVE else summary.FULL if arm==summary.FULL_NATIVE else arm
                gain = (.03 if stream_index==0 else .01) if seed==SEEDS[1] and arm in (summary.AFTER,summary.AFTER_NATIVE) else 0.
                if arm==summary.AFTER_NATIVE:
                    gain += (.02 if stream_index==0 else .04) if seed==SEEDS[1] else -.01
                scientific = dict(seed=seed, domain='family', arm=arm, history_arm=history)
                protocols = {}
                for protocol in ('old', 'new'):
                    values = metrics(gain)
                    scientific[protocol] = values
                    scientific[protocol+'_status'] = 'complete'
                    scientific[protocol+'_deployed'] = 'initial:0' if native else 'neighbor:2'
                    rows = [dict(reference_id=ref['reference_id'], policy_id=ref['policy_id'], roles=ref['roles'],
                        shortlisted=native, attempted=native, certified=native, comparison_attempted=False,
                        deployed=native and index==0, truth=dict(true_feasible=True, quality=.55),
                        quality_minus_actual_deployed=.05-gain) for index, ref in enumerate(references)]
                    protocols[protocol] = dict(status='complete', reference_rows=rows,
                        deployed_policy_id='P0' if native else 'P2', terminal_structure=dict(shortlist=[], candidate_budgets=[120,120,64]))
                    outcomes.append(dict(seed=seed, stream=stream, domain='family', arm=arm,
                        protocol=protocol, status='complete', costs=dict(physical_verification_calls=240 if native else 256)))
                science_rows.append(scientific)
                arms.append(dict(arm=arm, history_arm=history, terminal_backend='native' if native else 'role', protocols=protocols))
            tasks.append(dict(seed=seed, domain='family', stream=stream, status='complete', references=references, arms=arms))
        streams.append(dict(name=stream, rows=science_rows))
    return tasks, plan, dict(streams=streams, outcomes=outcomes, costs=dict(total_physical_calls=1234))


class ReferenceRetentionSummaryTests(unittest.TestCase):
    def test_signed_effects_average_streams_before_geometry_and_keep_all_controls(self):
        tasks, plan, science = fixture()
        result = summary.aggregate(tasks, plan, scientific_summary=science)
        self.assertEqual(result['registered_outcomes'], 72)
        self.assertEqual(len(result['groups']), 18)
        audit = result['audit_primary']
        stats = audit['metrics']['G']['geometry_statistics']
        self.assertEqual(stats['registered_geometries'], 2)
        self.assertAlmostEqual(stats['all_registered_mean'], .01)
        self.assertEqual((stats['positive'], stats['negative']), (1,1))
        self.assertAlmostEqual(stats['leave_largest_positive_out']['all_registered_mean'], -.01)
        self.assertEqual(audit['metrics']['G']['task_stream_description']['registered_task_streams'], 4)
        self.assertTrue(all(row['same_history_arm'] for row in audit['rows']))
        self.assertTrue(all(row['saved_cost_delta']['physical_verification_calls']==-16 for row in audit['rows']))
        self.assertFalse(audit['causal_shortlist_effect_identified'])
        self.assertEqual(result['original_scientific_primary']['left'], summary.AFTER)
        self.assertEqual(result['original_scientific_primary']['right'], summary.POINT)
        self.assertAlmostEqual(result['original_scientific_primary']['metrics']['G']['geometry_statistics']['all_registered_mean'], .01)

    def test_overlapping_reference_roles_are_not_extra_physical_points_or_gain_events(self):
        tasks, plan, science = fixture()
        result = summary.aggregate(tasks, plan, scientific_summary=science)
        group = next(row for row in result['groups'] if row['arm']==summary.AFTER_NATIVE and row['protocol']=='old')
        by_role = {row['role']:row for row in group['reference_coverage']}
        self.assertEqual(by_role['all_initial']['reference_occurrences'], 8)
        self.assertEqual(by_role['empirical_incumbent']['reference_occurrences'], 4)
        self.assertEqual(by_role['stop_primary']['reference_occurrences'], 4)
        self.assertEqual(by_role['all_initial']['stages']['shortlisted']['yes'], 8)
        account = next(row for row in result['audit_primary']['reference_change_accounting']
                       if row['role']=='all_initial' and row['stage']=='shortlisted')
        added = next(row for row in account['categories'] if row['category']=='native_added_only')
        self.assertEqual(added['task_streams'], 4)
        self.assertAlmostEqual(added['G']['defined_sum'], .04)
        self.assertEqual((added['G']['positive'],added['G']['negative']), (2,2))
        self.assertEqual(result['costs']['inherited_scientific_account']['total_physical_calls'], 1234)
        compact = summary.compact_summary(result)
        self.assertNotIn('cells', compact)
        self.assertEqual(len(compact['audit_primary']['rows']), 4)
        self.assertEqual(len(compact['groups']), 18)
        self.assertEqual(compact['costs']['historical_accounts'], ['older/completion.json'])

    def test_missing_analysis_does_not_erase_science_and_missing_stream_stays_null(self):
        tasks, plan, science = fixture()
        tasks[0].update(status='failed', arms=[], error='saved path unavailable')
        row = next(row for row in science['streams'][1]['rows'] if row['seed']==SEEDS[1] and row['arm']==summary.AFTER_NATIVE)
        row['old']['G'] = None
        false = next(row for row in science['streams'][1]['rows'] if row['seed']==SEEDS[1] and row['arm']==summary.FULL)
        false['new'] = dict(A=1,C=0,false_certificate=1,R=0.,G=0.,P=0.,L=0.)
        failed = next(row for row in science['streams'][1]['rows'] if row['seed']==SEEDS[1] and row['arm']=='same_source_gp')
        failed.update(new_status='failed',new=dict.fromkeys(summary.METRICS))
        outcome = next(row for row in science['outcomes'] if row['stream']==STREAMS[1] and row['seed']==SEEDS[1]
                       and row['arm']=='same_source_gp' and row['protocol']=='new')
        outcome['status']='failed'
        result = summary.aggregate(tasks, plan, scientific_summary=science)
        self.assertEqual(len(result['analysis_failures']),1)
        self.assertEqual(len(result['outcome_failures']),1)
        self.assertEqual(len(result['false_certificates']),1)
        cell = next(row for row in result['cells'] if row['seed']==SEEDS[0] and row['stream']==STREAMS[0]
                    and row['arm']==summary.AFTER_NATIVE and row['protocol']=='old')
        self.assertEqual(cell['metrics']['G'],-.01)
        self.assertEqual(cell['status'],'complete')
        self.assertTrue(all(row['shortlisted'] is None for row in cell['references']))
        stats = result['audit_primary']['metrics']['G']['geometry_statistics']
        self.assertIsNone(stats['all_registered_mean'])
        self.assertEqual(stats['defined_geometries'],1)
        self.assertEqual(result['costs']['inherited_scientific_account']['total_physical_calls'],1234)
        with self.assertRaisesRegex(ValueError,'Both registered stream'):
            summary.aggregate(tasks[:-1],plan,scientific_summary=science)


if __name__=='__main__':
    unittest.main()
