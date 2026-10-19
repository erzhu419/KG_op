"""Saved-metric fixtures: no simulation, model restoration or truth evaluation."""
from copy import deepcopy
import unittest

from performance import summarize_mainline_terminal_recording_replication as summary


ARMS = [summary.AFTER, summary.POINT, summary.FULL, 'same_source_gp', 'legacy_old',
        'structured', 'native_after', 'native_full', 'stop']


def metrics(gain):
    return dict(A=1, C=1, false_certificate=0, R=.5+gain, G=gain,
                P=max(gain,0.), L=max(-gain,0.))


def replace_outcome(row, field, value):
    row[field] = value
    row['new_minus_old'] = summary.difference(row['new'],row['old'])
    row['third_challenge'] = summary.difference(row['new'],row['B'])
    row['prechange'] = summary.difference(row['B'],row['old'])


def make_row(seed, arm, replicate):
    # Old AP is positive while AP-new and AFTER new-old are negative. This
    # catches accidentally reusing the former terminal-change primary.
    gain = (.01*seed + .02*(replicate-1)) if arm == summary.AFTER else 0.
    old, new = metrics(gain), metrics(.05 if arm == summary.POINT else 0.)
    row = dict(seed=seed, domain='family_'+str(seed), arm=arm, history_arm=arm,
        status='complete', old_status='complete', new_status='complete',
        quality_baseline_defined=True, quality_reference_status='defined', best_initial_quality=.5,
        old=old, new=new, B=old, old_reported_deployed=arm+':history:10',
        new_reported_deployed='initial:0', B_deployed=arm+':history:10',
        old_deployed=arm+':history:10', new_deployed='initial:0')
    replace_outcome(row,'old',old)
    return row


def refresh_outcomes(stream):
    stream['summary']['outcomes'] = [dict(seed=r['seed'],domain=r['domain'],arm=r['arm'],
        protocol=p,status=r[p+'_status'],metrics=deepcopy(r[p]),
        costs=dict(logical_total_calls=365,new_verification_calls=7),
        files=dict(verification_file='saved/'+r['arm']+'/'+p+'.json'))
        for r in stream['summary']['rows'] for p in ('old','new')]


def fixture():
    plan = dict(tasks=[dict(geometry_task_seed=s) for s in (1,2)], terminal_arms=ARMS,
        statistical_unit='Two fixture geometries, with two streams averaged within each geometry.',
        analysis=dict(primary='Old/default AFTER-minus-POINT C and signed G'),
        reference_manifest='opened-reference.json', historical_accounts=['old-registration/completion.json'])
    streams = []
    for replicate in (1,2):
        stream = dict(name='observation_'+str(replicate),replicate=replicate,
            output_dir='streams/'+str(replicate),
            summary=dict(status='complete',rows=[make_row(seed,arm,replicate)
                for seed in (1,2) for arm in ARMS],mechanism=dict(aggregate=dict(tasks=2))),
            completion=dict(new_work=dict(new_target_initial_calls=20,new_target_search_calls=6,
                new_verification_calls=54,new_truth_function_calls=11,new_source_fits=0,
                source_model_restorations=None,randomized_new_observations=True),
                new_physical_target_calls=80,recorded_cpu_seconds=10.,
                failed_attempts=[],execution_failures=[],terminal_costs=dict(old_logical_calls=54)))
        refresh_outcomes(stream)
        streams.append(stream)
    reference_rows = summary.origin_rows([make_row(seed,arm,1) for seed in (1,2) for arm in ARMS])
    for row in reference_rows:
        if row['arm'] == summary.AFTER:
            replace_outcome(row,'old',metrics(100.))
    reference = summary.summarize_unit(plan,summary.origin_rows(reference_rows))
    return plan,streams,reference


class TerminalRecordingReplicationSummaryTests(unittest.TestCase):
    def test_old_ap_primary_uses_geometry_units_and_excludes_opened_reference(self):
        plan,streams,reference = fixture()
        result = summary.summarize_loaded(plan,streams,reference)
        primary = result['geometry_averaged']['primary']
        self.assertEqual(primary['protocol'],'old')
        self.assertEqual(primary['left'],summary.AFTER)
        self.assertEqual(primary['right'],summary.POINT)
        estimate = primary['metrics']['G']
        self.assertEqual(estimate['registered_geometries'],2)
        self.assertEqual(estimate['conditional_available']['degrees_of_freedom'],1)
        self.assertAlmostEqual(estimate['all_registered_mean'],.025)
        self.assertEqual([round(r['metrics']['G'],3) for r in primary['rows']],[.02,.03])
        self.assertLess(result['geometry_averaged']['secondary']['AFTER_minus_POINT']
                        ['new_contrast']['G']['all_registered_mean'],0.)
        self.assertEqual(len(result['outcomes']),72)
        compact = summary.compact_summary(result)
        self.assertEqual(len(compact['cell_accounts']),72)
        self.assertEqual(len(compact['geometry_averaged']['arms']),9)
        self.assertAlmostEqual(compact['geometry_averaged']['primary']['metrics']['G']['largest_positive_share'],.6)
        drop = compact['geometry_averaged']['primary']['metrics']['G']['drop_largest_positive']
        self.assertEqual(drop['excluded_seed'],2)
        self.assertAlmostEqual(drop['all_registered_leave_one_out_mean'],.02)
        self.assertFalse(drop['primary_sample_changed'])

    def test_missing_stream_or_baseline_never_uses_the_other_stream_as_replacement(self):
        plan,streams,reference = fixture()
        row = next(r for r in streams[1]['summary']['rows'] if r['seed']==1 and r['arm']==summary.AFTER)
        row.update(quality_baseline_defined=False,quality_reference_status='no_true_feasible_initial',
                   best_initial_quality=None)
        for field in summary.FIELDS:
            for metric in ('G','P','L'):
                row[field][metric] = None
        refresh_outcomes(streams[1])
        result = summary.summarize_loaded(plan,streams,reference)
        primary = result['geometry_averaged']['primary']
        self.assertIsNone(primary['rows'][0]['metrics']['G'])
        self.assertEqual(primary['rows'][0]['metrics']['C'],0.)
        self.assertIsNone(primary['metrics']['G']['all_registered_mean'])
        self.assertEqual(primary['metrics']['G']['conditional_available']['n'],1)
        self.assertEqual(primary['metrics']['C']['conditional_available']['n'],2)
        with self.assertRaisesRegex(ValueError,'Both registered new streams'):
            summary.summarize_loaded(plan,streams[:1],reference)
        streams[1]['summary']['rows'].pop()
        with self.assertRaisesRegex(ValueError,'geometry-arm grid'):
            summary.summarize_loaded(plan,streams,reference)

    def test_signed_losses_false_certificates_failures_and_cost_scopes_are_preserved(self):
        plan,streams,reference = fixture()
        loss = next(r for r in streams[0]['summary']['rows'] if r['seed']==1 and r['arm']==summary.AFTER)
        replace_outcome(loss,'old',metrics(-.02))
        false = next(r for r in streams[1]['summary']['rows'] if r['seed']==1 and r['arm']==summary.FULL)
        replace_outcome(false,'old',dict(A=1,C=0,false_certificate=1,R=0.,G=0.,P=0.,L=0.))
        failed = next(r for r in streams[1]['summary']['rows'] if r['seed']==2 and r['arm']=='same_source_gp')
        failed.update(old_status='failed',status='incomplete_pair')
        replace_outcome(failed,'old',dict.fromkeys(summary.METRICS))
        streams[1]['summary']['status'] = 'complete_with_missing_cells'
        streams[1]['completion']['failed_attempts'] = [dict(status='failed',cpu_seconds=1.,new_verification_calls=7)]
        for stream in streams:
            refresh_outcomes(stream)
        result = summary.summarize_loaded(plan,streams,reference)
        self.assertEqual(result['status'],'complete_with_missing_cells')
        self.assertEqual(result['completed_outcome_cells'],71)
        self.assertEqual(len(result['failures']),1)
        self.assertEqual(len(result['false_certificates']),1)
        enriched = result['streams'][0]['rows'][0]
        self.assertEqual(enriched['origin_contributions']['dynamic']['old']['G'],-.02)
        false_row = next(r for r in result['streams'][1]['rows'] if r['seed']==1 and r['arm']==summary.FULL)
        self.assertEqual(false_row['origin_contributions']['dynamic']['old']['false_certificate'],1)
        self.assertEqual(false_row['origin_contributions']['dynamic']['old']['G'],0.)
        account = summary.completion(plan,result,[dict(status='complete',cpu_seconds=3.,stage='outer-summary')])
        self.assertEqual(account['new_physical_target_calls'],160)
        self.assertEqual(account['new_work']['new_truth_function_calls'],22)
        self.assertIsNone(account['new_work']['source_model_restorations'])
        self.assertEqual(account['recorded_cpu_seconds'],23.)
        self.assertEqual(account['historical_accounts'],['old-registration/completion.json'])
        compact = summary.compact_summary(result)
        self.assertEqual(len(compact['cell_accounts']),72)
        self.assertEqual(compact['costs']['streams'][1]['failed_attempt_count'],1)
        self.assertEqual(compact['failures'][0]['costs']['new_verification_calls'],7)


if __name__ == '__main__':
    unittest.main()
