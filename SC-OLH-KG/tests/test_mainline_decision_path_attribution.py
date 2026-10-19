"""Small substitutes for null quality, nested losses and saved-score attribution."""
from copy import deepcopy
import json

import pytest

from performance.mainline_decision_path_attribution import (
    ROLES, analyze_task, attribute_arm, opportunity_gaps, quality_set,
)


def fixture():
    history = [dict(point=[i], raw=[float(i), -1.]) for i in range(13)]
    qualities = [.1]*9 + [.4, .8, .7, 1., 1.2]
    truth = {(i,): dict(quality=q, objective=-q, true_feasible=True)
             for i, q in enumerate(qualities)}
    registry = {(i,): f'P{i:04d}' for i in range(14)}
    shortlist = [dict(point=[i], shortlist_position=j+1) for j, i in enumerate((10, 11, 9))]
    item = dict(frozen_shortlist=shortlist)
    record = dict(deployed=[9], verification=dict(certified=True,
        attempts=[dict(candidate_index=1, certified=False, upper_margin=.1),
                  dict(candidate_index=2, certified=True, upper_margin=-.1)],
        objective_incumbent_position=3, objective_comparison={}))
    case = dict(arm=ROLES[1], history_arm=ROLES[1], terminal_backend='role',
        old=dict(status='complete', item=item, record=record),
        new=dict(status='complete', item=deepcopy(item), record=deepcopy(record)))
    states = [dict(arm=ROLES[1], paid_prefix=i, status='captured',
        candidates=[dict(policy_id=f'P{j:04d}') for j in range(14)]) for i in (10, 11, 12)]
    metrics = dict(A=1, C=1, false_certificate=0, R=.4, G=0., P=0., L=0.)
    recording = dict(status='complete', issues=[], observed=[dict(observed_index=i,
        objective_mean=i+.23, objective_probability_violation=.2,
        posterior_rank=13-i, objective_eligible=True,
        disposition='incumbent_insertion_displaced' if i == 12 else 'included') for i in range(13)])
    return dict(case=case, history=history, frozen=dict(terminal=dict(frozen_shortlist=shortlist)),
        truth=truth, baseline=dict(status='defined', truth=truth[(9,)]), registry=registry,
        origins={(i,): 'initial' if i < 10 else 'dynamic' for i in range(14)}, chains=[], states=states,
        saved_metrics=dict(old=metrics, new=deepcopy(metrics)), recording=recording)


def test_nested_opportunity_losses_telescope_and_saved_ranks_are_not_reranked():
    value = attribute_arm(**fixture())
    protocol = value['protocols']['old']
    gaps = protocol['opportunity_gaps']
    assert [gaps[k] for k in ('search', 'shortlist', 'coverage', 'safety', 'comparison')] == pytest.approx([.2, .2, .1, .3, 0])
    assert gaps['terminal_total'] == pytest.approx(.6)
    assert gaps['total'] == pytest.approx(.8)
    assert gaps['telescoping_residual'] == pytest.approx(0)
    assert value['discovery']['quality_gain'] == pytest.approx(.6)
    assert value['observed'][0]['final_scores']['terminal_order_rank'] == 13
    assert value['observed'][0]['final_scores']['objective'] == .23
    assert value['observed'][12]['selection_record']['disposition'] == 'incumbent_insertion_displaced'
    assert value['observed'][0]['quality_minus_deployed']['old'] == pytest.approx(-.3)
    assert protocol['counts']['better_than_deployed_dispositions'] == {
        'not_on_frozen_shortlist': 1, 'not_verified_after_earlier_terminal_decision': 1,
        'safety_certificate_failed': 1}
    # A path's opportunity is not summed into the actual deployment metric.
    assert protocol['metrics']['G'] == 0


def test_no_certificate_keeps_quality_null_and_retains_actual_zero_gain():
    args = fixture()
    for name in ('old', 'new'):
        args['case'][name]['record']['verification']['certified'] = False
        for row in args['case'][name]['record']['verification']['attempts']:
            row['certified'] = False
        args['saved_metrics'][name].update(A=0, C=0, R=0, G=0)
    value = attribute_arm(**args)['protocols']['old']
    assert value['sets']['deployed']['max_quality'] is None
    assert value['sets']['safety_passed']['max_quality'] is None
    assert value['opportunity_gaps']['safety'] is None
    assert value['opportunity_gaps']['comparison'] is None
    assert value['opportunity_gaps']['terminal_total'] is None
    assert value['metrics']['C'] == value['metrics']['G'] == 0
    assert value['no_deployment_reason'] == 'all_attempted_safety_certificates_failed'


def test_incomplete_truth_never_reports_an_optimistic_maximum():
    args = fixture()
    del args['truth'][(13,)]
    value = attribute_arm(**args)
    assert value['sets']['pool']['status'] == 'truth_incomplete'
    assert value['sets']['pool']['max_quality'] is None
    assert value['sets']['pool']['best_policy_ids'] == []
    assert value['protocols']['old']['opportunity_gaps']['search'] is None
    assert value['protocols']['old']['opportunity_gaps']['terminal_total'] == pytest.approx(.6)
    del args['truth'][(11,)]
    value = attribute_arm(**args)
    assert value['sets']['observed']['max_quality'] is None
    assert value['protocols']['old']['sets']['shortlist']['max_quality'] is None
    assert value['discovery']['improving_safe_new_policy_count'] is None
    assert value['discovery']['known_improving_safe_new_policy_count'] == 2


def test_same_history_native_shares_pool_but_gp_never_borrows_a_role_pool():
    args = fixture()
    role = attribute_arm(**args)
    args['case']['arm'] += '_native_terminal'
    args['case']['terminal_backend'] = 'native'
    native = attribute_arm(**args)
    assert native['sets']['pool'] == role['sets']['pool']
    args['case'].update(arm='same_source_gp', history_arm='same_source_gp', terminal_backend='original')
    args['states'] = []
    gp = attribute_arm(**args)
    assert gp['sets']['pool']['status'] == 'full_pool_not_saved'
    assert gp['protocols']['old']['opportunity_gaps']['search'] is None
    assert gp['protocols']['old']['opportunity_gaps']['terminal_total'] == pytest.approx(.6)


def test_empty_true_safe_set_and_false_certificate_have_no_quality_zero():
    args = fixture()
    for value in args['truth'].values():
        value['true_feasible'] = False
    args['baseline'] = dict(status='no_true_feasible_initial', truth=None)
    for name in ('old', 'new'):
        args['saved_metrics'][name].update(C=0, false_certificate=1, R=0, G=None, P=None, L=None)
    value = attribute_arm(**args)
    assert value['sets']['observed']['status'] == 'no_true_safe_policy'
    assert value['sets']['observed']['max_quality'] is None
    assert value['protocols']['old']['no_deployment_reason'] == 'false_certificate'
    assert value['protocols']['old']['metrics']['G'] is None
    assert all(row['quality_minus_deployed']['old'] is None for row in value['observed'])


def test_reader_keeps_all_registered_controls_and_uses_corrected_recording(tmp_path):
    args = fixture()
    def save(path, value):
        target = tmp_path/path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value))
    cells = [dict(name=args['case']['arm'], history_arm=args['case']['history_arm'], terminal_backend='role'),
        dict(name='same_source_gp', history_arm='same_source_gp', terminal_backend='original')]
    cases = []
    summary = []
    audit = []
    for cell in cells:
        arm = cell['name']
        case = dict(seed=17, domain='fixture', arm=arm, history_arm=cell['history_arm'], source_case='task/'+arm)
        save('task/'+arm+'/search.json', dict(history=args['history'], **args['frozen']))
        for protocol in ('old', 'new'):
            save(arm+'/'+protocol+'_input.json', args['case'][protocol]['item'])
            save(arm+'/'+protocol+'_verification.json', args['case'][protocol]['record'])
            case[protocol] = dict(status='complete', input_file=arm+'/'+protocol+'_input.json',
                verification_file=arm+'/'+protocol+'_verification.json')
        cases.append(case)
        summary.append(dict(seed=17, arm=arm, **args['saved_metrics']))
        audit.append(dict(seed=17, arm=arm, **args['recording']))
    save('cohort/execution_status.json', dict(status='sealed', cases=cases))
    save('cohort/summary.json', dict(rows=summary))
    save('cohort/mechanism/seed_17/analysis.json', dict(policies=[dict(point=list(p),
        policy_id=pid, origin=args['origins'][p]) for p, pid in args['registry'].items()],
        states=args['states'], query_chains=[]))
    save('task/initialization.json', dict(history=args['history'][:10]))
    save('task/truth.json', [dict(point=list(p), truth=t) for p,t in args['truth'].items()])
    save('corrected/revision_1/details.json', dict(cases=audit))
    result = analyze_task(tmp_path, dict(output_dir='cohort', terminal_arms=cells), 17,
        stream='observation_1', terminal_recording_file=tmp_path/'corrected/revision_1/details.json')
    result = json.loads(json.dumps(result))
    assert result['coverage']['arms'] == 2 and result['coverage']['terminal_cells'] == 4
    assert result['stream'] == 'observation_1'
    assert result['arms'][0]['observed'][12]['selection_record']['posterior_rank'] == 1
    assert result['arms'][1]['protocols']['old']['metrics']['C'] == 1
    assert result['arms'][1]['sets']['pool']['max_quality'] is None
