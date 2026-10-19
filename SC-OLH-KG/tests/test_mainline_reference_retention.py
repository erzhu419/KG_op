"""Substitutes for truth-free reference freezing and saved terminal path joins."""
from copy import deepcopy
import json

import pytest

from performance import mainline_reference_retention as audit


def stop_input():
    observed = [dict(point=[i], observation_indices=[i], replicate_count=1) for i in range(10)]
    return dict(seed=7013, domain='fixture', arm=audit.STOP, search_evaluation_count=10,
        objective_incumbent_position=2, frozen_shortlist=[dict(point=[i], shortlist_role=f'slot{j+1}')
            for j, i in enumerate((2, 1, 0))], candidate_budgets=[80, 128, 128],
        familywise_delta=.05, objective_comparison_budget=8, objective_comparison_delta=.05/3,
        verification_seed=17, terminal_recording=dict(status='frozen', paid_prefix=10,
            observed=observed, history_observed_indices=list(range(10)), stages=[
                dict(phase='primary_order', selected=[0]),
                dict(phase='safe_fallback', selected=dict(point=[0]))]))


def fixture(tmp_path):
    item = stop_input()
    frozen = audit.references_from_input(item, stream='stream1', input_file='stop_input.json')
    task = dict(status='complete', seed=7013, domain='fixture', stream='stream1',
        baseline=dict(status='defined', truth=dict(quality=.8)),
        policies=[dict(policy_id=f'P{i:04d}', point=[i]) for i in range(11)], arms=[])
    for arm, backend, indices, deployed in (
            ('legacy_role_after_init_point_only', 'role', [10, 1, 3], 10),
            ('legacy_role_after_init_point_only_native_terminal', 'native', [10, 1, 0], 0),
            (audit.STOP, 'original', [2, 1, 0], 0),
            ('same_source_gp', 'original', [10, 1, 0], 0)):
        current = dict(arm=arm, history_arm='legacy_role_after_init_point_only'
            if backend == 'native' else arm, terminal_backend=backend,
            search_status='complete', recording=dict(status='complete'), observed=[], protocols={})
        for protocol in audit.PROTOCOLS:
            saved_input = deepcopy(item)
            saved_input.update(arm=arm, search_evaluation_count=10 if arm == audit.STOP else 13,
                frozen_shortlist=[dict(point=[i], shortlist_role=f'slot{j+1}') for j, i in enumerate(indices)],
                candidate_budgets=[80, 128, 128] if protocol == 'old' else [80, 120, 120],
                objective_comparison_delta=.05/3 if protocol == 'old' else .05/6)
            filename = f'{arm}_{protocol}.json'
            (tmp_path/filename).write_text(json.dumps(saved_input))
            safety = [dict(candidate_index=0, certified=True, upper_margin=-.1, verification_budget=80),
                dict(candidate_index=1, certified=False, upper_margin=.1, verification_budget=128)]
            if deployed == 0:
                safety.append(dict(candidate_index=2, certified=True, upper_margin=-.1, verification_budget=128))
            execution = dict(status='complete', files=dict(input_file=filename),
                safety=safety, comparisons=[dict(block_index=0, executed=False,
                    status='incumbent_not_safety_certified', challenger_index=0, incumbent_index=1)],
                costs=dict(new_verification_calls=208 if protocol == 'old' else 120))
            current['protocols'][protocol] = dict(status='complete',
                metrics=dict(A=1, C=1, G=-.1 if deployed == 10 else 0., false_certificate=0),
                deployed_policy_id=f'P{deployed:04d}', reported_policy_id=f'P{deployed:04d}',
                no_deployment_reason=None, sets=dict(deployed=dict(max_quality=.7 if deployed == 10 else .8)),
                execution=execution)
        for i in range(10 if arm == audit.STOP else 11):
            slot = indices.index(i)+1 if i in indices else None
            path = dict(status='complete', shortlist_position=slot,
                disposition='not_on_frozen_shortlist' if slot is None else 'included')
            if slot is not None:
                attempt = next((r for r in current['protocols']['old']['execution']['safety']
                    if r['candidate_index'] == slot-1), None)
                path.update(safety_attempted=attempt is not None,
                    safety_passed=None if attempt is None else attempt['certified'], deployed=i == deployed)
            current['observed'].append(dict(policy_id=f'P{i:04d}',
                origin='initial' if i < 10 else 'dynamic', observation_indices=[i],
                truth=dict(quality=.8 if i == 0 else .7,
                    true_feasible=True), selection_record=dict(posterior_rank=i+1,
                    pre_incumbent_slot=3 if i == 0 else slot, post_incumbent_slot=slot,
                    final_slot=slot, disposition='incumbent_insertion_displaced'
                        if i == 0 and slot is None else 'included'),
                protocols={p: deepcopy(path) for p in audit.PROTOCOLS}))
        task['arms'].append(current)
    return task, frozen


def test_freeze_uses_initial_primary_before_slot_transform_and_keeps_all_initial_points():
    frozen = audit.references_from_input(stop_input(), stream='stream1', input_file='input.json')
    assert len(frozen['references']) == 10
    refs = {tuple(row['point']): row for row in frozen['references']}
    assert refs[(0,)]['roles'] == ['stop_primary', 'stop_safe_fallback', 'stop_final_slot3']
    assert refs[(1,)]['roles'] == ['empirical_incumbent', 'stop_final_slot2']
    assert refs[(2,)]['roles'] == ['stop_final_slot1']
    assert refs[(9,)]['roles'] == []


def test_reference_identity_does_not_change_with_truth_or_verification_winner():
    item = stop_input()
    before = audit.references_from_input(item, stream='stream1', input_file='input.json')
    item.update(truth=[dict(point=[9], quality=999., true_feasible=True)],
        verification=dict(deployed=[9], certified=True), deployed=[9])
    assert audit.references_from_input(item, stream='stream1', input_file='input.json') == before


def test_initial_replicates_are_one_physical_reference_and_roles_overlap():
    item = stop_input()
    item['terminal_recording']['observed'][0]['observation_indices'].append(9)
    item['terminal_recording']['observed'].pop()
    frozen = audit.references_from_input(item, stream='stream1', input_file='input.json')
    assert frozen['initial_observation_count'] == 10
    assert frozen['initial_physical_policy_count'] == 9
    assert frozen['references'][0]['observation_indices'] == [0, 9]
    assert len(frozen['references'][0]['roles']) == 3


def test_registration_reads_frozen_stop_input_without_verification_or_truth(tmp_path, monkeypatch):
    seal = dict(status='sealed', cases=[dict(seed=7013, arm=audit.STOP,
        old=dict(input_file='input.json', verification_file='forbidden.json'))])
    calls = []
    def read(path):
        calls.append(str(path))
        if str(path).endswith('execution_status.json'):
            return seal
        if str(path).endswith('input.json'):
            return stop_input()
        raise AssertionError('Registration read a truth or verification artifact')
    monkeypatch.setattr(audit, 'read', read)
    frozen = audit.freeze_references(tmp_path, dict(output_dir='cohort'), 7013, stream='stream1')
    assert frozen['status'] == 'frozen'
    assert len(calls) == 2


def test_existing_retention_coverage_and_quality_net_are_distinct(tmp_path):
    task, frozen = fixture(tmp_path)
    # Native can retain an initial reference while losing a queried policy.
    # Keep both sides of that saved tradeoff; never synthesize a new deployment.
    new_policy = task['arms'][1]['observed'][10]
    new_policy['protocols'] = {p: dict(status='complete', shortlist_position=None,
        disposition='not_on_frozen_shortlist') for p in audit.PROTOCOLS}
    for descriptor in task['arms'][1]['protocols'].values():
        filename = descriptor['execution']['files']['input_file']
        item = json.loads((tmp_path/filename).read_text())
        item['frozen_shortlist'][0]['point'] = [9]
        (tmp_path/filename).write_text(json.dumps(item))
    result = audit.analyze_task(tmp_path, task, frozen)
    assert json.loads(json.dumps(result)) == result
    assert json.loads(json.dumps(frozen)) == frozen
    role, native = result['arms'][:2]
    omitted = role['protocols']['old']['reference_rows'][0]
    rescued = native['protocols']['old']['reference_rows'][0]
    assert omitted['shortlisted'] is False and omitted['attempted'] is False
    assert omitted['pre_incumbent_slot'] == 3 and omitted['post_incumbent_slot'] is None
    assert omitted['quality_minus_actual_deployed'] == pytest.approx(.1)
    assert rescued['shortlisted'] is True and rescued['attempted'] is True
    assert rescued['certified'] is True and rescued['deployed'] is True
    assert rescued['quality_minus_actual_deployed'] == 0.
    assert len(result['references']) == 10 and len(result['arms']) == 4
    assert role['protocols']['old']['reference_counts']['shortlisted'] == 2
    # Groups overlap without duplicating the physical point in the universe.
    assert len(rescued['roles']) == 3
    assert role['protocols']['old']['metrics']['G'] == -.1
    queried_role = role['protocols']['old']['other_observed_policies']
    queried_native = native['protocols']['old']['other_observed_policies']
    assert [row['policy_id'] for row in queried_role] == ['P0010']
    assert queried_role[0]['path']['deployed'] is True
    assert queried_native[0]['path']['disposition'] == 'not_on_frozen_shortlist'
    assert queried_native[0]['truth'] == task['arms'][1]['observed'][10]['truth']


def test_actual_budgets_slots_comparisons_and_costs_remain_separate(tmp_path):
    task, frozen = fixture(tmp_path)
    role = task['arms'][0]
    execution = role['protocols']['new']['execution']
    execution['comparisons'] = [dict(block_index=1, executed=True,
        challenger_index=2, incumbent_index=0, statistics=dict(challenger_dominates=False))]
    result = audit.analyze_task(tmp_path, task, frozen)['arms'][0]['protocols']
    old, new = result['old']['terminal_structure'], result['new']['terminal_structure']
    assert old['candidate_budgets'] == [80, 128, 128]
    assert new['candidate_budgets'] == [80, 120, 120]
    assert new['objective_comparison_delta'] == .05/6
    assert old['costs']['new_verification_calls'] == 208
    assert new['costs']['new_verification_calls'] == 120
    comparison = new['comparisons'][0]
    assert comparison['block_index'] == 1
    assert comparison['challenger_policy_id'] == 'P0003'
    assert result['new']['reference_rows'][3]['comparison_attempted'] is True
    assert result['new']['reference_rows'][1]['comparison_attempted'] is False


def test_no_certificate_failure_and_false_certificate_are_not_reference_benefits(tmp_path):
    task, frozen = fixture(tmp_path)
    descriptor = task['arms'][0]['protocols']['old']
    descriptor.update(metrics=dict(A=0, C=0, G=0., false_certificate=0),
        deployed_policy_id=None, no_deployment_reason='all_safety_failed')
    descriptor['sets']['deployed']['max_quality'] = None
    task['arms'][0]['protocols']['new']['status'] = 'verification_failed'
    reference = task['arms'][1]['observed'][0]
    reference['truth']['true_feasible'] = False
    result = audit.analyze_task(tmp_path, task, frozen)
    old, new = result['arms'][0]['protocols']['old'], result['arms'][0]['protocols']['new']
    assert old['metrics']['C'] == old['metrics']['G'] == 0
    assert old['reference_rows'][0]['quality_minus_actual_deployed'] is None
    assert new['reference_rows'][0]['shortlisted'] is False
    assert new['reference_rows'][0]['attempted'] is None
    assert new['reference_rows'][0]['certified'] is None
    false = result['arms'][1]['protocols']['old']['reference_rows'][0]
    assert false['certified'] is True and false['true_certified'] is False
    assert false['deployed'] is True and false['true_deployed'] is False


def test_new_search_policy_cannot_be_selected_as_an_initial_reference():
    item = stop_input()
    item['terminal_recording']['stages'][0]['selected'] = [10]
    with pytest.raises(ValueError, match='outside the initial physical universe'):
        audit.references_from_input(item, stream='stream1', input_file='input.json')
