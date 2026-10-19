"""Small saved-data substitutes for physical joins and actual terminal coverage."""
from copy import deepcopy
import json

import numpy as np
import pytest

from performance import mainline_precision_propagation_paths as paths
from performance.mainline_pathway_replication import _query
from test_mainline_frozen_propagation_paths import selection_record


def save(root, filename, value):
    target = root / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value))


def fixture(root):
    names = [paths.AFTER, paths.POINT, paths.FULL, paths.AFTER + '_native_terminal',
        paths.FULL + '_native_terminal', 'legacy_native_neighbors', 'same_source_gp',
        'current_structural_mixture', paths.STOP]
    configs = [dict(name=arm, history_arm=arm.removesuffix('_native_terminal'),
        terminal_backend='native' if arm.endswith('_native_terminal') else 'role') for arm in names]
    plan = dict(source_manifest='source.json', source_output_dir='source', terminal_arms=configs,
        tasks=[dict(geometry_task_seed=7, domain='fixture')],
        streams=[dict(name='observation_1'), dict(name='observation_2')])
    save(root, 'source.json', dict(output_dir='source', **{key: plan[key] for key in ('tasks', 'streams', 'terminal_arms')}))
    save(root, 'source/mechanism/inventory.json', dict(status='sealed', truth_access_allowed=True, registered_terminal_cells=18))
    inventory = 'source/mechanism/seed_7/policies.npz'
    (root / inventory).parent.mkdir(parents=True, exist_ok=True)
    np.savez(root / inventory, points=np.arange(14).reshape(-1, 1))
    truth = {i: dict(objective=-.7 if i == 11 else -.6 if i == 0 else -.5,
        quality=.7 if i == 11 else .6 if i == 0 else .5, true_feasible=True) for i in range(14)}
    save(root, 'source/cases/seed_7/truth.json', [dict(point=[i], truth=value) for i, value in truth.items()])
    streams, entries = [], []
    for stream in plan['streams']:
        name, cases, outcomes, histories, states = stream['name'], [], [], {}, []
        for config in configs:
            arm, history_arm = config['name'], config['history_arm']
            chosen = (0, 0, 0) if history_arm == paths.POINT else (10, 11, 11)
            if history_arm == 'same_source_gp':
                chosen = (10, 11, 99)  # Existing control observation outside the scored role union.
            history = [dict(point=[i], raw=[float(i), -1.]) for i in range(10)]
            if history_arm != paths.STOP:
                history += [dict(point=[i], raw=[float(i), -1.]) for i in chosen]
            histories[history_arm] = history
            source_case = f'search/{name}/{history_arm}'
            save(root, source_case + '/search.json', dict(history=history))
            if history_arm in paths.ROLES:
                audit = dict(stages=[dict(paid_prefix=prefix, action_ids=list(range(14)), score=dict(
                    total=[float(i) for i in range(14)], probability_violation=[.1] * 14,
                    stochastic_margin_mean=[-.2] * 14)) for prefix in (10, 11, 12)])
                save(root, source_case + '/query_audit.json', audit)
            shortlist = [dict(point=[i], posterior_rank=j + 1) for j, i in enumerate(
                (0, 1, 2) if history_arm in (paths.POINT, paths.STOP) else (10, 1, 0))]
            backend = 'legacy_native_v69' if history_arm == 'legacy_native_neighbors' else history_arm
            context = 0 if config['terminal_backend'] == 'native' or history_arm in (paths.STOP, 'legacy_native_neighbors') else 1
            recording = selection_record(history, shortlist, backend, context)
            item = dict(frozen_shortlist=shortlist, objective_incumbent_position=2, terminal_recording=recording)
            selected = shortlist[0]['point'][0]
            comparison = dict(status='challenger_dominates', simulation_calls=128,
                challenger_dominates=True, sample_mean_difference=-.1, one_sided_upper_confidence_bound=-.01)
            record = dict(deployed=[selected], new_verification_calls=336, verification=dict(
                certified=True, objective_incumbent_position=2, attempts=[
                    dict(candidate_index=i, certified=True, upper_margin=-.1) for i in (0, 1)],
                objective_comparison=comparison))
            input_file, verification_file = f'inputs/{name}/{arm}.json', f'verification/{name}/{arm}.json'
            save(root, input_file, item)
            save(root, verification_file, record)
            cases.append(dict(seed=7, domain='fixture', arm=arm, history_arm=history_arm, source_case=source_case,
                old=dict(status='complete', input_file=input_file, verification_file=verification_file)))
            gain = truth[selected]['quality'] - .6
            outcomes.append(dict(arm=arm, history_arm=history_arm, protocol='old', status='complete',
                metrics=dict(A=1, C=1, false_certificate=0, R=truth[selected]['quality'], G=gain, P=max(gain, 0), L=max(-gain, 0)),
                quality_reference_status='defined', best_initial_quality=.6, reported_policy_id=f'P{selected:04d}',
                deployed_policy_id=f'P{selected:04d}', actual_deployed_quality=truth[selected]['quality']))
        for arm in paths.ROLES:
            history = histories[arm]
            for prefix in (10, 11, 12):
                selected = history[prefix]['point'][0]
                observed = {row['point'][0] for row in history[:prefix]}
                candidates = []
                for i in range(14):
                    eligible = not (arm == paths.POINT and i == 10)
                    candidates.append(dict(policy_id=f'L{i}', bank_id=i, observed=i in observed,
                        initially_unsampled=i >= 10, eligible=eligible, selected=i == selected,
                        true_feasible=True, truth=truth[i], origin='dynamic' if i >= 10 else 'initial',
                        objective_mean=float(i) / 10 + (0.1 if arm == paths.POINT else 0.),
                        sampled_objective=-1. if i == selected else float(i), sampled_margin=-.1 if eligible else .1,
                        objective_epistemic=.1, constraint_mean=0., constraint_aleatoric=.1,
                        constraint_epistemic=.2, sampled_constraint=-.2,
                        mean_rank_within_eligible=i + 1 if eligible else None,
                        sampled_rank_within_eligible=1 if i == selected else i + 2 if eligible else None))
                states.append(dict(arm=arm, paid_prefix=prefix, status='captured', candidates=candidates,
                    selected=deepcopy(candidates[selected]), counts=dict(safe_eligible_unobserved=3),
                    replay=dict(valid=True, observation_returned=True, returned_policy_matches=True,
                                mode='feasible_objective')))
        comparisons = [dict(contrast=label, left=left, right=right, paid_prefix=prefix, status='available',
            same_paid_history=prefix == 10, same_physical_pool=True, both_observations_returned=True,
            query_changed=left == paths.AFTER, first_divergence=left == paths.AFTER and prefix == 10,
            selected={arm: f'L{histories[arm][prefix]["point"][0]}' for arm in (left, right)})
            for label, left, right in (('AP', paths.AFTER, paths.POINT), ('FA', paths.FULL, paths.AFTER))
            for prefix in (10, 11, 12)]
        raw = dict(states=states, comparisons=comparisons, policies=[dict(policy_id=f'L{i}', point=[i]) for i in range(14)])
        save(root, f'source/cases/seed_7/{name}/analysis.json', raw)
        queries = [_query(state, {f'L{i}': f'P{i:04d}' for i in range(14)}) for state in states]
        # Stage85 attaches the first-returned discovery class also to later repeats.
        for query in queries:
            if query['selected_policy_id'] in ('P0010', 'P0011'):
                query['discovery_category'] = paths.MULTI
        streams.append(dict(name=name, status='complete', outcomes=outcomes, queries=queries,
            policy_paths=[], coverage=dict(captured_states=9), quality_reference_status='defined'))
        entries.append(dict(name=name, cases=cases, status='frozen'))
    save(root, 'source/mechanism/seed_7/index.json', dict(union_policies_file=inventory, streams=entries))
    save(root, 'source/cases/seed_7/result.json', dict(seed=7, domain='fixture', status='complete', streams=streams))
    return plan


def test_saved_end_to_end_all_controls_identity_common_state_and_terminal_omissions(tmp_path, monkeypatch):
    plan = fixture(tmp_path)
    monkeypatch.setattr(np.random, 'default_rng', lambda *a, **k: pytest.fail('No RNG admitted'))
    before = {str(path): path.stat().st_mtime_ns for path in tmp_path.rglob('*') if path.is_file()}
    result = paths.analyze_task(tmp_path, plan, 7)
    assert result['status'] == 'complete' and len(result['streams']) == 2
    for stream in result['streams']:
        assert len(stream['outcomes']) == len(stream['policy_terminal_paths']) == 9
        assert all(cell['selection_record_status'] == 'complete' for cell in stream['policy_terminal_paths'])
        init = stream['initialization_contrast']
        assert init['common_state_first_divergence'] and init['initialization_propagation_delta'] is None
        focal = next(row for row in init['focal_policies'] if row['policy_id'] == 'P0010')
        assert focal['eligibility_transition'] == 'left_only_eligible'
        assert focal['saved_left_minus_right']['objective_mean'] == pytest.approx(-.1)
        assert focal['left']['total'] == 10 and focal['right']['probability_violation'] == .1
        assert not any(row['common_state_first_divergence'] for row in stream['query_comparisons'] if row['paid_prefix'] > 10)
        cell = next(row for row in stream['policy_terminal_paths'] if row['arm'] == paths.AFTER)
        good = next(row for row in cell['observed'] if row['policy_id'] == 'P0011')
        assert good['quality_gain_to_initial'] == pytest.approx(.1)
        assert good['terminal_path']['terminal_reason'] == 'not_on_frozen_shortlist'
        third = next(row for row in cell['observed'] if row['policy_id'] == 'P0000')
        assert third['initially_observed'] and third['terminal_path']['shortlist_position'] == 3
        assert third['terminal_path']['terminal_reason'] == 'not_verified_after_earlier_terminal_decision'
        assert third['terminal_path']['objective_comparison_coverage'] == []
        gp = next(row for row in stream['policy_terminal_paths'] if row['arm'] == 'same_source_gp')
        unknown = next(row for row in gp['observed'] if row['policy_id'].startswith('H'))
        assert unknown['truth'] is None and gp['final_observed_pool_truth']['max_true_safe_quality'] is None
        events = [row for row in stream['multi_safe_events'] if row['arm'] == paths.AFTER]
        assert len(events) == 3 and len({row['policy_id'] for row in events}) == 2
        assert events[-1]['query_action_kind'] == 'repeat'
    assert before == {str(path): path.stat().st_mtime_ns for path in tmp_path.rglob('*') if path.is_file()}
    encoded = json.dumps(result)
    assert '"point"' not in encoded and '"raw"' not in encoded


@pytest.mark.parametrize('executed', [False, True])
def test_unexecuted_objective_branch_is_not_a_failed_comparison(tmp_path, executed):
    plan = fixture(tmp_path)
    path = tmp_path / f'verification/observation_1/{paths.AFTER}.json'
    record = json.loads(path.read_text())
    record['deployed'] = [1]
    record['verification']['attempts'][0]['certified'] = executed
    record['verification']['objective_comparison'] = (dict(status='challenger_not_dominant', simulation_calls=128,
        challenger_dominates=False, sample_mean_difference=-.01, one_sided_upper_confidence_bound=.03)
        if executed else dict(status='challenger_not_safety_certified', simulation_calls=0))
    path.write_text(json.dumps(record))
    filename = 'source/cases/seed_7/result.json'
    saved = paths.read(tmp_path / filename)
    outcome = next(row for row in saved['streams'][0]['outcomes'] if row['arm'] == paths.AFTER)
    outcome.update(reported_policy_id='P0001', deployed_policy_id='P0001')
    save(tmp_path, filename, saved)
    result = paths.analyze_task(tmp_path, plan, 7)
    cell = next(row for row in result['streams'][0]['policy_terminal_paths'] if row['arm'] == paths.AFTER)
    challenger = next(row for row in cell['observed'] if row['policy_id'] == 'P0010')['terminal_path']
    assert challenger['safety_passed'] == executed
    assert challenger['terminal_reason'] == ('actual_objective_comparison_retained_other_policy'
        if executed else 'safety_certificate_failed')
    assert challenger['objective_comparison_coverage'][0]['executed'] == executed


def test_no_certificate_null_baseline_and_path_failure_preserve_scientific_outcomes(tmp_path):
    plan = fixture(tmp_path)
    filename = 'source/cases/seed_7/result.json'
    saved = paths.read(tmp_path / filename)
    after = next(row for row in saved['streams'][0]['outcomes'] if row['arm'] == paths.AFTER)
    after.update(metrics=dict(A=0, C=0, false_certificate=0, R=0, G=None, P=None, L=None),
        best_initial_quality=None, actual_deployed_quality=None, deployed_policy_id=None,
        quality_reference_status='no_true_feasible_initial')
    save(tmp_path, filename, saved)
    (tmp_path / 'source/cases/seed_7/observation_2/analysis.json').write_text('{invalid')
    result = paths.analyze_task(tmp_path, plan, 7)
    assert result['status'] == 'complete_with_retained_path_failures'
    assert len(result['streams'][1]['outcomes']) == 9
    assert result['streams'][1]['status'] == 'path_analysis_failed'
    assert result['streams'][0]['comparisons'][0]['difference']['G'] is None
    assert result['streams'][0]['outcomes'][0]['actual_deployed_quality'] is None


def test_incomplete_global_seal_blocks_reading_saved_truth(tmp_path):
    plan = fixture(tmp_path)
    save(tmp_path, 'source/mechanism/inventory.json', dict(status='sealed', truth_access_allowed=True,
        registered_terminal_cells=17))
    with pytest.raises(ValueError, match='global seal'):
        paths.analyze_task(tmp_path, plan, 7)
