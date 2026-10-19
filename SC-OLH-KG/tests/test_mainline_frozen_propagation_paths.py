"""Small saved-record fixtures for physical identity, unknowns, and attribution."""
from copy import deepcopy
import json

import numpy as np
import pytest

from performance import mainline_frozen_propagation_paths as paths


def save(root, path, value):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value))


def stage(prefix, selected, count=14):
    mean = [float(i) for i in range(count)]
    sampled = [float(i + 10) for i in range(count)]
    sampled[selected] = -1.
    margin = [-1.] * (count - 1) + [1.]
    _, total, mode, feasible = paths.score_choice(sampled, margin)
    score = {name: [0.] * count for name in paths.SCORE_FIELDS}
    score.update(objective_mean=mean, sampled_objective=sampled, sampled_margin=margin,
                 sampled_feasibility_mode=mode, sampled_feasible_count=feasible,
                 total=total.tolist())
    return dict(paid_prefix=prefix, action_ids=list(range(count)),
        action_sources=['fixture'] * count, selected_index=selected, selected_id=selected,
        observed_ids=list(range(prefix)), score=score)


def bundle(queries=(10, 11, 11)):
    history = [dict(point=[i], raw=[float(i), -1.]) for i in range(10)]
    history += [dict(point=[i], raw=[float(i), -1.]) for i in queries]
    return dict(history=history, bank=[(i,) for i in range(14)],
                audit=dict(stages=[stage(prefix, chosen) for prefix, chosen in zip((10, 11, 12), queries)]))


def selection_record(history, shortlist, arm, context):
    points = list(dict.fromkeys(tuple(row['point']) for row in history))
    observed = [dict(point=list(point), observation_indices=[i for i, row in enumerate(history)
        if tuple(row['point']) == point], replicate_count=sum(tuple(row['point']) == point for row in history))
        for point in points]
    pool = [list(point) for point in points]
    components = dict(objective=[float(point[0]) for point in points], probability_violation=[.1] * len(points))
    primary = shortlist[0]['point']
    ranked = [primary] + [point for point in pool if point != primary]
    phases = [dict(phase='start'),
        dict(phase='primary_pool', pool=pool, components=components, available_components=list(components)),
        dict(phase='primary_order', order=[pool.index(point) for point in ranked], ranked_points=ranked,
             selected=primary, maximum_violation_probability=.5),
        dict(phase='objective_gate', pool=pool, components=components,
             available_components=list(components), maximum_violation_probability=.5),
        dict(phase='objective_choice', selected=dict(point=shortlist[1]['point'])),
        dict(phase='safe_pool', pool=pool, components=components, available_components=list(components),
             probability_slack=.05, candidate_universe='frozen_observed_history'),
        dict(phase='safe_fallback', selected=dict(point=shortlist[2]['point'])),
        dict(phase='pre_incumbent', shortlist=shortlist, primary=primary),
        dict(phase='post_incumbent', shortlist=shortlist, incumbent_audit={}),
        dict(phase='frozen_shortlist', shortlist=shortlist),
        dict(phase='same_policy_transform', before=shortlist, after=shortlist, changed=False,
             incumbent_position_before=2, incumbent_position_after=2)]
    return dict(status='frozen', paid_prefix=len(history), backend_arm=arm, posterior_context=context,
        observed=observed, history_observed_indices=[points.index(tuple(row['point'])) for row in history], stages=phases)


def fixture(root):
    names = [paths.FULL, paths.FULL + '_native_terminal', paths.AFTER,
        paths.AFTER + '_native_terminal', 'legacy_native_neighbors', 'same_source_gp',
        paths.POINT, 'current_structural_mixture', paths.STOP]
    cells = [dict(name=name, history_arm=name.removesuffix('_native_terminal'),
        terminal_backend='native' if name.endswith('_native_terminal') else 'role') for name in names]
    plan = dict(reference_manifest='source.json', reference_output_dir='source', terminal_arms=cells,
                terminal_variants=list(paths.VARIANTS), tasks=[dict(geometry_task_seed=7, domain='fixture')])
    source = dict(output_dir='source', tasks=plan['tasks'], terminal_arms=cells)
    save(root, 'source.json', source)
    truth = {(i,): dict(quality=.5 if i == 0 else .4, objective=-.5 if i == 0 else -.4,
                        true_feasible=True) for i in range(10)}
    truth[(10,)] = dict(quality=.7, objective=-.7, true_feasible=True)
    # Policy11 was queried, but the frozen truth union did not include it.
    save(root, 'source/truth/seed_7/truth.json', [dict(point=list(point), truth=value) for point, value in truth.items()])
    frozen, sealed, scored = [], [], []
    for cell in cells:
        arm, history_arm = cell['name'], cell['history_arm']
        b = bundle((0, 1, 1) if history_arm == paths.POINT else (10, 11, 11))
        if history_arm == paths.STOP:
            b['history'] = b['history'][:10]
        source_case = 'search/' + history_arm
        save(root, source_case + '/search.json', dict(history=b['history']))
        if history_arm != paths.STOP:
            save(root, source_case + '/query_audit.json', b['audit'])
            np.savez(root / source_case / 'query_policies.npz', points=np.asarray(b['bank']))
        frozen.append(dict(seed=7, arm=arm, history_arm=history_arm, source_case=source_case))
        chosen = 1 if history_arm == paths.POINT else 0 if history_arm == paths.STOP else 10
        shortlist = [dict(point=[p], shortlist_position=i + 1) for i, p in
            enumerate((0, 1, 3) if chosen != 10 else (10, 0, 3))]
        native = cell['terminal_backend'] == 'native' or history_arm in ('legacy_native_neighbors', paths.STOP)
        backend = 'legacy_native_v69' if history_arm == 'legacy_native_neighbors' else history_arm
        recording = selection_record(b['history'], shortlist, backend, 0 if native else 1)
        g = -.1 if chosen == 1 else .2 if chosen == 10 else 0.
        for variant in paths.VARIANTS:
            item = dict(frozen_shortlist=deepcopy(shortlist), terminal_recording=deepcopy(recording),
                        reference_coverage=dict(reference_shortlist_changed=True, selected_initial_rank=1))
            if variant.startswith('reference_'):
                item['frozen_shortlist'][2]['point'] = [4]
            report = dict(certified=True, objective_incumbent_position=2,
                attempts=[dict(candidate_index=i, certified=i == 0 or i == 1, upper_margin=-.1 if i < 2 else .1)
                          for i in range(3 if variant.endswith('two_comparison') else 2)])
            comparison = dict(status='challenger_dominates' if chosen == 10 else 'challenger_not_dominant',
                              challenger_dominates=chosen == 10,
                              sample_mean_difference=.1, one_sided_upper_confidence_bound=.2)
            if variant.endswith('two_comparison'):
                report['objective_comparisons'] = [dict(comparison, block_index=0,
                    challenger_index=0, incumbent_index=1, winner_index=1)]
            else:
                report['objective_comparison'] = comparison
            input_path, result_path = f'inputs/{arm}/{variant}.json', f'results/{arm}/{variant}.json'
            save(root, input_path, item)
            save(root, result_path, dict(deployed=[chosen], verification=report))
            sealed.append(dict(seed=7, domain='fixture', arm=arm, history_arm=history_arm,
                variant=variant, status='complete', input_file=input_path, result_file=result_path))
            scored.append(dict(seed=7, domain='fixture', arm=arm, history_arm=history_arm,
                variant=variant, status='complete', metrics=dict(A=1, C=1, false_certificate=0,
                    R=.5 + g, G=g, P=max(g, 0), L=max(-g, 0)),
                quality_reference_status='defined', best_initial_quality=.5,
                safety_attempts=report['attempts']))
    save(root, 'source/execution_status.json', dict(status='sealed', truth_access_allowed=True,
        registered_terminal_cells=36, cells=sealed))
    save(root, 'source/frozen_inputs.json', dict(rows=frozen))
    save(root, 'source/truth/seed_7/scored.json', dict(domain='fixture', outcomes=scored))
    return plan


def test_read_only_reader_keeps36_cells_physical_repeats_and_unknown_truth(tmp_path, monkeypatch):
    plan = fixture(tmp_path)
    monkeypatch.setattr(np.random, 'default_rng', lambda *a, **k: pytest.fail('No random stream is admitted'))
    before = {str(path): path.stat().st_mtime_ns for path in tmp_path.rglob('*') if path.is_file()}
    result = paths.analyze_task(tmp_path, plan, 7)
    assert len(result['outcomes']) == len(result['cells']) == 36
    assert len(result['histories']) == 7
    assert all(row['original_selection_record_status'] == 'complete' for row in result['cells'])
    unknown = next(row for row in result['policies'] if row['truth_status'] == 'unknown')
    assert unknown['truth'] is None
    assert any(row['observation_index'] == 11 for row in unknown['observed_locations'])
    assert any(row['observation_index'] == 12 for row in unknown['observed_locations'])
    after = next(row for row in result['histories'] if row['history_arm'] == paths.AFTER)
    assert after['observed_truth']['max_quality'] is None
    assert all(row['pool_truth']['max_true_safe_quality'] is None for row in after['queries'])
    assert all(row['selection_replay']['matched'] for row in after['queries'])
    assert next(row for row in after['policy_paths'] if row['policy_id'] == unknown['policy_id'])[
        'adaptive_query_history_indices'] == [11, 12]
    assert before == {str(path): path.stat().st_mtime_ns for path in tmp_path.rglob('*') if path.is_file()}
    serialized = json.dumps(result)
    assert '"point"' not in serialized and '"raw"' not in serialized


def test_signed_quality_decomposition_separates_gain_from_avoided_loss():
    left = dict(G=.3, P=.3, L=0, C=1)
    right = dict(G=-.2, P=0, L=.2, C=1)
    result = paths.metric_difference(left, right)
    assert result['difference']['G'] == pytest.approx(.5)
    assert result['positive_gain_component'] == pytest.approx(.3)
    assert result['avoided_loss_component'] == pytest.approx(.2)
    assert result['decomposition_residual'] == pytest.approx(0)
    reverse = paths.metric_difference(right, left)
    assert reverse['positive_gain_component'] == pytest.approx(-.3)
    assert reverse['avoided_loss_component'] == pytest.approx(-.2)
    assert paths.metric_difference(dict(G=None, P=None, L=None), right)['difference']['G'] is None


def test_no_certificate_keeps_metric_zero_but_deployed_quality_unknown(tmp_path):
    plan = fixture(tmp_path)
    score_path = tmp_path / 'source/truth/seed_7/scored.json'
    scores = paths.read(score_path)
    selected = next(row for row in scores['outcomes'] if row['arm'] == paths.AFTER and row['variant'] == 'original_old')
    selected['metrics'].update(A=0, C=0, R=0, G=0, P=0, L=0)
    save(tmp_path, 'source/truth/seed_7/scored.json', scores)
    result_path = f'results/{paths.AFTER}/original_old.json'
    record = paths.read(tmp_path / result_path)
    record['verification']['certified'] = False
    for attempt in record['verification']['attempts']:
        attempt['certified'] = False
    save(tmp_path, result_path, record)
    result = paths.analyze_task(tmp_path, plan, 7)
    row = next(row for row in result['outcomes'] if row['arm'] == paths.AFTER and row['variant'] == 'original_old')
    assert row['metrics']['G'] == row['metrics']['C'] == 0
    assert row['actual_deployed_quality'] is None and row['deployed_policy_id'] is None
    assert row['reported_policy_truth']['quality'] == .7
    assert row['newly_queried_deployment'] is False


def test_only_same_state_first_divergence_can_attribute_saved_score_difference():
    bundles = {paths.AFTER: bundle(), paths.POINT: bundle((0, 1, 1))}
    registry = {(i,): str(i) for i in range(14)}
    comparison = paths.compare_queries(paths.AFTER, paths.POINT, bundles, registry)
    first, later, last = comparison['states']
    assert first['first_divergence'] and first['common_state_first_divergence']
    assert first['same_paid_history'] and first['same_physical_pool']
    assert later['same_paid_history'] is False and later['common_state_first_divergence'] is False
    assert later['interpretation'] == 'later_histories_or_pools_differ'
    assert last['earlier_query_divergence']
    assert comparison['initialization_propagation_delta'] is None


def test_risk_exclusion_objective_competition_and_missing_scores_remain_distinct():
    b = bundle()
    risky = paths.saved_candidate_trace((13,), b)['states'][0]
    eligible = paths.saved_candidate_trace((12,), b)['states'][0]
    selected = paths.saved_candidate_trace((10,), b)['states'][0]
    assert risky['reason'] == 'risk_excluded' and risky['sampled_feasible'] is False
    assert eligible['reason'] == 'objective_competition' and eligible['sampled_feasible'] is True
    assert selected['selected'] and selected['previously_observed'] is False
    b['audit']['stages'][0]['score'].pop('objective_mean')
    partial = paths.saved_candidate_trace((10,), b)['states'][0]
    assert partial['status'] == 'partial_saved_scores' and partial['objective_mean'] is None
    assert partial['initialization_propagation_delta'] is None


def test_old_third_not_tested_and_two_comparison_safety_failure_are_not_conflated(tmp_path):
    result = paths.analyze_task(tmp_path, fixture(tmp_path), 7)
    cells = {row['variant']: row for row in result['cells'] if row['arm'] == paths.AFTER}
    third = cells['original_old']['shortlist_policy_ids'][2]
    old = next(row for row in cells['original_old']['observed'] if row['policy_id'] == third)['terminal_path']
    two = next(row for row in cells['original_two_comparison']['observed'] if row['policy_id'] == third)['terminal_path']
    assert old['disposition'] == 'not_verified_after_earlier_terminal_decision'
    assert old['safety_attempted'] is False and old['counterfactual_certificate'] is None
    assert two['disposition'] == 'safety_certificate_failed' and two['safety_attempted']
    replaced = next(row for row in cells['reference_old']['observed'] if row['policy_id'] == third)['terminal_path']
    assert replaced['disposition'] == 'not_on_frozen_shortlist'
    assert replaced['counterfactual_objective_comparison'] is None


def test_unsealed_cohort_or_missing_registered_cell_blocks_attribution(tmp_path):
    plan = fixture(tmp_path)
    path = tmp_path / 'source/execution_status.json'
    seal = paths.read(path)
    seal['truth_access_allowed'] = False
    save(tmp_path, 'source/execution_status.json', seal)
    with pytest.raises(ValueError, match='admitted target truth'):
        paths.analyze_task(tmp_path, plan, 7)
    seal['truth_access_allowed'] = True
    seal['cells'].pop()
    save(tmp_path, 'source/execution_status.json', seal)
    with pytest.raises(ValueError, match='registered terminal cell'):
        paths.analyze_task(tmp_path, plan, 7)
