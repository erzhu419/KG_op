"""Tiny saved artifacts cover both real adapters and the fixed three-stream unit."""
from copy import deepcopy
import json

import pytest

from performance import mainline_pathway_replication as audit


def write(root, name, value):
    path = root/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root):
    arms = [dict(name=arm, history_arm=arm) for arm in audit.ROLES]
    arms += [dict(name=arm+'_native_terminal', history_arm=arm) for arm in (audit.AFTER, audit.FULL)]
    arms += [dict(name=arm, history_arm=arm) for arm in
             ('same_source_gp', 'legacy_native_neighbors', 'current_structural_mixture', 'initialization_only')]
    plan = dict(terminal_arms=arms, streams=[])
    for stream_index, name in enumerate(('reference', 'observation_1', 'observation_2')):
        ids = {i: f'L{stream_index}_{12-i}' for i in range(12)}
        values = {i: dict(true_feasible=True, objective=1.-q, quality=q)
                  for i, q in enumerate([.5]+[.6]*9+[.7, .8])}
        catalog = [dict(policy_id=ids[i], point=[i], origin='initial' if i < 10 else 'shared_neighbor')
                   for i in range(12)]
        sequences = {audit.AFTER: [10, 10, 11], audit.POINT: [0, 1, 1], audit.FULL: [10, 11, 11]}
        states = []
        for arm, sequence in sequences.items():
            for step, selected in enumerate(sequence):
                seen = set([*range(10), *sequence[:step]])
                mode = 'minimum_violation' if arm == audit.AFTER and step == 2 else 'feasible_objective'
                candidates = [dict(policy_id=ids[i], observed=i in seen,
                    initially_unsampled=i >= 10, true_feasible=True, eligible=mode != 'minimum_violation',
                    selected=i == selected, truth=values[i], origin=catalog[i]['origin'],
                    objective_mean=1.-values[i]['quality'], sampled_objective=.1 if i == selected else .9,
                    sampled_margin=.1 if mode == 'minimum_violation' else -.1,
                    mean_rank_within_eligible=i+1, sampled_rank_within_eligible=1 if i == selected else 2)
                    for i in range(12)]
                count = sum(not c['observed'] and c['eligible'] for c in candidates)
                states.append(dict(arm=arm, paid_prefix=10+step, status='captured', search_status='complete',
                    replay=dict(valid=True, observation_returned=True, returned_policy_matches=True,
                                mode=mode, sampled_feasible_count=12 if mode != 'minimum_violation' else 0),
                    candidates=candidates, selected=candidates[selected], counts=dict(candidates=12,
                        truth_unknown=0, safe_eligible_unobserved=count),
                    safe_competition=dict(status='saved', objective_regret=0.123 if count >= 2 else None),
                    pairs=dict(all=dict(pairs=1, mean_misorder_rate=.25))))
        comparison = dict(contrast='AP', left=audit.AFTER, right=audit.POINT, paid_prefix=10,
            status='available', same_paid_history=True, same_physical_pool=True, query_changed=True,
            first_divergence=True, scope='same_history_same_pool',
            selected={audit.AFTER:ids[10], audit.POINT:ids[0]}, substitutions=[dict(not_reused=True)])
        mechanism = dict(seed=7, domain='fixture', status='complete', policies=catalog,
                         states=states, query_chains=[], comparisons=[comparison])
        path_arms, old_comparisons = [], []
        for cell in arms:
            history = cell['history_arm']
            sequence = sequences.get(history, [10] if history == 'same_source_gp' else [])
            history_points = [*range(10), *sequence]
            chosen = 10 if history in (audit.AFTER, 'same_source_gp') else 11 if history == audit.FULL else 0
            q = values[chosen]['quality']
            gain = q-.6
            metrics = dict(A=1, C=1, false_certificate=0, R=q, G=gain, P=max(gain, 0), L=max(-gain, 0))
            observed = []
            for point in dict.fromkeys(history_points):
                path = dict(status='complete', shortlist_position=1 if point == chosen else 3,
                    deployed=point == chosen, safety_attempted=True, safety_passed=point == chosen,
                    disposition='deployed' if point == chosen else 'safety_certificate_failed',
                    counterfactual_certificate=None, actual_objective_comparisons=[])
                observed.append(dict(policy_id=ids[point], origin=catalog[point]['origin'],
                    observation_indices=[i for i,p in enumerate(history_points) if p == point],
                    truth=values[point], quality_gain_to_initial=values[point]['quality']-.6,
                    improving_safe=values[point]['quality']>.6,
                    protocols={p:deepcopy(path) for p in ('old', 'new')}))
            protocols = {p:dict(status='complete', metrics=deepcopy(metrics),
                deployed_policy_id=ids[chosen], reported_policy_id=ids[chosen],
                no_deployment_reason=None, execution=dict(logical_verification_calls=352)) for p in ('old', 'new')}
            path_arms.append(dict(arm=cell['name'], history_arm=history, observed=observed, protocols=protocols))
            old_comparisons.append(dict(arm=cell['name'], old=deepcopy(metrics), new=deepcopy(metrics),
                protocols={p:dict(status='complete', certified=True, deployed_policy_id=ids[chosen],
                    reported_policy_id=ids[chosen], saved_costs=dict(logical_verification_calls=352))
                    for p in ('old', 'new')}))
        attribution = dict(seed=7, domain='fixture', status='complete', policies=catalog,
            baseline=dict(status='defined', truth=values[9]), arms=path_arms)
        kind = 'observed_terminal' if name == 'reference' else 'decision_path'
        saved = dict(seed=7, shortlists=attribution, comparisons=dict(arms=old_comparisons)) if name == 'reference' else attribution
        write(root, name+'/execution_status.json', dict(status='sealed'))
        write(root, name+'/mechanism/seed_7/analysis.json', mechanism)
        write(root, name+'/case_7.json', saved)
        plan['streams'].append(dict(name=name, batch='reference' if name == 'reference' else 'independent',
            source_output_dir=name, source_manifest=name+'.json', attribution_kind=kind,
            attribution_file_pattern=name+'/case_{seed}.json'))
    return plan


def test_three_saved_streams_keep54_cells_physical_identity_and_no_reanalysis(tmp_path, monkeypatch):
    plan = fixture(tmp_path)
    from performance import mainline_safe_unsampled_competition as competition
    def forbidden(*args, **kwargs):
        raise AssertionError('Stage83 must reuse saved results, not rebuild choices or pairs')
    for name in ('analyze_task', 'pair_records', '_swaps', 'score_choice'):
        monkeypatch.setattr(competition, name, forbidden)
    before = {p: p.stat().st_mtime_ns for p in tmp_path.rglob('*.json')}
    result = audit.analyze_task(tmp_path, plan, 7)
    assert result['status'] == 'complete'
    assert len(result['streams']) == 3 and sum(len(s['outcomes']) for s in result['streams']) == 54
    assert all(len(s['queries']) == 9 for s in result['streams'])
    after = [next(o for o in s['outcomes'] if o['arm'] == audit.AFTER and o['protocol'] == 'old')
             for s in result['streams']]
    assert len({o['deployed_policy_id'] for o in after}) == 1
    assert all(o['discovery_category'] == 'multiple_safe_unobserved_competition' for o in after)
    assert all(o['returned_repeat_count'] == 1 for o in after)
    assert all(next(o for o in s['outcomes'] if o['arm'] == 'same_source_gp')['discovery_category'] ==
               'unavailable_saved_path' for s in result['streams'])
    assert result['streams'][0]['queries'][0]['saved_pair_summary']['all']['mean_misorder_rate'] == .25
    def data_keys(value):
        if isinstance(value, dict):
            return set(value).union(*(data_keys(child) for child in value.values()))
        if isinstance(value, list):
            return set().union(*(data_keys(child) for child in value))
        return set()
    assert not data_keys(result) & {'point', 'points', 'physical_vector', 'physical_vectors', 'substitutions'}
    assert before == {p: p.stat().st_mtime_ns for p in tmp_path.rglob('*.json')}


def test_all_returned_policies_include_minimum_violation_discovery_that_failed_terminal(tmp_path):
    result = audit.analyze_task(tmp_path, fixture(tmp_path), 7)['streams'][1]
    paths = [p for p in result['policy_paths'] if p['history_arm'] == audit.AFTER]
    assert len(paths) == 2
    missed = next(p for p in paths if p['first_query_prefix'] == 12)
    assert missed['discovery_category'] == 'minimum_violation'
    assert missed['returned_repeat_count'] == 0 and missed['improving_safe']
    terminal = next(t for t in missed['terminals'] if t['arm'] == audit.AFTER and t['protocol'] == 'old')
    assert terminal['path']['disposition'] == 'safety_certificate_failed'
    point = next(e for e in missed['exposures'] if e['history_arm'] == audit.POINT)
    assert point['returned_observation_indices'] == []
    assert all(s['present'] for s in point['states'])
    # These paths are descriptive; they do not each receive the deployed P value.
    assert all('metrics' not in p for p in paths)


def test_classification_needs_actual_return_and_complete_truth_for_safe_competition(tmp_path):
    fixture(tmp_path)
    state = audit.read(tmp_path/'reference/mechanism/seed_7/analysis.json')['states'][0]
    assert audit.classify_first_query(state) == 'multiple_safe_unobserved_competition'
    state['replay']['observation_returned'] = False
    assert audit.classify_first_query(state) == 'unknown'
    state['replay']['observation_returned'] = True
    state['candidates'][-1]['truth'] = None
    assert audit.classify_first_query(state) == 'unknown'
    state['replay']['mode'] = 'minimum_violation'
    assert audit.classify_first_query(state) == 'minimum_violation'
    state['replay']['returned_policy_matches'] = False
    assert audit.classify_first_query(state) == 'unknown'


def test_single_safe_initial_and_native_history_categories_remain_distinct(tmp_path):
    result = audit.analyze_task(tmp_path, fixture(tmp_path), 7)['streams'][0]
    rows = {(o['arm'], o['protocol']):o for o in result['outcomes']}
    assert rows[audit.FULL, 'old']['discovery_category'] == 'single_safe_unobserved_competition'
    assert rows[audit.POINT, 'old']['discovery_category'] == 'initial'
    assert rows[audit.POINT, 'old']['returned_repeat_count'] == 1
    repeated_initial = next(p for p in result['policy_paths'] if p['history_arm'] == audit.POINT and
                            p['observation_indices'] == [1, 11, 12])
    assert repeated_initial['discovery_category'] == 'initial'
    assert repeated_initial['returned_repeat_count'] == 2
    assert rows['initialization_only', 'old']['discovery_category'] == 'initial'
    assert rows['initialization_only', 'old']['returned_repeat_count'] == 0
    assert rows[audit.AFTER+'_native_terminal', 'old']['discovery_category'] == rows[audit.AFTER, 'old']['discovery_category']
    comparison = next(c for c in result['comparisons'] if c['contrast'] == 'AP' and c['protocol'] == 'old')
    assert comparison['positive_gain_component'] == pytest.approx(.1)
    assert comparison['avoided_loss_component'] == pytest.approx(.1)
    assert comparison['difference']['G'] == pytest.approx(.2)
    assert comparison['decomposition_residual'] == pytest.approx(0)


def test_uncertified_and_unknown_baseline_do_not_become_zero_quality(tmp_path):
    plan = fixture(tmp_path)
    path = tmp_path/'observation_1/case_7.json'
    saved = audit.read(path)
    after = next(a for a in saved['arms'] if a['arm'] == audit.AFTER)
    after['protocols']['old']['metrics'].update(A=0, C=0, R=0, G=0, P=0, L=0)
    after['protocols']['old']['no_deployment_reason'] = 'all_attempted_safety_certificates_failed'
    path.write_text(json.dumps(saved))
    path = tmp_path/'observation_2/case_7.json'
    saved = audit.read(path)
    saved['baseline'] = dict(status='no_true_feasible_initial', truth=None)
    for arm in saved['arms']:
        for p in arm['protocols'].values():p['metrics'].update(G=None, P=None, L=None)
    path.write_text(json.dumps(saved))
    result = audit.analyze_task(tmp_path, plan, 7)
    uncertified = next(o for o in result['streams'][1]['outcomes'] if o['arm'] == audit.AFTER and o['protocol'] == 'old')
    assert uncertified['metrics']['G'] == 0 and uncertified['actual_deployed_quality'] is None
    assert uncertified['deployed_policy_id'] is None and uncertified['reported_policy_id'] is not None
    assert uncertified['discovery_category'] == 'no_true_certified_deployment'
    assert result['streams'][2]['comparisons'][0]['positive_gain_component'] is None
    assert result['streams'][2]['comparisons'][0]['avoided_loss_component'] is None


def test_failure_control_is_retained_and_unsealed_source_blocks_reader(tmp_path):
    plan = fixture(tmp_path)
    path = tmp_path/'observation_1/case_7.json'
    saved = audit.read(path)
    failed = next(a for a in saved['arms'] if a['arm'] == 'same_source_gp')['protocols']['old']
    failed.update(status='failed', metrics=dict.fromkeys(audit.METRICS), deployed_policy_id=None)
    path.write_text(json.dumps(saved))
    result = audit.analyze_task(tmp_path, plan, 7)
    row = next(o for o in result['streams'][1]['outcomes'] if o['arm'] == 'same_source_gp' and o['protocol'] == 'old')
    assert row['status'] == 'failed' and row['metrics']['G'] is None
    assert len(result['streams'][1]['outcomes']) == 18
    write(tmp_path, 'reference/execution_status.json', dict(status='running'))
    with pytest.raises(ValueError, match='sealed'):
        audit.analyze_task(tmp_path, plan, 7)
