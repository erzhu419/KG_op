"""Read saved terminal selection records and actual outcomes, without inference."""
from collections import Counter
from pathlib import Path

from performance.mainline_saved_terminal_comparison import comparison_diagnostics, comparison_slots
from performance.run_mainline_legacy_backend_bridge import read


PHASE_FIELDS = {
    'primary_pool': ('pool', 'components', 'available_components'),
    'primary_order': ('order', 'ranked_points', 'selected', 'maximum_violation_probability'),
    'objective_gate': ('pool', 'components', 'available_components', 'maximum_violation_probability'),
    'objective_choice': ('selected',),
    'safe_pool': ('pool', 'components', 'available_components', 'probability_slack', 'candidate_universe'),
    'safe_fallback': ('selected',),
    'pre_incumbent': ('shortlist', 'primary'), 'post_incumbent': ('shortlist', 'incumbent_audit'),
    'frozen_shortlist': ('shortlist',),
    'same_policy_transform': ('before', 'after', 'changed', 'incumbent_position_before', 'incumbent_position_after'),
}


def points(rows):
    return [tuple(row['point']) for row in rows]


def audit_record(record, history, final_shortlist, expected_context, expected_backend_arm=None):
    """Validate physical identities and saved selection chain; never rank anew."""
    issues = []
    result = dict(status='missing_record' if record is None else 'invalid_record',
        issues=issues, observed=[], scope='Saved objective order is not a target-truth ranking.',
        disposition_scope='Shortlist membership and challenger eligibility; ineligibility is not a unique omission cause because safe fallback can bypass the objective gate.')
    if record is None:
        issues.append('terminal_recording_missing')
        return result
    result['record_status'] = record.get('status')
    for name in ('status', 'observed', 'history_observed_indices', 'paid_prefix', 'posterior_context', 'stages'):
        if name not in record:
            issues.append('missing:' + name)
    if issues:
        return result
    if record['status'] != 'frozen':
        issues.append('record_not_frozen')
    if record['posterior_context'] != expected_context:
        issues.append('posterior_context_mismatch')
    if expected_backend_arm is not None and record.get('backend_arm') != expected_backend_arm:
        issues.append('backend_arm_mismatch')
    history_points = points(history)
    observed = list(dict.fromkeys(history_points))
    expected = [dict(point=list(p), observation_indices=[i for i, x in enumerate(history_points) if x == p],
        replicate_count=history_points.count(p)) for p in observed]
    if record['observed'] != expected:
        issues.append('observed_history_physical_or_repeat_mismatch')
    if record['paid_prefix'] != len(history) or record['history_observed_indices'] != [observed.index(p) for p in history_points]:
        issues.append('paid_prefix_or_observation_indices_mismatch')
    stages = {stage['phase']: stage for stage in record['stages']}
    expected_phases = ['start', *PHASE_FIELDS]
    if [stage['phase'] for stage in record['stages']] != expected_phases:
        issues.append('selection_stage_chain_incomplete_or_reordered')
    for phase, fields in PHASE_FIELDS.items():
        for field in fields:
            if field not in stages.get(phase, {}):
                issues.append('missing:' + phase + '.' + field)
    if any(issue.startswith('missing:') for issue in issues):
        return result
    for phase in ('primary_pool', 'objective_gate', 'safe_pool'):
        stage = stages[phase]
        pool, components = [tuple(p) for p in stage['pool']], stage['components']
        expected_pool = observed
        pool_issue = phase + ':incomplete_observed_physical_pool'
        if phase == 'safe_pool':
            # This frozen protocol's initial atlas is the first ten paid observations.
            if stage['candidate_universe'] == 'frozen_initial_atlas':
                expected_pool = list(dict.fromkeys(history_points[:10]))
                pool_issue = 'safe_pool:incomplete_initial_physical_pool'
            elif stage['candidate_universe'] != 'frozen_observed_history':
                issues.append('safe_pool:unknown_candidate_universe')
        if len(pool) != len(expected_pool) or set(pool) != set(expected_pool):
            issues.append(pool_issue)
        if set(stage['available_components']) != set(components):
            issues.append(phase + ':component_key_inventory_mismatch')
        for field in ('objective', 'probability_violation'):
            if field not in components or len(components[field]) != len(pool):
                issues.append(phase + ':missing_or_unaligned_' + field)
    if any(':missing_or_unaligned_' in issue for issue in issues):
        return result
    primary, order = stages['primary_pool'], stages['primary_order']
    indices = order['order']
    if sorted(indices) != list(range(len(primary['pool']))):
        issues.append('primary_order_not_pool_permutation')
    elif order['ranked_points'] != [primary['pool'][i] for i in indices]:
        issues.append('ranked_points_do_not_match_saved_order')
    if not order['ranked_points'] or order['selected'] != order['ranked_points'][0]:
        issues.append('selected_not_first_saved_rank')
    if stages['post_incumbent']['shortlist'] != stages['frozen_shortlist']['shortlist']:
        issues.append('post_incumbent_not_frozen_shortlist')
    transform = stages['same_policy_transform']
    if transform['before'] != stages['frozen_shortlist']['shortlist']:
        issues.append('transform_not_connected_to_frozen_shortlist')
    if transform['after'] != final_shortlist:
        issues.append('transformed_shortlist_not_actual_input')
    pre, post, final = (points(stages['pre_incumbent']['shortlist']),
        points(stages['post_incumbent']['shortlist']), points(final_shortlist))
    if stages['pre_incumbent']['primary'] != order['selected'] or tuple(order['selected']) not in pre:
        issues.append('primary_not_connected_to_pre_incumbent')
    for phase in ('objective_choice', 'safe_fallback'):
        chosen = stages[phase]['selected'].get('point')
        if chosen is not None and tuple(chosen) not in pre:
            issues.append(phase + ':selected_not_in_pre_incumbent')
    if not transform['changed'] and transform['before'] != transform['after']:
        issues.append('unchanged_transform_changed_shortlist')
    if set(pre + post + final) - set(observed):
        issues.append('shortlist_policy_not_observed')
    gate = stages['objective_gate']
    by_primary = {tuple(p): i for i, p in enumerate(primary['pool'])}
    by_gate = {tuple(p): i for i, p in enumerate(gate['pool'])}
    ranked = [tuple(p) for p in order['ranked_points']]
    if set(observed) - set(by_primary) or set(observed) - set(by_gate):
        return result
    selected = stages['objective_choice']['selected'].get('point')
    selected_index = None if selected is None else by_gate.get(tuple(selected))
    selected_mean = None if selected_index is None else gate['components']['objective'][selected_index]
    for i, p in enumerate(observed):
        pi, gi = by_primary[p], by_gate[p]
        objective = gate['components']['objective'][gi]
        eligibility = gate['components']['probability_violation'][gi] <= gate['maximum_violation_probability']
        disposition = ('included' if p in final else 'incumbent_insertion_displaced' if p in pre and p not in post
            else 'omitted_and_objective_challenger_ineligible' if not eligibility else 'eligible_not_in_frozen_shortlist')
        result['observed'].append(dict(observed_index=i, observation_indices=expected[i]['observation_indices'],
            replicate_count=expected[i]['replicate_count'],
            posterior_rank=ranked.index(p)+1 if p in ranked else None,
            primary_probability_violation=primary['components']['probability_violation'][pi],
            primary_eligible=primary['components']['probability_violation'][pi] <= order['maximum_violation_probability'],
            objective_mean=objective, objective_probability_violation=gate['components']['probability_violation'][gi],
            objective_eligible=eligibility,
            objective_mean_minus_selected=None if selected_mean is None else objective-selected_mean,
            pre_incumbent_slot=pre.index(p)+1 if p in pre else None,
            post_incumbent_slot=post.index(p)+1 if p in post else None,
            final_slot=final.index(p)+1 if p in final else None,
            same_policy_slot_changed=p in post and p in final and post.index(p) != final.index(p),
            disposition=disposition))
    result.update(status='complete' if not issues else 'invalid_record',
        shortlist_observed_indices=[observed.index(p) if p in observed else None for p in final],
        same_policy_changed=transform['changed'])
    return result


def _outcome(protocol, descriptor, saved, item):
    result = dict(status=descriptor['status'], files={k: v for k, v in descriptor.items() if k.endswith('_file')})
    if descriptor['status'] != 'complete':
        result['failure'] = descriptor
        return result
    verification = saved['verification']
    shortlist = points(item['frozen_shortlist'])
    deployed = tuple(saved['deployed'])
    result.update(certified=verification['certified'], reported_deployed=saved['deployed'],
        reported_shortlist_index=shortlist.index(deployed) if deployed in shortlist else None,
        deployment_in_frozen_shortlist=deployed in shortlist,
        selected_shortlist_rank=verification.get('selected_shortlist_rank'),
        safety=[{k: row[k] for k in ('candidate_index', 'certified', 'upper_margin', 'verification_budget') if k in row}
            for row in verification['attempts']],
        comparisons=comparison_slots(protocol, descriptor, saved, item),
        costs={k: saved[k] for k in ('new_verification_calls', 'cumulative_new_verification_calls',
            'logical_verification_calls', 'reused_old_raw_calls', 'reused_new_raw_calls') if k in saved})
    comparisons = ({r['block_index']: r for r in verification['objective_comparisons']}
        if protocol == 'new' else {0: verification['objective_comparison']})
    for slot in result['comparisons']:
        if slot['executed']:
            slot['statistics'] = comparison_diagnostics(comparisons[slot['block_index']])
    return result


def audit_stream(root, plan, terminal_summary=None):
    """Audit every registered pair after scoring; read no truth/cache or model files."""
    root = Path(root)
    output = root / plan['output_dir']
    seal = read(output / 'execution_status.json')
    if seal['status'] != 'sealed':
        raise ValueError('Terminal recording audit requires the existing execution seal')
    summary = read(output / 'summary.json') if terminal_summary is None else terminal_summary
    references = {(r['seed'], r['arm']): r for r in summary['rows']}
    saved_cases = {(r['seed'], r['arm']): r for r in seal['cases']}
    histories, results = {}, []
    for task in plan['tasks']:
        seed = task['geometry_task_seed']
        for cell in plan['terminal_arms']:
            key = seed, cell['name']
            case = saved_cases.get(key)
            row = dict(seed=seed, arm=cell['name'], history_arm=cell['history_arm'],
                terminal_backend=cell['terminal_backend'], domain=task['domain'])
            if case is None:
                results.append(dict(row, status='missing_case', issues=['registered_pair_absent_from_seal']))
                continue
            items = {}
            for protocol in ('old', 'new'):
                path = case[protocol].get('input_file')
                if path and (root / path).is_file():
                    items[protocol] = read(root / path)
                elif case[protocol]['status'] == 'complete':
                    raise RuntimeError(f'Sealed {protocol} terminal input is absent: {path}')
            item = items.get('old') or items.get('new')
            reference = references.get(key, {})
            row['saved_results'] = {k: reference[k] for k in ('old', 'new', 'new_minus_old', 'old_deployed',
                'new_deployed', 'quality_reference_status', 'quality_baseline_defined') if k in reference}
            source = root / case['source_case']
            row['source_case'] = case['source_case']
            row['partial_record_files'] = [str(p.relative_to(root)) for pattern in
                ('failed_terminal_recording_*.json', 'failed_native_terminal_recording_*.json') for p in sorted(source.glob(pattern))]
            if item is None:
                results.append(dict(row, status='no_frozen_input', issues=['record_unavailable_after_execution_failure'],
                    protocols={p: dict(case[p], input_available=False) for p in ('old', 'new')}))
                continue
            history_file = root / item['source_case'] / 'search.json'
            if history_file not in histories:
                histories[history_file] = read(history_file)['history']
            native = cell['terminal_backend'] == 'native' or cell['history_arm'] in ('legacy_native_neighbors', 'initialization_only')
            backend = 'legacy_native_v69' if cell['history_arm'] == 'legacy_native_neighbors' else cell['history_arm']
            row.update(audit_record(item.get('terminal_recording'), histories[history_file], item['frozen_shortlist'],
                expected_context=0 if native else 1, expected_backend_arm=backend))
            if len(items) == 2 and (items['old'].get('terminal_recording') != items['new'].get('terminal_recording')
                    or items['old']['frozen_shortlist'] != items['new']['frozen_shortlist']):
                row['issues'].append('old_new_record_or_shortlist_mismatch')
                row['status'] = 'invalid_record'
            row['protocols'] = {}
            for protocol in ('old', 'new'):
                descriptor = case[protocol]
                saved = read(root / descriptor['verification_file']) if descriptor['status'] == 'complete' else None
                row['protocols'][protocol] = _outcome(protocol, descriptor, saved, items.get(protocol) or item)
                row['protocols'][protocol]['input_available'] = protocol in items
                if saved is not None and not row['protocols'][protocol]['deployment_in_frozen_shortlist']:
                    row['issues'].append(protocol + ':deployment_not_in_actual_shortlist')
                    row['status'] = 'invalid_record'
            results.append(row)
    names = [cell['name'] for cell in plan['terminal_arms']]
    by_arm = []
    for name in names:
        selected = [row for row in results if row['arm'] == name]
        by_arm.append(dict(arm=name, registered_pairs=len(selected),
            record_status_counts=dict(Counter(row['status'] for row in selected)),
            observed_disposition_counts=dict(Counter(p['disposition'] for row in selected for p in row.get('observed', []))),
            protocol_status_counts={p: dict(Counter(row.get('protocols', {}).get(p, {}).get('status', 'unknown') for row in selected))
                for p in ('old', 'new')}))
    return dict(status='complete', coverage=dict(registered_pairs=len(results), sealed_pairs=len(seal['cases']),
        valid_records=sum(r['status'] == 'complete' for r in results),
        issue_pairs=sum(bool(r['issues']) for r in results)), by_arm=by_arm, cases=results,
        new_scientific_calls=0, truth_read=False,
        interpretation='All registered controls retained. Missing records limit attribution and do not change outcome denominators; saved ranks alone do not establish true objective misordering.')


def compact_audit(result):
    return {**{k: v for k, v in result.items() if k != 'cases'},
        'issues': [{k: row[k] for k in ('seed', 'domain', 'arm', 'status', 'issues', 'source_case', 'partial_record_files') if k in row}
            for row in result['cases'] if row['issues']],
        'execution_failures': [dict(seed=row['seed'], arm=row['arm'], protocol=p, **value)
            for row in result['cases'] for p, value in row.get('protocols', {}).items() if value['status'] != 'complete']}
