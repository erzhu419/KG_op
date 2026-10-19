"""Read Stage85 saved candidate/terminal paths; never invoke a model or simulator."""
from copy import deepcopy
from pathlib import Path
import traceback

import numpy as np

from performance.mainline_frozen_propagation_paths import (
    AFTER, POINT, FULL, STOP, metric_difference, _selection_record, _pool_truth,
)
from performance.mainline_pathway_replication import ROLES, SCORES, _discovery
from performance.mainline_confirmation_query_attribution import terminal_path
from performance.mainline_saved_terminal_comparison import comparison_slots
from performance.run_mainline_legacy_backend_bridge import read


SAVED_FIELDS = (*SCORES, 'total', 'probability_violation', 'stochastic_margin_mean')
MULTI = 'multiple_safe_unobserved_competition'
LIMITS = [
    'Pre-initialization predictions are absent; temporal initialization deltas remain unknown.',
    'Same-history same-pool first divergence is a frozen ablation contrast; later differences include changed histories.',
    'Candidate scores, ranks, truth and first-discovery categories are saved values; no new predictions or pair tables.',
    'Query quality differences are not additive deployment contributions; signed effects use both actual deployments.',
    'An unexecuted comparison is not a failed objective test; unverified candidates have unknown counterfactual certificates.',
    'C=0 does not assign zero actual quality; missing baselines and unscored control-history policies remain unknown.',
]


def _load_optional(root, filename):
    path = None if not filename else root / filename
    return read(path) if path is not None and path.is_file() else None


def _pid(point, registry):
    point = tuple(point)
    if point not in registry:
        # The full scored inventory covers three role pools, not every control's
        # unselected observation. Preserve such observations without scoring them.
        registry[point] = f'H{sum(value.startswith("H") for value in registry.values()):04d}'
    return registry[point]


def _extra_scores(audit):
    result = {}
    for stage in (audit or {}).get('stages', []):
        actions, score = stage['action_ids'], stage.get('score') or {}
        for index, bank_id in enumerate(actions):
            result[stage['paid_prefix'], bank_id] = {
                key: values[index] for key in ('total', 'probability_violation', 'stochastic_margin_mean')
                if isinstance((values := score.get(key)), list) and len(values) == len(actions)}
    return result


def _candidate(candidate, state, extras):
    if candidate is None:
        return dict(present=False, observed=None, eligible=None, selected=None,
                    mode=state.get('replay', {}).get('mode'), **dict.fromkeys(SAVED_FIELDS))
    result = {key: deepcopy(candidate.get(key)) for key in (
        'observed', 'eligible', 'selected', 'initially_unsampled', 'origin', 'true_feasible', 'truth', *SAVED_FIELDS)}
    result.update(extras.get((state['paid_prefix'], candidate['bank_id']), {}))
    result.update(present=True, mode=state.get('replay', {}).get('mode'),
        objective_draw_innovation=(None if result['sampled_objective'] is None or result['objective_mean'] is None
                                   else result['sampled_objective'] - result['objective_mean']))
    return result


def _candidate_difference(policy, left, right, mapping, extras):
    a = next((row for row in left.get('candidates', []) if row['policy_id'] == policy), None)
    b = next((row for row in right.get('candidates', []) if row['policy_id'] == policy), None)
    a, b = _candidate(a, left, extras.get(left['arm'], {})), _candidate(b, right, extras.get(right['arm'], {}))
    fields = (*SAVED_FIELDS, 'objective_draw_innovation')
    differences = {key: None if a.get(key) is None or b.get(key) is None else a[key] - b[key] for key in fields}
    transition = ('not_in_both_pools' if not a['present'] or not b['present'] else 'unknown'
        if a['eligible'] is None or b['eligible'] is None else 'both_eligible' if a['eligible'] and b['eligible']
        else 'left_only_eligible' if a['eligible'] else 'right_only_eligible' if b['eligible'] else 'both_excluded')
    return dict(policy_id=mapping[policy], left=a, right=b, saved_left_minus_right=differences,
                eligibility_transition=transition)


def query_comparisons(raw, mapping, extras):
    """Project already saved comparisons; scalar subtraction does not rescore."""
    states = {(row['arm'], row['paid_prefix']): row for row in raw.get('states', [])}
    rows = []
    fields = ('contrast', 'left', 'right', 'paid_prefix', 'status', 'same_paid_history',
        'same_physical_pool', 'query_changed', 'first_divergence', 'first_recorded_divergence',
        'both_observations_returned', 'common_pool_count', 'left_only_pool_count', 'right_only_pool_count', 'scope')
    for saved in raw.get('comparisons', []):
        row = {key: deepcopy(saved.get(key)) for key in fields}
        row['common_state_first_divergence'] = bool(row['first_divergence'] and
            row['same_paid_history'] and row['same_physical_pool'] and row['both_observations_returned'])
        row['selected'] = {arm: mapping.get(policy) for arm, policy in saved.get('selected', {}).items()}
        row['initialization_propagation_delta'] = None
        row['focal_policies'], row['common_policy_changes'] = [], []
        if saved.get('status') == 'available':
            left, right = (states[arm, saved['paid_prefix']] for arm in (saved['left'], saved['right']))
            focal = list(dict.fromkeys(saved['selected'].values()))
            row['focal_policies'] = [_candidate_difference(policy, left, right, mapping, extras) for policy in focal]
            right_ids = {value['policy_id'] for value in right['candidates']}
            row['common_policy_changes'] = [_candidate_difference(value['policy_id'], left, right, mapping, extras)
                for value in left['candidates'] if value['policy_id'] in right_ids]
        rows.append(row)
    return rows


def _terminal(point, item, record, descriptor, outcome, policy):
    result = terminal_path(point, item, record, descriptor['status'])
    result['verification_selected'] = result.get('deployed')
    result['deployed'] = bool(outcome['metrics'].get('C') == 1 and outcome.get('deployed_policy_id') == policy)
    result['objective_comparison_coverage'] = []
    if record is not None and item is not None:
        index = result.get('shortlist_position')
        for slot in comparison_slots('old', descriptor, record, item):
            if index is not None and index - 1 in (slot['challenger_index'], slot['incumbent_index']):
                result['objective_comparison_coverage'].append(deepcopy(slot))
    executed = [slot for slot in result['objective_comparison_coverage'] if slot['executed']]
    if result.get('verification_selected') and not result['deployed']:
        reason = 'verified_but_not_true_certified_deployment'
    elif result['deployed']:
        reason = 'deployed'
    elif result.get('safety_passed') is True:
        reason = ('actual_objective_comparison_retained_other_policy' if executed else
                  'safety_certified_without_actual_objective_comparison')
    else:
        reason = result['disposition']
    result['terminal_reason'] = reason
    return result


def _terminal_cell(root, descriptor, config, outcome, history, states, local_ids, truth, registry):
    old = descriptor['old']
    item = _load_optional(root, old.get('input_file'))
    record = _load_optional(root, old.get('verification_file')) if old['status'] == 'complete' else None
    selection = (_selection_record(item, history, config) if item is not None else
                 dict(status='missing_record', issues=['terminal_input_missing'], observed=[]))
    selected = {row['observed_index']: row for row in selection.get('observed', [])}
    physical = list(dict.fromkeys(tuple(row['point']) for row in history))
    observed = []
    for index, point in enumerate(physical):
        policy = _pid(point, registry)
        indices = [i for i, value in enumerate(history) if tuple(value['point']) == point]
        discovery = _discovery(descriptor['history_arm'], dict(policy_id=local_ids.get(point), observation_indices=indices), states)
        value, baseline = truth.get(point), outcome.get('best_initial_quality')
        observed.append(dict(policy_id=policy, observation_indices=indices, truth=deepcopy(value), **discovery,
            quality_reference_status=outcome.get('quality_reference_status'),
            quality_gain_to_initial=None if value is None or baseline is None else value['quality'] - baseline,
            final_observed_pool_scores=deepcopy(selected.get(index)),
            terminal_path=_terminal(point, item, record, old, outcome, policy)))
    return dict(arm=descriptor['arm'], history_arm=descriptor['history_arm'], status=old['status'],
        selection_record_status=selection['status'], selection_issues=selection.get('issues', []),
        observed=observed, final_observed_pool_truth=_pool_truth(physical, truth),
        shortlist_policy_ids=[] if item is None else [_pid(row['point'], registry) for row in item['frozen_shortlist']],
        reported_policy_id=outcome.get('reported_policy_id'), deployed_policy_id=outcome.get('deployed_policy_id'),
        actual_objective_comparison=None if record is None else deepcopy(record['verification'].get('objective_comparison')),
        saved_costs={} if record is None else {key: record[key] for key in (
            'new_verification_calls', 'cumulative_new_verification_calls', 'logical_verification_calls',
            'reused_old_raw_calls', 'failed_simulation_calls') if key in record})


def _events(queries, cells):
    by_arm = {row['arm']: row for row in cells}
    events = []
    for query in queries:
        if (query.get('observation_returned') is not True or query.get('returned_policy_matches') is not True
                or query.get('discovery_category') != MULTI):
            continue
        observed = next((row for row in by_arm[query['arm']]['observed']
            if row['policy_id'] == query['selected_policy_id']), None)
        events.append(dict(arm=query['arm'], policy_id=query['selected_policy_id'], paid_prefix=query['paid_prefix'],
            query_action_kind=query['query_action_kind'], observation_returned=True,
            discovery_category=query['discovery_category'], pool_truth_complete=query.get('pool_truth_complete'),
            counts=deepcopy(query.get('counts')), query_discovery_category=(query['discovery_category']
                if query['query_action_kind'] == 'new' else None),
            discovery_gain=None if observed is None else observed['quality_gain_to_initial'],
            quality_gain_to_initial=None if observed is None else observed['quality_gain_to_initial'],
            returned_repeat_count=None if observed is None else observed.get('returned_repeat_count'),
            own_terminal_path=None if observed is None else deepcopy(observed['terminal_path'])))
    return events


def _stream(root, plan, output, seed, saved, index, registry, truth):
    result = dict(name=saved['name'], status='complete', source_status=saved['status'],
        outcomes=deepcopy(saved['outcomes']), queries=deepcopy(saved.get('queries', [])),
        policy_paths=deepcopy(saved.get('policy_paths', [])), coverage=deepcopy(saved.get('coverage')),
        quality_reference_status=saved.get('quality_reference_status'), query_comparisons=[],
        initialization_contrast=None, policy_terminal_paths=[], multi_safe_events=[])
    outcomes = {row['arm']: row for row in result['outcomes']}
    expected = {row['name'] for row in plan['terminal_arms']}
    if len(outcomes) != len(result['outcomes']) or set(outcomes) != expected:
        raise ValueError('All nine saved outcomes must be retained')
    result['comparisons'] = [dict(contrast=label, left=left, right=right,
        **metric_difference(outcomes[left]['metrics'], outcomes[right]['metrics']))
        for label, left, right in (('AP', AFTER, POINT), ('FA', FULL, AFTER))]
    try:
        path = output / 'cases' / f'seed_{seed}' / saved['name'] / 'analysis.json'
        raw = read(path) if path.is_file() else dict(states=[], comparisons=[], policies=[])
        mapping = {row['policy_id']: _pid(row['point'], registry) for row in raw['policies']}
        local_ids = {tuple(row['point']): row['policy_id'] for row in raw['policies']}
        states = {(row['arm'], row['paid_prefix']): row for row in raw['states']}
        histories, extras = {}, {}
        cases = index['cases']
        if {row['arm'] for row in cases} != expected or len(cases) != len(expected):
            raise ValueError('All nine sealed terminal descriptors must be retained')
        for case in cases:
            arm = case['history_arm']
            if arm not in histories:
                search = _load_optional(root, case['source_case'] + '/search.json')
                histories[arm] = [] if search is None else search['history']
                if arm in ROLES:
                    extras[arm] = _extra_scores(_load_optional(root, case['source_case'] + '/query_audit.json'))
        configs = {row['name']: row for row in plan['terminal_arms']}
        for case in cases:
            result['policy_terminal_paths'].append(_terminal_cell(root, case, configs[case['arm']],
                outcomes[case['arm']], histories[case['history_arm']], states, local_ids, truth, registry))
        result['query_comparisons'] = query_comparisons(raw, mapping, extras)
        result['initialization_contrast'] = next((row for row in result['query_comparisons']
            if row['contrast'] == 'AP' and row['paid_prefix'] == 10), None)
        result['multi_safe_events'] = _events(result['queries'], result['policy_terminal_paths'])
        if not path.is_file():
            result.update(status='saved_mechanism_unavailable', error='No saved full-pool mechanism analysis')
    except Exception:
        result.update(status='path_analysis_failed', error=traceback.format_exc())
    return result


def analyze_task(root, plan, seed):
    """Read one registered geometry. Caller schedules/meters/writes; no scientific work."""
    root = Path(root)
    source = read(root / plan['source_manifest'])
    if source['output_dir'] != plan['source_output_dir']:
        raise ValueError('Use the registered Stage85 saved output')
    output = root / plan['source_output_dir']
    inventory = read(output / 'mechanism/inventory.json')
    expected_cells = len(source['tasks']) * len(source['streams']) * len(source['terminal_arms'])
    if (inventory['status'] != 'sealed' or not inventory.get('truth_access_allowed')
            or inventory['registered_terminal_cells'] != expected_cells):
        raise ValueError('Complete Stage85 global seal must precede saved-data attribution')
    task = next(row for row in plan['tasks'] if row['geometry_task_seed'] == seed)
    saved = read(output / 'cases' / f'seed_{seed}' / 'result.json')
    index = read(output / 'mechanism' / f'seed_{seed}' / 'index.json')
    if saved['seed'] != seed or saved['domain'] != task['domain']:
        raise ValueError('Saved geometry identity differs from the registration')
    with np.load(root / index['union_policies_file'], allow_pickle=False) as arrays:
        registry = {tuple(point.tolist()): f'P{i:04d}' for i, point in enumerate(arrays['points'])}
    truth = {tuple(row['point']): row['truth'] for row in read(output / 'cases' / f'seed_{seed}' / 'truth.json')}
    names = [row['name'] for row in plan['streams']]
    if len(saved['streams']) != len(names) or {row['name'] for row in saved['streams']} != set(names):
        raise ValueError('Both registered observation streams must remain')
    entries = {row['name']: row for row in index['streams']}
    streams = [_stream(root, plan, output, seed, row, entries[row['name']], registry, truth) for row in saved['streams']]
    return dict(seed=seed, domain=task['domain'], status='complete' if all(row['status'] == 'complete'
        for row in streams) else 'complete_with_retained_path_failures', streams=streams, limits=LIMITS,
        source_result_file=str((output / 'cases' / f'seed_{seed}' / 'result.json').relative_to(root)),
        initialization_propagation_delta=None, new_scientific_calls=0)
