"""Connect frozen Stage81 records without simulation, inference, or new scores.

Only saved prediction arrays may be ranked. Target truth is limited to the
already scored initial/terminal union; a partial cache never defines a pool
maximum. Callers schedule, meter, and save the returned geometry record.
"""
from copy import deepcopy
from pathlib import Path

import numpy as np

from performance.diagnose_mainline_role_choice import score_choice
from performance.mainline_confirmation_query_attribution import candidate_path, terminal_path
from performance.mainline_decision_path_attribution import (
    METRICS, point_key, quality_set, read,
)
from performance.mainline_reference_coverage import VARIANTS
from performance.mainline_reference_coverage_cohort import require_truth_seal
from performance.mainline_terminal_recording_audit import audit_record

AFTER = 'legacy_role_after_init_point_only'
POINT = 'legacy_role_point_only'
FULL = 'legacy_role_full'
STOP = 'initialization_only'
SCORE_FIELDS = ('objective_mean', 'objective_epistemic', 'sampled_objective',
    'constraint_mean', 'constraint_aleatoric', 'constraint_epistemic',
    'sampled_constraint', 'sampled_margin', 'stochastic_margin_mean',
    'probability_violation')
LIMITS = [
    'Saved arithmetic and selection replay only; no new candidate predictions or random draws.',
    'Only the existing Stage81 truth union is known; unscored query and pool truth remains unknown.',
    'C=0 and metric G=0 do not assign a quality of zero to an uncertified deployment.',
    'A same-history, same-pool first query divergence permits a fixed-state comparison; later diverged histories do not.',
    'Pre-initialization predictions were not saved: initialization propagation deltas remain unknown.',
    'Positive gain differences are not new discoveries without a physical query-to-deployment path.',
    'No component substitutions, hypothetical certificate, or counterfactual trajectory is evaluated.',
]


def metric_difference(left, right):
    """Preserve both signed terms of delta G = delta P - delta L."""
    difference = {key: None if left.get(key) is None or right.get(key) is None
                  else left[key] - right[key] for key in METRICS}
    positive = difference['P']
    avoidance = None if difference['L'] is None else -difference['L']
    residual = (None if any(value is None for value in (difference['G'], positive, avoidance))
                else difference['G'] - positive - avoidance)
    return dict(difference=difference, positive_gain_component=positive,
        avoided_loss_component=avoidance, decomposition_residual=residual,
        positive_gain_is_new_discovery=None)


def _bundle(root, source_case):
    directory = root / source_case
    saved = read(directory / 'search.json') if (directory / 'search.json').exists() else None
    audit_path, bank_path = directory / 'query_audit.json', directory / 'query_policies.npz'
    audit = read(audit_path) if audit_path.exists() else None
    bank = None
    if bank_path.exists():
        with np.load(bank_path, allow_pickle=False) as arrays:
            bank = [point_key(point) for point in arrays['points']]
    return dict(source_case=source_case, search=saved,
                history=[] if saved is None else saved['history'], audit=audit, bank=bank)


def _stage_points(bundle, stage):
    if bundle['bank'] is None or stage is None:
        return None
    return [bundle['bank'][index] for index in stage['action_ids']]


def _pool_truth(points, truth):
    known = [point for point in set(points) if truth.get(point) is not None
             and all(truth[point].get(key) is not None for key in ('quality', 'true_feasible'))]
    complete = len(known) == len(set(points))
    safe = [point for point in known if truth[point]['true_feasible']]
    return dict(physical_policy_count=len(set(points)), cached_truth_count=len(known),
        unknown_truth_count=len(set(points)) - len(known), truth_complete=complete,
        max_true_safe_quality=(max(truth[point]['quality'] for point in safe)
                               if complete and safe else None),
        maximum_status='truth_incomplete' if not complete else 'defined' if safe else 'no_true_safe_policy')


def saved_candidate_trace(point, bundle):
    """Use the established path helper when its saved fields are available."""
    history = bundle['history']
    observed = [index for index, row in enumerate(history) if point_key(row['point']) == point]
    stages = [] if bundle['audit'] is None else bundle['audit'].get('stages', [])
    by_prefix = {stage['paid_prefix']: stage for stage in stages}
    rows = []
    for prefix in (10, 11, 12):
        stage = by_prefix.get(prefix)
        actions = _stage_points(bundle, stage)
        base = dict(paid_prefix=prefix, previously_observed=point in {
            point_key(row['point']) for row in history[:prefix]})
        if actions is None:
            rows.append(dict(base, status='query_record_unavailable', present=None,
                             selected=None, sampled_feasible=None, objective_mean=None,
                             sampled_objective=None, initialization_propagation_delta=None))
            continue
        if point not in actions:
            rows.append(dict(base, status='saved', present=False, selected=False,
                             sampled_feasible=None, initialization_propagation_delta=None))
            continue
        score, index = stage.get('score') or {}, actions.index(point)
        missing = [field for field in SCORE_FIELDS if field not in score
                   or not isinstance(score[field], list) or len(score[field]) != len(actions)]
        if not missing and 'sampled_feasibility_mode' in score:
            one = dict(bundle, audit=dict(stages=[stage]))
            row = candidate_path(point, one)['states'][0]
            row.update(status='saved', missing_score_fields=[])
            eligible = np.asarray(score['sampled_margin']) <= 0.
            means = np.asarray(score['objective_mean'])
            row['objective_mean_rank_within_sampled_feasible'] = (
                int(1 + np.sum(eligible & (means < means[index]))) if eligible[index] else None)
            row['objective_mean_minus_selected'] = float(means[index] - means[stage['selected_index']])
            row['sampled_objective_minus_selected'] = float(
                score['sampled_objective'][index] - score['sampled_objective'][stage['selected_index']])
        else:
            row = dict(base, status='partial_saved_scores', present=True,
                selected=index == stage['selected_index'], sampled_feasible=None,
                reason='score_fields_unavailable', missing_score_fields=missing)
            row.update({field: (score[field][index] if field not in missing else None)
                        for field in SCORE_FIELDS})
        row['initialization_propagation_delta'] = None
        row['objective_draw_innovation'] = (None if row.get('sampled_objective') is None
            or row.get('objective_mean') is None else row['sampled_objective'] - row['objective_mean'])
        rows.append(row)
    return dict(observation_history_indices=observed,
        adaptive_query_history_indices=[index for index in observed if index >= 10],
        initially_unobserved=not any(index < 10 for index in observed), states=rows)


def _query_states(bundle, registry, truth):
    if bundle['audit'] is None or bundle['bank'] is None:
        return []
    states = []
    for stage in bundle['audit'].get('stages', []):
        points = _stage_points(bundle, stage)
        selected = points[stage['selected_index']]
        prefix = stage['paid_prefix']
        actual = (None if prefix >= len(bundle['history'])
                  else point_key(bundle['history'][prefix]['point']))
        if stage['selected_id'] != stage['action_ids'][stage['selected_index']] or actual != selected:
            raise ValueError('Saved query identity does not match its returned observation')
        score = stage.get('score') or {}
        replay = dict(status='score_fields_unavailable', matched=None, max_total_error=None)
        if all(field in score for field in ('sampled_objective', 'sampled_margin', 'total',
                                           'sampled_feasibility_mode')):
            chosen, totals, mode, count = score_choice(score['sampled_objective'], score['sampled_margin'])
            error = float(np.max(np.abs(totals - np.asarray(score['total']))))
            matched = chosen == stage['selected_index'] and mode == score['sampled_feasibility_mode'] and error <= 1e-12
            replay = dict(status='matched' if matched else 'mismatch', matched=matched,
                          max_total_error=error, mode=mode, sampled_feasible_count=count)
            if not matched:
                raise ValueError('Saved selection replay changed the original query')
        states.append(dict(paid_prefix=prefix, selected_policy_id=registry[selected],
            actual_query_returned=True, selected_before_observed=selected in {
                point_key(row['point']) for row in bundle['history'][:prefix]},
            selected_trace=next(row for row in saved_candidate_trace(selected, bundle)['states']
                                if row['paid_prefix'] == prefix),
            candidate_count=len(points), pool_truth=_pool_truth(points, truth),
            selection_replay=replay))
    return states


def compare_queries(left_name, right_name, bundles, registry):
    """Compare actual saved states only, without risk/objective substitutions."""
    left, right = bundles[left_name], bundles[right_name]
    left_stages = {row['paid_prefix']: row for row in (left['audit'] or {}).get('stages', [])}
    right_stages = {row['paid_prefix']: row for row in (right['audit'] or {}).get('stages', [])}
    rows, diverged = [], False
    for prefix in (10, 11, 12):
        a, b = left_stages.get(prefix), right_stages.get(prefix)
        pa, pb = _stage_points(left, a), _stage_points(right, b)
        if pa is None or pb is None:
            rows.append(dict(paid_prefix=prefix, status='saved_query_unavailable',
                same_paid_history=None, same_physical_pool=None, same_pool_order=None,
                query_changed=None, first_divergence=None, common_state_first_divergence=False))
            continue
        selected_a, selected_b = pa[a['selected_index']], pb[b['selected_index']]
        changed = selected_a != selected_b
        first = changed and not diverged
        same_history = left['history'][:prefix] == right['history'][:prefix]
        same_pool = set(pa) == set(pb)
        focal = list(dict.fromkeys((selected_a, selected_b)))
        row = dict(paid_prefix=prefix, status='saved', same_paid_history=same_history,
            same_physical_pool=same_pool, same_pool_order=pa == pb,
            query_changed=changed, first_divergence=first,
            common_state_first_divergence=bool(first and same_history and same_pool),
            earlier_query_divergence=diverged,
            interpretation=('same_state_first_divergence' if first and same_history and same_pool
                            else 'later_histories_or_pools_differ' if not (same_history and same_pool)
                            else 'same_state_saved_comparison'),
            selected={left_name: registry[selected_a], right_name: registry[selected_b]}, candidates=[])
        for point in focal:
            traces = {name: next(value for value in saved_candidate_trace(point, bundles[name])['states']
                                if value['paid_prefix'] == prefix) for name in (left_name, right_name)}
            differences = {field: None if traces[left_name].get(field) is None or traces[right_name].get(field) is None
                else traces[left_name][field] - traces[right_name][field]
                for field in ('objective_mean', 'sampled_objective', 'sampled_margin', 'probability_violation')}
            row['candidates'].append(dict(policy_id=registry[point], traces=traces,
                                          saved_left_minus_right=differences))
        rows.append(row)
        diverged |= changed
    return dict(left=left_name, right=right_name, states=rows,
        changed_prefixes=[row['paid_prefix'] for row in rows if row['query_changed']],
        initialization_propagation_delta=None)


def _selection_record(item, history, cell):
    native = cell['terminal_backend'] == 'native' or cell['history_arm'] in ('legacy_native_neighbors', STOP)
    backend = 'legacy_native_v69' if cell['history_arm'] == 'legacy_native_neighbors' else cell['history_arm']
    return audit_record(item.get('terminal_recording'), history, item['frozen_shortlist'],
                        expected_context=0 if native else 1, expected_backend_arm=backend)


def _cell(descriptor, scored, item, record, history, selection, truth, registry, initial):
    metrics = deepcopy(scored['metrics'])
    shortlist = [] if item is None else [point_key(row['point']) for row in item['frozen_shortlist']]
    physical = list(dict.fromkeys(point_key(row['point']) for row in history))
    observed = []
    selected_rows = {row['observed_index']: row for row in selection.get('observed', [])}
    for index, point in enumerate(physical):
        path = terminal_path(point, item, record if descriptor['status'] == 'complete' else None,
                             descriptor['status'])
        observed.append(dict(policy_id=registry[point], truth=truth.get(point),
            final_observed_pool_scores=deepcopy(selected_rows.get(index)), terminal_path=path))
    reported = point_key(record['deployed']) if record is not None and 'deployed' in record else None
    deployment_truth = truth.get(reported)
    deployed = reported if metrics.get('C') == 1 else None
    quality = None if deployed is None or deployment_truth is None else deployment_truth.get('quality')
    query_indices = [] if reported is None else [index for index, row in enumerate(history)
        if index >= 10 and point_key(row['point']) == reported]
    outcome = dict(seed=descriptor['seed'], domain=descriptor['domain'], arm=descriptor['arm'],
        history_arm=descriptor['history_arm'], variant=descriptor['variant'],
        status=descriptor['status'], metrics=metrics, costs=deepcopy(scored.get('costs', {})),
        quality_reference_status=scored.get('quality_reference_status'),
        best_initial_quality=scored.get('best_initial_quality'),
        reported_policy_id=registry.get(reported), deployed_policy_id=registry.get(deployed),
        actual_deployed_quality=quality, reported_policy_truth=deployment_truth,
        newly_queried_deployment=bool(deployed is not None and reported not in initial and query_indices),
        deployment_query_indices=query_indices if deployed is not None else [],
        no_true_certified_deployment=metrics.get('C') == 0)
    verification = {} if record is None else record.get('verification', {})
    attempted = [shortlist[row['candidate_index']] for row in verification.get('attempts', [])]
    passed = [shortlist[row['candidate_index']] for row in verification.get('attempts', []) if row['certified']]
    sets = {name: quality_set(points, truth, registry, available=available) for name, points, available in (
        ('observed', physical, bool(history)), ('shortlist', shortlist, item is not None),
        ('attempted', attempted, record is not None), ('safety_passed', passed, record is not None))}
    cell = dict(arm=descriptor['arm'], history_arm=descriptor['history_arm'],
        variant=descriptor['variant'], status=descriptor['status'],
        shortlist_policy_ids=[registry[point] for point in shortlist],
        original_selection_record_status=selection['status'],
        selection_issues=selection.get('issues', []), observed=observed, quality_sets=sets,
        safety_attempts=deepcopy(scored.get('safety_attempts', [])),
        actual_objective_comparisons=deepcopy(verification.get('objective_comparisons',
            [verification['objective_comparison']] if 'objective_comparison' in verification else [])),
        reported_policy_id=outcome['reported_policy_id'], deployed_policy_id=outcome['deployed_policy_id'],
        reference_replacement=(None if item is None else dict(
            changed=item.get('reference_coverage', {}).get('reference_shortlist_changed'),
            selected_initial_rank=item.get('reference_coverage', {}).get('selected_initial_rank'),
            actual_reference_variant=descriptor['variant'].startswith('reference_'))))
    return outcome, cell


def analyze_task(root, plan, seed):
    """Read one geometry from the complete frozen cohort; write nothing."""
    root = Path(root)
    source = read(root / plan['reference_manifest'])
    output = root / plan['reference_output_dir']
    if plan['reference_output_dir'] != source['output_dir']:
        raise ValueError('Attribution must use the registered completed source output')
    seal = require_truth_seal(source, root=root)
    source_cells = [row for row in seal['cells'] if row['seed'] == seed]
    expected = {(cell['name'], variant) for cell in plan['terminal_arms'] for variant in VARIANTS}
    if len(source_cells) != len(expected) or {(row['arm'], row['variant']) for row in source_cells} != expected:
        raise ValueError('Attribution lost a registered terminal cell')
    scored = read(output / 'truth' / f'seed_{seed}' / 'scored.json')
    values = {(row['arm'], row['variant']): row for row in scored['outcomes']}
    if set(values) != expected:
        raise ValueError('Saved scores lost a registered terminal cell')
    truth = {point_key(row['point']): row['truth']
             for row in read(output / 'truth' / f'seed_{seed}' / 'truth.json')}
    frozen = [row for row in read(output / 'frozen_inputs.json')['rows'] if row['seed'] == seed]
    source_cases = {row['history_arm']: row['source_case'] for row in frozen}
    bundles = {name: _bundle(root, path) for name, path in source_cases.items()}
    initial_rows = bundles[STOP]['history'][:10]
    initial = {point_key(row['point']) for row in initial_rows}
    registry = {}

    def add(point):
        point = point_key(point)
        if point not in registry:
            registry[point] = f'P{len(registry):03d}'

    for row in initial_rows:
        add(row['point'])
    for bundle in bundles.values():
        for row in bundle['history']:
            add(row['point'])
    loaded = {}
    for descriptor in source_cells:
        item = read(root / descriptor['input_file']) if descriptor.get('input_file') else None
        record = read(root / descriptor['result_file']) if descriptor.get('result_file') else None
        loaded[descriptor['arm'], descriptor['variant']] = item, record
        for row in ([] if item is None else item['frozen_shortlist']):
            add(row['point'])
        if record is not None and 'deployed' in record:
            add(record['deployed'])
    policies = []
    for point, label in registry.items():
        locations = [dict(history_arm=name, observation_index=index)
            for name, bundle in bundles.items() for index, row in enumerate(bundle['history'])
            if point_key(row['point']) == point]
        indices = [index for index, row in enumerate(initial_rows) if point_key(row['point']) == point]
        policies.append(dict(policy_id=label, initial_indices=indices,
            origin='initial' if indices else 'queried_noninitial', observed_locations=locations,
            truth=deepcopy(truth.get(point)), truth_status='cached' if point in truth else 'unknown'))
    histories = []
    for name, bundle in bundles.items():
        histories.append(dict(history_arm=name, source_case=bundle['source_case'],
            status='complete' if bundle['search'] is not None else 'missing',
            observations=[dict(observation_index=index, policy_id=registry[point_key(row['point'])])
                          for index, row in enumerate(bundle['history'])],
            queries=_query_states(bundle, registry, truth),
            policy_paths=[] if name == STOP else [dict(policy_id=registry[point],
                **saved_candidate_trace(point, bundle)) for point in registry],
            observed_truth=quality_set([row['point'] for row in bundle['history']], truth, registry,
                                      available=bundle['search'] is not None)))
    selections = {}
    for cell in plan['terminal_arms']:
        item, _ = loaded[cell['name'], 'original_old']
        selections[cell['name']] = (dict(status='missing_record', issues=['input_unavailable'], observed=[])
            if item is None else _selection_record(item, bundles[cell['history_arm']]['history'], cell))
    outcomes, cells = [], []
    for descriptor in source_cells:
        key = descriptor['arm'], descriptor['variant']
        item, record = loaded[key]
        outcome, cell = _cell(descriptor, values[key], item, record,
            bundles[descriptor['history_arm']]['history'], selections[descriptor['arm']],
            truth, registry, initial)
        outcomes.append(outcome); cells.append(cell)
    by_key = {(row['arm'], row['variant']): row for row in outcomes}
    comparisons = []
    for left, right in ((AFTER, POINT), (FULL, AFTER)):
        comparisons.append(dict(left=left, right=right, variants={variant:
            dict(metric_difference(by_key[left, variant]['metrics'], by_key[right, variant]['metrics']),
                left_deployed_policy_id=by_key[left, variant]['deployed_policy_id'],
                right_deployed_policy_id=by_key[right, variant]['deployed_policy_id'],
                left_newly_queried_deployment=by_key[left, variant]['newly_queried_deployment'],
                right_newly_queried_deployment=by_key[right, variant]['newly_queried_deployment'])
            for variant in VARIANTS}))
    return dict(status='complete', seed=seed, domain=scored['domain'],
        registered_terminal_cells=len(expected), outcomes=outcomes, cells=cells,
        histories=histories, policies=policies, comparisons=comparisons,
        query_comparisons=[compare_queries(left, right, bundles, registry)
                           for left, right in ((AFTER, POINT), (FULL, AFTER))],
        coverage=dict(histories=len(histories), terminal_cells=len(cells),
            cached_truth_policies=len(truth), traced_physical_policies=len(policies),
            traced_unknown_truth_policies=sum(row['truth_status'] == 'unknown' for row in policies)),
        limits=LIMITS, new_simulation_calls=0, new_candidate_scores=0,
        new_source_fits=0, posterior_restorations=0, new_truth_function_calls=0,
        new_rng_draws=0)
