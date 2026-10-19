"""Read sealed histories and terminal records; never reconstruct a posterior.

Policy-arm rows describe exposure routes, not independent experimental samples.
Only an explicitly saved native table can supply native final scores. Default
scores that were not saved remain unknown, including for omitted good policies.
"""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from statistics import mean

from performance.mainline_confirmation_query_attribution import terminal_path
from performance.mainline_three_stream_query_attribution import register_policy


def read(path):
    return json.loads(Path(path).read_text())


def point_key(point):
    return tuple(map(int, point))


def final_evidence(frozen, history):
    """Saved per-point scores plus logical constraints of the saved full-pool gate."""
    terminal = frozen.get('terminal', {})
    shortlist = terminal.get('frozen_shortlist', [])
    observed = list(dict.fromkeys(point_key(row['point']) for row in history))
    table = {point_key(row['point']): dict(row) for row in frozen.get('observed', [])}
    challenger = dict(terminal.get('objective_challenger') or {})
    chosen = challenger.get('point')
    if chosen is None:
        chosen = next((row['point'] for row in shortlist
            if row.get('selected_posterior_objective') is not None and
            row.get('selection_contract') ==
            'posterior_median_feasible_minimum_posterior_objective'), None)
    if chosen is None:
        # Initial-incumbent insertion overwrites selection_contract when the
        # selected challenger is also that incumbent, but retains this role.
        chosen = next((row['point'] for row in shortlist if row.get('shortlist_role') in (
            'posterior_objective_primary', 'posterior_objective_verification_challenger')), None)
    primary = frozen.get('primary')
    whole_pool = bool(observed and primary is not None and point_key(primary) in observed)
    count = challenger.get('eligible_count')
    aggregate_eligible = (True if count == len(observed) else False if count == 0 else None) if whole_pool else None
    threshold = challenger.get('maximum_violation_probability')
    for row in shortlist:
        point = point_key(row['point'])
        saved = table.setdefault(point, {})
        for field, source in (('objective', 'selected_posterior_objective'),
                              ('probability_violation', 'selected_probability_violation'),
                              ('posterior_rank', 'posterior_rank')):
            if saved.get(field) is None and row.get(source) is not None:
                saved[field] = row[source]
    scores = {}
    for point in observed:
        row = table.get(point, {})
        probability = row.get('probability_violation')
        eligible = (probability <= threshold if probability is not None and threshold is not None
                    else row.get('eligible', aggregate_eligible))
        eligibility_source = ('saved_point_probability' if probability is not None and threshold is not None
            else 'saved_point_eligibility' if row.get('eligible') is not None
            else 'saved_full_pool_eligible_count' if aggregate_eligible is not None else 'unknown')
        scores[point] = dict(objective=row.get('objective'),
            probability_violation=probability, risk=row.get('risk'),
            expected_violation=row.get('expected_violation'),
            terminal_order_rank=row.get('posterior_rank'), eligible=eligible,
            eligibility_source=eligibility_source)
    full_objective = bool(observed) and all(scores[p]['objective'] is not None for p in observed)
    full_eligible_objective = bool(observed) and all(scores[p]['eligible'] is not None and
        (not scores[p]['eligible'] or scores[p]['objective'] is not None) for p in observed)
    for point, row in scores.items():
        objective = row['objective']
        row['objective_rank_all'] = (1 + sum(scores[p]['objective'] < objective for p in observed)
            if full_objective else None)
        row['objective_rank_eligible'] = (1 + sum(scores[p]['eligible'] and
            scores[p]['objective'] < objective for p in observed)
            if full_eligible_objective and row['eligible'] else None)
    challenger.update(point=None if chosen is None else list(chosen),
        candidate_count=len(observed) if whole_pool else None,
        all_observed_eligible=aggregate_eligible is True,
        no_observed_eligible=aggregate_eligible is False,
        whole_observed_pool_contract=whole_pool)
    return scores, challenger


def challenger_reason(point, scores, challenger):
    """Explain this branch only; its risk gate does not forbid fallback inclusion."""
    row = scores[point]
    chosen = challenger.get('point')
    if chosen is not None and point_key(chosen) == point:
        return 'selected_objective_challenger'
    if row['eligible'] is False:
        return 'excluded_from_objective_challenger_by_final_risk'
    if row['eligible'] is None:
        return 'unknown_final_risk_eligibility'
    if chosen is None:
        return 'eligible_challenger_identity_unavailable'
    objective = row['objective']
    selected = challenger.get('selected_posterior_objective')
    if objective is None or selected is None:
        return 'eligible_not_selected_objective_score_unavailable'
    if objective > selected:
        return 'eligible_higher_posterior_objective_than_challenger'
    if objective == selected:
        return 'eligible_objective_tie_not_selected'
    return 'saved_objective_order_inconsistent_with_selected_challenger'


def analyze_arm(case, history, frozen, truth, baseline, registry, origins, chains, *,
                search_status='complete', score_source=None):
    """Pure saved-data attribution for one arm, including every observed policy."""
    scores, challenger = final_evidence(frozen, history)
    positions = {}
    for index, row in enumerate(history):
        positions.setdefault(point_key(row['point']), []).append(index)
    terminal = frozen.get('terminal', {})
    frozen_points = {point_key(row['point']) for row in terminal.get('frozen_shortlist', [])}
    primary = frozen.get('primary')
    challenger_point = challenger.get('point')
    guaranteed = {point_key(p) for p in (primary, challenger_point) if p is not None}
    observed = []
    for point, indices in positions.items():
        pid = register_policy(registry, point)
        value = truth.get(point)
        raws = [history[i].get('raw') for i in indices]
        complete_raws = all(raw is not None and len(raw) >= 2 for raw in raws)
        empirical = dict(count=len(indices), objective_mean=mean(r[0] for r in raws) if complete_raws else None,
            constraint_mean=mean(r[1] for r in raws) if complete_raws else None)
        gain = None if value is None or baseline is None else value['quality']-baseline['quality']
        improving = (None if value is None or baseline is None else
                     bool(value['true_feasible'] and value['objective'] < baseline['objective']))
        protocols = {}
        for name in ('old', 'new'):
            descriptor = case[name]
            record = descriptor.get('record') if descriptor['status'] == 'complete' else None
            protocols[name] = terminal_path(point, descriptor.get('item'), record, descriptor['status'])
        guaranteed_missing = point in guaranteed and point not in frozen_points
        relevant_chains = [row for row in chains if row['policy_id'] == pid and
            row['arm'] == case.get('history_arm', case['arm'])]
        observed.append(dict(policy_id=pid, origin=origins.get(point, 'dynamic'),
            observation_indices=indices, first_observation_index=indices[0],
            first_adaptive_observation_index=next((i for i in indices if i >= 10), None),
            empirical=empirical, truth=value, quality_gain_to_initial=gain, improving_safe=improving,
            final_scores=scores[point], objective_challenger_reason=challenger_reason(point, scores, challenger),
            shortlist_membership_reason=('saved_guaranteed_candidate_missing_unresolved' if guaranteed_missing else
                'selected_by_frozen_shortlist' if point in frozen_points else
                'not_selected_by_frozen_shortlist_detailed_cause_unknown'),
            protocols=protocols, query_chains=relevant_chains))
    summary = dict(challenger)
    summary['policy_id'] = None if challenger_point is None else register_policy(registry, challenger_point)
    summary.pop('point', None)
    return dict(arm=case['arm'], history_arm=case.get('history_arm', case['arm']),
        terminal_backend=case.get('terminal_backend'), search_status=search_status,
        score_source=score_source, objective_challenger=summary,
        primary_policy_id=None if primary is None else register_policy(registry, primary),
        original_shortlist=[dict(policy_id=register_policy(registry, row['point']),
            position=row.get('shortlist_position'), role=row.get('shortlist_role'),
            terminal_order_rank=row.get('posterior_rank')) for row in terminal.get('frozen_shortlist', [])],
        score_coverage=dict(observed=len(observed), full_table_rows=len(frozen.get('observed', [])),
            objective_known=sum(r['final_scores']['objective'] is not None for r in observed),
            eligibility_known=sum(r['final_scores']['eligible'] is not None for r in observed)),
        observed=observed)


def analyze_task(root, plan, seed, policy_registry=None):
    """Read one registered geometry on scheduler workers; no engine is loaded."""
    root = Path(root)
    cohort = root / plan['cohort_dir']
    seal = read(cohort / 'execution_status.json')
    if seal['status'] != 'sealed':
        raise ValueError('Terminal outcomes must already be globally sealed')
    cases = [row for row in seal['cases'] if row['seed'] == seed]
    registry = policy_registry if policy_registry is not None else {}
    analysis = read(cohort / 'mechanism' / f'seed_{seed}' / 'analysis.json')
    origins = {}
    for row in analysis['policies']:
        point = point_key(row['point'])
        if point in registry and registry[point] != row['policy_id']:
            raise ValueError('Shared registry conflicts with saved physical policy identity')
        registry[point] = row['policy_id']
        origins[point] = row['origin']
    directory = (root / cases[0]['source_case']).parent
    initial = read(directory / 'initialization.json')['history']
    truth = {point_key(row['point']):row['truth'] for row in read(directory / 'truth.json')}
    values = [truth.get(point_key(row['point'])) for row in initial]
    baseline, status = None, 'initial_reference_unavailable'
    if len(initial) == 10 and all(value is not None for value in values):
        baseline = min((v for v in values if v['true_feasible']), key=lambda v:v['objective'], default=None)
        status = 'defined' if baseline is not None else 'no_true_feasible_initial'
    arms, histories = [], {}
    for original in cases:
        case = deepcopy(original)
        cell = next(row for row in plan['terminal_arms'] if row['name'] == case['arm'])
        case['terminal_backend'] = cell['terminal_backend']
        for protocol in ('old', 'new'):
            descriptor = case[protocol]
            path = descriptor.get('input_file')
            descriptor['item'] = read(root/path) if path and (root/path).exists() else None
            path = descriptor.get('verification_file') if descriptor['status'] == 'complete' else None
            descriptor['record'] = read(root/path) if path else None
        source = root / case['source_case']
        if case['source_case'] not in histories:
            if (source/'search.json').exists():
                search = read(source/'search.json')
                histories[case['source_case']] = (search, search['history'], 'complete')
            else:
                failed = sorted(source.glob('failed_search_raw_*.json'))
                raws = read(failed[-1]) if failed else []
                history = raws if case.get('history_arm') == 'initialization_only' else [*initial, *raws]
                histories[case['source_case']] = ({}, history, 'partial' if failed else 'missing')
        search, history, search_status = histories[case['source_case']]
        native = case.get('terminal_backend') == 'native'
        path = source / ('native_terminal.json' if native else 'search.json')
        frozen = read(path) if native and path.exists() else {} if native else search
        arms.append(analyze_arm(case, history, frozen, truth, baseline, registry, origins,
            analysis['query_chains'], search_status=search_status,
            score_source=str(path.relative_to(root)) if path.exists() else None))
    return dict(seed=seed, domain=cases[0]['domain'], status='complete',
        baseline=dict(status=status, truth=baseline), arms=arms,
        coverage=dict(arms=len(arms), terminal_cells=2*len(arms),
            observed_policy_arm_rows=sum(len(arm['observed']) for arm in arms),
            cached_truth_policies=len(truth)),
        policies=[dict(policy_id=pid, point=list(point)) for point,pid in registry.items()],
        limits=['Final posterior fields absent from saved records remain unknown; query-prefix scores are not substituted.',
            'A challenger risk exclusion concerns that branch; safe-interior and ranked fallback have separate roles.',
            'True-safe improving observed policies need not pass an unexecuted certificate or comparison.',
            'Other control histories can contain policies outside the saved full three-role truth inventory.'])


def compact_task(raw):
    result = {key:deepcopy(value) for key,value in raw.items() if key != 'policies'}
    for arm in result['arms']:
        for row in arm['observed']:
            row['query_chains'] = [{key:chain.get(key) for key in
                ('arm', 'paid_prefix', 'previously_observed', 'eligible', 'observation_returned', 'backend_conditioned')}
                for chain in row['query_chains']]
            row['protocols'] = {name:{key:path.get(key) for key in
                ('status', 'shortlist_position', 'shortlist_role', 'disposition', 'safety_attempted',
                 'safety_passed', 'safety_upper_margin', 'deployed')}
                for name,path in row['protocols'].items()}
    return result


def aggregate_tasks(tasks):
    """Keep all geometry counts; no policy-level confidence interval or oracle gain."""
    arms = []
    for name in dict.fromkeys(arm['arm'] for task in tasks for arm in task['arms']):
        per_task = []
        for task in tasks:
            arm = next(a for a in task['arms'] if a['arm'] == name)
            rows = arm['observed']
            good = [row for row in rows if row['improving_safe'] is True]
            per_task.append(dict(seed=task['seed'], domain=task['domain'],
                baseline_status=task['baseline']['status'], search_status=arm['search_status'],
                observed=len(rows), truth_unknown=sum(row['truth'] is None for row in rows),
                improving_safe=len(good) if task['baseline']['status'] == 'defined' else None,
                good_policy_ids=[row['policy_id'] for row in good],
                score_coverage=arm['score_coverage'],
                good_challenger_reasons=dict(Counter(row['objective_challenger_reason'] for row in good)),
                protocols={protocol:dict(
                    all_dispositions=dict(Counter(row['protocols'][protocol]['disposition'] for row in rows)),
                    good_dispositions=dict(Counter(row['protocols'][protocol]['disposition'] for row in good)),
                    deployed_good_policy_ids=[row['policy_id'] for row in good if row['protocols'][protocol].get('deployed')],
                    omitted_good_policy_ids=[row['policy_id'] for row in good
                        if row['protocols'][protocol]['disposition'] == 'not_on_frozen_shortlist'])
                    for protocol in ('old', 'new')}))
        arms.append(dict(arm=name, geometries=len(per_task), per_task=per_task,
            undefined_baseline_seeds=[r['seed'] for r in per_task if r['improving_safe'] is None],
            geometries_with_observed_good=sum(bool(r['improving_safe']) for r in per_task),
            protocols={protocol:dict(
                geometries_with_omitted_good=sum(bool(r['protocols'][protocol]['omitted_good_policy_ids']) for r in per_task),
                geometries_with_deployed_good=sum(bool(r['protocols'][protocol]['deployed_good_policy_ids']) for r in per_task),
                good_dispositions=dict(sum((Counter(r['protocols'][protocol]['good_dispositions']) for r in per_task), Counter())))
                for protocol in ('old', 'new')}))
    return dict(coverage=dict(geometries=len(tasks), arms=sum(len(t['arms']) for t in tasks),
        terminal_cells=sum(t['coverage']['terminal_cells'] for t in tasks)), arms=arms,
        statistical_unit='Geometry; policy-arm counts describe repeated route exposures.')
