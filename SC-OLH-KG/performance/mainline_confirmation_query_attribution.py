"""Stage65 query/deployment attribution from saved arrays; no scientific calls."""
from pathlib import Path
from collections import Counter

import numpy as np
from scipy.stats import norm

from performance.diagnose_mainline_role_terminal import policy_key
from performance.diagnose_mainline_three_family_queries import (
    AFTER, ARMS, FULL, POINT, audit_task,
)
from performance.run_mainline_legacy_backend_bridge import read

ARM_LABEL = {FULL: 'FULL', AFTER: 'AFTER', POINT: 'POINT'}


def _short_policy(label):
    if label is None:
        return None
    for arm, name in ARM_LABEL.items():
        label = label.replace(arm + ':', name + ':')
    return label


def compact_task(result):
    """Compact one raw result without vectors; preserves all actual query outcomes."""
    queries = []
    for row in result['states']:
        selected = row['choices']['actual']['selected']
        identity = selected['identity']
        truth = selected.get('truth')
        queries.append(dict(arm=ARM_LABEL[row['arm']], prefix=row['paid_prefix'],
            policy=_short_policy(identity['policy']), origin=identity['origin'],
            mode=row['choices']['actual']['mode'], truth_known=truth is not None,
            true_feasible=truth['true_feasible'] if truth else None,
            quality=truth['quality'] if truth else None, new_good=selected['new_good']))
    comparisons = []
    for left, right in ((AFTER, POINT), (FULL, AFTER)):
        pairs = [p for p in result['paired_states'] if p['left']==left and p['right']==right]
        first = next((p for p in pairs if p['first_divergence']), None)
        detail = None
        if first:
            detail = dict(prefix=first['paid_prefix'],same_history=first['same_paid_history'],
                same_pool=first['same_physical_pool'],selected={ARM_LABEL[a]:
                    _short_policy(first['selected'][a]['identity']['policy']) for a in (left,right)},
                swaps=[dict(receiver=ARM_LABEL[s['receiver']],donor=ARM_LABEL[s['donor']],
                    component='risk' if s['substitution'].startswith('risk_') else 'objective',
                    donor_recovered=s['donor_choice_recovered'],changed=s['selection_changed'],
                    selected_neighbor=s['selected']['neighbor_index'],
                    selected_left_bank_id=s['selected']['left_bank_id'])
                    for s in first['substitutions']])
        comparisons.append(dict(left=ARM_LABEL[left],right=ARM_LABEL[right],states=len(pairs),
            changed_prefixes=[p['paid_prefix'] for p in pairs if p['query_changed']],
            same_history_states=sum(p['same_paid_history'] for p in pairs),
            same_pool_states=sum(p['same_physical_pool'] for p in pairs),first_divergence=detail))
    deployed = next((r for r in result['deployment_paths'] if r['protocol']=='new'), None)
    deployment = None
    if deployed and deployed.get('after_policy'):
        policy = deployed['after_policy']
        paths = {}
        for path in deployed['paths']:
            search = path['search']
            search_compact = None if search is None else dict(
                in_bank=search['present_in_saved_bank'],
                first_candidate_prefix=search['first_candidate_paid_prefix'],
                observed_indices=search['observation_history_indices'],
                adaptive_query_indices=search['adaptive_query_history_indices'],
                states=[dict(prefix=s['paid_prefix'],present=False) if not s['present'] else
                    dict(prefix=s['paid_prefix'],present=True,observed=s['previously_observed'],
                        selected=s['selected'],reason=s['reason'],margin=s['sampled_margin'],
                        ts_rank=s['sampled_objective_rank_all'],
                        feasible_ts_rank=s['sampled_objective_rank_within_feasible'])
                    for s in search['states']])
            terminals = {protocol:{k:row.get(k) for k in ('status','shortlist_position',
                'disposition','safety_attempted','safety_passed','deployed')}
                for protocol,row in path['terminals'].items()}
            paths[ARM_LABEL[path['arm']]] = dict(search=search_compact,terminals=terminals)
        deployment = dict(policy=_short_policy(policy['policy']),origin=policy['origin'],
            certified=deployed['after_certified'],truth=policy['truth'],paths=paths)
    task = result['task']
    return dict(status=result['status'],seed=result['seed'],domain=result['domain'],
        baseline_defined=task['quality_baseline_defined'],
        reconstruction=result['reconstruction'],actual_queries=queries,comparisons=comparisons,
        after_new=deployment,truth_coverage=dict(cached_policies=task['cached_truth_count'],
            candidate_exposures=sum(r['candidate_count'] for r in result['states']),
            cached_truth_exposures=sum(r['cached_truth_action_count'] for r in result['states']),
            unknown_truth_exposures=sum(r['unknown_truth_action_count'] for r in result['states']),
            actual_queries=len(queries),known_query_truth=sum(r['truth_known'] for r in queries),
            scope='Saved task-local cache only; candidate exposures repeat physical policies'))


def aggregate_queries(compact_tasks):
    """Aggregate compact_task outputs; first-divergence substitutions count tasks."""
    comparisons = []
    for left,right in (('AFTER','POINT'),('FULL','AFTER')):
        selected = [(task['seed'],c) for task in compact_tasks for c in task['comparisons']
                    if c['left']==left and c['right']==right]
        first = [(seed,c['first_divergence']) for seed,c in selected if c['first_divergence']]
        causal = [(seed,f) for seed,f in first if f['same_history'] and f['same_pool']]
        swaps = []
        for receiver,donor in ((left,right),(right,left)):
            categories = dict(risk_only=[],objective_only=[],both_alone=[],neither_alone=[])
            for seed,f in causal:
                values = {s['component']:s['donor_recovered'] for s in f['swaps']
                    if s['receiver']==receiver and s['donor']==donor}
                risk,objective = values['risk'],values['objective']
                category = ('both_alone' if risk and objective else 'risk_only' if risk else
                            'objective_only' if objective else 'neither_alone')
                categories[category].append(seed)
            swaps.append(dict(receiver=receiver,donor=donor,tasks=len(causal),
                all_risk_recovers_donor=len(categories['risk_only'])+len(categories['both_alone']),
                all_objective_recovers_donor=len(categories['objective_only'])+len(categories['both_alone']),
                counts={k:len(v) for k,v in categories.items()},seeds=categories))
        comparisons.append(dict(left=left,right=right,registered_tasks=len(compact_tasks),
            available_tasks=sum(c['states']>0 for _,c in selected),
            paired_states=sum(c['states'] for _,c in selected),
            changed_queries=sum(len(c['changed_prefixes']) for _,c in selected),
            tasks_with_query_change=sum(bool(c['changed_prefixes']) for _,c in selected),
            first_divergences=len(first),same_history_same_pool_first_divergences=len(causal),
            first_divergence_by_prefix=dict(Counter(str(f['prefix']) for _,f in first)),
            unavailable_task_seeds=[seed for seed,c in selected if not c['states']],swaps=swaps))
    coverage_fields = ('cached_policies','candidate_exposures','cached_truth_exposures',
        'unknown_truth_exposures','actual_queries','known_query_truth')
    return dict(tasks=len(compact_tasks),comparisons=comparisons,
        reconstruction=dict(recorded_choices=sum(t['reconstruction']['recorded_choices']
            for t in compact_tasks),max_score_error=max((t['reconstruction']['max_score_error']
            for t in compact_tasks),default=0.)),
        truth_coverage={k:sum(t['truth_coverage'][k] for t in compact_tasks) for k in coverage_fields},
        undefined_baseline_seeds=[t['seed'] for t in compact_tasks if not t['baseline_defined']],
        limits=['Cached policies are task-local, not the whole candidate universe.',
            'Candidate exposures repeat policies; fixed-state recovery counts are tasks.',
            'No counterfactual trajectory, certificate or deployment is inferred.'])


def candidate_path(point, bundle):
    """Trace one exact physical policy through the three saved pre-query pools."""
    bank, audit, history = bundle['bank'], bundle['audit'], bundle['history']
    rows = []
    for stage in audit['stages']:
        actions = [bank[i] for i in stage['action_ids']]
        base = dict(paid_prefix=stage['paid_prefix'], present=point in actions)
        if point not in actions:
            rows.append(base)
            continue
        j = actions.index(point)
        score = stage['score']
        sampled_f = np.asarray(score['sampled_objective'])
        mean_f = np.asarray(score['objective_mean'])
        margin = np.asarray(score['sampled_margin'])
        eligible = margin <= 0.
        selected = j == stage['selected_index']
        mode = score['sampled_feasibility_mode']
        reason = ('selected' if selected else 'minimum_violation_competition'
            if mode == 'minimum_violation' else 'risk_excluded' if not eligible[j]
            else 'objective_competition')
        rows.append(dict(**base, bank_id=stage['action_ids'][j],
            action_source=stage['action_sources'][j],
            previously_observed=point in {policy_key(r['point'])
                for r in history[:stage['paid_prefix']]}, selected=selected,
            reason=reason, selection_mode=mode, sampled_feasible=bool(eligible[j]),
            sampled_objective_rank_all=int(1+np.sum(sampled_f < sampled_f[j])),
            sampled_objective_rank_within_feasible=(
                int(1+np.sum(eligible & (sampled_f < sampled_f[j]))) if eligible[j] else None),
            objective_mean_rank_all=int(1+np.sum(mean_f < mean_f[j])),
            **{k:float(score[k][j]) for k in ('objective_mean', 'objective_epistemic',
                'sampled_objective', 'constraint_mean', 'constraint_aleatoric',
                'constraint_epistemic', 'sampled_constraint', 'sampled_margin',
                'stochastic_margin_mean', 'probability_violation')}))
    observed = [i for i, row in enumerate(history) if policy_key(row['point']) == point]
    prefixes = [r['paid_prefix'] for r in rows if r['present']]
    return dict(present_in_saved_bank=point in bank,
        first_candidate_paid_prefix=min(prefixes) if prefixes else None,
        candidate_paid_prefixes=prefixes, observation_history_indices=observed,
        first_observation_history_index=observed[0] if observed else None,
        adaptive_query_history_indices=[i for i in observed if i >= 10], states=rows)


def terminal_path(point, item, record, status):
    """Saved exposure/outcome only; absent tests have unknown counterfactual results."""
    if item is None:
        return dict(status=status, disposition='terminal_input_unavailable',
            counterfactual_certificate=None, counterfactual_objective_comparison=None)
    match = next(((i, r) for i,r in enumerate(item['frozen_shortlist'])
        if policy_key(r['point']) == point), None)
    if match is None:
        return dict(status=status, shortlist_position=None, disposition='not_on_frozen_shortlist',
            counterfactual_certificate=None, counterfactual_objective_comparison=None)
    index, entry = match
    result = dict(status=status, shortlist_position=index+1,
        shortlist_role=entry.get('shortlist_role'), posterior_rank=entry.get('posterior_rank'),
        recorded_probability_violation=entry.get('selected_posterior_violation_probability',
            entry.get('selected_probability_violation')),
        recorded_posterior_objective=entry.get('selected_posterior_objective'))
    if record is None:
        return dict(**result, disposition='verification_unavailable',
            counterfactual_certificate=None, counterfactual_objective_comparison=None)
    verification = record['verification']
    safety = next((r for r in verification['attempts'] if r['candidate_index']==index), None)
    deployed = bool(verification['certified'] and policy_key(record['deployed'])==point)
    disposition = ('deployed' if deployed else 'not_verified_after_earlier_terminal_decision'
        if safety is None else 'safety_certificate_failed' if not safety['certified']
        else 'objective_guard_retained_other_policy')
    comparisons = verification.get('objective_comparisons')
    if comparisons is None:
        comparison = verification.get('objective_comparison', {})
        comparisons = [dict(comparison, challenger_index=0,
            incumbent_index=verification['objective_incumbent_position']-1)]
    fields = ('block_index','challenger_index','incumbent_index','winner_index','status',
              'challenger_dominates','sample_mean_difference','one_sided_upper_confidence_bound')
    relevant = [{k:r[k] for k in fields if k in r} for r in comparisons
        if index in (r.get('challenger_index'), r.get('incumbent_index'))]
    return dict(**result, disposition=disposition, deployed=deployed,
        safety_attempted=safety is not None,
        safety_passed=safety['certified'] if safety else None,
        safety_upper_margin=safety['upper_margin'] if safety else None,
        actual_objective_comparisons=relevant,
        counterfactual_certificate=None, counterfactual_objective_comparison=None)


def analyze_task(root, plan, seed):
    """Return task, states, paired_states, and AFTER deployment_paths for one seed.

    ``plan.source_manifest`` points to Stage65 and ``source_output_dir`` is optional.
    All data are read-only; caller owns scheduling, metering and output writes.
    """
    root = Path(root)
    source = read(root / plan['source_manifest'])
    output = root / plan.get('source_output_dir', source['output_dir'])
    seal = read(output / 'execution_status.json')
    cases = {r['arm']:r for r in seal['cases'] if r['seed']==seed and r['arm'] in ARMS}
    anchor = cases[AFTER]
    directory = (root / anchor['source_case']).parent
    terminals, bundles, failed = {}, {}, []
    for arm in ARMS:
        case = cases.get(arm)
        path = root / case['source_case'] if case else directory / arm
        if not all((path/name).is_file() for name in
                   ('search.json','query_audit.json','query_policies.npz')):
            failed.append(arm)
        else:
            with np.load(path/'query_policies.npz') as data:
                bank = [policy_key(p) for p in data['points']]
            bundles[arm] = dict(bank=bank, audit=read(path/'query_audit.json'),
                history=read(path/'search.json')['history'])
        for protocol in ('old', 'new'):
            descriptor = case[protocol] if case else dict(status='missing')
            item = read(root/descriptor['input_file']) if descriptor.get('input_file') else None
            record = (read(root/descriptor['verification_file'])
                if descriptor['status']=='complete' else None)
            terminals[arm, protocol] = item, record, descriptor['status']
    alpha = terminals[AFTER, 'new'][0]['task']['alpha']
    task, states, pairs, errors = audit_task(anchor['domain'], seed, directory,
        float(norm.ppf(1-alpha)), failed_arms=failed)
    # Keep old fixed-state score diagnostics; do not use truth-removal oracle patches.
    for state in states:
        for candidate in state.get('good_candidates') or []:
            candidate.pop('candidate_only_risk_removal', None)

    initial = [policy_key(r['point']) for r in read(directory/'initialization.json')['history']]
    with np.load(directory/'neighbors.npz') as data:
        neighbors = [policy_key(p) for p in data['points']]
    truth = {policy_key(r['point']):r['truth'] for r in read(directory/'truth.json')}

    def identity(point):
        ii = [i for i,p in enumerate(initial) if p==point]
        ni = neighbors.index(point) if point in neighbors else None
        locations = [dict(arm=arm, history_index=i) for arm,b in bundles.items()
            for i,r in enumerate(b['history']) if policy_key(r['point'])==point]
        origin = 'initial' if ii else 'shared_neighbor' if ni is not None else 'dynamic'
        first = locations[0] if locations else None
        label = (f'initial:{ii[0]}' if ii else f'neighbor:{ni}' if ni is not None
            else f'{first["arm"]}:history:{first["history_index"]}' if first else None)
        return dict(policy=label, origin=origin, initial_indices=ii, neighbor_index=ni,
            observed_locations=locations, truth=truth.get(point))

    for state in states:
        bank = bundles[state['arm']]['bank']
        for choice in state['choices'].values():
            choice['selected']['identity'] = identity(bank[choice['selected']['policy_id']])
        for candidate in state.get('good_candidates') or []:
            candidate['identity'] = identity(bank[candidate['policy_id']])
    for pair in pairs:
        pair['scope'] = ('same_history_same_pool' if pair['same_paid_history'] and
            pair['same_physical_pool'] else 'different_adaptive_histories_or_pools')
        causal_first = (pair['first_divergence'] and pair['same_paid_history'] and
                        pair['same_physical_pool'])
        pair['substitutions'] = [s for s in pair['substitutions'] if causal_first and
            s['substitution'] in ('risk_risk_mean+noise_margin+risk_draw', 'all_objective')]
        for arm in (pair['left'],pair['right']):
            bundle = bundles[arm]
            stage = next(s for s in bundle['audit']['stages']
                if s['paid_prefix']==pair['paid_prefix'])
            pair['selected'][arm]['identity'] = identity(bundle['bank'][stage['selected_id']])

    deployments = []
    for protocol in ('old','new'):
        _, record, status = terminals[AFTER,protocol]
        if record is None:
            deployments.append(dict(protocol=protocol,status=status,after_policy=None,paths=[]))
            continue
        point = policy_key(record['deployed'])
        paths = []
        for arm in ARMS:
            paths.append(dict(arm=arm, search=(candidate_path(point,bundles[arm])
                if arm in bundles else None),
                terminals={p:terminal_path(point,*terminals[arm,p]) for p in ('old','new')}))
        deployments.append(dict(protocol=protocol,status=status,
            after_certified=bool(record['verification']['certified']), after_policy=identity(point),
            policy_identity_basis='Exact physical integer vector, never cross-arm history label equality',
            paths=paths))
    return dict(status='complete',seed=seed,domain=anchor['domain'], task=task,
        states=states,paired_states=pairs,deployment_paths=deployments,
        reconstruction=dict(recorded_choices=len(errors),max_score_error=max(errors,default=0.)),
        limits=['Only cached truth is used; absent dynamic/shared-neighbor truth remains unknown.',
            'Same-history, same-pool first divergence isolates saved score changes; later contrasts are descriptive.',
            'Component substitutions are fixed-state choices, not a new trajectory or deployment outcome.',
            'Prefix13 final query scores were not saved; absent certificates and objective tests remain unknown.'])
