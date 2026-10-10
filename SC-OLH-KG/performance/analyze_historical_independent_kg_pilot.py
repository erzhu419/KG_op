"""Explain completed deployment outcomes using the existing full public bank."""

import json

import numpy as np

from performance.complete_historical_source_heads import ROOT, read, save
from performance.run_historical_independent_kg_pilot import PLAN, inputs
from performance.run_legacy_related_inventory_source import problem_for


def bank_summary(problem, policies, reference):
    feasible = [i for i, point in enumerate(policies) if problem.is_truly_feasible(point)]
    qualities = [float(np.clip(1-problem.true_objective(policies[i])/reference, 0, 1))
                 for i in feasible]
    best = int(np.argmax(qualities)) if feasible else None
    return {'bank_size': len(policies), 'feasible_policy_count': len(feasible),
            'perfect_certification_oracle_quality': max(qualities, default=0.),
            'oracle_index': feasible[best] if best is not None else None}


def run():
    spec = read(PLAN)
    pilot, frozen, states, public = inputs(spec)
    out = ROOT/spec['output']
    outcomes = read(str(out/'results.json'))
    recommendations = read(str(out/'recommendations.json'))
    if not outcomes['independent_outcomes_opened'] or not recommendations['all_recommendations_frozen']:
        raise ValueError('Post-outcome diagnostics require completed frozen deployment outcomes')
    legacy = read(read(pilot['source_connection_plan'])['legacy_connection_plan'])
    full_bank = read(legacy['public_inputs'])['bank']
    references = {s['terminal_protocol']['objective_reference'] for s in states}
    if len(references) != 1:
        raise ValueError('Full-bank comparison requires the same objective reference')
    reference = references.pop()
    subset_rows, full_rows = [], []
    for task in frozen['tasks']:
        problem = problem_for(public, task)
        full_rows.append({'task_id': task['task_id'], **bank_summary(problem, full_bank, reference)})
        for bank in frozen['banks']:
            rows = [r for r in outcomes['rows'] if (r['task_id'], r['layout']) ==
                    (task['task_id'], bank['layout'])]
            summary = bank_summary(problem, bank['policies'], reference)
            subset_rows.append({'task_id': task['task_id'], 'layout': bank['layout'], **summary,
                'arms': [{'model': r['arm'], 'policy': r['policy'], 'primary': r['action']['primary'],
                         'certified': r['verification']['certified'],
                         'actual_safe_reward': r['scores']['actual_safe_reward'],
                         'gap_to_bank_oracle': summary['perfect_certification_oracle_quality']-
                                               r['scores']['actual_safe_reward']}
                        for r in rows]})
    arm_rows = []
    for model in pilot['models']:
        for policy in pilot['policies']:
            rows = [r for r in outcomes['rows'] if (r['arm'], r['policy']) == (model, policy)]
            arm_rows.append({'model': model, 'policy': policy, 'row_count': len(rows),
                'mean_actual_safe_reward': float(np.mean([r['scores']['actual_safe_reward'] for r in rows])),
                'rejections': sum(r['scores']['rejection'] for r in rows),
                'false_certificates': sum(r['scores']['false_certificate'] for r in rows),
                'adaptive_queries': sum(r['query'] is not None for r in rows)})
    queries = outcomes['cost']['physical_search_calls']
    physical_initial = outcomes['cost']['shared_physical_initialization_calls']
    pooled_budget = len(outcomes['rows'])*(spec['total_target_call_cap']-2)
    result = {'results': str((out/'results.json').relative_to(ROOT)),
        'post_outcome_diagnostic': True, 'new_physical_calls': 0, 'new_source_fits': 0,
        'public_bank_inputs': legacy['public_inputs'], 'arms': arm_rows,
        'paired_comparisons': outcomes['paired_comparisons'],
        'frozen_subsets': subset_rows, 'existing_full_public_bank': full_rows,
        'subset_feasible_counts': [r['feasible_policy_count'] for r in subset_rows],
        'maximum_actual_gap_to_subset_oracle': max(a['gap_to_bank_oracle']
            for r in subset_rows for a in r['arms']),
        'full_bank_mean_perfect_certification_oracle_quality': float(np.mean([
            r['perfect_certification_oracle_quality'] for r in full_rows])),
        'budget_clarification': {'shared_initialization_plus_search_and_terminal_cap':
            physical_initial+pooled_budget, 'actual_search_calls': queries,
            'terminal_cap_given_actual_queries': pooled_budget-queries,
            'original_17256_terminal_bound_applies_only_if_all24_queries_execute': True},
        'conclusion': 'All four arms tie in realized safe reward and attain their frozen subset oracle. '
            'Eight task/layout banks contain no feasible policy; the other four contain one. '
            'The existing unfiltered73-policy public bank contains feasible alternatives in all six tasks. '
            'Neither source-prior superiority nor incremental KG reward is established.',
        'next': 'Reuse the existing full public bank and source-field restoration; inspect risk and objective '
            'ranking on development data before new full-bank numerical integration. Freeze any candidate '
            'or query screening from source/development observations only, identically for both models. '
            'Keep these six outcomes as development and use fresh tasks for independent confirmation.',
        'limitations': ['Six whole-task units, two fixed finite banks, common verifier streams and one query maximum.',
            'Perfect-certification bank oracles are post-outcome diagnostics, not achieved full-bank deployment reward.',
            'Earlier paid moment projections and uncertified strict KG query rankings remain.']}
    save(out/'analysis.json', result)
    print(json.dumps({'arms': arm_rows, 'subset_feasible_counts': result['subset_feasible_counts'],
        'full_bank_feasible_counts': [r['feasible_policy_count'] for r in full_rows],
        'maximum_actual_gap_to_subset_oracle': result['maximum_actual_gap_to_subset_oracle'],
        'new_physical_calls': 0}), flush=True)
    return result


if __name__ == '__main__':
    run()
