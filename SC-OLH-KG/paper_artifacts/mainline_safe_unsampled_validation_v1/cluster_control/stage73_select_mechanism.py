"""Select existing Stage73 JSON fields without recomputing any scientific metric."""
import json
from pathlib import Path
import sys

AFTER = 'legacy_role_after_init_point_only'
POINT = 'legacy_role_point_only'


def take(row, fields):
    return {field: row[field] for field in fields if field in row}


saved = json.loads(Path(sys.argv[1]).read_text())
tasks = []
for task in saved['mechanism']['tasks']:
    states = []
    for row in task['states']:
        item = take(row, ('arm', 'paid_prefix', 'status', 'replay', 'counts',
                         'safe_competition', 'safe_omission_counts'))
        if 'selected' in row:
            item['selected'] = take(row['selected'], ('policy_id', 'origin', 'observed',
                'true_feasible', 'eligible', 'truth', 'mean_rank_within_eligible',
                'sampled_rank_within_eligible'))
        states.append(item)
    chains = []
    for row in task['query_chains']:
        if row['arm'] not in (AFTER, POINT):
            continue
        item = take(row, ('arm', 'paid_prefix', 'policy_id', 'origin', 'truth',
            'previously_observed', 'eligible', 'observation_returned',
            'returned_policy_matches', 'backend_conditioned', 'exposures'))
        item['terminals'] = [dict(arm=terminal['arm'],
            history_arm=terminal.get('history_arm'), old=terminal['protocols']['old'])
            for terminal in row['terminals'] if terminal['arm'] in (AFTER, POINT)]
        chains.append(item)
    comparisons = []
    for row in task['comparisons']:
        if row['contrast'] != 'AP' or not (row['paid_prefix'] == 10 or row.get('first_divergence')):
            continue
        item = take(row, ('contrast', 'paid_prefix', 'status', 'same_paid_history',
            'same_physical_pool', 'common_pool_count', 'common_unobserved_count',
            'query_changed', 'selected', 'first_divergence', 'scope', 'substitutions'))
        item['eligibility_groups'] = [take(group, ('group', 'count', 'true_feasible',
            'truth_unknown', 'origins')) for group in row.get('eligibility_groups', [])]
        comparisons.append(item)
    tasks.append(dict(seed=task['seed'], domain=task['domain'], status=task['status'],
        coverage=task['coverage'], states=states, query_chains=chains,
        terminal_effects=[row for row in task['terminal_effects']
            if row['contrast'] == 'AP' and row['protocol'] == 'old'],
        comparisons=comparisons))
Path(sys.argv[2]).write_text(json.dumps(dict(status=saved['status'], tasks=tasks), separators=(',', ':'))+'\n')
