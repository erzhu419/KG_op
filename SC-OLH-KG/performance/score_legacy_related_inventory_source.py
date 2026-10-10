"""Safe reward accounting for the saved pilot; no refit or simulation."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.guarded_deployment_kg import (
    conditional_comparison_power, conditional_safety_power, shortlist_outcomes)
from performance.benchmark_lodo_meta_prior import build_scalarized_problem
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.replay_legacy_mixture_budget import terminal_at_prefix


def safe_terminal_values(terminal, truth, feasible, remaining):
    """Zero utility at truly infeasible policies, preserving comparison rules."""
    protocol = terminal.protocol
    quality = np.clip(1-truth[:, 0]/protocol['objective_reference'], 0, 1) * feasible
    sf, sg = np.exp(truth[:, 1]/2), np.exp(truth[:, 3]/2)
    rewards, costs = [], []
    for _, table, incumbents, budgets in terminal.allocations(remaining):
        delta = protocol['familywise_delta']/len(budgets)
        powers = {n: conditional_safety_power(truth[:, 2], sg, n, protocol['chance_alpha'], delta)
                  for n in set(budgets)}
        for indices, incumbent in zip(table, incumbents):
            primary, support = indices[0], indices[incumbent]
            comparison = conditional_comparison_power(truth[primary, 0]-truth[support, 0],
                abs(sf[primary]-sf[support]), protocol['comparison_budget'], protocol['comparison_delta'])
            p = np.array([powers[n][i] for i, n in zip(indices, budgets)])
            r, c = shortlist_outcomes(quality[indices], p, comparison, budgets,
                                      int(incumbent), protocol['comparison_budget'])
            rewards.append(float(r)); costs.append(float(c))
    return np.asarray(rewards), np.asarray(costs)


def add_safe_scores(result, terminal, fold, rows):
    truth = np.asarray(fold['scoring_truth'])
    feasible = np.isin(np.arange(len(truth)), fold['feasible_policies'])
    cap = result['execution']['total_target_call_cap']
    count = result['execution']['adaptation_calls']
    safe, costs = safe_terminal_values(terminal, truth, feasible, cap-count)
    original, _, actions = terminal.evaluate(truth[None], cap-count)
    best = int(np.lexsort((costs, -safe))[0])
    fold['best_safe_same_class_action'] = {'action': actions[best], 'reward': float(safe[best]),
                                          'total_calls': float(costs[best]+count)}
    for row in rows:
        index = row['action_index']
        if actions[index] != row['action']:
            raise ValueError('saved action does not match the reconstructed frozen terminal class')
        row['truth']['safe_reward'] = float(safe[index])
        row['truth']['false_certification_reward_contribution'] = float(original[index]-safe[index])
        row['truth']['loss_to_best_safe_same_class_action'] = float(safe[best]-safe[index])
    result['scoring_convention'] = ('truth.reward retains certificate-weighted quality, including false certificates; '
        'truth.safe_reward gives infeasible policies zero utility with identical verifier/comparison branches. '
        'All recommendations remain those frozen before scoring.')


def run():
    folder = ROOT / 'paper_artifacts/legacy_related_inventory_source_v1'
    path = folder / 'results.json'
    result = json.loads(path.read_text()); raw = json.loads((folder/'observations.json').read_text())
    plan, execution = result['plan'], result['execution']
    source_spec = json.loads((ROOT/execution['legacy_replay']).read_text())['source_fit_contract']
    bank = [tuple(x) for x in raw['bank']]
    for fold in result['folds']:
        task = next(t for t in raw['tasks'] if t['task_id'] == fold['task_id'])
        d = plan['task_distribution']
        provider = no_outcomes(build_scalarized_problem(task['family'], plan['source_dimension'], d['L'],
            d['sigma'], d['alpha'], d['weights'], problem_kwargs=task['problem_kwargs']))
        count = execution['adaptation_calls']
        observed = [v[0] for v in task['Y_replicates'][:count]]
        terminal, _ = terminal_at_prefix(provider, bank[:count], {'paid_observations': observed}, count,
                                         source_spec, len(bank))
        terminal.retain_recommendation(fold['retained_policy'])
        add_safe_scores(result, terminal, fold, [r for r in result['rows'] if r['task_id'] == fold['task_id']])
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'scored_decisions': len(result['rows']), 'new_simulator_calls': 0, 'source_refits': 0}))


if __name__ == '__main__':
    run()
