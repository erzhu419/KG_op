"""Small exact-native direct-action probe for the four reached primaries."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.budgeted_guarded_kg import BudgetedTerminalActions
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.check_legacy_source_layout_robustness import models_at_query, ARMS
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.replay_legacy_mixture_budget import terminal_at_prefix
from performance.run_legacy_joint_population_independent_pilot import target_problem
from performance.run_legacy_related_inventory_source import save
from performance.score_legacy_related_inventory_source import safe_terminal_values

EXECUTION = 'performance/manifests/legacy_joint_population_source_layout_reached_actions_v1.json'


class ReachedDirectActions(BudgetedTerminalActions):
    def __init__(self, protocol, candidates):
        super().__init__(np.asarray(candidates)[:,None], np.zeros(len(candidates), dtype=int), protocol)

    def allocations(self, remaining_calls):
        return [('direct', self.shortlists, self.incumbents, [int(n)])
                for n in dict.fromkeys([min(128, remaining_calls), remaining_calls])]


def action_list(terminal, remaining):
    return [{'kind': kind, 'primary': int(indices[0]), 'shortlist': indices.tolist(),
        'incumbent_position': int(incumbent), 'budgets': budgets}
        for kind, table, inc, budgets in terminal.allocations(remaining) for indices, incumbent in zip(table, inc)]


def run(execution=EXECUTION):
    started = time.process_time(); spec = read(execution); source = read(spec['source_execution'])
    snapshot = read(spec['target_snapshot'])['snapshots']; inputs = read(spec['target_inputs'])
    reference_execution = read(spec['reference_execution'])
    risk_cache = read(spec['risk_component_cache'])['rows']; gp_cache = read(spec['gp_component_cache'])['rows']
    reference = read(spec['reference_recommendations'])['rows']
    bank = [tuple(p) for p in inputs['bank']]; remaining = spec['total_target_call_cap']-spec['target_prefix']
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True); rows, jobs = [], {}
    native_components = []
    for target in inputs['targets']:
        task = target['task_id']; provider = no_outcomes(target_problem(inputs['target_plan'], target['config']))
        original, _ = terminal_at_prefix(provider, bank[:2],
            {'paid_observations': target['initialization']['paired_observations']}, 2, inputs['source_spec'], len(bank))
        original.retain_recommendation(reference_execution['public_retained_candidate'])
        terminal = ReachedDirectActions(original.protocol, spec['candidates']); jobs[task] = terminal
        actions = action_list(terminal, remaining); full_actions = action_list(original, remaining)
        ix = [full_actions.index(a) for a in actions]
        for arm in ARMS:
            old = (next(r for r in gp_cache if r['task_id']==task) if arm=='joint_gp'
                   else next(r for r in risk_cache if r['task_id']==task and r['mean_arm']==arm))
            reward = np.asarray(old['identity_weights'])@np.asarray(old['rewards'])[:,ix]
            cost = np.asarray(old['identity_weights'])@np.asarray(old['costs'])[:,ix]
            best = int(np.lexsort((cost,-reward))[0])
            prior = next(r for r in reference if r['task_id']==task and r['mean_arm']==arm)
            if actions[best] != prior['action']:
                raise ValueError('reached-action subset does not reproduce original layout0 choice')
            rows.append({'layout':'0','task_id':task,'mean_arm':arm,'action_index':best,'action':actions[best],
                         'predicted_reward':float(reward[best]),'predicted_verification_calls':float(cost[best])})
    for name in source['layouts']:
        folder = Path(source['output'])/f'layout{name}'
        members = [(read(folder/'inputs.json'),read(folder/'source_fit.json'),read(folder/'risk_source_fit.json'))]
        members += [(read(folder/f'task{i}'/'inputs.json'),read(folder/f'task{i}'/'gp_source_fit.json'),
                     read(folder/f'task{i}'/'risk_source_fit.json')) for i in source['training_tasks']]
        for target in inputs['targets']:
            task = target['task_id']; frozen = next(r for r in snapshot if r['layout']==name and r['query']['task_id']==task)
            query = frozen['query']; terminal = jobs[task]
            for arm, model in models_at_query(members, frozen['mean_queries'], frozen['risk_queries']):
                for j,value in zip(query['candidate_indices'],query['paired_observations']):
                    model.update_raw(j,[value[0],value[1]-query['tau']])
                r,c,actions = model.terminal_components(terminal,remaining,spec['terminal_draw_power'],spec['integration_seed'])
                reward,cost = model.weights@r,model.weights@c; best = int(np.lexsort((cost,-reward))[0])
                rows.append({'layout':name,'task_id':task,'mean_arm':arm,'action_index':best,'action':actions[best],
                             'predicted_reward':float(reward[best]),'predicted_verification_calls':float(cost[best])})
                native_components.append({'layout':name,'task_id':task,'mean_arm':arm,
                    'identity_weights':model.weights.tolist(),'rewards':r.tolist(),'costs':c.tolist()})
            print('reached-action choices frozen', name, task, [r['action']['primary'] for r in rows[-3:]], flush=True)
    save(out/'recommendations.json',{'execution':spec,'rows':rows})
    save(out/'component_utilities.json',{'execution':spec,'rows':native_components})
    # All54 choices precede truth scoring; source0 comparison scores open only now.
    scoring = read(spec['target_scoring']); reference_scores = read(spec['reference_results'])['rows']
    for target in inputs['targets']:
        task = target['task_id']; terminal = jobs[task]
        fold = next(f for f in scoring['folds'] if f['task_id']==task)
        truth = np.asarray(fold['scoring_truth']); feasible = np.isin(np.arange(len(bank)),fold['feasible_policies'])
        safe,calls = safe_terminal_values(terminal,truth,feasible,remaining)
        for row in (r for r in rows if r['task_id']==task):
            j = row['action_index']; row['scores'] = {'conditional_safe_reward':float(safe[j]),
                'conditional_total_calls':float(2+calls[j]),'primary_feasible':bool(feasible[row['action']['primary']])}
            if row['layout']=='0':
                old = next(r for r in reference_scores if r['task_id']==task and r['mean_arm']==row['mean_arm'])
                for k in ('conditional_safe_reward','conditional_total_calls'):
                    np.testing.assert_allclose(row['scores'][k],old['scores'][k],rtol=1e-8,atol=1e-8)
    summary = {name:{arm:{'feasible_primary_count':sum(r['scores']['primary_feasible'] for r in rows if r['layout']==name and r['mean_arm']==arm),
        **{k:float(np.mean([r['scores'][k] for r in rows if r['layout']==name and r['mean_arm']==arm]))
            for k in ('conditional_safe_reward','conditional_total_calls')}} for arm in ARMS} for name in spec['layouts']}
    ids = [t['task_id'] for t in inputs['targets']]
    draws = np.random.default_rng(spec['bootstrap_seed']).integers(len(ids),size=(spec['bootstrap_draws'],len(ids)))
    comparisons = []
    for name in spec['layouts']:
        for arm in ARMS[1:]:
            d = np.array([next(r for r in rows if r['layout']==name and r['task_id']==task and r['mean_arm']=='joint_structural')['scores']['conditional_safe_reward']
                -next(r for r in rows if r['layout']==name and r['task_id']==task and r['mean_arm']==arm)['scores']['conditional_safe_reward'] for task in ids])
            comparisons.append({'layout':name,'reference':arm,'task_differences':d.tolist(),'mean_difference':float(d.mean()),
                'task_bootstrap_95_interval':np.quantile(d[draws].mean(axis=1),[.025,.975]).tolist()})
    result = {'execution':spec,'rows':rows,'summary':summary,'paired_comparisons':comparisons,
        'layout0_recommendation_and_score_reproduction_passed':True,
        'cost':{'new_source_fits':0,'new_simulator_calls':0,'new_verification_calls':0,
            'new_component_integrations':252,'reused_layout0_choices':18,'actions_per_component':8},
        'cpu_seconds':time.process_time()-started}
    save(out/'results.json',result); print({'summary':summary,'cpu_seconds':result['cpu_seconds']},flush=True)


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else EXECUTION)
