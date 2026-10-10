"""Frozen cached choices under single and source-uncertainty risk laws."""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import models_at_query, read
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.replay_legacy_mixture_budget import terminal_at_prefix
from performance.run_legacy_joint_population_independent_pilot import target_problem
from performance.run_legacy_related_inventory_source import save
from performance.score_legacy_related_inventory_source import safe_terminal_values

EXECUTION = 'performance/manifests/legacy_joint_population_public_mean_risk_uncertainty_decision_v1.json'


def run():
    spec = read(EXECUTION); stage = read(spec['prediction_results'])
    if not stage['baseline_reproduction_passed']:
        raise RuntimeError('Stage35 single-risk forecasts must reproduce before action attribution')
    prediction = read(spec['prediction_execution']); frozen = read(spec['prediction_inputs'])
    inputs, means = read(prediction['source_inputs']), read(prediction['mean_fit'])
    target_inputs = read(spec['target_inputs'])
    forecasts = read(Path(prediction['output'])/'target_predictions.json')['rows']
    bank = [tuple(p) for p in target_inputs['bank']]
    n0 = spec['target_prefix']; remaining = spec['total_target_call_cap']-n0
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    rows, component_values, terminals = [], [], {}; started = time.process_time()
    for target in target_inputs['targets']:
        task = target['task_id']; query = next(q for q in frozen['queries'] if q['task_id'] == task)
        provider = no_outcomes(target_problem(target_inputs['target_plan'], target['config']))
        terminal, _ = terminal_at_prefix(provider, bank[:n0],
            {'paid_observations': query['paired_observations']}, n0, target_inputs['source_spec'], len(bank))
        terminal.retain_recommendation(spec['public_retained_candidate']); terminals[task] = terminal
        for a, case, model in models_at_query(prediction, inputs, means, frozen['risk_members'], query):
            if case != spec['risk_cases'][1]:
                continue
            old = next(r for r in forecasts if r['task_id'] == task and r['mean_arm'] == a and r['risk_case'] == case)
            mean, covariance = model.predict_joint()
            variance = np.diag(covariance.reshape(4*len(bank), 4*len(bank))).reshape(-1, 4)
            for field, value in [('mean', mean), ('variance', variance)]:
                np.testing.assert_allclose(value[:, [0, 2, 3, 1]], old[field], rtol=1e-8, atol=1e-9)
            rewards, costs, actions = model.terminal_components(terminal, remaining,
                spec['terminal_draw_power'], spec['integration_seed'])
            component_values.append({'task_id': task, 'mean_arm': a, 'identity_names': model.names,
                'identity_weights': model.weights.tolist(), 'rewards': rewards.tolist(), 'costs': costs.tolist()})
            for risk, reward, cost in [(spec['risk_cases'][0], rewards[0], costs[0]),
                                      (case, model.weights@rewards, model.weights@costs)]:
                best = int(np.lexsort((cost, -reward))[0])
                order = np.lexsort((cost, -reward))
                rows.append({'task_id': task, 'mean_arm': a, 'risk_case': risk,
                    'arm': a+'/'+risk, 'action_index': best, 'action': actions[best],
                    'predicted_reward': float(reward[best]), 'predicted_verification_calls': float(cost[best]),
                    'predicted_first_second_reward_gap': float(reward[best]-reward[order[1]])})
            print('risk uncertainty actions frozen', task, a,
                  [(r['risk_case'], r['action']['primary']) for r in rows[-2:]], flush=True)
            save(out/'recommendations.json', {'execution': spec, 'all_choices_frozen': False, 'rows': rows})
            save(out/'component_utilities.json', {'execution': spec, 'rows': component_values})
    # Truth is first opened here, after all36 choices and component values are saved.
    save(out/'recommendations.json', {'execution': spec, 'all_choices_frozen': True, 'rows': rows})
    scoring = read(spec['target_scoring']); oracle = []
    for target in target_inputs['targets']:
        task = target['task_id']; terminal = terminals[task]
        fold = next(f for f in scoring['folds'] if f['task_id'] == task)
        truth = np.asarray(fold['scoring_truth'])
        feasible = np.isin(np.arange(len(bank)), fold['feasible_policies'])
        safe, calls = safe_terminal_values(terminal, truth, feasible, remaining)
        _, _, actions = terminal.evaluate(truth[None], remaining)
        best = int(np.lexsort((calls, -safe))[0])
        oracle.append({'task_id': task, 'conditional_safe_reward': float(safe[best]), 'action': actions[best]})
        for row in (r for r in rows if r['task_id'] == task):
            j = row['action_index']
            if row['action'] != actions[j]:
                raise ValueError('frozen action differs from reconstructed scoring class')
            row['scores'] = {'conditional_safe_reward': float(safe[j]), 'conditional_total_calls': float(n0+calls[j]),
                'primary_feasible': bool(feasible[row['action']['primary']]),
                'loss_to_best_same_class': float(safe[best]-safe[j])}
    ids = [t['task_id'] for t in target_inputs['targets']]; names = list(dict.fromkeys(r['arm'] for r in rows))
    summary = {a: {'feasible_primary_count': sum(r['scores']['primary_feasible'] for r in rows if r['arm'] == a),
        **{k: float(np.mean([r['scores'][k] for r in rows if r['arm'] == a]))
           for k in ('conditional_safe_reward', 'conditional_total_calls', 'loss_to_best_same_class')}} for a in names}
    draws = np.random.default_rng(spec['bootstrap_seed']).integers(len(ids), size=(spec['bootstrap_draws'], len(ids)))
    pairs = [(a+'/'+spec['risk_cases'][1], a+'/'+spec['risk_cases'][0]) for a in spec['mean_arms']]
    pairs += [('joint_structural/'+c, a+'/'+c) for c in spec['risk_cases'] for a in ('flat', 'joint_gp')]
    comparisons = []
    for a, b in pairs:
        metrics = {}
        for k in ('conditional_safe_reward', 'conditional_total_calls'):
            d = np.array([next(r for r in rows if r['task_id'] == t and r['arm'] == a)['scores'][k]
                         -next(r for r in rows if r['task_id'] == t and r['arm'] == b)['scores'][k] for t in ids])
            metrics[k] = {'task_differences': d.tolist(), 'mean_difference': float(d.mean()),
                         'task_bootstrap_95_interval': np.quantile(d[draws].mean(axis=1), [.025, .975]).tolist()}
        comparisons.append({'arm': a, 'reference': b, 'metrics': metrics})
    result = {'execution': spec, 'rows': rows, 'summary': summary, 'paired_comparisons': comparisons,
        'same_class_oracle': oracle, 'stage35_forecast_reproduction_passed': True,
        'cost': {'historical_source_calls_per_arm': 144, 'new_source_fits': 0, 'new_simulator_calls': 0,
            'new_verification_calls': 0, 'new_terminal_integrations': len(component_values)*7,
            'reused_single_risk_integrations': len(component_values)}, 'cpu_seconds': time.process_time()-started}
    save(out/'results.json', result)
    print(json.dumps({'summary': summary, 'cpu_seconds': result['cpu_seconds']}), flush=True)


if __name__ == '__main__':
    run()
