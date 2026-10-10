"""Matched cached actions after the frozen public-mean prediction gate passes."""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_coordinates import components
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.replay_legacy_mixture_budget import terminal_at_prefix
from performance.run_legacy_joint_population_independent_pilot import target_problem
from performance.run_legacy_related_inventory_source import save
from performance.score_legacy_related_inventory_source import safe_terminal_values
from representation.joint_response_mixture import JointResponseMixture


def run():
    spec = json.loads((ROOT/'performance/manifests/legacy_joint_population_public_mean_decision_v1.json').read_text())
    stage = json.loads((ROOT/spec['prediction_results']).read_text())
    if not stage['terminal_progress_gate_passed']:
        raise RuntimeError('public mean source and target prediction gates must pass before target choices')
    source = stage['execution']
    inputs = json.loads((ROOT/spec['inputs']).read_text())
    fits = json.loads((ROOT/spec['source_fit']).read_text())
    old_inputs = json.loads((ROOT/source['source_inputs']).read_text())
    old_fits = json.loads((ROOT/source['source_fit']).read_text())
    bank = [tuple(p) for p in inputs['bank']]
    names = spec['arms']; rows, jobs = [], {}; started = time.process_time()
    n0 = spec['target_prefix']; remaining = spec['total_target_call_cap']-n0
    for target in inputs['targets']:
        provider = no_outcomes(target_problem(inputs['target_plan'], target['config']))
        model = JointResponseMixture(names, components(source, inputs, old_inputs, fits, old_fits,
            np.asarray(target['mean_psi']), np.asarray(target['risk_psi'])))
        observed = target['initialization']['paired_observations']
        for j, value in zip(target['initialization']['candidate_indices'], observed):
            model.update_raw(j, [value[0], value[1]-provider.tau])
        terminal, _ = terminal_at_prefix(provider, bank[:n0], {'paid_observations': observed}, n0,
                                        inputs['source_spec'], len(bank))
        terminal.retain_recommendation(spec['public_retained_candidate'])
        reward, cost, actions = model.terminal_components(terminal, remaining,
            spec['terminal_draw_power'], spec['integration_seed'])
        forecasts = {}
        for k, (name, (f, g)) in enumerate(zip(names, model.components)):
            best = int(np.lexsort((cost[k], -reward[k]))[0])
            fm, fv = f.predict(); gm, gv = g.predict()
            forecasts[name] = (np.column_stack([fm, gm]), np.column_stack([fv, gv]))
            rows.append({'task_id': target['task_id'], 'arm': name, 'action_index': best,
                'action': actions[best], 'predicted_reward': float(reward[k, best]),
                'predicted_verification_calls': float(cost[k, best])})
        jobs[target['task_id']] = (terminal, forecasts)
        print('public mean target actions', target['task_id'],
              [r['action']['primary'] for r in rows[-len(names):]], flush=True)
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    # All 42 actions are frozen before reading cached outcome/feasibility truth.
    save(out/'recommendations.json', {'execution': spec, 'rows': rows})
    scoring = json.loads((ROOT/spec['target_scoring']).read_text())
    for target in inputs['targets']:
        fold = next(f for f in scoring['folds'] if f['task_id'] == target['task_id'])
        terminal, forecasts = jobs[target['task_id']]
        truth = np.asarray(fold['scoring_truth'])  # [f, eta_f, g, eta_g]
        feasible = np.isin(np.arange(len(bank)), fold['feasible_policies'])
        safe, calls = safe_terminal_values(terminal, truth, feasible, remaining)
        _, _, actions = terminal.evaluate(truth[None], remaining)
        for row in (r for r in rows if r['task_id'] == target['task_id']):
            a = row['action_index']
            if actions[a] != row['action']:
                raise ValueError('frozen public mean action differs from the scored class')
            row['scores'] = {'conditional_safe_reward': float(safe[a]),
                'conditional_total_calls': float(n0+calls[a]),
                'primary_feasible': bool(feasible[row['action']['primary']])}
            mean, variance = forecasts[row['arm']]; error = mean[n0:]-truth[n0:]
            row['forecast'] = {'head_order': ['f', 'eta_f', 'g', 'eta_g'],
                'truth_mse': (error**2).mean(axis=0).tolist(),
                'moment_interval_95_truth_coverage': (abs(error) <= 1.96*np.sqrt(variance[n0:])).mean(axis=0).tolist(),
                'diagnostic_points': [{'candidate': j, 'mean': mean[j].tolist(),
                    'sd': np.sqrt(variance[j]).tolist(), 'scoring_truth': truth[j].tolist()}
                    for j in dict.fromkeys([row['action']['primary'], 27, 60, 63])]}
    # The unchanged legacy controls must reproduce the preceding action contract.
    reference = json.loads((ROOT/spec['reference_decisions']).read_text())
    for name, old_name in (('legacy_mean_joint_structural', 'joint_common_structural'),
                           ('legacy_full_joint_rbf', 'joint_common_rbf')):
        for row in (r for r in rows if r['arm'] == name):
            old = next(r for r in reference['rows'] if r['task_id'] == row['task_id'] and r['arm'] == old_name)
            if row['action'] != old['action']:
                raise ValueError('unchanged legacy control does not reproduce its saved action')
            for metric in ('conditional_safe_reward', 'conditional_total_calls'):
                np.testing.assert_allclose(row['scores'][metric], old['scores'][metric], rtol=1e-8, atol=1e-8)
    ids = [t['task_id'] for t in inputs['targets']]
    draws = np.random.default_rng(spec['bootstrap_seed']).integers(len(ids),
        size=(spec['bootstrap_draws'], len(ids)))
    summary = {name: {'feasible_primary_count': sum(r['scores']['primary_feasible'] for r in rows if r['arm'] == name),
        **{m: float(np.mean([r['scores'][m] for r in rows if r['arm'] == name]))
           for m in ('conditional_safe_reward', 'conditional_total_calls')},
        'truth_mse': np.mean([r['forecast']['truth_mse'] for r in rows if r['arm'] == name], axis=0).tolist()}
        for name in names}
    comparisons = []; candidate = spec['main_arm']
    for name in (n for n in names if n != candidate):
        metrics = {}
        for metric in ('conditional_safe_reward', 'conditional_total_calls'):
            differences = np.array([next(r for r in rows if r['task_id'] == t and r['arm'] == candidate)['scores'][metric]
                -next(r for r in rows if r['task_id'] == t and r['arm'] == name)['scores'][metric] for t in ids])
            metrics[metric] = {'task_differences': differences.tolist(), 'mean_difference': float(differences.mean()),
                'task_bootstrap_95_interval': np.quantile(differences[draws].mean(axis=1), [.025, .975]).tolist()}
        comparisons.append({'arm': candidate, 'reference': name, 'metrics': metrics})
    result = {'execution': spec, 'rows': rows, 'summary': summary, 'paired_comparisons': comparisons,
        'legacy_control_reproduction_passed': True,
        'cost': {'new_source_calls': 0, 'new_target_calls': 0, 'new_verification_calls': 0,
            'historical_source_response_calls_per_arm': 288, 'new_source_hyperparameter_fits': 0,
            'new_terminal_integrations': len(names)*len(ids)}, 'cpu_seconds': time.process_time()-started}
    save(out/'results.json', result)
    print(json.dumps({'summary': summary, 'cpu_seconds': result['cpu_seconds']}), flush=True)


if __name__ == '__main__':
    run()
