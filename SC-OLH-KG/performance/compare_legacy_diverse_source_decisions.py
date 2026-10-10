"""Frozen cached target attribution after the diversified-source gate passes."""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_diverse_source_design import components, encode_diverse_source
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.replay_legacy_mixture_budget import terminal_at_prefix
from performance.run_legacy_joint_population_independent_pilot import target_problem
from performance.run_legacy_related_inventory_source import problem_for, save
from performance.score_legacy_related_inventory_source import safe_terminal_values
from representation.joint_response_mixture import JointResponseMixture
from representation.legacy_observable_prior import legacy_exposure_descriptors


def run():
    spec = json.loads((ROOT/'performance/manifests/legacy_joint_population_diverse_source_decision_v1.json').read_text())
    source_results = json.loads((ROOT/spec['source_results']).read_text())
    if not source_results['source_gate']['passed']:
        raise RuntimeError('diversified source calibration must pass before target choices')
    source = source_results['execution']
    source_inputs = json.loads((ROOT/spec['source_inputs']).read_text())
    fits = json.loads((ROOT/spec['source_fit']).read_text())
    data = json.loads((ROOT/source['source_data']).read_text())
    public = json.loads((ROOT/source['source_data']).with_name('frozen_inputs.json').read_text())
    source_bank = [tuple(x) for x in public['bank']]
    descriptors = {i: legacy_exposure_descriptors(no_outcomes(problem_for(public['plan'], public['tasks'][i])),
        source_bank, 50) for i in source['training_tasks']}
    coordinate, tasks, center, scale = encode_diverse_source(data, source['training_tasks'],
        source['source_site_indices_by_task'], descriptors, public['plan']['task_distribution']['L'])
    # Reconstruct the saved representation; do not re-optimize source priors.
    np.testing.assert_allclose(center, source_inputs['source_feature_center'], atol=0., rtol=0.)
    np.testing.assert_allclose(scale, source_inputs['source_feature_scale'], atol=0., rtol=0.)
    inputs = json.loads((ROOT/spec['target_inputs']).read_text()); bank = [tuple(x) for x in inputs['bank']]
    started = time.process_time(); names = spec['arms']; rows, jobs = [], []
    n0 = spec['target_prefix']; remaining = spec['total_target_call_cap']-n0
    for target in inputs['targets']:
        provider = no_outcomes(target_problem(inputs['target_plan'], target['config']))
        psi = coordinate.transform(legacy_exposure_descriptors(provider, bank, 50))
        model = JointResponseMixture(names, components(source_inputs, fits, psi))
        observed = target['initialization']['paired_observations']
        for j, value in zip(target['initialization']['candidate_indices'], observed):
            model.update_raw(j, [value[0], value[1]-provider.tau])
        terminal, _ = terminal_at_prefix(provider, bank[:n0], {'paid_observations': observed}, n0,
                                         inputs['source_spec'], len(bank))
        terminal.retain_recommendation(n0)
        reward, cost, actions = model.terminal_components(terminal, remaining,
            spec['terminal_draw_power'], spec['integration_seed'])
        forecasts = {}
        for k, (name, (f, g)) in enumerate(zip(names, model.components)):
            best = int(np.lexsort((cost[k], -reward[k]))[0])
            fm, fv = f.predict(); gm, gv = g.predict()
            forecasts[name] = (np.column_stack([fm, gm]), np.column_stack([fv, gv]))
            rows.append({'task_id': target['task_id'], 'arm': name, 'action_index': best, 'action': actions[best],
                'predicted_reward': float(reward[k, best]), 'predicted_verification_calls': float(cost[k, best])})
        jobs.append((terminal, forecasts))
        print('diversified target actions', target['task_id'], [r['action']['primary'] for r in rows[-len(names):]], flush=True)
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    save(out/'recommendations.json', {'execution': spec, 'rows': rows})
    scoring = json.loads((ROOT/spec['target_scoring']).read_text())
    for target, fold, (terminal, forecasts) in zip(inputs['targets'], scoring['folds'], jobs):
        truth = np.asarray(fold['scoring_truth']); feasible = np.isin(np.arange(len(bank)), fold['feasible_policies'])
        safe, calls = safe_terminal_values(terminal, truth, feasible, remaining)
        _, _, actions = terminal.evaluate(truth[None], remaining)
        for row in (r for r in rows if r['task_id'] == target['task_id']):
            a = row['action_index']
            if actions[a] != row['action']:
                raise ValueError('frozen diversified action differs from the scored class')
            row['scores'] = {'conditional_safe_reward': float(safe[a]), 'conditional_total_calls': float(n0+calls[a]),
                             'primary_feasible': bool(feasible[row['action']['primary']])}
            mean, variance = forecasts[row['arm']]; error = mean[n0:]-truth[n0:]
            row['forecast'] = {'head_order': ['f', 'eta_f', 'g', 'eta_g'],
                'truth_mse': (error**2).mean(axis=0).tolist(),
                'moment_interval_95_truth_coverage': (abs(error) <= 1.96*np.sqrt(variance[n0:])).mean(axis=0).tolist(),
                'diagnostic_points': [{'candidate': j, 'mean': mean[j].tolist(), 'sd': np.sqrt(variance[j]).tolist(),
                    'scoring_truth': truth[j].tolist()} for j in list(dict.fromkeys([row['action']['primary'], 27, 63]))]}
    ids = [t['task_id'] for t in inputs['targets']]
    draws = np.random.default_rng(spec['bootstrap_seed']).integers(len(ids), size=(spec['bootstrap_draws'], len(ids)))
    summary = {name: {'feasible_primary_count': sum(r['scores']['primary_feasible'] for r in rows if r['arm'] == name),
        **{m: float(np.mean([r['scores'][m] for r in rows if r['arm'] == name]))
           for m in ('conditional_safe_reward', 'conditional_total_calls')},
        'truth_mse': np.mean([r['forecast']['truth_mse'] for r in rows if r['arm'] == name], axis=0).tolist()}
        for name in names}
    comparisons = []
    for reference in (names[0], names[2], names[3], names[4]):
        metrics = {}
        for metric in ('conditional_safe_reward', 'conditional_total_calls'):
            difference = np.array([next(r for r in rows if r['task_id'] == t and r['arm'] == names[1])['scores'][metric]
                -next(r for r in rows if r['task_id'] == t and r['arm'] == reference)['scores'][metric] for t in ids])
            metrics[metric] = {'task_differences': difference.tolist(), 'mean_difference': float(difference.mean()),
                'task_bootstrap_95_interval': np.quantile(difference[draws].mean(axis=1), [.025, .975]).tolist()}
        comparisons.append({'arm': names[1], 'reference': reference, 'metrics': metrics})
    result = {'execution': spec, 'rows': rows, 'summary': summary, 'paired_comparisons': comparisons,
        'cost': {'new_source_calls': 0, 'new_target_calls': 0, 'new_verification_calls': 0,
                 'historical_source_response_calls_per_arm': 288, 'new_source_hyperparameter_fits': 0},
        'cpu_seconds': time.process_time()-started}
    save(out/'results.json', result)
    print(json.dumps({'summary': summary, 'cpu_seconds': result['cpu_seconds']}), flush=True)


if __name__ == '__main__':
    run()
