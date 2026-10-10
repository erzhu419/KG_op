"""Stopping versus two more cached paid observations, with a fixed action class."""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.replay_legacy_mixture_budget import terminal_at_prefix
from performance.run_legacy_joint_population_independent_pilot import restore_source, target_problem
from performance.run_legacy_joint_population_matched_baselines import paired_fields
from performance.run_legacy_related_inventory_source import save
from performance.score_legacy_related_inventory_source import safe_terminal_values
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.joint_response_mixture import JointResponseMixture
from representation.source_kernel_population import predict_kernel_population


def prediction_diagnostic():
    """Replay moments only; decompose already frozen conditional decisions."""
    started = time.process_time()
    spec = json.loads((ROOT/'performance/manifests/legacy_joint_population_paid_prefix_attribution_v1.json').read_text())
    inputs = json.loads((ROOT/spec['inputs']).read_text())
    gp = json.loads((ROOT/spec['kernel_population_fit']).read_text())
    out = ROOT/spec['output']
    decisions = json.loads((out/'results.json').read_text())
    scoring = json.loads((ROOT/'paper_artifacts/legacy_joint_population_independent_pilot_v1/results.json').read_text())
    _, priors = restore_source(inputs['target_plan'], inputs['source_spec'])
    bank = [tuple(x) for x in inputs['bank']]
    forecasts, decompositions = [], []
    for target, fold in zip(inputs['targets'], scoring['folds']):
        provider = no_outcomes(target_problem(inputs['target_plan'], target['config']))
        psi = (np.asarray(target['source_features'])*np.asarray(inputs['source_feature_scale'])
               +np.asarray(inputs['source_feature_center']))
        components = [(JointFunctionRiskPosterior([priors[name][0]], [priors[name][3]], psi),
                       JointFunctionRiskPosterior([priors[name][1]], [priors[name][2]], psi))
                       for name in spec['arms'][:2]]
        components.append(paired_fields([predict_kernel_population(gp['source_features'],
                            target['source_features'], h) for h in gp['heads']]))
        model = JointResponseMixture(spec['arms'], components)
        observed = target['initialization']['paired_observations']
        truth = np.asarray(fold['scoring_truth'])
        feasible = np.isin(np.arange(len(bank)), fold['feasible_policies'])
        task_rows = [r for r in decisions['rows'] if r['task_id'] == target['task_id']]
        terminal, _ = terminal_at_prefix(provider, bank[:2], {'paid_observations': observed[:2]},
                                         2, inputs['source_spec'], len(bank))
        terminal.retain_recommendation(task_rows[0]['retained_candidate'])
        debited_rewards, debited_costs = safe_terminal_values(terminal, truth, feasible, 358)
        for count, (index, value) in enumerate(zip(target['initialization']['candidate_indices'], observed), 1):
            model.update_raw(index, [value[0], value[1]-provider.tau])
            if count not in spec['prefixes']:
                continue
            for name, (f, g) in zip(spec['arms'], model.components):
                row = next(r for r in task_rows if r['arm'] == name and r['paid_prefix'] == count)
                early = next(r for r in task_rows if r['arm'] == name and r['paid_prefix'] == 2)
                fm, fv = f.predict(); gm, gv = g.predict()
                mean, variance = np.column_stack([fm, gm]), np.column_stack([fv, gv])
                error = mean[4:]-truth[4:]
                points = list(dict.fromkeys([row['action']['primary'], early['action']['primary'], row['retained_candidate']]))
                forecasts.append({'task_id': target['task_id'], 'arm': name, 'paid_prefix': count,
                    'head_order': ['f', 'eta_f', 'g', 'eta_g'],
                    'common_unobserved_test_indices': [4, len(bank)-1],
                    'truth_mse': (error**2).mean(axis=0).tolist(),
                    'moment_interval_95_truth_coverage': (abs(error) <= 1.96*np.sqrt(variance[4:])).mean(axis=0).tolist(),
                    'selected_points': [{'candidate': i, 'posterior_mean': mean[i].tolist(),
                                         'posterior_sd': np.sqrt(variance[i]).tolist(),
                                         'scoring_truth': truth[i].tolist()} for i in points]})
                if count == 4:
                    a = early['action_index']
                    debit_reward, debit_calls = float(debited_rewards[a]), float(4+debited_costs[a])
                    decompositions.append({'task_id': target['task_id'], 'arm': name,
                        'frozen_two_point_primary': early['action']['primary'],
                        'four_point_primary': row['action']['primary'],
                        'budget_debited_same_action_reward': debit_reward,
                        'reward_change_from_budget_debit': debit_reward-early['scores']['conditional_safe_reward'],
                        'reward_change_from_selected_action': row['scores']['conditional_safe_reward']-debit_reward,
                        'calls_change_from_budget_debit': debit_calls-early['scores']['conditional_total_calls'],
                        'calls_change_from_selected_action': row['scores']['conditional_total_calls']-debit_calls})
    summaries = {}
    for name in spec['arms']:
        chosen = [d for d in decompositions if d['arm'] == name]
        summaries[name] = {'mean_decomposition': {m: float(np.mean([d[m] for d in chosen])) for m in
            ['reward_change_from_budget_debit', 'reward_change_from_selected_action',
             'calls_change_from_budget_debit', 'calls_change_from_selected_action']},
            'prefix_truth_mse': {str(count): np.mean([f['truth_mse'] for f in forecasts
                             if f['arm'] == name and f['paid_prefix'] == count], axis=0).tolist()
                             for count in spec['prefixes']}}
    result = {'scope': 'all frozen decisions on the opened cached path; not KG expected value or a new algorithm',
              'forecasts': forecasts, 'paid_path_decomposition': decompositions, 'summary': summaries,
              'cost': {'new_source_calls': 0, 'new_target_calls': 0, 'new_verification_calls': 0,
                       'source_optimizations': 0, 'source_encoder_reconstructions': 1},
              'cpu_seconds': time.process_time()-started}
    save(out/'prediction_diagnostic.json', result)
    print(json.dumps({'prediction_diagnostic': summaries, 'cpu_seconds': result['cpu_seconds']}), flush=True)


def run():
    started = time.process_time()
    spec = json.loads((ROOT/'performance/manifests/legacy_joint_population_paid_prefix_attribution_v1.json').read_text())
    inputs = json.loads((ROOT/spec['inputs']).read_text())
    gp = json.loads((ROOT/spec['kernel_population_fit']).read_text())
    _, priors = restore_source(inputs['target_plan'], inputs['source_spec'])
    bank = [tuple(x) for x in inputs['bank']]
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    rows, jobs = [], []
    for target in inputs['targets']:
        provider = no_outcomes(target_problem(inputs['target_plan'], target['config']))
        psi = (np.asarray(target['source_features'])*np.asarray(inputs['source_feature_scale'])
               +np.asarray(inputs['source_feature_center']))
        def pair(name):
            p = priors[name]
            return (JointFunctionRiskPosterior([p[0]], [p[3]], psi),
                    JointFunctionRiskPosterior([p[1]], [p[2]], psi))
        reference = pair('common_mean_structural')
        components = [pair(name) for name in spec['arms'][:2]]
        components.append(paired_fields([predict_kernel_population(gp['source_features'],
                            target['source_features'], h) for h in gp['heads']]))
        model = JointResponseMixture(spec['arms'], components)
        observed = target['initialization']['paired_observations']
        terminal, _ = terminal_at_prefix(provider, bank[:2], {'paid_observations': observed[:2]},
                                         2, inputs['source_spec'], len(bank))
        retained = None
        for count, (index, value) in enumerate(zip(target['initialization']['candidate_indices'], observed), 1):
            model.update_raw(index, [value[0], value[1]-provider.tau])
            if count <= 2:
                reference[0].update_batch(index, value[0], None, 1)
                reference[1].update_batch(index, value[1]-provider.tau, None, 1)
            if count not in spec['prefixes']:
                continue
            remaining = spec['total_target_call_cap']-count
            if count == 2:
                r, c, actions = terminal.evaluate_posterior(*reference, remaining,
                                spec['terminal_draw_power'], spec['integration_seed'])
                retained = actions[int(np.lexsort((c, -r))[0])]['primary']
                terminal.retain_recommendation(retained)
            rewards, costs, actions = model.terminal_components(terminal, remaining,
                                spec['terminal_draw_power'], spec['integration_seed'])
            for i, name in enumerate(spec['arms']):
                best = int(np.lexsort((costs[i], -rewards[i]))[0])
                rows.append({'task_id': target['task_id'], 'arm': name, 'paid_prefix': count,
                    'remaining_calls': remaining, 'retained_candidate': retained,
                    'action_index': best, 'action': actions[best],
                    'predicted_reward': float(rewards[i, best]),
                    'predicted_verification_calls': float(costs[i, best])})
            jobs.append((target['task_id'], count, terminal, remaining))
            print('paid-prefix recommendations prepared', target['task_id'], count,
                  [r['action']['primary'] for r in rows[-3:]], flush=True)
    save(out/'recommendations.json', {'execution': spec, 'rows': rows})
    # Source and recommendation calculations above never load target truth.
    old = json.loads((ROOT/'paper_artifacts/legacy_joint_population_independent_pilot_v1/results.json').read_text())
    for task_id, count, terminal, remaining in jobs:
        fold = next(f for f in old['folds'] if f['task_id'] == task_id)
        truth = np.asarray(fold['scoring_truth'])
        feasible = np.isin(np.arange(len(bank)), fold['feasible_policies'])
        safe, costs = safe_terminal_values(terminal, truth, feasible, remaining)
        _, _, actions = terminal.evaluate(truth[None], remaining)
        best = int(np.lexsort((costs, -safe))[0])
        for row in [r for r in rows if r['task_id'] == task_id and r['paid_prefix'] == count]:
            a = row['action_index']
            if row['action'] != actions[a]:
                raise ValueError('frozen paid-prefix action differs from the scored action class')
            row['scores'] = {'conditional_safe_reward': float(safe[a]),
                'conditional_total_calls': float(count+costs[a]),
                'primary_feasible': bool(feasible[row['action']['primary']]),
                'conditional_loss_to_best_same_class': float(safe[best]-safe[a])}
    ids = [t['task_id'] for t in inputs['targets']]
    draws = np.random.default_rng(spec['bootstrap_seed']).integers(len(ids),
                           size=(spec['bootstrap_draws'], len(ids)))
    summaries, comparisons = {}, []
    for name in spec['arms']:
        summaries[name] = {}
        for count in spec['prefixes']:
            selected = [r for r in rows if r['arm'] == name and r['paid_prefix'] == count]
            summaries[name][str(count)] = {'feasible_primary_count': sum(r['scores']['primary_feasible'] for r in selected),
                **{m: float(np.mean([r['scores'][m] for r in selected])) for m in
                   ['conditional_safe_reward', 'conditional_total_calls', 'conditional_loss_to_best_same_class']}}
        metrics = {}
        for m in ['conditional_safe_reward', 'conditional_total_calls']:
            differences = []
            for task_id in ids:
                selected = [r for r in rows if r['task_id'] == task_id and r['arm'] == name]
                early = next(r for r in selected if r['paid_prefix'] == 2)
                late = next(r for r in selected if r['paid_prefix'] == 4)
                differences.append(late['scores'][m]-early['scores'][m])
            difference = np.asarray(differences)
            metrics[m] = {'task_differences': differences, 'mean_difference': float(difference.mean()),
                'task_bootstrap_95_interval': np.quantile(difference[draws].mean(axis=1), [.025, .975]).tolist()}
        comparisons.append({'arm': name, 'comparison': 'four paid observations minus two', 'metrics': metrics})
    result = {'execution': spec, 'rows': rows, 'summary': summaries, 'paired_comparisons': comparisons,
              'cost': {'new_source_calls': 0, 'new_target_calls': 0, 'new_verification_calls': 0,
                       'source_optimizations': 0, 'cached_logical_initialization_calls': sum(r['paid_prefix'] for r in rows)},
              'cpu_seconds': time.process_time()-started}
    save(out/'results.json', result)
    print(json.dumps({'summary': summaries, 'cpu_seconds': result['cpu_seconds']}), flush=True)


if __name__ == '__main__':
    {'run': run, 'diagnose': prediction_diagnostic}[sys.argv[1] if len(sys.argv) > 1 else 'run']()
