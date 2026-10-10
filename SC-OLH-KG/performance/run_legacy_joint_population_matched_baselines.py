"""Prepare once, fit official source kernels separately, compare cached paths."""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.transfer_models import _kernel
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.replay_legacy_mixture_budget import terminal_at_prefix
from performance.run_legacy_joint_population_independent_pilot import target_problem
from performance.run_legacy_related_inventory_source import problem_for, save, training_view
from performance.score_legacy_related_inventory_source import safe_terminal_values
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.joint_response_mixture import JointResponseMixture
from representation.legacy_observable_prior import LegacyObservableCoordinate, legacy_exposure_descriptors


EXECUTION = ROOT/'performance/manifests/legacy_joint_population_matched_baseline_execution_v1.json'


def standardize(values):
    center, scale = values.mean(axis=0), values.std(axis=0)
    scale = np.where(scale > 0, scale, 1.)
    return (values-center)/scale, center, scale


def gp_kernel(features, lengthscale, variance):
    difference = (features[:, None]-features[None, :])/np.asarray(lengthscale)
    return variance*np.exp(-.5*np.sum(difference**2, axis=2))


def paired_fields(heads):
    n = len(heads[0][0]); components = []
    for a, b in [(0, 3), (1, 2)]:
        mean = np.column_stack([heads[a][0], heads[b][0]])
        covariance = np.zeros((n, 2, n, 2))
        covariance[:, 0, :, 0], covariance[:, 1, :, 1] = heads[a][1], heads[b][1]
        components.append(JointFunctionRiskPosterior.from_moments(mean, covariance))
    return tuple(components)


def prepare():
    execution = json.loads(EXECUTION.read_text())
    plan = json.loads((ROOT/execution['plan']).read_text())
    folder = ROOT/plan['target_cache']
    inputs = json.loads((folder/'frozen_inputs.json').read_text())
    initial = json.loads((folder/'recommendations.json').read_text())
    replay = json.loads((ROOT/inputs['execution']['legacy_replay']).read_text())
    source_spec = replay['source_fit_contract']
    data = json.loads((ROOT/plan['source_data']).read_text())
    source_plan = json.loads(((ROOT/plan['source_data']).parent/'frozen_inputs.json').read_text())['plan']
    tasks = training_view(data, execution['training_tasks'], 100)
    source_bank = [tuple(x) for x in data['bank']]
    descriptors = np.vstack([legacy_exposure_descriptors(no_outcomes(problem_for(source_plan, data['tasks'][i])),
        source_bank, source_spec['dynamics_grid_dimension']) for i in execution['training_tasks']])
    coordinate = LegacyObservableCoordinate().fit(descriptors, tasks)
    psi, center, scale = standardize(coordinate.transform(descriptors))
    source_records = [dict(name=t['name'], features=x.tolist(), response=t['response'].tolist(), noise=t['noise'].tolist())
                      for t, x in zip(tasks, np.split(psi, len(tasks)))]
    bank = [tuple(x) for x in inputs['bank']]
    target_records = []
    for config, observations in zip(inputs['tasks'], initial['initializations']):
        provider = no_outcomes(target_problem(inputs['plan'], config))
        raw = legacy_exposure_descriptors(provider, bank, source_spec['dynamics_grid_dimension'])
        public_features, _, _ = standardize(raw)
        source_features = (coordinate.transform(raw)-center)/scale
        reference = next(r for r in initial['rows'] if r['task_id'] == config['task_id'])
        target_records.append({'task_id': config['task_id'], 'config': config,
            'public_features': public_features.tolist(), 'source_features': source_features.tolist(),
            'initialization': observations, 'retained_candidate': reference['retained_candidate'],
            'tau': provider.tau, 'public_sigma_level': provider.sigma_level})
    out = ROOT/execution['output']; out.mkdir(parents=True, exist_ok=True)
    save(out/'inputs.json', {'execution': execution, 'target_plan': inputs['plan'],
        'source_spec': source_spec, 'source_feature_center': center.tolist(), 'source_feature_scale': scale.tolist(),
        'source_tasks': source_records, 'targets': target_records, 'bank': inputs['bank']})
    print('baseline inputs prepared without scoring truth', len(source_records), len(target_records), flush=True)


def summarize_pairs(rows, original, execution):
    task_ids = list(dict.fromkeys(r['task_id'] for r in rows))
    draws = np.random.default_rng(execution['bootstrap_seed']).integers(len(task_ids),
                size=(execution['bootstrap_draws'], len(task_ids)))
    pairs = []
    for reference in execution['arms']+['joint_population_full']:
        a = [next(r for r in original if r['task_id'] == t and r['arm'] == 'joint_population_structural') for t in task_ids]
        candidates = original if reference == 'joint_population_full' else rows
        b = [next(r for r in candidates if r['task_id'] == t and r['arm'] == reference) for t in task_ids]
        metrics = {}
        for name in ['conditional_safe_reward', 'conditional_total_calls']:
            diff = np.array([x['scores'][name]-y['scores'][name] for x, y in zip(a, b)])
            metrics[name] = {'task_differences': diff.tolist(), 'mean_difference': float(diff.mean()),
                'task_bootstrap_95_interval': np.quantile(diff[draws].mean(axis=1), [.025, .975]).tolist()}
        pairs.append({'arm': 'joint_population_structural', 'reference': reference, 'metrics': metrics})
    return pairs


def compare():
    execution = json.loads(EXECUTION.read_text()); out = ROOT/execution['output']
    inputs = json.loads((out/'inputs.json').read_text())
    fit = json.loads((out/'source_gp_fit.json').read_text())
    def target_only(target):
        public = np.asarray(target['public_features'])
        return [(np.full(len(public), 0. if h < 2 else np.log(target['public_sigma_level']**2)),
                 _kernel(public, public, lengthscale=1., outputscale=4., kind='rbf')) for h in range(4)]
    def source_prior(target):
        psi = np.asarray(target['source_features']); heads = []
        for d in fit['heads']:
            normalizer = d['normalization']; h = d['hyperparameters']
            mean = normalizer['mean']+normalizer['scale']*h['constant']
            covariance = normalizer['scale']**2*gp_kernel(psi, h['lengthscale'], h['signal_variance'])
            heads.append((np.full(len(psi), mean), covariance))
        return heads
    return evaluate_baselines(execution, inputs, dict(zip(execution['arms'], [target_only, source_prior])),
                              fit['cpu_seconds'], len(fit['heads']))


def evaluate_baselines(execution, inputs, prior_builders, source_cpu, fit_count):
    """The shared cached-data/terminal contract for the implemented GP priors."""
    started = time.process_time(); out = ROOT/execution['output']; out.mkdir(parents=True, exist_ok=True)
    bank = [tuple(x) for x in inputs['bank']]
    names = execution['arms']; count, remaining = 4, 358
    rows, jobs = [], []
    for target in inputs['targets']:
        provider = no_outcomes(target_problem(inputs['target_plan'], target['config']))
        model = JointResponseMixture(names, [paired_fields(prior_builders[name](target)) for name in names])
        observed = target['initialization']['paired_observations']
        for index, value in zip(target['initialization']['candidate_indices'], observed):
            model.update_raw(index, [value[0], value[1]-provider.tau])
        terminal, _ = terminal_at_prefix(provider, bank[:count], {'paid_observations': observed}, count,
                    inputs['source_spec'], len(bank))
        terminal.retain_recommendation(target['retained_candidate'])
        rewards, costs, actions = model.terminal_components(terminal, remaining,
                    execution['terminal_draw_power'], execution['integration_seed'])
        predictions = {}
        for i, (name, (f, g)) in enumerate(zip(names, model.components)):
            best = int(np.lexsort((costs[i], -rewards[i]))[0])
            fm, fv = f.predict(); gm, gv = g.predict()
            predictions[name] = {'mean': np.column_stack([fm, gm]), 'variance': np.column_stack([fv, gv])}
            rows.append({'task_id': target['task_id'], 'arm': name, 'action_index': best, 'action': actions[best],
                'predicted_reward': float(rewards[i, best]), 'predicted_verification_calls': float(costs[i, best]),
                'logical_initialization_calls': count, 'retained_candidate': target['retained_candidate']})
        jobs.append((terminal, predictions))
        print('baseline recommendations prepared', target['task_id'], [r['action']['primary'] for r in rows[-len(names):]], flush=True)
    save(out/'recommendations.json', {'rows': rows})
    # All baseline recommendations are on disk before cached truth is loaded.
    plan = json.loads((ROOT/execution['plan']).read_text())
    old = json.loads((ROOT/plan['target_cache']/'results.json').read_text())
    folds = []
    for target, (terminal, predictions), fold in zip(inputs['targets'], jobs, old['folds']):
        truth = np.asarray(fold['scoring_truth']); feasible = np.isin(np.arange(len(bank)), fold['feasible_policies'])
        safe, costs = safe_terminal_values(terminal, truth, feasible, remaining)
        _, _, actions = terminal.evaluate(truth[None], remaining)
        best = int(np.lexsort((costs, -safe))[0]); forecast = {}
        for row in [r for r in rows if r['task_id'] == target['task_id']]:
            a = row['action_index']
            if actions[a] != row['action']:
                raise ValueError('baseline frozen action and scored terminal class differ')
            row['scores'] = {'conditional_safe_reward': float(safe[a]), 'conditional_total_calls': float(count+costs[a]),
                'primary_feasible': bool(feasible[row['action']['primary']]),
                'conditional_loss_to_best_same_class': float(safe[best]-safe[a]),
                'conditional_reward_error': float(row['predicted_reward']-safe[a]),
                'conditional_call_error': float(row['predicted_verification_calls']-costs[a])}
            p = predictions[row['arm']]; error = p['mean'][count:]-truth[count:]
            forecast[row['arm']] = {'head_order': ['f', 'eta_f', 'g', 'eta_g'],
                'truth_mse': (error**2).mean(axis=0).tolist(), 'truth_bias': error.mean(axis=0).tolist(),
                'moment_interval_95_truth_coverage': (abs(error) <= 1.96*np.sqrt(p['variance'][count:])).mean(axis=0).tolist()}
        folds.append({'task_id': target['task_id'], 'predictions': forecast,
                      'best_safe_same_class_action': fold['best_safe_same_class_action']})
    summary = {}
    for name in names:
        selected = [r for r in rows if r['arm'] == name]
        summary[name] = {'feasible_primary_count': sum(r['scores']['primary_feasible'] for r in selected),
            **{m: float(np.mean([r['scores'][m] for r in selected])) for m in ['conditional_safe_reward',
                'conditional_total_calls', 'conditional_loss_to_best_same_class', 'conditional_reward_error', 'conditional_call_error']},
            'truth_mse': np.mean([f['predictions'][name]['truth_mse'] for f in folds], axis=0).tolist()}
    results = {'execution': execution, 'rows': rows, 'folds': folds, 'summary': summary,
        'paired_comparisons': summarize_pairs(rows, old['rows'], execution),
        'cost': {'new_source_calls': 0, 'new_target_calls': 0, 'new_verification_calls': 0,
                 'cached_logical_initialization_calls': count*len(rows), 'source_gp_head_fits': fit_count},
        'cpu_seconds': time.process_time()-started, 'source_fit_cpu_seconds': source_cpu}
    save(out/'results.json', results)
    print(json.dumps({'summary': summary, 'cpu_seconds': results['cpu_seconds']}), flush=True)


if __name__ == '__main__':
    {'prepare': prepare, 'compare': compare}[sys.argv[1]]()
