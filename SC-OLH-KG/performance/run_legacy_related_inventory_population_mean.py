"""Matched common-mean attribution and six source-only task holdouts."""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.replay_legacy_mixture_budget import terminal_at_prefix
from performance.run_legacy_related_inventory_source import (
    head_moments, prediction_metrics, problem_for, save, training_view)
from performance.score_legacy_related_inventory_source import add_safe_scores
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.joint_response_mixture import JointResponseMixture
from representation.legacy_observable_prior import LegacyObservableCoordinate, legacy_exposure_descriptors
from representation.orthogonal_function_prior import (
    OrthogonalLowOrderBasis, fit_orthogonal_function_prior, same_source_orthogonal_full_prior)
from representation.source_function_prior import head_batches
from representation.source_population_mean import fit_source_population_mean


def fit_arms(tasks, descriptors, source_spec):
    coordinate = LegacyObservableCoordinate().fit(descriptors, tasks)
    transformed = coordinate.transform(descriptors)
    adapted, offset = [], 0
    for task in tasks:
        n = len(task['response'])
        adapted.append(dict(task, psi=transformed[offset:offset+n])); offset += n
    basis = OrthogonalLowOrderBasis(transformed)
    original, common = [], []
    for h in range(4):
        batches, noise = head_batches(adapted, h), [t['noise'][:, h] for t in adapted]
        prior = fit_orthogonal_function_prior(batches, noise, basis,
                    max_iterations=source_spec['source_max_iterations'])
        original.append(prior)
        common.append(fit_source_population_mean(prior, batches, noise))
    return coordinate, {'zero_mean_structural': original,
        'zero_mean_full': [same_source_orthogonal_full_prior(p) for p in original],
        'common_mean_structural': common,
        'common_mean_full': [same_source_orthogonal_full_prior(p) for p in common]}


def source_preflight(data, plan, execution, source_plan, source_spec, descriptors, out, fit_priors=fit_arms):
    started = time.process_time(); records = []
    for heldout in plan['training_tasks']:
        train = [i for i in plan['training_tasks'] if i != heldout]
        tasks = training_view(data, train, source_plan['task_distribution']['L'])
        coordinate, arms = fit_priors(tasks, np.vstack([descriptors[i] for i in train]), source_spec)
        test = training_view(data, [heldout], source_plan['task_distribution']['L'])[0]
        psi = coordinate.transform(descriptors[heldout])
        diagnostics = {a: [p.diagnostics for p in ps] for a, ps in arms.items()}
        row = {'heldout_task_id': test['name'], 'training_task_ids': [t['name'] for t in tasks],
               'coordinate': coordinate.diagnostics(), 'fit_diagnostics': diagnostics, 'arms': {}}
        # The withheld observations are scoring labels, never used by the fit.
        for name in execution['arms']:
            heads = []
            for h, prior in enumerate(arms[name]):
                design = np.column_stack([np.ones(len(psi)), prior.basis.transform(psi)])
                mean = design @ prior.mean
                latent_variance = np.sum((design @ prior.covariance)*design, axis=1) + prior.discrepancy_variance
                observation_variance = latent_variance + test['noise'][:, h]
                error = mean-test['response'][:, h]
                heads.append({'head': ['f', 'g', 'eta_g', 'eta_f'][h],
                    'bias': float(error.mean()), 'observed_label_mse': float(np.mean(error**2)),
                    'mean_observation_variance': float(observation_variance.mean()),
                    'gaussian_label_log_score': float(np.mean(-.5*(np.log(2*np.pi*observation_variance)
                                                                 + error**2/observation_variance))),
                    'observation_interval_95_coverage': float(np.mean(abs(error) <= 1.96*np.sqrt(observation_variance))),
                    'noise_subtracted_residual_mse': float(np.mean(error**2-test['noise'][:, h])),
                    'discrepancy_variance': prior.discrepancy_variance,
                    'mean_observed_label_noise': float(test['noise'][:, h].mean())})
            row['arms'][name] = heads
        records.append(row)
        print('source-only preflight complete', test['name'], 'g MSE',
              {a: row['arms'][a][1]['observed_label_mse'] for a in execution['arms']}, flush=True)
    result = {'records': records, 'cpu_seconds': time.process_time()-started,
              'source_task_count': len(plan['training_tasks']), 'new_simulator_calls': 0}
    save(out/'source_preflight.json', result)
    return result


def run(execution_path='performance/manifests/legacy_related_inventory_population_mean_execution_v1.json',
        fit_priors=fit_arms):
    started = time.process_time()
    execution = json.loads((ROOT/execution_path).read_text())
    plan = json.loads((ROOT/execution['plan']).read_text())
    data = json.loads((ROOT/plan['data']).read_text())
    reference_folder = (ROOT/plan['data']).parent
    frozen = json.loads((reference_folder/'frozen_inputs.json').read_text())
    source_plan = frozen['plan']
    source_spec = json.loads((ROOT/frozen['execution']['legacy_replay']).read_text())['source_fit_contract']
    # This artifact contains stage17 pre-scoring decisions, not scoring truth.
    old_choices = json.loads((reference_folder/'recommendations.json').read_text())
    names = execution['arms']
    out = ROOT/execution['output']; out.mkdir(parents=True, exist_ok=True)
    bank = [tuple(x) for x in data['bank']]
    descriptors = {i: legacy_exposure_descriptors(no_outcomes(problem_for(source_plan, data['tasks'][i])),
        bank, source_spec['dynamics_grid_dimension']) for i in plan['training_tasks']}
    preflight = source_preflight(data, plan, execution, source_plan, source_spec, descriptors, out, fit_priors)
    tasks = training_view(data, plan['training_tasks'], source_plan['task_distribution']['L'])
    coordinate, arms = fit_priors(tasks, np.vstack([descriptors[i] for i in plan['training_tasks']]), source_spec)
    source = {'training_task_ids': [t['name'] for t in tasks], 'coordinate': coordinate.diagnostics(),
              'fit_diagnostics': {a: [p.diagnostics for p in ps] for a, ps in arms.items()}}
    save(out/'source_fit.json', source)
    print('six-task source priors frozen', flush=True)
    folds, rows, jobs = [], [], []
    count, cap = execution['adaptation_calls'], execution['total_target_call_cap']
    for i in plan['opened_development_holdouts']:
        task = data['tasks'][i]; provider = no_outcomes(problem_for(source_plan, task))
        desc = legacy_exposure_descriptors(provider, bank, source_spec['dynamics_grid_dimension'])
        psi = coordinate.transform(desc)
        components = [(JointFunctionRiskPosterior([arms[name][0]], [arms[name][3]], psi),
                       JointFunctionRiskPosterior([arms[name][1]], [arms[name][2]], psi)) for name in names]
        model = JointResponseMixture(names, components)
        observed = np.asarray([value[0] for value in task['Y_replicates'][:count]])
        for index, value in enumerate(observed):
            model.update_raw(index, [value[0], value[1]-provider.tau])
        terminal, _ = terminal_at_prefix(provider, bank[:count], {'paid_observations': observed.tolist()},
                                         count, source_spec, len(bank))
        old = next(f for f in old_choices['folds'] if f['task_id'] == task['task_id'])
        retained = old['retained_policy']; terminal.retain_recommendation(retained)
        reward, cost, actions = model.terminal_components(terminal, cap-count,
            execution['terminal_draw_power'], execution['integration_seed'])
        fold = {'task_id': task['task_id'], 'retained_policy': retained, 'action_count': len(actions),
                'paid_history': model.history, 'prediction': {name: {'mean': m.tolist(), 'variance': v.tolist()}
                    for name, (m, v) in head_moments(model).items() if name in names}}
        folds.append(fold); current = []
        for k, name in enumerate(names):
            best = int(np.lexsort((cost[k], -reward[k]))[0])
            row = {'task_id': task['task_id'], 'selector': name, 'search_calls': count,
                'total_target_call_cap': cap, 'action_index': best, 'action': actions[best],
                'predicted_reward': float(reward[k, best]), 'predicted_total_calls': float(cost[k, best]+count)}
            rows.append(row); current.append(row)
        jobs.append((task, model, terminal, fold, current))
        print('development decisions frozen', task['task_id'], [r['action']['primary'] for r in current], flush=True)
    save(out/'recommendations.json', {'execution': execution, 'source': source, 'folds': folds, 'rows': rows})
    reference = json.loads((ROOT/plan['reference']).read_text())
    result = {'plan': plan, 'execution': execution, 'source': source, 'folds': folds, 'rows': rows}
    for task, model, terminal, fold, current in jobs:
        old = next(f for f in reference['folds'] if f['task_id'] == task['task_id'])
        fold['scoring_truth'], fold['feasible_policies'] = old['scoring_truth'], old['feasible_policies']
        truth = np.asarray(fold['scoring_truth'])
        reward, cost, actions = terminal.evaluate(truth[None], cap-count)
        best = int(np.lexsort((cost, -reward))[0])
        fold['best_same_class_action'] = {'action': actions[best], 'reward': float(reward[best]),
                                          'total_calls': float(cost[best]+count)}
        fold['calibration'] = {name: value for name, value in prediction_metrics(model, task, bank,
            source_plan['task_distribution']['L'], truth, count).items() if name in names}
        for row in current:
            a = row['action_index']
            row['truth'] = {'reward': float(reward[a]), 'total_calls': float(cost[a]+count),
                'primary_feasible': row['action']['primary'] in fold['feasible_policies'],
                'reward_estimation_error': float(row['predicted_reward']-reward[a]),
                'call_estimation_error': float(row['predicted_total_calls']-cost[a]-count),
                'loss_to_best_same_class_action': float(reward[best]-reward[a])}
        add_safe_scores(result, terminal, fold, current)
        print('development scoring complete', task['task_id'], flush=True)
    result['source_preflight_cpu_seconds'] = preflight['cpu_seconds']
    result['cpu_seconds'] = time.process_time()-started
    result['cost'] = {'new_source_calls': 0, 'new_target_calls': 0, 'new_terminal_calls': 0,
                      'reused_training_source_calls': 1314, 'reused_adaptation_calls': len(jobs)*count,
                      'native_source_head_fits': execution['source_head_fits_per_fold']*(len(plan['training_tasks'])+1),
                      'new_independent_certificates': 0}
    save(out/'results.json', result)
    print(json.dumps({'decisions': len(rows), 'cpu_seconds': result['cpu_seconds'], 'cost': result['cost']}), flush=True)
    return result


if __name__ == '__main__':
    run()
