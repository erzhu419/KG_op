"""Frozen d50 source priors, six new d1000 tasks, independent verification."""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.transfer_archive import FrozenTransferArchive, resample_normalized_profiles
from core.designs import load_frozen_source_informed_design
from performance.benchmark_lodo_meta_prior import build_scalarized_problem
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.replay_legacy_mixture_budget import terminal_at_prefix
from performance.run_legacy_one_step_closed_loop import execute_terminal
from performance.run_legacy_related_inventory_source import problem_for, save, training_view
from performance.score_legacy_related_inventory_source import safe_terminal_values
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.joint_response_mixture import JointResponseMixture
from representation.legacy_observable_prior import LegacyObservableCoordinate, legacy_exposure_descriptors
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.source_function_prior import SourceFunctionPrior, head_batches
from representation.source_population_restore import restore_common_mean_prior


def restore_source(plan, source_spec):
    data = json.loads((ROOT/plan['source_data']).read_text())
    source_plan = json.loads(((ROOT/plan['source_data']).parent/'frozen_inputs.json').read_text())['plan']
    saved = json.loads((ROOT/plan['source_fit']).read_text())
    tasks = training_view(data, plan['training_tasks'], 100)
    if [t['name'] for t in tasks] != saved['training_task_ids']:
        raise ValueError('frozen source fit and registered training tasks differ')
    bank = [tuple(x) for x in data['bank']]
    descriptors = np.vstack([legacy_exposure_descriptors(no_outcomes(problem_for(source_plan, data['tasks'][i])),
                bank, source_spec['dynamics_grid_dimension']) for i in plan['training_tasks']])
    coordinate = LegacyObservableCoordinate().fit(descriptors, tasks)
    psi = coordinate.transform(descriptors)
    adapted = [dict(t, psi=p) for t, p in zip(tasks, np.split(psi, len(tasks)))]
    basis = OrthogonalLowOrderBasis(psi)
    arms = {}
    for name in plan['arms']:
        priors = []
        for h, d in enumerate(saved['fit_diagnostics'][name]):
            if d['selected_names'] != basis.names:
                raise ValueError('restored basis coefficient order differs from the frozen fit')
            np.testing.assert_allclose(basis.roughness, d['roughness_eigenvalues'], rtol=1e-10, atol=1e-10)
            if name == 'common_mean_structural':
                p = restore_common_mean_prior(basis, d, head_batches(adapted, h), [t['noise'][:, h] for t in adapted])
            else:
                covariance = np.asarray(d['population_mean_covariance'])+np.diag(d['residual_coefficient_diagonal'])
                p = SourceFunctionPrior(basis, np.asarray(d['coefficient_mean']), covariance,
                                        d['discrepancy_variance'], dict(d))
            priors.append(p)
        arms[name] = priors
    return coordinate, arms


def target_problem(plan, config):
    return build_scalarized_problem(plan['family'], plan['target_dimension'], 100, .04, .05,
                                    [.5, .5], problem_kwargs=config['problem_kwargs'])


def public_bank(plan, replay):
    base = json.loads((ROOT/replay['protocol']['paid_paths']).read_text())['protocol']
    inputs = ROOT/'paper_artifacts/legacy_joint_bridge_v1/inputs'/plan['family']
    archive = FrozenTransferArchive.load(inputs/'archive.json')
    initial, _ = load_frozen_source_informed_design(inputs/'design.json', heldout=plan['family'],
                    seed=base['target_seed'], n0=10, dimension=plan['target_dimension'])
    provider = no_outcomes(build_scalarized_problem(plan['family'], plan['target_dimension'],
                        base['L'], base['sigma'], base['alpha'], base['weights']))
    profiles = resample_normalized_profiles(np.vstack([t.X for t in archive.tasks]), provider.d)
    return list(dict.fromkeys(list(initial)+[tuple(map(int, provider.continuous_to_int(x))) for x in profiles]))


def paired_summary(rows, names, execution):
    rng = np.random.default_rng(execution['bootstrap_seed'])
    task_ids = list(dict.fromkeys(r['task_id'] for r in rows))
    draws = rng.integers(len(task_ids), size=(execution['bootstrap_draws'], len(task_ids)))
    metrics = ('actual_safe_reward', 'actual_total_calls', 'conditional_safe_reward', 'conditional_total_calls')
    comparisons = []
    for first, second in [(names[1], names[0]), (names[1], names[2])]:
        a = [next(r for r in rows if r['task_id'] == t and r['arm'] == first) for t in task_ids]
        b = [next(r for r in rows if r['task_id'] == t and r['arm'] == second) for t in task_ids]
        result = {'arm': first, 'reference': second, 'task_count': len(task_ids), 'metrics': {}}
        for metric in metrics:
            difference = np.array([x['scores'][metric]-y['scores'][metric] for x, y in zip(a, b)])
            result['metrics'][metric] = {'mean_difference': float(difference.mean()),
                'task_differences': difference.tolist(),
                'task_bootstrap_95_interval': np.quantile(difference[draws].mean(axis=1), [.025, .975]).tolist()}
        comparisons.append(result)
    return comparisons


def run():
    started = time.process_time()
    execution = json.loads((ROOT/'performance/manifests/legacy_joint_population_independent_pilot_execution_v1.json').read_text())
    plan = json.loads((ROOT/execution['plan']).read_text())
    out = ROOT/execution['output']; out.mkdir(parents=True, exist_ok=True)
    if (out/'recommendations.json').exists():
        raise ValueError('independent pilot already has frozen recommendations; reuse saved results rather than repeat paid streams')
    replay = json.loads((ROOT/execution['legacy_replay']).read_text())
    source_spec = replay['source_fit_contract']
    coordinate, arms = restore_source(plan, source_spec)
    bank = public_bank(plan, replay)
    if len(bank) != 73:
        raise ValueError('registered legacy bank is no longer the73-policy bank')
    rng = np.random.default_rng(plan['task_seed'])
    configs = [{'task_id': f'independent_inventory_{i:02d}',
        'problem_kwargs': {'task_geometry_shift': rng.uniform(-.035, .035, 3).tolist(),
                          'task_geometry_radius_scale': float(rng.uniform(.95, 1.05))}}
        for i in range(plan['new_task_count'])]
    save(out/'frozen_inputs.json', {'plan': plan, 'execution': execution,
         'tasks': configs, 'bank': [list(x) for x in bank], 'source_task_ids': [f'related_inventory_{i:02d}' for i in plan['training_tasks']]})
    print('source priors restored without optimization; public bank frozen', len(bank), flush=True)
    count, cap = execution['adaptation_calls'], execution['total_target_call_cap']
    remaining = cap-count; names = plan['arms']
    jobs, rows, initial_records = [], [], []
    counters = [0]*len(configs)
    for task_index, config in enumerate(configs):
        problem = target_problem(plan, config)
        simulator = problem.simulate
        def counted(point, rng=None, simulator=simulator, task_index=task_index):
            counters[task_index] += 1
            return simulator(point, rng)
        problem.simulate = counted
        provider = no_outcomes(target_problem(plan, config))
        psi = coordinate.transform(legacy_exposure_descriptors(provider, bank, source_spec['dynamics_grid_dimension']))
        components = [(JointFunctionRiskPosterior([arms[n][0]], [arms[n][3]], psi),
                       JointFunctionRiskPosterior([arms[n][1]], [arms[n][2]], psi)) for n in names]
        model = JointResponseMixture(names, components)
        observation_rng = np.random.default_rng(np.random.SeedSequence([execution['initialization_seed'], task_index]))
        observed = np.asarray([problem.simulate(p, observation_rng) for p in bank[:count]])
        for i, value in enumerate(observed):
            model.update_raw(i, [value[0], value[1]-provider.tau])
        initial_records.append({'task_id': config['task_id'], 'candidate_indices': list(range(count)),
                                'paired_observations': observed.tolist(), 'physical_calls': count})
        terminal, _ = terminal_at_prefix(provider, bank[:count], {'paid_observations': observed.tolist()}, count, source_spec, len(bank))
        reward, cost, actions = terminal.evaluate_posterior(*components[0], remaining,
                    execution['terminal_draw_power'], execution['integration_seed'])
        retained = actions[int(np.lexsort((cost, -reward))[0])]['primary']
        terminal.retain_recommendation(retained)
        rewards, costs, actions = model.terminal_components(terminal, remaining,
                    execution['terminal_draw_power'], execution['integration_seed'])
        predictions = {}
        for arm_index, name in enumerate(names):
            best = int(np.lexsort((costs[arm_index], -rewards[arm_index]))[0])
            f, g = components[arm_index]
            fm, fv = f.predict(); gm, gv = g.predict()
            predictions[name] = {'mean': np.column_stack([fm, gm]), 'variance': np.column_stack([fv, gv])}
            rows.append({'task_id': config['task_id'], 'task_index': task_index, 'arm': name,
                'arm_index': arm_index, 'retained_candidate': retained, 'action': actions[best], 'action_index': best,
                'predicted_reward': float(rewards[arm_index, best]), 'predicted_verification_calls': float(costs[arm_index, best]),
                'verification_seed': int(np.random.SeedSequence([execution['verification_seed'], task_index, arm_index]).generate_state(1)[0]),
                'logical_initialization_calls': count})
        jobs.append((problem, terminal, predictions))
        print('recommendations prepared', config['task_id'], [r['action']['primary'] for r in rows[-3:]], flush=True)
    # This disk barrier precedes every independent verifier and every scoring oracle.
    save(out/'recommendations.json', {'initializations': initial_records, 'rows': rows})
    print('all18 recommendations frozen; independent verification starts', flush=True)
    for row in rows:
        problem, terminal, _ = jobs[row['task_index']]
        deployed, verification = execute_terminal(problem, bank, row['action'], count,
                    row['verification_seed'], terminal.protocol, cap)
        row['deployed'] = deployed; row['verification'] = verification
        save(out/'verification_rows.json', {'rows': [r for r in rows if 'verification' in r]})
        print('verification complete', row['task_id'], row['arm'], verification['certified'],
              verification['total_evaluation_count'], flush=True)
    terminal_calls = sum(r['verification']['verification_budget'] for r in rows)
    if sum(counters) != count*len(configs)+terminal_calls or sum(counters) > plan['maximum_new_calls']:
        raise RuntimeError('physical simulator count and registered budget accounting differ')
    folds = []
    for task_index, (problem, terminal, predictions) in enumerate(jobs):
        truth = np.array([[problem.true_objective(p), np.log(problem.true_sigma(p)[0]**2),
                   problem.true_constraint_mean(p)-problem.tau, np.log(problem.true_sigma(p)[1]**2)] for p in bank])
        feasible = np.array([problem.is_truly_feasible(p) for p in bank], dtype=bool)
        safe, costs = safe_terminal_values(terminal, truth, feasible, remaining)
        original, _, actions = terminal.evaluate(truth[None], remaining)
        best = int(np.lexsort((costs, -safe))[0])
        fold = {'task_id': configs[task_index]['task_id'], 'feasible_policies': np.flatnonzero(feasible).tolist(),
                'scoring_truth': truth.tolist(), 'best_safe_same_class_action': {'action': actions[best],
                    'reward': float(safe[best]), 'total_calls': float(count+costs[best])}, 'predictions': {}}
        for row in [r for r in rows if r['task_index'] == task_index]:
            a, d = row['action_index'], row['deployed']
            if actions[a] != row['action']:
                raise ValueError('frozen action index differs from the scored terminal class')
            certified = row['verification']['certified']
            quality = float(np.clip(1-truth[d, 0]/terminal.protocol['objective_reference'], 0, 1))
            row['scores'] = {'actual_safe_reward': float(certified*feasible[d]*quality),
                'actual_certified_reward': float(certified*quality),
                'actual_total_calls': row['verification']['total_evaluation_count'],
                'false_certificate': bool(certified and not feasible[d]), 'deployed_feasible': bool(feasible[d]),
                'primary_feasible': bool(feasible[row['action']['primary']]),
                'conditional_safe_reward': float(safe[a]), 'conditional_total_calls': float(count+costs[a]),
                'conditional_false_certification_reward': float(original[a]-safe[a]),
                'conditional_reward_error': float(row['predicted_reward']-safe[a]),
                'conditional_call_error': float(count+row['predicted_verification_calls']-count-costs[a]),
                'conditional_loss_to_best_same_class': float(safe[best]-safe[a])}
            pred = predictions[row['arm']]
            test = np.arange(count, len(bank)); error = pred['mean'][test]-truth[test]
            fold['predictions'][row['arm']] = {'head_order': ['f', 'eta_f', 'g', 'eta_g'],
                'truth_bias': error.mean(axis=0).tolist(), 'truth_mse': (error**2).mean(axis=0).tolist(),
                'moment_interval_95_truth_coverage': (abs(error) <= 1.96*np.sqrt(pred['variance'][test])).mean(axis=0).tolist(),
                'feasible_policy_posteriors': [{'candidate': int(i), 'mean': pred['mean'][i].tolist(),
                    'sd': np.sqrt(pred['variance'][i]).tolist()} for i in np.flatnonzero(feasible)]}
        folds.append(fold)
    summary = {}
    for name in names:
        selected = [r for r in rows if r['arm'] == name]
        summary[name] = {'certified_count': sum(r['verification']['certified'] for r in selected),
            'safe_deployment_count': sum(r['verification']['certified'] and r['scores']['deployed_feasible'] for r in selected),
            'false_certificate_count': sum(r['scores']['false_certificate'] for r in selected),
            **{m: float(np.mean([r['scores'][m] for r in selected])) for m in
               ['actual_safe_reward', 'actual_total_calls', 'conditional_safe_reward', 'conditional_total_calls',
                'conditional_reward_error', 'conditional_call_error', 'conditional_loss_to_best_same_class']}}
    result = {'plan': plan, 'execution': execution, 'rows': rows, 'folds': folds, 'summary': summary,
        'paired_comparisons': paired_summary(rows, names, execution),
        'cost': {'new_source_calls': 0, 'source_optimizations': 0, 'source_encoder_reconstructions': 1,
            'shared_physical_initialization_calls': count*len(configs),
            'logical_initialization_calls': count*len(rows), 'physical_terminal_calls': terminal_calls,
            'physical_total_calls': sum(counters), 'logical_total_calls': sum(r['verification']['total_evaluation_count'] for r in rows)},
        'cpu_seconds': time.process_time()-started}
    save(out/'results.json', result)
    print(json.dumps({'summary': summary, 'cost': result['cost'], 'cpu_seconds': result['cpu_seconds']}), flush=True)


if __name__ == '__main__':
    run()
