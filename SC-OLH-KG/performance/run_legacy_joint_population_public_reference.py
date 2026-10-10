"""Keep the orthogonal measure on the public catalog, with only16 source labels."""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.replay_legacy_mixture_budget import terminal_at_prefix
from performance.run_legacy_joint_population_independent_pilot import target_problem
from performance.run_legacy_joint_population_source_budget import array_tasks, restore_joint
from performance.run_legacy_related_inventory_source import save
from performance.score_legacy_related_inventory_source import safe_terminal_values
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.joint_response_mixture import JointResponseMixture
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis, fit_orthogonal_function_prior
from representation.source_function_prior import head_batches
from representation.source_joint_population import fit_joint_source_population, same_source_full_residual_prior


def run():
    spec = json.loads((ROOT/'performance/manifests/legacy_joint_population_public_reference_attribution_v1.json').read_text())
    inputs = json.loads((ROOT/spec['inputs']).read_text())
    geometry = json.loads((ROOT/spec['public_geometry']).read_text())
    basis = OrthogonalLowOrderBasis(geometry['public_reference_psi'])
    tasks = array_tasks(inputs['source_tasks'])
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    source_path = out/'source_fit.json'
    started = time.process_time()
    fit = json.loads(source_path.read_text()) if source_path.exists() else {'regularizers': [], 'heads': [], 'cpu_seconds': 0.}
    for h in range(len(fit['heads']), 4):
        batches, noises = head_batches(tasks, h), [t['noise'][:, h] for t in tasks]
        regularizer = fit_orthogonal_function_prior(batches, noises, basis, max_iterations=spec['source_max_iterations'])
        prior = fit_joint_source_population(regularizer, batches, noises, max_iterations=spec['source_max_iterations'])
        fit['regularizers'].append({'diagnostics': regularizer.diagnostics,
                                    'coefficient_diagonal': np.diag(regularizer.covariance).tolist()})
        fit['heads'].append(prior.diagnostics)
        save(source_path, {**fit, 'cpu_seconds': fit['cpu_seconds']+time.process_time()-started})
        print('public-reference source head', h, 'regularizer/joint convergence',
              regularizer.diagnostics['optimizer_converged'], prior.diagnostics['optimizer_converged'], flush=True)
    fit['cpu_seconds'] += time.process_time()-started
    save(source_path, fit)
    if not all(r['diagnostics']['optimizer_converged'] for r in fit['regularizers']) or not all(d['optimizer_converged'] for d in fit['heads']):
        raise RuntimeError('public-reference source fit needs same-objective completion before target comparison')
    started = time.process_time()
    structural = [restore_joint(basis, d) for d in fit['heads']]
    priors = {'joint_population_structural': structural,
              'joint_population_full': [same_source_full_residual_prior(p) for p in structural]}
    bank = [tuple(x) for x in inputs['bank']]
    names = spec['arms']; remaining = spec['total_target_call_cap']-spec['target_prefix']
    rows, jobs = [], []
    for target in inputs['targets']:
        provider = no_outcomes(target_problem(inputs['target_plan'], target['config']))
        psi = np.asarray(target['psi'])
        components = [(JointFunctionRiskPosterior([priors[name][0]], [priors[name][3]], psi),
                       JointFunctionRiskPosterior([priors[name][1]], [priors[name][2]], psi)) for name in names]
        model = JointResponseMixture(names, components)
        observed = target['initialization']['paired_observations']
        for i, v in zip(target['initialization']['candidate_indices'], observed):
            model.update_raw(i, [v[0], v[1]-provider.tau])
        terminal, _ = terminal_at_prefix(provider, bank[:2], {'paid_observations': observed}, 2,
                                         inputs['source_spec'], len(bank))
        terminal.retain_recommendation(2)
        rewards, costs, actions = model.terminal_components(terminal, remaining,
                                        spec['terminal_draw_power'], spec['integration_seed'])
        forecasts = {}
        for k, (name, (f, g)) in enumerate(zip(names, components)):
            best = int(np.lexsort((costs[k], -rewards[k]))[0])
            fm, fv = f.predict(); gm, gv = g.predict()
            forecasts[name] = (np.column_stack([fm, gm]), np.column_stack([fv, gv]))
            rows.append({'task_id': target['task_id'], 'arm': name, 'action_index': best,
                'action': actions[best], 'predicted_reward': float(rewards[k, best]),
                'predicted_verification_calls': float(costs[k, best])})
        jobs.append((terminal, forecasts))
        print('public-reference recommendations prepared', target['task_id'],
              [r['action']['primary'] for r in rows[-2:]], flush=True)
    save(out/'recommendations.json', {'execution': spec, 'rows': rows})
    previous = json.loads((ROOT/spec['previous_results']).read_text())
    scoring = json.loads((ROOT/'paper_artifacts/legacy_joint_population_independent_pilot_v1/results.json').read_text())
    for target, fold, (terminal, forecasts) in zip(inputs['targets'], scoring['folds'], jobs):
        truth = np.asarray(fold['scoring_truth'])
        feasible = np.isin(np.arange(len(bank)), fold['feasible_policies'])
        safe, costs = safe_terminal_values(terminal, truth, feasible, remaining)
        _, _, actions = terminal.evaluate(truth[None], remaining)
        for row in [r for r in rows if r['task_id'] == target['task_id']]:
            a = row['action_index']
            if actions[a] != row['action']:
                raise ValueError('frozen public-reference action differs from the scored class')
            row['scores'] = {'conditional_safe_reward': float(safe[a]), 'conditional_total_calls': float(2+costs[a]),
                             'primary_feasible': bool(feasible[row['action']['primary']])}
            mean, variance = forecasts[row['arm']]; error = mean[2:]-truth[2:]
            row['forecast'] = {'head_order': ['f', 'eta_f', 'g', 'eta_g'], 'truth_mse': (error**2).mean(axis=0).tolist(),
                'moment_interval_95_truth_coverage': (abs(error) <= 1.96*np.sqrt(variance[2:])).mean(axis=0).tolist()}
    ids = [t['task_id'] for t in inputs['targets']]
    draws = np.random.default_rng(spec['bootstrap_seed']).integers(len(ids), size=(spec['bootstrap_draws'], len(ids)))
    summary, comparisons = {}, []
    for name in names:
        selected = [r for r in rows if r['arm'] == name]
        summary[name] = {'feasible_primary_count': sum(r['scores']['primary_feasible'] for r in selected),
            **{m: float(np.mean([r['scores'][m] for r in selected])) for m in ['conditional_safe_reward', 'conditional_total_calls']},
            'truth_mse': np.mean([r['forecast']['truth_mse'] for r in selected], axis=0).tolist()}
        for size, reference in [(16, name), (16, 'joint_population_rbf_shared_raw'), (73, name)]:
            metrics = {}
            for metric in ['conditional_safe_reward', 'conditional_total_calls']:
                differences = np.asarray([r['scores'][metric]-next(old for old in previous['rows']
                    if old['task_id'] == r['task_id'] and old['source_site_count'] == size and old['arm'] == reference)['scores'][metric]
                    for r in selected])
                metrics[metric] = {'task_differences': differences.tolist(), 'mean_difference': float(differences.mean()),
                    'task_bootstrap_95_interval': np.quantile(differences[draws].mean(axis=1), [.025, .975]).tolist()}
            comparisons.append({'arm': name, 'reference': [size, reference], 'metrics': metrics})
    metrics = {}
    for metric in ['conditional_safe_reward', 'conditional_total_calls']:
        differences = np.asarray([next(r for r in rows if r['task_id'] == task and r['arm'] == names[0])['scores'][metric]
                                  -next(r for r in rows if r['task_id'] == task and r['arm'] == names[1])['scores'][metric]
                                  for task in ids])
        metrics[metric] = {'task_differences': differences.tolist(), 'mean_difference': float(differences.mean()),
                          'task_bootstrap_95_interval': np.quantile(differences[draws].mean(axis=1), [.025, .975]).tolist()}
    comparisons.append({'arm': names[0], 'reference': ['public_reference16', names[1]], 'metrics': metrics})
    result = {'execution': spec, 'rows': rows, 'summary': summary, 'paired_comparisons': comparisons,
        'cost': {'new_source_calls': 0, 'new_target_calls': 0, 'new_verification_calls': 0,
                 'source_response_archive_calls': 288, 'new_native_source_head_fits': 8,
                 'cached_logical_initialization_calls': 2*len(rows)},
        'source_fit_cpu_seconds': fit['cpu_seconds'], 'cpu_seconds': time.process_time()-started}
    save(out/'results.json', result)
    print(json.dumps({'summary': summary, 'cpu_seconds': result['cpu_seconds']}), flush=True)


if __name__ == '__main__':
    run()
