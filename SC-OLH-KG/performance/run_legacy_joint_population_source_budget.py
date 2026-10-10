"""Source-label budget attribution, including supervised representation refits."""

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
from performance.run_legacy_joint_population_matched_baselines import paired_fields, standardize
from performance.run_legacy_related_inventory_source import problem_for, save, training_view
from performance.score_legacy_related_inventory_source import safe_terminal_values
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.joint_response_mixture import JointResponseMixture
from representation.legacy_observable_prior import LegacyObservableCoordinate, legacy_exposure_descriptors
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis, fit_orthogonal_function_prior
from representation.source_function_prior import SourceFunctionPrior, head_batches
from representation.source_joint_population import fit_joint_source_population, same_source_full_residual_prior
from representation.source_kernel_population import fit_kernel_population, predict_kernel_population
from representation.source_known_noise_gp import fit_known_noise_gp


EXECUTION = ROOT/'performance/manifests/legacy_joint_population_source_budget_execution_v1.json'


def source_subset(data, indices, count):
    """Select raw records before any response statistic or supervised fit."""
    return {'bank': data['bank'][:count], 'tasks': [
        {'task_id': data['tasks'][i]['task_id'], 'tau': data['tasks'][i]['tau'],
         'Y_replicates': data['tasks'][i]['Y_replicates'][:count]} for i in indices]}


def encode_subset(data, indices, count, descriptors, L):
    selected = source_subset(data, indices, count)
    tasks = training_view(selected, range(len(indices)), L)
    desc = np.vstack([descriptors[i][:count] for i in indices])
    coordinate = LegacyObservableCoordinate().fit(desc, tasks)
    psi = coordinate.transform(desc)
    features, center, scale = standardize(psi)
    records = []
    for task, p, x in zip(tasks, np.split(psi, len(tasks)), np.split(features, len(tasks))):
        records.append({'name': task['name'], 'psi': p, 'features': x,
                        'response': task['response'], 'noise': task['noise']})
    return coordinate, records, center, scale


def array_tasks(records):
    return [{**t, **{k: np.asarray(t[k]) for k in ('psi', 'features', 'response', 'noise')}} for t in records]


def restore_joint(basis, diagnostic):
    covariance = np.asarray(diagnostic['population_mean_covariance'])
    covariance = covariance+np.diag(diagnostic['residual_coefficient_diagonal'])
    return SourceFunctionPrior(basis, np.asarray(diagnostic['coefficient_mean']), covariance,
                               diagnostic['discrepancy_variance'], dict(diagnostic))


def prepare():
    started = time.process_time()
    execution = json.loads(EXECUTION.read_text())
    data = json.loads((ROOT/execution['source_data']).read_text())
    source_plan = json.loads(((ROOT/execution['source_data']).parent/'frozen_inputs.json').read_text())['plan']
    old = json.loads((ROOT/execution['target_inputs']).read_text())
    count = execution['source_site_count']; ids = execution['training_tasks']
    bank = [tuple(p) for p in data['bank'][:count]]
    descriptors = {i: legacy_exposure_descriptors(no_outcomes(problem_for(source_plan, data['tasks'][i])),
                    bank, old['source_spec']['dynamics_grid_dimension']) for i in ids}
    coordinate, tasks, center, scale = encode_subset(data, ids, count, descriptors,
                                                source_plan['task_distribution']['L'])
    targets = []
    target_bank = [tuple(p) for p in old['bank']]
    for target in old['targets']:
        provider = no_outcomes(target_problem(old['target_plan'], target['config']))
        raw = legacy_exposure_descriptors(provider, target_bank, old['source_spec']['dynamics_grid_dimension'])
        psi = coordinate.transform(raw)
        initial = target['initialization']; n0 = execution['target_prefix']
        targets.append({'task_id': target['task_id'], 'config': target['config'],
            'tau': target['tau'], 'psi': psi.tolist(), 'source_features': ((psi-center)/scale).tolist(),
            'initialization': {'candidate_indices': initial['candidate_indices'][:n0],
                              'paired_observations': initial['paired_observations'][:n0]}})
    out = ROOT/execution['output']; out.mkdir(parents=True, exist_ok=True)
    save(out/'inputs.json', {'execution': execution, 'source_tasks': [
        {**t, **{k: t[k].tolist() for k in ('psi', 'features', 'response', 'noise')}} for t in tasks],
        'source_site_indices': list(range(count)), 'coordinate': coordinate.diagnostics(),
        'source_feature_center': center.tolist(), 'source_feature_scale': scale.tolist(),
        'targets': targets, 'target_plan': old['target_plan'], 'source_spec': old['source_spec'],
        'bank': old['bank'], 'cpu_seconds': time.process_time()-started})
    print('subset supervised representation prepared', len(tasks)*count, 'records',
          count*len(tasks)*3, 'historical episodes', flush=True)


def fit():
    execution = json.loads(EXECUTION.read_text()); out = ROOT/execution['output']
    inputs = json.loads((out/'inputs.json').read_text())
    tasks = array_tasks(inputs['source_tasks'])
    basis = OrthogonalLowOrderBasis(np.vstack([t['psi'] for t in tasks]))
    fits = {}
    for name in ('gp_anchor', 'regularizer', 'joint_structural', 'gp_population'):
        path = out/(name+'_fit.json')
        recorded = json.loads(path.read_text()) if path.exists() else {'heads': [], 'cpu_seconds': 0.}
        started = time.process_time()
        for h in range(len(recorded['heads']), 4):
            if name == 'gp_anchor':
                record = fit_known_noise_gp(tasks, h, execution['generic_anchor_max_iterations'])
                diagnostic = record
            elif name == 'regularizer':
                p = fit_orthogonal_function_prior(head_batches(tasks, h), [t['noise'][:, h] for t in tasks],
                                        basis, max_iterations=execution['structural_max_iterations'])
                record = {'diagnostics': p.diagnostics, 'coefficient_diagonal': np.diag(p.covariance).tolist()}
                diagnostic = p.diagnostics
            elif name == 'joint_structural':
                r = fits['regularizer']['heads'][h]; d = r['diagnostics']
                regularizer = SourceFunctionPrior(basis, np.asarray(d['coefficient_mean']),
                    np.diag(r['coefficient_diagonal']), d['discrepancy_variance'], dict(d))
                p = fit_joint_source_population(regularizer, head_batches(tasks, h),
                    [t['noise'][:, h] for t in tasks], max_iterations=execution['structural_max_iterations'])
                record = diagnostic = p.diagnostics
            else:
                record = diagnostic = fit_kernel_population(tasks, h, fits['gp_anchor']['heads'][h],
                                            optimizer=execution['generic_population_optimizer'])
            recorded['heads'].append(record)
            save(path, {**recorded, 'cpu_seconds': recorded['cpu_seconds']+time.process_time()-started})
            print(name, 'source head', h, 'converged', diagnostic['optimizer_converged'],
                  'iterations', diagnostic['optimizer_iterations'], flush=True)
        recorded['cpu_seconds'] += time.process_time()-started
        save(path, recorded)
        fits[name] = recorded
        diagnostics = [r['diagnostics'] if name == 'regularizer' else r for r in recorded['heads']]
        if not all(d['optimizer_converged'] for d in diagnostics):
            raise RuntimeError(name+' has recorded source convergence failures; complete the same source objective before target comparison')


def compare():
    started = time.process_time()
    execution = json.loads(EXECUTION.read_text()); out = ROOT/execution['output']
    inputs = json.loads((out/'inputs.json').read_text())
    fits = {name: json.loads((out/(name+'_fit.json')).read_text())
            for name in ('gp_anchor', 'regularizer', 'joint_structural', 'gp_population')}
    for name, fit in fits.items():
        diagnostics = [r['diagnostics'] if name == 'regularizer' else r for r in fit['heads']]
        if len(diagnostics) != 4 or not all(d['optimizer_converged'] for d in diagnostics):
            raise RuntimeError('target comparison needs four completed '+name+' heads')
    basis = OrthogonalLowOrderBasis(np.vstack([t['psi'] for t in inputs['source_tasks']]))
    structural = [restore_joint(basis, d) for d in fits['joint_structural']['heads']]
    small = {'joint_population_structural': structural,
             'joint_population_full': [same_source_full_residual_prior(p) for p in structural]}
    full_coordinate, full = restore_source(inputs['target_plan'], inputs['source_spec'])
    full_inputs = json.loads((ROOT/execution['target_inputs']).read_text())
    full_gp = json.loads((ROOT/execution['full_kernel_population_fit']).read_text())
    bank = [tuple(x) for x in inputs['bank']]
    names = execution['arms']; n0 = execution['target_prefix']; remaining = execution['total_target_call_cap']-n0
    rows, jobs = [], []
    for target, full_target in zip(inputs['targets'], full_inputs['targets']):
        provider = no_outcomes(target_problem(inputs['target_plan'], target['config']))
        raw = legacy_exposure_descriptors(provider, bank, inputs['source_spec']['dynamics_grid_dimension'])
        full_psi = full_coordinate.transform(raw)
        terminal, _ = terminal_at_prefix(provider, bank[:n0],
            {'paid_observations': target['initialization']['paired_observations']}, n0, inputs['source_spec'], len(bank))
        terminal.retain_recommendation(n0)
        predictions = {}
        for size, priors, psi, source_x, target_x, gp_heads in [
            (execution['source_site_count'], small, np.asarray(target['psi']),
             inputs['source_tasks'][0]['features'], target['source_features'], fits['gp_population']['heads']),
            (execution['full_source_site_count'], full, full_psi,
             full_gp['source_features'], full_target['source_features'], full_gp['heads'])]:
            components = [(JointFunctionRiskPosterior([priors[name][0]], [priors[name][3]], psi),
                           JointFunctionRiskPosterior([priors[name][1]], [priors[name][2]], psi)) for name in names[:2]]
            components.append(paired_fields([predict_kernel_population(source_x, target_x, h) for h in gp_heads]))
            model = JointResponseMixture(names, components)
            for index, value in zip(target['initialization']['candidate_indices'], target['initialization']['paired_observations']):
                model.update_raw(index, [value[0], value[1]-provider.tau])
            rewards, costs, actions = model.terminal_components(terminal, remaining,
                                        execution['terminal_draw_power'], execution['integration_seed'])
            for i, (name, (f, g)) in enumerate(zip(names, model.components)):
                best = int(np.lexsort((costs[i], -rewards[i]))[0])
                fm, fv = f.predict(); gm, gv = g.predict()
                predictions[(size, name)] = (np.column_stack([fm, gm]), np.column_stack([fv, gv]))
                rows.append({'task_id': target['task_id'], 'source_site_count': size, 'arm': name,
                    'action_index': best, 'action': actions[best], 'public_retained_candidate': n0,
                    'predicted_reward': float(rewards[i, best]), 'predicted_verification_calls': float(costs[i, best])})
            print('source-budget recommendations prepared', target['task_id'], size,
                  [r['action']['primary'] for r in rows[-3:]], flush=True)
        jobs.append((terminal, predictions))
    save(out/'recommendations.json', {'execution': execution, 'rows': rows})
    scoring = json.loads((ROOT/'paper_artifacts/legacy_joint_population_independent_pilot_v1/results.json').read_text())
    for target, fold, (terminal, predictions) in zip(inputs['targets'], scoring['folds'], jobs):
        truth = np.asarray(fold['scoring_truth'])
        feasible = np.isin(np.arange(len(bank)), fold['feasible_policies'])
        safe, costs = safe_terminal_values(terminal, truth, feasible, remaining)
        _, _, actions = terminal.evaluate(truth[None], remaining)
        best = int(np.lexsort((costs, -safe))[0])
        for row in [r for r in rows if r['task_id'] == target['task_id']]:
            a = row['action_index']
            if actions[a] != row['action']:
                raise ValueError('frozen source-budget action differs from the scored class')
            row['scores'] = {'conditional_safe_reward': float(safe[a]), 'conditional_total_calls': float(n0+costs[a]),
                'primary_feasible': bool(feasible[row['action']['primary']]),
                'conditional_loss_to_best_same_class': float(safe[best]-safe[a])}
            mean, variance = predictions[(row['source_site_count'], row['arm'])]
            error = mean[n0:]-truth[n0:]
            points = list(dict.fromkeys([row['action']['primary'], n0]))
            row['forecast'] = {'head_order': ['f', 'eta_f', 'g', 'eta_g'],
                'truth_mse': (error**2).mean(axis=0).tolist(),
                'moment_interval_95_truth_coverage': (abs(error) <= 1.96*np.sqrt(variance[n0:])).mean(axis=0).tolist(),
                'selected_points': [{'candidate': j, 'mean': mean[j].tolist(), 'sd': np.sqrt(variance[j]).tolist(),
                                     'scoring_truth': truth[j].tolist()} for j in points]}
    ids = [t['task_id'] for t in inputs['targets']]
    draws = np.random.default_rng(execution['bootstrap_seed']).integers(len(ids),
                size=(execution['bootstrap_draws'], len(ids)))
    sizes = [execution['source_site_count'], execution['full_source_site_count']]
    summary, comparisons = {}, []
    for size in sizes:
        summary[str(size)] = {}
        for name in names:
            selected = [r for r in rows if r['source_site_count'] == size and r['arm'] == name]
            summary[str(size)][name] = {'feasible_primary_count': sum(r['scores']['primary_feasible'] for r in selected),
                **{m: float(np.mean([r['scores'][m] for r in selected])) for m in
                    ['conditional_safe_reward', 'conditional_total_calls', 'conditional_loss_to_best_same_class']},
                'truth_mse': np.mean([r['forecast']['truth_mse'] for r in selected], axis=0).tolist()}
    contrasts = [((size, names[0]), (size, ref)) for size in sizes for ref in names[1:]]
    contrasts += [((sizes[0], name), (sizes[1], name)) for name in names]
    for (sa, na), (sb, nb) in contrasts:
        metrics = {}
        for metric in ['conditional_safe_reward', 'conditional_total_calls']:
            difference = []
            for task_id in ids:
                a = next(r for r in rows if r['task_id'] == task_id and r['source_site_count'] == sa and r['arm'] == na)
                b = next(r for r in rows if r['task_id'] == task_id and r['source_site_count'] == sb and r['arm'] == nb)
                difference.append(a['scores'][metric]-b['scores'][metric])
            difference = np.asarray(difference)
            metrics[metric] = {'mean_difference': float(difference.mean()), 'task_differences': difference.tolist(),
                'task_bootstrap_95_interval': np.quantile(difference[draws].mean(axis=1), [.025, .975]).tolist()}
        comparisons.append({'arm': [sa, na], 'reference': [sb, nb], 'metrics': metrics})
    result = {'execution': execution, 'rows': rows, 'summary': summary, 'paired_comparisons': comparisons,
        'cost': {'new_source_calls': 0, 'new_target_calls': 0, 'new_verification_calls': 0,
                 'historical_response_archive_calls': {'16': 288, '73': 1314},
                 'new_native_source_head_fits': 16, 'cached_logical_initialization_calls': n0*len(rows)},
        'source_fit_cpu_seconds': sum(d['cpu_seconds'] for d in fits.values()),
        'cpu_seconds': time.process_time()-started}
    save(out/'results.json', result)
    print(json.dumps({'summary': summary, 'cpu_seconds': result['cpu_seconds']}), flush=True)


if __name__ == '__main__':
    {'prepare': prepare, 'fit': fit, 'compare': compare}[sys.argv[1]]()
