"""Cached same-budget decisions after source-only joint covariance learning."""

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
from performance.run_legacy_joint_population_matched_baselines import paired_fields
from performance.run_legacy_joint_population_source_budget import encode_subset, restore_joint
from performance.run_legacy_related_inventory_source import problem_for, save
from performance.score_legacy_related_inventory_source import safe_terminal_values
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.joint_response_mixture import JointResponseMixture
from representation.legacy_observable_prior import legacy_exposure_descriptors
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.source_joint_population import same_source_full_residual_prior
from representation.source_kernel_population import predict_kernel_population, predict_joint_common_kernel_population


def run():
    spec = json.loads((ROOT/'performance/manifests/legacy_joint_population_joint_common_decision_v1.json').read_text())
    source = json.loads((ROOT/spec['source_execution']).read_text())
    data = json.loads((ROOT/source['source_data']).read_text())
    public = json.loads((ROOT/source['source_data']).with_name('frozen_inputs.json').read_text())
    old_fit = json.loads((ROOT/spec['old_source_fit']).read_text())
    new_fit = json.loads((ROOT/spec['new_source_fit']).read_text())
    inputs = json.loads((ROOT/spec['target_inputs']).read_text())
    plan = public['plan']; ids, sites = source['training_tasks'], source['source_sites']
    source_bank = [tuple(p) for p in public['bank']]
    descriptors = {i: legacy_exposure_descriptors(no_outcomes(problem_for(plan, public['tasks'][i])), source_bank, 50) for i in ids}
    subset = {'bank': [data['bank'][j] for j in sites], 'tasks': [
        {'task_id': data['tasks'][i]['task_id'], 'tau': data['tasks'][i]['tau'],
         'Y_replicates': [data['tasks'][i]['Y_replicates'][j] for j in sites]} for i in ids]}
    coordinate, tasks, center, scale = encode_subset(subset, list(range(len(ids))), len(sites),
        {k: descriptors[i][sites] for k, i in enumerate(ids)}, plan['task_distribution']['L'])
    basis = OrthogonalLowOrderBasis(coordinate.transform(descriptors[ids[0]]))
    old_structural = [restore_joint(basis, h['joint']) for h in old_fit['heads']]
    new_structural = [restore_joint(basis, h['structural']) for h in new_fit['heads']]
    priors = [old_structural, new_structural, [same_source_full_residual_prior(p) for p in new_structural]]
    names = spec['arms']; bank = [tuple(p) for p in inputs['bank']]
    started = time.process_time(); rows, jobs = [], []
    for target in inputs['targets']:
        provider = no_outcomes(target_problem(inputs['target_plan'], target['config']))
        raw = legacy_exposure_descriptors(provider, bank, 50); psi = coordinate.transform(raw); x = (psi-center)/scale
        components = [(JointFunctionRiskPosterior([p[0]], [p[3]], psi),
                       JointFunctionRiskPosterior([p[1]], [p[2]], psi)) for p in priors]
        components.extend([paired_fields([prediction(tasks[0]['features'], x, h['gp']) for h in fit['heads']])
                           for prediction, fit in [(predict_kernel_population, old_fit), (predict_joint_common_kernel_population, new_fit)]])
        model = JointResponseMixture(names, components)
        observed = target['initialization']['paired_observations']
        for j, v in zip(target['initialization']['candidate_indices'], observed):
            model.update_raw(j, [v[0], v[1]-provider.tau])
        terminal, _ = terminal_at_prefix(provider, bank[:2], {'paid_observations': observed}, 2,
                                         inputs['source_spec'], len(bank))
        terminal.retain_recommendation(2)
        reward, cost, actions = model.terminal_components(terminal, 360, spec['terminal_draw_power'], spec['integration_seed'])
        forecasts = {}
        for k, (name, (f, g)) in enumerate(zip(names, model.components)):
            best = int(np.lexsort((cost[k], -reward[k]))[0])
            fm, fv = f.predict(); gm, gv = g.predict()
            forecasts[name] = (np.column_stack([fm, gm]), np.column_stack([fv, gv]))
            rows.append({'task_id': target['task_id'], 'arm': name, 'action_index': best, 'action': actions[best],
                'predicted_reward': float(reward[k, best]), 'predicted_verification_calls': float(cost[k, best])})
        jobs.append((terminal, forecasts))
        print('joint-common target actions frozen in memory', target['task_id'], [r['action']['primary'] for r in rows[-5:]], flush=True)
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    save(out/'recommendations.json', {'execution': spec, 'rows': rows})
    scoring = json.loads((ROOT/'paper_artifacts/legacy_joint_population_independent_pilot_v1/results.json').read_text())
    for target, fold, (terminal, forecasts) in zip(inputs['targets'], scoring['folds'], jobs):
        truth = np.asarray(fold['scoring_truth']); feasible = np.isin(np.arange(len(bank)), fold['feasible_policies'])
        safe, calls = safe_terminal_values(terminal, truth, feasible, 360)
        _, _, actions = terminal.evaluate(truth[None], 360)
        for row in [r for r in rows if r['task_id'] == target['task_id']]:
            a = row['action_index']
            if actions[a] != row['action']:
                raise ValueError('frozen joint-common action differs from scored class')
            row['scores'] = {'conditional_safe_reward': float(safe[a]), 'conditional_total_calls': float(2+calls[a]),
                             'primary_feasible': bool(feasible[row['action']['primary']])}
            mean, variance = forecasts[row['arm']]; error = mean[2:]-truth[2:]
            row['forecast'] = {'head_order': ['f', 'eta_f', 'g', 'eta_g'],
                'truth_mse': (error**2).mean(axis=0).tolist(),
                'moment_interval_95_truth_coverage': (abs(error) <= 1.96*np.sqrt(variance[2:])).mean(axis=0).tolist(),
                'selected_points': [{'candidate': j, 'mean': mean[j].tolist(), 'sd': np.sqrt(variance[j]).tolist(),
                                     'scoring_truth': truth[j].tolist()}
                                    for j in list(dict.fromkeys([row['action']['primary'], 63]))]}
    ids = [t['task_id'] for t in inputs['targets']]
    draws = np.random.default_rng(spec['bootstrap_seed']).integers(len(ids), size=(spec['bootstrap_draws'], len(ids)))
    summary, comparisons = {}, []
    for name in names:
        selected = [r for r in rows if r['arm'] == name]
        summary[name] = {'feasible_primary_count': sum(r['scores']['primary_feasible'] for r in selected),
            **{m: float(np.mean([r['scores'][m] for r in selected])) for m in ['conditional_safe_reward', 'conditional_total_calls']},
            'truth_mse': np.mean([r['forecast']['truth_mse'] for r in selected], axis=0).tolist()}
    for a, b in [(names[1], ref) for ref in [names[0], names[2], names[3], names[4]]]+[(names[4], names[3])]:
        metrics = {}
        for metric in ['conditional_safe_reward', 'conditional_total_calls']:
            differences = np.asarray([next(r for r in rows if r['task_id'] == task and r['arm'] == a)['scores'][metric]
                                     -next(r for r in rows if r['task_id'] == task and r['arm'] == b)['scores'][metric] for task in ids])
            metrics[metric] = {'task_differences': differences.tolist(), 'mean_difference': float(differences.mean()),
                'task_bootstrap_95_interval': np.quantile(differences[draws].mean(axis=1), [.025, .975]).tolist()}
        comparisons.append({'arm': a, 'reference': b, 'metrics': metrics})
    result = {'execution': spec, 'rows': rows, 'summary': summary, 'paired_comparisons': comparisons,
        'cost': {'new_source_calls': 0, 'new_target_calls': 0, 'new_verification_calls': 0,
                 'source_response_archive_calls_per_arm': 288, 'new_source_hyperparameter_fits': 0},
        'cpu_seconds': time.process_time()-started}
    save(out/'results.json', result)
    print(json.dumps({'summary': summary, 'cpu_seconds': result['cpu_seconds']}), flush=True)


if __name__ == '__main__':
    run()
