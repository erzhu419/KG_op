"""Source-selected complete GP uncertainty control with cached component reuse."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read, forecast
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.replay_legacy_mixture_budget import terminal_at_prefix
from performance.run_legacy_joint_population_independent_pilot import target_problem
from performance.run_legacy_joint_population_matched_baselines import paired_fields
from performance.run_legacy_joint_population_source_budget import restore_joint
from performance.run_legacy_related_inventory_source import save
from performance.score_legacy_related_inventory_source import safe_terminal_values
from representation.joint_response_mixture import JointResponseMixture
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.public_control_mean import public_control_mean_coordinates, source_prior_moments
from representation.source_kernel_population import predict_unequal_site_kernel_population

EXECUTION = 'performance/manifests/legacy_joint_population_complete_gp_control_v1.json'
CASE = 'full_plus_six_complete_source_deletions'
RISK_ONLY = 'full_plus_six_relearned_map_deletion_risks'


def run():
    started = time.process_time(); spec = read(EXECUTION); source_result = read(spec['source_results'])
    gp = [c for c in source_result['contrasts'] if c['mean_arm'] == 'joint_gp']
    if not all(c['g_mse_difference'] <= 0 and c['joint_batch_score_difference'] >= 0 and c['g_batch_score_difference'] > 0 for c in gp):
        raise RuntimeError('complete-GP source predictions must pass on both opened holdouts')
    source = read(spec['source_control']); frozen = read(spec['risk_prediction_inputs'])
    members = [m for m in frozen['risk_members'] if m['mode'] in ('full', source['risk_mode'])]
    mean_inputs = [read(source['full_mean_inputs'])]+[read(Path(source['output'])/f'task{i}'/'inputs.json') for i in source['deletion_tasks']]
    means = [read(source['full_mean_fit'])]+[read(Path(source['output'])/f'task{i}'/'source_fit.json') for i in source['deletion_tasks']]
    inputs = read(spec['target_inputs']); reference = read(spec['reference_decisions'])
    saved = read(Path(reference['output'])/'recommendations.json')['rows']
    cached = read(Path(reference['output'])/'component_utilities.json')['rows']
    old_predictions = read(spec['reference_predictions'])['rows']
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    bank = [tuple(p) for p in inputs['bank']]; remaining = spec['total_target_call_cap']-spec['target_prefix']
    predictions, jobs = [], {}
    for target in inputs['targets']:
        task = target['task_id']; provider = no_outcomes(target_problem(inputs['target_plan'], target['config']))
        raw = public_control_mean_coordinates(provider, bank); fields = []
        query = next(q for q in frozen['queries'] if q['task_id'] == task)
        for member, inp, fit in zip(members, mean_inputs, means):
            psi = (raw-np.asarray(inp['mean_coordinate_center']))/np.asarray(inp['mean_coordinate_scale'])
            mean = [predict_unequal_site_kernel_population(psi, h['joint_gp']) for h in fit['heads']]
            risk_inputs, risks = read(member['inputs']), read(member['fit'])
            basis = OrthogonalLowOrderBasis(risk_inputs['public_risk_reference'])
            eta = [source_prior_moments(restore_joint(basis, h['joint_structural']),
                np.asarray(member['psi'][task])) for h in risks['heads']]
            fields.append(paired_fields(mean+eta))
        model = JointResponseMixture([m['name'] for m in members], fields)
        for j, value in zip(query['candidate_indices'], query['paired_observations']):
            model.update_raw(j, [value[0], value[1]-query['tau']])
        full = forecast(query, 'joint_gp', 'single_full144_risk', JointResponseMixture(['full144'], [model.components[0]]))
        old = next(r for r in old_predictions if r['task_id'] == task and r['mean_arm'] == 'joint_gp' and r['risk_case'] == 'single_full144_risk')
        for k in ('mean', 'variance'):
            np.testing.assert_allclose(full[k], old[k], rtol=1e-8, atol=1e-9)
        predictions.append(forecast(query, 'joint_gp', CASE, model))
        terminal, _ = terminal_at_prefix(provider, bank[:2], {'paid_observations': query['paired_observations']},
                                        2, inputs['source_spec'], len(bank))
        terminal.retain_recommendation(spec['public_retained_candidate']); jobs[task] = (model, terminal)
    save(out/'target_predictions.json', {'execution': spec, 'rows': predictions})
    # All six complete-GP forecasts freeze before the36 new component integrations.
    rows = [{**r} for r in saved if r['mean_arm'] in ('joint_structural', 'flat') and r['risk_case'] == RISK_ONLY]
    components = []
    for target in inputs['targets']:
        task = target['task_id']; model, terminal = jobs[task]
        old = next(r for r in cached if r['task_id'] == task and r['mean_arm'] == 'joint_gp')
        rewards, costs = [old['rewards'][0]], [old['costs'][0]]
        for f, g in model.components[1:]:
            r, c, actions = terminal.evaluate_posterior(f, g, remaining, spec['terminal_draw_power'], spec['integration_seed'])
            rewards.append(r); costs.append(c)
        for previous in (r for r in saved if r['task_id'] == task):
            if previous['action'] != actions[previous['action_index']]:
                raise ValueError('current action indices differ from the cached component contract')
        reward, cost = model.weights@np.asarray(rewards), model.weights@np.asarray(costs)
        best = int(np.lexsort((cost, -reward))[0])
        rows.append({'task_id': task, 'mean_arm': 'joint_gp', 'risk_case': CASE, 'arm': 'joint_gp/'+CASE,
            'action_index': best, 'action': actions[best], 'predicted_reward': float(reward[best]),
            'predicted_verification_calls': float(cost[best])})
        components.append({'task_id': task, 'identity_names': model.names, 'identity_weights': model.weights.tolist(),
            'rewards': np.asarray(rewards).tolist(), 'costs': np.asarray(costs).tolist()})
        save(out/'recommendations.json', {'execution': spec, 'all_choices_frozen': False, 'rows': rows})
        print('complete GP choice frozen', task, actions[best], flush=True)
    save(out/'component_utilities.json', {'execution': spec, 'rows': components})
    save(out/'recommendations.json', {'execution': spec, 'all_choices_frozen': True, 'rows': rows})
    scoring = read(spec['target_scoring']); diagnostics = []
    for target, prediction in zip(inputs['targets'], predictions):
        task = target['task_id']; _, terminal = jobs[task]
        fold = next(f for f in scoring['folds'] if f['task_id'] == task)
        truth = np.asarray(fold['scoring_truth']); feasible = np.isin(np.arange(len(bank)), fold['feasible_policies'])
        safe, calls = safe_terminal_values(terminal, truth, feasible, remaining)
        for row in (r for r in rows if r['task_id'] == task):
            j = row['action_index']; row['scores'] = {'conditional_safe_reward': float(safe[j]),
                'conditional_total_calls': float(2+calls[j]), 'primary_feasible': bool(feasible[row['action']['primary']])}
        mean, variance = np.asarray(prediction['mean']), np.asarray(prediction['variance'])
        error = mean-truth[:, [0, 2, 3, 1]]
        prediction['scores'] = {'truth_mse': (error[2:]**2).mean(axis=0).tolist(),
            'moment_interval_95_truth_coverage': (abs(error[2:]) <= 1.96*np.sqrt(variance[2:])).mean(axis=0).tolist()}
        for j in spec['diagnostic_candidates']:
            diagnostics.append({'task_id': task, 'candidate': j, 'g_bias': float(error[j, 1]), 'eta_g_bias': float(error[j, 2]),
                'mean_eta_plugin_margin': float(mean[j, 1]+norm.ppf(1-target['alpha'])*np.exp(mean[j, 2]/2)),
                'true_chance_margin': float(truth[j, 2]+norm.ppf(1-target['alpha'])*np.exp(truth[j, 3]/2))})
    names = list(dict.fromkeys(r['arm'] for r in rows)); ids = [t['task_id'] for t in inputs['targets']]
    summary = {a: {'feasible_primary_count': sum(r['scores']['primary_feasible'] for r in rows if r['arm'] == a),
        **{k: float(np.mean([r['scores'][k] for r in rows if r['arm'] == a]))
           for k in ('conditional_safe_reward', 'conditional_total_calls')}} for a in names}
    draws = np.random.default_rng(spec['bootstrap_seed']).integers(len(ids), size=(spec['bootstrap_draws'], len(ids)))
    reference_scores = read(Path(reference['output'])/'results.json')['rows']
    comparisons = []; main = 'joint_structural/'+RISK_ONLY; gp_name = 'joint_gp/'+CASE
    for a, b, pool in [(main, gp_name, rows), ('flat/'+RISK_ONLY, gp_name, rows),
        (gp_name, 'joint_gp/'+RISK_ONLY, rows+reference_scores)]:
        d = np.array([next(r for r in pool if r['task_id'] == t and r['arm'] == a)['scores']['conditional_safe_reward']
                     -next(r for r in pool if r['task_id'] == t and r['arm'] == b)['scores']['conditional_safe_reward']
                     for t in ids])
        comparisons.append({'arm': a, 'reference': b, 'task_differences': d.tolist(), 'mean_difference': float(d.mean()),
            'task_bootstrap_95_interval': np.quantile(d[draws].mean(axis=1), [.025, .975]).tolist()})
    save(out/'calibration_profiles.json', {'rows': predictions})
    result = {'execution': spec, 'rows': rows, 'summary': summary, 'paired_comparisons': comparisons,
        'gp_target_prediction_summary': {k: np.mean([p['scores'][k] for p in predictions], axis=0).tolist() for k in predictions[0]['scores']},
        'diagnostic_rows': diagnostics, 'full_component_forecast_reproduction_passed': True,
        'cost': {'historical_source_calls_per_arm': 144, 'new_source_fits': 0, 'new_simulator_calls': 0,
            'new_verification_calls': 0, 'new_terminal_integrations': 36, 'reused_full_gp_integrations': 6,
            'reused_structural_flat_choices': 12}, 'cpu_seconds': time.process_time()-started}
    save(out/'results.json', result); print({'summary': summary, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
