"""Source-flat covariance attribution with cheap, frozen target diagnostics."""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_joint_common_covariance import restore_regularizer
from performance.check_legacy_diverse_source_design import encode_diverse_source, score_row
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.run_legacy_joint_population_independent_pilot import target_problem
from performance.run_legacy_joint_population_source_budget import restore_joint
from performance.run_legacy_related_inventory_source import problem_for, save, training_view
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.joint_response_mixture import JointResponseMixture
from representation.legacy_observable_prior import legacy_exposure_descriptors
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.source_function_prior import head_batches
from representation.source_joint_population import fit_flat_common_source_population

EXECUTION = ROOT/'performance/manifests/legacy_joint_population_common_flat_covariance_execution_v1.json'


def fields(priors, psi):
    return (JointFunctionRiskPosterior([priors[0]], [priors[3]], psi),
            JointFunctionRiskPosterior([priors[1]], [priors[2]], psi))


def forecast(model, task_id, task_index):
    f, g = model.components[0]; fm, fv = f.predict(); gm, gv = g.predict()
    return {'task_id': task_id, 'task_index': task_index, 'arm': 'joint_common_flat',
        'mean': np.column_stack([fm[:, 0], gm[:, 0], gm[:, 1], fm[:, 1]]).tolist(),
        'variance': np.column_stack([fv[:, 0], gv[:, 0], gv[:, 1], fv[:, 1]]).tolist(),
        'local_joint_components': [[[{'weight': w, 'mean': m.tolist(), 'covariance': c.tolist()}
            for w, m, c in head.local_components(j)] for head in (f, g)] for j in range(len(fm))]}


def fit(spec, inputs, old_fit):
    out = ROOT/spec['output']; path = out/'source_fit.json'
    recorded = json.loads(path.read_text()) if path.exists() else {'heads': [], 'cpu_seconds': 0.}
    tasks = [{**t, **{k: np.asarray(t[k]) for k in ('psi', 'features', 'response', 'noise')}}
             for t in inputs['source_tasks']]
    basis = OrthogonalLowOrderBasis(inputs['public_reference_psi'])
    for h in range(len(recorded['heads']), 4):
        started = time.process_time()
        regularizer = restore_regularizer(basis, old_fit['heads'][h]['regularizer'], tasks, h)
        prior = fit_flat_common_source_population(regularizer, head_batches(tasks, h),
            [t['noise'][:, h] for t in tasks], old_fit['heads'][h]['joint_structural'], optimizer=spec['optimizer'])
        recorded['heads'].append(prior.diagnostics); recorded['cpu_seconds'] += time.process_time()-started
        save(path, recorded)
        print('flat source head', h, prior.diagnostics['optimizer_converged'],
              prior.diagnostics['optimizer_iterations'], prior.diagnostics['optimizer_projected_gradient_max'], flush=True)
    if not all(h['optimizer_converged'] for h in recorded['heads']):
        raise RuntimeError('incomplete flat source fit blocks forecasts')
    return recorded, [restore_joint(basis, h) for h in recorded['heads']]


def source_check(spec, inputs, priors):
    started = time.process_time(); data = json.loads((ROOT/spec['source_data']).read_text()); rows = []
    for i in spec['heldout_tasks']:
        model = JointResponseMixture(['joint_common_flat'], [fields(priors, np.asarray(inputs['heldout_public_psi'][str(i)]))])
        task = data['tasks'][i]
        for j in (0, 1):
            value = task['Y_replicates'][j][0]; model.update_raw(j, [value[0], value[1]-task['tau']])
        rows.append(forecast(model, task['task_id'], i))
    out = ROOT/spec['output']; save(out/'source_predictions.json', {'execution': spec, 'rows': rows})
    public = json.loads((ROOT/spec['source_data']).with_name('frozen_inputs.json').read_text())
    reference = json.loads((ROOT/spec['reference_source_results']).read_text()); comparisons = []
    for row in rows:
        labels = training_view(data, [row['task_index']], public['plan']['task_distribution']['L'])[0]
        score_row(row, labels, 3)
        mean, variance = np.asarray(row['mean']), np.asarray(row['variance'])
        old = next(r for r in reference['rows'] if r['task_id'] == row['task_id'] and r['arm'] == 'joint_common_structural')
        checks = {'g_mse_improved': row['scores']['observed_head_mse'][1] < old['scores']['observed_head_mse'][1],
            'joint_density_improved': row['scores']['joint_batch_log_score'] > old['scores']['joint_batch_log_score'],
            'constraint_density_improved': row['scores']['response_batch_log_scores'][1] > old['scores']['response_batch_log_scores'][1]}
        profile = row['calibration_profile']; profile['diagnostic_points'].append({'candidate': 60,
            'g_mean': float(mean[60, 1]), 'g_observed': float(labels['response'][60, 1]),
            'g_standardized_residual': profile['standardized_residual'][60][1]})
        comparisons.append({'task_id': row['task_id'], 'checks': checks, 'reference': old['scores'], 'flat': row['scores']})
    summary = {k: np.mean([r['scores'][k] for r in rows], axis=0).tolist() for k in rows[0]['scores']}
    result = {'summary': summary, 'paired_tasks': comparisons, 'source_gate_passed': all(all(c['checks'].values()) for c in comparisons),
        'cpu_seconds': time.process_time()-started}
    save(out/'source_calibration_profiles.json', {'rows': [{k: v for k, v in r.items() if k != 'local_joint_components'} for r in rows]})
    save(out/'source_results.json', result)
    print('flat source results', json.dumps(result), flush=True)
    return result


def target_diagnosis(spec, inputs, priors):
    started = time.process_time()
    source = json.loads((ROOT/spec['source_execution']).read_text())
    data = json.loads((ROOT/spec['source_data']).read_text())
    public = json.loads((ROOT/spec['source_data']).with_name('frozen_inputs.json').read_text())
    bank50 = [tuple(p) for p in public['bank']]
    desc = {i: legacy_exposure_descriptors(no_outcomes(problem_for(public['plan'], public['tasks'][i])), bank50, 50)
            for i in source['training_tasks']}
    coordinate, _, center, scale = encode_diverse_source(data, source['training_tasks'],
        source['source_site_indices_by_task'], desc, public['plan']['task_distribution']['L'])
    np.testing.assert_allclose(center, inputs['source_feature_center'], atol=0., rtol=0.)
    np.testing.assert_allclose(scale, inputs['source_feature_scale'], atol=0., rtol=0.)
    target_inputs = json.loads((ROOT/spec['target_inputs']).read_text()); bank = [tuple(p) for p in target_inputs['bank']]
    n = len(bank); rows, geometry = [], []
    basis = priors[1].basis; source_features = basis.transform(inputs['public_reference_psi'])
    old_fit = json.loads((ROOT/spec['source_fit']).read_text())
    coefficients = {'joint_common_structural': np.asarray(old_fit['heads'][1]['joint_structural']['coefficient_mean']),
                    'joint_common_flat': priors[1].mean}
    for i, target in enumerate(target_inputs['targets']):
        provider = no_outcomes(target_problem(target_inputs['target_plan'], target['config']))
        target_desc = legacy_exposure_descriptors(provider, bank, 50)
        psi = coordinate.transform(target_desc)
        delta = basis.transform(psi)-source_features
        for j in spec['diagnostic_candidates']:
            impacts = {}
            for arm, coefficient in coefficients.items():
                contribution = delta[j]*coefficient[1:]
                grouped = np.bincount(basis.group_index, weights=contribution)
                top = np.argsort(-abs(grouped))[:5]
                impacts[arm] = {'prior_g_input_shift': float(contribution.sum()),
                    'top_function_groups': [{'group': basis.group_names[k], 'contribution': float(grouped[k])} for k in top]}
            geometry.append({'task_id': target['task_id'], 'candidate': j,
                'raw_source_control_channel_means': desc[source['training_tasks'][0]][j, :3].tolist(),
                'raw_target_control_channel_means': target_desc[j, :3].tolist(),
                'standardized_coordinate_shift': ((psi[j]-np.asarray(inputs['public_reference_psi'])[j])/scale).tolist(),
                'coordinate_axes': inputs['coordinate']['axes'], 'model_impacts': impacts})
        model = JointResponseMixture(['joint_common_flat'], [fields(priors, np.vstack([psi, inputs['public_reference_psi']]))])
        before = model.components[0][1].predict()[0].copy()
        for j in spec['diagnostic_candidates']:
            np.testing.assert_allclose(before[j, 0]-before[n+j, 0], geometry[-len(spec['diagnostic_candidates'])
                +spec['diagnostic_candidates'].index(j)]['model_impacts']['joint_common_flat']['prior_g_input_shift'],
                atol=1e-11, rtol=1e-10)
        for j, value in zip(target['initialization']['candidate_indices'], target['initialization']['paired_observations']):
            model.update_raw(j, [value[0], value[1]-provider.tau])
        row = forecast(model, target['task_id'], i)
        row['prior_g_eta_mean'] = before.tolist(); row['alpha'] = provider.alpha; rows.append(row)
    out = ROOT/spec['output']; save(out/'target_predictions.json', {'execution': spec, 'rows': rows})
    save(out/'input_geometry_profiles.json', {'rows': geometry,
        'interpretation': 'public inputs and cached source coefficient means only; no new fitting, ranking or outcome queries'})
    reference = json.loads((ROOT/spec['reference_target_results']).read_text()); diagnostic = []
    for row in rows:
        old = next(r for r in reference['rows'] if r['task_id'] == row['task_id'] and r['arm'] == 'joint_common_structural')
        mean, before = np.asarray(row['mean']), np.asarray(row['prior_g_eta_mean'])
        for j in spec['diagnostic_candidates']:
            old_point = next(p for p in old['forecast']['diagnostic_points'] if p['candidate'] == j)
            truth = old_point['scoring_truth']; z = norm.ppf(1-row['alpha'])
            diagnostic.append({'task_id': row['task_id'], 'candidate': j,
                'flat_g_mean': float(mean[j, 1]), 'structural_g_mean': old_point['mean'][2], 'target_true_g': truth[2],
                'flat_g_bias': float(mean[j, 1]-truth[2]), 'structural_g_bias': float(old_point['mean'][2]-truth[2]),
                'flat_source_input_prior_to_target_truth_bias': float(before[n+j, 0]-truth[2]),
                'flat_prior_input_transfer_mean_shift': float(before[j, 0]-before[n+j, 0]),
                'flat_paid_update_g_mean_shift': float(mean[j, 1]-before[j, 0]),
                'flat_mean_eta_plugin_margin': float(mean[j, 1]+z*np.exp(mean[j, 2]/2)),
                'structural_mean_eta_plugin_margin': float(old_point['mean'][2]+z*np.exp(old_point['mean'][3]/2)),
                'true_chance_margin': float(truth[2]+z*np.exp(truth[3]/2))})
    summary = {str(j): {k: float(np.mean([r[k] for r in diagnostic if r['candidate'] == j])) for k in
        ('flat_g_bias', 'structural_g_bias', 'flat_source_input_prior_to_target_truth_bias', 'flat_prior_input_transfer_mean_shift',
         'flat_paid_update_g_mean_shift', 'flat_mean_eta_plugin_margin', 'structural_mean_eta_plugin_margin', 'true_chance_margin')}
        for j in spec['diagnostic_candidates']}
    selected = [r for r in diagnostic if r['candidate'] == 60]
    result = {'rows': diagnostic, 'summary': summary,
        'selected_bias_gate_passed': all(abs(r['flat_g_bias']) < abs(r['structural_g_bias']) for r in selected),
        'interpretation': 'opened targets at previously selected points, no new ranking or decision claim; mean-eta margins are descriptive',
        'cpu_seconds': time.process_time()-started}
    save(out/'target_diagnostic.json', result); print('flat target diagnosis', json.dumps({'summary': summary,
        'selected_bias_gate_passed': result['selected_bias_gate_passed']}), flush=True)
    return result


def run():
    spec = json.loads(EXECUTION.read_text()); out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    inputs = json.loads((ROOT/spec['source_inputs']).read_text()); old_fit = json.loads((ROOT/spec['source_fit']).read_text())
    fitted, priors = fit(spec, inputs, old_fit)
    source_path = out/'source_results.json'
    source = json.loads(source_path.read_text()) if source_path.exists() else source_check(spec, inputs, priors)
    target = target_diagnosis(spec, inputs, priors)
    result = {'execution': spec, 'source': source, 'target': target,
        'terminal_progress_gate_passed': source['source_gate_passed'] and target['selected_bias_gate_passed'],
        'source_fit_cpu_seconds': fitted['cpu_seconds'],
        'cost': {'new_source_calls': 0, 'new_target_calls': 0, 'new_verification_calls': 0,
                 'new_native_source_head_fits': 4, 'new_terminal_integrations': 0, 'historical_source_calls_per_arm': 288}}
    save(out/'results.json', result)
    print('terminal progress gate', result['terminal_progress_gate_passed'], flush=True)


if __name__ == '__main__':
    run()
