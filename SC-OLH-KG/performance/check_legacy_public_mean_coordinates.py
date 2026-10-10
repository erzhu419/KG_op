"""Public mean coordinates with frozen source risk heads and matched controls."""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_diverse_source_design import encode_diverse_source, score_row
from performance.check_legacy_joint_common_covariance import restore_regularizer
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.run_legacy_joint_population_independent_pilot import target_problem
from performance.run_legacy_joint_population_matched_baselines import paired_fields, standardize
from performance.run_legacy_joint_population_source_budget import restore_joint
from performance.run_legacy_related_inventory_source import problem_for, save, training_view
from representation.joint_response_mixture import JointResponseMixture
from representation.legacy_observable_prior import legacy_exposure_descriptors
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis, fit_orthogonal_function_prior
from representation.public_control_mean import public_control_mean_coordinates, source_prior_moments
from representation.source_function_prior import head_batches
from representation.source_joint_population import (fit_joint_source_population, fit_joint_common_source_population,
    continue_joint_common_source_population, fit_flat_common_source_population)
from representation.source_kernel_population import (fit_unequal_site_kernel_population,
    fit_joint_common_unequal_site_kernel_population, predict_unequal_site_kernel_population)
from representation.source_known_noise_gp import fit_known_noise_gp

EXECUTION = ROOT/'performance/manifests/legacy_joint_population_public_mean_execution_v1.json'


def prepare(spec, old_inputs):
    started = time.process_time(); source = json.loads((ROOT/spec['source_execution']).read_text())
    data = json.loads((ROOT/spec['source_data']).read_text())
    public = json.loads((ROOT/spec['source_data']).with_name('frozen_inputs.json').read_text())
    bank = [tuple(p) for p in public['bank']]; raw, risk_desc = {}, {}
    for i, task in enumerate(public['tasks']):
        provider = no_outcomes(problem_for(public['plan'], task))
        raw[i] = public_control_mean_coordinates(provider, bank)
        risk_desc[i] = legacy_exposure_descriptors(provider, bank, 50)
    ids, layouts = source['training_tasks'], source['source_site_indices_by_task']
    stacked = np.vstack([raw[i][sites] for i, sites in zip(ids, layouts)])
    mean_psi, center, scale = standardize(stacked)
    tasks = [{**t, 'psi': p.tolist(), 'features': p.tolist()} for t, p in
             zip(old_inputs['source_tasks'], np.split(mean_psi, len(ids)))]
    coordinate, _, risk_center, risk_scale = encode_diverse_source(data, ids, layouts, risk_desc,
        public['plan']['task_distribution']['L'])
    np.testing.assert_allclose(risk_center, old_inputs['source_feature_center'], atol=0., rtol=0.)
    np.testing.assert_allclose(risk_scale, old_inputs['source_feature_scale'], atol=0., rtol=0.)
    target_inputs = json.loads((ROOT/spec['target_inputs']).read_text()); targets = []
    target_bank = [tuple(p) for p in target_inputs['bank']]
    for target in target_inputs['targets']:
        provider = no_outcomes(target_problem(target_inputs['target_plan'], target['config']))
        targets.append({**target,
            'mean_psi': ((public_control_mean_coordinates(provider, target_bank)-center)/scale).tolist(),
            'risk_psi': coordinate.transform(legacy_exposure_descriptors(provider, target_bank, 50)).tolist(),
            'alpha': provider.alpha})
    reference = (raw[ids[0]]-center)/scale; basis = OrthogonalLowOrderBasis(reference)
    design = np.column_stack([np.ones(len(mean_psi)), basis.transform(mean_psi)])
    result = {'execution': spec, 'source_tasks': tasks,
        'mean_coordinate_center': center.tolist(), 'mean_coordinate_scale': scale.tolist(),
        'mean_coordinate_names': ['stock_control_mean', 'reorder_control_mean', 'safety_control_mean', 'policy_sd'],
        'public_mean_reference': reference.tolist(), 'mean_basis': basis.diagnostics(),
        'mean_source_design_rank': int(np.linalg.matrix_rank(design)),
        'heldout_mean_psi': {str(i): ((raw[i]-center)/scale).tolist() for i in spec['heldout_tasks']},
        'targets': targets, 'target_plan': target_inputs['target_plan'], 'source_spec': target_inputs['source_spec'],
        'bank': target_inputs['bank'], 'cpu_seconds': time.process_time()-started}
    save(ROOT/spec['output']/'inputs.json', result)
    print('public mean inputs prepared', mean_psi.shape, 'functions', basis.feature_dim,
          'design rank', result['mean_source_design_rank'], flush=True)
    return result


def fit(spec, inputs):
    out = ROOT/spec['output']; path = out/'source_fit.json'
    fitted = json.loads(path.read_text()) if path.exists() else {'heads': [], 'cpu_seconds': 0.}
    tasks = [{**t, **{k: np.asarray(t[k]) for k in ('psi', 'features', 'response', 'noise')}} for t in inputs['source_tasks']]
    basis = OrthogonalLowOrderBasis(inputs['public_mean_reference'])
    for h in range(len(fitted['heads']), 2):
        started = time.process_time(); batches = head_batches(tasks, h); noises = [t['noise'][:, h] for t in tasks]
        regularizer = fit_orthogonal_function_prior(batches, noises, basis, max_iterations=spec['regularizer_max_iterations'])
        fixed = fit_joint_source_population(regularizer, batches, noises, max_iterations=spec['fixed_population_max_iterations'])
        joint = fit_joint_common_source_population(regularizer, batches, noises, fixed.diagnostics, optimizer=spec['optimizer'])
        if not joint.diagnostics['optimizer_converged']:
            save(out/f'head{h}_joint_before_completion.json', joint.diagnostics)
            joint = continue_joint_common_source_population(regularizer, batches, noises, joint.diagnostics, optimizer=spec['optimizer'])
        flat = fit_flat_common_source_population(regularizer, batches, noises, joint.diagnostics, optimizer=spec['optimizer'])
        anchor = fit_known_noise_gp(tasks, h, spec['gp_anchor_max_iterations'])
        gp_fixed = fit_unequal_site_kernel_population(tasks, h, anchor, optimizer=spec['optimizer'])
        gp_joint = fit_joint_common_unequal_site_kernel_population(tasks, h, gp_fixed, optimizer=spec['optimizer'])
        record = {'regularizer': regularizer.diagnostics, 'fixed_structural': fixed.diagnostics,
            'joint_structural': joint.diagnostics, 'flat': flat.diagnostics,
            'anchor': anchor, 'fixed_gp': gp_fixed, 'joint_gp': gp_joint}
        fitted['heads'].append(record); fitted['cpu_seconds'] += time.process_time()-started; save(path, fitted)
        print('public mean source head', h, {k: (d['optimizer_converged'], d['optimizer_iterations']) for k, d in record.items()}, flush=True)
    if not all(d['optimizer_converged'] for h in fitted['heads'] for d in h.values()):
        raise RuntimeError('complete capped/failed source objectives before public-mean forecasts')
    return fitted


def components(spec, inputs, old_inputs, fits, old_fit, mean_psi, risk_psi):
    mean_basis = OrthogonalLowOrderBasis(inputs['public_mean_reference'])
    risk_basis = OrthogonalLowOrderBasis(old_inputs['public_reference_psi'])
    risk_priors = [restore_joint(risk_basis, old_fit['heads'][h]['joint_structural']) for h in (2, 3)]
    risk_moments = [source_prior_moments(p, risk_psi) for p in risk_priors]
    fields = []
    for key in ('fixed_structural', 'joint_structural', 'flat'):
        mean_priors = [restore_joint(mean_basis, h[key]) for h in fits['heads']]
        means = [source_prior_moments(p, mean_psi) for p in mean_priors]
        fields.append(paired_fields(means+risk_moments))
    for key in ('fixed_gp', 'joint_gp'):
        fields.append(paired_fields([predict_unequal_site_kernel_population(mean_psi, h[key])
                                    for h in fits['heads']]+risk_moments))
    fields.append(paired_fields([source_prior_moments(restore_joint(risk_basis, h['joint_structural']), risk_psi)
                                for h in old_fit['heads']]))
    legacy_x = (risk_psi-np.asarray(old_inputs['source_feature_center']))/np.asarray(old_inputs['source_feature_scale'])
    fields.append(paired_fields([predict_unequal_site_kernel_population(legacy_x, h['joint_gp']) for h in old_fit['heads']]))
    return fields


def forecast_rows(model, task_id, task_index):
    rows = []
    for arm, (f, g) in zip(model.names, model.components):
        fm, fv = f.predict(); gm, gv = g.predict()
        rows.append({'task_id': task_id, 'task_index': task_index, 'arm': arm,
            'mean': np.column_stack([fm[:, 0], gm[:, 0], gm[:, 1], fm[:, 1]]).tolist(),
            'variance': np.column_stack([fv[:, 0], gv[:, 0], gv[:, 1], fv[:, 1]]).tolist(),
            'local_joint_components': [[[{'weight': w, 'mean': m.tolist(), 'covariance': c.tolist()}
                for w, m, c in head.local_components(j)] for head in (f, g)] for j in range(len(fm))]})
    return rows


def source_check(spec, inputs, old_inputs, fits, old_fit):
    started = time.process_time(); data = json.loads((ROOT/spec['source_data']).read_text()); rows = []
    for i in spec['heldout_tasks']:
        model = JointResponseMixture(spec['arms'], components(spec, inputs, old_inputs, fits, old_fit,
            np.asarray(inputs['heldout_mean_psi'][str(i)]), np.asarray(old_inputs['heldout_public_psi'][str(i)])))
        task = data['tasks'][i]
        for j in (0, 1):
            value = task['Y_replicates'][j][0]; model.update_raw(j, [value[0], value[1]-task['tau']])
        rows.extend(forecast_rows(model, task['task_id'], i))
    out = ROOT/spec['output']; save(out/'source_predictions.json', {'execution': spec, 'rows': rows})
    original = json.loads((ROOT/spec['reference_source_predictions']).read_text())
    for r in (r for r in rows if r['arm'] == 'legacy_mean_joint_structural'):
        old = next(x for x in original['rows'] if x['task_id'] == r['task_id'] and x['arm'] == 'joint_common_structural')
        for key in ('mean', 'variance'):
            np.testing.assert_allclose(r[key], old[key], rtol=1e-8, atol=1e-9)
    public = json.loads((ROOT/spec['source_data']).with_name('frozen_inputs.json').read_text())
    for r in rows:
        labels = training_view(data, [r['task_index']], public['plan']['task_distribution']['L'])[0]
        score_row(r, labels, 3)
        r['calibration_profile']['diagnostic_points'].append({'candidate': 60, 'g_mean': r['mean'][60][1],
            'g_observed': float(labels['response'][60, 1]), 'g_standardized_residual': r['calibration_profile']['standardized_residual'][60][1]})
    summary = {a: {k: np.mean([r['scores'][k] for r in rows if r['arm'] == a], axis=0).tolist()
                   for k in rows[0]['scores']} for a in spec['arms']}
    pairs = []
    for i in spec['heldout_tasks']:
        a = next(r for r in rows if r['task_index'] == i and r['arm'] == 'public_mean_joint_structural')['scores']
        b = next(r for r in rows if r['task_index'] == i and r['arm'] == 'legacy_mean_joint_structural')['scores']
        pairs.append({'task_index': i, 'checks': {'g_mse_improved': a['observed_head_mse'][1] < b['observed_head_mse'][1],
            'joint_density_improved': a['joint_batch_log_score'] > b['joint_batch_log_score'],
            'constraint_density_improved': a['response_batch_log_scores'][1] > b['response_batch_log_scores'][1]}})
    result = {'summary': summary, 'paired_tasks': pairs,
        'source_gate_passed': all(all(p['checks'].values()) for p in pairs), 'cpu_seconds': time.process_time()-started}
    save(out/'source_calibration_profiles.json', {'rows': [{k: v for k, v in r.items() if k != 'local_joint_components'} for r in rows]})
    save(out/'source_results.json', result)
    print('public mean source results', json.dumps(result), flush=True)
    return result


def target_diagnosis(spec, inputs, old_inputs, fits, old_fit):
    started = time.process_time(); rows = []
    source_mean = np.asarray(inputs['public_mean_reference']); source_risk = np.asarray(old_inputs['public_reference_psi'])
    n = len(inputs['bank'])
    for i, target in enumerate(inputs['targets']):
        model = JointResponseMixture(spec['arms'], components(spec, inputs, old_inputs, fits, old_fit,
            np.vstack([target['mean_psi'], source_mean]), np.vstack([target['risk_psi'], source_risk])))
        before = {a: g.predict()[0].copy() for a, (f, g) in zip(spec['arms'], model.components)}
        for j, value in zip(target['initialization']['candidate_indices'], target['initialization']['paired_observations']):
            model.update_raw(j, [value[0], value[1]-target['tau']])
        for row in forecast_rows(model, target['task_id'], i):
            row['prior_g_eta_mean'] = before[row['arm']].tolist(); rows.append(row)
    out = ROOT/spec['output']; save(out/'target_predictions.json', {'execution': spec, 'rows': rows})
    scoring = json.loads((ROOT/spec['target_scoring']).read_text())
    reference = json.loads((ROOT/spec['reference_target_results']).read_text()); diagnostic = []
    for row in rows:
        target = inputs['targets'][row['task_index']]
        fold = next(f for f in scoring['folds'] if f['task_id'] == row['task_id'])
        truth = np.asarray(fold['scoring_truth'])[:, [0, 2, 3, 1]]
        # Cached truth is [f,eta_f,g,eta_g]; forecast is [f,g,eta_g,eta_f].
        mean = np.asarray(row['mean']); variance = np.asarray(row['variance']); before = np.asarray(row['prior_g_eta_mean'])
        error = mean[:n]-truth
        row['scores'] = {'head_order': ['f', 'g', 'eta_g', 'eta_f'],
            'truth_mse': (error[2:]**2).mean(axis=0).tolist(),
            'moment_interval_95_truth_coverage': (abs(error[2:]) <= 1.96*np.sqrt(variance[2:n])).mean(axis=0).tolist()}
        for j in spec['diagnostic_candidates']:
            diagnostic.append({'task_id': row['task_id'], 'arm': row['arm'], 'candidate': j,
                'g_mean': float(mean[j, 1]), 'g_bias': float(error[j, 1]), 'g_true': float(truth[j, 1]),
                'source_input_prior_to_target_truth_bias': float(before[n+j, 0]-truth[j, 1]),
                'prior_input_transfer_mean_shift': float(before[j, 0]-before[n+j, 0]),
                'paid_update_g_mean_shift': float(mean[j, 1]-before[j, 0]),
                'mean_eta_plugin_margin': float(mean[j, 1]+norm.ppf(1-target['alpha'])*np.exp(mean[j, 2]/2)),
                'true_chance_margin': float(truth[j, 1]+norm.ppf(1-target['alpha'])*np.exp(truth[j, 2]/2))})
        if row['arm'] == 'legacy_mean_joint_structural':
            old = next(r for r in reference['rows'] if r['task_id'] == row['task_id'] and r['arm'] == 'joint_common_structural')
            for point in old['forecast']['diagnostic_points']:
                np.testing.assert_allclose(mean[point['candidate'], [0, 3, 1, 2]], point['mean'], rtol=1e-8, atol=1e-9)
    checks = []
    for target in inputs['targets']:
        points = []
        for j in (60, 63):
            a = next(r for r in diagnostic if r['task_id'] == target['task_id'] and r['candidate'] == j and r['arm'] == 'public_mean_joint_structural')
            b = next(r for r in diagnostic if r['task_id'] == target['task_id'] and r['candidate'] == j and r['arm'] == 'legacy_mean_joint_structural')
            points.append({'candidate': j, 'absolute_g_bias_improved': abs(a['g_bias']) < abs(b['g_bias'])})
        checks.append({'task_id': target['task_id'], 'points': points})
    summary = {arm: {'truth_mse': np.mean([r['scores']['truth_mse'] for r in rows if r['arm'] == arm], axis=0).tolist(),
        'points': {str(j): {k: float(np.mean([r[k] for r in diagnostic if r['arm'] == arm and r['candidate'] == j]))
            for k in ('g_bias', 'prior_input_transfer_mean_shift', 'mean_eta_plugin_margin', 'true_chance_margin')}
            for j in spec['diagnostic_candidates']}} for arm in spec['arms']}
    result = {'summary': summary, 'paired_tasks': checks, 'diagnostic_rows': diagnostic,
        'selected_bias_gate_passed': all(p['absolute_g_bias_improved'] for t in checks for p in t['points']),
        'interpretation': 'opened-target local diagnostics only, no new rankings or deployment; mean-eta margin is descriptive',
        'cpu_seconds': time.process_time()-started}
    save(out/'target_calibration_profiles.json', {'rows': [{k: v for k, v in r.items() if k != 'local_joint_components'} for r in rows]})
    save(out/'target_diagnostic.json', result)
    print('public mean target results', json.dumps({'summary': summary, 'selected_bias_gate_passed': result['selected_bias_gate_passed']}), flush=True)
    return result


def run():
    spec = json.loads(EXECUTION.read_text()); out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    old_inputs = json.loads((ROOT/spec['source_inputs']).read_text()); old_fit = json.loads((ROOT/spec['source_fit']).read_text())
    path = out/'inputs.json'; inputs = json.loads(path.read_text()) if path.exists() else prepare(spec, old_inputs)
    fits = fit(spec, inputs)
    source_path = out/'source_results.json'
    source = json.loads(source_path.read_text()) if source_path.exists() else source_check(spec, inputs, old_inputs, fits, old_fit)
    target = target_diagnosis(spec, inputs, old_inputs, fits, old_fit)
    result = {'execution': spec, 'source': source, 'target': target,
        'terminal_progress_gate_passed': source['source_gate_passed'] and target['selected_bias_gate_passed'],
        'source_fit_cpu_seconds': fits['cpu_seconds'],
        'cost': {'new_source_calls': 0, 'new_target_calls': 0, 'new_verification_calls': 0,
            'new_native_source_head_fits': 14+sum('continued_from_optimizer_iterations' in h['joint_structural'] for h in fits['heads']),
            'new_terminal_integrations': 0, 'historical_source_calls_per_arm': 288}}
    save(out/'results.json', result); print('public mean terminal progress gate', result['terminal_progress_gate_passed'], flush=True)


if __name__ == '__main__':
    run()
