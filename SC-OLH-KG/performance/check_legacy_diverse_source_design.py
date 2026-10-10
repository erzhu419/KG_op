"""Matched-budget diversified source fits and opened source-task forecasts."""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.check_legacy_joint_common_covariance import restore_regularizer
from performance.run_legacy_joint_population_matched_baselines import paired_fields, standardize
from performance.run_legacy_joint_population_source_budget import restore_joint
from performance.run_legacy_related_inventory_source import problem_for, save, training_view
from representation.joint_function_risk import JointFunctionRiskPosterior, joint_batch_log_score
from representation.joint_response_mixture import JointResponseMixture
from representation.legacy_function_prior import replicated_source_labels
from representation.legacy_observable_prior import LegacyObservableCoordinate, legacy_exposure_descriptors
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis, fit_orthogonal_function_prior
from representation.source_function_prior import SourceFunctionPrior, head_batches
from representation.source_joint_population import (fit_joint_source_population,
    fit_joint_common_source_population, continue_joint_common_source_population, same_source_full_residual_prior)
from representation.source_kernel_population import (fit_unequal_site_kernel_population,
    fit_joint_common_unequal_site_kernel_population, predict_unequal_site_kernel_population)
from representation.source_known_noise_gp import fit_known_noise_gp

EXECUTION = ROOT/'performance/manifests/legacy_joint_population_diverse_source_execution_v1.json'


def encode_diverse_source(data, ids, layouts, descriptors, L):
    """Slice raw observations and public inputs together before fitting maps."""
    labels, selected = [], []
    for i, sites in zip(ids, layouts):
        task = data['tasks'][i]
        labels.append(replicated_source_labels(task['task_id'], np.asarray(data['bank'])[sites]/L,
            [task['Y_replicates'][j] for j in sites], task['tau']))
        selected.append(descriptors[i][sites])
    desc = np.vstack(selected)
    coordinate = LegacyObservableCoordinate().fit(desc, labels)
    psi = coordinate.transform(desc)
    features, center, scale = standardize(psi)
    stops = np.cumsum([len(s) for s in layouts])[:-1]
    tasks = [{**t, 'psi': p, 'features': x} for t, p, x in
             zip(labels, np.split(psi, stops), np.split(features, stops))]
    return coordinate, tasks, center, scale


def prepare(spec):
    started = time.process_time()
    data = json.loads((ROOT/spec['source_data']).read_text())
    public = json.loads((ROOT/spec['source_data']).with_name('frozen_inputs.json').read_text())
    bank = [tuple(p) for p in public['bank']]
    desc = {i: legacy_exposure_descriptors(no_outcomes(problem_for(public['plan'], t)), bank, 50)
            for i, t in enumerate(public['tasks'])}
    coordinate, tasks, center, scale = encode_diverse_source(data, spec['training_tasks'],
        spec['source_site_indices_by_task'], desc, public['plan']['task_distribution']['L'])
    reference = coordinate.transform(desc[spec['training_tasks'][0]])
    basis = OrthogonalLowOrderBasis(reference)
    source_design = np.column_stack([np.ones(sum(len(t['psi']) for t in tasks)),
                                    basis.transform(np.vstack([t['psi'] for t in tasks]))])
    record = {'execution': spec, 'source_tasks': [{**t, **{k: t[k].tolist()
        for k in ('psi', 'features', 'response', 'noise')}} for t in tasks],
        'coordinate': coordinate.diagnostics(), 'public_reference_psi': reference.tolist(),
        'source_feature_center': center.tolist(), 'source_feature_scale': scale.tolist(),
        'heldout_public_psi': {str(i): coordinate.transform(desc[i]).tolist() for i in spec['heldout_tasks']},
        'source_common_design_rank': int(np.linalg.matrix_rank(source_design)),
        'source_coefficient_count': source_design.shape[1], 'basis': basis.diagnostics(),
        'cpu_seconds': time.process_time()-started}
    save(ROOT/spec['output']/'inputs.json', record)
    print('prepared selected source records', len(source_design), 'rank', record['source_common_design_rank'], flush=True)
    return record


def fit(spec, inputs):
    out = ROOT/spec['output']; path = out/'source_fit.json'
    recorded = json.loads(path.read_text()) if path.exists() else {'heads': [], 'cpu_seconds': 0.}
    tasks = [{**t, **{k: np.asarray(t[k]) for k in ('psi', 'features', 'response', 'noise')}}
             for t in inputs['source_tasks']]
    basis = OrthogonalLowOrderBasis(inputs['public_reference_psi'])
    for h in range(len(recorded['heads']), 4):
        started = time.process_time(); batches = head_batches(tasks, h)
        noises = [t['noise'][:, h] for t in tasks]
        regularizer = fit_orthogonal_function_prior(batches, noises, basis,
                                        max_iterations=spec['regularizer_max_iterations'])
        fixed = fit_joint_source_population(regularizer, batches, noises,
                                        max_iterations=spec['fixed_structural_max_iterations'])
        joint = fit_joint_common_source_population(regularizer, batches, noises,
                                        fixed.diagnostics, optimizer=spec['population_optimizer'])
        anchor = fit_known_noise_gp(tasks, h, spec['gp_anchor_max_iterations'])
        gp_fixed = fit_unequal_site_kernel_population(tasks, h, anchor, optimizer=spec['population_optimizer'])
        gp_joint = fit_joint_common_unequal_site_kernel_population(tasks, h, gp_fixed,
                                        optimizer=spec['population_optimizer'])
        head = {'regularizer': regularizer.diagnostics, 'fixed_structural': fixed.diagnostics,
                'joint_structural': joint.diagnostics, 'anchor': anchor,
                'fixed_gp': gp_fixed, 'joint_gp': gp_joint}
        recorded['heads'].append(head); recorded['cpu_seconds'] += time.process_time()-started
        save(path, recorded)
        print('source head', h, {k: (v['optimizer_converged'], v['optimizer_iterations'])
                                for k, v in head.items()}, flush=True)
    for h, head in enumerate(recorded['heads']):
        previous = head['joint_structural']
        if not previous['optimizer_converged'] and 'continued_from_optimizer_iterations' not in previous:
            started = time.process_time()
            regularizer = restore_regularizer(basis, head['regularizer'], tasks, h)
            completed = continue_joint_common_source_population(regularizer, head_batches(tasks, h),
                [t['noise'][:, h] for t in tasks], previous, optimizer=spec['population_optimizer'])
            save(out/f'head{h}_joint_before_completion.json', previous)
            head['joint_structural'] = completed.diagnostics
            recorded['cpu_seconds'] += time.process_time()-started
            save(path, recorded)
            print('source joint head completion', h, completed.diagnostics['optimizer_converged'],
                  completed.diagnostics['optimizer_iterations'], flush=True)
    if not all(v['optimizer_converged'] for h in recorded['heads'] for v in h.values()):
        raise RuntimeError('complete failed source objectives before opening source forecasts')
    return recorded


def components(inputs, fits, psi):
    basis = OrthogonalLowOrderBasis(inputs['public_reference_psi'])
    fixed = [restore_joint(basis, h['fixed_structural']) for h in fits['heads']]
    joint = [restore_joint(basis, h['joint_structural']) for h in fits['heads']]
    laws = [fixed, joint, [same_source_full_residual_prior(p) for p in joint]]
    fields = [(JointFunctionRiskPosterior([p[0]], [p[3]], psi),
               JointFunctionRiskPosterior([p[1]], [p[2]], psi)) for p in laws]
    x = (psi-np.asarray(inputs['source_feature_center']))/np.asarray(inputs['source_feature_scale'])
    fields.extend(paired_fields([predict_unequal_site_kernel_population(x, h[k])
                                for h in fits['heads']]) for k in ('fixed_gp', 'joint_gp'))
    return fields


def score_row(row, labels, repeats):
    response = labels['response']; mean, variance = np.asarray(row['mean']), np.asarray(row['variance'])
    error = mean-response; scores = []; observation_variance = variance.copy()
    for j, local in enumerate(row['local_joint_components']):
        point = []
        for k, (a, b) in enumerate([(0, 3), (1, 2)]):
            components = [(c['weight'], np.asarray(c['mean']), np.asarray(c['covariance'])) for c in local[k]]
            point.append(joint_batch_log_score(components, response[j, a], response[j, b], repeats))
            observation_variance[j, a] += sum(w*np.exp(m[1]+.5*c[1, 1])/repeats for w, m, c in components)
            observation_variance[j, b] += labels['noise'][j, b]
        scores.append(point)
    standardized = error/np.sqrt(observation_variance)
    tested = slice(2, None); scores = np.asarray(scores)
    row['scores'] = {'observed_head_mse': (error[tested]**2).mean(axis=0).tolist(),
        'joint_batch_log_score': float(scores[tested].sum(axis=1).mean()),
        'response_batch_log_scores': scores[tested].mean(axis=0).tolist(),
        'predictive_moment_band_95_observed_coverage': (abs(standardized[tested]) <= 1.96).mean(axis=0).tolist(),
        'predictive_standardized_residual_rms': np.sqrt((standardized[tested]**2).mean(axis=0)).tolist()}
    row['calibration_profile'] = {'head_order': ['f', 'g', 'eta_g', 'eta_f'],
        'observed_response': response.tolist(), 'standardized_residual': standardized.tolist(),
        'response_batch_log_scores': scores.tolist(),
        'diagnostic_points': [{'candidate': j, 'g_mean': float(mean[j, 1]), 'g_observed': float(response[j, 1]),
            'g_standardized_residual': float(standardized[j, 1]),
            'g_predicted_moment_band_95_contains_observation': bool(abs(standardized[j, 1]) <= 1.96)}
            for j in (27, 63)]}


def source_check(spec, inputs, fits):
    started = time.process_time(); data = json.loads((ROOT/spec['source_data']).read_text()); rows = []
    for i in spec['heldout_tasks']:
        model = JointResponseMixture(spec['arms'], components(inputs, fits, np.asarray(inputs['heldout_public_psi'][str(i)])))
        task = data['tasks'][i]
        for j in (0, 1):
            value = task['Y_replicates'][j][0]; model.update_raw(j, [value[0], value[1]-task['tau']])
        for name, (f, g) in zip(spec['arms'], model.components):
            fm, fv = f.predict(); gm, gv = g.predict()
            rows.append({'task_id': task['task_id'], 'task_index': i, 'arm': name,
                'mean': np.column_stack([fm[:, 0], gm[:, 0], gm[:, 1], fm[:, 1]]).tolist(),
                'variance': np.column_stack([fv[:, 0], gv[:, 0], gv[:, 1], fv[:, 1]]).tolist(),
                'local_joint_components': [[[{'weight': w, 'mean': m.tolist(), 'covariance': c.tolist()}
                    for w, m, c in head.local_components(j)] for head in (f, g)] for j in range(len(fm))]})
    out = ROOT/spec['output']; save(out/'predictions.json', {'execution': spec, 'rows': rows})
    public = json.loads((ROOT/spec['source_data']).with_name('frozen_inputs.json').read_text())
    for row in rows:
        score_row(row, training_view(data, [row['task_index']], public['plan']['task_distribution']['L'])[0],
                  spec['replicates_per_site'])
    keys = list(rows[0]['scores'])
    summary = {a: {k: np.mean([r['scores'][k] for r in rows if r['arm'] == a], axis=0).tolist()
                   for k in keys} for a in spec['arms']}
    old = json.loads((ROOT/spec['previous_source_results']).read_text()); comparisons = []
    for row in (r for r in rows if r['arm'] == 'joint_common_structural'):
        before = next(r for r in old['rows'] if r['task_id'] == row['task_id'] and r['arm'] == row['arm'])['scores']
        after = row['scores']; points = row['calibration_profile']['diagnostic_points']
        checks = {'g_mse_improved': after['observed_head_mse'][1] < before['observed_head_mse'][1],
            'joint_density_improved': after['joint_batch_log_score'] > before['joint_batch_log_score'],
            'constraint_density_improved': after['response_batch_log_scores'][1] > before['response_batch_log_scores'][1],
            'reached_points_within_predictive_moment_bands': all(p['g_predicted_moment_band_95_contains_observation'] for p in points),
            'reached_points_g_order_correct': (points[0]['g_mean']-points[1]['g_mean'])*(points[0]['g_observed']-points[1]['g_observed']) > 0}
        comparisons.append({'task_id': row['task_id'], 'checks': checks, 'before': before, 'after': after})
    result = {'execution': spec, 'summary': summary, 'source_gate': {'passed': all(all(c['checks'].values()) for c in comparisons),
        'tasks': comparisons, 'interpretation': 'opened development holdouts; predictive moment bands are not exact credible intervals'},
        'rows': [{k: r[k] for k in ('task_id', 'task_index', 'arm', 'scores')} for r in rows],
        'source_fit_cpu_seconds': fits['cpu_seconds'], 'prediction_scoring_cpu_seconds': time.process_time()-started,
        'cost': {'new_source_calls': 0, 'new_target_calls': 0, 'new_verification_calls': 0,
                 'historical_source_response_calls_per_arm': 288,
                 'new_native_source_head_fits': 24+sum('continued_from_optimizer_iterations' in h['joint_structural'] for h in fits['heads'])}}
    save(out/'calibration_profiles.json', {'rows': [{k: v for k, v in r.items() if k != 'local_joint_components'} for r in rows]})
    save(out/'results.json', result)
    print(json.dumps({'summary': summary, 'source_gate': result['source_gate']}), flush=True)


def run():
    spec = json.loads(EXECUTION.read_text()); out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    path = out/'inputs.json'
    inputs = json.loads(path.read_text()) if path.exists() else prepare(spec)
    fits = fit(spec, inputs)
    source_check(spec, inputs, fits)


if __name__ == '__main__':
    run()
