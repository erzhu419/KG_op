"""Compare public source-site coverage using cached source-task holdouts."""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.run_legacy_joint_population_matched_baselines import paired_fields
from performance.run_legacy_joint_population_source_budget import encode_subset, restore_joint
from performance.run_legacy_related_inventory_source import problem_for, save, training_view
from representation.joint_function_risk import JointFunctionRiskPosterior, joint_batch_log_score
from representation.joint_response_mixture import JointResponseMixture
from representation.legacy_observable_prior import legacy_exposure_descriptors
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis, fit_orthogonal_function_prior
from representation.source_function_prior import head_batches
from representation.source_joint_population import fit_joint_source_population, same_source_full_residual_prior
from representation.source_kernel_population import fit_kernel_population, predict_kernel_population
from representation.source_known_noise_gp import fit_known_noise_gp


def run():
    spec = json.loads((ROOT/'performance/manifests/legacy_joint_population_public_design_source_check_v1.json').read_text())
    data = json.loads((ROOT/spec['source_data']).read_text())
    public = json.loads((ROOT/spec['source_data']).with_name('frozen_inputs.json').read_text())
    original = json.loads((ROOT/spec['original_inputs']).read_text())
    plan = public['plan']; ids = spec['training_tasks']; sites = spec['source_sites']
    bank = [tuple(p) for p in public['bank']]
    # These providers expose only supported public state, never outcomes.
    descriptors = {i: legacy_exposure_descriptors(no_outcomes(problem_for(plan, task)), bank, 50)
                   for i, task in enumerate(public['tasks'])}
    prefix_coordinate, prefix_tasks, prefix_center, prefix_scale = encode_subset(
        data, ids, spec['source_site_count'], descriptors, plan['task_distribution']['L'])
    for old, rebuilt in zip(original['source_tasks'], prefix_tasks):
        np.testing.assert_array_equal(old['psi'], rebuilt['psi'])
        np.testing.assert_array_equal(old['response'], rebuilt['response'])
    # Select raw labels before any statistics, encoder or source fit.
    subset = {'bank': [data['bank'][j] for j in sites], 'tasks': [
        {'task_id': data['tasks'][i]['task_id'], 'tau': data['tasks'][i]['tau'],
         'Y_replicates': [data['tasks'][i]['Y_replicates'][j] for j in sites]} for i in ids]}
    selected_descriptors = {k: descriptors[i][sites] for k, i in enumerate(ids)}
    coordinate, tasks, center, scale = encode_subset(subset, list(range(len(ids))), len(sites),
                                selected_descriptors, plan['task_distribution']['L'])
    basis = OrthogonalLowOrderBasis(coordinate.transform(descriptors[ids[0]]))
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    source_path = out/'source_fit.json'
    started = time.process_time()
    fit = json.loads(source_path.read_text()) if source_path.exists() else {'heads': [], 'cpu_seconds': 0.}
    for h in range(len(fit['heads']), 4):
        batches, noises = head_batches(tasks, h), [t['noise'][:, h] for t in tasks]
        regularizer = fit_orthogonal_function_prior(batches, noises, basis, max_iterations=spec['source_max_iterations'])
        joint = fit_joint_source_population(regularizer, batches, noises, max_iterations=spec['source_max_iterations'])
        anchor = fit_known_noise_gp(tasks, h, spec['generic_anchor_max_iterations'])
        gp = fit_kernel_population(tasks, h, anchor, optimizer=spec['generic_population_optimizer'])
        fit['heads'].append({'regularizer': regularizer.diagnostics, 'joint': joint.diagnostics,
                             'anchor': anchor, 'gp': gp})
        save(source_path, {**fit, 'cpu_seconds': fit['cpu_seconds']+time.process_time()-started})
        print('public-design source head', h, 'convergence',
              [d['optimizer_converged'] for d in fit['heads'][-1].values()], flush=True)
    fit['cpu_seconds'] += time.process_time()-started
    save(source_path, fit)
    if not all(d['optimizer_converged'] for r in fit['heads'] for d in r.values()):
        raise RuntimeError('source optimizer completion required before heldout scoring')
    prefix_basis = OrthogonalLowOrderBasis(prefix_coordinate.transform(descriptors[ids[0]]))
    prefix_fit = json.loads((ROOT/spec['original_public_reference_fit']).read_text())
    prefix_gp = json.loads((ROOT/spec['original_gp_fit']).read_text())
    designs = [
        ('first16', prefix_coordinate, prefix_tasks, prefix_center, prefix_scale,
         [restore_joint(prefix_basis, d) for d in prefix_fit['heads']], prefix_gp['heads']),
        ('public_maxmin16', coordinate, tasks, center, scale,
         [restore_joint(basis, d['joint']) for d in fit['heads']], [d['gp'] for d in fit['heads']])]
    started = time.process_time(); rows = []
    for heldout in spec['heldout_tasks']:
        for design, encoder, training, mu, sd, structured, gp in designs:
            psi = encoder.transform(descriptors[heldout])
            full = [same_source_full_residual_prior(p) for p in structured]
            components = [(JointFunctionRiskPosterior([p[0]], [p[3]], psi),
                           JointFunctionRiskPosterior([p[1]], [p[2]], psi)) for p in [structured, full]]
            components.append(paired_fields([predict_kernel_population(training[0]['features'],
                                            (psi-mu)/sd, h) for h in gp]))
            model = JointResponseMixture(spec['arms'], components)
            task = data['tasks'][heldout]
            for j in [0, 1]:
                v = task['Y_replicates'][j][0]
                model.update_raw(j, [v[0], v[1]-task['tau']])
            for name, (f, g) in zip(spec['arms'], model.components):
                fm, fv = f.predict(); gm, gv = g.predict()
                rows.append({'task_id': task['task_id'], 'task_index': heldout, 'design': design, 'arm': name,
                    'mean': np.column_stack([fm[:, 0], gm[:, 0], gm[:, 1], fm[:, 1]]).tolist(),
                    'variance': np.column_stack([fv[:, 0], gv[:, 0], gv[:, 1], fv[:, 1]]).tolist(),
                    'local_joint_components': [[f.local_components(j), g.local_components(j)] for j in range(len(bank))]})
    # Save only local joint distributions; no need to retain full covariance.
    for row in rows:
        row['local_joint_components'] = [[[{'weight': w, 'mean': m.tolist(), 'covariance': c.tolist()}
                                            for w, m, c in head] for head in point]
                                          for point in row['local_joint_components']]
    save(out/'predictions.json', {'execution': spec, 'head_order': ['f', 'g', 'eta_g', 'eta_f'], 'rows': rows})
    # Previously opened source tasks; unobserved labels enter only after freezing predictions.
    labels = {i: training_view(data, [i], plan['task_distribution']['L'])[0] for i in spec['heldout_tasks']}
    for row in rows:
        response = labels[row['task_index']]['response']
        error = np.asarray(row['mean'])[2:]-response[2:]
        log_scores = []
        for j in range(2, len(bank)):
            scores = []
            for h, (a, b) in enumerate([(0, 3), (1, 2)]):
                components = [(d['weight'], np.asarray(d['mean']), np.asarray(d['covariance']))
                              for d in row['local_joint_components'][j][h]]
                scores.append(joint_batch_log_score(components, response[j, a], response[j, b], spec['replicates_per_source_site']))
            log_scores.append(scores)
        row['scores'] = {'observed_head_mse': (error**2).mean(axis=0).tolist(),
                         'joint_batch_log_score': float(np.asarray(log_scores).sum(axis=1).mean()),
                         'response_batch_log_scores': np.mean(log_scores, axis=0).tolist()}
    summary = {design: {arm: {
        'observed_head_mse': np.mean([r['scores']['observed_head_mse'] for r in rows if r['design'] == design and r['arm'] == arm], axis=0).tolist(),
        'joint_batch_log_score': float(np.mean([r['scores']['joint_batch_log_score'] for r in rows if r['design'] == design and r['arm'] == arm]))}
        for arm in spec['arms']} for design in spec['designs']}
    # Forecasts already live in predictions.json; results retain small scores.
    score_rows = [{k: r[k] for k in ['task_id', 'task_index', 'design', 'arm', 'scores']} for r in rows]
    result = {'execution': spec, 'head_order': ['f', 'g', 'eta_g', 'eta_f'], 'rows': score_rows, 'summary': summary,
        'cost': {'new_source_calls': 0, 'new_target_calls': 0, 'new_verification_calls': 0,
                 'source_response_archive_calls_per_design': len(ids)*len(sites)*3,
                 'cached_logical_adaptation_records_per_arm_design': 2*len(spec['heldout_tasks']),
                 'new_native_source_head_fits': 16},
        'source_fit_cpu_seconds': fit['cpu_seconds'], 'prediction_scoring_cpu_seconds': time.process_time()-started}
    save(out/'results.json', result)
    print(json.dumps({'summary': summary, 'cpu_seconds': result['prediction_scoring_cpu_seconds']}), flush=True)


if __name__ == '__main__':
    run()
