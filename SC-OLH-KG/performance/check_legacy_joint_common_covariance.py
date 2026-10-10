"""Source-only joint common/residual covariance attribution on cached holdouts."""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.run_legacy_joint_population_matched_baselines import paired_fields
from performance.run_legacy_joint_population_source_budget import encode_subset, restore_joint
from performance.run_legacy_related_inventory_source import problem_for, save, training_view
from representation.joint_function_risk import JointFunctionRiskPosterior, joint_batch_log_score
from representation.joint_response_mixture import JointResponseMixture
from representation.legacy_observable_prior import legacy_exposure_descriptors
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis, spectral_weights
from representation.source_function_prior import SourceFunctionPrior, head_batches
from representation.source_joint_population import (JointCovariancePopulationLikelihood,
    fit_joint_common_source_population, same_source_full_residual_prior)
from representation.source_kernel_population import (JointCommonKernelLikelihood,
    fit_joint_common_kernel_population, predict_joint_common_kernel_population)


def restore_regularizer(basis, diagnostic, tasks, head):
    """Reconstruct the recorded diagonal law, including its GLS intercept scale."""
    groups = np.asarray(diagnostic['source_group_variances'])
    weights = spectral_weights(basis, diagnostic['spectral_exponent'])[0]
    tail = .01*diagnostic['source_signal_scale']/basis.feature_dim
    diagonal = groups[basis.group_index]*weights+tail
    intercepts = []
    for task in tasks:
        z = basis.transform(task['psi'])
        covariance = (z*diagonal) @ z.T+np.diag(task['noise'][:, head]+diagnostic['discrepancy_variance'])
        precision_one = cho_solve(cho_factor(covariance, lower=True), np.ones(len(z)))
        intercepts.append(float(precision_one @ task['response'][:, head]/precision_one.sum()))
    intercept_variance = max(float(np.var(intercepts, ddof=1)), .25)
    return SourceFunctionPrior(basis, np.asarray(diagnostic['coefficient_mean']),
        np.diag(np.r_[intercept_variance, diagonal]), diagnostic['discrepancy_variance'], dict(diagnostic))


def run():
    spec = json.loads((ROOT/'performance/manifests/legacy_joint_population_joint_common_covariance_execution_v1.json').read_text())
    data = json.loads((ROOT/spec['source_data']).read_text())
    public = json.loads((ROOT/spec['source_data']).with_name('frozen_inputs.json').read_text())
    old_fit = json.loads((ROOT/spec['previous_fit']).read_text())
    plan = public['plan']; ids, sites = spec['training_tasks'], spec['source_sites']
    bank = [tuple(p) for p in public['bank']]
    descriptors = {i: legacy_exposure_descriptors(no_outcomes(problem_for(plan, task)), bank, 50)
                   for i, task in enumerate(public['tasks'])}
    subset = {'bank': [data['bank'][j] for j in sites], 'tasks': [
        {'task_id': data['tasks'][i]['task_id'], 'tau': data['tasks'][i]['tau'],
         'Y_replicates': [data['tasks'][i]['Y_replicates'][j] for j in sites]} for i in ids]}
    coordinate, tasks, center, scale = encode_subset(subset, list(range(len(ids))), len(sites),
        {k: descriptors[i][sites] for k, i in enumerate(ids)}, plan['task_distribution']['L'])
    basis = OrthogonalLowOrderBasis(coordinate.transform(descriptors[ids[0]]))
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    source_path = out/'source_fit.json'
    fit = json.loads(source_path.read_text()) if source_path.exists() else {'heads': [], 'cpu_seconds': 0.}
    started = time.process_time()
    for h in range(len(fit['heads']), 4):
        old = old_fit['heads'][h]
        regularizer = restore_regularizer(basis, old['regularizer'], tasks, h)
        batches, noises = head_batches(tasks, h), [t['noise'][:, h] for t in tasks]
        structural_likelihood = JointCovariancePopulationLikelihood(regularizer, batches, noises)
        initial = structural_likelihood.initial_parameters(old['joint'])
        np.testing.assert_allclose(structural_likelihood.evaluate(initial)[0],
                                   old['joint']['source_negative_log_likelihood_per_label'], atol=1e-8, rtol=1e-8)
        anchor = old['gp']['anchor']; normalizer = anchor['normalization']
        responses = [(t['response'][:, h]-normalizer['mean'])/normalizer['scale'] for t in tasks]
        errors = [t['noise'][:, h]/normalizer['scale']**2 for t in tasks]
        gp_likelihood = JointCommonKernelLikelihood(tasks[0]['features'], responses, errors, anchor)
        gp_initial = np.r_[anchor['signal_variance'], old['gp']['residual_parameters']]
        np.testing.assert_allclose(gp_likelihood.evaluate(gp_initial)[0],
                                   old['gp']['source_negative_log_likelihood_per_label'], atol=1e-8, rtol=1e-8)
        joint = fit_joint_common_source_population(regularizer, batches, noises, old['joint'], optimizer=spec['optimizer'])
        gp = fit_joint_common_kernel_population(tasks, h, old['gp'], optimizer=spec['optimizer'])
        fit['heads'].append({'structural': joint.diagnostics, 'gp': gp})
        save(source_path, {**fit, 'cpu_seconds': fit['cpu_seconds']+time.process_time()-started})
        print('joint-common source head', h, 'convergence', joint.diagnostics['optimizer_converged'], gp['optimizer_converged'],
              'objective changes', joint.diagnostics['source_negative_log_likelihood_per_label']-joint.diagnostics['initial_source_negative_log_likelihood_per_label'],
              gp['source_negative_log_likelihood_per_label']-gp['initial_source_negative_log_likelihood_per_label'], flush=True)
    fit['cpu_seconds'] += time.process_time()-started
    save(source_path, fit)
    if not all(d['optimizer_converged'] for r in fit['heads'] for d in r.values()):
        raise RuntimeError('source optimization completion required before heldout forecasts')
    started = time.process_time(); rows = []
    structured = [restore_joint(basis, d['structural']) for d in fit['heads']]
    full = [same_source_full_residual_prior(p) for p in structured]
    for heldout in spec['heldout_tasks']:
        psi = coordinate.transform(descriptors[heldout])
        components = [(JointFunctionRiskPosterior([p[0]], [p[3]], psi),
                       JointFunctionRiskPosterior([p[1]], [p[2]], psi)) for p in [structured, full]]
        components.append(paired_fields([predict_joint_common_kernel_population(tasks[0]['features'],
                                        (psi-center)/scale, d['gp']) for d in fit['heads']]))
        model = JointResponseMixture(spec['arms'], components)
        task = data['tasks'][heldout]
        for j in [0, 1]:
            value = task['Y_replicates'][j][0]
            model.update_raw(j, [value[0], value[1]-task['tau']])
        for name, (f, g) in zip(spec['arms'], model.components):
            fm, fv = f.predict(); gm, gv = g.predict()
            local = [[[{'weight': w, 'mean': m.tolist(), 'covariance': c.tolist()}
                       for w, m, c in head.local_components(j)] for head in [f, g]] for j in range(len(bank))]
            rows.append({'task_id': task['task_id'], 'task_index': heldout, 'arm': name,
                'mean': np.column_stack([fm[:, 0], gm[:, 0], gm[:, 1], fm[:, 1]]).tolist(),
                'variance': np.column_stack([fv[:, 0], gv[:, 0], gv[:, 1], fv[:, 1]]).tolist(),
                'local_joint_components': local})
    save(out/'predictions.json', {'execution': spec, 'rows': rows})
    labels = {i: training_view(data, [i], plan['task_distribution']['L'])[0] for i in spec['heldout_tasks']}
    for row in rows:
        response = labels[row['task_index']]['response']
        error = np.asarray(row['mean'])[2:]-response[2:]
        scores = []
        for j in range(2, len(bank)):
            point = []
            for h, (a, b) in enumerate([(0, 3), (1, 2)]):
                local = [(c['weight'], np.asarray(c['mean']), np.asarray(c['covariance']))
                         for c in row['local_joint_components'][j][h]]
                point.append(joint_batch_log_score(local, response[j, a], response[j, b], spec['replicates_per_source_site']))
            scores.append(point)
        row['scores'] = {'observed_head_mse': (error**2).mean(axis=0).tolist(),
                         'joint_batch_log_score': float(np.asarray(scores).sum(axis=1).mean()),
                         'response_batch_log_scores': np.mean(scores, axis=0).tolist()}
    summary = {arm: {'observed_head_mse': np.mean([r['scores']['observed_head_mse'] for r in rows if r['arm'] == arm], axis=0).tolist(),
        'joint_batch_log_score': float(np.mean([r['scores']['joint_batch_log_score'] for r in rows if r['arm'] == arm])),
        'response_batch_log_scores': np.mean([r['scores']['response_batch_log_scores'] for r in rows if r['arm'] == arm], axis=0).tolist()}
        for arm in spec['arms']}
    old_results = json.loads((ROOT/spec['previous_results']).read_text())
    contrasts = []
    references = ['joint_population_structural', 'joint_population_full', 'joint_population_rbf_shared_raw']
    for arm, reference in zip(spec['arms'], references):
        paired = []
        for row in [r for r in rows if r['arm'] == arm]:
            old = next(r for r in old_results['rows'] if r['task_id'] == row['task_id']
                       and r['design'] == 'public_maxmin16' and r['arm'] == reference)
            paired.append({'task_id': row['task_id'],
                'g_observed_mse_change': row['scores']['observed_head_mse'][1]-old['scores']['observed_head_mse'][1],
                'joint_batch_log_score_change': row['scores']['joint_batch_log_score']-old['scores']['joint_batch_log_score'],
                'constraint_batch_log_score_change': row['scores']['response_batch_log_scores'][1]-old['scores']['response_batch_log_scores'][1]})
        contrasts.append({'arm': arm, 'fixed_common_reference': reference, 'paired_tasks': paired})
    result = {'execution': spec, 'summary': summary, 'paired_comparisons': contrasts,
        'rows': [{k: r[k] for k in ['task_id', 'task_index', 'arm', 'scores']} for r in rows],
        'cost': {'new_source_calls': 0, 'new_target_calls': 0, 'new_verification_calls': 0,
                 'source_response_archive_calls': len(ids)*len(sites)*3, 'new_native_source_head_fits': 8},
        'source_fit_cpu_seconds': fit['cpu_seconds'], 'prediction_scoring_cpu_seconds': time.process_time()-started}
    save(out/'results.json', result)
    print(json.dumps({'summary': summary, 'cpu_seconds': result['prediction_scoring_cpu_seconds']}), flush=True)


if __name__ == '__main__':
    run()
