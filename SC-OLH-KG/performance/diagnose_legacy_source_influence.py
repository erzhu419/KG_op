"""Linear source influence and conditional mean-uncertainty attribution."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_joint_common_covariance import restore_regularizer
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_covered_mean_attribution import design
from performance.run_legacy_related_inventory_source import save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.source_function_prior import head_batches
from representation.source_joint_population import JointCovariancePopulationLikelihood, FlatCovariancePopulationLikelihood

EXECUTION = 'performance/manifests/legacy_joint_population_source_influence_v1.json'


def run():
    started = time.process_time(); spec = read(EXECUTION); source = read(spec['source_execution'])
    snapshots = read(spec['target_snapshot'])['snapshots']; profiles = read(spec['target_profiles'])['rows']; rows = []
    for layout in spec['layouts']:
        folder = Path(source['output'])/f'layout{layout}'; inputs = read(folder/'inputs.json'); fits = read(folder/'source_fit.json')
        tasks = [{**t, **{k: np.asarray(t[k]) for k in ('psi', 'features', 'response', 'noise')}} for t in inputs['source_tasks']]
        basis = OrthogonalLowOrderBasis(inputs['public_mean_reference'])
        for head, head_name in enumerate(('f', 'g')):
            regularizer = restore_regularizer(basis, fits['heads'][head]['regularizer'], tasks, head)
            for arm in spec['mean_arms']:
                fit = fits['heads'][head][arm]
                cls = JointCovariancePopulationLikelihood if arm == 'joint_structural' else FlatCovariancePopulationLikelihood
                likelihood = cls(regularizer, head_batches(tasks, head), [t['noise'][:, head] for t in tasks])
                value, _, state = likelihood.evaluate(np.asarray(fit['optimizer_parameters']))
                np.testing.assert_allclose(value, fit['source_negative_log_likelihood_per_label'], atol=1e-8)
                np.testing.assert_allclose(state['mean'], fit['coefficient_mean'], atol=1e-8)
                covariance = state['mean_covariance']; common_precision = np.diag(1/state['common_diagonal'])
                for frozen in (s for s in snapshots if s['layout'] == layout):
                    task_id = frozen['query']['task_id']; query = design(basis, frozen['mean_queries'][0])[spec['diagnostic_candidates']]
                    posterior_row = next(r for r in profiles if r['layout'] == layout and r['task_id'] == task_id and r['mean_arm'] == arm)
                    mean = query@state['mean']; influence = []; labels = []; noise = []; source_ids = []
                    origin = query@regularizer.mean; source_task_variance = np.zeros(len(query)); discrepancy_variance = np.zeros(len(query))
                    sample_noise_variance = np.zeros(len(query)); weights_by_task = []
                    function_groups = {'intercept': [0], **{g: (np.flatnonzero(basis.group_index == k)+1).tolist()
                        for k, g in enumerate(basis.group_names)}}
                    task_group_variance = {g: np.zeros(len(query)) for g in function_groups}
                    for position, (x, error, observation_noise) in enumerate(likelihood.blocks):
                        task_covariance = (x*state['residual_diagonal'])@x.T
                        variance = task_covariance+np.diag(observation_noise+state['discrepancy'])
                        weight = query@covariance@cho_solve(cho_factor(variance, lower=True), x).T
                        baseline = x@regularizer.mean
                        origin -= weight@baseline
                        influence.append(weight); labels.extend((error+baseline).tolist()); noise.extend(observation_noise.tolist())
                        source_ids.extend((source['training_tasks'][position], j) for j in source['layouts'][layout][position])
                        sample_noise_variance += np.sum(weight**2*observation_noise, axis=1)
                        source_task_variance += np.einsum('ij,jk,ik->i', weight, task_covariance, weight)
                        propagated = weight@x
                        for g, columns in function_groups.items():
                            task_group_variance[g] += np.sum(propagated[:, columns]**2*state['residual_diagonal'][columns], axis=1)
                        discrepancy_variance += np.sum(weight**2, axis=1)*state['discrepancy']
                        weights_by_task.append({'source_task': source['training_tasks'][position], 'absolute_weight_sum': np.abs(weight).sum(axis=1).tolist()})
                    influence = np.hstack(influence)
                    np.testing.assert_allclose(origin+influence@np.asarray(labels), mean, atol=1e-8)
                    np.testing.assert_allclose(mean, np.asarray(posterior_row['prior']['mean'])[spec['diagnostic_candidates'], head], atol=1e-8)
                    shrinkage = np.einsum('ij,jk,ik->i', query@covariance, common_precision, query@covariance)
                    common_variance = np.einsum('ij,jk,ik->i', query, covariance, query)
                    # A missing source/discrepancy component would break the posterior covariance identity.
                    np.testing.assert_allclose(sample_noise_variance+source_task_variance+discrepancy_variance+shrinkage,
                        common_variance, atol=1e-10, rtol=1e-8)
                    target_variance = np.sum(query**2*state['residual_diagonal'], axis=1)+state['discrepancy']
                    np.testing.assert_allclose(common_variance+target_variance,
                        np.asarray(posterior_row['prior']['variance'])[spec['diagnostic_candidates'], head], atol=1e-8)
                    rows.append({'layout': layout, 'task_id': task_id, 'head': head_name, 'mean_arm': arm,
                        'source_ids': source_ids, 'source_labels': labels, 'source_label_noise': noise,
                        'influence_weights': influence.tolist(), 'prior_origin': origin.tolist(), 'prior_mean': mean.tolist(),
                        'weights_by_source_task': weights_by_task,
                        'source_task_variance_by_function_group': {g: v.tolist() for g, v in task_group_variance.items()},
                        'variance_components': {'source_sample_mean_noise': sample_noise_variance.tolist(),
                            'source_task_functions': source_task_variance.tolist(), 'source_discrepancy': discrepancy_variance.tolist(),
                            'prior_shrinkage': shrinkage.tolist(), 'common_posterior': common_variance.tolist(),
                            'new_target_task_and_discrepancy': target_variance.tolist()}})
    contrasts = []
    for head in ('f', 'g'):
        for arm in spec['mean_arms']:
            for task_id in dict.fromkeys(s['query']['task_id'] for s in snapshots):
                a, b = [next(r for r in rows if r['layout'] == l and r['head'] == head and r['mean_arm'] == arm and r['task_id'] == task_id)
                    for l in spec['layouts']]
                weights, label_noise, labels = {}, {}, {}
                for sign, row in ((-1, a), (1, b)):
                    for j, key in enumerate(map(tuple, row['source_ids'])):
                        if key in labels:
                            np.testing.assert_allclose(labels[key], row['source_labels'][j], atol=1e-8)
                            np.testing.assert_allclose(label_noise[key], row['source_label_noise'][j], atol=1e-8)
                        labels[key] = row['source_labels'][j]; label_noise[key] = row['source_label_noise'][j]
                        weights[key] = weights.get(key, np.zeros(len(spec['diagnostic_candidates'])))+sign*np.asarray(row['influence_weights'])[:, j]
                shift = np.asarray(b['prior_mean'])-a['prior_mean']
                reconstructed = np.asarray(b['prior_origin'])-a['prior_origin']+sum(weights[k]*labels[k] for k in weights)
                np.testing.assert_allclose(reconstructed, shift, atol=1e-8)
                noise_sd = np.sqrt(sum(weights[k]**2*label_noise[k] for k in weights))
                contrasts.append({'task_id': task_id, 'head': head, 'mean_arm': arm,
                    'assignment_prior_mean_shift': shift.tolist(), 'fixed_parameter_sample_noise_sensitivity_sd': noise_sd.tolist()})
    result = {'execution': spec, 'rows': rows, 'assignment_contrasts': contrasts,
        'cost': {'new_source_fits': 0, 'new_simulator_calls': 0, 'new_verification_calls': 0, 'new_terminal_integrations': 0},
        'cpu_seconds': time.process_time()-started}
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True); save(out/'results.json', result)
    print({'cpu_seconds': result['cpu_seconds'], 'g_structural_assignment': [r for r in contrasts if r['head'] == 'g' and r['mean_arm'] == 'joint_structural']}, flush=True)


if __name__ == '__main__':
    run()
