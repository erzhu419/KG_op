"""Risk-smoothed source mean noise with whole-source-task exclusion."""

import copy
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_covered_complete_mean_uncertainty import fit_means
from performance.check_legacy_diverse_source_design import encode_diverse_source
from performance.check_legacy_public_mean_risk_uncertainty import read, forecast, score_source
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.run_legacy_joint_population_matched_baselines import paired_fields
from performance.run_legacy_joint_population_source_budget import restore_joint
from performance.run_legacy_related_inventory_source import problem_for, save, training_view
from representation.joint_response_mixture import JointResponseMixture
from representation.legacy_observable_prior import legacy_exposure_descriptors
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.public_control_mean import source_prior_moments
from representation.variance_only_moments import variance_only_moments

EXECUTION = 'performance/manifests/legacy_joint_population_crossfit_variance_noise_v1.json'


def task_noise(data, public, source, excluded, full, branch_inputs, risk_fit, descriptors, count):
    keep = [(i, s) for i, s in zip(source['training_tasks'], source['source_site_indices_by_task']) if i != excluded]
    coordinate, _, center, scale = encode_diverse_source(data, [i for i, _ in keep], [s for _, s in keep],
        descriptors, public['plan']['task_distribution']['L'])
    # A mismatched saved map would invalidate the claim that current-task labels were excluded.
    np.testing.assert_array_equal(center, branch_inputs['risk_coordinate_center'])
    np.testing.assert_array_equal(scale, branch_inputs['risk_coordinate_scale'])
    np.testing.assert_array_equal(coordinate.transform(descriptors[source['training_tasks'][0]]), branch_inputs['public_risk_reference'])
    position = source['training_tasks'].index(excluded); sites = source['source_site_indices_by_task'][position]
    psi = coordinate.transform(descriptors[excluded][sites]); basis = OrthogonalLowOrderBasis(branch_inputs['public_risk_reference'])
    original = np.asarray(full['source_tasks'][position]['noise']); corrected = original.copy(); rows = []
    for mean_head, risk_head in ((0, 1), (1, 0)):
        prior_mean, covariance = source_prior_moments(restore_joint(basis, risk_fit['heads'][risk_head]['joint_structural']), psi)
        for k, site in enumerate(sites):
            result = variance_only_moments(float(prior_mean[k]), float(covariance[k, k]), float(count*original[k, mean_head]), count)
            corrected[k, mean_head] = result['expected_episode_variance']/count
            rows.append({'source_task': excluded, 'candidate': site, 'mean_head': mean_head,
                'original_mean_noise': float(original[k, mean_head]), 'corrected_mean_noise': float(corrected[k, mean_head]),
                'risk_prior_mean': float(prior_mean[k]), 'risk_prior_variance': float(covariance[k, k]), **result})
    return corrected, rows


def risk_models(group, means, mean_psi, risk_queries, arms):
    basis = OrthogonalLowOrderBasis(group[0][0]['public_mean_reference'])
    moments = {a: [source_prior_moments(restore_joint(basis, h[a]), mean_psi) for h in means['heads']] for a in arms}
    fields = {a: [] for a in arms}
    for (inp, _, risk), psi in zip(group, risk_queries):
        rb = OrthogonalLowOrderBasis(inp['public_risk_reference'])
        eta = [source_prior_moments(restore_joint(rb, h['joint_structural']), psi) for h in risk['heads']]
        for a in arms:
            fields[a].append(paired_fields(moments[a]+eta))
    names = ['full144']+[f'drop{i}' for i in range(6)]
    return [(a, JointResponseMixture(names, fields[a])) for a in arms]


def run():
    started = time.process_time(); spec = read(EXECUTION); source = read(spec['source_execution'])
    data = read(source['source_data']); public = read(Path(source['source_data']).with_name('frozen_inputs.json'))
    descriptors = {i: legacy_exposure_descriptors(no_outcomes(problem_for(public['plan'], t)),
        [tuple(p) for p in public['bank']], 50) for i, t in enumerate(public['tasks'])}
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True); groups, views, estimates = {}, {}, []
    for name in spec['layouts']:
        folder = Path(source['output'])/f'layout{name}'
        group = [(read(folder/'inputs.json'), read(folder/'source_fit.json'), read(folder/'risk_source_fit.json'))]
        group += [(read(folder/f'task{i}'/'inputs.json'), read(folder/f'task{i}'/'gp_source_fit.json'),
                   read(folder/f'task{i}'/'risk_source_fit.json')) for i in source['training_tasks']]
        full = group[0][0]; view = copy.deepcopy(full); local_source = full['execution']
        for position, (i, (inp, _, risk)) in enumerate(zip(source['training_tasks'], group[1:])):
            noise, local = task_noise(data, public, local_source, i, full, inp, risk, descriptors, spec['replicates'])
            view['source_tasks'][position]['noise'] = noise.tolist(); estimates.extend(dict(r, layout=name) for r in local)
        groups[name], views[name] = group, view
    save(out/'noise_estimates.json', {'execution': spec, 'rows': estimates})
    # Freeze BOTH complete corrected input views before any fitted mean parameter changes.
    save(out/'inputs.json', {'execution': spec, 'views': views})
    fits = {name: fit_means(view['execution'], view, out/f'layout{name}') for name, view in views.items()}
    for name, view in views.items():
        save(out/f'layout{name}'/'inputs.json', view)
    rows, jobs = [], {}
    for name, group in groups.items():
        for i in source['heldout_tasks']:
            task = data['tasks'][i]; query = {'task_id': task['task_id'], 'task_index': i, 'kind': 'source',
                'tau': task['tau'], 'candidate_indices': [0, 1], 'paired_observations': [task['Y_replicates'][j][0] for j in (0, 1)]}
            for a, model in risk_models(group, fits[name], group[0][0]['heldout_mean_psi'][str(i)],
                    [g[0]['heldout_risk_psi'][str(i)] for g in group], spec['mean_arms']):
                prior = forecast(query, a, 'crossfit_variance_risk_mixture', model)
                for j, value in zip(query['candidate_indices'], query['paired_observations']):
                    model.update_raw(j, [value[0], value[1]-query['tau']])
                row = forecast(query, a, 'crossfit_variance_risk_mixture', model)
                row.update(layout=name, prior=prior); rows.append(row); jobs[(name, i, a)] = model
    save(out/'source_predictions.json', {'execution': spec, 'rows': rows})
    references = read(spec['reference_profiles'])['rows']; contrasts = []
    labels = {i: training_view(data, [i], public['plan']['task_distribution']['L'])[0] for i in source['heldout_tasks']}
    for r in rows:
        i, a, name = r['task_index'], r['mean_arm'], r['layout']; score_source(r, labels[i], jobs[(name, i, a)], 3)
        old = next(z for z in references if z['layout'] == name and z['task_index'] == i and z['mean_arm'] == a)
        c = {'layout': name, 'task_id': r['task_id'], 'mean_arm': a,
            'g_mse_difference': r['scores']['observed_head_mse'][1]-old['scores']['observed_head_mse'][1],
            'joint_batch_score_difference': r['scores']['joint_batch_log_score']-old['scores']['joint_batch_log_score'],
            'g_batch_score_difference': r['scores']['response_batch_log_scores'][1]-old['scores']['response_batch_log_scores'][1]}
        c['passed'] = c['g_mse_difference'] <= 0 and c['joint_batch_score_difference'] >= 0 and c['g_batch_score_difference'] > 0
        contrasts.append(c)
    gates = {a: all(c['passed'] for c in contrasts if c['mean_arm'] == a) for a in spec['mean_arms']}
    summary = {name: {a: {k: np.mean([r['scores'][k] for r in rows if r['layout'] == name and r['mean_arm'] == a], axis=0).tolist()
        for k in rows[0]['scores']} for a in spec['mean_arms']} for name in spec['layouts']}
    ratios = [r['corrected_mean_noise']/r['original_mean_noise'] for r in estimates]
    save(out/'calibration_profiles.json', {'execution': spec, 'rows': rows})
    result = {'execution': spec, 'summary': summary, 'contrasts': contrasts, 'source_gates': gates,
        'corrected_to_original_noise_ratio_quantiles': np.quantile(ratios, [0, .5, .9, 1]).tolist(),
        'maximum_variance_integral_relative_error': max(r['integration_relative_error'] for r in estimates),
        'native_mean_fits': 16+sum('continued_from_optimizer_iterations' in h['joint_structural'] for f in fits.values() for h in f['heads']),
        'source_mean_fit_cpu_seconds': sum(f['cpu_seconds'] for f in fits.values()),
        'cost': {'new_source_risk_fits': 0, 'new_simulator_calls': 0, 'new_verification_calls': 0, 'new_terminal_integrations': 0},
        'cpu_seconds': time.process_time()-started}
    save(out/'results.json', result); print({'summary': summary, 'source_gates': gates, 'contrasts': contrasts,
        'noise_ratios': result['corrected_to_original_noise_ratio_quantiles'], 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
