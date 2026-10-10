"""Matched predictive test of cached source-risk uncertainty hypotheses."""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_diverse_source_design import encode_diverse_source
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.run_legacy_joint_population_matched_baselines import paired_fields
from performance.run_legacy_joint_population_source_budget import restore_joint
from performance.run_legacy_related_inventory_source import problem_for, save, training_view
from representation.joint_response_mixture import JointResponseMixture
from representation.legacy_observable_prior import legacy_exposure_descriptors
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.public_control_mean import source_prior_moments
from representation.source_kernel_population import predict_unequal_site_kernel_population

EXECUTION = 'performance/manifests/legacy_joint_population_public_mean_risk_uncertainty_execution_v1.json'
OLD_ARM = {'joint_structural': 'public_mean_joint_structural',
           'flat': 'public_mean_common_flat', 'joint_gp': 'public_mean_joint_rbf'}


def read(path):
    return json.loads((ROOT/path).read_text())


def prepare(spec, inputs):
    deletion = read(spec['deletion_execution']); source = read(deletion['source_execution'])
    data = read(deletion['source_data']); public = read(Path(deletion['source_data']).with_name('frozen_inputs.json'))
    target_inputs = read(spec['target_inputs']); bank = [tuple(p) for p in public['bank']]
    descriptors = {i: legacy_exposure_descriptors(no_outcomes(problem_for(public['plan'], t)), bank, 50)
                   for i, t in enumerate(public['tasks'])}
    queries = []
    for i in source['heldout_tasks']:
        task = data['tasks'][i]; provider = no_outcomes(problem_for(public['plan'], public['tasks'][i]))
        queries.append({'task_id': task['task_id'], 'task_index': i, 'kind': 'source',
            'mean_psi': inputs['heldout_mean_psi'][str(i)], 'tau': task['tau'], 'alpha': provider.alpha,
            'candidate_indices': [0, 1], 'paired_observations': [task['Y_replicates'][j][0] for j in (0, 1)]})
    queries.extend({'task_id': t['task_id'], 'task_index': i, 'kind': 'target',
        'mean_psi': t['mean_psi'], 'tau': t['tau'], 'alpha': t['alpha'],
        **t['initialization']} for i, t in enumerate(target_inputs['targets']))
    full_psi = {data['tasks'][i]['task_id']: inputs['heldout_risk_psi'][str(i)] for i in source['heldout_tasks']}
    full_psi.update({t['task_id']: t['risk_psi'] for t in target_inputs['targets']})
    members = [{'name': 'full144', 'mode': 'full', 'inputs': spec['source_inputs'],
                'fit': spec['risk_fit'], 'psi': full_psi}]
    for mode in deletion['modes']:
        for deleted in deletion['deletion_tasks']:
            branch = Path(spec['deletion_artifacts'])/mode/f'task{deleted}'
            branch_inputs = read(branch/'inputs.json')
            if mode == deletion['modes'][0]:
                psi = full_psi
            else:
                keep = [(i, s) for i, s in zip(source['training_tasks'], source['source_site_indices_by_task']) if i != deleted]
                coordinate, _, center, scale = encode_diverse_source(data, [i for i, _ in keep],
                    [s for _, s in keep], descriptors, public['plan']['task_distribution']['L'])
                # These maps must match the ones used by the cached branch laws.
                np.testing.assert_array_equal(center, branch_inputs['risk_coordinate_center'])
                np.testing.assert_array_equal(scale, branch_inputs['risk_coordinate_scale'])
                np.testing.assert_array_equal(coordinate.transform(descriptors[source['training_tasks'][0]]),
                                              branch_inputs['public_risk_reference'])
                psi = {data['tasks'][i]['task_id']: coordinate.transform(descriptors[i]).tolist()
                       for i in source['heldout_tasks']}
                for t in target_inputs['targets']:
                    # Reuse the public target descriptors from the saved policy map.
                    from performance.run_legacy_joint_population_independent_pilot import target_problem
                    provider = no_outcomes(target_problem(target_inputs['target_plan'], t['config']))
                    desc = legacy_exposure_descriptors(provider, [tuple(p) for p in target_inputs['bank']], 50)
                    psi[t['task_id']] = coordinate.transform(desc).tolist()
            members.append({'name': f'drop{deleted}', 'mode': mode, 'inputs': str(branch/'inputs.json'),
                            'fit': str(branch/'risk_source_fit.json'), 'psi': psi})
    save(ROOT/spec['output']/'inputs.json', {'execution': spec, 'queries': queries, 'risk_members': members})
    return queries, members


def models_at_query(spec, inputs, means, members, query):
    mean_basis = OrthogonalLowOrderBasis(inputs['public_mean_reference']); psi = np.asarray(query['mean_psi'])
    mean_moments = {a: ([predict_unequal_site_kernel_population(psi, h[a]) for h in means['heads']]
        if a == 'joint_gp' else [source_prior_moments(restore_joint(mean_basis, h[a]), psi) for h in means['heads']])
        for a in spec['mean_arms']}
    eta = []
    for member in members:
        branch_inputs, fit = read(member['inputs']), read(member['fit'])
        basis = OrthogonalLowOrderBasis(branch_inputs['public_risk_reference'])
        eta.append([source_prior_moments(restore_joint(basis, h['joint_structural']),
                    np.asarray(member['psi'][query['task_id']])) for h in fit['heads']])
    models = []
    for a in spec['mean_arms']:
        for case in spec['risk_cases']:
            selected = ([0] if case == spec['risk_cases'][0] else [0]+[j for j, m in enumerate(members)
                if m['mode'] == spec['risk_modes'][spec['risk_cases'].index(case)-1]])
            model = JointResponseMixture([members[j]['name'] for j in selected],
                [paired_fields(mean_moments[a]+eta[j]) for j in selected])
            for j, value in zip(query['candidate_indices'], query['paired_observations']):
                model.update_raw(j, [value[0], value[1]-query['tau']])
            models.append((a, case, model))
    return models


def forecast(query, a, case, model):
    mean, covariance = model.predict_joint(); n = len(mean)
    variance = np.diag(covariance.reshape(4*n, 4*n)).reshape(n, 4)
    expected_episode_variance = np.zeros((n, 2))
    for weight, pair in zip(model.weights, model.components):
        for h, field in enumerate(pair):
            for j in range(n):
                expected_episode_variance[j, h] += weight*sum(w*np.exp(m[1]+.5*c[1, 1])
                    for w, m, c in field.local_components(j))
    return {'task_id': query['task_id'], 'task_index': query['task_index'], 'kind': query['kind'],
        'mean_arm': a, 'risk_case': case, 'head_order': ['f', 'g', 'eta_g', 'eta_f'],
        'mean': mean[:, [0, 2, 3, 1]].tolist(), 'variance': variance[:, [0, 2, 3, 1]].tolist(),
        'expected_episode_variance': expected_episode_variance.tolist(),
        'identity_names': model.names, 'identity_weights': model.weights.tolist(),
        'paid_predictive_log_score': float(sum(r['mixture_log_score'] for r in model.history)),
        'paid_history': list(model.history)}


def baseline_check(spec, rows):
    references = {'source': read(spec['reference_source_predictions']), 'target': read(spec['reference_target_predictions'])}
    for r in (r for r in rows if r['risk_case'] == spec['risk_cases'][0]):
        old = next(x for x in references[r['kind']]['rows'] if x['task_id'] == r['task_id'] and x['arm'] == OLD_ARM[r['mean_arm']])
        for k in ('mean', 'variance'):
            np.testing.assert_allclose(r[k], old[k], rtol=1e-8, atol=1e-9)


def score_source(row, labels, model, replicates):
    mean, variance = np.asarray(row['mean']), np.asarray(row['variance'])
    error = mean-np.asarray(labels['response']); observation_variance = variance.copy()
    observation_variance[:, :2] += np.asarray(row['expected_episode_variance'])/replicates
    observation_variance[:, 2:] += labels['noise'][:, 2:]
    standardized = error/np.sqrt(observation_variance); scores = []
    for j in range(2, len(mean)):
        value = labels['response'][j]
        scores.append(model.batch_log_scores(j, value[:2], value[[3, 2]], replicates))
    row['scores'] = {'observed_head_mse': (error[2:]**2).mean(axis=0).tolist(),
        'joint_batch_log_score': float(np.mean([s['joint'] for s in scores])),
        'response_batch_log_scores': np.mean([s['heads'] for s in scores], axis=0).tolist(),
        'predictive_moment_band_95_observed_coverage': (abs(standardized[2:]) <= 1.96).mean(axis=0).tolist(),
        'predictive_standardized_residual_rms': np.sqrt((standardized[2:]**2).mean(axis=0)).tolist()}


def run():
    started = time.process_time(); spec = read(EXECUTION); out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    inputs, means = read(spec['source_inputs']), read(spec['mean_fit'])
    queries, members = prepare(spec, inputs); rows, jobs = [], {}
    for query in queries:
        for a, case, model in models_at_query(spec, inputs, means, members, query):
            row = forecast(query, a, case, model); rows.append(row)
            if query['kind'] == 'source':
                jobs[(query['task_id'], a, case)] = model
        print('risk uncertainty forecasts frozen', query['kind'], query['task_id'], flush=True)
    save(out/'source_predictions.json', {'execution': spec, 'rows': [r for r in rows if r['kind'] == 'source']})
    save(out/'target_predictions.json', {'execution': spec, 'rows': [r for r in rows if r['kind'] == 'target']})
    baseline_check(spec, rows)
    # All18 source and54 target forecasts are saved before unobserved scoring.
    deletion = read(spec['deletion_execution']); data = read(deletion['source_data'])
    public = read(Path(deletion['source_data']).with_name('frozen_inputs.json'))
    scoring = read(spec['target_scoring']); diagnostics = []
    labels = {i: training_view(data, [i], public['plan']['task_distribution']['L'])[0] for i in spec['source_holdouts']}
    for r in rows:
        if r['kind'] == 'source':
            score_source(r, labels[r['task_index']], jobs[(r['task_id'], r['mean_arm'], r['risk_case'])], 3)
        else:
            query = next(q for q in queries if q['task_id'] == r['task_id'])
            fold = next(f for f in scoring['folds'] if f['task_id'] == r['task_id'])
            truth = np.asarray(fold['scoring_truth'])[:, [0, 2, 3, 1]]
            mean, variance = np.asarray(r['mean']), np.asarray(r['variance']); error = mean-truth
            r['scores'] = {'truth_mse': (error[2:]**2).mean(axis=0).tolist(),
                'moment_interval_95_truth_coverage': (abs(error[2:]) <= 1.96*np.sqrt(variance[2:])).mean(axis=0).tolist()}
            for j in spec['diagnostic_candidates']:
                diagnostics.append({'task_id': r['task_id'], 'mean_arm': r['mean_arm'], 'risk_case': r['risk_case'],
                    'candidate': j, 'g_bias': float(error[j, 1]), 'eta_g_bias': float(error[j, 2]),
                    'mean_eta_plugin_margin': float(mean[j, 1]+norm.ppf(1-query['alpha'])*np.exp(mean[j, 2]/2)),
                    'true_chance_margin': float(truth[j, 1]+norm.ppf(1-query['alpha'])*np.exp(truth[j, 2]/2))})
    summary = {kind: {a: {case: {k: np.mean([r['scores'][k] for r in rows if r['kind'] == kind
        and r['mean_arm'] == a and r['risk_case'] == case], axis=0).tolist() for k in next(r for r in rows if r['kind'] == kind)['scores']}
        for case in spec['risk_cases']} for a in spec['mean_arms']} for kind in ('source', 'target')}
    save(out/'calibration_profiles.json', {'rows': rows})
    result = {'execution': spec, 'summary': summary, 'diagnostic_rows': diagnostics, 'baseline_reproduction_passed': True,
        'cost': {'historical_source_calls_per_arm': 144, 'new_source_fits': 0, 'source_encoder_reconstructions': 6,
            'new_simulator_calls': 0, 'new_verification_calls': 0, 'new_terminal_integrations': 0},
        'cpu_seconds': time.process_time()-started}
    save(out/'results.json', result)
    print(json.dumps({'summary': summary, 'cpu_seconds': result['cpu_seconds']}), flush=True)


if __name__ == '__main__':
    run()
