"""Source-only complete f/g/eta estimation uncertainty, with cached risk laws."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_coordinates import fit as fit_mean
from performance.check_legacy_public_mean_risk_uncertainty import read, forecast, score_source
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.run_legacy_joint_population_matched_baselines import paired_fields, standardize
from performance.run_legacy_joint_population_source_budget import restore_joint
from performance.run_legacy_related_inventory_source import problem_for, save, training_view
from representation.joint_response_mixture import JointResponseMixture
from representation.legacy_function_prior import replicated_source_labels
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.public_control_mean import public_control_mean_coordinates, source_prior_moments
from representation.source_kernel_population import predict_unequal_site_kernel_population

EXECUTION = 'performance/manifests/legacy_joint_population_complete_source_uncertainty_v1.json'
CASE = 'full_plus_six_complete_source_deletions'


def mean_views(data, public, source, deleted):
    """Delete response labels before fitting mean laws and their normalizers."""
    bank = [tuple(p) for p in public['bank']]
    raw = {i: public_control_mean_coordinates(no_outcomes(problem_for(public['plan'], t)), bank)
           for i, t in enumerate(public['tasks'])}
    keep = [(i, s) for i, s in zip(source['training_tasks'], source['source_site_indices_by_task']) if i != deleted]
    labels = [replicated_source_labels(data['tasks'][i]['task_id'], np.asarray(data['bank'])[s]/
        public['plan']['task_distribution']['L'], [data['tasks'][i]['Y_replicates'][j] for j in s],
        data['tasks'][i]['tau']) for i, s in keep]
    psi, center, scale = standardize(np.vstack([raw[i][s] for i, s in keep]))
    stops = np.cumsum([len(s) for _, s in keep])[:-1]
    tasks = [{**t, 'psi': p, 'features': p} for t, p in zip(labels, np.split(psi, stops))]
    reference = (raw[source['training_tasks'][0]]-center)/scale
    basis = OrthogonalLowOrderBasis(reference)
    design = np.column_stack([np.ones(len(psi)), basis.transform(psi)])
    return {'deleted_task': deleted, 'source_tasks': [{**t, **{k: t[k].tolist()
        for k in ('psi', 'features', 'response', 'noise')}} for t in tasks],
        'public_mean_reference': reference.tolist(), 'mean_coordinate_center': center.tolist(),
        'mean_coordinate_scale': scale.tolist(), 'mean_source_design_rank': int(np.linalg.matrix_rank(design)),
        'heldout_mean_psi': {str(i): ((raw[i]-center)/scale).tolist() for i in source['heldout_tasks']}}


def run():
    started = time.process_time(); spec = read(EXECUTION); source = read(spec['source_execution'])
    data = read(spec['source_data']); public = read(Path(spec['source_data']).with_name('frozen_inputs.json'))
    frozen = read(spec['risk_prediction_inputs']); out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    members = [m for m in frozen['risk_members'] if m['mode'] in ('full', spec['risk_mode'])]
    inputs = [read(spec['full_mean_inputs'])]; means = [read(spec['full_mean_fit'])]
    for deleted in spec['deletion_tasks']:
        branch = out/f'task{deleted}'; branch.mkdir(parents=True, exist_ok=True)
        view = mean_views(data, public, source, deleted); save(branch/'inputs.json', view)
        branch_spec = dict(source, output=str(branch.relative_to(ROOT)))
        inputs.append(view); means.append(fit_mean(branch_spec, view))
        print('complete source mean fits frozen', deleted, 'rank', view['mean_source_design_rank'], flush=True)
    rows, models = [], {}; full_reference = read(spec['reference_predictions'])['rows']
    queries = [q for q in frozen['queries'] if q['kind'] == 'source']
    for query in queries:
        for a in spec['mean_arms']:
            fields = []
            for m, inp, fit in zip(members, inputs, means):
                psi = np.asarray(inp['heldout_mean_psi'][str(query['task_index'])])
                basis = OrthogonalLowOrderBasis(inp['public_mean_reference'])
                mean = ([predict_unequal_site_kernel_population(psi, h[a]) for h in fit['heads']]
                    if a == 'joint_gp' else [source_prior_moments(restore_joint(basis, h[a]), psi) for h in fit['heads']])
                risk_inputs, risks = read(m['inputs']), read(m['fit'])
                basis = OrthogonalLowOrderBasis(risk_inputs['public_risk_reference'])
                eta = [source_prior_moments(restore_joint(basis, h['joint_structural']),
                    np.asarray(m['psi'][query['task_id']])) for h in risks['heads']]
                fields.append(paired_fields(mean+eta))
            model = JointResponseMixture([m['name'] for m in members], fields)
            for j, value in zip(query['candidate_indices'], query['paired_observations']):
                model.update_raw(j, [value[0], value[1]-query['tau']])
            full = forecast(query, a, 'single_full144_risk', JointResponseMixture(['full144'], [model.components[0]]))
            old = next(r for r in full_reference if r['task_id'] == query['task_id'] and
                       r['mean_arm'] == a and r['risk_case'] == 'single_full144_risk')
            for k in ('mean', 'variance'):
                np.testing.assert_allclose(full[k], old[k], rtol=1e-8, atol=1e-9)
            rows.append(forecast(query, a, CASE, model)); models[(query['task_id'], a)] = model
    save(out/'source_predictions.json', {'execution': spec, 'rows': rows})
    # Every source prior and all six forecasts are saved before holdout-label scoring.
    reference = read(spec['reference_profiles'])['rows']; contrasts = []
    labels = {i: training_view(data, [i], public['plan']['task_distribution']['L'])[0] for i in spec['query_tasks']}
    for row in rows:
        score_source(row, labels[row['task_index']], models[(row['task_id'], row['mean_arm'])], 3)
        old = next(r for r in reference if r['task_id'] == row['task_id'] and r['mean_arm'] == row['mean_arm']
                   and r['risk_case'] == 'full_plus_six_relearned_map_deletion_risks')
        contrasts.append({'task_id': row['task_id'], 'mean_arm': row['mean_arm'],
            'g_mse_difference': row['scores']['observed_head_mse'][1]-old['scores']['observed_head_mse'][1],
            'joint_batch_score_difference': row['scores']['joint_batch_log_score']-old['scores']['joint_batch_log_score'],
            'g_batch_score_difference': row['scores']['response_batch_log_scores'][1]-old['scores']['response_batch_log_scores'][1]})
    summary = {a: {k: np.mean([r['scores'][k] for r in rows if r['mean_arm'] == a], axis=0).tolist()
                  for k in rows[0]['scores']} for a in spec['mean_arms']}
    main = [c for c in contrasts if c['mean_arm'] == 'joint_structural']
    gate = all(c['g_mse_difference'] <= 0 and c['joint_batch_score_difference'] >= 0 and c['g_batch_score_difference'] > 0 for c in main)
    save(out/'source_calibration_profiles.json', {'rows': rows})
    result = {'execution': spec, 'summary': summary, 'contrasts': contrasts,
        'full_component_reproduction_passed': True, 'source_progress_gate_passed': gate,
        'cost': {'historical_source_calls_per_arm': 144, 'new_native_mean_head_fits': 84,
            'new_source_risk_fits': 0, 'new_simulator_calls': 0, 'new_verification_calls': 0,
            'new_terminal_integrations': 0}, 'source_mean_fit_cpu_seconds': sum(m['cpu_seconds'] for m in means[1:]),
        'cpu_seconds': time.process_time()-started}
    save(out/'results.json', result); print(result, flush=True)


if __name__ == '__main__':
    run()
