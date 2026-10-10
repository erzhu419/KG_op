"""Source-label efficiency with all supervised maps and risk priors refitted."""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_diverse_source_design import encode_diverse_source, score_row
from performance.check_legacy_public_mean_coordinates import fit as fit_mean, forecast_rows
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.run_legacy_joint_population_matched_baselines import paired_fields, standardize
from performance.run_legacy_joint_population_source_budget import restore_joint
from performance.run_legacy_related_inventory_source import problem_for, save, training_view
from representation.joint_response_mixture import JointResponseMixture
from representation.legacy_observable_prior import legacy_exposure_descriptors
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis, fit_orthogonal_function_prior
from representation.public_control_mean import public_control_mean_coordinates, source_prior_moments
from representation.source_function_prior import head_batches
from representation.source_joint_population import (fit_joint_source_population,
    fit_joint_common_source_population, continue_joint_common_source_population)
from representation.source_kernel_population import predict_unequal_site_kernel_population

EXECUTION = ROOT/'performance/manifests/legacy_joint_population_public_mean_source_half_execution_v1.json'


def source_views(data, public, spec):
    bank = [tuple(p) for p in public['bank']]; raw, descriptors = {}, {}
    for i, task in enumerate(public['tasks']):
        provider = no_outcomes(problem_for(public['plan'], task))
        raw[i] = public_control_mean_coordinates(provider, bank)
        descriptors[i] = legacy_exposure_descriptors(provider, bank, 50)
    ids, layouts = spec['training_tasks'], spec['source_site_indices_by_task']
    coordinate, risk_tasks, risk_center, risk_scale = encode_diverse_source(
        data, ids, layouts, descriptors, public['plan']['task_distribution']['L'])
    mean_psi, center, scale = standardize(np.vstack([raw[i][sites] for i, sites in zip(ids, layouts)]))
    stops = np.cumsum([len(s) for s in layouts])[:-1]
    mean_tasks = [{**t, 'psi': p, 'features': p} for t, p in zip(risk_tasks, np.split(mean_psi, stops))]
    mean_reference = (raw[ids[0]]-center)/scale
    risk_reference = coordinate.transform(descriptors[ids[0]])
    mean_basis, risk_basis = OrthogonalLowOrderBasis(mean_reference), OrthogonalLowOrderBasis(risk_reference)
    mean_design = np.column_stack([np.ones(len(mean_psi)), mean_basis.transform(mean_psi)])
    risk_psi = np.vstack([t['psi'] for t in risk_tasks])
    risk_design = np.column_stack([np.ones(len(risk_psi)), risk_basis.transform(risk_psi)])
    return {'source_tasks': [{**t, **{k: t[k].tolist() for k in ('psi', 'features', 'response', 'noise')}}
                            for t in mean_tasks],
        'risk_source_tasks': [{**t, **{k: t[k].tolist() for k in ('psi', 'features', 'response', 'noise')}}
                             for t in risk_tasks],
        'public_mean_reference': mean_reference.tolist(), 'public_risk_reference': risk_reference.tolist(),
        'mean_coordinate_center': center.tolist(), 'mean_coordinate_scale': scale.tolist(),
        'risk_coordinate_center': risk_center.tolist(), 'risk_coordinate_scale': risk_scale.tolist(),
        'coordinate': coordinate.diagnostics(), 'mean_basis': mean_basis.diagnostics(), 'risk_basis': risk_basis.diagnostics(),
        'mean_source_design_rank': int(np.linalg.matrix_rank(mean_design)),
        'risk_source_design_rank': int(np.linalg.matrix_rank(risk_design)),
        'heldout_mean_psi': {str(i): ((raw[i]-center)/scale).tolist() for i in spec['heldout_tasks']},
        'heldout_risk_psi': {str(i): coordinate.transform(descriptors[i]).tolist() for i in spec['heldout_tasks']}}


def prepare(spec):
    started = time.process_time()
    data = json.loads((ROOT/spec['source_data']).read_text())
    public = json.loads((ROOT/spec['source_data']).with_name('frozen_inputs.json').read_text())
    inputs = {'execution': spec, **source_views(data, public, spec), 'cpu_seconds': time.process_time()-started}
    save(ROOT/spec['output']/'inputs.json', inputs)
    print('source half prepared', len(inputs['source_tasks'])*len(inputs['source_tasks'][0]['psi']),
        'mean/risk ranks', inputs['mean_source_design_rank'], inputs['risk_source_design_rank'], flush=True)
    return inputs


def fit_risk(spec, inputs):
    out = ROOT/spec['output']; path = out/'risk_source_fit.json'
    fitted = json.loads(path.read_text()) if path.exists() else {'heads': [], 'cpu_seconds': 0.}
    tasks = [{**t, **{k: np.asarray(t[k]) for k in ('psi', 'features', 'response', 'noise')}}
             for t in inputs['risk_source_tasks']]
    basis = OrthogonalLowOrderBasis(inputs['public_risk_reference'])
    for h in (2, 3)[len(fitted['heads']):]:
        started = time.process_time(); batches = head_batches(tasks, h); noises = [t['noise'][:, h] for t in tasks]
        regularizer = fit_orthogonal_function_prior(batches, noises, basis,
            max_iterations=spec['regularizer_max_iterations'])
        fixed = fit_joint_source_population(regularizer, batches, noises,
            max_iterations=spec['fixed_population_max_iterations'])
        joint = fit_joint_common_source_population(regularizer, batches, noises,
            fixed.diagnostics, optimizer=spec['optimizer'])
        if not joint.diagnostics['optimizer_converged']:
            save(out/f'risk_head{h}_joint_before_completion.json', joint.diagnostics)
            joint = continue_joint_common_source_population(regularizer, batches, noises,
                joint.diagnostics, optimizer=spec['optimizer'])
        record = {'head': h, 'regularizer': regularizer.diagnostics,
                  'fixed_structural': fixed.diagnostics, 'joint_structural': joint.diagnostics}
        fitted['heads'].append(record); fitted['cpu_seconds'] += time.process_time()-started; save(path, fitted)
        print('source half risk head', h, {k: (d['optimizer_converged'], d['optimizer_iterations'])
            for k, d in record.items() if k != 'head'}, flush=True)
    if not all(d['optimizer_converged'] for h in fitted['heads'] for k, d in h.items() if k != 'head'):
        raise RuntimeError('complete failed source risk objectives before forecasts')
    return fitted


def components(inputs, means, risks, mean_psi, risk_psi):
    mean_basis = OrthogonalLowOrderBasis(inputs['public_mean_reference'])
    risk_basis = OrthogonalLowOrderBasis(inputs['public_risk_reference'])
    eta = [source_prior_moments(restore_joint(risk_basis, h['joint_structural']), risk_psi)
           for h in risks['heads']]
    fields = [paired_fields([source_prior_moments(restore_joint(mean_basis, h[key]), mean_psi)
                            for h in means['heads']]+eta) for key in ('joint_structural', 'flat')]
    fields.append(paired_fields([predict_unequal_site_kernel_population(mean_psi, h['joint_gp'])
                                for h in means['heads']]+eta))
    return fields


def source_check(spec, inputs, means, risks):
    started = time.process_time(); data = json.loads((ROOT/spec['source_data']).read_text()); rows = []
    for i in spec['heldout_tasks']:
        model = JointResponseMixture(spec['arms'], components(inputs, means, risks,
            np.asarray(inputs['heldout_mean_psi'][str(i)]), np.asarray(inputs['heldout_risk_psi'][str(i)])))
        task = data['tasks'][i]
        for j in (0, 1):
            value = task['Y_replicates'][j][0]; model.update_raw(j, [value[0], value[1]-task['tau']])
        rows.extend(forecast_rows(model, task['task_id'], i))
    out = ROOT/spec['output']; save(out/'source_predictions.json', {'execution': spec, 'rows': rows})
    public = json.loads((ROOT/spec['source_data']).with_name('frozen_inputs.json').read_text())
    for row in rows:
        labels = training_view(data, [row['task_index']], public['plan']['task_distribution']['L'])[0]
        score_row(row, labels, spec['label_replicates_per_record'])
        row['calibration_profile']['diagnostic_points'].append({'candidate': 60,
            'g_mean': row['mean'][60][1], 'g_observed': float(labels['response'][60, 1]),
            'g_standardized_residual': row['calibration_profile']['standardized_residual'][60][1]})
    summary = {a: {k: np.mean([r['scores'][k] for r in rows if r['arm'] == a], axis=0).tolist()
                   for k in rows[0]['scores']} for a in spec['arms']}
    reference = json.loads((ROOT/spec['reference_profiles']).read_text())
    budget, matched = [], []
    for row in rows:
        old = next(r for r in reference['rows'] if r['task_id'] == row['task_id'] and r['arm'] == row['arm'])
        budget.append({'task_id': row['task_id'], 'arm': row['arm'], 'source_calls': [144, 288],
            'scores_144': row['scores'], 'scores_288': old['scores']})
    for i in spec['heldout_tasks']:
        main = next(r for r in rows if r['task_index'] == i and r['arm'] == spec['arms'][0])
        for a in spec['arms'][1:]:
            ref = next(r for r in rows if r['task_index'] == i and r['arm'] == a)
            matched.append({'task_id': main['task_id'], 'arm': main['arm'], 'reference': a,
                'g_mse_difference': main['scores']['observed_head_mse'][1]-ref['scores']['observed_head_mse'][1],
                'joint_batch_score_difference': main['scores']['joint_batch_log_score']-ref['scores']['joint_batch_log_score'],
                'constraint_batch_score_difference': main['scores']['response_batch_log_scores'][1]-ref['scores']['response_batch_log_scores'][1]})
    save(out/'source_calibration_profiles.json', {'rows': [{k: v for k, v in r.items() if k != 'local_joint_components'} for r in rows]})
    result = {'summary': summary, 'budget_contrasts': budget, 'matched_contrasts': matched,
              'cpu_seconds': time.process_time()-started}
    save(out/'source_results.json', result); print('source half results', json.dumps(result), flush=True)
    return result


def run():
    spec = json.loads(EXECUTION.read_text()); out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    inputs_path = out/'inputs.json'
    inputs = json.loads(inputs_path.read_text()) if inputs_path.exists() else prepare(spec)
    means = fit_mean(spec, inputs); risks = fit_risk(spec, inputs)
    source = source_check(spec, inputs, means, risks)
    continuations = sum('continued_from_optimizer_iterations' in h['joint_structural']
                        for h in means['heads']+risks['heads'])
    result = {'execution': spec, 'source': source,
        'source_fit_cpu_seconds': means['cpu_seconds']+risks['cpu_seconds'],
        'cost': {'new_source_calls': 0, 'new_target_calls': 0, 'new_verification_calls': 0,
            'historical_source_calls_per_arm': spec['source_response_calls_per_arm'],
            'new_native_source_head_fits': 20+continuations, 'new_terminal_integrations': 0}}
    save(out/'results.json', result)


if __name__ == '__main__':
    run()
