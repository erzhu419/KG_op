"""Training-only moment-matched task spectrum; unchanged development gate."""

import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.calibrate_legacy_task_transfer import NATIVE, CALIBRATED, forecast_branch, mean_density
from performance.check_legacy_joint_common_covariance import restore_regularizer
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.check_legacy_public_mean_source_half import components
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.run_legacy_related_inventory_source import problem_for, save
from representation.joint_function_risk import joint_batch_log_score
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.public_control_mean import public_control_mean_coordinates
from representation.source_function_prior import head_batches
from representation.source_joint_population import JointCovariancePopulationLikelihood
from representation.task_spectral_moments import integrate_task_spectrum

EXECUTION = 'performance/manifests/legacy_joint_population_task_spectral_moments_execution_v1.json'
INTEGRATED = 'conditional_task_spectral_moments'
METRICS = ('g_mse', 'g_gaussian_mean_density', 'g_raw_batch_density',
    'raw_paid_log_evidence', 'sequential_transcript_density')


def score(row, branch, fields):
    scores = {}
    for stage, field in zip(('prior', 'raw_paid'), fields):
        prediction = row['predictions'][stage]
        scores[stage] = {
            'g_mse': float(np.mean((np.asarray(prediction['mean'])-branch['observed'][:, 1])**2)),
            'g_gaussian_mean_density': mean_density(prediction, branch),
            'g_raw_batch_density': float(np.mean([joint_batch_log_score(field.local_components(j), value[1], value[2], 3)
                for j, value in zip(branch['sites'], branch['observed'])]))}
    paid = scores['raw_paid']; count = len(branch['sites'])
    paid['raw_paid_log_evidence'] = row['predictions']['raw_paid']['paid_predictive_log_score']
    paid['sequential_transcript_density'] = (paid['raw_paid_log_evidence']+count*paid['g_gaussian_mean_density'])/(count+len(branch['paid_sites']))
    return scores


def run():
    started = time.process_time(); spec = read(EXECUTION)
    scalar = read(spec['spectral_results']); profile = read(spec['profile_execution'])
    projection = read(profile['projection_execution']); gain = read(projection['gain_execution'])
    adaptation = read(gain['adaptation_execution']); coverage = read(adaptation['coverage_execution'])
    parent = read(coverage['source_execution']); previous = read(Path(profile['output'])/'calibration_profiles.json')['rows']
    calibration_spec = read(spec['calibration_execution'])
    calibration_rows = read(Path(calibration_spec['output'])/'calibration_profiles.json')['rows']
    rejected = read('paper_artifacts/legacy_joint_population_task_recondition_common_v1/results.json')
    public = read(Path(coverage['source_data']).with_name('frozen_inputs.json')); bank = [tuple(p) for p in public['bank']]
    raw = read(coverage['source_data'])
    values = {i: [raw['tasks'][i]['Y_replicates'][j][0][1]-raw['tasks'][i]['tau'] for j in spec['paid_candidates']]
        for i in spec['training_tasks']}; del raw
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    branches, rows, fields, laws = {}, [], {}, []
    for layout in spec['layouts']:
        source = read(Path(coverage['output'])/f'layout{layout}'/'inputs.json')
        for deleted in spec['training_tasks']:
            inputs = read(Path(coverage['output'])/f'layout{layout}'/f'task{deleted}'/'inputs.json')
            means = read(Path(profile['output'])/f'layout{layout}'/f'task{deleted}'/'selected_source_fit.json')
            risk = read(Path(parent['output'])/f"layout{adaptation['parent_layouts'][layout]}"/f'task{deleted}'/'risk_source_fit.json')
            native = means['heads'][1]['joint_structural']; basis = OrthogonalLowOrderBasis(inputs['public_mean_reference'])
            tasks = [{**t, **{k: np.asarray(t[k]) for k in ('psi', 'features', 'response', 'noise')}} for t in inputs['source_tasks']]
            prior = restore_regularizer(basis, means['heads'][1]['regularizer'], tasks, 1)
            likelihood = JointCovariancePopulationLikelihood(prior, head_batches(tasks, 1), [t['noise'][:, 1] for t in tasks])
            reference = next(r for r in scalar['rows'] if r['layout'] == layout and r['deleted_task'] == deleted)
            law, diagnostics = integrate_task_spectrum(likelihood, native, spec['quadrature'], reference)
            laws.append({'layout': layout, 'deleted_task': deleted, 'law': law, 'diagnostics': diagnostics})
            provider = no_outcomes(problem_for(public['plan'], public['tasks'][deleted]))
            psi = (public_control_mean_coordinates(provider, bank)-inputs['mean_coordinate_center'])/inputs['mean_coordinate_scale']
            template = components(inputs, means, risk, psi, np.asarray(inputs['public_risk_reference']))[0][1]
            labels = source['source_tasks'][deleted]
            branch = {'layout': layout, 'deleted_task': deleted, 'template': template,
                'design': np.column_stack([np.ones(len(psi)), basis.transform(psi)]), 'native': native,
                'sites': coverage['layouts'][layout][deleted], 'paid_sites': spec['paid_candidates'],
                'paid_values': values[deleted], 'observed': np.asarray(labels['response']), 'noise': np.asarray(labels['noise'])[:, 1]}
            predictions, native_fields = forecast_branch(branch, [1., 1.]); original_paid = deepcopy(template)
            for j, y in zip(branch['paid_sites'], branch['paid_values']):
                original_paid.update_batch(j, float(y), None, 1)
            cached = next(r for r in previous if r['layout'] == layout and r['deleted_task'] == deleted and r['case'] == NATIVE)
            for stage, field, expected in zip(('prior', 'raw_paid'), native_fields, (template, original_paid)):
                for actual, original in zip(field.predict(full_cov=True), expected.predict(full_cov=True)):
                    np.testing.assert_allclose(actual, original, atol=1e-8, rtol=1e-8)
                for key in ('mean', 'covariance'):
                    np.testing.assert_allclose(predictions[stage][key], cached['predictions'][stage][key], atol=1e-8, rtol=1e-8)
            row = {'layout': layout, 'deleted_task': deleted, 'case': NATIVE,
                'scoring_sites': branch['sites'], 'predictions': predictions}
            rows.append(row); fields[(layout, deleted, NATIVE)] = native_fields
            branch['native'] = law; integrated, integrated_fields = forecast_branch(branch, [1., 1.])
            native_mean, native_cov = template.predict(full_cov=True)
            new_mean, new_cov = integrated_fields[0].predict(full_cov=True)
            np.testing.assert_array_equal(new_mean[:, 1], native_mean[:, 1])
            np.testing.assert_array_equal(new_cov[:, 1, :, 1], native_cov[:, 1, :, 1])
            rows.append({'layout': layout, 'deleted_task': deleted, 'case': INTEGRATED,
                'scoring_sites': branch['sites'], 'predictions': integrated})
            fields[(layout, deleted, INTEGRATED)] = integrated_fields
            branches[(layout, deleted)] = branch
        print('six integrated source laws and native endpoint replays frozen', layout, flush=True)
    save(out/'integrated_source_laws.json', {'execution': spec, 'rows': laws})
    save(out/'predictions.json', {'execution': spec, 'all12_integrated_forecasts_frozen_before_scoring': True,
        'native_all73_source_laws_and_scalar_integrals_reproduced': True, 'eta_laws_unchanged': True, 'rows': rows})
    for row in rows:
        key = (row['layout'], row['deleted_task']); row['scores'] = score(row, branches[key], fields[(*key, row['case'])])
        if row['case'] == NATIVE:
            cached = next(r for r in previous if r['layout'] == key[0] and r['deleted_task'] == key[1] and r['case'] == NATIVE)
            for stage in ('prior', 'raw_paid'):
                for metric, expected in cached['scores'][stage].items():
                    np.testing.assert_allclose(row['scores'][stage][metric], expected, atol=1e-8, rtol=1e-8)
    summary = {layout: {case: {stage: {metric: float(np.mean([r['scores'][stage][metric]
        for r in rows if r['layout'] == layout and r['case'] == case]))
        for metric in rows[0]['scores'][stage]} for stage in ('prior', 'raw_paid')}
        for case in (NATIVE, INTEGRATED)} for layout in spec['layouts']}
    benchmarks = {}
    for layout in spec['layouts']:
        local = [r for r in calibration_rows if r['layout'] == layout and r['case'] == CALIBRATED]
        for r in local:
            key = (layout, r['deleted_task']); count = len(branches[key]['sites'])
            paid = r['scores']['raw_paid']; paid['raw_paid_log_evidence'] = r['predictions']['raw_paid']['paid_predictive_log_score']
            paid['sequential_transcript_density'] = (paid['raw_paid_log_evidence']+count*paid['g_gaussian_mean_density'])/(count+2)
        benchmarks[layout] = {'stage80_calibrated': {metric: float(np.mean([r['scores']['raw_paid'][metric] for r in local]))
            for metric in METRICS}, 'stage82_reconditioned': rejected['summary'][layout]['reconditioned_common']}
    preferred, contrasts = {}, []
    for layout in spec['layouts']:
        new, old = (summary[layout][case]['raw_paid'] for case in (INTEGRATED, NATIVE))
        preferred[layout] = new['g_mse'] <= old['g_mse'] and new['g_raw_batch_density'] > old['g_raw_batch_density'] and new['sequential_transcript_density'] >= old['sequential_transcript_density']
    for row in rows:
        if row['case'] != INTEGRATED:
            continue
        old = next(r for r in rows if r['case'] == NATIVE and r['layout'] == row['layout'] and r['deleted_task'] == row['deleted_task'])
        contrasts.append({'layout': row['layout'], 'deleted_task': row['deleted_task'],
            'integrated_minus_native': {stage: {metric: row['scores'][stage][metric]-old['scores'][stage][metric]
                for metric in row['scores'][stage]} for stage in ('prior', 'raw_paid')}})
    result = {'execution': spec, 'summary': summary, 'benchmarks': benchmarks, 'contrasts': contrasts,
        'preferred_by_layout': preferred, 'spectral_moments_preferred': all(preferred.values()),
        'native_source_laws_all73_moments_scores_and_scalar_integrals_reproduced': True,
        'cost': {**spec['cost'], 'actual_conditional_source_integrals': len(laws),
            'conditional_likelihood_evaluations': sum(r['diagnostics']['likelihood_evaluations'] for r in laws)},
        'cpu_seconds': time.process_time()-started, 'source_holdouts_evaluated': False,
        'target_promoted': False, 'independent_confirmation': False}
    save(out/'calibration_profiles.json', {'execution': spec, 'rows': rows}); save(out/'results.json', result)
    print({'spectral_moments_preferred': result['spectral_moments_preferred'], 'summary': summary,
        'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
