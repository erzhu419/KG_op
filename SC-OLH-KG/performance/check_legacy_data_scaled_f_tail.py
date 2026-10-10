"""Original f source family with the unchanged1% tail tied to source data."""

import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_joint_common_covariance import restore_regularizer
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.check_legacy_public_mean_source_half import components
from performance.check_legacy_source_mean_covariance_swap import g_prediction
from performance.check_legacy_task_bank_centering import prepare, verify_dense
from performance.run_legacy_related_inventory_source import save
from performance.validate_legacy_paired_taskout import gaussian_score
from representation.joint_function_risk import joint_batch_log_score
from representation.source_function_prior import head_batches
from representation.task_bank_centering import CenteredTaskPopulationLikelihood, centered_g_field, fit_centered_task_population

EXECUTION = 'performance/manifests/legacy_joint_population_data_scaled_f_tail_execution_v1.json'
NATIVE, DATA = 'native_f', 'data_scaled_f_tail'


def forecast_field(field, sites, paid_sites, values):
    paid = deepcopy(field); evidence = 0.
    for j, value in zip(paid_sites, values):
        evidence += paid.update_batch(j, float(value), None, 1)
    predictions = {}
    for stage, model in zip(('prior', 'raw_paid'), (field, paid)):
        mean, covariance = model.predict(full_cov=True)
        predictions[stage] = g_prediction(mean[:, 0], covariance[:, 0, :, 0], sites)
    predictions['raw_paid']['paid_predictive_log_score'] = float(evidence)
    return predictions, (field, paid)


def score_f(row, fields, labels):
    observed, noise = np.asarray(labels['response']), np.asarray(labels['noise'])[:, 0]; scores = {}
    for stage, field in zip(('prior', 'raw_paid'), fields):
        p = row['predictions'][stage]; n = len(p['mean'])
        scores[stage] = {'f_mse': float(np.mean((np.asarray(p['mean'])-observed[:, 0])**2)),
            'f_gaussian_mean_density': gaussian_score(np.asarray(p['mean']), np.asarray(p['covariance'])+np.diag(noise), observed[:, 0])/n,
            'f_raw_batch_density': float(np.mean([joint_batch_log_score(field.local_components(j), value[0], value[3], 3)
                for j, value in zip(row['scoring_sites'], observed)]))}
    paid = scores['raw_paid']; paid['raw_paid_log_evidence'] = row['predictions']['raw_paid']['paid_predictive_log_score']
    paid['sequential_transcript_density'] = (paid['raw_paid_log_evidence']+n*paid['f_gaussian_mean_density'])/(n+2)
    return scores


def run():
    started = time.process_time(); spec = read(EXECUTION); centering = read(spec['centering_execution'])
    coverage, branches = prepare(centering); scales = read(spec['signal_results'])['rows']
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True); laws, verification = [], []
    for (layout, deleted), b in branches.items():
        native = b['means']['heads'][0]['joint_structural']; prior = restore_regularizer(b['basis'], b['means']['heads'][0]['regularizer'], b['tasks'], 0)
        likelihood = CenteredTaskPopulationLikelihood(prior, head_batches(b['tasks'], 0), [t['noise'][:, 0] for t in b['tasks']],
            [np.zeros(b['basis'].feature_dim) for _ in b['tasks']])
        parameters = np.asarray(native['optimizer_parameters']); value, _, state = likelihood.evaluate(parameters)
        np.testing.assert_allclose(value, native['source_negative_log_likelihood_per_label'], atol=1e-8, rtol=1e-8)
        for key, saved in (('mean', 'coefficient_mean'), ('mean_covariance', 'population_mean_covariance'), ('residual_diagonal', 'residual_coefficient_diagonal')):
            np.testing.assert_allclose(state[key], native[saved], atol=1e-8, rtol=1e-8)
        scale = next(r for r in scales if r['layout'] == layout and r['deleted_task'] == deleted and r['head'] == 'f')['nonnegative_data_signal_scale']
        assert scale > 0.
        likelihood.tail = .01*scale/b['basis'].feature_dim
        if deleted == 1:
            verification.append({'layout': layout, 'deleted_task': deleted, **verify_dense(likelihood, parameters)})
        folder = out/f'layout{layout}'/f'task{deleted}'; folder.mkdir(parents=True, exist_ok=True)
        assert b['inputs']['execution']['optimizer'] == spec['optimizer']
        law = fit_centered_task_population(likelihood, native, spec['optimizer'])
        law.update(population_model='original common/task f family with source-data1% weak tail',
            tail_signal_scale=scale, tail_strength_fraction=.01, query_task_covariance_requires_public_bank_center=False)
        b['f_law'] = law; save(folder/'source_fit.json', law); laws.append({'layout': layout, 'deleted_task': deleted, 'law': law})
        print('source-data f tail fit', layout, deleted, law['optimizer_converged'], law['optimizer_iterations'], flush=True)
    save(out/'source_laws.json', {'execution': spec, 'all12_laws_frozen_before_forecasts': True, 'verification': verification, 'rows': laws})
    cost = {**spec['cost'], 'actual_new_f_source_fits': len(laws)}
    if not all(r['law']['optimizer_converged'] for r in laws):
        result = {'execution': spec, 'data_tail_preferred': False, 'all_fits_complete': False, 'forecasts_scored': False,
            'cost': cost, 'cpu_seconds': time.process_time()-started, 'source_holdouts_evaluated': False, 'target_promoted': False, 'independent_confirmation': False}
        save(out/'results.json', result); print('incomplete fits retained; no scoring or completion', flush=True); return
    raw = read(coverage['source_data']); values = {i: [raw['tasks'][i]['Y_replicates'][j][0][0] for j in spec['paid_candidates']] for i in spec['training_tasks']}; del raw
    old_spec = read('performance/manifests/legacy_joint_population_training_paid_adaptation_execution_v1.json')
    old_rows = read(Path(old_spec['output'])/'predictions.json')['rows']
    rows, fields = [], {}
    for (layout, deleted), b in branches.items():
        template = components(b['inputs'], b['means'], read(b['risk_path']), b['mean_psi'], np.asarray(b['inputs']['public_risk_reference']))[0][0]
        design = b['designs'][b['task_name']]
        actual = centered_g_field(template, design, b['f_law'], np.zeros(b['basis'].feature_dim))
        m, c = actual.predict(full_cov=True); om, oc = template.predict(full_cov=True)
        np.testing.assert_array_equal(m[:, 1], om[:, 1]); np.testing.assert_array_equal(c[:, 1, :, 1], oc[:, 1, :, 1])
        diagonal = np.asarray(b['f_law']['residual_coefficient_diagonal'])
        assert np.all(diagonal[1:] >= .01*b['f_law']['tail_signal_scale']/b['basis'].feature_dim)
        coefficient_covariance = np.asarray(b['f_law']['population_mean_covariance'])+np.diag(diagonal)
        np.testing.assert_allclose(coefficient_covariance, coefficient_covariance.T, atol=1e-10, rtol=1e-10)
        assert np.linalg.eigvalsh(coefficient_covariance).min() >= -1e-10
        for case, field in ((NATIVE, template), (DATA, actual)):
            predictions, stored = forecast_field(field, b['sites'], spec['paid_candidates'], values[deleted])
            if case == NATIVE:
                cached = next(r for r in old_rows if r['layout'] == layout and r['deleted_task'] == deleted and r['mean_arm'] == 'joint_structural')
                for stage, model in zip(('prior', 'raw_paid'), stored):
                    expected = cached['prior'] if stage == 'prior' else cached
                    mean, covariance = model.predict(full_cov=True)
                    np.testing.assert_allclose(mean, np.asarray(expected['mean'])[:, [0, 3]], atol=1e-8, rtol=1e-8)
                    np.testing.assert_allclose(np.diag(covariance.reshape(2*len(mean), 2*len(mean))).reshape(-1, 2),
                        np.asarray(expected['variance'])[:, [0, 3]], atol=1e-8, rtol=1e-8)
            rows.append({'layout': layout, 'deleted_task': deleted, 'case': case, 'scoring_sites': b['sites'], 'predictions': predictions})
            fields[(layout, deleted, case)] = stored
    save(out/'predictions.json', {'execution': spec, 'all12_candidate_native_forecasts_frozen_before_scoring': True,
        'native_all73_f_eta_prior_paid_reproduced': True, 'eta_f_prior_unchanged': True, 'rows': rows})
    source = {layout: read(Path(coverage['output'])/f'layout{layout}'/'inputs.json') for layout in spec['layouts']}
    old_profiles = read(Path(old_spec['output'])/'calibration_profiles.json')['rows']
    for row in rows:
        key = (row['layout'], row['deleted_task']); row['scores'] = score_f(row, fields[(*key, row['case'])], source[key[0]]['source_tasks'][key[1]])
        if row['case'] == NATIVE:
            cached = next(r for r in old_profiles if r['layout'] == key[0] and r['deleted_task'] == key[1] and r['mean_arm'] == 'joint_structural')
            for stage, saved in (('prior', 'prior_scores'), ('raw_paid', 'scores')):
                for metric, expected in (('f_mse', cached[saved]['observed_head_mse'][0]), ('f_raw_batch_density', cached[saved]['response_batch_log_scores'][0])):
                    np.testing.assert_allclose(row['scores'][stage][metric], expected, atol=1e-8, rtol=1e-8)
    summary = {layout: {case: {stage: {k: float(np.mean([r['scores'][stage][k] for r in rows if r['layout'] == layout and r['case'] == case]))
        for k in rows[0]['scores'][stage]} for stage in ('prior', 'raw_paid')} for case in (NATIVE, DATA)} for layout in spec['layouts']}
    preferred, contrasts = {}, []
    for layout in spec['layouts']:
        new, old = (summary[layout][case]['raw_paid'] for case in (DATA, NATIVE))
        preferred[layout] = new['f_mse'] <= old['f_mse'] and new['f_raw_batch_density'] > old['f_raw_batch_density'] and new['sequential_transcript_density'] >= old['sequential_transcript_density']
    for row in rows:
        if row['case'] != DATA: continue
        old = next(r for r in rows if r['layout'] == row['layout'] and r['deleted_task'] == row['deleted_task'] and r['case'] == NATIVE)
        contrasts.append({'layout': row['layout'], 'deleted_task': row['deleted_task'], 'data_minus_native': {stage:
            {k: row['scores'][stage][k]-old['scores'][stage][k] for k in row['scores'][stage]} for stage in ('prior', 'raw_paid')}})
    g = read(Path(centering['output'])/'results.json')['summary']
    joint = {layout: {case: summary[layout][case]['raw_paid']['f_raw_batch_density']+g[layout]['public_bank_centered_task_population']['raw_paid']['g_raw_batch_density']
        for case in (NATIVE, DATA)} for layout in spec['layouts']}
    result = {'execution': spec, 'summary': summary, 'contrasts': contrasts, 'joint_raw_density_with_identical_stage88_g': joint,
        'preferred_by_layout': preferred, 'data_tail_preferred': all(preferred.values()), 'all_fits_complete': True, 'forecasts_scored': True,
        'native_all73_moments_source_scores_and_dense_gradients_reproduced': True, 'cost': cost,
        'cpu_seconds': time.process_time()-started, 'source_holdouts_evaluated': False, 'target_promoted': False, 'independent_confirmation': False}
    save(out/'calibration_profiles.json', {'execution': spec, 'rows': rows}); save(out/'results.json', result)
    print({'data_tail_preferred': result['data_tail_preferred'], 'summary': summary, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
