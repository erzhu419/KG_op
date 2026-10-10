"""Training-task contrast: infer common curves under the calibrated task law."""

import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.calibrate_legacy_task_transfer import NATIVE, forecast_branch, mean_density
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
from representation.task_transfer_calibration import recondition_common, scaled_task_diagonal

EXECUTION = 'performance/manifests/legacy_joint_population_task_recondition_common_execution_v1.json'
FIXED, RECONDITIONED = 'fixed_common', 'reconditioned_common'


def run():
    started = time.process_time(); spec = read(EXECUTION); calibration_spec = read(spec['calibration_execution'])
    calibration = read(spec['calibration_results']); profile = read(spec['profile_execution'])
    projection = read(profile['projection_execution']); gain = read(projection['gain_execution'])
    adaptation = read(gain['adaptation_execution']); coverage = read(adaptation['coverage_execution'])
    parent = read(coverage['source_execution']); fixed_rows = read(spec['calibration_profiles'])['rows']
    public = read(Path(coverage['source_data']).with_name('frozen_inputs.json')); bank = [tuple(p) for p in public['bank']]
    raw = read(coverage['source_data'])
    values = {i: [raw['tasks'][i]['Y_replicates'][j][0][1]-raw['tasks'][i]['tau'] for j in spec['paid_candidates']]
        for i in spec['training_tasks']}; del raw
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    branches, rows, models, laws = {}, [], {}, []
    for layout in spec['layouts']:
        source = read(Path(coverage['output'])/f'layout{layout}'/'inputs.json')
        parameters = calibration['fits'][layout]['multipliers']
        for deleted in spec['training_tasks']:
            inputs = read(Path(coverage['output'])/f'layout{layout}'/f'task{deleted}'/'inputs.json')
            means = read(Path(profile['output'])/f'layout{layout}'/f'task{deleted}'/'selected_source_fit.json')
            risk = read(Path(parent['output'])/f"layout{adaptation['parent_layouts'][layout]}"/f'task{deleted}'/'risk_source_fit.json')
            native = means['heads'][1]['joint_structural']; basis = OrthogonalLowOrderBasis(inputs['public_mean_reference'])
            tasks = [{**t, **{k: np.asarray(t[k]) for k in ('psi', 'features', 'response', 'noise')}} for t in inputs['source_tasks']]
            regularizer = restore_regularizer(basis, means['heads'][1]['regularizer'], tasks, 1)
            likelihood = JointCovariancePopulationLikelihood(regularizer, head_batches(tasks, 1), [t['noise'][:, 1] for t in tasks])
            original = recondition_common(likelihood, native, [1., 1.])
            for key in ('coefficient_mean', 'population_mean_covariance', 'residual_coefficient_diagonal',
                    'discrepancy_variance', 'common_prior_coefficient_diagonal', 'source_negative_log_likelihood_per_label'):
                np.testing.assert_allclose(original[key], native[key], atol=1e-8, rtol=1e-8)
            conditioned = recondition_common(likelihood, native, parameters)
            np.testing.assert_allclose(conditioned['residual_coefficient_diagonal'],
                scaled_task_diagonal(native, basis.feature_dim, parameters), atol=1e-12, rtol=1e-12)
            for key in ('common_prior_coefficient_diagonal', 'common_group_variances', 'common_spectral_exponent',
                    'spectral_exponent', 'discrepancy_variance'):
                np.testing.assert_array_equal(conditioned[key], original[key])
            laws.append({'layout': layout, 'deleted_task': deleted, 'law': conditioned,
                'source_nll_difference': conditioned['source_negative_log_likelihood_per_label']-native['source_negative_log_likelihood_per_label']})
            provider = no_outcomes(problem_for(public['plan'], public['tasks'][deleted]))
            psi = (public_control_mean_coordinates(provider, bank)-inputs['mean_coordinate_center'])/inputs['mean_coordinate_scale']
            template = components(inputs, means, risk, psi, np.asarray(inputs['public_risk_reference']))[0][1]
            label = source['source_tasks'][deleted]
            branch = {'layout': layout, 'deleted_task': deleted, 'template': template,
                'design': np.column_stack([np.ones(len(psi)), basis.transform(psi)]), 'native': original,
                'sites': coverage['layouts'][layout][deleted], 'paid_sites': spec['paid_candidates'],
                'paid_values': values[deleted], 'observed': np.asarray(label['response']), 'noise': np.asarray(label['noise'])[:, 1]}
            _, native_fields = forecast_branch(branch, [1., 1.]); native_paid = deepcopy(template)
            for j, y in zip(branch['paid_sites'], branch['paid_values']):
                native_paid.update_batch(j, float(y), None, 1)
            for field, expected in zip(native_fields, (template, native_paid)):
                for actual, prior in zip(field.predict(full_cov=True), expected.predict(full_cov=True)):
                    np.testing.assert_allclose(actual, prior, atol=1e-8, rtol=1e-8)
            branch['native'] = native; fixed, fixed_fields = forecast_branch(branch, parameters)
            cached = next(r for r in fixed_rows if r['layout'] == layout and r['deleted_task'] == deleted)
            for stage in ('prior', 'raw_paid'):
                for key in ('mean', 'covariance'):
                    np.testing.assert_allclose(fixed[stage][key], cached['predictions'][stage][key], atol=1e-8, rtol=1e-8)
            np.testing.assert_allclose(fixed['raw_paid']['paid_predictive_log_score'],
                cached['predictions']['raw_paid']['paid_predictive_log_score'], atol=1e-8)
            branch['native'] = conditioned; predictions, fields = forecast_branch(branch, [1., 1.])
            branches[(layout, deleted)] = branch
            for case, predicted, stored in ((FIXED, fixed, fixed_fields), (RECONDITIONED, predictions, fields)):
                rows.append({'layout': layout, 'deleted_task': deleted, 'case': case,
                    'scoring_sites': branch['sites'], 'predictions': predicted})
                models[(layout, deleted, case)] = stored
        print('common laws and paired forecasts frozen', layout, flush=True)
    save(out/'conditioned_source_laws.json', {'execution': spec, 'rows': laws})
    save(out/'predictions.json', {'execution': spec, 'all24_forecasts_frozen': True,
        'native_objectives_laws_all73_moments_and_fixed_endpoint_reproduced': True, 'rows': rows})
    for row in rows:
        layout, deleted, case = row['layout'], row['deleted_task'], row['case']; b = branches[(layout, deleted)]
        if case == FIXED:
            row['scores'] = deepcopy(next(r['scores'] for r in fixed_rows if r['layout'] == layout and r['deleted_task'] == deleted))
        else:
            row['scores'] = {}
            for stage, model in zip(('prior', 'raw_paid'), models[(layout, deleted, case)]):
                prediction = row['predictions'][stage]
                row['scores'][stage] = {'g_mse': float(np.mean((np.asarray(prediction['mean'])-b['observed'][:, 1])**2)),
                    'g_gaussian_mean_density': mean_density(prediction, b),
                    'g_raw_batch_density': float(np.mean([joint_batch_log_score(model.local_components(j), value[1], value[2], 3)
                        for j, value in zip(row['scoring_sites'], b['observed'])]))}
        paid = row['predictions']['raw_paid']; score = row['scores']['raw_paid']; count = len(b['sites'])
        score['raw_paid_log_evidence'] = paid['paid_predictive_log_score']
        score['sequential_transcript_density'] = (score['raw_paid_log_evidence']+count*score['g_gaussian_mean_density'])/(count+2)
    metrics = ('g_mse', 'g_gaussian_mean_density', 'g_raw_batch_density', 'raw_paid_log_evidence', 'sequential_transcript_density')
    summary = {layout: {case: {key: float(np.mean([r['scores']['raw_paid'][key] for r in rows if r['layout'] == layout and r['case'] == case]))
        for key in metrics} for case in (FIXED, RECONDITIONED)} for layout in spec['layouts']}
    preferred = {}; contrasts = []
    for layout in spec['layouts']:
        new, fixed = summary[layout][RECONDITIONED], summary[layout][FIXED]
        baseline = calibration['summary'][layout][NATIVE]['raw_paid']
        preferred[layout] = (all(new['g_mse'] <= ref['g_mse'] and new['g_raw_batch_density'] > ref['g_raw_batch_density']
            for ref in (fixed, baseline)) and new['sequential_transcript_density'] >= fixed['sequential_transcript_density'])
    for row in rows:
        if row['case'] != RECONDITIONED:
            continue
        old = next(r for r in rows if r['layout'] == row['layout'] and r['deleted_task'] == row['deleted_task'] and r['case'] == FIXED)
        contrasts.append({'layout': row['layout'], 'deleted_task': row['deleted_task'],
            'reconditioned_minus_fixed': {key: row['scores']['raw_paid'][key]-old['scores']['raw_paid'][key] for key in metrics}})
    result = {'execution': spec, 'summary': summary, 'contrasts': contrasts, 'preferred_by_layout': preferred,
        'reconditioned_common_preferred': all(preferred.values()), 'native_and_fixed_endpoint_reproduction_passed': True,
        'cost': {**spec['cost'], 'actual_common_conditionings': len(laws)}, 'cpu_seconds': time.process_time()-started,
        'target_promoted': False, 'independent_confirmation': False}
    save(out/'calibration_profiles.json', {'execution': spec, 'rows': rows}); save(out/'results.json', result)
    print({'reconditioned_common_preferred': result['reconditioned_common_preferred'], 'summary': summary,
        'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
