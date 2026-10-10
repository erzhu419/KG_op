"""Two source-only transfer multipliers, with the original development gate."""

import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.check_legacy_public_mean_source_half import components
from performance.check_legacy_source_mean_covariance_swap import g_prediction
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.run_legacy_related_inventory_source import problem_for, save
from performance.validate_legacy_paired_taskout import gaussian_score
from representation.joint_function_risk import joint_batch_log_score
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.public_control_mean import public_control_mean_coordinates
from representation.task_transfer_calibration import calibrated_g_field, scaled_task_diagonal

EXECUTION = 'performance/manifests/legacy_joint_population_task_transfer_calibration_execution_v1.json'
NATIVE, CALIBRATED = 'native_source_nll_selection', 'source_transfer_calibration'


def forecast_branch(branch, multipliers):
    prior = calibrated_g_field(branch['template'], branch['design'], branch['native'], multipliers)
    paid = deepcopy(prior); paid_evidence = 0.
    for j, y in zip(branch['paid_sites'], branch['paid_values']):
        paid_evidence += paid.update_batch(j, float(y), None, 1)
    predictions = {}
    for stage, model in zip(('prior', 'raw_paid'), (prior, paid)):
        mean, covariance = model.predict(full_cov=True)
        predictions[stage] = g_prediction(mean[:, 0], covariance[:, 0, :, 0], branch['sites'])
    predictions['raw_paid']['paid_predictive_log_score'] = float(paid_evidence)
    return predictions, (prior, paid)


def mean_density(prediction, branch):
    return gaussian_score(np.asarray(prediction['mean']),
        np.asarray(prediction['covariance'])+np.diag(branch['noise']), branch['observed'][:, 1])/len(branch['sites'])


def conditional_loss(predictions, branch):
    return -mean_density(predictions['raw_paid'], branch)


def run(execution=EXECUTION, calibration_loss=conditional_loss):
    started = time.process_time(); spec = read(execution); profile = read(spec['profile_execution'])
    projection = read(profile['projection_execution']); gain = read(projection['gain_execution'])
    adaptation = read(gain['adaptation_execution']); coverage = read(adaptation['coverage_execution'])
    parent = read(coverage['source_execution']); previous = read(spec['training_control_profiles'])['rows']
    public = read(Path(coverage['source_data']).with_name('frozen_inputs.json')); bank = [tuple(p) for p in public['bank']]
    raw = read(coverage['source_data'])
    values = {i: [raw['tasks'][i]['Y_replicates'][j][0][1]-raw['tasks'][i]['tau'] for j in spec['paid_candidates']]
        for i in spec['training_tasks']}; del raw
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    branches = {}
    for layout in spec['layouts']:
        source = read(Path(coverage['output'])/f'layout{layout}'/'inputs.json'); local = []
        for deleted in spec['training_tasks']:
            inputs = read(Path(coverage['output'])/f'layout{layout}'/f'task{deleted}'/'inputs.json')
            means = read(Path(profile['output'])/f'layout{layout}'/f'task{deleted}'/'selected_source_fit.json')
            risk = read(Path(parent['output'])/f"layout{adaptation['parent_layouts'][layout]}"/f'task{deleted}'/'risk_source_fit.json')
            native = means['heads'][1]['joint_structural']; basis = OrthogonalLowOrderBasis(inputs['public_mean_reference'])
            provider = no_outcomes(problem_for(public['plan'], public['tasks'][deleted]))
            psi = (public_control_mean_coordinates(provider, bank)-inputs['mean_coordinate_center'])/inputs['mean_coordinate_scale']
            design = np.column_stack([np.ones(len(psi)), basis.transform(psi)])
            template = components(inputs, means, risk, psi, np.asarray(inputs['public_risk_reference']))[0][1]
            labels = source['source_tasks'][deleted]
            cached = next(r for r in previous if r['layout'] == layout and r['deleted_task'] == deleted and r['case'] == NATIVE)
            branch = {'layout': layout, 'deleted_task': deleted, 'template': template, 'design': design,
                'native': native, 'sites': coverage['layouts'][layout][deleted], 'paid_sites': spec['paid_candidates'],
                'paid_values': values[deleted], 'observed': np.asarray(labels['response']),
                'noise': np.asarray(labels['noise'])[:, 1], 'cached': cached}
            predictions, fields = forecast_branch(branch, [1., 1.]); original_paid = deepcopy(template)
            for j, y in zip(branch['paid_sites'], branch['paid_values']):
                original_paid.update_batch(j, float(y), None, 1)
            for stage, actual, original in zip(('prior', 'raw_paid'), fields, (template, original_paid)):
                for a, b in zip(actual.predict(full_cov=True), original.predict(full_cov=True)):
                    np.testing.assert_allclose(a, b, atol=1e-8, rtol=1e-8)
                for k in ('mean', 'covariance'):
                    np.testing.assert_allclose(predictions[stage][k], cached['predictions'][stage][k], atol=1e-8, rtol=1e-8)
                np.testing.assert_allclose(mean_density(predictions[stage], branch),
                    cached['scores'][stage]['g_gaussian_mean_density'], atol=1e-8, rtol=1e-8)
            zero_excess = scaled_task_diagonal(native, basis.feature_dim, [1., 0.])
            np.testing.assert_array_equal(zero_excess[1:], np.full(basis.feature_dim, .01*native['source_signal_scale']/basis.feature_dim))
            assert np.all(zero_excess[1:] > 0)
            local.append(branch)
        branches[layout] = local
        print('native all73 endpoint and fixed-noise objective reproduced', layout, flush=True)
    fits = {}
    for layout, local in branches.items():
        history = []
        def objective(parameters):
            value = float(np.mean([calibration_loss(forecast_branch(b, parameters)[0], b) for b in local]))
            history.append({'multipliers': np.asarray(parameters).tolist(), 'calibration_objective': value})
            return value
        optimizer = spec['optimizer']; initial_value = objective(optimizer['initial'])
        fitted = minimize(objective, optimizer['initial'], method=optimizer['method'], bounds=optimizer['bounds'],
            options={k: optimizer[k] for k in ('maxiter', 'maxfun', 'ftol', 'gtol')}
            | {'eps': optimizer['finite_difference_step']})
        fits[layout] = {'multipliers': fitted.x.tolist(), 'optimizer_converged': bool(fitted.success),
            'optimizer_message': str(fitted.message), 'iterations': int(fitted.nit), 'evaluations': int(fitted.nfev),
            'initial_objective': initial_value, 'final_objective': float(fitted.fun), 'final_gradient': fitted.jac.tolist(),
            'objective_history': history}
        save(out/f'layout{layout}_calibration.json', fits[layout])
        print('source calibration frozen', layout, {k: v for k, v in fits[layout].items() if k != 'objective_history'}, flush=True)
    save(out/'calibration_fits.json', {'execution': spec, 'fits': fits})
    if not all(f['optimizer_converged'] for f in fits.values()):
        result = {'execution': spec, 'fits': fits, 'calibration_preferred': False,
            'forecasts_scored': False, 'incomplete_calibration': True, 'cost': spec['cost'],
            'cpu_seconds': time.process_time()-started, 'target_promoted': False, 'independent_confirmation': False}
        save(out/'results.json', result); print('incomplete fit; native retained, no score or continuation', flush=True); return
    rows, models = [], {}
    for layout, local in branches.items():
        for b in local:
            predictions, fields = forecast_branch(b, fits[layout]['multipliers'])
            rows.append({'layout': layout, 'deleted_task': b['deleted_task'], 'case': CALIBRATED,
                'scoring_sites': b['sites'], 'predictions': predictions})
            models[(layout, b['deleted_task'])] = fields
    save(out/'predictions.json', {'execution': spec, 'both_calibrations_and_all12_forecasts_frozen': True,
        'native_all73_and_fixed_noise_objective_reproduced': True, 'rows': rows})
    for row in rows:
        b = next(b for b in branches[row['layout']] if b['deleted_task'] == row['deleted_task'])
        row['scores'] = {}
        for stage, field in zip(('prior', 'raw_paid'), models[(row['layout'], row['deleted_task'])]):
            prediction = row['predictions'][stage]
            row['scores'][stage] = {'g_mse': float(np.mean((np.asarray(prediction['mean'])-b['observed'][:, 1])**2)),
                'g_gaussian_mean_density': mean_density(prediction, b),
                'g_raw_batch_density': float(np.mean([joint_batch_log_score(field.local_components(j), value[1], value[2], 3)
                    for j, value in zip(row['scoring_sites'], b['observed'])]))}
    summary = {layout: {case: {stage: {k: float(np.mean([r['scores'][stage][k] for r in
        (rows if case == CALIBRATED else previous) if r['layout'] == layout and r['case'] == case]))
        for k in rows[0]['scores'][stage]} for stage in ('prior', 'raw_paid')} for case in (NATIVE, CALIBRATED)}
        for layout in spec['layouts']}
    contrasts = []
    for row in rows:
        cached = next(b['cached'] for b in branches[row['layout']] if b['deleted_task'] == row['deleted_task'])
        contrasts.append({'layout': row['layout'], 'deleted_task': row['deleted_task'],
            'calibrated_minus_native': {stage: {k: row['scores'][stage][k]-cached['scores'][stage][k]
                for k in row['scores'][stage]} for stage in ('prior', 'raw_paid')}})
    preferred = {layout: summary[layout][CALIBRATED]['raw_paid']['g_mse'] <= summary[layout][NATIVE]['raw_paid']['g_mse']
        and summary[layout][CALIBRATED]['raw_paid']['g_raw_batch_density'] > summary[layout][NATIVE]['raw_paid']['g_raw_batch_density']
        for layout in spec['layouts']}
    result = {'execution': spec, 'fits': fits, 'summary': summary, 'contrasts': contrasts,
        'preferred_by_layout': preferred, 'calibration_preferred': all(preferred.values()),
        'forecasts_scored': True, 'native_all73_and_fixed_noise_objective_reproduced': True,
        'cost': spec['cost'], 'cpu_seconds': time.process_time()-started,
        'target_promoted': False, 'independent_confirmation': False}
    save(out/'calibration_profiles.json', {'execution': spec, 'rows': rows}); save(out/'results.json', result)
    print({'calibration_preferred': result['calibration_preferred'], 'summary': summary,
        'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
