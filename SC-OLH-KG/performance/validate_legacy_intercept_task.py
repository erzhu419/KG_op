"""Training-task validation of restricted task variation, with frozen risk laws."""

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
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.run_legacy_related_inventory_source import problem_for, save
from performance.validate_legacy_paired_taskout import gaussian_score
from representation.intercept_task_population import InterceptTaskPopulationLikelihood, fit_intercept_task_population
from representation.joint_function_risk import joint_batch_log_score
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.public_control_mean import public_control_mean_coordinates
from representation.source_function_prior import head_batches

EXECUTION = 'performance/manifests/legacy_joint_population_intercept_task_validation_execution_v1.json'
CASES = ('native_source_nll_selection', 'intercept_task')


def run():
    started = time.process_time(); spec = read(EXECUTION); profile = read(spec['profile_execution'])
    projection = read(profile['projection_execution']); gain = read(projection['gain_execution'])
    adaptation = read(gain['adaptation_execution']); coverage = read(adaptation['coverage_execution'])
    parent = read(coverage['source_execution'])
    previous = read(Path(profile['output'])/'calibration_profiles.json')['rows']
    public = read(Path(coverage['source_data']).with_name('frozen_inputs.json')); bank = [tuple(p) for p in public['bank']]
    raw = read(coverage['source_data']); paid_sites = spec['paid_candidates']
    values = {i: [raw['tasks'][i]['Y_replicates'][j][0][1]-raw['tasks'][i]['tau'] for j in paid_sites]
        for i in spec['training_tasks']}; del raw
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    rows, models, attribution, fit_count = [], {}, [], 0
    for layout in spec['layouts']:
        for deleted in spec['training_tasks']:
            inputs = read(Path(coverage['output'])/f'layout{layout}'/f'task{deleted}'/'inputs.json')
            means = read(Path(profile['output'])/f'layout{layout}'/f'task{deleted}'/'selected_source_fit.json')
            risk = read(Path(parent['output'])/f"layout{adaptation['parent_layouts'][layout]}"/f'task{deleted}'/'risk_source_fit.json')
            native = means['heads'][1]['joint_structural']; basis = OrthogonalLowOrderBasis(inputs['public_mean_reference'])
            tasks = [{**t, **{k: np.asarray(t[k]) for k in ('psi', 'features', 'response', 'noise')}} for t in inputs['source_tasks']]
            prior = restore_regularizer(basis, means['heads'][1]['regularizer'], tasks, 1)
            likelihood = InterceptTaskPopulationLikelihood(prior, head_batches(tasks, 1), [t['noise'][:, 1] for t in tasks])
            folder = out/f'layout{layout}'/f'task{deleted}'; folder.mkdir(parents=True, exist_ok=True)
            fitted = fit_intercept_task_population(likelihood, native, optimizer=inputs['execution']['optimizer']).diagnostics
            fit_count += 1
            if not fitted['optimizer_converged']:
                save(folder/'before_completion.json', fitted)
                fitted = fit_intercept_task_population(likelihood, fitted, optimizer=inputs['execution']['optimizer']).diagnostics
                fit_count += 1
            save(folder/'restricted_g_fit.json', fitted)
            if not fitted['optimizer_converged']:
                raise RuntimeError(f'restricted task fit incomplete at {layout}/{deleted}; no scores')
            candidate = deepcopy(means); candidate['heads'][1]['joint_structural'] = fitted
            save(folder/'source_fit.json', candidate)
            attribution.append({'layout': layout, 'deleted_task': deleted, 'candidate': fitted,
                'restricted_minus_native_source_nll': fitted['source_negative_log_likelihood_per_label']-native['source_negative_log_likelihood_per_label']})
            provider = no_outcomes(problem_for(public['plan'], public['tasks'][deleted]))
            psi = (public_control_mean_coordinates(provider, bank)-inputs['mean_coordinate_center'])/inputs['mean_coordinate_scale']
            sites = coverage['layouts'][layout][deleted]
            cached = next(r for r in previous if r['layout'] == layout and r['deleted_task'] == deleted and r['case'] == CASES[0])
            for case, source_fit in zip(CASES, (means, candidate)):
                field = components(inputs, source_fit, risk, psi, np.asarray(inputs['public_risk_reference']))[0][1]
                pm, pc = field.predict(full_cov=True); paid = deepcopy(field)
                for j, y in zip(paid_sites, values[deleted]):
                    paid.update_batch(j, float(y), None, 1)
                qm, qc = paid.predict(full_cov=True)
                predictions = {'prior': g_prediction(pm[:, 0], pc[:, 0, :, 0], sites),
                    'raw_paid': g_prediction(qm[:, 0], qc[:, 0, :, 0], sites)}
                if case == CASES[0]:
                    for stage, prediction in predictions.items():
                        for key in ('mean', 'covariance'):
                            np.testing.assert_allclose(prediction[key], cached['predictions'][stage][key], atol=1e-8, rtol=1e-8)
                rows.append({'layout': layout, 'deleted_task': deleted, 'case': case, 'scoring_sites': sites, 'predictions': predictions})
                models[(layout, deleted, case)] = field, paid
            print('restricted task forecasts frozen', layout, deleted, flush=True)
    save(out/'predictions.json', {'execution': spec, 'all12_restricted_branches_frozen': True,
        'native_scoring_site_mean_full_covariance_reproduced': True, 'rows': rows})
    source = {layout: read(Path(coverage['output'])/f'layout{layout}'/'inputs.json') for layout in spec['layouts']}
    for row in rows:
        layout, deleted, case = row['layout'], row['deleted_task'], row['case']
        if case == CASES[0]:
            cached = next(r for r in previous if r['layout'] == layout and r['deleted_task'] == deleted and r['case'] == case)
            row['scores'] = cached['scores']; continue
        labels = source[layout]['source_tasks'][deleted]; observed = np.asarray(labels['response']); noise = np.asarray(labels['noise'])[:, 1]
        row['scores'] = {}
        for stage, model in zip(('prior', 'raw_paid'), models[(layout, deleted, case)]):
            prediction = row['predictions'][stage]; predicted = np.asarray(prediction['mean']); covariance = np.asarray(prediction['covariance'])
            row['scores'][stage] = {'g_mse': float(np.mean((predicted-observed[:, 1])**2)),
                'g_gaussian_mean_density': gaussian_score(predicted, covariance+np.diag(noise), observed[:, 1])/len(predicted),
                'g_raw_batch_density': float(np.mean([joint_batch_log_score(model.local_components(j), value[1], value[2], 3)
                    for j, value in zip(row['scoring_sites'], observed)]))}
    summary = {layout: {case: {stage: {k: np.mean([r['scores'][stage][k] for r in rows if r['layout'] == layout and r['case'] == case]).item()
        for k in rows[0]['scores'][stage]} for stage in ('prior', 'raw_paid')} for case in CASES} for layout in spec['layouts']}
    contrasts = []
    for row in rows:
        if row['case'] == CASES[0]:
            continue
        baseline = next(r for r in rows if r['layout'] == row['layout'] and r['deleted_task'] == row['deleted_task'] and r['case'] == CASES[0])
        contrasts.append({'layout': row['layout'], 'deleted_task': row['deleted_task'],
            'restricted_minus_native': {stage: {k: row['scores'][stage][k]-baseline['scores'][stage][k]
                for k in row['scores'][stage]} for stage in ('prior', 'raw_paid')}})
    preferred = {layout: summary[layout][CASES[1]]['raw_paid']['g_mse'] <= summary[layout][CASES[0]]['raw_paid']['g_mse']
        and summary[layout][CASES[1]]['raw_paid']['g_raw_batch_density'] > summary[layout][CASES[0]]['raw_paid']['g_raw_batch_density']
        for layout in spec['layouts']}
    result = {'execution': spec, 'summary': summary, 'contrasts': contrasts, 'fit_attribution': attribution,
        'preferred_by_layout': preferred, 'restricted_family_preferred': all(preferred.values()),
        'cost': {**spec['cost'], 'actual_new_g_source_fits': fit_count}, 'cpu_seconds': time.process_time()-started,
        'independent_confirmation': False, 'target_promoted': False}
    save(out/'calibration_profiles.json', {'execution': spec, 'rows': rows}); save(out/'results.json', result)
    print({'restricted_family_preferred': result['restricted_family_preferred'], 'summary': summary,
        'new_g_source_fits': fit_count, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
