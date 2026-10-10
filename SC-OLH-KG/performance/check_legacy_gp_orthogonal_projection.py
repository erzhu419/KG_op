"""Represent the existing GP control in native orthogonal source spaces."""

import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.check_legacy_public_mean_source_half import components
from performance.check_legacy_source_mean_covariance_swap import g_prediction
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.run_legacy_related_inventory_source import problem_for, save
from performance.validate_legacy_paired_taskout import gaussian_score
from representation.joint_function_risk import JointFunctionRiskPosterior, joint_batch_log_score
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.orthogonal_gp_projection import project_gaussian_field
from representation.public_control_mean import public_control_mean_coordinates

EXECUTION = 'performance/manifests/legacy_joint_population_gp_orthogonal_projection_execution_v1.json'
CASES = ('native_gp', 'full_dictionary', 'native_common_groups')


def run():
    started = time.process_time(); spec = read(EXECUTION); gain = read(spec['gain_execution'])
    adaptation = read(gain['adaptation_execution']); coverage = read(adaptation['coverage_execution'])
    deletion, parent = read(adaptation['deleted_mean_execution']), read(coverage['source_execution'])
    public = read(Path(coverage['source_data']).with_name('frozen_inputs.json')); bank = [tuple(p) for p in public['bank']]
    old = read(Path(read(spec['swap_results'])['execution']['output'])/'calibration_profiles.json')['rows']
    all73 = read(Path(read(gain['adaptation_results'])['execution']['output'])/'predictions.json')['rows']
    raw = read(coverage['source_data']); paid_sites = spec['paid_candidates']
    values = {i: [raw['tasks'][i]['Y_replicates'][j][0][1]-raw['tasks'][i]['tau'] for j in paid_sites] for i in spec['training_tasks']}
    del raw
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True); rows, projections, models = [], [], {}
    for layout in spec['layouts']:
        for deleted in spec['training_tasks']:
            branch = Path(coverage['output'])/f'layout{layout}'/f'task{deleted}'
            inputs, gp = read(branch/'inputs.json'), read(branch/'gp_source_fit.json')
            means = read(Path(deletion['output'])/f'layout{layout}'/f'task{deleted}'/'source_fit.json')
            native_mean = means['heads'][1]['joint_structural']
            means = dict(means, heads=[dict(h, joint_gp=g['joint_gp']) for h, g in zip(means['heads'], gp['heads'])])
            risk = read(Path(parent['output'])/f"layout{adaptation['parent_layouts'][layout]}"/f'task{deleted}'/'risk_source_fit.json')
            provider = no_outcomes(problem_for(public['plan'], public['tasks'][deleted]))
            psi = (public_control_mean_coordinates(provider, bank)-inputs['mean_coordinate_center'])/inputs['mean_coordinate_scale']
            native_field = components(inputs, means, risk, psi, np.asarray(inputs['public_risk_reference']))[2][1]
            m, c = native_field.predict(full_cov=True); n = len(m); mean, covariance = m[:, 0], c[:, 0, :, 0]
            native_paid = deepcopy(native_field)
            for j, y in zip(paid_sites, values[deleted]):
                native_paid.update_batch(j, float(y), None, 1)
            pm, pc = native_paid.predict(full_cov=True)
            reference = next(r for r in all73 if r['layout'] == layout and r['deleted_task'] == deleted and r['mean_arm'] == 'joint_gp')
            for actual_m, actual_c, prior in ((m, c, reference['prior']), (pm, pc, reference)):
                np.testing.assert_allclose(actual_m, np.asarray(prior['mean'])[:, [1, 2]], atol=1e-8)
                np.testing.assert_allclose(np.diag(actual_c.reshape(2*n, 2*n)).reshape(n, 2), np.asarray(prior['variance'])[:, [1, 2]], atol=1e-8)
            cached = next(r for r in old if r['layout'] == layout and r['deleted_task'] == deleted and r['case'] == 'GG')
            sites = cached['scoring_sites']
            rows.append({'layout': layout, 'deleted_task': deleted, 'case': 'native_gp', 'scoring_sites': sites,
                'predictions': {stage: cached['predictions'][stage] for stage in ('prior', 'raw_paid')}})
            basis = OrthogonalLowOrderBasis(inputs['public_mean_reference'])
            design = np.column_stack([np.ones(n), basis.transform(psi)])
            active = np.asarray(native_mean['common_group_variances']) > 0
            selected_columns = [0]+[j+1 for j, group in enumerate(basis.group_index) if active[group]]
            scopes = {'full_dictionary': list(range(design.shape[1])), 'native_common_groups': selected_columns}
            fit = gp['heads'][1]['joint_gp']
            discrepancy = float(fit['anchor']['normalization']['scale']**2*fit['residual_parameters'][2])
            for case, columns in scopes.items():
                projected = project_gaussian_field(design[:, columns], mean, covariance, discrepancy)
                record = {'layout': layout, 'deleted_task': deleted, 'case': case, 'columns': columns,
                    'coefficient_names': [(['intercept']+basis.names)[j] for j in columns],
                    'active_native_common_groups': [g for g, a in zip(basis.group_names, active) if a],
                    'coefficient_mean': projected['coefficient_mean'].tolist(),
                    'coefficient_covariance': projected['coefficient_covariance'].tolist(), 'diagnostics': projected['diagnostics']}
                projections.append(record)
                field_mean = np.column_stack([projected['mean'], m[:, 1]])
                field_covariance = np.zeros((n, 2, n, 2))
                field_covariance[:, 0, :, 0] = projected['covariance']
                field_covariance[:, 1, :, 1] = c[:, 1, :, 1]
                prior = JointFunctionRiskPosterior.from_moments(field_mean, field_covariance); paid = deepcopy(prior)
                for j, y in zip(paid_sites, values[deleted]):
                    paid.update_batch(j, float(y), None, 1)
                paid_mean, paid_covariance = paid.predict(full_cov=True)
                diagnostic = dict(projected['diagnostics'], mean_paid_variance_reduction_fraction=float(np.mean(
                    1-np.diag(paid_covariance[:, 0, :, 0])[sites]/np.diag(projected['covariance'])[sites])))
                rows.append({'layout': layout, 'deleted_task': deleted, 'case': case, 'scoring_sites': sites,
                    'projection_diagnostics': diagnostic, 'predictions': {
                        'prior': g_prediction(projected['mean'], projected['covariance'], sites),
                        'raw_paid': g_prediction(paid_mean[:, 0], paid_covariance[:, 0, :, 0], sites)}})
                models[(layout, deleted, case)] = prior, paid
        print('orthogonal GP projection forecasts frozen', layout, flush=True)
    save(out/'projections.json', {'execution': spec, 'rows': projections})
    save(out/'predictions.json', {'execution': spec, 'all24_projected_branches_frozen': True, 'native_GG_all73_reproduction_passed': True, 'rows': rows})
    source = {layout: read(Path(coverage['output'])/f'layout{layout}'/'inputs.json') for layout in spec['layouts']}
    for row in rows:
        layout, deleted, case = row['layout'], row['deleted_task'], row['case']
        if case == 'native_gp':
            cached = next(r for r in old if r['layout'] == layout and r['deleted_task'] == deleted and r['case'] == 'GG')
            row['scores'] = {stage: cached['scores'][stage] for stage in ('prior', 'raw_paid')}
            continue
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
    diagnostics = {layout: {case: {k: np.mean([r['projection_diagnostics'][k] for r in rows if r['layout'] == layout and r['case'] == case]).item()
        for k in rows[1]['projection_diagnostics']} for case in CASES[1:]} for layout in spec['layouts']}
    contrasts = []
    for row in rows:
        if row['case'] == 'native_gp':
            continue
        native = next(r for r in rows if r['layout'] == row['layout'] and r['deleted_task'] == row['deleted_task'] and r['case'] == 'native_gp')
        for stage in ('prior', 'raw_paid'):
            a, b = row['scores'][stage], native['scores'][stage]
            contrasts.append({'layout': row['layout'], 'deleted_task': row['deleted_task'], 'case': row['case'], 'stage': stage,
                'projection_minus_native': {k: a[k]-b[k] for k in a}})
    result = {'execution': spec, 'summary': summary, 'projection_summary': diagnostics, 'contrasts': contrasts,
        'native_GG_reproduction_passed': True, 'no_model_selected': True, 'cost': spec['cost'], 'cpu_seconds': time.process_time()-started}
    save(out/'calibration_profiles.json', {'execution': spec, 'rows': rows}); save(out/'results.json', result)
    print({'summary': summary, 'projection_summary': diagnostics, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
