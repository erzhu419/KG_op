"""Existing source GP f fields projected into original orthogonal dictionaries."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_data_scaled_f_tail import NATIVE, forecast_field, score_f
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.check_legacy_public_mean_source_half import components
from performance.check_legacy_task_bank_centering import prepare
from performance.diagnose_legacy_predictive_residual_geometry import geometry
from performance.run_legacy_related_inventory_source import save
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.orthogonal_gp_projection import project_gaussian_field

EXECUTION = 'performance/manifests/legacy_joint_population_f_gp_orthogonal_projection_execution_v1.json'


def run():
    started = time.process_time(); spec = read(EXECUTION)
    coverage, branches = prepare(read(spec['centering_execution']))
    native_all73 = read(spec['native_gp_all73'])['rows']; raw = read(coverage['source_data'])
    paid_values = {i: [raw['tasks'][i]['Y_replicates'][j][0][0] for j in spec['paid_candidates']] for i in spec['training_tasks']}; del raw
    rows, projections, fields = [], [], {}
    for (layout, deleted), b in branches.items():
        fit = read(Path(coverage['output'])/f'layout{layout}'/f'task{deleted}'/'gp_source_fit.json')
        means = dict(b['means'], heads=[dict(h, joint_gp=g['joint_gp']) for h, g in zip(b['means']['heads'], fit['heads'])])
        gp = components(b['inputs'], means, read(b['risk_path']), b['mean_psi'], np.asarray(b['inputs']['public_risk_reference']))[2][0]
        predictions, models = forecast_field(gp, b['sites'], spec['paid_candidates'], paid_values[deleted])
        cached = next(r for r in native_all73 if r['layout'] == layout and r['deleted_task'] == deleted and r['mean_arm'] == 'joint_gp')
        for stage, model in zip(('prior', 'raw_paid'), models):
            mean, covariance = model.predict(full_cov=True); expected = cached['prior'] if stage == 'prior' else cached
            np.testing.assert_allclose(mean, np.asarray(expected['mean'])[:, [0, 3]], atol=1e-8, rtol=1e-8)
            np.testing.assert_allclose(np.diag(covariance.reshape(2*len(mean), 2*len(mean))).reshape(-1, 2),
                np.asarray(expected['variance'])[:, [0, 3]], atol=1e-8, rtol=1e-8)
        rows.append({'layout': layout, 'deleted_task': deleted, 'case': 'native_gp', 'scoring_sites': b['sites'], 'predictions': predictions})
        fields[(layout, deleted, 'native_gp')] = models
        m, c = gp.predict(full_cov=True); design = b['designs'][b['task_name']]; active = np.asarray(b['means']['heads'][0]['joint_structural']['common_group_variances']) > 0.
        columns = {'full_dictionary': list(range(design.shape[1])), 'native_common_groups': [0]+[j+1 for j, g in enumerate(b['basis'].group_index) if active[g]]}
        gp_law = fit['heads'][0]['joint_gp']
        white = float(gp_law['anchor']['normalization']['scale']**2*gp_law['residual_parameters'][2])
        for case, selected in columns.items():
            projected = project_gaussian_field(design[:, selected], m[:, 0], c[:, 0, :, 0], white)
            record = {'layout': layout, 'deleted_task': deleted, 'case': case, 'columns': selected,
                'coefficient_names': [(['intercept']+b['basis'].names)[j] for j in selected],
                'coefficient_mean': projected['coefficient_mean'].tolist(), 'coefficient_covariance': projected['coefficient_covariance'].tolist(),
                'diagnostics': projected['diagnostics']}; projections.append(record)
            field_mean = np.column_stack([projected['mean'], m[:, 1]])
            field_cov = np.zeros_like(c); field_cov[:, 0, :, 0] = projected['covariance']; field_cov[:, 1, :, 1] = c[:, 1, :, 1]
            field = JointFunctionRiskPosterior.from_moments(field_mean, field_cov)
            predictions, models = forecast_field(field, b['sites'], spec['paid_candidates'], paid_values[deleted])
            rows.append({'layout': layout, 'deleted_task': deleted, 'case': case, 'scoring_sites': b['sites'], 'predictions': predictions,
                'projection_diagnostics': projected['diagnostics']})
            fields[(layout, deleted, case)] = models
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    save(out/'projections.json', {'execution': spec, 'rows': projections})
    save(out/'predictions.json', {'execution': spec, 'all24_projected_and12_native_forecasts_frozen_before_scores': True,
        'native_gp_f_eta_prior_paid_all73_reproduced': True, 'rows': rows})
    sources = {layout: read(Path(coverage['output'])/f'layout{layout}'/'inputs.json') for layout in spec['layouts']}
    for row in rows:
        key = row['layout'], row['deleted_task'], row['case']; labels = sources[key[0]]['source_tasks'][key[1]]
        row['scores'] = score_f(row, fields[key], labels)
        row['residual_geometry'] = {stage: geometry(p['mean'], p['covariance'], np.asarray(labels['response'])[:, 0], np.asarray(labels['noise'])[:, 0])
            for stage, p in row['predictions'].items()}
    summary = {layout: {case: {stage: {metric: float(np.mean([r['scores'][stage][metric] for r in rows if r['layout'] == layout and r['case'] == case]))
        for metric in rows[0]['scores'][stage]} for stage in ('prior', 'raw_paid')} for case in spec['cases']} for layout in spec['layouts']}
    contrasts = []
    for row in rows:
        if row['case'] == 'native_gp': continue
        old = next(r for r in rows if r['layout'] == row['layout'] and r['deleted_task'] == row['deleted_task'] and r['case'] == 'native_gp')
        contrasts.append({'layout': row['layout'], 'deleted_task': row['deleted_task'], 'case': row['case'], 'projected_minus_gp': {stage:
            {k: row['scores'][stage][k]-old['scores'][stage][k] for k in row['scores'][stage]} for stage in ('prior', 'raw_paid')}})
    native = read(spec['native_f_profiles'])['rows']
    benchmark = {layout: {stage: {k: float(np.mean([r['scores'][stage][k] for r in native if r['layout'] == layout and r['case'] == NATIVE]))
        for k in native[0]['scores'][stage]} for stage in ('prior', 'raw_paid')} for layout in spec['layouts']}
    result = {'execution': spec, 'summary': summary, 'contrasts': contrasts, 'native_structural_benchmark': benchmark,
        'no_model_selected': True, 'cost': spec['cost'], 'cpu_seconds': time.process_time()-started,
        'source_holdouts_evaluated': False, 'target_promoted': False, 'independent_confirmation': False}
    save(out/'calibration_profiles.json', {'execution': spec, 'rows': rows}); save(out/'results.json', result)
    print({'summary': summary, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
