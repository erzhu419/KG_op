"""Fixed projected common f field, comparing only task covariance geometry."""

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
from representation.orthogonal_function_prior import spectral_weights
from representation.orthogonal_gp_projection import project_gaussian_field
from representation.source_kernel_population import rbf_cross

EXECUTION = 'performance/manifests/legacy_joint_population_f_projected_task_spectrum_execution_v1.json'


def run():
    started = time.process_time(); spec = read(EXECUTION)
    coverage, branches = prepare(read(spec['centering_execution']))
    cached = read(spec['projection_profiles'])['rows']; raw = read(coverage['source_data'])
    paid = {i: [raw['tasks'][i]['Y_replicates'][j][0][0] for j in spec['paid_candidates']] for i in spec['training_tasks']}; del raw
    rows, laws, fields = [], [], {}
    for (layout, deleted), b in branches.items():
        native = b['means']['heads'][0]['joint_structural']
        gp_fit = read(Path(coverage['output'])/f'layout{layout}'/f'task{deleted}'/'gp_source_fit.json')
        means = dict(b['means'], heads=[dict(h, joint_gp=g['joint_gp']) for h, g in zip(b['means']['heads'], gp_fit['heads'])])
        gp = components(b['inputs'], means, read(b['risk_path']), b['mean_psi'], np.asarray(b['inputs']['public_risk_reference']))[2][0]
        m, c = gp.predict(full_cov=True); n = len(m); design = b['designs'][b['task_name']]
        operator = np.linalg.lstsq(design, np.eye(n), rcond=None)[0]
        fit = gp_fit['heads'][0]['joint_gp']; scale = fit['anchor']['normalization']['scale']**2
        amplitude, intercept, discrepancy = np.asarray(fit['residual_parameters'])*scale
        kernel = amplitude*rbf_cross(b['mean_psi'], b['mean_psi'], fit['anchor']['lengthscale'])+intercept
        task = operator@kernel@operator.T; task = .5*(task+task.T)
        common = operator@(c[:, 0, :, 0]-kernel-discrepancy*np.eye(n))@operator.T; common = .5*(common+common.T)
        projected = project_gaussian_field(design, m[:, 0], c[:, 0, :, 0], discrepancy)
        np.testing.assert_allclose(common+task, projected['coefficient_covariance'], atol=1e-10, rtol=1e-10)
        diagonal = np.diag(task).copy(); tail = .01*native['source_signal_scale']/b['basis'].feature_dim
        sizes = np.bincount(b['basis'].group_index); mass = np.bincount(b['basis'].group_index, weights=diagonal[1:])
        excess = np.maximum(mass-sizes*tail, 0.); weights, _ = spectral_weights(b['basis'], native['spectral_exponent'])
        structured = np.r_[diagonal[0], excess[b['basis'].group_index]*weights+tail]
        actual_mass = np.bincount(b['basis'].group_index, weights=structured[1:])
        np.testing.assert_allclose(actual_mass, excess+sizes*tail, atol=1e-12)
        variants = {'projected_full_task': task, 'diagonal_task': np.diag(diagonal),
            'native_group_spectrum_tail': np.diag(structured), 'native_task_law': np.diag(native['residual_coefficient_diagonal'])}
        for matrix in [common]+list(variants.values()):
            np.testing.assert_allclose(matrix, matrix.T, atol=1e-10)
            assert np.linalg.eigvalsh(matrix).min() >= -1e-10
        laws.append({'layout': layout, 'deleted_task': deleted, 'projected_common_mean': projected['coefficient_mean'].tolist(),
            'common_covariance': common.tolist(), 'task_covariances': {case: value.tolist() for case, value in variants.items()}})
        old = next(r for r in cached if r['layout'] == layout and r['deleted_task'] == deleted and r['case'] == 'full_dictionary')
        for case, matrix in variants.items():
            covariance = design@(common+matrix)@design.T+discrepancy*np.eye(n)
            fm = np.column_stack([projected['mean'], m[:, 1]]); fc = np.zeros_like(c)
            fc[:, 0, :, 0], fc[:, 1, :, 1] = covariance, c[:, 1, :, 1]
            field = JointFunctionRiskPosterior.from_moments(fm, fc)
            predictions, stored = forecast_field(field, b['sites'], spec['paid_candidates'], paid[deleted])
            if case == 'projected_full_task':
                for stage in ('prior', 'raw_paid'):
                    for key in ('mean', 'covariance'):
                        np.testing.assert_allclose(predictions[stage][key], old['predictions'][stage][key], atol=1e-8, rtol=1e-8)
                np.testing.assert_allclose(predictions['raw_paid']['paid_predictive_log_score'],
                    old['predictions']['raw_paid']['paid_predictive_log_score'], atol=1e-8, rtol=1e-8)
            off = task-np.diag(diagonal)
            diagnostics = {'task_intercept_variance': float(matrix[0, 0]), 'native_task_intercept_variance': native['residual_coefficient_diagonal'][0],
                'projected_task_off_diagonal_squared_mass_fraction': float(np.sum(off**2)/np.sum(task**2)),
                'native_task_spectral_exponent': native['spectral_exponent'], 'native_tail': float(tail),
                'projected_task_group_mass': mass.tolist(), 'native_spectral_task_group_mass': actual_mass.tolist(),
                'task_trace_added_by_native_tail': float(np.sum(actual_mass-mass))}
            rows.append({'layout': layout, 'deleted_task': deleted, 'case': case, 'scoring_sites': b['sites'],
                'diagnostics': diagnostics, 'predictions': predictions})
            fields[(layout, deleted, case)] = stored
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    save(out/'covariance_laws.json', {'execution': spec, 'rows': laws})
    save(out/'predictions.json', {'execution': spec, 'all48_forecasts_frozen_before_scores': True,
        'projected_full_endpoint_reproduced': True, 'rows': rows})
    source = {layout: read(Path(coverage['output'])/f'layout{layout}'/'inputs.json') for layout in spec['layouts']}
    for row in rows:
        key = row['layout'], row['deleted_task'], row['case']; labels = source[key[0]]['source_tasks'][key[1]]
        if key[2] == 'projected_full_task':
            row['scores'] = next(r for r in cached if r['layout'] == key[0] and r['deleted_task'] == key[1] and r['case'] == 'full_dictionary')['scores']
        else:
            row['scores'] = score_f(row, fields[key], labels)
        row['residual_geometry'] = {stage: geometry(p['mean'], p['covariance'], np.asarray(labels['response'])[:, 0], np.asarray(labels['noise'])[:, 0])
            for stage, p in row['predictions'].items()}
    summary = {layout: {case: {stage: {k: float(np.mean([r['scores'][stage][k] for r in rows if r['layout'] == layout and r['case'] == case]))
        for k in rows[0]['scores'][stage]} for stage in ('prior', 'raw_paid')} for case in spec['cases']} for layout in spec['layouts']}
    contrasts = []
    for row in rows:
        if row['case'] == 'projected_full_task': continue
        old = next(r for r in rows if r['layout'] == row['layout'] and r['deleted_task'] == row['deleted_task'] and r['case'] == 'projected_full_task')
        contrasts.append({'layout': row['layout'], 'deleted_task': row['deleted_task'], 'case': row['case'], 'compressed_minus_full': {stage:
            {k: row['scores'][stage][k]-old['scores'][stage][k] for k in row['scores'][stage]} for stage in ('prior', 'raw_paid')}})
    native_rows = read(spec['native_f_profiles'])['rows']
    benchmark = {layout: {stage: {k: float(np.mean([r['scores'][stage][k] for r in native_rows if r['layout'] == layout and r['case'] == NATIVE]))
        for k in native_rows[0]['scores'][stage]} for stage in ('prior', 'raw_paid')} for layout in spec['layouts']}
    result = {'execution': spec, 'summary': summary, 'contrasts': contrasts, 'native_structural_benchmark': benchmark,
        'projected_full_endpoint_reproduced': True, 'no_model_selected': True, 'cost': spec['cost'], 'cpu_seconds': time.process_time()-started,
        'source_holdouts_evaluated': False, 'target_promoted': False, 'independent_confirmation': False}
    save(out/'calibration_profiles.json', {'execution': spec, 'rows': rows}); save(out/'results.json', result)
    print({'summary': summary, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
