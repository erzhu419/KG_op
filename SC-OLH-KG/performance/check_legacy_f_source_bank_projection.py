"""Shared source-bank common function compression, excluding the query bank."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_data_scaled_f_tail import forecast_field, score_f
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.check_legacy_public_mean_source_half import components
from performance.check_legacy_task_bank_centering import prepare
from performance.run_legacy_related_inventory_source import save
from representation.source_kernel_population import shared_kernel
from representation.task_bank_centering import centered_g_field

EXECUTION = 'performance/manifests/legacy_joint_population_f_source_bank_projection_execution_v1.json'


def common_moments(psi, fit):
    anchor = fit['anchor']; cross = shared_kernel(psi, np.asarray(fit['source_features']), anchor)
    scale = anchor['normalization']['scale']
    mean = anchor['normalization']['mean']+scale*(anchor['constant']+cross@np.asarray(fit['common_alpha']))
    covariance = scale**2*(shared_kernel(psi, psi, anchor)-cross@np.asarray(fit['common_precision'])@cross.T)
    return mean, .5*(covariance+covariance.T)


def run():
    started = time.process_time(); spec = read(EXECUTION)
    coverage, branches = prepare(read(spec['centering_execution']))
    cached = read(spec['common_profiles'])['rows']; local_laws = read(spec['covariance_laws'])['rows']
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True); laws, verification = [], []
    for (layout, deleted), b in branches.items():
        native = b['means']['heads'][0]['joint_structural']
        fit = read(Path(coverage['output'])/f'layout{layout}'/f'task{deleted}'/'gp_source_fit.json')['heads'][0]['joint_gp']
        names = [t['name'] for t in b['tasks']]; assert b['task_name'] not in names
        psi = np.vstack([b['mean_psi_by_task'][name] for name in names]); design = np.vstack([b['designs'][name] for name in names])
        mean, covariance = common_moments(psi, fit); query_m, query_c = common_moments(b['mean_psi'], fit)
        query_design = b['designs'][b['task_name']]; q = query_design.shape[1]
        if deleted == 1:
            operator = np.linalg.lstsq(query_design, np.eye(len(query_design)), rcond=None)[0]
            old = next(r for r in local_laws if r['layout'] == layout and r['deleted_task'] == deleted)
            np.testing.assert_allclose(operator@query_m, old['projected_common_mean'], atol=1e-10, rtol=1e-10)
            np.testing.assert_allclose(operator@query_c@operator.T, old['common_covariance'], atol=1e-10, rtol=1e-10)
            verification.append({'layout': layout, 'direct_common_query_law_reproduced': True})
        active = np.asarray(native['common_group_variances']) > 0.
        scopes = {'source_bank_gp_common': list(range(q)),
            'source_bank_native_common_groups': [0]+[j+1 for j, group in enumerate(b['basis'].group_index) if active[group]]}
        b['source_projection_laws'] = {}
        for case, columns in scopes.items():
            operator, _, rank, _ = np.linalg.lstsq(design[:, columns], np.eye(len(design)), rcond=None)
            beta = np.zeros(q); h = np.zeros((q, q)); beta[columns] = operator@mean
            h[np.ix_(columns, columns)] = operator@covariance@operator.T; h = .5*(h+h.T)
            np.testing.assert_allclose(h, h.T, atol=1e-10); assert np.linalg.eigvalsh(h).min() >= -1e-10
            query_projected = query_design@beta; query_cov = query_design@h@query_design.T
            law = dict(native, coefficient_mean=beta.tolist(), population_mean_covariance=h.tolist())
            record = {'layout': layout, 'deleted_task': deleted, 'case': case, 'source_task_names': names,
                'query_bank_excluded_from_projection_operator': True, 'source_bank_point_count': len(design), 'rank': int(rank), 'columns': columns,
                'source_bank_distinct_coordinate_count': int(len(np.unique(psi, axis=0))),
                'source_bank_max_coordinate_difference_from_query': float(max(np.max(np.abs(b['mean_psi_by_task'][name]-b['mean_psi'])) for name in names)),
                'coefficient_common_mean': beta.tolist(), 'coefficient_common_covariance': h.tolist(),
                'source_bank_gp_mean_projection_rms': float(np.sqrt(np.mean((design@beta-mean)**2))),
                'query_gp_mean_projection_rms': float(np.sqrt(np.mean((query_projected-query_m)**2))),
                'query_gp_common_covariance_relative_frobenius_error': float(np.linalg.norm(query_cov-query_c)/np.linalg.norm(query_c))}
            laws.append(record); b['source_projection_laws'][case] = law
    save(out/'source_laws.json', {'execution': spec, 'all24_source_bank_laws_frozen_before_query_paid_outcomes': True,
        'verification': verification, 'rows': laws})
    raw = read(coverage['source_data'])
    paid = {i: [raw['tasks'][i]['Y_replicates'][j][0][0] for j in spec['paid_candidates']] for i in spec['training_tasks']}; del raw
    rows, fields = [], {}
    for (layout, deleted), b in branches.items():
        template = components(b['inputs'], b['means'], read(b['risk_path']), b['mean_psi'], np.asarray(b['inputs']['public_risk_reference']))[0][0]
        for case, law in b['source_projection_laws'].items():
            field = centered_g_field(template, b['designs'][b['task_name']], law, np.zeros(b['basis'].feature_dim))
            predictions, stored = forecast_field(field, b['sites'], spec['paid_candidates'], paid[deleted])
            rows.append({'layout': layout, 'deleted_task': deleted, 'case': case, 'scoring_sites': b['sites'], 'predictions': predictions})
            fields[(layout, deleted, case)] = stored
    controls = [dict(r, case='query_bank_gp_common' if r['case'] == 'gp_common_mean_and_covariance' else 'native_common')
        for r in cached if r['case'] in ('native_common', 'gp_common_mean_and_covariance')]
    save(out/'predictions.json', {'execution': spec, 'all24_new_forecasts_frozen_before_scores': True, 'rows': rows+controls})
    sources = {layout: read(Path(coverage['output'])/f'layout{layout}'/'inputs.json') for layout in spec['layouts']}
    for row in rows:
        key = row['layout'], row['deleted_task'], row['case']; row['scores'] = score_f(row, fields[key], sources[key[0]]['source_tasks'][key[1]])
    all_rows = rows+controls
    summary = {layout: {case: {stage: {k: float(np.mean([r['scores'][stage][k] for r in all_rows if r['layout'] == layout and r['case'] == case]))
        for k in all_rows[0]['scores'][stage]} for stage in ('prior', 'raw_paid')} for case in spec['cases']} for layout in spec['layouts']}
    contrasts = []
    for row in rows:
        query = next(r for r in controls if r['layout'] == row['layout'] and r['deleted_task'] == row['deleted_task'] and r['case'] == 'query_bank_gp_common')
        contrasts.append({'layout': row['layout'], 'deleted_task': row['deleted_task'], 'case': row['case'], 'source_bank_minus_query_bank': {stage:
            {k: row['scores'][stage][k]-query['scores'][stage][k] for k in row['scores'][stage]} for stage in ('prior', 'raw_paid')}})
    result = {'execution': spec, 'summary': summary, 'contrasts': contrasts, 'projection_diagnostics': laws, 'verification': verification,
        'no_model_selected': True, 'cost': spec['cost'], 'cpu_seconds': time.process_time()-started,
        'source_holdouts_evaluated': False, 'target_promoted': False, 'independent_confirmation': False}
    save(out/'calibration_profiles.json', {'execution': spec, 'rows': all_rows}); save(out/'results.json', result)
    print({'summary': summary, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
