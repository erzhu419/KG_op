"""Common f mean/uncertainty factorial, with native task law and white term fixed."""

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
from representation.task_bank_centering import centered_g_field

EXECUTION = 'performance/manifests/legacy_joint_population_f_common_attribution_execution_v1.json'


def run():
    started = time.process_time(); spec = read(EXECUTION)
    coverage, branches = prepare(read(spec['centering_execution']))
    projected = read(spec['covariance_laws'])['rows']; native_rows = read(spec['native_f_profiles'])['rows']
    raw = read(coverage['source_data'])
    paid = {i: [raw['tasks'][i]['Y_replicates'][j][0][0] for j in spec['paid_candidates']] for i in spec['training_tasks']}; del raw
    rows, fields = [], {}
    for (layout, deleted), b in branches.items():
        native = b['means']['heads'][0]['joint_structural']
        gp = next(r for r in projected if r['layout'] == layout and r['deleted_task'] == deleted)
        cached = next(r for r in native_rows if r['layout'] == layout and r['deleted_task'] == deleted and r['case'] == NATIVE)
        template = components(b['inputs'], b['means'], read(b['risk_path']), b['mean_psi'], np.asarray(b['inputs']['public_risk_reference']))[0][0]
        for case, mean_swap, covariance_swap in zip(spec['cases'], (False, True, False, True), (False, False, True, True)):
            law = dict(native,
                coefficient_mean=gp['projected_common_mean'] if mean_swap else native['coefficient_mean'],
                population_mean_covariance=gp['common_covariance'] if covariance_swap else native['population_mean_covariance'])
            field = centered_g_field(template, b['designs'][b['task_name']], law, np.zeros(b['basis'].feature_dim))
            predictions, stored = forecast_field(field, b['sites'], spec['paid_candidates'], paid[deleted])
            if case == 'native_common':
                for stage in ('prior', 'raw_paid'):
                    for key in ('mean', 'covariance'):
                        np.testing.assert_allclose(predictions[stage][key], cached['predictions'][stage][key], atol=1e-8, rtol=1e-8)
                np.testing.assert_allclose(predictions['raw_paid']['paid_predictive_log_score'],
                    cached['predictions']['raw_paid']['paid_predictive_log_score'], atol=1e-8, rtol=1e-8)
            rows.append({'layout': layout, 'deleted_task': deleted, 'case': case, 'scoring_sites': b['sites'],
                'native_discrepancy_variance': native['discrepancy_variance'], 'predictions': predictions})
            fields[(layout, deleted, case)] = stored
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    save(out/'predictions.json', {'execution': spec, 'all48_forecasts_frozen_before_scores': True,
        'native_endpoint_reproduced': True, 'native_R_discrepancy_eta_prior_fixed': True, 'rows': rows})
    source = {layout: read(Path(coverage['output'])/f'layout{layout}'/'inputs.json') for layout in spec['layouts']}
    for row in rows:
        key = row['layout'], row['deleted_task'], row['case']; labels = source[key[0]]['source_tasks'][key[1]]
        if key[2] == 'native_common':
            row['scores'] = next(r for r in native_rows if r['layout'] == key[0] and r['deleted_task'] == key[1] and r['case'] == NATIVE)['scores']
        else:
            row['scores'] = score_f(row, fields[key], labels)
        row['residual_geometry'] = {stage: geometry(p['mean'], p['covariance'], np.asarray(labels['response'])[:, 0], np.asarray(labels['noise'])[:, 0])
            for stage, p in row['predictions'].items()}
    summary = {layout: {case: {stage: {k: float(np.mean([r['scores'][stage][k] for r in rows if r['layout'] == layout and r['case'] == case]))
        for k in rows[0]['scores'][stage]} for stage in ('prior', 'raw_paid')} for case in spec['cases']} for layout in spec['layouts']}
    contrasts = []
    for layout in spec['layouts']:
        for deleted in spec['training_tasks']:
            arms = [next(r for r in rows if r['layout'] == layout and r['deleted_task'] == deleted and r['case'] == case) for case in spec['cases']]
            contrasts.append({'layout': layout, 'deleted_task': deleted, 'stages': {stage: {metric:
                {'mean_only_minus_native': arms[1]['scores'][stage][metric]-arms[0]['scores'][stage][metric],
                 'covariance_only_minus_native': arms[2]['scores'][stage][metric]-arms[0]['scores'][stage][metric],
                 'both_minus_native': arms[3]['scores'][stage][metric]-arms[0]['scores'][stage][metric],
                 'factorial_interaction': arms[3]['scores'][stage][metric]-arms[1]['scores'][stage][metric]-arms[2]['scores'][stage][metric]+arms[0]['scores'][stage][metric]}
                for metric in arms[0]['scores'][stage]} for stage in ('prior', 'raw_paid')}})
    result = {'execution': spec, 'summary': summary, 'contrasts': contrasts,
        'native_endpoint_reproduced': True, 'no_model_selected': True, 'cost': spec['cost'], 'cpu_seconds': time.process_time()-started,
        'source_holdouts_evaluated': False, 'target_promoted': False, 'independent_confirmation': False}
    save(out/'calibration_profiles.json', {'execution': spec, 'rows': rows}); save(out/'results.json', result)
    print({'summary': summary, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
