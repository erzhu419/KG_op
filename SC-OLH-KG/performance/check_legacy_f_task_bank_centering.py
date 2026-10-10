"""Original f weak tail with source-learned public-bank level/shape geometry."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_data_scaled_f_tail import NATIVE, forecast_field, score_f
from performance.check_legacy_joint_common_covariance import restore_regularizer
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.check_legacy_public_mean_source_half import components
from performance.check_legacy_task_bank_centering import prepare, verify_dense
from performance.run_legacy_related_inventory_source import save
from representation.source_function_prior import head_batches
from representation.task_bank_centering import CenteredTaskPopulationLikelihood, centered_g_field, fit_centered_task_population

EXECUTION = 'performance/manifests/legacy_joint_population_f_task_bank_centering_execution_v1.json'
CENTERED = 'public_bank_centered_f_task_population'


def run():
    started = time.process_time(); spec = read(EXECUTION); centering = read(spec['centering_execution'])
    coverage, branches = prepare(centering); out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    laws, verification = [], []
    for (layout, deleted), b in branches.items():
        native = b['means']['heads'][0]['joint_structural']
        prior = restore_regularizer(b['basis'], b['means']['heads'][0]['regularizer'], b['tasks'], 0)
        likelihood = CenteredTaskPopulationLikelihood(prior, head_batches(b['tasks'], 0),
            [t['noise'][:, 0] for t in b['tasks']], [b['centers'][t['name']] for t in b['tasks']])
        if deleted == 1:
            verification.append({'layout': layout, 'deleted_task': deleted,
                **verify_dense(likelihood, np.asarray(native['optimizer_parameters']))})
        assert b['inputs']['execution']['optimizer'] == spec['optimizer']
        folder = out/f'layout{layout}'/f'task{deleted}'; folder.mkdir(parents=True, exist_ok=True)
        law = fit_centered_task_population(likelihood, native, spec['optimizer'])
        law['source_bank_centers'] = {t['name']: b['centers'][t['name']].tolist() for t in b['tasks']}
        law['source_task_names'] = [t['name'] for t in b['tasks']]
        b['f_law'] = law; save(folder/'source_fit.json', law); laws.append({'layout': layout, 'deleted_task': deleted, 'law': law})
        print('centered f source fit', layout, deleted, law['optimizer_converged'], law['optimizer_iterations'], flush=True)
    save(out/'source_laws.json', {'execution': spec, 'all12_laws_frozen_before_forecasts': True, 'verification': verification, 'rows': laws})
    cost = {**spec['cost'], 'actual_new_f_source_fits': len(laws)}
    if not all(r['law']['optimizer_converged'] for r in laws):
        result = {'execution': spec, 'all_fits_complete': False, 'forecasts_scored': False, 'centering_preferred': False,
            'cost': cost, 'cpu_seconds': time.process_time()-started, 'source_holdouts_evaluated': False,
            'target_promoted': False, 'independent_confirmation': False}
        save(out/'results.json', result); print('incomplete fit retained; no scoring or completion', flush=True); return
    raw = read(coverage['source_data'])
    paid_values = {i: [raw['tasks'][i]['Y_replicates'][j][0][0] for j in spec['paid_candidates']] for i in spec['training_tasks']}; del raw
    tail = read(spec['tail_results']); native_rows = [r for r in read(Path(tail['execution']['output'])/'calibration_profiles.json')['rows'] if r['case'] == NATIVE]
    rows, fields = [], {}
    for (layout, deleted), b in branches.items():
        template = components(b['inputs'], b['means'], read(b['risk_path']), b['mean_psi'], np.asarray(b['inputs']['public_risk_reference']))[0][0]
        design, center = b['designs'][b['task_name']], b['centers'][b['task_name']]
        field = centered_g_field(template, design, b['f_law'], center)
        mean, cov = field.predict(full_cov=True); old_mean, old_cov = template.predict(full_cov=True)
        np.testing.assert_array_equal(mean[:, 1], old_mean[:, 1]); np.testing.assert_array_equal(cov[:, 1, :, 1], old_cov[:, 1, :, 1])
        predictions, stored = forecast_field(field, b['sites'], spec['paid_candidates'], paid_values[deleted])
        rows.append({'layout': layout, 'deleted_task': deleted, 'case': CENTERED,
            'scoring_sites': b['sites'], 'query_bank_center': center.tolist(), 'predictions': predictions})
        fields[(layout, deleted)] = stored
    save(out/'predictions.json', {'execution': spec, 'all12_candidate_forecasts_frozen_before_scoring': True,
        'eta_f_prior_unchanged': True, 'native_frozen_reference': str(Path(tail['execution']['output'])/'calibration_profiles.json'), 'rows': rows+native_rows})
    sources = {layout: read(Path(coverage['output'])/f'layout{layout}'/'inputs.json') for layout in spec['layouts']}
    for row in rows:
        key = row['layout'], row['deleted_task']
        row['scores'] = score_f(row, fields[key], sources[key[0]]['source_tasks'][key[1]])
    summary = {layout: {case: {stage: {metric: float(np.mean([r['scores'][stage][metric] for r in members if r['layout'] == layout]))
        for metric in members[0]['scores'][stage]} for stage in ('prior', 'raw_paid')} for case, members in ((NATIVE, native_rows), (CENTERED, rows))}
        for layout in spec['layouts']}
    preferred, contrasts = {}, []
    for layout in spec['layouts']:
        new, old = (summary[layout][case]['raw_paid'] for case in (CENTERED, NATIVE))
        preferred[layout] = new['f_mse'] <= old['f_mse'] and new['f_raw_batch_density'] > old['f_raw_batch_density'] and new['sequential_transcript_density'] >= old['sequential_transcript_density']
    for row in rows:
        old = next(r for r in native_rows if r['layout'] == row['layout'] and r['deleted_task'] == row['deleted_task'])
        contrasts.append({'layout': row['layout'], 'deleted_task': row['deleted_task'], 'centered_minus_native': {stage:
            {k: row['scores'][stage][k]-old['scores'][stage][k] for k in row['scores'][stage]} for stage in ('prior', 'raw_paid')}})
    others = {layout: {k: float(np.mean([r['centered_minus_native']['raw_paid'][k] for r in contrasts if r['layout'] == layout and r['deleted_task'] != 1]))
        for k in contrasts[0]['centered_minus_native']['raw_paid']} for layout in spec['layouts']}
    g = read(Path(centering['output'])/'results.json')['summary']; gp = read('paper_artifacts/legacy_joint_population_training_paid_adaptation_v1/results.json')['summary']
    result = {'execution': spec, 'summary': summary, 'contrasts': contrasts, 'other5_descriptive_contrasts': others,
        'joint_raw_density_with_identical_stage88_g': {layout: {case: summary[layout][case]['raw_paid']['f_raw_batch_density']
            +g[layout]['public_bank_centered_task_population']['raw_paid']['g_raw_batch_density'] for case in (NATIVE, CENTERED)} for layout in spec['layouts']},
        'benchmarks': {'rejected_stage93_tail': tail['summary'], 'same_source_gp': {layout: gp[layout]['joint_gp'] for layout in spec['layouts']}},
        'preferred_by_layout': preferred, 'centering_preferred': all(preferred.values()), 'all_fits_complete': True, 'forecasts_scored': True,
        'verification': verification, 'cost': cost, 'cpu_seconds': time.process_time()-started,
        'source_holdouts_evaluated': False, 'target_promoted': False, 'independent_confirmation': False}
    save(out/'calibration_profiles.json', {'execution': spec, 'rows': rows+native_rows}); save(out/'results.json', result)
    print({'centering_preferred': result['centering_preferred'], 'summary': summary, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
