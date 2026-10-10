"""Scoring-noise resolution and conditional-only calibration objective."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.calibrate_legacy_task_transfer import NATIVE, CALIBRATED
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.check_legacy_public_mean_source_half import components
from performance.check_legacy_source_mean_covariance_swap import g_prediction
from performance.diagnose_legacy_counterbalanced_mse_uncertainty import paired_interval
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.run_legacy_related_inventory_source import problem_for, save, training_view
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.public_control_mean import public_control_mean_coordinates
from representation.task_transfer_calibration import calibrated_g_field, scaled_task_diagonal

EXECUTION = 'performance/manifests/legacy_joint_population_task_transfer_diagnosis_execution_v1.json'


def run():
    started = time.process_time(); spec = read(EXECUTION); calibration_spec = read(spec['calibration_execution'])
    calibration = read(spec['calibration_results']); holdout = read(spec['holdout_execution'])
    coverage = read(holdout['coverage_execution']); parent = read(coverage['source_execution'])
    profile = read(calibration_spec['profile_execution'])
    public = read(Path(coverage['source_data']).with_name('frozen_inputs.json')); bank = [tuple(p) for p in public['bank']]
    raw = read(coverage['source_data']); labels = {i: training_view(raw, [i], public['plan']['task_distribution']['L'])[0]
        for i in spec['heldout_tasks']}
    native = read(spec['native_profiles'])['rows']; candidate = read(spec['holdout_profiles'])['rows']
    intervals = []
    for row in candidate:
        old = next(r for r in native if r['layout'] == row['layout'] and r['task_index'] == row['task_index'])
        label = labels[row['task_index']]
        interval, _ = paired_interval(np.asarray(row['mean'])[2:, 1], np.asarray(old['mean'])[2:, 1],
            label['response'][2:, 1], label['noise'][2:, 1], 3)
        np.testing.assert_allclose(interval['paired_mse_difference'],
            row['scores']['observed_head_mse'][1]-old['scores']['observed_head_mse'][1], atol=1e-12)
        intervals.append({'layout': row['layout'], 'task_index': row['task_index'], **interval})
    original_profiles = read(calibration_spec['training_control_profiles'])['rows']
    calibrated_profiles = read(Path(calibration_spec['output'])/'calibration_profiles.json')['rows']
    source_rows = []
    for layout in spec['layouts']:
        for deleted in spec['training_tasks']:
            inputs = read(Path(coverage['output'])/f'layout{layout}'/f'task{deleted}'/'inputs.json')
            means = read(Path(profile['output'])/f'layout{layout}'/f'task{deleted}'/'selected_source_fit.json')
            risk = read(Path(parent['output'])/f"layout{holdout['parent_layouts'][layout]}"/f'task{deleted}'/'risk_source_fit.json')
            law = means['heads'][1]['joint_structural']; basis = OrthogonalLowOrderBasis(inputs['public_mean_reference'])
            provider = no_outcomes(problem_for(public['plan'], public['tasks'][deleted]))
            psi = (public_control_mean_coordinates(provider, bank)-inputs['mean_coordinate_center'])/inputs['mean_coordinate_scale']
            design = np.column_stack([np.ones(len(psi)), basis.transform(psi)])
            template = components(inputs, means, risk, psi, np.asarray(inputs['public_risk_reference']))[0][1]
            sites = coverage['layouts'][layout][deleted]
            for case, parameters, cached_rows in ((NATIVE, [1., 1.], original_profiles),
                    (CALIBRATED, calibration['fits'][layout]['multipliers'], calibrated_profiles)):
                field = calibrated_g_field(template, design, law, parameters)
                cached = next(r for r in cached_rows if r['layout'] == layout and r['deleted_task'] == deleted and r['case'] == case)
                paid_evidence = []
                for stage in ('prior', 'raw_paid'):
                    if stage == 'raw_paid':
                        task = raw['tasks'][deleted]
                        for j in calibration_spec['paid_candidates']:
                            paid_evidence.append(field.update_batch(j, task['Y_replicates'][j][0][1]-task['tau'], None, 1))
                    mean, covariance = field.predict(full_cov=True)
                    prediction = g_prediction(mean[:, 0], covariance[:, 0, :, 0], sites)
                    for k in ('mean', 'covariance'):
                        np.testing.assert_allclose(prediction[k], cached['predictions'][stage][k], atol=1e-8, rtol=1e-8)
                conditional = cached['scores']['raw_paid']['g_gaussian_mean_density']
                source_rows.append({'layout': layout, 'deleted_task': deleted, 'case': case,
                    'raw_paid_log_evidence': float(sum(paid_evidence)), 'raw_paid_step_log_evidence': paid_evidence,
                    'paid_gaussian_mean_density_per_label': conditional,
                    'sequential_transcript_density_per_label': float((sum(paid_evidence)+len(sites)*conditional)/(len(sites)+2)),
                    'task_intercept_variance': float(scaled_task_diagonal(law, basis.feature_dim, parameters)[0])})
    summary = {layout: {case: {k: float(np.mean([r[k] for r in source_rows if r['layout'] == layout and r['case'] == case]))
        for k in ('raw_paid_log_evidence', 'paid_gaussian_mean_density_per_label', 'sequential_transcript_density_per_label')}
        for case in (NATIVE, CALIBRATED)} for layout in spec['layouts']}
    result = {'execution': spec, 'holdout_measurement_intervals': intervals, 'source_summary': summary,
        'source_rows': source_rows, 'cached_training_moments_and_holdout_mse_reproduced': True,
        'source_gate_remains_failed': True, 'cost': spec['cost'], 'cpu_seconds': time.process_time()-started,
        'target_promoted': False, 'independent_confirmation': False}
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True); save(out/'results.json', result)
    print({'holdout_measurement_intervals': intervals, 'source_summary': summary, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
