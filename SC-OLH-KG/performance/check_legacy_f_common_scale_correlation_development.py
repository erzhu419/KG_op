"""Unchanged development gate for the frozen joint common geometry source laws."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_data_scaled_f_tail import NATIVE, forecast_field, score_f
from performance.check_legacy_f_gp_profile_start import snapshot
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.run_legacy_joint_population_source_budget import restore_joint
from performance.run_legacy_related_inventory_source import problem_for, save
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.public_control_mean import public_control_mean_coordinates, source_prior_moments

EXECUTION = 'performance/manifests/legacy_joint_population_f_common_scale_correlation_development_execution_v1.json'
CANDIDATE = 'joint_common_scale_correlation'


def f_field(basis, psi, law, eta):
    f = source_prior_moments(restore_joint(basis, law), psi); n = len(psi)
    mean = np.column_stack([f[0], eta[0]]); covariance = np.zeros((n, 2, n, 2))
    covariance[:, 0, :, 0], covariance[:, 1, :, 1] = f[1], eta[1]
    return JointFunctionRiskPosterior.from_moments(mean, covariance)


def run():
    started = time.process_time(); spec = read(EXECUTION)
    fitted = read(spec['source_results']); assert fitted['all_fits_complete'] and fitted['all12_source_laws_verified_and_frozen']
    frozen = read(spec['source_laws']); assert frozen['all12_laws_frozen_before_development_queries']
    source = read(spec['source_execution']); centering = read(source['centering_execution'])
    coverage = read(centering['coverage_execution']); profile = read(centering['profile_execution'])
    adaptation = read(spec['adaptation_execution']); risk_source = read(coverage['source_execution'])
    public = read(Path(coverage['source_data']).with_name('frozen_inputs.json')); bank = [tuple(p) for p in public['bank']]
    coordinates = {i: public_control_mean_coordinates(no_outcomes(problem_for(public['plan'], public['tasks'][i])), bank)
        for i in spec['training_tasks']}
    raw = read(coverage['source_data'])
    paid = {i: [raw['tasks'][i]['Y_replicates'][j][0][0] for j in spec['paid_candidates']] for i in spec['training_tasks']}; del raw
    controls = [r for r in read(spec['native_f_profiles'])['rows'] if r['case'] == NATIVE]
    rows, fields, verification = [], {}, []
    for layout in spec['layouts']:
        for deleted in spec['training_tasks']:
            suffix = Path(f'layout{layout}')/f'task{deleted}'
            inputs = read(Path(coverage['output'])/suffix/'inputs.json')
            means = read(Path(profile['output'])/suffix/'selected_source_fit.json')['heads'][0]
            risk_fit = read(Path(risk_source['output'])/f"layout{adaptation['parent_layouts'][layout]}"/f'task{deleted}'/'risk_source_fit.json')
            basis = OrthogonalLowOrderBasis(inputs['public_mean_reference'])
            risk_basis = OrthogonalLowOrderBasis(inputs['public_risk_reference'])
            eta_law = next(r['joint_structural'] for r in risk_fit['heads'] if r['head'] == 3)
            eta = source_prior_moments(restore_joint(risk_basis, eta_law), np.asarray(inputs['public_risk_reference']))
            psi = (coordinates[deleted]-inputs['mean_coordinate_center'])/inputs['mean_coordinate_scale']
            sites = coverage['layouts'][layout][deleted]
            control = next(r for r in controls if r['layout'] == layout and r['deleted_task'] == deleted)
            np.testing.assert_array_equal(sites, control['scoring_sites'])
            if deleted == 1:
                predicted, _ = forecast_field(f_field(basis, psi, means['joint_structural'], eta), sites, spec['paid_candidates'], paid[deleted])
                for stage in ('prior', 'raw_paid'):
                    for key in ('mean', 'covariance'):
                        np.testing.assert_allclose(predicted[stage][key], control['predictions'][stage][key], atol=1e-8, rtol=1e-8)
                np.testing.assert_allclose(predicted['raw_paid']['paid_predictive_log_score'],
                    control['predictions']['raw_paid']['paid_predictive_log_score'], atol=1e-8, rtol=1e-8)
                verification.append({'layout': layout, 'deleted_task': deleted, 'native_f_prior_paid_moments_and_evidence_reproduced': True})
            law = next(r['law'] for r in frozen['rows'] if r['layout'] == layout and r['deleted_task'] == deleted)
            field = f_field(basis, psi, law, eta)
            mean, covariance = field.predict(full_cov=True)
            np.testing.assert_array_equal(mean[:, 1], eta[0]); np.testing.assert_allclose(covariance[:, 1, :, 1], eta[1], atol=1e-14)
            predictions, stored = forecast_field(field, sites, spec['paid_candidates'], paid[deleted])
            rows.append({'layout': layout, 'deleted_task': deleted, 'case': CANDIDATE, 'scoring_sites': sites,
                'predictions': predictions, 'all73': {stage: snapshot(model) for stage, model in zip(('prior', 'raw_paid'), stored)}})
            fields[(layout, deleted)] = stored
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    save(out/'predictions.json', {'execution': spec, 'all12_forecasts_frozen_before_scores': True,
        'source_laws_unchanged': True, 'eta_f_prior_unchanged': True, 'verification': verification, 'rows': rows})
    labels = {layout: read(Path(coverage['output'])/f'layout{layout}'/'inputs.json')['source_tasks'] for layout in spec['layouts']}
    contrasts = []
    for row in rows:
        layout, deleted = row['layout'], row['deleted_task']
        row['scores'] = score_f(row, fields[(layout, deleted)], labels[layout][deleted])
        old = next(r for r in controls if r['layout'] == layout and r['deleted_task'] == deleted)
        contrasts.append({'layout': layout, 'deleted_task': deleted, 'candidate_minus_native': {stage:
            {key: row['scores'][stage][key]-old['scores'][stage][key] for key in row['scores'][stage]} for stage in ('prior', 'raw_paid')}})
    summary = {layout: {case: {stage: {key: float(np.mean([r['scores'][stage][key] for r in members if r['layout'] == layout]))
        for key in members[0]['scores'][stage]} for stage in ('prior', 'raw_paid')}
        for case, members in ((NATIVE, controls), (CANDIDATE, rows))} for layout in spec['layouts']}
    gates = {}
    for layout in spec['layouts']:
        new, old = (summary[layout][case]['raw_paid'] for case in (CANDIDATE, NATIVE))
        gates[layout] = {'paid_f_mse_nonincrease': new['f_mse'] <= old['f_mse'],
            'raw_batch_density_improved': new['f_raw_batch_density'] > old['f_raw_batch_density'],
            'sequential_transcript_density_nondecrease': new['sequential_transcript_density'] >= old['sequential_transcript_density']}
    passed = all(all(g.values()) for g in gates.values())
    result = {'execution': spec, 'summary': summary, 'contrasts': contrasts, 'gates_by_layout': gates,
        'passes_both_layouts': passed, 'candidate_stopped_after_development_failure': not passed,
        'other5_descriptive_contrasts': {layout: {key: float(np.mean([r['candidate_minus_native']['raw_paid'][key]
            for r in contrasts if r['layout'] == layout and r['deleted_task'] != 1])) for key in contrasts[0]['candidate_minus_native']['raw_paid']}
            for layout in spec['layouts']},
        'verification': verification, 'no_model_adopted': True, 'cost': spec['cost'], 'cpu_seconds': time.process_time()-started,
        'source6_7_evaluated': False, 'target_promoted': False, 'independent_confirmation': False}
    save(out/'calibration_profiles.json', {'execution': spec, 'rows': rows})
    save(out/'results.json', result)
    print({'passes_both_layouts': passed, 'gates': gates, 'summary': summary, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
