"""Original f source likelihood from a source GP prior/task variance initial point."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_data_scaled_f_tail import NATIVE, forecast_field, score_f
from performance.check_legacy_gp_profile_start import initial_profile
from performance.check_legacy_joint_common_covariance import restore_regularizer
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.check_legacy_public_mean_source_half import components
from performance.check_legacy_task_bank_centering import prepare
from performance.run_legacy_related_inventory_source import save
from representation.source_function_prior import head_batches
from representation.source_joint_population import JointCovariancePopulationLikelihood, _solve_joint_common_population
from representation.task_bank_centering import centered_g_field

EXECUTION = 'performance/manifests/legacy_joint_population_f_gp_profile_start_execution_v1.json'
SELECTED = 'source_nll_selected_f_profile'


def snapshot(field):
    mean, covariance = field.predict(full_cov=True)
    n = len(mean)
    return {'mean': mean.tolist(), 'variance': np.diag(covariance.reshape(2*n, 2*n)).reshape(n, 2).tolist()}


def run():
    started = time.process_time(); spec = read(EXECUTION); centering = read(spec['centering_execution'])
    coverage, branches = prepare(centering); out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    fits, verification = [], []
    for (layout, deleted), b in branches.items():
        native = b['means']['heads'][0]['joint_structural']
        prior = restore_regularizer(b['basis'], b['means']['heads'][0]['regularizer'], b['tasks'], 0)
        likelihood = JointCovariancePopulationLikelihood(prior, head_batches(b['tasks'], 0), [t['noise'][:, 0] for t in b['tasks']])
        value, _, state = likelihood.evaluate(np.asarray(native['optimizer_parameters']))
        np.testing.assert_allclose(value, native['source_negative_log_likelihood_per_label'], atol=1e-8, rtol=1e-8)
        for key, saved in (('mean', 'coefficient_mean'), ('mean_covariance', 'population_mean_covariance'),
            ('residual_diagonal', 'residual_coefficient_diagonal'), ('discrepancy', 'discrepancy_variance')):
            np.testing.assert_allclose(state[key], native[saved], atol=1e-8, rtol=1e-8)
        gp = read(Path(coverage['output'])/f'layout{layout}'/f'task{deleted}'/'gp_source_fit.json')['heads'][0]['joint_gp']
        initial, profile = initial_profile(b['designs'][b['task_name']], b['mean_psi'], b['basis'], gp, native, likelihood)
        _, _, initial_state = likelihood.evaluate(initial); count = len(b['basis'].group_names)
        sizes = np.bincount(b['basis'].group_index)
        for params, mass in ((initial[:count], profile['projected_common_prior_group_mass']),
            (initial[count+2:2*count+2], profile['projected_task_group_mass'])):
            np.testing.assert_allclose(params*likelihood.signal, np.maximum(np.asarray(mass)-sizes*likelihood.tail, 0.), atol=1e-12)
        np.testing.assert_allclose(initial_state['residual_diagonal'][0], profile['projected_task_intercept_variance'], atol=1e-12)
        np.testing.assert_allclose(initial_state['common_diagonal'][0], native['common_intercept_variance_fixed'], atol=1e-12)
        np.testing.assert_allclose(likelihood.tail*b['basis'].feature_dim, .01*native['source_signal_scale'], atol=1e-12)
        assert b['inputs']['execution']['optimizer'] == spec['optimizer']
        folder = out/f'layout{layout}'/f'task{deleted}'; folder.mkdir(parents=True, exist_ok=True)
        candidate = _solve_joint_common_population(likelihood, initial,
            native['fixed_common_source_negative_log_likelihood_per_label'], spec['optimizer']).diagnostics
        save(folder/'candidate_fit.json', candidate)
        improvement = value-candidate['source_negative_log_likelihood_per_label']
        selected = candidate['optimizer_converged'] and improvement > spec['optimizer']['ftol']
        b['f_law'] = candidate if selected else native
        save(folder/'selected_source_fit.json', b['f_law'])
        fits.append({'layout': layout, 'deleted_task': deleted, 'selected_gp_start': bool(selected),
            'source_nll_improvement_per_label': float(improvement), 'initial_profile': profile,
            'native': native, 'candidate': candidate, 'selected': b['f_law']})
        verification.append({'layout': layout, 'deleted_task': deleted, 'native_law_and_physical_variance_profile_reproduced': True})
        print('f source profile fit', layout, deleted, candidate['optimizer_converged'], candidate['optimizer_iterations'], bool(selected), improvement, flush=True)
    save(out/'source_laws.json', {'execution': spec, 'all12_selected_laws_frozen_before_paid_outcomes': True,
        'verification': verification, 'rows': fits})
    cost = {**spec['cost'], 'actual_new_f_source_fits': len(fits)}
    if not all(r['candidate']['optimizer_converged'] for r in fits):
        result = {'execution': spec, 'profile_preferred': False, 'all_fits_complete': False, 'forecasts_scored': False,
            'cost': cost, 'cpu_seconds': time.process_time()-started, 'source_holdouts_evaluated': False,
            'target_promoted': False, 'independent_confirmation': False}
        save(out/'results.json', result); print('incomplete fits retained; no completion or scoring', flush=True); return
    raw = read(coverage['source_data'])
    values = {i: [raw['tasks'][i]['Y_replicates'][j][0][0] for j in spec['paid_candidates']] for i in spec['training_tasks']}; del raw
    native_rows = [r for r in read(spec['native_f_profiles'])['rows'] if r['case'] == NATIVE]
    rows, fields = [], {}
    for (layout, deleted), b in branches.items():
        template = components(b['inputs'], b['means'], read(b['risk_path']), b['mean_psi'], np.asarray(b['inputs']['public_risk_reference']))[0][0]
        field = centered_g_field(template, b['designs'][b['task_name']], b['f_law'], np.zeros(b['basis'].feature_dim))
        m, c = field.predict(full_cov=True); om, oc = template.predict(full_cov=True)
        np.testing.assert_array_equal(m[:, 1], om[:, 1]); np.testing.assert_array_equal(c[:, 1, :, 1], oc[:, 1, :, 1])
        predictions, stored = forecast_field(field, b['sites'], spec['paid_candidates'], values[deleted])
        rows.append({'layout': layout, 'deleted_task': deleted, 'case': SELECTED,
            'scoring_sites': b['sites'], 'predictions': predictions,
            'all73': {stage: snapshot(model) for stage, model in zip(('prior', 'raw_paid'), stored)}})
        fields[(layout, deleted)] = stored
    save(out/'predictions.json', {'execution': spec, 'all12_selected_forecasts_frozen_before_scores': True,
        'eta_f_prior_unchanged': True, 'native_frozen_reference': spec['native_f_profiles'], 'rows': rows+native_rows})
    source = {layout: read(Path(coverage['output'])/f'layout{layout}'/'inputs.json') for layout in spec['layouts']}
    for row in rows:
        key = row['layout'], row['deleted_task']; row['scores'] = score_f(row, fields[key], source[key[0]]['source_tasks'][key[1]])
    summary = {layout: {case: {stage: {k: float(np.mean([r['scores'][stage][k] for r in members if r['layout'] == layout]))
        for k in members[0]['scores'][stage]} for stage in ('prior', 'raw_paid')}
        for case, members in ((NATIVE, native_rows), (SELECTED, rows))} for layout in spec['layouts']}
    preferred, contrasts = {}, []
    for layout in spec['layouts']:
        new, old = (summary[layout][case]['raw_paid'] for case in (SELECTED, NATIVE))
        preferred[layout] = new['f_mse'] <= old['f_mse'] and new['f_raw_batch_density'] > old['f_raw_batch_density'] and new['sequential_transcript_density'] >= old['sequential_transcript_density']
    for row in rows:
        old = next(r for r in native_rows if r['layout'] == row['layout'] and r['deleted_task'] == row['deleted_task'])
        contrasts.append({'layout': row['layout'], 'deleted_task': row['deleted_task'], 'selected_minus_native': {stage:
            {k: row['scores'][stage][k]-old['scores'][stage][k] for k in row['scores'][stage]} for stage in ('prior', 'raw_paid')}})
    others = {layout: {k: float(np.mean([r['selected_minus_native']['raw_paid'][k] for r in contrasts if r['layout'] == layout and r['deleted_task'] != 1]))
        for k in contrasts[0]['selected_minus_native']['raw_paid']} for layout in spec['layouts']}
    g = read(Path(centering['output'])/'results.json')['summary']
    result = {'execution': spec, 'summary': summary, 'contrasts': contrasts, 'other5_descriptive_contrasts': others,
        'joint_raw_density_with_identical_stage88_g': {layout: {case: summary[layout][case]['raw_paid']['f_raw_batch_density']
            +g[layout]['public_bank_centered_task_population']['raw_paid']['g_raw_batch_density'] for case in (NATIVE, SELECTED)} for layout in spec['layouts']},
        'source_fit_attribution': fits, 'selected_gp_start_count': sum(r['selected_gp_start'] for r in fits),
        'benchmarks': {name: read(f'paper_artifacts/legacy_joint_population_{name}_v1/results.json')['summary']
            for name in ('data_scaled_f_tail', 'f_task_bank_centering', 'f_gp_orthogonal_projection')},
        'preferred_by_layout': preferred, 'profile_preferred': all(preferred.values()), 'all_fits_complete': True, 'forecasts_scored': True,
        'verification': verification, 'cost': cost, 'cpu_seconds': time.process_time()-started,
        'source_holdouts_evaluated': False, 'target_promoted': False, 'independent_confirmation': False}
    save(out/'calibration_profiles.json', {'execution': spec, 'rows': rows+native_rows}); save(out/'results.json', result)
    print({'profile_preferred': result['profile_preferred'], 'selected_gp_start_count': result['selected_gp_start_count'],
        'summary': summary, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
