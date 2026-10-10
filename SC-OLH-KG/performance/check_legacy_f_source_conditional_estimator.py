"""Source conditional composite fitting, followed by the original f task gate."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_data_scaled_f_tail import NATIVE, forecast_field, score_f
from performance.check_legacy_f_gp_profile_start import snapshot
from performance.check_legacy_joint_common_covariance import restore_regularizer
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.check_legacy_public_mean_source_half import components
from performance.check_legacy_task_bank_centering import prepare
from performance.run_legacy_related_inventory_source import save
from representation.source_conditional_population import SourceTaskConditionalLikelihood, fit_source_conditional_population
from representation.source_function_prior import head_batches
from representation.source_joint_population import JointCovariancePopulationLikelihood
from representation.task_bank_centering import centered_g_field

EXECUTION = 'performance/manifests/legacy_joint_population_f_source_conditional_estimator_execution_v1.json'
CONDITIONAL = 'source_conditional_f_estimator'


def verify_conditionals(likelihood):
    full = likelihood.full; count = len(full.basis.group_names)
    point = np.full(2*count+4, .1); point[count] = point[-2] = 2.
    value, gradient, state = likelihood.evaluate(point)
    design = np.vstack([b[0] for b in full.blocks]); error = np.concatenate([b[1] for b in full.blocks])
    covariance = (design*state['common_diagonal'])@design.T+block_diag(*[
        (d*state['residual_diagonal'])@d.T+np.diag(noise+state['discrepancy']) for d, _, noise in full.blocks])
    factor = cho_factor(covariance, lower=True); precision = cho_solve(factor, np.eye(len(error))); alpha = precision@error
    total, offset = 0., 0
    for block in full.blocks:
        n = len(block[1]); sl = slice(offset, offset+n); conditional_precision = precision[sl, sl]
        pf = cho_factor(conditional_precision, lower=True)
        total += .5*(n*np.log(2*np.pi)-2*np.log(np.diag(pf[0])).sum()+alpha[sl]@cho_solve(pf, alpha[sl]))
        offset += n
    dense = total/full.label_count
    np.testing.assert_allclose(value, dense, atol=1e-8, rtol=0.)
    eps = 1e-5; finite = []
    for j in range(len(point)):
        a, b = point.copy(), point.copy(); a[j] += eps; b[j] -= eps
        finite.append((likelihood.evaluate(a)[0]-likelihood.evaluate(b)[0])/(2*eps))
    finite = np.asarray(finite); np.testing.assert_allclose(gradient, finite, atol=1e-6, rtol=1e-5)
    return {'dense_conditional_density_error': float(abs(value-dense)),
        'all_parameter_gradient_max_error': float(np.max(np.abs(gradient-finite))),
        'full_and_subset_signal_tail_units_identical': all(s.signal == full.signal and s.tail == full.tail for s in likelihood.subsets)}


def run():
    started = time.process_time(); spec = read(EXECUTION); centering = read(spec['centering_execution'])
    coverage, branches = prepare(centering); out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    verification = []
    for (layout, deleted), b in branches.items():
        native = b['means']['heads'][0]['joint_structural']
        prior = restore_regularizer(b['basis'], b['means']['heads'][0]['regularizer'], b['tasks'], 0)
        full = JointCovariancePopulationLikelihood(prior, head_batches(b['tasks'], 0), [t['noise'][:, 0] for t in b['tasks']])
        np.testing.assert_allclose(full.evaluate(np.asarray(native['optimizer_parameters']))[0],
            native['source_negative_log_likelihood_per_label'], atol=1e-8, rtol=1e-8)
        assert b['inputs']['execution']['optimizer'] == spec['optimizer']
        b['conditional'] = SourceTaskConditionalLikelihood(full)
        if deleted == 1:
            verification.append({'layout': layout, 'deleted_task': deleted, **verify_conditionals(b['conditional'])})
    save(out/'source_verification.json', {'execution': spec, 'verification': verification,
        'dense_conditionals_all_gradients_and_native_source_objectives_passed': True})
    print('conditional Gaussian block densities and all parameter gradients verified', flush=True)
    laws = []
    for (layout, deleted), b in branches.items():
        folder = out/f'layout{layout}'/f'task{deleted}'; folder.mkdir(parents=True, exist_ok=True)
        native = b['means']['heads'][0]['joint_structural']
        law = fit_source_conditional_population(b['conditional'], native, spec['optimizer'])
        b['f_law'] = law; save(folder/'source_fit.json', law); laws.append({'layout': layout, 'deleted_task': deleted, 'law': law})
        print('conditional source f fit', layout, deleted, law['optimizer_converged'], law['optimizer_iterations'],
            law['source_conditional_composite_negative_log_likelihood_per_label'], flush=True)
    save(out/'source_laws.json', {'execution': spec, 'all12_laws_frozen_before_paid_outcomes': True, 'rows': laws})
    cost = {**spec['cost'], 'actual_new_f_source_fits': len(laws)}
    if not all(r['law']['optimizer_converged'] for r in laws):
        result = {'execution': spec, 'conditional_preferred': False, 'all_fits_complete': False, 'forecasts_scored': False,
            'cost': cost, 'cpu_seconds': time.process_time()-started, 'source_holdouts_evaluated': False,
            'target_promoted': False, 'independent_confirmation': False}
        save(out/'results.json', result); print('incomplete source fits; no completion or scoring', flush=True); return
    raw = read(coverage['source_data'])
    values = {i: [raw['tasks'][i]['Y_replicates'][j][0][0] for j in spec['paid_candidates']] for i in spec['training_tasks']}; del raw
    native_rows = [r for r in read(spec['native_f_profiles'])['rows'] if r['case'] == NATIVE]
    rows, fields = [], {}
    for (layout, deleted), b in branches.items():
        template = components(b['inputs'], b['means'], read(b['risk_path']), b['mean_psi'], np.asarray(b['inputs']['public_risk_reference']))[0][0]
        field = centered_g_field(template, b['designs'][b['task_name']], b['f_law'], np.zeros(b['basis'].feature_dim))
        predictions, stored = forecast_field(field, b['sites'], spec['paid_candidates'], values[deleted])
        rows.append({'layout': layout, 'deleted_task': deleted, 'case': CONDITIONAL, 'scoring_sites': b['sites'],
            'predictions': predictions, 'all73': {stage: snapshot(model) for stage, model in zip(('prior', 'raw_paid'), stored)}})
        fields[(layout, deleted)] = stored
    save(out/'predictions.json', {'execution': spec, 'all12_forecasts_frozen_before_scores': True,
        'eta_f_prior_and_original_function_family_preserved': True, 'rows': rows+native_rows})
    source = {layout: read(Path(coverage['output'])/f'layout{layout}'/'inputs.json') for layout in spec['layouts']}
    for row in rows:
        key = row['layout'], row['deleted_task']; row['scores'] = score_f(row, fields[key], source[key[0]]['source_tasks'][key[1]])
    summary = {layout: {case: {stage: {k: float(np.mean([r['scores'][stage][k] for r in members if r['layout'] == layout]))
        for k in members[0]['scores'][stage]} for stage in ('prior', 'raw_paid')} for case, members in ((NATIVE, native_rows), (CONDITIONAL, rows))}
        for layout in spec['layouts']}
    preferred, contrasts = {}, []
    for layout in spec['layouts']:
        new, old = (summary[layout][case]['raw_paid'] for case in (CONDITIONAL, NATIVE))
        preferred[layout] = new['f_mse'] <= old['f_mse'] and new['f_raw_batch_density'] > old['f_raw_batch_density'] and new['sequential_transcript_density'] >= old['sequential_transcript_density']
    for row in rows:
        old = next(r for r in native_rows if r['layout'] == row['layout'] and r['deleted_task'] == row['deleted_task'])
        contrasts.append({'layout': row['layout'], 'deleted_task': row['deleted_task'], 'conditional_minus_native': {stage:
            {k: row['scores'][stage][k]-old['scores'][stage][k] for k in row['scores'][stage]} for stage in ('prior', 'raw_paid')}})
    result = {'execution': spec, 'summary': summary, 'contrasts': contrasts,
        'other5_descriptive_contrasts': {layout: {k: float(np.mean([r['conditional_minus_native']['raw_paid'][k] for r in contrasts if r['layout'] == layout and r['deleted_task'] != 1]))
            for k in contrasts[0]['conditional_minus_native']['raw_paid']} for layout in spec['layouts']},
        'preferred_by_layout': preferred, 'conditional_preferred': all(preferred.values()), 'all_fits_complete': True, 'forecasts_scored': True,
        'source_objective_values': [{'layout': r['layout'], 'deleted_task': r['deleted_task'],
            'initial_conditional': r['law']['initial_source_conditional_composite_negative_log_likelihood_per_label'],
            'fitted_conditional': r['law']['source_conditional_composite_negative_log_likelihood_per_label'],
            'initial_marginal': r['law']['initial_source_negative_log_likelihood_per_label'],
            'fitted_marginal': r['law']['source_negative_log_likelihood_per_label']} for r in laws],
        'verification': verification, 'cost': cost, 'cpu_seconds': time.process_time()-started,
        'source_holdouts_evaluated': False, 'target_promoted': False, 'independent_confirmation': False}
    save(out/'calibration_profiles.json', {'execution': spec, 'rows': rows+native_rows}); save(out/'results.json', result)
    print({'conditional_preferred': result['conditional_preferred'], 'summary': summary, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
