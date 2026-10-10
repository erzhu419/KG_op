"""Fixed-marginal common-prior fit and unchanged f/g development prediction gates."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_data_scaled_f_tail import forecast_field, score_f
from performance.check_legacy_f_gp_profile_start import snapshot
from performance.check_legacy_joint_common_covariance import restore_regularizer
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.check_legacy_public_mean_source_half import components
from performance.check_legacy_task_bank_centering import CENTERED, prepare
from performance.check_legacy_task_spectral_moments import score as score_g
from performance.run_legacy_related_inventory_source import save
from representation.source_common_correlation import CommonPriorCorrelationLikelihood, fit_common_correlation
from representation.source_function_prior import head_batches
from representation.task_bank_centering import CenteredTaskPopulationLikelihood, centered_g_field

EXECUTION = 'performance/manifests/legacy_joint_population_common_correlation_execution_v1.json'
CORRELATED, CONTROL = 'source_common_correlation', 'native_control'


def verify_dense(likelihood, p, epsilon=None):
    source = likelihood.source; value, derivative, state = likelihood.evaluate(p)
    a = state['common_prior_covariance']; np.testing.assert_array_equal(np.diag(a), likelihood.diagonal)
    assert np.linalg.eigvalsh(a).min() >= -1e-12
    np.testing.assert_array_equal(state['residual_diagonal'], likelihood.native['residual_coefficient_diagonal'])
    np.testing.assert_allclose(state['discrepancy'], likelihood.native['discrepancy_variance'], atol=1e-14)
    design = np.vstack([b[0] for b in source.blocks]); error = np.concatenate([b[1] for b in source.blocks])
    covariance = design@a@design.T+block_diag(*[(z*state['residual_diagonal'])@z.T+np.diag(b[2]+state['discrepancy'])
        for b, z in zip(source.blocks, source.task_designs)])
    factor = cho_factor(covariance, lower=True); alpha = cho_solve(factor, error); n = source.label_count
    dense_value = .5*(n*np.log(2*np.pi)+2*np.log(np.diag(factor[0])).sum()+error@alpha)/n
    dense_mean = source.prior.mean+a@design.T@alpha
    dense_h = a-a@design.T@cho_solve(factor, design)@a
    dense_derivative = .5*np.sum((cho_solve(factor, np.eye(n))-np.outer(alpha, alpha))*(design@likelihood.direction@design.T).T)/n
    np.testing.assert_allclose(value, dense_value, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(state['mean'], dense_mean, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(state['mean_covariance'], dense_h, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(derivative, dense_derivative, atol=1e-9, rtol=1e-7)
    record = {'p': p, 'nll_dense_error': float(abs(value-dense_value)),
        'common_posterior_dense_max_error': float(np.max(np.abs(state['mean_covariance']-dense_h))),
        'derivative_dense_error': float(abs(derivative-dense_derivative)), 'marginals_task_covariance_and_psd_verified': True}
    if epsilon is not None:
        finite = (likelihood.evaluate(p+epsilon)[0]-likelihood.evaluate(p-epsilon)[0])/(2*epsilon)
        np.testing.assert_allclose(derivative, finite, atol=1e-7, rtol=1e-5)
        record['derivative_finite_difference_error'] = float(abs(derivative-finite))
    return record


def run():
    started = time.process_time(); spec = read(EXECUTION); centering = read(spec['centering_execution'])
    geometry = read(spec['geometry_results']); assert geometry['all24_directions_defined_and_verified']
    coverage, branches = prepare(centering); out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    laws, verification = [], []
    for (layout, deleted), b in branches.items():
        b['correlated_laws'] = {}
        for head, name in enumerate(spec['heads']):
            if name == 'g':
                native = read(Path(centering['output'])/f'layout{layout}'/f'task{deleted}'/'source_fit.json')
                centers = [np.asarray(native['source_bank_centers'][t['name']]) for t in b['tasks']]
            else:
                native = b['means']['heads'][head]['joint_structural']; centers = [np.zeros(b['basis'].feature_dim) for _ in b['tasks']]
            prior = restore_regularizer(b['basis'], b['means']['heads'][head]['regularizer'], b['tasks'], head)
            source = CenteredTaskPopulationLikelihood(prior, head_batches(b['tasks'], head), [t['noise'][:, head] for t in b['tasks']], centers)
            direction = next(r['coefficient_covariance_direction'] for r in geometry['rows']
                if r['layout'] == layout and r['deleted_task'] == deleted and r['head'] == name)
            likelihood = CommonPriorCorrelationLikelihood(source, native, direction)
            if deleted == 1:
                verification.append({'layout': layout, 'deleted_task': deleted, 'head': name, 'point': 'interior',
                    **verify_dense(likelihood, spec['gradient_verification_p'], spec['gradient_verification_epsilon'])})
            law = fit_common_correlation(likelihood, spec['optimizer'])
            verification.append({'layout': layout, 'deleted_task': deleted, 'head': name, 'point': 'fitted',
                **verify_dense(likelihood, law['common_correlation_strength'])})
            b['correlated_laws'][name] = law
            laws.append({'layout': layout, 'deleted_task': deleted, 'head': name, 'law': law})
        print('f/g common-correlation fits complete', layout, deleted, flush=True)
    save(out/'source_laws.json', {'execution': spec, 'all24_laws_frozen_before_paid_outcomes': True, 'rows': laws, 'verification': verification})
    cost = {**spec['cost'], 'actual_new_f_source_fits': 12, 'actual_new_g_source_fits': 12}
    if not all(r['law']['optimizer_converged'] for r in laws):
        result = {'execution': spec, 'all_fits_complete': False, 'forecasts_scored': False, 'no_model_adopted': True,
            'cost': cost, 'cpu_seconds': time.process_time()-started, 'source_holdouts_evaluated': False,
            'target_promoted': False, 'independent_confirmation': False}
        save(out/'results.json', result); print('incomplete scalar source fits; no scoring', flush=True); return
    raw = read(coverage['source_data'])
    values = {i: {'f': [raw['tasks'][i]['Y_replicates'][j][0][0] for j in spec['paid_candidates']],
        'g': [raw['tasks'][i]['Y_replicates'][j][0][1]-raw['tasks'][i]['tau'] for j in spec['paid_candidates']]}
        for i in spec['training_tasks']}; del raw
    controls = [dict(r, head='f', case=CONTROL) for r in read(spec['native_f_profiles'])['rows'] if r['case'] == 'native_f']
    controls += [dict(r, head='g', case=CONTROL) for r in read(spec['native_g_profiles'])['rows'] if r['case'] == CENTERED]
    rows, fields = [], {}
    for (layout, deleted), b in branches.items():
        templates = components(b['inputs'], b['means'], read(b['risk_path']), b['mean_psi'], np.asarray(b['inputs']['public_risk_reference']))[0]
        for head, name in enumerate(spec['heads']):
            center = b['centers'][b['task_name']] if name == 'g' else np.zeros(b['basis'].feature_dim)
            field = centered_g_field(templates[head], b['designs'][b['task_name']], b['correlated_laws'][name], center)
            predictions, stored = forecast_field(field, b['sites'], spec['paid_candidates'], values[deleted][name])
            rows.append({'layout': layout, 'deleted_task': deleted, 'head': name, 'case': CORRELATED,
                'scoring_sites': b['sites'], 'predictions': predictions,
                'all73': {stage: snapshot(model) for stage, model in zip(('prior', 'raw_paid'), stored)}})
            fields[(layout, deleted, name)] = stored
    save(out/'predictions.json', {'execution': spec, 'all24_forecasts_frozen_before_scores': True, 'eta_priors_unchanged': True, 'rows': rows+controls})
    sources = {layout: read(Path(coverage['output'])/f'layout{layout}'/'inputs.json') for layout in spec['layouts']}
    contrasts = []
    for row in rows:
        layout, deleted, head = row['layout'], row['deleted_task'], row['head']; labels = sources[layout]['source_tasks'][deleted]
        if head == 'f':
            row['scores'] = score_f(row, fields[(layout, deleted, head)], labels)
        else:
            b = branches[(layout, deleted)]; b['observed'] = np.asarray(labels['response']); b['noise'] = np.asarray(labels['noise'])[:, 1]
            row['scores'] = score_g(row, b, fields[(layout, deleted, head)])
        control = next(r for r in controls if r['layout'] == layout and r['deleted_task'] == deleted and r['head'] == head)
        contrasts.append({'layout': layout, 'deleted_task': deleted, 'head': head, 'correlated_minus_control': {stage:
            {k: row['scores'][stage][k]-control['scores'][stage][k] for k in row['scores'][stage]} for stage in ('prior', 'raw_paid')}})
    summary, preferred = {}, {}
    for layout in spec['layouts']:
        summary[layout] = {}; preferred[layout] = {}
        for head in spec['heads']:
            summary[layout][head] = {case: {stage: {k: float(np.mean([r['scores'][stage][k] for r in members if r['layout'] == layout and r['head'] == head]))
                for k in next(r for r in members if r['head'] == head)['scores'][stage]} for stage in ('prior', 'raw_paid')}
                for case, members in ((CONTROL, controls), (CORRELATED, rows))}
            new, old = (summary[layout][head][case]['raw_paid'] for case in (CORRELATED, CONTROL))
            preferred[layout][head] = new[head+'_mse'] <= old[head+'_mse'] and new[head+'_raw_batch_density'] > old[head+'_raw_batch_density'] and new['sequential_transcript_density'] >= old['sequential_transcript_density']
    result = {'execution': spec, 'summary': summary, 'contrasts': contrasts, 'preferred_by_layout_and_head': preferred,
        'head_passes_both_layouts': {head: all(preferred[l][head] for l in spec['layouts']) for head in spec['heads']},
        'other5_descriptive_contrasts': {layout: {head: {k: float(np.mean([r['correlated_minus_control']['raw_paid'][k]
            for r in contrasts if r['layout'] == layout and r['head'] == head and r['deleted_task'] != 1]))
            for k in next(r for r in contrasts if r['head'] == head)['correlated_minus_control']['raw_paid']}
            for head in spec['heads']} for layout in spec['layouts']},
        'source_fits': [{'layout': r['layout'], 'deleted_task': r['deleted_task'], 'head': r['head'],
            'p': r['law']['common_correlation_strength'],
            'total_source_log_likelihood_gain': r['law']['source_label_count']*(r['law']['initial_source_negative_log_likelihood_per_label']-r['law']['source_negative_log_likelihood_per_label'])}
            for r in laws], 'verification': verification, 'all_fits_complete': True, 'forecasts_scored': True,
        'no_model_adopted': True, 'cost': cost, 'cpu_seconds': time.process_time()-started,
        'source_holdouts_evaluated': False, 'target_promoted': False, 'independent_confirmation': False}
    save(out/'calibration_profiles.json', {'execution': spec, 'rows': rows+controls}); save(out/'results.json', result)
    print({'preferred': preferred, 'summary': summary, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
